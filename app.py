"""
RuVoiCer — Gradio GUI for English → Russian audio translation
with voice cloning and intonation preservation.

Pipeline:
  1. ASR  (faster-whisper)   → English transcript + timestamps
  2. Translation (Ollama / NLLB-200)  → Russian text
  3. TTS  (XTTS-v2 voice clone)       → Russian speech in original voice
  4. Assembly  (pydub)                 → final .wav
"""

import faulthandler
import sys
faulthandler.enable(file=sys.stderr, all_threads=True)

import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ruvoicer")

# ── Pre-load Whisper BEFORE importing gradio/tqdm ────────────────────
# ctranslate2 crashes with "access violation" if CUDA init happens
# while any background threads are running (tqdm monitor, gradio analytics).
# We must load the model before those packages are even imported.
import config  # noqa: F401  (sets env vars on import)
from core.transcriber import Transcriber

_TRANSCRIBER: Transcriber | None = None


def get_transcriber(model_size: str) -> Transcriber:
    global _TRANSCRIBER
    if _TRANSCRIBER is None or _TRANSCRIBER.model_size != model_size:
        if _TRANSCRIBER is not None:
            _TRANSCRIBER.unload()
        _TRANSCRIBER = Transcriber(model_size=model_size)
        _TRANSCRIBER.load()
    return _TRANSCRIBER


logger.info("Pre-loading Whisper model (before Gradio starts)...")
get_transcriber("large-v3-turbo")
logger.info("Whisper ready.")

# ── Now safe to import gradio and everything else ────────────────────
import gradio as gr
import torch

from core.translator import NLLBTranslator, OllamaTranslator
from core.synthesizer import Synthesizer
from core.assembler import Assembler
from core.diarizer import Diarizer
from core.separator import VocalSeparator
from utils.audio import convert_to_wav, extract_reference_audio, merge_references

OUTPUT_DIR = Path(config.OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════
#  System checks
# ═════════════════════════════════════════════════════════════════════════

def system_info() -> str:
    """Return a short markdown summary of the runtime environment."""
    lines = []
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f"**GPU:** {gpu}  ({vram:.1f} GB VRAM)")
    else:
        lines.append("**GPU:** не обнаружена (будет использоваться CPU — медленно)")
    lines.append(f"**PyTorch:** {torch.__version__}")
    lines.append(f"**CUDA available:** {torch.cuda.is_available()}")
    return "\n".join(lines)


def check_ollama(base_url: str) -> str:
    translator = OllamaTranslator(base_url=base_url)
    if translator.check_available():
        return "Ollama доступен ✔"
    return "Ollama недоступен ✘ — убедитесь, что `ollama serve` запущен"


def _resolve_paths(file_list):
    """Extract string paths from gr.File objects (handles str and NamedString)."""
    if not file_list:
        return []
    paths = []
    for f in file_list:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "name"):
            paths.append(f.name)
        else:
            paths.append(str(f))
    return paths


# ═════════════════════════════════════════════════════════════════════════
#  Voice template management
# ═════════════════════════════════════════════════════════════════════════

VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)


def create_voice_template(template_name, voice_files):
    """Create a merged voice reference template from multiple audio files."""
    if not template_name or not template_name.strip():
        return "❌ Введите имя шаблона", _list_voice_templates()
    if not voice_files:
        return "❌ Загрузите аудиофайлы", _list_voice_templates()

    name = template_name.strip().replace(" ", "_")
    template_dir = VOICES_DIR / name
    template_dir.mkdir(exist_ok=True)

    paths = _resolve_paths(voice_files)

    # convert all to wav
    wav_refs = []
    for i, p in enumerate(paths):
        wav = convert_to_wav(p, template_dir / f"src_{i:03d}.wav")
        ref = extract_reference_audio(wav, template_dir / f"ref_{i:03d}.wav")
        wav_refs.append(str(ref))

    # merge into single reference
    merged = merge_references(wav_refs, template_dir / "voice.wav", max_total_sec=45)

    return f"✅ Шаблон «{name}» создан ({len(paths)} файлов → {merged})", _list_voice_templates()


def _list_voice_templates():
    """Return list of available voice template names."""
    templates = []
    for d in sorted(VOICES_DIR.iterdir()):
        if d.is_dir() and (d / "voice.wav").exists():
            templates.append(d.name)
    return templates


def get_voice_choices():
    return ["(авто из загруженных)"] + _list_voice_templates()


# ═════════════════════════════════════════════════════════════════════════
#  Batch processing pipeline (generator for live log)
# ═════════════════════════════════════════════════════════════════════════

def process_batch(
    audio_files,
    reference_files,
    voice_template,
    output_folder,
    whisper_model,
    source_language,
    translation_method,
    ollama_model,
    ollama_url,
    multi_speaker,
    hf_token,
):
    """Generator — yields (log_text, result_files, details_text) at each step."""
    log_lines = []

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")

    def state(files=None, details="", zip_file=None, audio=None):
        return "\n".join(log_lines), files, details, zip_file, audio

    audio_paths = _resolve_paths(audio_files)
    if not audio_paths:
        log("❌ Загрузите хотя бы один аудиофайл!")
        yield state()
        return

    n = len(audio_paths)
    log(f"📁 Загружено файлов: {n}")

    # determine output folder
    out_dir = Path(output_folder.strip()) if output_folder and output_folder.strip() else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"📂 Папка результатов: {out_dir}")
    else:
        out_dir = Path(tempfile.mkdtemp(dir=OUTPUT_DIR))
        log(f"📂 Папка результатов: {out_dir}")
    yield state()

    batch_dir = Path(tempfile.mkdtemp(dir=OUTPUT_DIR))

    # ── 0. Convert all to WAV ────────────────────────────────────────────
    wav_paths = []
    for i, f in enumerate(audio_paths):
        fname = Path(f).name
        log(f"🔄 [{i+1}/{n}] Конвертация: {fname}")
        yield state()
        wav = convert_to_wav(f, batch_dir / f"input_{i:03d}.wav")
        wav_paths.append(wav)

    # ── 0.5. Vocal separation (Demucs) ──────────────────────────────────
    log("🎵 Разделение аудио на голос и фон (Demucs)…")
    yield state()
    separator = VocalSeparator()
    separator.load()
    vocals_paths = []  # clean vocals per file
    bg_paths = []      # background per file
    for i, wav in enumerate(wav_paths):
        fname = Path(audio_paths[i]).name
        log(f"🎵 [{i+1}/{n}] Разделение: {fname}")
        yield state()
        sep_dir = batch_dir / f"separated_{i:03d}"
        result = separator.separate(str(wav), str(sep_dir))
        vocals_paths.append(result["vocals"])
        bg_paths.append(result["background"])
    separator.unload()
    log("🎵 Demucs выгружен из VRAM")
    yield state()

    # ── 1. Diarization (if multi-speaker) ───────────────────────────────
    all_diarization = []  # per-file diarization results
    diarizer = None
    if multi_speaker:
        if not hf_token or not hf_token.strip():
            log("❌ Для мультиспикера нужен HuggingFace токен!")
            yield state()
            return
        log("👥 Режим мультиспикера: запуск диаризации…")
        yield state()
        diarizer = Diarizer(hf_token=hf_token.strip())
        diarizer.load()
        log("👥 Модель диаризации загружена")
        yield state()

        for i, wav in enumerate(wav_paths):
            fname = Path(audio_paths[i]).name
            log(f"👥 [{i+1}/{n}] Диаризация: {fname}")
            yield state()
            dseg = diarizer.diarize(vocals_paths[i])
            all_diarization.append(dseg)
            speakers = {s["speaker"] for s in dseg}
            log(f"   → {len(dseg)} фрагментов, спикеров: {len(speakers)} ({', '.join(sorted(speakers))})")
            yield state()

        diarizer.unload()
        log("👥 Модель диаризации выгружена из VRAM")
        yield state()

    # ── 2. Build voice reference(s) ──────────────────────────────────────
    # In multi-speaker mode: extract per-speaker reference from each file
    # In single-speaker mode: use template or merge from all files
    speaker_refs: dict[str, str] = {}  # speaker_id -> ref wav path

    use_template = voice_template and voice_template != "(авто из загруженных)"
    if use_template and not multi_speaker:
        template_path = VOICES_DIR / voice_template / "voice.wav"
        if template_path.exists():
            speaker_refs["_default"] = str(template_path)
            log(f"🎤 Используется шаблон голоса: «{voice_template}»")
            yield state()
        else:
            log(f"⚠ Шаблон «{voice_template}» не найден, извлекаю из файлов…")
            use_template = False

    if multi_speaker:
        # collect all speakers across all files
        all_speakers = set()
        for dseg in all_diarization:
            for s in dseg:
                all_speakers.add(s["speaker"])

        log(f"🎤 Извлечение голосов для {len(all_speakers)} спикеров…")
        yield state()

        for spk in sorted(all_speakers):
            # find the file with the longest turn for this speaker
            best_file_idx = None
            best_turn_len = 0
            for i, dseg in enumerate(all_diarization):
                for s in dseg:
                    if s["speaker"] == spk:
                        dur = s["end"] - s["start"]
                        if dur > best_turn_len:
                            best_turn_len = dur
                            best_file_idx = i

            if best_file_idx is not None:
                ref_path = str(batch_dir / f"ref_{spk}.wav")
                extracted = Diarizer.extract_speaker_audio(
                    vocals_paths[best_file_idx],
                    all_diarization[best_file_idx],
                    spk,
                    ref_path,
                )
                if extracted:
                    speaker_refs[spk] = extracted
                    log(f"   🎤 {spk}: {best_turn_len:.1f}с эталон из файла {best_file_idx+1}")

        yield state()

    elif not use_template:
        log(f"🎤 Извлечение эталонного голоса из {n} файлов…")
        yield state()

        ref_clips = []
        for i, wav in enumerate(wav_paths):
            ref = extract_reference_audio(vocals_paths[i], batch_dir / f"ref_{i:03d}.wav")
            ref_clips.append(str(ref))

        extra_paths = _resolve_paths(reference_files)
        for i, rf in enumerate(extra_paths):
            ref = convert_to_wav(rf, batch_dir / f"ref_extra_{i:03d}.wav")
            ref_clips.append(str(ref))

        merged = str(merge_references(ref_clips, batch_dir / "voice_merged.wav"))
        speaker_refs["_default"] = merged
        log(f"🎤 Собран эталон: {len(ref_clips)} клипов → 1 файл")
        yield state()

    # ── 3. Transcribe all files ──────────────────────────────────────────
    log("📝 Инициализация Whisper…")
    yield state()
    transcriber = get_transcriber(whisper_model)

    all_segments = []
    for i, wav in enumerate(wav_paths):
        fname = Path(audio_paths[i]).name
        log(f"📝 [{i+1}/{n}] Транскрибирование: {fname}")
        yield state()
        segs, info = transcriber.transcribe(str(wav), language=source_language)

        # assign speaker labels if multi-speaker
        if multi_speaker and i < len(all_diarization):
            segs = Diarizer.assign_speakers(segs, all_diarization[i])
            spk_counts = {}
            for s in segs:
                spk_counts[s["speaker"]] = spk_counts.get(s["speaker"], 0) + 1
            spk_info = ", ".join(f"{k}: {v}" for k, v in sorted(spk_counts.items()))
            log(f"   → {len(segs)} сегм. | спикеры: {spk_info}")
        else:
            log(f"   → {len(segs)} сегм., язык: {info.language} ({info.language_probability * 100:.0f}%)")

        all_segments.append(segs)
        yield state()

    total_segs = sum(len(s) for s in all_segments)
    log(f"📝 Итого: {total_segs} сегментов из {n} файлов")
    yield state()

    # ── 4. Translate ─────────────────────────────────────────────────────
    if translation_method == "Ollama (LLM)":
        translator = OllamaTranslator(model=ollama_model, base_url=ollama_url)
        if not translator.check_available():
            log("❌ Ollama недоступен! Запустите `ollama serve`")
            yield state()
            return
        log(f"🌐 Ollama подключён: {ollama_model}")
    else:
        log("🌐 Загрузка модели NLLB-200…")
        yield state()
        translator = NLLBTranslator()
        translator.load()
        log("🌐 NLLB-200 загружена")
    yield state()

    all_translated = []
    for i, segs in enumerate(all_segments):
        fname = Path(audio_paths[i]).name
        log(f"🌐 [{i+1}/{n}] Перевод: {fname} ({len(segs)} сегм.)")
        yield state()
        translated = translator.translate_segments(segs)
        all_translated.append(translated)
        if translated:
            sample = translated[0]
            log(f'   → "{sample["text"][:50]}" → "{sample["translated_text"][:50]}"')
            yield state()

    if hasattr(translator, "unload"):
        translator.unload()
        log("🌐 Переводчик выгружен из VRAM")
        yield state()

    # ── 5. Synthesize ────────────────────────────────────────────────────
    log("🔊 Загрузка TTS модели (XTTS-v2)…")
    yield state()
    synthesizer = Synthesizer()
    synthesizer.load()
    log("🔊 TTS модель загружена")
    yield state()

    all_synthesized = []
    for i, translated in enumerate(all_translated):
        fname = Path(audio_paths[i]).name
        seg_dir = batch_dir / f"segments_{i:03d}"
        seg_dir.mkdir()
        log(f"🔊 [{i+1}/{n}] Синтез речи: {fname} ({len(translated)} сегм.)")
        yield state()

        if multi_speaker:
            # synthesize each segment with the correct speaker's voice
            synthesized = []
            for j, seg in enumerate(translated):
                text = seg.get("translated_text", "").strip()
                spk = seg.get("speaker", "_default")
                ref = speaker_refs.get(spk, speaker_refs.get("_default", next(iter(speaker_refs.values()))))

                if text:
                    out_path = str(seg_dir / f"seg_{j:04d}.wav")
                    try:
                        synthesizer.synthesize(text, ref, out_path, language="ru")
                        synthesized.append({**seg, "audio_path": out_path})
                    except Exception as e:
                        logger.error("Synthesis failed for segment %d: %s", j, e)
                        synthesized.append({**seg, "audio_path": None})
                else:
                    synthesized.append({**seg, "audio_path": None})
        else:
            ref = speaker_refs.get("_default", next(iter(speaker_refs.values())))
            synthesized = synthesizer.synthesize_segments(
                translated, ref, str(seg_dir), language="ru"
            )

        all_synthesized.append(synthesized)
        ok = sum(1 for s in synthesized if s.get("audio_path"))
        log(f"   → Синтезировано {ok}/{len(translated)} сегментов")
        yield state()

    synthesizer.unload()
    log("🔊 TTS модель выгружена из VRAM")
    yield state()

    # ── 6. Assemble ──────────────────────────────────────────────────────
    result_paths = []
    assembler = Assembler()
    for i, synthesized in enumerate(all_synthesized):
        fname = Path(audio_paths[i]).stem
        result_name = f"{fname}_ru.wav"
        out_path = str(out_dir / result_name)
        log(f"🔗 [{i+1}/{n}] Сборка: {result_name}")
        yield state()
        result = assembler.assemble(synthesized, out_path, background_path=bg_paths[i])
        if result:
            result_paths.append(result)

    # ── Build details text ───────────────────────────────────────────────
    details_lines = []
    for i in range(n):
        fname = Path(audio_paths[i]).name
        details_lines.append(f"{'═' * 50}")
        details_lines.append(f" Файл {i+1}: {fname}")
        details_lines.append(f"{'═' * 50}")
        for seg in all_translated[i]:
            spk = seg.get("speaker", "")
            spk_tag = f" [{spk}]" if spk else ""
            details_lines.append(f"[{seg['start']:.1f}–{seg['end']:.1f}s]{spk_tag}")
            details_lines.append(f"  EN: {seg['text']}")
            details_lines.append(f"  RU: {seg['translated_text']}")
        details_lines.append("")

    log(f"✅ Готово! Обработано: {n}, результатов: {len(result_paths)}")
    if out_dir:
        log(f"📂 Файлы сохранены в: {out_dir.resolve()}")

    # Create ZIP for download
    zip_path = None
    if result_paths:
        zip_path = str(batch_dir / "results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rp in result_paths:
                zf.write(rp, Path(rp).name)
        log(f"📦 ZIP архив: {zip_path}")

    # Pick first result for audio player preview
    first_audio = str(result_paths[0]) if result_paths else None
    yield state(files=result_paths, details="\n".join(details_lines), zip_file=zip_path, audio=first_audio)


# ═════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ═════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
)

# JS to auto-scroll log textarea to bottom on every update
AUTO_SCROLL_JS = """
() => {
    const box = document.querySelector('#log_box textarea');
    if (!box) return;
    const observer = new MutationObserver(() => {
        box.scrollTop = box.scrollHeight;
    });
    observer.observe(box, {childList: true, characterData: true, subtree: true});
    // Also poll for value changes (Gradio updates value property)
    setInterval(() => { box.scrollTop = box.scrollHeight; }, 500);
}
"""

with gr.Blocks(title="RuVoiCer") as app:

    # ── TAB 1: Main processing ───────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("🎙 Обработка"):
            gr.Markdown(
                "# 🎙 RuVoiCer\n"
                "### Пакетный перевод аудио EN → RU с клонированием голоса\n"
            )
            gr.Markdown(system_info())

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    audio_input = gr.File(
                        label="🎧 Аудиофайлы (можно несколько)",
                        file_count="multiple",
                        file_types=["audio"],
                    )
                    reference_input = gr.File(
                        label="🎤 Доп. эталон голоса (опционально)",
                        file_count="multiple",
                        file_types=["audio"],
                    )
                    voice_template = gr.Dropdown(
                        choices=get_voice_choices(),
                        value="(авто из загруженных)",
                        label="🗣 Шаблон голоса",
                        allow_custom_value=False,
                    )
                    output_folder = gr.Textbox(
                        value="",
                        label="📂 Папка для результатов (пусто = output/tmp…)",
                        placeholder="D:\\MyDub\\episode1",
                    )

                    with gr.Accordion("Настройки", open=False):
                        whisper_model = gr.Dropdown(
                            choices=["large-v3", "large-v3-turbo", "medium", "small", "base"],
                            value="large-v3-turbo",
                            label="Модель Whisper",
                        )
                        source_language = gr.Dropdown(
                            choices=["en", "auto"],
                            value="en",
                            label="Язык исходного аудио",
                        )
                        translation_method = gr.Radio(
                            choices=["Ollama (LLM)", "NLLB-200 (офлайн)"],
                            value="Ollama (LLM)",
                            label="Метод перевода",
                        )
                        ollama_model = gr.Textbox(
                            value="gemma3:4b",
                            label="Модель Ollama",
                        )
                        ollama_url = gr.Textbox(
                            value="http://localhost:11434",
                            label="URL Ollama",
                        )
                        ollama_status = gr.Textbox(
                            label="Статус Ollama", interactive=False
                        )
                        check_btn = gr.Button("Проверить Ollama")

                    with gr.Accordion("👥 Мультиспикер (несколько голосов)", open=False):
                        multi_speaker = gr.Checkbox(
                            value=False,
                            label="Включить диаризацию (определение спикеров)",
                            info="Каждый спикер получит свой голос в дубляже. Требуется HuggingFace токен.",
                        )
                        hf_token = gr.Textbox(
                            value="",
                            label="🔑 HuggingFace токен",
                            type="password",
                            placeholder="hf_...",
                            info="Нужно принять лицензию pyannote на huggingface.co/pyannote/speaker-diarization-3.1",
                        )

                    process_btn = gr.Button(
                        "▶  Запустить обработку",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=1):
                    log_output = gr.Textbox(
                        label="📋 Лог обработки",
                        lines=25,
                        max_lines=60,
                        interactive=False,
                        elem_id="log_box",
                    )
                    audio_player = gr.Audio(
                        label="🔊 Прослушать результат",
                        interactive=False,
                        type="filepath",
                    )
                    download_zip = gr.File(
                        label="⬇ Скачать все результаты (ZIP)",
                        interactive=False,
                        visible=True,
                    )
                    result_files = gr.File(
                        label="📦 Отдельные файлы",
                        file_count="multiple",
                        interactive=False,
                    )
                    with gr.Accordion("Транскрипт и перевод", open=False):
                        details_output = gr.Textbox(
                            label="Детали (EN → RU)",
                            lines=15,
                            interactive=False,
                        )

        # ── TAB 2: Voice Templates ───────────────────────────────────────
        with gr.Tab("🗣 Шаблоны голосов"):
            gr.Markdown(
                "## Создание шаблона голоса\n"
                "Загрузите несколько записей одного персонажа. "
                "Шаблон объединит все в один эталон для максимально точного клонирования.\n\n"
                "Затем выберите шаблон на вкладке **Обработка** — не придётся каждый раз загружать эталоны."
            )
            with gr.Row():
                with gr.Column():
                    vt_name = gr.Textbox(
                        label="Имя шаблона",
                        placeholder="trevor_gta5",
                    )
                    vt_files = gr.File(
                        label="Аудиофайлы персонажа (3-10 записей, чистая речь)",
                        file_count="multiple",
                        file_types=["audio"],
                    )
                    vt_btn = gr.Button("💾 Создать шаблон", variant="primary")
                with gr.Column():
                    vt_status = gr.Textbox(label="Статус", interactive=False)
                    vt_list = gr.JSON(label="Доступные шаблоны", value=_list_voice_templates())

    # ── Event handlers ───────────────────────────────────────────────────

    def toggle_ollama_ui(method):
        show = method == "Ollama (LLM)"
        return (
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=show),
        )

    translation_method.change(
        toggle_ollama_ui,
        inputs=[translation_method],
        outputs=[ollama_model, ollama_url, ollama_status, check_btn],
    )

    check_btn.click(check_ollama, inputs=[ollama_url], outputs=[ollama_status])

    # main pipeline
    process_btn.click(
        process_batch,
        inputs=[
            audio_input,
            reference_input,
            voice_template,
            output_folder,
            whisper_model,
            source_language,
            translation_method,
            ollama_model,
            ollama_url,
            multi_speaker,
            hf_token,
        ],
        outputs=[log_output, result_files, details_output, download_zip, audio_player],
    )

    # voice template creation
    def on_create_template(name, files):
        status, templates = create_voice_template(name, files)
        return status, templates, gr.update(choices=get_voice_choices())

    vt_btn.click(
        on_create_template,
        inputs=[vt_name, vt_files],
        outputs=[vt_status, vt_list, voice_template],
    )


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860, theme=THEME, js=AUTO_SCROLL_JS)
