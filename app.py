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
import tempfile
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
from utils.audio import convert_to_wav, extract_reference_audio

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


# ═════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════════

def process_audio(
    audio_file,
    reference_file,
    whisper_model,
    source_language,
    translation_method,
    ollama_model,
    ollama_url,
    progress=gr.Progress(),
):
    if audio_file is None:
        raise gr.Error("Загрузите аудиофайл!")

    # ── prepare working directory ────────────────────────────────────────
    job_dir = Path(tempfile.mkdtemp(dir=OUTPUT_DIR))
    seg_dir = job_dir / "segments"
    seg_dir.mkdir()

    progress(0.02, desc="Подготовка аудио…")
    wav_path = convert_to_wav(audio_file, job_dir / "input.wav")

    # reference: use provided file or auto-extract first 15 s
    if reference_file:
        ref_path = convert_to_wav(reference_file, job_dir / "reference.wav")
    else:
        progress(0.05, desc="Извлечение эталонного голоса…")
        ref_path = extract_reference_audio(wav_path, job_dir / "reference.wav")

    # ── 1. Transcribe ────────────────────────────────────────────────────
    progress(0.08, desc="Загрузка Whisper…")
    transcriber = get_transcriber(whisper_model)

    progress(0.12, desc="Транскрибирование…")
    segments, info = transcriber.transcribe(str(wav_path), language=source_language)

    if not segments:
        raise gr.Error("Whisper не распознал речь в аудиофайле.")

    transcript_text = "\n".join(
        f"[{s['start']:.1f}–{s['end']:.1f}s]  {s['text']}" for s in segments
    )

    # ── 2. Translate ─────────────────────────────────────────────────────
    progress(0.30, desc="Перевод текста…")
    if translation_method == "Ollama (LLM)":
        translator = OllamaTranslator(model=ollama_model, base_url=ollama_url)
        if not translator.check_available():
            raise gr.Error(
                "Ollama недоступен! Запустите `ollama serve` или выберите NLLB-200."
            )
    else:
        translator = NLLBTranslator()
        translator.load()

    def on_translate(cur, total):
        progress(0.30 + cur / total * 0.25, desc=f"Перевод ({cur}/{total})…")

    translated = translator.translate_segments(segments, progress_callback=on_translate)

    if hasattr(translator, "unload"):
        translator.unload()

    translation_text = "\n".join(
        f"[{s['start']:.1f}s]  {s['translated_text']}" for s in translated
    )

    # ── 3. Synthesize ────────────────────────────────────────────────────
    progress(0.58, desc="Загрузка модели TTS…")
    synthesizer = Synthesizer()
    synthesizer.load()

    def on_synth(cur, total):
        progress(0.60 + cur / total * 0.32, desc=f"Синтез речи ({cur}/{total})…")

    synthesized = synthesizer.synthesize_segments(
        translated, str(ref_path), str(seg_dir), language="ru", progress_callback=on_synth
    )
    synthesizer.unload()

    # ── 4. Assemble ──────────────────────────────────────────────────────
    progress(0.95, desc="Сборка финального аудио…")
    assembler = Assembler()
    output_path = str(job_dir / "result.wav")
    assembler.assemble(synthesized, output_path)

    progress(1.0, desc="Готово!")
    return output_path, transcript_text, translation_text, "Обработка завершена успешно ✔"


# ═════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ═════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
)

with gr.Blocks(title="RuVoiCer") as app:
    gr.Markdown(
        "# RuVoiCer\n"
        "### Перевод аудио EN → RU с сохранением голоса и интонации\n"
        "Пайплайн: Whisper → LLM/NLLB перевод → XTTS-v2 голосовое клонирование"
    )

    # system info bar
    gr.Markdown(system_info())

    with gr.Row(equal_height=False):
        # ── left column: inputs & settings ───────────────────────────────
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Исходное аудио (английский)",
                type="filepath",
            )
            reference_input = gr.Audio(
                label="Эталон голоса (опционально, 5–30 с чистой речи)",
                type="filepath",
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
                    value="qwen2.5:7b",
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

            process_btn = gr.Button(
                "▶  Запустить обработку",
                variant="primary",
                size="lg",
            )

        # ── right column: outputs ────────────────────────────────────────
        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Статус", interactive=False)
            audio_output = gr.Audio(label="Результат", type="filepath")

            with gr.Accordion("Транскрипт и перевод", open=True):
                transcript_output = gr.Textbox(
                    label="Оригинальный текст (EN)",
                    lines=10,
                    interactive=False,
                )
                translation_output = gr.Textbox(
                    label="Перевод (RU)",
                    lines=10,
                    interactive=False,
                )

    # ── visibility toggles ───────────────────────────────────────────────

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

    # ── main pipeline trigger ────────────────────────────────────────────

    process_btn.click(
        process_audio,
        inputs=[
            audio_input,
            reference_input,
            whisper_model,
            source_language,
            translation_method,
            ollama_model,
            ollama_url,
        ],
        outputs=[audio_output, transcript_output, translation_output, status_output],
    )


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860, theme=THEME)
