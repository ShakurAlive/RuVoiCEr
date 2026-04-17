"""Text-to-Speech with voice cloning via CosyVoice 3.0 (FunAudioLLM).

Replaces XTTS-v2 for significantly better prosody and speaker similarity
in cross-lingual (EN → RU) voice cloning.
"""

import gc
import logging
import os
import sys
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)

# Add CosyVoice and its Matcha-TTS dependency to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COSYVOICE_DIR = _PROJECT_ROOT / "third_party" / "CosyVoice"
_MATCHA_DIR = _COSYVOICE_DIR / "third_party" / "Matcha-TTS"

for _p in [str(_COSYVOICE_DIR), str(_MATCHA_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Synthesizer:
    """CosyVoice 3.0 wrapper with VRAM management."""

    MAX_CHARS = 300  # CosyVoice handles longer inputs better than XTTS

    # CosyVoice3 (Qwen2-based) requires <|endofprompt|> token in cross-lingual text
    _PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"

    MODEL_DIR = str(_PROJECT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B")

    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.sample_rate = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def load(self):
        from cosyvoice.cli.cosyvoice import AutoModel

        logger.info("Loading CosyVoice3 model…")
        self.model = AutoModel(model_dir=self.MODEL_DIR)
        self.sample_rate = self.model.sample_rate
        logger.info("CosyVoice3 loaded (sample_rate=%d)", self.sample_rate)

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("CosyVoice3 unloaded, VRAM freed")

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _split_text(text: str, max_chars: int = 250) -> list[str]:
        """Split long text into chunks at sentence boundaries."""
        if len(text) <= max_chars:
            return [text]

        delimiters = [". ", "! ", "? ", "; ", ", "]
        chunks: list[str] = []
        remaining = text

        while len(remaining) > max_chars:
            # find the last delimiter within max_chars
            split_pos = -1
            for delim in delimiters:
                pos = remaining[:max_chars].rfind(delim)
                if pos > split_pos:
                    split_pos = pos + len(delim)

            if split_pos <= 0:
                # no delimiter found — hard split at max_chars
                split_pos = max_chars

            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    # ── synthesis ────────────────────────────────────────────────────────

    def synthesize(self, text: str, reference_audio: str | list[str], output_path: str, language="ru"):
        """Synthesize one piece of text using CosyVoice3 cross-lingual mode."""
        if self.model is None:
            self.load()

        # CosyVoice expects a single reference path
        ref_path = reference_audio if isinstance(reference_audio, str) else reference_audio[0]

        chunks = self._split_text(text, self.MAX_CHARS)

        all_audio = []
        for chunk in chunks:
            # CosyVoice3 requires <|endofprompt|> token in text
            tagged_chunk = self._PROMPT_PREFIX + chunk
            for result in self.model.inference_cross_lingual(
                tagged_chunk,
                ref_path,
                stream=False,
            ):
                all_audio.append(result["tts_speech"])

        if not all_audio:
            raise RuntimeError(f"CosyVoice produced no audio for: {text[:50]}")

        # Concatenate all chunks
        combined = torch.cat(all_audio, dim=-1)
        torchaudio.save(output_path, combined, self.sample_rate)

        return output_path

    def synthesize_segments(
        self,
        segments: list[dict],
        reference_audio: str | list[str],
        output_dir: str,
        language: str = "ru",
        progress_callback=None,
    ) -> list[dict]:
        """Synthesize every translated segment. Returns enriched segment dicts."""
        os.makedirs(output_dir, exist_ok=True)
        if self.model is None:
            self.load()

        results: list[dict] = []
        for i, seg in enumerate(segments):
            text = seg.get("translated_text", "").strip()
            if text:
                out_path = os.path.join(output_dir, f"seg_{i:04d}.wav")
                try:
                    self.synthesize(text, reference_audio, out_path, language)
                    results.append({**seg, "audio_path": out_path})
                except Exception as e:
                    logger.error("Synthesis failed for segment %d: %s", i, e)
                    results.append({**seg, "audio_path": None})
            else:
                results.append({**seg, "audio_path": None})

            if progress_callback:
                progress_callback(i + 1, len(segments))

        return results
