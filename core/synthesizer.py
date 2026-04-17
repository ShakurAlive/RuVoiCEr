"""Text-to-Speech with voice cloning via Coqui XTTS-v2."""

import gc
import logging
import os

import torch

logger = logging.getLogger(__name__)

# Accept Coqui TTS license
os.environ["COQUI_TOS_AGREED"] = "1"


class Synthesizer:
    """XTTS-v2 wrapper with VRAM management and long-text splitting."""

    MAX_CHARS = 250  # XTTS produces best quality on shorter inputs

    def __init__(
        self,
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        device="cuda",
    ):
        self.model_name = model_name
        self.device = device
        self.tts = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def load(self):
        # TTS 0.22.0 uses torch.load without weights_only=False,
        # but PyTorch >=2.6 defaults to weights_only=True.
        # Monkey-patch to restore old behaviour for trusted Coqui checkpoints.
        import functools
        _orig_load = torch.load
        @functools.wraps(_orig_load)
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load

        # torchaudio patch is applied globally in config.py

        from TTS.api import TTS

        logger.info("Loading TTS model: %s", self.model_name)
        self.tts = TTS(self.model_name).to(self.device)
        logger.info("TTS model loaded")

        # Restore original torch.load
        torch.load = _orig_load

    def unload(self):
        if self.tts is not None:
            del self.tts
            self.tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("TTS model unloaded, VRAM freed")

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
        """Synthesize one piece of text; handles automatic chunk splitting."""
        if self.tts is None:
            self.load()

        chunks = self._split_text(text, self.MAX_CHARS)

        if len(chunks) == 1:
            self.tts.tts_to_file(
                text=chunks[0],
                speaker_wav=reference_audio,
                language=language,
                file_path=output_path,
            )
        else:
            from pydub import AudioSegment

            combined = AudioSegment.empty()
            for idx, chunk in enumerate(chunks):
                chunk_path = output_path.replace(".wav", f"_c{idx}.wav")
                self.tts.tts_to_file(
                    text=chunk,
                    speaker_wav=reference_audio,
                    language=language,
                    file_path=chunk_path,
                )
                combined += AudioSegment.from_file(chunk_path)
                os.remove(chunk_path)
            combined.export(output_path, format="wav")

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
        if self.tts is None:
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
