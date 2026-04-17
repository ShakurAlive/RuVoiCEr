"""Speech recognition using faster-whisper."""

import gc
import logging

import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    """Whisper-based ASR with explicit VRAM management."""

    def __init__(self, model_size="large-v3-turbo", device="cuda", compute_type="float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def load(self):
        logger.info("Loading Whisper model: %s", self.model_size)
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=1,
            num_workers=1,
        )
        logger.info("Whisper model loaded")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper model unloaded, VRAM freed")

    # ── main API ─────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str, language: str = "en"):
        """Return (segments_list, info).

        Each segment: {"start": float, "end": float, "text": str}
        """
        if self.model is None:
            self.load()

        logger.info("Transcribing: %s", audio_path)
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = []
        for seg in segments_gen:
            text = seg.text.strip()
            if text:
                segments.append(
                    {"start": seg.start, "end": seg.end, "text": text}
                )

        logger.info(
            "Transcribed %d segments  |  detected lang: %s (%.0f%%)",
            len(segments),
            info.language,
            info.language_probability * 100,
        )
        return segments, info
