"""Assemble synthesized audio segments into a single output file."""

import logging
import os

from pydub import AudioSegment

logger = logging.getLogger(__name__)


class Assembler:
    """Concatenate TTS segments preserving the original pause structure."""

    def assemble(self, segments: list[dict], output_path: str, background_path: str | None = None) -> str | None:
        combined = AudioSegment.empty()
        prev_end = 0.0

        for seg in segments:
            # silence gap between the previous segment and this one
            gap_ms = max(0, int((seg["start"] - prev_end) * 1000))
            if gap_ms > 50:
                combined += AudioSegment.silent(duration=gap_ms)

            if seg.get("audio_path") and os.path.exists(seg["audio_path"]):
                audio = AudioSegment.from_file(seg["audio_path"])
                combined += audio
                prev_end = seg["start"] + len(audio) / 1000.0
            else:
                # fill with silence matching original duration
                dur_ms = max(int((seg["end"] - seg["start"]) * 1000), 100)
                combined += AudioSegment.silent(duration=dur_ms)
                prev_end = seg["end"]

        if len(combined) == 0:
            logger.error("No audio segments to assemble")
            return None

        # light loudness normalization
        combined = combined.normalize()

        # Mix with background track if provided
        if background_path and os.path.exists(background_path):
            bg = AudioSegment.from_file(background_path)
            # Match lengths: pad or trim background
            if len(bg) < len(combined):
                bg += AudioSegment.silent(duration=len(combined) - len(bg))
            elif len(bg) > len(combined):
                bg = bg[:len(combined)]
            # Lower background volume so speech is clear
            bg = bg - 3  # reduce by 3 dB
            combined = combined.overlay(bg)
            logger.info("Mixed with background track: %s", background_path)

        combined.export(output_path, format="wav")
        logger.info("Assembled %.1f s → %s", len(combined) / 1000, output_path)
        return output_path
