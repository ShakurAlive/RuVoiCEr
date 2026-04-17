"""Audio helper utilities (format conversion, reference extraction)."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_to_wav(input_path: str | Path, output_path: str | Path, sample_rate: int = 22050) -> Path:
    """Convert any audio file to mono WAV via FFmpeg (pydub fallback)."""
    input_path, output_path = str(input_path), str(output_path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-i", input_path,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return Path(output_path)
        logger.warning("FFmpeg failed, falling back to pydub: %s", result.stderr[:200])

    from pydub import AudioSegment
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    audio.export(output_path, format="wav")
    return Path(output_path)


def extract_reference_audio(
    input_path: str | Path,
    output_path: str | Path,
    start_sec: float = 0,
    duration_sec: float = 15,
) -> Path:
    """Extract a short segment of clean speech for voice-cloning reference."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(input_path))
    start_ms = int(start_sec * 1000)
    end_ms = min(int((start_sec + duration_sec) * 1000), len(audio))

    segment = audio[start_ms:end_ms]
    segment = segment.set_frame_rate(22050).set_channels(1)
    segment.export(str(output_path), format="wav")

    logger.info("Extracted reference audio: %.1f s", len(segment) / 1000)
    return Path(output_path)
