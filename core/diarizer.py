"""Speaker diarization using pyannote-audio."""

import gc
import logging
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


class Diarizer:
    """Identify who speaks when using pyannote speaker-diarization-3.1."""

    def __init__(self, hf_token: str, device: str = "cuda"):
        self.hf_token = hf_token
        self.device = device
        self.pipeline = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def load(self):
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote speaker-diarization-3.1 …")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.hf_token,
        )
        self.pipeline.to(torch.device(self.device))
        logger.info("Diarization pipeline loaded on %s", self.device)

    def unload(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Diarization pipeline unloaded")

    # ── main API ─────────────────────────────────────────────────────────

    def diarize(self, audio_path: str) -> list[dict]:
        """Run diarization and return speaker-labeled segments.

        Returns list of dicts:
            {"start": float, "end": float, "speaker": str}
        sorted by start time.
        """
        if self.pipeline is None:
            self.load()

        logger.info("Diarizing: %s", audio_path)
        waveform, sample_rate = torchaudio.load(audio_path)
        result = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # pyannote ≥3.1 returns DiarizeOutput dataclass; extract Annotation
        annotation = getattr(result, "speaker_diarization", result)

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        speakers = {s["speaker"] for s in segments}
        logger.info(
            "Diarization: %d turns, %d speakers detected",
            len(segments), len(speakers),
        )
        return segments

    # ── helper: assign speaker labels to transcription segments ──────────

    @staticmethod
    def assign_speakers(
        transcription_segments: list[dict],
        diarization_segments: list[dict],
    ) -> list[dict]:
        """Assign a speaker label to each transcription segment based on
        maximum overlap with diarization turns.

        Each transcription segment gets a new key ``"speaker"``.
        """
        result = []
        for tseg in transcription_segments:
            t_start, t_end = tseg["start"], tseg["end"]
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0

            for dseg in diarization_segments:
                overlap_start = max(t_start, dseg["start"])
                overlap_end = min(t_end, dseg["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = dseg["speaker"]

            result.append({**tseg, "speaker": best_speaker})
        return result

    @staticmethod
    def extract_speaker_audio(
        audio_path: str,
        diarization_segments: list[dict],
        speaker: str,
        output_path: str,
        max_duration_sec: float = 24.0,
    ) -> str | None:
        """Extract the longest contiguous speech of a given speaker for
        voice-cloning reference. Returns output path or None."""
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        speaker_turns = [s for s in diarization_segments if s["speaker"] == speaker]

        if not speaker_turns:
            return None

        # pick the longest turn
        speaker_turns.sort(key=lambda s: s["end"] - s["start"], reverse=True)
        best = speaker_turns[0]

        start_ms = int(best["start"] * 1000)
        end_ms = int(best["end"] * 1000)
        # cap duration
        max_ms = int(max_duration_sec * 1000)
        if end_ms - start_ms > max_ms:
            end_ms = start_ms + max_ms

        clip = audio[start_ms:end_ms]
        clip = clip.set_frame_rate(22050).set_channels(1).normalize()
        clip.export(output_path, format="wav")

        logger.info(
            "Extracted %.1f s reference for %s → %s",
            len(clip) / 1000, speaker, Path(output_path).name,
        )
        return output_path
