"""Vocal/background separation using Demucs (Meta)."""

import gc
import logging
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


class VocalSeparator:
    """Separate audio into vocals and background using Demucs htdemucs model."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load(self):
        from demucs.pretrained import get_model

        logger.info("Loading Demucs htdemucs model…")
        self.model = get_model("htdemucs")
        self.model.to(torch.device(self.device))
        self.model.eval()
        logger.info("Demucs loaded on %s", self.device)

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Demucs unloaded")

    def separate(self, audio_path: str, output_dir: str) -> dict[str, str]:
        """Separate audio into vocals and background.

        Returns dict with keys 'vocals' and 'background', values are file paths.
        """
        if self.model is None:
            self.load()

        from demucs.apply import apply_model

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load audio at model's sample rate
        waveform, sr = torchaudio.load(audio_path)

        # Demucs expects the model's native sample rate (44100)
        model_sr = self.model.samplerate
        if sr != model_sr:
            waveform = torchaudio.functional.resample(waveform, sr, model_sr)

        # Demucs expects (batch, channels, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)  # mono → stereo for demucs
        waveform = waveform.unsqueeze(0)  # add batch dim

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(
                self.model,
                waveform.to(self.device),
                shifts=1,
                overlap=0.25,
            )

        # Undo normalization
        sources = sources * ref.std() + ref.mean()
        sources = sources.cpu()

        # htdemucs sources order: drums, bass, other, vocals
        source_names = self.model.sources  # ['drums', 'bass', 'other', 'vocals']
        vocals_idx = source_names.index("vocals")

        vocals = sources[0, vocals_idx]  # (channels, time)
        # Background = everything except vocals
        bg_indices = [i for i in range(len(source_names)) if i != vocals_idx]
        background = sum(sources[0, i] for i in bg_indices)

        # Save as WAV
        vocals_path = str(output_dir / "vocals.wav")
        bg_path = str(output_dir / "background.wav")

        torchaudio.save(vocals_path, vocals, model_sr)
        torchaudio.save(bg_path, background, model_sr)

        logger.info(
            "Separated: vocals → %s, background → %s",
            vocals_path, bg_path,
        )
        return {"vocals": vocals_path, "background": bg_path}
