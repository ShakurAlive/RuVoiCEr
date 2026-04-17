"""RuVoiCer — configuration defaults."""

import os

# Accept Coqui XTTS license automatically
os.environ["COQUI_TOS_AGREED"] = "1"

# ── Patch torchaudio 2.11+ ──────────────────────────────────────────────
# torchaudio 2.11 hardcodes torchcodec as the only backend.
# torchcodec requires FFmpeg shared DLLs which are not available on Windows.
# Replace torchaudio.load / torchaudio.save with soundfile-based versions.
def _patch_torchaudio():
    try:
        import torchaudio
        import soundfile as sf
        import numpy as np
        import torch
    except ImportError:
        return

    # Only patch if the current load() calls torchcodec
    if not hasattr(torchaudio, "load_with_torchcodec"):
        return

    def _soundfile_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                        channels_first=True, format=None, buffer_size=4096,
                        backend=None):
        data, sr = sf.read(str(uri), start=frame_offset,
                           stop=None if num_frames == -1 else frame_offset + num_frames,
                           dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T if channels_first else data)
        return tensor, sr

    def _soundfile_save(uri, src, sample_rate, channels_first=True,
                        format=None, encoding=None, bits_per_sample=None,
                        buffer_size=4096, backend=None, compression=None):
        if isinstance(src, torch.Tensor):
            arr = src.cpu().numpy()
        else:
            arr = np.asarray(src)
        if channels_first and arr.ndim == 2:
            arr = arr.T
        sf.write(str(uri), arr, sample_rate)

    torchaudio.load = _soundfile_load
    torchaudio.save = _soundfile_save

_patch_torchaudio()

# ── Paths ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"

# ── Whisper (ASR) ────────────────────────────────────────────────────────
WHISPER_MODEL = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# ── Translation ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"
NLLB_MODEL = "facebook/nllb-200-3.3B"

# ── TTS (voice cloning) ─────────────────────────────────────────────────
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# ── Reference audio ─────────────────────────────────────────────────────
REFERENCE_DURATION_SEC = 15
REFERENCE_SAMPLE_RATE = 22050
