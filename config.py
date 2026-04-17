"""RuVoiCer — configuration defaults."""

import os

# Accept Coqui XTTS license automatically
os.environ["COQUI_TOS_AGREED"] = "1"

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
