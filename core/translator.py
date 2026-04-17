"""Translation backends — Ollama (LLM) and NLLB-200 (Seq2Seq)."""

import gc
import logging
import re

import requests
import torch

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
#  Ollama (local LLM)
# ═════════════════════════════════════════════════════════════════════════

# Regex patterns for cleaning up model output
_RE_PARENTHETICAL = re.compile(r"\s*\((?:[A-Z]|Note|Translation|English|literally|Translated|This is)[^)]*\)\s*", re.IGNORECASE)
_RE_ENGLISH_LINE = re.compile(r"^[A-Za-z\s\'\",.:;!?-]{10,}$")
_RE_META_PREFIX = re.compile(r"^(Note:|Translation:|Here|However|Remember|I\'m|I am|My |Please|If you|Happy|The |This is|Output:)", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "Translate the following English text into natural spoken Russian. "
    "Rules:\n"
    "1. Output ONLY the Russian translation — one line, no quotes, no notes, no parentheses, no English.\n"
    "2. Translate profanity directly: fuck→ебать/блять, shit→дерьмо/говно, "
    "bitch→сука, ass→жопа, damn→чёрт, bastard→ублюдок, motherfucker→сукин сын.\n"
    "3. Use colloquial spoken Russian, not formal/literary.\n"
    "4. Never add explanations, commentary, or the original English."
)


def _clean_translation(raw: str, source: str) -> str:
    """Strip model garbage: English notes, parenthetical commentary, meta-text."""
    # Split into lines, process each
    lines = raw.strip().splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove parenthetical English notes
        line = _RE_PARENTHETICAL.sub("", line).strip()
        if not line:
            continue
        # Skip lines that are pure English (model echoed source or added notes)
        if _RE_ENGLISH_LINE.match(line):
            continue
        # Skip meta-commentary lines
        if _RE_META_PREFIX.match(line):
            continue
        # Skip if line is identical to source (model echoed input)
        if line.lower().strip('"\'') == source.lower().strip('"\''):
            continue
        cleaned.append(line)

    result = " ".join(cleaned).strip()

    # If cleaning removed everything, return raw first line as fallback
    if not result and lines:
        result = lines[0].strip()
        result = _RE_PARENTHETICAL.sub("", result).strip()

    return result


class OllamaTranslator:
    """Translate via a locally-running Ollama instance."""

    def __init__(self, model="gemma3:4b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    # ── helpers ──────────────────────────────────────────────────────────

    def check_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ── translation ──────────────────────────────────────────────────────

    def translate(self, text: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Translate: {text}"},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 512},
            },
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        result = _clean_translation(raw, text)
        logger.debug("Translate: %r → raw=%r → clean=%r", text, raw, result)
        return result

    def translate_segments(self, segments, progress_callback=None):
        translated = []
        for i, seg in enumerate(segments):
            if seg["text"]:
                try:
                    result = self.translate(seg["text"])
                except Exception as e:
                    logger.error("Ollama translation failed for segment %d: %s", i, e)
                    result = seg["text"]
            else:
                result = ""
            translated.append({**seg, "translated_text": result})
            if progress_callback:
                progress_callback(i + 1, len(segments))
        return translated


# ═════════════════════════════════════════════════════════════════════════
#  NLLB-200 (fully offline Seq2Seq)
# ═════════════════════════════════════════════════════════════════════════

class NLLBTranslator:
    """Offline translation using Meta NLLB-200."""

    def __init__(self, model_name="facebook/nllb-200-3.3B", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading NLLB model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            revision="refs/pr/17",
        ).to(self.device)
        logger.info("NLLB model loaded")

    def unload(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("NLLB model unloaded, VRAM freed")

    # ── translation ──────────────────────────────────────────────────────

    def translate(self, text: str, src_lang="eng_Latn", tgt_lang="rus_Cyrl") -> str:
        if self.model is None:
            self.load()

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=512,
                num_beams=5,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_segments(self, segments, progress_callback=None):
        if self.model is None:
            self.load()

        translated = []
        for i, seg in enumerate(segments):
            if seg["text"]:
                try:
                    result = self.translate(seg["text"])
                except Exception as e:
                    logger.error("NLLB translation failed for segment %d: %s", i, e)
                    result = seg["text"]
            else:
                result = ""
            translated.append({**seg, "translated_text": result})
            if progress_callback:
                progress_callback(i + 1, len(segments))
        return translated
