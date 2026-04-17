"""Translation backends — Ollama (LLM) and NLLB-200 (Seq2Seq)."""

import gc
import logging

import requests
import torch

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
#  Ollama (local LLM)
# ═════════════════════════════════════════════════════════════════════════

class OllamaTranslator:
    """Translate via a locally-running Ollama instance."""

    def __init__(self, model="qwen2.5:7b", base_url="http://localhost:11434"):
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
        prompt = (
            "Translate the following English text into natural, literary Russian. "
            "Preserve the tone, emotions, style and nuances of the original. "
            "Return ONLY the translation, nothing else.\n\n"
            f"{text}"
        )
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1024},
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

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
