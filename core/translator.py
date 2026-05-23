"""
core/translator.py
Translation pipeline: memory lookup → NLLB-200 inference → save result.

For every speech bubble text this module:
  1. Checks series translation memory (approved=True)
  2. Checks global translation memory (approved=True)
  3. Falls back to NLLB-200 model inference
  4. Saves result to series memory as unapproved
  5. Returns bubble data for the review page
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Ensure project root is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.language_detector import detect_language, LANG_EN
from memory.override_checker import OverrideChecker


@dataclass
class BubbleResult:
    """Holds translation result for a single speech bubble."""
    source_text: str
    translated_text: str
    source_lang: str
    confidence: float          # 0.0 – 100.0
    source: str                # "series_memory" | "global_memory" | "model"
    approved: bool = False
    edited: bool = False
    memory_id: Optional[int] = None   # DB row id for update


@dataclass
class PageResult:
    """All bubble results for a single comic page."""
    page_path: str             # path to the translated page image
    bubbles: List[BubbleResult] = field(default_factory=list)


class Translator:
    """
    Stateful translator that holds the model in GPU memory for efficiency.
    Instantiate once per batch; call .translate_text() per bubble.
    """

    def __init__(self, series: str, cfg: dict):
        """
        Parameters
        ----------
        series : Series name (used for memory lookup and saving).
        cfg    : Loaded settings.yaml dict.
                 Optional keys:
                   target_lang          – NLLB code for output language (default eng_Latn)
                   source_lang_override – NLLB code to force as source (skip auto-detect)
                   translation_engine   – "nllb" (default), "deepl", "google", "baidu", "sarvam", "papago", or "ollama"
                   deepl_api_key        – DeepL API auth key (required when engine=deepl)
                   google_api_key       – Gemini API key (required when engine=google)
                   baidu_app_id         – Baidu APP ID (required when engine=baidu)
                   baidu_secret_key     – Baidu Secret Key (required when engine=baidu)
                   sarvam_api_key       – Sarvam AI API key (required when engine=sarvam)
                   ollama_model         – Ollama model name (required when engine=ollama)
                   ollama_url           – Ollama API URL (optional, default localhost:11434)
        """
        self.series = series
        self.cfg = cfg

        self._target_lang = cfg.get("target_lang", LANG_EN)
        self._source_lang_override = cfg.get("source_lang_override", None)
        self._engine = cfg.get("translation_engine", "nllb").lower()

        self._checker = OverrideChecker()
        self._model = None
        self._tokenizer = None
        self._deepl = None
        self._google = None
        self._baidu = None
        self._sarvam = None
        self._papago = None
        self._ollama = None
        self._groq   = None
        self._device = cfg.get("gpu_device", "cuda")

    # ── Public API ────────────────────────────────────────────────────────────

    def translate_text(self, source_text: str) -> BubbleResult:
        """
        Translate a single bubble's text through the full lookup chain.
        """
        source_text = source_text.strip()
        if not source_text:
            return BubbleResult(
                source_text="", translated_text="",
                source_lang="", confidence=100.0,
                source="empty",
            )

        source_lang = self._source_lang_override or detect_language(source_text)

        # Step 1 & 2: Memory lookup (series → global) — skip for cloud API engines
        _cloud_engines = {"deepl", "google", "gemini", "baidu", "sarvam", "papago", "ollama", "groq_llama"}
        if self._engine not in _cloud_engines:
            memory_result = self._checker.lookup(source_text, self.series)
            if memory_result:
                translated, conf, src, mem_id = memory_result
                # Increment usage counter
                self._checker.memory_manager.increment_usage(mem_id, src == "series_memory", series=self.series)
                return BubbleResult(
                    source_text=source_text,
                    translated_text=translated,
                    source_lang=source_lang,
                    confidence=100.0,   # memory hit = perfect confidence
                    source=src,
                    approved=True,
                    memory_id=mem_id,
                )

        # Step 3: Translation inference (NLLB, DeepL, Google, Baidu, or Sarvam)
        if self._engine == "manual":
            # Skip translation entirely, just dump the OCR text into Review UI
            translated, confidence = "", 0.0
            engine_source = "manual"
        elif self._engine == "deepl":
            self._ensure_deepl_loaded()
            translated, confidence = self._deepl.translate(source_text, source_lang)
            engine_source = "deepl"
        elif self._engine in ("google", "gemini"):
            self._ensure_google_loaded()
            translated, confidence = self._google.translate(source_text, source_lang)
            engine_source = "google"
        elif self._engine == "baidu":
            self._ensure_baidu_loaded()
            translated, confidence = self._baidu.translate(source_text, source_lang)
            engine_source = "baidu"
        elif self._engine == "sarvam":
            self._ensure_sarvam_loaded()
            translated, confidence = self._sarvam.translate(source_text, source_lang)
            engine_source = "sarvam"
        elif self._engine == "papago":
            self._ensure_papago_loaded()
            translated, confidence = self._papago.translate(source_text, source_lang)
            engine_source = "papago"
        elif self._engine == "ollama":
            self._ensure_ollama_loaded()
            translated, confidence = self._ollama.translate(source_text, source_lang)
            engine_source = "ollama"
        elif self._engine == "groq_llama":
            self._ensure_groq_loaded()
            translated, confidence = self._groq.translate(source_text, source_lang)
            engine_source = "groq_llama"
        else:
            self._ensure_model_loaded()
            translated, confidence = self._run_inference(source_text, source_lang)
            engine_source = "model"

        # Step 4: Save to series memory as unapproved
        mem_id = self._checker.memory_manager.save(
            source_lang=source_lang,
            source_text=source_text,
            translated_text=translated,
            approved=False,
            edited=False,
            confidence_score=confidence,
            series=self.series,
        )

        return BubbleResult(
            source_text=source_text,
            translated_text=translated,
            source_lang=source_lang,
            confidence=confidence,
            source=engine_source,
            approved=False,
            memory_id=mem_id,
        )

    def translate_batch(self, texts: List[str]) -> List[BubbleResult]:
        """
        Translate a list of strings in a single batch (optimised for Gemini/Cloud).
        """
        if not texts:
            return []

        source_lang = self._source_lang_override or detect_language("\n".join(texts[:5])) # detect from first few

        # Step 1: Check if engine supports native batching (Gemini or Groq)
        if self._engine in ("google", "gemini"):
            self._ensure_google_loaded()
            raw_results = self._google.translate_batch(texts, source_lang)
            engine_source = "google"

            results = []
            for i, (translated, confidence) in enumerate(raw_results):
                mem_id = self._checker.memory_manager.save(
                    source_lang=source_lang,
                    source_text=texts[i],
                    translated_text=translated,
                    approved=False,
                    edited=False,
                    confidence_score=confidence,
                    series=self.series,
                )
                results.append(BubbleResult(
                    source_text=texts[i],
                    translated_text=translated,
                    source_lang=source_lang,
                    confidence=confidence,
                    source=engine_source,
                    approved=False,
                    memory_id=mem_id,
                ))
            return results

        if self._engine == "groq_llama":
            self._ensure_groq_loaded()
            raw_results = self._groq.translate_batch(texts, source_lang)

            results = []
            for i, (translated, confidence) in enumerate(raw_results):
                mem_id = self._checker.memory_manager.save(
                    source_lang=source_lang,
                    source_text=texts[i],
                    translated_text=translated,
                    approved=False,
                    edited=False,
                    confidence_score=confidence,
                    series=self.series,
                )
                results.append(BubbleResult(
                    source_text=texts[i],
                    translated_text=translated,
                    source_lang=source_lang,
                    confidence=confidence,
                    source="groq_llama",
                    approved=False,
                    memory_id=mem_id,
                ))
            return results

        # Fallback: one-by-one for other engines
        return [self.translate_text(t) for t in texts]

    def unload_model(self):
        """Free GPU memory when done with a batch."""
        if self._model is not None:
            try:
                import torch
                del self._model
                del self._tokenizer
                self._model = None
                self._tokenizer = None
                torch.cuda.empty_cache()
                logger.info("Model unloaded, VRAM freed.")
            except Exception as exc:
                logger.warning("Error unloading model: %s", exc)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_deepl_loaded(self):
        if self._deepl is None:
            from core.deepl_translator import DeepLTranslator
            api_key = self.cfg.get("deepl_api_key", "")
            self._deepl = DeepLTranslator(
                api_key=api_key,
                target_lang=self._target_lang,
                source_lang_override=self._source_lang_override,
            )
            logger.info("DeepL translator loaded.")

    def _ensure_google_loaded(self):
        if self._google is None:
            from core.google_translator import GoogleTranslator
            api_key = self.cfg.get("google_api_key", "")
            self._google = GoogleTranslator(
                api_key=api_key,
                target_lang=self._target_lang,
                source_lang_override=self._source_lang_override,
                system_prompt=self.cfg.get("google_system_prompt"),
            )
            logger.info("Google/Gemini translator loaded.")

    def _ensure_baidu_loaded(self):
        if self._baidu is None:
            from core.baidu_translator import BaiduTranslator
            self._baidu = BaiduTranslator(
                app_id=self.cfg.get("baidu_app_id", ""),
                secret_key=self.cfg.get("baidu_secret_key", ""),
                target_lang=self._target_lang,
                source_lang_override=self._source_lang_override,
            )
            logger.info("Baidu translator loaded.")

    def _ensure_sarvam_loaded(self):
        if self._sarvam is None:
            from core.sarvam_translator import SarvamTranslator
            self._sarvam = SarvamTranslator(
                api_key=self.cfg.get("sarvam_api_key", ""),
                target_lang=self._target_lang,
                source_lang_override=self._source_lang_override,
            )
            logger.info("Sarvam translator loaded.")

    def _ensure_papago_loaded(self):
        if self._papago is None:
            from core.papago_translator import PapagoTranslator
            self._papago = PapagoTranslator(
                target_lang=self._target_lang,
                source_lang_override=self._source_lang_override,
            )
            logger.info("Papago translator loaded.")

    def _ensure_ollama_loaded(self):
        if self._ollama is None:
            from core.ollama_translator import OllamaTranslator
            self._ollama = OllamaTranslator(
                model=self.cfg.get("ollama_model", "llama3"),
                base_url=self.cfg.get("ollama_url", "http://localhost:11434"),
                target_lang=self._target_lang,
                source_lang_override=self._source_lang_override,
            )
            logger.info("[VRAM] Loading Ollama translator (model=%s)...", self.cfg.get("ollama_model", "llama3"))
            logger.info("Ollama translator loaded.")

    def _ensure_groq_loaded(self):
        if self._groq is None:
            from core.groq_translator import GroqTranslator
            api_key = self.cfg.get("groq_api_key", "")
            if not api_key:
                raise ValueError(
                    "Groq API key is missing. Please add it in Settings → Groq API Key."
                )
            self._groq = GroqTranslator(
                api_key=api_key,
                target_lang=self._target_lang,
                source_lang_override=self._source_lang_override,
            )
            logger.info("[Groq] Groq Llama 3.3 70B translator loaded.")

    def _ensure_model_loaded(self):
        if self._model is None:
            from model.model_loader import ModelLoader
            loader = ModelLoader(self.cfg)
            logger.info("[VRAM] Loading NLLB-200 Translation model onto %s...", self._device)
            self._model, self._tokenizer = loader.load()
            logger.info("Translation model loaded.")

    def unload(self):
        """Unload the translation model from VRAM."""
        if self._model is not None:
            logger.info("[VRAM] Unloading NLLB-200 Translation model from VRAM...")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("[VRAM] VRAM Cache cleared after translation.")
            except ImportError:
                pass

    def _run_inference(self, text: str, source_lang: str) -> Tuple[str, float]:
        """
        Run NLLB-200 translation and compute confidence score.

        Confidence is derived from per-token softmax probability:
          confidence = mean(max_probs) ** 0.5  scaled to 0–100.
        Higher entropy (uncertain tokens) → lower confidence.
        """
        import torch

        src_lang = source_lang
        tgt_lang = self._target_lang

        try:
            # NLLB tokenizer requires src_lang set as an attribute, not a kwarg
            self._tokenizer.src_lang = src_lang

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(self._device)

            # Resolve forced BOS token — use the stable convert_tokens_to_ids API
            # (lang_code_to_id is not present on all NLLB tokenizer versions)
            forced_bos = self._tokenizer.convert_tokens_to_ids(tgt_lang)
            if forced_bos == self._tokenizer.unk_token_id:
                # Fallback: try the special-token mapping directly
                sp_map = getattr(self._tokenizer, "added_tokens_encoder", {})
                forced_bos = sp_map.get(tgt_lang, forced_bos)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=256,
                    num_beams=4,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Decode
            generated_ids = outputs.sequences[0]
            translated = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Confidence: mean of per-token max softmax probs → 0-100
            scores = outputs.scores  # tuple of (vocab_size,) tensors
            if scores:
                import torch.nn.functional as F
                max_probs = [
                    F.softmax(s[0], dim=-1).max().item()
                    for s in scores
                ]
                mean_prob = sum(max_probs) / len(max_probs)
                # Apply square-root to stretch low-confidence range
                confidence = round(math.sqrt(mean_prob) * 100, 1)
            else:
                confidence = 75.0  # default if scores unavailable

            return translated, confidence

        except Exception as exc:
            import traceback
            logger.error("Inference error: %s\n%s", exc, traceback.format_exc())
            return "[Translation Error]", 0.0
