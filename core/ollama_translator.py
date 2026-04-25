"""
core/ollama_translator.py
Ollama local LLM translation backend for the standard pipeline.

Uses the Ollama REST API (http://localhost:11434) to translate text
using any locally installed model (llama3, qwen2.5, mistral, etc.).

No API key required — Ollama must be running locally.

Provides the same translate() interface as other engine backends:
    translate(source_text, source_lang) → (translated_text, confidence)

Usage in settings.yaml:
    translation_engine: ollama
    ollama_model: "llama3"           # or "qwen2.5-coder:3b", etc.
    ollama_url: "http://localhost:11434"  # optional, default
"""

import json
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Language name mapping: NLLB-200 code → human-readable for prompt ─────────

_NLLB_TO_LANG_NAME = {
    "jpn_Jpan": "Japanese",
    "zho_Hans": "Simplified Chinese",
    "zho_Hant": "Traditional Chinese",
    "kor_Hang": "Korean",
    "eng_Latn": "English",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "spa_Latn": "Spanish",
    "por_Latn": "Portuguese",
    "ita_Latn": "Italian",
    "rus_Cyrl": "Russian",
    "ara_Arab": "Arabic",
    "hin_Deva": "Hindi",
    "tha_Thai": "Thai",
    "vie_Latn": "Vietnamese",
    "ind_Latn": "Indonesian",
    "tur_Latn": "Turkish",
    "pol_Latn": "Polish",
    "nld_Latn": "Dutch",
    "ukr_Cyrl": "Ukrainian",
    "ben_Beng": "Bengali",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "guj_Gujr": "Gujarati",
    "pan_Guru": "Punjabi",
    "urd_Arab": "Urdu",
}


class OllamaTranslator:
    """
    Translates text using a locally running Ollama instance.

    Supports any model installed in Ollama (llama3, qwen2.5, mistral, etc.).
    No API key required — just needs Ollama running on localhost.
    """

    _MAX_RETRIES = 3

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        target_lang: str = "eng_Latn",
        source_lang_override: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model               : Ollama model name (e.g. "llama3", "qwen2.5-coder:3b").
        base_url            : Ollama API base URL (default: http://localhost:11434).
        target_lang         : NLLB-200 code for the desired output language.
        source_lang_override: NLLB code to force as source (None = auto-detect).
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._target_lang_nllb = target_lang
        self._source_lang_override = source_lang_override

        # Resolve human-readable language names for the prompt
        self._target_lang_name = _NLLB_TO_LANG_NAME.get(target_lang, "English")
        self._source_lang_name = (
            _NLLB_TO_LANG_NAME.get(source_lang_override)
            if source_lang_override
            else None
        )

        import importlib
        self._requests = importlib.import_module("requests")

        logger.info(
            "Ollama Translator initialised (model=%s, target=%s, source=%s, url=%s)",
            self._model,
            self._target_lang_name,
            self._source_lang_name or "auto-detect",
            self._base_url,
        )

    def translate(self, text: str, source_lang: str) -> Tuple[str, float]:
        """
        Translate *text* using the local Ollama model.
        Returns (translated_text, confidence).
        """
        if not text or not text.strip():
            return "", 100.0

        # Build source description for the prompt
        src_name = self._source_lang_name or _NLLB_TO_LANG_NAME.get(
            source_lang, "the source language"
        )

        prompt = (
            f"Translate the following {src_name} manga/comic dialogue text to "
            f"{self._target_lang_name}. "
            "Return ONLY the translated text with no explanation, notes, quotes, "
            "or extra formatting:\n\n"
            f"{text}"
        )

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = self._requests.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 512,
                        },
                    },
                    timeout=120,  # local models can be slow
                )

                if resp.status_code == 404:
                    logger.error(
                        "Ollama model '%s' not found. "
                        "Run: ollama pull %s",
                        self._model, self._model,
                    )
                    return "[Model Not Found]", 0.0

                resp.raise_for_status()

                data = resp.json()
                translated = data.get("response", "").strip()

                # Clean up common LLM artifacts
                # Remove wrapping quotes if present
                if (translated.startswith('"') and translated.endswith('"')) or \
                   (translated.startswith("'") and translated.endswith("'")):
                    translated = translated[1:-1].strip()

                if not translated:
                    logger.error("Ollama: empty response for: %r", text[:40])
                    return "[Translation Error]", 0.0

                logger.debug("Ollama (%s): %r → %r", self._model, text[:40], translated[:40])
                return translated, 85.0

            except self._requests.exceptions.ConnectionError:
                logger.error(
                    "Cannot connect to Ollama at %s. "
                    "Make sure Ollama is running (ollama serve).",
                    self._base_url,
                )
                return "[Ollama Not Running]", 0.0

            except Exception as exc:
                logger.warning(
                    "Ollama translation error (attempt %d/%d): %s",
                    attempt, self._MAX_RETRIES, exc,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(1.0 * attempt)
                    continue
                logger.error("Ollama: all retries exhausted for: %r", text[:40])
                return "[Translation Error]", 0.0

        return "[Translation Error]", 0.0

    def list_models(self) -> list:
        """List all models available in the local Ollama instance."""
        try:
            resp = self._requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return []
