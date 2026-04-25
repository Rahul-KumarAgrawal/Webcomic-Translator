"""
core/sarvam_translator.py
Sarvam AI translation backend for the standard pipeline.

Uses the Sarvam AI Translate API (api.sarvam.ai/translate) with
the mayura:v1 model for high-quality Indian-language translations.

Supports bidirectional translation between English and 10 Indian
languages: Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi,
Odia, Punjabi, Tamil, and Telugu.

Provides the same translate() interface as other engine backends:
    translate(source_text, source_lang) → (translated_text, confidence)

Usage in settings.yaml:
    translation_engine: sarvam
    sarvam_api_key: "sk_..."
"""

import json
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Language code mapping: NLLB-200 code → Sarvam language code ──────────────

_NLLB_TO_SARVAM = {
    "eng_Latn": "en-IN",
    "hin_Deva": "hi-IN",
    "ben_Beng": "bn-IN",
    "guj_Gujr": "gu-IN",
    "kan_Knda": "kn-IN",
    "mal_Mlym": "ml-IN",
    "mar_Deva": "mr-IN",
    "pan_Guru": "pa-IN",
    "tam_Taml": "ta-IN",
    "tel_Telu": "te-IN",
    # Odia — NLLB uses "ory_Orya"
    "ory_Orya": "od-IN",
}

_SARVAM_ENDPOINT = "https://api.sarvam.ai/translate"
_SARVAM_MODEL = "mayura:v1"


class SarvamTranslator:
    """
    Translates text using the Sarvam AI translation API.

    Best suited for English ↔ Indian language translation pairs.
    Instantiate once per batch for connection reuse.
    """

    _MIN_INTERVAL = 0.5   # seconds between consecutive API calls
    _MAX_RETRIES  = 3     # retry attempts on 429 before giving up

    def __init__(
        self,
        api_key: str,
        target_lang: str = "eng_Latn",
        source_lang_override: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        api_key             : Sarvam AI API key (sk_...).
        target_lang         : NLLB-200 code for the desired output language.
        source_lang_override: NLLB code to force as source (None = auto-detect).
        """
        if not api_key:
            raise ValueError(
                "Sarvam API key is required. "
                "Set it in Settings → Translation Engine → Sarvam API Key."
            )

        self._api_key = api_key
        self._target_lang_nllb = target_lang
        self._source_lang_override = source_lang_override

        # Resolve Sarvam language codes
        self._target_lang = _NLLB_TO_SARVAM.get(target_lang)
        if not self._target_lang:
            logger.warning(
                "Target language '%s' not natively supported by Sarvam. "
                "Falling back to English (en-IN).",
                target_lang,
            )
            self._target_lang = "en-IN"

        self._source_lang = (
            _NLLB_TO_SARVAM.get(source_lang_override)
            if source_lang_override
            else None
        )

        import importlib
        self._requests = importlib.import_module("requests")

        # Rate-limit state
        self._last_call_time = 0.0

        logger.info(
            "Sarvam Translator initialised (target=%s, source=%s, model=%s)",
            self._target_lang,
            self._source_lang or "auto",
            _SARVAM_MODEL,
        )

    def _wait_for_rate_limit(self):
        """Block until the minimum inter-request interval has elapsed."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._MIN_INTERVAL:
            sleep_for = self._MIN_INTERVAL - elapsed
            logger.debug("Sarvam rate-limit: sleeping %.2fs", sleep_for)
            time.sleep(sleep_for)
        self._last_call_time = time.monotonic()

    def translate(self, text: str, source_lang: str) -> Tuple[str, float]:
        """
        Translate *text* using Sarvam AI and return (translated_text, confidence).

        Automatically rate-limits and retries on 429 with exponential backoff.
        """
        if not text or not text.strip():
            return "", 100.0

        # Resolve source language
        src_code = self._source_lang or _NLLB_TO_SARVAM.get(source_lang)
        if not src_code:
            # Sarvam supports auto-detect
            src_code = "auto"

        payload = {
            "input": text,
            "source_language_code": src_code,
            "target_language_code": self._target_lang,
            "model": _SARVAM_MODEL,
            "mode": "formal",
        }

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self._api_key,
        }

        backoff = 2.0  # initial backoff on 429

        for attempt in range(1, self._MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                resp = self._requests.post(
                    _SARVAM_ENDPOINT,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=30,
                )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else backoff
                    logger.warning(
                        "Sarvam 429 rate limit (attempt %d/%d). "
                        "Waiting %.1fs before retry...",
                        attempt, self._MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)
                    self._last_call_time = 0.0
                    continue

                resp.raise_for_status()

                data = resp.json()
                translated = data.get("translated_text", "").strip()
                if not translated:
                    logger.error("Sarvam: empty response body: %s", data)
                    return "[Translation Error]", 0.0

                logger.debug("Sarvam: %r → %r", text[:40], translated[:40])
                return translated, 90.0

            except Exception as exc:
                logger.error("Sarvam translation error: %s", exc)
                return "[Translation Error]", 0.0

        logger.error(
            "Sarvam: max retries (%d) exceeded for text: %r",
            self._MAX_RETRIES, text[:40],
        )
        return "[Translation Error]", 0.0
