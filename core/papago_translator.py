"""
core/papago_translator.py
Naver Papago translation backend using the PentaGo library.

Uses the unofficial Papago API via the PentaGo package — no API key required.
Supports 16+ languages including Japanese, Chinese (Simplified/Traditional),
Korean, English, Hindi, and more.

Provides the same translate() interface as other engine backends:
    translate(source_text, source_lang) → (translated_text, confidence)

Usage in settings.yaml:
    translation_engine: papago
"""

import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Language code mapping: NLLB-200 code → PentaGo/Papago code ───────────────

_NLLB_TO_PAPAGO = {
    "jpn_Jpan": "ja",
    "zho_Hans": "zh-CN",
    "zho_Hant": "zh-TW",
    "kor_Hang": "ko",
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "spa_Latn": "es",
    "por_Latn": "pt",
    "ita_Latn": "it",
    "rus_Cyrl": "ru",
    "ara_Arab": "ar",
    "hin_Deva": "hi",
    "tha_Thai": "th",
    "vie_Latn": "vi",
    "ind_Latn": "id",
}


class PapagoTranslator:
    """
    Translates text using Naver Papago via the PentaGo library.

    No API key required. Supports auto language detection.
    Uses synchronous translate_sync() to fit the existing pipeline.
    Instantiate once per batch.
    """

    _MIN_INTERVAL = 0.3   # seconds between calls to be respectful
    _MAX_RETRIES  = 3

    def __init__(
        self,
        target_lang: str = "eng_Latn",
        source_lang_override: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        target_lang         : NLLB-200 code for the desired output language.
        source_lang_override: NLLB code to force as source (None = auto-detect).
        """
        from pentago import Pentago
        from pentago import lang as plang

        self._Pentago = Pentago
        self._plang = plang

        # Resolve Papago target language code
        self._target_lang_papago = _NLLB_TO_PAPAGO.get(target_lang, "en")
        self._target_lang_nllb = target_lang

        # Resolve source language
        self._source_lang_override = source_lang_override
        self._source_lang_papago = (
            _NLLB_TO_PAPAGO.get(source_lang_override)
            if source_lang_override
            else None
        )

        # Rate-limit state
        self._last_call_time = 0.0

        logger.info(
            "Papago Translator initialised (target=%s, source=%s)",
            self._target_lang_papago,
            self._source_lang_papago or "auto",
        )

    def _wait_for_rate_limit(self):
        """Block until the minimum inter-request interval has elapsed."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._MIN_INTERVAL:
            sleep_for = self._MIN_INTERVAL - elapsed
            logger.debug("Papago rate-limit: sleeping %.2fs", sleep_for)
            time.sleep(sleep_for)
        self._last_call_time = time.monotonic()

    def translate(self, text: str, source_lang: str) -> Tuple[str, float]:
        """
        Translate *text* using Papago and return (translated_text, confidence).

        Uses the synchronous API (translate_sync) from PentaGo.
        """
        if not text or not text.strip():
            return "", 100.0

        # Resolve source language for this call
        src_code = self._source_lang_papago or _NLLB_TO_PAPAGO.get(source_lang, "auto")

        for attempt in range(1, self._MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                translator = self._Pentago(src_code, self._target_lang_papago)
                result = translator.translate_sync(text)

                # PentaGo returns a dict with 'translatedText'
                if isinstance(result, dict):
                    translated = result.get("translatedText", "").strip()
                else:
                    # Some versions may return the result object differently
                    translated = str(result).strip()

                if not translated:
                    logger.error("Papago: empty translation for: %r", text[:40])
                    return "[Translation Error]", 0.0

                logger.debug("Papago: %r → %r", text[:40], translated[:40])
                return translated, 92.0

            except Exception as exc:
                logger.warning(
                    "Papago translation error (attempt %d/%d): %s",
                    attempt, self._MAX_RETRIES, exc,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(1.0 * attempt)  # simple backoff
                    continue
                logger.error("Papago: all retries exhausted for: %r", text[:40])
                return "[Translation Error]", 0.0

        return "[Translation Error]", 0.0
