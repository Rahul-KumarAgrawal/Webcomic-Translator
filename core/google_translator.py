"""
core/google_translator.py
Google Gemini API translation backend for the standard pipeline.

Uses the Gemini generative AI API (generativelanguage.googleapis.com),
which is what Google AI Studio API keys (AIzaSy...) grant access to.

This is distinct from the paid Cloud Translation API v2 — Gemini is
free-tier accessible via AI Studio keys and produces high-quality
context-aware translations.

Provides the same translate() interface as DeepLTranslator:
    translate(source_text, source_lang) → (translated_text, confidence)

Usage in settings.yaml:
    translation_engine: google
    google_api_key: "AIzaSy..."
"""

import json
import logging
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
    "swe_Latn": "Swedish",
    "dan_Latn": "Danish",
    "fin_Latn": "Finnish",
    "nor_Latn": "Norwegian",
    "ces_Latn": "Czech",
    "hun_Latn": "Hungarian",
    "ron_Latn": "Romanian",
    "bul_Cyrl": "Bulgarian",
    "hrv_Latn": "Croatian",
    "slk_Latn": "Slovak",
    "cat_Latn": "Catalan",
    "heb_Hebr": "Hebrew",
    "msa_Latn": "Malay",
    "fil_Latn": "Filipino",
    "ben_Beng": "Bengali",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "guj_Gujr": "Gujarati",
    "pan_Guru": "Punjabi",
    "urd_Arab": "Urdu",
    "fas_Arab": "Persian",
    "swh_Latn": "Swahili",
    "afr_Latn": "Afrikaans",
    "isl_Latn": "Icelandic",
    "lat_Latn": "Latin",
    "lit_Latn": "Lithuanian",
    "lav_Latn": "Latvian",
    "est_Latn": "Estonian",
    "mkd_Cyrl": "Macedonian",
    "srp_Cyrl": "Serbian",
    "slv_Latn": "Slovenian",
    "bos_Latn": "Bosnian",
    "alb_Latn": "Albanian",
    "glg_Latn": "Galician",
    "eus_Latn": "Basque",
    "aze_Latn": "Azerbaijani",
    "kaz_Cyrl": "Kazakh",
    "uzb_Latn": "Uzbek",
    "geo_Geor": "Georgian",
    "hye_Armn": "Armenian",
    "khm_Khmr": "Khmer",
    "mya_Mymr": "Burmese",
    "sin_Sinh": "Sinhala",
    "nep_Deva": "Nepali",
    "mon_Mong": "Mongolian",
}

# Gemini model to use — flash is fast and free-tier
_GEMINI_MODEL = "gemini-2.0-flash"
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{_GEMINI_MODEL}:generateContent"
)


class GoogleTranslator:
    """
    Translates manga/manhwa speech-bubble text using the Google Gemini API.

    Uses the same AIzaSy... key issued by Google AI Studio — no billing
    or Cloud Console setup required for the free tier (up to 1,500 RPD).

    Instantiate once per batch for efficiency.
    """

    # Gemini free tier: 15 RPM → minimum 4 s between calls
    _MIN_INTERVAL = 4.0   # seconds between consecutive API calls
    _MAX_RETRIES  = 4     # retry attempts on 429 before giving up

    def __init__(
        self,
        api_key: str,
        target_lang: str = "eng_Latn",
        source_lang_override: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        api_key             : Google AI Studio key (AIzaSy...).
        target_lang         : NLLB-200 code for the desired output language.
        source_lang_override: NLLB code to force as source (None = auto-detect).
        """
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set it in Settings → Translation Engine → Google API Key."
            )

        self._api_key = api_key
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

        # Rate-limit state: track when the last call was made
        self._last_call_time = 0.0

        logger.info(
            "Gemini Translator initialised (target=%s, source=%s, min_interval=%.1fs)",
            self._target_lang_name,
            self._source_lang_name or "auto-detect",
            self._MIN_INTERVAL,
        )

    def _wait_for_rate_limit(self):
        """Block until the minimum inter-request interval has elapsed."""
        import time
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._MIN_INTERVAL:
            sleep_for = self._MIN_INTERVAL - elapsed
            logger.debug("Gemini rate-limit: sleeping %.2fs", sleep_for)
            time.sleep(sleep_for)
        self._last_call_time = time.monotonic()

    def translate(self, text: str, source_lang: str) -> Tuple[str, float]:
        """
        Translate *text* using Gemini and return (translated_text, confidence).

        Automatically rate-limits to ≤15 RPM and retries up to
        _MAX_RETRIES times with exponential backoff on 429 responses.
        """
        import time

        if not text or not text.strip():
            return "", 100.0

        # Build source description for the prompt
        src_name = self._source_lang_name or _NLLB_TO_LANG_NAME.get(
            source_lang, "the source language"
        )

        prompt = (
            f"Translate the following {src_name} manga/comic dialogue text to "
            f"{self._target_lang_name}. "
            "Return ONLY the translated text with no explanation, notes, or quotes:\n\n"
            f"{text}"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
            },
        }

        backoff = self._MIN_INTERVAL  # start backoff at the base interval

        for attempt in range(1, self._MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                resp = self._requests.post(
                    _GEMINI_ENDPOINT,
                    params={"key": self._api_key},
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=30,
                )

                if resp.status_code == 429:
                    # Check if Retry-After header is present
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else backoff
                    logger.warning(
                        "Gemini 429 rate limit (attempt %d/%d). "
                        "Waiting %.1fs before retry...",
                        attempt, self._MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)  # cap at 60 s
                    self._last_call_time = 0.0      # force rate-limiter to re-measure
                    continue

                resp.raise_for_status()

                data = resp.json()
                translated = (
                    data["candidates"][0]["content"]["parts"][0]["text"].strip()
                )
                logger.debug("Gemini: %r → %r", text[:40], translated[:40])
                return translated, 90.0

            except Exception as exc:
                # Non-429 error — don't retry
                logger.error("Gemini translation error: %s", exc)
                return "[Translation Error]", 0.0

        logger.error("Gemini: max retries (%d) exceeded for text: %r", self._MAX_RETRIES, text[:40])
        return "[Translation Error]", 0.0
