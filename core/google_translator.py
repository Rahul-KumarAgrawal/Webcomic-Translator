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
import os
from typing import Optional, Tuple, List, Dict, Any, Union

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
        system_prompt: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        api_key             : Google AI Studio key (AIzaSy...).
        target_lang         : NLLB-200 code for the desired output language.
        source_lang_override: NLLB code to force as source (None = auto-detect).
        system_prompt       : Custom translation directives (None = use expert default).
        """
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set it in Settings → Translation Engine → Google API Key."
            )

        self._api_key = api_key
        self._target_lang_nllb = target_lang
        self._source_lang_override = source_lang_override
        self._system_prompt = system_prompt

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
        """Translate *text* using Gemini and return (translated_text, confidence)."""
        results = self.translate_batch([text], source_lang)
        return results[0]

    def translate_batch(self, texts: List[str], source_lang: str) -> List[Tuple[str, float]]:
        """
        Translate a list of strings in a single batch request to Gemini.
        Returns a list of (translated_text, confidence) tuples.
        """
        import time
        if not texts:
            return []
        
        # Filter out empty strings but keep indices
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return [("", 100.0)] * len(texts)

        valid_texts = [texts[i] for i in valid_indices]
        
        # Build source description
        src_name = self._source_lang_name or _NLLB_TO_LANG_NAME.get(
            source_lang, "the source language"
        )

        # Build base prompt (custom or expert default)
        if self._system_prompt and self._system_prompt.strip():
            base_prompt = self._system_prompt
            # Ensure target language is injected if placeholder [TARGET_LANG] is used
            base_prompt = base_prompt.replace("[TARGET_LANG]", self._target_lang_name)
            base_prompt = base_prompt.replace("[SOURCE_LANG]", src_name)
        else:
            base_prompt = (
                f"Act as an expert manga, manhua, and webtoon translator and localizer across all genres. "
                f"Your primary task is to translate raw text blocks into natural, fluent, and highly accurate {self._target_lang_name}, "
                f"prioritizing flow and contextual meaning over rigid, literal machine translation.\n\n"
                f"Core Directives:\n"
                f"1. Clarity & Readability: Keep the translated sentences clear, concise, and easy to digest. Comic dialogue should be punchy and natural.\n"
                f"2. Genre & Context Accuracy: Use precise terminology tailored to the specific genre. Maintain strict consistency in terminology, character titles, and specialized vocabulary across the entire batch.\n"
                f"3. OCR Noise Handling: For any line that is clearly garbled OCR nonsense, watermarks, or credits, return '[Skipped]' for that entry.\n"
                f"4. Localization & Tone: Capture emotional weight and character voice. Adapt idioms and slang for native impact without losing original cultural flavor.\n"
                f"5. Strict Preservation: Never censor or modify the underlying meaning. Enhance flow but keep the core truth strictly intact.\n\n"
                f"Output Format: Respond ONLY with a JSON array of strings containing the translations in the exact same order as the input list. "
                f"Each string in the array must correspond to one input text. Do not include introductory text or explanations."
            )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": base_prompt},
                        {"text": json.dumps(valid_texts, ensure_ascii=False)}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        }

        results = [("[Translation Error]", 0.0)] * len(texts)
        # Mark non-valid as empty
        for i in range(len(texts)):
            if i not in valid_indices:
                results[i] = ("", 100.0)

        backoff = self._MIN_INTERVAL
        for attempt in range(1, self._MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                resp = self._requests.post(
                    _GEMINI_ENDPOINT,
                    params={"key": self._api_key},
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=60,
                )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else backoff
                    logger.warning("Gemini 429 (batch attempt %d/%d). Waiting %.1fs", attempt, self._MAX_RETRIES, wait)
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)
                    self._last_call_time = 0.0
                    continue

                resp.raise_for_status()
                data = resp.json()
                
                raw_json = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                translated_list = json.loads(raw_json)

                if not isinstance(translated_list, list):
                    raise ValueError("Gemini did not return a JSON list")

                # Map back to results
                for i, idx in enumerate(valid_indices):
                    if i < len(translated_list):
                        results[idx] = (translated_list[i], 90.0)
                
                return results

            except Exception as exc:
                logger.error("Gemini batch translation error (attempt %d): %s", attempt, exc)
                if attempt == self._MAX_RETRIES:
                    return results
                time.sleep(backoff)
                backoff *= 2

        return results

    def detect_language_from_image(self, image_path: str) -> Optional[str]:
        """
        Multimodal detection: Gemini looks at the image and identifies the language.
        Returns ISO-639-1 code (e.g., 'ja', 'ko', 'zh-CN').
        """
        import base64
        import time

        if not os.path.exists(image_path):
            return None

        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")

            prompt = (
                "Look at this manga/comic page. What is the primary language of the text in the speech bubbles? "
                "Return ONLY the ISO-639-1 language code (e.g., 'ja' for Japanese, 'ko' for Korean, 'zh-CN' for Simplified Chinese, 'en' for English). "
                "If no text is found, return 'unknown'."
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/png" if image_path.lower().endswith(".png") else "image/jpeg",
                                    "data": img_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                },
            }

            self._wait_for_rate_limit()
            resp = self._requests.post(
                _GEMINI_ENDPOINT,
                params={"key": self._api_key},
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30,
            )
            resp.raise_for_status()

            data = resp.json()
            detected = data["candidates"][0]["content"]["parts"][0].get("text", "").strip().lower()
            
            # Clean up potential extra noise from Gemini
            if "ja" in detected: return "ja"
            if "ko" in detected: return "ko"
            if "zh-cn" in detected or "zh_cn" in detected: return "zh-CN"
            if "zh-tw" in detected or "zh_tw" in detected: return "zh-TW"
            if "zh" in detected: return "zh-CN"
            if "en" in detected: return "en"
            
            if len(detected) <= 5:
                return detected
                
            return None

        except Exception as e:
            logger.error("Gemini multimodal detection failed: %s", e)
            return None
