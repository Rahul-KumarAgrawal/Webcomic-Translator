"""
core/groq_translator.py
Groq Cloud LLM translation backend using llama-3.3-70b-versatile.

Sends bubbles in batches of up to 250 at a time as a JSON array.
One API request translates up to 250 bubbles simultaneously.
Includes automatic 5-second delays between batches to respect
Groq's 12K tokens/minute free-tier rate limit.

Rate-limit safety (Groq free tier):
  - 30 requests / minute  → 1 batch = 1 request ✅
  - 12,000 tokens / minute → 250 bubbles ≈ 7,500 tokens ✅
  - 100,000 tokens / day  → 700 bubbles = ~3 batches = ~21,000 tokens ✅

Provides the same translate() / translate_batch() interface as other
engine backends:
    translate(source_text, source_lang) → (translated_text, confidence)
    translate_batch(texts, source_lang) → [(translated_text, confidence), ...]

Usage in settings.yaml / UI:
    translation_engine: groq_llama
    groq_api_key: <your-groq-key>   (reuses the same key as auto-detection)
"""

import json
import logging
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Language name mapping: NLLB-200 code → human-readable for prompt ──────────
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

_GROQ_MODEL = "llama-3.3-70b-versatile"
_BATCH_SIZE = 250          # bubbles per API call
_INTER_BATCH_DELAY = 5.0   # seconds between batches to respect 12K TPM limit
_MAX_RETRIES = 3


class GroqTranslator:
    """
    Translates manga/comic speech bubble text using Groq's Llama 3.3 70B model.

    Key behaviour:
    - Accepts single texts via translate() or large lists via translate_batch().
    - translate_batch() chunks the input into groups of _BATCH_SIZE (250) and
      sends each chunk as a single JSON-array prompt, then sleeps _INTER_BATCH_DELAY
      seconds between chunks to stay within the free-tier token-per-minute limit.
    - The LLM is instructed to return a JSON array of translated strings with
      the same length as the input, so results map 1-to-1 with the source texts.
    """

    def __init__(
        self,
        api_key: str,
        target_lang: str = "eng_Latn",
        source_lang_override: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("Groq API key is required for the Groq translation engine.")

        self._api_key = api_key
        self._target_lang_nllb = target_lang
        self._source_lang_override = source_lang_override

        self._target_lang_name = _NLLB_TO_LANG_NAME.get(target_lang, "English")
        self._source_lang_name = (
            _NLLB_TO_LANG_NAME.get(source_lang_override)
            if source_lang_override
            else None
        )

        # Lazy import groq SDK
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError(
                "The 'groq' Python package is required. Run: pip install groq"
            )

        logger.info(
            "[Groq] Translator initialised (model=%s, target=%s, source=%s)",
            _GROQ_MODEL,
            self._target_lang_name,
            self._source_lang_name or "auto-detect",
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def translate(self, text: str, source_lang: str) -> Tuple[str, float]:
        """Translate a single bubble text. Wraps translate_batch for simplicity."""
        if not text or not text.strip():
            return "", 100.0
        results = self.translate_batch([text], source_lang)
        return results[0] if results else ("[Translation Error]", 0.0)

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
    ) -> List[Tuple[str, float]]:
        """
        Translate a list of bubble texts in safe batches of _BATCH_SIZE.

        Returns a list of (translated_text, confidence) tuples, same length
        and order as the input `texts`.
        """
        if not texts:
            return []

        src_name = self._source_lang_name or _NLLB_TO_LANG_NAME.get(
            source_lang, "the source language"
        )

        all_results: List[Tuple[str, float]] = []
        total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

        for batch_num, start in enumerate(range(0, len(texts), _BATCH_SIZE), 1):
            chunk = texts[start: start + _BATCH_SIZE]
            logger.info(
                "[Groq] Translating batch %d/%d (%d bubbles)...",
                batch_num, total_batches, len(chunk),
            )

            batch_results = self._call_groq_batch(chunk, src_name)
            all_results.extend(batch_results)

            # Rate-limit guard: sleep between batches (skip after the last one)
            if batch_num < total_batches:
                logger.info(
                    "[Groq] Batch %d/%d done — waiting %.1fs before next batch...",
                    batch_num, total_batches, _INTER_BATCH_DELAY,
                )
                time.sleep(_INTER_BATCH_DELAY)

        logger.info("[Groq] All %d bubbles translated.", len(texts))
        return all_results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _call_groq_batch(
        self,
        texts: List[str],
        src_name: str,
    ) -> List[Tuple[str, float]]:
        """
        Send one batch of texts to Groq and parse the JSON response.

        The prompt asks the model to return ONLY a JSON array of translated
        strings in the same order as the input, with no extra commentary.
        """
        # Encode the texts as a JSON array so the model knows the exact count
        input_json = json.dumps(texts, ensure_ascii=False)

        system_prompt = (
            f"You are an expert manga/comic translator. "
            f"Translate {src_name} dialogue into {self._target_lang_name}. "
            f"You will receive a JSON array of strings. "
            f"Return ONLY a valid JSON array of translated strings, "
            f"same length and same order as the input. "
            f"No explanations, no notes, no markdown — ONLY the JSON array."
        )

        user_prompt = (
            f"Translate each string in this JSON array from {src_name} to "
            f"{self._target_lang_name}. "
            f"Return ONLY a JSON array with {len(texts)} translated strings:\n\n"
            f"{input_json}"
        )

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=_GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_completion_tokens=4096,
                )

                raw = response.choices[0].message.content.strip()

                # Strip markdown code fences if the model adds them
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1]
                    if raw.endswith("```"):
                        raw = raw[: raw.rfind("```")]
                    raw = raw.strip()

                translated_list = json.loads(raw)

                if not isinstance(translated_list, list):
                    raise ValueError("Groq response is not a JSON array.")

                # Pad or truncate to match input length (safety net)
                if len(translated_list) < len(texts):
                    logger.warning(
                        "[Groq] Response has %d items but input had %d — padding with errors.",
                        len(translated_list), len(texts),
                    )
                    translated_list += ["[Translation Error]"] * (len(texts) - len(translated_list))
                elif len(translated_list) > len(texts):
                    translated_list = translated_list[: len(texts)]

                return [(str(t).strip(), 88.0) for t in translated_list]

            except json.JSONDecodeError as exc:
                logger.warning(
                    "[Groq] JSON parse error (attempt %d/%d): %s — raw: %r",
                    attempt, _MAX_RETRIES, exc, raw[:200] if 'raw' in dir() else "N/A",
                )
            except Exception as exc:
                logger.warning(
                    "[Groq] API error (attempt %d/%d): %s",
                    attempt, _MAX_RETRIES, exc,
                )
                # On rate-limit (429), wait longer before retrying
                if "429" in str(exc) or "rate_limit" in str(exc).lower():
                    wait = 60.0 * attempt
                    logger.warning("[Groq] Rate limited — waiting %.0fs before retry...", wait)
                    time.sleep(wait)
                elif attempt < _MAX_RETRIES:
                    time.sleep(2.0 * attempt)

        # All retries exhausted — return error placeholders
        logger.error("[Groq] All retries exhausted for batch of %d bubbles.", len(texts))
        return [("[Translation Error]", 0.0)] * len(texts)
