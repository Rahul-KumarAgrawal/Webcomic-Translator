"""
core/deepl_translator.py
DeepL API translation backend.

Provides the same translate interface as the NLLB model path:
    translate(source_text, source_lang) → (translated_text, confidence)

Requires a DeepL API key (Free or Pro) configured in settings.yaml.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Language code mapping: NLLB-200 → DeepL ──────────────────────────────────

_NLLB_TO_DEEPL_SOURCE = {
    "jpn_Jpan": "JA",
    "zho_Hans": "ZH",
    "zho_Hant": "ZH",
    "kor_Hang": "KO",
    "eng_Latn": "EN",
    "fra_Latn": "FR",
    "deu_Latn": "DE",
    "spa_Latn": "ES",
    "por_Latn": "PT",
    "ita_Latn": "IT",
    "rus_Cyrl": "RU",
    "ara_Arab": "AR",
    "hin_Deva": "HI",
    "tha_Thai": "TH",
    "vie_Latn": "VI",
    "ind_Latn": "ID",
    "tur_Latn": "TR",
    "pol_Latn": "PL",
    "nld_Latn": "NL",
    "ukr_Cyrl": "UK",
}

_NLLB_TO_DEEPL_TARGET = {
    "eng_Latn": "EN-US",
    "jpn_Jpan": "JA",
    "zho_Hans": "ZH-HANS",
    "zho_Hant": "ZH-HANT",
    "kor_Hang": "KO",
    "fra_Latn": "FR",
    "deu_Latn": "DE",
    "spa_Latn": "ES",
    "por_Latn": "PT-BR",
    "ita_Latn": "IT",
    "rus_Cyrl": "RU",
    "ara_Arab": "AR",
    "hin_Deva": "HI",
    "tha_Thai": "TH",  # Note: Thai not supported by DeepL yet
    "vie_Latn": "VI",  # Note: Vietnamese not supported by DeepL yet
    "ind_Latn": "ID",
    "tur_Latn": "TR",
    "pol_Latn": "PL",
    "nld_Latn": "NL",
    "ukr_Cyrl": "UK",
}


class DeepLTranslator:
    """
    Translates text using the DeepL API.

    Instantiate once per batch for connection reuse.
    """

    def __init__(self, api_key: str, target_lang: str = "eng_Latn",
                 source_lang_override: Optional[str] = None):
        """
        Parameters
        ----------
        api_key             : DeepL authentication key (Free keys end with ':fx').
        target_lang         : NLLB-200 language code for the output language.
        source_lang_override: NLLB code to force as source (None = let DeepL auto-detect).
        """
        if not api_key:
            raise ValueError(
                "DeepL API key is required. Set it in Settings → Translation Engine → DeepL API Key."
            )

        import deepl
        self._translator = deepl.Translator(api_key)
        self._target_lang_nllb = target_lang
        self._source_lang_override = source_lang_override

        # Resolve DeepL target language code
        self._target_lang = _NLLB_TO_DEEPL_TARGET.get(target_lang, "EN-US")
        self._source_lang = (
            _NLLB_TO_DEEPL_SOURCE.get(source_lang_override)
            if source_lang_override
            else None
        )

        logger.info(
            "DeepL translator initialised (target=%s, source=%s)",
            self._target_lang,
            self._source_lang or "auto-detect",
        )

    def translate(self, text: str, source_lang: str) -> Tuple[str, float]:
        """
        Translate *text* and return (translated_text, confidence).

        DeepL does not expose a confidence score, so we use a fixed 95.0
        to indicate high-quality API translation.
        """
        if not text or not text.strip():
            return "", 100.0

        try:
            # Use override if set, otherwise map the detected source lang
            src = self._source_lang or _NLLB_TO_DEEPL_SOURCE.get(source_lang)

            result = self._translator.translate_text(
                text,
                source_lang=src,       # None = auto-detect
                target_lang=self._target_lang,
            )

            translated = result.text
            logger.debug("DeepL: %r → %r", text[:40], translated[:40])
            return translated, 95.0

        except Exception as exc:
            logger.error("DeepL translation error: %s", exc)
            return "[Translation Error]", 0.0
