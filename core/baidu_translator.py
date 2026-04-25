"""
core/baidu_translator.py
Baidu Translate API (通用翻译API) backend for the standard pipeline.

Provides the same translate() interface as DeepLTranslator / GoogleTranslator:
    translate(source_text, source_lang) → (translated_text, confidence)

Baidu Translate API details:
  - Endpoint : https://fanyi-api.baidu.com/api/trans/vip/translate
  - Auth     : APP ID + MD5(appid + q + salt + secretKey)
  - Free tier: 2,000,000 chars / month (Standard) — no billing card needed
  - Rate     : 10 QPS on free tier (no per-minute cap, just concurrent)

Get credentials at: https://fanyi-api.baidu.com/

Usage in settings.yaml:
    translation_engine: baidu
    baidu_app_id: "2015063000000001"
    baidu_secret_key: "12345678"
"""

import hashlib
import logging
import random
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Language code mapping: NLLB-200 → Baidu lang codes ───────────────────────

_NLLB_TO_BAIDU = {
    "eng_Latn": "en",
    "jpn_Jpan": "jp",
    "zho_Hans": "zh",
    "zho_Hant": "cht",
    "kor_Hang": "kor",
    "fra_Latn": "fra",
    "deu_Latn": "de",
    "spa_Latn": "spa",
    "por_Latn": "pt",
    "ita_Latn": "it",
    "rus_Cyrl": "ru",
    "ara_Arab": "ara",
    "hin_Deva": "hi",
    "tha_Thai": "th",
    "vie_Latn": "vie",
    "ind_Latn": "id",
    "tur_Latn": "tr",
    "pol_Latn": "pl",
    "nld_Latn": "nl",
    "ukr_Cyrl": "ukr",
    "swe_Latn": "swe",
    "dan_Latn": "dan",
    "fin_Latn": "fin",
    "nor_Latn": "nor",
    "ces_Latn": "cs",
    "hun_Latn": "hu",
    "ron_Latn": "rom",
    "bul_Cyrl": "bul",
    "hrv_Latn": "hrv",
    "slk_Latn": "sk",
    "heb_Hebr": "heb",
    "msa_Latn": "may",
    "ben_Beng": "ben",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
    "kan_Knda": "kan",
    "mal_Mlym": "mal",
    "mar_Deva": "mar",
    "nep_Deva": "nep",
    "urd_Arab": "urd",
    "fil_Latn": "fil",
    "ukr_Cyrl": "ukr",
    "mon_Mong": "mon",
    "afr_Latn": "afr",
    "est_Latn": "est",
    "mkd_Cyrl": "mac",
    "srp_Cyrl": "srp",
    "slv_Latn": "slo",
    "alb_Latn": "alb",
    "geo_Geor": "geo",
    "hye_Armn": "arm",
    "khm_Khmr": "khm",
    "sin_Sinh": "sin",
}

_ENDPOINT = "https://fanyi-api.baidu.com/api/trans/vip/translate"

# Free tier: 10 QPS — we stay conservative at 1 request / 0.15 s
_MIN_INTERVAL = 0.15


def _sign(app_id: str, query: str, salt: str, secret_key: str) -> str:
    """Compute Baidu's MD5 signature: MD5(appid + q + salt + secretKey)."""
    raw = app_id + query + salt + secret_key
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class BaiduTranslator:
    """
    Translates text using the Baidu Translate General API.

    No third-party SDK needed — pure stdlib + requests.
    Instantiate once per batch for connection reuse.
    """

    _MAX_RETRIES = 3

    def __init__(
        self,
        app_id: str,
        secret_key: str,
        target_lang: str = "eng_Latn",
        source_lang_override: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        app_id              : Baidu APP ID from the developer console.
        secret_key          : Baidu Secret Key from the developer console.
        target_lang         : NLLB-200 code for the desired output language.
        source_lang_override: NLLB code to force as source (None = auto-detect).
        """
        if not app_id or not secret_key:
            raise ValueError(
                "Baidu APP ID and Secret Key are required. "
                "Set them in Settings → Translation Engine → Baidu."
            )

        self._app_id    = app_id
        self._secret_key = secret_key

        self._target_lang = _NLLB_TO_BAIDU.get(target_lang, "en")
        self._source_lang = (
            _NLLB_TO_BAIDU.get(source_lang_override)
            if source_lang_override
            else None          # "auto" = Baidu auto-detects
        )

        import importlib
        self._requests = importlib.import_module("requests")
        self._last_call_time = 0.0

        logger.info(
            "Baidu Translator initialised (target=%s, source=%s)",
            self._target_lang,
            self._source_lang or "auto-detect",
        )

    def _wait_for_rate_limit(self):
        """Enforce minimum inter-request gap."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_call_time = time.monotonic()

    def translate(self, text: str, source_lang: str) -> Tuple[str, float]:
        """
        Translate *text* and return (translated_text, confidence).

        Baidu does not expose a confidence score; returns a fixed 93.0.
        Retries up to _MAX_RETRIES times on transient errors (52001 / 52002).
        """
        if not text or not text.strip():
            return "", 100.0

        src = self._source_lang or _NLLB_TO_BAIDU.get(source_lang) or "auto"
        backoff = 1.0

        for attempt in range(1, self._MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                salt = str(random.randint(32768, 65536))
                sign = _sign(self._app_id, text, salt, self._secret_key)

                params = {
                    "q":    text,
                    "from": src,
                    "to":   self._target_lang,
                    "appid": self._app_id,
                    "salt": salt,
                    "sign": sign,
                }

                resp = self._requests.post(
                    _ENDPOINT, params=params, timeout=15
                )
                resp.raise_for_status()
                data = resp.json()

                # Baidu error codes
                error_code = data.get("error_code")
                if error_code:
                    # 52001 = request timeout, 52002 = system error → retry
                    if error_code in ("52001", "52002") and attempt < self._MAX_RETRIES:
                        logger.warning(
                            "Baidu error %s (attempt %d/%d), retrying in %.1fs…",
                            error_code, attempt, self._MAX_RETRIES, backoff,
                        )
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                        continue
                    logger.error(
                        "Baidu API error %s: %s",
                        error_code, data.get("error_msg", "unknown"),
                    )
                    return "[Translation Error]", 0.0

                # Success: collect all sentence parts
                results = data.get("trans_result", [])
                translated = "\n".join(r.get("dst", "") for r in results).strip()
                logger.debug("Baidu: %r → %r", text[:40], translated[:40])
                return translated, 93.0

            except Exception as exc:
                logger.error("Baidu translation error: %s", exc)
                return "[Translation Error]", 0.0

        logger.error("Baidu: max retries exceeded for text: %r", text[:40])
        return "[Translation Error]", 0.0
