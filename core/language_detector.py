"""
core/language_detector.py
Detect whether extracted text is Chinese (ZH), Japanese (JA), or Korean (KO).

Strategy:
  1. Unicode block ranges (fast, character-level heuristic)
  2. langdetect library (statistical, sentence-level)

Returns NLLB-200 language codes:
  ZH → "zho_Hans"  (Simplified Chinese — most common in manhwa/manhua)
  JA → "jpn_Jpan"
  KO → "kor_Hang"
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# NLLB-200 language codes
LANG_ZH = "zho_Hans"
LANG_JA = "jpn_Jpan"
LANG_KO = "kor_Hang"
LANG_EN = "eng_Latn"

# Unicode block ranges
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # Extension A
    (0x20000, 0x2A6DF),  # Extension B
    (0xF900, 0xFAFF),    # Compatibility Ideographs
]
_HIRAGANA = (0x3040, 0x309F)
_KATAKANA = (0x30A0, 0x30FF)
_HANGUL   = (0xAC00, 0xD7AF)


def _count_range(text: str, lo: int, hi: int) -> int:
    return sum(1 for c in text if lo <= ord(c) <= hi)


def _unicode_heuristic(text: str) -> Optional[str]:
    """Return lang code if a clear Unicode-block majority exists."""
    if not text.strip():
        return None

    cjk      = sum(_count_range(text, lo, hi) for lo, hi in _CJK_RANGES)
    hiragana = _count_range(text, *_HIRAGANA)
    katakana = _count_range(text, *_KATAKANA)
    hangul   = _count_range(text, *_HANGUL)
    japanese = hiragana + katakana

    totals = {LANG_JA: japanese, LANG_KO: hangul, LANG_ZH: cjk}
    best_lang, best_count = max(totals.items(), key=lambda x: x[1])

    if best_count == 0:
        return None

    # Japanese is identified by kana; CJK alone could be ZH or JA
    if best_lang == LANG_ZH and japanese > 0:
        # Mixed CJK+kana → Japanese
        return LANG_JA

    if best_count >= 2:
        return best_lang

    return None


def detect_language(text: str) -> str:
    """
    Detect language of *text* and return NLLB-200 language code.
    Falls back to LANG_ZH if detection fails.
    """
    if not text or not text.strip():
        return LANG_ZH  # sensible default

    # Step 1: Unicode heuristic (fast)
    lang = _unicode_heuristic(text)
    if lang:
        logger.debug("Unicode heuristic → %s for: %.40r", lang, text)
        return lang

    # Step 2: langdetect (statistical)
    try:
        from langdetect import detect
        ld_lang = detect(text)
        mapping = {
            "zh-cn": LANG_ZH, "zh-tw": LANG_ZH, "zh": LANG_ZH,
            "ja": LANG_JA,
            "ko": LANG_KO,
        }
        if ld_lang in mapping:
            logger.debug("langdetect → %s (%s) for: %.40r", mapping[ld_lang], ld_lang, text)
            return mapping[ld_lang]
    except Exception as exc:
        logger.debug("langdetect failed: %s", exc)

    logger.debug("Detection inconclusive, defaulting to ZH for: %.40r", text)
    return LANG_ZH
