"""
memory/override_checker.py
Implements the three-step translation lookup chain:
  1. Series memory (approved=True)
  2. Global memory (approved=True)
  3. NLLB-200 model (caller's responsibility)

Returns (translated_text, confidence, source, memory_id) or None.
"""

import logging
from typing import Optional, Tuple

from memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

LookupResult = Optional[Tuple[str, float, str, int]]
# (translated_text, confidence, source_label, memory_id)


class OverrideChecker:
    """
    Encapsulates the series → global → model lookup chain.
    """

    def __init__(self):
        self.memory_manager = MemoryManager()

    def lookup(self, source_text: str, series: str) -> LookupResult:
        """
        Look up *source_text* in translation memories.

        Parameters
        ----------
        source_text : Raw OCR text from the speech bubble.
        series      : Series name (used to locate the series DB).

        Returns
        -------
        (translated_text, confidence, source, row_id)  if found
        None  if no approved match exists (caller should run model)
        """
        # ── Step 1: Series memory ──────────────────────────────────────────────
        row = self.memory_manager.lookup_series(source_text, series)
        if row:
            logger.debug("Series memory hit for: %.40r", source_text)
            return (
                row["translated_text"],
                100.0,
                "series_memory",
                row["id"],
            )

        # ── Step 2: Global memory ──────────────────────────────────────────────
        row = self.memory_manager.lookup_global(source_text)
        if row:
            logger.debug("Global memory hit for: %.40r", source_text)
            return (
                row["translated_text"],
                100.0,
                "global_memory",
                row["id"],
            )

        # ── Step 3: No match — caller handles model inference ──────────────────
        logger.debug("No memory match for: %.40r", source_text)
        return None

    def promote_to_global(self, source_text: str, series: str) -> bool:
        """
        Copy an approved series memory entry to the global memory.
        Useful for frequently reused phrases across series.

        Returns True if the entry was copied, False if it wasn't found.
        """
        mm  = self.memory_manager
        row = mm.lookup_series(source_text, series)
        if not row:
            return False

        mm.save(
            source_lang     = row["source_lang"],
            source_text     = row["source_text"],
            translated_text = row["translated_text"],
            approved        = True,
            edited          = bool(row["edited"]),
            confidence_score= row["confidence_score"],
            series          = None,  # save to global
        )
        logger.info(
            "Promoted to global memory: %.40r → %.40r",
            source_text, row["translated_text"],
        )
        return True
