"""
memory/memory_manager.py
CRUD interface for translation memory with SQLite ↔ JSON dual-storage sync.

Schema (identical for series and global DBs):
  id, source_lang, source_text, translated_text, approved (bool),
  edited (bool), confidence_score (float), times_used (int),
  created_at (ISO timestamp), updated_at (ISO timestamp)
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GLOBAL_DB_PATH   = os.path.join(_ROOT, "memory", "global", "translation_memory.db")
GLOBAL_JSON_PATH = os.path.join(_ROOT, "memory", "global", "translation_memory.json")
SERIES_DIR       = os.path.join(_ROOT, "memory", "series")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS translation_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_lang     TEXT    NOT NULL,
    source_text     TEXT    NOT NULL,
    translated_text TEXT    NOT NULL,
    approved        INTEGER NOT NULL DEFAULT 0,
    edited          INTEGER NOT NULL DEFAULT 0,
    confidence_score REAL   NOT NULL DEFAULT 0.0,
    times_used      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    UNIQUE(source_text, source_lang)
);
"""


class MemoryManager:
    """
    Provides CRUD operations on translation memory databases.
    Every write operation keeps SQLite and JSON files in sync.
    """

    def __init__(self):
        # Ensure global DB exists
        self.init_db(GLOBAL_DB_PATH)

    # ── DB Initialisation ─────────────────────────────────────────────────────

    def init_db(self, db_path: str) -> None:
        """Create the DB file and schema if they don't exist."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect(db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        # Ensure matching JSON file exists
        json_path = db_path.replace(".db", ".json")
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        logger.debug("DB initialised: %s", db_path)

    def _series_db(self, series: str) -> str:
        safe = self._safe_series_name(series)
        path = os.path.join(SERIES_DIR, safe, "translation_memory.db")
        self.init_db(path)
        return path

    def _series_json(self, series: str) -> str:
        safe = self._safe_series_name(series)
        return os.path.join(SERIES_DIR, safe, "translation_memory.json")

    @staticmethod
    def _safe_series_name(name: str) -> str:
        import re
        return re.sub(r'[<>:"/\\|?*]', "_", name).strip()

    # ── Lookup ────────────────────────────────────────────────────────────────

    def lookup_series(
        self, source_text: str, series: str
    ) -> Optional[Dict[str, Any]]:
        """Return approved series memory row or None."""
        db = self._series_db(series)
        return self._lookup(db, source_text, approved_only=True)

    def lookup_global(self, source_text: str) -> Optional[Dict[str, Any]]:
        """Return approved global memory row or None."""
        return self._lookup(GLOBAL_DB_PATH, source_text, approved_only=True)

    def _lookup(
        self, db_path: str, source_text: str, approved_only: bool = True
    ) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM translation_memory WHERE source_text = ?"
        if approved_only:
            sql += " AND approved = 1"
        sql += " LIMIT 1"
        with self._connect(db_path) as conn:
            row = conn.execute(sql, (source_text,)).fetchone()
        return dict(row) if row else None

    # ── Save / Upsert ─────────────────────────────────────────────────────────

    def save(
        self,
        source_lang: str,
        source_text: str,
        translated_text: str,
        approved: bool,
        edited: bool,
        confidence_score: float,
        series: str = None,
    ) -> int:
        """
        Insert or update a translation pair.
        If series is None, saves to global DB. Otherwise to series DB.
        Returns the row id.
        Syncs JSON immediately.
        """
        if series:
            db_path   = self._series_db(series)
            json_path = self._series_json(series)
        else:
            db_path   = GLOBAL_DB_PATH
            json_path = GLOBAL_JSON_PATH
        now = datetime.now().isoformat()

        with self._connect(db_path) as conn:
            # Try update first (UNIQUE constraint on source_text + source_lang)
            existing = conn.execute(
                "SELECT id FROM translation_memory WHERE source_text=? AND source_lang=?",
                (source_text, source_lang),
            ).fetchone()

            if existing:
                row_id = existing["id"]
                conn.execute(
                    """UPDATE translation_memory
                       SET translated_text=?, approved=?, edited=?,
                           confidence_score=?, updated_at=?
                       WHERE id=?""",
                    (translated_text, int(approved), int(edited),
                     confidence_score, now, row_id),
                )
            else:
                cursor = conn.execute(
                    """INSERT INTO translation_memory
                       (source_lang, source_text, translated_text,
                        approved, edited, confidence_score,
                        times_used, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,0,?,?)""",
                    (source_lang, source_text, translated_text,
                     int(approved), int(edited), confidence_score, now, now),
                )
                row_id = cursor.lastrowid
            conn.commit()

        self._sync_json(db_path, json_path)
        return row_id

    # ── Approve / Reject / Edit ───────────────────────────────────────────────

    def approve(self, row_id: int, series: Optional[str] = None) -> None:
        db_path   = self._series_db(series) if series else GLOBAL_DB_PATH
        json_path = self._series_json(series) if series else GLOBAL_JSON_PATH
        now = datetime.now().isoformat()
        with self._connect(db_path) as conn:
            conn.execute(
                "UPDATE translation_memory SET approved=1, updated_at=? WHERE id=?",
                (now, row_id),
            )
            conn.commit()
        self._sync_json(db_path, json_path)

    def reject(self, row_id: int, series: Optional[str] = None) -> None:
        """Mark as not approved (keeps record for reference)."""
        db_path   = self._series_db(series) if series else GLOBAL_DB_PATH
        json_path = self._series_json(series) if series else GLOBAL_JSON_PATH
        now = datetime.now().isoformat()
        with self._connect(db_path) as conn:
            conn.execute(
                "UPDATE translation_memory SET approved=0, updated_at=? WHERE id=?",
                (now, row_id),
            )
            conn.commit()
        self._sync_json(db_path, json_path)

    def edit(
        self, row_id: int, new_text: str, series: Optional[str] = None
    ) -> None:
        """Update translated_text and mark as edited + approved."""
        db_path   = self._series_db(series) if series else GLOBAL_DB_PATH
        json_path = self._series_json(series) if series else GLOBAL_JSON_PATH
        now = datetime.now().isoformat()
        with self._connect(db_path) as conn:
            conn.execute(
                """UPDATE translation_memory
                   SET translated_text=?, edited=1, approved=1, updated_at=?
                   WHERE id=?""",
                (new_text, now, row_id),
            )
            conn.commit()
        self._sync_json(db_path, json_path)

    def increment_usage(self, row_id: int, is_series: bool, series: str = None) -> None:
        db_path   = self._series_db(series) if (is_series and series) else GLOBAL_DB_PATH
        json_path = self._series_json(series) if (is_series and series) else GLOBAL_JSON_PATH
        now = datetime.now().isoformat()
        with self._connect(db_path) as conn:
            conn.execute(
                "UPDATE translation_memory SET times_used=times_used+1, updated_at=? WHERE id=?",
                (now, row_id),
            )
            conn.commit()
        self._sync_json(db_path, json_path)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_all(
        self, series: Optional[str] = None, approved_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Return all records from series or global memory."""
        db_path = self._series_db(series) if series else GLOBAL_DB_PATH
        sql = "SELECT * FROM translation_memory"
        if approved_only:
            sql += " WHERE approved=1"
        sql += " ORDER BY updated_at DESC"
        with self._connect(db_path) as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def count_approved_all_series(self) -> int:
        """Count total approved pairs across ALL series + global."""
        total = 0
        # Global
        total += self._count_approved(GLOBAL_DB_PATH)
        # All series
        if os.path.exists(SERIES_DIR):
            for series_dir in Path(SERIES_DIR).iterdir():
                db_path = str(series_dir / "translation_memory.db")
                if os.path.exists(db_path):
                    total += self._count_approved(db_path)
        return total

    def _count_approved(self, db_path: str) -> int:
        try:
            with self._connect(db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM translation_memory WHERE approved=1"
                ).fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def get_all_approved_for_training(self) -> List[Dict[str, Any]]:
        """Return all approved pairs across every series + global (for trainer)."""
        results = []
        results.extend(self.get_all(series=None, approved_only=True))
        if os.path.exists(SERIES_DIR):
            for series_dir in Path(SERIES_DIR).iterdir():
                db_path = str(series_dir / "translation_memory.db")
                if os.path.exists(db_path):
                    sql = "SELECT * FROM translation_memory WHERE approved=1"
                    with self._connect(db_path) as conn:
                        rows = conn.execute(sql).fetchall()
                    results.extend([dict(r) for r in rows])
        return results

    def list_series(self) -> List[str]:
        """Return list of series names that have a memory folder."""
        if not os.path.exists(SERIES_DIR):
            return []
        return [d.name for d in Path(SERIES_DIR).iterdir() if d.is_dir()]

    def delete_series(self, series: str) -> None:
        """Delete an entire series memory."""
        import shutil
        safe = self._safe_series_name(series)
        series_path = os.path.join(SERIES_DIR, safe)
        if os.path.exists(series_path):
            shutil.rmtree(series_path)

    # ── JSON sync ─────────────────────────────────────────────────────────────

    def _sync_json(self, db_path: str, json_path: str) -> None:
        """Write all records from SQLite to JSON (full replace)."""
        try:
            with self._connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM translation_memory ORDER BY id"
                ).fetchall()
            data = [dict(r) for r in rows]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("JSON sync error for %s: %s", json_path, exc)

    # ── SQLite helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")  # safer concurrent writes
        return conn
