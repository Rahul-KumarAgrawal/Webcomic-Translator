"""
setup_init_db.py
Initialises the global translation memory SQLite database.
Run by setup.bat — safe to run multiple times (idempotent).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

try:
    from memory.memory_manager import MemoryManager
    mm = MemoryManager()
    print("  Global translation memory database initialised.")
except Exception as e:
    print(f"[ERROR] DB init failed: {e}")
    sys.exit(1)
