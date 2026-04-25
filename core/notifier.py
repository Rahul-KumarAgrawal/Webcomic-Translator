"""
core/notifier.py
Send Windows toast notifications for key events.
Falls back to plyer if win10toast is unavailable.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_APP_NAME = "CBZ Translator"
_ICON_PATH: Optional[str] = None  # Optionally set to .ico path


def notify(title: str, message: str, duration: int = 8) -> None:
    """
    Send a Windows toast notification.

    Parameters
    ----------
    title   : Notification title string.
    message : Body text.
    duration: How long (seconds) the toast stays visible.
    """
    try:
        _notify_win10toast(title, message, duration)
        return
    except Exception as exc:
        logger.debug("win10toast failed (%s), trying plyer...", exc)

    try:
        _notify_plyer(title, message, duration)
        return
    except Exception as exc:
        logger.debug("plyer also failed (%s)", exc)

    # Silent fallback — log only
    logger.info("[NOTIFICATION] %s: %s", title, message)


def notify_batch_done(cbz_name: str, num_bubbles: int, elapsed_secs: float) -> None:
    m = int(elapsed_secs // 60)
    s = int(elapsed_secs % 60)
    notify(
        f"✅ Batch Complete — {cbz_name}",
        f"Translated {num_bubbles} bubbles in {m}m {s}s.",
    )


def notify_finetune_done(checkpoint_version: int, num_pairs: int) -> None:
    notify(
        "🧠 Fine-Tuning Complete",
        f"Checkpoint v{checkpoint_version} saved using {num_pairs} approved pairs.",
    )


def notify_backup_done(backup_path: str) -> None:
    notify(
        "💾 Backup Complete",
        f"Saved to: {backup_path}",
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _notify_win10toast(title: str, message: str, duration: int) -> None:
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    toaster.show_toast(
        title,
        message,
        icon_path=_ICON_PATH,
        duration=duration,
        threaded=True,
    )
    logger.debug("win10toast sent: %s", title)


def _notify_plyer(title: str, message: str, duration: int) -> None:
    from plyer import notification
    notification.notify(
        title=title,
        message=message,
        app_name=_APP_NAME,
        timeout=duration,
    )
    logger.debug("plyer notification sent: %s", title)
