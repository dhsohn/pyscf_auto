"""Notification system for pyscf_auto."""

from .notifier import Notifier, make_notify_callback
from .events import (
    EVT_RUN_STARTED,
    EVT_ATTEMPT_COMPLETED,
    EVT_RUN_COMPLETED,
    EVT_RUN_FAILED,
    EVT_HEARTBEAT,
)

__all__ = [
    "Notifier",
    "make_notify_callback",
    "EVT_RUN_STARTED",
    "EVT_ATTEMPT_COMPLETED",
    "EVT_RUN_COMPLETED",
    "EVT_RUN_FAILED",
    "EVT_HEARTBEAT",
]
