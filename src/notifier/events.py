"""Notification event types and message rendering."""

from __future__ import annotations

from typing import Any

EVT_RUN_STARTED = "run_started"
EVT_ATTEMPT_COMPLETED = "attempt_completed"
EVT_RUN_COMPLETED = "run_completed"
EVT_RUN_FAILED = "run_failed"
EVT_RUN_INTERRUPTED = "run_interrupted"
EVT_HEARTBEAT = "heartbeat"


def render_message(event: dict[str, Any]) -> str:
    """Render an event dict into a Telegram message string.

    Format: ``[pyscf_auto] <event_type> | run_id=<id> | <key>=<value> ...``
    """
    event_type = event.get("event_type", "unknown")
    run_id = event.get("run_id", "?")

    parts = [f"[pyscf_auto] {event_type}", f"run_id={run_id}"]

    # Add relevant fields based on event type
    skip_keys = {"event_type", "run_id", "event_id", "timestamp"}
    for key, value in event.items():
        if key in skip_keys:
            continue
        if value is not None:
            parts.append(f"{key}={value}")

    return " | ".join(parts)


def make_event(
    event_type: str,
    run_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a notification event dict.

    Args:
        event_type: One of the EVT_* constants.
        run_id: The run identifier.
        **kwargs: Additional event data.

    Returns:
        Event dictionary ready for the notifier.
    """
    event: dict[str, Any] = {
        "event_type": event_type,
        "run_id": run_id,
    }
    # Build event_id for deduplication
    suffix = kwargs.pop("event_suffix", "")
    event["event_id"] = f"{run_id}:{event_type}:{suffix}" if suffix else f"{run_id}:{event_type}"

    event.update(kwargs)
    return event
