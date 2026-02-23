"""Async notification sender with deduplication and heartbeat."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable

from app_config import MonitoringConfig
from .events import EVT_HEARTBEAT, make_event, render_message
from .telegram_client import send_with_retry

logger = logging.getLogger(__name__)


class Notifier:
    """Manages Telegram notifications with async delivery and heartbeat.

    Usage::

        notifier = Notifier(monitoring_config)
        notifier.start()
        notifier.send_event({"event_type": "run_started", "run_id": "abc"})
        # ... later ...
        notifier.stop()
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self._config = config
        self._telegram = config.telegram
        self._heartbeat_config = config.heartbeat
        self._delivery = config.delivery

        # Resolve credentials from environment
        self._token = os.environ.get(self._telegram.bot_token_env, "")
        self._chat_id = os.environ.get(self._telegram.chat_id_env, "")

        # State
        self._dedup: dict[str, float] = {}
        self._dedup_ttl = self._delivery.dedup_ttl_hours * 3600
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._enabled = bool(
            config.enabled and self._token and self._chat_id
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        """Start the notifier (heartbeat thread if configured)."""
        if not self._enabled:
            logger.info(
                "Notifications disabled (missing credentials or config)."
            )

    def stop(self) -> None:
        """Stop the notifier and flush pending messages."""
        self._stop_heartbeat()

    def send_event(self, event: dict[str, Any]) -> None:
        """Send a notification event.

        Events are deduplicated by event_id and sent via Telegram.
        """
        if not self._enabled:
            return

        event_id = event.get("event_id", "")
        if self._is_duplicate(event_id):
            logger.debug("Skipping duplicate event: %s", event_id)
            return

        message = render_message(event)
        self._send(message)
        self._mark_sent(event_id)

    def start_heartbeat(self, state_fn: Callable[[], dict[str, Any]]) -> None:
        """Start periodic heartbeat notifications.

        Args:
            state_fn: Callable that returns current run state for heartbeat.
        """
        if not self._enabled or not self._heartbeat_config.enabled:
            return

        interval = self._heartbeat_config.interval_minutes * 60

        def _heartbeat_loop():
            while not self._heartbeat_stop.wait(interval):
                try:
                    state = state_fn()
                    event = make_event(
                        EVT_HEARTBEAT,
                        state.get("run_id", "?"),
                        status=state.get("status", "unknown"),
                        attempts=len(state.get("attempts", [])),
                        elapsed_sec=state.get("elapsed_sec"),
                        event_suffix=f"hb_{int(time.time())}",
                    )
                    self.send_event(event)
                except Exception:
                    logger.debug("Heartbeat event failed", exc_info=True)

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=_heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)

    def _send(self, message: str) -> None:
        """Send a message via Telegram."""
        try:
            send_with_retry(
                self._token,
                self._chat_id,
                message,
                timeout=self._telegram.timeout,
                max_retries=self._telegram.max_retries,
                base_delay=self._telegram.base_delay,
                jitter=self._telegram.jitter,
            )
        except Exception:
            logger.debug("Failed to send Telegram message", exc_info=True)

    def _is_duplicate(self, event_id: str) -> bool:
        """Check if an event has already been sent (within TTL)."""
        if not event_id:
            return False
        sent_at = self._dedup.get(event_id)
        if sent_at is None:
            return False
        if time.time() - sent_at > self._dedup_ttl:
            del self._dedup[event_id]
            return False
        return True

    def _mark_sent(self, event_id: str) -> None:
        """Record that an event was sent."""
        if event_id:
            self._dedup[event_id] = time.time()
        # Periodic cleanup
        if len(self._dedup) > 1000:
            self._cleanup_dedup()

    def _cleanup_dedup(self) -> None:
        """Remove expired dedup entries."""
        now = time.time()
        expired = [
            k for k, v in self._dedup.items() if now - v > self._dedup_ttl
        ]
        for k in expired:
            del self._dedup[k]


def make_notify_callback(
    notifier: Notifier | None,
) -> Callable[[dict[str, Any]], None]:
    """Create a notification callback suitable for the orchestrator.

    Returns a no-op callback if the notifier is None or disabled.
    """
    if notifier is None or not notifier.enabled:
        return lambda event: None
    return notifier.send_event
