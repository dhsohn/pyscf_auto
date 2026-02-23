"""Telegram API client using only stdlib (urllib)."""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 3500
_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass
class SendResult:
    """Result of a Telegram send attempt."""

    ok: bool
    status_code: int | None = None
    error: str | None = None


def send_message(
    token: str,
    chat_id: str,
    text: str,
    timeout: float = 5.0,
) -> SendResult:
    """Send a single message to Telegram.

    Args:
        token: Bot API token.
        chat_id: Target chat ID.
        text: Message text (truncated to 3500 chars).
        timeout: HTTP timeout in seconds.

    Returns:
        A ``SendResult`` indicating success or failure.
    """
    text = _truncate_text(text, _MAX_MESSAGE_LENGTH)
    url = _API_URL.format(token=token)
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return SendResult(ok=True, status_code=resp.status)
    except urllib.error.HTTPError as exc:
        error_msg = _sanitize_token(
            f"HTTP {exc.code}: {exc.reason}", token
        )
        logger.warning("Telegram send failed: %s", error_msg)
        return SendResult(ok=False, status_code=exc.code, error=error_msg)
    except Exception as exc:
        error_msg = _sanitize_token(str(exc), token)
        logger.warning("Telegram send error: %s", error_msg)
        return SendResult(ok=False, error=error_msg)


def send_with_retry(
    token: str,
    chat_id: str,
    text: str,
    timeout: float = 5.0,
    max_retries: int = 2,
    base_delay: float = 1.0,
    jitter: float = 0.3,
) -> SendResult:
    """Send a message with exponential backoff retry.

    Args:
        token: Bot API token.
        chat_id: Target chat ID.
        text: Message text.
        timeout: HTTP timeout per attempt.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries (seconds).
        jitter: Random jitter added to delay (seconds).

    Returns:
        The ``SendResult`` from the last attempt.
    """
    last_result = send_message(token, chat_id, text, timeout)
    if last_result.ok:
        return last_result

    for attempt in range(max_retries):
        delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
        time.sleep(delay)
        last_result = send_message(token, chat_id, text, timeout)
        if last_result.ok:
            return last_result

    return last_result


def _truncate_text(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sanitize_token(msg: str, token: str) -> str:
    """Remove bot token from error messages."""
    if token:
        return msg.replace(token, "<TOKEN>")
    return msg
