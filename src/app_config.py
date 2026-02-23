"""Global application configuration for pyscf_auto.

Loaded from ``~/.pyscf_auto/config.yaml`` or the path specified by the
``PYSCF_AUTO_CONFIG`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_DIR = os.path.expanduser("~/.pyscf_auto")
_DEFAULT_CONFIG_PATH = os.path.join(_DEFAULT_CONFIG_DIR, "config.yaml")


@dataclass
class TelegramTransportConfig:
    bot_token_env: str = "PYSCF_AUTO_TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "PYSCF_AUTO_TELEGRAM_CHAT_ID"
    timeout: float = 5.0
    max_retries: int = 2
    base_delay: float = 1.0
    jitter: float = 0.3


@dataclass
class HeartbeatConfig:
    enabled: bool = True
    interval_minutes: int = 30


@dataclass
class DeliveryConfig:
    queue_size: int = 1000
    flush_timeout: float = 3.0
    dedup_ttl_hours: int = 24


@dataclass
class MonitoringConfig:
    telegram: TelegramTransportConfig = field(
        default_factory=TelegramTransportConfig
    )
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    enabled: bool = True


@dataclass
class RuntimeConfig:
    allowed_root: str = os.path.expanduser("~/pyscf_runs")
    organized_root: str = os.path.expanduser("~/pyscf_outputs")
    default_max_retries: int = 5


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def load_app_config(config_path: str | None = None) -> AppConfig:
    """Load application configuration from a YAML file.

    Looks for configuration in this order:
    1. ``config_path`` argument
    2. ``PYSCF_AUTO_CONFIG`` environment variable
    3. ``~/.pyscf_auto/config.yaml``

    Returns a default ``AppConfig`` if no config file is found.
    """
    if config_path is None:
        config_path = os.environ.get("PYSCF_AUTO_CONFIG", _DEFAULT_CONFIG_PATH)

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        return AppConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return AppConfig()

    return _parse_app_config(raw)


def _parse_app_config(raw: dict[str, Any]) -> AppConfig:
    """Parse a raw dictionary into an AppConfig."""
    runtime_raw = raw.get("runtime", {})
    monitoring_raw = raw.get("monitoring", {})

    runtime = RuntimeConfig(
        allowed_root=os.path.expanduser(
            runtime_raw.get("allowed_root", "~/pyscf_runs")
        ),
        organized_root=os.path.expanduser(
            runtime_raw.get("organized_root", "~/pyscf_outputs")
        ),
        default_max_retries=runtime_raw.get("default_max_retries", 5),
    )

    monitoring = _parse_monitoring_config(monitoring_raw)

    return AppConfig(runtime=runtime, monitoring=monitoring)


def _parse_monitoring_config(raw: dict[str, Any]) -> MonitoringConfig:
    """Parse monitoring configuration section."""
    if not raw:
        return MonitoringConfig()

    telegram_raw = raw.get("telegram", {})
    heartbeat_raw = raw.get("heartbeat", {})
    delivery_raw = raw.get("delivery", {})

    telegram = TelegramTransportConfig(
        bot_token_env=telegram_raw.get(
            "bot_token_env", "PYSCF_AUTO_TELEGRAM_BOT_TOKEN"
        ),
        chat_id_env=telegram_raw.get(
            "chat_id_env", "PYSCF_AUTO_TELEGRAM_CHAT_ID"
        ),
        timeout=telegram_raw.get("timeout", 5.0),
        max_retries=telegram_raw.get("max_retries", 2),
        base_delay=telegram_raw.get("base_delay", 1.0),
        jitter=telegram_raw.get("jitter", 0.3),
    )

    heartbeat = HeartbeatConfig(
        enabled=heartbeat_raw.get("enabled", True),
        interval_minutes=heartbeat_raw.get("interval_minutes", 30),
    )

    delivery = DeliveryConfig(
        queue_size=delivery_raw.get("queue_size", 1000),
        flush_timeout=delivery_raw.get("flush_timeout", 3.0),
        dedup_ttl_hours=delivery_raw.get("dedup_ttl_hours", 24),
    )

    return MonitoringConfig(
        telegram=telegram,
        heartbeat=heartbeat,
        delivery=delivery,
        enabled=raw.get("enabled", True),
    )
