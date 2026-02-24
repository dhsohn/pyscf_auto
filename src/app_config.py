"""Global application configuration for pyscf_auto."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG_DIR = os.path.expanduser("~/.pyscf_auto")
_DEFAULT_CONFIG_PATH = os.path.join(_DEFAULT_CONFIG_DIR, "config.yaml")
_DEFAULT_ALLOWED_ROOT = os.path.expanduser("~/pyscf_runs")
_DEFAULT_ORGANIZED_ROOT = os.path.expanduser("~/pyscf_outputs")

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:\\")
_WSL_WINDOWS_MOUNT_RE = re.compile(r"^/mnt/[a-zA-Z](/|$)")

_DEFAULT_KEEP_EXTENSIONS = [".inp", ".out", ".xyz", ".gbw", ".hess"]
_DEFAULT_KEEP_FILENAMES = ["run_state.json", "run_report.json", "run_report.md"]
_DEFAULT_REMOVE_PATTERNS = [
    "*.retry*.inp",
    "*.retry*.out",
    "*_trj.xyz",
    "*.densities",
    "*.engrad",
    "*.tmp",
    "*.prop",
    "*.scfp",
    "*.opt",
]


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
    telegram: TelegramTransportConfig = field(default_factory=TelegramTransportConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    enabled: bool = False


@dataclass
class RuntimeConfig:
    allowed_root: str = _DEFAULT_ALLOWED_ROOT
    organized_root: str = _DEFAULT_ORGANIZED_ROOT
    default_max_retries: int = 5


@dataclass
class CleanupConfig:
    keep_extensions: list[str] = field(default_factory=lambda: list(_DEFAULT_KEEP_EXTENSIONS))
    keep_filenames: list[str] = field(default_factory=lambda: list(_DEFAULT_KEEP_FILENAMES))
    remove_patterns: list[str] = field(default_factory=lambda: list(_DEFAULT_REMOVE_PATTERNS))


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)


def load_app_config(config_path: str | None = None) -> AppConfig:
    """Load application configuration from YAML.

    Config search order:
    1. ``config_path`` argument
    2. ``PYSCF_AUTO_CONFIG`` environment variable
    3. ``~/.pyscf_auto/config.yaml``

    Returns defaults when the target file does not exist.
    Raises ``ValueError`` for invalid YAML or invalid schema.
    """
    path = _resolve_config_path(config_path)
    if not path.exists():
        return AppConfig()

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Missing dependency 'PyYAML'. Install project dependencies "
            "(e.g. conda install pyyaml or activate the pyscf_auto environment)."
        ) from exc

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config: {path} ({exc})") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read config file: {path} ({exc})") from exc

    if raw is None:
        return AppConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    cfg = _parse_app_config(raw)
    _validate_app_config(cfg)
    return cfg


def _resolve_config_path(config_path: str | None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()
    env_path = os.environ.get("PYSCF_AUTO_CONFIG", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(_DEFAULT_CONFIG_PATH).expanduser().resolve()


def _parse_app_config(raw: dict[str, Any]) -> AppConfig:
    runtime_raw = _as_mapping(raw.get("runtime"))
    monitoring_raw = _as_mapping(raw.get("monitoring"))
    cleanup_raw = _as_mapping(raw.get("cleanup"))

    runtime = RuntimeConfig(
        allowed_root=_normalize_runtime_path(
            runtime_raw.get("allowed_root"),
            default=_DEFAULT_ALLOWED_ROOT,
            field_name="runtime.allowed_root",
        ),
        organized_root=_normalize_runtime_path(
            runtime_raw.get("organized_root"),
            default=_DEFAULT_ORGANIZED_ROOT,
            field_name="runtime.organized_root",
        ),
        default_max_retries=_as_int(
            runtime_raw.get("default_max_retries"),
            default=RuntimeConfig.default_max_retries,
            field_name="runtime.default_max_retries",
        ),
    )

    monitoring = _parse_monitoring_config(monitoring_raw)
    cleanup = _parse_cleanup_config(cleanup_raw)

    return AppConfig(runtime=runtime, monitoring=monitoring, cleanup=cleanup)


def _parse_monitoring_config(raw: dict[str, Any]) -> MonitoringConfig:
    telegram_raw = _as_mapping(raw.get("telegram"))
    heartbeat_raw = _as_mapping(raw.get("heartbeat"))
    delivery_raw = _as_mapping(raw.get("delivery"))

    return MonitoringConfig(
        enabled=_as_bool(raw.get("enabled"), default=MonitoringConfig.enabled, field_name="monitoring.enabled"),
        telegram=TelegramTransportConfig(
            bot_token_env=_as_env_name(
                telegram_raw.get("bot_token_env"),
                default=TelegramTransportConfig.bot_token_env,
                field_name="monitoring.telegram.bot_token_env",
            ),
            chat_id_env=_as_env_name(
                telegram_raw.get("chat_id_env"),
                default=TelegramTransportConfig.chat_id_env,
                field_name="monitoring.telegram.chat_id_env",
            ),
            timeout=_as_float(
                telegram_raw.get("timeout"),
                default=TelegramTransportConfig.timeout,
                field_name="monitoring.telegram.timeout",
            ),
            max_retries=_as_int(
                telegram_raw.get("max_retries"),
                default=TelegramTransportConfig.max_retries,
                field_name="monitoring.telegram.max_retries",
            ),
            base_delay=_as_float(
                telegram_raw.get("base_delay"),
                default=TelegramTransportConfig.base_delay,
                field_name="monitoring.telegram.base_delay",
            ),
            jitter=_as_float(
                telegram_raw.get("jitter"),
                default=TelegramTransportConfig.jitter,
                field_name="monitoring.telegram.jitter",
            ),
        ),
        heartbeat=HeartbeatConfig(
            enabled=_as_bool(
                heartbeat_raw.get("enabled"),
                default=HeartbeatConfig.enabled,
                field_name="monitoring.heartbeat.enabled",
            ),
            interval_minutes=_as_int(
                heartbeat_raw.get("interval_minutes"),
                default=HeartbeatConfig.interval_minutes,
                field_name="monitoring.heartbeat.interval_minutes",
            ),
        ),
        delivery=DeliveryConfig(
            queue_size=_as_int(
                delivery_raw.get("queue_size"),
                default=DeliveryConfig.queue_size,
                field_name="monitoring.delivery.queue_size",
            ),
            flush_timeout=_as_float(
                delivery_raw.get("flush_timeout"),
                default=DeliveryConfig.flush_timeout,
                field_name="monitoring.delivery.flush_timeout",
            ),
            dedup_ttl_hours=_as_int(
                delivery_raw.get("dedup_ttl_hours"),
                default=DeliveryConfig.dedup_ttl_hours,
                field_name="monitoring.delivery.dedup_ttl_hours",
            ),
        ),
    )


def _parse_cleanup_config(raw: dict[str, Any]) -> CleanupConfig:
    return CleanupConfig(
        keep_extensions=_normalize_extensions(
            raw.get("keep_extensions"),
            defaults=_DEFAULT_KEEP_EXTENSIONS,
            field_name="cleanup.keep_extensions",
        ),
        keep_filenames=_normalize_string_list(
            raw.get("keep_filenames"),
            defaults=_DEFAULT_KEEP_FILENAMES,
            field_name="cleanup.keep_filenames",
        ),
        remove_patterns=_normalize_string_list(
            raw.get("remove_patterns"),
            defaults=_DEFAULT_REMOVE_PATTERNS,
            field_name="cleanup.remove_patterns",
        ),
    )


def _validate_app_config(cfg: AppConfig) -> None:
    _validate_runtime_config(cfg.runtime)
    _validate_monitoring_config(cfg.monitoring)
    _validate_cleanup_config(cfg.cleanup)


def _validate_runtime_config(runtime: RuntimeConfig) -> None:
    if runtime.default_max_retries < 0:
        raise ValueError(
            f"runtime.default_max_retries must be >= 0 (got {runtime.default_max_retries})"
        )
    allowed_root = Path(runtime.allowed_root).resolve()
    organized_root = Path(runtime.organized_root).resolve()
    if _is_subpath(allowed_root, organized_root) or _is_subpath(organized_root, allowed_root):
        raise ValueError(
            "runtime.allowed_root and runtime.organized_root must not contain each other: "
            f"allowed_root={allowed_root}, organized_root={organized_root}"
        )


def _validate_monitoring_config(monitoring: MonitoringConfig) -> None:
    telegram = monitoring.telegram
    if telegram.timeout <= 0:
        raise ValueError(f"monitoring.telegram.timeout must be > 0 (got {telegram.timeout})")
    if telegram.max_retries < 0:
        raise ValueError(
            f"monitoring.telegram.max_retries must be >= 0 (got {telegram.max_retries})"
        )
    if telegram.base_delay < 0:
        raise ValueError(
            f"monitoring.telegram.base_delay must be >= 0 (got {telegram.base_delay})"
        )
    if telegram.jitter < 0:
        raise ValueError(f"monitoring.telegram.jitter must be >= 0 (got {telegram.jitter})")

    delivery = monitoring.delivery
    if delivery.queue_size < 1:
        raise ValueError(
            f"monitoring.delivery.queue_size must be >= 1 (got {delivery.queue_size})"
        )
    if delivery.flush_timeout <= 0:
        raise ValueError(
            "monitoring.delivery.flush_timeout must be > 0 "
            f"(got {delivery.flush_timeout})"
        )
    if delivery.dedup_ttl_hours < 1:
        raise ValueError(
            "monitoring.delivery.dedup_ttl_hours must be >= 1 "
            f"(got {delivery.dedup_ttl_hours})"
        )

    heartbeat = monitoring.heartbeat
    if heartbeat.enabled and heartbeat.interval_minutes < 1:
        raise ValueError(
            "monitoring.heartbeat.interval_minutes must be >= 1 when heartbeat is enabled "
            f"(got {heartbeat.interval_minutes})"
        )


def _validate_cleanup_config(cleanup: CleanupConfig) -> None:
    if not cleanup.keep_extensions:
        raise ValueError("cleanup.keep_extensions must not be empty")
    if not cleanup.keep_filenames:
        raise ValueError("cleanup.keep_filenames must not be empty")


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    raise ValueError("Config sections must be mappings")


def _as_bool(value: Any, *, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _as_int(value: Any, *, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer (got {value!r})") from exc


def _as_float(value: Any, *, default: float, field_name: str) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number (got {value!r})") from exc


def _as_env_name(value: Any, *, default: str, field_name: str) -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{field_name} must be a non-empty string")


def _normalize_extensions(
    value: Any,
    *,
    defaults: list[str],
    field_name: str,
) -> list[str]:
    if value is None:
        return list(defaults)
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in seen:
            seen.add(ext)
            result.append(ext)
    return result


def _normalize_string_list(
    value: Any,
    *,
    defaults: list[str],
    field_name: str,
) -> list[str]:
    if value is None:
        return list(defaults)
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        text = item.strip()
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _normalize_runtime_path(value: Any, *, default: str, field_name: str) -> str:
    raw = default if value is None else value
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    path_text = raw.strip()
    if _is_windows_style_path(path_text):
        raise ValueError(
            f"{field_name} must be a POSIX path (Windows-style paths are unsupported): {path_text!r}"
        )
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path: {path_text!r}")
    return str(path.resolve())


def _is_windows_style_path(path_text: str) -> bool:
    return bool(
        _WINDOWS_DRIVE_RE.match(path_text)
        or _WSL_WINDOWS_MOUNT_RE.match(path_text)
    )


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
