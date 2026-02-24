"""Lazy-loaded execution plugins.

This module is the only place that resolves stage implementations and
optional exporters at runtime. Keeping this boundary explicit prevents
direct cross-imports from spreading through the codebase.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

from env_compat import env_truthy

logger = logging.getLogger(__name__)


class FeatureUnavailableError(RuntimeError):
    """Raised when an optional workflow feature is disabled or unavailable."""


_STAGE_TARGETS: dict[str, tuple[str, str]] = {
    "optimization": ("execution.stage_opt", "run_optimization_stage"),
    "single_point": ("execution.stage_sp", "run_single_point_stage"),
    "frequency": ("execution.stage_freq", "run_frequency_stage"),
    "irc": ("execution.stage_irc", "run_irc_stage"),
    "scan": ("execution.stage_scan", "run_scan_stage"),
}

_DISABLE_ENV: dict[str, str] = {
    "frequency": "PYSCF_AUTO_DISABLE_FREQUENCY",
    "irc": "PYSCF_AUTO_DISABLE_IRC",
    "scan": "PYSCF_AUTO_DISABLE_SCAN",
    "qcschema": "PYSCF_AUTO_DISABLE_QCSCHEMA",
}


def _is_disabled(feature: str) -> bool:
    env_name = _DISABLE_ENV.get(feature)
    return bool(env_name and env_truthy(env_name))


def _resolve_callable(module_name: str, attr_name: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name, None)
    if not callable(fn):
        raise FeatureUnavailableError(
            f"Invalid plugin target: {module_name}.{attr_name} is not callable."
        )
    return fn


def load_stage_runner(stage_name: str) -> Callable[..., Any]:
    normalized = str(stage_name).strip().lower()
    if normalized not in _STAGE_TARGETS:
        available = ", ".join(sorted(_STAGE_TARGETS))
        raise FeatureUnavailableError(
            f"Unsupported stage '{stage_name}'. Available stages: {available}."
        )
    if _is_disabled(normalized):
        env_name = _DISABLE_ENV.get(normalized)
        raise FeatureUnavailableError(
            f"Stage '{normalized}' is disabled by {env_name}=1."
        )
    module_name, attr_name = _STAGE_TARGETS[normalized]
    try:
        return _resolve_callable(module_name, attr_name)
    except ModuleNotFoundError as exc:
        raise FeatureUnavailableError(
            f"Stage '{normalized}' is unavailable: missing dependency ({exc.name})."
        ) from exc


def run_stage(stage_name: str, *args: Any, **kwargs: Any) -> Any:
    runner = load_stage_runner(stage_name)
    return runner(*args, **kwargs)


def export_qcschema_result(*args: Any, **kwargs: Any) -> Any:
    if _is_disabled("qcschema"):
        logger.info("Skipping QCSchema export (PYSCF_AUTO_DISABLE_QCSCHEMA=1).")
        return None
    try:
        from qcschema_export import export_qcschema_result as _export_qcschema_result
    except ModuleNotFoundError as exc:
        logger.warning(
            "QCSchema export skipped: missing dependency (%s).",
            exc.name,
        )
        return None
    return _export_qcschema_result(*args, **kwargs)


__all__ = [
    "FeatureUnavailableError",
    "export_qcschema_result",
    "load_stage_runner",
    "run_stage",
]

