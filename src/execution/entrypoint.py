"""Stable runner-facing entrypoint for execution engine calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from run_opt_config import build_run_config

from . import run as run_execution


@dataclass(frozen=True)
class ExecutionAttemptResult:
    """Execution result visible to the retry runner."""

    metadata_path: str
    metadata: dict[str, Any] | None


def _build_args_namespace(xyz_path: str, run_dir: str, *, run_id: str | None, profile: bool):
    return SimpleNamespace(
        xyz_file=xyz_path,
        config=None,
        solvent_map="solvent_dielectric.json",
        run_dir=run_dir,
        run_id=run_id,
        resume=None,
        profile=profile,
        queue_priority=0,
        queue_max_runtime=None,
        no_background=True,
        background=False,
        force_resume=False,
        scan_dimension=None,
        scan_grid=None,
        scan_mode=None,
        scan_result_csv=None,
        queue_runner=False,
    )


def _load_attempt_metadata(run_dir: str) -> dict[str, Any] | None:
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def execute_attempt(
    *,
    config_dict: dict[str, Any],
    xyz_path: str,
    run_dir: str,
    run_id: str | None = None,
    profile: bool = False,
) -> ExecutionAttemptResult:
    """Execute one attempt with a stable interface from runner -> execution."""
    run_config = build_run_config(config_dict)
    args = _build_args_namespace(
        xyz_path=xyz_path,
        run_dir=run_dir,
        run_id=run_id,
        profile=profile,
    )
    config_raw = json.dumps(config_dict, indent=2)
    run_execution(args, run_config, config_raw, None, False)
    metadata_path = os.path.join(run_dir, "metadata.json")
    metadata = _load_attempt_metadata(run_dir)
    return ExecutionAttemptResult(metadata_path=metadata_path, metadata=metadata)


__all__ = ["ExecutionAttemptResult", "execute_attempt"]

