"""Main orchestration for pyscf_auto run-inp execution."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from app_config import AppConfig, load_app_config
from inp.parser import parse_inp_file
from notifier.events import (
    EVT_RUN_COMPLETED,
    EVT_RUN_FAILED,
    EVT_RUN_STARTED,
    make_event,
)
from notifier.notifier import Notifier, make_notify_callback
from .attempt_engine import run_attempts
from .run_lock import acquire_run_lock
from .state_machine import (
    finalize_state,
    load_or_create_state,
    load_state,
    new_state,
    save_state,
    write_report_files,
)

logger = logging.getLogger(__name__)
RETRY_INP_RE = re.compile(r"\.retry\d+$", re.IGNORECASE)


def cmd_run_inp(
    reaction_dir: str,
    max_retries: int | None = None,
    force: bool = False,
    json_output: bool = False,
    app_config: AppConfig | None = None,
) -> int:
    """Execute the run-inp command.

    1. Find the .inp file in the reaction directory
    2. Parse and validate it
    3. Check for existing completed results
    4. Acquire run lock
    5. Start notifier
    6. Run attempt loop
    7. Generate reports

    Args:
        reaction_dir: Path to the reaction directory.
        max_retries: Maximum retry attempts (overrides config).
        force: Re-run even if completed result exists.
        json_output: Output progress as JSON.
        app_config: Global application configuration.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    if app_config is None:
        app_config = load_app_config()

    reaction_dir = str(Path(reaction_dir).expanduser().resolve())
    if not os.path.isdir(reaction_dir):
        logger.error("Reaction directory not found: %s", reaction_dir)
        return 1
    allowed_root = str(Path(app_config.runtime.allowed_root).expanduser().resolve())
    if not _is_subpath(Path(reaction_dir), Path(allowed_root)):
        logger.error(
            "Reaction directory must be under allowed_root: %s (got %s)",
            allowed_root,
            reaction_dir,
        )
        return 1

    # Find .inp file
    inp_path = _find_inp_file(reaction_dir)
    if inp_path is None:
        logger.error("No .inp file found in %s", reaction_dir)
        return 1

    # Parse .inp file
    try:
        inp_config = parse_inp_file(inp_path)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Failed to parse %s: %s", inp_path, exc)
        return 1

    logger.info("Parsed input: %s (%s %s/%s)", inp_path,
                inp_config.job_type, inp_config.functional, inp_config.basis)

    # Resolve max retries
    if max_retries is None:
        max_retries = app_config.runtime.default_max_retries
    max_retries = max(0, int(max_retries))

    if not force:
        completed_out = _existing_completed_out(inp_path)
        if completed_out is not None:
            state = new_state(reaction_dir, str(inp_path), max_retries)
            state = finalize_state(
                state,
                "completed",
                analyzer_status="completed",
                reason="existing_out_completed",
                resumed=None,
                extra={
                    "last_out_path": completed_out["out_path"],
                    "skipped_execution": True,
                    "last_attempt_status": "completed",
                },
            )
            save_state(reaction_dir, state)
            write_report_files(state, reaction_dir)
            payload = _build_run_payload(reaction_dir, inp_path, state)
            _emit(payload, as_json=json_output)
            return 0

    # Setup notifier
    notifier = Notifier(app_config.monitoring)
    notifier.start()
    notify_fn = make_notify_callback(notifier)

    exit_code = 1
    resumed = False
    state: dict[str, Any] | None = None
    try:
        with acquire_run_lock(reaction_dir):
            if force:
                state = new_state(reaction_dir, str(inp_path), max_retries)
                save_state(reaction_dir, state)
                resumed = False
            else:
                state, resumed = load_or_create_state(
                    reaction_dir=reaction_dir,
                    selected_inp=str(inp_path),
                    max_retries=max_retries,
                )

            # Send run_started event
            notify_fn(make_event(
                EVT_RUN_STARTED,
                state["run_id"],
                dir=os.path.basename(reaction_dir),
            ))

            # Start heartbeat
            notifier.start_heartbeat(
                lambda: {
                    "run_id": state["run_id"],
                    "status": state.get("status", "running"),
                    "attempts": state.get("attempts", []),
                    "elapsed_sec": int(
                        time.time()
                        - time.mktime(
                            time.strptime(
                                state["started_at"][:19], "%Y-%m-%dT%H:%M:%S"
                            )
                        )
                    )
                    if state.get("started_at")
                    else 0,
                }
            )

            # Run attempts
            state = run_attempts(
                state=state,
                inp_config=inp_config,
                reaction_dir=reaction_dir,
                max_retries=max_retries,
                notify_fn=notify_fn,
                resumed=resumed,
            )

            # Determine exit code
            if state.get("status") == "completed":
                exit_code = 0
                notify_fn(make_event(
                    EVT_RUN_COMPLETED,
                    state["run_id"],
                    attempts=len(state["attempts"]),
                    reason=state.get("final_result", {}).get("reason", ""),
                ))
            else:
                notify_fn(make_event(
                    EVT_RUN_FAILED,
                    state["run_id"],
                    status=state.get("status", "failed"),
                    reason=state.get("final_result", {}).get("reason", ""),
                ))

    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Run interrupted by user.")

        if state is not None:
            state = finalize_state(state, "interrupted", reason="keyboard_interrupt")
            save_state(reaction_dir, state)
            exit_code = 130
        else:
            return 130
    finally:
        notifier.stop()
        if state is not None:
            write_report_files(state, reaction_dir)

    payload = _build_run_payload(reaction_dir, inp_path, state)
    _emit(payload, as_json=json_output)
    return exit_code


def cmd_status(
    reaction_dir: str,
    json_output: bool = False,
    app_config: AppConfig | None = None,
) -> int:
    """Display the status of a run in a reaction directory.

    Returns:
        Exit code (0 = found, 1 = not found).
    """
    if app_config is None:
        app_config = load_app_config()

    reaction_dir = str(Path(reaction_dir).expanduser().resolve())
    allowed_root = str(Path(app_config.runtime.allowed_root).expanduser().resolve())
    if not _is_subpath(Path(reaction_dir), Path(allowed_root)):
        logger.error(
            "Reaction directory must be under allowed_root: %s (got %s)",
            allowed_root,
            reaction_dir,
        )
        return 1

    state = load_state(reaction_dir)

    if state is None:
        logger.error("No run state found in %s", reaction_dir)
        return 1

    if json_output:
        print(json.dumps(state, indent=2))
    else:
        payload = _build_status_payload(reaction_dir, state)
        _emit(payload, as_json=False)

    return 0


def _find_inp_file(reaction_dir: str) -> str | None:
    """Find the most recently modified .inp file in the directory.

    Filters out .retryNN.inp files if a base .inp exists.
    """
    pattern = os.path.join(reaction_dir, "*.inp")
    inp_files = glob.glob(pattern)

    if not inp_files:
        return None

    # Filter out retry files
    base_files = [f for f in inp_files if ".retry" not in os.path.basename(f)]
    if base_files:
        inp_files = base_files

    # Return the most recently modified
    return max(inp_files, key=os.path.getmtime)


def _existing_completed_out(selected_inp: str) -> dict[str, str] | None:
    selected_path = Path(selected_inp)
    base_stem = RETRY_INP_RE.sub("", selected_path.stem) or selected_path.stem

    out_candidates = list(selected_path.parent.glob(f"{base_stem}.out"))
    out_candidates.extend(selected_path.parent.glob(f"{base_stem}.retry*.out"))
    out_candidates.sort(key=lambda p: (p.stat().st_mtime_ns, p.name.lower()), reverse=True)

    seen: set[Path] = set()
    for out_path in out_candidates:
        resolved = out_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not _looks_like_completed_out(out_path):
            continue
        return {"out_path": str(out_path)}
    return None


def _looks_like_completed_out(out_path: Path) -> bool:
    try:
        text = out_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    lowered = text.lower()
    completion_markers = (
        "normal termination",
        "terminated normally",
        "calculation completed",
        "status: completed",
    )
    return any(marker in lowered for marker in completion_markers)


def _is_subpath(path: Path, root: Path) -> bool:
    """Return True if path is root itself or located under root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    """Emit command result payload in text or JSON."""
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    for key in [
        "status",
        "reaction_dir",
        "selected_inp",
        "attempt_count",
        "reason",
        "run_state",
        "report_json",
        "report_md",
    ]:
        if key in payload:
            print(f"{key}: {payload[key]}")


def _build_status_payload(
    reaction_dir: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    final = state.get("final_result")
    if not isinstance(final, dict):
        final = {}
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        attempts = []

    return {
        "status": state.get("status", ""),
        "reaction_dir": reaction_dir,
        "selected_inp": state.get("selected_inp", ""),
        "attempt_count": len(attempts),
        "reason": final.get("reason", ""),
        "run_state": os.path.join(reaction_dir, "run_state.json"),
    }


def _build_run_payload(
    reaction_dir: str,
    selected_inp: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    payload = _build_status_payload(reaction_dir, state)
    payload["selected_inp"] = state.get("selected_inp", selected_inp)
    payload["report_json"] = os.path.join(reaction_dir, "run_report.json")
    payload["report_md"] = os.path.join(reaction_dir, "run_report.md")
    return payload
