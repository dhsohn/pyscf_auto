"""Core retry loop for running PySCF calculations with automatic retry."""

from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from inp.parser import InpConfig, inp_config_to_dict, inp_config_to_xyz_content
from run_opt_config import build_run_config
from .retry_strategies import apply_retry_strategy
from .state_machine import (
    decide_attempt_outcome,
    RunState,
    finalize_state,
    record_attempt,
    save_state,
    update_status,
)
from .types import AttemptRecord

logger = logging.getLogger(__name__)


def run_attempts(
    state: RunState,
    inp_config: InpConfig,
    reaction_dir: str,
    max_retries: int,
    notify_fn: Callable[[dict[str, Any]], None],
    resumed: bool = False,
) -> RunState:
    """Execute the calculation with automatic retry on failure.

    This is the core retry loop that:
    1. Converts InpConfig to RunConfig
    2. Runs the calculation through the execution engine
    3. Analyzes the result
    4. Retries with modified config if needed
    5. Records each attempt in the state

    Args:
        state: Current run state.
        inp_config: Parsed .inp file configuration.
        reaction_dir: Path to the reaction directory.
        max_retries: Maximum number of retry attempts.
        notify_fn: Callback for sending notification events.
        resumed: Whether this run is resuming from an in-progress state.

    Returns:
        Updated run state after all attempts.
    """
    max_attempts = 1 + max_retries
    base_config_dict = inp_config_to_dict(inp_config)
    current_config_dict = base_config_dict

    # Prepare xyz file
    xyz_path = _prepare_xyz_file(inp_config, reaction_dir)

    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
        state["attempts"] = attempts

    if resumed and attempts:
        last_attempt = attempts[-1]
        if not isinstance(last_attempt, dict):
            last_attempt = {}
        retries_used = max(0, len(attempts) - 1)
        analyzer_status = str(last_attempt.get("analyzer_status", "")).strip()
        analyzer_reason = str(last_attempt.get("analyzer_reason", "")).strip()
        decision = decide_attempt_outcome(
            analyzer_status=analyzer_status,
            analyzer_reason=analyzer_reason,
            retries_used=retries_used,
            max_retries=max_retries,
        )
        if decision is not None:
            logger.info(
                "Resume detected terminal previous attempt: status=%s, reason=%s",
                decision.analyzer_status,
                decision.reason,
            )
            state = finalize_state(
                state,
                decision.run_status,
                analyzer_status=decision.analyzer_status,
                reason=decision.reason,
                energy=last_attempt.get("energy"),
                resumed=resumed,
                extra={"last_attempt_status": decision.analyzer_status},
            )
            save_state(reaction_dir, state)
            return state

    start_attempt = len(attempts) + 1
    if start_attempt > max_attempts:
        last_attempt = attempts[-1] if attempts else {}
        if not isinstance(last_attempt, dict):
            last_attempt = {}
        state = finalize_state(
            state,
            "failed",
            analyzer_status=str(last_attempt.get("analyzer_status", "unknown")),
            reason="retry_limit_reached",
            energy=last_attempt.get("energy"),
            resumed=resumed,
            extra={
                "last_attempt_status": str(
                    last_attempt.get("analyzer_status", "unknown"),
                ),
            },
        )
        save_state(reaction_dir, state)
        return state

    state = update_status(state, "running" if start_attempt == 1 else "retrying")
    save_state(reaction_dir, state)

    for attempt_num in range(start_attempt, max_attempts + 1):
        attempt_dir = os.path.join(reaction_dir, f"attempt_{attempt_num:03d}")
        os.makedirs(attempt_dir, exist_ok=True)

        # Apply retry strategy if not first attempt
        patch_actions: list[str] = []
        if attempt_num > 1:
            last_attempt = state["attempts"][-1] if state["attempts"] else {}
            failure_reason = last_attempt.get("analyzer_reason", "unknown")
            current_config_dict, patch_actions = apply_retry_strategy(
                base_config_dict, attempt_num, failure_reason
            )
            state = update_status(state, "retrying")
            save_state(reaction_dir, state)

        logger.info(
            "Starting attempt %d/%d%s",
            attempt_num,
            max_attempts,
            f" (patches: {patch_actions})" if patch_actions else "",
        )

        attempt_record = _run_single_attempt(
            config_dict=current_config_dict,
            xyz_path=xyz_path,
            attempt_dir=attempt_dir,
            attempt_num=attempt_num,
            patch_actions=patch_actions,
        )

        state = record_attempt(state, attempt_record)
        save_state(reaction_dir, state)

        # Notify attempt completed
        from notifier.events import EVT_ATTEMPT_COMPLETED, make_event

        notify_fn(make_event(
            EVT_ATTEMPT_COMPLETED,
            state["run_id"],
            attempt=attempt_num,
            status=attempt_record.get("analyzer_status", "unknown"),
            event_suffix=f"attempt_{attempt_num}",
        ))

        # Check if completed
        analyzer_status = attempt_record.get("analyzer_status", "")
        if analyzer_status == "completed":
            state = finalize_state(
                state,
                "completed",
                analyzer_status="completed",
                reason="normal_termination",
                energy=attempt_record.get("energy"),
                resumed=resumed,
                extra={"last_attempt_status": "completed"},
            )
            save_state(reaction_dir, state)
            return state

        # Check for non-recoverable errors
        if analyzer_status in (
            "error_multiplicity_impossible",
            "error_basis_not_found",
            "error_functional_not_found",
        ):
            state = finalize_state(
                state,
                "failed",
                analyzer_status=analyzer_status,
                reason=attempt_record.get("analyzer_reason", "non_recoverable"),
                resumed=resumed,
                extra={"last_attempt_status": analyzer_status},
            )
            save_state(reaction_dir, state)
            return state

        logger.warning(
            "Attempt %d failed: %s (%s)",
            attempt_num,
            analyzer_status,
            attempt_record.get("analyzer_reason", ""),
        )

    # Exhausted all retries
    last_attempt = state["attempts"][-1] if state["attempts"] else {}
    state = finalize_state(
        state,
        "failed",
        analyzer_status=last_attempt.get("analyzer_status", "unknown"),
        reason="retry_limit_reached",
        resumed=resumed,
        extra={
            "last_attempt_status": str(last_attempt.get("analyzer_status", "unknown")),
        },
    )
    save_state(reaction_dir, state)
    return state


def _prepare_xyz_file(inp_config: InpConfig, reaction_dir: str) -> str:
    """Prepare a .xyz file for the calculation.

    If the geometry is inline, writes a temporary _geometry.xyz file.
    If it references an external file, returns that path.
    """
    if inp_config.xyz_source != "inline":
        return inp_config.xyz_source

    xyz_path = os.path.join(reaction_dir, "_geometry.xyz")
    content = inp_config_to_xyz_content(inp_config)
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(content)
    return xyz_path


def _run_single_attempt(
    config_dict: dict[str, Any],
    xyz_path: str,
    attempt_dir: str,
    attempt_num: int,
    patch_actions: list[str],
) -> AttemptRecord:
    """Execute a single calculation attempt.

    This bridges the retry runner to the execution engine.

    Returns:
        An AttemptRecord with the outcome.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    attempt: AttemptRecord = {
        "index": attempt_num,
        "started_at": started_at,
        "patch_actions": patch_actions,
        "attempt_dir": attempt_dir,
    }

    try:
        # Build RunConfig from the config dictionary
        run_config = build_run_config(config_dict)

        # Create an args-like namespace for the execution engine entrypoint.
        args = _build_engine_args(xyz_path, attempt_dir)

        # Import and run the execution engine.
        import execution

        config_raw = __import__("json").dumps(config_dict, indent=2)
        execution.run(args, run_config, config_raw, None, False)

        # If we get here, the calculation completed without exception
        attempt["analyzer_status"] = "completed"
        attempt["analyzer_reason"] = "normal_termination"
        attempt["converged"] = True

        # Try to extract energy from metadata
        energy = _extract_energy_from_attempt(attempt_dir)
        if energy is not None:
            attempt["energy"] = energy

    except Exception as exc:
        error_msg = str(exc)
        attempt["analyzer_status"] = _classify_error(error_msg)
        attempt["analyzer_reason"] = error_msg[:500]
        attempt["converged"] = False
        attempt["error"] = traceback.format_exc()[:2000]
        logger.warning("Attempt %d failed: %s", attempt_num, error_msg)

    attempt["ended_at"] = datetime.now(timezone.utc).isoformat()
    return attempt


def _build_engine_args(xyz_path: str, run_dir: str):
    """Build an argparse-compatible namespace for the execution engine."""
    from types import SimpleNamespace

    return SimpleNamespace(
        xyz_file=xyz_path,
        config=None,
        solvent_map="solvent_dielectric.json",
        run_dir=run_dir,
        run_id=None,
        resume=None,
        profile=False,
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


def _classify_error(error_msg: str) -> str:
    """Classify an error message into an analyzer status."""
    lower = error_msg.lower()
    if "scf" in lower and ("converge" in lower or "convergence" in lower):
        return "error_scf_convergence"
    if "not converged" in lower:
        return "error_opt_not_converged"
    if "memory" in lower or "memoryerror" in lower:
        return "error_memory"
    if "basis" in lower and "not found" in lower:
        return "error_basis_not_found"
    if "functional" in lower and "not found" in lower:
        return "error_functional_not_found"
    if "multiplicity" in lower or "spin" in lower:
        return "error_multiplicity_impossible"
    return "unknown_failure"


def _extract_energy_from_attempt(attempt_dir: str) -> float | None:
    """Try to extract the final energy from attempt metadata."""
    import json

    metadata_path = os.path.join(attempt_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("energy") or metadata.get("final_energy")
    except (OSError, json.JSONDecodeError):
        return None
