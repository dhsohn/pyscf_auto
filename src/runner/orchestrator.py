"""Main orchestration for pyscf_auto run-inp workflow."""

from __future__ import annotations

import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from app_config import AppConfig, load_app_config
from inp.parser import InpConfig, parse_inp_file
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
    is_completed,
    load_state,
    new_state,
    save_state,
    write_report_files,
)

logger = logging.getLogger(__name__)


def cmd_run_inp(
    reaction_dir: str,
    max_retries: int | None = None,
    force: bool = False,
    json_output: bool = False,
    profile: bool = False,
    verbose: bool = False,
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
        profile: Enable profiling.
        verbose: Enable verbose logging.
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

    # Check for existing result
    existing_state = load_state(reaction_dir)
    if existing_state and is_completed(existing_state) and not force:
        status = existing_state.get("status", "unknown")
        logger.info(
            "Run already %s (run_id=%s). Use --force to re-run.",
            status,
            existing_state.get("run_id", "?"),
        )
        return 0 if status == "completed" else 1

    # Resolve max retries
    if max_retries is None:
        max_retries = app_config.runtime.default_max_retries

    # Create run state
    state = new_state(reaction_dir, str(inp_path), max_retries)
    save_state(reaction_dir, state)

    # Setup notifier
    notifier = Notifier(app_config.monitoring)
    notifier.start()
    notify_fn = make_notify_callback(notifier)

    # Send run_started event
    notify_fn(make_event(
        EVT_RUN_STARTED,
        state["run_id"],
        dir=os.path.basename(reaction_dir),
    ))

    exit_code = 1
    try:
        with acquire_run_lock(reaction_dir):
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
        from .state_machine import finalize_state

        state = finalize_state(state, "interrupted", reason="keyboard_interrupt")
        save_state(reaction_dir, state)
    finally:
        notifier.stop()
        write_report_files(state, reaction_dir)

    # Print summary
    _print_summary(state, json_output)
    return exit_code


def cmd_status(
    reaction_dir: str,
    json_output: bool = False,
) -> int:
    """Display the status of a run in a reaction directory.

    Returns:
        Exit code (0 = found, 1 = not found).
    """
    reaction_dir = str(Path(reaction_dir).expanduser().resolve())
    state = load_state(reaction_dir)

    if state is None:
        logger.error("No run state found in %s", reaction_dir)
        return 1

    if json_output:
        import json

        print(json.dumps(state, indent=2))
    else:
        _print_summary(state, json_output=False)

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


def _print_summary(state: dict[str, Any], json_output: bool) -> None:
    """Print a human-readable summary of the run state."""
    if json_output:
        import json

        print(json.dumps(state, indent=2))
        return

    run_id = state.get("run_id", "?")
    status = state.get("status", "unknown")
    attempts = state.get("attempts", [])
    final = state.get("final_result")

    print(f"\nRun: {run_id}")
    print(f"Status: {status}")
    print(f"Attempts: {len(attempts)}")

    if attempts:
        last = attempts[-1]
        print(f"Last attempt: {last.get('analyzer_status', '?')} "
              f"({last.get('analyzer_reason', '')})")

    if final:
        print(f"Final: {final.get('status', '?')} - {final.get('reason', '')}")
        if final.get("energy") is not None:
            print(f"Energy: {final['energy']:.8f}")

    print()
