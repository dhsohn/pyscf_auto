"""Run state machine and persistence for pyscf_auto.

Manages ``run_state.json`` in each reaction directory with attempt history,
resume-aware state transitions, and report generation.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import AttemptRecord, RunState

_STATE_FILE = "run_state.json"
_REPORT_JSON = "run_report.json"
_REPORT_MD = "run_report.md"

RESUMABLE_RUN_STATUSES = {"running", "retrying"}
NON_RECOVERABLE_ANALYZER_STATUSES = {
    "error_multiplicity_impossible",
    "error_basis_not_found",
    "error_functional_not_found",
}


@dataclass(frozen=True)
class AttemptDecision:
    """Terminal decision inferred from the last attempt result."""

    run_status: str
    reason: str
    analyzer_status: str


def generate_run_id() -> str:
    """Generate a unique run ID: ``run_YYYYMMDD_HHMMSS_<8hex>``."""
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"run_{ts}_{suffix}"


def new_state(
    reaction_dir: str,
    selected_inp: str,
    max_retries: int = 5,
) -> RunState:
    """Create a fresh run state."""
    return {
        "run_id": generate_run_id(),
        "reaction_dir": str(Path(reaction_dir).resolve()),
        "selected_inp": selected_inp,
        "max_retries": max(0, int(max_retries)),
        "status": "created",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "attempts": [],
        "final_result": None,
    }


def load_state(reaction_dir: str) -> RunState | None:
    """Load run state from ``run_state.json`` in the reaction directory.

    Returns ``None`` if no state file exists.
    """
    state_path = os.path.join(reaction_dir, _STATE_FILE)
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_state(reaction_dir: str, state: RunState) -> None:
    """Persist run state atomically (tmp + rename + fsync)."""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path = os.path.join(reaction_dir, _STATE_FILE)
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, state_path)


def update_status(state: RunState, status: str) -> RunState:
    """Update the status field and timestamp."""
    state["status"] = status
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return state


def record_attempt(
    state: RunState,
    attempt: AttemptRecord,
) -> RunState:
    """Append an attempt record to the state."""
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
        state["attempts"] = attempts
    attempts.append(attempt)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return state


def finalize_state(
    state: RunState,
    status: str,
    analyzer_status: str | None = None,
    reason: str | None = None,
    energy: float | None = None,
    resumed: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> RunState:
    """Mark the run as terminal (completed or failed)."""
    state["status"] = status
    final_result: dict[str, Any] = {
        "status": status,
        "analyzer_status": analyzer_status or status,
        "reason": reason or "",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "energy": energy,
    }
    if resumed is not None:
        final_result["resumed"] = resumed
    if isinstance(extra, dict):
        protected = {"status", "analyzer_status", "reason", "completed_at", "energy"}
        final_result.update({k: v for k, v in extra.items() if k not in protected})
    state["final_result"] = final_result
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return state


def _resolve_path_text(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def state_matches_selected(state: RunState, selected_inp: str) -> bool:
    """Check if stored selected input path matches the current selection."""
    previous = state.get("selected_inp")
    if not isinstance(previous, str) or not previous.strip():
        return False
    try:
        return _resolve_path_text(previous) == _resolve_path_text(selected_inp)
    except OSError:
        return False


def load_or_create_state(
    reaction_dir: str,
    selected_inp: str,
    max_retries: int,
) -> tuple[RunState, bool]:
    """Load existing state when resumable, or create a fresh one.

    Returns:
        ``(state, resumed)`` where ``resumed`` is ``True`` only when an existing
        state for the same selected input is in a resumable status.
    """
    state = load_state(reaction_dir)
    resumed = False
    if not state or not state_matches_selected(state, selected_inp):
        state = new_state(reaction_dir, selected_inp, max_retries=max_retries)
    elif str(state.get("status")) in RESUMABLE_RUN_STATUSES:
        resumed = True
    else:
        state = new_state(reaction_dir, selected_inp, max_retries=max_retries)

    state["max_retries"] = max(0, int(max_retries))
    if not isinstance(state.get("attempts"), list):
        state["attempts"] = []
    save_state(reaction_dir, state)
    return state, resumed


def decide_attempt_outcome(
    *,
    analyzer_status: str,
    analyzer_reason: str,
    retries_used: int,
    max_retries: int,
) -> AttemptDecision | None:
    """Determine whether the latest attempt already implies a terminal run."""
    status_text = str(analyzer_status or "").strip()
    reason_text = str(analyzer_reason or "").strip() or "unknown"

    if status_text == "completed":
        return AttemptDecision(
            run_status="completed",
            reason=reason_text,
            analyzer_status="completed",
        )
    if status_text in NON_RECOVERABLE_ANALYZER_STATUSES:
        return AttemptDecision(
            run_status="failed",
            reason=reason_text,
            analyzer_status=status_text,
        )
    if retries_used >= max(0, int(max_retries)):
        return AttemptDecision(
            run_status="failed",
            reason="retry_limit_reached",
            analyzer_status=status_text or "unknown_failure",
        )
    return None


def write_report_files(state: RunState, reaction_dir: str) -> None:
    """Generate ``run_report.json`` and ``run_report.md``."""
    _write_report_json(state, reaction_dir)
    _write_report_md(state, reaction_dir)


def _write_report_json(state: RunState, reaction_dir: str) -> None:
    """Write the JSON report."""
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    report = {
        "run_id": state["run_id"],
        "reaction_dir": state["reaction_dir"],
        "status": state["status"],
        "attempt_count": len(attempts),
        "max_retries": state["max_retries"],
        "started_at": state["started_at"],
        "updated_at": state["updated_at"],
        "attempts": attempts,
        "final_result": state.get("final_result"),
    }
    report_path = os.path.join(reaction_dir, _REPORT_JSON)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _write_report_md(state: RunState, reaction_dir: str) -> None:
    """Write the Markdown report."""
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        attempts = []

    lines = [
        "# PySCF Run Report",
        "",
        f"- run_id: `{state['run_id']}`",
        f"- status: `{state['status']}`",
        f"- attempt_count: `{len(attempts)}`",
        f"- max_retries: `{state['max_retries']}`",
        f"- started_at: `{state['started_at']}`",
        "",
        "## Attempts",
        "",
        "| # | status | reason | energy | patch_actions |",
        "|--:|--------|--------|--------|---------------|",
    ]

    for i, attempt in enumerate(attempts, start=1):
        status = attempt.get("analyzer_status", "?")
        reason = attempt.get("analyzer_reason", "")
        energy = attempt.get("energy")
        energy_str = f"{energy:.8f}" if energy is not None else "-"
        patch_actions = attempt.get("patch_actions")
        patches = ", ".join(patch_actions) if isinstance(patch_actions, list) else ""
        lines.append(f"| {i} | `{status}` | {reason} | {energy_str} | {patches} |")

    final = state.get("final_result")
    if final:
        lines.extend([
            "",
            "## Final Result",
            "",
            f"- status: `{final.get('status', '?')}`",
            f"- reason: {final.get('reason', '-')}",
        ])
        if final.get("analyzer_status"):
            lines.append(f"- analyzer_status: `{final['analyzer_status']}`")
        if final.get("energy") is not None:
            lines.append(f"- energy: {final['energy']:.8f}")
        if final.get("resumed") is not None:
            lines.append(f"- resumed: `{bool(final['resumed'])}`")
        if final.get("last_attempt_status"):
            lines.append(f"- last_attempt_status: `{final['last_attempt_status']}`")

    lines.append("")
    report_path = os.path.join(reaction_dir, _REPORT_MD)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def is_completed(state: RunState) -> bool:
    """Check if the run has reached a terminal state."""
    return state.get("status") in ("completed", "failed", "interrupted")
