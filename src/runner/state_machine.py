"""Run state machine and persistence for pyscf_auto.

Manages ``run_state.json`` in each reaction directory with full attempt
history and status tracking.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .types import AttemptRecord, RunState

_STATE_FILE = "run_state.json"
_REPORT_JSON = "run_report.json"
_REPORT_MD = "run_report.md"


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
        "max_retries": max_retries,
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
    state["attempts"].append(attempt)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return state


def finalize_state(
    state: RunState,
    status: str,
    analyzer_status: str | None = None,
    reason: str | None = None,
    energy: float | None = None,
) -> RunState:
    """Mark the run as terminal (completed or failed)."""
    state["status"] = status
    state["final_result"] = {
        "status": status,
        "analyzer_status": analyzer_status or status,
        "reason": reason or "",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "energy": energy,
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return state


def write_report_files(state: RunState, reaction_dir: str) -> None:
    """Generate ``run_report.json`` and ``run_report.md``."""
    _write_report_json(state, reaction_dir)
    _write_report_md(state, reaction_dir)


def _write_report_json(state: RunState, reaction_dir: str) -> None:
    """Write the JSON report."""
    report = {
        "run_id": state["run_id"],
        "reaction_dir": state["reaction_dir"],
        "status": state["status"],
        "attempt_count": len(state["attempts"]),
        "max_retries": state["max_retries"],
        "started_at": state["started_at"],
        "updated_at": state["updated_at"],
        "attempts": state["attempts"],
        "final_result": state.get("final_result"),
    }
    report_path = os.path.join(reaction_dir, _REPORT_JSON)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _write_report_md(state: RunState, reaction_dir: str) -> None:
    """Write the Markdown report."""
    lines = [
        "# PySCF Run Report",
        "",
        f"- run_id: `{state['run_id']}`",
        f"- status: `{state['status']}`",
        f"- attempt_count: `{len(state['attempts'])}`",
        f"- max_retries: `{state['max_retries']}`",
        f"- started_at: `{state['started_at']}`",
        "",
        "## Attempts",
        "",
        "| # | status | reason | energy | patch_actions |",
        "|--:|--------|--------|--------|---------------|",
    ]

    for i, attempt in enumerate(state["attempts"], start=1):
        status = attempt.get("analyzer_status", "?")
        reason = attempt.get("analyzer_reason", "")
        energy = attempt.get("energy")
        energy_str = f"{energy:.8f}" if energy is not None else "-"
        patches = ", ".join(attempt.get("patch_actions", []))
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
        if final.get("energy") is not None:
            lines.append(f"- energy: {final['energy']:.8f}")

    lines.append("")
    report_path = os.path.join(reaction_dir, _REPORT_MD)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def is_completed(state: RunState) -> bool:
    """Check if the run has reached a terminal state."""
    return state.get("status") in ("completed", "failed", "interrupted")
