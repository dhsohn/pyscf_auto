"""Type definitions for the runner system."""

from __future__ import annotations

from typing import TypedDict


class AttemptRecord(TypedDict, total=False):
    """Record of a single calculation attempt."""

    index: int
    started_at: str
    ended_at: str
    analyzer_status: str
    analyzer_reason: str
    patch_actions: list[str]
    energy: float | None
    converged: bool | None
    error: str | None
    attempt_dir: str


class RunFinalResult(TypedDict, total=False):
    """Terminal result of a run."""

    status: str
    analyzer_status: str
    reason: str
    completed_at: str
    energy: float | None
    resumed: bool
    last_attempt_status: str


class RunState(TypedDict, total=False):
    """Full run state persisted to run_state.json."""

    run_id: str
    reaction_dir: str
    selected_inp: str
    max_retries: int
    status: str
    started_at: str
    updated_at: str
    attempts: list[AttemptRecord]
    final_result: RunFinalResult | None
