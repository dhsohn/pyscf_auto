"""Cleanup helpers for organized output directories."""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runner.state_machine import load_state

logger = logging.getLogger(__name__)


@dataclass
class CleanupFileEntry:
    path: Path
    size_bytes: int


@dataclass
class CleanupPlan:
    reaction_dir: Path
    run_id: str
    files_to_remove: list[CleanupFileEntry] = field(default_factory=list)
    keep_count: int = 0
    total_remove_bytes: int = 0


@dataclass
class CleanupSkipReason:
    reaction_dir: str
    reason: str


@dataclass
class CleanupResult:
    reaction_dir: str
    run_id: str
    files_removed: int = 0
    bytes_freed: int = 0
    errors: list[str] = field(default_factory=list)


def _should_keep(
    file_path: Path,
    keep_extensions: set[str],
    keep_filenames: set[str],
    remove_patterns: list[str],
) -> bool:
    name = file_path.name
    for pattern in remove_patterns:
        if fnmatch.fnmatch(name, pattern):
            return False
    if name in keep_filenames:
        return True
    return file_path.suffix.lower() in keep_extensions


def _state_artifact_path_texts(state: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    selected_inp = state.get("selected_inp")
    if isinstance(selected_inp, str) and selected_inp.strip():
        texts.append(selected_inp)

    attempts = state.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            inp_path = attempt.get("inp_path")
            if isinstance(inp_path, str) and inp_path.strip():
                texts.append(inp_path)
            out_path = attempt.get("out_path")
            if isinstance(out_path, str) and out_path.strip():
                texts.append(out_path)

    final_result = state.get("final_result")
    if isinstance(final_result, dict):
        last_out_path = final_result.get("last_out_path")
        if isinstance(last_out_path, str) and last_out_path.strip():
            texts.append(last_out_path)

    return texts


def _artifact_candidates(path_text: str, reaction_dir: Path) -> list[Path]:
    raw = path_text.strip()
    if not raw:
        return []

    candidate = Path(raw)
    if candidate.is_absolute():
        return [candidate, reaction_dir / candidate.name]
    return [reaction_dir / candidate, reaction_dir / candidate.name]


def _collect_protected_artifacts(
    state: dict[str, Any],
    reaction_dir: Path,
) -> tuple[set[Path], set[str]]:
    protected_paths: set[Path] = set()
    protected_names: set[str] = set()

    for path_text in _state_artifact_path_texts(state):
        raw = path_text.strip()
        name = Path(raw).name
        if name:
            protected_names.add(name)

        for candidate in _artifact_candidates(raw, reaction_dir):
            if not candidate.exists() or not candidate.is_file():
                continue
            try:
                protected_paths.add(candidate.resolve())
            except OSError:
                continue

    return protected_paths, protected_names


def check_cleanup_eligibility(
    reaction_dir: Path,
) -> tuple[Any, CleanupSkipReason | None]:
    state = load_state(str(reaction_dir))
    if state is None:
        return None, CleanupSkipReason(str(reaction_dir), "state_missing_or_invalid")

    run_id = state.get("run_id")
    status = state.get("status")
    if not isinstance(run_id, str) or not run_id.strip():
        return None, CleanupSkipReason(str(reaction_dir), "state_schema_invalid")
    if not isinstance(status, str) or not status.strip():
        return None, CleanupSkipReason(str(reaction_dir), "state_schema_invalid")
    if status != "completed":
        return None, CleanupSkipReason(str(reaction_dir), "not_completed")

    return state, None


def compute_cleanup_plan(
    reaction_dir: Path,
    state: Any,
    keep_extensions: set[str],
    keep_filenames: set[str],
    remove_patterns: list[str],
) -> CleanupPlan:
    plan = CleanupPlan(reaction_dir=reaction_dir, run_id=str(state.get("run_id", "unknown")))
    protected_paths, protected_names = _collect_protected_artifacts(state, reaction_dir)

    for file_path in sorted(reaction_dir.iterdir()):
        if not file_path.is_file():
            continue
        try:
            resolved = file_path.resolve()
        except OSError:
            resolved = file_path

        if resolved in protected_paths or file_path.name in protected_names:
            plan.keep_count += 1
            continue

        if _should_keep(file_path, keep_extensions, keep_filenames, remove_patterns):
            plan.keep_count += 1
            continue

        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0
        plan.files_to_remove.append(CleanupFileEntry(path=file_path, size_bytes=size))
        plan.total_remove_bytes += size

    return plan


def plan_cleanup_single(
    reaction_dir: Path,
    keep_extensions: set[str],
    keep_filenames: set[str],
    remove_patterns: list[str],
) -> tuple[CleanupPlan | None, CleanupSkipReason | None]:
    state, skip = check_cleanup_eligibility(reaction_dir)
    if skip is not None:
        return None, skip
    assert state is not None

    plan = compute_cleanup_plan(
        reaction_dir,
        state,
        keep_extensions=keep_extensions,
        keep_filenames=keep_filenames,
        remove_patterns=remove_patterns,
    )
    if not plan.files_to_remove:
        return None, CleanupSkipReason(str(reaction_dir), "nothing_to_clean")
    return plan, None


def plan_cleanup_root_scan(
    organized_root: Path,
    keep_extensions: set[str],
    keep_filenames: set[str],
    remove_patterns: list[str],
) -> tuple[list[CleanupPlan], list[CleanupSkipReason]]:
    plans: list[CleanupPlan] = []
    skips: list[CleanupSkipReason] = []
    root_resolved = organized_root.resolve()
    index_root = (organized_root / "index").resolve()

    if not organized_root.exists():
        return plans, skips

    for dirpath, dirnames, filenames in os.walk(organized_root, followlinks=False):
        current_dir = Path(dirpath)

        # Never descend into symlink directories.
        dirnames[:] = [name for name in dirnames if not (current_dir / name).is_symlink()]

        try:
            current_resolved = current_dir.resolve()
        except OSError:
            skips.append(CleanupSkipReason(str(current_dir), "resolve_failed"))
            dirnames[:] = []
            continue

        if not _is_subpath(current_resolved, root_resolved):
            skips.append(CleanupSkipReason(str(current_dir), "outside_organized_root"))
            dirnames[:] = []
            continue

        if current_resolved == index_root or _is_subpath(current_resolved, index_root):
            dirnames[:] = []
            continue

        if "run_state.json" not in filenames:
            continue

        state_file = current_dir / "run_state.json"
        if state_file.is_symlink():
            skips.append(CleanupSkipReason(str(current_dir), "symlink_state_file"))
            continue

        plan, skip = plan_cleanup_single(
            current_resolved,
            keep_extensions=keep_extensions,
            keep_filenames=keep_filenames,
            remove_patterns=remove_patterns,
        )
        if plan is not None:
            plans.append(plan)
        if skip is not None:
            skips.append(skip)

    return plans, skips


def execute_cleanup(plan: CleanupPlan) -> CleanupResult:
    result = CleanupResult(reaction_dir=str(plan.reaction_dir), run_id=plan.run_id)
    for entry in plan.files_to_remove:
        try:
            entry.path.unlink()
            result.files_removed += 1
            result.bytes_freed += entry.size_bytes
        except OSError as exc:
            result.errors.append(f"{entry.path.name}: {exc}")
            logger.error("Failed to remove %s: %s", entry.path, exc)
    return result


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
