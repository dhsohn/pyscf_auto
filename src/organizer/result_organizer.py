"""Organize completed runs into a clean directory structure."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.state_machine import save_state, write_report_files
from .index_store import (
    acquire_index_lock,
    append_record,
    find_by_job_type,
    find_by_run_id,
    load_index,
    now_utc_iso,
    rebuild_index,
)
from .molecule_key import derive_molecule_key

logger = logging.getLogger(__name__)

# Map job types to directory names
_JOB_TYPE_DIRS = {
    "optimization": "optimization",
    "single_point": "single_point",
    "frequency": "frequency",
    "irc": "irc",
    "scan": "scan",
}

_ORGANIZE_TERMINAL_STATUSES = {"completed"}


@dataclass(frozen=True)
class SkipReason:
    reaction_dir: str
    reason: str


@dataclass(frozen=True)
class OrganizePlan:
    source_dir: Path
    target_abs_path: Path
    target_rel_path: str
    run_id: str
    job_type: str
    molecule_key: str
    status: str
    state: dict[str, Any]


def organize_run(
    reaction_dir: str,
    organized_root: str,
    apply: bool = False,
) -> dict[str, Any] | None:
    """Organize one reaction directory."""
    source_dir = Path(reaction_dir).expanduser().resolve()
    organized_root_path = Path(organized_root).expanduser().resolve()

    plan, _skip = plan_single(source_dir, organized_root_path)
    if plan is None:
        return None

    if not apply:
        return _plan_to_result(plan, applied=False, action="dry_run")

    applied, failures = apply_plans([plan], organized_root_path)
    if applied:
        return applied[0]
    if failures:
        reason = failures[0].get("reason", "apply_failed")
        return _plan_to_result(plan, applied=False, action="failed", reason=reason)
    return None


def organize_all(
    root: str,
    organized_root: str,
    apply: bool = False,
) -> list[dict[str, Any]]:
    """Organize all eligible runs under the root directory."""
    root_path = Path(root).expanduser().resolve()
    organized_root_path = Path(organized_root).expanduser().resolve()

    plans, _skips = plan_root_scan(root_path, organized_root_path)
    if not apply:
        return [_plan_to_result(plan, applied=False, action="dry_run") for plan in plans]

    applied, failures = apply_plans(plans, organized_root_path)
    if not failures:
        return applied

    failure_by_run_id = {
        str(item.get("run_id", "")): str(item.get("reason", "apply_failed"))
        for item in failures
    }
    merged: list[dict[str, Any]] = []
    by_run_id = {result["run_id"]: result for result in applied}
    for plan in plans:
        if plan.run_id in by_run_id:
            merged.append(by_run_id[plan.run_id])
            continue
        reason = failure_by_run_id.get(plan.run_id, "apply_failed")
        merged.append(
            _plan_to_result(plan, applied=False, action="failed", reason=reason),
        )
    return merged


def find_organized_runs(
    organized_root: str,
    run_id: str | None = None,
    job_type: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Search organized outputs by run_id/job_type."""
    organized_root_path = Path(organized_root).expanduser().resolve()

    if run_id:
        record = find_by_run_id(organized_root_path, run_id)
        return [record] if record is not None else []

    effective_limit = max(0, int(limit or 0))
    if job_type:
        return find_by_job_type(organized_root_path, job_type, limit=effective_limit)

    records = list(load_index(organized_root_path).values())
    if effective_limit > 0:
        return records[:effective_limit]
    return records


def rebuild_organized_index(organized_root: str) -> int:
    """Rebuild index records by scanning organized outputs."""
    organized_root_path = Path(organized_root).expanduser().resolve()
    return rebuild_index(organized_root_path)


def plan_single(
    reaction_dir: Path,
    organized_root: Path,
) -> tuple[OrganizePlan | None, SkipReason | None]:
    """Build an organize plan for one reaction directory."""
    state = _load_state(reaction_dir)
    if state is None:
        return None, SkipReason(str(reaction_dir), "state_missing_or_invalid")

    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return None, SkipReason(str(reaction_dir), "state_schema_invalid")

    status = str(state.get("status", "")).strip()
    if status not in _ORGANIZE_TERMINAL_STATUSES:
        return None, SkipReason(str(reaction_dir), "not_completed")

    job_type = _infer_job_type(reaction_dir)
    job_type_dir = _JOB_TYPE_DIRS.get(job_type, job_type or "other")

    atom_spec = _extract_atom_spec(reaction_dir)
    tag = _extract_tag(reaction_dir)
    molecule_key = derive_molecule_key(
        atom_spec or "",
        tag=tag,
        reaction_dir_name=reaction_dir.name,
    )

    target_rel_path = f"{job_type_dir}/{molecule_key}/{run_id}"
    target_abs_path = organized_root / target_rel_path

    return (
        OrganizePlan(
            source_dir=reaction_dir,
            target_abs_path=target_abs_path,
            target_rel_path=target_rel_path,
            run_id=run_id,
            job_type=job_type_dir,
            molecule_key=molecule_key,
            status=status,
            state=state,
        ),
        None,
    )


def plan_root_scan(
    root: Path,
    organized_root: Path,
) -> tuple[list[OrganizePlan], list[SkipReason]]:
    """Build organize plans for every directory under root."""
    plans: list[OrganizePlan] = []
    skips: list[SkipReason] = []

    if not root.exists() or not root.is_dir():
        logger.error("Root directory not found: %s", root)
        return plans, skips

    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if not entry.is_dir() or entry.is_symlink():
            continue
        if entry == organized_root or _is_subpath(entry, organized_root):
            continue
        plan, skip = plan_single(entry, organized_root)
        if plan is not None:
            plans.append(plan)
        elif skip is not None:
            skips.append(skip)
    return plans, skips


def check_conflict(plan: OrganizePlan, index: dict[str, dict[str, Any]]) -> str | None:
    """Return conflict reason or None if safe to apply."""
    existing = index.get(plan.run_id)
    if existing is not None:
        if existing.get("organized_path") == plan.target_rel_path:
            return "already_organized"
        return "index_conflict"
    if plan.target_abs_path.exists():
        return "path_occupied"
    return None


def apply_plans(
    plans: list[OrganizePlan],
    organized_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply organize plans with index lock and rollback on failure."""
    applied: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for plan in plans:
        moved = False
        try:
            with acquire_index_lock(organized_root):
                index = load_index(organized_root)
                conflict = check_conflict(plan, index)
                if conflict == "already_organized":
                    applied.append(
                        _plan_to_result(
                            plan,
                            applied=False,
                            action="skipped",
                            reason=conflict,
                        ),
                    )
                    continue
                if conflict:
                    failures.append({"run_id": plan.run_id, "reason": conflict})
                    continue

                execute_move(plan)
                moved = True
                state_after_move = sync_state_after_move(plan)
                append_record(organized_root, _build_index_record(plan, state_after_move))
                applied.append(_plan_to_result(plan, applied=True, action="moved"))
        except Exception as exc:
            logger.error("Organize apply failed for %s: %s", plan.run_id, exc)
            reason = f"apply_failed:{exc}"
            if moved:
                try:
                    rollback_move(plan)
                    sync_state_after_rollback(plan)
                    reason = f"{reason};rolled_back=true"
                except Exception as rollback_exc:
                    reason = f"{reason};rollback_failed:{rollback_exc}"
            failures.append({"run_id": plan.run_id, "reason": reason})

    return applied, failures


def _plan_to_result(
    plan: OrganizePlan,
    *,
    applied: bool,
    action: str,
    reason: str | None = None,
) -> dict[str, Any]:
    result = {
        "source": str(plan.source_dir),
        "target": str(plan.target_abs_path),
        "target_rel_path": plan.target_rel_path,
        "run_id": plan.run_id,
        "job_type": plan.job_type,
        "molecule_key": plan.molecule_key,
        "status": plan.status,
        "applied": applied,
        "action": action,
    }
    if reason:
        result["reason"] = reason
    return result


def _build_index_record(plan: OrganizePlan, state: dict[str, Any]) -> dict[str, Any]:
    final_result = state.get("final_result")
    if not isinstance(final_result, dict):
        final_result = {}
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    last_attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}

    analyzer_status = final_result.get("analyzer_status")
    if not isinstance(analyzer_status, str) or not analyzer_status:
        analyzer_status = str(last_attempt.get("analyzer_status", ""))

    reason = final_result.get("reason")
    if not isinstance(reason, str):
        reason = ""

    return {
        "run_id": plan.run_id,
        "reaction_dir": str(plan.target_abs_path),
        "status": plan.status,
        "analyzer_status": analyzer_status,
        "reason": reason,
        "job_type": plan.job_type,
        "molecule_key": plan.molecule_key,
        "selected_inp": _to_relative_path(
            state.get("selected_inp", ""),
            reaction_dir=plan.target_abs_path,
        ),
        "last_attempt_status": (
            last_attempt.get("analyzer_status", "")
            if isinstance(last_attempt, dict)
            else ""
        ),
        "attempt_count": len(attempts),
        "completed_at": final_result.get("completed_at", ""),
        "organized_at": now_utc_iso(),
        "organized_path": plan.target_rel_path,
    }


def _to_relative_path(path_value: Any, *, reaction_dir: Path) -> str:
    if not isinstance(path_value, str):
        return ""
    raw = path_value.strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute():
        try:
            return str(path.relative_to(reaction_dir))
        except ValueError:
            return path.name
    return str(path)


def execute_move(plan: OrganizePlan) -> None:
    """Move source directory to target path.

    Falls back to copytree+rmtree on cross-device moves.
    """
    plan.target_abs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(str(plan.source_dir), str(plan.target_abs_path))
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copytree(str(plan.source_dir), str(plan.target_abs_path))
        shutil.rmtree(str(plan.source_dir))


def rollback_move(plan: OrganizePlan) -> None:
    """Rollback a previously moved directory."""
    if not plan.target_abs_path.exists():
        return
    if plan.source_dir.exists():
        raise RuntimeError(f"Rollback blocked: source already exists: {plan.source_dir}")
    plan.source_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(str(plan.target_abs_path), str(plan.source_dir))
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copytree(str(plan.target_abs_path), str(plan.source_dir))
        shutil.rmtree(str(plan.target_abs_path))


def sync_state_after_move(plan: OrganizePlan) -> dict[str, Any]:
    """Sync state/report files after a successful move."""
    return _sync_state_after_relocation(
        state_dir=plan.target_abs_path,
        source_dir=plan.source_dir,
        target_dir=plan.target_abs_path,
    )


def sync_state_after_rollback(plan: OrganizePlan) -> dict[str, Any]:
    """Restore state/report paths after rollback."""
    return _sync_state_after_relocation(
        state_dir=plan.source_dir,
        source_dir=plan.target_abs_path,
        target_dir=plan.source_dir,
    )


def _sync_state_after_relocation(
    *,
    state_dir: Path,
    source_dir: Path,
    target_dir: Path,
) -> dict[str, Any]:
    state = _load_state(state_dir)
    if state is None:
        raise RuntimeError(f"Relocated directory has invalid state: {state_dir}")

    state["reaction_dir"] = str(target_dir)

    selected_inp = state.get("selected_inp")
    if isinstance(selected_inp, str):
        state["selected_inp"] = _normalize_moved_artifact_path(
            selected_inp,
            source_dir=source_dir,
            target_dir=target_dir,
        )

    attempts = state.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            for key in ("attempt_dir", "inp_path", "out_path"):
                value = attempt.get(key)
                if isinstance(value, str):
                    attempt[key] = _normalize_moved_artifact_path(
                        value,
                        source_dir=source_dir,
                        target_dir=target_dir,
                    )

    final_result = state.get("final_result")
    if isinstance(final_result, dict):
        last_out = final_result.get("last_out_path")
        if isinstance(last_out, str):
            final_result["last_out_path"] = _normalize_moved_artifact_path(
                last_out,
                source_dir=source_dir,
                target_dir=target_dir,
            )

    save_state(str(state_dir), state)
    write_report_files(state, str(state_dir))
    return state


def _remap_moved_path(
    path_text: str,
    *,
    source_dir: Path,
    target_dir: Path,
) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path_text
    try:
        rel = path.relative_to(source_dir)
    except ValueError:
        return path_text
    return str(target_dir / rel)


def _normalize_moved_artifact_path(
    path_text: str,
    *,
    source_dir: Path,
    target_dir: Path,
) -> str:
    remapped = _remap_moved_path(
        path_text,
        source_dir=source_dir,
        target_dir=target_dir,
    )
    remapped_path = Path(remapped)
    if not remapped_path.is_absolute():
        return remapped
    resolved = _resolve_existing_artifact(remapped, target_dir)
    if resolved is None:
        return remapped
    return str(resolved)


def _resolve_existing_artifact(path_text: str, reaction_dir: Path) -> Path | None:
    raw = path_text.strip()
    if not raw:
        return None

    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        candidates.append(reaction_dir / path.name)
    else:
        candidates.append(reaction_dir / path)
        candidates.append(reaction_dir / path.name)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return resolved
    return None


def _load_state(reaction_dir: Path) -> dict[str, Any] | None:
    """Load run_state.json from a reaction directory."""
    state_path = reaction_dir / "run_state.json"
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _infer_job_type(reaction_dir: Path) -> str:
    """Infer job type from .inp file in the reaction directory."""
    inp_files = sorted(reaction_dir.glob("*.inp"), key=lambda path: path.name)
    if not inp_files:
        return "other"

    # Read the first .inp file and find the route line.
    try:
        with inp_files[0].open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("!"):
                    from inp.route_line import parse_route_line

                    result = parse_route_line(stripped)
                    job_type = result.job_type
                    if (
                        job_type == "optimization"
                        and result.optimizer_mode == "transition_state"
                    ):
                        return "ts_optimization"
                    return job_type
    except Exception:
        return "other"
    return "other"


def _extract_atom_spec(reaction_dir: Path) -> str | None:
    """Try to extract atom spec from the xyz file in the reaction dir."""
    xyz_path = reaction_dir / "_geometry.xyz"
    if not xyz_path.exists():
        xyz_files = sorted(reaction_dir.glob("*.xyz"), key=lambda path: path.name)
        if not xyz_files:
            return None
        xyz_path = xyz_files[0]

    try:
        lines = xyz_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 3:
            return None
        return "\n".join(lines[2:]).strip()
    except OSError:
        return None


def _extract_tag(reaction_dir: Path) -> str | None:
    """Try to extract TAG from .inp file."""
    import re

    inp_files = sorted(reaction_dir.glob("*.inp"), key=lambda path: path.name)
    if not inp_files:
        return None

    tag_re = re.compile(r"^#\s*TAG\s*:\s*(.+)$", re.IGNORECASE)
    try:
        with inp_files[0].open("r", encoding="utf-8") as handle:
            for line in handle:
                match = tag_re.match(line.strip())
                if match:
                    return match.group(1).strip()
    except OSError:
        return None
    return None


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
