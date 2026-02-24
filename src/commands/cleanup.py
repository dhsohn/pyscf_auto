from __future__ import annotations

import json
import logging
from typing import Any

from app_config import load_app_config
from organizer.result_cleaner import (
    CleanupPlan,
    CleanupResult,
    CleanupSkipReason,
    execute_cleanup,
    plan_cleanup_root_scan,
    plan_cleanup_single,
)
from ._helpers import (
    _MAX_SAMPLE_FILES,
    human_bytes,
    validate_cleanup_reaction_dir,
    validate_organized_root_dir,
)

logger = logging.getLogger(__name__)


def _emit_cleanup(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    for key in [
        "action",
        "to_clean",
        "skipped",
        "cleaned",
        "failed",
        "total_files_removed",
        "total_bytes_freed_human",
    ]:
        if key in payload:
            print(f"{key}: {payload[key]}")
    for plan in payload.get("plans", []):
        print(f"  {plan['reaction_dir']}: {plan['remove_count']} files, {plan['bytes_human']}")
    for skip in payload.get("skip_reasons", []):
        print(f"  SKIP {skip['reaction_dir']}: {skip['reason']}")
    for failure in payload.get("failures", []):
        print(f"  FAIL {failure['run_id']}: {failure['errors']}")


def _cleanup_plan_to_dict(plan: CleanupPlan) -> dict[str, Any]:
    file_names = [entry.path.name for entry in plan.files_to_remove]
    return {
        "run_id": plan.run_id,
        "reaction_dir": str(plan.reaction_dir),
        "remove_count": len(plan.files_to_remove),
        "keep_count": plan.keep_count,
        "total_remove_bytes": plan.total_remove_bytes,
        "bytes_human": human_bytes(plan.total_remove_bytes),
        "sample_files": file_names[:_MAX_SAMPLE_FILES],
    }


def _cmd_cleanup_apply(
    plans: list[CleanupPlan],
    skips: list[CleanupSkipReason],
    as_json: bool,
) -> int:
    results: list[CleanupResult] = []
    failures: list[dict[str, Any]] = []

    for plan in plans:
        try:
            result = execute_cleanup(plan)
            results.append(result)
            if result.errors:
                failures.append({"run_id": result.run_id, "errors": result.errors})
        except Exception as exc:
            logger.error("Cleanup failed for %s: %s", plan.run_id, exc)
            failures.append({"run_id": plan.run_id, "errors": [str(exc)]})

    total_files = sum(result.files_removed for result in results)
    total_bytes = sum(result.bytes_freed for result in results)
    cleaned_count = len([result for result in results if not result.errors])

    summary = {
        "action": "apply",
        "cleaned": cleaned_count,
        "skipped": len(skips),
        "failed": len(failures),
        "total_files_removed": total_files,
        "total_bytes_freed": total_bytes,
        "total_bytes_freed_human": human_bytes(total_bytes),
        "failures": failures,
    }
    _emit_cleanup(summary, as_json=as_json)
    return 1 if failures else 0


def cmd_cleanup(args: Any) -> int:
    cfg = load_app_config(getattr(args, "config", None))
    reaction_dir_raw = getattr(args, "reaction_dir", None)
    root_raw = getattr(args, "root", None)

    if reaction_dir_raw and root_raw:
        logger.error("--reaction-dir and --root are mutually exclusive")
        return 1
    if not reaction_dir_raw and not root_raw:
        root_raw = cfg.runtime.organized_root

    keep_extensions = set(cfg.cleanup.keep_extensions)
    keep_filenames = set(cfg.cleanup.keep_filenames)
    remove_patterns = list(cfg.cleanup.remove_patterns)

    if reaction_dir_raw:
        try:
            reaction_dir = validate_cleanup_reaction_dir(cfg, reaction_dir_raw)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        plan, skip = plan_cleanup_single(
            reaction_dir,
            keep_extensions=keep_extensions,
            keep_filenames=keep_filenames,
            remove_patterns=remove_patterns,
        )
        plans = [plan] if plan else []
        skips = [skip] if skip else []
    else:
        try:
            root = validate_organized_root_dir(cfg, str(root_raw))
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        plans, skips = plan_cleanup_root_scan(
            root,
            keep_extensions=keep_extensions,
            keep_filenames=keep_filenames,
            remove_patterns=remove_patterns,
        )

    if not getattr(args, "apply", False):
        total_bytes = sum(plan.total_remove_bytes for plan in plans)
        summary = {
            "action": "dry_run",
            "to_clean": len(plans),
            "skipped": len(skips),
            "total_bytes_freed": total_bytes,
            "total_bytes_freed_human": human_bytes(total_bytes),
            "plans": [_cleanup_plan_to_dict(plan) for plan in plans],
            "skip_reasons": [{"reaction_dir": skip.reaction_dir, "reason": skip.reason} for skip in skips],
        }
        _emit_cleanup(summary, as_json=args.json)
        return 0

    return _cmd_cleanup_apply(plans, skips, as_json=args.json)

