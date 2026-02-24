from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app_config import load_app_config
from organizer.result_organizer import (
    apply_plans,
    find_organized_runs,
    plan_root_scan,
    plan_single,
    rebuild_organized_index,
)
from ._helpers import validate_reaction_dir, validate_root_scan_dir

logger = logging.getLogger(__name__)


def _emit_organize(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    for key in ["action", "to_organize", "skipped", "organized", "failed", "records_count"]:
        if key in payload:
            print(f"{key}: {payload[key]}")
    for plan in payload.get("plans", []):
        print(f"  {plan['source_dir']} -> {plan['target_rel_path']}")
    for skip in payload.get("skip_reasons", []):
        print(f"  SKIP {skip['reaction_dir']}: {skip['reason']}")
    for failure in payload.get("failures", []):
        print(f"  FAIL {failure.get('run_id', '?')}: {failure.get('reason', 'unknown')}")
    for key in ["run_id", "job_type", "molecule_key", "organized_path", "count"]:
        if key in payload:
            print(f"{key}: {payload[key]}")
    for result in payload.get("results", []):
        print(f"  {result.get('run_id', '?')}: {result.get('organized_path', '?')}")


def _plan_to_dict(plan: Any) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "source_dir": str(plan.source_dir),
        "target_rel_path": plan.target_rel_path,
        "target_abs_path": str(plan.target_abs_path),
        "job_type": plan.job_type,
        "molecule_key": plan.molecule_key,
    }


def cmd_organize(args: Any) -> int:
    cfg = load_app_config(getattr(args, "config", None))
    organized_root = Path(cfg.runtime.organized_root).expanduser().resolve()

    if getattr(args, "rebuild_index", False):
        count = rebuild_organized_index(str(organized_root))
        _emit_organize({"action": "rebuild_index", "records_count": count}, as_json=args.json)
        return 0

    if getattr(args, "find", False):
        run_id = getattr(args, "run_id", None)
        job_type = getattr(args, "job_type", None)
        limit = max(0, int(getattr(args, "limit", 0) or 0))

        if run_id:
            records = find_organized_runs(str(organized_root), run_id=run_id)
            if not records:
                logger.error("run_id not found: %s", run_id)
                return 1
            _emit_organize(records[0], as_json=args.json)
            return 0

        if job_type:
            records = find_organized_runs(str(organized_root), job_type=job_type, limit=limit)
            _emit_organize({"results": records, "count": len(records)}, as_json=args.json)
            return 0

        logger.error("--find requires --run-id or --job-type")
        return 1

    reaction_dir_raw = getattr(args, "reaction_dir", None)
    root_raw = getattr(args, "root", None)

    if reaction_dir_raw and root_raw:
        logger.error("--reaction-dir and --root are mutually exclusive")
        return 1
    if not reaction_dir_raw and not root_raw:
        logger.error("Either --reaction-dir or --root is required")
        return 1

    if reaction_dir_raw:
        try:
            reaction_dir = validate_reaction_dir(cfg, reaction_dir_raw)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        plan, skip = plan_single(reaction_dir, organized_root)
        plans = [plan] if plan is not None else []
        skips = [skip] if skip is not None else []
    else:
        try:
            root = validate_root_scan_dir(cfg, str(root_raw))
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        plans, skips = plan_root_scan(root, organized_root)

    if not getattr(args, "apply", False):
        summary = {
            "action": "dry_run",
            "to_organize": len(plans),
            "skipped": len(skips),
            "plans": [_plan_to_dict(plan) for plan in plans],
            "skip_reasons": [{"reaction_dir": skip.reaction_dir, "reason": skip.reason} for skip in skips],
        }
        _emit_organize(summary, as_json=args.json)
        return 0

    results, failures = apply_plans(plans, organized_root)
    summary = {
        "action": "apply",
        "organized": len([result for result in results if result.get("action") == "moved"]),
        "skipped": len(skips) + len([result for result in results if result.get("action") == "skipped"]),
        "failed": len(failures),
        "failures": failures,
    }
    _emit_organize(summary, as_json=args.json)
    return 1 if failures else 0
