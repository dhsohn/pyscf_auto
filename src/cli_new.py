"""CLI for pyscf_auto with .inp file-based runs.

User-facing command surface is intentionally aligned with orca_auto:
- run-inp
- status
- organize
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyscf_auto")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pyscf_auto.yaml",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run-inp ---
    run_parser = subparsers.add_parser(
        "run-inp",
        help="Run a PySCF calculation from a .inp file.",
    )
    run_parser.add_argument(
        "--reaction-dir",
        required=True,
        help="Directory under the configured allowed_root containing input files",
    )
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum retry attempts on failure (default: from config).",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if existing output is completed",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Output progress as JSON.",
    )

    # --- status ---
    status_parser = subparsers.add_parser(
        "status",
        help="Check the status of a run.",
    )
    status_parser.add_argument(
        "--reaction-dir",
        required=True,
        help="Directory under the configured allowed_root",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON.",
    )

    # --- organize ---
    organize_parser = subparsers.add_parser(
        "organize",
        help="Organize completed runs into a clean directory structure.",
    )
    organize_parser.add_argument(
        "--reaction-dir",
        default=None,
        help="Single reaction directory to organize",
    )
    organize_parser.add_argument(
        "--root",
        default=None,
        help="Root directory to scan (mutually exclusive with --reaction-dir)",
    )
    organize_parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually move files (default is dry-run)",
    )
    organize_parser.add_argument(
        "--rebuild-index",
        action="store_true",
        default=False,
        help="Rebuild JSONL index from organized directories.",
    )
    organize_parser.add_argument(
        "--find",
        action="store_true",
        default=False,
        help="Search the index",
    )
    organize_parser.add_argument(
        "--run-id",
        default=None,
        help="Find by run_id (with --find)",
    )
    organize_parser.add_argument(
        "--job-type",
        default=None,
        help="Filter by job_type (with --find)",
    )
    organize_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit results (with --find)",
    )
    organize_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        command_map = {
            "run-inp": _cmd_run_inp,
            "status": _cmd_status,
            "organize": _cmd_organize,
        }
        handler = command_map.get(args.command)
        if handler is None:
            parser.print_help()
            return 1
        return int(handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.error("%s", exc)
        return 1


def _cmd_run_inp(args: argparse.Namespace) -> int:
    from app_config import load_app_config
    from runner.orchestrator import cmd_run_inp

    app_config = load_app_config(getattr(args, "config", None))
    exit_code = cmd_run_inp(
        reaction_dir=args.reaction_dir,
        max_retries=args.max_retries,
        force=args.force,
        json_output=args.json,
        app_config=app_config,
    )
    return int(exit_code)


def _cmd_status(args: argparse.Namespace) -> int:
    from app_config import load_app_config
    from runner.orchestrator import cmd_status

    app_config = load_app_config(getattr(args, "config", None))
    exit_code = cmd_status(
        reaction_dir=args.reaction_dir,
        json_output=args.json,
        app_config=app_config,
    )
    return int(exit_code)


def _cmd_organize(args: argparse.Namespace) -> int:
    from app_config import load_app_config
    from organizer.result_organizer import (
        apply_plans,
        find_organized_runs,
        plan_root_scan,
        plan_single,
        rebuild_organized_index,
    )

    app_config = load_app_config(getattr(args, "config", None))
    organized_root = _resolve_dir(app_config.runtime.organized_root)
    allowed_root = _resolve_dir(app_config.runtime.allowed_root)

    if getattr(args, "rebuild_index", False):
        count = rebuild_organized_index(str(organized_root))
        payload = {"action": "rebuild_index", "records_count": count}
        _emit_organize(payload, as_json=args.json)
        return 0

    if getattr(args, "find", False):
        run_id = getattr(args, "run_id", None)
        job_type = getattr(args, "job_type", None)
        limit = max(0, int(getattr(args, "limit", 0) or 0))

        if run_id:
            records = find_organized_runs(str(organized_root), run_id=run_id)
            if not records:
                logging.error("run_id not found: %s", run_id)
                return 1
            _emit_organize(records[0], as_json=args.json)
            return 0
        if job_type:
            records = find_organized_runs(
                str(organized_root),
                job_type=job_type,
                limit=limit,
            )
            _emit_organize(
                {"results": records, "count": len(records)},
                as_json=args.json,
            )
            return 0

        logging.error("--find requires --run-id or --job-type")
        return 1

    reaction_dir_raw = getattr(args, "reaction_dir", None)
    root_raw = getattr(args, "root", None)

    if reaction_dir_raw and root_raw:
        logging.error("--reaction-dir and --root are mutually exclusive")
        return 1
    if not reaction_dir_raw and not root_raw:
        logging.error("Either --reaction-dir or --root is required")
        return 1

    if reaction_dir_raw:
        reaction_dir = _validate_reaction_dir(reaction_dir_raw, allowed_root)
        if reaction_dir is None:
            return 1
        plan, skip = plan_single(reaction_dir, organized_root)
        plans = [plan] if plan is not None else []
        skips = [skip] if skip is not None else []
    else:
        root = _validate_root_scan_dir(root_raw, allowed_root)
        if root is None:
            return 1
        plans, skips = plan_root_scan(root, organized_root)

    if not getattr(args, "apply", False):
        summary = {
            "action": "dry_run",
            "to_organize": len(plans),
            "skipped": len(skips),
            "plans": [_plan_to_dict(plan) for plan in plans],
            "skip_reasons": [
                {"reaction_dir": skip.reaction_dir, "reason": skip.reason}
                for skip in skips
            ],
        }
        _emit_organize(summary, as_json=args.json)
        return 0

    results, failures = apply_plans(plans, organized_root)
    organized_count = len(
        [result for result in results if result.get("action") == "moved"],
    )
    skipped_count = len(skips) + len(
        [result for result in results if result.get("action") == "skipped"],
    )
    summary = {
        "action": "apply",
        "organized": organized_count,
        "skipped": skipped_count,
        "failed": len(failures),
        "failures": failures,
    }
    _emit_organize(summary, as_json=args.json)
    return 1 if failures else 0


def _emit_organize(payload: dict[str, Any], *, as_json: bool) -> None:
    import json

    if as_json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    for key in ["action", "to_organize", "skipped", "organized", "failed", "records_count"]:
        if key in payload:
            print(f"{key}: {payload[key]}")

    for plan in payload.get("plans", []):
        if isinstance(plan, dict):
            print(f"  {plan['source_dir']} -> {plan['target_rel_path']}")

    for skip in payload.get("skip_reasons", []):
        if isinstance(skip, dict):
            print(f"  SKIP {skip['reaction_dir']}: {skip['reason']}")

    for failure in payload.get("failures", []):
        if isinstance(failure, dict):
            print(f"  FAIL {failure.get('run_id', '?')}: {failure.get('reason', 'unknown')}")

    for key in ["run_id", "job_type", "molecule_key", "organized_path", "count"]:
        if key in payload:
            print(f"{key}: {payload[key]}")

    for result in payload.get("results", []):
        if isinstance(result, dict):
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


def _resolve_dir(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def _validate_reaction_dir(
    reaction_dir_raw: str,
    allowed_root: Path,
) -> Path | None:
    reaction_dir = _resolve_dir(reaction_dir_raw)
    if not reaction_dir.exists() or not reaction_dir.is_dir():
        logging.error("Reaction directory not found: %s", reaction_dir)
        return None
    if not _is_subpath(reaction_dir, allowed_root):
        logging.error(
            "Reaction directory must be under allowed_root: %s (got %s)",
            allowed_root,
            reaction_dir,
        )
        return None
    return reaction_dir


def _validate_root_scan_dir(root_raw: str, allowed_root: Path) -> Path | None:
    root = _resolve_dir(root_raw)
    if not root.exists() or not root.is_dir():
        logging.error("Root directory not found: %s", root)
        return None
    if root != allowed_root:
        logging.error(
            "--root must exactly match allowed_root: %s (got %s)",
            allowed_root,
            root,
        )
        return None
    return root


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
