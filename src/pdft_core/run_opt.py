"""CLI entrypoint module for pDFT."""

__all__ = ["main"]

import json
import logging
import os
import sys
from pathlib import Path

from . import cli, interactive, queue, workflow
from .run_opt_config import (
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_QUEUE_RUNNER_LOCK_PATH,
    DEFAULT_RUN_METADATA_PATH,
    build_run_config,
    load_run_config,
)

TERMINAL_RESUME_STATUSES = {"completed", "failed", "timeout", "canceled"}


def _load_resume_checkpoint(resume_dir):
    resume_path = Path(resume_dir).expanduser().resolve()
    if not resume_path.exists() or not resume_path.is_dir():
        raise ValueError(f"--resume path is not a directory: {resume_path}")
    checkpoint_path = resume_path / "checkpoint.json"
    if not checkpoint_path.exists():
        raise ValueError(f"Missing checkpoint.json in resume directory: {resume_path}")
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint.json in {resume_path}: {exc}") from exc
    xyz_file = checkpoint.get("xyz_file")
    if not xyz_file:
        raise ValueError("checkpoint.json is missing required key: xyz_file")
    xyz_path = Path(xyz_file).expanduser()
    if not xyz_path.is_absolute():
        xyz_path = (resume_path / xyz_path).resolve()
    else:
        xyz_path = xyz_path.resolve()
    if not xyz_path.exists():
        raise ValueError(f"XYZ file from checkpoint.json not found: {xyz_path}")

    config_used_path = resume_path / "config_used.json"
    config_raw = None
    config_source_path = None
    if config_used_path.exists():
        config_raw = config_used_path.read_text(encoding="utf-8")
        config_source_path = config_used_path
    else:
        config_raw = checkpoint.get("config_raw")
        config_source_path = checkpoint.get("config_source_path")
        if config_source_path:
            config_source_path = Path(config_source_path).expanduser()
            if not config_source_path.is_absolute():
                config_source_path = (resume_path / config_source_path).resolve()
            else:
                config_source_path = config_source_path.resolve()
            if config_source_path.exists() and config_raw is None:
                config_raw = config_source_path.read_text(encoding="utf-8")

    if not config_raw:
        raise ValueError("Unable to reconstruct config_raw from checkpoint/config_used.json.")

    return {
        "resume_dir": resume_path,
        "checkpoint_path": checkpoint_path,
        "xyz_file": str(xyz_path),
        "config_source_path": config_source_path,
        "config_raw": config_raw,
    }


def _load_resume_status(run_metadata_path):
    if not run_metadata_path:
        return None
    try:
        if not os.path.exists(run_metadata_path):
            return None
        with open(run_metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return None
    return metadata.get("status")


def _parse_scan_dimension(spec):
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError("Scan dimension spec must not be empty.")
    dim_type = parts[0].lower()
    if dim_type not in ("bond", "angle", "dihedral"):
        raise ValueError("Scan dimension type must be bond, angle, or dihedral.")
    index_count = {"bond": 2, "angle": 3, "dihedral": 4}[dim_type]
    expected_len = 1 + index_count + 3
    if len(parts) != expected_len:
        raise ValueError(
            "Scan dimension spec '{spec}' must have {expected} fields.".format(
                spec=spec, expected=expected_len
            )
        )
    try:
        indices = [int(value) for value in parts[1 : 1 + index_count]]
    except ValueError as exc:
        raise ValueError("Scan dimension indices must be integers.") from exc
    try:
        start, end, step = (float(value) for value in parts[1 + index_count :])
    except ValueError as exc:
        raise ValueError("Scan dimension start/end/step must be numbers.") from exc
    dimension = {"type": dim_type, "start": start, "end": end, "step": step}
    for key, value in zip(("i", "j", "k", "l"), indices, strict=False):
        dimension[key] = value
    return dimension


def _apply_scan_cli_overrides(config, args):
    if not (args.scan_dimension or args.scan_grid or args.scan_mode):
        return config
    if args.scan_dimension is None:
        raise ValueError("--scan-dimension is required when using scan options.")
    dimensions = [_parse_scan_dimension(spec) for spec in args.scan_dimension]
    if len(dimensions) not in (1, 2):
        raise ValueError("Scan mode currently supports 1D or 2D dimensions only.")
    scan_config = {}
    if len(dimensions) == 1:
        scan_config = dimensions[0]
    else:
        scan_config["dimensions"] = dimensions
    if args.scan_grid:
        if len(args.scan_grid) != len(dimensions):
            raise ValueError("--scan-grid entries must match scan dimension count.")
        grid = []
        for entry in args.scan_grid:
            values = [value.strip() for value in entry.split(",") if value.strip()]
            if not values:
                raise ValueError("--scan-grid entries must contain values.")
            try:
                grid.append([float(value) for value in values])
            except ValueError as exc:
                raise ValueError("--scan-grid values must be numbers.") from exc
        scan_config["grid"] = grid
    if args.scan_mode:
        scan_config["mode"] = args.scan_mode
    config = dict(config)
    config["scan"] = scan_config
    config["calculation_mode"] = "scan"
    return config


def main():
    """
    Main function to run the geometry optimization.
    """
    parser = cli.build_parser()
    args = parser.parse_args(cli._normalize_cli_args(sys.argv[1:]))

    if args.doctor:
        workflow.run_doctor()
        return

    run_in_background = bool(args.background and not args.no_background)
    if args.interactive and args.non_interactive:
        raise ValueError("--interactive and --non-interactive cannot be used together.")
    if args.resume and args.run_dir:
        raise ValueError("--resume and --run-dir cannot be used together.")
    if args.resume and args.scan_dimension:
        raise ValueError("--scan-dimension cannot be used with --resume.")
    if args.resume and args.scan_grid:
        raise ValueError("--scan-grid cannot be used with --resume.")
    if args.resume and args.scan_mode:
        raise ValueError("--scan-mode cannot be used with --resume.")
    if args.resume and args.xyz_file:
        raise ValueError("xyz_file cannot be provided when using --resume.")
    if args.resume and args.interactive:
        raise ValueError("--interactive cannot be used with --resume.")
    if args.validate_only and args.interactive:
        raise ValueError("--validate-only cannot be used with --interactive.")
    if args.interactive is None:
        args.interactive = not args.non_interactive
    if args.validate_only:
        args.interactive = False
        args.non_interactive = True
    if args.resume:
        args.interactive = False
        args.non_interactive = True
    if args.queue_status:
        queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
        with queue.queue_lock(DEFAULT_QUEUE_LOCK_PATH):
            queue_state = queue.load_queue(DEFAULT_QUEUE_PATH)
        queue.format_queue_status(queue_state)
        return

    try:
        if args.queue_runner:
            queue.run_queue_worker(
                os.path.abspath(sys.argv[0]),
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
                DEFAULT_QUEUE_RUNNER_LOCK_PATH,
            )
            return

        if args.queue_cancel:
            queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
            canceled, error = queue.cancel_queue_entry(
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
                args.queue_cancel,
            )
            if not canceled:
                raise ValueError(error)
            print(f"Canceled queued run: {args.queue_cancel}")
            return
        if args.queue_retry:
            queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
            retried, error = queue.requeue_queue_entry(
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
                args.queue_retry,
                reason="retry",
            )
            if not retried:
                raise ValueError(error)
            print(f"Re-queued run: {args.queue_retry}")
            return
        if args.queue_requeue_failed:
            queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
            count = queue.requeue_failed_entries(DEFAULT_QUEUE_PATH, DEFAULT_QUEUE_LOCK_PATH)
            print(f"Re-queued failed runs: {count}")
            return

        if args.status_recent:
            queue.print_recent_statuses(args.status_recent)
            return

        if args.status:
            queue.print_status(args.status, DEFAULT_RUN_METADATA_PATH)
            return

        config_source_path = None
        if args.resume:
            resume_state = _load_resume_checkpoint(args.resume)
            args.xyz_file = resume_state["xyz_file"]
            args.run_dir = str(resume_state["resume_dir"])
            config_raw = resume_state["config_raw"]
            config_source_path = resume_state["config_source_path"]
            if config_source_path is not None:
                args.config = str(config_source_path)
            else:
                args.config = str(resume_state["checkpoint_path"])
            try:
                config = json.loads(config_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid config JSON in resume data: {exc}") from exc
        elif args.interactive:
            config, config_raw, config_source_path = interactive.prompt_config(args)
            if config_source_path is not None:
                config_source_path = Path(config_source_path).resolve()
        else:
            if not args.xyz_file and not args.validate_only:
                raise ValueError(
                    "xyz_file is required unless --status or --validate-only is used."
                )
            config_path = Path(args.config).expanduser().resolve()
            config, config_raw = load_run_config(config_path)
            args.config = str(config_path)
            config_source_path = config_path
        if args.scan_dimension or args.scan_grid or args.scan_mode:
            if args.interactive:
                raise ValueError("--scan-* options cannot be used with --interactive.")
            config = _apply_scan_cli_overrides(config, args)
            config_raw = json.dumps(config, indent=2, ensure_ascii=False)

        try:
            config = build_run_config(config)
        except ValueError as error:
            message = str(error)
            print(message, file=sys.stderr)
            logging.error(message)
            raise
        if args.resume:
            metadata_candidate = config.run_metadata_file or DEFAULT_RUN_METADATA_PATH
            if os.path.isabs(metadata_candidate):
                run_metadata_path = Path(metadata_candidate)
            else:
                run_metadata_path = Path(args.run_dir) / metadata_candidate
            previous_status = _load_resume_status(str(run_metadata_path))
            args.resume_previous_status = previous_status
            if previous_status in TERMINAL_RESUME_STATUSES and not args.force_resume:
                raise ValueError(
                    "Refusing to resume a {status} run without --force-resume.".format(
                        status=previous_status
                    )
                )
            if previous_status in TERMINAL_RESUME_STATUSES and args.force_resume:
                print(
                    "Warning: resuming a {status} run with --force-resume.".format(
                        status=previous_status
                    ),
                    file=sys.stderr,
                )
        if args.validate_only:
            print(f"Config validation passed: {args.config}")
            return

        workflow.run(args, config, config_raw, config_source_path, run_in_background)
    except Exception:
        logging.exception("Run failed.")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        message = str(error)
        if message:
            print(message, file=sys.stderr)
        sys.exit(1)
