"""CLI entrypoint module for DFTFlow."""

__all__ = ["main"]

import argparse
import itertools
import json
import logging
import os
import re
import sys
from pathlib import Path

import cli
import interactive
import run_queue
import workflow
from run_opt_config import (
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_QUEUE_RUNNER_LOCK_PATH,
    DEFAULT_RUN_METADATA_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
    build_run_config,
    load_run_config,
    load_solvent_map,
)
from run_opt_resources import create_run_directory

TERMINAL_RESUME_STATUSES = {"completed", "failed", "timeout", "canceled"}

SMOKE_TEST_XYZ = """3
DFTFlow smoke-test water molecule.
O     -0.1659811139    2.0308399200   -0.0000031757
H     -2.5444712639    1.0182403326    0.6584512591
H     -1.0147968531    2.4412472248   -2.0058431625
"""
SMOKE_TEST_MODES = ("single_point", "optimization", "frequency", "irc", "scan")
SMOKE_TEST_SOLVENT_MODELS = (None, "pcm", "smd")
SMOKE_TEST_DISPERSION_MODELS = (None, "d3bj", "d3zero", "d4")


def _build_smoke_test_config(base_config, mode, overrides):
    config = dict(base_config)
    config["calculation_mode"] = mode
    config["basis"] = overrides["basis"]
    config["xc"] = overrides["xc"]
    config["solvent_model"] = overrides["solvent_model"]
    config["solvent"] = overrides["solvent"]
    config["dispersion"] = overrides["dispersion"]
    if mode != "scan":
        config.pop("scan", None)
        config.pop("scan2d", None)
    scf_config = dict(config.get("scf") or {})
    scf_config["max_cycle"] = 1
    config["scf"] = scf_config
    single_point_config = dict(config.get("single_point") or {})
    single_point_scf = dict(single_point_config.get("scf") or {})
    single_point_scf["max_cycle"] = 1
    single_point_config["scf"] = single_point_scf
    config["single_point"] = single_point_config
    optimizer_config = dict(config.get("optimizer") or {})
    optimizer_ase = dict(optimizer_config.get("ase") or {})
    optimizer_ase["steps"] = 1
    optimizer_config["ase"] = optimizer_ase
    config["optimizer"] = optimizer_config
    if mode == "optimization":
        config["frequency_enabled"] = False
        config["single_point_enabled"] = False
    if mode == "irc":
        config["irc"] = {
            "steps": 1,
            "step_size": 0.05,
            "force_threshold": 0.1,
        }
    if mode == "scan":
        config.pop("scan2d", None)
        config["scan"] = {
            "mode": "single_point",
            "dimensions": [
                {
                    "type": "bond",
                    "i": 0,
                    "j": 1,
                    "start": 0.9,
                    "end": 0.9,
                    "step": 0.1,
                }
            ],
        }
    return config


def _unique_values(values):
    seen = set()
    ordered = []
    for value in values:
        key = value if value is not None else "__none__"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def _prepare_smoke_test_suite(args):
    config_path = Path(args.config).expanduser().resolve()
    base_config, _base_raw = load_run_config(config_path)
    basis_options = _unique_values(
        [*interactive.BASIS_SET_OPTIONS, base_config.get("basis")]
    )
    basis_options = [basis for basis in basis_options if basis]
    xc_options = _unique_values(
        [*interactive.XC_FUNCTIONAL_OPTIONS, base_config.get("xc")]
    )
    xc_options = [xc for xc in xc_options if xc]
    solvent_model_options = _unique_values(
        [*SMOKE_TEST_SOLVENT_MODELS, base_config.get("solvent_model")]
    )
    dispersion_options = _unique_values(
        [*SMOKE_TEST_DISPERSION_MODELS, base_config.get("dispersion")]
    )
    solvent_map_path = base_config.get("solvent_map") or DEFAULT_SOLVENT_MAP_PATH
    solvent_options = sorted(load_solvent_map(solvent_map_path).keys())
    modes = list(SMOKE_TEST_MODES)
    cases = []
    for basis, xc, solvent_model, dispersion in itertools.product(
        basis_options, xc_options, solvent_model_options, dispersion_options
    ):
        if solvent_model:
            for solvent in solvent_options:
                cases.append(
                    {
                        "basis": basis,
                        "xc": xc,
                        "solvent_model": solvent_model,
                        "solvent": solvent,
                        "dispersion": dispersion,
                    }
                )
        else:
            cases.append(
                {
                    "basis": basis,
                    "xc": xc,
                    "solvent_model": None,
                    "solvent": "vacuum",
                    "dispersion": dispersion,
                }
            )
    return base_config, config_path, modes, cases


def _prepare_smoke_test_run_dir(base_run_dir, mode, overrides, index):
    parts = [
        f"{index:04d}",
        _slugify(mode),
        _slugify(overrides["basis"]),
        _slugify(overrides["xc"]),
        _slugify(overrides["solvent_model"] or "vacuum"),
        _slugify(overrides["solvent"]),
        _slugify(overrides["dispersion"] or "none"),
    ]
    run_dir = Path(base_run_dir) / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    base_scan = config.get("scan")
    if isinstance(base_scan, dict):
        scan_config.update(base_scan)
    for key in ("type", "i", "j", "k", "l", "start", "end", "step", "dimensions"):
        scan_config.pop(key, None)
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
    else:
        existing_grid = base_scan.get("grid") if isinstance(base_scan, dict) else None
        if isinstance(existing_grid, list) and len(existing_grid) == len(dimensions):
            scan_config["grid"] = existing_grid
        else:
            scan_config.pop("grid", None)
    if args.scan_mode:
        scan_config["mode"] = args.scan_mode
    config = dict(config)
    config["scan"] = scan_config
    config.pop("scan2d", None)
    config["calculation_mode"] = "scan"
    return config


def main():
    """
    Main function to run the geometry optimization.
    """
    parser = cli.build_parser()
    args = parser.parse_args(cli._normalize_cli_args(sys.argv[1:]))

    try:
        if args.command == "doctor":
            workflow.run_doctor()
            return

        if args.command == "queue":
            if args.queue_command == "status":
                run_queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
                with run_queue.queue_lock(DEFAULT_QUEUE_LOCK_PATH):
                    queue_state = run_queue.load_queue(DEFAULT_QUEUE_PATH)
                run_queue.format_queue_status(queue_state)
                return
            if args.queue_command == "cancel":
                run_queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
                canceled, error = run_queue.cancel_queue_entry(
                    DEFAULT_QUEUE_PATH,
                    DEFAULT_QUEUE_LOCK_PATH,
                    args.run_id,
                )
                if not canceled:
                    raise ValueError(error)
                print(f"Canceled queued run: {args.run_id}")
                return
            if args.queue_command == "retry":
                run_queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
                retried, error = run_queue.requeue_queue_entry(
                    DEFAULT_QUEUE_PATH,
                    DEFAULT_QUEUE_LOCK_PATH,
                    args.run_id,
                    reason="retry",
                )
                if not retried:
                    raise ValueError(error)
                print(f"Re-queued run: {args.run_id}")
                return
            if args.queue_command == "requeue-failed":
                run_queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
                count = run_queue.requeue_failed_entries(
                    DEFAULT_QUEUE_PATH, DEFAULT_QUEUE_LOCK_PATH
                )
                print(f"Re-queued failed runs: {count}")
                return
            if args.queue_command == "prune":
                run_queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
                run_queue.load_queue(DEFAULT_QUEUE_PATH)
                removed, remaining = run_queue.prune_queue_entries(
                    DEFAULT_QUEUE_PATH,
                    DEFAULT_QUEUE_LOCK_PATH,
                    args.keep_days,
                    {"completed", "failed", "timeout", "canceled"},
                )
                print(f"Pruned queue entries: {removed} removed, {remaining} remaining.")
                return
            if args.queue_command == "archive":
                run_queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
                archive_path = run_queue.archive_queue(
                    DEFAULT_QUEUE_PATH,
                    DEFAULT_QUEUE_LOCK_PATH,
                    args.path,
                )
                print(f"Archived queue entries to: {archive_path}")
                return

        if args.command == "status":
            if args.recent and args.run_path:
                raise ValueError("--recent cannot be used with a run path.")
            if args.recent:
                run_queue.print_recent_statuses(args.recent)
                return
            if args.run_path:
                run_queue.print_status(args.run_path, DEFAULT_RUN_METADATA_PATH)
                return
            raise ValueError("status requires a run path or --recent.")

        if args.command == "validate-config":
            config_path = args.config_path or args.config
            config_path = Path(config_path).expanduser().resolve()
            config, _config_raw = load_run_config(config_path)
            try:
                build_run_config(config)
            except ValueError as error:
                message = str(error)
                print(message, file=sys.stderr)
                logging.error(message)
                raise
            print(f"Config validation passed: {config_path}")
            return

        if args.command == "smoke-test":
            base_config, config_path, modes, cases = _prepare_smoke_test_suite(args)
            solvent_map_path = base_config.get("solvent_map") or DEFAULT_SOLVENT_MAP_PATH
            base_run_dir = args.run_dir or create_run_directory()
            base_run_dir = str(Path(base_run_dir).expanduser().resolve())
            os.makedirs(base_run_dir, exist_ok=True)
            xyz_path = Path(base_run_dir) / "smoke_test_water.xyz"
            xyz_path.write_text(SMOKE_TEST_XYZ, encoding="utf-8")
            total_cases = len(modes) * len(cases)
            failures = []
            case_index = 1
            for overrides in cases:
                for mode in modes:
                    run_dir = _prepare_smoke_test_run_dir(
                        base_run_dir, mode, overrides, case_index
                    )
                    case_index += 1
                    smoke_config = _build_smoke_test_config(base_config, mode, overrides)
                    smoke_config_raw = json.dumps(
                        smoke_config, indent=2, ensure_ascii=False
                    )
                    smoke_config_path = run_dir / "config_smoke_test.json"
                    smoke_config_path.write_text(smoke_config_raw, encoding="utf-8")
                    run_args = argparse.Namespace(
                        xyz_file=str(xyz_path),
                        solvent_map=solvent_map_path,
                        config=str(smoke_config_path),
                        interactive=False,
                        non_interactive=True,
                        background=False,
                        no_background=True,
                        run_dir=str(run_dir),
                        resume=None,
                        run_id=None,
                        force_resume=False,
                        queue_priority=0,
                        queue_max_runtime=None,
                        scan_dimension=None,
                        scan_grid=None,
                        scan_mode=None,
                        scan_result_csv=None,
                        queue_runner=False,
                    )
                    try:
                        config = build_run_config(smoke_config)
                        workflow.run(
                            run_args,
                            config,
                            smoke_config_raw,
                            str(smoke_config_path),
                            False,
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "run_dir": str(run_dir),
                                "mode": mode,
                                "basis": overrides["basis"],
                                "xc": overrides["xc"],
                                "solvent": overrides["solvent"],
                                "solvent_model": overrides["solvent_model"],
                                "dispersion": overrides["dispersion"],
                                "error": str(error),
                            }
                        )
                        continue
            if failures:
                print(
                    "Smoke test completed with failures: "
                    f"{len(failures)}/{total_cases}"
                )
                for failure in failures:
                    print(
                        "  {mode} {basis} {xc} {solvent_model}/{solvent} "
                        "{dispersion} -> {run_dir} ({error})".format(**failure)
                    )
                raise SystemExit(1)
            print(f"Smoke test completed: {base_run_dir} ({total_cases} cases)")
            return

        run_in_background = bool(args.background and not args.no_background)
        if args.queue_runner:
            run_queue.run_queue_worker(
                os.path.abspath(sys.argv[0]),
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
                DEFAULT_QUEUE_RUNNER_LOCK_PATH,
            )
            return

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
        if args.interactive is None:
            args.interactive = not args.non_interactive
        if args.resume:
            args.interactive = False
            args.non_interactive = True

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
            if not args.xyz_file:
                raise ValueError("xyz_file is required unless --resume is used.")
            config_path = Path(args.config).expanduser().resolve()
            config, config_raw = load_run_config(config_path)
            args.config = str(config_path)
            config_source_path = config_path
        if args.scan_dimension or args.scan_grid or args.scan_mode:
            config = _apply_scan_cli_overrides(config, args)
            config_raw = json.dumps(config, indent=2, ensure_ascii=False)
        if args.scan_result_csv:
            config = dict(config)
            config["scan_result_csv_file"] = args.scan_result_csv
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
