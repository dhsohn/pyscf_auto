"""New CLI for pyscf_auto with .inp file-based workflow.

Usage::

    pyscf_auto run-inp --reaction-dir ~/pyscf_runs/ts_water
    pyscf_auto status --reaction-dir ~/pyscf_runs/ts_water
    pyscf_auto organize --root ~/pyscf_runs --apply
    pyscf_auto doctor
    pyscf_auto validate input.inp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyscf_auto",
        description="PySCF calculation workflow with .inp file input.",
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
        help="Directory containing the .inp file and geometry.",
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
        help="Re-run even if a completed result exists.",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output progress as JSON.",
    )
    run_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable SCF/gradient timing profiling.",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    run_parser.add_argument(
        "--config",
        default=None,
        help="Path to app config YAML (default: ~/.pyscf_auto/config.yaml).",
    )

    # --- status ---
    status_parser = subparsers.add_parser(
        "status",
        help="Check the status of a run.",
    )
    status_parser.add_argument(
        "--reaction-dir",
        required=True,
        help="Reaction directory to check.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON.",
    )

    # --- organize ---
    organize_parser = subparsers.add_parser(
        "organize",
        help="Organize completed runs into a clean directory structure.",
    )
    org_group = organize_parser.add_mutually_exclusive_group()
    org_group.add_argument(
        "--reaction-dir",
        help="Organize a single reaction directory.",
    )
    org_group.add_argument(
        "--root",
        help="Organize all reaction directories under this root.",
    )
    organize_parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy files (default is dry-run preview).",
    )
    organize_parser.add_argument(
        "--find",
        dest="find_query",
        help="Search organized outputs by run_id.",
    )
    organize_parser.add_argument(
        "--job-type",
        help="Filter search by job type.",
    )
    organize_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum search results.",
    )
    organize_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON.",
    )
    organize_parser.add_argument(
        "--config",
        default=None,
        help="Path to app config YAML.",
    )

    # --- doctor ---
    subparsers.add_parser(
        "doctor",
        help="Run environment diagnostics.",
    )

    # --- validate ---
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a .inp file without running.",
    )
    validate_parser.add_argument(
        "inp_file",
        help="Path to the .inp file to validate.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        if args.command == "run-inp":
            _cmd_run_inp(args)
        elif args.command == "status":
            _cmd_status(args)
        elif args.command == "organize":
            _cmd_organize(args)
        elif args.command == "doctor":
            _cmd_doctor()
        elif args.command == "validate":
            _cmd_validate(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logging.error("%s", exc)
        sys.exit(1)


def _cmd_run_inp(args: argparse.Namespace) -> None:
    from app_config import load_app_config
    from runner.orchestrator import cmd_run_inp

    app_config = load_app_config(getattr(args, "config", None))
    exit_code = cmd_run_inp(
        reaction_dir=args.reaction_dir,
        max_retries=args.max_retries,
        force=args.force,
        json_output=args.json_output,
        profile=args.profile,
        verbose=args.verbose,
        app_config=app_config,
    )
    sys.exit(exit_code)


def _cmd_status(args: argparse.Namespace) -> None:
    from runner.orchestrator import cmd_status

    exit_code = cmd_status(
        reaction_dir=args.reaction_dir,
        json_output=args.json_output,
    )
    sys.exit(exit_code)


def _cmd_organize(args: argparse.Namespace) -> None:
    from app_config import load_app_config
    from organizer.result_organizer import (
        find_organized_runs,
        organize_all,
        organize_run,
    )

    app_config = load_app_config(getattr(args, "config", None))
    organized_root = app_config.runtime.organized_root

    # Search mode
    if args.find_query:
        results = find_organized_runs(
            organized_root,
            run_id=args.find_query,
            job_type=args.job_type,
            limit=args.limit,
        )
        if args.json_output:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(f"  {r.get('run_id', '?')} | {r.get('job_type', '?')} | "
                      f"{r.get('molecule_key', '?')} | {r.get('target', '?')}")
            print(f"\nFound {len(results)} runs.")
        return

    # Organize mode
    if args.reaction_dir:
        result = organize_run(args.reaction_dir, organized_root, apply=args.apply)
        if result:
            if args.json_output:
                print(json.dumps(result, indent=2))
            else:
                action = "Organized" if result["applied"] else "Would organize"
                print(f"{action}: {result['source']} -> {result['target']}")
    elif args.root:
        results = organize_all(args.root, organized_root, apply=args.apply)
        if args.json_output:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                action = "Organized" if r["applied"] else "Would organize"
                print(f"  {action}: {r.get('run_id', '?')} -> {r['target']}")
            print(f"\nTotal: {len(results)} runs.")
    else:
        # Default: organize from allowed_root
        root = app_config.runtime.allowed_root
        results = organize_all(root, organized_root, apply=args.apply)
        if args.json_output:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                action = "Organized" if r["applied"] else "Would organize"
                print(f"  {action}: {r.get('run_id', '?')} -> {r['target']}")
            print(f"\nTotal: {len(results)} runs.")
            if not args.apply and results:
                print("Use --apply to actually copy files.")


def _cmd_doctor() -> None:
    import workflow

    workflow.run_doctor()


def _cmd_validate(args: argparse.Namespace) -> None:
    from inp.parser import inp_config_to_dict, parse_inp_file
    from run_opt_config import build_run_config

    try:
        inp = parse_inp_file(args.inp_file)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Validation FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Job type:    {inp.job_type}")
    print(f"Functional:  {inp.functional}")
    print(f"Basis:       {inp.basis}")
    print(f"Charge:      {inp.charge}")
    print(f"Mult:        {inp.multiplicity}")
    print(f"Dispersion:  {inp.dispersion or 'none'}")
    print(f"Solvent:     {inp.solvent_name or 'vacuum'} ({inp.solvent_model or '-'})")
    print(f"Opt mode:    {inp.optimizer_mode or '-'}")

    if inp.scf:
        print(f"SCF:         {inp.scf}")
    if inp.runtime:
        print(f"Runtime:     {inp.runtime}")

    # Test conversion to RunConfig
    config_dict = inp_config_to_dict(inp)
    try:
        build_run_config(config_dict)
        print("\nRunConfig validation: PASSED")
    except ValueError as exc:
        print(f"\nRunConfig validation FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDerived config:\n{json.dumps(config_dict, indent=2)}")


if __name__ == "__main__":
    main()
