"""CLI for pyscf_auto with orca_auto-aligned command names."""

from __future__ import annotations

import argparse
import logging
import sys

from commands._helpers import default_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyscf_auto")
    parser.add_argument("--config", default=default_config_path(), help="Path to pyscf_auto.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-inp", help="Run a PySCF calculation from a .inp file.")
    run_parser.add_argument(
        "--reaction-dir",
        required=True,
        help="Directory under the configured allowed_root containing input files",
    )
    run_parser.add_argument("--max-retries", type=int, default=None)
    run_parser.add_argument("--force", action="store_true", help="Force re-run even if existing output is completed")
    run_parser.add_argument("--json", action="store_true")

    status_parser = subparsers.add_parser("status", help="Check the status of a run.")
    status_parser.add_argument("--reaction-dir", required=True, help="Directory under the configured allowed_root")
    status_parser.add_argument("--json", action="store_true")

    organize_parser = subparsers.add_parser(
        "organize",
        help="Organize completed runs into a clean directory structure.",
    )
    organize_parser.add_argument("--reaction-dir", default=None, help="Single reaction directory to organize")
    organize_parser.add_argument(
        "--root",
        default=None,
        help="Root directory to scan (mutually exclusive with --reaction-dir)",
    )
    organize_parser.add_argument("--apply", action="store_true", default=False, help="Actually move files (default is dry-run)")
    organize_parser.add_argument("--rebuild-index", action="store_true", default=False, help="Rebuild JSONL index from organized directories.")
    organize_parser.add_argument("--find", action="store_true", default=False, help="Search the index")
    organize_parser.add_argument("--run-id", default=None, help="Find by run_id (with --find)")
    organize_parser.add_argument("--job-type", default=None, help="Filter by job_type (with --find)")
    organize_parser.add_argument("--limit", type=int, default=0, help="Limit results (with --find)")
    organize_parser.add_argument("--json", action="store_true")

    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument(
        "--reaction-dir",
        default=None,
        help="Single reaction directory under organized_root to clean",
    )
    cleanup_parser.add_argument(
        "--root",
        default=None,
        help="Root directory to scan (must match organized_root)",
    )
    cleanup_parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually delete files (default is dry-run)",
    )
    cleanup_parser.add_argument("--json", action="store_true")

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

    command_map = {
        "run-inp": _cmd_run_inp,
        "status": _cmd_status,
        "organize": _cmd_organize,
        "cleanup": _cmd_cleanup,
    }
    try:
        return int(command_map[args.command](args))
    except KeyboardInterrupt:
        return 130
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or str(exc))
        logging.error(
            "Missing Python dependency: %s. Activate/install pyscf_auto environment first.",
            missing,
        )
        return 1
    except ValueError as exc:
        logging.error("%s", exc)
        return 1
    except Exception:
        logging.exception("Unexpected error")
        return 1


def _cmd_run_inp(args: argparse.Namespace) -> int:
    from commands.run_inp import cmd_run_inp

    return int(cmd_run_inp(args))


def _cmd_status(args: argparse.Namespace) -> int:
    from commands.run_inp import cmd_status

    return int(cmd_status(args))


def _cmd_organize(args: argparse.Namespace) -> int:
    from commands.organize import cmd_organize

    return int(cmd_organize(args))


def _cmd_cleanup(args: argparse.Namespace) -> int:
    from commands.cleanup import cmd_cleanup

    return int(cmd_cleanup(args))


if __name__ == "__main__":
    raise SystemExit(main())
