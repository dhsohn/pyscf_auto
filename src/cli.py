import argparse

from run_opt_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
)


def _normalize_cli_args(argv):
    if not argv:
        return argv
    command = argv[0]
    commands = {
        "run",
        "doctor",
        "validate-config",
        "status",
        "queue",
        "smoke-test",
        "scan-point",
        "list-runs",
    }
    if command in {"-h", "--help"}:
        return argv
    if command in commands:
        return argv

    if "--queue-runner" in argv:
        return ["run", *argv]
    return argv


def build_parser():
    parser = argparse.ArgumentParser(
        description="Optimize molecular geometry using PySCF and ASE."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run geometry optimization or scan workflows."
    )
    run_parser.add_argument(
        "xyz_file",
        nargs="?",
        help="Path to the .xyz file with the initial molecular geometry.",
    )
    run_parser.add_argument(
        "--solvent-map",
        default=DEFAULT_SOLVENT_MAP_PATH,
        help=(
            "Path to JSON/YAML/TOML file mapping solvent names to dielectric constants "
            f"(default: {DEFAULT_SOLVENT_MAP_PATH})."
        ),
    )
    run_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to JSON/YAML/TOML config file for runtime settings "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    run_parser.add_argument(
        "--background",
        action="store_true",
        help="Launch in the background queue (optional).",
    )
    run_parser.add_argument(
        "--profile",
        action="store_true",
        help="Record SCF/gradient/Hessian timing and cycle counts in metadata.",
    )
    run_dir_group = run_parser.add_mutually_exclusive_group()
    run_dir_group.add_argument(
        "--run-dir",
        help="Optional run directory to use (useful for background launches).",
    )
    run_dir_group.add_argument(
        "--resume",
        help=(
            "Resume from an existing run directory (loads checkpoint.json and "
            "config_used.json). Requires --force-resume for completed/failed runs."
        ),
    )
    run_parser.add_argument(
        "--resume-config-mismatch",
        choices=("warn", "error", "ignore"),
        default="warn",
        help=(
            "How to handle resume config mismatches between checkpoint/config_used "
            "and the current config (default: warn)."
        ),
    )
    run_parser.add_argument(
        "--run-id",
        help="Optional run ID to use (useful for background launches).",
    )
    run_parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Allow resuming runs marked as completed/failed/timeout/canceled.",
    )
    run_parser.add_argument(
        "--queue-priority",
        type=int,
        default=0,
        help="Priority for the queued run (higher runs first).",
    )
    run_parser.add_argument(
        "--queue-max-runtime",
        type=int,
        help="Max runtime in seconds for queued runs (timeout if exceeded).",
    )
    run_parser.add_argument(
        "--scan-dimension",
        action="append",
        metavar="SPEC",
        help=(
            "Scan dimension spec: 'type,i,j[,k[,l]],start,end,step' "
            "(e.g., bond,0,1,1.0,2.0,0.1). Provide twice for 2D scans."
        ),
    )
    run_parser.add_argument(
        "--scan-mode",
        choices=["optimization", "single_point"],
        help="Scan mode to run at each point (optimization or single_point).",
    )
    run_parser.add_argument(
        "--scan-grid",
        action="append",
        metavar="VALUES",
        help=(
            "Optional explicit grid values for each scan dimension "
            "(comma-separated, provide once per dimension)."
        ),
    )
    run_parser.add_argument(
        "--scan-result-csv",
        help="Optional CSV output path for scan results (default: scan_result.csv).",
    )
    run_parser.add_argument("--no-background", action="store_true", help=argparse.SUPPRESS)
    run_parser.add_argument("--queue-runner", action="store_true", help=argparse.SUPPRESS)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Run environment diagnostics and exit."
    )
    doctor_parser.set_defaults()

    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate the JSON config file and exit."
    )
    validate_parser.add_argument(
        "config_path",
        nargs="?",
        help=(
            "Optional JSON config path for validation "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    validate_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to JSON config file for runtime settings "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run a minimal water single-point calculation (1 SCF cycle).",
    )
    smoke_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Base config file to derive the smoke-test settings "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    smoke_parser.add_argument(
        "--run-dir",
        help="Optional base run directory for smoke-test output.",
    )
    smoke_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a smoke-test run by skipping completed cases in --run-dir.",
    )
    smoke_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the smoke-test run immediately on the first failure.",
    )
    smoke_parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Run smoke-test cases in the same process (default: isolated subprocesses).",
    )
    smoke_parser.add_argument(
        "--smoke-mode",
        choices=("quick", "full"),
        default="quick",
        help="Smoke-test scope (quick reduces the Cartesian product; default: quick).",
    )
    smoke_parser.add_argument(
        "--watch",
        action="store_true",
        help="Monitor smoke-test progress and auto-resume on stalled logs.",
    )
    smoke_parser.add_argument(
        "--watch-interval",
        type=int,
        default=30,
        help="Polling interval in seconds for smoke-test watch mode (default: 30).",
    )
    smoke_parser.add_argument(
        "--watch-timeout",
        type=int,
        default=900,
        help="Seconds without log updates before auto-resume (default: 900).",
    )
    smoke_parser.add_argument(
        "--watch-max-restarts",
        type=int,
        default=0,
        help="Max auto-resume attempts in watch mode (0 = unlimited).",
    )

    scan_point_parser = subparsers.add_parser(
        "scan-point",
        help=argparse.SUPPRESS,
    )
    scan_point_parser.add_argument(
        "--manifest",
        required=True,
        help=argparse.SUPPRESS,
    )
    scan_point_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help=argparse.SUPPRESS,
    )

    status_parser = subparsers.add_parser(
        "status", help="Show run summaries or recent status listings."
    )
    status_parser.add_argument(
        "run_path",
        nargs="?",
        help="Run directory or metadata JSON file to summarize.",
    )
    status_parser.add_argument(
        "--recent",
        type=int,
        metavar="N",
        help="Show a summary list for the most recent N runs.",
    )

    list_runs_parser = subparsers.add_parser(
        "list-runs", help="List recent runs as JSON (for GUI/remote clients)."
    )
    list_runs_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max runs to return (default: 20).",
    )
    list_runs_parser.add_argument(
        "--runs-dir",
        help="Optional runs directory override (defaults to PYSCF_AUTO_BASE_DIR/runs).",
    )

    queue_parser = subparsers.add_parser(
        "queue", help="Manage the background run queue."
    )
    queue_subparsers = queue_parser.add_subparsers(dest="queue_command", required=True)
    queue_subparsers.add_parser(
        "status", help="Show queue status (includes foreground runs)."
    )
    queue_cancel_parser = queue_subparsers.add_parser(
        "cancel", help="Cancel a queued reservation by run ID."
    )
    queue_cancel_parser.add_argument("run_id", metavar="RUN_ID")
    queue_retry_parser = queue_subparsers.add_parser(
        "retry", help="Retry a run by re-queuing it (failed/timeout/canceled runs)."
    )
    queue_retry_parser.add_argument("run_id", metavar="RUN_ID")
    queue_subparsers.add_parser(
        "requeue-failed", help="Re-queue all failed/timeout runs."
    )
    queue_prune_parser = queue_subparsers.add_parser(
        "prune", help="Remove old completed/failed queue entries."
    )
    queue_prune_parser.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Keep queue entries newer than N days (default: 30).",
    )
    queue_archive_parser = queue_subparsers.add_parser(
        "archive", help="Archive queue entries and reset the queue."
    )
    queue_archive_parser.add_argument(
        "--path",
        help=(
            "Optional archive output path (default: "
            "queue.json.YYYYMMDDHHMMSS.archive.json)."
        ),
    )
    return parser


__all__ = ["build_parser", "_normalize_cli_args"]
