import argparse
import warnings

from run_opt_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
)


def _normalize_cli_args(argv):
    if not argv:
        return argv
    command = argv[0]
    commands = {"run", "doctor", "validate-config", "status", "queue"}
    if command in {"-h", "--help"}:
        return argv
    if command in commands:
        return argv

    if "--doctor" in argv:
        warnings.warn(
            "`--doctor` is deprecated; use `doctor` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        remaining = [arg for arg in argv if arg != "--doctor"]
        return ["doctor", *remaining]

    if "--validate-only" in argv:
        warnings.warn(
            "`--validate-only` is deprecated; use `validate-config` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        remaining = [arg for arg in argv if arg != "--validate-only"]
        return ["validate-config", *remaining]

    queue_command = None
    queue_run_id = None
    if "--queue-status" in argv:
        queue_command = "status"
    if "--queue-cancel" in argv:
        queue_command = "cancel"
        index = argv.index("--queue-cancel")
        if index + 1 < len(argv):
            queue_run_id = argv[index + 1]
    if "--queue-retry" in argv:
        queue_command = "retry"
        index = argv.index("--queue-retry")
        if index + 1 < len(argv):
            queue_run_id = argv[index + 1]
    if "--queue-requeue-failed" in argv:
        queue_command = "requeue-failed"
    if queue_command:
        warnings.warn(
            "Queue flags are deprecated; use `queue <command>` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        remaining = [
            arg
            for arg in argv
            if arg
            not in {
                "--queue-status",
                "--queue-cancel",
                "--queue-retry",
                "--queue-requeue-failed",
            }
        ]
        if queue_run_id and queue_run_id in remaining:
            remaining.remove(queue_run_id)
        normalized = ["queue", queue_command]
        if queue_run_id:
            normalized.append(queue_run_id)
        normalized.extend(remaining)
        return normalized

    if "--status" in argv or "--status-recent" in argv:
        warnings.warn(
            "`--status` flags are deprecated; use `status` subcommands instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        remaining = list(argv)
        status_target = None
        recent_value = None
        if "--status" in remaining:
            index = remaining.index("--status")
            if index + 1 < len(remaining):
                status_target = remaining[index + 1]
            del remaining[index : index + 2]
        if "--status-recent" in remaining:
            index = remaining.index("--status-recent")
            if index + 1 < len(remaining):
                recent_value = remaining[index + 1]
            del remaining[index : index + 2]
        normalized = ["status"]
        if status_target:
            normalized.append(status_target)
        if recent_value:
            normalized.extend(["--recent", recent_value])
        normalized.extend(remaining)
        return normalized

    if "--queue-runner" in argv:
        return ["run", *argv]

    warnings.warn(
        "Default command is deprecated; use `run` explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ["run", *argv]


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
        "--interactive",
        action="store_true",
        default=None,
        help="Prompt for run settings interactively (default).",
    )
    run_parser.add_argument(
        "--non-interactive",
        "--advanced",
        action="store_true",
        help="Run with explicit inputs/configs without prompts (advanced).",
    )
    run_parser.add_argument(
        "--background",
        action="store_true",
        help="Launch in the background queue (optional).",
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
