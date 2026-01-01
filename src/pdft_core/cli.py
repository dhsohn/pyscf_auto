import argparse

from .run_opt_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
)


def _normalize_cli_args(argv):
    if not argv:
        return argv
    command = argv[0]
    if command == "doctor":
        return ["--doctor", *argv[1:]]
    if command != "validate-config":
        return argv
    remaining = argv[1:]
    config_path = None
    if remaining and not remaining[0].startswith("-"):
        config_path = remaining[0]
        remaining = remaining[1:]
    normalized = ["--validate-only"]
    if config_path:
        normalized.extend(["--config", config_path])
    normalized.extend(remaining)
    return normalized


def build_parser():
    parser = argparse.ArgumentParser(
        description="Optimize molecular geometry using PySCF and ASE."
    )
    parser.add_argument(
        "xyz_file",
        nargs="?",
        help="Path to the .xyz file with the initial molecular geometry.",
    )
    parser.add_argument(
        "--solvent-map",
        default=DEFAULT_SOLVENT_MAP_PATH,
        help=(
            "Path to JSON file mapping solvent names to dielectric constants "
            f"(default: {DEFAULT_SOLVENT_MAP_PATH})."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to JSON config file for runtime settings "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=None,
        help="Prompt for run settings interactively (default).",
    )
    parser.add_argument(
        "--non-interactive",
        "--advanced",
        action="store_true",
        help="Run with explicit inputs/configs without prompts (advanced).",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Launch in the background queue (optional).",
    )
    run_dir_group = parser.add_mutually_exclusive_group()
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
    parser.add_argument(
        "--run-id",
        help="Optional run ID to use (useful for background launches).",
    )
    parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Allow resuming runs marked as completed/failed/timeout/canceled.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the JSON config file and exit without running a calculation.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run environment diagnostics and exit (e.g., python run_opt.py --doctor).",
    )
    parser.add_argument(
        "--status",
        help=(
            "Show a summary for a run directory or metadata JSON file "
            "(e.g., --status runs/2024-01-01_120000)."
        ),
    )
    parser.add_argument(
        "--queue-status",
        action="store_true",
        help="Show queue status (includes foreground runs).",
    )
    parser.add_argument(
        "--queue-cancel",
        metavar="RUN_ID",
        help="Cancel a queued reservation by run ID.",
    )
    parser.add_argument(
        "--queue-retry",
        metavar="RUN_ID",
        help="Retry a run by re-queuing it (failed/timeout/canceled runs).",
    )
    parser.add_argument(
        "--queue-requeue-failed",
        action="store_true",
        help="Re-queue all failed/timeout runs.",
    )
    parser.add_argument(
        "--queue-priority",
        type=int,
        default=0,
        help="Priority for the queued run (higher runs first).",
    )
    parser.add_argument(
        "--queue-max-runtime",
        type=int,
        help="Max runtime in seconds for queued runs (timeout if exceeded).",
    )
    parser.add_argument(
        "--status-recent",
        type=int,
        metavar="N",
        help="Show a summary list for the most recent N runs.",
    )
    parser.add_argument(
        "--scan-dimension",
        action="append",
        metavar="SPEC",
        help=(
            "Scan dimension spec: 'type,i,j[,k[,l]],start,end,step' "
            "(e.g., bond,0,1,1.0,2.0,0.1). Provide twice for 2D scans."
        ),
    )
    parser.add_argument(
        "--scan-mode",
        choices=["optimization", "single_point"],
        help="Scan mode to run at each point (optimization or single_point).",
    )
    parser.add_argument(
        "--scan-grid",
        action="append",
        metavar="VALUES",
        help=(
            "Optional explicit grid values for each scan dimension "
            "(comma-separated, provide once per dimension)."
        ),
    )
    parser.add_argument(
        "--scan-result-csv",
        help="Optional CSV output path for scan results (default: scan_result.csv).",
    )
    parser.add_argument("--no-background", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--queue-runner", action="store_true", help=argparse.SUPPRESS)
    return parser


__all__ = ["build_parser", "_normalize_cli_args"]
