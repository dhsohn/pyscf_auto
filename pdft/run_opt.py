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
    if args.validate_only and args.interactive:
        raise ValueError("--validate-only cannot be used with --interactive.")
    if args.interactive is None:
        args.interactive = not args.non_interactive
    if args.validate_only:
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
        if args.interactive:
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

        try:
            config = build_run_config(config)
        except ValueError as error:
            message = str(error)
            print(message, file=sys.stderr)
            logging.error(message)
            raise
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
