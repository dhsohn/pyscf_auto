import os
import sys
import traceback
from datetime import datetime

from run_queue import (
    enqueue_run,
    ensure_queue_runner_started,
    record_status_event,
)
from run_opt_config import (
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_QUEUE_RUNNER_LOG_PATH,
)
from run_opt_metadata import write_run_metadata
from .types import RunContext


def enqueue_background_run(args, context: RunContext):
    queued_at = datetime.now().isoformat()
    queue_priority = args.queue_priority
    max_runtime_seconds = args.queue_max_runtime
    queued_metadata = {
        "status": "queued",
        "run_directory": context["run_dir"],
        "run_id": context["run_id"],
        "attempt": context["attempt"],
        "run_id_history": context["run_id_history"],
        "xyz_file": args.xyz_file,
        "config_file": args.config,
        "run_metadata_file": context["run_metadata_path"],
        "log_file": context["log_path"],
        "event_log_file": context["event_log_path"],
        "queued_at": queued_at,
        "priority": queue_priority,
        "max_runtime_seconds": max_runtime_seconds,
    }
    write_run_metadata(context["run_metadata_path"], queued_metadata)
    queue_entry = {
        "status": "queued",
        "run_directory": context["run_dir"],
        "run_id": context["run_id"],
        "attempt": context["attempt"],
        "run_id_history": context["run_id_history"],
        "xyz_file": args.xyz_file,
        "config_file": args.config,
        "solvent_map": args.solvent_map,
        "run_metadata_file": context["run_metadata_path"],
        "log_file": context["log_path"],
        "event_log_file": context["event_log_path"],
        "queued_at": queued_at,
        "priority": queue_priority,
        "max_runtime_seconds": max_runtime_seconds,
        "retry_count": 0,
    }
    position = enqueue_run(queue_entry, DEFAULT_QUEUE_PATH, DEFAULT_QUEUE_LOCK_PATH)
    record_status_event(
        context["event_log_path"],
        context["run_id"],
        context["run_dir"],
        "queued",
        previous_status=context.get("previous_status"),
        details={
            "priority": queue_priority,
            "max_runtime_seconds": max_runtime_seconds,
        },
    )
    runner_command = [
        sys.executable,
        os.path.abspath(sys.argv[0]),
        "--queue-runner",
    ]
    ensure_queue_runner_started(runner_command, DEFAULT_QUEUE_RUNNER_LOG_PATH)
    print("Background run queued.")
    print(f"  Run ID       : {context['run_id']}")
    print(f"  Queue pos    : {position}")
    print(f"  Run dir      : {context['run_dir']}")
    print(f"  Metadata     : {context['run_metadata_path']}")
    print(f"  Log file     : {context['log_path']}")
    print(f"  Queue runner : {DEFAULT_QUEUE_RUNNER_LOG_PATH}")


def finalize_metadata(
    run_metadata_path,
    event_log_path,
    run_id,
    run_dir,
    metadata,
    status,
    previous_status,
    queue_update_fn=None,
    exit_code=None,
    details=None,
    error=None,
):
    metadata["status"] = status
    metadata["run_ended_at"] = datetime.now().isoformat()
    metadata["run_updated_at"] = datetime.now().isoformat()
    if error is not None:
        metadata["error"] = str(error)
        metadata["traceback"] = traceback.format_exc()
    write_run_metadata(run_metadata_path, metadata)
    record_status_event(
        event_log_path,
        run_id,
        run_dir,
        status,
        previous_status=previous_status,
        details=details,
    )
    if queue_update_fn:
        queue_update_fn(status, exit_code=exit_code)
