import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime

from .run_opt_config import (
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_QUEUE_RUNNER_LOCK_PATH,
    DEFAULT_RUN_METADATA_PATH,
)
from .run_opt_metadata import write_run_metadata
from .run_opt_resources import ensure_parent_dir


def _queue_priority_value(entry):
    try:
        return int(entry.get("priority", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _parse_iso_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _queue_entry_sort_key(entry):
    priority = _queue_priority_value(entry)
    queued_at = _parse_iso_timestamp(entry.get("queued_at")) or datetime.min
    return (-priority, queued_at, entry.get("run_id") or "")


def _queue_backup_path(queue_path):
    return f"{queue_path}.bak"


def _queue_corrupt_path(queue_path):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{queue_path}.{timestamp}.corrupt"


def _load_queue_from_path(queue_path):
    with open(queue_path, "r", encoding="utf-8") as queue_file:
        return json.load(queue_file)


def _handle_corrupt_queue(queue_path):
    backup_path = _queue_backup_path(queue_path)
    try:
        backup_state = _load_queue_from_path(backup_path)
    except (OSError, json.JSONDecodeError):
        backup_state = None
    if backup_state is not None:
        corrupt_path = _queue_corrupt_path(queue_path)
        try:
            os.replace(queue_path, corrupt_path)
        except OSError:
            logging.warning("Failed to move corrupt queue file to %s", corrupt_path)
        else:
            logging.warning("Moved corrupt queue file to %s", corrupt_path)
        try:
            shutil.copy2(backup_path, queue_path)
        except OSError:
            logging.warning("Failed to restore queue from backup at %s", backup_path)
        else:
            logging.warning("Restored queue from backup at %s", backup_path)
        return backup_state
    corrupt_path = _queue_corrupt_path(queue_path)
    try:
        os.replace(queue_path, corrupt_path)
    except OSError:
        logging.warning("Failed to move corrupt queue file to %s", corrupt_path)
    else:
        logging.warning("Moved corrupt queue file to %s", corrupt_path)
    return {"entries": [], "updated_at": None}


def _load_queue(queue_path):
    if not os.path.exists(queue_path):
        return {"entries": [], "updated_at": None}
    try:
        return _load_queue_from_path(queue_path)
    except OSError:
        logging.warning("Failed to read queue file: %s", queue_path)
        return {"entries": [], "updated_at": None}
    except json.JSONDecodeError:
        logging.warning("Queue file is corrupt: %s", queue_path)
        return _handle_corrupt_queue(queue_path)


def _write_queue(queue_path, queue_state):
    queue_state["updated_at"] = datetime.now().isoformat()
    ensure_parent_dir(queue_path)
    queue_dir = os.path.dirname(queue_path) or "."
    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=queue_dir,
        prefix=".queue.json.",
        suffix=".tmp",
        delete=False,
    )
    try:
        backup_path = _queue_backup_path(queue_path)
        if os.path.exists(queue_path):
            try:
                shutil.copy2(queue_path, backup_path)
            except OSError:
                logging.warning("Failed to create queue backup at %s", backup_path)
        with temp_handle as queue_file:
            json.dump(queue_state, queue_file, indent=2)
            queue_file.flush()
            os.fsync(queue_file.fileno())
        os.replace(temp_handle.name, queue_path)
    finally:
        if os.path.exists(temp_handle.name):
            try:
                os.remove(temp_handle.name)
            except FileNotFoundError:
                pass


def _read_lock_info(lock_path):
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            contents = handle.read().strip()
    except OSError:
        return None, None
    if not contents:
        return None, None
    parts = contents.split()
    try:
        pid = int(parts[0])
    except ValueError:
        pid = None
    timestamp = parts[1] if len(parts) > 1 else None
    return pid, timestamp


def _is_pid_running(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _is_lock_stale(lock_path, stale_timeout):
    pid, _timestamp = _read_lock_info(lock_path)
    if pid and _is_pid_running(pid):
        return False
    try:
        mtime = os.path.getmtime(lock_path)
    except OSError:
        return False
    return (time.time() - mtime) > stale_timeout


def _acquire_lock(lock_path, timeout=10, delay=0.1, stale_timeout=60):
    start_time = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _is_lock_stale(lock_path, stale_timeout):
                try:
                    os.remove(lock_path)
                    continue
                except FileNotFoundError:
                    continue
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}") from None
            time.sleep(delay)
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()} {datetime.now().isoformat()}")
            return


@contextmanager
def _queue_lock(lock_path):
    _acquire_lock(lock_path)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _ensure_queue_file(queue_path):
    ensure_parent_dir(queue_path)
    if not os.path.exists(queue_path):
        _write_queue(queue_path, {"entries": [], "updated_at": datetime.now().isoformat()})


def _read_runner_pid(lock_path):
    if not os.path.exists(lock_path):
        return None
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _ensure_queue_runner_started(command, log_path):
    ensure_parent_dir(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
    if log_path:
        ensure_parent_dir(log_path)
    lock_exists = os.path.exists(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
    existing_pid = _read_runner_pid(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
    if existing_pid is None and lock_exists:
        try:
            os.remove(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
        except FileNotFoundError:
            pass
    if existing_pid and _is_pid_running(existing_pid):
        return
    if existing_pid and not _is_pid_running(existing_pid):
        try:
            os.remove(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
        except FileNotFoundError:
            pass
    if log_path:
        with open(log_path, "a", encoding="utf-8") as log_file:
            subprocess.Popen(
                command,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
    else:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _enqueue_run(entry, queue_path, lock_path):
    _ensure_queue_file(queue_path)
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        entries = queue_state.get("entries") or []
        if any(item.get("run_id") == entry["run_id"] for item in entries):
            raise ValueError(f"Run ID already queued: {entry['run_id']}")
        entries.append(entry)
        queue_state["entries"] = entries
        _write_queue(queue_path, queue_state)
        queued_positions = [item for item in entries if item.get("status") == "queued"]
        return len(queued_positions)


def _register_foreground_run(entry, queue_path, lock_path):
    _ensure_queue_file(queue_path)
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        entries = queue_state.get("entries") or []
        if any(item.get("run_id") == entry["run_id"] for item in entries):
            raise ValueError(f"Run ID already queued: {entry['run_id']}")
        entries.append(entry)
        queue_state["entries"] = entries
        _write_queue(queue_path, queue_state)


def _update_queue_entry(queue_path, lock_path, run_id, updater):
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        entries = queue_state.get("entries") or []
        updated = False
        for item in entries:
            if item.get("run_id") == run_id:
                updater(item)
                updated = True
                break
        if updated:
            _write_queue(queue_path, queue_state)
        return updated


def _update_queue_status(queue_path, lock_path, run_id, status, exit_code=None):
    timestamp = datetime.now().isoformat()

    def _apply_update(entry):
        if entry.get("run_id") == run_id:
            entry["status"] = status
            if status in ("running", "started"):
                entry["started_at"] = entry.get("started_at") or timestamp
            if status in ("completed", "failed", "timeout", "canceled"):
                entry["ended_at"] = timestamp
                entry["exit_code"] = exit_code

    _update_queue_entry(queue_path, lock_path, run_id, _apply_update)


def _load_run_metadata(metadata_path):
    if not metadata_path or not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def _append_event_log(event_log_path, payload):
    if not event_log_path:
        return
    try:
        ensure_parent_dir(event_log_path)
        with open(event_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        logging.warning("Failed to write event log: %s", event_log_path)


def _record_status_event(event_log_path, run_id, run_dir, status, previous_status=None, details=None):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "event": "status_transition",
        "run_id": run_id,
        "run_directory": run_dir,
        "status": status,
        "previous_status": previous_status,
    }
    if details:
        payload["details"] = details
    _append_event_log(event_log_path, payload)


def _cancel_queue_entry(queue_path, lock_path, run_id):
    event_log_path = None
    run_dir = None

    def _apply_cancel(entry):
        if entry.get("status") == "queued":
            entry["status"] = "canceled"
            entry["canceled_at"] = datetime.now().isoformat()
            entry["exit_code"] = None

    updated = _update_queue_entry(queue_path, lock_path, run_id, _apply_cancel)
    if not updated:
        return False, "Run ID not found in queue."

    metadata_path = None
    status = None
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        for item in queue_state.get("entries") or []:
            if item.get("run_id") == run_id:
                metadata_path = item.get("run_metadata_file")
                status = item.get("status")
                event_log_path = item.get("event_log_file")
                run_dir = item.get("run_directory")
                break
    if status != "canceled":
        return False, "Run is not queued (already running or completed)."
    if metadata_path:
        metadata = _load_run_metadata(metadata_path) or {}
        metadata["status"] = "canceled"
        metadata["run_ended_at"] = datetime.now().isoformat()
        metadata["canceled_at"] = metadata["run_ended_at"]
        write_run_metadata(metadata_path, metadata)
    _record_status_event(
        event_log_path,
        run_id,
        run_dir,
        "canceled",
        previous_status="queued",
    )
    return True, None


def _requeue_queue_entry(queue_path, lock_path, run_id, reason):
    requeued_at = datetime.now().isoformat()
    event_log_path = None
    run_dir = None
    metadata_path = None
    previous_status = None

    def _apply_requeue(entry):
        nonlocal event_log_path, run_dir, metadata_path, previous_status
        previous_status = entry.get("status")
        entry["status"] = "queued"
        entry["queued_at"] = requeued_at
        entry["started_at"] = None
        entry["ended_at"] = None
        entry["exit_code"] = None
        entry["retry_count"] = int(entry.get("retry_count", 0) or 0) + 1
        entry["requeued_at"] = requeued_at
        event_log_path = entry.get("event_log_file")
        run_dir = entry.get("run_directory")
        metadata_path = entry.get("run_metadata_file")

    updated = _update_queue_entry(queue_path, lock_path, run_id, _apply_requeue)
    if not updated:
        return False, "Run ID not found in queue."
    if previous_status == "queued":
        return False, "Run is already queued."
    if metadata_path:
        metadata = _load_run_metadata(metadata_path) or {}
        metadata["status"] = "queued"
        metadata["queued_at"] = requeued_at
        metadata["run_started_at"] = None
        metadata["run_ended_at"] = None
        metadata["requeued_at"] = requeued_at
        write_run_metadata(metadata_path, metadata)
    _record_status_event(
        event_log_path,
        run_id,
        run_dir,
        "queued",
        previous_status=previous_status,
        details={"reason": reason},
    )
    return True, None


def _requeue_failed_entries(queue_path, lock_path):
    requeued = []
    failed_statuses = {"failed", "timeout"}
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        entries = queue_state.get("entries") or []
        for entry in entries:
            status_before = entry.get("status")
            if status_before in failed_statuses:
                entry["status"] = "queued"
                entry["queued_at"] = datetime.now().isoformat()
                entry["started_at"] = None
                entry["ended_at"] = None
                entry["exit_code"] = None
                entry["retry_count"] = int(entry.get("retry_count", 0) or 0) + 1
                entry["requeued_at"] = entry["queued_at"]
                requeued.append({"entry": dict(entry), "previous_status": status_before})
        if requeued:
            _write_queue(queue_path, queue_state)
    for item in requeued:
        entry = item["entry"]
        metadata_path = entry.get("run_metadata_file")
        if metadata_path:
            metadata = _load_run_metadata(metadata_path) or {}
            metadata["status"] = "queued"
            metadata["queued_at"] = entry.get("queued_at")
            metadata["run_started_at"] = None
            metadata["run_ended_at"] = None
            metadata["requeued_at"] = entry.get("requeued_at")
            write_run_metadata(metadata_path, metadata)
        _record_status_event(
            entry.get("event_log_file"),
            entry.get("run_id"),
            entry.get("run_directory"),
            "queued",
            previous_status=item.get("previous_status"),
            details={"reason": "requeue_failed"},
        )
    return len(requeued)


def _format_queue_status(queue_state):
    entries = queue_state.get("entries") or []
    if not entries:
        print("Queue is empty.")
        return
    print("Queue status")
    queued_index = 0
    for entry in entries:
        status = entry.get("status", "unknown")
        priority = _queue_priority_value(entry)
        max_runtime_seconds = entry.get("max_runtime_seconds")
        if status == "queued":
            queued_index += 1
            position = f"{queued_index}"
        else:
            position = "-"
        timestamp_label = "queued_at"
        timestamp_value = entry.get("queued_at")
        if status in ("running", "started"):
            timestamp_label = "started_at"
            timestamp_value = entry.get("started_at") or entry.get("run_started_at")
        elif status not in ("queued",):
            timestamp_label = "ended_at"
            timestamp_value = entry.get("ended_at")
        exit_code = entry.get("exit_code")
        exit_code_label = f", exit_code={exit_code}" if exit_code is not None else ""
        print(
            "  [{pos}] {run_id} {status} ({timestamp}={timestamp_value}, priority={priority}{exit_code})".format(
                pos=position,
                run_id=entry.get("run_id"),
                status=status,
                timestamp=timestamp_label,
                timestamp_value=timestamp_value,
                priority=priority,
                exit_code=exit_code_label,
            )
        )
        run_dir = entry.get("run_directory")
        if run_dir:
            print(f"        run_dir={run_dir}")
        if max_runtime_seconds:
            print(f"        max_runtime_seconds={max_runtime_seconds}")


def _run_queue_worker(script_path, queue_path, lock_path, runner_lock_path):
    ensure_parent_dir(queue_path)
    try:
        fd = os.open(runner_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except OSError:
        return
    try:
        while True:
            entry = None
            entry_status_before = None
            with _queue_lock(lock_path):
                queue_state = _load_queue(queue_path)
                entries = queue_state.get("entries") or []
                candidates = [
                    (index, item)
                    for index, item in enumerate(entries)
                    if item.get("status") == "queued"
                ]
                if candidates:
                    index, item = min(
                        candidates,
                        key=lambda candidate: _queue_entry_sort_key(candidate[1]),
                    )
                    entry_status_before = item.get("status")
                    item["status"] = "running"
                    item["started_at"] = datetime.now().isoformat()
                    entry = dict(item)
                    entries[index] = item
                    _write_queue(queue_path, queue_state)
            if entry is None:
                break
            _record_status_event(
                entry.get("event_log_file"),
                entry.get("run_id"),
                entry.get("run_directory"),
                "running",
                previous_status=entry_status_before,
                details={
                    "priority": _queue_priority_value(entry),
                    "max_runtime_seconds": entry.get("max_runtime_seconds"),
                },
            )
            command = [
                sys.executable,
                script_path,
                entry["xyz_file"],
                "--config",
                entry["config_file"],
                "--solvent-map",
                entry["solvent_map"],
                "--run-dir",
                entry["run_directory"],
                "--run-id",
                entry["run_id"],
                "--no-background",
                "--non-interactive",
            ]
            timeout_seconds = entry.get("max_runtime_seconds")
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    timeout=timeout_seconds if timeout_seconds else None,
                )
                status = "completed" if result.returncode == 0 else "failed"
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
                exit_code = -1
            finished_at = datetime.now().isoformat()

            def _apply_update(
                item,
                run_id=entry["run_id"],
                update_status=status,
                finished_time=finished_at,
                update_exit_code=exit_code,
            ):
                if item.get("run_id") == run_id:
                    item["status"] = update_status
                    item["ended_at"] = finished_time
                    item["exit_code"] = update_exit_code

            _update_queue_entry(queue_path, lock_path, entry["run_id"], _apply_update)
            _record_status_event(
                entry.get("event_log_file"),
                entry.get("run_id"),
                entry.get("run_directory"),
                status,
                previous_status="running",
                details={"exit_code": exit_code},
            )
    finally:
        try:
            os.remove(runner_lock_path)
        except FileNotFoundError:
            pass


def _resolve_status_metadata_path(status_target, default_metadata_name):
    if os.path.isdir(status_target):
        candidate = os.path.join(status_target, default_metadata_name)
        if os.path.exists(candidate):
            return candidate
        matches = sorted(glob.glob(os.path.join(status_target, "*metadata*.json")))
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise FileNotFoundError(
                "Multiple metadata files found in {path}: {matches}".format(
                    path=status_target, matches=", ".join(matches)
                )
            )
        raise FileNotFoundError(
            "No metadata JSON found in {path}.".format(path=status_target)
        )
    if os.path.isfile(status_target):
        return status_target
    raise FileNotFoundError(f"Status target not found: {status_target}")


def _format_elapsed(elapsed_seconds):
    if elapsed_seconds is None:
        return None
    seconds = int(elapsed_seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _tail_last_line(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = b""
            while position > 0:
                read_size = min(4096, position)
                position -= read_size
                handle.seek(position)
                buffer = handle.read(read_size) + buffer
                if b"\n" in buffer:
                    break
    except OSError:
        return None
    for line in reversed(buffer.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped.decode("utf-8", errors="ignore")
    return None


def _print_status(status_target, default_metadata_name):
    metadata_path = _resolve_status_metadata_path(status_target, default_metadata_name)
    metadata = _load_run_metadata(metadata_path)
    if not metadata:
        raise FileNotFoundError(f"Metadata file is empty: {metadata_path}")
    summary = metadata.get("summary") or {}
    run_dir = metadata.get("run_directory") or os.path.dirname(metadata_path)
    run_id = metadata.get("run_id")
    status = metadata.get("status") or "unknown"
    run_started_at = metadata.get("run_started_at")
    run_ended_at = metadata.get("run_ended_at")
    elapsed_seconds = summary.get("elapsed_seconds")
    if elapsed_seconds is None and run_started_at and status in ("started", "running", "queued"):
        try:
            started_dt = datetime.fromisoformat(run_started_at)
            elapsed_seconds = (datetime.now() - started_dt).total_seconds()
        except ValueError:
            elapsed_seconds = None
    n_steps = summary.get("n_steps") if summary else metadata.get("n_steps")
    final_energy = summary.get("final_energy") if summary else None
    log_path = metadata.get("log_file")
    last_log_line = _tail_last_line(log_path)
    print("Run status summary")
    print(f"  Status       : {status}")
    if run_id:
        print(f"  Run ID       : {run_id}")
    if run_dir:
        print(f"  Run dir      : {run_dir}")
    if run_started_at:
        print(f"  Started at   : {run_started_at}")
    if run_ended_at:
        print(f"  Ended at     : {run_ended_at}")
    if elapsed_seconds is not None:
        print(f"  Elapsed      : {_format_elapsed(elapsed_seconds)}")
    if n_steps is not None:
        print(f"  Steps        : {n_steps}")
    if final_energy is not None:
        print(f"  Final energy : {final_energy}")
    if log_path:
        print(f"  Log file     : {log_path}")
    if last_log_line:
        print(f"  Last log     : {last_log_line}")
    optimized_xyz = metadata.get("optimized_xyz_file")
    if optimized_xyz:
        print(f"  Optimized XYZ: {optimized_xyz}")
    print(f"  Metadata     : {metadata_path}")


def _print_recent_statuses(count, base_dir="runs"):
    if count is None or count <= 0:
        raise ValueError("Recent status count must be a positive integer.")
    metadata_paths = sorted(glob.glob(os.path.join(base_dir, "*", "metadata*.json")))
    items = []
    for path in metadata_paths:
        metadata = _load_run_metadata(path)
        if not metadata:
            continue
        status = metadata.get("status") or "unknown"
        run_id = metadata.get("run_id")
        run_dir = metadata.get("run_directory") or os.path.dirname(path)
        run_started_at = metadata.get("run_started_at")
        run_ended_at = metadata.get("run_ended_at")
        summary = metadata.get("summary") or {}
        elapsed_seconds = summary.get("elapsed_seconds")
        if elapsed_seconds is None and run_started_at and status in ("started", "running", "queued"):
            started_dt = _parse_iso_timestamp(run_started_at)
            if started_dt:
                elapsed_seconds = (datetime.now() - started_dt).total_seconds()
        sort_key = _parse_iso_timestamp(run_started_at) or _parse_iso_timestamp(
            metadata.get("run_updated_at")
        )
        if sort_key is None:
            try:
                sort_key = datetime.fromtimestamp(os.path.getmtime(path))
            except OSError:
                sort_key = datetime.min
        items.append(
            {
                "path": path,
                "run_id": run_id,
                "run_dir": run_dir,
                "status": status,
                "run_started_at": run_started_at,
                "run_ended_at": run_ended_at,
                "elapsed": _format_elapsed(elapsed_seconds) if elapsed_seconds is not None else None,
                "final_energy": summary.get("final_energy"),
                "n_steps": summary.get("n_steps"),
                "sort_key": sort_key,
            }
        )
    if not items:
        print("No recent runs found.")
        return
    items.sort(key=lambda item: item["sort_key"], reverse=True)
    print(f"Recent runs (latest {min(count, len(items))})")
    for item in items[:count]:
        print(
            "  {status:9} {run_id} (started={started}, ended={ended})".format(
                status=item["status"],
                run_id=item["run_id"],
                started=item["run_started_at"],
                ended=item["run_ended_at"],
            )
        )
        print(f"        run_dir={item['run_dir']}")
        if item["elapsed"]:
            print(f"        elapsed={item['elapsed']}")
        if item["n_steps"] is not None:
            print(f"        steps={item['n_steps']}")
        if item["final_energy"] is not None:
            print(f"        final_energy={item['final_energy']}")
        print(f"        metadata={item['path']}")


def ensure_queue_file(queue_path=DEFAULT_QUEUE_PATH):
    return _ensure_queue_file(queue_path)


def queue_lock(lock_path=DEFAULT_QUEUE_LOCK_PATH):
    return _queue_lock(lock_path)


def enqueue_run(entry, queue_path=DEFAULT_QUEUE_PATH, lock_path=DEFAULT_QUEUE_LOCK_PATH):
    return _enqueue_run(entry, queue_path, lock_path)


def register_foreground_run(
    entry, queue_path=DEFAULT_QUEUE_PATH, lock_path=DEFAULT_QUEUE_LOCK_PATH
):
    return _register_foreground_run(entry, queue_path, lock_path)


def update_queue_status(
    queue_path, lock_path, run_id, status, exit_code=None
):
    return _update_queue_status(queue_path, lock_path, run_id, status, exit_code=exit_code)


def cancel_queue_entry(queue_path, lock_path, run_id):
    return _cancel_queue_entry(queue_path, lock_path, run_id)


def requeue_queue_entry(queue_path, lock_path, run_id, reason):
    return _requeue_queue_entry(queue_path, lock_path, run_id, reason)


def requeue_failed_entries(queue_path, lock_path):
    return _requeue_failed_entries(queue_path, lock_path)


def format_queue_status(queue_state):
    return _format_queue_status(queue_state)


def run_queue_worker(script_path, queue_path, lock_path, runner_lock_path):
    return _run_queue_worker(script_path, queue_path, lock_path, runner_lock_path)


def print_status(status_target, default_metadata_name=DEFAULT_RUN_METADATA_PATH):
    return _print_status(status_target, default_metadata_name)


def print_recent_statuses(count, base_dir="runs"):
    return _print_recent_statuses(count, base_dir=base_dir)


def record_status_event(
    event_log_path, run_id, run_dir, status, previous_status=None, details=None
):
    return _record_status_event(
        event_log_path,
        run_id,
        run_dir,
        status,
        previous_status=previous_status,
        details=details,
    )


def load_run_metadata(metadata_path):
    return _load_run_metadata(metadata_path)


def ensure_queue_runner_started(command, log_path):
    return _ensure_queue_runner_started(command, log_path)


def load_queue(queue_path=DEFAULT_QUEUE_PATH):
    return _load_queue(queue_path)


__all__ = [
    "cancel_queue_entry",
    "ensure_queue_file",
    "ensure_queue_runner_started",
    "enqueue_run",
    "format_queue_status",
    "load_run_metadata",
    "print_recent_statuses",
    "print_status",
    "queue_lock",
    "record_status_event",
    "register_foreground_run",
    "requeue_failed_entries",
    "requeue_queue_entry",
    "run_queue_worker",
    "update_queue_status",
    "load_queue",
]
