"""Index storage helpers for organized outputs."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

INDEX_DIR_NAME = "index"
RECORDS_FILE_NAME = "records.jsonl"
LOCK_FILE_NAME = "index.lock"


def now_utc_iso() -> str:
    """Return a UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def index_dir(organized_root: Path) -> Path:
    """Return the index directory path under organized_root."""
    return organized_root / INDEX_DIR_NAME


def records_path(organized_root: Path) -> Path:
    """Return the JSONL records path under organized_root."""
    return index_dir(organized_root) / RECORDS_FILE_NAME


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def load_index(organized_root: Path) -> dict[str, dict[str, Any]]:
    """Load index records keyed by run_id."""
    path = records_path(organized_root)
    if not path.exists():
        return {}

    data: dict[str, dict[str, Any]] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return data

    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("Malformed JSON at %s line %d", path, line_number)
            continue
        if not isinstance(record, dict):
            continue
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            data[run_id] = record
    return data


def append_record(organized_root: Path, record: dict[str, Any]) -> None:
    """Append one index record in JSONL format."""
    path = records_path(organized_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def find_by_run_id(organized_root: Path, run_id: str) -> dict[str, Any] | None:
    """Return one record by exact run_id."""
    return load_index(organized_root).get(run_id)


def find_by_job_type(
    organized_root: Path,
    job_type: str,
    *,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Return records filtered by exact job_type."""
    records = [
        record
        for record in load_index(organized_root).values()
        if record.get("job_type") == job_type
    ]
    if limit > 0:
        return records[:limit]
    return records


def _to_reaction_relative_path(path_value: Any, reaction_dir: Path) -> str:
    if not isinstance(path_value, str):
        return ""
    raw = path_value.strip()
    if not raw:
        return ""

    path = Path(raw)
    if path.is_absolute():
        try:
            return str(path.relative_to(reaction_dir))
        except ValueError:
            return path.name
    return str(path)


def rebuild_index(organized_root: Path) -> int:
    """Rebuild index records by scanning organized outputs."""
    path = records_path(organized_root)
    records_by_run_id: dict[str, dict[str, Any]] = {}

    if organized_root.exists():
        for state_file in sorted(organized_root.rglob("run_state.json")):
            if INDEX_DIR_NAME in state_file.parts:
                continue
            reaction_dir = state_file.parent
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(state, dict):
                continue

            run_id = state.get("run_id")
            if not isinstance(run_id, str) or not run_id.strip():
                continue

            final_result = state.get("final_result")
            if not isinstance(final_result, dict):
                final_result = {}

            attempts = state.get("attempts")
            if not isinstance(attempts, list):
                attempts = []
            last_attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}

            try:
                rel = reaction_dir.relative_to(organized_root)
                rel_parts = rel.parts
                organized_path = str(rel)
            except ValueError:
                rel_parts = ()
                organized_path = str(reaction_dir)

            job_type = rel_parts[0] if len(rel_parts) >= 1 else "other"
            molecule_key = rel_parts[1] if len(rel_parts) >= 2 else "unknown"

            records_by_run_id[run_id] = {
                "run_id": run_id,
                "reaction_dir": str(reaction_dir),
                "status": state.get("status", ""),
                "analyzer_status": final_result.get("analyzer_status", ""),
                "reason": final_result.get("reason", ""),
                "job_type": job_type,
                "molecule_key": molecule_key,
                "selected_inp": _to_reaction_relative_path(
                    state.get("selected_inp", ""),
                    reaction_dir,
                ),
                "last_attempt_status": (
                    last_attempt.get("analyzer_status", "")
                    if isinstance(last_attempt, dict)
                    else ""
                ),
                "attempt_count": len(attempts),
                "completed_at": final_result.get("completed_at", ""),
                "organized_at": now_utc_iso(),
                "organized_path": organized_path,
            }

    lines = [
        json.dumps(records_by_run_id[run_id], ensure_ascii=True)
        for run_id in sorted(records_by_run_id)
    ]
    content = ("\n".join(lines) + "\n") if lines else ""
    _atomic_write_text(path, content)
    return len(records_by_run_id)


@contextmanager
def acquire_index_lock(
    organized_root: Path,
    *,
    timeout_seconds: int = 30,
    poll_interval_seconds: float = 0.1,
) -> Iterator[None]:
    """Acquire a coarse-grained lock for index mutations."""
    lock_path = index_dir(organized_root) / LOCK_FILE_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    acquired = False

    payload = json.dumps(
        {
            "pid": os.getpid(),
            "started_at": now_utc_iso(),
        },
        ensure_ascii=True,
    )

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            acquired = True
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out acquiring index lock: {lock_path}",
                ) from None
            time.sleep(poll_interval_seconds)

    try:
        yield
    finally:
        if acquired:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove index lock: %s", lock_path)
