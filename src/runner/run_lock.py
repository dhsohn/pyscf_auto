"""File-based locking to prevent concurrent runs on the same directory."""

from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

_LOCK_FILE = "run.lock"


@contextlib.contextmanager
def acquire_run_lock(reaction_dir: str) -> Generator[None, None, None]:
    """Acquire an exclusive lock on the reaction directory.

    Raises:
        RuntimeError: If the lock cannot be acquired (another run is active).
    """
    lock_path = Path(reaction_dir) / _LOCK_FILE
    payload = {"pid": os.getpid(), "started_at": _now_utc_iso()}
    start_ticks = _current_process_start_ticks()
    if start_ticks is not None:
        payload["process_start_ticks"] = start_ticks

    try:
        try:
            _acquire_lock_file(lock_path, payload)
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


def _acquire_lock_file(lock_path: Path, payload: dict[str, object]) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    while True:
        try:
            fd = os.open(str(lock_path), flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return
        except FileExistsError:
            try:
                if lock_path.is_symlink():
                    lock_path.unlink()
                    continue
            except OSError as exc:
                raise RuntimeError(
                    f"Lock path is a symlink and could not be removed: {lock_path}. error={exc}"
                ) from exc

            info = _parse_lock_info(lock_path)
            lock_pid = info.get("pid")
            started_at = info.get("started_at")
            lock_ticks = info.get("process_start_ticks")

            if isinstance(lock_pid, int):
                alive = _is_process_alive(lock_pid)
                if alive and isinstance(lock_ticks, int):
                    observed_ticks = _process_start_ticks(lock_pid)
                    if observed_ticks is not None and observed_ticks != lock_ticks:
                        alive = False
                if alive:
                    started = started_at if isinstance(started_at, str) and started_at else "unknown"
                    raise RuntimeError(
                        "Another run is active in this directory "
                        f"(pid={lock_pid}, started_at={started}). Lock file: {lock_path}"
                    ) from None
                try:
                    lock_path.unlink()
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        "Detected stale lock but failed to remove it "
                        f"(pid={lock_pid}). Lock file: {lock_path}. error={exc}"
                    ) from exc

            raise RuntimeError(
                f"Lock file exists but owner PID is unreadable. Remove manually: {lock_path}"
            ) from None


def _parse_lock_info(lock_path: Path) -> dict[str, int | str | None]:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {"pid": None, "started_at": None, "process_start_ticks": None}
    if not raw:
        return {"pid": None, "started_at": None, "process_start_ticks": None}

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        pid = parsed.get("pid")
        started_at = parsed.get("started_at")
        start_ticks = parsed.get("process_start_ticks")
        return {
            "pid": int(pid) if isinstance(pid, int) else _try_int(pid),
            "started_at": started_at if isinstance(started_at, str) else None,
            "process_start_ticks": (
                int(start_ticks) if isinstance(start_ticks, int) else _try_int(start_ticks)
            ),
        }

    return {"pid": _try_int(raw.splitlines()[0]), "started_at": None, "process_start_ticks": None}


def _try_int(value: object) -> int | None:
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
            return parsed if parsed > 0 else None
        except ValueError:
            return None
    return None


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _process_start_ticks(pid: int) -> int | None:
    if pid <= 0:
        return None
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    if not raw:
        return None
    right_paren = raw.rfind(")")
    if right_paren < 0:
        return None
    fields_after_comm = raw[right_paren + 2 :].split()
    if len(fields_after_comm) <= 19:
        return None
    try:
        value = int(fields_after_comm[19])
    except ValueError:
        return None
    return value if value > 0 else None


def _current_process_start_ticks() -> int | None:
    return _process_start_ticks(os.getpid())


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
