"""File-based locking to prevent concurrent runs on the same directory."""

from __future__ import annotations

import contextlib
import fcntl
import os
from typing import Generator

_LOCK_FILE = "run.lock"


@contextlib.contextmanager
def acquire_run_lock(reaction_dir: str) -> Generator[None, None, None]:
    """Acquire an exclusive lock on the reaction directory.

    Uses ``fcntl.flock`` for POSIX file locking. The lock is released
    when the context manager exits.

    Raises:
        RuntimeError: If the lock cannot be acquired (another run is active).
    """
    lock_path = os.path.join(reaction_dir, _LOCK_FILE)
    lock_fd = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"Another run is active in {reaction_dir}. "
                f"Remove {lock_path} if this is incorrect."
            ) from exc
        # Write PID for debugging
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        os.fsync(lock_fd)
        yield
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)
            # Clean up lock file
            try:
                os.unlink(lock_path)
            except OSError:
                pass
