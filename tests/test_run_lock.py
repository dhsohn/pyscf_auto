from __future__ import annotations

import json
from pathlib import Path

import pytest

import runner.run_lock as run_lock


def test_acquire_run_lock_removes_stale_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reaction_dir = tmp_path / "rxn"
    reaction_dir.mkdir()
    lock_path = reaction_dir / "run.lock"
    lock_path.write_text(
        json.dumps({"pid": 999999, "started_at": "2026-02-24T00:00:00+00:00"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_lock, "_is_process_alive", lambda _pid: False)

    with run_lock.acquire_run_lock(str(reaction_dir)):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_acquire_run_lock_rejects_alive_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reaction_dir = tmp_path / "rxn"
    reaction_dir.mkdir()
    lock_path = reaction_dir / "run.lock"
    lock_path.write_text(
        json.dumps({"pid": 12345, "started_at": "2026-02-24T00:00:00+00:00"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_lock, "_is_process_alive", lambda _pid: True)
    monkeypatch.setattr(run_lock, "_process_start_ticks", lambda _pid: None)

    with pytest.raises(RuntimeError, match="Another run is active"):
        with run_lock.acquire_run_lock(str(reaction_dir)):
            pass

