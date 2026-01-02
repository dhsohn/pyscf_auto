import json
from pathlib import Path

from run_queue import load_queue


def test_load_queue_corrupt_with_backup(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    backup_path = Path(f"{queue_path}.bak")
    backup_state = {"entries": [{"run_id": "run-1"}], "updated_at": "2024-01-01"}

    queue_path.write_text("not valid json", encoding="utf-8")
    backup_path.write_text(json.dumps(backup_state), encoding="utf-8")

    loaded_state = load_queue(str(queue_path))

    assert loaded_state == backup_state
    corrupt_files = list(tmp_path.glob("queue.json.*.corrupt"))
    assert len(corrupt_files) == 1
    with queue_path.open("r", encoding="utf-8") as handle:
        restored_state = json.load(handle)
    assert restored_state == backup_state


def test_load_queue_corrupt_without_backup(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    queue_path.write_text("not valid json", encoding="utf-8")

    loaded_state = load_queue(str(queue_path))

    assert loaded_state == {"entries": [], "updated_at": None}
    corrupt_files = list(tmp_path.glob("queue.json.*.corrupt"))
    assert len(corrupt_files) == 1
