import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path, args):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "run_opt", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_only_passes(tmp_path):
    config_text = (REPO_ROOT / "run_config.json").read_text(encoding="utf-8")
    (tmp_path / "run_config.json").write_text(config_text, encoding="utf-8")

    result = run_cli(tmp_path, ["--validate-only", "--config", "run_config.json"])

    assert result.returncode == 0
    assert "Config validation passed" in result.stdout


def test_queue_status_empty_when_missing_queue_file(tmp_path):
    result = run_cli(tmp_path, ["--queue-status"])

    assert result.returncode == 0
    assert "Queue is empty." in result.stdout


def test_queue_status_shows_entries(tmp_path):
    queue_path = tmp_path / "runs" / "queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "run_id": "run-123",
                        "status": "queued",
                        "queued_at": "2024-01-01T00:00:00",
                    }
                ],
                "updated_at": "2024-01-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(tmp_path, ["--queue-status"])

    assert result.returncode == 0
    assert "run-123" in result.stdout


def test_queue_cancel_updates_queue_file(tmp_path):
    run_id = "run-456"
    queue_path = tmp_path / "runs" / "queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "run_id": run_id,
                        "status": "queued",
                        "queued_at": "2024-01-01T00:00:00",
                    }
                ],
                "updated_at": "2024-01-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(tmp_path, ["--queue-cancel", run_id])

    assert result.returncode == 0
    assert f"Canceled queued run: {run_id}" in result.stdout

    updated_state = json.loads(queue_path.read_text(encoding="utf-8"))
    assert updated_state["entries"][0]["status"] == "canceled"
