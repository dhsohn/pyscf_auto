from __future__ import annotations

import json
from typing import Any

import execution.entrypoint as entrypoint


def test_execute_attempt_uses_stable_runner_contract(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_build_run_config(config_dict):
        captured["config_dict"] = dict(config_dict)
        return {"built": True}

    def _fake_run_execution(args, run_config, config_raw, config_source_path, run_in_background):
        captured["args"] = args
        captured["run_config"] = run_config
        captured["config_raw"] = json.loads(config_raw)
        captured["config_source_path"] = config_source_path
        captured["run_in_background"] = run_in_background
        (tmp_path / "metadata.json").write_text(
            json.dumps({"status": "completed", "summary": {"final_energy": -1.23}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(entrypoint, "build_run_config", _fake_build_run_config)
    monkeypatch.setattr(entrypoint, "run_execution", _fake_run_execution)

    result = entrypoint.execute_attempt(
        config_dict={"basis": "def2-svp", "xc": "b3lyp"},
        xyz_path="input.xyz",
        run_dir=str(tmp_path),
    )

    args: Any = captured["args"]
    assert args.xyz_file == "input.xyz"
    assert args.run_dir == str(tmp_path)
    assert args.no_background is True
    assert captured["run_in_background"] is False
    assert captured["config_source_path"] is None
    assert captured["config_raw"] == {"basis": "def2-svp", "xc": "b3lyp"}
    assert result.metadata_path == str(tmp_path / "metadata.json")
    assert result.metadata == {
        "status": "completed",
        "summary": {"final_energy": -1.23},
    }
