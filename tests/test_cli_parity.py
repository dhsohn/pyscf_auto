from __future__ import annotations

import json
from pathlib import Path

import pytest

from app_config import AppConfig, RuntimeConfig
from cli_new import (
    _validate_root_scan_dir,
    build_parser,
    main,
)
from organizer.result_organizer import (
    find_organized_runs,
    organize_run,
    rebuild_organized_index,
)
from runner.orchestrator import cmd_run_inp, cmd_status


def _write_minimal_completed_run(reaction_dir: Path, run_id: str) -> None:
    reaction_dir.mkdir(parents=True, exist_ok=True)
    (reaction_dir / "rxn.inp").write_text(
        "! SP B3LYP def2-SVP\n* xyz 0 1\nH 0 0 0\n*\n",
        encoding="utf-8",
    )
    (reaction_dir / "_geometry.xyz").write_text(
        "1\ncomment\nH 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (reaction_dir / "rxn.out").write_text("normal termination\n", encoding="utf-8")
    attempt_dir = reaction_dir / "attempt_001"
    attempt_dir.mkdir(exist_ok=True)
    (reaction_dir / "run_report.json").write_text("{}", encoding="utf-8")
    (reaction_dir / "run_report.md").write_text("# report\n", encoding="utf-8")

    state = {
        "run_id": run_id,
        "reaction_dir": str(reaction_dir),
        "selected_inp": str(reaction_dir / "rxn.inp"),
        "status": "completed",
        "started_at": "2026-02-22T10:00:00+00:00",
        "updated_at": "2026-02-22T10:00:10+00:00",
        "max_retries": 3,
        "attempts": [
            {
                "index": 1,
                "analyzer_status": "completed",
                "analyzer_reason": "normal_termination",
                "attempt_dir": str(attempt_dir),
            },
        ],
        "final_result": {
            "status": "completed",
            "analyzer_status": "completed",
            "reason": "normal_termination",
            "completed_at": "2026-02-22T10:00:10+00:00",
            "last_out_path": str(reaction_dir / "rxn.out"),
            "resumed": False,
            "last_attempt_status": "completed",
        },
    }
    (reaction_dir / "run_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_minimal_failed_run(reaction_dir: Path, run_id: str) -> None:
    _write_minimal_completed_run(reaction_dir, run_id)
    state_path = reaction_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "failed"
    if isinstance(state.get("final_result"), dict):
        state["final_result"]["status"] = "failed"
        state["final_result"]["analyzer_status"] = "unknown_failure"
        state["final_result"]["reason"] = "failed_for_test"
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_parser_has_global_config_and_verbose() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "/tmp/custom.yaml",
            "-v",
            "status",
            "--reaction-dir",
            "/tmp/reaction",
        ],
    )
    assert args.config == "/tmp/custom.yaml"
    assert args.verbose is True
    assert args.command == "status"


def test_parser_rejects_legacy_doctor_and_validate_commands() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["doctor"])
    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "input.inp"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run-inp", "--reaction-dir", "/tmp/rxn", "--profile"])


def test_cmd_status_requires_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()

    run_id = "run_20260223_100000_deadbeef"
    _write_minimal_completed_run(outside_root / "rxn1", run_id)

    app_config = AppConfig(
        runtime=RuntimeConfig(
            allowed_root=str(allowed_root),
            organized_root=str(tmp_path / "organized"),
            default_max_retries=3,
        ),
    )

    exit_code = cmd_status(
        reaction_dir=str(outside_root / "rxn1"),
        app_config=app_config,
    )
    assert exit_code == 1


def test_validate_root_scan_dir_requires_exact_allowed_root(tmp_path: Path) -> None:
    allowed_root = (tmp_path / "runs").resolve()
    allowed_root.mkdir()
    subdir = allowed_root / "batch1"
    subdir.mkdir()

    assert _validate_root_scan_dir(str(subdir), allowed_root) is None
    assert _validate_root_scan_dir(str(allowed_root), allowed_root) == allowed_root


def test_organize_run_builds_index_and_detects_duplicate(tmp_path: Path) -> None:
    reaction_dir = tmp_path / "runs" / "rxn1"
    organized_root = tmp_path / "outputs"
    run_id = "run_20260223_110000_cafebabe"
    _write_minimal_completed_run(reaction_dir, run_id)

    first = organize_run(str(reaction_dir), str(organized_root), apply=True)
    assert first is not None
    assert first["applied"] is True
    assert first["action"] == "moved"
    assert not reaction_dir.exists()

    _write_minimal_completed_run(reaction_dir, run_id)
    second = organize_run(str(reaction_dir), str(organized_root), apply=True)
    assert second is not None
    assert second["applied"] is False
    assert second["action"] == "skipped"
    assert second["reason"] == "already_organized"


def test_organize_move_syncs_state_paths(tmp_path: Path) -> None:
    reaction_dir = tmp_path / "runs" / "rxn_sync"
    organized_root = tmp_path / "outputs"
    run_id = "run_20260223_111111_syncsync"
    _write_minimal_completed_run(reaction_dir, run_id)

    result = organize_run(str(reaction_dir), str(organized_root), apply=True)
    assert result is not None
    assert result["action"] == "moved"

    target = Path(result["target"])
    state = json.loads((target / "run_state.json").read_text(encoding="utf-8"))
    assert state["reaction_dir"] == str(target)
    assert state["selected_inp"] == str(target / "rxn.inp")
    assert state["attempts"][0]["attempt_dir"] == str(target / "attempt_001")
    assert state["final_result"]["last_out_path"] == str(target / "rxn.out")
    assert not reaction_dir.exists()


def test_organize_move_syncs_legacy_absolute_paths(tmp_path: Path) -> None:
    reaction_dir = tmp_path / "runs" / "rxn_legacy"
    organized_root = tmp_path / "outputs"
    run_id = "run_20260223_113333_legacypa"
    _write_minimal_completed_run(reaction_dir, run_id)

    state_path = reaction_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["reaction_dir"] = "/mnt/c/pyscf_runs/rxn_legacy"
    state["selected_inp"] = "/mnt/c/pyscf_runs/rxn_legacy/rxn.inp"
    state["attempts"][0]["attempt_dir"] = "/mnt/c/pyscf_runs/rxn_legacy/attempt_001"
    state["attempts"][0]["inp_path"] = "/mnt/c/pyscf_runs/rxn_legacy/rxn.inp"
    state["attempts"][0]["out_path"] = "/mnt/c/pyscf_runs/rxn_legacy/rxn.out"
    state["final_result"]["last_out_path"] = "/mnt/c/pyscf_runs/rxn_legacy/rxn.out"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    result = organize_run(str(reaction_dir), str(organized_root), apply=True)
    assert result is not None
    assert result["action"] == "moved"

    target = Path(result["target"])
    moved = json.loads((target / "run_state.json").read_text(encoding="utf-8"))
    assert moved["reaction_dir"] == str(target)
    assert moved["selected_inp"] == str(target / "rxn.inp")
    assert moved["attempts"][0]["attempt_dir"] == str(target / "attempt_001")
    assert moved["attempts"][0]["inp_path"] == str(target / "rxn.inp")
    assert moved["attempts"][0]["out_path"] == str(target / "rxn.out")
    assert moved["final_result"]["last_out_path"] == str(target / "rxn.out")


def test_organize_skips_failed_runs(tmp_path: Path) -> None:
    reaction_dir = tmp_path / "runs" / "rxn_failed"
    organized_root = tmp_path / "outputs"
    _write_minimal_failed_run(reaction_dir, "run_20260223_114444_failonly")

    result = organize_run(str(reaction_dir), str(organized_root), apply=False)
    assert result is None
    assert reaction_dir.exists()


def test_rebuild_index_and_find_records(tmp_path: Path) -> None:
    reaction_dir = tmp_path / "runs" / "rxn2"
    organized_root = tmp_path / "outputs"
    run_id = "run_20260223_120000_01234567"
    _write_minimal_completed_run(reaction_dir, run_id)
    organize_run(str(reaction_dir), str(organized_root), apply=True)

    count = rebuild_organized_index(str(organized_root))
    assert count == 1

    by_id = find_organized_runs(str(organized_root), run_id=run_id)
    assert len(by_id) == 1
    assert by_id[0]["run_id"] == run_id

    by_job_type = find_organized_runs(str(organized_root), job_type="single_point")
    assert len(by_job_type) == 1
    assert by_job_type[0]["job_type"] == "single_point"


def test_cli_organize_find_requires_selector(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    allowed_root.mkdir()
    organized_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                f"  allowed_root: {allowed_root}",
                f"  organized_root: {organized_root}",
            ],
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path), "organize", "--find"])
    assert exit_code == 1


def test_cli_organize_apply_returns_nonzero_on_failure(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    reaction_dir = allowed_root / "rxn_fail"
    allowed_root.mkdir()
    organized_root.mkdir()
    _write_minimal_completed_run(reaction_dir, "run_20260223_130000_failfail")

    preview = organize_run(str(reaction_dir), str(organized_root), apply=False)
    assert preview is not None
    Path(preview["target"]).mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                f"  allowed_root: {allowed_root}",
                f"  organized_root: {organized_root}",
            ],
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "organize",
            "--reaction-dir",
            str(reaction_dir),
            "--apply",
        ],
    )
    assert exit_code == 1


def test_cli_organize_dry_run_emits_orca_style_keys(tmp_path: Path, capsys) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    allowed_root.mkdir()
    organized_root.mkdir()
    _write_minimal_completed_run(allowed_root / "rxn_ok", "run_20260223_131111_okokokok")
    _write_minimal_failed_run(allowed_root / "rxn_fail", "run_20260223_131212_failfail")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                f"  allowed_root: {allowed_root}",
                f"  organized_root: {organized_root}",
            ],
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "organize",
            "--root",
            str(allowed_root),
        ],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "action: dry_run" in out
    assert "to_organize: 1" in out
    assert "skipped: 1" in out


def test_cmd_run_inp_terminal_state_emits_standard_payload(tmp_path: Path, capsys) -> None:
    allowed_root = tmp_path / "runs"
    reaction_dir = allowed_root / "rxn_done"
    allowed_root.mkdir()
    _write_minimal_completed_run(reaction_dir, "run_20260223_140000_doneeeee")

    app_config = AppConfig(
        runtime=RuntimeConfig(
            allowed_root=str(allowed_root),
            organized_root=str(tmp_path / "organized"),
            default_max_retries=3,
        ),
    )
    exit_code = cmd_run_inp(
        reaction_dir=str(reaction_dir),
        app_config=app_config,
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "status: completed" in out
    assert "reaction_dir:" in out
    assert "attempt_count:" in out
    assert "run_state:" in out


def test_cmd_run_inp_existing_out_short_circuits_execution(tmp_path: Path, monkeypatch) -> None:
    import runner.orchestrator as orchestrator

    allowed_root = tmp_path / "runs"
    reaction_dir = allowed_root / "rxn_existing_out"
    reaction_dir.mkdir(parents=True)
    (reaction_dir / "rxn.inp").write_text(
        "! SP B3LYP def2-SVP\n* xyz 0 1\nH 0 0 0\n*\n",
        encoding="utf-8",
    )
    (reaction_dir / "rxn.out").write_text("normal termination\n", encoding="utf-8")

    app_config = AppConfig(
        runtime=RuntimeConfig(
            allowed_root=str(allowed_root),
            organized_root=str(tmp_path / "organized"),
            default_max_retries=3,
        ),
    )

    def _should_not_run_attempts(*_args, **_kwargs):
        raise AssertionError("existing completed output must skip execution")

    monkeypatch.setattr(orchestrator, "run_attempts", _should_not_run_attempts)

    exit_code = cmd_run_inp(
        reaction_dir=str(reaction_dir),
        app_config=app_config,
    )
    assert exit_code == 0

    state = json.loads((reaction_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["final_result"]["reason"] == "existing_out_completed"
    assert state["final_result"]["skipped_execution"] is True
