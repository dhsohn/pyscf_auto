from __future__ import annotations

import runner.attempt_engine as attempt_engine
from app_config import AppConfig, RuntimeConfig
from inp.parser import InpConfig
from runner.orchestrator import cmd_run_inp
from runner.state_machine import (
    load_or_create_state,
    new_state,
    save_state,
)


def _minimal_inp_config() -> InpConfig:
    return InpConfig(
        job_type="single_point",
        functional="b3lyp",
        basis="def2-svp",
        charge=0,
        multiplicity=1,
        atom_spec="H 0.0 0.0 0.0",
        xyz_source="inline",
    )


def test_run_attempts_resume_continues_from_next_attempt(tmp_path, monkeypatch):
    state = new_state(str(tmp_path), str(tmp_path / "job.inp"), max_retries=3)
    state["attempts"] = [
        {
            "index": 1,
            "analyzer_status": "error_scf_convergence",
            "analyzer_reason": "scf_failed",
        }
    ]

    called_attempt_numbers: list[int] = []

    def _fake_run_single_attempt(*, attempt_num: int, **_kwargs):
        called_attempt_numbers.append(attempt_num)
        return {
            "index": attempt_num,
            "analyzer_status": "completed",
            "analyzer_reason": "normal_termination",
            "energy": -1.23,
        }

    monkeypatch.setattr(attempt_engine, "_run_single_attempt", _fake_run_single_attempt)

    result = attempt_engine.run_attempts(
        state=state,
        inp_config=_minimal_inp_config(),
        reaction_dir=str(tmp_path),
        max_retries=3,
        notify_fn=lambda _evt: None,
        resumed=True,
    )

    assert called_attempt_numbers == [2]
    assert result["status"] == "completed"
    assert len(result["attempts"]) == 2
    assert result["final_result"]["reason"] == "normal_termination"


def test_run_attempts_resume_terminal_previous_completed(tmp_path, monkeypatch):
    state = new_state(str(tmp_path), str(tmp_path / "job.inp"), max_retries=3)
    state["status"] = "running"
    state["attempts"] = [
        {
            "index": 1,
            "analyzer_status": "completed",
            "analyzer_reason": "normal_termination",
            "energy": -5.0,
        }
    ]

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("A completed resumed run must not execute another attempt")

    monkeypatch.setattr(attempt_engine, "_run_single_attempt", _should_not_run)

    result = attempt_engine.run_attempts(
        state=state,
        inp_config=_minimal_inp_config(),
        reaction_dir=str(tmp_path),
        max_retries=3,
        notify_fn=lambda _evt: None,
        resumed=True,
    )

    assert result["status"] == "completed"
    assert len(result["attempts"]) == 1
    assert result["final_result"]["reason"] == "normal_termination"


def test_run_attempts_resume_retry_limit_reached(tmp_path, monkeypatch):
    state = new_state(str(tmp_path), str(tmp_path / "job.inp"), max_retries=1)
    state["status"] = "retrying"
    state["attempts"] = [
        {
            "index": 1,
            "analyzer_status": "error_scf_convergence",
            "analyzer_reason": "first_failure",
        },
        {
            "index": 2,
            "analyzer_status": "unknown_failure",
            "analyzer_reason": "second_failure",
        },
    ]

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("Retry limit reached during resume must not run another attempt")

    monkeypatch.setattr(attempt_engine, "_run_single_attempt", _should_not_run)

    result = attempt_engine.run_attempts(
        state=state,
        inp_config=_minimal_inp_config(),
        reaction_dir=str(tmp_path),
        max_retries=1,
        notify_fn=lambda _evt: None,
        resumed=True,
    )

    assert result["status"] == "failed"
    assert result["final_result"]["reason"] == "retry_limit_reached"
    assert result["final_result"]["analyzer_status"] == "unknown_failure"


def test_load_or_create_state_resumes_same_selected_input(tmp_path):
    reaction_dir = tmp_path / "reaction"
    reaction_dir.mkdir()

    selected_inp = reaction_dir / "job.inp"
    selected_inp.write_text("! SP B3LYP def2-SVP\n* xyz 0 1\nH 0 0 0\n*\n", encoding="utf-8")

    state = new_state(str(reaction_dir), str(selected_inp), max_retries=5)
    state["status"] = "running"
    save_state(str(reaction_dir), state)

    loaded, resumed = load_or_create_state(
        reaction_dir=str(reaction_dir),
        selected_inp=str(selected_inp),
        max_retries=2,
    )

    assert resumed is True
    assert loaded["run_id"] == state["run_id"]
    assert loaded["max_retries"] == 2


def test_load_or_create_state_restarts_when_selected_input_changes(tmp_path):
    reaction_dir = tmp_path / "reaction"
    reaction_dir.mkdir()

    selected_a = reaction_dir / "a.inp"
    selected_b = reaction_dir / "b.inp"
    selected_a.write_text("! SP B3LYP def2-SVP\n* xyz 0 1\nH 0 0 0\n*\n", encoding="utf-8")
    selected_b.write_text("! SP B3LYP def2-SVP\n* xyz 0 1\nH 0 0 0\n*\n", encoding="utf-8")

    state = new_state(str(reaction_dir), str(selected_a), max_retries=5)
    state["status"] = "running"
    save_state(str(reaction_dir), state)

    loaded, resumed = load_or_create_state(
        reaction_dir=str(reaction_dir),
        selected_inp=str(selected_b),
        max_retries=2,
    )

    assert resumed is False
    assert loaded["run_id"] != state["run_id"]
    assert loaded["selected_inp"] == str(selected_b)


def test_cmd_run_inp_requires_allowed_root(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    app_config = AppConfig(
        runtime=RuntimeConfig(
            allowed_root=str(allowed_root),
            organized_root=str(tmp_path / "organized"),
            default_max_retries=1,
        )
    )

    exit_code = cmd_run_inp(
        reaction_dir=str(outside_dir),
        app_config=app_config,
    )

    assert exit_code == 1
