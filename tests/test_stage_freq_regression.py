import json
import time
from pathlib import Path

import execution.stage_freq as stage_freq
from execution.utils import _update_checkpoint_scf


def _build_stage_context(
    tmp_path: Path,
    *,
    ts_quality,
    irc_enabled: bool,
    single_point_enabled: bool,
) -> dict:
    input_xyz = tmp_path / "input.xyz"
    input_xyz.write_text("1\ncharge=0 spin=0\nH 0.0 0.0 0.0\n", encoding="utf-8")

    return {
        "run_start": time.perf_counter(),
        "metadata": {
            "single_point": {},
            "calculation_mode": "frequency",
        },
        "profiling_enabled": False,
        "irc_enabled": irc_enabled,
        "single_point_enabled": single_point_enabled,
        "mol": object(),
        "calc_basis": "def2-svp",
        "calc_xc": "b3lyp",
        "calc_scf_config": {"max_cycle": 50},
        "calc_solvent_name": "vacuum",
        "calc_solvent_model": None,
        "calc_eps": None,
        "calc_dispersion_model": None,
        "freq_dispersion_mode": "numerical",
        "freq_dispersion_step": 0.005,
        "optimizer_ase_config": {},
        "thermo": None,
        "verbose": False,
        "memory_mb": None,
        "constraints": None,
        "run_dir": str(tmp_path),
        "optimizer_mode": "transition_state",
        "multiplicity": 1,
        "ts_quality": ts_quality,
        "frequency_output_path": str(tmp_path / "frequency_result.json"),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "pyscf_chkfile": str(tmp_path / "scf.chk"),
        "irc_output_path": str(tmp_path / "irc_result.json"),
        "irc_config": None,
        "memory_limit_enforced": False,
        "qcschema_output_path": str(tmp_path / "qcschema_result.json"),
        "input_xyz": str(input_xyz),
        "run_metadata_path": str(tmp_path / "metadata.json"),
        "event_log_path": str(tmp_path / "events.jsonl"),
        "run_id": "run-test",
    }


def test_frequency_stage_ts_quality_enforced_skips_irc_and_single_point(
    tmp_path, monkeypatch
):
    stage_context = _build_stage_context(
        tmp_path,
        ts_quality={"enforce": True},
        irc_enabled=True,
        single_point_enabled=True,
    )

    monkeypatch.setattr(
        stage_freq.DEFAULT_ENGINE_ADAPTER,
        "compute_frequencies",
        lambda *_args, **_kwargs: {
            "energy": -10.0,
            "converged": True,
            "cycles": 8,
            "imaginary_count": 1,
            "imaginary_check": {"status": "one_imaginary", "message": "ok"},
            "ts_quality": {
                "status": "fail",
                "message": "TS quality checks did not pass.",
                "allow_irc": False,
                "allow_single_point": False,
            },
            "dispersion": None,
            "thermochemistry": None,
            "profiling": None,
        },
    )

    sp_calls = []
    irc_calls = []
    finalized = {}

    def _fail_if_called(*_args, **_kwargs):
        sp_calls.append(1)
        raise AssertionError("single-point should not run when ts_quality.enforce=true")

    def _irc_if_called(*_args, **_kwargs):
        irc_calls.append(1)
        raise AssertionError("IRC should not run when ts_quality.enforce=true")

    def _capture_finalize(
        _run_metadata_path,
        _event_log_path,
        _run_id,
        _run_dir,
        metadata,
        status,
        previous_status,
        queue_update_fn=None,
        exit_code=None,
        details=None,
        error=None,
    ):
        finalized["metadata"] = metadata
        finalized["status"] = status
        finalized["previous_status"] = previous_status
        finalized["exit_code"] = exit_code
        finalized["details"] = details
        finalized["error"] = error
        if queue_update_fn is not None:
            queue_update_fn(status, exit_code=exit_code)

    monkeypatch.setattr(
        stage_freq.DEFAULT_ENGINE_ADAPTER, "compute_single_point_energy", _fail_if_called
    )
    monkeypatch.setattr(stage_freq, "run_irc_stage", _irc_if_called)
    monkeypatch.setattr(stage_freq, "finalize_metadata", _capture_finalize)
    monkeypatch.setattr(stage_freq, "export_qcschema_result", lambda *_a, **_k: None)

    queue_updates = []
    stage_freq.run_frequency_stage(
        stage_context,
        lambda status, exit_code=None: queue_updates.append((status, exit_code)),
    )

    assert not sp_calls
    assert not irc_calls
    assert finalized["status"] == "completed"
    assert queue_updates == [("completed", 0)]
    assert (
        finalized["metadata"]["single_point"]["status"] == "skipped"
    )
    assert (
        finalized["metadata"]["irc"]["status"] == "skipped"
    )

    frequency_payload = json.loads(
        Path(stage_context["frequency_output_path"]).read_text(encoding="utf-8")
    )
    expected_keys = {
        "status",
        "output_file",
        "units",
        "versions",
        "basis",
        "xc",
        "scf",
        "solvent",
        "solvent_model",
        "solvent_eps",
        "dispersion",
        "dispersion_mode",
        "dispersion_step",
        "thermochemistry",
        "results",
        "single_point",
    }
    assert expected_keys.issubset(frequency_payload.keys())
    assert frequency_payload["single_point"]["status"] == "skipped"


def test_frequency_stage_ts_quality_not_enforced_runs_single_point(
    tmp_path, monkeypatch
):
    stage_context = _build_stage_context(
        tmp_path,
        ts_quality={"enforce": False},
        irc_enabled=False,
        single_point_enabled=True,
    )

    monkeypatch.setattr(
        stage_freq.DEFAULT_ENGINE_ADAPTER,
        "compute_frequencies",
        lambda *_args, **_kwargs: {
            "energy": -20.0,
            "converged": True,
            "cycles": 6,
            "imaginary_count": 0,
            "imaginary_check": {"status": "no_imaginary", "message": "ok"},
            "ts_quality": {
                "status": "fail",
                "message": "quality mismatch",
                "allow_irc": False,
                "allow_single_point": False,
            },
            "dispersion": None,
            "thermochemistry": None,
            "profiling": None,
        },
    )

    sp_calls = []
    checkpoint_calls = []
    finalized = {}

    def _single_point_stub(*_args, **_kwargs):
        sp_calls.append(1)
        return {
            "energy": -19.5,
            "converged": True,
            "cycles": 11,
            "dispersion": None,
            "profiling": None,
        }

    def _capture_checkpoint(*_args, **kwargs):
        checkpoint_calls.append(kwargs)

    def _capture_finalize(
        _run_metadata_path,
        _event_log_path,
        _run_id,
        _run_dir,
        metadata,
        status,
        previous_status,
        queue_update_fn=None,
        exit_code=None,
        details=None,
        error=None,
    ):
        finalized["metadata"] = metadata
        finalized["status"] = status
        finalized["details"] = details
        finalized["error"] = error
        if queue_update_fn is not None:
            queue_update_fn(status, exit_code=exit_code)

    monkeypatch.setattr(
        stage_freq.DEFAULT_ENGINE_ADAPTER, "compute_single_point_energy", _single_point_stub
    )
    monkeypatch.setattr(stage_freq, "_update_checkpoint_scf", _capture_checkpoint)
    monkeypatch.setattr(stage_freq, "finalize_metadata", _capture_finalize)
    monkeypatch.setattr(stage_freq, "export_qcschema_result", lambda *_a, **_k: None)

    queue_updates = []
    stage_freq.run_frequency_stage(
        stage_context,
        lambda status, exit_code=None: queue_updates.append((status, exit_code)),
    )

    assert len(sp_calls) == 1
    assert finalized["status"] == "completed"
    assert queue_updates == [("completed", 0)]
    assert finalized["metadata"]["single_point"]["status"] == "executed"
    assert finalized["metadata"]["summary"]["final_energy"] == -19.5
    assert len(checkpoint_calls) >= 2
    assert checkpoint_calls[0]["scf_energy"] == -20.0
    assert checkpoint_calls[-1]["scf_energy"] == -19.5


def test_update_checkpoint_scf_preserves_existing_payload(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "created_at": "2026-02-06T00:00:00",
                "nested": {"a": 1},
                "last_scf_energy": 0.0,
            }
        ),
        encoding="utf-8",
    )

    _update_checkpoint_scf(
        str(checkpoint_path),
        pyscf_chkfile="scf.chk",
        scf_energy=-1.25,
        scf_converged=True,
    )

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["created_at"] == "2026-02-06T00:00:00"
    assert payload["nested"] == {"a": 1}
    assert payload["pyscf_chkfile"] == "scf.chk"
    assert payload["last_scf_energy"] == -1.25
    assert payload["last_scf_converged"] is True

    _update_checkpoint_scf(
        str(checkpoint_path),
        pyscf_chkfile=None,
        scf_energy=None,
        scf_converged=None,
    )
    payload_after_noop = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload_after_noop == payload
