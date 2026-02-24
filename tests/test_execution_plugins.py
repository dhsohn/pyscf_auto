from __future__ import annotations

import pytest

from execution.plugins import FeatureUnavailableError, export_qcschema_result, load_stage_runner


def test_load_stage_runner_rejects_unknown_stage() -> None:
    with pytest.raises(FeatureUnavailableError):
        load_stage_runner("not-a-stage")


def test_load_stage_runner_respects_disable_flags(monkeypatch) -> None:
    monkeypatch.setenv("PYSCF_AUTO_DISABLE_SCAN", "1")
    with pytest.raises(FeatureUnavailableError):
        load_stage_runner("scan")


def test_export_qcschema_result_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("PYSCF_AUTO_DISABLE_QCSCHEMA", "1")
    assert export_qcschema_result(None, None, None) is None

