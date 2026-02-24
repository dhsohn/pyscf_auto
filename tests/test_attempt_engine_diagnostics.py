from __future__ import annotations

from runner.attempt_engine import _classify_error, _classify_from_metadata


def test_classify_error_detects_disk_and_timeout() -> None:
    assert _classify_error("No space left on device") == "error_disk_io"
    assert _classify_error("calculation timed out") == "timeout"


def test_classify_from_metadata_prefers_metadata_status() -> None:
    status, reason = _classify_from_metadata(
        {
            "status": "failed",
            "summary": {"reason": "SCF convergence failed"},
        }
    )
    assert status == "error_scf_convergence"
    assert reason == "SCF convergence failed"

