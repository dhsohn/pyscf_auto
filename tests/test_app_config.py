from __future__ import annotations

from pathlib import Path

import pytest

from app_config import load_app_config


def test_load_app_config_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    cfg = load_app_config(str(missing_path))
    assert cfg.runtime.default_max_retries == 5
    assert cfg.runtime.allowed_root.endswith("pyscf_runs")
    assert cfg.runtime.organized_root.endswith("pyscf_outputs")


def test_load_app_config_rejects_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("runtime: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML config"):
        load_app_config(str(config_path))


def test_load_app_config_rejects_relative_runtime_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  allowed_root: ./runs",
                f"  organized_root: {tmp_path / 'outputs'}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime.allowed_root must be an absolute path"):
        load_app_config(str(config_path))


def test_load_app_config_rejects_nested_runtime_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "runs"
    organized = allowed / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                f"  allowed_root: {allowed}",
                f"  organized_root: {organized}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not contain each other"):
        load_app_config(str(config_path))


def test_load_app_config_rejects_empty_cleanup_keep_extensions(tmp_path: Path) -> None:
    allowed = tmp_path / "runs"
    organized = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                f"  allowed_root: {allowed}",
                f"  organized_root: {organized}",
                "cleanup:",
                "  keep_extensions: []",
                "  keep_filenames: [run_state.json]",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cleanup.keep_extensions must not be empty"):
        load_app_config(str(config_path))


def test_load_app_config_parses_cleanup_overrides(tmp_path: Path) -> None:
    allowed = tmp_path / "runs"
    organized = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                f"  allowed_root: {allowed}",
                f"  organized_root: {organized}",
                "  default_max_retries: 7",
                "monitoring:",
                "  enabled: true",
                "cleanup:",
                "  keep_extensions: [inp, out]",
                "  keep_filenames: [run_state.json, run_report.json]",
                "  remove_patterns: [\"*.tmp\"]",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_app_config(str(config_path))
    assert cfg.runtime.default_max_retries == 7
    assert cfg.monitoring.enabled is True
    assert cfg.cleanup.keep_extensions == [".inp", ".out"]
    assert cfg.cleanup.keep_filenames == ["run_state.json", "run_report.json"]
    assert cfg.cleanup.remove_patterns == ["*.tmp"]

