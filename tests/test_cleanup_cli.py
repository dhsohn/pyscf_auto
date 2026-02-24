from __future__ import annotations

import json
from pathlib import Path

from cli_new import main


def _write_config(path: Path, allowed_root: Path, organized_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "runtime:",
                f"  allowed_root: {allowed_root}",
                f"  organized_root: {organized_root}",
            ]
        ),
        encoding="utf-8",
    )


def _write_organized_run(reaction_dir: Path) -> None:
    reaction_dir.mkdir(parents=True, exist_ok=True)
    (reaction_dir / "rxn.inp").write_text("! SP B3LYP def2-SVP\n* xyz 0 1\nH 0 0 0\n*\n", encoding="utf-8")
    (reaction_dir / "rxn.out").write_text("normal termination\n", encoding="utf-8")
    (reaction_dir / "rxn.xyz").write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    (reaction_dir / "rxn.gbw").write_bytes(b"\x00" * 16)
    (reaction_dir / "rxn.hess").write_text("hessian\n", encoding="utf-8")
    (reaction_dir / "rxn.densities").write_text("junk\n", encoding="utf-8")
    (reaction_dir / "rxn.tmp").write_text("tmp\n", encoding="utf-8")
    (reaction_dir / "rxn.engrad").write_text("engrad\n", encoding="utf-8")
    (reaction_dir / "rxn.retry01.inp").write_text("! SP\n", encoding="utf-8")
    (reaction_dir / "rxn.retry01.out").write_text("failed\n", encoding="utf-8")
    (reaction_dir / "rxn_trj.xyz").write_text("traj\n", encoding="utf-8")
    (reaction_dir / "run_report.json").write_text("{}", encoding="utf-8")
    (reaction_dir / "run_report.md").write_text("# report\n", encoding="utf-8")

    state = {
        "run_id": "run_20260224_010101_deadbeef",
        "reaction_dir": str(reaction_dir),
        "selected_inp": str(reaction_dir / "rxn.inp"),
        "status": "completed",
        "attempts": [
            {
                "index": 1,
                "inp_path": str(reaction_dir / "rxn.inp"),
                "out_path": str(reaction_dir / "rxn.out"),
            }
        ],
        "final_result": {
            "status": "completed",
            "analyzer_status": "completed",
            "reason": "normal_termination",
            "last_out_path": str(reaction_dir / "rxn.out"),
        },
    }
    (reaction_dir / "run_state.json").write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def test_cleanup_dry_run_does_not_delete_files(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    allowed_root.mkdir()
    organized_root.mkdir()
    reaction_dir = organized_root / "single_point" / "H" / "run_001"
    _write_organized_run(reaction_dir)

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, allowed_root, organized_root)

    rc = main(
        [
            "--config",
            str(config_path),
            "cleanup",
            "--reaction-dir",
            str(reaction_dir),
            "--json",
        ]
    )
    assert rc == 0
    assert (reaction_dir / "rxn.densities").exists()
    assert (reaction_dir / "rxn.tmp").exists()
    assert (reaction_dir / "rxn.retry01.inp").exists()


def test_cleanup_apply_removes_junk_and_keeps_essentials(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    allowed_root.mkdir()
    organized_root.mkdir()
    reaction_dir = organized_root / "single_point" / "H" / "run_001"
    _write_organized_run(reaction_dir)

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, allowed_root, organized_root)

    rc = main(
        [
            "--config",
            str(config_path),
            "cleanup",
            "--reaction-dir",
            str(reaction_dir),
            "--apply",
        ]
    )
    assert rc == 0
    assert not (reaction_dir / "rxn.densities").exists()
    assert not (reaction_dir / "rxn.tmp").exists()
    assert not (reaction_dir / "rxn.engrad").exists()
    assert not (reaction_dir / "rxn.retry01.inp").exists()
    assert not (reaction_dir / "rxn.retry01.out").exists()
    assert not (reaction_dir / "rxn_trj.xyz").exists()
    assert (reaction_dir / "rxn.inp").exists()
    assert (reaction_dir / "rxn.out").exists()
    assert (reaction_dir / "run_state.json").exists()
    assert (reaction_dir / "run_report.json").exists()
    assert (reaction_dir / "run_report.md").exists()


def test_cleanup_apply_preserves_state_referenced_retry_files(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    allowed_root.mkdir()
    organized_root.mkdir()
    reaction_dir = organized_root / "single_point" / "H" / "run_001"
    _write_organized_run(reaction_dir)

    state_path = reaction_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["selected_inp"] = "/legacy/path/rxn.retry01.inp"
    state["attempts"] = [
        {
            "index": 1,
            "inp_path": "/legacy/path/rxn.retry01.inp",
            "out_path": "/legacy/path/rxn.retry01.out",
        }
    ]
    state["final_result"]["last_out_path"] = "/legacy/path/rxn.retry01.out"
    state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, allowed_root, organized_root)

    rc = main(
        [
            "--config",
            str(config_path),
            "cleanup",
            "--reaction-dir",
            str(reaction_dir),
            "--apply",
        ]
    )
    assert rc == 0
    assert (reaction_dir / "rxn.retry01.inp").exists()
    assert (reaction_dir / "rxn.retry01.out").exists()
    assert not (reaction_dir / "rxn.densities").exists()


def test_cleanup_uses_default_organized_root_when_root_not_given(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    allowed_root.mkdir()
    organized_root.mkdir()
    _write_organized_run(organized_root / "single_point" / "H" / "run_001")

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, allowed_root, organized_root)

    rc = main(["--config", str(config_path), "cleanup", "--json"])
    assert rc == 0


def test_cleanup_rejects_wrong_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    allowed_root.mkdir()
    organized_root.mkdir()

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, allowed_root, organized_root)

    rc = main(
        [
            "--config",
            str(config_path),
            "cleanup",
            "--root",
            str(allowed_root),
        ]
    )
    assert rc == 1


def test_cleanup_root_scan_ignores_symlinked_external_directory(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    organized_root = tmp_path / "outputs"
    allowed_root.mkdir()
    organized_root.mkdir()

    internal = organized_root / "single_point" / "H" / "run_001"
    _write_organized_run(internal)

    outside = tmp_path / "outside_run"
    _write_organized_run(outside)
    outside_junk = outside / "rxn.tmp"
    assert outside_junk.exists()

    symlink_path = organized_root / "linked_outside"
    try:
        symlink_path.symlink_to(outside, target_is_directory=True)
    except OSError:
        # Environments that disallow symlink creation should skip this hardening test.
        return

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, allowed_root, organized_root)

    rc = main(
        [
            "--config",
            str(config_path),
            "cleanup",
            "--root",
            str(organized_root),
            "--apply",
        ]
    )
    assert rc == 0
    assert outside_junk.exists()
