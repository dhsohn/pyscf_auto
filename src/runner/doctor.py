"""Runtime environment diagnostics for pyscf_auto."""

from __future__ import annotations

import importlib.util
import json
import sys

from run_opt_config import (
    DEFAULT_SOLVENT_MAP_PATH,
    load_solvent_map_from_path,
    load_solvent_map_from_resource,
)
from run_opt_resources import inspect_thread_settings


def run_doctor() -> None:
    """Validate optional/runtime dependencies and environment settings."""

    def format_doctor_result(label: str, status: bool, remedy: str | None = None) -> str:
        status_label = "OK" if status else "FAIL"
        separator = "  " if status_label == "OK" else " "
        if status:
            return f"{status_label}{separator}{label}"
        if remedy:
            return f"{status_label}{separator}{label} ({remedy})"
        return f"{status_label}{separator}{label}"

    failures: list[str] = []

    def _record_check(label: str, ok: bool, remedy: str | None = None) -> None:
        if not ok:
            failures.append(label)
        print(format_doctor_result(label, ok, remedy))

    def _check_import(module_name: str, hint: str, label: str | None = None) -> bool:
        spec = importlib.util.find_spec(module_name)
        ok = spec is not None
        display_label = label or module_name
        _record_check(display_label, ok, hint if not ok else None)
        return ok

    def _solvent_map_path_hint(error: Exception) -> str:
        if isinstance(error, FileNotFoundError):
            return (
                "Missing solvent map file. Provide --solvent-map or restore "
                f"{DEFAULT_SOLVENT_MAP_PATH}."
            )
        if isinstance(error, json.JSONDecodeError):
            return "Invalid JSON in solvent map. Fix the JSON syntax."
        return "Unable to read solvent map. Check file permissions and path."

    def _solvent_map_resource_hint(error: Exception) -> str:
        if isinstance(error, FileNotFoundError):
            return (
                "Missing solvent map package resource. Reinstall with packaged data "
                "or update the installation."
            )
        if isinstance(error, json.JSONDecodeError):
            return "Invalid JSON in package solvent map. Reinstall package data."
        return "Unable to read package solvent map. Check package installation."

    try:
        load_solvent_map_from_path(DEFAULT_SOLVENT_MAP_PATH)
        _record_check("solvent_map (file path)", True)
    except Exception as exc:
        _record_check("solvent_map (file path)", False, _solvent_map_path_hint(exc))

    try:
        load_solvent_map_from_resource()
        _record_check("solvent_map (package resource)", True)
    except Exception as exc:
        _record_check(
            "solvent_map (package resource)",
            False,
            _solvent_map_resource_hint(exc),
        )

    checks = [
        ("ase", "Install with: conda install -c daehyupsohn -c conda-forge ase"),
        ("ase.io", "Install with: conda install -c daehyupsohn -c conda-forge ase"),
        ("pyscf", "Install with: conda install -c daehyupsohn -c conda-forge pyscf"),
        ("pyscf.dft", "Install with: conda install -c daehyupsohn -c conda-forge pyscf"),
        ("pyscf.gto", "Install with: conda install -c daehyupsohn -c conda-forge pyscf"),
        (
            "pyscf.hessian.thermo",
            "Install with: conda install -c daehyupsohn -c conda-forge pyscf",
        ),
        ("dftd3", "Install with: conda install -c daehyupsohn -c conda-forge dftd3-python"),
        ("dftd4", "Install with: conda install -c daehyupsohn -c conda-forge dftd4-python"),
        (
            "sella",
            "Install with: conda install -c daehyupsohn -c conda-forge sella",
            "sella (TS optimizer)",
        ),
    ]
    for module_name, hint, *label in checks:
        _check_import(module_name, hint, label[0] if label else None)

    thread_status = inspect_thread_settings()
    print("INFO thread environment settings:")
    for env_name, env_value in thread_status["env"].items():
        print(f"  {env_name}={env_value}")
    print(f"INFO requested thread count = {thread_status['requested']}")
    print(f"INFO pyscf.lib.num_threads() = {thread_status['effective_threads']}")
    print(f"INFO openmp_available = {thread_status['openmp_available']}")

    if failures:
        print(f"FAIL {len(failures)} checks failed: {', '.join(failures)}")
        sys.exit(1)
    print("OK  all checks passed")
