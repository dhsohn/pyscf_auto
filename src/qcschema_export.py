import json
import os
import socket
from typing import Any, Mapping

from run_opt_engine import load_xyz, normalized_symbol
from run_opt_metadata import collect_git_metadata, get_package_version
from run_opt_resources import collect_environment_snapshot
from run_opt_resources import ensure_parent_dir

HARTREE_TO_EV = 27.211386245988


def _driver_from_mode(calculation_mode: str | None) -> str:
    if calculation_mode == "frequency":
        return "hessian"
    if calculation_mode in ("optimization", "irc"):
        return "gradient"
    return "energy"


def _atom_spec_to_molecule(atom_spec: str, charge: int, multiplicity: int | None) -> dict[str, Any]:
    symbols: list[str] = []
    geometry: list[float] = []
    for line in atom_spec.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(normalized_symbol(parts[0]))
        geometry.extend([float(parts[1]), float(parts[2]), float(parts[3])])
    return {
        "symbols": symbols,
        "geometry": geometry,
        "charge": charge,
        "multiplicity": multiplicity,
        "units": "angstrom",
    }


def build_atomic_input(
    calculation_metadata: Mapping[str, Any],
    input_xyz: str,
) -> dict[str, Any]:
    atom_spec, charge, _spin, multiplicity = load_xyz(input_xyz)
    molecule = _atom_spec_to_molecule(atom_spec, charge, multiplicity)
    model = _model_from_metadata(calculation_metadata, sp_result=None, frequency_payload=None)
    keywords = {
        "scf": calculation_metadata.get("scf_config") or calculation_metadata.get("scf_settings"),
        "solvent": calculation_metadata.get("solvent"),
        "solvent_model": calculation_metadata.get("solvent_model"),
        "solvent_eps": calculation_metadata.get("solvent_eps"),
        "dispersion": calculation_metadata.get("dispersion"),
    }
    return {
        "schema_name": "qcschema_input",
        "schema_version": "1.0",
        "driver": _driver_from_mode(calculation_metadata.get("calculation_mode")),
        "model": model,
        "molecule": molecule,
        "keywords": {key: value for key, value in keywords.items() if value is not None},
    }


def _model_from_metadata(
    calculation_metadata: Mapping[str, Any],
    *,
    sp_result: Mapping[str, Any] | None,
    frequency_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model_source: Mapping[str, Any] = calculation_metadata
    if isinstance(calculation_metadata, dict):
        if sp_result is not None or (frequency_payload and frequency_payload.get("results")):
            sp_metadata = calculation_metadata.get("single_point")
            if isinstance(sp_metadata, dict):
                model_source = sp_metadata
    model = {
        "method": model_source.get("xc"),
        "basis": model_source.get("basis"),
        "solvent": model_source.get("solvent"),
        "solvent_model": model_source.get("solvent_model"),
        "solvent_eps": model_source.get("solvent_eps"),
    }
    return {key: value for key, value in model.items() if value is not None}


def _resolve_return_result(
    calculation_metadata: Mapping[str, Any],
    sp_result: Mapping[str, Any] | None,
    frequency_payload: Mapping[str, Any] | None,
) -> tuple[float | None, str | None]:
    if sp_result and sp_result.get("energy") is not None:
        return float(sp_result.get("energy")), "Hartree"
    if frequency_payload:
        frequency_result = frequency_payload.get("results") or {}
        if frequency_result.get("energy") is not None:
            return float(frequency_result.get("energy")), "Hartree"
    summary = calculation_metadata.get("summary") if isinstance(calculation_metadata, dict) else None
    if isinstance(summary, dict) and summary.get("final_energy") is not None:
        final_energy = float(summary.get("final_energy"))
        if calculation_metadata.get("calculation_mode") == "irc":
            return final_energy / HARTREE_TO_EV, "Hartree"
        return final_energy, "Hartree"
    return None, None


def _build_properties(
    return_result: float | None,
    gradient: Any | None,
) -> dict[str, Any]:
    properties = {}
    units = {}
    if return_result is not None:
        properties["return_energy"] = return_result
        properties["scf_total_energy"] = return_result
        units["return_energy"] = "Hartree"
        units["scf_total_energy"] = "Hartree"
    if gradient is not None:
        properties["gradient"] = gradient
        units["gradient"] = "Hartree/Bohr"
    if units:
        properties["units"] = units
    return properties


def _build_provenance(calculation_metadata: Mapping[str, Any]) -> dict[str, Any]:
    thread_count = (
        calculation_metadata.get("thread_count")
        if isinstance(calculation_metadata, dict)
        else None
    )
    environment_snapshot = (
        calculation_metadata.get("environment")
        if isinstance(calculation_metadata, dict)
        else None
    )
    if environment_snapshot is None:
        environment_snapshot = collect_environment_snapshot(thread_count or 1)
    git_metadata = (
        calculation_metadata.get("git") if isinstance(calculation_metadata, dict) else None
    )
    if git_metadata is None:
        git_metadata = collect_git_metadata(os.getcwd())
    version = get_package_version("dftflow")
    routine = ["DFTFlow"]
    calculation_mode = calculation_metadata.get("calculation_mode")
    if calculation_mode:
        routine.append(f"mode={calculation_mode}")
    python_version = environment_snapshot.get("python_version") if environment_snapshot else None
    if python_version:
        routine.append(f"python={python_version.split()[0]}")
    if git_metadata:
        commit = git_metadata.get("commit")
        if commit:
            routine.append(f"git={commit}")
    summary = calculation_metadata.get("summary") if isinstance(calculation_metadata, dict) else None
    walltime = summary.get("elapsed_seconds") if isinstance(summary, dict) else None
    provenance = {
        "creator": "DFTFlow",
        "version": version,
        "routine": routine,
        "walltime": walltime,
        "hostname": socket.gethostname(),
    }
    return {key: value for key, value in provenance.items() if value is not None}


def build_atomic_result(
    calculation_metadata: Mapping[str, Any],
    input_xyz: str,
    *,
    frequency_payload: Mapping[str, Any] | None = None,
    irc_payload: Mapping[str, Any] | None = None,
    sp_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    atomic_input = build_atomic_input(calculation_metadata, input_xyz)
    return_result, _ = _resolve_return_result(
        calculation_metadata, sp_result, frequency_payload
    )
    gradient = calculation_metadata.get("gradient") if isinstance(calculation_metadata, dict) else None
    properties = _build_properties(return_result, gradient)
    extras = {
        "dftflow": {
            "calculation_metadata": dict(calculation_metadata),
            "frequency_payload": frequency_payload,
            "irc_payload": irc_payload,
            "sp_result": sp_result,
            "input_xyz": input_xyz,
        }
    }
    model = _model_from_metadata(
        calculation_metadata,
        sp_result=sp_result,
        frequency_payload=frequency_payload,
    )
    provenance = _build_provenance(calculation_metadata)
    return {
        **atomic_input,
        "schema_name": "qcschema_output",
        "schema_version": "1.0",
        "success": calculation_metadata.get("status") != "failed",
        "return_result": return_result,
        "properties": properties,
        "provenance": provenance,
        "model": model,
        "extras": extras,
    }


def export_qcschema_result(
    output_path: str | None,
    calculation_metadata: Mapping[str, Any],
    input_xyz: str | None,
    *,
    frequency_payload: Mapping[str, Any] | None = None,
    irc_payload: Mapping[str, Any] | None = None,
    sp_result: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not output_path or not input_xyz:
        return None
    ensure_parent_dir(output_path)
    payload = build_atomic_result(
        calculation_metadata,
        input_xyz,
        frequency_payload=frequency_payload,
        irc_payload=irc_payload,
        sp_result=sp_result,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
