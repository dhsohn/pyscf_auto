import json
from typing import Any, Mapping

from .run_opt_engine import load_xyz, normalized_symbol
from .run_opt_resources import ensure_parent_dir


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
    model = {
        "method": calculation_metadata.get("xc"),
        "basis": calculation_metadata.get("basis"),
    }
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


def _resolve_return_result(
    calculation_metadata: Mapping[str, Any],
    sp_result: Mapping[str, Any] | None,
    frequency_payload: Mapping[str, Any] | None,
) -> float | None:
    if sp_result and sp_result.get("energy") is not None:
        return sp_result.get("energy")
    if frequency_payload:
        frequency_result = frequency_payload.get("results") or {}
        if frequency_result.get("energy") is not None:
            return frequency_result.get("energy")
    summary = calculation_metadata.get("summary") if isinstance(calculation_metadata, dict) else None
    if isinstance(summary, dict) and summary.get("final_energy") is not None:
        return summary.get("final_energy")
    return None


def build_atomic_result(
    calculation_metadata: Mapping[str, Any],
    input_xyz: str,
    *,
    frequency_payload: Mapping[str, Any] | None = None,
    irc_payload: Mapping[str, Any] | None = None,
    sp_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    atomic_input = build_atomic_input(calculation_metadata, input_xyz)
    return_result = _resolve_return_result(calculation_metadata, sp_result, frequency_payload)
    properties = {}
    if return_result is not None:
        properties["scf_total_energy"] = return_result
    extras = {
        "pdft": {
            "calculation_metadata": dict(calculation_metadata),
            "frequency_payload": frequency_payload,
            "irc_payload": irc_payload,
            "sp_result": sp_result,
            "input_xyz": input_xyz,
        }
    }
    provenance = {
        "creator": "pDFT",
        "version": (
            calculation_metadata.get("versions", {}).get("pyscf")
            if isinstance(calculation_metadata, dict)
            else None
        ),
    }
    return {
        **atomic_input,
        "schema_name": "qcschema_output",
        "schema_version": "1.0",
        "success": calculation_metadata.get("status") != "failed",
        "return_result": return_result,
        "properties": properties,
        "provenance": {key: value for key, value in provenance.items() if value is not None},
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
