import json
import logging
import os
import re

from ..run_opt_metadata import get_package_version, write_checkpoint
from ..run_opt_resources import resolve_run_path


def _xc_includes_dispersion(xc):
    if not xc:
        return False
    normalized = re.sub(r"[\s_\-]+", "", str(xc)).lower()
    return normalized.endswith(("d", "d2", "d3", "d4"))


def _is_vacuum_solvent(name):
    return name is not None and name.strip().lower() == "vacuum"


def _normalize_dispersion_settings(stage_label, xc, dispersion_model, allow_dispersion=True):
    if dispersion_model is None:
        return None
    normalized = str(dispersion_model).lower()
    if not allow_dispersion:
        logging.warning(
            "%s 단계에서는 dispersion 입력을 지원하지 않습니다. '%s' 설정을 무시합니다.",
            stage_label,
            dispersion_model,
        )
        return None
    if _xc_includes_dispersion(xc):
        logging.warning(
            "%s XC '%s'에는 dispersion이 포함되어 있어 요청된 '%s'를 무시합니다.",
            stage_label,
            xc,
            normalized,
        )
        return None
    return normalized


def _frequency_units():
    return {
        "frequencies_wavenumber": "cm^-1",
        "frequencies_au": "a.u.",
        "energy": "Hartree",
        "zpe": "Hartree",
        "thermochemistry_temperature": "K",
        "thermochemistry_pressure": "config unit",
        "thermochemistry_zpe": "Hartree",
        "thermochemistry_thermal_correction_enthalpy": "Hartree",
        "thermochemistry_entropy": "Hartree/K",
        "thermochemistry_gibbs_correction": "Hartree",
        "thermochemistry_gibbs_free_energy": "Hartree",
        "min_frequency": "cm^-1",
        "max_frequency": "cm^-1",
        "dispersion_energy_hartree": "Hartree",
        "dispersion_energy_ev": "eV",
    }


def _resolve_scf_chkfile(scf_config, run_dir):
    if not scf_config:
        return None
    chkfile = scf_config.get("chkfile")
    if not chkfile:
        return None
    resolved = resolve_run_path(run_dir, chkfile)
    scf_config["chkfile"] = resolved
    return resolved


def _warn_missing_chkfile(resume_label, chkfile_path):
    if chkfile_path and not os.path.exists(chkfile_path):
        logging.warning(
            "%s PySCF chkfile not found at %s; continuing without chkfile init.",
            resume_label,
            chkfile_path,
        )


def _frequency_versions():
    return {
        "ase": get_package_version("ase"),
        "pyscf": get_package_version("pyscf"),
        "pyscf_hessian_thermo": get_package_version("pyscf"),
        "dftd3": get_package_version("dftd3"),
        "dftd4": get_package_version("dftd4"),
    }


def _thermochemistry_payload(thermo_config, thermochemistry):
    if thermochemistry is not None:
        return thermochemistry
    if not thermo_config:
        return None
    if hasattr(thermo_config, "to_dict"):
        thermo_data = thermo_config.to_dict()
    elif isinstance(thermo_config, dict):
        thermo_data = thermo_config
    else:
        thermo_data = {}
    return {
        "temperature": thermo_data.get("T"),
        "pressure": thermo_data.get("P"),
        "pressure_unit": thermo_data.get("unit"),
        "zpe": None,
        "thermal_correction_enthalpy": None,
        "entropy": None,
        "gibbs_correction": None,
        "gibbs_free_energy": None,
    }


def _normalize_frequency_dispersion_mode(mode_value):
    if mode_value is None:
        return "none"
    normalized = re.sub(r"[\s_\-]+", "", str(mode_value)).lower()
    if normalized in ("none", "no", "off", "false"):
        return "none"
    raise ValueError(
        "Unsupported frequency dispersion mode '{value}'. Use 'none'.".format(
            value=mode_value
        )
    )


def _normalize_solvent_settings(stage_label, solvent_name, solvent_model):
    if not solvent_name:
        return None, None
    if _is_vacuum_solvent(solvent_name):
        if solvent_model:
            logging.warning(
                "%s 단계에서 solvent '%s'는 vacuum으로 처리됩니다. solvent_model '%s'를 무시합니다.",
                stage_label,
                solvent_name,
                solvent_model,
            )
        return solvent_name, None
    return solvent_name, solvent_model


def _evaluate_irc_profile(
    profile,
    ts_energy_ev=None,
    min_drop_ev=1.0e-3,
    endpoint_tol_ev=1.0e-3,
):
    warnings = []
    if not profile:
        return {
            "status": "fail",
            "details": {
                "reason": "empty_profile",
                "ts_energy_ev": ts_energy_ev,
                "criteria": {
                    "min_drop_ev": min_drop_ev,
                    "endpoint_tol_ev": endpoint_tol_ev,
                },
                "directions": {},
            },
            "warnings": ["missing_profile"],
        }
    energies = [
        entry.get("energy_ev")
        for entry in profile
        if entry.get("energy_ev") is not None
    ]
    if not energies:
        return {
            "status": "fail",
            "details": {
                "reason": "missing_energy_values",
                "ts_energy_ev": ts_energy_ev,
                "criteria": {
                    "min_drop_ev": min_drop_ev,
                    "endpoint_tol_ev": endpoint_tol_ev,
                },
                "directions": {},
            },
            "warnings": ["missing_energy_values"],
        }
    if ts_energy_ev is None:
        ts_energy_ev = max(energies)
        warnings.append("ts_energy_missing_used_profile_max")

    direction_details = {}
    for direction in ("forward", "reverse"):
        entries = [
            entry
            for entry in profile
            if entry.get("direction") == direction and entry.get("energy_ev") is not None
        ]
        if not entries:
            direction_details[direction] = {
                "status": "fail",
                "reason": "missing_direction",
            }
            warnings.append(f"missing_direction:{direction}")
            continue
        entries = sorted(entries, key=lambda entry: entry.get("step", 0))
        endpoint_energy = entries[-1]["energy_ev"]
        min_energy = min(entry["energy_ev"] for entry in entries)
        endpoint_drop = ts_energy_ev - endpoint_energy
        min_drop = ts_energy_ev - min_energy
        endpoint_below_ts = endpoint_drop > min_drop_ev
        min_below_ts = min_drop > min_drop_ev
        endpoint_near_min = endpoint_energy <= min_energy + endpoint_tol_ev
        status = (
            "pass"
            if endpoint_below_ts and min_below_ts and endpoint_near_min
            else "fail"
        )
        direction_details[direction] = {
            "status": status,
            "step_count": len(entries),
            "endpoint_step": entries[-1].get("step"),
            "endpoint_energy_ev": endpoint_energy,
            "min_energy_ev": min_energy,
            "endpoint_drop_from_ts_ev": endpoint_drop,
            "min_drop_from_ts_ev": min_drop,
            "endpoint_below_ts": endpoint_below_ts,
            "min_below_ts": min_below_ts,
            "endpoint_near_min": endpoint_near_min,
        }

    overall_status = (
        "pass"
        if direction_details
        and all(
            detail.get("status") == "pass" for detail in direction_details.values()
        )
        else "fail"
    )
    if warnings and overall_status == "pass":
        overall_status = "warn"

    return {
        "status": overall_status,
        "details": {
            "ts_energy_ev": ts_energy_ev,
            "criteria": {
                "min_drop_ev": min_drop_ev,
                "endpoint_tol_ev": endpoint_tol_ev,
            },
            "directions": direction_details,
        },
        "warnings": warnings,
    }


def _normalize_optimizer_mode(mode_value):
    if not mode_value:
        return "minimum"
    normalized = re.sub(r"[\s_\-]+", "", str(mode_value)).lower()
    if normalized in ("minimum", "min", "geometry", "geom", "opt", "optimization"):
        return "minimum"
    if normalized in (
        "transitionstate",
        "transition",
        "ts",
        "saddle",
        "saddlepoint",
        "tsopt",
    ):
        return "transition_state"
    raise ValueError(
        "Unsupported optimizer mode '{value}'. Use 'minimum' or 'transition_state'.".format(
            value=mode_value
        )
    )


def _normalize_calculation_mode(mode_value):
    if not mode_value:
        return "optimization"
    normalized = re.sub(r"[\s_\-]+", "", str(mode_value)).lower()
    if normalized in (
        "optimization",
        "opt",
        "geometry",
        "geom",
        "structure",
        "structureoptimization",
        "구조최적화",
        "구조최적",
    ):
        return "optimization"
    if normalized in (
        "singlepoint",
        "singlepointenergy",
        "singlepointenergycalculation",
        "singlepointenergycalc",
        "singlepointcalc",
        "single_point",
        "single",
        "sp",
        "단일점",
        "단일점에너지",
        "단일점에너지계산",
    ):
        return "single_point"
    if normalized in (
        "frequency",
        "frequencies",
        "freq",
        "vibration",
        "vibrational",
        "프리퀀시",
        "진동",
    ):
        return "frequency"
    if normalized in (
        "irc",
        "intrinsicreactioncoordinate",
        "reactionpath",
        "reactioncoordinate",
    ):
        return "irc"
    if normalized in ("scan", "scanning", "scanmode"):
        return "scan"
    raise ValueError(
        "Unsupported calculation mode '{value}'. Use 'optimization', "
        "'single_point', 'frequency', 'irc', or 'scan'.".format(value=mode_value)
    )


def _normalize_scan_mode(mode_value):
    if not mode_value:
        return "optimization"
    normalized = re.sub(r"[\s_\-]+", "", str(mode_value)).lower()
    if normalized in ("optimization", "opt", "geometry", "geom", "optimize"):
        return "optimization"
    if normalized in ("singlepoint", "single_point", "single", "sp"):
        return "single_point"
    raise ValueError(
        "Unsupported scan mode '{value}'. Use 'optimization' or 'single_point'.".format(
            value=mode_value
        )
    )


def _generate_scan_values(start, end, step):
    if step == 0:
        raise ValueError("Scan step must be non-zero.")
    values = []
    current = float(start)
    end_value = float(end)
    step_value = float(step)
    tolerance = abs(step_value) * 1.0e-6
    if step_value > 0 and current > end_value + tolerance:
        raise ValueError("Scan start must be <= end for positive step.")
    if step_value < 0 and current < end_value - tolerance:
        raise ValueError("Scan start must be >= end for negative step.")
    if step_value > 0:
        while current <= end_value + tolerance:
            values.append(current)
            current += step_value
    else:
        while current >= end_value - tolerance:
            values.append(current)
            current += step_value
    if not values:
        raise ValueError("Scan produced no values; check start/end/step.")
    return values


def _dimension_key(dimension):
    indices = dimension["indices"]
    return "{type}:{indices}".format(
        type=dimension["type"],
        indices=",".join(str(index) for index in indices),
    )


def _apply_scan_geometry(atoms, dimensions, values):
    for dimension, value in zip(dimensions, values, strict=True):
        dimension_type = dimension["type"]
        indices = dimension["indices"]
        if dimension_type == "bond":
            atoms.set_distance(indices[0], indices[1], value, fix=0.5)
        elif dimension_type == "angle":
            atoms.set_angle(indices[0], indices[1], indices[2], value)
        else:
            atoms.set_dihedral(indices[0], indices[1], indices[2], indices[3], value)
    return atoms


def _atoms_to_atom_spec(atoms):
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    lines = []
    for symbol, position in zip(symbols, positions, strict=True):
        lines.append(
            "{symbol} {x:.10f} {y:.10f} {z:.10f}".format(
                symbol=symbol, x=position[0], y=position[1], z=position[2]
            )
        )
    return "\n".join(lines)


def _build_scan_constraints(dimensions, values):
    constraints = {}
    for dimension, value in zip(dimensions, values, strict=True):
        dim_type = dimension["type"]
        if dim_type == "bond":
            constraints.setdefault("bonds", []).append(
                {"i": dimension["indices"][0], "j": dimension["indices"][1], "length": value}
            )
        elif dim_type == "angle":
            constraints.setdefault("angles", []).append(
                {
                    "i": dimension["indices"][0],
                    "j": dimension["indices"][1],
                    "k": dimension["indices"][2],
                    "angle": value,
                }
            )
        else:
            constraints.setdefault("dihedrals", []).append(
                {
                    "i": dimension["indices"][0],
                    "j": dimension["indices"][1],
                    "k": dimension["indices"][2],
                    "l": dimension["indices"][3],
                    "dihedral": value,
                }
            )
    return constraints


def _merge_constraints(base_constraints, scan_constraints):
    if not base_constraints:
        return scan_constraints or None
    merged = {}
    for key in ("bonds", "angles", "dihedrals"):
        combined = []
        base_items = base_constraints.get(key) if isinstance(base_constraints, dict) else None
        if base_items:
            combined.extend([dict(item) for item in base_items])
        scan_items = scan_constraints.get(key)
        if scan_items:
            combined.extend(scan_items)
        if combined:
            merged[key] = combined
    return merged or None


def _parse_scan_dimensions(scan_config):
    if not isinstance(scan_config, dict):
        raise ValueError("Scan configuration must be an object.")
    raw_dimensions = scan_config.get("dimensions")
    if raw_dimensions is None:
        raw_dimensions = [scan_config]
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise ValueError("Scan dimensions must be a non-empty list.")
    dimensions = []
    for idx, dimension in enumerate(raw_dimensions):
        if not isinstance(dimension, dict):
            raise ValueError(f"Scan dimension {idx} must be an object.")
        dim_type = dimension.get("type")
        if dim_type not in ("bond", "angle", "dihedral"):
            raise ValueError(
                "Scan dimension {idx} must set type to bond, angle, or dihedral.".format(
                    idx=idx
                )
            )
        required_keys = {"bond": ("i", "j"), "angle": ("i", "j", "k"), "dihedral": ("i", "j", "k", "l")}
        indices = []
        for key in required_keys[dim_type]:
            value = dimension.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    "Scan dimension {idx} field '{key}' must be an integer.".format(
                        idx=idx, key=key
                    )
                )
            indices.append(value)
        dimensions.append(
            {
                "type": dim_type,
                "indices": indices,
                "start": dimension.get("start"),
                "end": dimension.get("end"),
                "step": dimension.get("step"),
            }
        )
    grid = scan_config.get("grid")
    if grid is not None:
        if not isinstance(grid, list) or len(grid) != len(dimensions):
            raise ValueError("Scan grid must match the number of dimensions.")
        grid_values = []
        for idx, values in enumerate(grid):
            if not isinstance(values, list) or not values:
                raise ValueError(f"Scan grid entry {idx} must be a non-empty list.")
            grid_values.append([float(value) for value in values])
        return dimensions, grid_values
    values_list = []
    for dimension in dimensions:
        if dimension["start"] is None or dimension["end"] is None or dimension["step"] is None:
            raise ValueError("Scan dimensions require start/end/step when grid is not set.")
        values_list.append(
            _generate_scan_values(dimension["start"], dimension["end"], dimension["step"])
        )
    return dimensions, values_list


def _normalize_stage_flags(config, calculation_mode):
    frequency_enabled = config.frequency_enabled
    single_point_enabled = config.single_point_enabled
    if calculation_mode != "optimization":
        frequency_enabled = False
        single_point_enabled = False
    if frequency_enabled is None and calculation_mode == "optimization":
        frequency_enabled = True
    if single_point_enabled is None:
        single_point_enabled = True
    return frequency_enabled, single_point_enabled


def _read_json_file(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_run_identity(resume_dir, run_metadata_path, checkpoint_path, override_run_id=None):
    run_id = override_run_id
    run_id_history = []
    attempt = 1
    existing_metadata = _read_json_file(run_metadata_path) if resume_dir else None
    existing_checkpoint = _read_json_file(checkpoint_path) if resume_dir else None
    existing_run_id = None
    if existing_metadata and existing_metadata.get("run_id"):
        existing_run_id = existing_metadata.get("run_id")
    elif existing_checkpoint and existing_checkpoint.get("run_id"):
        existing_run_id = existing_checkpoint.get("run_id")
    for source in (existing_metadata, existing_checkpoint):
        if not isinstance(source, dict):
            continue
        history = source.get("run_id_history")
        if isinstance(history, list):
            for item in history:
                if item and item not in run_id_history:
                    run_id_history.append(item)
    if resume_dir:
        prior_attempt = None
        for source in (existing_metadata, existing_checkpoint):
            if not isinstance(source, dict):
                continue
            candidate = source.get("attempt")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                prior_attempt = candidate
                break
        if run_id is None:
            run_id = existing_run_id
        if existing_run_id and run_id and existing_run_id != run_id:
            if existing_run_id not in run_id_history:
                run_id_history.append(existing_run_id)
        attempt = (prior_attempt or 1) + 1
    return run_id, run_id_history, attempt, existing_checkpoint


def _update_checkpoint_scf(checkpoint_path, pyscf_chkfile=None, scf_energy=None, scf_converged=None):
    if not checkpoint_path:
        return
    checkpoint_payload = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
                checkpoint_payload = json.load(checkpoint_file)
        except (OSError, json.JSONDecodeError):
            checkpoint_payload = {}
    if pyscf_chkfile is not None:
        checkpoint_payload["pyscf_chkfile"] = pyscf_chkfile
    if scf_energy is not None:
        checkpoint_payload["last_scf_energy"] = scf_energy
    if scf_converged is not None:
        checkpoint_payload["last_scf_converged"] = scf_converged
    write_checkpoint(checkpoint_path, checkpoint_payload)
