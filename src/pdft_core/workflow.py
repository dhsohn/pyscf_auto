import importlib.util
import itertools
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime

from .ase_backend import _run_ase_irc, _run_ase_optimizer
from .queue import (
    enqueue_run,
    ensure_queue_runner_started,
    record_status_event,
    register_foreground_run,
    update_queue_status,
)
from .run_opt_engine import (
    apply_scf_settings,
    apply_solvent_model,
    compute_frequencies,
    compute_imaginary_mode,
    compute_single_point_energy,
    load_xyz,
    normalize_xc_functional,
    run_capability_check,
    select_ks_type,
    total_electron_count,
)
from .run_opt_config import (
    DEFAULT_EVENT_LOG_PATH,
    DEFAULT_FREQUENCY_PATH,
    DEFAULT_IRC_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_OPTIMIZED_XYZ_PATH,
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_QUEUE_RUNNER_LOG_PATH,
    DEFAULT_RUN_METADATA_PATH,
    DEFAULT_SCAN_RESULT_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
    DEFAULT_THREAD_COUNT,
    RunConfig,
    load_solvent_map,
    load_solvent_map_from_path,
    load_solvent_map_from_resource,
)
from .run_opt_logging import ensure_stream_newlines, setup_logging
from .run_opt_metadata import (
    build_run_summary,
    collect_git_metadata,
    compute_file_hash,
    compute_text_hash,
    get_package_version,
    parse_single_point_cycle_count,
    write_checkpoint,
    write_optimized_xyz,
    write_run_metadata,
)
from .run_opt_resources import (
    apply_memory_limit,
    apply_thread_settings,
    collect_environment_snapshot,
    create_run_directory,
    ensure_parent_dir,
    format_log_path,
    inspect_thread_settings,
    resolve_run_path,
)


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


def run_doctor():
    def format_doctor_result(label, status, remedy=None):
        status_label = "OK" if status else "FAIL"
        separator = "  " if status_label == "OK" else " "
        if status:
            return f"{status_label}{separator}{label}"
        if remedy:
            return f"{status_label}{separator}{label} ({remedy})"
        return f"{status_label}{separator}{label}"

    failures = []

    def _record_check(label, ok, remedy=None):
        if not ok:
            failures.append(label)
        print(format_doctor_result(label, ok, remedy))

    def _check_import(module_name, hint, label=None):
        spec = importlib.util.find_spec(module_name)
        ok = spec is not None
        display_label = label or module_name
        _record_check(display_label, ok, hint if not ok else None)
        return ok

    def _solvent_map_path_hint(error):
        if isinstance(error, FileNotFoundError):
            return (
                "Missing solvent map file. Provide --solvent-map or restore "
                f"{DEFAULT_SOLVENT_MAP_PATH}."
            )
        if isinstance(error, json.JSONDecodeError):
            return "Invalid JSON in solvent map. Fix the JSON syntax."
        return "Unable to read solvent map. Check file permissions and path."

    def _solvent_map_resource_hint(error):
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
        _record_check("solvent_map (package resource)", False, _solvent_map_resource_hint(exc))

    checks = [
        ("ase", "Install with: pip install ase"),
        ("ase.io", "Install with: pip install ase"),
        ("pyscf", "Install with: pip install pyscf"),
        ("pyscf.dft", "Install with: pip install pyscf"),
        ("pyscf.gto", "Install with: pip install pyscf"),
        ("pyscf.hessian.thermo", "Install with: pip install pyscf"),
        ("dftd3", "Install with: pip install dftd3"),
        ("dftd4", "Install with: pip install dftd4"),
        ("sella", "Install with: pip install sella", "sella (TS optimizer)"),
    ]
    for module_name, hint, *label in checks:
        _check_import(module_name, hint, label[0] if label else None)

    thread_status = inspect_thread_settings()
    print("INFO thread environment settings:")
    for env_name, env_value in thread_status["environment"].items():
        print(f"  {env_name}={env_value}")
    print(f"INFO requested thread count = {thread_status['requested']}")
    print(f"INFO pyscf.lib.num_threads() = {thread_status['effective_threads']}")
    print(f"INFO openmp_available = {thread_status['openmp_available']}")

    if failures:
        print(f"FAIL {len(failures)} checks failed: {', '.join(failures)}")
        sys.exit(1)
    print("OK  all checks passed")


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


def prepare_run_context(args, config: RunConfig, config_raw):
    config_dict = config.to_dict()
    calculation_mode = _normalize_calculation_mode(config.calculation_mode)
    basis = config.basis
    xc = normalize_xc_functional(config.xc)
    solvent_name = config.solvent
    solvent_model = config.solvent_model
    dispersion_model = config.dispersion
    optimizer_config = config.optimizer
    optimizer_ase_config = optimizer_config.ase if optimizer_config else None
    optimizer_ase_dict = optimizer_ase_config.to_dict() if optimizer_ase_config else {}
    constraints = config.constraints
    scan_config = config.scan2d or config.scan
    if config.scan and config.scan2d:
        raise ValueError("Config must not define both 'scan' and 'scan2d'.")
    if scan_config and calculation_mode != "scan":
        raise ValueError("Scan configuration requires calculation_mode='scan'.")
    if calculation_mode == "scan" and not scan_config:
        raise ValueError("calculation_mode='scan' requires a scan configuration.")
    scan_mode = None
    if scan_config:
        scan_mode = _normalize_scan_mode(scan_config.get("mode"))
    optimizer_mode = None
    if calculation_mode in ("optimization", "irc") or (
        calculation_mode == "scan" and scan_mode == "optimization"
    ):
        optimizer_mode = _normalize_optimizer_mode(
            optimizer_config.mode if optimizer_config else ("transition_state" if calculation_mode == "irc" else None)
        )
    solvent_map_path = config.solvent_map or DEFAULT_SOLVENT_MAP_PATH
    single_point_config = config.single_point
    thermo_config = config.thermo
    frequency_enabled, single_point_enabled = _normalize_stage_flags(
        config, calculation_mode
    )
    if calculation_mode == "optimization":
        irc_enabled = bool(config.irc_enabled)
    else:
        irc_enabled = calculation_mode == "irc"
    irc_config = config.irc
    if not basis:
        raise ValueError("Config must define 'basis' in the JSON config file.")
    if not xc:
        raise ValueError("Config must define 'xc' in the JSON config file.")
    if solvent_model and not solvent_name:
        raise ValueError("Config must define 'solvent' when 'solvent_model' is set.")
    if solvent_name:
        if not solvent_model and not _is_vacuum_solvent(solvent_name):
            raise ValueError("Config must define 'solvent_model' when 'solvent' is set.")
        if solvent_model and solvent_model.lower() not in ("pcm", "smd"):
            raise ValueError("Config 'solvent_model' must be one of: pcm, smd.")
    thread_count = config.threads if config.threads is not None else DEFAULT_THREAD_COUNT
    memory_gb = config.memory_gb
    verbose = bool(config.verbose)
    resume_dir = getattr(args, "resume", None)
    run_dir = args.run_dir or resume_dir or create_run_directory()
    os.makedirs(run_dir, exist_ok=True)
    log_path = resolve_run_path(run_dir, config.log_file or DEFAULT_LOG_PATH)
    log_path = format_log_path(log_path)
    scf_config = config.scf.to_dict() if config.scf else {}
    pyscf_chkfile = _resolve_scf_chkfile(scf_config, run_dir)
    optimized_xyz_path = resolve_run_path(
        run_dir, config.optimized_xyz_file or DEFAULT_OPTIMIZED_XYZ_PATH
    )
    run_metadata_path = resolve_run_path(
        run_dir, config.run_metadata_file or DEFAULT_RUN_METADATA_PATH
    )
    frequency_output_path = resolve_run_path(
        run_dir, config.frequency_file or DEFAULT_FREQUENCY_PATH
    )
    irc_output_path = resolve_run_path(run_dir, config.irc_file or DEFAULT_IRC_PATH)
    scan_result_path = resolve_run_path(run_dir, DEFAULT_SCAN_RESULT_PATH)
    ensure_parent_dir(log_path)
    ensure_parent_dir(optimized_xyz_path)
    ensure_parent_dir(run_metadata_path)
    ensure_parent_dir(frequency_output_path)
    ensure_parent_dir(irc_output_path)
    ensure_parent_dir(scan_result_path)

    if args.interactive:
        config_used_path = resolve_run_path(run_dir, "config_used.json")
        ensure_parent_dir(config_used_path)
        with open(config_used_path, "w", encoding="utf-8") as config_used_file:
            config_used_file.write(config_raw)
        args.config = config_used_path

    checkpoint_path = resolve_run_path(run_dir, "checkpoint.json")
    run_id, run_id_history, attempt, checkpoint_payload = _resolve_run_identity(
        resume_dir,
        run_metadata_path,
        checkpoint_path,
        override_run_id=args.run_id,
    )
    if not run_id:
        run_id = str(uuid.uuid4())
    args.run_id = run_id
    if not resume_dir:
        checkpoint_payload = {
            "created_at": datetime.now().isoformat(),
            "run_dir": run_dir,
            "run_id": args.run_id,
            "attempt": attempt,
            "run_id_history": run_id_history,
            "xyz_file": os.path.abspath(args.xyz_file) if args.xyz_file else None,
            "config_source_path": str(args.config) if args.config else None,
            "config_raw": config_raw,
        }
        write_checkpoint(checkpoint_path, checkpoint_payload)
    elif checkpoint_payload is not None:
        updated = False
        if checkpoint_payload.get("run_id") != run_id:
            checkpoint_payload["run_id"] = run_id
            updated = True
        if checkpoint_payload.get("attempt") != attempt:
            checkpoint_payload["attempt"] = attempt
            updated = True
        if run_id_history and checkpoint_payload.get("run_id_history") != run_id_history:
            checkpoint_payload["run_id_history"] = run_id_history
            updated = True
        if updated:
            write_checkpoint(checkpoint_path, checkpoint_payload)
    elif pyscf_chkfile:
        _warn_missing_chkfile("Resume mode:", pyscf_chkfile)

    event_log_path = resolve_run_path(
        run_dir, config.event_log_file or DEFAULT_EVENT_LOG_PATH
    )
    if event_log_path:
        event_log_path = format_log_path(event_log_path)
        ensure_parent_dir(event_log_path)

    return {
        "config_dict": config_dict,
        "config_raw": config_raw,
        "calculation_mode": calculation_mode,
        "basis": basis,
        "xc": xc,
        "solvent_name": solvent_name,
        "solvent_model": solvent_model,
        "dispersion_model": dispersion_model,
        "optimizer_config": optimizer_config,
        "optimizer_ase_dict": optimizer_ase_dict,
        "optimizer_mode": optimizer_mode,
        "constraints": constraints,
        "scan_config": scan_config,
        "scan_mode": scan_mode,
        "solvent_map_path": solvent_map_path,
        "single_point_config": single_point_config,
        "thermo": thermo_config,
        "frequency_enabled": frequency_enabled,
        "single_point_enabled": single_point_enabled,
        "thread_count": thread_count,
        "memory_gb": memory_gb,
        "verbose": verbose,
        "run_dir": run_dir,
        "log_path": log_path,
        "scf_config": scf_config,
        "optimized_xyz_path": optimized_xyz_path,
        "run_metadata_path": run_metadata_path,
        "frequency_output_path": frequency_output_path,
        "irc_output_path": irc_output_path,
        "scan_result_path": scan_result_path,
        "event_log_path": event_log_path,
        "run_id": run_id,
        "attempt": attempt,
        "run_id_history": run_id_history,
        "resume_dir": resume_dir,
        "pyscf_chkfile": pyscf_chkfile,
        "irc_enabled": irc_enabled,
        "irc_config": irc_config,
        "previous_status": getattr(args, "resume_previous_status", None),
        "checkpoint_path": checkpoint_path,
    }


def _enqueue_background_run(args, context):
    queued_at = datetime.now().isoformat()
    queue_priority = args.queue_priority
    max_runtime_seconds = args.queue_max_runtime
    queued_metadata = {
        "status": "queued",
        "run_directory": context["run_dir"],
        "run_id": context["run_id"],
        "attempt": context["attempt"],
        "run_id_history": context["run_id_history"],
        "xyz_file": args.xyz_file,
        "config_file": args.config,
        "run_metadata_file": context["run_metadata_path"],
        "log_file": context["log_path"],
        "event_log_file": context["event_log_path"],
        "queued_at": queued_at,
        "priority": queue_priority,
        "max_runtime_seconds": max_runtime_seconds,
    }
    write_run_metadata(context["run_metadata_path"], queued_metadata)
    queue_entry = {
        "status": "queued",
        "run_directory": context["run_dir"],
        "run_id": context["run_id"],
        "attempt": context["attempt"],
        "run_id_history": context["run_id_history"],
        "xyz_file": args.xyz_file,
        "config_file": args.config,
        "solvent_map": args.solvent_map,
        "run_metadata_file": context["run_metadata_path"],
        "log_file": context["log_path"],
        "event_log_file": context["event_log_path"],
        "queued_at": queued_at,
        "priority": queue_priority,
        "max_runtime_seconds": max_runtime_seconds,
        "retry_count": 0,
    }
    position = enqueue_run(queue_entry, DEFAULT_QUEUE_PATH, DEFAULT_QUEUE_LOCK_PATH)
    record_status_event(
        context["event_log_path"],
        context["run_id"],
        context["run_dir"],
        "queued",
        previous_status=context.get("previous_status"),
        details={
            "priority": queue_priority,
            "max_runtime_seconds": max_runtime_seconds,
        },
    )
    runner_command = [
        sys.executable,
        os.path.abspath(sys.argv[0]),
        "--queue-runner",
    ]
    ensure_queue_runner_started(runner_command, DEFAULT_QUEUE_RUNNER_LOG_PATH)
    print("Background run queued.")
    print(f"  Run ID       : {context['run_id']}")
    print(f"  Queue pos    : {position}")
    print(f"  Run dir      : {context['run_dir']}")
    print(f"  Metadata     : {context['run_metadata_path']}")
    print(f"  Log file     : {context['log_path']}")
    print(f"  Queue runner : {DEFAULT_QUEUE_RUNNER_LOG_PATH}")


def build_molecule_context(args, context, memory_mb):
    from pyscf import dft, gto

    atom_spec, charge, spin, multiplicity = load_xyz(args.xyz_file)
    if context["optimizer_mode"] == "transition_state" and multiplicity is None:
        if args.interactive:
            logging.info("TS 모드: multiplicity 입력 강제")
            while True:
                raw_value = input("Multiplicity(2S+1)를 입력하세요: ").strip()
                try:
                    multiplicity = int(raw_value)
                except ValueError:
                    print("Multiplicity는 양의 정수여야 합니다.")
                    continue
                if multiplicity < 1:
                    print("Multiplicity는 양의 정수여야 합니다.")
                    continue
                break
        else:
            raise ValueError(
                "Transition-state mode requires multiplicity; "
                "provide it in the XYZ comment line or run with --interactive."
            )
    total_electrons = total_electron_count(atom_spec, charge)
    if multiplicity is not None:
        if multiplicity < 1:
            raise ValueError("Multiplicity must be a positive integer (2S+1).")
        multiplicity_spin = multiplicity - 1
        if spin is not None and spin != multiplicity_spin:
            raise ValueError(
                "Spin and multiplicity are inconsistent. "
                f"spin={spin} implies multiplicity={spin + 1}, "
                f"but multiplicity={multiplicity} was provided."
            )
        spin = multiplicity_spin
    if spin is None:
        spin = total_electrons % 2
        logging.warning(
            "Auto spin estimation enabled: spin not specified; using parity "
            "(spin = total_electrons %% 2). For TS/radical/metal/diradical cases, "
            "set multiplicity in the XYZ comment line to avoid incorrect states."
        )
    elif spin < 0 or spin > total_electrons:
        raise ValueError(
            "Spin is outside the valid electron count range. "
            f"Total electrons: {total_electrons}, spin: {spin}. "
            "Spin must be between 0 and total electrons."
        )
    elif (total_electrons - spin) % 2 != 0:
        raise ValueError(
            "Spin is inconsistent with electron count. "
            f"Total electrons: {total_electrons}, spin: {spin}. "
            "Spin must satisfy (Nalpha - Nbeta) with Nalpha+Nbeta=total electrons."
        )
    if multiplicity is None:
        multiplicity = spin + 1
    mol = gto.M(atom=atom_spec, basis=context["basis"], charge=charge, spin=spin)
    if memory_mb:
        mol.max_memory = memory_mb

    ks_type = select_ks_type(
        mol=mol,
        scf_config=context["scf_config"],
        optimizer_mode=context["optimizer_mode"],
        multiplicity=multiplicity,
    )
    if ks_type == "RKS":
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = context["xc"]

    return {
        "atom_spec": atom_spec,
        "charge": charge,
        "spin": spin,
        "multiplicity": multiplicity,
        "mol": mol,
        "mf": mf,
        "ks_type": ks_type,
        "total_electrons": total_electrons,
    }


def finalize_metadata(
    run_metadata_path,
    event_log_path,
    run_id,
    run_dir,
    metadata,
    status,
    previous_status,
    queue_update_fn=None,
    exit_code=None,
    details=None,
    error=None,
):
    metadata["status"] = status
    metadata["run_ended_at"] = datetime.now().isoformat()
    metadata["run_updated_at"] = datetime.now().isoformat()
    if error is not None:
        metadata["error"] = str(error)
        metadata["traceback"] = traceback.format_exc()
    write_run_metadata(run_metadata_path, metadata)
    record_status_event(
        event_log_path,
        run_id,
        run_dir,
        status,
        previous_status=previous_status,
        details=details,
    )
    if queue_update_fn:
        queue_update_fn(status, exit_code=exit_code)


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


def run_single_point_stage(stage_context, queue_update_fn):
    logging.info("Starting single-point energy calculation...")
    run_start = stage_context["run_start"]
    calculation_metadata = stage_context["metadata"]
    try:
        sp_result = compute_single_point_energy(
            stage_context["mol"],
            stage_context["calc_basis"],
            stage_context["calc_xc"],
            stage_context["calc_scf_config"],
            stage_context["calc_solvent_model"],
            stage_context["calc_solvent_name"],
            stage_context["calc_eps"],
            stage_context["calc_dispersion_model"],
            stage_context["verbose"],
            stage_context["memory_mb"],
            run_dir=stage_context["run_dir"],
            optimizer_mode=stage_context["optimizer_mode"],
            multiplicity=stage_context["multiplicity"],
            log_override=False,
        )
        calculation_metadata["dispersion_info"] = sp_result.get("dispersion")
        energy = sp_result.get("energy")
        sp_converged = sp_result.get("converged")
        sp_cycles = sp_result.get("cycles")
        summary = {
            "elapsed_seconds": time.perf_counter() - run_start,
            "n_steps": sp_cycles,
            "final_energy": energy,
            "opt_final_energy": energy,
            "final_sp_energy": energy,
            "final_sp_converged": sp_converged,
            "final_sp_cycles": sp_cycles,
            "scf_converged": sp_converged,
            "opt_converged": None,
            "converged": bool(sp_converged) if sp_converged is not None else True,
        }
        calculation_metadata["summary"] = summary
        calculation_metadata["summary"]["memory_limit_enforced"] = stage_context[
            "memory_limit_enforced"
        ]
        _update_checkpoint_scf(
            stage_context.get("checkpoint_path"),
            pyscf_chkfile=stage_context.get("pyscf_chkfile"),
            scf_energy=energy,
            scf_converged=sp_converged,
        )
        finalize_metadata(
            stage_context["run_metadata_path"],
            stage_context["event_log_path"],
            stage_context["run_id"],
            stage_context["run_dir"],
            calculation_metadata,
            status="completed",
            previous_status="running",
            queue_update_fn=queue_update_fn,
            exit_code=0,
        )
    except Exception as exc:
        logging.exception("Calculation failed.")
        finalize_metadata(
            stage_context["run_metadata_path"],
            stage_context["event_log_path"],
            stage_context["run_id"],
            stage_context["run_dir"],
            calculation_metadata,
            status="failed",
            previous_status="running",
            queue_update_fn=queue_update_fn,
            exit_code=1,
            details={"error": str(exc)},
            error=exc,
        )
        raise


def run_frequency_stage(stage_context, queue_update_fn):
    logging.info("Starting frequency calculation...")
    run_start = stage_context["run_start"]
    calculation_metadata = stage_context["metadata"]
    try:
        frequency_result = compute_frequencies(
            stage_context["mol"],
            stage_context["calc_basis"],
            stage_context["calc_xc"],
            stage_context["calc_scf_config"],
            stage_context["calc_solvent_model"],
            stage_context["calc_solvent_name"],
            stage_context["calc_eps"],
            stage_context["calc_dispersion_model"],
            stage_context["freq_dispersion_mode"],
            stage_context["thermo"],
            stage_context["verbose"],
            stage_context["memory_mb"],
            run_dir=stage_context["run_dir"],
            optimizer_mode=stage_context["optimizer_mode"],
            multiplicity=stage_context["multiplicity"],
            log_override=False,
        )
        imaginary_check = frequency_result.get("imaginary_check") or {}
        imaginary_status = imaginary_check.get("status")
        imaginary_message = imaginary_check.get("message")
        if imaginary_message:
            if imaginary_status == "one_imaginary":
                logging.info("Imaginary frequency check: %s", imaginary_message)
            else:
                logging.warning("Imaginary frequency check: %s", imaginary_message)
        frequency_payload = {
            "status": "completed",
            "output_file": stage_context["frequency_output_path"],
            "units": _frequency_units(),
            "versions": _frequency_versions(),
            "basis": stage_context["calc_basis"],
            "xc": stage_context["calc_xc"],
            "scf": stage_context["calc_scf_config"],
            "solvent": stage_context["calc_solvent_name"],
            "solvent_model": stage_context["calc_solvent_model"]
            if stage_context["calc_solvent_name"]
            else None,
            "solvent_eps": stage_context["calc_eps"],
            "dispersion": stage_context["calc_dispersion_model"],
            "dispersion_mode": stage_context["freq_dispersion_mode"],
            "thermochemistry": _thermochemistry_payload(
                stage_context["thermo"], frequency_result.get("thermochemistry")
            ),
            "results": frequency_result,
        }
        with open(
            stage_context["frequency_output_path"], "w", encoding="utf-8"
        ) as handle:
            json.dump(frequency_payload, handle, indent=2)
        calculation_metadata["frequency"] = frequency_payload
        calculation_metadata["dispersion_info"] = frequency_result.get("dispersion")
        energy = frequency_result.get("energy")
        sp_converged = frequency_result.get("converged")
        sp_cycles = frequency_result.get("cycles")
        summary = {
            "elapsed_seconds": time.perf_counter() - run_start,
            "n_steps": sp_cycles,
            "final_energy": energy,
            "opt_final_energy": energy,
            "final_sp_energy": energy,
            "final_sp_converged": sp_converged,
            "final_sp_cycles": sp_cycles,
            "scf_converged": sp_converged,
            "opt_converged": None,
            "converged": bool(sp_converged) if sp_converged is not None else True,
        }
        calculation_metadata["summary"] = summary
        calculation_metadata["summary"]["memory_limit_enforced"] = stage_context[
            "memory_limit_enforced"
        ]
        _update_checkpoint_scf(
            stage_context.get("checkpoint_path"),
            pyscf_chkfile=stage_context.get("pyscf_chkfile"),
            scf_energy=frequency_result.get("energy"),
            scf_converged=frequency_result.get("converged"),
        )
        finalize_metadata(
            stage_context["run_metadata_path"],
            stage_context["event_log_path"],
            stage_context["run_id"],
            stage_context["run_dir"],
            calculation_metadata,
            status="completed",
            previous_status="running",
            queue_update_fn=queue_update_fn,
            exit_code=0,
        )
    except Exception as exc:
        logging.exception("Calculation failed.")
        finalize_metadata(
            stage_context["run_metadata_path"],
            stage_context["event_log_path"],
            stage_context["run_id"],
            stage_context["run_dir"],
            calculation_metadata,
            status="failed",
            previous_status="running",
            queue_update_fn=queue_update_fn,
            exit_code=1,
            details={"error": str(exc)},
            error=exc,
        )
        raise


def run_irc_stage(stage_context, queue_update_fn):
    logging.info("Starting IRC calculation...")
    run_start = stage_context["run_start"]
    calculation_metadata = stage_context["metadata"]
    try:
        irc_result = _run_ase_irc(
            stage_context["input_xyz"],
            stage_context["run_dir"],
            stage_context["charge"],
            stage_context["spin"],
            stage_context["multiplicity"],
            stage_context["calc_basis"],
            stage_context["calc_xc"],
            stage_context["calc_scf_config"],
            stage_context["calc_solvent_model"],
            stage_context["calc_solvent_name"],
            stage_context["calc_eps"],
            stage_context["calc_dispersion_model"],
            stage_context["verbose"],
            stage_context["memory_mb"],
            stage_context["optimizer_ase_config"],
            stage_context["optimizer_mode"],
            stage_context["constraints"],
            stage_context["mode_vector"],
            stage_context["irc_steps"],
            stage_context["irc_step_size"],
            stage_context["irc_force_threshold"],
        )
        irc_payload = {
            "status": "completed",
            "output_file": stage_context["irc_output_path"],
            "forward_xyz": irc_result.get("forward_xyz"),
            "reverse_xyz": irc_result.get("reverse_xyz"),
            "steps": stage_context["irc_steps"],
            "step_size": stage_context["irc_step_size"],
            "force_threshold": stage_context["irc_force_threshold"],
            "mode_eigenvalue": stage_context.get("mode_eigenvalue"),
            "profile": irc_result.get("profile", []),
        }
        with open(stage_context["irc_output_path"], "w", encoding="utf-8") as handle:
            json.dump(irc_payload, handle, indent=2)
        calculation_metadata["irc"] = irc_payload
        energy_summary = None
        if irc_payload["profile"]:
            energy_summary = {
                "start_energy_ev": irc_payload["profile"][0]["energy_ev"],
                "end_energy_ev": irc_payload["profile"][-1]["energy_ev"],
            }
        summary = {
            "elapsed_seconds": time.perf_counter() - run_start,
            "n_steps": len(irc_payload["profile"]),
            "final_energy": energy_summary["end_energy_ev"] if energy_summary else None,
            "opt_final_energy": None,
            "final_sp_energy": None,
            "final_sp_converged": None,
            "final_sp_cycles": None,
            "scf_converged": None,
            "opt_converged": None,
            "converged": True,
        }
        calculation_metadata["summary"] = summary
        calculation_metadata["summary"]["memory_limit_enforced"] = stage_context[
            "memory_limit_enforced"
        ]
        finalize_metadata(
            stage_context["run_metadata_path"],
            stage_context["event_log_path"],
            stage_context["run_id"],
            stage_context["run_dir"],
            calculation_metadata,
            status="completed",
            previous_status="running",
            queue_update_fn=queue_update_fn,
            exit_code=0,
        )
    except Exception as exc:
        logging.exception("IRC calculation failed.")
        finalize_metadata(
            stage_context["run_metadata_path"],
            stage_context["event_log_path"],
            stage_context["run_id"],
            stage_context["run_dir"],
            calculation_metadata,
            status="failed",
            previous_status="running",
            queue_update_fn=queue_update_fn,
            exit_code=1,
            details={"error": str(exc)},
            error=exc,
        )
        raise


def run_scan_stage(
    args,
    context,
    molecule_context,
    memory_mb,
    memory_limit_status,
    memory_limit_enforced,
    openmp_available,
    effective_threads,
    queue_update_fn,
):
    from ase.io import read as ase_read
    from ase.io import write as ase_write
    from pyscf import gto

    scan_config = context["scan_config"] or {}
    scan_mode = context["scan_mode"]
    dimensions, values_grid = _parse_scan_dimensions(scan_config)
    scan_points = list(itertools.product(*values_grid))

    run_dir = context["run_dir"]
    log_path = context["log_path"]
    run_metadata_path = context["run_metadata_path"]
    scan_result_path = context["scan_result_path"]
    event_log_path = context["event_log_path"]
    run_id = context["run_id"]
    charge = molecule_context["charge"]
    spin = molecule_context["spin"]
    multiplicity = molecule_context["multiplicity"]
    verbose = context["verbose"]
    thread_count = context["thread_count"]

    scan_dir = resolve_run_path(run_dir, "scan")
    os.makedirs(scan_dir, exist_ok=True)

    calc_basis = context["basis"]
    calc_xc = context["xc"]
    calc_scf_config = context["scf_config"]
    calc_solvent_name = context["solvent_name"]
    calc_solvent_model = context["solvent_model"]
    calc_eps = context["eps"]
    calc_dispersion_model = context["dispersion_model"]
    optimizer_mode = context["optimizer_mode"]
    optimizer_ase_dict = context["optimizer_ase_dict"]

    if scan_mode == "single_point":
        calc_basis = context["sp_basis"]
        calc_xc = context["sp_xc"]
        calc_scf_config = context["sp_scf_config"]
        calc_solvent_name = context["sp_solvent_name"]
        calc_solvent_model = context["sp_solvent_model"]
        calc_eps = context["sp_eps"]
        calc_dispersion_model = context["sp_dispersion_model"]

    scan_summary = {
        "status": "running",
        "run_directory": run_dir,
        "run_started_at": datetime.now().isoformat(),
        "run_id": run_id,
        "attempt": context["attempt"],
        "run_id_history": context["run_id_history"],
        "pid": os.getpid(),
        "xyz_file": args.xyz_file,
        "xyz_file_hash": compute_file_hash(args.xyz_file),
        "basis": calc_basis,
        "xc": calc_xc,
        "solvent": calc_solvent_name,
        "solvent_model": calc_solvent_model if calc_solvent_name else None,
        "solvent_eps": calc_eps,
        "dispersion": calc_dispersion_model,
        "scan": {
            "mode": scan_mode,
            "dimensions": dimensions,
            "grid_counts": [len(values) for values in values_grid],
            "total_points": len(scan_points),
        },
        "scan_result_file": scan_result_path,
        "calculation_mode": "scan",
        "charge": charge,
        "spin": spin,
        "multiplicity": multiplicity,
        "thread_count": thread_count,
        "effective_thread_count": effective_threads,
        "openmp_available": openmp_available,
        "memory_mb": memory_mb,
        "memory_limit_status": memory_limit_status,
        "log_file": log_path,
        "event_log_file": event_log_path,
        "run_metadata_file": run_metadata_path,
        "config_file": args.config,
        "config": context["config_dict"],
        "config_raw": context["config_raw"],
        "config_hash": compute_text_hash(context["config_raw"]),
        "scf_config": calc_scf_config,
        "scf_settings": calc_scf_config,
        "environment": collect_environment_snapshot(thread_count),
        "git": collect_git_metadata(os.getcwd()),
        "versions": {
            "ase": get_package_version("ase"),
            "pyscf": get_package_version("pyscf"),
            "dftd3": get_package_version("dftd3"),
            "dftd4": get_package_version("dftd4"),
        },
    }
    scan_summary["run_updated_at"] = datetime.now().isoformat()
    write_run_metadata(run_metadata_path, scan_summary)
    record_status_event(
        event_log_path,
        run_id,
        run_dir,
        "running",
        previous_status=context.get("previous_status"),
    )

    mol = molecule_context["mol"]
    run_capability_check(
        mol,
        calc_basis,
        calc_xc,
        calc_scf_config,
        calc_solvent_model if calc_solvent_name else None,
        calc_solvent_name,
        calc_eps,
        None,
        "none",
        require_hessian=False,
        verbose=verbose,
        memory_mb=memory_mb,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
    )

    logging.info("Starting scan (%s mode) with %d points.", scan_mode, len(scan_points))
    if thread_count:
        logging.info("Using threads: %s", thread_count)
    if memory_mb:
        logging.info("Memory target: %s MB (PySCF max_memory)", memory_mb)
    if log_path:
        logging.info("Log file: %s", log_path)
    if scan_result_path:
        logging.info("Scan results file: %s", scan_result_path)

    run_start = time.perf_counter()
    base_atoms = ase_read(args.xyz_file)
    results = []
    try:
        for index, values in enumerate(scan_points):
            point_label = {"index": index}
            for dimension, value in zip(dimensions, values, strict=True):
                point_label[_dimension_key(dimension)] = value
            atoms = base_atoms.copy()
            _apply_scan_geometry(atoms, dimensions, values)
            input_xyz_path = resolve_run_path(scan_dir, f"scan_{index:03d}_input.xyz")
            ase_write(input_xyz_path, atoms)
            output_xyz_path = None
            n_steps = None
            scf_result = None
            if scan_mode == "optimization":
                output_xyz_path = resolve_run_path(
                    scan_dir, f"scan_{index:03d}_optimized.xyz"
                )
                scan_constraints = _build_scan_constraints(dimensions, values)
                merged_constraints = _merge_constraints(
                    context["constraints"], scan_constraints
                )
                n_steps = _run_ase_optimizer(
                    input_xyz_path,
                    output_xyz_path,
                    run_dir,
                    charge,
                    spin,
                    multiplicity,
                    calc_basis,
                    calc_xc,
                    calc_scf_config,
                    calc_solvent_model.lower() if calc_solvent_model else None,
                    calc_solvent_name,
                    calc_eps,
                    calc_dispersion_model,
                    verbose,
                    memory_mb,
                    optimizer_ase_dict,
                    optimizer_mode,
                    merged_constraints,
                )
                atoms = ase_read(output_xyz_path)
            atom_spec = _atoms_to_atom_spec(atoms)
            mol_scan = gto.M(
                atom=atom_spec,
                basis=calc_basis,
                charge=charge,
                spin=spin,
            )
            if memory_mb:
                mol_scan.max_memory = memory_mb
            scf_result = compute_single_point_energy(
                mol_scan,
                calc_basis,
                calc_xc,
                calc_scf_config,
                calc_solvent_model if calc_solvent_name else None,
                calc_solvent_name,
                calc_eps,
                calc_dispersion_model,
                verbose,
                memory_mb,
                run_dir=run_dir,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
                log_override=False,
            )
            point_result = {
                "index": index,
                "values": point_label,
                "mode": scan_mode,
                "energy": scf_result.get("energy") if scf_result else None,
                "converged": scf_result.get("converged") if scf_result else None,
                "cycles": scf_result.get("cycles") if scf_result else None,
                "optimizer_steps": n_steps,
                "input_xyz": input_xyz_path,
                "output_xyz": output_xyz_path,
            }
            results.append(point_result)
            with open(scan_result_path, "w", encoding="utf-8") as handle:
                json.dump({"results": results}, handle, indent=2)
            scan_summary["scan"]["completed_points"] = len(results)
            scan_summary["run_updated_at"] = datetime.now().isoformat()
            write_run_metadata(run_metadata_path, scan_summary)
        elapsed_seconds = time.perf_counter() - run_start
        converged_points = sum(
            1 for item in results if item.get("converged") is True
        )
        scan_summary["summary"] = {
            "elapsed_seconds": elapsed_seconds,
            "n_points": len(results),
            "converged_points": converged_points,
            "final_energy": results[-1]["energy"] if results else None,
            "converged": converged_points == len(results) if results else False,
        }
        scan_summary["summary"]["memory_limit_enforced"] = memory_limit_enforced
        finalize_metadata(
            run_metadata_path,
            event_log_path,
            run_id,
            run_dir,
            scan_summary,
            status="completed",
            previous_status="running",
            queue_update_fn=queue_update_fn,
            exit_code=0,
        )
    except Exception as exc:
        logging.exception("Scan calculation failed.")
        finalize_metadata(
            run_metadata_path,
            event_log_path,
            run_id,
            run_dir,
            scan_summary,
            status="failed",
            previous_status="running",
            queue_update_fn=queue_update_fn,
            exit_code=1,
            details={"error": str(exc)},
            error=exc,
        )
        raise


def run_optimization_stage(
    args,
    context,
    molecule_context,
    memory_mb,
    memory_limit_status,
    memory_limit_enforced,
    openmp_available,
    effective_threads,
    queue_update_fn,
):
    from pyscf import gto

    basis = context["basis"]
    xc = context["xc"]
    scf_config = context["scf_config"]
    solvent_name = context["solvent_name"]
    solvent_model = context["solvent_model"]
    solvent_map_path = context["solvent_map_path"]
    dispersion_model = context["dispersion_model"]
    optimizer_config = context["optimizer_config"]
    optimizer_ase_dict = context["optimizer_ase_dict"]
    optimizer_mode = context["optimizer_mode"]
    frequency_enabled = context["frequency_enabled"]
    single_point_enabled = context["single_point_enabled"]
    thread_count = context["thread_count"]
    memory_gb = context["memory_gb"]
    verbose = context["verbose"]
    run_dir = context["run_dir"]
    log_path = context["log_path"]
    optimized_xyz_path = context["optimized_xyz_path"]
    run_metadata_path = context["run_metadata_path"]
    frequency_output_path = context["frequency_output_path"]
    irc_output_path = context["irc_output_path"]
    event_log_path = context["event_log_path"]
    run_id = context["run_id"]
    irc_enabled = context["irc_enabled"]
    irc_config = context["irc_config"]
    checkpoint_path = context["checkpoint_path"]
    mol = molecule_context["mol"]
    mf = molecule_context["mf"]
    charge = molecule_context["charge"]
    spin = molecule_context["spin"]
    multiplicity = molecule_context["multiplicity"]
    ks_type = molecule_context["ks_type"]

    logging.info("Running capability check for geometry optimization (SCF + gradient)...")
    run_capability_check(
        mol,
        basis,
        xc,
        scf_config,
        solvent_model,
        solvent_name,
        context["eps"],
        None,
        "none",
        require_hessian=False,
        verbose=verbose,
        memory_mb=memory_mb,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
    )
    if frequency_enabled:
        logging.info(
            "Running capability check for frequency calculation (SCF + gradient + Hessian)..."
        )
        run_capability_check(
            mol,
            context["sp_basis"],
            context["sp_xc"],
            context["sp_scf_config"],
            context["sp_solvent_model"] if context["sp_solvent_name"] else None,
            context["sp_solvent_name"],
            context["sp_eps"],
            context["freq_dispersion_model"],
            context["freq_dispersion_mode"],
            require_hessian=True,
            verbose=verbose,
            memory_mb=memory_mb,
            optimizer_mode=optimizer_mode,
            multiplicity=multiplicity,
        )

    logging.info("Starting geometry optimization...")
    logging.info("Run ID: %s", run_id)
    logging.info("Run directory: %s", run_dir)
    logging.info("Optimization mode: %s", optimizer_mode)
    if thread_count:
        logging.info("Using threads: %s", thread_count)
        if openmp_available is False:
            effective_display = (
                str(effective_threads) if effective_threads is not None else "unknown"
            )
            logging.warning(
                "OpenMP appears unavailable; requested threads may have no effect "
                "(effective threads: %s).",
                effective_display,
            )
    if memory_gb:
        logging.info("Memory target: %s GB (PySCF max_memory)", memory_gb)
        if memory_limit_status:
            if memory_limit_status["applied"]:
                limit_gb = memory_limit_status["limit_bytes"] / (1024 ** 3)
                limit_name = memory_limit_status["limit_name"] or "unknown"
                logging.info(
                    "OS hard memory limit: applied at %.2f GB (%s)",
                    limit_gb,
                    limit_name,
                )
            else:
                log_fn = logging.warning
                if memory_limit_status["reason"] == "disabled by config":
                    log_fn = logging.info
                log_fn(
                    "OS hard memory limit: not applied (%s).",
                    memory_limit_status["reason"],
                )
    logging.info("Verbose logging: %s", "enabled" if verbose else "disabled")
    logging.info("Log file: %s", log_path)
    if event_log_path:
        logging.info("Event log file: %s", event_log_path)
    if context["applied_scf"]:
        logging.info("SCF settings: %s", context["applied_scf"])
    if dispersion_model:
        logging.info("Dispersion correction: %s", dispersion_model)
        if context["dispersion_info"]:
            logging.info("Dispersion details: %s", context["dispersion_info"])
    if context["sp_xc"] != xc:
        logging.info("Single-point XC override: %s", context["sp_xc"])
    if context["sp_basis"] != basis:
        logging.info("Single-point basis override: %s", context["sp_basis"])
    if context["sp_scf_config"] != scf_config:
        logging.info("Single-point SCF override: %s", context["sp_scf_config"])
    if context["sp_solvent_name"] != solvent_name or context["sp_solvent_model"] != solvent_model:
        logging.info(
            "Single-point solvent override: %s (%s)",
            context["sp_solvent_name"],
            context["sp_solvent_model"],
        )
    if (
        context["sp_dispersion_model"] is not None
        and context["sp_dispersion_model"] != dispersion_model
    ):
        logging.info("Single-point dispersion override: %s", context["sp_dispersion_model"])
    if frequency_enabled:
        logging.info("Frequency dispersion mode: %s", context["freq_dispersion_mode"])
    run_start = time.perf_counter()
    optimization_metadata = {
        "status": "running",
        "run_directory": run_dir,
        "run_started_at": datetime.now().isoformat(),
        "run_id": run_id,
        "attempt": context["attempt"],
        "run_id_history": context["run_id_history"],
        "pid": os.getpid(),
        "xyz_file": args.xyz_file,
        "xyz_file_hash": compute_file_hash(args.xyz_file),
        "basis": basis,
        "xc": xc,
        "solvent": solvent_name,
        "solvent_model": solvent_model if solvent_name else None,
        "solvent_eps": context["eps"],
        "solvent_map": solvent_map_path,
        "dispersion": dispersion_model,
        "dispersion_info": context["dispersion_info"],
        "frequency_dispersion_mode": context["freq_dispersion_mode"] if frequency_enabled else None,
        "optimizer": {
            "mode": optimizer_mode,
            "output_xyz": optimizer_config.output_xyz if optimizer_config else None,
            "ase": optimizer_ase_dict or None,
        },
        "single_point": {
            "basis": context["sp_basis"],
            "xc": context["sp_xc"],
            "scf": context["sp_scf_config"],
            "solvent": context["sp_solvent_name"],
            "solvent_model": context["sp_solvent_model"]
            if context["sp_solvent_name"]
            else None,
            "solvent_eps": context["sp_eps"],
            "solvent_map": context["sp_solvent_map_path"],
            "dispersion": context["sp_dispersion_model"],
            "frequency_dispersion_mode": context["freq_dispersion_mode"]
            if frequency_enabled
            else None,
        },
        "single_point_enabled": single_point_enabled,
        "frequency_enabled": frequency_enabled,
        "irc_enabled": irc_enabled,
        "calculation_mode": "optimization",
        "charge": charge,
        "spin": spin,
        "multiplicity": multiplicity,
        "ks_type": ks_type,
        "thread_count": thread_count,
        "effective_thread_count": effective_threads,
        "openmp_available": openmp_available,
        "memory_gb": memory_gb,
        "memory_mb": memory_mb,
        "memory_limit_status": memory_limit_status,
        "log_file": log_path,
        "event_log_file": event_log_path,
        "optimized_xyz_file": optimized_xyz_path,
        "frequency_file": frequency_output_path,
        "irc_file": irc_output_path,
        "run_metadata_file": run_metadata_path,
        "config_file": args.config,
        "config": context["config_dict"],
        "config_raw": context["config_raw"],
        "config_hash": compute_text_hash(context["config_raw"]),
        "scf_config": scf_config,
        "scf_settings": context["applied_scf"],
        "environment": collect_environment_snapshot(thread_count),
        "git": collect_git_metadata(os.getcwd()),
        "versions": {
            "ase": get_package_version("ase"),
            "pyscf": get_package_version("pyscf"),
            "dftd3": get_package_version("dftd3"),
            "dftd4": get_package_version("dftd4"),
        },
    }
    optimization_metadata["run_updated_at"] = datetime.now().isoformat()
    n_steps = {"value": 0}
    n_steps_source = None

    write_run_metadata(run_metadata_path, optimization_metadata)
    record_status_event(
        event_log_path,
        run_id,
        run_dir,
        "running",
        previous_status=context.get("previous_status"),
    )

    last_metadata_write = {
        "time": time.monotonic(),
        "step": 0,
    }
    checkpoint_base = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
                checkpoint_base = json.load(checkpoint_file)
        except (OSError, json.JSONDecodeError):
            checkpoint_base = {}

    optimizer_name = (optimizer_ase_dict.get("optimizer") or "").lower()
    if not optimizer_name:
        optimizer_name = "sella" if optimizer_mode == "transition_state" else "bfgs"
    optimizer_fmax = optimizer_ase_dict.get("fmax", 0.05)
    optimizer_steps = optimizer_ase_dict.get("steps", 200)

    input_xyz_name = os.path.basename(args.xyz_file) if args.xyz_file else None
    input_xyz_path = (
        resolve_run_path(run_dir, input_xyz_name) if input_xyz_name else None
    )
    output_xyz_setting = (
        optimizer_config.output_xyz if optimizer_config else None
    ) or "ase_optimized.xyz"
    output_xyz_path = resolve_run_path(run_dir, output_xyz_setting)
    pyscf_chkfile = None
    if scf_config and scf_config.get("chkfile"):
        pyscf_chkfile = resolve_run_path(run_dir, scf_config.get("chkfile"))

    def _write_checkpoint(
        atoms=None,
        atom_spec=None,
        step=None,
        status=None,
        error_message=None,
        scf_energy=None,
        scf_converged=None,
    ):
        checkpoint_payload = dict(checkpoint_base)
        if atom_spec is not None:
            checkpoint_payload["last_geometry"] = atom_spec
        elif atoms is not None:
            checkpoint_payload["last_geometry"] = _atoms_to_atom_spec(atoms)
        if step is not None:
            checkpoint_payload["last_step"] = step
        checkpoint_payload.update(
            {
                "optimizer": optimizer_name,
                "fmax": optimizer_fmax,
                "steps": optimizer_steps,
                "run_dir": run_dir,
                "calculation_mode": context.get("calculation_mode", "optimization"),
                "timestamp": datetime.now().isoformat(),
                "input_xyz": input_xyz_path,
                "output_xyz": output_xyz_path,
                "pyscf_chkfile": pyscf_chkfile,
            }
        )
        if scf_energy is not None:
            checkpoint_payload["last_scf_energy"] = scf_energy
        if scf_converged is not None:
            checkpoint_payload["last_scf_converged"] = scf_converged
        if status is not None:
            checkpoint_payload["status"] = status
        if error_message is not None:
            checkpoint_payload["error"] = error_message
        write_checkpoint(checkpoint_path, checkpoint_payload)

    def _step_callback(*_args, **_kwargs):
        n_steps["value"] += 1
        step_value = n_steps["value"]
        now = time.monotonic()
        should_write = (step_value - last_metadata_write["step"] >= 5) or (
            now - last_metadata_write["time"] >= 5.0
        )
        if should_write:
            optimizer = _args[0] if _args else None
            atoms = getattr(optimizer, "atoms", None) if optimizer is not None else None
            optimization_metadata["n_steps"] = step_value
            optimization_metadata["n_steps_source"] = "ase"
            optimization_metadata["status"] = "running"
            optimization_metadata["run_updated_at"] = datetime.now().isoformat()
            write_run_metadata(run_metadata_path, optimization_metadata)
            _write_checkpoint(atoms=atoms, step=step_value, status="running")
            last_metadata_write["time"] = now
            last_metadata_write["step"] = step_value

    try:
        if args.xyz_file and input_xyz_path:
            if os.path.abspath(args.xyz_file) != os.path.abspath(input_xyz_path):
                shutil.copy2(args.xyz_file, input_xyz_path)
        ensure_parent_dir(output_xyz_path)
        n_steps_value = _run_ase_optimizer(
            input_xyz_path,
            output_xyz_path,
            run_dir,
            charge,
            spin,
            multiplicity,
            basis,
            xc,
            scf_config,
            solvent_model.lower() if solvent_model else None,
            solvent_name,
            context["eps"],
            dispersion_model,
            verbose,
            memory_mb,
            optimizer_ase_dict,
            optimizer_mode,
            context["constraints"],
            step_callback=_step_callback,
        )
        optimized_atom_spec, _, _, _ = load_xyz(output_xyz_path)
        mol_optimized = gto.M(
            atom=optimized_atom_spec,
            basis=basis,
            charge=charge,
            spin=spin,
        )
        if memory_mb:
            mol_optimized.max_memory = memory_mb
        if n_steps_value is not None:
            n_steps["value"] = n_steps_value
        n_steps_source = "ase"
    except Exception as exc:
        logging.exception("Geometry optimization failed.")
        n_steps_value = n_steps["value"] if n_steps_source else None
        _write_checkpoint(status="failed", error_message=str(exc))
        optimization_metadata["status"] = "failed"
        optimization_metadata["run_ended_at"] = datetime.now().isoformat()
        optimization_metadata["error"] = str(exc)
        optimization_metadata["traceback"] = traceback.format_exc()
        optimization_metadata["last_geometry_source"] = "mf.mol"
        optimization_metadata["n_steps"] = n_steps_value
        optimization_metadata["n_steps_source"] = n_steps_source
        elapsed_seconds = time.perf_counter() - run_start
        optimization_metadata["summary"] = build_run_summary(
            mf,
            getattr(mf, "mol", mol),
            elapsed_seconds,
            completed=False,
            n_steps=n_steps_value,
            final_sp_energy=None,
            final_sp_converged=None,
            final_sp_cycles=None,
        )
        optimization_metadata["summary"]["memory_limit_enforced"] = memory_limit_enforced
        write_run_metadata(run_metadata_path, optimization_metadata)
        record_status_event(
            event_log_path,
            run_id,
            run_dir,
            "failed",
            previous_status="running",
            details={"error": str(exc)},
        )
        queue_update_fn("failed", exit_code=1)
        raise

    logging.info("Optimization finished.")
    logging.info("Optimized geometry (in Angstrom):")
    logging.info("%s", mol_optimized.tostring(format="xyz"))
    write_optimized_xyz(optimized_xyz_path, mol_optimized)
    ensure_stream_newlines()
    final_sp_energy = None
    final_sp_converged = None
    final_sp_cycles = None
    last_scf_energy = None
    last_scf_converged = None
    optimization_metadata["status"] = "completed"
    optimization_metadata["run_ended_at"] = datetime.now().isoformat()
    _write_checkpoint(
        atom_spec=optimized_atom_spec,
        step=n_steps_value,
        status="completed",
    )
    elapsed_seconds = time.perf_counter() - run_start
    n_steps_value = n_steps["value"] if n_steps_source else None
    imaginary_count = None
    frequency_payload = None
    if frequency_enabled:
        logging.info("Calculating harmonic frequencies for optimized geometry...")
        try:
            frequency_result = compute_frequencies(
                mol_optimized,
                context["sp_basis"],
                context["sp_xc"],
                context["sp_scf_config"],
                context["sp_solvent_model"] if context["sp_solvent_name"] else None,
                context["sp_solvent_name"],
                context["sp_eps"],
                context["freq_dispersion_model"],
                context["freq_dispersion_mode"],
                context["thermo"],
                verbose,
                memory_mb,
                run_dir=run_dir,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
            )
            last_scf_energy = frequency_result.get("energy")
            last_scf_converged = frequency_result.get("converged")
            imaginary_count = frequency_result.get("imaginary_count")
            imaginary_check = frequency_result.get("imaginary_check") or {}
            imaginary_status = imaginary_check.get("status")
            imaginary_message = imaginary_check.get("message")
            if imaginary_message:
                if imaginary_status == "one_imaginary":
                    logging.info("Imaginary frequency check: %s", imaginary_message)
                else:
                    logging.warning("Imaginary frequency check: %s", imaginary_message)
            frequency_payload = {
                "status": "completed",
                "output_file": frequency_output_path,
                "units": _frequency_units(),
                "versions": _frequency_versions(),
                "basis": context["sp_basis"],
                "xc": context["sp_xc"],
                "scf": context["sp_scf_config"],
                "solvent": context["sp_solvent_name"],
                "solvent_model": context["sp_solvent_model"]
                if context["sp_solvent_name"]
                else None,
                "solvent_eps": context["sp_eps"],
                "dispersion": context["freq_dispersion_model"],
                "dispersion_mode": context["freq_dispersion_mode"],
                "thermochemistry": _thermochemistry_payload(
                    context["thermo"], frequency_result.get("thermochemistry")
                ),
                "results": frequency_result,
            }
            with open(frequency_output_path, "w", encoding="utf-8") as handle:
                json.dump(frequency_payload, handle, indent=2)
            optimization_metadata["frequency"] = frequency_payload
        except Exception as exc:
            logging.exception("Frequency calculation failed.")
            failure_reason = str(exc) or "Frequency calculation failed."
            frequency_payload = {
                "status": "failed",
                "output_file": frequency_output_path,
                "reason": failure_reason,
                "units": _frequency_units(),
                "versions": _frequency_versions(),
                "basis": context["sp_basis"],
                "xc": context["sp_xc"],
                "scf": context["sp_scf_config"],
                "solvent": context["sp_solvent_name"],
                "solvent_model": context["sp_solvent_model"]
                if context["sp_solvent_name"]
                else None,
                "solvent_eps": context["sp_eps"],
                "dispersion": context["freq_dispersion_model"],
                "dispersion_mode": context["freq_dispersion_mode"],
                "thermochemistry": _thermochemistry_payload(context["thermo"], None),
                "results": None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            with open(frequency_output_path, "w", encoding="utf-8") as handle:
                json.dump(frequency_payload, handle, indent=2)
            optimization_metadata["frequency"] = frequency_payload
    else:
        frequency_payload = {
            "status": "skipped",
            "output_file": frequency_output_path,
            "reason": "Frequency calculation disabled.",
            "units": _frequency_units(),
            "versions": _frequency_versions(),
            "thermochemistry": _thermochemistry_payload(context["thermo"], None),
            "results": None,
        }
        with open(frequency_output_path, "w", encoding="utf-8") as handle:
            json.dump(frequency_payload, handle, indent=2)
        optimization_metadata["frequency"] = frequency_payload
    irc_status = "skipped"
    irc_skip_reason = None
    if irc_enabled:
        expected_imaginary = 1 if optimizer_mode == "transition_state" else 0
        if frequency_enabled and imaginary_count is not None:
            if imaginary_count != expected_imaginary:
                irc_skip_reason = (
                    "Imaginary frequency count does not match expected "
                    f"{expected_imaginary}."
                )
                logging.warning("Skipping IRC: %s", irc_skip_reason)
            else:
                irc_status = "pending"
        else:
            irc_status = "pending"
            if optimizer_mode == "transition_state" and not frequency_enabled:
                logging.warning(
                    "IRC requested without frequency check; proceeding without "
                    "imaginary mode validation."
                )
    else:
        irc_skip_reason = "IRC calculation disabled."
    run_single_point = False
    sp_status = "skipped"
    sp_skip_reason = None
    if single_point_enabled:
        if frequency_enabled:
            expected_imaginary = 1 if optimizer_mode == "transition_state" else 0
            if imaginary_count is None:
                logging.warning(
                    "Skipping single-point calculation because imaginary frequency "
                    "count is unavailable."
                )
                sp_skip_reason = "Imaginary frequency count unavailable."
            elif imaginary_count == expected_imaginary:
                run_single_point = True
            else:
                logging.warning(
                    "Skipping single-point calculation because imaginary frequency "
                    "count %s does not match expected %s.",
                    imaginary_count,
                    expected_imaginary,
                )
                sp_skip_reason = (
                    "Imaginary frequency count does not match expected "
                    f"{expected_imaginary}."
                )
        else:
            run_single_point = True
    else:
        logging.info("Skipping single-point energy calculation (disabled).")
        sp_skip_reason = "Single-point calculation disabled."

    if run_single_point:
        sp_status = "executed"
        sp_skip_reason = None

    optimization_metadata["single_point"]["status"] = sp_status
    optimization_metadata["single_point"]["skip_reason"] = sp_skip_reason
    optimization_metadata["irc"] = {
        "status": irc_status,
        "skip_reason": irc_skip_reason,
        "output_file": irc_output_path,
    }
    if frequency_payload is not None:
        frequency_payload["single_point"] = {
            "status": sp_status,
            "skip_reason": sp_skip_reason,
        }
        with open(frequency_output_path, "w", encoding="utf-8") as handle:
            json.dump(frequency_payload, handle, indent=2)

    try:
        if irc_status == "pending":
            logging.info("Running IRC for optimized geometry...")
            irc_steps = 10
            irc_step_size = 0.05
            irc_force_threshold = 0.01
            if irc_config:
                if irc_config.steps is not None:
                    irc_steps = irc_config.steps
                if irc_config.step_size is not None:
                    irc_step_size = irc_config.step_size
                if irc_config.force_threshold is not None:
                    irc_force_threshold = irc_config.force_threshold
            try:
                mode_result = compute_imaginary_mode(
                    mol_optimized,
                    context["sp_basis"],
                    context["sp_xc"],
                    context["sp_scf_config"],
                    context["sp_solvent_model"] if context["sp_solvent_name"] else None,
                    context["sp_solvent_name"],
                    context["sp_eps"],
                    verbose,
                    memory_mb,
                    run_dir=run_dir,
                    optimizer_mode=optimizer_mode,
                    multiplicity=multiplicity,
                )
                if mode_result.get("eigenvalue", 0.0) >= 0:
                    logging.warning(
                        "IRC mode eigenvalue is non-negative (%.6f); "
                        "structure may not be a first-order saddle point.",
                        mode_result.get("eigenvalue", 0.0),
                    )
                irc_result = _run_ase_irc(
                    output_xyz_path,
                    run_dir,
                    charge,
                    spin,
                    multiplicity,
                    context["sp_basis"],
                    context["sp_xc"],
                    context["sp_scf_config"],
                    context["sp_solvent_model"] if context["sp_solvent_name"] else None,
                    context["sp_solvent_name"],
                    context["sp_eps"],
                    context["sp_dispersion_model"],
                    verbose,
                    memory_mb,
                    optimizer_ase_dict,
                    optimizer_mode,
                    context["constraints"],
                    mode_result["mode"],
                    irc_steps,
                    irc_step_size,
                    irc_force_threshold,
                )
                irc_payload = {
                    "status": "completed",
                    "output_file": irc_output_path,
                    "forward_xyz": irc_result.get("forward_xyz"),
                    "reverse_xyz": irc_result.get("reverse_xyz"),
                    "steps": irc_steps,
                    "step_size": irc_step_size,
                    "force_threshold": irc_force_threshold,
                    "mode_eigenvalue": mode_result.get("eigenvalue"),
                    "profile": irc_result.get("profile", []),
                }
                irc_status = "executed"
            except Exception as exc:
                logging.exception("IRC calculation failed.")
                irc_payload = {
                    "status": "failed",
                    "output_file": irc_output_path,
                    "steps": irc_steps,
                    "step_size": irc_step_size,
                    "force_threshold": irc_force_threshold,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                irc_status = "failed"
            with open(irc_output_path, "w", encoding="utf-8") as handle:
                json.dump(irc_payload, handle, indent=2)
            optimization_metadata["irc"] = irc_payload
        elif irc_status == "skipped":
            logging.info("Skipping IRC calculation.")
        if run_single_point:
            logging.info("Calculating single-point energy for optimized geometry...")
            sp_result = compute_single_point_energy(
                mol_optimized,
                context["sp_basis"],
                context["sp_xc"],
                context["sp_scf_config"],
                context["sp_solvent_model"] if context["sp_solvent_name"] else None,
                context["sp_solvent_name"],
                context["sp_eps"],
                context["freq_dispersion_model"],
                verbose,
                memory_mb,
                run_dir=run_dir,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
            )
            final_sp_energy = sp_result["energy"]
            final_sp_converged = sp_result["converged"]
            final_sp_cycles = sp_result["cycles"]
            last_scf_energy = final_sp_energy
            last_scf_converged = final_sp_converged
            optimization_metadata["single_point"]["dispersion_info"] = sp_result.get(
                "dispersion"
            )
            if final_sp_cycles is None:
                final_sp_cycles = parse_single_point_cycle_count(log_path)
        elif single_point_enabled:
            logging.info("Skipping single-point energy calculation.")
    except Exception:
        logging.exception("Post-optimization calculations failed.")
        if run_single_point:
            final_sp_energy = None
            final_sp_converged = None
            final_sp_cycles = parse_single_point_cycle_count(log_path)
    optimization_metadata["n_steps"] = n_steps_value
    optimization_metadata["n_steps_source"] = n_steps_source
    optimization_metadata["summary"] = build_run_summary(
        mf,
        mol_optimized,
        elapsed_seconds,
        completed=True,
        n_steps=n_steps_value,
        final_sp_energy=final_sp_energy,
        final_sp_converged=final_sp_converged,
        final_sp_cycles=final_sp_cycles,
    )
    optimization_metadata["summary"]["memory_limit_enforced"] = memory_limit_enforced
    _write_checkpoint(
        scf_energy=last_scf_energy,
        scf_converged=last_scf_converged,
    )
    write_run_metadata(run_metadata_path, optimization_metadata)
    record_status_event(
        event_log_path,
        run_id,
        run_dir,
        "completed",
        previous_status="running",
    )
    queue_update_fn("completed", exit_code=0)

def run(args, config: RunConfig, config_raw, config_source_path, run_in_background):
    context = prepare_run_context(args, config, config_raw)
    calculation_mode = context["calculation_mode"]
    config_dict = context["config_dict"]
    basis = context["basis"]
    xc = context["xc"]
    solvent_name = context["solvent_name"]
    solvent_model = context["solvent_model"]
    dispersion_model = context["dispersion_model"]
    optimizer_mode = context["optimizer_mode"]
    scan_mode = context["scan_mode"]
    solvent_map_path = context["solvent_map_path"]
    single_point_config = context["single_point_config"]
    config_raw = context["config_raw"]
    thread_count = context["thread_count"]
    memory_gb = context["memory_gb"]
    verbose = context["verbose"]
    run_dir = context["run_dir"]
    log_path = context["log_path"]
    scf_config = context["scf_config"]
    run_metadata_path = context["run_metadata_path"]
    frequency_output_path = context["frequency_output_path"]
    event_log_path = context["event_log_path"]
    run_id = context["run_id"]
    if run_in_background:
        _enqueue_background_run(args, context)
        return
    setup_logging(
        log_path,
        verbose,
        run_id=run_id,
        event_log_path=event_log_path,
    )
    if config_source_path is not None:
        logging.info("Loaded config file: %s", config_source_path)
    queue_tracking = False
    if not run_in_background and not args.no_background:
        started_at = datetime.now().isoformat()
        foreground_entry = {
            "status": "running",
            "run_directory": context["run_dir"],
            "run_id": context["run_id"],
            "xyz_file": args.xyz_file,
            "config_file": args.config,
            "solvent_map": args.solvent_map,
            "run_metadata_file": context["run_metadata_path"],
            "log_file": context["log_path"],
            "event_log_file": context["event_log_path"],
            "started_at": started_at,
            "priority": args.queue_priority,
            "max_runtime_seconds": args.queue_max_runtime,
            "retry_count": 0,
        }
        register_foreground_run(
            foreground_entry,
            DEFAULT_QUEUE_PATH,
            DEFAULT_QUEUE_LOCK_PATH,
        )
        queue_tracking = True

    def _update_foreground_queue(status, exit_code=None):
        if queue_tracking:
            update_queue_status(
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
                context["run_id"],
                status,
                exit_code=exit_code,
            )

    try:
        thread_status = apply_thread_settings(context["thread_count"])
        openmp_available = thread_status.get("openmp_available")
        effective_threads = thread_status.get("effective_threads")
        enforce_os_memory_limit = bool(config.enforce_os_memory_limit)
        memory_mb, memory_limit_status = apply_memory_limit(
            context["memory_gb"], enforce_os_memory_limit
        )
        memory_limit_enforced = bool(memory_limit_status and memory_limit_status.get("applied"))

        molecule_context = build_molecule_context(args, context, memory_mb)
        mol = molecule_context["mol"]
        mf = molecule_context["mf"]
        charge = molecule_context["charge"]
        spin = molecule_context["spin"]
        multiplicity = molecule_context["multiplicity"]

        calculation_label = {
            "optimization": "Geometry optimization",
            "single_point": "Single-point",
            "frequency": "Frequency",
            "irc": "IRC",
            "scan": "Scan",
        }[calculation_mode]
        dispersion_model = _normalize_dispersion_settings(
            calculation_label,
            xc,
            dispersion_model,
            allow_dispersion=True,
        )

        solvent_name, solvent_model = _normalize_solvent_settings(
            calculation_label,
            solvent_name,
            solvent_model,
        )

        eps = None
        if solvent_name:
            solvent_key = solvent_name.lower()
            if not _is_vacuum_solvent(solvent_key):
                solvent_model_lower = solvent_model.lower() if solvent_model else None
                if solvent_model_lower == "pcm":
                    solvent_map = load_solvent_map(solvent_map_path)
                    eps = solvent_map.get(solvent_key)
                    if eps is None:
                        available = ", ".join(sorted(solvent_map.keys()))
                        raise ValueError(
                            "Solvent '{name}' not found in solvent map {path}. "
                            "Available solvents: {available}.".format(
                                name=solvent_name,
                                path=solvent_map_path,
                                available=available,
                            )
                        )
                mf = apply_solvent_model(
                    mf,
                    solvent_model_lower,
                    solvent_key,
                    eps,
                )
        dispersion_info = None
        context["eps"] = eps
        context["dispersion_info"] = dispersion_info
        if verbose:
            mf.verbose = 4
        applied_scf = scf_config or None
        if calculation_mode == "optimization" or (
            calculation_mode == "scan" and scan_mode == "optimization"
        ):
            applied_scf = apply_scf_settings(mf, scf_config)
        context["applied_scf"] = applied_scf

        sp_basis = (
            single_point_config.basis if single_point_config and single_point_config.basis else basis
        )
        sp_xc = normalize_xc_functional(
            single_point_config.xc if single_point_config and single_point_config.xc else xc
        )
        sp_scf_config = (
            single_point_config.scf.to_dict()
            if single_point_config and single_point_config.scf
            else scf_config
        )
        sp_chkfile = _resolve_scf_chkfile(sp_scf_config, run_dir)
        if context.get("resume_dir") and sp_chkfile and sp_chkfile != context.get("pyscf_chkfile"):
            _warn_missing_chkfile("Resume mode (single-point):", sp_chkfile)
        sp_solvent_name = (
            single_point_config.solvent
            if single_point_config and single_point_config.solvent
            else solvent_name
        )
        sp_solvent_model = (
            single_point_config.solvent_model
            if single_point_config and single_point_config.solvent_model
            else solvent_model
        )
        if single_point_config and "dispersion" in single_point_config.raw:
            sp_dispersion_model = single_point_config.dispersion
        else:
            sp_dispersion_model = dispersion_model
        sp_solvent_map_path = (
            single_point_config.solvent_map
            if single_point_config and single_point_config.solvent_map
            else solvent_map_path
        )
        if not sp_basis:
            raise ValueError(
                "Single-point config must define 'basis' or fall back to base 'basis'."
            )
        if sp_xc is None:
            raise ValueError("Single-point config must define 'xc' or fall back to base 'xc'.")
        if sp_solvent_model and not sp_solvent_name:
            raise ValueError(
                "Single-point config must define 'solvent' when 'solvent_model' is set."
            )
        if sp_solvent_name:
            if not sp_solvent_model and not _is_vacuum_solvent(sp_solvent_name):
                raise ValueError(
                    "Single-point config must define 'solvent_model' when 'solvent' is set."
                )
            if sp_solvent_model and sp_solvent_model.lower() not in ("pcm", "smd"):
                raise ValueError("Single-point config 'solvent_model' must be one of: pcm, smd.")

        sp_eps = None
        sp_solvent_key = None
        if sp_solvent_name:
            sp_solvent_key = sp_solvent_name.lower()
            if not _is_vacuum_solvent(sp_solvent_key):
                sp_solvent_model = sp_solvent_model.lower() if sp_solvent_model else None
                if sp_solvent_model == "pcm":
                    sp_solvent_map = load_solvent_map(sp_solvent_map_path)
                    sp_eps = sp_solvent_map.get(sp_solvent_key)
                    if sp_eps is None:
                        available = ", ".join(sorted(sp_solvent_map.keys()))
                        raise ValueError(
                            "Single-point solvent '{name}' not found in solvent map {path}. "
                            "Available solvents: {available}.".format(
                                name=sp_solvent_name,
                                path=sp_solvent_map_path,
                                available=available,
                            )
                        )
        sp_dispersion_model = _normalize_dispersion_settings(
            "Single-point",
            sp_xc,
            sp_dispersion_model,
            allow_dispersion=True,
        )

        sp_solvent_name, sp_solvent_model = _normalize_solvent_settings(
            "Single-point",
            sp_solvent_name,
            sp_solvent_model,
        )
        context["sp_basis"] = sp_basis
        context["sp_xc"] = sp_xc
        context["sp_scf_config"] = sp_scf_config
        context["sp_solvent_name"] = sp_solvent_name
        context["sp_solvent_model"] = sp_solvent_model
        context["sp_dispersion_model"] = sp_dispersion_model
        context["sp_solvent_map_path"] = sp_solvent_map_path
        context["sp_eps"] = sp_eps

        frequency_config = config.frequency
        freq_dispersion_mode = _normalize_frequency_dispersion_mode(
            frequency_config.dispersion if frequency_config else None
        )
        if frequency_config and "dispersion_model" in frequency_config.raw:
            freq_dispersion_raw = frequency_config.dispersion_model
        else:
            freq_dispersion_raw = sp_dispersion_model

        freq_dispersion_model = _normalize_dispersion_settings(
            "Frequency",
            sp_xc,
            freq_dispersion_raw,
            allow_dispersion=True,
        )
        context["freq_dispersion_mode"] = freq_dispersion_mode
        context["freq_dispersion_model"] = freq_dispersion_model

        if calculation_mode == "scan":
            run_scan_stage(
                args,
                context,
                molecule_context,
                memory_mb,
                memory_limit_status,
                memory_limit_enforced,
                openmp_available,
                effective_threads,
                _update_foreground_queue,
            )
            return
        if calculation_mode != "optimization":
            calc_basis = sp_basis
            calc_xc = sp_xc
            calc_scf_config = sp_scf_config
            calc_solvent_name = sp_solvent_name
            calc_solvent_model = sp_solvent_model
            calc_solvent_map_path = sp_solvent_map_path
            calc_eps = sp_eps
            calc_dispersion_model = (
                freq_dispersion_model if calculation_mode == "frequency" else sp_dispersion_model
            )
            calc_ks_type = select_ks_type(
                mol=mol,
                scf_config=calc_scf_config,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
            )
            logging.info(
                "Running capability check for %s calculation (SCF%s)...",
                "single-point"
                if calculation_mode == "single_point"
                else "frequency"
                if calculation_mode == "frequency"
                else "IRC",
                " + Hessian" if calculation_mode in ("frequency", "irc") else "",
            )
            run_capability_check(
                mol,
                calc_basis,
                calc_xc,
                calc_scf_config,
                calc_solvent_model if calc_solvent_name else None,
                calc_solvent_name,
                calc_eps,
                calc_dispersion_model if calculation_mode == "frequency" else None,
                freq_dispersion_mode if calculation_mode == "frequency" else "none",
                require_hessian=calculation_mode in ("frequency", "irc"),
                verbose=verbose,
                memory_mb=memory_mb,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
            )
            logging.info("Run ID: %s", run_id)
            logging.info("Run directory: %s", run_dir)
            if thread_count:
                logging.info("Using threads: %s", thread_count)
                if openmp_available is False:
                    effective_display = (
                        str(effective_threads) if effective_threads is not None else "unknown"
                    )
                    logging.warning(
                        "OpenMP appears unavailable; requested threads may have no effect "
                        "(effective threads: %s).",
                        effective_display,
                    )
            if memory_gb:
                logging.info("Memory target: %s GB (PySCF max_memory)", memory_gb)
                if memory_limit_status:
                    if memory_limit_status["applied"]:
                        limit_gb = memory_limit_status["limit_bytes"] / (1024 ** 3)
                        limit_name = memory_limit_status["limit_name"] or "unknown"
                        logging.info(
                            "OS hard memory limit: applied at %.2f GB (%s)",
                            limit_gb,
                            limit_name,
                        )
                    else:
                        log_fn = logging.warning
                        if memory_limit_status["reason"] == "disabled by config":
                            log_fn = logging.info
                        log_fn(
                            "OS hard memory limit: not applied (%s).",
                            memory_limit_status["reason"],
                        )
            logging.info("Verbose logging: %s", "enabled" if verbose else "disabled")
            logging.info("Log file: %s", log_path)
            if event_log_path:
                logging.info("Event log file: %s", event_log_path)
            if calc_scf_config:
                logging.info("SCF settings: %s", calc_scf_config)
            if calc_dispersion_model:
                logging.info("Dispersion correction: %s", calc_dispersion_model)
            if calculation_mode == "frequency":
                logging.info("Frequency dispersion mode: %s", freq_dispersion_mode)
            if calc_xc != xc:
                logging.info("XC override: %s", calc_xc)
            if calc_basis != basis:
                logging.info("Basis override: %s", calc_basis)
            if calc_scf_config != scf_config:
                logging.info("SCF override: %s", calc_scf_config)
            if calc_solvent_name != solvent_name or calc_solvent_model != solvent_model:
                logging.info(
                    "Solvent override: %s (%s)",
                    calc_solvent_name,
                    calc_solvent_model,
                )

            run_start = time.perf_counter()
            calculation_metadata = {
                "status": "running",
                "run_directory": run_dir,
                "run_started_at": datetime.now().isoformat(),
                "run_id": run_id,
                "attempt": context["attempt"],
                "run_id_history": context["run_id_history"],
                "pid": os.getpid(),
                "xyz_file": args.xyz_file,
                "xyz_file_hash": compute_file_hash(args.xyz_file),
                "basis": calc_basis,
                "xc": calc_xc,
                "solvent": calc_solvent_name,
                "solvent_model": calc_solvent_model if calc_solvent_name else None,
                "solvent_eps": calc_eps,
                "solvent_map": calc_solvent_map_path,
                "dispersion": calc_dispersion_model,
                "frequency_dispersion_mode": freq_dispersion_mode
                if calculation_mode == "frequency"
                else None,
                "dispersion_info": dispersion_info,
                "single_point": {
                    "basis": calc_basis,
                    "xc": calc_xc,
                    "scf": calc_scf_config,
                    "solvent": calc_solvent_name,
                    "solvent_model": calc_solvent_model if calc_solvent_name else None,
                    "solvent_eps": calc_eps,
                    "solvent_map": calc_solvent_map_path,
                    "dispersion": calc_dispersion_model,
                    "frequency_dispersion_mode": freq_dispersion_mode
                    if calculation_mode == "frequency"
                    else None,
                },
                "single_point_enabled": calculation_mode == "single_point",
                "calculation_mode": calculation_mode,
                "charge": charge,
                "spin": spin,
                "multiplicity": multiplicity,
                "ks_type": calc_ks_type,
                "thread_count": thread_count,
                "effective_thread_count": effective_threads,
                "openmp_available": openmp_available,
                "memory_gb": memory_gb,
                "memory_mb": memory_mb,
                "memory_limit_status": memory_limit_status,
                "log_file": log_path,
                "event_log_file": event_log_path,
                "frequency_file": frequency_output_path,
                "irc_file": context["irc_output_path"],
                "run_metadata_file": run_metadata_path,
                "config_file": args.config,
                "config": config_dict,
                "config_raw": config_raw,
                "config_hash": compute_text_hash(config_raw),
                "scf_config": calc_scf_config,
                "scf_settings": calc_scf_config,
                "environment": collect_environment_snapshot(thread_count),
                "git": collect_git_metadata(os.getcwd()),
                "versions": {
                    "ase": get_package_version("ase"),
                    "pyscf": get_package_version("pyscf"),
                    "dftd3": get_package_version("dftd3"),
                    "dftd4": get_package_version("dftd4"),
                },
            }
            calculation_metadata["run_updated_at"] = datetime.now().isoformat()
            write_run_metadata(run_metadata_path, calculation_metadata)
            record_status_event(
                event_log_path,
                run_id,
                run_dir,
                "running",
                previous_status=None,
            )
            stage_context = {
                "mol": mol,
                "calc_basis": calc_basis,
                "calc_xc": calc_xc,
                "calc_scf_config": calc_scf_config,
                "calc_solvent_name": calc_solvent_name,
                "calc_solvent_model": calc_solvent_model if calc_solvent_name else None,
                "calc_eps": calc_eps,
                "calc_dispersion_model": calc_dispersion_model,
                "freq_dispersion_mode": freq_dispersion_mode,
                "thermo": context["thermo"],
                "verbose": verbose,
                "memory_mb": memory_mb,
                "optimizer_mode": optimizer_mode,
                "multiplicity": multiplicity,
                "run_start": run_start,
                "metadata": calculation_metadata,
                "memory_limit_enforced": memory_limit_enforced,
                "run_metadata_path": run_metadata_path,
                "event_log_path": event_log_path,
                "run_id": run_id,
                "run_dir": run_dir,
                "checkpoint_path": context["checkpoint_path"],
                "pyscf_chkfile": calc_scf_config.get("chkfile") if calc_scf_config else None,
                "frequency_output_path": frequency_output_path,
                "irc_output_path": context["irc_output_path"],
                "input_xyz": args.xyz_file,
                "charge": charge,
                "spin": spin,
                "optimizer_ase_config": context["optimizer_ase_dict"],
                "constraints": context["constraints"],
            }
            if calculation_mode == "single_point":
                run_single_point_stage(stage_context, _update_foreground_queue)
            elif calculation_mode == "frequency":
                run_frequency_stage(stage_context, _update_foreground_queue)
            else:
                irc_config = context["irc_config"]
                irc_steps = 10
                irc_step_size = 0.05
                irc_force_threshold = 0.01
                if irc_config:
                    if irc_config.steps is not None:
                        irc_steps = irc_config.steps
                    if irc_config.step_size is not None:
                        irc_step_size = irc_config.step_size
                    if irc_config.force_threshold is not None:
                        irc_force_threshold = irc_config.force_threshold
                mode_result = compute_imaginary_mode(
                    mol,
                    calc_basis,
                    calc_xc,
                    calc_scf_config,
                    calc_solvent_model if calc_solvent_name else None,
                    calc_solvent_name,
                    calc_eps,
                    verbose,
                    memory_mb,
                    run_dir=run_dir,
                    optimizer_mode=optimizer_mode,
                    multiplicity=multiplicity,
                )
                if mode_result.get("eigenvalue", 0.0) >= 0:
                    logging.warning(
                        "IRC mode eigenvalue is non-negative (%.6f); "
                        "structure may not be a first-order saddle point.",
                        mode_result.get("eigenvalue", 0.0),
                    )
                stage_context.update(
                    {
                        "mode_vector": mode_result["mode"],
                        "mode_eigenvalue": mode_result.get("eigenvalue"),
                        "irc_steps": irc_steps,
                        "irc_step_size": irc_step_size,
                        "irc_force_threshold": irc_force_threshold,
                    }
                )
                run_irc_stage(stage_context, _update_foreground_queue)
            return

        if calculation_mode == "optimization":
            run_optimization_stage(
                args,
                context,
                molecule_context,
                memory_mb,
                memory_limit_status,
                memory_limit_enforced,
                openmp_available,
                effective_threads,
                _update_foreground_queue,
            )
    except Exception:
        logging.exception("Run failed.")
        if queue_tracking:
            _update_foreground_queue("failed", exit_code=1)
        raise
    finally:
        ensure_stream_newlines()
