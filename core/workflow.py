import importlib.util
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

from .ase_backend import _run_ase_optimizer
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
    DEFAULT_LOG_PATH,
    DEFAULT_OPTIMIZED_XYZ_PATH,
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_QUEUE_RUNNER_LOG_PATH,
    DEFAULT_RUN_METADATA_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
    DEFAULT_THREAD_COUNT,
    RunConfig,
    load_solvent_map,
)
from .run_opt_logging import ensure_stream_newlines, setup_logging
from .run_opt_metadata import (
    build_run_summary,
    collect_git_metadata,
    compute_file_hash,
    compute_text_hash,
    get_package_version,
    parse_single_point_cycle_count,
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
    raise ValueError(
        "Unsupported calculation mode '{value}'. Use 'optimization', "
        "'single_point', or 'frequency'.".format(value=mode_value)
    )


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

    def _check_import(module_name, hint):
        spec = importlib.util.find_spec(module_name)
        ok = spec is not None
        _record_check(module_name, ok, hint if not ok else None)
        return ok

    def _solvent_map_hint(error):
        if isinstance(error, FileNotFoundError):
            return (
                "Missing solvent map file. Provide --solvent-map or restore "
                f"{DEFAULT_SOLVENT_MAP_PATH}."
            )
        if isinstance(error, json.JSONDecodeError):
            return "Invalid JSON in solvent map. Fix the JSON syntax."
        return "Unable to read solvent map. Check file permissions and path."

    try:
        load_solvent_map(DEFAULT_SOLVENT_MAP_PATH)
        _record_check("solvent_map", True)
    except Exception as exc:
        _record_check("solvent_map", False, _solvent_map_hint(exc))

    checks = [
        ("pyscf", "Install with: pip install pyscf"),
        ("pyscf.dft", "Install with: pip install pyscf"),
        ("pyscf.gto", "Install with: pip install pyscf"),
        ("pyscf.hessian.thermo", "Install with: pip install pyscf"),
        ("dftd3", "Install with: pip install dftd3"),
        ("dftd4", "Install with: pip install dftd4"),
    ]
    for module_name, hint in checks:
        _check_import(module_name, hint)

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
    optimizer_mode = None
    if calculation_mode == "optimization":
        optimizer_mode = _normalize_optimizer_mode(
            optimizer_config.mode if optimizer_config else None
        )
    solvent_map_path = config.solvent_map or DEFAULT_SOLVENT_MAP_PATH
    single_point_config = config.single_point
    thermo_config = config.thermo
    frequency_enabled, single_point_enabled = _normalize_stage_flags(
        config, calculation_mode
    )
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
    run_dir = args.run_dir or create_run_directory()
    os.makedirs(run_dir, exist_ok=True)
    log_path = resolve_run_path(run_dir, config.log_file or DEFAULT_LOG_PATH)
    log_path = format_log_path(log_path)
    scf_config = config.scf.to_dict() if config.scf else {}
    optimized_xyz_path = resolve_run_path(
        run_dir, config.optimized_xyz_file or DEFAULT_OPTIMIZED_XYZ_PATH
    )
    run_metadata_path = resolve_run_path(
        run_dir, config.run_metadata_file or DEFAULT_RUN_METADATA_PATH
    )
    frequency_output_path = resolve_run_path(
        run_dir, config.frequency_file or DEFAULT_FREQUENCY_PATH
    )
    ensure_parent_dir(log_path)
    ensure_parent_dir(optimized_xyz_path)
    ensure_parent_dir(run_metadata_path)
    ensure_parent_dir(frequency_output_path)

    if args.interactive:
        config_used_path = resolve_run_path(run_dir, "config_used.json")
        ensure_parent_dir(config_used_path)
        with open(config_used_path, "w", encoding="utf-8") as config_used_file:
            config_used_file.write(config_raw)
        args.config = config_used_path

    run_id = args.run_id or str(uuid.uuid4())
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
        "event_log_path": event_log_path,
        "run_id": run_id,
    }


def _enqueue_background_run(args, context):
    queued_at = datetime.now().isoformat()
    queue_priority = args.queue_priority
    max_runtime_seconds = args.queue_max_runtime
    queued_metadata = {
        "status": "queued",
        "run_directory": context["run_dir"],
        "run_id": context["run_id"],
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
        previous_status=None,
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
    event_log_path = context["event_log_path"]
    run_id = context["run_id"]
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
        previous_status=None,
    )

    last_metadata_write = {
        "time": time.monotonic(),
        "step": 0,
    }

    def _step_callback(*_args, **_kwargs):
        n_steps["value"] += 1
        step_value = n_steps["value"]
        now = time.monotonic()
        should_write = (step_value - last_metadata_write["step"] >= 5) or (
            now - last_metadata_write["time"] >= 5.0
        )
        if should_write:
            optimization_metadata["n_steps"] = step_value
            optimization_metadata["n_steps_source"] = "ase"
            optimization_metadata["status"] = "running"
            optimization_metadata["run_updated_at"] = datetime.now().isoformat()
            write_run_metadata(run_metadata_path, optimization_metadata)
            last_metadata_write["time"] = now
            last_metadata_write["step"] = step_value

    try:
        input_xyz_name = os.path.basename(args.xyz_file)
        input_xyz_path = resolve_run_path(run_dir, input_xyz_name)
        if os.path.abspath(args.xyz_file) != os.path.abspath(input_xyz_path):
            shutil.copy2(args.xyz_file, input_xyz_path)
        output_xyz_setting = (
            optimizer_config.output_xyz if optimizer_config else None
        ) or "ase_optimized.xyz"
        output_xyz_path = resolve_run_path(run_dir, output_xyz_setting)
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
    optimization_metadata["status"] = "completed"
    optimization_metadata["run_ended_at"] = datetime.now().isoformat()
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
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
            )
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
    if frequency_payload is not None:
        frequency_payload["single_point"] = {
            "status": sp_status,
            "skip_reason": sp_skip_reason,
        }
        with open(frequency_output_path, "w", encoding="utf-8") as handle:
            json.dump(frequency_payload, handle, indent=2)

    try:
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
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
            )
            final_sp_energy = sp_result["energy"]
            final_sp_converged = sp_result["converged"]
            final_sp_cycles = sp_result["cycles"]
            optimization_metadata["single_point"]["dispersion_info"] = sp_result.get(
                "dispersion"
            )
            if final_sp_cycles is None:
                final_sp_cycles = parse_single_point_cycle_count(log_path)
        elif single_point_enabled:
            logging.info("Skipping single-point energy calculation.")
    except Exception:
        logging.exception("Single-point energy calculation failed.")
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
        if calculation_mode == "optimization":
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

        if calculation_mode != "optimization":
            calc_basis = sp_basis
            calc_xc = sp_xc
            calc_scf_config = sp_scf_config
            calc_solvent_name = sp_solvent_name
            calc_solvent_model = sp_solvent_model
            calc_solvent_map_path = sp_solvent_map_path
            calc_eps = sp_eps
            calc_dispersion_model = (
                sp_dispersion_model if calculation_mode == "single_point" else freq_dispersion_model
            )
            calc_ks_type = select_ks_type(
                mol=mol,
                scf_config=calc_scf_config,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
            )
            logging.info(
                "Running capability check for %s calculation (SCF%s)...",
                "single-point" if calculation_mode == "single_point" else "frequency",
                " + Hessian" if calculation_mode == "frequency" else "",
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
                require_hessian=calculation_mode == "frequency",
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
                "frequency_output_path": frequency_output_path,
            }
            if calculation_mode == "single_point":
                run_single_point_stage(stage_context, _update_foreground_queue)
            else:
                run_frequency_stage(stage_context, _update_foreground_queue)
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
