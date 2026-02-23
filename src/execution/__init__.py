import logging
import os
import time
from datetime import datetime

from env_compat import getenv_with_legacy
from run_queue import record_status_event, register_foreground_run, update_queue_status
from run_opt_config import (
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    RunConfig,
    load_solvent_map,
)
from run_opt_engine import (
    apply_density_fit_setting,
    apply_scf_settings,
    apply_solvent_model,
    normalize_xc_functional,
    select_ks_type,
)
from run_opt_logging import ensure_stream_newlines, setup_logging_context
from run_opt_metadata import (
    collect_git_metadata,
    compute_file_hash,
    compute_text_hash,
    get_package_version,
    write_run_metadata,
)
from run_opt_resources import (
    apply_memory_limit,
    apply_thread_settings,
    collect_environment_snapshot,
)
from .context import build_molecule_context, prepare_run_context
from .engine_adapter import WorkflowEngineAdapter
from .events import enqueue_background_run
from .stage_freq import run_frequency_stage
from .stage_irc import run_irc_stage
from .stage_opt import run_optimization_stage
from .stage_scan import run_scan_stage
from .stage_sp import run_single_point_stage
from .types import RunContext
from .utils import (
    _disable_smd_solvent_settings,
    _is_vacuum_solvent,
    _normalize_dispersion_settings,
    _normalize_frequency_dispersion_mode,
    _normalize_solvent_settings,
    _prepare_frequency_scf_config,
    _recommend_density_fit,
    _resolve_d3_params,
    _resolve_scf_chkfile,
    _warn_missing_chkfile,
)


__all__ = ["run"]
DEFAULT_ENGINE_ADAPTER = WorkflowEngineAdapter()


def _build_foreground_entry(args, context, started_at):
    return {
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


def _register_foreground_tracking(args, context, run_in_background):
    if run_in_background or args.no_background:
        return False
    register_foreground_run(
        _build_foreground_entry(args, context, datetime.now().isoformat()),
        DEFAULT_QUEUE_PATH,
        DEFAULT_QUEUE_LOCK_PATH,
    )
    return True


def _update_foreground_queue_status(queue_tracking, run_id, status, exit_code=None):
    if not queue_tracking:
        return
    update_queue_status(
        DEFAULT_QUEUE_PATH,
        DEFAULT_QUEUE_LOCK_PATH,
        run_id,
        status,
        exit_code=exit_code,
    )


def _prepare_runtime_environment(config: RunConfig, context):
    thread_status = apply_thread_settings(context["thread_count"])
    openmp_available = thread_status.get("openmp_available")
    effective_threads = thread_status.get("effective_threads")
    enforce_os_memory_limit = bool(config.enforce_os_memory_limit)
    memory_mb, memory_limit_status = apply_memory_limit(
        context["memory_gb"], enforce_os_memory_limit
    )
    memory_limit_enforced = bool(memory_limit_status and memory_limit_status.get("applied"))
    return {
        "openmp_available": openmp_available,
        "effective_threads": effective_threads,
        "memory_mb": memory_mb,
        "memory_limit_status": memory_limit_status,
        "memory_limit_enforced": memory_limit_enforced,
    }


def _prepare_base_runtime_state(
    *,
    args,
    context,
    calculation_mode,
    scan_mode,
    xc,
    scf_config,
    solvent_name,
    solvent_model,
    solvent_map_path,
    dispersion_model,
    verbose,
    memory_mb,
):
    molecule_context = build_molecule_context(args, context, memory_mb)
    mol = molecule_context["mol"]
    mf = molecule_context["mf"]
    _recommend_density_fit(scf_config, mol, label="Base SCF")

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
    solvent_name, solvent_model = _disable_smd_solvent_settings(
        calculation_label,
        solvent_name,
        solvent_model,
    )
    eps = None
    apply_scf = calculation_mode == "optimization" or (
        calculation_mode == "scan" and scan_mode == "optimization"
    )
    if apply_scf:
        mf, _ = apply_density_fit_setting(mf, scf_config)
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
    if verbose:
        mf.verbose = 4
    applied_scf = scf_config or None
    if apply_scf:
        mf, applied_scf = apply_scf_settings(mf, scf_config, apply_density_fit=False)
    return {
        "molecule_context": molecule_context,
        "solvent_name": solvent_name,
        "solvent_model": solvent_model,
        "dispersion_model": dispersion_model,
        "eps": eps,
        "dispersion_info": dispersion_info,
        "applied_scf": applied_scf,
    }


def _prepare_single_point_runtime_state(
    *,
    context,
    basis,
    xc,
    scf_config,
    single_point_config,
    run_dir,
    solvent_name,
    solvent_model,
    solvent_map_path,
    dispersion_model,
    mol,
):
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
    if sp_scf_config is not scf_config:
        _recommend_density_fit(sp_scf_config, mol, label="Single-point SCF")
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
        raise ValueError("Single-point config must define 'basis' or fall back to base 'basis'.")
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
    sp_solvent_name, sp_solvent_model = _disable_smd_solvent_settings(
        "Single-point",
        sp_solvent_name,
        sp_solvent_model,
    )
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
    return {
        "sp_basis": sp_basis,
        "sp_xc": sp_xc,
        "sp_scf_config": sp_scf_config,
        "sp_chkfile": sp_chkfile,
        "sp_solvent_name": sp_solvent_name,
        "sp_solvent_model": sp_solvent_model,
        "sp_dispersion_model": sp_dispersion_model,
        "sp_solvent_map_path": sp_solvent_map_path,
        "sp_eps": sp_eps,
    }


def _prepare_frequency_runtime_state(
    *,
    config: RunConfig,
    sp_scf_config,
    run_dir,
    sp_dispersion_model,
    sp_xc,
):
    frequency_config = config.frequency
    frequency_use_chkfile = True
    if frequency_config and frequency_config.use_chkfile is not None:
        frequency_use_chkfile = bool(frequency_config.use_chkfile)
    freq_scf_config = _prepare_frequency_scf_config(
        sp_scf_config,
        run_dir,
        frequency_use_chkfile,
    )
    freq_dispersion_mode = _normalize_frequency_dispersion_mode(
        frequency_config.dispersion if frequency_config else None
    )
    freq_dispersion_step = None
    if frequency_config and frequency_config.dispersion_step is not None:
        freq_dispersion_step = frequency_config.dispersion_step
    if freq_dispersion_mode == "none":
        if frequency_config and (
            "dispersion_model" in frequency_config.raw
            or frequency_config.dispersion_step is not None
        ):
            logging.info("Frequency dispersion disabled; ignoring dispersion overrides.")
        freq_dispersion_model = None
        freq_dispersion_step = None
    else:
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
        if freq_dispersion_mode != "numerical":
            if freq_dispersion_step is not None:
                logging.info(
                    "Frequency dispersion mode '%s' ignores dispersion_step.",
                    freq_dispersion_mode,
                )
            freq_dispersion_step = None
    return {
        "frequency_use_chkfile": frequency_use_chkfile,
        "freq_scf_config": freq_scf_config,
        "freq_dispersion_mode": freq_dispersion_mode,
        "freq_dispersion_model": freq_dispersion_model,
        "freq_dispersion_step": freq_dispersion_step,
    }


def _prepare_runtime_state(
    args,
    config: RunConfig,
    context,
    calculation_mode,
    scan_mode,
    basis,
    xc,
    solvent_name,
    solvent_model,
    dispersion_model,
    single_point_config,
    solvent_map_path,
    scf_config,
    run_dir,
    verbose,
):
    runtime_environment = _prepare_runtime_environment(config, context)
    memory_mb = runtime_environment["memory_mb"]
    base_runtime = _prepare_base_runtime_state(
        args=args,
        context=context,
        calculation_mode=calculation_mode,
        scan_mode=scan_mode,
        xc=xc,
        scf_config=scf_config,
        solvent_name=solvent_name,
        solvent_model=solvent_model,
        solvent_map_path=solvent_map_path,
        dispersion_model=dispersion_model,
        verbose=verbose,
        memory_mb=memory_mb,
    )
    molecule_context = base_runtime["molecule_context"]
    sp_runtime = _prepare_single_point_runtime_state(
        context=context,
        basis=basis,
        xc=xc,
        scf_config=scf_config,
        single_point_config=single_point_config,
        run_dir=run_dir,
        solvent_name=base_runtime["solvent_name"],
        solvent_model=base_runtime["solvent_model"],
        solvent_map_path=solvent_map_path,
        dispersion_model=base_runtime["dispersion_model"],
        mol=molecule_context["mol"],
    )
    freq_runtime = _prepare_frequency_runtime_state(
        config=config,
        sp_scf_config=sp_runtime["sp_scf_config"],
        run_dir=run_dir,
        sp_dispersion_model=sp_runtime["sp_dispersion_model"],
        sp_xc=sp_runtime["sp_xc"],
    )
    context["solvent_name"] = base_runtime["solvent_name"]
    context["solvent_model"] = base_runtime["solvent_model"]
    context["eps"] = base_runtime["eps"]
    context["dispersion_info"] = base_runtime["dispersion_info"]
    context["applied_scf"] = base_runtime["applied_scf"]
    context["sp_basis"] = sp_runtime["sp_basis"]
    context["sp_xc"] = sp_runtime["sp_xc"]
    context["sp_scf_config"] = sp_runtime["sp_scf_config"]
    context["sp_chkfile"] = sp_runtime["sp_chkfile"]
    context["sp_solvent_name"] = sp_runtime["sp_solvent_name"]
    context["sp_solvent_model"] = sp_runtime["sp_solvent_model"]
    context["sp_dispersion_model"] = sp_runtime["sp_dispersion_model"]
    context["sp_solvent_map_path"] = sp_runtime["sp_solvent_map_path"]
    context["sp_eps"] = sp_runtime["sp_eps"]
    context["frequency_use_chkfile"] = freq_runtime["frequency_use_chkfile"]
    context["freq_scf_config"] = freq_runtime["freq_scf_config"]
    context["freq_dispersion_mode"] = freq_runtime["freq_dispersion_mode"]
    context["freq_dispersion_model"] = freq_runtime["freq_dispersion_model"]
    context["freq_dispersion_step"] = freq_runtime["freq_dispersion_step"]

    return {
        "molecule_context": molecule_context,
        "openmp_available": runtime_environment["openmp_available"],
        "effective_threads": runtime_environment["effective_threads"],
        "memory_mb": runtime_environment["memory_mb"],
        "memory_limit_status": runtime_environment["memory_limit_status"],
        "memory_limit_enforced": runtime_environment["memory_limit_enforced"],
        "solvent_name": base_runtime["solvent_name"],
        "solvent_model": base_runtime["solvent_model"],
        "dispersion_model": base_runtime["dispersion_model"],
        "dispersion_info": base_runtime["dispersion_info"],
        "sp_basis": sp_runtime["sp_basis"],
        "sp_xc": sp_runtime["sp_xc"],
        "sp_scf_config": sp_runtime["sp_scf_config"],
        "sp_solvent_name": sp_runtime["sp_solvent_name"],
        "sp_solvent_model": sp_runtime["sp_solvent_model"],
        "sp_dispersion_model": sp_runtime["sp_dispersion_model"],
        "sp_solvent_map_path": sp_runtime["sp_solvent_map_path"],
        "sp_eps": sp_runtime["sp_eps"],
        "freq_dispersion_mode": freq_runtime["freq_dispersion_mode"],
        "freq_dispersion_model": freq_runtime["freq_dispersion_model"],
    }


def _run_non_optimization_mode(
    args,
    context,
    runtime_state,
    calculation_mode,
    basis,
    xc,
    scf_config,
    optimizer_mode,
    config_dict,
    config_raw,
    thread_count,
    memory_gb,
    verbose,
    run_id,
    run_dir,
    log_path,
    event_log_path,
    run_metadata_path,
    qcschema_output_path,
    frequency_output_path,
    profiling_enabled,
    update_foreground_queue,
    engine_adapter: WorkflowEngineAdapter = DEFAULT_ENGINE_ADAPTER,
):
    molecule_context = runtime_state["molecule_context"]
    mol = molecule_context["mol"]
    charge = molecule_context["charge"]
    spin = molecule_context["spin"]
    multiplicity = molecule_context["multiplicity"]
    memory_mb = runtime_state["memory_mb"]
    memory_limit_status = runtime_state["memory_limit_status"]
    memory_limit_enforced = runtime_state["memory_limit_enforced"]
    openmp_available = runtime_state["openmp_available"]
    effective_threads = runtime_state["effective_threads"]
    solvent_name = runtime_state["solvent_name"]
    solvent_model = runtime_state["solvent_model"]
    dispersion_info = runtime_state["dispersion_info"]
    sp_basis = runtime_state["sp_basis"]
    sp_xc = runtime_state["sp_xc"]
    sp_scf_config = runtime_state["sp_scf_config"]
    sp_solvent_name = runtime_state["sp_solvent_name"]
    sp_solvent_model = runtime_state["sp_solvent_model"]
    sp_dispersion_model = runtime_state["sp_dispersion_model"]
    sp_solvent_map_path = runtime_state["sp_solvent_map_path"]
    sp_eps = runtime_state["sp_eps"]
    freq_dispersion_mode = runtime_state["freq_dispersion_mode"]
    freq_dispersion_model = runtime_state["freq_dispersion_model"]

    calc_basis = sp_basis
    calc_xc = sp_xc
    if calculation_mode == "frequency":
        calc_scf_config = context.get("freq_scf_config") or sp_scf_config
    else:
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
    skip_capability_check = bool(
        getenv_with_legacy(
            "PYSCF_AUTO_SKIP_CAPABILITY_CHECK",
            "DFTFLOW_SKIP_CAPABILITY_CHECK",
        )
    )
    if skip_capability_check:
        logging.warning("Skipping capability check (PYSCF_AUTO_SKIP_CAPABILITY_CHECK=1).")
    else:
        logging.info(
            "Running capability check for %s calculation (SCF%s)...",
            "single-point"
            if calculation_mode == "single_point"
            else "frequency"
            if calculation_mode == "frequency"
            else "IRC",
            " + Hessian" if calculation_mode in ("frequency", "irc") else "",
        )
        engine_adapter.run_capability_check(
            mol,
            calc_basis,
            calc_xc,
            calc_scf_config,
            calc_solvent_model if calc_solvent_name else None,
            calc_solvent_name,
            calc_eps,
            None
            if calculation_mode == "frequency" and freq_dispersion_mode == "none"
            else calc_dispersion_model,
            freq_dispersion_mode if calculation_mode == "frequency" else "none",
            dispersion_params=_resolve_d3_params(context.get("optimizer_ase_dict")),
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
            effective_display = str(effective_threads) if effective_threads is not None else "unknown"
            logging.warning(
                "OpenMP appears unavailable; requested threads may have no effect "
                "(effective threads: %s).",
                effective_display,
            )
    if memory_gb:
        logging.info("Memory target: %s GB (PySCF max_memory)", memory_gb)
        if memory_limit_status:
            if memory_limit_status["applied"]:
                limit_gb = memory_limit_status["limit_bytes"] / (1024**3)
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
        "frequency_dispersion_mode": freq_dispersion_mode if calculation_mode == "frequency" else None,
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
            "frequency_dispersion_mode": freq_dispersion_mode if calculation_mode == "frequency" else None,
        },
        "single_point_enabled": calculation_mode == "single_point"
        or bool(context.get("single_point_enabled")),
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
        "irc_profile_csv_file": context["irc_profile_csv_path"],
        "run_metadata_file": run_metadata_path,
        "qcschema_output_file": qcschema_output_path,
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
        "freq_dispersion_step": context.get("freq_dispersion_step"),
        "thermo": context["thermo"],
        "ts_quality": context.get("ts_quality"),
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
        "resume_dir": context["resume_dir"],
        "checkpoint_path": context["checkpoint_path"],
        "pyscf_chkfile": calc_scf_config.get("chkfile") if calc_scf_config else None,
        "frequency_output_path": frequency_output_path,
        "irc_output_path": context["irc_output_path"],
        "irc_profile_csv_path": context["irc_profile_csv_path"],
        "qcschema_output_path": qcschema_output_path,
        "input_xyz": args.xyz_file,
        "charge": charge,
        "spin": spin,
        "optimizer_ase_config": context["optimizer_ase_dict"],
        "constraints": context["constraints"],
        "profiling_enabled": profiling_enabled,
        "snapshot_interval_steps": context.get("snapshot_interval_steps"),
        "snapshot_mode": context.get("snapshot_mode"),
        "single_point_enabled": context.get("single_point_enabled"),
        "irc_enabled": context.get("irc_enabled"),
        "irc_config": context.get("irc_config"),
    }
    if calculation_mode == "single_point":
        run_single_point_stage(
            stage_context,
            update_foreground_queue,
            engine_adapter=engine_adapter,
        )
        return
    if calculation_mode == "frequency":
        run_frequency_stage(
            stage_context,
            update_foreground_queue,
            engine_adapter=engine_adapter,
        )
        return

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
    mode_result = engine_adapter.compute_imaginary_mode(
        mol,
        calc_basis,
        calc_xc,
        calc_scf_config,
        calc_solvent_model if calc_solvent_name else None,
        calc_solvent_name,
        calc_eps,
        verbose,
        memory_mb,
        dispersion=calc_dispersion_model,
        dispersion_hessian_step=context.get("freq_dispersion_step"),
        constraints=context["constraints"],
        dispersion_params=_resolve_d3_params(context.get("optimizer_ase_dict")),
        run_dir=run_dir,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        profiling_enabled=profiling_enabled,
        return_hessian=True,
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
            "mode_hessian": mode_result.get("hessian"),
            "mode_eigenvalue": mode_result.get("eigenvalue"),
            "mode_profiling": mode_result.get("profiling") if profiling_enabled else None,
            "irc_steps": irc_steps,
            "irc_step_size": irc_step_size,
            "irc_force_threshold": irc_force_threshold,
        }
    )
    run_irc_stage(
        stage_context,
        update_foreground_queue,
        engine_adapter=engine_adapter,
    )


def _create_foreground_queue_updater(context, queue_tracking):
    def _update_foreground_queue(status, exit_code=None):
        _update_foreground_queue_status(
            queue_tracking,
            context["run_id"],
            status,
            exit_code=exit_code,
        )

    return _update_foreground_queue


def _prepare_runtime_state_for_run(args, config: RunConfig, context):
    return _prepare_runtime_state(
        args,
        config,
        context,
        context["calculation_mode"],
        context["scan_mode"],
        context["basis"],
        context["xc"],
        context["solvent_name"],
        context["solvent_model"],
        context["dispersion_model"],
        context["single_point_config"],
        context["solvent_map_path"],
        context["scf_config"],
        context["run_dir"],
        context["verbose"],
    )


def _dispatch_stage_for_mode(
    args,
    context,
    runtime_state,
    update_foreground_queue,
    engine_adapter: WorkflowEngineAdapter = DEFAULT_ENGINE_ADAPTER,
):
    calculation_mode = context["calculation_mode"]
    molecule_context = runtime_state["molecule_context"]
    memory_mb = runtime_state["memory_mb"]
    memory_limit_status = runtime_state["memory_limit_status"]
    memory_limit_enforced = runtime_state["memory_limit_enforced"]
    openmp_available = runtime_state["openmp_available"]
    effective_threads = runtime_state["effective_threads"]

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
            update_foreground_queue,
            engine_adapter=engine_adapter,
        )
        return

    if calculation_mode != "optimization":
        _run_non_optimization_mode(
            args=args,
            context=context,
            runtime_state=runtime_state,
            calculation_mode=calculation_mode,
            basis=context["basis"],
            xc=context["xc"],
            scf_config=context["scf_config"],
            optimizer_mode=context["optimizer_mode"],
            config_dict=context["config_dict"],
            config_raw=context["config_raw"],
            thread_count=context["thread_count"],
            memory_gb=context["memory_gb"],
            verbose=context["verbose"],
            run_id=context["run_id"],
            run_dir=context["run_dir"],
            log_path=context["log_path"],
            event_log_path=context["event_log_path"],
            run_metadata_path=context["run_metadata_path"],
            qcschema_output_path=context["qcschema_output_path"],
            frequency_output_path=context["frequency_output_path"],
            profiling_enabled=bool(context.get("profiling_enabled")),
            update_foreground_queue=update_foreground_queue,
            engine_adapter=engine_adapter,
        )
        return

    run_optimization_stage(
        args,
        context,
        molecule_context,
        memory_mb,
        memory_limit_status,
        memory_limit_enforced,
        openmp_available,
        effective_threads,
        update_foreground_queue,
        engine_adapter=engine_adapter,
    )


def _run_foreground(args, config: RunConfig, context, config_source_path, run_in_background):
    with setup_logging_context(
        context["log_path"],
        context["verbose"],
        run_id=context["run_id"],
        event_log_path=context["event_log_path"],
    ):
        if config_source_path is not None:
            logging.info("Loaded config file: %s", config_source_path)
        queue_tracking = _register_foreground_tracking(args, context, run_in_background)
        update_foreground_queue = _create_foreground_queue_updater(context, queue_tracking)
        try:
            runtime_state = _prepare_runtime_state_for_run(args, config, context)
            _dispatch_stage_for_mode(args, context, runtime_state, update_foreground_queue)
        except Exception:
            logging.exception("Run failed.")
            if queue_tracking:
                update_foreground_queue("failed", exit_code=1)
            raise
        finally:
            ensure_stream_newlines()


def run(args, config: RunConfig, config_raw, config_source_path, run_in_background):
    context: RunContext = prepare_run_context(args, config, config_raw)
    if run_in_background:
        enqueue_background_run(args, context)
        return
    _run_foreground(args, config, context, config_source_path, run_in_background)
