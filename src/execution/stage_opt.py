import json
import logging
import os
import shutil
import time
import traceback
from datetime import datetime
from typing import Any, cast

from env_compat import env_truthy
from run_opt_logging import ensure_stream_newlines
from run_opt_metadata import (
    build_run_summary,
    collect_git_metadata,
    compute_file_hash,
    compute_text_hash,
    format_xyz_comment,
    get_package_version,
    parse_single_point_cycle_count,
    write_checkpoint,
    write_optimized_xyz,
    write_xyz_snapshot,
)
from run_opt_resources import collect_environment_snapshot, ensure_parent_dir, resolve_run_path
from run_opt_utils import is_ts_quality_enforced as _is_ts_quality_enforced
from .engine_adapter import WorkflowEngineAdapter
from .metadata_recorder import RunMetadataRecorder
from .plugins import export_qcschema_result
from .types import MoleculeContext, OptimizationStageContext, RunContext
from .utils import (
    _atoms_to_atom_spec,
    _evaluate_irc_profile,
    _frequency_units,
    _frequency_versions,
    _resolve_d3_params,
    _seed_scf_checkpoint,
    _thermochemistry_payload,
)


DEFAULT_ENGINE_ADAPTER = WorkflowEngineAdapter()
DEFAULT_METADATA_RECORDER = RunMetadataRecorder()


def _build_optimization_stage_context(context: RunContext) -> OptimizationStageContext:
    required_keys = (
        "config_dict",
        "config_raw",
        "calculation_mode",
        "basis",
        "xc",
        "solvent_name",
        "solvent_model",
        "dispersion_model",
        "optimizer_config",
        "optimizer_ase_dict",
        "optimizer_mode",
        "constraints",
        "solvent_map_path",
        "ts_quality",
        "thermo",
        "frequency_enabled",
        "single_point_enabled",
        "thread_count",
        "memory_gb",
        "verbose",
        "profiling_enabled",
        "io_write_interval_steps",
        "io_write_interval_seconds",
        "snapshot_interval_steps",
        "snapshot_mode",
        "run_dir",
        "log_path",
        "scf_config",
        "optimized_xyz_path",
        "run_metadata_path",
        "qcschema_output_path",
        "frequency_output_path",
        "irc_output_path",
        "irc_profile_csv_path",
        "event_log_path",
        "run_id",
        "attempt",
        "run_id_history",
        "checkpoint_path",
        "solvent_map_path",
        "scf_config",
        "optimizer_config",
        "optimizer_ase_dict",
        "optimizer_mode",
        "constraints",
        "frequency_enabled",
        "single_point_enabled",
        "irc_enabled",
        "irc_config",
        "thermo",
        "sp_basis",
        "sp_xc",
        "sp_scf_config",
        "sp_solvent_name",
        "sp_solvent_model",
        "sp_dispersion_model",
        "sp_solvent_map_path",
        "sp_eps",
        "freq_dispersion_mode",
        "freq_dispersion_model",
        "pyscf_chkfile",
    )
    optional_keys = (
        "resume_dir",
        "previous_status",
        "dispersion_info",
        "applied_scf",
        "sp_chkfile",
        "eps",
        "freq_dispersion_step",
        "freq_scf_config",
        "frequency_use_chkfile",
    )
    narrowed: dict[str, Any] = {}
    missing = [key for key in required_keys if key not in context]
    if missing:
        raise KeyError(
            "Missing required optimization context keys: {keys}".format(
                keys=", ".join(sorted(missing))
            )
        )
    for key in required_keys:
        narrowed[key] = context[key]
    for key in optional_keys:
        if key in context:
            narrowed[key] = context[key]
    return cast(OptimizationStageContext, narrowed)


def _run_optimization_capability_checks(
    *,
    context: OptimizationStageContext,
    mol,
    basis,
    xc,
    scf_config,
    optimizer_mode,
    multiplicity,
    freq_scf_config,
    optimizer_ase_dict,
    engine_adapter: WorkflowEngineAdapter,
    verbose,
    memory_mb,
) -> None:
    skip_capability_check = env_truthy("PYSCF_AUTO_SKIP_CAPABILITY_CHECK")
    if skip_capability_check:
        logging.warning(
            "Skipping capability check (PYSCF_AUTO_SKIP_CAPABILITY_CHECK=1)."
        )
        return

    logging.info(
        "Running capability check for geometry optimization (SCF + gradient)..."
    )
    engine_adapter.run_capability_check(
        mol,
        basis,
        xc,
        scf_config,
        context["solvent_model"],
        context["solvent_name"],
        context["eps"],
        context["dispersion_model"],
        "none",
        dispersion_params=_resolve_d3_params(optimizer_ase_dict),
        require_hessian=False,
        verbose=verbose,
        memory_mb=memory_mb,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
    )
    if not context["frequency_enabled"]:
        return

    logging.info(
        "Running capability check for frequency calculation (SCF + gradient + Hessian)..."
    )
    engine_adapter.run_capability_check(
        mol,
        context["sp_basis"],
        context["sp_xc"],
        freq_scf_config,
        context["sp_solvent_model"] if context["sp_solvent_name"] else None,
        context["sp_solvent_name"],
        context["sp_eps"],
        None if context["freq_dispersion_mode"] == "none" else context["freq_dispersion_model"],
        context["freq_dispersion_mode"],
        dispersion_params=_resolve_d3_params(optimizer_ase_dict),
        require_hessian=True,
        verbose=verbose,
        memory_mb=memory_mb,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
    )


def _log_optimization_start(
    *,
    context: OptimizationStageContext,
    run_id,
    run_dir,
    optimizer_mode,
    thread_count,
    openmp_available,
    effective_threads,
    memory_gb,
    memory_limit_status,
    verbose,
    log_path,
    event_log_path,
    basis,
    xc,
    scf_config,
    solvent_name,
    solvent_model,
    dispersion_model,
) -> None:
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
    if context["frequency_enabled"]:
        logging.info("Frequency dispersion mode: %s", context["freq_dispersion_mode"])


def _build_optimization_metadata(
    *,
    args,
    context: OptimizationStageContext,
    run_dir,
    run_id,
    basis,
    xc,
    solvent_name,
    solvent_model,
    solvent_map_path,
    dispersion_model,
    frequency_enabled,
    single_point_enabled,
    irc_enabled,
    charge,
    spin,
    multiplicity,
    ks_type,
    thread_count,
    effective_threads,
    openmp_available,
    memory_gb,
    memory_mb,
    memory_limit_status,
    log_path,
    event_log_path,
    optimized_xyz_path,
    frequency_output_path,
    irc_output_path,
    run_metadata_path,
    scf_config,
) -> dict[str, Any]:
    optimization_metadata: dict[str, Any] = {
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
        "frequency_dispersion_step": context.get("freq_dispersion_step")
        if frequency_enabled
        else None,
        "optimizer": {
            "mode": context["optimizer_mode"],
            "output_xyz": context["optimizer_config"].output_xyz if context["optimizer_config"] else None,
            "ase": context["optimizer_ase_dict"] or None,
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
            "frequency_dispersion_step": context.get("freq_dispersion_step")
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
        "irc_profile_csv_file": context["irc_profile_csv_path"],
        "run_metadata_file": run_metadata_path,
        "qcschema_output_file": context.get("qcschema_output_path"),
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
    return optimization_metadata


def _run_frequency_for_optimized_geometry(
    *,
    context: OptimizationStageContext,
    engine_adapter: WorkflowEngineAdapter,
    mol_optimized,
    freq_scf_config,
    optimizer_ase_dict,
    optimizer_mode,
    multiplicity,
    run_dir,
    frequency_output_path,
    verbose,
    memory_mb,
    profiling_enabled,
    optimization_metadata: dict[str, Any],
) -> tuple[dict[str, Any], int | None, float | None, bool | None]:
    frequency_enabled = context["frequency_enabled"]
    frequency_payload: dict[str, Any]
    imaginary_count: int | None = None
    last_scf_energy: float | None = None
    last_scf_converged: bool | None = None

    if frequency_enabled:
        logging.info("Calculating harmonic frequencies for optimized geometry...")
        try:
            frequency_result = engine_adapter.compute_frequencies(
                mol_optimized,
                context["sp_basis"],
                context["sp_xc"],
                freq_scf_config,
                context["sp_solvent_model"] if context["sp_solvent_name"] else None,
                context["sp_solvent_name"],
                context["sp_eps"],
                context["freq_dispersion_model"],
                context["freq_dispersion_mode"],
                context.get("freq_dispersion_step"),
                _resolve_d3_params(optimizer_ase_dict),
                context["thermo"],
                verbose,
                memory_mb,
                context["constraints"],
                run_dir=run_dir,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
                ts_quality=context.get("ts_quality"),
                profiling_enabled=profiling_enabled,
            )
            last_scf_energy = frequency_result.get("energy")
            last_scf_converged = frequency_result.get("converged")
            imaginary_count = frequency_result.get("imaginary_count")
            imaginary_check = frequency_result.get("imaginary_check") or {}
            imaginary_status = imaginary_check.get("status")
            imaginary_message = imaginary_check.get("message")
            ts_quality_result = frequency_result.get("ts_quality") or {}
            ts_quality_status = ts_quality_result.get("status")
            ts_quality_message = ts_quality_result.get("message")
            if imaginary_message:
                if imaginary_status == "one_imaginary":
                    logging.info("Imaginary frequency check: %s", imaginary_message)
                else:
                    logging.warning("Imaginary frequency check: %s", imaginary_message)
            if ts_quality_message:
                if ts_quality_status in ("pass", "warn"):
                    logging.info("TS quality check: %s", ts_quality_message)
                else:
                    logging.warning("TS quality check: %s", ts_quality_message)
            frequency_payload = {
                "status": "completed",
                "output_file": frequency_output_path,
                "units": _frequency_units(),
                "versions": _frequency_versions(),
                "basis": context["sp_basis"],
                "xc": context["sp_xc"],
                "scf": freq_scf_config,
                "solvent": context["sp_solvent_name"],
                "solvent_model": context["sp_solvent_model"]
                if context["sp_solvent_name"]
                else None,
                "solvent_eps": context["sp_eps"],
                "dispersion": context["freq_dispersion_model"],
                "dispersion_mode": context["freq_dispersion_mode"],
                "dispersion_step": context.get("freq_dispersion_step"),
                "profiling": frequency_result.get("profiling"),
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
                "scf": freq_scf_config,
                "solvent": context["sp_solvent_name"],
                "solvent_model": context["sp_solvent_model"]
                if context["sp_solvent_name"]
                else None,
                "solvent_eps": context["sp_eps"],
                "dispersion": context["freq_dispersion_model"],
                "dispersion_mode": context["freq_dispersion_mode"],
                "dispersion_step": context.get("freq_dispersion_step"),
                "thermochemistry": _thermochemistry_payload(context["thermo"], None),
                "results": None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            with open(frequency_output_path, "w", encoding="utf-8") as handle:
                json.dump(frequency_payload, handle, indent=2)
            optimization_metadata["frequency"] = frequency_payload
        return frequency_payload, imaginary_count, last_scf_energy, last_scf_converged

    frequency_payload = {
        "status": "skipped",
        "output_file": frequency_output_path,
        "reason": "Frequency calculation disabled.",
        "units": _frequency_units(),
        "versions": _frequency_versions(),
        "dispersion_step": context.get("freq_dispersion_step"),
        "thermochemistry": _thermochemistry_payload(context["thermo"], None),
        "results": None,
    }
    with open(frequency_output_path, "w", encoding="utf-8") as handle:
        json.dump(frequency_payload, handle, indent=2)
    optimization_metadata["frequency"] = frequency_payload
    return frequency_payload, imaginary_count, last_scf_energy, last_scf_converged


def _determine_irc_and_single_point_plan(
    *,
    context: OptimizationStageContext,
    optimizer_mode,
    frequency_payload,
    imaginary_count,
) -> tuple[str, str | None, bool, str, str | None]:
    frequency_enabled = context["frequency_enabled"]
    irc_enabled = context["irc_enabled"]
    single_point_enabled = context["single_point_enabled"]
    ts_quality_enforced = _is_ts_quality_enforced(context.get("ts_quality"))
    if ts_quality_enforced and not frequency_enabled:
        logging.warning(
            "TS quality enforcement requested but frequency calculation disabled; "
            "proceeding without gating."
        )

    irc_status = "skipped"
    irc_skip_reason = None
    if irc_enabled:
        expected_imaginary = 1 if optimizer_mode == "transition_state" else 0
        if frequency_enabled:
            if imaginary_count is None:
                if ts_quality_enforced:
                    irc_skip_reason = (
                        "Imaginary frequency count unavailable; skipping IRC."
                    )
                    logging.warning("Skipping IRC: %s", irc_skip_reason)
                else:
                    logging.warning(
                        "Imaginary frequency count unavailable; proceeding with IRC "
                        "because ts_quality.enforce is false."
                    )
                    irc_status = "pending"
            else:
                ts_quality_result = (
                    frequency_payload.get("results", {}).get("ts_quality")
                    if frequency_payload
                    else None
                )
                if ts_quality_result is None:
                    ts_quality_result = {}
                allow_irc = ts_quality_result.get("allow_irc")
                if optimizer_mode == "transition_state" and allow_irc is not None:
                    if not allow_irc:
                        message = ts_quality_result.get("message") or (
                            "TS quality checks did not pass."
                        )
                        if ts_quality_enforced:
                            irc_skip_reason = message
                            logging.warning("Skipping IRC: %s", irc_skip_reason)
                        else:
                            logging.warning(
                                "TS quality checks did not pass; proceeding with IRC "
                                "because ts_quality.enforce is false. %s",
                                message,
                            )
                            irc_status = "pending"
                    else:
                        irc_status = "pending"
                elif imaginary_count != expected_imaginary:
                    if ts_quality_enforced:
                        irc_skip_reason = (
                            "Imaginary frequency count does not match expected "
                            f"{expected_imaginary}."
                        )
                        logging.warning("Skipping IRC: %s", irc_skip_reason)
                    else:
                        logging.warning(
                            "Imaginary frequency count does not match expected %s; "
                            "proceeding with IRC because ts_quality.enforce is false.",
                            expected_imaginary,
                        )
                        irc_status = "pending"
                else:
                    irc_status = "pending"
        else:
            irc_status = "pending"
            if optimizer_mode == "transition_state":
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
                if ts_quality_enforced:
                    logging.warning(
                        "Skipping single-point calculation because imaginary "
                        "frequency count is unavailable."
                    )
                    sp_skip_reason = "Imaginary frequency count unavailable."
                else:
                    logging.warning(
                        "Imaginary frequency count unavailable; proceeding with "
                        "single-point because ts_quality.enforce is false."
                    )
                    run_single_point = True
            elif optimizer_mode == "transition_state":
                ts_quality_result = (
                    frequency_payload.get("results", {}).get("ts_quality")
                    if frequency_payload
                    else None
                )
                if ts_quality_result is None:
                    ts_quality_result = {}
                allow_sp = ts_quality_result.get("allow_single_point")
                if allow_sp is None:
                    allow_sp = imaginary_count == expected_imaginary
                if allow_sp:
                    run_single_point = True
                else:
                    message = ts_quality_result.get("message") or (
                        "TS quality checks did not pass."
                    )
                    if ts_quality_enforced:
                        logging.warning(
                            "Skipping single-point calculation due to TS quality "
                            "checks."
                        )
                        sp_skip_reason = message
                    else:
                        logging.warning(
                            "TS quality checks did not pass; proceeding with "
                            "single-point because ts_quality.enforce is false. %s",
                            message,
                        )
                        run_single_point = True
            elif imaginary_count == expected_imaginary:
                run_single_point = True
            else:
                if ts_quality_enforced:
                    logging.warning(
                        "Skipping single-point calculation because imaginary "
                        "frequency count %s does not match expected %s.",
                        imaginary_count,
                        expected_imaginary,
                    )
                    sp_skip_reason = (
                        "Imaginary frequency count does not match expected "
                        f"{expected_imaginary}."
                    )
                else:
                    logging.warning(
                        "Imaginary frequency count %s does not match expected %s; "
                        "proceeding with single-point because ts_quality.enforce is "
                        "false.",
                        imaginary_count,
                        expected_imaginary,
                    )
                    run_single_point = True
        else:
            run_single_point = True
    else:
        logging.info("Skipping single-point energy calculation (disabled).")
        sp_skip_reason = "Single-point calculation disabled."

    if run_single_point:
        sp_status = "executed"
        sp_skip_reason = None

    return irc_status, irc_skip_reason, run_single_point, sp_status, sp_skip_reason


def _estimate_ts_energy(frequency_payload, last_scf_energy):
    ts_energy_ev = None
    ts_energy_hartree = None
    if not isinstance(frequency_payload, dict):
        frequency_payload = {}
    freq_results = frequency_payload.get("results") or {}
    ts_energy_hartree = freq_results.get("energy")
    if ts_energy_hartree is None:
        ts_energy_hartree = last_scf_energy
    if ts_energy_hartree is not None:
        try:
            from ase import units
        except Exception:
            ts_energy_ev = ts_energy_hartree * 27.211386245988
        else:
            ts_energy_ev = ts_energy_hartree * units.Hartree
    return ts_energy_ev, ts_energy_hartree


def _normalize_snapshot_settings(
    snapshot_interval_steps, snapshot_mode
) -> tuple[int, str, bool, bool]:
    normalized_interval = snapshot_interval_steps
    if normalized_interval is None or normalized_interval <= 0:
        normalized_interval = 1
    normalized_mode = (snapshot_mode or "all").lower()
    if normalized_mode not in ("none", "last", "all"):
        logging.warning("Unknown snapshot_mode '%s'; defaulting to 'all'.", normalized_mode)
        normalized_mode = "all"
    return (
        normalized_interval,
        normalized_mode,
        normalized_mode == "all",
        normalized_mode in ("all", "last"),
    )


def _build_snapshot_paths(run_dir: str) -> dict[str, str]:
    return {
        "snapshot_dir": resolve_run_path(run_dir, "snapshots"),
        "opt_steps_snapshot": resolve_run_path(run_dir, "snapshots/optimization_steps.xyz"),
        "opt_last_snapshot": resolve_run_path(run_dir, "snapshots/optimization_last.xyz"),
        "irc_forward_steps_snapshot": resolve_run_path(
            run_dir, "snapshots/irc_forward_steps.xyz"
        ),
        "irc_reverse_steps_snapshot": resolve_run_path(
            run_dir, "snapshots/irc_reverse_steps.xyz"
        ),
        "irc_forward_last_snapshot": resolve_run_path(
            run_dir, "snapshots/irc_forward_last.xyz"
        ),
        "irc_reverse_last_snapshot": resolve_run_path(
            run_dir, "snapshots/irc_reverse_last.xyz"
        ),
    }


def _load_checkpoint_base(checkpoint_path) -> dict[str, Any]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
            data = json.load(checkpoint_file)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _extract_resume_optimization_step(context: OptimizationStageContext, checkpoint_base) -> int | None:
    if not context.get("resume_dir") or not checkpoint_base:
        return None
    resume_step = checkpoint_base.get("optimization_last_step")
    if resume_step is None and checkpoint_base.get("last_stage") == "optimization":
        resume_step = checkpoint_base.get("last_step")
    if resume_step is None:
        return None
    try:
        step_value = int(resume_step)
    except (TypeError, ValueError):
        return None
    if step_value < 0:
        return None
    return step_value


def _resolve_resume_xyz_path(
    *,
    context: OptimizationStageContext,
    checkpoint_base: dict[str, Any],
    run_dir,
    charge,
    spin,
    multiplicity,
) -> str | None:
    if not context.get("resume_dir") or not checkpoint_base:
        return None
    resume_candidate = checkpoint_base.get("optimization_last_xyz")
    if resume_candidate:
        resume_candidate = resolve_run_path(run_dir, resume_candidate)
        if os.path.exists(resume_candidate):
            return resume_candidate
    resume_candidate = checkpoint_base.get("last_geometry_xyz")
    if resume_candidate and checkpoint_base.get("last_stage") == "optimization":
        resume_candidate = resolve_run_path(run_dir, resume_candidate)
        if os.path.exists(resume_candidate):
            return resume_candidate
    if checkpoint_base.get("last_geometry") and checkpoint_base.get("last_stage") == "optimization":
        resume_xyz_path = resolve_run_path(run_dir, "resume_last_geometry.xyz")
        comment = format_xyz_comment(
            charge=charge,
            spin=spin,
            multiplicity=multiplicity,
            extra="resume=checkpoint",
        )
        write_xyz_snapshot(
            resume_xyz_path,
            checkpoint_base.get("last_geometry"),
            comment=comment,
        )
        return resume_xyz_path
    return None


def _build_optimization_checkpoint_common(
    *,
    context: OptimizationStageContext,
    run_dir,
    input_xyz_path,
    optimizer_input_xyz,
    resume_xyz_path,
    output_xyz_path,
    pyscf_chkfile,
    optimizer_name,
    optimizer_fmax,
    optimizer_steps,
    snapshot_mode,
    snapshot_write_steps,
    snapshot_write_last,
    snapshot_paths: dict[str, str],
    charge,
    spin,
    multiplicity,
) -> dict[str, Any]:
    return {
        "optimizer_name": optimizer_name,
        "optimizer_fmax": optimizer_fmax,
        "optimizer_steps": optimizer_steps,
        "run_dir": run_dir,
        "calculation_mode": context.get("calculation_mode", "optimization"),
        "input_xyz_path": input_xyz_path,
        "optimizer_input_xyz": optimizer_input_xyz,
        "resume_xyz_path": resume_xyz_path,
        "output_xyz_path": output_xyz_path,
        "pyscf_chkfile": pyscf_chkfile,
        "snapshot_mode": snapshot_mode,
        "snapshot_write_steps": snapshot_write_steps,
        "snapshot_write_last": snapshot_write_last,
        "opt_steps_snapshot": snapshot_paths["opt_steps_snapshot"],
        "opt_last_snapshot": snapshot_paths["opt_last_snapshot"],
        "snapshot_dir": snapshot_paths["snapshot_dir"],
        "charge": charge,
        "spin": spin,
        "multiplicity": multiplicity,
    }


def _update_optimization_checkpoint(
    checkpoint_base: dict[str, Any],
    checkpoint_path,
    checkpoint_common: dict[str, Any],
    *,
    atoms=None,
    atom_spec=None,
    step=None,
    status=None,
    error_message=None,
    scf_energy=None,
    scf_converged=None,
) -> None:
    checkpoint_payload = dict(checkpoint_base)
    if atom_spec is not None:
        checkpoint_payload["last_geometry"] = atom_spec
    elif atoms is not None:
        checkpoint_payload["last_geometry"] = _atoms_to_atom_spec(atoms)
    if step is not None:
        checkpoint_payload["last_step"] = step
    checkpoint_payload.update(
        {
            "optimizer": checkpoint_common["optimizer_name"],
            "fmax": checkpoint_common["optimizer_fmax"],
            "steps": checkpoint_common["optimizer_steps"],
            "run_dir": checkpoint_common["run_dir"],
            "calculation_mode": checkpoint_common["calculation_mode"],
            "timestamp": datetime.now().isoformat(),
            "input_xyz": checkpoint_common["input_xyz_path"],
            "optimizer_input_xyz": checkpoint_common["optimizer_input_xyz"],
            "resume_from": checkpoint_common["resume_xyz_path"],
            "output_xyz": checkpoint_common["output_xyz_path"],
            "pyscf_chkfile": checkpoint_common["pyscf_chkfile"],
            "snapshot_dir": checkpoint_common["snapshot_dir"],
        }
    )
    if checkpoint_common["snapshot_write_steps"]:
        checkpoint_payload["optimization_steps_xyz"] = checkpoint_common["opt_steps_snapshot"]
    else:
        checkpoint_payload.pop("optimization_steps_xyz", None)
    if atoms is not None or atom_spec is not None:
        checkpoint_payload["last_stage"] = "optimization"
        if checkpoint_common["snapshot_write_last"]:
            checkpoint_payload["last_geometry_xyz"] = checkpoint_common["opt_last_snapshot"]
            checkpoint_payload["optimization_last_xyz"] = checkpoint_common["opt_last_snapshot"]
        else:
            checkpoint_payload.pop("last_geometry_xyz", None)
            checkpoint_payload.pop("optimization_last_xyz", None)
    if scf_energy is not None:
        checkpoint_payload["last_scf_energy"] = scf_energy
    if scf_converged is not None:
        checkpoint_payload["last_scf_converged"] = scf_converged
    if status is not None:
        checkpoint_payload["status"] = status
    if error_message is not None:
        checkpoint_payload["error"] = error_message
    if step is not None:
        checkpoint_payload["last_step"] = step
        checkpoint_payload["last_step_stage"] = "optimization"
        checkpoint_payload["optimization_last_step"] = step
    checkpoint_base.clear()
    checkpoint_base.update(checkpoint_payload)
    write_checkpoint(checkpoint_path, checkpoint_base)


def _write_optimization_snapshot(
    atom_spec, step_value, checkpoint_common: dict[str, Any], force_last_only=False
) -> None:
    if checkpoint_common["snapshot_mode"] == "none" or not atom_spec:
        return
    comment = format_xyz_comment(
        charge=checkpoint_common["charge"],
        spin=checkpoint_common["spin"],
        multiplicity=checkpoint_common["multiplicity"],
        extra=f"step={step_value}",
    )
    if checkpoint_common["snapshot_write_steps"] and not force_last_only:
        write_xyz_snapshot(
            checkpoint_common["opt_steps_snapshot"],
            atom_spec,
            comment=comment,
            append=True,
        )
    if checkpoint_common["snapshot_write_last"] or force_last_only:
        write_xyz_snapshot(
            checkpoint_common["opt_last_snapshot"],
            atom_spec,
            comment=comment,
        )


def _build_optimization_step_callback(
    *,
    n_steps: dict[str, int],
    last_metadata_write: dict[str, int | float],
    io_write_interval_steps,
    io_write_interval_seconds,
    snapshot_interval_steps,
    optimization_metadata: dict[str, Any],
    run_metadata_path,
    metadata_recorder: RunMetadataRecorder,
    checkpoint_base: dict[str, Any],
    checkpoint_path,
    checkpoint_common: dict[str, Any],
):
    last_snapshot_write = {"step": n_steps["value"]}
    last_checkpoint_write = {"step": n_steps["value"]}
    first_checkpoint_step = n_steps["value"] + 1

    def _step_callback(*_args, **_kwargs):
        n_steps["value"] += 1
        step_value = n_steps["value"]
        now = time.monotonic()
        should_write = False
        if io_write_interval_steps:
            should_write = (
                step_value - int(last_metadata_write["step"]) >= io_write_interval_steps
            )
        if not should_write and io_write_interval_seconds is not None:
            should_write = (
                now - float(last_metadata_write["time"]) >= io_write_interval_seconds
            )
        optimizer = _args[0] if _args else None
        atoms = getattr(optimizer, "atoms", None) if optimizer is not None else None
        if atoms is not None:
            should_snapshot = (
                step_value - last_snapshot_write["step"] >= snapshot_interval_steps
            )
            should_checkpoint = (
                step_value - last_checkpoint_write["step"] >= snapshot_interval_steps
            )
            if step_value == first_checkpoint_step:
                should_checkpoint = True
            if should_snapshot or should_checkpoint:
                atom_spec = _atoms_to_atom_spec(atoms)
                if should_snapshot:
                    _write_optimization_snapshot(atom_spec, step_value, checkpoint_common)
                    last_snapshot_write["step"] = step_value
                if should_checkpoint:
                    _update_optimization_checkpoint(
                        checkpoint_base,
                        checkpoint_path,
                        checkpoint_common,
                        atom_spec=atom_spec,
                        step=step_value,
                        status="running",
                    )
                    last_checkpoint_write["step"] = step_value
        if should_write:
            optimization_metadata["n_steps"] = step_value
            optimization_metadata["n_steps_source"] = "ase"
            optimization_metadata["status"] = "running"
            optimization_metadata["run_updated_at"] = datetime.now().isoformat()
            metadata_recorder.write(run_metadata_path, optimization_metadata)
            last_metadata_write["time"] = now
            last_metadata_write["step"] = step_value

    return _step_callback


def _resolve_irc_settings(irc_config) -> tuple[int, float, float]:
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
    return irc_steps, irc_step_size, irc_force_threshold


def _parse_nonnegative_step(value) -> int | None:
    try:
        step_value = int(value)
    except (TypeError, ValueError):
        return None
    return step_value if step_value >= 0 else None


def _initialize_irc_profile_state(
    context: OptimizationStageContext,
    checkpoint_base: dict[str, Any],
) -> dict[str, Any]:
    irc_profile_cache: list[dict[str, Any]] = []
    irc_profile_keys = set()
    if context.get("resume_dir") and checkpoint_base.get("irc_profile"):
        irc_profile_cache = list(checkpoint_base.get("irc_profile", []))
        for entry in irc_profile_cache:
            irc_profile_keys.add((entry.get("direction"), entry.get("step")))
    return {
        "profile_cache": irc_profile_cache,
        "profile_keys": irc_profile_keys,
        "last_snapshot_step": {"forward": -1, "reverse": -1},
        "last_checkpoint_step": {"forward": -1, "reverse": -1},
        "last_geometry_cache": {"forward": None, "reverse": None},
        "last_step_cache": {"forward": None, "reverse": None},
    }


def _build_irc_resume_state_for_optimized_geometry(
    *,
    context: OptimizationStageContext,
    checkpoint_base: dict[str, Any],
    run_dir,
    charge,
    spin,
    multiplicity,
    irc_state: dict[str, Any],
) -> dict[str, object] | None:
    if not context.get("resume_dir") or not checkpoint_base:
        return None
    resume_state: dict[str, object] = {
        "forward_completed": bool(checkpoint_base.get("irc_forward_completed")),
        "reverse_completed": bool(checkpoint_base.get("irc_reverse_completed")),
    }
    for direction in ("forward", "reverse"):
        step_key = f"irc_{direction}_step"
        xyz_key = f"irc_{direction}_last_xyz"
        geom_key = f"irc_{direction}_last_geometry"
        step_value_raw = checkpoint_base.get(step_key)
        step_value = _parse_nonnegative_step(step_value_raw)
        if step_value is not None:
            irc_state["last_snapshot_step"][direction] = step_value
            irc_state["last_checkpoint_step"][direction] = step_value
        xyz_value = checkpoint_base.get(xyz_key)
        if xyz_value:
            resolved = resolve_run_path(run_dir, xyz_value)
            if os.path.exists(resolved):
                resume_state[direction] = {
                    "step": step_value if step_value is not None else step_value_raw,
                    "xyz": resolved,
                }
                continue
        atom_spec = checkpoint_base.get(geom_key)
        if atom_spec and step_value is not None:
            resume_xyz = resolve_run_path(
                run_dir, f"resume_irc_{direction}_last_geometry.xyz"
            )
            comment = format_xyz_comment(
                charge=charge,
                spin=spin,
                multiplicity=multiplicity,
                extra=f"resume=checkpoint direction={direction}",
            )
            write_xyz_snapshot(
                resume_xyz,
                atom_spec,
                comment=comment,
            )
            resume_state[direction] = {"step": step_value, "xyz": resume_xyz}
    return resume_state


def _build_irc_checkpoint_callbacks_for_optimized_geometry(
    *,
    checkpoint_path,
    checkpoint_base: dict[str, Any],
    snapshot_interval_steps,
    snapshot_mode,
    snapshot_write_steps,
    snapshot_write_last,
    snapshot_dir,
    irc_forward_steps_snapshot,
    irc_reverse_steps_snapshot,
    irc_forward_last_snapshot,
    irc_reverse_last_snapshot,
    charge,
    spin,
    multiplicity,
    irc_state: dict[str, Any],
):
    def _persist_checkpoint():
        if not checkpoint_path:
            return
        write_checkpoint(checkpoint_path, checkpoint_base)

    def _record_irc_step(direction, step_index, atoms, energy_ev, energy_hartree):
        atom_spec = _atoms_to_atom_spec(atoms)
        irc_state["last_geometry_cache"][direction] = atom_spec
        irc_state["last_step_cache"][direction] = step_index
        steps_path = (
            irc_forward_steps_snapshot
            if direction == "forward"
            else irc_reverse_steps_snapshot
        )
        last_path = (
            irc_forward_last_snapshot
            if direction == "forward"
            else irc_reverse_last_snapshot
        )
        entry_key = (direction, step_index)
        if entry_key not in irc_state["profile_keys"]:
            irc_state["profile_keys"].add(entry_key)
            irc_state["profile_cache"].append(
                {
                    "direction": direction,
                    "step": step_index,
                    "energy_ev": float(energy_ev),
                    "energy_hartree": float(energy_hartree),
                }
            )
        should_snapshot = (
            step_index - irc_state["last_snapshot_step"][direction] >= snapshot_interval_steps
        )
        should_checkpoint = (
            irc_state["last_checkpoint_step"][direction] < 0
            or step_index - irc_state["last_checkpoint_step"][direction]
            >= snapshot_interval_steps
        )
        if should_snapshot and snapshot_mode != "none":
            comment = format_xyz_comment(
                charge=charge,
                spin=spin,
                multiplicity=multiplicity,
                extra=f"step={step_index} direction={direction}",
            )
            if snapshot_write_steps:
                write_xyz_snapshot(
                    steps_path,
                    atom_spec,
                    comment=comment,
                    append=True,
                )
            if snapshot_write_last:
                write_xyz_snapshot(last_path, atom_spec, comment=comment)
            irc_state["last_snapshot_step"][direction] = step_index
        if should_checkpoint:
            checkpoint_base.update(
                {
                    "last_stage": "irc",
                    "last_step": step_index,
                    "last_step_stage": "irc",
                    "last_step_direction": direction,
                    "last_geometry": atom_spec,
                    "snapshot_dir": snapshot_dir,
                    "irc_direction": direction,
                    f"irc_{direction}_step": step_index,
                    f"irc_{direction}_last_geometry": atom_spec,
                }
            )
            if snapshot_write_steps:
                checkpoint_base["irc_forward_steps_xyz"] = irc_forward_steps_snapshot
                checkpoint_base["irc_reverse_steps_xyz"] = irc_reverse_steps_snapshot
            else:
                checkpoint_base.pop("irc_forward_steps_xyz", None)
                checkpoint_base.pop("irc_reverse_steps_xyz", None)
            if snapshot_write_last:
                checkpoint_base["last_geometry_xyz"] = last_path
                checkpoint_base[f"irc_{direction}_last_xyz"] = last_path
            else:
                checkpoint_base.pop("last_geometry_xyz", None)
                checkpoint_base.pop(f"irc_{direction}_last_xyz", None)
            checkpoint_base["irc_profile"] = irc_state["profile_cache"]
            _persist_checkpoint()
            irc_state["last_checkpoint_step"][direction] = step_index

    def _mark_direction_complete(direction, last_step):
        cached_step = irc_state["last_step_cache"].get(direction)
        if cached_step is not None:
            last_step = cached_step
        atom_spec = irc_state["last_geometry_cache"].get(direction)
        if atom_spec:
            comment = format_xyz_comment(
                charge=charge,
                spin=spin,
                multiplicity=multiplicity,
                extra=f"step={last_step} direction={direction}",
            )
            last_path = (
                irc_forward_last_snapshot
                if direction == "forward"
                else irc_reverse_last_snapshot
            )
            if snapshot_write_last:
                write_xyz_snapshot(last_path, atom_spec, comment=comment)
            checkpoint_base.update(
                {
                    "last_stage": "irc",
                    "last_step": last_step,
                    "last_step_stage": "irc",
                    "last_step_direction": direction,
                    "last_geometry": atom_spec,
                    "snapshot_dir": snapshot_dir,
                    "irc_direction": direction,
                    f"irc_{direction}_last_geometry": atom_spec,
                }
            )
            if snapshot_write_steps:
                checkpoint_base["irc_forward_steps_xyz"] = irc_forward_steps_snapshot
                checkpoint_base["irc_reverse_steps_xyz"] = irc_reverse_steps_snapshot
            else:
                checkpoint_base.pop("irc_forward_steps_xyz", None)
                checkpoint_base.pop("irc_reverse_steps_xyz", None)
            if snapshot_write_last:
                checkpoint_base["last_geometry_xyz"] = last_path
                checkpoint_base[f"irc_{direction}_last_xyz"] = last_path
            else:
                checkpoint_base.pop("last_geometry_xyz", None)
                checkpoint_base.pop(f"irc_{direction}_last_xyz", None)
        checkpoint_base["irc_profile"] = irc_state["profile_cache"]
        checkpoint_base[f"irc_{direction}_completed"] = True
        if last_step is not None:
            checkpoint_base[f"irc_{direction}_step"] = last_step
        _persist_checkpoint()

    return _record_irc_step, _mark_direction_complete


def _compute_optimized_geometry_mode(
    *,
    context: OptimizationStageContext,
    engine_adapter: WorkflowEngineAdapter,
    mol_optimized,
    optimizer_ase_dict,
    optimizer_mode,
    multiplicity,
    run_dir,
    verbose,
    memory_mb,
    profiling_enabled,
):
    mode_result = engine_adapter.compute_imaginary_mode(
        mol_optimized,
        context["sp_basis"],
        context["sp_xc"],
        context["sp_scf_config"],
        context["sp_solvent_model"] if context["sp_solvent_name"] else None,
        context["sp_solvent_name"],
        context["sp_eps"],
        verbose,
        memory_mb,
        dispersion=context["sp_dispersion_model"],
        dispersion_hessian_step=context.get("freq_dispersion_step"),
        constraints=context["constraints"],
        dispersion_params=_resolve_d3_params(optimizer_ase_dict),
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
    return mode_result


def _run_ase_irc_for_optimized_geometry(
    *,
    context: OptimizationStageContext,
    engine_adapter: WorkflowEngineAdapter,
    output_xyz_path,
    run_dir,
    charge,
    spin,
    multiplicity,
    verbose,
    memory_mb,
    optimizer_ase_dict,
    optimizer_mode,
    mode_result,
    irc_steps,
    irc_step_size,
    irc_force_threshold,
    profiling_enabled,
    record_irc_step,
    mark_direction_complete,
    irc_resume_state,
):
    return engine_adapter.run_ase_irc(
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
        mode_result.get("hessian"),
        irc_steps,
        irc_step_size,
        irc_force_threshold,
        profiling_enabled=profiling_enabled,
        step_callback=record_irc_step,
        direction_callback=mark_direction_complete,
        resume_state=irc_resume_state,
    )


def _build_success_irc_payload_for_optimized_geometry(
    *,
    irc_output_path,
    irc_result,
    profile,
    irc_steps,
    irc_step_size,
    irc_force_threshold,
    mode_result,
    mode_profiling,
    profiling_enabled,
    ts_energy_ev,
):
    irc_payload = {
        "status": "completed",
        "output_file": irc_output_path,
        "forward_xyz": irc_result.get("forward_xyz"),
        "reverse_xyz": irc_result.get("reverse_xyz"),
        "steps": irc_steps,
        "step_size": irc_step_size,
        "force_threshold": irc_force_threshold,
        "mode_eigenvalue": mode_result.get("eigenvalue"),
        "profile": profile,
        "profiling": {
            "mode": mode_profiling,
            "irc": irc_result.get("profiling"),
        }
        if profiling_enabled
        else None,
    }
    irc_payload["assessment"] = _evaluate_irc_profile(
        irc_payload["profile"],
        ts_energy_ev=ts_energy_ev,
    )
    return irc_payload


def _build_failed_irc_payload_for_optimized_geometry(
    *,
    irc_output_path,
    irc_steps,
    irc_step_size,
    irc_force_threshold,
    error,
):
    return {
        "status": "failed",
        "output_file": irc_output_path,
        "steps": irc_steps,
        "step_size": irc_step_size,
        "force_threshold": irc_force_threshold,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


def _write_irc_payload_file(irc_output_path, irc_payload) -> None:
    with open(irc_output_path, "w", encoding="utf-8") as handle:
        json.dump(irc_payload, handle, indent=2)


def _run_irc_for_optimized_geometry(
    *,
    context: OptimizationStageContext,
    engine_adapter: WorkflowEngineAdapter,
    mol_optimized,
    output_xyz_path,
    run_dir,
    charge,
    spin,
    multiplicity,
    verbose,
    memory_mb,
    optimizer_ase_dict,
    optimizer_mode,
    profiling_enabled,
    irc_status,
    irc_config,
    checkpoint_base: dict[str, Any],
    checkpoint_path,
    snapshot_interval_steps,
    snapshot_mode,
    snapshot_write_steps,
    snapshot_write_last,
    snapshot_dir,
    irc_forward_steps_snapshot,
    irc_reverse_steps_snapshot,
    irc_forward_last_snapshot,
    irc_reverse_last_snapshot,
    ts_energy_ev,
    irc_output_path,
) -> tuple[str, dict[str, Any] | None]:
    irc_payload = None
    if irc_status == "pending":
        logging.info("Running IRC for optimized geometry...")
        irc_steps, irc_step_size, irc_force_threshold = _resolve_irc_settings(irc_config)
        try:
            mode_result = _compute_optimized_geometry_mode(
                context=context,
                engine_adapter=engine_adapter,
                mol_optimized=mol_optimized,
                optimizer_ase_dict=optimizer_ase_dict,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
                run_dir=run_dir,
                verbose=verbose,
                memory_mb=memory_mb,
                profiling_enabled=profiling_enabled,
            )
            mode_profiling = mode_result.get("profiling") if profiling_enabled else None
            irc_state = _initialize_irc_profile_state(context, checkpoint_base)
            irc_resume_state = _build_irc_resume_state_for_optimized_geometry(
                context=context,
                checkpoint_base=checkpoint_base,
                run_dir=run_dir,
                charge=charge,
                spin=spin,
                multiplicity=multiplicity,
                irc_state=irc_state,
            )
            record_irc_step, mark_direction_complete = (
                _build_irc_checkpoint_callbacks_for_optimized_geometry(
                    checkpoint_path=checkpoint_path,
                    checkpoint_base=checkpoint_base,
                    snapshot_interval_steps=snapshot_interval_steps,
                    snapshot_mode=snapshot_mode,
                    snapshot_write_steps=snapshot_write_steps,
                    snapshot_write_last=snapshot_write_last,
                    snapshot_dir=snapshot_dir,
                    irc_forward_steps_snapshot=irc_forward_steps_snapshot,
                    irc_reverse_steps_snapshot=irc_reverse_steps_snapshot,
                    irc_forward_last_snapshot=irc_forward_last_snapshot,
                    irc_reverse_last_snapshot=irc_reverse_last_snapshot,
                    charge=charge,
                    spin=spin,
                    multiplicity=multiplicity,
                    irc_state=irc_state,
                )
            )
            irc_result = _run_ase_irc_for_optimized_geometry(
                context=context,
                engine_adapter=engine_adapter,
                output_xyz_path=output_xyz_path,
                run_dir=run_dir,
                charge=charge,
                spin=spin,
                multiplicity=multiplicity,
                verbose=verbose,
                memory_mb=memory_mb,
                optimizer_ase_dict=optimizer_ase_dict,
                optimizer_mode=optimizer_mode,
                mode_result=mode_result,
                irc_steps=irc_steps,
                irc_step_size=irc_step_size,
                irc_force_threshold=irc_force_threshold,
                profiling_enabled=profiling_enabled,
                record_irc_step=record_irc_step,
                mark_direction_complete=mark_direction_complete,
                irc_resume_state=irc_resume_state,
            )
            profile = irc_state["profile_cache"] or irc_result.get("profile", [])
            irc_payload = _build_success_irc_payload_for_optimized_geometry(
                irc_output_path=irc_output_path,
                irc_result=irc_result,
                profile=profile,
                irc_steps=irc_steps,
                irc_step_size=irc_step_size,
                irc_force_threshold=irc_force_threshold,
                mode_result=mode_result,
                mode_profiling=mode_profiling,
                profiling_enabled=profiling_enabled,
                ts_energy_ev=ts_energy_ev,
            )
            irc_status = "executed"
        except Exception as exc:
            logging.exception("IRC calculation failed.")
            irc_payload = _build_failed_irc_payload_for_optimized_geometry(
                irc_output_path=irc_output_path,
                irc_steps=irc_steps,
                irc_step_size=irc_step_size,
                irc_force_threshold=irc_force_threshold,
                error=exc,
            )
            irc_status = "failed"
        _write_irc_payload_file(irc_output_path, irc_payload)
    elif irc_status == "skipped":
        logging.info("Skipping IRC calculation.")

    return irc_status, irc_payload


def _run_single_point_for_optimized_geometry(
    *,
    run_single_point,
    single_point_enabled,
    context: OptimizationStageContext,
    engine_adapter: WorkflowEngineAdapter,
    mol_optimized,
    optimizer_ase_dict,
    optimizer_mode,
    multiplicity,
    run_dir,
    verbose,
    memory_mb,
    profiling_enabled,
    log_path,
    optimization_metadata: dict[str, Any],
    last_scf_energy,
    last_scf_converged,
) -> tuple[
    dict[str, Any] | None,
    float | None,
    bool | None,
    int | None,
    float | None,
    bool | None,
]:
    sp_result = None
    final_sp_energy = None
    final_sp_converged = None
    final_sp_cycles = None
    if run_single_point:
        logging.info("Calculating single-point energy for optimized geometry...")
        sp_result = engine_adapter.compute_single_point_energy(
            mol_optimized,
            context["sp_basis"],
            context["sp_xc"],
            context["sp_scf_config"],
            context["sp_solvent_model"] if context["sp_solvent_name"] else None,
            context["sp_solvent_name"],
            context["sp_eps"],
            context["freq_dispersion_model"],
            _resolve_d3_params(optimizer_ase_dict),
            verbose,
            memory_mb,
            run_dir=run_dir,
            optimizer_mode=optimizer_mode,
            multiplicity=multiplicity,
            profiling_enabled=profiling_enabled,
        )
        final_sp_energy = sp_result["energy"]
        final_sp_converged = sp_result["converged"]
        final_sp_cycles = sp_result["cycles"]
        last_scf_energy = final_sp_energy
        last_scf_converged = final_sp_converged
        optimization_metadata["single_point"]["dispersion_info"] = sp_result.get(
            "dispersion"
        )
        if profiling_enabled and sp_result.get("profiling"):
            optimization_metadata["single_point"]["profiling"] = sp_result.get(
                "profiling"
            )
        if final_sp_cycles is None:
            final_sp_cycles = parse_single_point_cycle_count(log_path)
    elif single_point_enabled:
        logging.info("Skipping single-point energy calculation.")

    return (
        sp_result,
        final_sp_energy,
        final_sp_converged,
        final_sp_cycles,
        last_scf_energy,
        last_scf_converged,
    )


def _run_geometry_optimization_phase(
    *,
    args,
    engine_adapter: WorkflowEngineAdapter,
    run_dir,
    input_xyz_path,
    optimizer_input_xyz,
    output_xyz_path,
    charge,
    spin,
    multiplicity,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    eps,
    dispersion_model,
    verbose,
    memory_mb,
    optimizer_ase_dict,
    optimizer_mode,
    constraints,
    profiling_enabled,
    step_callback,
    optimization_metadata: dict[str, Any],
    n_steps: dict[str, int],
) -> tuple[Any, Any, int | None, str]:
    from pyscf import gto

    if args.xyz_file and input_xyz_path:
        if os.path.abspath(args.xyz_file) != os.path.abspath(input_xyz_path):
            shutil.copy2(args.xyz_file, input_xyz_path)
    ensure_parent_dir(output_xyz_path)
    opt_result = engine_adapter.run_ase_optimizer(
        optimizer_input_xyz,
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
        eps,
        dispersion_model,
        verbose,
        memory_mb,
        optimizer_ase_dict,
        optimizer_mode,
        constraints,
        profiling_enabled=profiling_enabled,
        step_callback=step_callback,
    )
    n_steps_value = opt_result.get("n_steps")
    if profiling_enabled and opt_result.get("profiling"):
        optimization_metadata.setdefault("profiling", {})["optimizer"] = opt_result.get(
            "profiling"
        )
    optimized_atom_spec, _, _, _ = engine_adapter.load_xyz(output_xyz_path)
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
    return mol_optimized, optimized_atom_spec, n_steps_value, "ase"


def _handle_optimization_failure(
    *,
    error,
    metadata_recorder: RunMetadataRecorder,
    checkpoint_base: dict[str, Any],
    checkpoint_path,
    checkpoint_common: dict[str, Any],
    optimization_metadata: dict[str, Any],
    run_start,
    mf,
    mol,
    memory_limit_enforced,
    run_metadata_path,
    event_log_path,
    run_id,
    run_dir,
    queue_update_fn,
    n_steps: dict[str, int],
    n_steps_source,
) -> None:
    n_steps_value = n_steps["value"] if n_steps_source else None
    _update_optimization_checkpoint(
        checkpoint_base,
        checkpoint_path,
        checkpoint_common,
        status="failed",
        error_message=str(error),
    )
    optimization_metadata["status"] = "failed"
    optimization_metadata["run_ended_at"] = datetime.now().isoformat()
    optimization_metadata["error"] = str(error)
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
    metadata_recorder.write(run_metadata_path, optimization_metadata)
    metadata_recorder.record_status(
        event_log_path,
        run_id,
        run_dir,
        "failed",
        previous_status="running",
        details={"error": str(error)},
    )
    queue_update_fn("failed", exit_code=1)


def _mark_optimization_completed(
    *,
    mol_optimized,
    optimized_atom_spec,
    optimized_xyz_path,
    n_steps_value,
    n_steps: dict[str, int],
    n_steps_source,
    checkpoint_base: dict[str, Any],
    checkpoint_path,
    checkpoint_common: dict[str, Any],
    optimization_metadata: dict[str, Any],
    run_start,
) -> tuple[float, int | None]:
    logging.info("Optimization finished.")
    logging.info("Optimized geometry (in Angstrom):")
    logging.info("%s", mol_optimized.tostring(format="xyz"))
    write_optimized_xyz(optimized_xyz_path, mol_optimized)
    final_step_value = n_steps_value if n_steps_value is not None else n_steps["value"]
    if final_step_value is None:
        final_step_value = "final"
    _write_optimization_snapshot(
        optimized_atom_spec,
        final_step_value,
        checkpoint_common,
        force_last_only=True,
    )
    ensure_stream_newlines()
    optimization_metadata["status"] = "completed"
    optimization_metadata["run_ended_at"] = datetime.now().isoformat()
    _update_optimization_checkpoint(
        checkpoint_base,
        checkpoint_path,
        checkpoint_common,
        atom_spec=optimized_atom_spec,
        step=n_steps_value,
        status="completed",
    )
    elapsed_seconds = time.perf_counter() - run_start
    resolved_n_steps_value = n_steps["value"] if n_steps_source else None
    return elapsed_seconds, resolved_n_steps_value


def _run_frequency_and_plan_after_optimization(
    *,
    context: OptimizationStageContext,
    engine_adapter: WorkflowEngineAdapter,
    mol_optimized,
    freq_scf_config,
    optimizer_ase_dict,
    optimizer_mode,
    multiplicity,
    run_dir,
    frequency_output_path,
    verbose,
    memory_mb,
    profiling_enabled,
    optimization_metadata: dict[str, Any],
    irc_output_path,
):
    base_chkfile = context.get("pyscf_chkfile")
    sp_chkfile = context.get("sp_chkfile")
    if base_chkfile and sp_chkfile and base_chkfile != sp_chkfile:
        if context["basis"] == context["sp_basis"]:
            _seed_scf_checkpoint(
                base_chkfile,
                sp_chkfile,
                label="post-optimization",
            )
        else:
            logging.info(
                "Skipping SCF checkpoint seed (basis mismatch: %s vs %s).",
                context["basis"],
                context["sp_basis"],
            )
    (
        frequency_payload,
        imaginary_count,
        last_scf_energy,
        last_scf_converged,
    ) = _run_frequency_for_optimized_geometry(
        context=context,
        engine_adapter=engine_adapter,
        mol_optimized=mol_optimized,
        freq_scf_config=freq_scf_config,
        optimizer_ase_dict=optimizer_ase_dict,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        run_dir=run_dir,
        frequency_output_path=frequency_output_path,
        verbose=verbose,
        memory_mb=memory_mb,
        profiling_enabled=profiling_enabled,
        optimization_metadata=optimization_metadata,
    )
    (
        irc_status,
        irc_skip_reason,
        run_single_point,
        sp_status,
        sp_skip_reason,
    ) = _determine_irc_and_single_point_plan(
        context=context,
        optimizer_mode=optimizer_mode,
        frequency_payload=frequency_payload,
        imaginary_count=imaginary_count,
    )
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
    ts_energy_ev, _ = _estimate_ts_energy(frequency_payload, last_scf_energy)
    return {
        "frequency_payload": frequency_payload,
        "irc_status": irc_status,
        "run_single_point": run_single_point,
        "last_scf_energy": last_scf_energy,
        "last_scf_converged": last_scf_converged,
        "ts_energy_ev": ts_energy_ev,
    }


def _run_irc_and_single_point_after_optimization(
    *,
    run_single_point,
    single_point_enabled,
    context: OptimizationStageContext,
    engine_adapter: WorkflowEngineAdapter,
    mol_optimized,
    output_xyz_path,
    run_dir,
    charge,
    spin,
    multiplicity,
    verbose,
    memory_mb,
    optimizer_ase_dict,
    optimizer_mode,
    profiling_enabled,
    irc_status,
    irc_config,
    checkpoint_base: dict[str, Any],
    checkpoint_path,
    snapshot_interval_steps,
    snapshot_mode,
    snapshot_write_steps,
    snapshot_write_last,
    snapshot_dir,
    irc_forward_steps_snapshot,
    irc_reverse_steps_snapshot,
    irc_forward_last_snapshot,
    irc_reverse_last_snapshot,
    ts_energy_ev,
    irc_output_path,
    log_path,
    optimization_metadata: dict[str, Any],
    last_scf_energy,
    last_scf_converged,
):
    irc_payload = None
    sp_result = None
    final_sp_energy = None
    final_sp_converged = None
    final_sp_cycles = None
    try:
        irc_status, irc_payload = _run_irc_for_optimized_geometry(
            context=context,
            engine_adapter=engine_adapter,
            mol_optimized=mol_optimized,
            output_xyz_path=output_xyz_path,
            run_dir=run_dir,
            charge=charge,
            spin=spin,
            multiplicity=multiplicity,
            verbose=verbose,
            memory_mb=memory_mb,
            optimizer_ase_dict=optimizer_ase_dict,
            optimizer_mode=optimizer_mode,
            profiling_enabled=profiling_enabled,
            irc_status=irc_status,
            irc_config=irc_config,
            checkpoint_base=checkpoint_base,
            checkpoint_path=checkpoint_path,
            snapshot_interval_steps=snapshot_interval_steps,
            snapshot_mode=snapshot_mode,
            snapshot_write_steps=snapshot_write_steps,
            snapshot_write_last=snapshot_write_last,
            snapshot_dir=snapshot_dir,
            irc_forward_steps_snapshot=irc_forward_steps_snapshot,
            irc_reverse_steps_snapshot=irc_reverse_steps_snapshot,
            irc_forward_last_snapshot=irc_forward_last_snapshot,
            irc_reverse_last_snapshot=irc_reverse_last_snapshot,
            ts_energy_ev=ts_energy_ev,
            irc_output_path=irc_output_path,
        )
        if irc_payload is not None:
            optimization_metadata["irc"] = irc_payload
        (
            sp_result,
            final_sp_energy,
            final_sp_converged,
            final_sp_cycles,
            last_scf_energy,
            last_scf_converged,
        ) = _run_single_point_for_optimized_geometry(
            run_single_point=run_single_point,
            single_point_enabled=single_point_enabled,
            context=context,
            engine_adapter=engine_adapter,
            mol_optimized=mol_optimized,
            optimizer_ase_dict=optimizer_ase_dict,
            optimizer_mode=optimizer_mode,
            multiplicity=multiplicity,
            run_dir=run_dir,
            verbose=verbose,
            memory_mb=memory_mb,
            profiling_enabled=profiling_enabled,
            log_path=log_path,
            optimization_metadata=optimization_metadata,
            last_scf_energy=last_scf_energy,
            last_scf_converged=last_scf_converged,
        )
    except Exception:
        logging.exception("Post-optimization calculations failed.")
        if run_single_point:
            final_sp_energy = None
            final_sp_converged = None
            final_sp_cycles = parse_single_point_cycle_count(log_path)
    return {
        "irc_payload": irc_payload,
        "sp_result": sp_result,
        "final_sp_energy": final_sp_energy,
        "final_sp_converged": final_sp_converged,
        "final_sp_cycles": final_sp_cycles,
        "last_scf_energy": last_scf_energy,
        "last_scf_converged": last_scf_converged,
    }


def _finalize_optimization_stage_success(
    *,
    metadata_recorder: RunMetadataRecorder,
    optimization_metadata: dict[str, Any],
    n_steps_value,
    n_steps_source,
    mf,
    mol_optimized,
    elapsed_seconds,
    final_sp_energy,
    final_sp_converged,
    final_sp_cycles,
    memory_limit_enforced,
    checkpoint_base: dict[str, Any],
    checkpoint_path,
    checkpoint_common: dict[str, Any],
    last_scf_energy,
    last_scf_converged,
    run_metadata_path,
    context: OptimizationStageContext,
    args,
    optimized_xyz_path,
    frequency_payload,
    irc_payload,
    sp_result,
    event_log_path,
    run_id,
    run_dir,
    queue_update_fn,
) -> None:
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
    _update_optimization_checkpoint(
        checkpoint_base,
        checkpoint_path,
        checkpoint_common,
        scf_energy=last_scf_energy,
        scf_converged=last_scf_converged,
    )
    metadata_recorder.write(run_metadata_path, optimization_metadata)
    export_qcschema_result(
        context.get("qcschema_output_path"),
        optimization_metadata,
        args.xyz_file,
        geometry_xyz=optimized_xyz_path,
        frequency_payload=frequency_payload,
        irc_payload=irc_payload,
        sp_result=sp_result,
    )
    metadata_recorder.record_status(
        event_log_path,
        run_id,
        run_dir,
        "completed",
        previous_status="running",
    )
    queue_update_fn("completed", exit_code=0)


def _prepare_optimization_runtime_state(
    *,
    args,
    context: OptimizationStageContext,
    engine_adapter: WorkflowEngineAdapter,
    metadata_recorder: RunMetadataRecorder,
    molecule_context: MoleculeContext,
    memory_mb,
    memory_limit_status,
    openmp_available,
    effective_threads,
) -> dict[str, Any]:
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
    profiling_enabled = bool(context.get("profiling_enabled"))
    io_write_interval_steps = context.get("io_write_interval_steps", 5)
    io_write_interval_seconds = context.get("io_write_interval_seconds", 5.0)
    (
        snapshot_interval_steps,
        snapshot_mode,
        snapshot_write_steps,
        snapshot_write_last,
    ) = _normalize_snapshot_settings(
        context.get("snapshot_interval_steps", 1),
        context.get("snapshot_mode", "all"),
    )
    freq_scf_config = context.get("freq_scf_config") or context["sp_scf_config"]
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

    _run_optimization_capability_checks(
        context=context,
        mol=mol,
        basis=basis,
        xc=xc,
        scf_config=scf_config,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        freq_scf_config=freq_scf_config,
        optimizer_ase_dict=optimizer_ase_dict,
        engine_adapter=engine_adapter,
        verbose=verbose,
        memory_mb=memory_mb,
    )
    _log_optimization_start(
        context=context,
        run_id=run_id,
        run_dir=run_dir,
        optimizer_mode=optimizer_mode,
        thread_count=thread_count,
        openmp_available=openmp_available,
        effective_threads=effective_threads,
        memory_gb=memory_gb,
        memory_limit_status=memory_limit_status,
        verbose=verbose,
        log_path=log_path,
        event_log_path=event_log_path,
        basis=basis,
        xc=xc,
        scf_config=scf_config,
        solvent_name=solvent_name,
        solvent_model=solvent_model,
        dispersion_model=dispersion_model,
    )
    run_start = time.perf_counter()
    optimization_metadata = _build_optimization_metadata(
        args=args,
        context=context,
        run_dir=run_dir,
        run_id=run_id,
        basis=basis,
        xc=xc,
        solvent_name=solvent_name,
        solvent_model=solvent_model,
        solvent_map_path=solvent_map_path,
        dispersion_model=dispersion_model,
        frequency_enabled=frequency_enabled,
        single_point_enabled=single_point_enabled,
        irc_enabled=irc_enabled,
        charge=charge,
        spin=spin,
        multiplicity=multiplicity,
        ks_type=ks_type,
        thread_count=thread_count,
        effective_threads=effective_threads,
        openmp_available=openmp_available,
        memory_gb=memory_gb,
        memory_mb=memory_mb,
        memory_limit_status=memory_limit_status,
        log_path=log_path,
        event_log_path=event_log_path,
        optimized_xyz_path=optimized_xyz_path,
        frequency_output_path=frequency_output_path,
        irc_output_path=irc_output_path,
        run_metadata_path=run_metadata_path,
        scf_config=scf_config,
    )
    n_steps = {"value": 0}
    n_steps_source = None
    metadata_recorder.write(run_metadata_path, optimization_metadata)
    metadata_recorder.record_status(
        event_log_path,
        run_id,
        run_dir,
        "running",
        previous_status=context.get("previous_status"),
    )
    last_metadata_write = {"time": time.monotonic(), "step": 0}
    checkpoint_base = _load_checkpoint_base(checkpoint_path)
    resume_step = _extract_resume_optimization_step(context, checkpoint_base)
    if resume_step is not None:
        n_steps["value"] = resume_step
        last_metadata_write["step"] = resume_step
    snapshot_paths = _build_snapshot_paths(run_dir)
    snapshot_dir = snapshot_paths["snapshot_dir"]
    irc_forward_steps_snapshot = snapshot_paths["irc_forward_steps_snapshot"]
    irc_reverse_steps_snapshot = snapshot_paths["irc_reverse_steps_snapshot"]
    irc_forward_last_snapshot = snapshot_paths["irc_forward_last_snapshot"]
    irc_reverse_last_snapshot = snapshot_paths["irc_reverse_last_snapshot"]
    resume_xyz_path = _resolve_resume_xyz_path(
        context=context,
        checkpoint_base=checkpoint_base,
        run_dir=run_dir,
        charge=charge,
        spin=spin,
        multiplicity=multiplicity,
    )
    if resume_xyz_path:
        logging.info("Resuming optimization from snapshot: %s", resume_xyz_path)
    optimizer_name = (optimizer_ase_dict.get("optimizer") or "").lower()
    if not optimizer_name:
        optimizer_name = "sella" if optimizer_mode == "transition_state" else "bfgs"
    optimizer_fmax = optimizer_ase_dict.get("fmax", 0.05)
    optimizer_steps = optimizer_ase_dict.get("steps", 200)
    input_xyz_name = os.path.basename(args.xyz_file) if args.xyz_file else None
    input_xyz_path = resolve_run_path(run_dir, input_xyz_name) if input_xyz_name else None
    optimizer_input_xyz = resume_xyz_path or input_xyz_path
    output_xyz_setting = (
        optimizer_config.output_xyz if optimizer_config else None
    ) or "ase_optimized.xyz"
    output_xyz_path = resolve_run_path(run_dir, output_xyz_setting)
    pyscf_chkfile = None
    if scf_config and scf_config.get("chkfile"):
        pyscf_chkfile = resolve_run_path(run_dir, scf_config.get("chkfile"))
    checkpoint_common = _build_optimization_checkpoint_common(
        context=context,
        run_dir=run_dir,
        input_xyz_path=input_xyz_path,
        optimizer_input_xyz=optimizer_input_xyz,
        resume_xyz_path=resume_xyz_path,
        output_xyz_path=output_xyz_path,
        pyscf_chkfile=pyscf_chkfile,
        optimizer_name=optimizer_name,
        optimizer_fmax=optimizer_fmax,
        optimizer_steps=optimizer_steps,
        snapshot_mode=snapshot_mode,
        snapshot_write_steps=snapshot_write_steps,
        snapshot_write_last=snapshot_write_last,
        snapshot_paths=snapshot_paths,
        charge=charge,
        spin=spin,
        multiplicity=multiplicity,
    )
    step_callback = _build_optimization_step_callback(
        n_steps=n_steps,
        last_metadata_write=last_metadata_write,
        io_write_interval_steps=io_write_interval_steps,
        io_write_interval_seconds=io_write_interval_seconds,
        snapshot_interval_steps=snapshot_interval_steps,
        optimization_metadata=optimization_metadata,
        run_metadata_path=run_metadata_path,
        metadata_recorder=metadata_recorder,
        checkpoint_base=checkpoint_base,
        checkpoint_path=checkpoint_path,
        checkpoint_common=checkpoint_common,
    )
    return {
        "basis": basis,
        "xc": xc,
        "scf_config": scf_config,
        "solvent_name": solvent_name,
        "solvent_model": solvent_model,
        "dispersion_model": dispersion_model,
        "optimizer_ase_dict": optimizer_ase_dict,
        "optimizer_mode": optimizer_mode,
        "single_point_enabled": single_point_enabled,
        "profiling_enabled": profiling_enabled,
        "freq_scf_config": freq_scf_config,
        "run_dir": run_dir,
        "log_path": log_path,
        "optimized_xyz_path": optimized_xyz_path,
        "run_metadata_path": run_metadata_path,
        "frequency_output_path": frequency_output_path,
        "irc_output_path": irc_output_path,
        "event_log_path": event_log_path,
        "run_id": run_id,
        "irc_config": irc_config,
        "checkpoint_path": checkpoint_path,
        "mol": mol,
        "mf": mf,
        "charge": charge,
        "spin": spin,
        "multiplicity": multiplicity,
        "optimization_metadata": optimization_metadata,
        "run_start": run_start,
        "n_steps": n_steps,
        "n_steps_source": n_steps_source,
        "checkpoint_base": checkpoint_base,
        "snapshot_interval_steps": snapshot_interval_steps,
        "snapshot_mode": snapshot_mode,
        "snapshot_write_steps": snapshot_write_steps,
        "snapshot_write_last": snapshot_write_last,
        "snapshot_dir": snapshot_dir,
        "irc_forward_steps_snapshot": irc_forward_steps_snapshot,
        "irc_reverse_steps_snapshot": irc_reverse_steps_snapshot,
        "irc_forward_last_snapshot": irc_forward_last_snapshot,
        "irc_reverse_last_snapshot": irc_reverse_last_snapshot,
        "input_xyz_path": input_xyz_path,
        "optimizer_input_xyz": optimizer_input_xyz,
        "output_xyz_path": output_xyz_path,
        "checkpoint_common": checkpoint_common,
        "step_callback": step_callback,
        "engine_adapter": engine_adapter,
        "metadata_recorder": metadata_recorder,
    }


def run_optimization_stage(
    args,
    context: RunContext,
    molecule_context: MoleculeContext,
    memory_mb,
    memory_limit_status,
    memory_limit_enforced,
    openmp_available,
    effective_threads,
    queue_update_fn,
    engine_adapter: WorkflowEngineAdapter = DEFAULT_ENGINE_ADAPTER,
    metadata_recorder: RunMetadataRecorder = DEFAULT_METADATA_RECORDER,
):
    stage_context = _build_optimization_stage_context(context)
    runtime = _prepare_optimization_runtime_state(
        args=args,
        context=stage_context,
        engine_adapter=engine_adapter,
        metadata_recorder=metadata_recorder,
        molecule_context=molecule_context,
        memory_mb=memory_mb,
        memory_limit_status=memory_limit_status,
        openmp_available=openmp_available,
        effective_threads=effective_threads,
    )
    n_steps_source = runtime["n_steps_source"]

    try:
        (
            mol_optimized,
            optimized_atom_spec,
            n_steps_value,
            n_steps_source,
        ) = _run_geometry_optimization_phase(
            args=args,
            engine_adapter=runtime["engine_adapter"],
            run_dir=runtime["run_dir"],
            input_xyz_path=runtime["input_xyz_path"],
            optimizer_input_xyz=runtime["optimizer_input_xyz"],
            output_xyz_path=runtime["output_xyz_path"],
            charge=runtime["charge"],
            spin=runtime["spin"],
            multiplicity=runtime["multiplicity"],
            basis=runtime["basis"],
            xc=runtime["xc"],
            scf_config=runtime["scf_config"],
            solvent_model=runtime["solvent_model"],
            solvent_name=runtime["solvent_name"],
            eps=stage_context["eps"],
            dispersion_model=runtime["dispersion_model"],
            verbose=stage_context["verbose"],
            memory_mb=memory_mb,
            optimizer_ase_dict=runtime["optimizer_ase_dict"],
            optimizer_mode=runtime["optimizer_mode"],
            constraints=stage_context["constraints"],
            profiling_enabled=runtime["profiling_enabled"],
            step_callback=runtime["step_callback"],
            optimization_metadata=runtime["optimization_metadata"],
            n_steps=runtime["n_steps"],
        )
    except Exception as exc:
        logging.exception("Geometry optimization failed.")
        _handle_optimization_failure(
            error=exc,
            metadata_recorder=runtime["metadata_recorder"],
            checkpoint_base=runtime["checkpoint_base"],
            checkpoint_path=runtime["checkpoint_path"],
            checkpoint_common=runtime["checkpoint_common"],
            optimization_metadata=runtime["optimization_metadata"],
            run_start=runtime["run_start"],
            mf=runtime["mf"],
            mol=runtime["mol"],
            memory_limit_enforced=memory_limit_enforced,
            run_metadata_path=runtime["run_metadata_path"],
            event_log_path=runtime["event_log_path"],
            run_id=runtime["run_id"],
            run_dir=runtime["run_dir"],
            queue_update_fn=queue_update_fn,
            n_steps=runtime["n_steps"],
            n_steps_source=n_steps_source,
        )
        raise

    elapsed_seconds, n_steps_value = _mark_optimization_completed(
        mol_optimized=mol_optimized,
        optimized_atom_spec=optimized_atom_spec,
        optimized_xyz_path=runtime["optimized_xyz_path"],
        n_steps_value=n_steps_value,
        n_steps=runtime["n_steps"],
        n_steps_source=n_steps_source,
        checkpoint_base=runtime["checkpoint_base"],
        checkpoint_path=runtime["checkpoint_path"],
        checkpoint_common=runtime["checkpoint_common"],
        optimization_metadata=runtime["optimization_metadata"],
        run_start=runtime["run_start"],
    )

    post_plan = _run_frequency_and_plan_after_optimization(
        context=stage_context,
        engine_adapter=runtime["engine_adapter"],
        mol_optimized=mol_optimized,
        freq_scf_config=runtime["freq_scf_config"],
        optimizer_ase_dict=runtime["optimizer_ase_dict"],
        optimizer_mode=runtime["optimizer_mode"],
        multiplicity=runtime["multiplicity"],
        run_dir=runtime["run_dir"],
        frequency_output_path=runtime["frequency_output_path"],
        verbose=stage_context["verbose"],
        memory_mb=memory_mb,
        profiling_enabled=runtime["profiling_enabled"],
        optimization_metadata=runtime["optimization_metadata"],
        irc_output_path=runtime["irc_output_path"],
    )

    post_results = _run_irc_and_single_point_after_optimization(
        run_single_point=post_plan["run_single_point"],
        single_point_enabled=runtime["single_point_enabled"],
        context=stage_context,
        engine_adapter=runtime["engine_adapter"],
        mol_optimized=mol_optimized,
        output_xyz_path=runtime["output_xyz_path"],
        run_dir=runtime["run_dir"],
        charge=runtime["charge"],
        spin=runtime["spin"],
        multiplicity=runtime["multiplicity"],
        verbose=stage_context["verbose"],
        memory_mb=memory_mb,
        optimizer_ase_dict=runtime["optimizer_ase_dict"],
        optimizer_mode=runtime["optimizer_mode"],
        profiling_enabled=runtime["profiling_enabled"],
        irc_status=post_plan["irc_status"],
        irc_config=runtime["irc_config"],
        checkpoint_base=runtime["checkpoint_base"],
        checkpoint_path=runtime["checkpoint_path"],
        snapshot_interval_steps=runtime["snapshot_interval_steps"],
        snapshot_mode=runtime["snapshot_mode"],
        snapshot_write_steps=runtime["snapshot_write_steps"],
        snapshot_write_last=runtime["snapshot_write_last"],
        snapshot_dir=runtime["snapshot_dir"],
        irc_forward_steps_snapshot=runtime["irc_forward_steps_snapshot"],
        irc_reverse_steps_snapshot=runtime["irc_reverse_steps_snapshot"],
        irc_forward_last_snapshot=runtime["irc_forward_last_snapshot"],
        irc_reverse_last_snapshot=runtime["irc_reverse_last_snapshot"],
        ts_energy_ev=post_plan["ts_energy_ev"],
        irc_output_path=runtime["irc_output_path"],
        log_path=runtime["log_path"],
        optimization_metadata=runtime["optimization_metadata"],
        last_scf_energy=post_plan["last_scf_energy"],
        last_scf_converged=post_plan["last_scf_converged"],
    )

    _finalize_optimization_stage_success(
        metadata_recorder=runtime["metadata_recorder"],
        optimization_metadata=runtime["optimization_metadata"],
        n_steps_value=n_steps_value,
        n_steps_source=n_steps_source,
        mf=runtime["mf"],
        mol_optimized=mol_optimized,
        elapsed_seconds=elapsed_seconds,
        final_sp_energy=post_results["final_sp_energy"],
        final_sp_converged=post_results["final_sp_converged"],
        final_sp_cycles=post_results["final_sp_cycles"],
        memory_limit_enforced=memory_limit_enforced,
        checkpoint_base=runtime["checkpoint_base"],
        checkpoint_path=runtime["checkpoint_path"],
        checkpoint_common=runtime["checkpoint_common"],
        last_scf_energy=post_results["last_scf_energy"],
        last_scf_converged=post_results["last_scf_converged"],
        run_metadata_path=runtime["run_metadata_path"],
        context=stage_context,
        args=args,
        optimized_xyz_path=runtime["optimized_xyz_path"],
        frequency_payload=post_plan["frequency_payload"],
        irc_payload=post_results["irc_payload"],
        sp_result=post_results["sp_result"],
        event_log_path=runtime["event_log_path"],
        run_id=runtime["run_id"],
        run_dir=runtime["run_dir"],
        queue_update_fn=queue_update_fn,
    )
