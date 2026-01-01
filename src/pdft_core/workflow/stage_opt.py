import json
import logging
import os
import shutil
import time
import traceback
from datetime import datetime

from ..ase_backend import _run_ase_irc, _run_ase_optimizer
from ..queue import record_status_event
from ..run_opt_engine import (
    compute_frequencies,
    compute_imaginary_mode,
    compute_single_point_energy,
    load_xyz,
    run_capability_check,
)
from ..run_opt_logging import ensure_stream_newlines
from ..run_opt_metadata import (
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
from ..run_opt_resources import collect_environment_snapshot, ensure_parent_dir, resolve_run_path
from .utils import (
    _atoms_to_atom_spec,
    _frequency_units,
    _frequency_versions,
    _thermochemistry_payload,
)


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
