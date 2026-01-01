import logging
import os
import uuid
from datetime import datetime

from ..run_opt_config import (
    DEFAULT_EVENT_LOG_PATH,
    DEFAULT_FREQUENCY_PATH,
    DEFAULT_IRC_PATH,
    DEFAULT_IRC_PROFILE_CSV_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_OPTIMIZED_XYZ_PATH,
    DEFAULT_QCSCHEMA_OUTPUT_PATH,
    DEFAULT_RUN_METADATA_PATH,
    DEFAULT_SCAN_RESULT_CSV_PATH,
    DEFAULT_SCAN_RESULT_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
    DEFAULT_THREAD_COUNT,
    RunConfig,
)
from ..run_opt_engine import (
    load_xyz,
    normalize_xc_functional,
    select_ks_type,
    total_electron_count,
)
from ..run_opt_metadata import write_checkpoint
from ..run_opt_resources import (
    create_run_directory,
    ensure_parent_dir,
    format_log_path,
    resolve_run_path,
)
from .types import MoleculeContext, RunContext
from .utils import (
    _is_vacuum_solvent,
    _normalize_calculation_mode,
    _normalize_optimizer_mode,
    _normalize_scan_mode,
    _normalize_stage_flags,
    _resolve_run_identity,
    _resolve_scf_chkfile,
    _warn_missing_chkfile,
)


def prepare_run_context(args, config: RunConfig, config_raw) -> RunContext:
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
    qcschema_output_path = resolve_run_path(
        run_dir, config.qcschema_output_file or DEFAULT_QCSCHEMA_OUTPUT_PATH
    )
    frequency_output_path = resolve_run_path(
        run_dir, config.frequency_file or DEFAULT_FREQUENCY_PATH
    )
    irc_output_path = resolve_run_path(run_dir, config.irc_file or DEFAULT_IRC_PATH)
    irc_profile_csv_path = resolve_run_path(
        run_dir, config.irc_profile_csv_file or DEFAULT_IRC_PROFILE_CSV_PATH
    )
    scan_result_path = resolve_run_path(run_dir, DEFAULT_SCAN_RESULT_PATH)
    scan_result_csv_path = resolve_run_path(
        run_dir, config.scan_result_csv_file or DEFAULT_SCAN_RESULT_CSV_PATH
    )
    ensure_parent_dir(log_path)
    ensure_parent_dir(optimized_xyz_path)
    ensure_parent_dir(run_metadata_path)
    ensure_parent_dir(qcschema_output_path)
    ensure_parent_dir(frequency_output_path)
    ensure_parent_dir(irc_output_path)
    ensure_parent_dir(irc_profile_csv_path)
    ensure_parent_dir(scan_result_path)
    ensure_parent_dir(scan_result_csv_path)

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
        "qcschema_output_path": qcschema_output_path,
        "frequency_output_path": frequency_output_path,
        "irc_output_path": irc_output_path,
        "irc_profile_csv_path": irc_profile_csv_path,
        "scan_result_path": scan_result_path,
        "scan_result_csv_path": scan_result_csv_path,
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


def build_molecule_context(args, context: RunContext, memory_mb) -> MoleculeContext:
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
