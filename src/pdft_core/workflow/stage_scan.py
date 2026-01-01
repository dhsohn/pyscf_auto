import csv
import itertools
import json
import logging
import os
import time
from datetime import datetime

from ..ase_backend import _run_ase_optimizer
from ..queue import record_status_event
from ..run_opt_engine import compute_single_point_energy, run_capability_check
from ..run_opt_metadata import (
    collect_git_metadata,
    compute_file_hash,
    compute_text_hash,
    get_package_version,
    write_run_metadata,
)
from ..run_opt_resources import collect_environment_snapshot, resolve_run_path
from .events import finalize_metadata
from .types import MoleculeContext, RunContext
from .utils import (
    _apply_scan_geometry,
    _atoms_to_atom_spec,
    _build_scan_constraints,
    _dimension_key,
    _merge_constraints,
    _parse_scan_dimensions,
)


def run_scan_stage(
    args,
    context: RunContext,
    molecule_context: MoleculeContext,
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
    scan_result_csv_path = context["scan_result_csv_path"]
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
        "scan_result_csv_file": scan_result_csv_path,
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
    if scan_result_csv_path:
        logging.info("Scan results CSV file: %s", scan_result_csv_path)

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
            if scan_result_csv_path:
                value_key_map = {}
                for item in results:
                    values = item.get("values") or {}
                    for key in values:
                        if key == "index":
                            continue
                        normalized_key = key.replace(":", "_").replace(",", "_")
                        value_key_map.setdefault(normalized_key, key)
                fieldnames = [
                    "index",
                    *value_key_map.keys(),
                    "energy",
                    "converged",
                    "cycles",
                    "optimizer_steps",
                    "input_xyz",
                    "output_xyz",
                ]
                with open(scan_result_csv_path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in results:
                        values = item.get("values") or {}
                        row = {"index": item.get("index")}
                        for normalized_key, original_key in value_key_map.items():
                            row[normalized_key] = values.get(original_key)
                        row.update(
                            {
                                "energy": item.get("energy"),
                                "converged": item.get("converged"),
                                "cycles": item.get("cycles"),
                                "optimizer_steps": item.get("optimizer_steps"),
                                "input_xyz": item.get("input_xyz"),
                                "output_xyz": item.get("output_xyz"),
                            }
                        )
                        writer.writerow(row)
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
