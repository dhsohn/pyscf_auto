import concurrent.futures
import copy
import csv
import itertools
import json
import logging
import math
import os
import time
import traceback
from datetime import datetime

from ase_backend import _run_ase_optimizer
from run_queue import record_status_event
from run_opt_engine import compute_single_point_energy, run_capability_check
from run_opt_metadata import (
    collect_git_metadata,
    compute_file_hash,
    compute_text_hash,
    get_package_version,
    write_run_metadata,
)
from run_opt_resources import (
    apply_thread_settings,
    collect_environment_snapshot,
    ensure_parent_dir,
    resolve_run_path,
)
from .events import finalize_metadata
from .types import MoleculeContext, RunContext
from .utils import (
    _apply_scan_geometry,
    _atoms_to_atom_spec,
    _build_scan_constraints,
    _dimension_key,
    _merge_constraints,
    _parse_scan_dimensions,
    _resolve_d3_params,
    _seed_scf_checkpoint,
)


SCAN_EXECUTORS = ("serial", "local", "manifest")


def _normalize_scan_executor(value):
    if not value:
        return "local"
    normalized = str(value).strip().lower()
    if normalized not in SCAN_EXECUTORS:
        raise ValueError(
            "Unsupported scan executor '{value}'. Use serial, local, or manifest.".format(
                value=value
            )
        )
    return normalized


def _resolve_scan_worker_settings(scan_config, total_points, thread_count):
    cpu_count = os.cpu_count() or 1
    threads_per_worker = scan_config.get("threads_per_worker") if scan_config else None
    if threads_per_worker is None:
        threads_per_worker = thread_count if thread_count else 1
    if not isinstance(threads_per_worker, int) or isinstance(threads_per_worker, bool):
        raise ValueError("scan.threads_per_worker must be a positive integer.")
    if threads_per_worker < 1:
        raise ValueError("scan.threads_per_worker must be a positive integer.")
    if threads_per_worker > cpu_count:
        logging.warning(
            "scan.threads_per_worker=%s exceeds CPU count %s; clamping to %s.",
            threads_per_worker,
            cpu_count,
            cpu_count,
        )
        threads_per_worker = cpu_count
    max_workers = scan_config.get("max_workers") if scan_config else None
    if max_workers is None:
        max_workers = max(1, cpu_count // threads_per_worker)
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise ValueError("scan.max_workers must be a positive integer.")
    if max_workers < 1:
        raise ValueError("scan.max_workers must be a positive integer.")
    if total_points:
        max_workers = min(max_workers, total_points)
    max_parallel = max(1, cpu_count // threads_per_worker)
    if max_workers > max_parallel:
        logging.warning(
            "scan.max_workers=%s with threads_per_worker=%s oversubscribes CPU (%s); "
            "using %s workers.",
            max_workers,
            threads_per_worker,
            cpu_count,
            max_parallel,
        )
        max_workers = max_parallel
    return max_workers, threads_per_worker


def _resolve_scan_threads_per_worker(scan_config, thread_count):
    threads_per_worker = scan_config.get("threads_per_worker") if scan_config else None
    if threads_per_worker is None:
        threads_per_worker = thread_count if thread_count else 1
    if not isinstance(threads_per_worker, int) or isinstance(threads_per_worker, bool):
        raise ValueError("scan.threads_per_worker must be a positive integer.")
    if threads_per_worker < 1:
        raise ValueError("scan.threads_per_worker must be a positive integer.")
    return threads_per_worker


def _resolve_scan_batch_size(scan_config, total_points, max_workers):
    batch_size = scan_config.get("batch_size") if scan_config else None
    if batch_size is None:
        if not total_points or not max_workers:
            return 1
        target = math.ceil(total_points / (max_workers * 4))
        return max(1, min(25, target))
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError("scan.batch_size must be a positive integer.")
    if batch_size < 1:
        raise ValueError("scan.batch_size must be a positive integer.")
    if total_points:
        return min(batch_size, total_points)
    return batch_size


def _build_scan_batches(scan_points, batch_size):
    batches = []
    for start in range(0, len(scan_points), batch_size):
        batch = []
        for index in range(start, min(start + batch_size, len(scan_points))):
            batch.append((index, scan_points[index]))
        batches.append(batch)
    return batches


def _scan_point_dir(scan_dir, index):
    return os.path.join(scan_dir, "points", f"point_{index:03d}")


def _scan_point_result_path(scan_dir, index):
    return os.path.join(_scan_point_dir(scan_dir, index), "result.json")


def _build_point_label(dimensions, values, index):
    point_label = {"index": index}
    for dimension, value in zip(dimensions, values, strict=True):
        point_label[_dimension_key(dimension)] = value
    return point_label


def _prepare_point_scf_config(base_scf_config, run_dir, parallel):
    scf_config = copy.deepcopy(base_scf_config or {})
    if not scf_config:
        return scf_config
    chkfile = scf_config.get("chkfile")
    if chkfile and parallel and os.path.isabs(str(chkfile)):
        scf_config["chkfile"] = os.path.join(run_dir, os.path.basename(str(chkfile)))
    return scf_config


def _write_point_result(path, result):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def _write_scan_results(results_by_index, scan_result_path, scan_result_csv_path):
    results = [results_by_index[index] for index in sorted(results_by_index)]
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
    return results


def _run_scan_point(
    *,
    index,
    values,
    dimensions,
    scan_mode,
    xyz_file,
    atoms_template=None,
    seed_chkfile=None,
    scan_dir,
    run_dir,
    charge,
    spin,
    multiplicity,
    basis,
    xc,
    scf_config,
    solvent_name,
    solvent_model,
    solvent_eps,
    dispersion_model,
    optimizer_mode,
    optimizer_ase_dict,
    constraints,
    verbose,
    memory_mb,
    thread_count,
    parallel,
    profiling_enabled=False,
):
    from ase.io import read as ase_read
    from ase.io import write as ase_write
    from pyscf import gto

    if thread_count and parallel:
        apply_thread_settings(thread_count)
    point_label = _build_point_label(dimensions, values, index)
    point_run_dir = run_dir
    os.makedirs(point_run_dir, exist_ok=True)
    input_xyz_path = resolve_run_path(scan_dir, f"scan_{index:03d}_input.xyz")
    output_xyz_path = None
    if atoms_template is None:
        atoms = ase_read(xyz_file).copy()
    else:
        atoms = atoms_template.copy()
    _apply_scan_geometry(atoms, dimensions, values)
    ase_write(input_xyz_path, atoms)
    point_scf_config = _prepare_point_scf_config(scf_config, point_run_dir, parallel)
    if seed_chkfile and point_scf_config.get("chkfile"):
        target_chkfile = resolve_run_path(
            point_run_dir, point_scf_config.get("chkfile")
        )
        _seed_scf_checkpoint(
            seed_chkfile,
            target_chkfile,
            label=f"scan point {index}",
        )
    n_steps = None
    optimizer_profile = None
    if scan_mode == "optimization":
        output_xyz_path = resolve_run_path(
            scan_dir, f"scan_{index:03d}_optimized.xyz"
        )
        scan_constraints = _build_scan_constraints(dimensions, values)
        merged_constraints = _merge_constraints(constraints, scan_constraints)
        opt_result = _run_ase_optimizer(
            input_xyz_path,
            output_xyz_path,
            point_run_dir,
            charge,
            spin,
            multiplicity,
            basis,
            xc,
            point_scf_config,
            solvent_model.lower() if solvent_model else None,
            solvent_name,
            solvent_eps,
            dispersion_model,
            verbose,
            memory_mb,
            optimizer_ase_dict,
            optimizer_mode,
            merged_constraints,
            profiling_enabled=profiling_enabled,
        )
        n_steps = opt_result.get("n_steps")
        optimizer_profile = opt_result.get("profiling") if profiling_enabled else None
        atoms = ase_read(output_xyz_path)
    atom_spec = _atoms_to_atom_spec(atoms)
    mol_scan = gto.M(
        atom=atom_spec,
        basis=basis,
        charge=charge,
        spin=spin,
    )
    if memory_mb:
        mol_scan.max_memory = memory_mb
    scf_result = compute_single_point_energy(
        mol_scan,
        basis,
        xc,
        point_scf_config,
        solvent_model if solvent_name else None,
        solvent_name,
        solvent_eps,
        dispersion_model,
        _resolve_d3_params(optimizer_ase_dict),
        verbose,
        memory_mb,
        run_dir=point_run_dir,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=False,
        profiling_enabled=profiling_enabled,
    )
    point_result = {
        "index": index,
        "values": point_label,
        "mode": scan_mode,
        "energy": scf_result.get("energy") if scf_result else None,
        "converged": scf_result.get("converged") if scf_result else None,
        "cycles": scf_result.get("cycles") if scf_result else None,
        "optimizer_steps": n_steps,
        "profiling": {
            "optimizer": optimizer_profile,
            "single_point": scf_result.get("profiling") if scf_result else None,
        }
        if profiling_enabled
        else None,
        "input_xyz": input_xyz_path,
        "output_xyz": output_xyz_path,
    }
    point_result_path = _scan_point_result_path(scan_dir, index)
    _write_point_result(point_result_path, point_result)
    return point_result


def _run_scan_batch(
    *,
    batch_items,
    dimensions,
    scan_mode,
    xyz_file,
    seed_chkfile,
    scan_dir,
    run_dir,
    charge,
    spin,
    multiplicity,
    basis,
    xc,
    scf_config,
    solvent_name,
    solvent_model,
    solvent_eps,
    dispersion_model,
    optimizer_mode,
    optimizer_ase_dict,
    constraints,
    verbose,
    memory_mb,
    thread_count,
    parallel,
    profiling_enabled=False,
):
    results = []
    errors = []
    for index, values in batch_items:
        point_run_dir = _scan_point_dir(scan_dir, index) if parallel else run_dir
        try:
            result = _run_scan_point(
                index=index,
                values=values,
                dimensions=dimensions,
                scan_mode=scan_mode,
                xyz_file=xyz_file,
                seed_chkfile=seed_chkfile,
                scan_dir=scan_dir,
                run_dir=point_run_dir,
                charge=charge,
                spin=spin,
                multiplicity=multiplicity,
                basis=basis,
                xc=xc,
                scf_config=scf_config,
                solvent_name=solvent_name,
                solvent_model=solvent_model,
                solvent_eps=solvent_eps,
                dispersion_model=dispersion_model,
                optimizer_mode=optimizer_mode,
                optimizer_ase_dict=optimizer_ase_dict,
                constraints=constraints,
                verbose=verbose,
                memory_mb=memory_mb,
                thread_count=thread_count,
                parallel=parallel,
                profiling_enabled=profiling_enabled,
            )
            results.append(result)
        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return {"results": results, "errors": errors}


def _write_scan_manifest(
    *,
    manifest_path,
    scan_dir,
    run_dir,
    xyz_file,
    scan_mode,
    dimensions,
    scan_points,
    settings,
):
    points = []
    for index, values in enumerate(scan_points):
        point_dir = _scan_point_dir(scan_dir, index)
        point_result_path = _scan_point_result_path(scan_dir, index)
        points.append(
            {
                "index": index,
                "values": list(values),
                "label": _build_point_label(dimensions, values, index),
                "work_dir": point_dir,
                "input_xyz": resolve_run_path(scan_dir, f"scan_{index:03d}_input.xyz"),
                "output_xyz": resolve_run_path(
                    scan_dir, f"scan_{index:03d}_optimized.xyz"
                )
                if scan_mode == "optimization"
                else None,
                "result_file": point_result_path,
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "run_dir": run_dir,
        "scan_dir": scan_dir,
        "xyz_file": xyz_file,
        "scan_mode": scan_mode,
        "dimensions": dimensions,
        "executor": "manifest",
        "settings": settings,
        "points": points,
        "command_template": [
            "dftflow",
            "scan-point",
            "--manifest",
            manifest_path,
            "--index",
            "{index}",
        ],
    }
    ensure_parent_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_scan_point_from_manifest(manifest_path, index):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    points = manifest.get("points") or []
    target = None
    for item in points:
        if item.get("index") == index:
            target = item
            break
    if target is None:
        raise ValueError(f"Scan point {index} not found in manifest.")
    settings = manifest.get("settings") or {}
    thread_count = settings.get("threads_per_worker") or settings.get("thread_count") or 1
    return _run_scan_point(
        index=index,
        values=target.get("values") or [],
        dimensions=manifest.get("dimensions") or [],
        scan_mode=manifest.get("scan_mode"),
        xyz_file=manifest.get("xyz_file"),
        seed_chkfile=settings.get("seed_chkfile"),
        scan_dir=manifest.get("scan_dir"),
        run_dir=target.get("work_dir") or manifest.get("run_dir"),
        charge=settings.get("charge"),
        spin=settings.get("spin"),
        multiplicity=settings.get("multiplicity"),
        basis=settings.get("basis"),
        xc=settings.get("xc"),
        scf_config=settings.get("scf_config"),
        solvent_name=settings.get("solvent"),
        solvent_model=settings.get("solvent_model"),
        solvent_eps=settings.get("solvent_eps"),
        dispersion_model=settings.get("dispersion"),
        optimizer_mode=settings.get("optimizer_mode"),
        optimizer_ase_dict=settings.get("optimizer_ase"),
        constraints=settings.get("constraints"),
        verbose=bool(settings.get("verbose")),
        memory_mb=settings.get("memory_mb"),
        thread_count=thread_count,
        parallel=True,
        profiling_enabled=bool(settings.get("profiling_enabled")),
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
    profiling_enabled = bool(context.get("profiling_enabled"))
    scan_write_interval_points = context.get("scan_write_interval_points", 1)

    scan_dir = resolve_run_path(run_dir, "scan")
    os.makedirs(scan_dir, exist_ok=True)
    points_dir = os.path.join(scan_dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    scan_executor = _normalize_scan_executor(scan_config.get("executor"))
    scan_max_workers = None
    scan_batch_size = None
    manifest_setting = scan_config.get("manifest_file")
    if manifest_setting:
        scan_manifest_path = resolve_run_path(run_dir, manifest_setting)
    else:
        scan_manifest_path = resolve_run_path(scan_dir, "scan_manifest.json")
    scan_thread_count = thread_count
    if scan_executor == "local":
        scan_max_workers, scan_thread_count = _resolve_scan_worker_settings(
            scan_config,
            len(scan_points),
            thread_count,
        )
        scan_batch_size = _resolve_scan_batch_size(
            scan_config,
            len(scan_points),
            scan_max_workers,
        )
    elif scan_executor == "manifest":
        scan_thread_count = _resolve_scan_threads_per_worker(scan_config, thread_count)

    if scan_executor in ("local", "manifest"):
        if thread_count and thread_count != scan_thread_count:
            logging.warning(
                "Scan executor %s overrides thread_count=%s to threads_per_worker=%s.",
                scan_executor,
                thread_count,
                scan_thread_count,
            )
        thread_status = apply_thread_settings(scan_thread_count)
        effective_threads = thread_status.get("effective_threads")
        openmp_available = thread_status.get("openmp_available")

    calc_basis = context["basis"]
    calc_xc = context["xc"]
    calc_scf_config = context["scf_config"]
    calc_solvent_name = context["solvent_name"]
    calc_solvent_model = context["solvent_model"]
    calc_eps = context["eps"]
    calc_dispersion_model = context["dispersion_model"]
    optimizer_mode = context["optimizer_mode"]
    optimizer_ase_dict = context["optimizer_ase_dict"]
    constraints = context["constraints"]
    scan_seed_chkfile = (
        calc_scf_config.get("chkfile") if isinstance(calc_scf_config, dict) else None
    )

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
        "thread_count": scan_thread_count,
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
        "environment": collect_environment_snapshot(scan_thread_count),
        "git": collect_git_metadata(os.getcwd()),
        "versions": {
            "ase": get_package_version("ase"),
            "pyscf": get_package_version("pyscf"),
            "dftd3": get_package_version("dftd3"),
            "dftd4": get_package_version("dftd4"),
        },
    }
    scan_summary["scan"].update(
        {
            "executor": scan_executor,
            "max_workers": scan_max_workers if scan_executor == "local" else None,
            "manifest_file": scan_manifest_path if scan_executor == "manifest" else None,
            "point_result_dir": points_dir,
            "threads_per_worker": scan_thread_count,
            "batch_size": scan_batch_size if scan_executor == "local" else None,
            "write_interval_points": scan_write_interval_points,
        }
    )
    scan_summary["run_updated_at"] = datetime.now().isoformat()
    write_run_metadata(run_metadata_path, scan_summary)
    record_status_event(
        event_log_path,
        run_id,
        run_dir,
        "running",
        previous_status=context.get("previous_status"),
    )

    if scan_executor != "manifest":
        mol = molecule_context["mol"]
        run_capability_check(
            mol,
            calc_basis,
            calc_xc,
            calc_scf_config,
            calc_solvent_model if calc_solvent_name else None,
            calc_solvent_name,
            calc_eps,
            calc_dispersion_model,
            "none",
            dispersion_params=_resolve_d3_params(optimizer_ase_dict),
            require_hessian=False,
            verbose=verbose,
            memory_mb=memory_mb,
            optimizer_mode=optimizer_mode,
            multiplicity=multiplicity,
        )
    else:
        logging.info("Manifest executor selected; skipping capability check.")

    logging.info("Starting scan (%s mode) with %d points.", scan_mode, len(scan_points))
    logging.info("Scan executor: %s", scan_executor)
    if scan_executor == "local":
        logging.info("Scan workers: %s", scan_max_workers)
        if scan_batch_size:
            logging.info("Scan batch size: %s", scan_batch_size)
    if scan_thread_count:
        logging.info("Threads per worker: %s", scan_thread_count)
    if memory_mb:
        logging.info("Memory target: %s MB (PySCF max_memory)", memory_mb)
    if log_path:
        logging.info("Log file: %s", log_path)
    if scan_result_path:
        logging.info("Scan results file: %s", scan_result_path)
    if scan_result_csv_path:
        logging.info("Scan results CSV file: %s", scan_result_csv_path)

    run_start = time.perf_counter()
    results_by_index = {}
    last_scan_write = {"count": 0}

    def _maybe_write_scan_results(force=False):
        if not results_by_index:
            return
        completed = len(results_by_index)
        if (
            force
            or scan_write_interval_points <= 1
            or completed - last_scan_write["count"] >= scan_write_interval_points
        ):
            _write_scan_results(results_by_index, scan_result_path, scan_result_csv_path)
            scan_summary["scan"]["completed_points"] = completed
            scan_summary["run_updated_at"] = datetime.now().isoformat()
            write_run_metadata(run_metadata_path, scan_summary)
            last_scan_write["count"] = completed
    try:
        if scan_executor == "manifest":
            manifest_settings = {
                "basis": calc_basis,
                "xc": calc_xc,
                "scf_config": calc_scf_config,
                "seed_chkfile": scan_seed_chkfile,
                "solvent": calc_solvent_name,
                "solvent_model": calc_solvent_model if calc_solvent_name else None,
                "solvent_eps": calc_eps,
                "dispersion": calc_dispersion_model,
                "optimizer_mode": optimizer_mode,
                "optimizer_ase": optimizer_ase_dict,
                "constraints": constraints,
                "charge": charge,
                "spin": spin,
                "multiplicity": multiplicity,
                "memory_mb": memory_mb,
                "verbose": verbose,
                "thread_count": scan_thread_count,
                "threads_per_worker": scan_thread_count,
                "write_interval_points": scan_write_interval_points,
                "profiling_enabled": profiling_enabled,
            }
            _write_scan_manifest(
                manifest_path=scan_manifest_path,
                scan_dir=scan_dir,
                run_dir=run_dir,
                xyz_file=args.xyz_file,
                scan_mode=scan_mode,
                dimensions=dimensions,
                scan_points=scan_points,
                settings=manifest_settings,
            )
            logging.info("Wrote scan manifest to %s", scan_manifest_path)
            scan_summary["scan"]["completed_points"] = 0
            scan_summary["summary"] = {
                "elapsed_seconds": time.perf_counter() - run_start,
                "n_points": 0,
                "planned_points": len(scan_points),
                "converged_points": 0,
                "final_energy": None,
                "manifest_only": True,
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
            return

        if scan_executor == "local":
            error = None
            batch_size = scan_batch_size or 1
            batches = _build_scan_batches(scan_points, batch_size)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=scan_max_workers
            ) as executor:
                futures = {}
                for batch in batches:
                    futures[
                        executor.submit(
                            _run_scan_batch,
                            batch_items=batch,
                            dimensions=dimensions,
                            scan_mode=scan_mode,
                            xyz_file=args.xyz_file,
                            seed_chkfile=scan_seed_chkfile,
                            scan_dir=scan_dir,
                            run_dir=run_dir,
                            charge=charge,
                            spin=spin,
                            multiplicity=multiplicity,
                            basis=calc_basis,
                            xc=calc_xc,
                            scf_config=calc_scf_config,
                            solvent_name=calc_solvent_name,
                            solvent_model=calc_solvent_model,
                            solvent_eps=calc_eps,
                            dispersion_model=calc_dispersion_model,
                            optimizer_mode=optimizer_mode,
                            optimizer_ase_dict=optimizer_ase_dict,
                            constraints=constraints,
                            verbose=verbose,
                            memory_mb=memory_mb,
                            thread_count=scan_thread_count,
                            parallel=True,
                            profiling_enabled=profiling_enabled,
                        )
                    ] = batch
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_result = future.result()
                    except Exception as exc:
                        logging.exception("Scan batch failed.")
                        error = exc
                        continue
                    for point_result in batch_result.get("results", []):
                        results_by_index[point_result["index"]] = point_result
                    for failure in batch_result.get("errors", []):
                        logging.error(
                            "Scan point %s failed: %s",
                            failure.get("index"),
                            failure.get("error"),
                        )
                        if failure.get("traceback"):
                            logging.error("%s", failure.get("traceback"))
                        if error is None:
                            error = RuntimeError(
                                "Scan point {index} failed: {error}".format(
                                    index=failure.get("index"),
                                    error=failure.get("error"),
                                )
                            )
                    if batch_result.get("results") or batch_result.get("errors"):
                        _maybe_write_scan_results(force=bool(batch_result.get("errors")))
            if error:
                _maybe_write_scan_results(force=True)
                raise error
        else:
            atoms_template = None
            if scan_executor == "serial":
                from ase.io import read as ase_read

                atoms_template = ase_read(args.xyz_file)
            for index, values in enumerate(scan_points):
                point_result = _run_scan_point(
                    index=index,
                    values=values,
                    dimensions=dimensions,
                    scan_mode=scan_mode,
                    xyz_file=args.xyz_file,
                    atoms_template=atoms_template,
                    seed_chkfile=scan_seed_chkfile,
                    scan_dir=scan_dir,
                    run_dir=run_dir,
                    charge=charge,
                    spin=spin,
                    multiplicity=multiplicity,
                    basis=calc_basis,
                    xc=calc_xc,
                    scf_config=calc_scf_config,
                    solvent_name=calc_solvent_name,
                    solvent_model=calc_solvent_model,
                    solvent_eps=calc_eps,
                    dispersion_model=calc_dispersion_model,
                    optimizer_mode=optimizer_mode,
                    optimizer_ase_dict=optimizer_ase_dict,
                    constraints=constraints,
                    verbose=verbose,
                    memory_mb=memory_mb,
                    thread_count=scan_thread_count,
                    parallel=False,
                    profiling_enabled=profiling_enabled,
                )
                results_by_index[index] = point_result
                _maybe_write_scan_results()
        results = _write_scan_results(
            results_by_index, scan_result_path, scan_result_csv_path
        )
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
