import csv
import json
import logging
import os
import time

from run_opt_metadata import format_xyz_comment, write_checkpoint, write_xyz_snapshot
from run_opt_resources import resolve_run_path
from .engine_adapter import WorkflowEngineAdapter
from .events import finalize_metadata
from .utils import (
    _atoms_to_atom_spec,
    _evaluate_irc_profile,
    _resolve_d3_params,
    _update_checkpoint_scf,
)


DEFAULT_ENGINE_ADAPTER = WorkflowEngineAdapter()


def _normalize_snapshot_settings(snapshot_interval_steps, snapshot_mode):
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


def _build_irc_snapshot_paths(run_dir):
    return {
        "snapshot_dir": resolve_run_path(run_dir, "snapshots"),
        "forward_steps_snapshot": resolve_run_path(
            run_dir, "snapshots/irc_forward_steps.xyz"
        ),
        "reverse_steps_snapshot": resolve_run_path(
            run_dir, "snapshots/irc_reverse_steps.xyz"
        ),
        "forward_last_snapshot": resolve_run_path(run_dir, "snapshots/irc_forward_last.xyz"),
        "reverse_last_snapshot": resolve_run_path(run_dir, "snapshots/irc_reverse_last.xyz"),
    }


def _load_checkpoint_base(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
            loaded = json.load(checkpoint_file)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return dict(loaded)


def _parse_nonnegative_step(value):
    try:
        step_value = int(value)
    except (TypeError, ValueError):
        return None
    return step_value if step_value >= 0 else None


def _prepare_irc_resume_state(
    *,
    resume_dir,
    checkpoint_base,
    run_dir,
    charge,
    spin,
    multiplicity,
):
    profile_cache = []
    profile_keys = set()
    if resume_dir and checkpoint_base.get("irc_profile"):
        profile_cache = list(checkpoint_base.get("irc_profile", []))
        for entry in profile_cache:
            profile_keys.add((entry.get("direction"), entry.get("step")))
    irc_last_snapshot_step = {"forward": -1, "reverse": -1}
    irc_last_checkpoint_step = {"forward": -1, "reverse": -1}
    irc_last_geometry_cache = {"forward": None, "reverse": None}
    irc_last_step_cache = {"forward": None, "reverse": None}

    resume_state = None
    if resume_dir and checkpoint_base:
        resume_state = {
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
                irc_last_snapshot_step[direction] = step_value
                irc_last_checkpoint_step[direction] = step_value
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
                resume_state[direction] = {
                    "step": step_value,
                    "xyz": resume_xyz,
                }

    return {
        "resume_state": resume_state,
        "profile_cache": profile_cache,
        "profile_keys": profile_keys,
        "irc_last_snapshot_step": irc_last_snapshot_step,
        "irc_last_checkpoint_step": irc_last_checkpoint_step,
        "irc_last_geometry_cache": irc_last_geometry_cache,
        "irc_last_step_cache": irc_last_step_cache,
    }


def _build_irc_callbacks(
    *,
    checkpoint_path,
    checkpoint_base,
    profile_cache,
    profile_keys,
    irc_last_snapshot_step,
    irc_last_checkpoint_step,
    irc_last_geometry_cache,
    irc_last_step_cache,
    snapshot_interval_steps,
    snapshot_mode,
    snapshot_write_steps,
    snapshot_write_last,
    charge,
    spin,
    multiplicity,
    snapshot_paths,
):
    def _persist_checkpoint():
        if not checkpoint_path:
            return
        write_checkpoint(checkpoint_path, checkpoint_base)

    def _record_irc_step(direction, step_index, atoms, energy_ev, energy_hartree):
        atom_spec = _atoms_to_atom_spec(atoms)
        irc_last_geometry_cache[direction] = atom_spec
        irc_last_step_cache[direction] = step_index
        steps_path = (
            snapshot_paths["forward_steps_snapshot"]
            if direction == "forward"
            else snapshot_paths["reverse_steps_snapshot"]
        )
        last_path = (
            snapshot_paths["forward_last_snapshot"]
            if direction == "forward"
            else snapshot_paths["reverse_last_snapshot"]
        )
        entry_key = (direction, step_index)
        if entry_key not in profile_keys:
            profile_keys.add(entry_key)
            profile_cache.append(
                {
                    "direction": direction,
                    "step": step_index,
                    "energy_ev": float(energy_ev),
                    "energy_hartree": float(energy_hartree),
                }
            )
        should_snapshot = (
            step_index - irc_last_snapshot_step[direction] >= snapshot_interval_steps
        )
        should_checkpoint = (
            irc_last_checkpoint_step[direction] < 0
            or step_index - irc_last_checkpoint_step[direction] >= snapshot_interval_steps
        )
        if should_snapshot and snapshot_mode != "none":
            comment = format_xyz_comment(
                charge=charge,
                spin=spin,
                multiplicity=multiplicity,
                extra=f"step={step_index} direction={direction}",
            )
            if snapshot_write_steps:
                write_xyz_snapshot(steps_path, atom_spec, comment=comment, append=True)
            if snapshot_write_last:
                write_xyz_snapshot(last_path, atom_spec, comment=comment)
            irc_last_snapshot_step[direction] = step_index
        if should_checkpoint:
            checkpoint_base.update(
                {
                    "last_stage": "irc",
                    "last_step": step_index,
                    "last_step_stage": "irc",
                    "last_step_direction": direction,
                    "last_geometry": atom_spec,
                    "snapshot_dir": snapshot_paths["snapshot_dir"],
                    "irc_direction": direction,
                    f"irc_{direction}_step": step_index,
                    f"irc_{direction}_last_geometry": atom_spec,
                }
            )
            if snapshot_write_steps:
                checkpoint_base["irc_forward_steps_xyz"] = snapshot_paths[
                    "forward_steps_snapshot"
                ]
                checkpoint_base["irc_reverse_steps_xyz"] = snapshot_paths[
                    "reverse_steps_snapshot"
                ]
            else:
                checkpoint_base.pop("irc_forward_steps_xyz", None)
                checkpoint_base.pop("irc_reverse_steps_xyz", None)
            if snapshot_write_last:
                checkpoint_base["last_geometry_xyz"] = last_path
                checkpoint_base[f"irc_{direction}_last_xyz"] = last_path
            else:
                checkpoint_base.pop("last_geometry_xyz", None)
                checkpoint_base.pop(f"irc_{direction}_last_xyz", None)
            checkpoint_base["irc_profile"] = profile_cache
            _persist_checkpoint()
            irc_last_checkpoint_step[direction] = step_index

    def _mark_direction_complete(direction, last_step):
        cached_step = irc_last_step_cache.get(direction)
        if cached_step is not None:
            last_step = cached_step
        atom_spec = irc_last_geometry_cache.get(direction)
        if atom_spec:
            comment = format_xyz_comment(
                charge=charge,
                spin=spin,
                multiplicity=multiplicity,
                extra=f"step={last_step} direction={direction}",
            )
            last_path = (
                snapshot_paths["forward_last_snapshot"]
                if direction == "forward"
                else snapshot_paths["reverse_last_snapshot"]
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
                    "snapshot_dir": snapshot_paths["snapshot_dir"],
                    "irc_direction": direction,
                    f"irc_{direction}_last_geometry": atom_spec,
                }
            )
            if snapshot_write_steps:
                checkpoint_base["irc_forward_steps_xyz"] = snapshot_paths[
                    "forward_steps_snapshot"
                ]
                checkpoint_base["irc_reverse_steps_xyz"] = snapshot_paths[
                    "reverse_steps_snapshot"
                ]
            else:
                checkpoint_base.pop("irc_forward_steps_xyz", None)
                checkpoint_base.pop("irc_reverse_steps_xyz", None)
            if snapshot_write_last:
                checkpoint_base["last_geometry_xyz"] = last_path
                checkpoint_base[f"irc_{direction}_last_xyz"] = last_path
            else:
                checkpoint_base.pop("last_geometry_xyz", None)
                checkpoint_base.pop(f"irc_{direction}_last_xyz", None)
        checkpoint_base["irc_profile"] = profile_cache
        checkpoint_base[f"irc_{direction}_completed"] = True
        if last_step is not None:
            checkpoint_base[f"irc_{direction}_step"] = last_step
        _persist_checkpoint()

    return _record_irc_step, _mark_direction_complete


def _run_ase_irc_with_callbacks(
    stage_context,
    profiling_enabled,
    step_callback,
    direction_callback,
    resume_state,
    engine_adapter,
):
    return engine_adapter.run_ase_irc(
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
        stage_context.get("mode_hessian"),
        stage_context["irc_steps"],
        stage_context["irc_step_size"],
        stage_context["irc_force_threshold"],
        profiling_enabled=profiling_enabled,
        step_callback=step_callback,
        direction_callback=direction_callback,
        resume_state=resume_state,
    )


def _build_irc_payload(
    stage_context,
    irc_result,
    profile,
    mode_profiling,
    profiling_enabled,
):
    irc_payload = {
        "status": "completed",
        "output_file": stage_context["irc_output_path"],
        "forward_xyz": irc_result.get("forward_xyz"),
        "reverse_xyz": irc_result.get("reverse_xyz"),
        "steps": stage_context["irc_steps"],
        "step_size": stage_context["irc_step_size"],
        "force_threshold": stage_context["irc_force_threshold"],
        "mode_eigenvalue": stage_context.get("mode_eigenvalue"),
        "profile": profile,
        "profile_csv_file": stage_context["irc_profile_csv_path"],
        "profiling": {
            "mode": mode_profiling,
            "irc": irc_result.get("profiling"),
        }
        if profiling_enabled
        else None,
    }
    irc_payload["assessment"] = _evaluate_irc_profile(irc_payload["profile"])
    return irc_payload


def _write_irc_profile_csv(irc_profile_csv_path, irc_payload):
    with open(irc_profile_csv_path, "w", encoding="utf-8", newline="") as handle:
        direction_assessment = (
            irc_payload.get("assessment", {}).get("details", {}).get("directions", {})
        )
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "direction",
                "step",
                "energy_ev",
                "energy_hartree",
                "direction_status",
                "direction_endpoint_energy_ev",
                "direction_min_energy_ev",
                "direction_drop_from_ts_ev",
                "direction_min_drop_from_ts_ev",
                "direction_endpoint_near_min",
            ],
        )
        writer.writeheader()
        for entry in irc_payload["profile"]:
            direction_detail = direction_assessment.get(entry.get("direction"), {})
            writer.writerow(
                {
                    "direction": entry.get("direction"),
                    "step": entry.get("step"),
                    "energy_ev": entry.get("energy_ev"),
                    "energy_hartree": entry.get("energy_hartree"),
                    "direction_status": direction_detail.get("status"),
                    "direction_endpoint_energy_ev": direction_detail.get("endpoint_energy_ev"),
                    "direction_min_energy_ev": direction_detail.get("min_energy_ev"),
                    "direction_drop_from_ts_ev": direction_detail.get(
                        "endpoint_drop_from_ts_ev"
                    ),
                    "direction_min_drop_from_ts_ev": direction_detail.get(
                        "min_drop_from_ts_ev"
                    ),
                    "direction_endpoint_near_min": direction_detail.get(
                        "endpoint_near_min"
                    ),
                }
            )


def _write_irc_outputs(stage_context, irc_payload):
    with open(stage_context["irc_output_path"], "w", encoding="utf-8") as handle:
        json.dump(irc_payload, handle, indent=2)
    _write_irc_profile_csv(stage_context["irc_profile_csv_path"], irc_payload)


def _record_irc_metadata(calculation_metadata, irc_payload, profiling_enabled):
    calculation_metadata["irc"] = irc_payload
    if profiling_enabled and irc_payload.get("profiling") is not None:
        calculation_metadata.setdefault("profiling", {})["irc"] = irc_payload.get("profiling")


def _execute_irc_single_point(
    stage_context,
    multiplicity,
    profiling_enabled,
    engine_adapter,
):
    return engine_adapter.compute_single_point_energy(
        stage_context["mol"],
        stage_context["calc_basis"],
        stage_context["calc_xc"],
        stage_context["calc_scf_config"],
        stage_context["calc_solvent_model"],
        stage_context["calc_solvent_name"],
        stage_context["calc_eps"],
        stage_context["calc_dispersion_model"],
        _resolve_d3_params(stage_context.get("optimizer_ase_config")),
        stage_context["verbose"],
        stage_context["memory_mb"],
        run_dir=stage_context["run_dir"],
        optimizer_mode=stage_context["optimizer_mode"],
        multiplicity=multiplicity,
        log_override=False,
        profiling_enabled=profiling_enabled,
    )


def _run_post_irc_single_point(
    *,
    stage_context,
    calculation_metadata,
    single_point_enabled,
    run_single_point,
    profiling_enabled,
    multiplicity,
    irc_payload,
    engine_adapter,
):
    sp_result = None
    sp_status = "skipped"
    sp_skip_reason = None

    if single_point_enabled:
        if irc_payload.get("status") != "completed":
            sp_skip_reason = "IRC did not complete; skipping single-point."
        else:
            logging.info("Calculating single-point energy after IRC...")
            try:
                sp_result = _execute_irc_single_point(
                    stage_context,
                    multiplicity,
                    profiling_enabled,
                    engine_adapter,
                )
                sp_status = "executed"
                if isinstance(calculation_metadata.get("single_point"), dict):
                    calculation_metadata["single_point"]["dispersion_info"] = sp_result.get(
                        "dispersion"
                    )
                    if profiling_enabled and sp_result.get("profiling") is not None:
                        calculation_metadata["single_point"]["profiling"] = sp_result.get(
                            "profiling"
                        )
                _update_checkpoint_scf(
                    stage_context.get("checkpoint_path"),
                    pyscf_chkfile=stage_context.get("pyscf_chkfile"),
                    scf_energy=sp_result.get("energy"),
                    scf_converged=sp_result.get("converged"),
                )
            except Exception:
                logging.exception("Single-point calculation failed.")
                sp_result = None
                sp_status = "failed"
                sp_skip_reason = "Single-point calculation failed."
    elif run_single_point:
        logging.info("Skipping single-point energy calculation (disabled).")
        sp_skip_reason = "Single-point calculation disabled."

    return sp_result, sp_status, sp_skip_reason


def _record_single_point_status(calculation_metadata, run_single_point, sp_status, sp_skip_reason):
    if run_single_point and isinstance(calculation_metadata.get("single_point"), dict):
        calculation_metadata["single_point"]["status"] = sp_status
        calculation_metadata["single_point"]["skip_reason"] = sp_skip_reason


def _build_irc_summary(run_start, irc_payload, sp_result):
    energy_summary = None
    if irc_payload["profile"]:
        energy_summary = {
            "start_energy_ev": irc_payload["profile"][0]["energy_ev"],
            "end_energy_ev": irc_payload["profile"][-1]["energy_ev"],
        }
    final_sp_energy = sp_result.get("energy") if sp_result else None
    final_sp_converged = sp_result.get("converged") if sp_result else None
    final_sp_cycles = sp_result.get("cycles") if sp_result else None
    return {
        "elapsed_seconds": time.perf_counter() - run_start,
        "n_steps": len(irc_payload["profile"]),
        "final_energy": energy_summary["end_energy_ev"] if energy_summary else None,
        "opt_final_energy": None,
        "final_sp_energy": final_sp_energy,
        "final_sp_converged": final_sp_converged,
        "final_sp_cycles": final_sp_cycles,
        "scf_converged": None,
        "opt_converged": None,
        "converged": True,
    }


def run_irc_stage(
    stage_context,
    queue_update_fn,
    *,
    finalize=True,
    update_summary=True,
    run_single_point=True,
    engine_adapter: WorkflowEngineAdapter = DEFAULT_ENGINE_ADAPTER,
):
    logging.info("Starting IRC calculation...")
    run_start = stage_context["run_start"]
    calculation_metadata = stage_context["metadata"]
    profiling_enabled = bool(stage_context.get("profiling_enabled"))
    mode_profiling = stage_context.get("mode_profiling")
    run_dir = stage_context["run_dir"]
    checkpoint_path = stage_context.get("checkpoint_path")
    resume_dir = stage_context.get("resume_dir")
    charge = stage_context.get("charge")
    spin = stage_context.get("spin")
    multiplicity = stage_context.get("multiplicity")
    (
        snapshot_interval_steps,
        snapshot_mode,
        snapshot_write_steps,
        snapshot_write_last,
    ) = _normalize_snapshot_settings(
        stage_context.get("snapshot_interval_steps", 1),
        stage_context.get("snapshot_mode", "all"),
    )
    single_point_enabled = bool(stage_context.get("single_point_enabled")) and run_single_point

    snapshot_paths = _build_irc_snapshot_paths(run_dir)
    checkpoint_base = _load_checkpoint_base(checkpoint_path)
    resume_data = _prepare_irc_resume_state(
        resume_dir=resume_dir,
        checkpoint_base=checkpoint_base,
        run_dir=run_dir,
        charge=charge,
        spin=spin,
        multiplicity=multiplicity,
    )
    record_step_callback, mark_direction_complete_callback = _build_irc_callbacks(
        checkpoint_path=checkpoint_path,
        checkpoint_base=checkpoint_base,
        profile_cache=resume_data["profile_cache"],
        profile_keys=resume_data["profile_keys"],
        irc_last_snapshot_step=resume_data["irc_last_snapshot_step"],
        irc_last_checkpoint_step=resume_data["irc_last_checkpoint_step"],
        irc_last_geometry_cache=resume_data["irc_last_geometry_cache"],
        irc_last_step_cache=resume_data["irc_last_step_cache"],
        snapshot_interval_steps=snapshot_interval_steps,
        snapshot_mode=snapshot_mode,
        snapshot_write_steps=snapshot_write_steps,
        snapshot_write_last=snapshot_write_last,
        charge=charge,
        spin=spin,
        multiplicity=multiplicity,
        snapshot_paths=snapshot_paths,
    )

    try:
        irc_result = _run_ase_irc_with_callbacks(
            stage_context,
            profiling_enabled,
            record_step_callback,
            mark_direction_complete_callback,
            resume_data["resume_state"],
            engine_adapter,
        )
        profile = resume_data["profile_cache"] or irc_result.get("profile", [])
        irc_payload = _build_irc_payload(
            stage_context,
            irc_result,
            profile,
            mode_profiling,
            profiling_enabled,
        )
        _write_irc_outputs(stage_context, irc_payload)
        _record_irc_metadata(calculation_metadata, irc_payload, profiling_enabled)

        sp_result, sp_status, sp_skip_reason = _run_post_irc_single_point(
            stage_context=stage_context,
            calculation_metadata=calculation_metadata,
            single_point_enabled=single_point_enabled,
            run_single_point=run_single_point,
            profiling_enabled=profiling_enabled,
            multiplicity=multiplicity,
            irc_payload=irc_payload,
            engine_adapter=engine_adapter,
        )
        _record_single_point_status(
            calculation_metadata,
            run_single_point,
            sp_status,
            sp_skip_reason,
        )

        summary = _build_irc_summary(run_start, irc_payload, sp_result)
        if update_summary:
            calculation_metadata["summary"] = summary
            calculation_metadata["summary"]["memory_limit_enforced"] = stage_context[
                "memory_limit_enforced"
            ]
        if finalize:
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
        return irc_payload
    except Exception as exc:
        logging.exception("IRC calculation failed.")
        if finalize:
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
