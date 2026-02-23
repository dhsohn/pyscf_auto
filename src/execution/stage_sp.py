import logging
import time

from qcschema_export import export_qcschema_result
from .engine_adapter import WorkflowEngineAdapter
from .events import finalize_metadata
from .utils import _resolve_d3_params, _update_checkpoint_scf


DEFAULT_ENGINE_ADAPTER = WorkflowEngineAdapter()


def run_single_point_stage(
    stage_context,
    queue_update_fn,
    engine_adapter: WorkflowEngineAdapter = DEFAULT_ENGINE_ADAPTER,
):
    logging.info("Starting single-point energy calculation...")
    run_start = stage_context["run_start"]
    calculation_metadata = stage_context["metadata"]
    profiling_enabled = bool(stage_context.get("profiling_enabled"))
    try:
        sp_result = engine_adapter.compute_single_point_energy(
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
            multiplicity=stage_context["multiplicity"],
            log_override=False,
            profiling_enabled=profiling_enabled,
        )
        calculation_metadata["dispersion_info"] = sp_result.get("dispersion")
        if profiling_enabled and sp_result.get("profiling") is not None:
            calculation_metadata.setdefault("profiling", {})["single_point"] = sp_result.get(
                "profiling"
            )
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
        export_qcschema_result(
            stage_context.get("qcschema_output_path"),
            calculation_metadata,
            stage_context.get("input_xyz"),
            geometry_xyz=stage_context.get("input_xyz"),
            sp_result=sp_result,
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
