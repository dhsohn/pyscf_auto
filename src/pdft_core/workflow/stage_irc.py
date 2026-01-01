import json
import logging
import time

from ..ase_backend import _run_ase_irc
from .events import finalize_metadata


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
