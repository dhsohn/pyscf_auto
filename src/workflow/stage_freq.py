import json
import logging
import time

from run_opt_engine import compute_frequencies
from qcschema_export import export_qcschema_result
from .events import finalize_metadata
from .utils import (
    _frequency_units,
    _frequency_versions,
    _thermochemistry_payload,
    _update_checkpoint_scf,
)


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
            run_dir=stage_context["run_dir"],
            optimizer_mode=stage_context["optimizer_mode"],
            multiplicity=stage_context["multiplicity"],
            ts_quality=stage_context.get("ts_quality"),
            log_override=False,
        )
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
        _update_checkpoint_scf(
            stage_context.get("checkpoint_path"),
            pyscf_chkfile=stage_context.get("pyscf_chkfile"),
            scf_energy=frequency_result.get("energy"),
            scf_converged=frequency_result.get("converged"),
        )
        export_qcschema_result(
            stage_context.get("qcschema_output_path"),
            calculation_metadata,
            stage_context.get("input_xyz"),
            frequency_payload=frequency_payload,
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
