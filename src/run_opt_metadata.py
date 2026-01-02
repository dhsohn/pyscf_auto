import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from importlib import metadata as importlib_metadata

from run_opt_resources import ensure_parent_dir
from run_opt_utils import extract_step_count

RUN_METADATA_SCHEMA_VERSION = 1


def _extract_energy(*candidates):
    for obj in candidates:
        if obj is None:
            continue
        if hasattr(obj, "e_tot"):
            return obj.e_tot
        if hasattr(obj, "energy_tot"):
            return obj.energy_tot
    return None


def _extract_opt_converged(mf):
    for candidate in (getattr(mf, "opt", None), getattr(mf, "optimizer", None)):
        if candidate is None:
            continue
        for attr in ("converged", "opt_converged"):
            if hasattr(candidate, attr):
                return getattr(candidate, attr)
    return None


def parse_single_point_cycle_count(log_path):
    if not log_path or not os.path.exists(log_path):
        return None
    sp_start_pattern = re.compile(r"Calculating single-point energy", re.IGNORECASE)
    cycle_pattern = re.compile(r"\bcycle=\s*\d+", re.IGNORECASE)
    extra_cycle_pattern = re.compile(r"\bextra cycle\b", re.IGNORECASE)
    found_sp_section = False
    cycle_count = 0
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
            for line in log_file:
                if not found_sp_section:
                    if sp_start_pattern.search(line):
                        found_sp_section = True
                    continue
                cycle_hits = cycle_pattern.findall(line)
                if cycle_hits:
                    cycle_count += len(cycle_hits)
                elif extra_cycle_pattern.search(line):
                    cycle_count += 1
    except OSError:
        return None
    return cycle_count or None


def compute_file_hash(path, algorithm="sha256", chunk_size=1024 * 1024):
    if not path or not os.path.exists(path):
        return None
    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        return None
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def compute_text_hash(text, algorithm="sha256"):
    if text is None:
        return None
    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        return None
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def collect_git_metadata(base_path):
    if not base_path:
        return None
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=base_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None
    def _run_git(args):
        result = subprocess.run(
            ["git", *args],
            cwd=base_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status),
    }


def build_run_summary(
    mf,
    mol_optimized,
    elapsed_seconds,
    completed,
    n_steps=None,
    final_sp_energy=None,
    final_sp_converged=None,
    final_sp_cycles=None,
):
    opt_final_energy = (
        final_sp_energy if final_sp_energy is not None else _extract_energy(mf, mol_optimized)
    )
    scf_converged = (
        final_sp_converged
        if final_sp_converged is not None
        else getattr(mf, "converged", None)
    )
    opt_converged = _extract_opt_converged(mf)
    if scf_converged is not None and opt_converged is not None:
        converged = bool(scf_converged and opt_converged)
    elif scf_converged is not None:
        converged = bool(scf_converged) if completed else scf_converged
    elif opt_converged is not None:
        converged = bool(opt_converged) if completed else opt_converged
    else:
        converged = bool(completed)
    n_steps_value = n_steps
    if n_steps_value is None:
        n_steps_value = extract_step_count(
            mf, getattr(mf, "opt", None), getattr(mf, "optimizer", None)
        )
    final_energy = final_sp_energy if final_sp_energy is not None else opt_final_energy
    return {
        "elapsed_seconds": elapsed_seconds,
        "n_steps": n_steps_value,
        "final_energy": final_energy,
        "opt_final_energy": opt_final_energy,
        "final_sp_energy": final_sp_energy,
        "final_sp_converged": final_sp_converged,
        "final_sp_cycles": final_sp_cycles,
        "scf_converged": scf_converged,
        "opt_converged": opt_converged,
        "converged": converged,
    }


def write_optimized_xyz(output_path, mol):
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            xyz_data = mol.tostring(format="xyz")
            if xyz_data and not xyz_data.endswith("\n"):
                xyz_data += "\n"
            output_file.write(xyz_data)
    except Exception as exc:
        logging.getLogger().error(
            "Failed to write optimized XYZ to %s: %s",
            output_path,
            exc,
        )


def get_package_version(package_name):
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def write_run_metadata(metadata_path, metadata):
    try:
        if metadata is None:
            return
        metadata.setdefault("schema_version", RUN_METADATA_SCHEMA_VERSION)
        ensure_parent_dir(metadata_path)
        metadata_dir = os.path.dirname(metadata_path) or "."
        temp_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metadata_dir,
            prefix=".metadata.json.",
            suffix=".tmp",
            delete=False,
        )
        try:
            with temp_handle as metadata_file:
                json.dump(metadata, metadata_file, indent=2)
                metadata_file.flush()
                os.fsync(metadata_file.fileno())
            os.replace(temp_handle.name, metadata_path)
        finally:
            if os.path.exists(temp_handle.name):
                try:
                    os.remove(temp_handle.name)
                except FileNotFoundError:
                    pass
    except Exception as exc:
        logging.getLogger().error(
            "Failed to write run metadata to %s: %s",
            metadata_path,
            exc,
        )


def write_checkpoint(checkpoint_path, checkpoint_payload):
    try:
        if checkpoint_payload is None:
            return
        ensure_parent_dir(checkpoint_path)
        checkpoint_dir = os.path.dirname(checkpoint_path) or "."
        temp_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=checkpoint_dir,
            prefix=".checkpoint.json.",
            suffix=".tmp",
            delete=False,
        )
        try:
            with temp_handle as checkpoint_file:
                json.dump(checkpoint_payload, checkpoint_file, indent=2, ensure_ascii=False)
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())
            os.replace(temp_handle.name, checkpoint_path)
        finally:
            if os.path.exists(temp_handle.name):
                try:
                    os.remove(temp_handle.name)
                except FileNotFoundError:
                    pass
    except Exception as exc:
        logging.getLogger().error(
            "Failed to write checkpoint to %s: %s",
            checkpoint_path,
            exc,
        )
