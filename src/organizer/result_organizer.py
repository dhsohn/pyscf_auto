"""Organize completed runs into a clean directory structure."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .molecule_key import derive_molecule_key

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.jsonl"

# Map job types to directory names
_JOB_TYPE_DIRS = {
    "optimization": "optimization",
    "single_point": "single_point",
    "frequency": "frequency",
    "irc": "irc",
    "scan": "scan",
}


def organize_run(
    reaction_dir: str,
    organized_root: str,
    apply: bool = False,
) -> dict[str, Any] | None:
    """Organize a completed run from reaction_dir to organized output.

    The target structure is::

        organized_root/<job_type>/<molecule_key>/<run_id>/
            run_state.json
            run_report.json
            run_report.md
            attempt_NNN/...

    Args:
        reaction_dir: Source reaction directory.
        organized_root: Root directory for organized outputs.
        apply: If True, actually move files. If False, dry-run preview.

    Returns:
        A dict describing the operation, or None if nothing to organize.
    """
    state = _load_state(reaction_dir)
    if state is None:
        logger.info("No run_state.json in %s, skipping.", reaction_dir)
        return None

    if state.get("status") not in ("completed", "failed"):
        logger.info(
            "Run in %s is %s (not terminal), skipping.",
            reaction_dir,
            state.get("status", "?"),
        )
        return None

    # Determine target path
    run_id = state.get("run_id", "unknown")
    job_type = _infer_job_type(reaction_dir)
    job_type_dir = _JOB_TYPE_DIRS.get(job_type, job_type or "other")

    # Get molecule key
    atom_spec = _extract_atom_spec(reaction_dir)
    tag = _extract_tag(reaction_dir)
    molecule_key = derive_molecule_key(
        atom_spec or "",
        tag=tag,
        reaction_dir_name=os.path.basename(reaction_dir),
    )

    target_dir = os.path.join(organized_root, job_type_dir, molecule_key, run_id)

    result = {
        "source": reaction_dir,
        "target": target_dir,
        "run_id": run_id,
        "job_type": job_type_dir,
        "molecule_key": molecule_key,
        "status": state.get("status"),
        "applied": False,
    }

    if apply:
        os.makedirs(target_dir, exist_ok=True)

        # Copy key files
        _copy_if_exists(reaction_dir, target_dir, "run_state.json")
        _copy_if_exists(reaction_dir, target_dir, "run_report.json")
        _copy_if_exists(reaction_dir, target_dir, "run_report.md")

        # Copy attempt directories
        for entry in sorted(os.listdir(reaction_dir)):
            if entry.startswith("attempt_") and os.path.isdir(
                os.path.join(reaction_dir, entry)
            ):
                src = os.path.join(reaction_dir, entry)
                dst = os.path.join(target_dir, entry)
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)

        # Copy log directory
        _copy_dir_if_exists(reaction_dir, target_dir, "log")

        # Append to index
        _append_to_index(organized_root, result)

        result["applied"] = True
        logger.info("Organized: %s -> %s", reaction_dir, target_dir)
    else:
        logger.info("Would organize: %s -> %s", reaction_dir, target_dir)

    return result


def organize_all(
    root: str,
    organized_root: str,
    apply: bool = False,
) -> list[dict[str, Any]]:
    """Organize all completed runs under a root directory.

    Args:
        root: Root directory containing reaction directories.
        organized_root: Target root for organized outputs.
        apply: If True, actually move files.

    Returns:
        List of operation descriptions.
    """
    root = str(Path(root).expanduser().resolve())
    results = []

    if not os.path.isdir(root):
        logger.error("Root directory not found: %s", root)
        return results

    for entry in sorted(os.listdir(root)):
        reaction_dir = os.path.join(root, entry)
        if not os.path.isdir(reaction_dir):
            continue
        # Check if it has a run_state.json
        if not os.path.exists(os.path.join(reaction_dir, "run_state.json")):
            continue
        result = organize_run(reaction_dir, organized_root, apply=apply)
        if result:
            results.append(result)

    return results


def find_organized_runs(
    organized_root: str,
    run_id: str | None = None,
    job_type: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Search organized outputs by criteria.

    Args:
        organized_root: Root directory of organized outputs.
        run_id: Filter by run ID (partial match).
        job_type: Filter by job type.
        limit: Maximum number of results.

    Returns:
        List of matching run entries from the index.
    """
    index_path = os.path.join(organized_root, _INDEX_FILE)
    if not os.path.exists(index_path):
        return []

    results = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if run_id and run_id not in entry.get("run_id", ""):
                continue
            if job_type and entry.get("job_type") != job_type:
                continue

            results.append(entry)
            if limit and len(results) >= limit:
                break

    return results


def _load_state(reaction_dir: str) -> dict[str, Any] | None:
    """Load run_state.json from a reaction directory."""
    state_path = os.path.join(reaction_dir, "run_state.json")
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _infer_job_type(reaction_dir: str) -> str:
    """Infer job type from .inp file in the reaction directory."""
    import glob

    inp_files = glob.glob(os.path.join(reaction_dir, "*.inp"))
    if not inp_files:
        return "other"

    # Read the first .inp file and find the route line
    try:
        with open(inp_files[0], "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("!"):
                    from inp.route_line import parse_route_line

                    result = parse_route_line(stripped)
                    job_type = result.job_type
                    if job_type == "optimization" and result.optimizer_mode == "transition_state":
                        return "ts_optimization"
                    return job_type
    except Exception:
        pass

    return "other"


def _extract_atom_spec(reaction_dir: str) -> str | None:
    """Try to extract atom spec from the xyz file in the reaction dir."""
    xyz_path = os.path.join(reaction_dir, "_geometry.xyz")
    if not os.path.exists(xyz_path):
        # Try other xyz files
        import glob

        xyz_files = glob.glob(os.path.join(reaction_dir, "*.xyz"))
        if not xyz_files:
            return None
        xyz_path = xyz_files[0]

    try:
        with open(xyz_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 3:
            return None
        return "".join(lines[2:]).strip()
    except OSError:
        return None


def _extract_tag(reaction_dir: str) -> str | None:
    """Try to extract TAG from .inp file."""
    import glob
    import re

    inp_files = glob.glob(os.path.join(reaction_dir, "*.inp"))
    if not inp_files:
        return None

    tag_re = re.compile(r"^#\s*TAG\s*:\s*(.+)$", re.IGNORECASE)
    try:
        with open(inp_files[0], "r", encoding="utf-8") as f:
            for line in f:
                m = tag_re.match(line.strip())
                if m:
                    return m.group(1).strip()
    except OSError:
        pass

    return None


def _copy_if_exists(src_dir: str, dst_dir: str, filename: str) -> None:
    """Copy a file if it exists."""
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)


def _copy_dir_if_exists(src_dir: str, dst_dir: str, dirname: str) -> None:
    """Copy a directory if it exists."""
    src = os.path.join(src_dir, dirname)
    dst = os.path.join(dst_dir, dirname)
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)


def _append_to_index(organized_root: str, entry: dict[str, Any]) -> None:
    """Append an entry to the JSONL index."""
    index_path = os.path.join(organized_root, _INDEX_FILE)
    os.makedirs(organized_root, exist_ok=True)
    record = {
        "run_id": entry.get("run_id"),
        "source": entry.get("source"),
        "target": entry.get("target"),
        "job_type": entry.get("job_type"),
        "molecule_key": entry.get("molecule_key"),
        "status": entry.get("status"),
    }
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
