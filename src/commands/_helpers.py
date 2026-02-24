from __future__ import annotations

import os
from pathlib import Path

from app_config import AppConfig


CONFIG_ENV_VAR = "PYSCF_AUTO_CONFIG"
_MAX_SAMPLE_FILES = 10


def default_config_path() -> str:
    env_path = os.getenv(CONFIG_ENV_VAR, "").strip()
    if env_path:
        return env_path

    repo_default = Path(__file__).resolve().parents[2] / "config" / "pyscf_auto.yaml"
    if repo_default.exists():
        return str(repo_default)

    home_default = Path.home() / ".pyscf_auto" / "config.yaml"
    return str(home_default)


def to_resolved_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_reaction_dir(cfg: AppConfig, reaction_dir_raw: str) -> Path:
    reaction_dir = to_resolved_path(reaction_dir_raw)
    if not reaction_dir.exists() or not reaction_dir.is_dir():
        raise ValueError(f"Reaction directory not found: {reaction_dir}")
    allowed_root = to_resolved_path(cfg.runtime.allowed_root)
    if not is_subpath(reaction_dir, allowed_root):
        raise ValueError(
            f"Reaction directory must be under allowed_root: {allowed_root}. got={reaction_dir}"
        )
    return reaction_dir


def validate_root_scan_dir(cfg: AppConfig, root_raw: str) -> Path:
    root = to_resolved_path(root_raw)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root directory not found: {root}")
    allowed_root = to_resolved_path(cfg.runtime.allowed_root)
    if root != allowed_root:
        raise ValueError(f"--root must exactly match allowed_root: {allowed_root}. got={root}")
    return root


def validate_organized_root_dir(cfg: AppConfig, root_raw: str) -> Path:
    root = to_resolved_path(root_raw)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root directory not found: {root}")
    organized_root = to_resolved_path(cfg.runtime.organized_root)
    if root != organized_root:
        raise ValueError(
            f"--root must exactly match organized_root: {organized_root}. got={root}"
        )
    return root


def validate_cleanup_reaction_dir(cfg: AppConfig, reaction_dir_raw: str) -> Path:
    reaction_dir = to_resolved_path(reaction_dir_raw)
    if not reaction_dir.exists() or not reaction_dir.is_dir():
        raise ValueError(f"Reaction directory not found: {reaction_dir}")
    organized_root = to_resolved_path(cfg.runtime.organized_root)
    if not is_subpath(reaction_dir, organized_root):
        raise ValueError(
            f"Reaction directory must be under organized_root: {organized_root}. got={reaction_dir}"
        )
    return reaction_dir


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(amount) < 1024.0:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TB"

