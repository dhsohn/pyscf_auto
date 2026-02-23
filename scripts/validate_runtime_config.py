#!/usr/bin/env python3
"""Validate runtime-level app config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="validate_runtime_config.py")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default loader order applies).",
    )
    return parser.parse_args()


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    _bootstrap_path()
    from app_config import load_app_config

    args = _parse_args()
    cfg = load_app_config(args.config)

    allowed_root = Path(cfg.runtime.allowed_root).expanduser().resolve()
    organized_root = Path(cfg.runtime.organized_root).expanduser().resolve()
    max_retries = int(cfg.runtime.default_max_retries)

    errors: list[str] = []
    if not allowed_root.is_absolute():
        errors.append(f"allowed_root must be absolute: {allowed_root}")
    if not organized_root.is_absolute():
        errors.append(f"organized_root must be absolute: {organized_root}")
    if _is_subpath(organized_root, allowed_root) or _is_subpath(allowed_root, organized_root):
        errors.append(
            "allowed_root and organized_root must not contain each other: "
            f"allowed_root={allowed_root}, organized_root={organized_root}",
        )
    if max_retries < 0:
        errors.append(f"default_max_retries must be >= 0: {max_retries}")

    print(f"allowed_root={allowed_root}")
    print(f"organized_root={organized_root}")
    print(f"default_max_retries={max_retries}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("OK: runtime config is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
