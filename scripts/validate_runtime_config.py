#!/usr/bin/env python3
"""Validate pyscf_auto runtime configuration."""

from __future__ import annotations

import argparse
import os
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
    config_text = args.config or os.environ.get("PYSCF_AUTO_CONFIG") or "~/.pyscf_auto/config.yaml"
    print(f"Config file: {Path(config_text).expanduser().resolve()}")
    try:
        cfg = load_app_config(args.config)
    except ValueError as exc:
        print(f"  [FAIL] Config validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"  [FAIL] Config load error: {exc}", file=sys.stderr)
        return 1

    allowed_root = Path(cfg.runtime.allowed_root).expanduser().resolve()
    organized_root = Path(cfg.runtime.organized_root).expanduser().resolve()
    max_retries = int(cfg.runtime.default_max_retries)

    errors = 0
    print("  [PASS] Config loaded successfully")
    print(f"         allowed_root: {allowed_root}")
    print(f"         organized_root: {organized_root}")
    print(f"         default_max_retries: {max_retries}")
    print(f"         monitoring.enabled: {bool(cfg.monitoring.enabled)}")

    if allowed_root.exists() and allowed_root.is_dir():
        print("  [PASS] allowed_root exists and is a directory")
    else:
        print(f"  [FAIL] allowed_root not found or not a directory: {allowed_root}", file=sys.stderr)
        errors += 1

    if organized_root.exists():
        if organized_root.is_dir():
            print("  [PASS] organized_root exists and is a directory")
        else:
            print(f"  [FAIL] organized_root exists but is not a directory: {organized_root}", file=sys.stderr)
            errors += 1
    else:
        print(f"  [WARN] organized_root does not exist yet (will be created on first organize): {organized_root}")

    if _is_subpath(organized_root, allowed_root) or _is_subpath(allowed_root, organized_root):
        print(
            "  [FAIL] allowed_root and organized_root must not contain each other: "
            f"allowed_root={allowed_root}, organized_root={organized_root}",
            file=sys.stderr,
        )
        errors += 1
    else:
        print("  [PASS] runtime roots are separated")

    if cfg.monitoring.enabled:
        token = os.environ.get(cfg.monitoring.telegram.bot_token_env, "").strip()
        chat = os.environ.get(cfg.monitoring.telegram.chat_id_env, "").strip()
        if token and chat:
            print("  [PASS] monitoring is enabled and Telegram credentials are present")
        else:
            print(
                "  [FAIL] monitoring is enabled but Telegram credentials are missing: "
                f"{cfg.monitoring.telegram.bot_token_env}, {cfg.monitoring.telegram.chat_id_env}",
                file=sys.stderr,
            )
            errors += 1
    else:
        print("  [PASS] monitoring is disabled")

    if errors:
        print(f"\n=== {errors} CHECK(S) FAILED ===")
        return 1
    print("\n=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
