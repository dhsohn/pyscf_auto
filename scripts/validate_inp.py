#!/usr/bin/env python3
"""Validate one .inp file without running a job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="validate_inp.py")
    parser.add_argument("inp_file", help="Path to .inp file")
    return parser.parse_args()


def main() -> int:
    _bootstrap_path()
    from inp.parser import inp_config_to_dict, parse_inp_file
    from run_opt_config import build_run_config

    args = _parse_args()

    try:
        inp = parse_inp_file(args.inp_file)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Validation FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"job_type: {inp.job_type}")
    print(f"functional: {inp.functional}")
    print(f"basis: {inp.basis}")
    print(f"charge: {inp.charge}")
    print(f"multiplicity: {inp.multiplicity}")

    config_dict = inp_config_to_dict(inp)
    try:
        build_run_config(config_dict)
    except ValueError as exc:
        print(f"run_config: FAILED ({exc})", file=sys.stderr)
        return 1

    print("run_config: OK")
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
