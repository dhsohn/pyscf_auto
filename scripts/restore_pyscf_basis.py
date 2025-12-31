#!/usr/bin/env python3
"""Restore missing PySCF basis data files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/pyscf/pyscf/master/pyscf/gto/basis"
KNOWN_BASIS = {
    "6-31G": "pople-basis/6-31G.dat",
}


def _resolve_pyscf_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    try:
        import pyscf  # type: ignore

        return Path(pyscf.__file__).resolve().parent
    except Exception as err:
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / "pyscf" / "pyscf"
        if candidate.exists():
            return candidate
        raise RuntimeError(
            "Unable to locate PySCF package. Provide --pyscf-root to point to it."
        ) from err


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore PySCF basis data files.")
    parser.add_argument(
        "--basis",
        default="6-31G",
        help="Basis set name to restore (default: 6-31G).",
    )
    parser.add_argument(
        "--pyscf-root",
        help="Path to the PySCF package root (the directory containing gto/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing basis files.",
    )
    args = parser.parse_args()

    basis_key = args.basis
    if basis_key not in KNOWN_BASIS:
        known = ", ".join(sorted(KNOWN_BASIS))
        raise SystemExit(f"Unknown basis '{basis_key}'. Known bases: {known}.")

    pyscf_root = _resolve_pyscf_root(args.pyscf_root)
    basis_relpath = KNOWN_BASIS[basis_key]
    basis_dest = pyscf_root / "gto" / "basis" / basis_relpath

    if basis_dest.exists() and not args.overwrite:
        print(f"Basis file already present: {basis_dest}")
        return 0

    url = f"{BASE_URL}/{basis_relpath}"
    print(f"Downloading {url} -> {basis_dest}")
    _download(url, basis_dest)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
