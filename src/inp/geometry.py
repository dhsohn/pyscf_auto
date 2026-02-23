"""Parse geometry blocks from .inp files."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GeometryResult:
    """Parsed geometry block contents."""

    charge: int
    multiplicity: int
    atom_spec: str  # PySCF-compatible atom specification string
    source_type: str  # "inline" or "xyzfile"
    source_path: str | None = None  # Path to .xyz file if xyzfile reference


def parse_geometry_block(lines: list[str], inp_dir: str) -> GeometryResult:
    """Parse a geometry block from .inp file lines.

    Supports two formats:

    1. Inline geometry::

        * xyz <charge> <multiplicity>
        O  0.0  0.0  0.0
        H  1.0  0.0  0.0
        H  0.0  1.0  0.0
        *

    2. External file reference::

        * xyzfile <charge> <multiplicity> <path>

    Args:
        lines: All lines of the .inp file.
        inp_dir: Directory containing the .inp file (for resolving relative paths).

    Returns:
        A ``GeometryResult`` with charge, multiplicity, and atom specification.

    Raises:
        ValueError: If the geometry block is missing or malformed.
    """
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("*") and not stripped.startswith("#"):
            # Check if this is a geometry start line
            parts = stripped.split()
            if len(parts) >= 2 and parts[0] == "*":
                keyword = parts[1].lower()
                if keyword in ("xyz", "xyzfile"):
                    start_idx = i
                    break

    if start_idx is None:
        raise ValueError(
            "No geometry block found. Expected '* xyz <charge> <mult>' or "
            "'* xyzfile <charge> <mult> <path>'."
        )

    header = lines[start_idx].strip()
    parts = header.split()

    # parts[0] = '*', parts[1] = 'xyz' or 'xyzfile'
    keyword = parts[1].lower()

    if len(parts) < 4:
        raise ValueError(
            f"Geometry header must have at least charge and multiplicity: {header!r}"
        )

    try:
        charge = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"Invalid charge in geometry header: {parts[2]!r}") from exc

    try:
        multiplicity = int(parts[3])
    except ValueError as exc:
        raise ValueError(
            f"Invalid multiplicity in geometry header: {parts[3]!r}"
        ) from exc

    if multiplicity < 1:
        raise ValueError(f"Multiplicity must be >= 1, got {multiplicity}")

    if keyword == "xyzfile":
        # External file reference: * xyzfile <charge> <mult> <path>
        if len(parts) < 5:
            raise ValueError(
                "xyzfile format requires a file path: "
                "'* xyzfile <charge> <mult> <path>'"
            )
        xyz_path = " ".join(parts[4:])  # Handle paths with spaces
        if not os.path.isabs(xyz_path):
            xyz_path = os.path.join(inp_dir, xyz_path)
        xyz_path = os.path.abspath(xyz_path)

        if not os.path.exists(xyz_path):
            raise ValueError(f"Referenced XYZ file not found: {xyz_path}")

        atom_spec = _load_atom_spec_from_xyz(xyz_path)
        return GeometryResult(
            charge=charge,
            multiplicity=multiplicity,
            atom_spec=atom_spec,
            source_type="xyzfile",
            source_path=xyz_path,
        )

    # Inline geometry: * xyz <charge> <mult> ... atoms ... *
    atom_lines = []
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "*":
            break
        if stripped and not stripped.startswith("#"):
            atom_lines.append(stripped)
    else:
        raise ValueError(
            "Inline geometry block not terminated. Expected closing '*'."
        )

    if not atom_lines:
        raise ValueError("Inline geometry block is empty (no atoms).")

    # Validate atom lines
    for idx, aline in enumerate(atom_lines, start=1):
        aparts = aline.split()
        if len(aparts) < 4:
            raise ValueError(
                f"Atom line {idx} must have at least 4 columns "
                f"(element x y z): {aline!r}"
            )
        # Validate coordinates
        for coord in aparts[1:4]:
            try:
                float(coord)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid coordinate in atom line {idx}: {aline!r}"
                ) from exc

    atom_spec = "\n".join(atom_lines)
    return GeometryResult(
        charge=charge,
        multiplicity=multiplicity,
        atom_spec=atom_spec,
        source_type="inline",
    )


def _load_atom_spec_from_xyz(xyz_path: str) -> str:
    """Load atom specification from an XYZ file.

    Reads the standard XYZ format (atom count, comment, atom lines)
    and returns the atom specification string.
    """
    with open(xyz_path, "r", encoding="utf-8") as f:
        file_lines = [line.rstrip("\n") for line in f]

    if not file_lines:
        raise ValueError(f"XYZ file is empty: {xyz_path}")

    try:
        expected_atoms = int(file_lines[0].strip())
    except ValueError as exc:
        raise ValueError(
            f"XYZ first line must be an integer atom count: {file_lines[0]!r}"
        ) from exc

    if expected_atoms <= 0:
        raise ValueError(f"XYZ atom count must be positive; got {expected_atoms}")

    # Skip comment line (line 1), read atom lines (line 2+)
    atom_lines = [line for line in file_lines[2:] if line.strip()]

    if len(atom_lines) != expected_atoms:
        raise ValueError(
            f"XYZ atom count mismatch: header says {expected_atoms}, "
            f"but found {len(atom_lines)} atom lines."
        )

    for idx, line in enumerate(atom_lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"XYZ atom line {idx} must have at least 4 columns: {line!r}"
            )

    return "\n".join(atom_lines)
