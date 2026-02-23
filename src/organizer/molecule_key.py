"""Derive molecule keys for organizing output directories."""

from __future__ import annotations

import re
from collections import Counter


def derive_molecule_key(
    atom_spec: str,
    tag: str | None = None,
    reaction_dir_name: str | None = None,
) -> str:
    """Derive a molecule identifier for directory organization.

    Priority:
    1. TAG from .inp file comment (``# TAG: <value>``)
    2. Hill notation formula from atom specification
    3. Reaction directory basename as fallback

    Args:
        atom_spec: PySCF atom specification string.
        tag: Optional TAG from .inp file.
        reaction_dir_name: Basename of the reaction directory.

    Returns:
        A string suitable for use as a directory name.
    """
    if tag:
        return _sanitize_dirname(tag)

    formula = hill_formula(atom_spec)
    if formula:
        return formula

    if reaction_dir_name:
        return _sanitize_dirname(reaction_dir_name)

    return "unknown"


def hill_formula(atom_spec: str) -> str:
    """Generate Hill notation formula from a PySCF atom specification.

    Hill notation: C first, H second, then all other elements alphabetically.

    Example::

        >>> hill_formula("O 0 0 0\\nH 1 0 0\\nH 0 1 0")
        'H2O'
    """
    lines = atom_spec.strip().split("\n")
    counts: Counter[str] = Counter()

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        # Element symbol is the first token (may have ghost prefix like "X-")
        element = parts[0]
        # Remove ghost atom prefix
        if element.startswith("X-") or element.startswith("x-"):
            element = element[2:]
        # Normalize: capitalize first letter, lowercase rest
        element = element[0].upper() + element[1:].lower() if len(element) > 1 else element.upper()
        # Remove any numeric prefix (e.g., "1H" -> "H")
        element = re.sub(r"^\d+", "", element)
        if element:
            counts[element] += 1

    if not counts:
        return ""

    # Hill ordering: C, H first, then alphabetical
    parts = []
    if "C" in counts:
        parts.append(("C", counts.pop("C")))
        if "H" in counts:
            parts.append(("H", counts.pop("H")))

    for element in sorted(counts.keys()):
        parts.append((element, counts[element]))

    formula_parts = []
    for element, count in parts:
        if count == 1:
            formula_parts.append(element)
        else:
            formula_parts.append(f"{element}{count}")

    return "".join(formula_parts)


def _sanitize_dirname(name: str) -> str:
    """Sanitize a string for use as a directory name."""
    # Replace problematic characters with underscores
    sanitized = re.sub(r"[/\\:*?\"<>|]+", "_", name)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "unknown"
