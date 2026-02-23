"""Shared helpers for run_opt runs."""

from typing import Any, Literal, TypeAlias

ConstraintStyle: TypeAlias = Literal["runtime", "config"]
ConstraintBond: TypeAlias = tuple[int, int, float]
ConstraintAngle: TypeAlias = tuple[int, int, int, float]
ConstraintDihedral: TypeAlias = tuple[int, int, int, int, float]


def extract_step_count(*candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ("n_steps", "nsteps", "nstep", "steps", "step_count"):
            if not hasattr(candidate, attr):
                continue
            value = getattr(candidate, attr)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, int):
                return value
    return None


def normalize_solvent_key(name: Any) -> str:
    return "".join(char for char in str(name).lower() if char.isalnum())


def is_ts_quality_enforced(ts_quality: Any) -> bool:
    if ts_quality is None:
        return False
    if hasattr(ts_quality, "enforce"):
        enforce_value = ts_quality.enforce
        if enforce_value is not None:
            return bool(enforce_value)
    if hasattr(ts_quality, "to_dict"):
        try:
            ts_quality_dict = ts_quality.to_dict()
        except Exception:
            ts_quality_dict = None
        if isinstance(ts_quality_dict, dict):
            enforce_value = ts_quality_dict.get("enforce")
            if enforce_value is not None:
                return bool(enforce_value)
    if isinstance(ts_quality, dict):
        enforce_value = ts_quality.get("enforce")
        if enforce_value is not None:
            return bool(enforce_value)
    return False


def _constraint_label(path: str, style: ConstraintStyle) -> str:
    if style == "config":
        return f"Config '{path}'"
    return path


def _constraint_entries(
    constraints: dict[str, Any],
    name: str,
    style: ConstraintStyle,
) -> list[Any]:
    raw_value = constraints.get(name)
    if style == "runtime":
        if not raw_value:
            return []
    elif raw_value is None:
        return []
    path = f"constraints.{name}"
    if not isinstance(raw_value, list):
        raise ValueError(f"{_constraint_label(path, style)} must be a list.")
    return raw_value


def _validate_constraint_index(
    value: Any,
    path: str,
    atom_count: int | None,
    style: ConstraintStyle,
) -> int:
    label = _constraint_label(path, style)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    if atom_count is None:
        if value < 0:
            raise ValueError(f"{label} must be >= 0.")
        return value
    if value < 0 or value >= atom_count:
        raise ValueError(f"{label} index {value} is out of range for {atom_count} atoms.")
    return value


def _validate_constraint_number(value: Any, path: str, style: ConstraintStyle) -> float:
    label = _constraint_label(path, style)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number.")
    return float(value)


def normalize_constraints(
    constraints: Any,
    atom_count: int | None = None,
    style: ConstraintStyle = "runtime",
) -> tuple[list[ConstraintBond], list[ConstraintAngle], list[ConstraintDihedral]]:
    if not constraints:
        return [], [], []
    if not isinstance(constraints, dict):
        raise ValueError("Config 'constraints' must be an object.")

    bond_entries: list[ConstraintBond] = []
    angle_entries: list[ConstraintAngle] = []
    dihedral_entries: list[ConstraintDihedral] = []

    bonds = _constraint_entries(constraints, "bonds", style)
    for idx, bond in enumerate(bonds):
        item_path = f"constraints.bonds[{idx}]"
        if not isinstance(bond, dict):
            raise ValueError(f"{_constraint_label(item_path, style)} must be an object.")
        for key in ("i", "j"):
            if key not in bond:
                raise ValueError(
                    f"{_constraint_label(item_path, style)} must define '{key}'."
                )
        if "length" not in bond:
            raise ValueError(f"{_constraint_label(item_path, style)} must define 'length'.")
        bond_i = _validate_constraint_index(
            bond["i"], f"{item_path}.i", atom_count, style
        )
        bond_j = _validate_constraint_index(
            bond["j"], f"{item_path}.j", atom_count, style
        )
        length = _validate_constraint_number(
            bond["length"], f"{item_path}.length", style
        )
        if length <= 0:
            raise ValueError(
                f"{_constraint_label(f'{item_path}.length', style)} must be > 0 (Angstrom)."
            )
        bond_entries.append((bond_i, bond_j, length))

    angles = _constraint_entries(constraints, "angles", style)
    for idx, angle in enumerate(angles):
        item_path = f"constraints.angles[{idx}]"
        if not isinstance(angle, dict):
            raise ValueError(f"{_constraint_label(item_path, style)} must be an object.")
        for key in ("i", "j", "k"):
            if key not in angle:
                raise ValueError(
                    f"{_constraint_label(item_path, style)} must define '{key}'."
                )
        if "angle" not in angle:
            raise ValueError(f"{_constraint_label(item_path, style)} must define 'angle'.")
        angle_i = _validate_constraint_index(
            angle["i"], f"{item_path}.i", atom_count, style
        )
        angle_j = _validate_constraint_index(
            angle["j"], f"{item_path}.j", atom_count, style
        )
        angle_k = _validate_constraint_index(
            angle["k"], f"{item_path}.k", atom_count, style
        )
        angle_value = _validate_constraint_number(
            angle["angle"], f"{item_path}.angle", style
        )
        if not (0 < angle_value <= 180):
            raise ValueError(
                f"{_constraint_label(f'{item_path}.angle', style)} must be between 0 and 180 degrees."
            )
        angle_entries.append((angle_i, angle_j, angle_k, angle_value))

    dihedrals = _constraint_entries(constraints, "dihedrals", style)
    for idx, dihedral in enumerate(dihedrals):
        item_path = f"constraints.dihedrals[{idx}]"
        if not isinstance(dihedral, dict):
            raise ValueError(f"{_constraint_label(item_path, style)} must be an object.")
        for key in ("i", "j", "k", "l"):
            if key not in dihedral:
                raise ValueError(
                    f"{_constraint_label(item_path, style)} must define '{key}'."
                )
        if "dihedral" not in dihedral:
            raise ValueError(
                f"{_constraint_label(item_path, style)} must define 'dihedral'."
            )
        dihedral_i = _validate_constraint_index(
            dihedral["i"], f"{item_path}.i", atom_count, style
        )
        dihedral_j = _validate_constraint_index(
            dihedral["j"], f"{item_path}.j", atom_count, style
        )
        dihedral_k = _validate_constraint_index(
            dihedral["k"], f"{item_path}.k", atom_count, style
        )
        dihedral_l = _validate_constraint_index(
            dihedral["l"], f"{item_path}.l", atom_count, style
        )
        dihedral_value = _validate_constraint_number(
            dihedral["dihedral"], f"{item_path}.dihedral", style
        )
        if not (-180 <= dihedral_value <= 180):
            raise ValueError(
                f"{_constraint_label(f'{item_path}.dihedral', style)} must be between -180 and 180 degrees."
            )
        dihedral_entries.append(
            (dihedral_i, dihedral_j, dihedral_k, dihedral_l, dihedral_value)
        )

    return bond_entries, angle_entries, dihedral_entries
