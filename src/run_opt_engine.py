import logging
import math
import os
import re
import time
from functools import lru_cache

from env_compat import getenv_str
from run_opt_config import (
    DEFAULT_CHARGE,
    DEFAULT_MULTIPLICITY,
    DEFAULT_SPIN,
    SMD_UNSUPPORTED_SOLVENT_KEYS,
)
from run_opt_dispersion import load_d3_calculator, parse_dispersion_settings
from run_opt_resources import ensure_parent_dir, resolve_run_path
from run_opt_utils import (
    extract_step_count,
    normalize_constraints,
    normalize_solvent_key,
)


def normalize_xc_functional(xc):
    if not xc:
        return xc
    normalized = re.sub(r"[\s_\-]+", "", str(xc)).lower()
    mapping = {
        "wb97xd": "WB97X_D",
        "wb97xd3": "WB97X_D3",
    }
    return mapping.get(normalized, xc)


def _compute_dispersion_energy(atoms, dispersion_settings):
    backend = dispersion_settings["backend"]
    settings = dispersion_settings["settings"]
    if backend == "d3":
        d3_cls, _ = load_d3_calculator()
        if d3_cls is None:
            raise ImportError(
                "DFTD3 dispersion requested but no DFTD3 calculator is available. "
                "Install `dftd3` (recommended)."
            )
        d3_calc = d3_cls(atoms=atoms, **settings)
        return d3_calc.get_potential_energy(atoms=atoms), "dftd3"
    from dftd4.ase import DFTD4

    d4_calc = DFTD4(atoms=atoms, **settings)
    return d4_calc.get_potential_energy(atoms=atoms), "ase-dftd4"


def _collect_constraint_jacobians(atoms, constraints):
    import numpy as np
    from ase.constraints import FixInternals

    bond_entries, angle_entries, dihedral_entries = normalize_constraints(
        constraints,
        atom_count=len(atoms),
    )
    if not bond_entries and not angle_entries and not dihedral_entries:
        return None

    fix = FixInternals(
        bonds=[[length, [i, j]] for i, j, length in bond_entries] or None,
        angles_deg=[[angle, [i, j, k]] for i, j, k, angle in angle_entries] or None,
        dihedrals_deg=[
            [dihedral, [i, j, k, atom_l]]
            for i, j, k, atom_l, dihedral in dihedral_entries
        ]
        or None,
    )
    fix.initialize(atoms)
    jacobians = []
    atom_count = len(atoms)
    for constraint in fix.constraints:
        constraint.setup_jacobian(atoms.positions)
        jac = np.asarray(constraint.jacobian, dtype=float).reshape(-1)
        if jac.shape[0] != atom_count * 3:
            raise ValueError("Constraint jacobian shape does not match coordinates.")
        jacobians.append(jac)
    if not jacobians:
        return None
    return np.vstack(jacobians)


def _project_hessian_constraints(hess, mol, constraint_jacobians):
    import numpy as np

    hess_array = np.asarray(hess, dtype=float)
    natm = mol.natm
    if hess_array.ndim == 4:
        hess_2d = hess_array.reshape(natm * 3, natm * 3)
    else:
        hess_2d = hess_array
    masses = np.asarray(mol.atom_mass_list(isotope_avg=True), dtype=float)
    if masses.size != natm:
        raise ValueError("Atomic mass list does not match atom count.")
    mass_vector = np.repeat(masses, 3)
    if np.any(mass_vector <= 0):
        raise ValueError("Invalid atomic masses encountered for constraint projection.")
    sqrt_mass = np.sqrt(mass_vector)
    hess_mw = hess_2d / np.outer(sqrt_mass, sqrt_mass)
    jac = np.asarray(constraint_jacobians, dtype=float)
    if jac.ndim == 1:
        jac = jac[None, :]
    if jac.shape[1] != hess_mw.shape[0]:
        raise ValueError("Constraint jacobian dimension does not match Hessian.")
    jac_mw = jac / sqrt_mass
    q, _ = np.linalg.qr(jac_mw.T)
    if q.size == 0:
        return hess
    projector = np.eye(hess_mw.shape[0]) - q @ q.T
    hess_mw_proj = projector.T @ hess_mw @ projector
    hess_mw_proj = 0.5 * (hess_mw_proj + hess_mw_proj.T)
    hess_proj = hess_mw_proj * np.outer(sqrt_mass, sqrt_mass)
    if hess_array.ndim == 4:
        return hess_proj.reshape(natm, natm, 3, 3)
    return hess_proj


def parse_xyz_metadata(xyz_lines):
    """
    Parse charge/spin metadata from the comment line of an XYZ file.

    Expected format in the second line (comment), for example:
      charge=0 spin=0
    Multiplicity is optional and parsed for compatibility.

    Returns (charge, spin, multiplicity), defaulting to charge 0 and None otherwise.
    """
    if len(xyz_lines) < 2:
        return DEFAULT_CHARGE, DEFAULT_SPIN, DEFAULT_MULTIPLICITY

    comment = xyz_lines[1]
    charge = DEFAULT_CHARGE
    spin = DEFAULT_SPIN
    multiplicity = DEFAULT_MULTIPLICITY

    metadata_pattern = re.compile(
        r"(charge|spin|multiplicity)\s*[:=]\s*([^\s,;]+)", re.I
    )
    integer_pattern = re.compile(r"[+-]?\d+$")
    for match in metadata_pattern.finditer(comment):
        key = match.group(1).lower()
        raw_value = match.group(2)
        if not integer_pattern.fullmatch(raw_value):
            raise ValueError(
                f"Invalid {key} value in XYZ comment: {raw_value!r}. "
                "Expected an integer (e.g., charge=0)."
            )
        value = int(raw_value)
        if key == "charge":
            charge = value
        elif key == "spin":
            spin = value
        elif key == "multiplicity":
            multiplicity = value

    return charge, spin, multiplicity


def load_xyz(xyz_path):
    """
    Load an XYZ file and return (atom_spec, charge, spin, multiplicity).
    """
    with open(xyz_path, "r", encoding="utf-8") as xyz_file:
        lines = [line.rstrip("\n") for line in xyz_file]

    if not lines:
        raise ValueError(f"XYZ file is empty: {xyz_path}")
    try:
        expected_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(
            f"XYZ first line must be an integer atom count: {lines[0]!r}"
        ) from exc
    if expected_atoms <= 0:
        raise ValueError(
            f"XYZ atom count must be positive; got {expected_atoms} in {xyz_path}"
        )
    if len(lines) < 2:
        raise ValueError(
            f"XYZ file is missing a comment line; expected at least 2 lines in {xyz_path}"
        )

    charge, spin, multiplicity = parse_xyz_metadata(lines)
    atom_lines = [line for line in lines[2:] if line.strip()]
    if len(atom_lines) != expected_atoms:
        raise ValueError(
            "XYZ atom count mismatch in {path}: header says {expected}, "
            "but found {actual} atom lines.".format(
                path=xyz_path,
                expected=expected_atoms,
                actual=len(atom_lines),
            )
        )
    for index, line in enumerate(atom_lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"XYZ atom line {index} must have at least 4 columns: {line!r}"
            )
        coords = parts[1:4]
        try:
            [float(value) for value in coords]
        except ValueError as exc:
            raise ValueError(
                f"XYZ atom line {index} has invalid coordinates: {line!r}"
            ) from exc
    atom_spec = "\n".join(atom_lines)
    return atom_spec, charge, spin, multiplicity


def normalized_symbol(symbol):
    return symbol[0].upper() + symbol[1:].lower()


def atomic_number_from_token(token):
    if re.fullmatch(r"\d+", token):
        return int(token)
    match = re.match(r"^(?:\d+)?([A-Za-z]+)", token)
    if not match:
        raise ValueError(f"Unable to parse element token '{token}'.")
    symbol = normalized_symbol(match.group(1))
    try:
        from pyscf.data import elements

        return elements.charge(symbol)
    except Exception as exc:
        raise ValueError(f"Unable to parse element symbol '{symbol}' from '{token}'.") from exc


def total_electron_count(atom_spec, charge):
    total_charge = 0
    for line in atom_spec.splitlines():
        parts = line.split()
        if not parts:
            continue
        total_charge += atomic_number_from_token(parts[0])
    return total_charge - charge


def is_density_fit_gradient_einsum_error(exc):
    if not isinstance(exc, ValueError):
        return False
    message = str(exc)
    return "not enough values to unpack" in message and "expected 4, got 3" in message


def _apply_density_fit_setting(mf, density_fit):
    if isinstance(density_fit, bool):
        if density_fit:
            if not hasattr(mf, "density_fit"):
                raise ValueError("Density fitting is not supported for this SCF object.")
            mf = mf.density_fit()
    elif isinstance(density_fit, str):
        if not density_fit.strip():
            raise ValueError(
                "Config 'scf.extra.density_fit' must be a boolean or non-empty string."
            )
        if not hasattr(mf, "density_fit"):
            raise ValueError("Density fitting is not supported for this SCF object.")
        density_fit_key = density_fit.strip().lower()
        if density_fit_key == "autoaux":
            import pyscf.df

            mf = mf.density_fit(auxbasis=pyscf.df.autoaux(mf.mol))
        else:
            mf = mf.density_fit(auxbasis=density_fit)
    else:
        raise ValueError("Config 'scf.extra.density_fit' must be a boolean or a string.")
    return mf


def apply_density_fit_setting(mf, scf_config):
    if not scf_config:
        return mf, {}
    extra = scf_config.get("extra") or {}
    if "density_fit" not in extra:
        return mf, {}
    density_fit = extra.get("density_fit")
    mf = _apply_density_fit_setting(mf, density_fit)
    return mf, {"density_fit": density_fit}


def apply_scf_settings(mf, scf_config, *, apply_density_fit=True):
    if not scf_config:
        return mf, {}
    applied = {}
    extra = scf_config.get("extra") or {}
    applied_extra = {}
    if "density_fit" in extra:
        density_fit = extra.get("density_fit")
        if apply_density_fit:
            mf = _apply_density_fit_setting(mf, density_fit)
            applied_extra["density_fit"] = density_fit
    max_cycle = scf_config.get("max_cycle")
    conv_tol = scf_config.get("conv_tol")
    diis = scf_config.get("diis")
    diis_preset = scf_config.get("diis_preset")
    level_shift = scf_config.get("level_shift")
    damping = scf_config.get("damping")

    if diis is None and diis_preset is not None:
        preset_key = _normalize_diis_preset(diis_preset)
        diis = DIIS_PRESET_VALUES[preset_key]
        applied["diis_preset"] = preset_key

    if max_cycle is not None:
        mf.max_cycle = int(max_cycle)
        applied["max_cycle"] = mf.max_cycle
    if conv_tol is not None:
        mf.conv_tol = float(conv_tol)
        applied["conv_tol"] = mf.conv_tol
    if level_shift is not None:
        mf.level_shift = float(level_shift)
        applied["level_shift"] = mf.level_shift
    if damping is not None:
        mf.damp = float(damping)
        applied["damping"] = mf.damp
    if diis is False:
        mf.diis = None
        applied["diis"] = False
    elif isinstance(diis, int):
        mf.diis_space = int(diis)
        applied["diis"] = mf.diis_space
    elif diis is True:
        applied["diis"] = True

    if extra:
        grids = extra.get("grids") or {}
        if grids:
            applied_grids = {}
            if "level" in grids:
                mf.grids.level = int(grids.get("level"))
                applied_grids["level"] = mf.grids.level
            if "prune" in grids:
                prune_value = grids.get("prune")
                allowed_prune_types = (bool, str, tuple, list, dict)
                if not (
                    prune_value is None
                    or callable(prune_value)
                    or isinstance(prune_value, allowed_prune_types)
                ):
                    raise ValueError(
                        "Config 'scf.extra.grids.prune' must be a bool, string, "
                        "tuple, list, dict, callable, or null."
                    )
                mf.grids.prune = prune_value
                applied_grids["prune"] = prune_value
            if applied_grids:
                applied_extra["grids"] = applied_grids
        if "init_guess" in extra:
            init_guess = extra.get("init_guess")
            mf.init_guess = init_guess
            applied_extra["init_guess"] = init_guess
    if applied_extra:
        applied["extra"] = applied_extra

    return mf, applied


def apply_scf_checkpoint(mf, scf_config, run_dir=None):
    if not scf_config:
        return None, None
    extra = scf_config.get("extra") or {}
    init_guess_setting = extra.get("init_guess") if isinstance(extra, dict) else None
    chkfile_setting = scf_config.get("chkfile")
    if not chkfile_setting:
        return None, None
    chkfile_path = resolve_run_path(run_dir, chkfile_setting) if run_dir else chkfile_setting
    ensure_parent_dir(chkfile_path)
    mf.chkfile = chkfile_path
    if chkfile_path and os.path.exists(chkfile_path):
        from pyscf.scf import chkfile as scf_chkfile

        try:
            dm0 = scf_chkfile.load(chkfile_path, "scf/dm")
        except Exception:
            dm0 = None
        if dm0 is None:
            if init_guess_setting is None:
                mf.init_guess = "chkfile"
                logging.info(
                    "SCF chkfile found at %s; using init_guess='chkfile'.",
                    chkfile_path,
                )
            else:
                logging.info(
                    "SCF chkfile found at %s; keeping user init_guess=%r.",
                    chkfile_path,
                    init_guess_setting,
                )
        elif init_guess_setting is not None:
            logging.info(
                "SCF chkfile found at %s; using density matrix from chkfile "
                "(init_guess=%r ignored).",
                chkfile_path,
                init_guess_setting,
            )
        return dm0, chkfile_path
    return None, chkfile_path


def _scf_retry_enabled():
    value = getenv_str("PYSCF_AUTO_SCF_RETRY", "1")
    value = value.strip().lower()
    return value not in ("0", "false", "no", "off")


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


DIIS_PRESET_VALUES = {
    "fast": 6,
    "default": 8,
    "stable": 12,
    "off": False,
}

DIIS_PRESET_ALIASES = {
    "conservative": "fast",
    "aggressive": "stable",
    "robust": "stable",
    "none": "off",
    "disabled": "off",
}

SCF_RETRY_PRESET_VALUES = {
    "fast": (
        {"level_shift": 0.3, "damping": 0.1, "max_cycle": 50},
    ),
    "default": (
        {"level_shift": 0.5, "damping": 0.2, "max_cycle": 50},
        {"level_shift": 1.0, "damping": 0.3, "max_cycle": 200},
    ),
    "stable": (
        {"level_shift": 0.5, "damping": 0.2, "max_cycle": 50},
        {"level_shift": 1.0, "damping": 0.3, "max_cycle": 200},
        {"level_shift": 2.0, "damping": 0.4, "max_cycle": 400},
    ),
    "off": (),
}

SCF_RETRY_PRESET_ALIASES = {
    "conservative": "fast",
    "aggressive": "stable",
    "robust": "stable",
    "none": "off",
    "disabled": "off",
}


def _normalize_diis_preset(value):
    if value is None:
        return None
    normalized = re.sub(r"[\s_\-]+", "", str(value)).lower()
    normalized = DIIS_PRESET_ALIASES.get(normalized, normalized)
    if normalized not in DIIS_PRESET_VALUES:
        allowed = ", ".join(sorted(DIIS_PRESET_VALUES))
        raise ValueError(f"Invalid scf.diis_preset: {value!r}. Allowed: {allowed}.")
    return normalized


def _normalize_scf_retry_preset(value):
    if value is None:
        return "default"
    normalized = re.sub(r"[\s_\-]+", "", str(value)).lower()
    normalized = SCF_RETRY_PRESET_ALIASES.get(normalized, normalized)
    if normalized not in SCF_RETRY_PRESET_VALUES:
        allowed = ", ".join(sorted(SCF_RETRY_PRESET_VALUES))
        raise ValueError(f"Invalid scf.retry_preset: {value!r}. Allowed: {allowed}.")
    return normalized


def _merge_scf_config(base_config, overrides):
    merged = dict(base_config or {})
    merged.update(overrides or {})
    return merged


def _build_scf_retry_overrides(scf_config):
    base_config = scf_config or {}
    preset = _normalize_scf_retry_preset(base_config.get("retry_preset"))
    retry_targets = SCF_RETRY_PRESET_VALUES.get(preset, ())
    if not retry_targets:
        return []
    base_level = _coerce_float(base_config.get("level_shift"))
    base_damping = _coerce_float(base_config.get("damping"))
    base_max = _coerce_int(base_config.get("max_cycle"))
    retries = []
    for target in retry_targets:
        overrides = {}
        if base_level is None or base_level < target["level_shift"]:
            overrides["level_shift"] = target["level_shift"]
        if base_damping is None or base_damping < target["damping"]:
            overrides["damping"] = target["damping"]
        if base_max is None or base_max < target["max_cycle"]:
            overrides["max_cycle"] = target["max_cycle"]
        if overrides:
            retries.append(overrides)
    return retries


def _format_scf_retry_overrides(overrides):
    parts = []
    for key in ("level_shift", "damping", "max_cycle"):
        if key in overrides:
            parts.append(f"{key}={overrides[key]}")
    return ", ".join(parts) if parts else "no changes"


def _is_scf_converged(mf):
    converged = getattr(mf, "converged", None)
    if converged is None:
        return True
    return bool(converged)


def _unpack_mf_builder_result(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def _run_scf_with_retries(build_mf, scf_config, run_dir, label):
    base_config = dict(scf_config or {})

    def _run_once(config):
        mf, info = _unpack_mf_builder_result(build_mf(config))
        dm0, _ = apply_scf_checkpoint(mf, config, run_dir=run_dir)
        if dm0 is not None:
            energy_value = mf.kernel(dm0=dm0)
        else:
            energy_value = mf.kernel()
        return energy_value, mf, info

    energy, mf, info = _run_once(base_config)
    if _is_scf_converged(mf):
        return energy, mf, info
    if not _scf_retry_enabled():
        return energy, mf, info
    retries = _build_scf_retry_overrides(base_config)
    if not retries:
        return energy, mf, info
    for attempt, overrides in enumerate(retries, start=1):
        retry_config = _merge_scf_config(base_config, overrides)
        logging.warning(
            "%s did not converge; retrying with SCF settings (%s).",
            label,
            _format_scf_retry_overrides(overrides),
        )
        energy, mf, info = _run_once(retry_config)
        if _is_scf_converged(mf):
            logging.info("%s converged after SCF retry %s.", label, attempt)
            return energy, mf, info
    logging.warning(
        "%s did not converge after %s retries; proceeding with last result.",
        label,
        len(retries),
    )
    return energy, mf, info


def select_ks_type(
    mol=None,
    spin=None,
    scf_config=None,
    optimizer_mode=None,
    multiplicity=None,
    log_override=True,
):
    if mol is None and spin is None:
        raise ValueError("select_ks_type requires either mol or spin.")
    resolved_spin = mol.spin if mol is not None else spin
    default_type = "RKS" if resolved_spin == 0 else "UKS"
    reference = None
    if scf_config is not None:
        reference = scf_config.get("reference")
    if reference is not None:
        reference = str(reference).strip().lower()
    if reference in ("rks", "uks"):
        requested_type = "RKS" if reference == "rks" else "UKS"
        if log_override:
            logging.warning(
                "SCF reference requested (%s): using %s (default %s).",
                reference,
                requested_type,
                default_type,
            )
        return requested_type
    return default_type


def _smd_available(mf):
    if not hasattr(mf, "SMD"):
        return False
    try:
        from pyscf.solvent import smd
    except Exception:
        return False
    return getattr(smd, "libsolvent", None) is not None


def apply_solvent_model(
    mf,
    solvent_model,
    solvent_name=None,
    solvent_eps=None,
):
    if solvent_model is None:
        return mf
    if solvent_name is None or solvent_name.strip().lower() == "vacuum":
        return mf
    model = solvent_model.lower()
    if model == "pcm":
        if solvent_eps is None:
            raise ValueError("PCM solvent model requires a dielectric constant.")
        mf = mf.PCM()
        mf.with_solvent.eps = solvent_eps
    elif model == "smd":
        if not solvent_name:
            raise ValueError("SMD solvent model requires a solvent name.")
        if not _smd_available(mf):
            raise ValueError(
                "SMD solvent model is unavailable in this PySCF build. "
                "Install the SMD-enabled PySCF package from the pyscf_auto conda channel."
            )
        supported = _supported_smd_solvents()
        supported_map = _cached_smd_supported_map()
        normalized = normalize_solvent_key(solvent_name)
        if normalized in SMD_UNSUPPORTED_SOLVENT_KEYS:
            raise ValueError(
                "SMD solvent '{name}' is not supported by PySCF SMD. "
                "Use PCM or choose another solvent.".format(name=solvent_name)
            )
        resolved = supported_map.get(normalized)
        if resolved is None:
            preview = ", ".join(sorted(supported)[:10])
            suggestion = _SMD_SOLVENT_ALIASES.get(normalized)
            hint = f" Try '{suggestion}'." if suggestion else ""
            raise ValueError(
                "SMD solvent '{name}' not found in supported list (showing first 10 "
                "of {count}: {preview}).{hint}".format(
                    name=solvent_name,
                    count=len(supported),
                    preview=preview,
                    hint=hint,
                )
            )
        mf = mf.SMD()
        mf.with_solvent.solvent = resolved
    else:
        raise ValueError(f"Unsupported solvent model '{solvent_model}'.")
    return mf

_SMD_SOLVENT_ALIASES = {
    normalize_solvent_key("diethyl ether"): "diethylether",
    normalize_solvent_key("isopropanol"): "2-propanol",
    normalize_solvent_key("dmso"): "dimethylsulfoxide",
    normalize_solvent_key("ethylene glycol"): "1,2-ethanediol",
    normalize_solvent_key("ethyl acetate"): "ethylethanoate",
    normalize_solvent_key("hexane"): "n-hexane",
    normalize_solvent_key("dmf"): "N,N-dimethylformamide",
    normalize_solvent_key("heptane"): "n-heptane",
}


def _build_smd_supported_map(supported=None):
    if supported is None:
        supported = _supported_smd_solvents()
    supported_map = {}
    for name in supported:
        key = normalize_solvent_key(name)
        if key and key not in supported_map:
            supported_map[key] = name
    for alias_key, canonical in _SMD_SOLVENT_ALIASES.items():
        canonical_key = normalize_solvent_key(canonical)
        canonical_name = supported_map.get(canonical_key)
        if canonical_name and alias_key not in supported_map:
            supported_map[alias_key] = canonical_name
    return supported_map


@lru_cache(maxsize=1)
def _cached_smd_supported_map():
    return _build_smd_supported_map(_supported_smd_solvents())


@lru_cache(maxsize=1)
def _supported_smd_solvents():
    try:
        from pyscf.solvent import smd
    except Exception as exc:
        raise ValueError("Unable to import PySCF SMD solvent data.") from exc

    candidates = []

    def add_candidates(value):
        if isinstance(value, dict):
            candidates.extend(value.keys())
        elif isinstance(value, (list, tuple, set)):
            candidates.extend(value)

    for obj in (smd, getattr(smd, "SMD", None)):
        if obj is None:
            continue
        for attr in (
            "solvent_db",
            "solvent_dict",
            "solvent_params",
            "solvent_param",
            "solvents",
            "SOLVENTS",
        ):
            add_candidates(getattr(obj, attr, None))
        for name in (
            "get_smd_solvents",
            "get_solvents",
            "solvent_list",
            "available_solvents",
        ):
            func = getattr(obj, name, None)
            if callable(func):
                try:
                    add_candidates(func())
                except Exception:
                    continue

    supported = sorted(
        {
            candidate
            for candidate in candidates
            if isinstance(candidate, str) and candidate.strip()
        }
    )
    if not supported:
        raise ValueError("Unable to determine supported SMD solvents from PySCF.")
    return tuple(supported)


def compute_single_point_energy(
    mol,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    dispersion,
    dispersion_params,
    verbose,
    memory_mb,
    run_dir=None,
    optimizer_mode=None,
    multiplicity=None,
    profiling_enabled=False,
    log_override=True,
):
    from ase import units
    from ase import Atoms
    from pyscf import dft

    xc = normalize_xc_functional(xc)
    mol_sp = mol.copy()
    if basis:
        mol_sp.basis = basis
        mol_sp.build()
    if memory_mb:
        mol_sp.max_memory = memory_mb
    def _build_mf_sp(scf_settings):
        ks_type = select_ks_type(
            mol=mol_sp,
            scf_config=scf_settings,
            optimizer_mode=optimizer_mode,
            multiplicity=multiplicity,
            log_override=log_override,
        )
        if ks_type == "RKS":
            mf_sp = dft.RKS(mol_sp)
        else:
            mf_sp = dft.UKS(mol_sp)
        mf_sp.xc = xc
        mf_sp, _ = apply_density_fit_setting(mf_sp, scf_settings)
        if solvent_model is not None:
            mf_sp = apply_solvent_model(
                mf_sp,
                solvent_model,
                solvent_name,
                solvent_eps,
            )
        if verbose:
            mf_sp.verbose = 4
        mf_sp, _ = apply_scf_settings(mf_sp, scf_settings, apply_density_fit=False)
        return mf_sp

    scf_start = time.perf_counter() if profiling_enabled else None
    energy, mf_sp, _ = _run_scf_with_retries(
        _build_mf_sp, scf_config, run_dir, "Single-point SCF"
    )
    scf_seconds = None
    if scf_start is not None:
        scf_seconds = time.perf_counter() - scf_start
    dispersion_info = None
    if dispersion is not None:
        dispersion_settings = parse_dispersion_settings(
            dispersion,
            xc,
            charge=mol_sp.charge,
            spin=mol_sp.spin,
            d3_params=dispersion_params,
        )
        positions = mol_sp.atom_coords(unit="Angstrom")
        if hasattr(mol_sp, "atom_symbols"):
            symbols = mol_sp.atom_symbols()
        else:
            symbols = [mol_sp.atom_symbol(i) for i in range(mol_sp.natm)]
        atoms = Atoms(symbols=symbols, positions=positions)
        dispersion_energy_ev, dispersion_backend = _compute_dispersion_energy(
            atoms, dispersion_settings
        )
        dispersion_energy_hartree = dispersion_energy_ev / units.Hartree
        energy += dispersion_energy_hartree
        dispersion_info = {
            "model": dispersion,
            "energy_hartree": dispersion_energy_hartree,
            "energy_ev": dispersion_energy_ev,
            "backend": dispersion_backend,
        }
    result = {
        "energy": energy,
        "converged": getattr(mf_sp, "converged", None),
        "cycles": extract_step_count(mf_sp),
        "dispersion": dispersion_info,
    }
    if profiling_enabled:
        result["profiling"] = {
            "scf_seconds": scf_seconds,
            "scf_cycles": extract_step_count(mf_sp),
        }
    return result


DEFAULT_TS_IMAG_FREQ_MIN_ABS = 50.0
DEFAULT_TS_IMAG_FREQ_MAX_ABS = 1500.0
DEFAULT_TS_MODE_PROJECTION_STEP = 0.01


def _extract_imaginary_mode_from_hessian(hess, mol, atomic_masses, atomic_numbers):
    import numpy as np

    hess = np.asarray(hess)
    natm = mol.natm
    if hess.ndim == 4:
        hess = hess.reshape(natm * 3, natm * 3)
    symbols = [mol.atom_symbol(i) for i in range(natm)]
    masses = [atomic_masses[atomic_numbers[symbol]] for symbol in symbols]
    mass_vector = np.repeat(np.asarray(masses, dtype=float), 3)
    sqrt_mass = np.sqrt(mass_vector)
    mass_weight = np.outer(sqrt_mass, sqrt_mass)
    hess_mw = hess / mass_weight
    eigvals, eigvecs = np.linalg.eigh(hess_mw)
    min_index = int(np.argmin(eigvals))
    mode_mw = eigvecs[:, min_index]
    mode_cart = mode_mw / sqrt_mass
    norm = np.linalg.norm(mode_cart)
    if norm == 0:
        raise ValueError("Failed to normalize imaginary mode; eigenvector norm is zero.")
    mode_cart = mode_cart / norm
    return {
        "mode": mode_cart.reshape(natm, 3),
        "eigenvalue": float(eigvals[min_index]),
        "natoms": natm,
        "symbols": symbols,
    }


def _evaluate_internal_coordinate(kind, positions, i, j, k=None, l_index=None):
    import numpy as np

    if kind == "bond":
        return float(np.linalg.norm(positions[i] - positions[j]))
    if kind == "angle":
        v1 = positions[i] - positions[j]
        v2 = positions[k] - positions[j]
        dot = np.dot(v1, v2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            raise ValueError("Zero-length vector encountered while computing angle.")
        cos_angle = np.clip(dot / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
    if kind == "dihedral":
        p0 = positions[i]
        p1 = positions[j]
        p2 = positions[k]
        p3 = positions[l_index]
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = np.linalg.norm(b1)
        if b1_norm == 0:
            raise ValueError("Zero-length bond encountered while computing dihedral.")
        b1_unit = b1 / b1_norm
        v = b0 - np.dot(b0, b1_unit) * b1_unit
        w = b2 - np.dot(b2, b1_unit) * b1_unit
        x = np.dot(v, w)
        y = np.dot(np.cross(b1_unit, v), w)
        return float(np.degrees(np.arctan2(y, x)))
    raise ValueError(f"Unsupported internal coordinate type: {kind}")


def _project_imaginary_mode_to_internal_coordinates(
    positions,
    mode,
    internal_coordinates,
    projection_step,
    projection_min_abs,
):
    import numpy as np

    if not internal_coordinates:
        return {
            "status": "not_configured",
            "projection_step": projection_step,
            "projection_min_abs": projection_min_abs,
            "coordinates": [],
        }
    results = []
    positions = np.asarray(positions, dtype=float)
    mode = np.asarray(mode, dtype=float)
    shifted_plus = positions + projection_step * mode
    shifted_minus = positions - projection_step * mode
    for entry in internal_coordinates:
        kind = entry.get("type")
        i = entry.get("i")
        j = entry.get("j")
        k = entry.get("k")
        l_index = entry.get("l")
        target = entry.get("target")
        direction = entry.get("direction")
        tolerance = entry.get("tolerance")
        current = _evaluate_internal_coordinate(kind, positions, i, j, k, l_index)
        value_plus = _evaluate_internal_coordinate(kind, shifted_plus, i, j, k, l_index)
        value_minus = _evaluate_internal_coordinate(kind, shifted_minus, i, j, k, l_index)
        projection = (value_plus - value_minus) / (2 * projection_step)
        desired_sign = None
        alignment = None
        delta = None
        if target is not None:
            delta = target - current
            if tolerance is not None and abs(delta) <= tolerance:
                alignment = "aligned"
            elif delta == 0:
                alignment = "aligned"
            else:
                desired_sign = 1.0 if delta > 0 else -1.0
        if desired_sign is None and direction:
            desired_sign = 1.0 if direction == "increase" else -1.0
        if alignment is None:
            if abs(projection) < projection_min_abs:
                alignment = "weak"
            elif desired_sign is None:
                alignment = "unknown"
            elif projection * desired_sign > 0:
                alignment = "aligned"
            else:
                alignment = "misaligned"
        results.append(
            {
                "type": kind,
                "i": i,
                "j": j,
                "k": k,
                "l": l_index,
                "current": current,
                "target": target,
                "delta_to_target": delta,
                "direction": direction,
                "tolerance": tolerance,
                "projection": float(projection),
                "status": alignment,
            }
        )
    statuses = {item["status"] for item in results}
    if "misaligned" in statuses:
        overall = "misaligned"
    elif "weak" in statuses:
        overall = "weak"
    elif "unknown" in statuses:
        overall = "unknown"
    else:
        overall = "aligned"
    return {
        "status": overall,
        "projection_step": projection_step,
        "projection_min_abs": projection_min_abs,
        "coordinates": results,
    }


def _atoms_from_molecule(mol_handle, atoms_cls):
    positions = mol_handle.atom_coords(unit="Angstrom")
    if hasattr(mol_handle, "atom_symbols"):
        symbols = mol_handle.atom_symbols()
    else:
        symbols = [mol_handle.atom_symbol(i) for i in range(mol_handle.natm)]
    return atoms_cls(symbols=symbols, positions=positions)


def _build_dispersion_calculator(atoms, dispersion_settings):
    backend = dispersion_settings["backend"]
    settings = dispersion_settings["settings"]
    if backend == "d3":
        d3_cls, _ = load_d3_calculator()
        if d3_cls is None:
            raise ImportError(
                "DFTD3 dispersion requested but no DFTD3 calculator is available. "
                "Install `dftd3` (recommended)."
            )
        return d3_cls(atoms=atoms, **settings), "dftd3"
    from dftd4.ase import DFTD4

    return DFTD4(atoms=atoms, **settings), "ase-dftd4"


def _compute_dispersion_hessian_numerical(atoms, step):
    import numpy as np

    base_positions = atoms.get_positions()
    natoms = len(base_positions)
    hessian = np.zeros((natoms, natoms, 3, 3), dtype=float)
    for atom_idx in range(natoms):
        for coord in range(3):
            pos_plus = base_positions.copy()
            pos_plus[atom_idx, coord] += step
            atoms.set_positions(pos_plus)
            forces_plus = atoms.get_forces()
            pos_minus = base_positions.copy()
            pos_minus[atom_idx, coord] -= step
            atoms.set_positions(pos_minus)
            forces_minus = atoms.get_forces()
            delta_forces = (forces_plus - forces_minus) / (2.0 * step)
            hessian[:, atom_idx, :, coord] = -delta_forces
    atoms.set_positions(base_positions)
    return hessian


def _prepare_frequency_dispersion_payload(
    *,
    mol_freq,
    atoms,
    atoms_cls,
    dispersion,
    xc,
    dispersion_hessian_mode,
    dispersion_hessian_step,
    dispersion_params,
    profiling,
    units,
):
    payload = {"energy_hartree": 0.0, "info": None, "hessian": None}
    if dispersion is None:
        return payload, atoms

    if atoms is None:
        atoms = _atoms_from_molecule(mol_freq, atoms_cls)
    dispersion_settings = parse_dispersion_settings(
        dispersion,
        xc,
        charge=mol_freq.charge,
        spin=mol_freq.spin,
        d3_params=dispersion_params,
    )
    dispersion_calc, dispersion_backend = _build_dispersion_calculator(
        atoms, dispersion_settings
    )
    atoms.calc = dispersion_calc
    dispersion_energy_ev = dispersion_calc.get_potential_energy(atoms=atoms)
    dispersion_energy_hartree = dispersion_energy_ev / units.Hartree
    payload["energy_hartree"] = dispersion_energy_hartree
    dispersion_info = {
        "model": dispersion,
        "energy_hartree": dispersion_energy_hartree,
        "energy_ev": dispersion_energy_ev,
        "backend": dispersion_backend,
        "hessian_mode": dispersion_hessian_mode,
    }
    if dispersion_hessian_mode == "numerical":
        step = dispersion_hessian_step if dispersion_hessian_step is not None else 0.005
        dispersion_hessian_start = time.perf_counter() if profiling else None
        dispersion_hessian = _compute_dispersion_hessian_numerical(atoms, step)
        if dispersion_hessian_start is not None:
            profiling["dispersion_hessian_seconds"] = (
                time.perf_counter() - dispersion_hessian_start
            )
        dispersion_hessian *= (1.0 / units.Hartree) / (units.Bohr ** 2)
        payload["hessian"] = dispersion_hessian
        dispersion_info["hessian_step"] = step
    payload["info"] = dispersion_info
    return payload, atoms


def _apply_dispersion_payload(energy_value, dispersion_payload):
    if dispersion_payload["info"] is None:
        return energy_value, None, None
    energy_value += dispersion_payload["energy_hartree"]
    return energy_value, dispersion_payload["info"], dispersion_payload["hessian"]


def _to_list(values):
    if values is None:
        return None
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _to_scalar(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _build_thermochemistry_payload(
    *,
    mf_freq,
    freq_au,
    thermo,
    zpe_value,
    energy,
    dispersion_info,
    solvent_model,
    solvent_name,
    nist,
    pyscf_thermo,
):
    thermo_settings = None
    if thermo:
        if hasattr(thermo, "to_dict"):
            thermo_settings = thermo.to_dict()
        elif isinstance(thermo, dict):
            thermo_settings = thermo
    if not thermo_settings:
        return None

    temperature = thermo_settings.get("T")
    pressure = thermo_settings.get("P")
    pressure_unit = thermo_settings.get("unit")
    thermo_result = None
    if freq_au is not None:
        try:
            thermo_result = pyscf_thermo.thermo(
                mf_freq,
                freq_au,
                temperature=temperature,
                pressure=pressure,
                unit=pressure_unit,
            )
        except TypeError as exc:
            if "unit" not in str(exc):
                raise
            thermo_result = pyscf_thermo.thermo(
                mf_freq,
                freq_au,
                temperature=temperature,
                pressure=pressure,
            )

    def _thermo_value(keys):
        if not thermo_result:
            return None
        for key in keys:
            if key in thermo_result:
                return thermo_result[key]
        return None

    zpe_thermo = _thermo_value(
        ("zpe", "ZPE", "zpve", "ZPVE", "zero_point_energy", "zero_point_energy_hartree")
    )
    enthalpy_total = _thermo_value(("H", "enthalpy"))
    gibbs_total = _thermo_value(("G", "gibbs"))
    entropy = _thermo_value(("S", "entropy"))
    zpe_for_thermo = _to_scalar(zpe_thermo) if zpe_thermo is not None else zpe_value
    enthalpy_total_value = _to_scalar(enthalpy_total)
    gibbs_total_value = _to_scalar(gibbs_total)
    entropy_value = _to_scalar(entropy)
    dispersion_energy_hartree = (
        dispersion_info.get("energy_hartree") if dispersion_info else None
    )
    if dispersion_energy_hartree is not None:
        if enthalpy_total_value is not None:
            enthalpy_total_value += dispersion_energy_hartree
        if gibbs_total_value is not None:
            gibbs_total_value += dispersion_energy_hartree

    standard_state = None
    standard_state_correction = None
    if solvent_model and solvent_name and str(solvent_name).strip().lower() != "vacuum":
        temperature_value = None
        pressure_value_pa = None
        temp_entry = thermo_result.get("temperature") if thermo_result else None
        if isinstance(temp_entry, (list, tuple)) and temp_entry:
            temperature_value = _to_scalar(temp_entry[0])
        elif temp_entry is not None:
            temperature_value = _to_scalar(temp_entry)
        if temperature_value is None and temperature is not None:
            temperature_value = _to_scalar(temperature)
        pressure_entry = thermo_result.get("pressure") if thermo_result else None
        if isinstance(pressure_entry, (list, tuple)) and pressure_entry:
            pressure_value_pa = _to_scalar(pressure_entry[0])
        elif pressure_entry is not None:
            pressure_value_pa = _to_scalar(pressure_entry)
        if pressure_value_pa is None and pressure is not None:
            unit = str(pressure_unit or "pa").strip().lower()
            if unit in ("atm", "atmosphere", "atmospheres"):
                pressure_value_pa = float(pressure) * 101325.0
            elif unit in ("bar", "bars"):
                pressure_value_pa = float(pressure) * 100000.0
            elif unit in ("kpa",):
                pressure_value_pa = float(pressure) * 1000.0
            elif unit in ("mpa",):
                pressure_value_pa = float(pressure) * 1_000_000.0
            elif unit in ("torr", "mmhg"):
                pressure_value_pa = float(pressure) * 133.322368
            else:
                pressure_value_pa = float(pressure)
        if temperature_value and pressure_value_pa:
            ratio = (
                1000.0
                * nist.AVOGADRO
                * nist.BOLTZMANN
                * float(temperature_value)
                / float(pressure_value_pa)
            )
            if ratio > 0:
                standard_state_correction = (
                    nist.BOLTZMANN
                    * float(temperature_value)
                    * math.log(ratio)
                    / nist.HARTREE2J
                )
                standard_state = "1M"
                if gibbs_total_value is not None:
                    gibbs_total_value += standard_state_correction

    thermal_correction_enthalpy = None
    gibbs_correction = None
    gibbs_free_energy = None
    if enthalpy_total_value is not None:
        thermal_correction_enthalpy = enthalpy_total_value - energy
    if gibbs_total_value is not None:
        gibbs_correction = gibbs_total_value - energy
        gibbs_free_energy = energy + gibbs_correction

    return {
        "temperature": _to_scalar(temperature),
        "pressure": _to_scalar(pressure),
        "pressure_unit": pressure_unit,
        "zpe": zpe_for_thermo,
        "thermal_correction_enthalpy": _to_scalar(thermal_correction_enthalpy),
        "entropy": entropy_value,
        "gibbs_correction": _to_scalar(gibbs_correction),
        "gibbs_free_energy": _to_scalar(gibbs_free_energy),
        "standard_state": standard_state,
        "standard_state_correction": _to_scalar(standard_state_correction),
    }


def _summarize_imaginary_frequencies(freq_wavenumber_list):
    imaginary_count = None
    imaginary_status = None
    imaginary_message = None
    min_frequency = None
    max_frequency = None
    imaginary_frequencies = []
    if freq_wavenumber_list:
        imaginary_frequencies = [value for value in freq_wavenumber_list if value < 0]
        imaginary_count = len(imaginary_frequencies)
        min_frequency = min(freq_wavenumber_list)
        max_frequency = max(freq_wavenumber_list)
        if imaginary_count == 0:
            imaginary_status = "no_imaginary"
            imaginary_message = (
                "No imaginary frequencies found; optimized structure may be a minimum."
            )
        elif imaginary_count == 1:
            imaginary_status = "one_imaginary"
            imaginary_message = (
                "One imaginary frequency found; optimized structure is consistent with a TS."
            )
        else:
            imaginary_status = "multiple_imaginary"
            imaginary_message = (
                f"{imaginary_count} imaginary frequencies found; "
                "optimized structure may not be a first-order saddle point."
            )
    else:
        imaginary_status = "unknown"
        imaginary_message = "No frequency data available to assess imaginary modes."
    return {
        "imaginary_count": imaginary_count,
        "imaginary_status": imaginary_status,
        "imaginary_message": imaginary_message,
        "min_frequency": min_frequency,
        "max_frequency": max_frequency,
        "imaginary_frequencies": imaginary_frequencies,
    }


def _build_ts_quality_payload(
    *,
    ts_quality,
    optimizer_mode,
    imaginary_frequencies,
    imaginary_count,
    hess,
    mol_freq,
    atomic_masses,
    atomic_numbers,
):
    ts_quality_settings = {}
    if ts_quality is not None:
        if hasattr(ts_quality, "to_dict"):
            ts_quality_settings = ts_quality.to_dict()
        elif isinstance(ts_quality, dict):
            ts_quality_settings = dict(ts_quality)
    expected_imaginary_count = ts_quality_settings.get("expected_imaginary_count")
    if expected_imaginary_count is None and optimizer_mode == "transition_state":
        expected_imaginary_count = 1
    min_abs = ts_quality_settings.get("imaginary_frequency_min_abs")
    max_abs = ts_quality_settings.get("imaginary_frequency_max_abs")
    if optimizer_mode == "transition_state":
        if min_abs is None:
            min_abs = DEFAULT_TS_IMAG_FREQ_MIN_ABS
        if max_abs is None:
            max_abs = DEFAULT_TS_IMAG_FREQ_MAX_ABS
    projection_step = ts_quality_settings.get("projection_step") or DEFAULT_TS_MODE_PROJECTION_STEP
    projection_min_abs = ts_quality_settings.get("projection_min_abs") or 0.0
    internal_coordinates = ts_quality_settings.get("internal_coordinates") or []
    if not (
        expected_imaginary_count is not None
        or min_abs is not None
        or max_abs is not None
        or internal_coordinates
        or optimizer_mode == "transition_state"
    ):
        return None

    imaginary_abs = None
    imaginary_value = None
    if imaginary_frequencies:
        imaginary_value = min(imaginary_frequencies)
        imaginary_abs = abs(imaginary_value)
    count_ok = None
    if expected_imaginary_count is not None and imaginary_count is not None:
        count_ok = imaginary_count == expected_imaginary_count
    range_status = "unknown"
    range_message = None
    if imaginary_abs is not None and (min_abs is not None or max_abs is not None):
        if min_abs is not None and imaginary_abs < min_abs:
            range_status = "too_small"
            range_message = (
                f"Imaginary frequency magnitude {imaginary_abs:.2f} cm^-1 "
                f"is below the minimum {min_abs:.2f} cm^-1."
            )
        elif max_abs is not None and imaginary_abs > max_abs:
            range_status = "too_large"
            range_message = (
                f"Imaginary frequency magnitude {imaginary_abs:.2f} cm^-1 "
                f"exceeds the maximum {max_abs:.2f} cm^-1."
            )
        else:
            range_status = "ok"
    alignment_result = {
        "status": "not_configured",
        "projection_step": projection_step,
        "projection_min_abs": projection_min_abs,
        "coordinates": [],
    }
    if internal_coordinates:
        try:
            mode_result = _extract_imaginary_mode_from_hessian(
                hess, mol_freq, atomic_masses, atomic_numbers
            )
            alignment_result = _project_imaginary_mode_to_internal_coordinates(
                positions=mol_freq.atom_coords(unit="Angstrom"),
                mode=mode_result["mode"],
                internal_coordinates=internal_coordinates,
                projection_step=projection_step,
                projection_min_abs=projection_min_abs,
            )
        except Exception as exc:
            alignment_result = {
                "status": "unknown",
                "projection_step": projection_step,
                "projection_min_abs": projection_min_abs,
                "coordinates": [],
                "error": str(exc),
            }
    status = "pass"
    messages = []
    if count_ok is False:
        status = "fail"
        messages.append(
            "Imaginary frequency count does not match expected "
            f"{expected_imaginary_count}."
        )
    if alignment_result.get("status") == "misaligned":
        status = "fail"
        messages.append("Imaginary mode is misaligned with target coordinates.")
    if range_status in ("too_small", "too_large"):
        if status != "fail":
            status = "warn"
        if range_message:
            messages.append(range_message)
    if alignment_result.get("status") in ("weak", "unknown"):
        if status == "pass":
            status = "warn"
        if alignment_result.get("status") == "weak":
            messages.append("Imaginary mode projection is weak for target coordinates.")
        else:
            messages.append("Imaginary mode alignment could not be determined.")
    if status == "pass" and not messages:
        messages.append("Transition-state quality checks passed.")
    allow_irc = None
    allow_single_point = None
    if optimizer_mode == "transition_state":
        allow_irc = status in ("pass", "warn")
        allow_single_point = allow_irc
    return {
        "status": status,
        "message": " ".join(messages) if messages else None,
        "expected_imaginary_count": expected_imaginary_count,
        "imaginary_count": imaginary_count,
        "imaginary_count_ok": count_ok,
        "imaginary_frequency": imaginary_value,
        "imaginary_frequency_abs": imaginary_abs,
        "imaginary_frequency_range": {
            "min_abs": min_abs,
            "max_abs": max_abs,
            "status": range_status,
            "message": range_message,
        },
        "internal_coordinate_alignment": alignment_result,
        "allow_irc": allow_irc,
        "allow_single_point": allow_single_point,
    }


def _init_frequency_profiling(profiling_enabled):
    if not profiling_enabled:
        return None
    return {
        "scf_seconds": None,
        "scf_cycles": None,
        "hessian_seconds": None,
        "dispersion_hessian_seconds": None,
    }


def _prepare_frequency_molecule(mol, basis, memory_mb):
    mol_freq = mol.copy()
    if basis:
        mol_freq.basis = basis
        mol_freq.build()
    if memory_mb:
        mol_freq.max_memory = memory_mb
    return mol_freq


def _build_frequency_mf(
    *,
    mol_freq,
    dft_module,
    xc,
    scf_settings,
    solvent_model,
    solvent_name,
    solvent_eps,
    apply_density_fit,
    verbose,
    optimizer_mode,
    multiplicity,
    log_override,
):
    ks_type = select_ks_type(
        mol=mol_freq,
        scf_config=scf_settings,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=log_override,
    )
    if ks_type == "RKS":
        mf_freq = dft_module.RKS(mol_freq)
    else:
        mf_freq = dft_module.UKS(mol_freq)
    mf_freq.xc = xc
    density_fit_applied = False
    if apply_density_fit:
        mf_freq, df_config = apply_density_fit_setting(mf_freq, scf_settings)
        density_fit_applied = bool(df_config.get("density_fit"))
    if solvent_model is not None:
        mf_freq = apply_solvent_model(
            mf_freq,
            solvent_model,
            solvent_name,
            solvent_eps,
        )
    if verbose:
        mf_freq.verbose = 4
    mf_freq, _ = apply_scf_settings(mf_freq, scf_settings, apply_density_fit=False)
    return mf_freq, {"density_fit_applied": density_fit_applied}


def _run_frequency_scf(
    *,
    mol_freq,
    dft_module,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    apply_density_fit,
    verbose,
    optimizer_mode,
    multiplicity,
    log_override,
    run_dir,
    label,
    profiling,
):
    scf_start = time.perf_counter() if profiling else None
    energy, mf_freq, info = _run_scf_with_retries(
        lambda scf_settings: _build_frequency_mf(
            mol_freq=mol_freq,
            dft_module=dft_module,
            xc=xc,
            scf_settings=scf_settings,
            solvent_model=solvent_model,
            solvent_name=solvent_name,
            solvent_eps=solvent_eps,
            apply_density_fit=apply_density_fit,
            verbose=verbose,
            optimizer_mode=optimizer_mode,
            multiplicity=multiplicity,
            log_override=log_override,
        ),
        scf_config,
        run_dir,
        label,
    )
    if scf_start is not None:
        elapsed = time.perf_counter() - scf_start
        profiling["scf_seconds"] = (profiling["scf_seconds"] or 0.0) + elapsed
        cycles = extract_step_count(mf_freq)
        if cycles is not None:
            profiling["scf_cycles"] = (profiling["scf_cycles"] or 0) + cycles
    return energy, mf_freq, info


def _compute_hessian_with_timing(mf_handle, pyscf_hessian, profiling):
    hess_start = time.perf_counter() if profiling else None
    if hasattr(mf_handle, "Hessian"):
        hess_value = mf_handle.Hessian().kernel()
    else:
        hess_value = pyscf_hessian.Hessian(mf_handle).kernel()
    hess_seconds = None
    if hess_start is not None:
        hess_seconds = time.perf_counter() - hess_start
    return hess_value, hess_seconds


def _run_frequency_hessian_with_retry(
    *,
    energy,
    mf_freq,
    scf_info,
    dispersion_payload,
    mol_freq,
    dft_module,
    pyscf_hessian,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    verbose,
    optimizer_mode,
    multiplicity,
    log_override,
    run_dir,
    profiling,
):
    density_fit_applied = bool(scf_info.get("density_fit_applied"))
    energy, dispersion_info, dispersion_hessian = _apply_dispersion_payload(
        energy, dispersion_payload
    )
    try:
        hess, hess_seconds = _compute_hessian_with_timing(
            mf_freq, pyscf_hessian, profiling
        )
    except Exception as exc:
        if not (
            density_fit_applied and is_density_fit_gradient_einsum_error(exc)
        ):
            raise
        logging.warning(
            "Density-fitting Hessian failed; retrying without density fitting."
        )
        energy, mf_freq, retry_info = _run_frequency_scf(
            mol_freq=mol_freq,
            dft_module=dft_module,
            xc=xc,
            scf_config=scf_config,
            solvent_model=solvent_model,
            solvent_name=solvent_name,
            solvent_eps=solvent_eps,
            apply_density_fit=False,
            verbose=verbose,
            optimizer_mode=optimizer_mode,
            multiplicity=multiplicity,
            log_override=log_override,
            run_dir=run_dir,
            label="Frequency SCF (no density fit)",
            profiling=profiling,
        )
        density_fit_applied = bool(retry_info.get("density_fit_applied"))
        energy, dispersion_info, dispersion_hessian = _apply_dispersion_payload(
            energy, dispersion_payload
        )
        hess, hess_seconds = _compute_hessian_with_timing(
            mf_freq, pyscf_hessian, profiling
        )
    if hess_seconds is not None and profiling is not None:
        profiling["hessian_seconds"] = (
            (profiling["hessian_seconds"] or 0.0) + hess_seconds
        )
    return energy, mf_freq, dispersion_info, dispersion_hessian, hess


def _project_frequency_constraints(*, hess, mol_freq, atoms, atoms_cls, constraints):
    if not constraints:
        return hess, None, atoms
    constraint_projection = {
        "requested": True,
        "projected": False,
        "count": 0,
        "error": None,
    }
    try:
        if atoms is None:
            atoms = _atoms_from_molecule(mol_freq, atoms_cls)
        jacobians = _collect_constraint_jacobians(atoms, constraints)
        if jacobians is None:
            constraint_projection["error"] = "no_constraints"
        else:
            constraint_projection["count"] = jacobians.shape[0]
            hess = _project_hessian_constraints(hess, mol_freq, jacobians)
            constraint_projection["projected"] = True
            logging.info(
                "Applied constraint-projected Hessian (%s constraints).",
                constraint_projection["count"],
            )
    except Exception as exc:
        constraint_projection["error"] = str(exc)
        logging.warning("Constraint-projected Hessian skipped: %s", exc)
    return hess, constraint_projection, atoms


def _extract_harmonic_terms(harmonic):
    freq_wavenumber = None
    freq_au = None
    zpe = None
    if harmonic:
        freq_wavenumber = harmonic.get("freq_wavenumber")
        freq_au = harmonic.get("freq_au")
        for key in (
            "zpe",
            "ZPE",
            "zero_point_energy",
            "zero_point_energy_hartree",
            "zpve",
            "ZPVE",
        ):
            if key in harmonic:
                zpe = harmonic.get(key)
                break
    return freq_wavenumber, freq_au, zpe


def _build_frequency_result_payload(
    *,
    energy,
    mf_freq,
    freq_wavenumber_list,
    freq_au_list,
    zpe_value,
    imaginary_summary,
    ts_quality_payload,
    dispersion_info,
    thermochemistry,
    constraint_projection,
    profiling,
):
    result = {
        "energy": energy,
        "converged": getattr(mf_freq, "converged", None),
        "cycles": extract_step_count(mf_freq),
        "frequencies_wavenumber": freq_wavenumber_list,
        "frequencies_au": freq_au_list,
        "zpe": zpe_value,
        "imaginary_count": imaginary_summary["imaginary_count"],
        "imaginary_check": {
            "status": imaginary_summary["imaginary_status"],
            "message": imaginary_summary["imaginary_message"],
        },
        "min_frequency": imaginary_summary["min_frequency"],
        "max_frequency": imaginary_summary["max_frequency"],
        "ts_quality": ts_quality_payload,
        "dispersion": dispersion_info,
        "thermochemistry": thermochemistry,
        "constraint_projection": constraint_projection,
    }
    if profiling:
        result["profiling"] = profiling
    return result


def compute_frequencies(
    mol,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    dispersion,
    dispersion_hessian_mode,
    dispersion_hessian_step,
    dispersion_params,
    thermo,
    verbose,
    memory_mb,
    constraints,
    run_dir=None,
    optimizer_mode=None,
    multiplicity=None,
    ts_quality=None,
    profiling_enabled=False,
    log_override=True,
):
    from ase import units
    from ase import Atoms
    from ase.data import atomic_masses, atomic_numbers
    from pyscf import dft, hessian as pyscf_hessian
    from pyscf.data import nist
    from pyscf.hessian import thermo as pyscf_thermo

    xc = normalize_xc_functional(xc)
    profiling = _init_frequency_profiling(profiling_enabled)
    if dispersion_hessian_mode == "none":
        dispersion = None
    mol_freq = _prepare_frequency_molecule(mol, basis, memory_mb)

    atoms = None
    if dispersion is not None or constraints:
        atoms = _atoms_from_molecule(mol_freq, Atoms)
    dispersion_payload, atoms = _prepare_frequency_dispersion_payload(
        mol_freq=mol_freq,
        atoms=atoms,
        atoms_cls=Atoms,
        dispersion=dispersion,
        xc=xc,
        dispersion_hessian_mode=dispersion_hessian_mode,
        dispersion_hessian_step=dispersion_hessian_step,
        dispersion_params=dispersion_params,
        profiling=profiling,
        units=units,
    )

    energy, mf_freq, scf_info = _run_frequency_scf(
        mol_freq=mol_freq,
        dft_module=dft,
        xc=xc,
        scf_config=scf_config,
        solvent_model=solvent_model,
        solvent_name=solvent_name,
        solvent_eps=solvent_eps,
        apply_density_fit=True,
        verbose=verbose,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=log_override,
        run_dir=run_dir,
        label="Frequency SCF",
        profiling=profiling,
    )
    energy, mf_freq, dispersion_info, dispersion_hessian, hess = (
        _run_frequency_hessian_with_retry(
            energy=energy,
            mf_freq=mf_freq,
            scf_info=scf_info,
            dispersion_payload=dispersion_payload,
            mol_freq=mol_freq,
            dft_module=dft,
            pyscf_hessian=pyscf_hessian,
            xc=xc,
            scf_config=scf_config,
            solvent_model=solvent_model,
            solvent_name=solvent_name,
            solvent_eps=solvent_eps,
            verbose=verbose,
            optimizer_mode=optimizer_mode,
            multiplicity=multiplicity,
            log_override=log_override,
            run_dir=run_dir,
            profiling=profiling,
        )
    )
    if dispersion_hessian is not None:
        hess = hess + dispersion_hessian
    hess, constraint_projection, atoms = _project_frequency_constraints(
        hess=hess,
        mol_freq=mol_freq,
        atoms=atoms,
        atoms_cls=Atoms,
        constraints=constraints,
    )
    harmonic = pyscf_thermo.harmonic_analysis(mol_freq, hess, imaginary_freq=False)
    freq_wavenumber, freq_au, zpe = _extract_harmonic_terms(harmonic)
    freq_wavenumber_list = _to_list(freq_wavenumber)
    freq_au_list = _to_list(freq_au)
    zpe_value = _to_scalar(zpe)
    thermochemistry = _build_thermochemistry_payload(
        mf_freq=mf_freq,
        freq_au=freq_au,
        thermo=thermo,
        zpe_value=zpe_value,
        energy=energy,
        dispersion_info=dispersion_info,
        solvent_model=solvent_model,
        solvent_name=solvent_name,
        nist=nist,
        pyscf_thermo=pyscf_thermo,
    )
    imaginary_summary = _summarize_imaginary_frequencies(freq_wavenumber_list)
    ts_quality_payload = _build_ts_quality_payload(
        ts_quality=ts_quality,
        optimizer_mode=optimizer_mode,
        imaginary_frequencies=imaginary_summary["imaginary_frequencies"],
        imaginary_count=imaginary_summary["imaginary_count"],
        hess=hess,
        mol_freq=mol_freq,
        atomic_masses=atomic_masses,
        atomic_numbers=atomic_numbers,
    )
    return _build_frequency_result_payload(
        energy=energy,
        mf_freq=mf_freq,
        freq_wavenumber_list=freq_wavenumber_list,
        freq_au_list=freq_au_list,
        zpe_value=zpe_value,
        imaginary_summary=imaginary_summary,
        ts_quality_payload=ts_quality_payload,
        dispersion_info=dispersion_info,
        thermochemistry=thermochemistry,
        constraint_projection=constraint_projection,
        profiling=profiling,
    )


def _init_imaginary_mode_profiling(profiling_enabled):
    if not profiling_enabled:
        return None
    return {
        "scf_seconds": None,
        "scf_cycles": None,
        "hessian_seconds": None,
    }


def _prepare_imaginary_mode_molecule(mol, basis, memory_mb):
    mol_mode = mol.copy()
    if basis:
        mol_mode.basis = basis
        mol_mode.build()
    if memory_mb:
        mol_mode.max_memory = memory_mb
    return mol_mode


def _build_imaginary_mode_mf(
    *,
    mol_mode,
    dft_module,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    apply_density_fit,
    verbose,
    ks_type,
):
    if ks_type == "RKS":
        mf_mode = dft_module.RKS(mol_mode)
    else:
        mf_mode = dft_module.UKS(mol_mode)
    mf_mode.xc = xc
    density_fit_applied = False
    if apply_density_fit:
        mf_mode, df_config = apply_density_fit_setting(mf_mode, scf_config)
        density_fit_applied = bool(df_config.get("density_fit"))
    if solvent_model is not None:
        mf_mode = apply_solvent_model(
            mf_mode,
            solvent_model,
            solvent_name,
            solvent_eps,
        )
    if verbose:
        mf_mode.verbose = 4
    mf_mode, _ = apply_scf_settings(mf_mode, scf_config, apply_density_fit=False)
    return mf_mode, density_fit_applied


def _run_imaginary_mode_scf(mf_mode, scf_config, run_dir, profiling):
    scf_start = time.perf_counter() if profiling else None
    dm0, _ = apply_scf_checkpoint(mf_mode, scf_config, run_dir=run_dir)
    if dm0 is not None:
        mf_mode.kernel(dm0=dm0)
    else:
        mf_mode.kernel()
    if scf_start is not None:
        profiling["scf_seconds"] = (profiling["scf_seconds"] or 0.0) + (
            time.perf_counter() - scf_start
        )
        cycles = extract_step_count(mf_mode)
        if cycles is not None:
            profiling["scf_cycles"] = (profiling["scf_cycles"] or 0) + cycles


def _compute_imaginary_mode_hessian(mf_mode, pyscf_hessian, profiling):
    hess, hess_seconds = _compute_hessian_with_timing(mf_mode, pyscf_hessian, profiling)
    if hess_seconds is not None and profiling is not None:
        profiling["hessian_seconds"] = (profiling["hessian_seconds"] or 0.0) + hess_seconds
    return hess


def _run_imaginary_mode_hessian_with_retry(
    *,
    mol_mode,
    dft_module,
    pyscf_hessian,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    verbose,
    ks_type,
    run_dir,
    profiling,
):
    mf_mode, density_fit_applied = _build_imaginary_mode_mf(
        mol_mode=mol_mode,
        dft_module=dft_module,
        xc=xc,
        scf_config=scf_config,
        solvent_model=solvent_model,
        solvent_name=solvent_name,
        solvent_eps=solvent_eps,
        apply_density_fit=True,
        verbose=verbose,
        ks_type=ks_type,
    )
    _run_imaginary_mode_scf(mf_mode, scf_config, run_dir, profiling)
    try:
        return _compute_imaginary_mode_hessian(mf_mode, pyscf_hessian, profiling)
    except Exception as exc:
        if not (density_fit_applied and is_density_fit_gradient_einsum_error(exc)):
            raise
        logging.warning(
            "Density-fitting Hessian failed; retrying without density fitting."
        )
        mf_mode, _ = _build_imaginary_mode_mf(
            mol_mode=mol_mode,
            dft_module=dft_module,
            xc=xc,
            scf_config=scf_config,
            solvent_model=solvent_model,
            solvent_name=solvent_name,
            solvent_eps=solvent_eps,
            apply_density_fit=False,
            verbose=verbose,
            ks_type=ks_type,
        )
        _run_imaginary_mode_scf(mf_mode, scf_config, run_dir, profiling)
        return _compute_imaginary_mode_hessian(mf_mode, pyscf_hessian, profiling)


def _compute_imaginary_mode_dispersion_hessian(
    *,
    mol_mode,
    atoms_cls,
    units_module,
    dispersion,
    xc,
    dispersion_hessian_step,
    dispersion_params,
):
    atoms = _atoms_from_molecule(mol_mode, atoms_cls)
    dispersion_settings = parse_dispersion_settings(
        dispersion,
        xc,
        charge=mol_mode.charge,
        spin=mol_mode.spin,
        d3_params=dispersion_params,
    )
    dispersion_calc, _dispersion_backend = _build_dispersion_calculator(
        atoms, dispersion_settings
    )
    atoms.calc = dispersion_calc
    step = dispersion_hessian_step if dispersion_hessian_step is not None else 0.005
    dispersion_hessian = _compute_dispersion_hessian_numerical(atoms, step)
    return dispersion_hessian * ((1.0 / units_module.Hartree) / (units_module.Bohr ** 2))


def _apply_imaginary_mode_constraints(hess, mol_mode, atoms_cls, constraints):
    if not constraints:
        return hess
    atoms = _atoms_from_molecule(mol_mode, atoms_cls)
    jacobians = _collect_constraint_jacobians(atoms, constraints)
    if jacobians is not None:
        hess = _project_hessian_constraints(hess, mol_mode, jacobians)
    return hess


def _finalize_imaginary_mode_result(result, *, hess, profiling, return_hessian):
    if return_hessian:
        result["hessian"] = hess
    if profiling:
        result["profiling"] = profiling
    return result


_CAPABILITY_SCF_ERROR = (
    "Capability check failed during SCF. "
    "The current XC/solvent/spin combination may be unsupported. "
    "Review the SCF setup or choose a different solvent model."
)
_CAPABILITY_GRAD_ERROR = (
    "Capability check failed during nuclear gradient evaluation. "
    "The current XC/solvent/spin combination may not support gradients. "
    "Adjust the solvent model or use a PySCF build with gradient support."
)
_CAPABILITY_HESS_ERROR = (
    "Capability check failed during Hessian evaluation. "
    "The current XC/solvent/spin combination may not support Hessians. "
    "Disable frequency calculations or use a compatible PySCF build."
)


def _prepare_capability_check_molecule(mol, basis, memory_mb):
    mol_check = mol.copy()
    if basis:
        mol_check.basis = basis
        mol_check.build()
    if memory_mb:
        mol_check.max_memory = memory_mb
    return mol_check


def _validate_capability_dispersion(mol_check, dispersion, xc, dispersion_params):
    if dispersion is None:
        return
    parse_dispersion_settings(
        dispersion,
        normalize_xc_functional(xc),
        charge=mol_check.charge,
        spin=mol_check.spin,
        d3_params=dispersion_params,
    )


def _prepare_capability_scf_override(scf_config, max_scf_cycles):
    scf_override = dict(scf_config or {})
    current_max = scf_override.get("max_cycle", max_scf_cycles)
    scf_override["max_cycle"] = min(current_max, max_scf_cycles)
    return scf_override


def _build_capability_check_mf(
    *,
    mol_check,
    dft_module,
    ks_type,
    xc,
    scf_config,
    scf_override,
    solvent_model,
    solvent_name,
    solvent_eps,
    verbose,
    apply_density_fit,
):
    if ks_type == "RKS":
        mf_check = dft_module.RKS(mol_check)
    else:
        mf_check = dft_module.UKS(mol_check)
    mf_check.xc = normalize_xc_functional(xc)
    density_fit_applied = False
    if apply_density_fit:
        mf_check, df_config = apply_density_fit_setting(mf_check, scf_config)
        density_fit_applied = bool(df_config.get("density_fit"))
    if solvent_model is not None:
        mf_check = apply_solvent_model(
            mf_check,
            solvent_model,
            solvent_name,
            solvent_eps,
        )
    if verbose:
        mf_check.verbose = 4
    mf_check, _ = apply_scf_settings(mf_check, scf_override, apply_density_fit=False)
    return mf_check, density_fit_applied


def _run_capability_scf_or_raise(mf_check):
    try:
        mf_check.kernel()
    except Exception as exc:
        raise RuntimeError(_CAPABILITY_SCF_ERROR) from exc


def _run_capability_gradient(mf_check):
    mf_check.nuc_grad_method().kernel()


def _run_capability_hessian(mf_check, pyscf_hessian):
    if hasattr(mf_check, "Hessian"):
        mf_check.Hessian().kernel()
    else:
        pyscf_hessian.Hessian(mf_check).kernel()


def _run_capability_gradient_with_retry(
    *, mf_check, density_fit_applied, build_mf_check
):
    try:
        _run_capability_gradient(mf_check)
    except Exception as exc:
        if density_fit_applied and is_density_fit_gradient_einsum_error(exc):
            try:
                mf_check, _ = build_mf_check(apply_density_fit=False)
                _run_capability_scf_or_raise(mf_check)
                _run_capability_gradient(mf_check)
                return mf_check, False
            except Exception as retry_exc:
                raise RuntimeError(_CAPABILITY_GRAD_ERROR) from retry_exc
        raise RuntimeError(_CAPABILITY_GRAD_ERROR) from exc
    return mf_check, density_fit_applied


def _run_capability_hessian_with_retry(
    *, mf_check, density_fit_applied, build_mf_check, pyscf_hessian
):
    try:
        _run_capability_hessian(mf_check, pyscf_hessian)
    except Exception as exc:
        if density_fit_applied and is_density_fit_gradient_einsum_error(exc):
            try:
                mf_check, _ = build_mf_check(apply_density_fit=False)
                _run_capability_scf_or_raise(mf_check)
                _run_capability_hessian(mf_check, pyscf_hessian)
                return mf_check, False
            except Exception as retry_exc:
                raise RuntimeError(_CAPABILITY_HESS_ERROR) from retry_exc
        raise RuntimeError(_CAPABILITY_HESS_ERROR) from exc
    return mf_check, density_fit_applied


def compute_imaginary_mode(
    mol,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    verbose,
    memory_mb,
    dispersion=None,
    dispersion_hessian_step=None,
    constraints=None,
    dispersion_params=None,
    run_dir=None,
    optimizer_mode=None,
    multiplicity=None,
    profiling_enabled=False,
    log_override=True,
    return_hessian=False,
):
    try:
        from ase import Atoms, units
        from ase.data import atomic_masses, atomic_numbers
    except ImportError as exc:
        raise ImportError(
            "IRC mode extraction requires ASE (ase.data). Install ASE to proceed."
        ) from exc
    from pyscf import dft, hessian as pyscf_hessian

    xc = normalize_xc_functional(xc)
    profiling = _init_imaginary_mode_profiling(profiling_enabled)
    mol_mode = _prepare_imaginary_mode_molecule(mol, basis, memory_mb)
    ks_type = select_ks_type(
        mol=mol_mode,
        scf_config=scf_config,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=log_override,
    )
    hess = _run_imaginary_mode_hessian_with_retry(
        mol_mode=mol_mode,
        dft_module=dft,
        pyscf_hessian=pyscf_hessian,
        xc=xc,
        scf_config=scf_config,
        solvent_model=solvent_model,
        solvent_name=solvent_name,
        solvent_eps=solvent_eps,
        verbose=verbose,
        ks_type=ks_type,
        run_dir=run_dir,
        profiling=profiling,
    )
    if dispersion is not None:
        dispersion_hessian = _compute_imaginary_mode_dispersion_hessian(
            mol_mode=mol_mode,
            atoms_cls=Atoms,
            units_module=units,
            dispersion=dispersion,
            xc=xc,
            dispersion_hessian_step=dispersion_hessian_step,
            dispersion_params=dispersion_params,
        )
        hess = hess + dispersion_hessian

    hess = _apply_imaginary_mode_constraints(hess, mol_mode, Atoms, constraints)

    result = _extract_imaginary_mode_from_hessian(
        hess, mol_mode, atomic_masses, atomic_numbers
    )
    return _finalize_imaginary_mode_result(
        result,
        hess=hess,
        profiling=profiling,
        return_hessian=return_hessian,
    )


def run_capability_check(
    mol,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    dispersion,
    dispersion_hessian_mode,
    dispersion_params=None,
    require_hessian=False,
    verbose=False,
    memory_mb=None,
    max_scf_cycles=1,
    optimizer_mode=None,
    multiplicity=None,
):
    from pyscf import dft, hessian as pyscf_hessian

    mol_check = _prepare_capability_check_molecule(mol, basis, memory_mb)
    _validate_capability_dispersion(mol_check, dispersion, xc, dispersion_params)
    ks_type = select_ks_type(
        mol=mol_check,
        scf_config=scf_config,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=False,
    )
    scf_override = _prepare_capability_scf_override(scf_config, max_scf_cycles)

    def build_mf_check(*, apply_density_fit):
        return _build_capability_check_mf(
            mol_check=mol_check,
            dft_module=dft,
            ks_type=ks_type,
            xc=xc,
            scf_config=scf_config,
            scf_override=scf_override,
            solvent_model=solvent_model,
            solvent_name=solvent_name,
            solvent_eps=solvent_eps,
            verbose=verbose,
            apply_density_fit=apply_density_fit,
        )

    mf_check, density_fit_applied = build_mf_check(apply_density_fit=True)
    _run_capability_scf_or_raise(mf_check)
    mf_check, density_fit_applied = _run_capability_gradient_with_retry(
        mf_check=mf_check,
        density_fit_applied=density_fit_applied,
        build_mf_check=build_mf_check,
    )
    if require_hessian:
        _run_capability_hessian_with_retry(
            mf_check=mf_check,
            density_fit_applied=density_fit_applied,
            build_mf_check=build_mf_check,
            pyscf_hessian=pyscf_hessian,
        )
