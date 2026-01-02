import logging
import os
import re

from run_opt_config import DEFAULT_CHARGE, DEFAULT_MULTIPLICITY, DEFAULT_SPIN
from run_opt_dispersion import load_d3_calculator, parse_dispersion_settings
from run_opt_resources import ensure_parent_dir, resolve_run_path
from run_opt_utils import extract_step_count


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


def parse_xyz_metadata(xyz_lines):
    """
    Parse charge/spin metadata from the comment line of an XYZ file.

    Expected format in the second line (comment), for example:
      charge=0 spin=1 multiplicity=2

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


def apply_scf_settings(mf, scf_config):
    if not scf_config:
        return mf, {}
    applied = {}
    extra = scf_config.get("extra") or {}
    applied_extra = {}
    if "density_fit" in extra:
        density_fit = extra.get("density_fit")
        if not isinstance(density_fit, bool):
            raise ValueError("Config 'scf.extra.density_fit' must be a boolean.")
        if density_fit:
            if not hasattr(mf, "density_fit"):
                raise ValueError("Density fitting is not supported for this SCF object.")
            mf = mf.density_fit()
        applied_extra["density_fit"] = density_fit
    max_cycle = scf_config.get("max_cycle")
    conv_tol = scf_config.get("conv_tol")
    diis = scf_config.get("diis")
    level_shift = scf_config.get("level_shift")
    damping = scf_config.get("damping")

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
    force_restricted = bool(scf_config.get("force_restricted")) if scf_config else False
    force_unrestricted = bool(scf_config.get("force_unrestricted")) if scf_config else False
    if force_restricted and force_unrestricted:
        raise ValueError(
            "Config 'scf' must not set both 'force_restricted' and 'force_unrestricted'."
        )
    if force_restricted or force_unrestricted:
        if optimizer_mode == "transition_state" and multiplicity and multiplicity > 1:
            if log_override:
                logging.warning(
                    "SCF override requested (%s) ignored for transition-state multiplicity %s; "
                    "using default %s.",
                    "force_restricted" if force_restricted else "force_unrestricted",
                    multiplicity,
                    default_type,
                )
            return default_type
        requested_type = "RKS" if force_restricted else "UKS"
        if log_override:
            logging.warning(
                "SCF override requested (%s): using %s (default %s).",
                "force_restricted" if force_restricted else "force_unrestricted",
                requested_type,
                default_type,
            )
        return requested_type
    return default_type


def _smd_available(mf):
    if not hasattr(mf, "SMD"):
        return False
    try:
        from pyscf.solvent import smd  # noqa: F401
    except Exception:
        return False
    return True


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
                "Install PySCF with solvent support or choose --solvent-model pcm."
            )
        supported = _supported_smd_solvents()
        normalized = solvent_name.strip().lower()
        supported_map = {name.strip().lower(): name for name in supported}
        if normalized not in supported_map:
            preview = ", ".join(sorted(supported)[:10])
            raise ValueError(
                "SMD solvent '{name}' not found in supported list (showing first 10 "
                "of {count}: {preview}).".format(
                    name=solvent_name, count=len(supported), preview=preview
                )
            )
        mf = mf.SMD()
        mf.with_solvent.solvent = supported_map[normalized]
    else:
        raise ValueError(f"Unsupported solvent model '{solvent_model}'.")
    return mf


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
    return supported


def compute_single_point_energy(
    mol,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    dispersion,
    verbose,
    memory_mb,
    run_dir=None,
    optimizer_mode=None,
    multiplicity=None,
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
    ks_type = select_ks_type(
        mol=mol_sp,
        scf_config=scf_config,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=log_override,
    )
    if ks_type == "RKS":
        mf_sp = dft.RKS(mol_sp)
    else:
        mf_sp = dft.UKS(mol_sp)
    mf_sp.xc = xc
    if solvent_model is not None:
        mf_sp = apply_solvent_model(
            mf_sp,
            solvent_model,
            solvent_name,
            solvent_eps,
        )
    if verbose:
        mf_sp.verbose = 4
    mf_sp, _ = apply_scf_settings(mf_sp, scf_config)
    dm0, _ = apply_scf_checkpoint(mf_sp, scf_config, run_dir=run_dir)
    if dm0 is not None:
        energy = mf_sp.kernel(dm0=dm0)
    else:
        energy = mf_sp.kernel()
    dispersion_info = None
    if dispersion is not None:
        dispersion_settings = parse_dispersion_settings(
            dispersion, xc, charge=mol_sp.charge, spin=mol_sp.spin
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
    return {
        "energy": energy,
        "converged": getattr(mf_sp, "converged", None),
        "cycles": extract_step_count(mf_sp),
        "dispersion": dispersion_info,
    }


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
    thermo,
    verbose,
    memory_mb,
    run_dir=None,
    optimizer_mode=None,
    multiplicity=None,
    ts_quality=None,
    log_override=True,
):
    from ase import units
    from ase import Atoms
    from ase.data import atomic_masses, atomic_numbers
    from pyscf import dft, hessian as pyscf_hessian
    from pyscf.hessian import thermo as pyscf_thermo

    xc = normalize_xc_functional(xc)
    mol_freq = mol.copy()
    if basis:
        mol_freq.basis = basis
        mol_freq.build()
    if memory_mb:
        mol_freq.max_memory = memory_mb
    ks_type = select_ks_type(
        mol=mol_freq,
        scf_config=scf_config,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=log_override,
    )
    if ks_type == "RKS":
        mf_freq = dft.RKS(mol_freq)
    else:
        mf_freq = dft.UKS(mol_freq)
    mf_freq.xc = xc
    dispersion_info = None
    if solvent_model is not None:
        mf_freq = apply_solvent_model(
            mf_freq,
            solvent_model,
            solvent_name,
            solvent_eps,
        )
    if verbose:
        mf_freq.verbose = 4
    mf_freq, _ = apply_scf_settings(mf_freq, scf_config)
    dm0, _ = apply_scf_checkpoint(mf_freq, scf_config, run_dir=run_dir)
    if dm0 is not None:
        energy = mf_freq.kernel(dm0=dm0)
    else:
        energy = mf_freq.kernel()
    if dispersion is not None and dispersion_hessian_mode == "none":
        dispersion_settings = parse_dispersion_settings(
            dispersion, xc, charge=mol_freq.charge, spin=mol_freq.spin
        )
        positions = mol_freq.atom_coords(unit="Angstrom")
        if hasattr(mol_freq, "atom_symbols"):
            symbols = mol_freq.atom_symbols()
        else:
            symbols = [mol_freq.atom_symbol(i) for i in range(mol_freq.natm)]
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
            "hessian_mode": dispersion_hessian_mode,
        }
    if hasattr(mf_freq, "Hessian"):
        hess = mf_freq.Hessian().kernel()
    else:
        hess = pyscf_hessian.Hessian(mf_freq).kernel()
    harmonic = pyscf_thermo.harmonic_analysis(mol_freq, hess, imaginary_freq=False)
    freq_wavenumber = None
    freq_au = None
    zpe = None
    if harmonic:
        freq_wavenumber = harmonic.get("freq_wavenumber")
        freq_au = harmonic.get("freq_au")
        for key in ("zpe", "ZPE", "zero_point_energy", "zero_point_energy_hartree", "zpve", "ZPVE"):
            if key in harmonic:
                zpe = harmonic.get(key)
                break

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

    freq_wavenumber_list = _to_list(freq_wavenumber)
    freq_au_list = _to_list(freq_au)
    zpe_value = _to_scalar(zpe)
    thermochemistry = None
    thermo_settings = None
    if thermo:
        if hasattr(thermo, "to_dict"):
            thermo_settings = thermo.to_dict()
        elif isinstance(thermo, dict):
            thermo_settings = thermo
    if thermo_settings:
        temperature = thermo_settings.get("T")
        pressure = thermo_settings.get("P")
        pressure_unit = thermo_settings.get("unit")
        thermo_result = None
        if freq_au is not None:
            thermo_result = pyscf_thermo.thermo(
                mol_freq,
                freq_au,
                temperature=temperature,
                pressure=pressure,
                unit=pressure_unit,
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
        thermal_correction_enthalpy = None
        gibbs_correction = None
        gibbs_free_energy = None
        if enthalpy_total_value is not None:
            thermal_correction_enthalpy = enthalpy_total_value - energy
        if gibbs_total_value is not None:
            gibbs_correction = gibbs_total_value - energy
            gibbs_free_energy = energy + gibbs_correction
        thermochemistry = {
            "temperature": _to_scalar(temperature),
            "pressure": _to_scalar(pressure),
            "pressure_unit": pressure_unit,
            "zpe": zpe_for_thermo,
            "thermal_correction_enthalpy": _to_scalar(thermal_correction_enthalpy),
            "entropy": entropy_value,
            "gibbs_correction": _to_scalar(gibbs_correction),
            "gibbs_free_energy": _to_scalar(gibbs_free_energy),
        }
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

    ts_quality_payload = None
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
    if (
        expected_imaginary_count is not None
        or min_abs is not None
        or max_abs is not None
        or internal_coordinates
        or optimizer_mode == "transition_state"
    ):
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
        mode_result = None
        alignment_error = None
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
                alignment_error = str(exc)
                alignment_result = {
                    "status": "unknown",
                    "projection_step": projection_step,
                    "projection_min_abs": projection_min_abs,
                    "coordinates": [],
                    "error": alignment_error,
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
        ts_quality_payload = {
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

    return {
        "energy": energy,
        "converged": getattr(mf_freq, "converged", None),
        "cycles": extract_step_count(mf_freq),
        "frequencies_wavenumber": freq_wavenumber_list,
        "frequencies_au": freq_au_list,
        "zpe": zpe_value,
        "imaginary_count": imaginary_count,
        "imaginary_check": {
            "status": imaginary_status,
            "message": imaginary_message,
        },
        "min_frequency": min_frequency,
        "max_frequency": max_frequency,
        "ts_quality": ts_quality_payload,
        "dispersion": dispersion_info,
        "thermochemistry": thermochemistry,
    }


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
    run_dir=None,
    optimizer_mode=None,
    multiplicity=None,
    log_override=True,
):
    try:
        from ase.data import atomic_masses, atomic_numbers
    except ImportError as exc:
        raise ImportError(
            "IRC mode extraction requires ASE (ase.data). Install ASE to proceed."
        ) from exc
    from pyscf import dft, hessian as pyscf_hessian

    xc = normalize_xc_functional(xc)
    mol_mode = mol.copy()
    if basis:
        mol_mode.basis = basis
        mol_mode.build()
    if memory_mb:
        mol_mode.max_memory = memory_mb
    ks_type = select_ks_type(
        mol=mol_mode,
        scf_config=scf_config,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=log_override,
    )
    if ks_type == "RKS":
        mf_mode = dft.RKS(mol_mode)
    else:
        mf_mode = dft.UKS(mol_mode)
    mf_mode.xc = xc
    if solvent_model is not None:
        mf_mode = apply_solvent_model(
            mf_mode,
            solvent_model,
            solvent_name,
            solvent_eps,
        )
    if verbose:
        mf_mode.verbose = 4
    mf_mode, _ = apply_scf_settings(mf_mode, scf_config)
    dm0, _ = apply_scf_checkpoint(mf_mode, scf_config, run_dir=run_dir)
    if dm0 is not None:
        mf_mode.kernel(dm0=dm0)
    else:
        mf_mode.kernel()
    if hasattr(mf_mode, "Hessian"):
        hess = mf_mode.Hessian().kernel()
    else:
        hess = pyscf_hessian.Hessian(mf_mode).kernel()
    return _extract_imaginary_mode_from_hessian(
        hess, mol_mode, atomic_masses, atomic_numbers
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
    require_hessian=False,
    verbose=False,
    memory_mb=None,
    max_scf_cycles=1,
    optimizer_mode=None,
    multiplicity=None,
):
    from pyscf import dft, hessian as pyscf_hessian

    mol_check = mol.copy()
    if basis:
        mol_check.basis = basis
        mol_check.build()
    if memory_mb:
        mol_check.max_memory = memory_mb
    ks_type = select_ks_type(
        mol=mol_check,
        scf_config=scf_config,
        optimizer_mode=optimizer_mode,
        multiplicity=multiplicity,
        log_override=False,
    )
    if ks_type == "RKS":
        mf_check = dft.RKS(mol_check)
    else:
        mf_check = dft.UKS(mol_check)
    mf_check.xc = normalize_xc_functional(xc)
    if solvent_model is not None:
        mf_check = apply_solvent_model(
            mf_check,
            solvent_model,
            solvent_name,
            solvent_eps,
        )
    if verbose:
        mf_check.verbose = 4
    scf_override = dict(scf_config or {})
    current_max = scf_override.get("max_cycle", max_scf_cycles)
    scf_override["max_cycle"] = min(current_max, max_scf_cycles)
    mf_check, _ = apply_scf_settings(mf_check, scf_override)
    try:
        mf_check.kernel()
    except Exception as exc:
        raise RuntimeError(
            "Capability check failed during SCF. "
            "The current XC/solvent/spin combination may be unsupported. "
            "Review the SCF setup or choose a different solvent model."
        ) from exc
    try:
        mf_check.nuc_grad_method().kernel()
    except Exception as exc:
        raise RuntimeError(
            "Capability check failed during nuclear gradient evaluation. "
            "The current XC/solvent/spin combination may not support gradients. "
            "Adjust the solvent model or use a PySCF build with gradient support."
        ) from exc
    if require_hessian:
        try:
            if hasattr(mf_check, "Hessian"):
                mf_check.Hessian().kernel()
            else:
                pyscf_hessian.Hessian(mf_check).kernel()
        except Exception as exc:
            raise RuntimeError(
                "Capability check failed during Hessian evaluation. "
                "The current XC/solvent/spin combination may not support Hessians. "
                "Disable frequency calculations or use a compatible PySCF build."
            ) from exc
