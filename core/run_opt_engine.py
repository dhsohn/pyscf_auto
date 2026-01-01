import logging
import re

from .run_opt_config import DEFAULT_CHARGE, DEFAULT_MULTIPLICITY, DEFAULT_SPIN
from .run_opt_dispersion import load_d3_calculator, parse_dispersion_settings
from .run_opt_utils import extract_step_count


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
        d3_cls, d3_backend = load_d3_calculator()
        if d3_cls is None:
            raise ImportError(
                "DFTD3 dispersion requested but no DFTD3 calculator is available. "
                "Install `dftd3` (recommended) or `ase` with the DFTD3 binary available."
            )
        d3_calc = d3_cls(atoms=atoms, **settings)
        try:
            label = "dftd3" if d3_backend == "dftd3" else "ase-dftd3"
            return d3_calc.get_potential_energy(atoms=atoms), label
        except FileNotFoundError:
            raise RuntimeError(
                "DFTD3 executable not found. Install `dftd3` or set optimizer.ase.d3_command "
                "(or optimizer.ase.dftd3_command) to the full path of the DFTD3 binary."
            ) from None
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
        return {}
    applied = {}
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

    return applied


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
    apply_scf_settings(mf_sp, scf_config)
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
    optimizer_mode=None,
    multiplicity=None,
    log_override=True,
):
    from ase import units
    from ase import Atoms
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
    apply_scf_settings(mf_freq, scf_config)
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
    if freq_wavenumber_list:
        imaginary_count = sum(1 for value in freq_wavenumber_list if value < 0)
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
        "dispersion": dispersion_info,
        "thermochemistry": thermochemistry,
    }


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
    apply_scf_settings(mf_check, scf_override)
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
