import shutil

from .run_opt_engine import (
    apply_scf_settings,
    apply_solvent_model,
    normalize_xc_functional,
    select_ks_type,
)
from .run_opt_dispersion import load_d3_calculator, parse_dispersion_settings
from .run_opt_resources import ensure_parent_dir, resolve_run_path


def _build_atom_spec_from_ase(atoms):
    lines = []
    for symbol, (x, y, z) in zip(
        atoms.get_chemical_symbols(),
        atoms.get_positions(),
        strict=True,
    ):
        lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(lines)


def _build_pyscf_calculator(
    *,
    atoms,
    charge,
    spin,
    multiplicity,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    dispersion_model,
    verbose,
    memory_mb,
    optimizer_config,
    optimization_mode,
):
    import numpy as np

    from ase import units
    from ase.calculators.calculator import Calculator, all_changes
    from pyscf import dft, gto

    xc = normalize_xc_functional(xc)
    d3_params = optimizer_config.get("d3_params") or optimizer_config.get("dftd3_params")
    prefer_d3_backend = optimizer_config.get("d3_backend") or optimizer_config.get("dftd3_backend")
    d3_command = optimizer_config.get("d3_command") or optimizer_config.get("dftd3_command")
    d3_command_validate = optimizer_config.get("d3_command_validate", True)
    if not prefer_d3_backend and d3_command:
        prefer_d3_backend = "ase"
    ks_type = select_ks_type(
        spin=spin,
        scf_config=scf_config,
        optimizer_mode=optimization_mode,
        multiplicity=multiplicity,
    )
    dispersion_settings = (
        parse_dispersion_settings(
            dispersion_model,
            xc,
            charge=charge,
            spin=spin,
            d3_params=d3_params,
            prefer_d3_backend=prefer_d3_backend,
        )
        if dispersion_model
        else None
    )

    class PySCFCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            atom_spec = _build_atom_spec_from_ase(atoms)
            try:
                mol = gto.M(
                    atom=atom_spec,
                    basis=basis,
                    charge=charge,
                    spin=spin,
                    unit="Angstrom",
                )
            except FileNotFoundError as exc:
                if "pople-basis" in str(exc):
                    raise FileNotFoundError(
                        f"{exc}\nMissing PySCF basis data. Run: "
                        f"python scripts/restore_pyscf_basis.py --basis {basis}"
                    ) from exc
                raise
            if memory_mb:
                mol.max_memory = memory_mb
            if ks_type == "RKS":
                mf = dft.RKS(mol)
            else:
                mf = dft.UKS(mol)
            mf.xc = xc
            if solvent_model:
                mf = apply_solvent_model(mf, solvent_model, solvent_name, solvent_eps)
            apply_scf_settings(mf, scf_config)
            if verbose:
                mf.verbose = 4
            energy_hartree = mf.kernel()
            grad = mf.nuc_grad_method().kernel()
            forces = -grad * (units.Hartree / units.Bohr)
            self.results["energy"] = energy_hartree * units.Hartree
            self.results["forces"] = forces

    class _SumCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, calculators, **kwargs):
            super().__init__(**kwargs)
            self.calculators = calculators

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            energy_total = 0.0
            forces_total = None
            for calculator in self.calculators:
                energy = calculator.get_property("energy", atoms)
                forces = calculator.get_property("forces", atoms)
                energy_total += energy
                if forces_total is None:
                    forces_total = np.array(forces, copy=True)
                else:
                    forces_total += forces
            self.results["energy"] = energy_total
            self.results["forces"] = forces_total

    base_calc = PySCFCalculator()
    if dispersion_settings:
        backend = dispersion_settings["backend"]
        settings = dict(dispersion_settings["settings"])
        if backend == "d3":
            d3_cls, d3_backend = load_d3_calculator(prefer_d3_backend)
            if d3_cls is None:
                raise ImportError(
                    "DFTD3 dispersion requested but no DFTD3 calculator is available. "
                    "Install `dftd3` (recommended) or `ase` with the DFTD3 binary available."
                )
            if d3_backend == "ase" and d3_command:
                d3_command_path = shutil.which(d3_command)
                if d3_command_validate and d3_command_path is None:
                    raise ValueError(
                        "DFTD3 command '{command}' was not found on PATH. "
                        "Install the DFTD3 binary and set optimizer.ase.d3_command "
                        "to the full path.".format(command=d3_command)
                    )
                settings["command"] = d3_command
            dispersion_calc = d3_cls(atoms=atoms, **settings)
        else:
            from dftd4.ase import DFTD4

            dispersion_calc = DFTD4(atoms=atoms, **settings)
        return _SumCalculator([base_calc, dispersion_calc])
    return base_calc


def _run_ase_optimizer(
    input_xyz,
    output_xyz,
    run_dir,
    charge,
    spin,
    multiplicity,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    dispersion_model,
    verbose,
    memory_mb,
    optimizer_config,
    optimization_mode,
    step_callback=None,
):
    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write
        from ase.optimize import BFGS, FIRE, GPMin, LBFGS, MDMin
    except ImportError as exc:
        raise ImportError(
            "ASE optimizer requested but ASE or required calculators are not installed. "
            "Install ASE with DFTD3 support (e.g., `conda install -c conda-forge ase`)."
        ) from exc

    atoms = ase_read(input_xyz)
    atoms.calc = _build_pyscf_calculator(
        atoms=atoms,
        charge=charge,
        spin=spin,
        multiplicity=multiplicity,
        basis=basis,
        xc=xc,
        scf_config=scf_config,
        solvent_model=solvent_model,
        solvent_name=solvent_name,
        solvent_eps=solvent_eps,
        dispersion_model=dispersion_model,
        verbose=verbose,
        memory_mb=memory_mb,
        optimizer_config=optimizer_config,
        optimization_mode=optimization_mode,
    )

    optimizer_name = (optimizer_config.get("optimizer") or "").lower()
    if not optimizer_name:
        optimizer_name = "sella" if optimization_mode == "transition_state" else "bfgs"
    fmax = optimizer_config.get("fmax", 0.05)
    steps = optimizer_config.get("steps", 200)
    trajectory = optimizer_config.get("trajectory")
    logfile = optimizer_config.get("logfile")
    if trajectory:
        trajectory = resolve_run_path(run_dir, trajectory)
        ensure_parent_dir(trajectory)
    if logfile:
        logfile = resolve_run_path(run_dir, logfile)
        ensure_parent_dir(logfile)

    if optimizer_name == "sella":
        import importlib.util

        if importlib.util.find_spec("sella") is None:
            raise ImportError(
                "Transition-state optimization requires the Sella optimizer. "
                "Install it with `pip install sella`."
            )
        from sella import Sella

        sella_config = optimizer_config.get("sella") or {}
        if not isinstance(sella_config, dict):
            raise ValueError("ASE optimizer config 'sella' must be an object.")
        sella_kwargs = dict(sella_config)
        order = sella_kwargs.pop("order", None)
        if order is None:
            order = 1 if optimization_mode == "transition_state" else 0
        if optimization_mode == "transition_state" and order < 1:
            raise ValueError(
                "Transition-state optimization requires Sella 'order' >= 1."
            )
        optimizer = Sella(
            atoms,
            order=order,
            trajectory=trajectory,
            logfile=logfile,
            **sella_kwargs,
        )
    else:
        if optimization_mode == "transition_state":
            raise ValueError(
                "Transition-state optimization currently supports only the Sella optimizer. "
                "Set optimizer.ase.optimizer='sella'."
            )
        optimizer_map = {
            "bfgs": BFGS,
            "lbfgs": LBFGS,
            "fire": FIRE,
            "gpmin": GPMin,
            "mdmin": MDMin,
        }
        optimizer_cls = optimizer_map.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(
                "Unsupported ASE optimizer '{name}'. Supported: {supported}.".format(
                    name=optimizer_name,
                    supported=", ".join(sorted(optimizer_map.keys())),
                )
            )
        optimizer = optimizer_cls(atoms, trajectory=trajectory, logfile=logfile)
    if step_callback is not None:
        optimizer.attach(step_callback, interval=1)
    optimizer.run(fmax=fmax, steps=steps)

    ase_write(output_xyz, atoms, format="xyz")
    return getattr(optimizer, "nsteps", None)


def _run_ase_irc(
    input_xyz,
    run_dir,
    charge,
    spin,
    multiplicity,
    basis,
    xc,
    scf_config,
    solvent_model,
    solvent_name,
    solvent_eps,
    dispersion_model,
    verbose,
    memory_mb,
    optimizer_config,
    optimization_mode,
    mode_vector,
    steps,
    step_size,
    force_threshold,
    output_prefix="irc",
):
    import numpy as np

    try:
        from ase import units
        from ase.io import read as ase_read
        from ase.io import write as ase_write
    except ImportError as exc:
        raise ImportError(
            "ASE IRC requested but ASE or required calculators are not installed. "
            "Install ASE with DFTD3 support (e.g., `conda install -c conda-forge ase`)."
        ) from exc

    atoms = ase_read(input_xyz)
    atoms.calc = _build_pyscf_calculator(
        atoms=atoms,
        charge=charge,
        spin=spin,
        multiplicity=multiplicity,
        basis=basis,
        xc=xc,
        scf_config=scf_config,
        solvent_model=solvent_model,
        solvent_name=solvent_name,
        solvent_eps=solvent_eps,
        dispersion_model=dispersion_model,
        verbose=verbose,
        memory_mb=memory_mb,
        optimizer_config=optimizer_config,
        optimization_mode=optimization_mode,
    )
    positions = atoms.get_positions()
    mode = np.asarray(mode_vector, dtype=float)
    if mode.shape != positions.shape:
        raise ValueError("IRC mode vector shape does not match atom coordinates.")
    mode_norm = np.linalg.norm(mode)
    if mode_norm == 0:
        raise ValueError("IRC mode vector is zero; cannot start IRC.")
    initial_direction = mode / mode_norm
    masses = atoms.get_masses()
    if np.any(masses <= 0):
        raise ValueError("Invalid atomic masses encountered for IRC.")
    profile = []
    outputs = {}

    for label, sign in (("forward", 1.0), ("reverse", -1.0)):
        step_atoms = atoms.copy()
        step_atoms.calc = atoms.calc
        direction_unit = initial_direction
        step_atoms.set_positions(positions + sign * step_size * direction_unit)
        output_xyz = resolve_run_path(run_dir, f"{output_prefix}_{label}.xyz")
        outputs[label] = output_xyz
        ensure_parent_dir(output_xyz)
        for step_index in range(steps):
            energy_ev = step_atoms.get_potential_energy()
            forces = step_atoms.get_forces()
            profile.append(
                {
                    "direction": label,
                    "step": step_index,
                    "energy_ev": float(energy_ev),
                    "energy_hartree": float(energy_ev / units.Hartree),
                }
            )
            ase_write(output_xyz, step_atoms, format="xyz", append=True)
            force_norm = np.linalg.norm(forces)
            if force_norm < force_threshold:
                break
            mass_weighted = forces / masses[:, None]
            mass_norm = np.linalg.norm(mass_weighted)
            if mass_norm == 0:
                break
            direction_unit = mass_weighted / mass_norm
            step_atoms.set_positions(step_atoms.get_positions() + step_size * direction_unit)

    return {
        "profile": profile,
        "forward_xyz": outputs.get("forward"),
        "reverse_xyz": outputs.get("reverse"),
    }


__all__ = ["_run_ase_optimizer", "_run_ase_irc"]
