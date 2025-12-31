import argparse
import importlib.util
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

from run_opt_chemistry import (
    apply_scf_settings,
    apply_solvent_model,
    compute_frequencies,
    compute_single_point_energy,
    load_xyz,
    normalize_xc_functional,
    run_capability_check,
    select_ks_type,
    total_electron_count,
)
from run_opt_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENT_LOG_PATH,
    DEFAULT_FREQUENCY_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_OPTIMIZED_XYZ_PATH,
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_QUEUE_RUNNER_LOCK_PATH,
    DEFAULT_QUEUE_RUNNER_LOG_PATH,
    DEFAULT_RUN_METADATA_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
    DEFAULT_THREAD_COUNT,
    load_run_config,
    load_solvent_map,
    validate_run_config,
)
from run_opt_logging import ensure_stream_newlines, setup_logging
from run_opt_metadata import (
    build_run_summary,
    collect_git_metadata,
    compute_file_hash,
    compute_text_hash,
    get_package_version,
    parse_single_point_cycle_count,
    write_optimized_xyz,
    write_run_metadata,
)
from run_opt_resources import (
    apply_memory_limit,
    apply_thread_settings,
    collect_environment_snapshot,
    create_run_directory,
    ensure_parent_dir,
    format_log_path,
    resolve_run_path,
)

from run_opt_dispersion import load_d3_calculator, parse_dispersion_settings


_SCRIPT_DIR = Path(__file__).resolve().parent
INTERACTIVE_CONFIG_MINIMUM = _SCRIPT_DIR / "run_config_ase.json"
INTERACTIVE_CONFIG_TS = _SCRIPT_DIR / "run_config_ts.json"
BASIS_SET_OPTIONS = [
    "6-31g",
    "6-31g*",
    "6-31g**",
    "def2-svp",
    "def2-tzvp",
    "def2-tzvpp",
    "cc-pvdz",
    "cc-pvtz",
]
XC_FUNCTIONAL_OPTIONS = [
    "b3lyp",
    "pbe0",
    "WB97X_D",
    "m06-2x",
    "pbe",
    "b97-d",
]
SOLVENT_MODEL_OPTIONS = ["pcm", "smd", "none (vacuum)"]
CALCULATION_MODE_OPTIONS = [
    "구조 최적화",
    "단일점 에너지 계산",
    "프리퀀시 계산",
]


def _build_atom_spec_from_ase(atoms):
    lines = []
    for symbol, (x, y, z) in zip(
        atoms.get_chemical_symbols(),
        atoms.get_positions(),
        strict=True,
    ):
        lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(lines)




def _xc_includes_dispersion(xc):
    if not xc:
        return False
    normalized = re.sub(r"[\s_\-]+", "", str(xc)).lower()
    return normalized.endswith(("d", "d2", "d3", "d4"))


def _is_vacuum_solvent(name):
    return name is not None and name.strip().lower() == "vacuum"


def _normalize_dispersion_settings(stage_label, xc, dispersion_model, allow_dispersion=True):
    if dispersion_model is None:
        return None
    normalized = str(dispersion_model).lower()
    if not allow_dispersion:
        logging.warning(
            "%s 단계에서는 dispersion 입력을 지원하지 않습니다. '%s' 설정을 무시합니다.",
            stage_label,
            dispersion_model,
        )
        return None
    if _xc_includes_dispersion(xc):
        logging.warning(
            "%s XC '%s'에는 dispersion이 포함되어 있어 요청된 '%s'를 무시합니다.",
            stage_label,
            xc,
            normalized,
        )
        return None
    return normalized


def _frequency_units():
    return {
        "frequencies_wavenumber": "cm^-1",
        "frequencies_au": "a.u.",
        "energy": "Hartree",
        "zpe": "Hartree",
        "min_frequency": "cm^-1",
        "max_frequency": "cm^-1",
        "dispersion_energy_hartree": "Hartree",
        "dispersion_energy_ev": "eV",
    }


def _frequency_versions():
    return {
        "ase": get_package_version("ase"),
        "pyscf": get_package_version("pyscf"),
        "dftd3": get_package_version("dftd3"),
        "dftd4": get_package_version("dftd4"),
    }


def _normalize_frequency_dispersion_mode(mode_value):
    if mode_value is None:
        return "none"
    normalized = re.sub(r"[\s_\-]+", "", str(mode_value)).lower()
    if normalized in ("none", "no", "off", "false"):
        return "none"
    raise ValueError(
        "Unsupported frequency dispersion mode '{value}'. Use 'none'.".format(
            value=mode_value
        )
    )


def _normalize_solvent_settings(stage_label, solvent_name, solvent_model):
    if not solvent_name:
        return None, None
    if _is_vacuum_solvent(solvent_name):
        if solvent_model:
            logging.warning(
                "%s 단계에서 solvent '%s'는 vacuum으로 처리됩니다. solvent_model '%s'를 무시합니다.",
                stage_label,
                solvent_name,
                solvent_model,
            )
        return solvent_name, None
    return solvent_name, solvent_model


def _normalize_optimizer_mode(mode_value):
    if not mode_value:
        return "minimum"
    normalized = re.sub(r"[\s_\-]+", "", str(mode_value)).lower()
    if normalized in ("minimum", "min", "geometry", "geom", "opt", "optimization"):
        return "minimum"
    if normalized in (
        "transitionstate",
        "transition",
        "ts",
        "saddle",
        "saddlepoint",
        "tsopt",
    ):
        return "transition_state"
    raise ValueError(
        "Unsupported optimizer mode '{value}'. Use 'minimum' or 'transition_state'.".format(
            value=mode_value
        )
    )


def _normalize_calculation_mode(mode_value):
    if not mode_value:
        return "optimization"
    normalized = re.sub(r"[\s_\-]+", "", str(mode_value)).lower()
    if normalized in (
        "optimization",
        "opt",
        "geometry",
        "geom",
        "structure",
        "structureoptimization",
        "구조최적화",
        "구조최적",
    ):
        return "optimization"
    if normalized in (
        "singlepoint",
        "singlepointenergy",
        "singlepointenergycalculation",
        "singlepointenergycalc",
        "singlepointcalc",
        "single_point",
        "single",
        "sp",
        "단일점",
        "단일점에너지",
        "단일점에너지계산",
    ):
        return "single_point"
    if normalized in (
        "frequency",
        "frequencies",
        "freq",
        "vibration",
        "vibrational",
        "프리퀀시",
        "진동",
    ):
        return "frequency"
    raise ValueError(
        "Unsupported calculation mode '{value}'. Use 'optimization', "
        "'single_point', or 'frequency'.".format(value=mode_value)
    )


def _prompt_choice(prompt, options, allow_custom=False, default_value=None):
    normalized_options = list(dict.fromkeys(options))
    if default_value and default_value not in normalized_options:
        normalized_options.insert(0, default_value)
    while True:
        print(prompt)
        for index, option in enumerate(normalized_options, start=1):
            print(f"{index}. {option}")
        if allow_custom:
            print("0. 직접 입력")
        choice = input("> ").strip()
        if not choice and default_value:
            return default_value
        if allow_custom and choice == "0":
            custom = input("값을 입력하세요: ").strip()
            if custom:
                return custom
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(normalized_options):
                return normalized_options[index - 1]
        print("유효한 번호를 입력하세요.")


def _prompt_yes_no(prompt, default=True):
    suffix = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} ({suffix}): ").strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("y 또는 n으로 입력하세요.")


def _prompt_interactive_config(args):
    calculation_choice = _prompt_choice(
        "어떤 계산을 진행할까요?",
        CALCULATION_MODE_OPTIONS,
    )
    calculation_mode = {
        "구조 최적화": "optimization",
        "단일점 에너지 계산": "single_point",
        "프리퀀시 계산": "frequency",
    }[calculation_choice]
    if calculation_mode == "optimization":
        optimization_choice = _prompt_choice(
            "최적화 유형을 선택하세요:",
            ["중간체 최적화", "전이상태 최적화"],
        )
        if optimization_choice == "전이상태 최적화":
            base_config_path = INTERACTIVE_CONFIG_TS
        else:
            base_config_path = INTERACTIVE_CONFIG_MINIMUM
    else:
        base_config_path = INTERACTIVE_CONFIG_MINIMUM

    config, _ = load_run_config(base_config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Failed to load base config from {base_config_path}.")

    if not args.xyz_file:
        input_dir = os.path.join(os.path.dirname(__file__), "input")
        input_xyz_files = sorted(
            filename
            for filename in os.listdir(input_dir)
            if filename.lower().endswith(".xyz")
            and os.path.isfile(os.path.join(input_dir, filename))
        )
        if not input_xyz_files:
            raise ValueError("input 디렉토리에 .xyz 파일이 없습니다.")
        selected_xyz = _prompt_choice(
            "인풋 파일을 선택하세요 (.xyz):",
            input_xyz_files,
        )
        args.xyz_file = os.path.join(input_dir, selected_xyz)

    basis = _prompt_choice(
        "basis set을 선택하세요:",
        BASIS_SET_OPTIONS,
        allow_custom=True,
        default_value=config.get("basis"),
    )
    xc = _prompt_choice(
        "함수를 선택하세요:",
        XC_FUNCTIONAL_OPTIONS,
        allow_custom=True,
        default_value=config.get("xc"),
    )
    xc = normalize_xc_functional(xc)
    solvent_model = _prompt_choice(
        "용매 모델을 선택하세요:",
        SOLVENT_MODEL_OPTIONS,
        allow_custom=True,
        default_value=config.get("solvent_model"),
    )
    if isinstance(solvent_model, str) and solvent_model.lower() in (
        "none",
        "none (vacuum)",
        "vacuum",
        "없음",
    ):
        solvent_model = None
    if solvent_model is None:
        solvent = "vacuum"
    else:
        solvent_map_path = config.get("solvent_map", DEFAULT_SOLVENT_MAP_PATH)
        solvent_map = load_solvent_map(solvent_map_path)
        solvent_options = list(solvent_map.keys())
        solvent = _prompt_choice(
            "용매를 선택하세요:",
            solvent_options,
            allow_custom=True,
            default_value=config.get("solvent", "vacuum"),
        )
    single_point_enabled = False
    frequency_enabled = False
    single_point_config = config.get("single_point") or {}
    if calculation_mode == "optimization":
        frequency_enabled = _prompt_yes_no("프리퀀시 계산을 실행할까요?", default=True)
        if frequency_enabled:
            single_point_enabled = _prompt_yes_no(
                "프리퀀시 계산에서 허수진동수가 적절하게 나오면 "
                "단일점 계산을 진행하시겠습니까?",
                default=True,
            )
        else:
            single_point_enabled = _prompt_yes_no(
                "단일점(single point) 계산을 실행할까요?",
                default=True,
            )
        if single_point_enabled:
            sp_basis = _prompt_choice(
                "단일점 계산용 basis set을 선택하세요:",
                BASIS_SET_OPTIONS,
                allow_custom=True,
                default_value=basis,
            )
            sp_xc = _prompt_choice(
                "단일점 계산용 함수를 선택하세요:",
                XC_FUNCTIONAL_OPTIONS,
                allow_custom=True,
                default_value=xc,
            )
            sp_xc = normalize_xc_functional(sp_xc)
            sp_solvent_model = _prompt_choice(
                "단일점 계산용 용매 모델을 선택하세요:",
                SOLVENT_MODEL_OPTIONS,
                allow_custom=True,
                default_value=solvent_model,
            )
            if isinstance(sp_solvent_model, str) and sp_solvent_model.lower() in (
                "none",
                "none (vacuum)",
                "vacuum",
                "없음",
            ):
                sp_solvent_model = None
            if sp_solvent_model is None:
                sp_solvent = "vacuum"
            else:
                solvent_map_path = config.get("solvent_map", DEFAULT_SOLVENT_MAP_PATH)
                solvent_map = load_solvent_map(solvent_map_path)
                solvent_options = list(solvent_map.keys())
                sp_solvent = _prompt_choice(
                    "단일점 계산용 용매를 선택하세요:",
                    solvent_options,
                    allow_custom=True,
                    default_value=solvent,
                )
            single_point_config = dict(single_point_config)
            single_point_config["basis"] = sp_basis
            single_point_config["xc"] = sp_xc
            single_point_config["solvent_model"] = sp_solvent_model
            single_point_config["solvent"] = sp_solvent

    config = json.loads(json.dumps(config))
    config["calculation_mode"] = calculation_mode
    config["basis"] = basis
    config["xc"] = xc
    config["solvent_model"] = solvent_model
    config["solvent"] = solvent
    config["frequency_enabled"] = frequency_enabled
    config["single_point_enabled"] = single_point_enabled
    if single_point_config:
        config["single_point"] = single_point_config
    config_raw = json.dumps(config, indent=2, ensure_ascii=False)
    args.config = "<interactive>"
    return config, config_raw, base_config_path


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
    import numpy as np

    try:
        from ase import units
        from ase.calculators.calculator import Calculator, all_changes
        from ase.io import read as ase_read
        from ase.io import write as ase_write
        from ase.optimize import BFGS, FIRE, GPMin, LBFGS, MDMin
    except ImportError as exc:
        raise ImportError(
            "ASE optimizer requested but ASE or required calculators are not installed. "
            "Install ASE with DFTD3 support (e.g., `conda install -c conda-forge ase`)."
        ) from exc

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

    atoms = ase_read(input_xyz)
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
        atoms.calc = _SumCalculator([base_calc, dispersion_calc])
    else:
        atoms.calc = base_calc

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


def _load_run_metadata(metadata_path):
    if not metadata_path or not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def _resolve_status_metadata_path(status_target, default_metadata_name):
    if os.path.isdir(status_target):
        candidate = os.path.join(status_target, default_metadata_name)
        if os.path.exists(candidate):
            return candidate
        matches = sorted(glob.glob(os.path.join(status_target, "*metadata*.json")))
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise FileNotFoundError(
                "Multiple metadata files found in {path}: {matches}".format(
                    path=status_target, matches=", ".join(matches)
                )
            )
        raise FileNotFoundError(
            "No metadata JSON found in {path}.".format(path=status_target)
        )
    if os.path.isfile(status_target):
        return status_target
    raise FileNotFoundError(f"Status target not found: {status_target}")


def _format_elapsed(elapsed_seconds):
    if elapsed_seconds is None:
        return None
    seconds = int(elapsed_seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _tail_last_line(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = b""
            while position > 0:
                read_size = min(4096, position)
                position -= read_size
                handle.seek(position)
                buffer = handle.read(read_size) + buffer
                if b"\n" in buffer:
                    break
    except OSError:
        return None
    for line in reversed(buffer.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped.decode("utf-8", errors="ignore")
    return None


def _append_event_log(event_log_path, payload):
    if not event_log_path:
        return
    try:
        ensure_parent_dir(event_log_path)
        with open(event_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        logging.warning("Failed to write event log: %s", event_log_path)


def _record_status_event(event_log_path, run_id, run_dir, status, previous_status=None, details=None):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "event": "status_transition",
        "run_id": run_id,
        "run_directory": run_dir,
        "status": status,
        "previous_status": previous_status,
    }
    if details:
        payload["details"] = details
    _append_event_log(event_log_path, payload)


def _parse_iso_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _queue_priority_value(entry):
    try:
        return int(entry.get("priority", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _queue_entry_sort_key(entry):
    priority = _queue_priority_value(entry)
    queued_at = _parse_iso_timestamp(entry.get("queued_at")) or datetime.min
    return (-priority, queued_at, entry.get("run_id") or "")


def _ensure_queue_file(queue_path):
    ensure_parent_dir(queue_path)
    if not os.path.exists(queue_path):
        _write_queue(queue_path, {"entries": [], "updated_at": datetime.now().isoformat()})


def _queue_backup_path(queue_path):
    return f"{queue_path}.bak"


def _queue_corrupt_path(queue_path):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{queue_path}.{timestamp}.corrupt"


def _load_queue_from_path(queue_path):
    with open(queue_path, "r", encoding="utf-8") as queue_file:
        return json.load(queue_file)


def _handle_corrupt_queue(queue_path):
    backup_path = _queue_backup_path(queue_path)
    try:
        backup_state = _load_queue_from_path(backup_path)
    except (OSError, json.JSONDecodeError):
        backup_state = None
    if backup_state is not None:
        corrupt_path = _queue_corrupt_path(queue_path)
        try:
            os.replace(queue_path, corrupt_path)
        except OSError:
            logging.warning("Failed to move corrupt queue file to %s", corrupt_path)
        else:
            logging.warning("Moved corrupt queue file to %s", corrupt_path)
        try:
            shutil.copy2(backup_path, queue_path)
        except OSError:
            logging.warning("Failed to restore queue from backup at %s", backup_path)
        else:
            logging.warning("Restored queue from backup at %s", backup_path)
        return backup_state
    corrupt_path = _queue_corrupt_path(queue_path)
    try:
        os.replace(queue_path, corrupt_path)
    except OSError:
        logging.warning("Failed to move corrupt queue file to %s", corrupt_path)
    else:
        logging.warning("Moved corrupt queue file to %s", corrupt_path)
    return {"entries": [], "updated_at": None}


def _load_queue(queue_path):
    if not os.path.exists(queue_path):
        return {"entries": [], "updated_at": None}
    try:
        return _load_queue_from_path(queue_path)
    except OSError:
        logging.warning("Failed to read queue file: %s", queue_path)
        return {"entries": [], "updated_at": None}
    except json.JSONDecodeError:
        logging.warning("Queue file is corrupt: %s", queue_path)
        return _handle_corrupt_queue(queue_path)


def _write_queue(queue_path, queue_state):
    queue_state["updated_at"] = datetime.now().isoformat()
    ensure_parent_dir(queue_path)
    queue_dir = os.path.dirname(queue_path) or "."
    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=queue_dir,
        prefix=".queue.json.",
        suffix=".tmp",
        delete=False,
    )
    try:
        backup_path = _queue_backup_path(queue_path)
        if os.path.exists(queue_path):
            try:
                shutil.copy2(queue_path, backup_path)
            except OSError:
                logging.warning("Failed to create queue backup at %s", backup_path)
        with temp_handle as queue_file:
            json.dump(queue_state, queue_file, indent=2)
            queue_file.flush()
            os.fsync(queue_file.fileno())
        os.replace(temp_handle.name, queue_path)
    finally:
        if os.path.exists(temp_handle.name):
            try:
                os.remove(temp_handle.name)
            except FileNotFoundError:
                pass


def _read_lock_info(lock_path):
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            contents = handle.read().strip()
    except OSError:
        return None, None
    if not contents:
        return None, None
    parts = contents.split()
    try:
        pid = int(parts[0])
    except ValueError:
        pid = None
    timestamp = parts[1] if len(parts) > 1 else None
    return pid, timestamp


def _is_lock_stale(lock_path, stale_timeout):
    pid, _timestamp = _read_lock_info(lock_path)
    if pid and _is_pid_running(pid):
        return False
    try:
        mtime = os.path.getmtime(lock_path)
    except OSError:
        return False
    return (time.time() - mtime) > stale_timeout


def _acquire_lock(lock_path, timeout=10, delay=0.1, stale_timeout=60):
    start_time = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _is_lock_stale(lock_path, stale_timeout):
                try:
                    os.remove(lock_path)
                    continue
                except FileNotFoundError:
                    continue
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}") from None
            time.sleep(delay)
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()} {datetime.now().isoformat()}")
            return


@contextmanager
def _queue_lock(lock_path):
    _acquire_lock(lock_path)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _is_pid_running(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _read_runner_pid(lock_path):
    if not os.path.exists(lock_path):
        return None
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _ensure_queue_runner_started(command, log_path):
    ensure_parent_dir(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
    if log_path:
        ensure_parent_dir(log_path)
    lock_exists = os.path.exists(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
    existing_pid = _read_runner_pid(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
    if existing_pid is None and lock_exists:
        try:
            os.remove(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
        except FileNotFoundError:
            pass
    if existing_pid and _is_pid_running(existing_pid):
        return
    if existing_pid and not _is_pid_running(existing_pid):
        try:
            os.remove(DEFAULT_QUEUE_RUNNER_LOCK_PATH)
        except FileNotFoundError:
            pass
    if log_path:
        with open(log_path, "a", encoding="utf-8") as log_file:
            subprocess.Popen(
                command,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
    else:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _enqueue_run(entry, queue_path, lock_path):
    _ensure_queue_file(queue_path)
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        entries = queue_state.get("entries") or []
        if any(item.get("run_id") == entry["run_id"] for item in entries):
            raise ValueError(f"Run ID already queued: {entry['run_id']}")
        entries.append(entry)
        queue_state["entries"] = entries
        _write_queue(queue_path, queue_state)
        queued_positions = [item for item in entries if item.get("status") == "queued"]
        return len(queued_positions)


def _register_foreground_run(entry, queue_path, lock_path):
    _ensure_queue_file(queue_path)
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        entries = queue_state.get("entries") or []
        if any(item.get("run_id") == entry["run_id"] for item in entries):
            raise ValueError(f"Run ID already queued: {entry['run_id']}")
        entries.append(entry)
        queue_state["entries"] = entries
        _write_queue(queue_path, queue_state)


def _update_queue_status(queue_path, lock_path, run_id, status, exit_code=None):
    timestamp = datetime.now().isoformat()

    def _apply_update(entry):
        if entry.get("run_id") == run_id:
            entry["status"] = status
            if status in ("running", "started"):
                entry["started_at"] = entry.get("started_at") or timestamp
            if status in ("completed", "failed", "timeout", "canceled"):
                entry["ended_at"] = timestamp
                entry["exit_code"] = exit_code

    _update_queue_entry(queue_path, lock_path, run_id, _apply_update)


def _update_queue_entry(queue_path, lock_path, run_id, updater):
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        entries = queue_state.get("entries") or []
        updated = False
        for item in entries:
            if item.get("run_id") == run_id:
                updater(item)
                updated = True
                break
        if updated:
            _write_queue(queue_path, queue_state)
        return updated


def _cancel_queue_entry(queue_path, lock_path, run_id):
    event_log_path = None
    run_dir = None

    def _apply_cancel(entry):
        if entry.get("status") == "queued":
            entry["status"] = "canceled"
            entry["canceled_at"] = datetime.now().isoformat()
            entry["exit_code"] = None

    updated = _update_queue_entry(queue_path, lock_path, run_id, _apply_cancel)
    if not updated:
        return False, "Run ID not found in queue."

    metadata_path = None
    status = None
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        for item in queue_state.get("entries") or []:
            if item.get("run_id") == run_id:
                metadata_path = item.get("run_metadata_file")
                status = item.get("status")
                event_log_path = item.get("event_log_file")
                run_dir = item.get("run_directory")
                break
    if status != "canceled":
        return False, "Run is not queued (already running or completed)."
    if metadata_path:
        metadata = _load_run_metadata(metadata_path) or {}
        metadata["status"] = "canceled"
        metadata["run_ended_at"] = datetime.now().isoformat()
        metadata["canceled_at"] = metadata["run_ended_at"]
        write_run_metadata(metadata_path, metadata)
    _record_status_event(
        event_log_path,
        run_id,
        run_dir,
        "canceled",
        previous_status="queued",
    )
    return True, None


def _requeue_queue_entry(queue_path, lock_path, run_id, reason):
    requeued_at = datetime.now().isoformat()
    event_log_path = None
    run_dir = None
    metadata_path = None
    previous_status = None

    def _apply_requeue(entry):
        nonlocal event_log_path, run_dir, metadata_path, previous_status
        previous_status = entry.get("status")
        entry["status"] = "queued"
        entry["queued_at"] = requeued_at
        entry["started_at"] = None
        entry["ended_at"] = None
        entry["exit_code"] = None
        entry["retry_count"] = int(entry.get("retry_count", 0) or 0) + 1
        entry["requeued_at"] = requeued_at
        event_log_path = entry.get("event_log_file")
        run_dir = entry.get("run_directory")
        metadata_path = entry.get("run_metadata_file")

    updated = _update_queue_entry(queue_path, lock_path, run_id, _apply_requeue)
    if not updated:
        return False, "Run ID not found in queue."
    if previous_status == "queued":
        return False, "Run is already queued."
    if metadata_path:
        metadata = _load_run_metadata(metadata_path) or {}
        metadata["status"] = "queued"
        metadata["queued_at"] = requeued_at
        metadata["run_started_at"] = None
        metadata["run_ended_at"] = None
        metadata["requeued_at"] = requeued_at
        write_run_metadata(metadata_path, metadata)
    _record_status_event(
        event_log_path,
        run_id,
        run_dir,
        "queued",
        previous_status=previous_status,
        details={"reason": reason},
    )
    return True, None


def _requeue_failed_entries(queue_path, lock_path):
    requeued = []
    failed_statuses = {"failed", "timeout"}
    with _queue_lock(lock_path):
        queue_state = _load_queue(queue_path)
        entries = queue_state.get("entries") or []
        for entry in entries:
            status_before = entry.get("status")
            if status_before in failed_statuses:
                entry["status"] = "queued"
                entry["queued_at"] = datetime.now().isoformat()
                entry["started_at"] = None
                entry["ended_at"] = None
                entry["exit_code"] = None
                entry["retry_count"] = int(entry.get("retry_count", 0) or 0) + 1
                entry["requeued_at"] = entry["queued_at"]
                requeued.append({"entry": dict(entry), "previous_status": status_before})
        if requeued:
            _write_queue(queue_path, queue_state)
    for item in requeued:
        entry = item["entry"]
        metadata_path = entry.get("run_metadata_file")
        if metadata_path:
            metadata = _load_run_metadata(metadata_path) or {}
            metadata["status"] = "queued"
            metadata["queued_at"] = entry.get("queued_at")
            metadata["run_started_at"] = None
            metadata["run_ended_at"] = None
            metadata["requeued_at"] = entry.get("requeued_at")
            write_run_metadata(metadata_path, metadata)
        _record_status_event(
            entry.get("event_log_file"),
            entry.get("run_id"),
            entry.get("run_directory"),
            "queued",
            previous_status=item.get("previous_status"),
            details={"reason": "requeue_failed"},
        )
    return len(requeued)


def _format_queue_status(queue_state):
    entries = queue_state.get("entries") or []
    if not entries:
        print("Queue is empty.")
        return
    print("Queue status")
    queued_index = 0
    for entry in entries:
        status = entry.get("status", "unknown")
        priority = _queue_priority_value(entry)
        max_runtime_seconds = entry.get("max_runtime_seconds")
        if status == "queued":
            queued_index += 1
            position = f"{queued_index}"
        else:
            position = "-"
        timestamp_label = "queued_at"
        timestamp_value = entry.get("queued_at")
        if status in ("running", "started"):
            timestamp_label = "started_at"
            timestamp_value = entry.get("started_at") or entry.get("run_started_at")
        elif status not in ("queued",):
            timestamp_label = "ended_at"
            timestamp_value = entry.get("ended_at")
        exit_code = entry.get("exit_code")
        exit_code_label = f", exit_code={exit_code}" if exit_code is not None else ""
        print(
            "  [{pos}] {run_id} {status} ({timestamp}={timestamp_value}, priority={priority}{exit_code})".format(
                pos=position,
                run_id=entry.get("run_id"),
                status=status,
                timestamp=timestamp_label,
                timestamp_value=timestamp_value,
                priority=priority,
                exit_code=exit_code_label,
            )
        )
        run_dir = entry.get("run_directory")
        if run_dir:
            print(f"        run_dir={run_dir}")
        if max_runtime_seconds:
            print(f"        max_runtime_seconds={max_runtime_seconds}")


def _run_queue_worker(script_path, queue_path, lock_path, runner_lock_path):
    ensure_parent_dir(queue_path)
    try:
        fd = os.open(runner_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except OSError:
        return
    try:
        while True:
            entry = None
            entry_status_before = None
            with _queue_lock(lock_path):
                queue_state = _load_queue(queue_path)
                entries = queue_state.get("entries") or []
                candidates = [
                    (index, item)
                    for index, item in enumerate(entries)
                    if item.get("status") == "queued"
                ]
                if candidates:
                    index, item = min(
                        candidates,
                        key=lambda candidate: _queue_entry_sort_key(candidate[1]),
                    )
                    entry_status_before = item.get("status")
                    item["status"] = "running"
                    item["started_at"] = datetime.now().isoformat()
                    entry = dict(item)
                    entries[index] = item
                    _write_queue(queue_path, queue_state)
            if entry is None:
                break
            _record_status_event(
                entry.get("event_log_file"),
                entry.get("run_id"),
                entry.get("run_directory"),
                "running",
                previous_status=entry_status_before,
                details={
                    "priority": _queue_priority_value(entry),
                    "max_runtime_seconds": entry.get("max_runtime_seconds"),
                },
            )
            command = [
                sys.executable,
                script_path,
                entry["xyz_file"],
                "--config",
                entry["config_file"],
                "--solvent-map",
                entry["solvent_map"],
                "--run-dir",
                entry["run_directory"],
                "--run-id",
                entry["run_id"],
                "--no-background",
                "--non-interactive",
            ]
            timeout_seconds = entry.get("max_runtime_seconds")
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    timeout=timeout_seconds if timeout_seconds else None,
                )
                status = "completed" if result.returncode == 0 else "failed"
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
                exit_code = -1
            finished_at = datetime.now().isoformat()

            def _apply_update(
                item,
                run_id=entry["run_id"],
                update_status=status,
                finished_time=finished_at,
                update_exit_code=exit_code,
            ):
                if item.get("run_id") == run_id:
                    item["status"] = update_status
                    item["ended_at"] = finished_time
                    item["exit_code"] = update_exit_code

            _update_queue_entry(queue_path, lock_path, entry["run_id"], _apply_update)
            _record_status_event(
                entry.get("event_log_file"),
                entry.get("run_id"),
                entry.get("run_directory"),
                status,
                previous_status="running",
                details={"exit_code": exit_code},
            )
    finally:
        try:
            os.remove(runner_lock_path)
        except FileNotFoundError:
            pass


def _print_status(status_target, default_metadata_name):
    metadata_path = _resolve_status_metadata_path(status_target, default_metadata_name)
    metadata = _load_run_metadata(metadata_path)
    if not metadata:
        raise FileNotFoundError(f"Metadata file is empty: {metadata_path}")
    summary = metadata.get("summary") or {}
    run_dir = metadata.get("run_directory") or os.path.dirname(metadata_path)
    run_id = metadata.get("run_id")
    status = metadata.get("status") or "unknown"
    run_started_at = metadata.get("run_started_at")
    run_ended_at = metadata.get("run_ended_at")
    elapsed_seconds = summary.get("elapsed_seconds")
    if elapsed_seconds is None and run_started_at and status in ("started", "running", "queued"):
        try:
            started_dt = datetime.fromisoformat(run_started_at)
            elapsed_seconds = (datetime.now() - started_dt).total_seconds()
        except ValueError:
            elapsed_seconds = None
    n_steps = summary.get("n_steps") if summary else metadata.get("n_steps")
    final_energy = summary.get("final_energy") if summary else None
    log_path = metadata.get("log_file")
    last_log_line = _tail_last_line(log_path)
    print("Run status summary")
    print(f"  Status       : {status}")
    if run_id:
        print(f"  Run ID       : {run_id}")
    if run_dir:
        print(f"  Run dir      : {run_dir}")
    if run_started_at:
        print(f"  Started at   : {run_started_at}")
    if run_ended_at:
        print(f"  Ended at     : {run_ended_at}")
    if elapsed_seconds is not None:
        print(f"  Elapsed      : {_format_elapsed(elapsed_seconds)}")
    if n_steps is not None:
        print(f"  Steps        : {n_steps}")
    if final_energy is not None:
        print(f"  Final energy : {final_energy}")
    if log_path:
        print(f"  Log file     : {log_path}")
    if last_log_line:
        print(f"  Last log     : {last_log_line}")
    optimized_xyz = metadata.get("optimized_xyz_file")
    if optimized_xyz:
        print(f"  Optimized XYZ: {optimized_xyz}")
    print(f"  Metadata     : {metadata_path}")


def _print_recent_statuses(count, base_dir="runs"):
    if count is None or count <= 0:
        raise ValueError("Recent status count must be a positive integer.")
    metadata_paths = sorted(glob.glob(os.path.join(base_dir, "*", "metadata*.json")))
    items = []
    for path in metadata_paths:
        metadata = _load_run_metadata(path)
        if not metadata:
            continue
        status = metadata.get("status") or "unknown"
        run_id = metadata.get("run_id")
        run_dir = metadata.get("run_directory") or os.path.dirname(path)
        run_started_at = metadata.get("run_started_at")
        run_ended_at = metadata.get("run_ended_at")
        summary = metadata.get("summary") or {}
        elapsed_seconds = summary.get("elapsed_seconds")
        if elapsed_seconds is None and run_started_at and status in ("started", "running", "queued"):
            started_dt = _parse_iso_timestamp(run_started_at)
            if started_dt:
                elapsed_seconds = (datetime.now() - started_dt).total_seconds()
        sort_key = _parse_iso_timestamp(run_started_at) or _parse_iso_timestamp(
            metadata.get("run_updated_at")
        )
        if sort_key is None:
            try:
                sort_key = datetime.fromtimestamp(os.path.getmtime(path))
            except OSError:
                sort_key = datetime.min
        items.append(
            {
                "path": path,
                "run_id": run_id,
                "run_dir": run_dir,
                "status": status,
                "run_started_at": run_started_at,
                "run_ended_at": run_ended_at,
                "elapsed": _format_elapsed(elapsed_seconds) if elapsed_seconds is not None else None,
                "final_energy": summary.get("final_energy"),
                "n_steps": summary.get("n_steps"),
                "sort_key": sort_key,
            }
        )
    if not items:
        print("No recent runs found.")
        return
    items.sort(key=lambda item: item["sort_key"], reverse=True)
    print(f"Recent runs (latest {min(count, len(items))})")
    for item in items[:count]:
        print(
            "  {status:9} {run_id} (started={started}, ended={ended})".format(
                status=item["status"],
                run_id=item["run_id"],
                started=item["run_started_at"],
                ended=item["run_ended_at"],
            )
        )
        print(f"        run_dir={item['run_dir']}")
        if item["elapsed"]:
            print(f"        elapsed={item['elapsed']}")
        if item["n_steps"] is not None:
            print(f"        steps={item['n_steps']}")
        if item["final_energy"] is not None:
            print(f"        final_energy={item['final_energy']}")
        print(f"        metadata={item['path']}")


def _normalize_cli_args(argv):
    if not argv:
        return argv
    command = argv[0]
    if command == "doctor":
        return ["--doctor", *argv[1:]]
    if command != "validate-config":
        return argv
    remaining = argv[1:]
    config_path = None
    if remaining and not remaining[0].startswith("-"):
        config_path = remaining[0]
        remaining = remaining[1:]
    normalized = ["--validate-only"]
    if config_path:
        normalized.extend(["--config", config_path])
    normalized.extend(remaining)
    return normalized


def _run_doctor():
    def format_doctor_result(label, status, remedy=None):
        status_label = "OK" if status else "FAIL"
        separator = "  " if status_label == "OK" else " "
        if status:
            return f"{status_label}{separator}{label}"
        if remedy:
            return f"{status_label}{separator}{label} ({remedy})"
        return f"{status_label}{separator}{label}"

    failures = []

    def _record_check(label, ok, remedy=None):
        if not ok:
            failures.append(label)
        print(format_doctor_result(label, ok, remedy))

    def _check_import(module_name, hint):
        spec = importlib.util.find_spec(module_name)
        ok = spec is not None
        _record_check(module_name, ok, hint if not ok else None)
        return ok

    def _solvent_map_hint(error):
        if isinstance(error, FileNotFoundError):
            return (
                "Missing solvent map file. Provide --solvent-map or restore "
                f"{DEFAULT_SOLVENT_MAP_PATH}."
            )
        if isinstance(error, json.JSONDecodeError):
            return "Invalid JSON in solvent map. Fix the JSON syntax."
        return "Unable to read solvent map. Check file permissions and path."

    try:
        load_solvent_map(DEFAULT_SOLVENT_MAP_PATH)
        _record_check("solvent_map", True)
    except Exception as exc:
        _record_check("solvent_map", False, _solvent_map_hint(exc))

    checks = [
        ("pyscf", "Install with: pip install pyscf"),
        ("pyscf.dft", "Install with: pip install pyscf"),
        ("pyscf.gto", "Install with: pip install pyscf"),
        ("pyscf.hessian.thermo", "Install with: pip install pyscf"),
        ("dftd3", "Install with: pip install dftd3"),
        ("dftd4", "Install with: pip install dftd4"),
    ]
    for module_name, hint in checks:
        _check_import(module_name, hint)

    if failures:
        print(f"FAIL {len(failures)} checks failed: {', '.join(failures)}")
        sys.exit(1)
    print("OK  all checks passed")


def main():
    """
    Main function to run the geometry optimization.
    """
    parser = argparse.ArgumentParser(description="Optimize molecular geometry using PySCF and ASE.")
    parser.add_argument(
        "xyz_file",
        nargs="?",
        help="Path to the .xyz file with the initial molecular geometry.",
    )
    parser.add_argument(
        "--solvent-map",
        default=DEFAULT_SOLVENT_MAP_PATH,
        help=(
            "Path to JSON file mapping solvent names to dielectric constants "
            f"(default: {DEFAULT_SOLVENT_MAP_PATH})."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to JSON config file for runtime settings "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=None,
        help="Prompt for run settings interactively (default).",
    )
    parser.add_argument(
        "--non-interactive",
        "--advanced",
        action="store_true",
        help="Run with explicit inputs/configs without prompts (advanced).",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Launch in the background queue (optional).",
    )
    parser.add_argument(
        "--run-dir",
        help="Optional run directory to use (useful for background launches).",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run ID to use (useful for background launches).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the JSON config file and exit without running a calculation.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run environment diagnostics and exit (e.g., python run_opt.py --doctor).",
    )
    parser.add_argument(
        "--status",
        help=(
            "Show a summary for a run directory or metadata JSON file "
            "(e.g., --status runs/2024-01-01_120000)."
        ),
    )
    parser.add_argument(
        "--queue-status",
        action="store_true",
        help="Show queue status (includes foreground runs).",
    )
    parser.add_argument(
        "--queue-cancel",
        metavar="RUN_ID",
        help="Cancel a queued reservation by run ID.",
    )
    parser.add_argument(
        "--queue-retry",
        metavar="RUN_ID",
        help="Retry a run by re-queuing it (failed/timeout/canceled runs).",
    )
    parser.add_argument(
        "--queue-requeue-failed",
        action="store_true",
        help="Re-queue all failed/timeout runs.",
    )
    parser.add_argument(
        "--queue-priority",
        type=int,
        default=0,
        help="Priority for the queued run (higher runs first).",
    )
    parser.add_argument(
        "--queue-max-runtime",
        type=int,
        help="Max runtime in seconds for queued runs (timeout if exceeded).",
    )
    parser.add_argument(
        "--status-recent",
        type=int,
        metavar="N",
        help="Show a summary list for the most recent N runs.",
    )
    parser.add_argument("--no-background", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--queue-runner", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(_normalize_cli_args(sys.argv[1:]))

    if args.doctor:
        _run_doctor()
        return

    run_in_background = bool(args.background and not args.no_background)
    if args.interactive and args.non_interactive:
        raise ValueError("--interactive and --non-interactive cannot be used together.")
    if args.validate_only and args.interactive:
        raise ValueError("--validate-only cannot be used with --interactive.")
    if args.interactive is None:
        args.interactive = not args.non_interactive
    if args.validate_only:
        args.interactive = False
        args.non_interactive = True
    if args.queue_status:
        _ensure_queue_file(DEFAULT_QUEUE_PATH)
        with _queue_lock(DEFAULT_QUEUE_LOCK_PATH):
            queue_state = _load_queue(DEFAULT_QUEUE_PATH)
        _format_queue_status(queue_state)
        return

    try:
        if args.queue_runner:
            _run_queue_worker(
                os.path.abspath(sys.argv[0]),
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
                DEFAULT_QUEUE_RUNNER_LOCK_PATH,
            )
            return

        if args.queue_cancel:
            _ensure_queue_file(DEFAULT_QUEUE_PATH)
            canceled, error = _cancel_queue_entry(
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
                args.queue_cancel,
            )
            if not canceled:
                raise ValueError(error)
            print(f"Canceled queued run: {args.queue_cancel}")
            return
        if args.queue_retry:
            _ensure_queue_file(DEFAULT_QUEUE_PATH)
            retried, error = _requeue_queue_entry(
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
                args.queue_retry,
                reason="retry",
            )
            if not retried:
                raise ValueError(error)
            print(f"Re-queued run: {args.queue_retry}")
            return
        if args.queue_requeue_failed:
            _ensure_queue_file(DEFAULT_QUEUE_PATH)
            count = _requeue_failed_entries(DEFAULT_QUEUE_PATH, DEFAULT_QUEUE_LOCK_PATH)
            print(f"Re-queued failed runs: {count}")
            return

        if args.status_recent:
            _print_recent_statuses(args.status_recent)
            return

        if args.status:
            _print_status(args.status, DEFAULT_RUN_METADATA_PATH)
            return

        config_source_path = None
        if args.interactive:
            config, config_raw, config_source_path = _prompt_interactive_config(args)
            if config_source_path is not None:
                config_source_path = config_source_path.resolve()
        else:
            if not args.xyz_file and not args.validate_only:
                raise ValueError(
                    "xyz_file is required unless --status or --validate-only is used."
                )
            config_path = Path(args.config).expanduser().resolve()
            config, config_raw = load_run_config(config_path)
            args.config = str(config_path)
            config_source_path = config_path

        try:
            validate_run_config(config)
        except ValueError as error:
            message = str(error)
            print(message, file=sys.stderr)
            logging.error(message)
            raise
        if args.validate_only:
            print(f"Config validation passed: {args.config}")
            return
        calculation_mode = _normalize_calculation_mode(config.get("calculation_mode"))
        basis = config.get("basis")
        xc = normalize_xc_functional(config.get("xc"))
        solvent_name = config.get("solvent")
        solvent_model = config.get("solvent_model")
        dispersion_model = config.get("dispersion")
        optimizer_config = config.get("optimizer") or {}
        optimizer_mode = None
        if calculation_mode == "optimization":
            optimizer_mode = _normalize_optimizer_mode(optimizer_config.get("mode"))
        solvent_map_path = config.get("solvent_map", DEFAULT_SOLVENT_MAP_PATH)
        single_point_config = config.get("single_point") or {}
        frequency_enabled = config.get("frequency_enabled")
        single_point_enabled = config.get("single_point_enabled", True)
        if calculation_mode != "optimization":
            frequency_enabled = False
            single_point_enabled = False
        if frequency_enabled is None and calculation_mode == "optimization":
            frequency_enabled = True
        if not basis:
            raise ValueError("Config must define 'basis' in the JSON config file.")
        if not xc:
            raise ValueError("Config must define 'xc' in the JSON config file.")
        if solvent_model and not solvent_name:
            raise ValueError("Config must define 'solvent' when 'solvent_model' is set.")
        if solvent_name:
            if not solvent_model and not _is_vacuum_solvent(solvent_name):
                raise ValueError("Config must define 'solvent_model' when 'solvent' is set.")
            if solvent_model and solvent_model.lower() not in ("pcm", "smd"):
                raise ValueError("Config 'solvent_model' must be one of: pcm, smd.")
        thread_count = config.get("threads", DEFAULT_THREAD_COUNT)
        memory_gb = config.get("memory_gb")
        verbose = bool(config.get("verbose", False))
        run_dir = args.run_dir or create_run_directory()
        os.makedirs(run_dir, exist_ok=True)
        log_path = resolve_run_path(run_dir, config.get("log_file", DEFAULT_LOG_PATH))
        log_path = format_log_path(log_path)
        scf_config = config.get("scf", {})
        optimized_xyz_path = resolve_run_path(
            run_dir, config.get("optimized_xyz_file", DEFAULT_OPTIMIZED_XYZ_PATH)
        )
        run_metadata_path = resolve_run_path(
            run_dir, config.get("run_metadata_file", DEFAULT_RUN_METADATA_PATH)
        )
        frequency_output_path = resolve_run_path(
            run_dir, config.get("frequency_file", DEFAULT_FREQUENCY_PATH)
        )
        ensure_parent_dir(log_path)
        ensure_parent_dir(optimized_xyz_path)
        ensure_parent_dir(run_metadata_path)
        ensure_parent_dir(frequency_output_path)

        if args.interactive:
            config_used_path = resolve_run_path(run_dir, "config_used.json")
            ensure_parent_dir(config_used_path)
            with open(config_used_path, "w", encoding="utf-8") as config_used_file:
                config_used_file.write(config_raw)
            args.config = config_used_path

        run_id = args.run_id or str(uuid.uuid4())
        event_log_path = resolve_run_path(
            run_dir, config.get("event_log_file", DEFAULT_EVENT_LOG_PATH)
        )
        if event_log_path:
            event_log_path = format_log_path(event_log_path)
            ensure_parent_dir(event_log_path)
        if run_in_background:
            queued_at = datetime.now().isoformat()
            queue_priority = args.queue_priority
            max_runtime_seconds = args.queue_max_runtime
            queued_metadata = {
                "status": "queued",
                "run_directory": run_dir,
                "run_id": run_id,
                "xyz_file": args.xyz_file,
                "config_file": args.config,
                "run_metadata_file": run_metadata_path,
                "log_file": log_path,
                "event_log_file": event_log_path,
                "queued_at": queued_at,
                "priority": queue_priority,
                "max_runtime_seconds": max_runtime_seconds,
            }
            write_run_metadata(run_metadata_path, queued_metadata)
            queue_entry = {
                "status": "queued",
                "run_directory": run_dir,
                "run_id": run_id,
                "xyz_file": args.xyz_file,
                "config_file": args.config,
                "solvent_map": args.solvent_map,
                "run_metadata_file": run_metadata_path,
                "log_file": log_path,
                "event_log_file": event_log_path,
                "queued_at": queued_at,
                "priority": queue_priority,
                "max_runtime_seconds": max_runtime_seconds,
                "retry_count": 0,
            }
            position = _enqueue_run(queue_entry, DEFAULT_QUEUE_PATH, DEFAULT_QUEUE_LOCK_PATH)
            _record_status_event(
                event_log_path,
                run_id,
                run_dir,
                "queued",
                previous_status=None,
                details={
                    "priority": queue_priority,
                    "max_runtime_seconds": max_runtime_seconds,
                },
            )
            runner_command = [
                sys.executable,
                os.path.abspath(sys.argv[0]),
                "--queue-runner",
            ]
            _ensure_queue_runner_started(runner_command, DEFAULT_QUEUE_RUNNER_LOG_PATH)
            print("Background run queued.")
            print(f"  Run ID       : {run_id}")
            print(f"  Queue pos    : {position}")
            print(f"  Run dir      : {run_dir}")
            print(f"  Metadata     : {run_metadata_path}")
            print(f"  Log file     : {log_path}")
            print(f"  Queue runner : {DEFAULT_QUEUE_RUNNER_LOG_PATH}")
            return
        setup_logging(log_path, verbose, run_id=run_id, event_log_path=event_log_path)
        if config_source_path is not None:
            logging.info("Loaded config file: %s", config_source_path)
        queue_tracking = False
        if not run_in_background and not args.no_background:
            started_at = datetime.now().isoformat()
            foreground_entry = {
                "status": "running",
                "run_directory": run_dir,
                "run_id": run_id,
                "xyz_file": args.xyz_file,
                "config_file": args.config,
                "solvent_map": args.solvent_map,
                "run_metadata_file": run_metadata_path,
                "log_file": log_path,
                "event_log_file": event_log_path,
                "started_at": started_at,
                "priority": args.queue_priority,
                "max_runtime_seconds": args.queue_max_runtime,
                "retry_count": 0,
            }
            _register_foreground_run(
                foreground_entry,
                DEFAULT_QUEUE_PATH,
                DEFAULT_QUEUE_LOCK_PATH,
            )
            queue_tracking = True

        def _update_foreground_queue(status, exit_code=None):
            if queue_tracking:
                _update_queue_status(
                    DEFAULT_QUEUE_PATH,
                    DEFAULT_QUEUE_LOCK_PATH,
                    run_id,
                    status,
                    exit_code=exit_code,
                )
        try:
            thread_status = apply_thread_settings(thread_count)
            openmp_available = thread_status.get("openmp_available")
            effective_threads = thread_status.get("effective_threads")
            enforce_os_memory_limit = config.get("enforce_os_memory_limit", False)
            memory_mb, memory_limit_status = apply_memory_limit(memory_gb, enforce_os_memory_limit)
            memory_limit_enforced = bool(memory_limit_status and memory_limit_status.get("applied"))

            from pyscf import dft, gto
            atom_spec, charge, spin, multiplicity = load_xyz(args.xyz_file)
            if optimizer_mode == "transition_state" and multiplicity is None:
                if args.interactive:
                    logging.info("TS 모드: multiplicity 입력 강제")
                    while True:
                        raw_value = input("Multiplicity(2S+1)를 입력하세요: ").strip()
                        try:
                            multiplicity = int(raw_value)
                        except ValueError:
                            print("Multiplicity는 양의 정수여야 합니다.")
                            continue
                        if multiplicity < 1:
                            print("Multiplicity는 양의 정수여야 합니다.")
                            continue
                        break
                else:
                    raise ValueError(
                        "Transition-state mode requires multiplicity; "
                        "provide it in the XYZ comment line or run with --interactive."
                    )
            total_electrons = total_electron_count(atom_spec, charge)
            if multiplicity is not None:
                if multiplicity < 1:
                    raise ValueError("Multiplicity must be a positive integer (2S+1).")
                multiplicity_spin = multiplicity - 1
                if spin is not None and spin != multiplicity_spin:
                    raise ValueError(
                        "Spin and multiplicity are inconsistent. "
                        f"spin={spin} implies multiplicity={spin + 1}, "
                        f"but multiplicity={multiplicity} was provided."
                    )
                spin = multiplicity_spin
            if spin is None:
                spin = total_electrons % 2
                logging.warning(
                    "Auto spin estimation enabled: spin not specified; using parity "
                    "(spin = total_electrons %% 2). For TS/radical/metal/diradical cases, "
                    "set multiplicity in the XYZ comment line to avoid incorrect states."
                )
            elif spin < 0 or spin > total_electrons:
                raise ValueError(
                    "Spin is outside the valid electron count range. "
                    f"Total electrons: {total_electrons}, spin: {spin}. "
                    "Spin must be between 0 and total electrons."
                )
            elif (total_electrons - spin) % 2 != 0:
                raise ValueError(
                    "Spin is inconsistent with electron count. "
                    f"Total electrons: {total_electrons}, spin: {spin}. "
                    "Spin must satisfy (Nalpha - Nbeta) with Nalpha+Nbeta=total electrons."
                )
            if multiplicity is None:
                multiplicity = spin + 1
            mol = gto.M(atom=atom_spec, basis=basis, charge=charge, spin=spin)
            if memory_mb:
                mol.max_memory = memory_mb

            ks_type = select_ks_type(
                mol=mol,
                scf_config=scf_config,
                optimizer_mode=optimizer_mode,
                multiplicity=multiplicity,
            )
            if ks_type == "RKS":
                mf = dft.RKS(mol)
            else:
                mf = dft.UKS(mol)
            mf.xc = xc

            calculation_label = {
                "optimization": "Geometry optimization",
                "single_point": "Single-point",
                "frequency": "Frequency",
            }[calculation_mode]
            dispersion_model = _normalize_dispersion_settings(
                calculation_label,
                xc,
                dispersion_model,
                allow_dispersion=True,
            )

            solvent_name, solvent_model = _normalize_solvent_settings(
                calculation_label,
                solvent_name,
                solvent_model,
            )

            eps = None
            if solvent_name:
                solvent_key = solvent_name.lower()
                if not _is_vacuum_solvent(solvent_key):
                    solvent_model_lower = solvent_model.lower() if solvent_model else None
                    if solvent_model_lower == "pcm":
                        solvent_map = load_solvent_map(solvent_map_path)
                        eps = solvent_map.get(solvent_key)
                        if eps is None:
                            available = ", ".join(sorted(solvent_map.keys()))
                            raise ValueError(
                                "Solvent '{name}' not found in solvent map {path}. "
                                "Available solvents: {available}.".format(
                                    name=solvent_name,
                                    path=solvent_map_path,
                                    available=available,
                                )
                            )
                    mf = apply_solvent_model(
                        mf,
                        solvent_model_lower,
                        solvent_key,
                        eps,
                    )
            dispersion_info = None
            if verbose:
                mf.verbose = 4
            applied_scf = scf_config or None
            if calculation_mode == "optimization":
                applied_scf = apply_scf_settings(mf, scf_config)

            sp_basis = single_point_config.get("basis") or basis
            sp_xc = normalize_xc_functional(single_point_config.get("xc") or xc)
            sp_scf_config = single_point_config.get("scf") or scf_config
            sp_solvent_name = single_point_config.get("solvent") or solvent_name
            sp_solvent_model = single_point_config.get("solvent_model") or solvent_model
            sp_dispersion_model = single_point_config.get("dispersion")
            sp_solvent_map_path = single_point_config.get("solvent_map") or solvent_map_path
            if not sp_basis:
                raise ValueError(
                    "Single-point config must define 'basis' or fall back to base 'basis'."
                )
            if sp_xc is None:
                raise ValueError("Single-point config must define 'xc' or fall back to base 'xc'.")
            if sp_solvent_model and not sp_solvent_name:
                raise ValueError(
                    "Single-point config must define 'solvent' when 'solvent_model' is set."
                )
            if sp_solvent_name:
                if not sp_solvent_model and not _is_vacuum_solvent(sp_solvent_name):
                    raise ValueError(
                        "Single-point config must define 'solvent_model' when 'solvent' is set."
                    )
                if sp_solvent_model and sp_solvent_model.lower() not in ("pcm", "smd"):
                    raise ValueError("Single-point config 'solvent_model' must be one of: pcm, smd.")

            sp_eps = None
            sp_solvent_key = None
            if sp_solvent_name:
                sp_solvent_key = sp_solvent_name.lower()
                if not _is_vacuum_solvent(sp_solvent_key):
                    sp_solvent_model = sp_solvent_model.lower() if sp_solvent_model else None
                    if sp_solvent_model == "pcm":
                        sp_solvent_map = load_solvent_map(sp_solvent_map_path)
                        sp_eps = sp_solvent_map.get(sp_solvent_key)
                        if sp_eps is None:
                            available = ", ".join(sorted(sp_solvent_map.keys()))
                            raise ValueError(
                                "Single-point solvent '{name}' not found in solvent map {path}. "
                                "Available solvents: {available}.".format(
                                    name=sp_solvent_name,
                                    path=sp_solvent_map_path,
                                    available=available,
                                )
                            )
            sp_dispersion_model = _normalize_dispersion_settings(
                "Single-point",
                sp_xc,
                sp_dispersion_model,
                allow_dispersion=True,
            )

            sp_solvent_name, sp_solvent_model = _normalize_solvent_settings(
                "Single-point",
                sp_solvent_name,
                sp_solvent_model,
            )

            frequency_config = config.get("frequency") or config.get("freq") or {}
            if not isinstance(frequency_config, dict):
                raise ValueError("Config 'frequency' (or 'freq') must be an object.")
            freq_dispersion_mode = _normalize_frequency_dispersion_mode(
                frequency_config.get("dispersion")
            )

            freq_dispersion_model = _normalize_dispersion_settings(
                "Frequency",
                sp_xc,
                sp_dispersion_model,
                allow_dispersion=True,
            )

            if calculation_mode != "optimization":
                calc_basis = sp_basis
                calc_xc = sp_xc
                calc_scf_config = sp_scf_config
                calc_solvent_name = sp_solvent_name
                calc_solvent_model = sp_solvent_model
                calc_solvent_map_path = sp_solvent_map_path
                calc_eps = sp_eps
                calc_dispersion_model = (
                    sp_dispersion_model if calculation_mode == "single_point" else freq_dispersion_model
                )
                calc_ks_type = select_ks_type(
                    mol=mol,
                    scf_config=calc_scf_config,
                    optimizer_mode=optimizer_mode,
                    multiplicity=multiplicity,
                )
                logging.info(
                    "Running capability check for %s calculation (SCF%s)...",
                    "single-point" if calculation_mode == "single_point" else "frequency",
                    " + Hessian" if calculation_mode == "frequency" else "",
                )
                run_capability_check(
                    mol,
                    calc_basis,
                    calc_xc,
                    calc_scf_config,
                    calc_solvent_model if calc_solvent_name else None,
                    calc_solvent_name,
                    calc_eps,
                    calc_dispersion_model if calculation_mode == "frequency" else None,
                    freq_dispersion_mode if calculation_mode == "frequency" else "none",
                    require_hessian=calculation_mode == "frequency",
                    verbose=verbose,
                    memory_mb=memory_mb,
                    optimizer_mode=optimizer_mode,
                    multiplicity=multiplicity,
                )
                if calculation_mode == "single_point":
                    logging.info("Starting single-point energy calculation...")
                else:
                    logging.info("Starting frequency calculation...")
                logging.info("Run ID: %s", run_id)
                logging.info("Run directory: %s", run_dir)
                if thread_count:
                    logging.info("Using threads: %s", thread_count)
                    if openmp_available is False:
                        effective_display = (
                            str(effective_threads) if effective_threads is not None else "unknown"
                        )
                        logging.warning(
                            "OpenMP appears unavailable; requested threads may have no effect "
                            "(effective threads: %s).",
                            effective_display,
                        )
                if memory_gb:
                    logging.info("Memory target: %s GB (PySCF max_memory)", memory_gb)
                    if memory_limit_status:
                        if memory_limit_status["applied"]:
                            limit_gb = memory_limit_status["limit_bytes"] / (1024 ** 3)
                            limit_name = memory_limit_status["limit_name"] or "unknown"
                            logging.info(
                                "OS hard memory limit: applied at %.2f GB (%s)",
                                limit_gb,
                                limit_name,
                            )
                        else:
                            log_fn = logging.warning
                            if memory_limit_status["reason"] == "disabled by config":
                                log_fn = logging.info
                            log_fn(
                                "OS hard memory limit: not applied (%s).",
                                memory_limit_status["reason"],
                            )
                logging.info("Verbose logging: %s", "enabled" if verbose else "disabled")
                logging.info("Log file: %s", log_path)
                if event_log_path:
                    logging.info("Event log file: %s", event_log_path)
                if calc_scf_config:
                    logging.info("SCF settings: %s", calc_scf_config)
                if calc_dispersion_model:
                    logging.info("Dispersion correction: %s", calc_dispersion_model)
                if calculation_mode == "frequency":
                    logging.info("Frequency dispersion mode: %s", freq_dispersion_mode)
                if calc_xc != xc:
                    logging.info("XC override: %s", calc_xc)
                if calc_basis != basis:
                    logging.info("Basis override: %s", calc_basis)
                if calc_scf_config != scf_config:
                    logging.info("SCF override: %s", calc_scf_config)
                if calc_solvent_name != solvent_name or calc_solvent_model != solvent_model:
                    logging.info(
                        "Solvent override: %s (%s)",
                        calc_solvent_name,
                        calc_solvent_model,
                    )

                run_start = time.perf_counter()
                calculation_metadata = {
                    "status": "running",
                    "run_directory": run_dir,
                    "run_started_at": datetime.now().isoformat(),
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "xyz_file": args.xyz_file,
                    "xyz_file_hash": compute_file_hash(args.xyz_file),
                    "basis": calc_basis,
                    "xc": calc_xc,
                    "solvent": calc_solvent_name,
                    "solvent_model": calc_solvent_model if calc_solvent_name else None,
                    "solvent_eps": calc_eps,
                    "solvent_map": calc_solvent_map_path,
                    "dispersion": calc_dispersion_model,
                    "frequency_dispersion_mode": freq_dispersion_mode
                    if calculation_mode == "frequency"
                    else None,
                    "dispersion_info": dispersion_info,
                    "single_point": {
                        "basis": calc_basis,
                        "xc": calc_xc,
                        "scf": calc_scf_config,
                        "solvent": calc_solvent_name,
                        "solvent_model": calc_solvent_model if calc_solvent_name else None,
                        "solvent_eps": calc_eps,
                        "solvent_map": calc_solvent_map_path,
                        "dispersion": calc_dispersion_model,
                        "frequency_dispersion_mode": freq_dispersion_mode
                        if calculation_mode == "frequency"
                        else None,
                    },
                    "single_point_enabled": calculation_mode == "single_point",
                    "calculation_mode": calculation_mode,
                    "charge": charge,
                    "spin": spin,
                    "multiplicity": multiplicity,
                    "ks_type": calc_ks_type,
                    "thread_count": thread_count,
                    "effective_thread_count": effective_threads,
                    "openmp_available": openmp_available,
                    "memory_gb": memory_gb,
                    "memory_mb": memory_mb,
                    "memory_limit_status": memory_limit_status,
                    "log_file": log_path,
                    "event_log_file": event_log_path,
                    "frequency_file": frequency_output_path,
                    "run_metadata_file": run_metadata_path,
                    "config_file": args.config,
                    "config": config,
                    "config_raw": config_raw,
                    "config_hash": compute_text_hash(config_raw),
                    "scf_config": calc_scf_config,
                    "scf_settings": calc_scf_config,
                    "environment": collect_environment_snapshot(thread_count),
                    "git": collect_git_metadata(os.getcwd()),
                    "versions": {
                        "ase": get_package_version("ase"),
                        "pyscf": get_package_version("pyscf"),
                        "dftd3": get_package_version("dftd3"),
                        "dftd4": get_package_version("dftd4"),
                    },
                }
                calculation_metadata["run_updated_at"] = datetime.now().isoformat()
                write_run_metadata(run_metadata_path, calculation_metadata)
                _record_status_event(
                    event_log_path,
                    run_id,
                    run_dir,
                    "running",
                    previous_status=None,
                )

                try:
                    if calculation_mode == "single_point":
                        sp_result = compute_single_point_energy(
                            mol,
                            calc_basis,
                            calc_xc,
                            calc_scf_config,
                            calc_solvent_model if calc_solvent_name else None,
                            calc_solvent_name,
                            calc_eps,
                            calc_dispersion_model,
                            verbose,
                            memory_mb,
                            optimizer_mode=optimizer_mode,
                            multiplicity=multiplicity,
                            log_override=False,
                        )
                        calculation_metadata["dispersion_info"] = sp_result.get("dispersion")
                        energy = sp_result.get("energy")
                        sp_converged = sp_result.get("converged")
                        sp_cycles = sp_result.get("cycles")
                        summary = {
                            "elapsed_seconds": time.perf_counter() - run_start,
                            "n_steps": sp_cycles,
                            "final_energy": energy,
                            "opt_final_energy": energy,
                            "final_sp_energy": energy,
                            "final_sp_converged": sp_converged,
                            "final_sp_cycles": sp_cycles,
                            "scf_converged": sp_converged,
                            "opt_converged": None,
                            "converged": bool(sp_converged)
                            if sp_converged is not None
                            else True,
                        }
                        calculation_metadata["summary"] = summary
                        calculation_metadata["summary"]["memory_limit_enforced"] = (
                            memory_limit_enforced
                        )
                    else:
                        frequency_result = compute_frequencies(
                            mol,
                            calc_basis,
                            calc_xc,
                            calc_scf_config,
                            calc_solvent_model if calc_solvent_name else None,
                            calc_solvent_name,
                            calc_eps,
                            calc_dispersion_model,
                            freq_dispersion_mode,
                            verbose,
                            memory_mb,
                            optimizer_mode=optimizer_mode,
                            multiplicity=multiplicity,
                            log_override=False,
                        )
                        imaginary_check = frequency_result.get("imaginary_check") or {}
                        imaginary_status = imaginary_check.get("status")
                        imaginary_message = imaginary_check.get("message")
                        if imaginary_message:
                            if imaginary_status == "one_imaginary":
                                logging.info("Imaginary frequency check: %s", imaginary_message)
                            else:
                                logging.warning("Imaginary frequency check: %s", imaginary_message)
                        frequency_payload = {
                            "status": "completed",
                            "output_file": frequency_output_path,
                            "units": _frequency_units(),
                            "versions": _frequency_versions(),
                            "basis": calc_basis,
                            "xc": calc_xc,
                            "scf": calc_scf_config,
                            "solvent": calc_solvent_name,
                            "solvent_model": calc_solvent_model if calc_solvent_name else None,
                            "solvent_eps": calc_eps,
                            "dispersion": calc_dispersion_model,
                            "dispersion_mode": freq_dispersion_mode,
                            "results": frequency_result,
                        }
                        with open(frequency_output_path, "w", encoding="utf-8") as handle:
                            json.dump(frequency_payload, handle, indent=2)
                        calculation_metadata["frequency"] = frequency_payload
                        calculation_metadata["dispersion_info"] = frequency_result.get("dispersion")
                        energy = frequency_result.get("energy")
                        sp_converged = frequency_result.get("converged")
                        sp_cycles = frequency_result.get("cycles")
                        summary = {
                            "elapsed_seconds": time.perf_counter() - run_start,
                            "n_steps": sp_cycles,
                            "final_energy": energy,
                            "opt_final_energy": energy,
                            "final_sp_energy": energy,
                            "final_sp_converged": sp_converged,
                            "final_sp_cycles": sp_cycles,
                            "scf_converged": sp_converged,
                            "opt_converged": None,
                            "converged": bool(sp_converged)
                            if sp_converged is not None
                            else True,
                        }
                        calculation_metadata["summary"] = summary
                        calculation_metadata["summary"]["memory_limit_enforced"] = (
                            memory_limit_enforced
                        )

                    calculation_metadata["status"] = "completed"
                    calculation_metadata["run_ended_at"] = datetime.now().isoformat()
                    calculation_metadata["run_updated_at"] = datetime.now().isoformat()
                    write_run_metadata(run_metadata_path, calculation_metadata)
                    _record_status_event(
                        event_log_path,
                        run_id,
                        run_dir,
                        "completed",
                        previous_status="running",
                    )
                    _update_foreground_queue("completed", exit_code=0)
                except Exception as exc:
                    logging.exception("Calculation failed.")
                    calculation_metadata["status"] = "failed"
                    calculation_metadata["run_ended_at"] = datetime.now().isoformat()
                    calculation_metadata["run_updated_at"] = datetime.now().isoformat()
                    calculation_metadata["error"] = str(exc)
                    calculation_metadata["traceback"] = traceback.format_exc()
                    write_run_metadata(run_metadata_path, calculation_metadata)
                    _record_status_event(
                        event_log_path,
                        run_id,
                        run_dir,
                        "failed",
                        previous_status="running",
                        details={"error": str(exc)},
                    )
                    _update_foreground_queue("failed", exit_code=1)
                    raise
                return

            if calculation_mode == "optimization":
                logging.info("Running capability check for geometry optimization (SCF + gradient)...")
                run_capability_check(
                    mol,
                    basis,
                    xc,
                    scf_config,
                    solvent_model,
                    solvent_name,
                    eps,
                    None,
                    "none",
                    require_hessian=False,
                    verbose=verbose,
                    memory_mb=memory_mb,
                    optimizer_mode=optimizer_mode,
                    multiplicity=multiplicity,
                )
                if frequency_enabled:
                    logging.info(
                        "Running capability check for frequency calculation (SCF + gradient + Hessian)..."
                    )
                    run_capability_check(
                        mol,
                        sp_basis,
                        sp_xc,
                        sp_scf_config,
                        sp_solvent_model if sp_solvent_name else None,
                        sp_solvent_name,
                        sp_eps,
                        freq_dispersion_model,
                        freq_dispersion_mode,
                        require_hessian=True,
                        verbose=verbose,
                        memory_mb=memory_mb,
                        optimizer_mode=optimizer_mode,
                        multiplicity=multiplicity,
                    )

                logging.info("Starting geometry optimization...")
                logging.info("Run ID: %s", run_id)
                logging.info("Run directory: %s", run_dir)
                logging.info("Optimization mode: %s", optimizer_mode)
                if thread_count:
                    logging.info("Using threads: %s", thread_count)
                    if openmp_available is False:
                        effective_display = (
                            str(effective_threads) if effective_threads is not None else "unknown"
                        )
                        logging.warning(
                            "OpenMP appears unavailable; requested threads may have no effect "
                            "(effective threads: %s).",
                            effective_display,
                        )
                if memory_gb:
                    logging.info("Memory target: %s GB (PySCF max_memory)", memory_gb)
                    if memory_limit_status:
                        if memory_limit_status["applied"]:
                            limit_gb = memory_limit_status["limit_bytes"] / (1024 ** 3)
                            limit_name = memory_limit_status["limit_name"] or "unknown"
                            logging.info(
                                "OS hard memory limit: applied at %.2f GB (%s)",
                                limit_gb,
                                limit_name,
                            )
                        else:
                            log_fn = logging.warning
                            if memory_limit_status["reason"] == "disabled by config":
                                log_fn = logging.info
                            log_fn(
                                "OS hard memory limit: not applied (%s).",
                                memory_limit_status["reason"],
                            )
                logging.info("Verbose logging: %s", "enabled" if verbose else "disabled")
                logging.info("Log file: %s", log_path)
                if event_log_path:
                    logging.info("Event log file: %s", event_log_path)
                if applied_scf:
                    logging.info("SCF settings: %s", applied_scf)
                if dispersion_model:
                    logging.info("Dispersion correction: %s", dispersion_model)
                    if dispersion_info:
                        logging.info("Dispersion details: %s", dispersion_info)
                if sp_xc != xc:
                    logging.info("Single-point XC override: %s", sp_xc)
                if sp_basis != basis:
                    logging.info("Single-point basis override: %s", sp_basis)
                if sp_scf_config != scf_config:
                    logging.info("Single-point SCF override: %s", sp_scf_config)
                if sp_solvent_name != solvent_name or sp_solvent_model != solvent_model:
                    logging.info(
                        "Single-point solvent override: %s (%s)",
                        sp_solvent_name,
                        sp_solvent_model,
                    )
                if sp_dispersion_model is not None and sp_dispersion_model != dispersion_model:
                    logging.info("Single-point dispersion override: %s", sp_dispersion_model)
                if frequency_enabled:
                    logging.info("Frequency dispersion mode: %s", freq_dispersion_mode)
                run_start = time.perf_counter()
                optimization_metadata = {
                    "status": "running",
                    "run_directory": run_dir,
                    "run_started_at": datetime.now().isoformat(),
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "xyz_file": args.xyz_file,
                    "xyz_file_hash": compute_file_hash(args.xyz_file),
                    "basis": basis,
                    "xc": xc,
                    "solvent": solvent_name,
                    "solvent_model": solvent_model if solvent_name else None,
                    "solvent_eps": eps,
                    "solvent_map": solvent_map_path,
                    "dispersion": dispersion_model,
                    "dispersion_info": dispersion_info,
                    "frequency_dispersion_mode": freq_dispersion_mode if frequency_enabled else None,
                    "optimizer": {
                        "mode": optimizer_mode,
                        "output_xyz": optimizer_config.get("output_xyz"),
                        "ase": optimizer_config.get("ase"),
                    },
                    "single_point": {
                        "basis": sp_basis,
                        "xc": sp_xc,
                        "scf": sp_scf_config,
                        "solvent": sp_solvent_name,
                        "solvent_model": sp_solvent_model if sp_solvent_name else None,
                        "solvent_eps": sp_eps,
                        "solvent_map": sp_solvent_map_path,
                        "dispersion": sp_dispersion_model,
                        "frequency_dispersion_mode": freq_dispersion_mode
                        if frequency_enabled
                        else None,
                    },
                    "single_point_enabled": single_point_enabled,
                    "frequency_enabled": frequency_enabled,
                    "calculation_mode": calculation_mode,
                    "charge": charge,
                    "spin": spin,
                    "multiplicity": multiplicity,
                    "ks_type": ks_type,
                    "thread_count": thread_count,
                    "effective_thread_count": effective_threads,
                    "openmp_available": openmp_available,
                    "memory_gb": memory_gb,
                    "memory_mb": memory_mb,
                    "memory_limit_status": memory_limit_status,
                    "log_file": log_path,
                    "event_log_file": event_log_path,
                    "optimized_xyz_file": optimized_xyz_path,
                    "frequency_file": frequency_output_path,
                    "run_metadata_file": run_metadata_path,
                    "config_file": args.config,
                    "config": config,
                    "config_raw": config_raw,
                    "config_hash": compute_text_hash(config_raw),
                    "scf_config": scf_config,
                    "scf_settings": applied_scf,
                    "environment": collect_environment_snapshot(thread_count),
                    "git": collect_git_metadata(os.getcwd()),
                    "versions": {
                        "ase": get_package_version("ase"),
                        "pyscf": get_package_version("pyscf"),
                        "dftd3": get_package_version("dftd3"),
                        "dftd4": get_package_version("dftd4"),
                    },
                }
                optimization_metadata["run_updated_at"] = datetime.now().isoformat()
                n_steps = {"value": 0}
                n_steps_source = None

                write_run_metadata(run_metadata_path, optimization_metadata)
                _record_status_event(
                    event_log_path,
                    run_id,
                    run_dir,
                    "running",
                    previous_status=None,
                )

                last_metadata_write = {
                    "time": time.monotonic(),
                    "step": 0,
                }

            def _step_callback(*_args, **_kwargs):
                n_steps["value"] += 1
                step_value = n_steps["value"]
                now = time.monotonic()
                should_write = (step_value - last_metadata_write["step"] >= 5) or (
                    now - last_metadata_write["time"] >= 5.0
                )
                if should_write:
                    optimization_metadata["n_steps"] = step_value
                    optimization_metadata["n_steps_source"] = "ase"
                    optimization_metadata["status"] = "running"
                    optimization_metadata["run_updated_at"] = datetime.now().isoformat()
                    write_run_metadata(run_metadata_path, optimization_metadata)
                    last_metadata_write["time"] = now
                    last_metadata_write["step"] = step_value

            try:
                input_xyz_name = os.path.basename(args.xyz_file)
                input_xyz_path = resolve_run_path(run_dir, input_xyz_name)
                if os.path.abspath(args.xyz_file) != os.path.abspath(input_xyz_path):
                    shutil.copy2(args.xyz_file, input_xyz_path)
                output_xyz_setting = optimizer_config.get("output_xyz") or "ase_optimized.xyz"
                output_xyz_path = resolve_run_path(run_dir, output_xyz_setting)
                ensure_parent_dir(output_xyz_path)
                n_steps_value = _run_ase_optimizer(
                    input_xyz_path,
                    output_xyz_path,
                    run_dir,
                    charge,
                    spin,
                    multiplicity,
                    basis,
                    xc,
                    scf_config,
                    solvent_model.lower() if solvent_model else None,
                    solvent_name,
                    eps,
                    dispersion_model,
                    verbose,
                    memory_mb,
                    optimizer_config.get("ase") or {},
                    optimizer_mode,
                    step_callback=_step_callback,
                )
                optimized_atom_spec, _, _, _ = load_xyz(output_xyz_path)
                mol_optimized = gto.M(
                    atom=optimized_atom_spec,
                    basis=basis,
                    charge=charge,
                    spin=spin,
                )
                if memory_mb:
                    mol_optimized.max_memory = memory_mb
                if n_steps_value is not None:
                    n_steps["value"] = n_steps_value
                n_steps_source = "ase"
            except Exception as exc:
                logging.exception("Geometry optimization failed.")
                n_steps_value = n_steps["value"] if n_steps_source else None
                optimization_metadata["status"] = "failed"
                optimization_metadata["run_ended_at"] = datetime.now().isoformat()
                optimization_metadata["error"] = str(exc)
                optimization_metadata["traceback"] = traceback.format_exc()
                optimization_metadata["last_geometry_source"] = "mf.mol"
                optimization_metadata["n_steps"] = n_steps_value
                optimization_metadata["n_steps_source"] = n_steps_source
                elapsed_seconds = time.perf_counter() - run_start
                optimization_metadata["summary"] = build_run_summary(
                    mf,
                    getattr(mf, "mol", mol),
                    elapsed_seconds,
                    completed=False,
                    n_steps=n_steps_value,
                    final_sp_energy=None,
                    final_sp_converged=None,
                    final_sp_cycles=None,
                )
                optimization_metadata["summary"]["memory_limit_enforced"] = memory_limit_enforced
                write_run_metadata(run_metadata_path, optimization_metadata)
                _record_status_event(
                    event_log_path,
                    run_id,
                    run_dir,
                    "failed",
                    previous_status="running",
                    details={"error": str(exc)},
                )
                _update_foreground_queue("failed", exit_code=1)
                raise

            logging.info("Optimization finished.")
            logging.info("Optimized geometry (in Angstrom):")
            logging.info("%s", mol_optimized.tostring(format="xyz"))
            write_optimized_xyz(optimized_xyz_path, mol_optimized)
            ensure_stream_newlines()
            final_sp_energy = None
            final_sp_converged = None
            final_sp_cycles = None
            optimization_metadata["status"] = "completed"
            optimization_metadata["run_ended_at"] = datetime.now().isoformat()
            elapsed_seconds = time.perf_counter() - run_start
            n_steps_value = n_steps["value"] if n_steps_source else None
            imaginary_count = None
            frequency_payload = None
            if frequency_enabled:
                logging.info("Calculating harmonic frequencies for optimized geometry...")
                try:
                    frequency_result = compute_frequencies(
                        mol_optimized,
                        sp_basis,
                        sp_xc,
                        sp_scf_config,
                        sp_solvent_model if sp_solvent_name else None,
                        sp_solvent_name,
                        sp_eps,
                        freq_dispersion_model,
                        freq_dispersion_mode,
                        verbose,
                        memory_mb,
                        optimizer_mode=optimizer_mode,
                        multiplicity=multiplicity,
                    )
                    imaginary_count = frequency_result.get("imaginary_count")
                    imaginary_check = frequency_result.get("imaginary_check") or {}
                    imaginary_status = imaginary_check.get("status")
                    imaginary_message = imaginary_check.get("message")
                    if imaginary_message:
                        if imaginary_status == "one_imaginary":
                            logging.info("Imaginary frequency check: %s", imaginary_message)
                        else:
                            logging.warning("Imaginary frequency check: %s", imaginary_message)
                    frequency_payload = {
                        "status": "completed",
                        "output_file": frequency_output_path,
                        "units": _frequency_units(),
                        "versions": _frequency_versions(),
                        "basis": sp_basis,
                        "xc": sp_xc,
                        "scf": sp_scf_config,
                        "solvent": sp_solvent_name,
                        "solvent_model": sp_solvent_model if sp_solvent_name else None,
                        "solvent_eps": sp_eps,
                        "dispersion": freq_dispersion_model,
                        "dispersion_mode": freq_dispersion_mode,
                        "results": frequency_result,
                    }
                    with open(frequency_output_path, "w", encoding="utf-8") as handle:
                        json.dump(frequency_payload, handle, indent=2)
                    optimization_metadata["frequency"] = frequency_payload
                except Exception as exc:
                    logging.exception("Frequency calculation failed.")
                    failure_reason = str(exc) or "Frequency calculation failed."
                    frequency_payload = {
                        "status": "failed",
                        "output_file": frequency_output_path,
                        "reason": failure_reason,
                        "units": _frequency_units(),
                        "versions": _frequency_versions(),
                        "basis": sp_basis,
                        "xc": sp_xc,
                        "scf": sp_scf_config,
                        "solvent": sp_solvent_name,
                        "solvent_model": sp_solvent_model if sp_solvent_name else None,
                        "solvent_eps": sp_eps,
                        "dispersion": freq_dispersion_model,
                        "dispersion_mode": freq_dispersion_mode,
                        "results": None,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    with open(frequency_output_path, "w", encoding="utf-8") as handle:
                        json.dump(frequency_payload, handle, indent=2)
                    optimization_metadata["frequency"] = frequency_payload
            else:
                frequency_payload = {
                    "status": "skipped",
                    "output_file": frequency_output_path,
                    "reason": "Frequency calculation disabled.",
                    "units": _frequency_units(),
                    "versions": _frequency_versions(),
                    "results": None,
                }
                with open(frequency_output_path, "w", encoding="utf-8") as handle:
                    json.dump(frequency_payload, handle, indent=2)
                optimization_metadata["frequency"] = frequency_payload
            run_single_point = False
            sp_status = "skipped"
            sp_skip_reason = None
            if single_point_enabled:
                if frequency_enabled:
                    expected_imaginary = 1 if optimizer_mode == "transition_state" else 0
                    if imaginary_count is None:
                        logging.warning(
                            "Skipping single-point calculation because imaginary frequency "
                            "count is unavailable."
                        )
                        sp_skip_reason = "Imaginary frequency count unavailable."
                    elif imaginary_count == expected_imaginary:
                        run_single_point = True
                    else:
                        logging.warning(
                            "Skipping single-point calculation because imaginary frequency "
                            "count %s does not match expected %s.",
                            imaginary_count,
                            expected_imaginary,
                        )
                        sp_skip_reason = (
                            "Imaginary frequency count does not match expected "
                            f"{expected_imaginary}."
                        )
                else:
                    run_single_point = True
            else:
                logging.info("Skipping single-point energy calculation (disabled).")
                sp_skip_reason = "Single-point calculation disabled."

            if run_single_point:
                sp_status = "executed"
                sp_skip_reason = None

            optimization_metadata["single_point"]["status"] = sp_status
            optimization_metadata["single_point"]["skip_reason"] = sp_skip_reason
            if frequency_payload is not None:
                frequency_payload["single_point"] = {
                    "status": sp_status,
                    "skip_reason": sp_skip_reason,
                }
                with open(frequency_output_path, "w", encoding="utf-8") as handle:
                    json.dump(frequency_payload, handle, indent=2)

            try:
                if run_single_point:
                    logging.info("Calculating single-point energy for optimized geometry...")
                    sp_result = compute_single_point_energy(
                        mol_optimized,
                        sp_basis,
                        sp_xc,
                        sp_scf_config,
                        sp_solvent_model if sp_solvent_name else None,
                        sp_solvent_name,
                        sp_eps,
                        freq_dispersion_model,
                        verbose,
                        memory_mb,
                        optimizer_mode=optimizer_mode,
                        multiplicity=multiplicity,
                    )
                    final_sp_energy = sp_result["energy"]
                    final_sp_converged = sp_result["converged"]
                    final_sp_cycles = sp_result["cycles"]
                    optimization_metadata["single_point"]["dispersion_info"] = sp_result.get(
                        "dispersion"
                    )
                    if final_sp_cycles is None:
                        final_sp_cycles = parse_single_point_cycle_count(log_path)
                elif single_point_enabled:
                    logging.info("Skipping single-point energy calculation.")
            except Exception:
                logging.exception("Single-point energy calculation failed.")
                if run_single_point:
                    final_sp_energy = None
                    final_sp_converged = None
                    final_sp_cycles = parse_single_point_cycle_count(log_path)
            optimization_metadata["n_steps"] = n_steps_value
            optimization_metadata["n_steps_source"] = n_steps_source
            optimization_metadata["summary"] = build_run_summary(
                mf,
                mol_optimized,
                elapsed_seconds,
                completed=True,
                n_steps=n_steps_value,
                final_sp_energy=final_sp_energy,
                final_sp_converged=final_sp_converged,
                final_sp_cycles=final_sp_cycles,
            )
            optimization_metadata["summary"]["memory_limit_enforced"] = memory_limit_enforced
            write_run_metadata(run_metadata_path, optimization_metadata)
            _record_status_event(
                event_log_path,
                run_id,
                run_dir,
                "completed",
                previous_status="running",
            )
            _update_foreground_queue("completed", exit_code=0)
        finally:
            ensure_stream_newlines()
    except Exception:
        logging.exception("Run failed.")
        if "queue_tracking" in locals() and queue_tracking:
            _update_foreground_queue("failed", exit_code=1)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        message = str(error)
        if message:
            print(message, file=sys.stderr)
        sys.exit(1)
