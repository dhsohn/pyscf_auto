import json
from pathlib import Path

from run_opt_engine import normalize_xc_functional
from run_opt_config import DEFAULT_SOLVENT_MAP_PATH, load_run_config, load_solvent_map


_REPO_ROOT = Path(__file__).resolve().parents[1]
INTERACTIVE_CONFIG = _REPO_ROOT / "run_config.json"
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
SOLVENT_MODEL_OPTIONS = ["pcm", "none (vacuum)"]
DISPERSION_MODEL_OPTIONS = ["none (disabled)", "d3bj", "d3zero", "d4"]
CALCULATION_MODE_OPTIONS = [
    "Geometry optimization",
    "Single-point energy",
    "Frequency analysis",
    "IRC calculation",
    "Scan calculation",
]


def _resolve_interactive_input_dir() -> Path | None:
    preferred_paths = [
        _REPO_ROOT / "input",
        Path(__file__).resolve().parent / "input",
    ]
    for path in preferred_paths:
        if path.is_dir():
            return path
    return None


def _prompt_choice(prompt, options, allow_custom=False, default_value=None):
    normalized_options = list(dict.fromkeys(options))
    if default_value and default_value not in normalized_options:
        normalized_options.insert(0, default_value)
    while True:
        print(prompt)
        for index, option in enumerate(normalized_options, start=1):
            print(f"{index}. {option}")
        if allow_custom:
            print("0. Enter custom value")
        choice = input("> ").strip()
        if not choice and default_value:
            return default_value
        if allow_custom and choice == "0":
            custom = input("Enter a value: ").strip()
            if custom:
                return custom
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(normalized_options):
                return normalized_options[index - 1]
        print("Enter a valid number.")


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
        print("Enter 'y' or 'n'.")


def _prompt_dispersion(stage_label, default_value=None):
    default_label = (
        DISPERSION_MODEL_OPTIONS[0] if default_value is None else default_value
    )
    choice = _prompt_choice(
        f"Select dispersion correction for {stage_label}:",
        DISPERSION_MODEL_OPTIONS,
        default_value=default_label,
    )
    if choice.strip().lower().startswith("none"):
        return None
    return choice


def _prompt_int_list(prompt, count):
    while True:
        raw = input(f"{prompt} (e.g., 0,1)\n> ").strip()
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) != count:
            print(f"Enter {count} indices.")
            continue
        try:
            values = [int(part) for part in parts]
        except ValueError:
            print("Enter integer indices.")
            continue
        if any(value < 0 for value in values):
            print("Indices must be non-negative integers.")
            continue
        return values


def _prompt_float(prompt):
    while True:
        raw = input(f"{prompt}\n> ").strip()
        try:
            return float(raw)
        except ValueError:
            print("Enter a number.")


def _prompt_float_list(prompt):
    while True:
        raw = input(f"{prompt} (e.g., 1.0,2.5,3.2)\n> ").strip()
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            print("Enter at least one number.")
            continue
        try:
            return [float(part) for part in parts]
        except ValueError:
            print("Enter the list of numbers in a valid format.")


def _prompt_scan_dimension():
    scan_type = _prompt_choice(
        "Select scan type:",
        ["bond", "angle", "dihedral"],
    )
    if scan_type == "bond":
        indices = _prompt_int_list("Enter bond indices i,j", 2)
    elif scan_type == "angle":
        indices = _prompt_int_list("Enter angle indices i,j,k", 3)
    else:
        indices = _prompt_int_list("Enter dihedral indices i,j,k,l", 4)
    start = _prompt_float("Enter start value")
    end = _prompt_float("Enter end value")
    step = _prompt_float("Enter step value")
    dimension = {"type": scan_type, "start": start, "end": end, "step": step}
    for key, value in zip(("i", "j", "k", "l"), indices, strict=False):
        dimension[key] = value
    return dimension


def _prompt_interactive_config(args):
    calculation_choice = _prompt_choice(
        "Which calculation would you like to run?",
        CALCULATION_MODE_OPTIONS,
    )
    calculation_mode = {
        "Geometry optimization": "optimization",
        "Single-point energy": "single_point",
        "Frequency analysis": "frequency",
        "IRC calculation": "irc",
        "Scan calculation": "scan",
    }[calculation_choice]
    optimization_choice = None
    scan_config = None
    if calculation_mode == "optimization":
        optimization_choice = _prompt_choice(
            "Select optimization type:",
            ["Minimum optimization", "Transition-state optimization"],
        )
    if calculation_mode == "scan":
        scan_mode_choice = _prompt_choice(
            "Select scan mode:",
            ["Optimization scan", "Single-point scan"],
        )
        scan_mode = (
            "optimization"
            if scan_mode_choice == "Optimization scan"
            else "single_point"
        )
        dimension_count_choice = _prompt_choice(
            "Select scan dimension:",
            ["1D", "2D"],
        )
        dimension_count = 1 if dimension_count_choice == "1D" else 2
        dimensions = [_prompt_scan_dimension() for _ in range(dimension_count)]
        grid_values = None
        if _prompt_yes_no("Enter grid values manually for each dimension?", default=False):
            grid_values = []
            for index in range(dimension_count):
                grid_values.append(
                    _prompt_float_list(
                        f"Enter grid values for dimension {index + 1}"
                    )
                )
        if dimension_count == 1:
            scan_config = dimensions[0]
            scan_config["mode"] = scan_mode
        else:
            scan_config = {"dimensions": dimensions, "mode": scan_mode}
        if grid_values is not None:
            scan_config["grid"] = grid_values
    base_config_path = INTERACTIVE_CONFIG

    config_filename = base_config_path.name
    base_config_path = _REPO_ROOT / config_filename
    if not base_config_path.is_file():
        raise FileNotFoundError(
            "Interactive base config file not found. "
            f"Expected file at: {base_config_path}"
        )

    config, _ = load_run_config(base_config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Failed to load base config from {base_config_path}.")
    if optimization_choice == "Transition-state optimization":
        optimizer_config = config.setdefault("optimizer", {})
        optimizer_config["mode"] = "transition_state"
        optimizer_config.setdefault("output_xyz", "ts_optimized.xyz")
        ase_config = optimizer_config.setdefault("ase", {})
        ase_config["optimizer"] = "sella"
        ase_config.setdefault("fmax", 0.05)
        ase_config.setdefault("steps", 200)
        ase_config.setdefault("trajectory", "ts_opt.traj")
        ase_config.setdefault("logfile", "ts_opt.log")
        ase_config.setdefault("sella", {"order": 1})

    constraints_default = config.get("constraints")
    if constraints_default:
        constraints_hint = json.dumps(constraints_default, ensure_ascii=False)
    else:
        constraints_hint = ""
    constraints_prompt = (
        "Enter constraints as JSON (e.g., "
        "{\"bonds\":[{\"i\":0,\"j\":1,\"length\":1.10}]})"
    )
    if constraints_hint:
        constraints_prompt += f" [default: {constraints_hint}]"
    constraints_input = input(f"{constraints_prompt}\n> ").strip()
    if not constraints_input and constraints_default is not None:
        constraints = constraints_default
    elif not constraints_input:
        constraints = None
    elif constraints_input.lower() in ("none", "null", "no"):
        constraints = None
    else:
        try:
            constraints = json.loads(constraints_input)
        except json.JSONDecodeError as exc:
            raise ValueError("constraints input is not valid JSON.") from exc
        if not isinstance(constraints, dict):
            raise ValueError("constraints input must be a JSON object.")

    if not args.xyz_file:
        input_dir = _resolve_interactive_input_dir()
        if input_dir is None:
            expected_repo = _REPO_ROOT / "input"
            expected_core = Path(__file__).resolve().parent / "input"
            expected_paths = [
                str(path) for path in (expected_repo, expected_core) if path is not None
            ]
            hint = ", ".join(expected_paths) if expected_paths else "input/"
            raise FileNotFoundError(
                "Interactive input directory not found. "
                f"Expected an input directory at: {hint}"
            )
        input_xyz_files = sorted(
            path.name
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".xyz"
        )
        if not input_xyz_files:
            raise ValueError("No .xyz files found in the input directory.")
        selected_xyz = _prompt_choice(
            "Select an input file (.xyz):",
            input_xyz_files,
        )
        args.xyz_file = str(input_dir / selected_xyz)

    basis = _prompt_choice(
        "Select a basis set:",
        BASIS_SET_OPTIONS,
        allow_custom=True,
        default_value=config.get("basis"),
    )
    xc = _prompt_choice(
        "Select an XC functional:",
        XC_FUNCTIONAL_OPTIONS,
        allow_custom=True,
        default_value=config.get("xc"),
    )
    xc = normalize_xc_functional(xc)
    base_dispersion = None
    if calculation_mode in ("optimization", "single_point", "frequency"):
        base_dispersion = _prompt_dispersion(
            {
                "optimization": "geometry optimization",
                "single_point": "single-point calculation",
                "frequency": "frequency analysis",
            }[calculation_mode],
            config.get("dispersion"),
        )
    default_solvent_model = config.get("solvent_model")
    if isinstance(default_solvent_model, str) and default_solvent_model.lower() == "smd":
        default_solvent_model = None
    solvent_model = _prompt_choice(
        "Select a solvent model:",
        SOLVENT_MODEL_OPTIONS,
        default_value=default_solvent_model,
    )
    if isinstance(solvent_model, str) and solvent_model.lower() in (
        "none",
        "none (vacuum)",
        "vacuum",
    ):
        solvent_model = None
    if solvent_model is None:
        solvent = "vacuum"
    else:
        solvent_map_path = config.get("solvent_map", DEFAULT_SOLVENT_MAP_PATH)
        solvent_map = load_solvent_map(solvent_map_path)
        solvent_options = list(solvent_map.keys())
        solvent = _prompt_choice(
            "Select a solvent:",
            solvent_options,
            allow_custom=True,
            default_value=config.get("solvent", "vacuum"),
        )
    single_point_enabled = False
    frequency_enabled = False
    single_point_config = config.get("single_point") or {}
    if calculation_mode == "optimization":
        frequency_enabled = _prompt_yes_no("Run a frequency calculation?", default=True)
        if frequency_enabled:
            single_point_enabled = _prompt_yes_no(
                "If the frequency calculation has acceptable imaginary modes, "
                "run a single-point calculation?",
                default=True,
            )
        else:
            single_point_enabled = _prompt_yes_no(
                "Run a single-point calculation?",
                default=True,
            )
        if single_point_enabled:
            sp_dispersion_default = (
                single_point_config.get("dispersion")
                if "dispersion" in single_point_config
                else base_dispersion
            )
            sp_basis = _prompt_choice(
                "Select a basis set for the single-point calculation:",
                BASIS_SET_OPTIONS,
                allow_custom=True,
                default_value=basis,
            )
            sp_xc = _prompt_choice(
                "Select an XC functional for the single-point calculation:",
                XC_FUNCTIONAL_OPTIONS,
                allow_custom=True,
                default_value=xc,
            )
            sp_xc = normalize_xc_functional(sp_xc)
            sp_solvent_model = _prompt_choice(
                "Select a solvent model for the single-point calculation:",
                SOLVENT_MODEL_OPTIONS,
                default_value=solvent_model,
            )
            sp_dispersion = _prompt_dispersion(
                "single-point calculation",
                sp_dispersion_default,
            )
            if isinstance(sp_solvent_model, str) and sp_solvent_model.lower() in (
                "none",
                "none (vacuum)",
                "vacuum",
            ):
                sp_solvent_model = None
            if sp_solvent_model is None:
                sp_solvent = "vacuum"
            else:
                solvent_map_path = config.get("solvent_map", DEFAULT_SOLVENT_MAP_PATH)
                solvent_map = load_solvent_map(solvent_map_path)
                solvent_options = list(solvent_map.keys())
                sp_solvent = _prompt_choice(
                    "Select a solvent for the single-point calculation:",
                    solvent_options,
                    allow_custom=True,
                    default_value=solvent,
                )
            single_point_config = dict(single_point_config)
            single_point_config["basis"] = sp_basis
            single_point_config["xc"] = sp_xc
            single_point_config["solvent_model"] = sp_solvent_model
            single_point_config["solvent"] = sp_solvent
            single_point_config["dispersion"] = sp_dispersion
        if frequency_enabled:
            frequency_config = dict(config.get("frequency") or {})
            if "dispersion_model" in frequency_config:
                freq_dispersion_default = frequency_config.get("dispersion_model")
            else:
                freq_dispersion_default = (
                    sp_dispersion if single_point_enabled else base_dispersion
                )
            freq_dispersion = _prompt_dispersion(
                "frequency analysis",
                freq_dispersion_default,
            )
            frequency_config["dispersion_model"] = freq_dispersion
            config["frequency"] = frequency_config
    if calculation_mode == "single_point":
        sp_dispersion = _prompt_dispersion(
            "single-point calculation",
            single_point_config.get("dispersion", base_dispersion),
        )
        single_point_config = dict(single_point_config)
        single_point_config["dispersion"] = sp_dispersion
    if calculation_mode == "frequency":
        frequency_config = dict(config.get("frequency") or {})
        if "dispersion_model" in frequency_config:
            freq_dispersion_default = frequency_config.get("dispersion_model")
        else:
            freq_dispersion_default = base_dispersion
        freq_dispersion = _prompt_dispersion(
            "frequency analysis",
            freq_dispersion_default,
        )
        frequency_config["dispersion_model"] = freq_dispersion
        config["frequency"] = frequency_config

    config = json.loads(json.dumps(config))
    config["calculation_mode"] = calculation_mode
    config["basis"] = basis
    config["xc"] = xc
    if calculation_mode in ("optimization", "single_point", "frequency"):
        config["dispersion"] = base_dispersion
    config["solvent_model"] = solvent_model
    config["solvent"] = solvent
    config["frequency_enabled"] = frequency_enabled
    config["single_point_enabled"] = single_point_enabled
    if scan_config:
        config["scan"] = scan_config
    if single_point_config:
        config["single_point"] = single_point_config
    if constraints is None:
        config.pop("constraints", None)
    else:
        config["constraints"] = constraints
    config_raw = json.dumps(config, indent=2, ensure_ascii=False)
    args.config = "<interactive>"
    return config, config_raw, base_config_path


def prompt_config(args):
    return _prompt_interactive_config(args)


__all__ = [
    "prompt_config",
    "_prompt_choice",
    "_prompt_yes_no",
    "_prompt_interactive_config",
    "INTERACTIVE_CONFIG",
    "BASIS_SET_OPTIONS",
    "XC_FUNCTIONAL_OPTIONS",
    "SOLVENT_MODEL_OPTIONS",
    "CALCULATION_MODE_OPTIONS",
]
