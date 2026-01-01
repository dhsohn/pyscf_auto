import json
from pathlib import Path

from .run_opt_engine import normalize_xc_functional
from .run_opt_config import DEFAULT_SOLVENT_MAP_PATH, load_run_config, load_solvent_map


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
SOLVENT_MODEL_OPTIONS = ["pcm", "smd", "none (vacuum)"]
DISPERSION_MODEL_OPTIONS = ["none (사용 안 함)", "d3bj", "d3zero", "d4"]
CALCULATION_MODE_OPTIONS = [
    "구조 최적화",
    "단일점 에너지 계산",
    "프리퀀시 계산",
    "IRC 계산",
    "스캔 계산",
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


def _prompt_dispersion(stage_label, default_value=None):
    default_label = (
        DISPERSION_MODEL_OPTIONS[0] if default_value is None else default_value
    )
    choice = _prompt_choice(
        f"{stage_label} dispersion 보정을 선택하세요:",
        DISPERSION_MODEL_OPTIONS,
        default_value=default_label,
    )
    if choice.strip().lower().startswith("none"):
        return None
    return choice


def _prompt_int_list(prompt, count):
    while True:
        raw = input(f"{prompt} (예: 0,1)\n> ").strip()
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) != count:
            print(f"{count}개의 인덱스를 입력하세요.")
            continue
        try:
            values = [int(part) for part in parts]
        except ValueError:
            print("정수 인덱스를 입력하세요.")
            continue
        if any(value < 0 for value in values):
            print("인덱스는 0 이상의 정수여야 합니다.")
            continue
        return values


def _prompt_float(prompt):
    while True:
        raw = input(f"{prompt}\n> ").strip()
        try:
            return float(raw)
        except ValueError:
            print("숫자를 입력하세요.")


def _prompt_scan_dimension():
    scan_type = _prompt_choice(
        "스캔 유형을 선택하세요:",
        ["bond", "angle", "dihedral"],
    )
    if scan_type == "bond":
        indices = _prompt_int_list("bond 인덱스 i,j를 입력하세요", 2)
    elif scan_type == "angle":
        indices = _prompt_int_list("angle 인덱스 i,j,k를 입력하세요", 3)
    else:
        indices = _prompt_int_list("dihedral 인덱스 i,j,k,l을 입력하세요", 4)
    start = _prompt_float("시작값(start)을 입력하세요")
    end = _prompt_float("종료값(end)을 입력하세요")
    step = _prompt_float("스텝(step)을 입력하세요")
    dimension = {"type": scan_type, "start": start, "end": end, "step": step}
    for key, value in zip(("i", "j", "k", "l"), indices, strict=False):
        dimension[key] = value
    return dimension


def _prompt_interactive_config(args):
    calculation_choice = _prompt_choice(
        "어떤 계산을 진행할까요?",
        CALCULATION_MODE_OPTIONS,
    )
    calculation_mode = {
        "구조 최적화": "optimization",
        "단일점 에너지 계산": "single_point",
        "프리퀀시 계산": "frequency",
        "IRC 계산": "irc",
        "스캔 계산": "scan",
    }[calculation_choice]
    optimization_choice = None
    scan_config = None
    if calculation_mode == "optimization":
        optimization_choice = _prompt_choice(
            "최적화 유형을 선택하세요:",
            ["중간체 최적화", "전이상태 최적화"],
        )
    if calculation_mode == "scan":
        scan_mode_choice = _prompt_choice(
            "스캔 계산 모드를 선택하세요:",
            ["최적화 스캔", "단일점 스캔"],
        )
        scan_mode = "optimization" if scan_mode_choice == "최적화 스캔" else "single_point"
        dimension_count_choice = _prompt_choice(
            "스캔 차원을 선택하세요:",
            ["1D", "2D"],
        )
        dimension_count = 1 if dimension_count_choice == "1D" else 2
        dimensions = [_prompt_scan_dimension() for _ in range(dimension_count)]
        if dimension_count == 1:
            scan_config = dimensions[0]
            scan_config["mode"] = scan_mode
        else:
            scan_config = {"dimensions": dimensions, "mode": scan_mode}
    base_config_path = INTERACTIVE_CONFIG

    config_filename = base_config_path.name
    base_config_path = _REPO_ROOT / config_filename
    if not base_config_path.is_file():
        raise FileNotFoundError(
            "인터랙티브 기본 설정 파일을 찾을 수 없습니다. "
            f"다음 경로에 파일이 있어야 합니다: {base_config_path}"
        )

    config, _ = load_run_config(base_config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Failed to load base config from {base_config_path}.")
    if optimization_choice == "전이상태 최적화":
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
        "constraints를 JSON으로 입력하세요 (예: "
        "{\"bonds\":[{\"i\":0,\"j\":1,\"length\":1.10}]})"
    )
    if constraints_hint:
        constraints_prompt += f" [기본값: {constraints_hint}]"
    constraints_input = input(f"{constraints_prompt}\n> ").strip()
    if not constraints_input and constraints_default is not None:
        constraints = constraints_default
    elif not constraints_input:
        constraints = None
    elif constraints_input.lower() in ("none", "null", "없음", "no"):
        constraints = None
    else:
        try:
            constraints = json.loads(constraints_input)
        except json.JSONDecodeError as exc:
            raise ValueError("constraints 입력이 올바른 JSON이 아닙니다.") from exc
        if not isinstance(constraints, dict):
            raise ValueError("constraints 입력은 JSON 객체여야 합니다.")

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
                "인터랙티브 입력 디렉토리를 찾을 수 없습니다. "
                f"다음 경로 중 하나에 input 디렉토리가 있어야 합니다: {hint}"
            )
        input_xyz_files = sorted(
            path.name
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".xyz"
        )
        if not input_xyz_files:
            raise ValueError("input 디렉토리에 .xyz 파일이 없습니다.")
        selected_xyz = _prompt_choice(
            "인풋 파일을 선택하세요 (.xyz):",
            input_xyz_files,
        )
        args.xyz_file = str(input_dir / selected_xyz)

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
    base_dispersion = None
    if calculation_mode in ("optimization", "single_point", "frequency"):
        base_dispersion = _prompt_dispersion(
            {
                "optimization": "구조 최적화",
                "single_point": "단일점 계산",
                "frequency": "프리퀀시 계산",
            }[calculation_mode],
            config.get("dispersion"),
        )
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
            sp_dispersion_default = (
                single_point_config.get("dispersion")
                if "dispersion" in single_point_config
                else base_dispersion
            )
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
            sp_dispersion = _prompt_dispersion(
                "단일점 계산",
                sp_dispersion_default,
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
                "프리퀀시 계산",
                freq_dispersion_default,
            )
            frequency_config["dispersion_model"] = freq_dispersion
            config["frequency"] = frequency_config
    if calculation_mode == "single_point":
        sp_dispersion = _prompt_dispersion(
            "단일점 계산",
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
            "프리퀀시 계산",
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
