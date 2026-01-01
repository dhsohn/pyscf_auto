import json
import os
from importlib import resources
from pathlib import Path

from .run_opt_chemistry import normalize_xc_functional
from .run_opt_config import DEFAULT_SOLVENT_MAP_PATH, load_run_config, load_solvent_map


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ASSET_ROOT = resources.files("pdft.assets")
INTERACTIVE_CONFIG_MINIMUM = _ASSET_ROOT / "run_config_ase.json"
INTERACTIVE_CONFIG_TS = _ASSET_ROOT / "run_config_ts.json"
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


def _resolve_interactive_config_path(filename: str) -> Path | None:
    asset_path = _ASSET_ROOT / filename
    if asset_path.is_file():
        return Path(asset_path)
    fallback_path = _REPO_ROOT / filename
    if fallback_path.is_file():
        return fallback_path
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

    config_filename = base_config_path.name
    base_config_path = _resolve_interactive_config_path(config_filename)
    if base_config_path is None or not base_config_path.is_file():
        expected_assets = _ASSET_ROOT / config_filename
        expected_repo = _REPO_ROOT / config_filename
        expected_paths = [
            str(path) for path in (expected_assets, expected_repo) if path is not None
        ]
        hint = ", ".join(expected_paths) if expected_paths else "run_config_ase.json"
        raise FileNotFoundError(
            "인터랙티브 기본 설정 파일을 찾을 수 없습니다. "
            f"다음 경로 중 하나에 파일이 있어야 합니다: {hint}"
        )

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


def prompt_config(args):
    return _prompt_interactive_config(args)


__all__ = [
    "prompt_config",
    "_prompt_choice",
    "_prompt_yes_no",
    "_prompt_interactive_config",
    "INTERACTIVE_CONFIG_MINIMUM",
    "INTERACTIVE_CONFIG_TS",
    "BASIS_SET_OPTIONS",
    "XC_FUNCTIONAL_OPTIONS",
    "SOLVENT_MODEL_OPTIONS",
    "CALCULATION_MODE_OPTIONS",
]
