import json
import os
import re
import tomllib
from pathlib import Path
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any, Mapping

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
import yaml

from run_opt_paths import get_app_base_dir, get_runs_base_dir

DEFAULT_CHARGE = 0
DEFAULT_SPIN = None
DEFAULT_MULTIPLICITY = None
DEFAULT_SPIN_MODE = "strict"
DEFAULT_CONFIG_PATH = "run_config.json"
DEFAULT_SOLVENT_MAP_PATH = "solvent_dielectric.json"
DEFAULT_THREAD_COUNT = None
DEFAULT_LOG_PATH = "log/run.log"
DEFAULT_EVENT_LOG_PATH = "log/run_events.jsonl"
DEFAULT_OPTIMIZED_XYZ_PATH = "optimized.xyz"
DEFAULT_FREQUENCY_PATH = "frequency_result.json"
DEFAULT_IRC_PATH = "irc_result.json"
DEFAULT_IRC_PROFILE_CSV_PATH = "irc_profile.csv"
DEFAULT_RUN_METADATA_PATH = "metadata.json"
DEFAULT_QCSCHEMA_OUTPUT_PATH = "qcschema_result.json"
DEFAULT_SCAN_RESULT_PATH = "scan_result.json"
DEFAULT_SCAN_RESULT_CSV_PATH = "scan_result.csv"
DEFAULT_SCF_CHKFILE = "scf.chk"
DEFAULT_CONFIG_USED_PATH = "config_used.json"


def _normalize_solvent_key(name):
    return "".join(char for char in str(name).lower() if char.isalnum())


SMD_UNSUPPORTED_SOLVENTS = (
    "propylene carbonate",
    "dimethyl carbonate",
    "nmp",
    "n-methyl-2-pyrrolidone",
    "n-methyl-2-pyrrolidinone",
)
SMD_UNSUPPORTED_SOLVENT_KEYS = {
    _normalize_solvent_key(name) for name in SMD_UNSUPPORTED_SOLVENTS
}

DEFAULT_APP_BASE_DIR = get_app_base_dir()
DEFAULT_RUNS_BASE_DIR = get_runs_base_dir()
DEFAULT_QUEUE_PATH = os.path.join(DEFAULT_RUNS_BASE_DIR, "queue.json")
DEFAULT_QUEUE_LOCK_PATH = os.path.join(DEFAULT_RUNS_BASE_DIR, "queue.lock")
DEFAULT_QUEUE_RUNNER_LOCK_PATH = os.path.join(DEFAULT_RUNS_BASE_DIR, "queue.runner.lock")
DEFAULT_QUEUE_RUNNER_LOG_PATH = os.path.join(
    DEFAULT_APP_BASE_DIR, "log", "queue_runner.log"
)

RUN_CONFIG_EXAMPLES = {
    "threads": "\"threads\": 4",
    "memory_gb": "\"memory_gb\": 8",
    "basis": "\"basis\": \"def2-svp\"",
    "xc": "\"xc\": \"b3lyp\"",
    "solvent": "\"solvent\": \"water\"",
    "solvent_model": "\"solvent_model\": \"pcm\"",
    "dispersion": "\"dispersion\": \"d3bj\"",
    "spin_mode": "\"spin_mode\": \"strict\"",
    "calculation_mode": "\"calculation_mode\": \"optimization\"",
    "irc_enabled": "\"irc_enabled\": true",
    "irc": "\"irc\": {\"steps\": 10, \"step_size\": 0.05, \"force_threshold\": 0.01}",
    "qcschema_output_file": "\"qcschema_output_file\": \"qcschema_result.json\"",
    "optimizer": (
        "\"optimizer\": {\"mode\": \"minimum\", \"output_xyz\": \"ase_optimized.xyz\", "
        "\"ase\": {\"d3_params\": {\"damping\": {\"s6\": 1.0}}, \"optimizer\": \"bfgs\"}}"
    ),
    "optimizer.ase": "\"ase\": {\"d3_params\": {\"damping\": {\"s6\": 1.0}}}",
    "scf": (
        "\"scf\": {\"max_cycle\": 200, \"conv_tol\": 1e-7, \"diis\": 8, "
        "\"chkfile\": \"scf.chk\", \"extra\": {\"grids\": {\"level\": 3}}}"
    ),
    "scf.retry_preset": "\"scf\": {\"retry_preset\": \"stable\"}",
    "scf.diis_preset": "\"scf\": {\"diis_preset\": \"stable\"}",
    "scf.reference": "\"scf\": {\"reference\": \"uks\"}",
    "single_point": (
        "\"single_point\": {\"basis\": \"def2-svp\", \"xc\": \"b3lyp\", "
        "\"solvent\": \"water\", \"dispersion\": \"d3bj\"}"
    ),
    "single_point.scf": (
        "\"scf\": {\"max_cycle\": 200, \"conv_tol\": 1e-7, \"diis\": 8, "
        "\"chkfile\": \"scf.chk\", \"extra\": {\"density_fit\": true}}"
    ),
    "frequency": "\"frequency\": {\"dispersion\": \"numerical\", \"dispersion_model\": \"d3bj\"}",
    "freq": "\"freq\": {\"dispersion\": \"numerical\", \"dispersion_model\": \"d3bj\"}",
    "frequency.dispersion_step": "\"frequency\": {\"dispersion_step\": 0.005}",
    "frequency.use_chkfile": "\"frequency\": {\"use_chkfile\": false}",
    "io.write_interval_steps": "\"io\": {\"write_interval_steps\": 10}",
    "io.write_interval_seconds": "\"io\": {\"write_interval_seconds\": 30}",
    "io.scan_write_interval_points": "\"io\": {\"scan_write_interval_points\": 5}",
    "thermo": "\"thermo\": {\"T\": 298.15, \"P\": 1.0, \"unit\": \"atm\"}",
    "constraints": (
        "\"constraints\": {\"bonds\": [{\"i\": 0, \"j\": 1, \"length\": 1.10}], "
        "\"angles\": [{\"i\": 0, \"j\": 1, \"k\": 2, \"angle\": 120.0}], "
        "\"dihedrals\": [{\"i\": 0, \"j\": 1, \"k\": 2, \"l\": 3, \"dihedral\": 180.0}]}"
    ),
    "scan": (
        "\"scan\": {\"type\": \"bond\", \"i\": 0, \"j\": 1, \"start\": 1.0, "
        "\"end\": 2.0, \"step\": 0.1, \"mode\": \"optimization\"}"
    ),
    "scan.executor": "\"scan\": {\"executor\": \"local\"}",
    "scan.max_workers": "\"scan\": {\"executor\": \"local\", \"max_workers\": 4}",
    "scan.threads_per_worker": "\"scan\": {\"threads_per_worker\": 2}",
    "scan.batch_size": "\"scan\": {\"batch_size\": 10}",
    "ts_quality": (
        "\"ts_quality\": {\"expected_imaginary_count\": 1, "
        "\"imaginary_frequency_min_abs\": 50.0, \"imaginary_frequency_max_abs\": 1500.0, "
        "\"projection_min_abs\": 0.01, "
        "\"internal_coordinates\": [{\"type\": \"bond\", \"i\": 0, \"j\": 1, "
        "\"target\": 2.0}]}"
    ),
}

def _ensure_dict(data: Mapping[str, Any] | None, label: str) -> dict[str, Any] | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"Config '{label}' must be an object.")
    return dict(data)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_unset=True)

    @property
    def raw(self) -> dict[str, Any]:
        return self.to_dict()


class SCFConfig(ConfigModel):
    max_cycle: int | None = None
    conv_tol: float | None = None
    level_shift: float | None = None
    damping: float | None = None
    diis: bool | int | None = None
    diis_preset: str | None = None
    retry_preset: str | None = None
    chkfile: str | None = None
    reference: str | None = None
    extra: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SCFConfig | None":
        data = _ensure_dict(data, "scf")
        if data is None:
            return None
        return cls.model_validate(data)


class OptimizerASEConfig(ConfigModel):
    d3_backend: str | None = None
    dftd3_backend: str | None = None
    d3_params: dict[str, Any] | None = None
    dftd3_params: dict[str, Any] | None = None
    optimizer: str | None = None
    fmax: float | None = None
    steps: int | None = None
    trajectory: str | None = None
    logfile: str | None = None
    sella: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "OptimizerASEConfig | None":
        data = _ensure_dict(data, "optimizer.ase")
        if data is None:
            return None
        return cls.model_validate(data)


class OptimizerConfig(ConfigModel):
    output_xyz: str | None = None
    mode: str | None = None
    ase: OptimizerASEConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "OptimizerConfig | None":
        data = _ensure_dict(data, "optimizer")
        if data is None:
            return None
        return cls.model_validate(data)


class SinglePointConfig(ConfigModel):
    basis: str | None = None
    xc: str | None = None
    solvent: str | None = None
    solvent_model: str | None = None
    solvent_map: str | None = None
    dispersion: str | None = None
    scf: SCFConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SinglePointConfig | None":
        data = _ensure_dict(data, "single_point")
        if data is None:
            return None
        return cls.model_validate(data)


class FrequencyConfig(ConfigModel):
    dispersion: str | None = None
    dispersion_model: str | None = None
    dispersion_step: float | None = None
    use_chkfile: bool | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "FrequencyConfig | None":
        data = _ensure_dict(data, "frequency")
        if data is None:
            return None
        return cls.model_validate(data)


class IrcConfig(ConfigModel):
    steps: int | None = None
    step_size: float | None = None
    force_threshold: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "IrcConfig | None":
        data = _ensure_dict(data, "irc")
        if data is None:
            return None
        return cls.model_validate(data)


class TSQualityConfig(ConfigModel):
    expected_imaginary_count: int | None = None
    imaginary_frequency_min_abs: float | None = None
    imaginary_frequency_max_abs: float | None = None
    projection_step: float | None = None
    projection_min_abs: float | None = None
    enforce: bool | None = None
    internal_coordinates: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TSQualityConfig | None":
        data = _ensure_dict(data, "ts_quality")
        if data is None:
            return None
        return cls.model_validate(data)


class ThermoConfig(ConfigModel):
    temperature: float | None = Field(default=None, alias="T")
    pressure: float | None = Field(default=None, alias="P")
    unit: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ThermoConfig | None":
        data = _ensure_dict(data, "thermo")
        if data is None:
            return None
        return cls.model_validate(data)


class IOConfig(ConfigModel):
    write_interval_steps: int | None = None
    write_interval_seconds: float | None = None
    scan_write_interval_points: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "IOConfig | None":
        data = _ensure_dict(data, "io")
        if data is None:
            return None
        return cls.model_validate(data)


class RunConfig(ConfigModel):
    threads: int | None = None
    memory_gb: float | None = None
    basis: str | None = None
    xc: str | None = None
    solvent: str | None = None
    solvent_model: str | None = None
    dispersion: str | None = None
    spin_mode: str | None = None
    calculation_mode: str | None = None
    enforce_os_memory_limit: bool | None = None
    verbose: bool | None = None
    single_point_enabled: bool | None = None
    frequency_enabled: bool | None = None
    irc_enabled: bool | None = None
    log_file: str | None = None
    event_log_file: str | None = None
    optimized_xyz_file: str | None = None
    run_metadata_file: str | None = None
    qcschema_output_file: str | None = None
    frequency_file: str | None = None
    irc_file: str | None = None
    irc_profile_csv_file: str | None = None
    scan_result_csv_file: str | None = None
    solvent_map: str | None = None
    optimizer: OptimizerConfig | None = None
    scf: SCFConfig | None = None
    single_point: SinglePointConfig | None = None
    frequency: FrequencyConfig | None = Field(
        default=None,
        validation_alias=AliasChoices("frequency", "freq"),
        serialization_alias="frequency",
    )
    irc: IrcConfig | None = None
    ts_quality: TSQualityConfig | None = None
    thermo: ThermoConfig | None = None
    io: IOConfig | None = None
    constraints: dict[str, Any] | None = None
    scan: dict[str, Any] | None = None
    scan2d: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        if not isinstance(data, dict):
            raise ValueError("Config must be an object/mapping.")
        return cls.model_validate(data)


def load_run_config(config_path):
    if not config_path:
        return {}, None
    if not os.path.isfile(config_path):
        default_missing = os.path.basename(str(config_path)) == DEFAULT_CONFIG_PATH
        example_configs = [
            name for name in ("run_config.json",) if os.path.isfile(name)
        ]
        example_hint = ", ".join(example_configs) if example_configs else "run_config.json"
        if default_missing:
            message = (
                f"Default config '{DEFAULT_CONFIG_PATH}' was not found at '{config_path}'. "
                f"Copy an example config (e.g., {example_hint}) to {DEFAULT_CONFIG_PATH}, "
                "or pass --config with a valid config file (JSON/YAML/TOML)."
            )
        else:
            message = (
                f"Config file not found: '{config_path}'. "
                "Use --config with a valid config file (JSON/YAML/TOML) "
                f"(examples: {example_hint})."
            )
        raise FileNotFoundError(message)
    with open(config_path, "r", encoding="utf-8") as config_file:
        raw_config = config_file.read()
        config, raw_config = _parse_config_contents(
            config_path, raw_config, file_label="config"
        )
    if not isinstance(config, dict):
        raise ValueError("Config must be an object/mapping.")
    return config, raw_config


def load_solvent_map(map_path):
    if not map_path:
        return {}
    resolved = resolve_solvent_map_path(map_path)
    if resolved is None:
        raise FileNotFoundError(f"Solvent map file not found: '{map_path}'.")
    return _load_solvent_map_from_resolved(resolved)


def resolve_solvent_map_path(map_path):
    if not map_path:
        return None
    if os.path.isfile(map_path):
        return map_path
    resource = resolve_solvent_map_resource()
    if resource and os.path.basename(str(map_path)) == DEFAULT_SOLVENT_MAP_PATH:
        return resource
    return map_path


def resolve_solvent_map_resource():
    module_dir = Path(__file__).resolve().parent
    candidate = module_dir / DEFAULT_SOLVENT_MAP_PATH
    if candidate.is_file():
        return candidate
    try:
        resource = resources.files(__package__).joinpath(DEFAULT_SOLVENT_MAP_PATH)
    except (ModuleNotFoundError, AttributeError, TypeError):
        return None
    if resource.is_file():
        return resource
    return None


def load_solvent_map_from_path(map_path):
    if not map_path:
        return {}
    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"Solvent map file not found: '{map_path}'.")
    return _load_solvent_map_from_resolved(map_path)


def load_solvent_map_from_resource():
    resource = resolve_solvent_map_resource()
    if resource is None:
        raise FileNotFoundError("Solvent map package resource not found.")
    return _load_solvent_map_from_resolved(resource)


def _load_solvent_map_from_resolved(resolved):
    with _open_solvent_map(resolved) as map_file:
        raw_map = map_file.read()
    map_data, _ = _parse_config_contents(
        str(resolved),
        raw_map,
        expect_raw_json=False,
        file_label="solvent map",
    )
    if not isinstance(map_data, dict):
        raise ValueError("Solvent map must be an object/mapping.")
    return map_data


def _open_solvent_map(resolved):
    if isinstance(resolved, Traversable):
        return resolved.open("r", encoding="utf-8")
    return open(resolved, "r", encoding="utf-8")


SUPPORTED_CONFIG_FORMATS = ".json/.yaml/.toml"


def _format_parse_error_prefix(path, file_label):
    label = f"{file_label} file" if file_label else "file"
    return f"Failed to parse {label} '{path}' (supported: {SUPPORTED_CONFIG_FORMATS})"


def _format_json_decode_error(path, error, file_label):
    location = f"line {error.lineno} column {error.colno}"
    message = error.msg
    if message == "Extra data":
        remainder = ""
        if error.doc and error.pos is not None:
            remainder = error.doc[error.pos : error.pos + 120]
        preview = repr(remainder) if remainder else "(empty)"
        common_causes = (
            "two JSON objects stuck together; extra text after JSON; "
            "comments are not allowed in standard JSON"
        )
        message = (
            "Extra data after the first JSON value. "
            f"remainder preview: {preview}. "
            f"common causes: {common_causes}. "
            "More than one JSON object detected in the file."
        )
    prefix = _format_parse_error_prefix(path, file_label)
    return f"{prefix} ({location}): {message}"


def _format_yaml_decode_error(path, error, file_label):
    prefix = _format_parse_error_prefix(path, file_label)
    return f"{prefix}: {error}"


def _format_toml_decode_error(path, error, file_label):
    prefix = _format_parse_error_prefix(path, file_label)
    return f"{prefix}: {error}"


def _parse_config_contents(path, raw_text, expect_raw_json=True, file_label="config"):
    extension = os.path.splitext(str(path))[1].lower()
    if extension in (".yaml", ".yml"):
        try:
            config = yaml.safe_load(raw_text)
        except yaml.YAMLError as error:
            raise ValueError(_format_yaml_decode_error(path, error, file_label)) from error
        if expect_raw_json:
            raw_text = json.dumps(config, indent=2, ensure_ascii=False)
        return config, raw_text
    if extension == ".toml":
        try:
            config = tomllib.loads(raw_text)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(_format_toml_decode_error(path, error, file_label)) from error
        if expect_raw_json:
            raw_text = json.dumps(config, indent=2, ensure_ascii=False)
        return config, raw_text
    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(_format_json_decode_error(path, error, file_label)) from error
    return config, raw_text


def _validate_fields(config, rules, prefix=""):
    for key, (predicate, message) in rules.items():
        if key in config and config[key] is not None:
            if not predicate(config[key]):
                name = f"{prefix}{key}"
                raise ValueError(message.format(name=name))


def _schema_example_for_path(path):
    if not path:
        return "See run_config.json for a complete example."
    if path in RUN_CONFIG_EXAMPLES:
        return RUN_CONFIG_EXAMPLES[path]
    leaf = path.split(".")[-1]
    return RUN_CONFIG_EXAMPLES.get(leaf, "See run_config.json for a complete example.")


def validate_run_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be an object/mapping.")

    def is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def is_bool(value):
        return isinstance(value, bool)

    def is_str(value):
        return isinstance(value, str)

    def is_diis(value):
        return isinstance(value, (bool, int))

    def is_dict(value):
        return isinstance(value, dict)

    def is_positive_int(value):
        return is_int(value) and value > 0

    def is_positive_number(value):
        return is_number(value) and value > 0

    def normalize_calc_mode(value):
        if not value:
            return None
        normalized = re.sub(r"[\s_\-]+", "", str(value)).lower()
        if normalized in (
            "optimization",
            "opt",
            "geometry",
            "geom",
            "structure",
            "structureoptimization",
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
        ):
            return "single_point"
        if normalized in (
            "frequency",
            "frequencies",
            "freq",
            "vibration",
            "vibrational",
        ):
            return "frequency"
        if normalized in (
            "irc",
            "intrinsicreactioncoordinate",
            "reactionpath",
            "reactioncoordinate",
        ):
            return "irc"
        if normalized in ("scan", "scanning"):
            return "scan"
        return None

    scf_retry_presets = {"fast", "default", "stable", "off"}
    scf_retry_aliases = {
        "conservative": "fast",
        "aggressive": "stable",
        "robust": "stable",
        "none": "off",
        "disabled": "off",
    }
    diis_presets = {"fast", "default", "stable", "off"}
    diis_aliases = {
        "conservative": "fast",
        "aggressive": "stable",
        "robust": "stable",
        "none": "off",
        "disabled": "off",
    }

    def normalize_preset(value, aliases, allowed, label):
        normalized = re.sub(r"[\s_\-]+", "", str(value)).lower()
        normalized = aliases.get(normalized, normalized)
        if normalized not in allowed:
            allowed_values = ", ".join(sorted(allowed))
            raise ValueError(
                "Config '{label}' must be one of: {values}.".format(
                    label=label, values=allowed_values
                )
            )
        return normalized

    def _validate_smd_solvent_support(solvent_model, solvent, field_label):
        if not solvent_model:
            return
        if str(solvent_model).lower() != "smd":
            return
        if not solvent:
            return
        if _normalize_solvent_key(solvent) in SMD_UNSUPPORTED_SOLVENT_KEYS:
            raise ValueError(
                "Config '{field}' uses SMD solvent '{solvent}', which is not supported "
                "by PySCF SMD. Use PCM or choose another solvent.".format(
                    field=field_label, solvent=solvent
                )
            )

    def _validate_scan_dimension(dim, path, require_bounds=True):
        if not isinstance(dim, dict):
            raise ValueError(f"Config '{path}' must be an object.")
        dim_type = dim.get("type")
        if not isinstance(dim_type, str):
            raise ValueError(f"Config '{path}.type' must be a string.")
        if dim_type not in ("bond", "angle", "dihedral"):
            raise ValueError(
                f"Config '{path}.type' must be one of: bond, angle, dihedral."
            )
        required_indices = {"bond": ("i", "j"), "angle": ("i", "j", "k"), "dihedral": ("i", "j", "k", "l")}
        for key in required_indices[dim_type]:
            value = dim.get(key)
            if not is_int(value) or isinstance(value, bool):
                raise ValueError(
                    "Config '{path}.{key}' must be an integer.".format(path=path, key=key)
                )
            if value < 0:
                raise ValueError(
                    "Config '{path}.{key}' must be >= 0.".format(path=path, key=key)
                )
        if require_bounds:
            for key in ("start", "end", "step"):
                value = dim.get(key)
                if not is_number(value):
                    raise ValueError(
                        "Config '{path}.{key}' must be a number.".format(path=path, key=key)
                    )
            if dim.get("step") == 0:
                raise ValueError(f"Config '{path}.step' must be non-zero.")

    def _validate_scan_config(scan, name, require_dimensions=False):
        if not isinstance(scan, dict):
            raise ValueError(f"Config '{name}' must be an object.")
        executor_value = scan.get("executor")
        if executor_value is not None:
            if not isinstance(executor_value, str):
                raise ValueError(f"Config '{name}.executor' must be a string.")
            if executor_value not in ("serial", "local", "manifest"):
                raise ValueError(
                    "Config '{name}.executor' must be one of: serial, local, manifest.".format(
                        name=name
                    )
                )
        max_workers = scan.get("max_workers")
        if max_workers is not None:
            if not is_int(max_workers) or max_workers < 1:
                raise ValueError(
                    "Config '{name}.max_workers' must be a positive integer.".format(
                        name=name
                    )
                )
        threads_per_worker = scan.get("threads_per_worker")
        if threads_per_worker is not None:
            if not is_int(threads_per_worker) or threads_per_worker < 1:
                raise ValueError(
                    "Config '{name}.threads_per_worker' must be a positive integer.".format(
                        name=name
                    )
                )
        batch_size = scan.get("batch_size")
        if batch_size is not None:
            if not is_int(batch_size) or batch_size < 1:
                raise ValueError(
                    "Config '{name}.batch_size' must be a positive integer.".format(
                        name=name
                    )
                )
        manifest_file = scan.get("manifest_file")
        if manifest_file is not None and not isinstance(manifest_file, str):
            raise ValueError(f"Config '{name}.manifest_file' must be a string.")
        mode_value = scan.get("mode")
        if mode_value is not None:
            if not isinstance(mode_value, str):
                raise ValueError(f"Config '{name}.mode' must be a string.")
            if mode_value not in ("optimization", "single_point"):
                raise ValueError(
                    f"Config '{name}.mode' must be one of: optimization, single_point."
                )
        dimensions = scan.get("dimensions")
        grid = scan.get("grid")
        if require_dimensions or dimensions is not None:
            if not isinstance(dimensions, list) or not dimensions:
                raise ValueError(f"Config '{name}.dimensions' must be a non-empty list.")
            if name == "scan2d" and len(dimensions) != 2:
                raise ValueError("Config 'scan2d.dimensions' must have exactly 2 entries.")
            require_bounds = grid is None
            for idx, dim in enumerate(dimensions):
                _validate_scan_dimension(dim, f"{name}.dimensions[{idx}]", require_bounds)
            if grid is not None:
                if not isinstance(grid, list):
                    raise ValueError(f"Config '{name}.grid' must be a list.")
                if len(grid) != len(dimensions):
                    raise ValueError(
                        "Config '{name}.grid' length must match dimensions.".format(
                            name=name
                        )
                    )
                for idx, values in enumerate(grid):
                    if not isinstance(values, list) or not values:
                        raise ValueError(
                            "Config '{name}.grid[{idx}]' must be a non-empty list.".format(
                                name=name, idx=idx
                            )
                        )
                    for value in values:
                        if not is_number(value):
                            raise ValueError(
                                "Config '{name}.grid[{idx}]' must contain numbers.".format(
                                    name=name, idx=idx
                                )
                            )
            return
        _validate_scan_dimension(scan, name, require_bounds=True)

    def _validate_scf_extra(extra, name):
        if extra is None:
            return
        if not isinstance(extra, dict):
            raise ValueError(f"Config '{name}.extra' must be an object.")
        allowed_keys = {"grids", "density_fit", "init_guess"}
        unknown = set(extra) - allowed_keys
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise ValueError(
                "Config '{name}.extra' has unsupported keys: {keys}. "
                "Allowed keys: grids, density_fit, init_guess, grids.level, grids.prune.".format(
                    name=name, keys=unknown_list
                )
            )
        if "grids" in extra:
            grids = extra.get("grids")
            if not isinstance(grids, dict):
                raise ValueError(f"Config '{name}.extra.grids' must be an object.")
            allowed_grid_keys = {"level", "prune"}
            grid_unknown = set(grids) - allowed_grid_keys
            if grid_unknown:
                grid_unknown_list = ", ".join(sorted(grid_unknown))
                raise ValueError(
                    "Config '{name}.extra.grids' has unsupported keys: {keys}. "
                    "Allowed keys: level, prune.".format(name=name, keys=grid_unknown_list)
                )

    validation_rules = {
        "threads": (is_positive_int, "Config '{name}' must be a positive integer."),
        "memory_gb": (is_positive_number, "Config '{name}' must be a positive number (int or float)."),
        "enforce_os_memory_limit": (is_bool, "Config '{name}' must be a boolean."),
        "verbose": (is_bool, "Config '{name}' must be a boolean."),
        "single_point_enabled": (is_bool, "Config '{name}' must be a boolean."),
        "frequency_enabled": (is_bool, "Config '{name}' must be a boolean."),
        "irc_enabled": (is_bool, "Config '{name}' must be a boolean."),
        "calculation_mode": (is_str, "Config '{name}' must be a string."),
        "log_file": (is_str, "Config '{name}' must be a string path."),
        "event_log_file": (is_str, "Config '{name}' must be a string path."),
        "optimized_xyz_file": (is_str, "Config '{name}' must be a string path."),
        "run_metadata_file": (is_str, "Config '{name}' must be a string path."),
        "qcschema_output_file": (is_str, "Config '{name}' must be a string path."),
        "frequency_file": (is_str, "Config '{name}' must be a string path."),
        "irc_file": (is_str, "Config '{name}' must be a string path."),
        "irc_profile_csv_file": (is_str, "Config '{name}' must be a string path."),
        "scan_result_csv_file": (is_str, "Config '{name}' must be a string path."),
        "basis": (is_str, "Config '{name}' must be a string."),
        "xc": (is_str, "Config '{name}' must be a string."),
        "solvent": (is_str, "Config '{name}' must be a string."),
        "solvent_model": (is_str, "Config '{name}' must be a string."),
        "solvent_map": (is_str, "Config '{name}' must be a string path."),
        "dispersion": (is_str, "Config '{name}' must be a string."),
        "spin_mode": (is_str, "Config '{name}' must be a string."),
    }
    _validate_fields(config, validation_rules)
    for required_key in ("basis", "xc", "solvent"):
        if config.get(required_key) in (None, ""):
            raise ValueError(
                "Config '{name}' is required. Example: {example}.".format(
                    name=required_key,
                    example=_schema_example_for_path(required_key),
                )
            )
    calculation_mode = config.get("calculation_mode")
    if calculation_mode is not None:
        allowed_modes = ("optimization", "single_point", "frequency", "irc", "scan")
        if calculation_mode not in allowed_modes:
            allowed_values = ", ".join(allowed_modes)
            raise ValueError(
                "Config 'calculation_mode' must be one of: {values}. "
                "Example: {example}.".format(
                    values=allowed_values,
                    example=_schema_example_for_path("calculation_mode"),
                )
            )
    spin_mode = config.get("spin_mode")
    if spin_mode is not None:
        allowed_spin_modes = ("auto", "strict")
        if spin_mode not in allowed_spin_modes:
            allowed_values = ", ".join(allowed_spin_modes)
            raise ValueError(
                "Config 'spin_mode' must be one of: {values}. "
                "Example: {example}.".format(
                    values=allowed_values,
                    example=_schema_example_for_path("spin_mode"),
                )
            )
    dispersion = config.get("dispersion")
    if dispersion is not None and dispersion not in ("d3bj", "d3zero", "d4"):
        raise ValueError(
            "Config 'dispersion' must be one of: d3bj, d3zero, d4. "
            "Example: {example}.".format(example=_schema_example_for_path("dispersion"))
        )
    solvent_model = config.get("solvent_model")
    if solvent_model:
        if not isinstance(solvent_model, str):
            raise ValueError("Config 'solvent_model' must be a string.")
        if solvent_model not in ("pcm", "smd"):
            raise ValueError(
                "Config 'solvent_model' must be one of: pcm, smd. "
                "Example: \"solvent_model\": \"pcm\"."
            )
        if not config.get("solvent"):
            raise ValueError(
                "Config 'solvent' is required when 'solvent_model' is set. "
                "Example: \"solvent\": \"water\"."
            )
        _validate_smd_solvent_support(solvent_model, config.get("solvent"), "solvent")
    if "optimizer" in config and config["optimizer"] is not None:
        if not isinstance(config["optimizer"], dict):
            raise ValueError("Config 'optimizer' must be an object.")
        optimizer_rules = {
            "output_xyz": (is_str, "Config '{name}' must be a string path."),
            "mode": (is_str, "Config '{name}' must be a string."),
        }
        _validate_fields(config["optimizer"], optimizer_rules, prefix="optimizer.")
        if "ase" in config["optimizer"] and config["optimizer"]["ase"] is not None:
            if not isinstance(config["optimizer"]["ase"], dict):
                raise ValueError("Config 'optimizer.ase' must be an object.")
            ase_config = config["optimizer"]["ase"]
            for backend_key in ("d3_backend", "dftd3_backend"):
                backend_value = ase_config.get(backend_key)
                if backend_value is None:
                    continue
                if not isinstance(backend_value, str):
                    raise ValueError(
                        f"Config 'optimizer.ase.{backend_key}' must be a string."
                    )
                if backend_value != "dftd3":
                    raise ValueError(
                        "Config 'optimizer.ase.{name}' must be 'dftd3'.".format(
                            name=backend_key
                        )
                    )
            d3_params = ase_config.get("d3_params")
            dftd3_params = ase_config.get("dftd3_params")
            if d3_params is not None and dftd3_params is not None:
                raise ValueError(
                    "Config must not define both 'optimizer.ase.d3_params' and "
                    "'optimizer.ase.dftd3_params'."
                )
            if d3_params is not None:
                chosen_params = d3_params
                param_prefix = "optimizer.ase.d3_params"
            else:
                chosen_params = dftd3_params
                param_prefix = "optimizer.ase.dftd3_params"
            if chosen_params is not None:
                if not isinstance(chosen_params, dict):
                    raise ValueError(
                        f"Config '{param_prefix}' must be an object/mapping."
                    )
                for key, value in chosen_params.items():
                    if key == "damping" and isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey in ("damping", "variant", "method"):
                                if not isinstance(subvalue, str):
                                    raise ValueError(
                                        f"Config '{param_prefix}.damping.{subkey}' must be a string."
                                    )
                                continue
                            if subkey == "parameters" and isinstance(subvalue, dict):
                                for param_key, param_value in subvalue.items():
                                    if not is_number(param_value):
                                        raise ValueError(
                                            "Config '{prefix}.damping.parameters.{name}' "
                                            "must be a number.".format(
                                                prefix=param_prefix, name=param_key
                                            )
                                        )
                                continue
                            if not is_number(subvalue):
                                raise ValueError(
                                    "Config '{prefix}.damping.{name}' "
                                    "must be a number.".format(prefix=param_prefix, name=subkey)
                                )
                        continue
                    if key == "parameters" and isinstance(value, dict):
                        for param_key, param_value in value.items():
                            if not is_number(param_value):
                                raise ValueError(
                                    "Config '{prefix}.parameters.{name}' must be a number.".format(
                                        prefix=param_prefix, name=param_key
                                    )
                                )
                        continue
                    if not is_number(value):
                        raise ValueError(
                            f"Config '{param_prefix}.{key}' must be a number."
                        )
    if "scf" in config and config["scf"] is not None:
        if not isinstance(config["scf"], dict):
            raise ValueError("Config 'scf' must be an object.")
        scf_validation_rules = {
            "max_cycle": (is_int, "Config '{name}' must be an integer."),
            "conv_tol": (is_number, "Config '{name}' must be a number (int or float)."),
            "level_shift": (is_number, "Config '{name}' must be a number (int or float)."),
            "damping": (is_number, "Config '{name}' must be a number (int or float)."),
            "diis": (is_diis, "Config '{name}' must be a boolean or integer."),
            "diis_preset": (is_str, "Config '{name}' must be a string."),
            "retry_preset": (is_str, "Config '{name}' must be a string."),
            "chkfile": (is_str, "Config '{name}' must be a string path."),
            "reference": (is_str, "Config '{name}' must be a string."),
            "extra": (is_dict, "Config '{name}' must be an object."),
        }
        _validate_fields(config["scf"], scf_validation_rules, prefix="scf.")
        _validate_scf_extra(config["scf"].get("extra"), "scf")
        reference_value = config["scf"].get("reference")
        if reference_value is not None:
            allowed_reference = ("auto", "rks", "uks")
            normalized_reference = str(reference_value).strip().lower()
            if normalized_reference not in allowed_reference:
                allowed_values = ", ".join(allowed_reference)
                raise ValueError(
                    "Config 'scf.reference' must be one of: {values}. "
                    "Example: {example}.".format(
                        values=allowed_values,
                        example=_schema_example_for_path("scf.reference"),
                    )
                )
        if config["scf"].get("retry_preset") is not None:
            normalize_preset(
                config["scf"].get("retry_preset"),
                scf_retry_aliases,
                scf_retry_presets,
                "scf.retry_preset",
            )
        if config["scf"].get("diis_preset") is not None:
            normalize_preset(
                config["scf"].get("diis_preset"),
                diis_aliases,
                diis_presets,
                "scf.diis_preset",
            )
    if "single_point" in config and config["single_point"] is not None:
        if not isinstance(config["single_point"], dict):
            raise ValueError("Config 'single_point' must be an object.")
        single_point_rules = {
            "basis": (is_str, "Config '{name}' must be a string."),
            "xc": (is_str, "Config '{name}' must be a string."),
            "solvent": (is_str, "Config '{name}' must be a string."),
            "solvent_model": (is_str, "Config '{name}' must be a string."),
            "solvent_map": (is_str, "Config '{name}' must be a string path."),
            "dispersion": (is_str, "Config '{name}' must be a string."),
        }
        _validate_fields(config["single_point"], single_point_rules, prefix="single_point.")
        _validate_smd_solvent_support(
            config["single_point"].get("solvent_model"),
            config["single_point"].get("solvent"),
            "single_point.solvent",
        )
        if "scf" in config["single_point"] and config["single_point"]["scf"] is not None:
            if not isinstance(config["single_point"]["scf"], dict):
                raise ValueError("Config 'single_point.scf' must be an object.")
            scf_validation_rules = {
                "max_cycle": (is_int, "Config '{name}' must be an integer."),
                "conv_tol": (is_number, "Config '{name}' must be a number (int or float)."),
                "level_shift": (is_number, "Config '{name}' must be a number (int or float)."),
                "damping": (is_number, "Config '{name}' must be a number (int or float)."),
                "diis": (is_diis, "Config '{name}' must be a boolean or integer."),
                "diis_preset": (is_str, "Config '{name}' must be a string."),
                "retry_preset": (is_str, "Config '{name}' must be a string."),
                "chkfile": (is_str, "Config '{name}' must be a string path."),
                "reference": (is_str, "Config '{name}' must be a string."),
                "extra": (is_dict, "Config '{name}' must be an object."),
            }
            _validate_fields(config["single_point"]["scf"], scf_validation_rules, prefix="single_point.scf.")
            _validate_scf_extra(config["single_point"]["scf"].get("extra"), "single_point.scf")
            reference_value = config["single_point"]["scf"].get("reference")
            if reference_value is not None:
                allowed_reference = ("auto", "rks", "uks")
                normalized_reference = str(reference_value).strip().lower()
                if normalized_reference not in allowed_reference:
                    allowed_values = ", ".join(allowed_reference)
                    raise ValueError(
                        "Config 'single_point.scf.reference' must be one of: {values}. "
                        "Example: {example}.".format(
                            values=allowed_values,
                            example=_schema_example_for_path("scf.reference"),
                        )
                    )
            if config["single_point"]["scf"].get("retry_preset") is not None:
                normalize_preset(
                    config["single_point"]["scf"].get("retry_preset"),
                    scf_retry_aliases,
                    scf_retry_presets,
                    "single_point.scf.retry_preset",
                )
            if config["single_point"]["scf"].get("diis_preset") is not None:
                normalize_preset(
                    config["single_point"]["scf"].get("diis_preset"),
                    diis_aliases,
                    diis_presets,
                    "single_point.scf.diis_preset",
                )
    for frequency_key in ("frequency", "freq"):
        if frequency_key in config and config[frequency_key] is not None:
            if not isinstance(config[frequency_key], dict):
                raise ValueError(f"Config '{frequency_key}' must be an object.")
            frequency_rules = {
                "dispersion": (is_str, "Config '{name}' must be a string."),
                "dispersion_model": (is_str, "Config '{name}' must be a string."),
                "dispersion_step": (
                    is_positive_number,
                    "Config '{name}' must be a positive number.",
                ),
                "use_chkfile": (is_bool, "Config '{name}' must be a boolean."),
            }
            _validate_fields(config[frequency_key], frequency_rules, prefix=f"{frequency_key}.")
    if "ts_quality" in config and config["ts_quality"] is not None:
        if not isinstance(config["ts_quality"], dict):
            raise ValueError("Config 'ts_quality' must be an object.")
        ts_rules = {
            "expected_imaginary_count": (
                is_int,
                "Config '{name}' must be an integer.",
            ),
            "imaginary_frequency_min_abs": (
                is_number,
                "Config '{name}' must be a number (int or float).",
            ),
            "imaginary_frequency_max_abs": (
                is_number,
                "Config '{name}' must be a number (int or float).",
            ),
            "projection_step": (
                is_positive_number,
                "Config '{name}' must be a positive number.",
            ),
            "projection_min_abs": (
                is_number,
                "Config '{name}' must be a number (int or float).",
            ),
            "enforce": (is_bool, "Config '{name}' must be a boolean."),
        }
        _validate_fields(config["ts_quality"], ts_rules, prefix="ts_quality.")
        expected_count = config["ts_quality"].get("expected_imaginary_count")
        if expected_count is not None and expected_count < 0:
            raise ValueError("Config 'ts_quality.expected_imaginary_count' must be >= 0.")
        min_abs = config["ts_quality"].get("imaginary_frequency_min_abs")
        max_abs = config["ts_quality"].get("imaginary_frequency_max_abs")
        if min_abs is not None and min_abs < 0:
            raise ValueError("Config 'ts_quality.imaginary_frequency_min_abs' must be >= 0.")
        if max_abs is not None and max_abs < 0:
            raise ValueError("Config 'ts_quality.imaginary_frequency_max_abs' must be >= 0.")
        if min_abs is not None and max_abs is not None and min_abs > max_abs:
            raise ValueError(
                "Config 'ts_quality.imaginary_frequency_min_abs' must be <= "
                "'ts_quality.imaginary_frequency_max_abs'."
            )
        internal_coords = config["ts_quality"].get("internal_coordinates")
        if internal_coords is not None:
            if not isinstance(internal_coords, list):
                raise ValueError("Config 'ts_quality.internal_coordinates' must be a list.")
            for idx, coord in enumerate(internal_coords):
                if not isinstance(coord, dict):
                    raise ValueError(
                        f"Config 'ts_quality.internal_coordinates[{idx}]' must be an object."
                    )
                coord_type = coord.get("type")
                if coord_type not in ("bond", "angle", "dihedral"):
                    raise ValueError(
                        "Config 'ts_quality.internal_coordinates[{idx}].type' "
                        "must be one of: bond, angle, dihedral.".format(idx=idx)
                    )
                required_indices = {
                    "bond": ("i", "j"),
                    "angle": ("i", "j", "k"),
                    "dihedral": ("i", "j", "k", "l"),
                }
                for key in required_indices[coord_type]:
                    value = coord.get(key)
                    if not is_int(value) or value < 0:
                        raise ValueError(
                            "Config 'ts_quality.internal_coordinates[{idx}].{key}' "
                            "must be an integer >= 0.".format(idx=idx, key=key)
                        )
                target = coord.get("target")
                if target is not None and not is_number(target):
                    raise ValueError(
                        "Config 'ts_quality.internal_coordinates[{idx}].target' "
                        "must be a number.".format(idx=idx)
                    )
                if coord.get("direction") is not None:
                    if coord.get("direction") not in ("increase", "decrease"):
                        raise ValueError(
                            "Config 'ts_quality.internal_coordinates[{idx}].direction' "
                            "must be 'increase' or 'decrease'.".format(idx=idx)
                        )
                if coord.get("target") is None and coord.get("direction") is None:
                    raise ValueError(
                        "Config 'ts_quality.internal_coordinates[{idx}]' must define "
                        "'target' or 'direction'.".format(idx=idx)
                    )
                tolerance = coord.get("tolerance")
                if tolerance is not None and tolerance < 0:
                    raise ValueError(
                        "Config 'ts_quality.internal_coordinates[{idx}].tolerance' "
                        "must be >= 0.".format(idx=idx)
                    )
    if "irc" in config and config["irc"] is not None:
        if not isinstance(config["irc"], dict):
            raise ValueError("Config 'irc' must be an object.")
        irc_rules = {
            "steps": (is_positive_int, "Config '{name}' must be a positive integer."),
            "step_size": (is_positive_number, "Config '{name}' must be a positive number."),
            "force_threshold": (is_positive_number, "Config '{name}' must be a positive number."),
        }
        _validate_fields(config["irc"], irc_rules, prefix="irc.")
    if "thermo" in config and config["thermo"] is not None:
        if not isinstance(config["thermo"], dict):
            raise ValueError("Config 'thermo' must be an object.")
        thermo_rules = {
            "T": (is_number, "Config '{name}' must be a number (int or float)."),
            "P": (is_number, "Config '{name}' must be a number (int or float)."),
            "unit": (is_str, "Config '{name}' must be a string."),
        }
        _validate_fields(config["thermo"], thermo_rules, prefix="thermo.")
        temperature = config["thermo"].get("T")
        pressure = config["thermo"].get("P")
        if temperature is None or pressure is None or config["thermo"].get("unit") is None:
            raise ValueError(
                "Config 'thermo' must define 'T', 'P', and 'unit'. "
                "Example: \"thermo\": {\"T\": 298.15, \"P\": 1.0, \"unit\": \"atm\"}."
            )
        if temperature <= 0:
            raise ValueError("Config 'thermo.T' must be a positive number.")
        if pressure <= 0:
            raise ValueError("Config 'thermo.P' must be a positive number.")
        unit = config["thermo"].get("unit")
        if unit not in ("atm", "bar", "Pa"):
            raise ValueError(
                "Config 'thermo.unit' must be one of: atm, bar, Pa. "
                "Example: \"thermo\": {\"T\": 298.15, \"P\": 1.0, \"unit\": \"atm\"}."
            )
    if "io" in config and config["io"] is not None:
        if not isinstance(config["io"], dict):
            raise ValueError("Config 'io' must be an object.")
        io_rules = {
            "write_interval_steps": (
                is_positive_int,
                "Config '{name}' must be a positive integer.",
            ),
            "write_interval_seconds": (
                is_positive_number,
                "Config '{name}' must be a positive number.",
            ),
            "scan_write_interval_points": (
                is_positive_int,
                "Config '{name}' must be a positive integer.",
            ),
        }
        _validate_fields(config["io"], io_rules, prefix="io.")
    if "constraints" in config and config["constraints"] is not None:
        if not isinstance(config["constraints"], dict):
            raise ValueError("Config 'constraints' must be an object.")
        constraints = config["constraints"]

        def _validate_constraint_list(list_name, items, index_keys, value_key, value_label):
            if not isinstance(items, list):
                raise ValueError(
                    "Config 'constraints.{name}' must be a list.".format(name=list_name)
                )
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError(
                        "Config 'constraints.{name}[{idx}]' must be an object.".format(
                            name=list_name, idx=idx
                        )
                    )
                for key in index_keys:
                    if key not in item:
                        raise ValueError(
                            "Config 'constraints.{name}[{idx}]' must define '{key}'.".format(
                                name=list_name, idx=idx, key=key
                            )
                        )
                    if not is_int(item[key]):
                        raise ValueError(
                            "Config 'constraints.{name}[{idx}].{key}' must be an integer.".format(
                                name=list_name, idx=idx, key=key
                            )
                        )
                    if item[key] < 0:
                        raise ValueError(
                            "Config 'constraints.{name}[{idx}].{key}' must be >= 0.".format(
                                name=list_name, idx=idx, key=key
                            )
                        )
                if value_key not in item:
                    raise ValueError(
                        "Config 'constraints.{name}[{idx}]' must define '{key}'.".format(
                            name=list_name, idx=idx, key=value_key
                        )
                    )
                value = item[value_key]
                if not is_number(value):
                    raise ValueError(
                        "Config 'constraints.{name}[{idx}].{key}' must be a number.".format(
                            name=list_name, idx=idx, key=value_key
                        )
                    )
                if value_label == "length" and value <= 0:
                    raise ValueError(
                        "Config 'constraints.{name}[{idx}].{key}' must be > 0 (Angstrom).".format(
                            name=list_name, idx=idx, key=value_key
                        )
                    )
                if value_label == "angle" and not (0 < value <= 180):
                    raise ValueError(
                        "Config 'constraints.{name}[{idx}].{key}' must be between 0 and 180 degrees.".format(
                            name=list_name, idx=idx, key=value_key
                        )
                    )
                if value_label == "dihedral" and not (-180 <= value <= 180):
                    raise ValueError(
                        "Config 'constraints.{name}[{idx}].{key}' must be between -180 and 180 degrees.".format(
                            name=list_name, idx=idx, key=value_key
                        )
                    )

        bonds = constraints.get("bonds")
        if bonds is not None:
            _validate_constraint_list("bonds", bonds, ("i", "j"), "length", "length")
        angles = constraints.get("angles")
        if angles is not None:
            _validate_constraint_list("angles", angles, ("i", "j", "k"), "angle", "angle")
        dihedrals = constraints.get("dihedrals")
        if dihedrals is not None:
            _validate_constraint_list(
                "dihedrals", dihedrals, ("i", "j", "k", "l"), "dihedral", "dihedral"
            )
    scan = config.get("scan")
    scan2d = config.get("scan2d")
    if scan is not None and scan2d is not None:
        raise ValueError("Config must not define both 'scan' and 'scan2d'.")
    if scan is not None:
        _validate_scan_config(scan, "scan")
    if scan2d is not None:
        _validate_scan_config(scan2d, "scan2d", require_dimensions=True)
    normalized_mode = normalize_calc_mode(config.get("calculation_mode"))
    if normalized_mode == "scan":
        if scan is None and scan2d is None:
            raise ValueError("Config 'calculation_mode' is 'scan' but no scan block exists.")
    elif scan is not None or scan2d is not None:
        raise ValueError("Config 'scan' requires 'calculation_mode' to be 'scan'.")


def build_run_config(config):
    validate_run_config(config)
    return RunConfig.from_dict(config)
