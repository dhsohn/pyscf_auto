import json
import os
import re

from jsonschema import Draft7Validator

DEFAULT_CHARGE = 0
DEFAULT_SPIN = None
DEFAULT_MULTIPLICITY = None
DEFAULT_CONFIG_PATH = "run_config.json"
DEFAULT_SOLVENT_MAP_PATH = "solvent_dielectric.json"
DEFAULT_THREAD_COUNT = None
DEFAULT_LOG_PATH = "log/run.log"
DEFAULT_EVENT_LOG_PATH = "log/run_events.jsonl"
DEFAULT_OPTIMIZED_XYZ_PATH = "optimized.xyz"
DEFAULT_FREQUENCY_PATH = "frequency_result.json"
DEFAULT_RUN_METADATA_PATH = "metadata.json"
DEFAULT_QUEUE_PATH = "runs/queue.json"
DEFAULT_QUEUE_LOCK_PATH = "runs/queue.lock"
DEFAULT_QUEUE_RUNNER_LOCK_PATH = "runs/queue.runner.lock"
DEFAULT_QUEUE_RUNNER_LOG_PATH = "log/queue_runner.log"

RUN_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["threads", "memory_gb", "basis", "xc", "solvent"],
    "properties": {
        "threads": {"type": "integer", "minimum": 1},
        "memory_gb": {"type": ["number", "integer"], "exclusiveMinimum": 0},
        "basis": {"type": "string", "minLength": 1},
        "xc": {"type": "string", "minLength": 1},
        "solvent": {"type": "string", "minLength": 1},
        "solvent_model": {"type": ["string", "null"], "enum": ["pcm", "smd", None]},
        "dispersion": {"type": ["string", "null"], "enum": ["d3bj", "d3zero", "d4", None]},
        "calculation_mode": {
            "type": "string",
            "enum": ["optimization", "single_point", "frequency"],
        },
    },
    "additionalProperties": True,
}

RUN_CONFIG_EXAMPLES = {
    "threads": "\"threads\": 4",
    "memory_gb": "\"memory_gb\": 8",
    "basis": "\"basis\": \"def2-svp\"",
    "xc": "\"xc\": \"b3lyp\"",
    "solvent": "\"solvent\": \"water\"",
    "solvent_model": "\"solvent_model\": \"pcm\"",
    "dispersion": "\"dispersion\": \"d3bj\"",
    "calculation_mode": "\"calculation_mode\": \"optimization\"",
}


def load_run_config(config_path):
    if not config_path:
        return {}, None
    if not os.path.isfile(config_path):
        default_missing = os.path.basename(str(config_path)) == DEFAULT_CONFIG_PATH
        example_configs = [
            name
            for name in ("run_config_ase.json", "run_config_ts.json")
            if os.path.isfile(name)
        ]
        example_hint = ", ".join(example_configs) if example_configs else "run_config_ase.json"
        if default_missing:
            message = (
                f"Default config '{DEFAULT_CONFIG_PATH}' was not found at '{config_path}'. "
                f"Copy an example config (e.g., {example_hint}) to {DEFAULT_CONFIG_PATH}, "
                "or pass --config with a valid JSON file."
            )
        else:
            message = (
                f"Config file not found: '{config_path}'. "
                f"Use --config with a valid JSON file (examples: {example_hint})."
            )
        raise FileNotFoundError(message)
    with open(config_path, "r", encoding="utf-8") as config_file:
        raw_config = config_file.read()
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError as error:
            raise ValueError(_format_json_decode_error(config_path, error)) from error
    return config, raw_config


def load_solvent_map(map_path):
    if not map_path:
        return {}
    with open(map_path, "r", encoding="utf-8") as map_file:
        try:
            return json.load(map_file)
        except json.JSONDecodeError as error:
            raise ValueError(_format_json_decode_error(map_path, error)) from error


def _format_json_decode_error(path, error):
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
            "파일에 JSON 객체가 두 개 이상 있음."
        )
    return f"Failed to parse JSON file '{path}' ({location}): {message}"


def _validate_fields(config, rules, prefix=""):
    for key, (predicate, message) in rules.items():
        if key in config and config[key] is not None:
            if not predicate(config[key]):
                name = f"{prefix}{key}"
                raise ValueError(message.format(name=name))


def _schema_example_for_path(path):
    if not path:
        return "See run_config_ase.json for a complete example."
    if path in RUN_CONFIG_EXAMPLES:
        return RUN_CONFIG_EXAMPLES[path]
    leaf = path.split(".")[-1]
    return RUN_CONFIG_EXAMPLES.get(leaf, "See run_config_ase.json for a complete example.")


def _format_schema_error(error):
    path = ".".join(str(part) for part in error.absolute_path)
    if error.validator == "required":
        missing = None
        if isinstance(error.params, dict):
            missing = error.params.get("required")
        if isinstance(missing, (list, tuple)):
            missing = missing[0] if missing else None
        if not missing:
            match = re.search(r"'(.+?)' is a required property", error.message)
            if match:
                missing = match.group(1)
        key_path = f"{path}.{missing}" if path and missing else (missing or path or "(root)")
        cause = "Missing required key."
        example = _schema_example_for_path(missing or path)
    elif error.validator == "enum":
        key_path = path or "(root)"
        allowed_values = ", ".join(repr(value) for value in error.validator_value)
        cause = f"Invalid value. Allowed values: {allowed_values}."
        example = _schema_example_for_path(path)
    else:
        key_path = path or "(root)"
        cause = error.message
        example = _schema_example_for_path(path)
    return f"Config schema error at '{key_path}': {cause} Example: {example}"


def _validate_schema(config):
    validator = Draft7Validator(RUN_CONFIG_SCHEMA)
    errors = sorted(validator.iter_errors(config), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError(_format_schema_error(errors[0]))


def validate_run_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")
    _validate_schema(config)

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

    def is_positive_int(value):
        return is_int(value) and value > 0

    def is_positive_number(value):
        return is_number(value) and value > 0

    validation_rules = {
        "threads": (is_positive_int, "Config '{name}' must be a positive integer."),
        "memory_gb": (is_positive_number, "Config '{name}' must be a positive number (int or float)."),
        "enforce_os_memory_limit": (is_bool, "Config '{name}' must be a boolean."),
        "verbose": (is_bool, "Config '{name}' must be a boolean."),
        "single_point_enabled": (is_bool, "Config '{name}' must be a boolean."),
        "calculation_mode": (is_str, "Config '{name}' must be a string."),
        "log_file": (is_str, "Config '{name}' must be a string path."),
        "event_log_file": (is_str, "Config '{name}' must be a string path."),
        "optimized_xyz_file": (is_str, "Config '{name}' must be a string path."),
        "run_metadata_file": (is_str, "Config '{name}' must be a string path."),
        "frequency_file": (is_str, "Config '{name}' must be a string path."),
        "basis": (is_str, "Config '{name}' must be a string."),
        "xc": (is_str, "Config '{name}' must be a string."),
        "solvent": (is_str, "Config '{name}' must be a string."),
        "solvent_model": (is_str, "Config '{name}' must be a string."),
        "solvent_map": (is_str, "Config '{name}' must be a string path."),
        "dispersion": (is_str, "Config '{name}' must be a string."),
    }
    _validate_fields(config, validation_rules)
    for required_key, example_value in (
        ("basis", '"def2-svp"'),
        ("xc", '"b3lyp"'),
    ):
        if config.get(required_key) in (None, ""):
            raise ValueError(
                "Config '{name}' is required. Example: {example}.".format(
                    name=required_key,
                    example=f"\"{required_key}\": {example_value}",
                )
            )
    solvent_model = config.get("solvent_model")
    if solvent_model:
        if not isinstance(solvent_model, str):
            raise ValueError("Config 'solvent_model' must be a string.")
        if solvent_model.lower() not in ("pcm", "smd"):
            raise ValueError(
                "Config 'solvent_model' must be one of: pcm, smd. "
                "Example: \"solvent_model\": \"pcm\"."
            )
        if not config.get("solvent"):
            raise ValueError(
                "Config 'solvent' is required when 'solvent_model' is set. "
                "Example: \"solvent\": \"water\"."
            )
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
            d3_backend = ase_config.get("d3_backend") or ase_config.get("dftd3_backend")
            d3_command = ase_config.get("d3_command") or ase_config.get("dftd3_command")
            d3_command_validate = ase_config.get("d3_command_validate", True)
            if "d3_command_validate" in ase_config and not isinstance(
                d3_command_validate, bool
            ):
                raise ValueError("Config 'optimizer.ase.d3_command_validate' must be a boolean.")
            if d3_command == "/path/to/dftd3":
                raise ValueError(
                    "Config 'optimizer.ase.d3_command' uses the placeholder '/path/to/dftd3'. "
                    "Replace it with null and set \"d3_backend\": \"dftd3\" (recommended), or "
                    "provide a real executable path such as \"/usr/local/bin/dftd3\". "
                    "Recommended: \"d3_backend\": \"dftd3\", \"d3_command\": null."
                )
            normalized_backend = None
            if d3_backend is not None:
                if not isinstance(d3_backend, str):
                    raise ValueError("Config 'optimizer.ase.d3_backend' must be a string.")
                normalized_backend = d3_backend.strip().lower()
                if normalized_backend not in ("dftd3", "ase"):
                    raise ValueError(
                        "Config 'optimizer.ase.d3_backend' must be one of: dftd3, ase. "
                        "Example: \"d3_backend\": \"dftd3\"."
                    )
            if normalized_backend == "dftd3":
                if d3_command not in (None, ""):
                    raise ValueError(
                        "Config 'optimizer.ase.d3_command' must be null or unset when "
                        "'optimizer.ase.d3_backend' is 'dftd3'. Recommended: "
                        "\"d3_backend\": \"dftd3\", \"d3_command\": null."
                    )
            if normalized_backend == "ase":
                if not d3_command:
                    raise ValueError(
                        "Config 'optimizer.ase.d3_command' is required when "
                        "'optimizer.ase.d3_backend' is 'ase'. Example: \"d3_command\": "
                        "\"/usr/local/bin/dftd3\". Recommended: \"d3_backend\": \"dftd3\", "
                        "\"d3_command\": null."
                    )
                if not isinstance(d3_command, str):
                    raise ValueError("Config 'optimizer.ase.d3_command' must be a string path.")
                if d3_command_validate and not (
                    os.path.isfile(d3_command) and os.access(d3_command, os.X_OK)
                ):
                    raise ValueError(
                        "Config 'optimizer.ase.d3_command' must point to an executable file. "
                        "Example: \"d3_command\": \"/usr/local/bin/dftd3\". Recommended: "
                        "\"d3_backend\": \"dftd3\", \"d3_command\": null."
                    )
            d3_params = ase_config.get("d3_params")
            dftd3_params = ase_config.get("dftd3_params")
            if d3_params is not None and dftd3_params is not None:
                raise ValueError(
                    "Config must not define both 'optimizer.ase.d3_params' and "
                    "'optimizer.ase.dftd3_params'."
                )
            chosen_params = d3_params if d3_params is not None else dftd3_params
            if chosen_params is not None:
                if not isinstance(chosen_params, dict):
                    raise ValueError(
                        "Config 'optimizer.ase.d3_params' must be a JSON object."
                    )
                for key, value in chosen_params.items():
                    if key == "damping" and isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey in ("damping", "variant", "method"):
                                if not isinstance(subvalue, str):
                                    raise ValueError(
                                        "Config 'optimizer.ase.d3_params.damping.{name}' "
                                        "must be a string.".format(name=subkey)
                                    )
                                continue
                            if subkey == "parameters" and isinstance(subvalue, dict):
                                for param_key, param_value in subvalue.items():
                                    if not is_number(param_value):
                                        raise ValueError(
                                            "Config 'optimizer.ase.d3_params.damping.parameters.{name}' "
                                            "must be a number.".format(name=param_key)
                                        )
                                continue
                            if not is_number(subvalue):
                                raise ValueError(
                                    "Config 'optimizer.ase.d3_params.damping.{name}' "
                                    "must be a number.".format(name=subkey)
                                )
                        continue
                    if key == "parameters" and isinstance(value, dict):
                        for param_key, param_value in value.items():
                            if not is_number(param_value):
                                raise ValueError(
                                    "Config 'optimizer.ase.d3_params.parameters.{name}' "
                                    "must be a number.".format(name=param_key)
                                )
                        continue
                    if not is_number(value):
                        raise ValueError(
                            "Config 'optimizer.ase.d3_params.{name}' must be a number.".format(
                                name=key
                            )
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
            "force_restricted": (is_bool, "Config '{name}' must be a boolean."),
            "force_unrestricted": (is_bool, "Config '{name}' must be a boolean."),
        }
        _validate_fields(config["scf"], scf_validation_rules, prefix="scf.")
        if config["scf"].get("force_restricted") and config["scf"].get("force_unrestricted"):
            raise ValueError(
                "Config 'scf' must not set both 'force_restricted' and 'force_unrestricted'."
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
        if "scf" in config["single_point"] and config["single_point"]["scf"] is not None:
            if not isinstance(config["single_point"]["scf"], dict):
                raise ValueError("Config 'single_point.scf' must be an object.")
            scf_validation_rules = {
                "max_cycle": (is_int, "Config '{name}' must be an integer."),
                "conv_tol": (is_number, "Config '{name}' must be a number (int or float)."),
                "level_shift": (is_number, "Config '{name}' must be a number (int or float)."),
                "damping": (is_number, "Config '{name}' must be a number (int or float)."),
                "diis": (is_diis, "Config '{name}' must be a boolean or integer."),
                "force_restricted": (is_bool, "Config '{name}' must be a boolean."),
                "force_unrestricted": (is_bool, "Config '{name}' must be a boolean."),
            }
            _validate_fields(config["single_point"]["scf"], scf_validation_rules, prefix="single_point.scf.")
            if (
                config["single_point"]["scf"].get("force_restricted")
                and config["single_point"]["scf"].get("force_unrestricted")
            ):
                raise ValueError(
                    "Config 'single_point.scf' must not set both 'force_restricted' and "
                    "'force_unrestricted'."
                )
    for frequency_key in ("frequency", "freq"):
        if frequency_key in config and config[frequency_key] is not None:
            if not isinstance(config[frequency_key], dict):
                raise ValueError(f"Config '{frequency_key}' must be an object.")
            frequency_rules = {
                "dispersion": (is_str, "Config '{name}' must be a string."),
            }
            _validate_fields(config[frequency_key], frequency_rules, prefix=f"{frequency_key}.")
