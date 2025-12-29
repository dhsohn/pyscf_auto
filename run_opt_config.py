import json
import os

DEFAULT_CHARGE = 0
DEFAULT_SPIN = None
DEFAULT_MULTIPLICITY = None
DEFAULT_CONFIG_PATH = "run_config.json"
DEFAULT_SOLVENT_MAP_PATH = "solvent_dielectric.json"
DEFAULT_THREAD_COUNT = None
DEFAULT_LOG_PATH = "log/run.log"
DEFAULT_EVENT_LOG_PATH = "log/run_events.jsonl"
DEFAULT_OPTIMIZED_XYZ_PATH = "optimized.xyz"
DEFAULT_FREQUENCY_PATH = "frequency.json"
DEFAULT_RUN_METADATA_PATH = "metadata.json"
DEFAULT_QUEUE_PATH = "runs/queue.json"
DEFAULT_QUEUE_LOCK_PATH = "runs/queue.lock"
DEFAULT_QUEUE_RUNNER_LOCK_PATH = "runs/queue.runner.lock"
DEFAULT_QUEUE_RUNNER_LOG_PATH = "log/queue_runner.log"


def load_run_config(config_path):
    if not config_path:
        return {}, None
    with open(config_path, "r", encoding="utf-8") as config_file:
        raw_config = config_file.read()
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError as error:
            location = f"line {error.lineno} column {error.colno}"
            raise ValueError(
                f"Failed to parse JSON config '{config_path}' ({location}): {error.msg}"
            ) from error
    return config, raw_config


def load_solvent_map(map_path):
    if not map_path:
        return {}
    with open(map_path, "r", encoding="utf-8") as map_file:
        return json.load(map_file)


def _validate_fields(config, rules, prefix=""):
    for key, (predicate, message) in rules.items():
        if key in config and config[key] is not None:
            if not predicate(config[key]):
                name = f"{prefix}{key}"
                raise ValueError(message.format(name=name))


def validate_run_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")
    is_int = lambda value: isinstance(value, int) and not isinstance(value, bool)
    is_number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
    is_bool = lambda value: isinstance(value, bool)
    is_str = lambda value: isinstance(value, str)
    is_diis = lambda value: isinstance(value, (bool, int))
    is_positive_int = lambda value: is_int(value) and value > 0
    is_positive_number = lambda value: is_number(value) and value > 0
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
            if d3_command == "/path/to/dftd3":
                raise ValueError(
                    "Config 'optimizer.ase.d3_command' uses the placeholder '/path/to/dftd3'. "
                    "Replace it with null and set \"d3_backend\": \"dftd3\", or provide a "
                    "real executable path such as \"/usr/local/bin/dftd3\"."
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
                        "'optimizer.ase.d3_backend' is 'dftd3'. Example: \"d3_command\": null."
                    )
            if normalized_backend == "ase":
                if not d3_command:
                    raise ValueError(
                        "Config 'optimizer.ase.d3_command' is required when "
                        "'optimizer.ase.d3_backend' is 'ase'. Example: \"d3_command\": "
                        "\"/usr/local/bin/dftd3\"."
                    )
                if not isinstance(d3_command, str):
                    raise ValueError("Config 'optimizer.ase.d3_command' must be a string path.")
                if not (os.path.isfile(d3_command) and os.access(d3_command, os.X_OK)):
                    raise ValueError(
                        "Config 'optimizer.ase.d3_command' must point to an executable file. "
                        "Example: \"d3_command\": \"/usr/local/bin/dftd3\"."
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
