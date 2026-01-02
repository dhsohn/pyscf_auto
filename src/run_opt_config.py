import json
import os
import re
import tomllib
from importlib import resources
from importlib.resources.abc import Traversable
from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import Draft7Validator
import yaml

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
DEFAULT_IRC_PATH = "irc_result.json"
DEFAULT_IRC_PROFILE_CSV_PATH = "irc_profile.csv"
DEFAULT_RUN_METADATA_PATH = "metadata.json"
DEFAULT_QCSCHEMA_OUTPUT_PATH = "qcschema_result.json"
DEFAULT_SCAN_RESULT_PATH = "scan_result.json"
DEFAULT_SCAN_RESULT_CSV_PATH = "scan_result.csv"
DEFAULT_QUEUE_PATH = "runs/queue.json"
DEFAULT_QUEUE_LOCK_PATH = "runs/queue.lock"
DEFAULT_QUEUE_RUNNER_LOCK_PATH = "runs/queue.runner.lock"
DEFAULT_QUEUE_RUNNER_LOG_PATH = "log/queue_runner.log"

SCAN_DIMENSION_SCHEMA = {
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {"type": "string", "enum": ["bond", "angle", "dihedral"]},
        "i": {"type": "integer", "minimum": 0},
        "j": {"type": "integer", "minimum": 0},
        "k": {"type": "integer", "minimum": 0},
        "l": {"type": "integer", "minimum": 0},
        "start": {"type": ["number", "integer"]},
        "end": {"type": ["number", "integer"]},
        "step": {"type": ["number", "integer"]},
    },
    "additionalProperties": True,
}

TS_INTERNAL_COORDINATE_SCHEMA = {
    "type": "object",
    "required": ["type", "i", "j"],
    "properties": {
        "type": {"type": "string", "enum": ["bond", "angle", "dihedral"]},
        "i": {"type": "integer", "minimum": 0},
        "j": {"type": "integer", "minimum": 0},
        "k": {"type": "integer", "minimum": 0},
        "l": {"type": "integer", "minimum": 0},
        "target": {"type": ["number", "integer", "null"]},
        "direction": {"type": ["string", "null"], "enum": ["increase", "decrease", None]},
        "tolerance": {"type": ["number", "integer", "null"], "minimum": 0},
        "label": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

RUN_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["basis", "xc", "solvent"],
    "properties": {
        "threads": {"type": ["integer", "null"], "minimum": 1},
        "memory_gb": {
            "type": ["number", "integer", "null"],
            "exclusiveMinimum": 0,
        },
        "basis": {"type": "string", "minLength": 1},
        "xc": {"type": "string", "minLength": 1},
        "solvent": {"type": "string", "minLength": 1},
        "solvent_model": {"type": ["string", "null"], "enum": ["pcm", "smd", None]},
        "dispersion": {"type": ["string", "null"], "enum": ["d3bj", "d3zero", "d4", None]},
        "calculation_mode": {
            "type": "string",
            "enum": ["optimization", "single_point", "frequency", "irc", "scan"],
        },
        "irc_enabled": {"type": ["boolean", "null"]},
        "irc_file": {"type": ["string", "null"]},
        "irc_profile_csv_file": {"type": ["string", "null"]},
        "scan_result_csv_file": {"type": ["string", "null"]},
        "qcschema_output_file": {"type": ["string", "null"]},
        "irc": {
            "type": ["object", "null"],
            "required": [],
            "properties": {
                "steps": {"type": ["integer", "null"], "minimum": 1},
                "step_size": {
                    "type": ["number", "integer", "null"],
                    "exclusiveMinimum": 0,
                },
                "force_threshold": {
                    "type": ["number", "integer", "null"],
                    "exclusiveMinimum": 0,
                },
            },
            "additionalProperties": True,
        },
        "optimizer": {
            "type": ["object", "null"],
            "required": [],
            "properties": {
                "output_xyz": {"type": ["string", "null"]},
                "mode": {"type": ["string", "null"]},
                "ase": {
                    "type": ["object", "null"],
                    "required": [],
                    "properties": {
                        "d3_backend": {
                            "type": ["string", "null"],
                            "enum": ["dftd3", None],
                        },
                        "dftd3_backend": {
                            "type": ["string", "null"],
                            "enum": ["dftd3", None],
                        },
                        "d3_params": {"type": ["object", "null"]},
                        "dftd3_params": {"type": ["object", "null"]},
                        "optimizer": {"type": ["string", "null"]},
                        "fmax": {"type": ["number", "integer", "null"]},
                        "steps": {"type": ["integer", "null"]},
                        "trajectory": {"type": ["string", "null"]},
                        "logfile": {"type": ["string", "null"]},
                        "sella": {"type": ["object", "null"]},
                    },
                    "additionalProperties": True,
                },
            },
            "additionalProperties": True,
        },
        "scf": {
            "type": ["object", "null"],
            "required": [],
            "properties": {
                "max_cycle": {"type": ["integer", "null"]},
                "conv_tol": {"type": ["number", "integer", "null"]},
                "level_shift": {"type": ["number", "integer", "null"]},
                "damping": {"type": ["number", "integer", "null"]},
                "diis": {"type": ["boolean", "integer", "null"]},
                "chkfile": {"type": ["string", "null"]},
                "force_restricted": {"type": ["boolean", "null"]},
                "force_unrestricted": {"type": ["boolean", "null"]},
                "extra": {"type": ["object", "null"]},
            },
            "additionalProperties": True,
        },
        "single_point": {
            "type": ["object", "null"],
            "required": [],
            "properties": {
                "basis": {"type": ["string", "null"]},
                "xc": {"type": ["string", "null"]},
                "solvent": {"type": ["string", "null"]},
                "solvent_model": {"type": ["string", "null"]},
                "solvent_map": {"type": ["string", "null"]},
                "dispersion": {"type": ["string", "null"]},
                "scf": {
                    "type": ["object", "null"],
                    "required": [],
                    "properties": {
                        "max_cycle": {"type": ["integer", "null"]},
                        "conv_tol": {"type": ["number", "integer", "null"]},
                        "level_shift": {"type": ["number", "integer", "null"]},
                        "damping": {"type": ["number", "integer", "null"]},
                        "diis": {"type": ["boolean", "integer", "null"]},
                        "chkfile": {"type": ["string", "null"]},
                        "force_restricted": {"type": ["boolean", "null"]},
                        "force_unrestricted": {"type": ["boolean", "null"]},
                        "extra": {"type": ["object", "null"]},
                    },
                    "additionalProperties": True,
                },
            },
            "additionalProperties": True,
        },
        "frequency": {
            "type": ["object", "null"],
            "required": [],
            "properties": {
                "dispersion": {"type": ["string", "null"]},
                "dispersion_model": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        },
        "freq": {
            "type": ["object", "null"],
            "required": [],
            "properties": {
                "dispersion": {"type": ["string", "null"]},
                "dispersion_model": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        },
        "thermo": {
            "type": "object",
            "required": ["T", "P", "unit"],
            "properties": {
                "T": {"type": ["number", "integer"]},
                "P": {"type": ["number", "integer"]},
                "unit": {"type": "string", "enum": ["atm", "bar", "Pa"]},
            },
        },
        "constraints": {
            "type": ["object", "null"],
            "properties": {
                "bonds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["i", "j", "length"],
                        "properties": {
                            "i": {"type": "integer", "minimum": 0},
                            "j": {"type": "integer", "minimum": 0},
                            "length": {
                                "type": ["number", "integer"],
                                "exclusiveMinimum": 0,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "angles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["i", "j", "k", "angle"],
                        "properties": {
                            "i": {"type": "integer", "minimum": 0},
                            "j": {"type": "integer", "minimum": 0},
                            "k": {"type": "integer", "minimum": 0},
                            "angle": {
                                "type": ["number", "integer"],
                                "minimum": 0,
                                "maximum": 180,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "dihedrals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["i", "j", "k", "l", "dihedral"],
                        "properties": {
                            "i": {"type": "integer", "minimum": 0},
                            "j": {"type": "integer", "minimum": 0},
                            "k": {"type": "integer", "minimum": 0},
                            "l": {"type": "integer", "minimum": 0},
                            "dihedral": {
                                "type": ["number", "integer"],
                                "minimum": -180,
                                "maximum": 180,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "scan": {
            "type": ["object", "null"],
            "properties": {
                "type": {"type": "string", "enum": ["bond", "angle", "dihedral"]},
                "i": {"type": "integer", "minimum": 0},
                "j": {"type": "integer", "minimum": 0},
                "k": {"type": "integer", "minimum": 0},
                "l": {"type": "integer", "minimum": 0},
                "start": {"type": ["number", "integer"]},
                "end": {"type": ["number", "integer"]},
                "step": {"type": ["number", "integer"]},
                "mode": {"type": "string", "enum": ["optimization", "single_point"]},
                "dimensions": {
                    "type": "array",
                    "minItems": 1,
                    "items": SCAN_DIMENSION_SCHEMA,
                },
                "grid": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": ["number", "integer"]},
                    },
                },
            },
            "additionalProperties": True,
        },
        "scan2d": {
            "type": ["object", "null"],
            "properties": {
                "mode": {"type": "string", "enum": ["optimization", "single_point"]},
                "dimensions": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": SCAN_DIMENSION_SCHEMA,
                },
                "grid": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": ["number", "integer"]},
                    },
                },
            },
            "additionalProperties": True,
        },
        "ts_quality": {
            "type": ["object", "null"],
            "required": [],
            "properties": {
                "expected_imaginary_count": {"type": ["integer", "null"], "minimum": 0},
                "imaginary_frequency_min_abs": {
                    "type": ["number", "integer", "null"],
                    "minimum": 0,
                },
                "imaginary_frequency_max_abs": {
                    "type": ["number", "integer", "null"],
                    "minimum": 0,
                },
                "projection_step": {
                    "type": ["number", "integer", "null"],
                    "exclusiveMinimum": 0,
                },
                "projection_min_abs": {
                    "type": ["number", "integer", "null"],
                    "minimum": 0,
                },
                "internal_coordinates": {
                    "type": "array",
                    "items": TS_INTERNAL_COORDINATE_SCHEMA,
                },
            },
            "additionalProperties": True,
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
    "single_point": (
        "\"single_point\": {\"basis\": \"def2-svp\", \"xc\": \"b3lyp\", "
        "\"solvent\": \"water\", \"dispersion\": \"d3bj\"}"
    ),
    "single_point.scf": (
        "\"scf\": {\"max_cycle\": 200, \"conv_tol\": 1e-7, \"diis\": 8, "
        "\"chkfile\": \"scf.chk\", \"extra\": {\"density_fit\": true}}"
    ),
    "frequency": "\"frequency\": {\"dispersion\": \"d3bj\", \"dispersion_model\": \"d3bj\"}",
    "freq": "\"freq\": {\"dispersion\": \"d3bj\", \"dispersion_model\": \"d3bj\"}",
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
    "ts_quality": (
        "\"ts_quality\": {\"expected_imaginary_count\": 1, "
        "\"imaginary_frequency_min_abs\": 50.0, \"imaginary_frequency_max_abs\": 1500.0, "
        "\"projection_min_abs\": 0.01, "
        "\"internal_coordinates\": [{\"type\": \"bond\", \"i\": 0, \"j\": 1, "
        "\"target\": 2.0}]}"
    ),
}


@dataclass(frozen=True)
class SCFConfig:
    raw: dict[str, Any]
    max_cycle: int | None = None
    conv_tol: float | None = None
    level_shift: float | None = None
    damping: float | None = None
    diis: bool | int | None = None
    force_restricted: bool | None = None
    force_unrestricted: bool | None = None
    extra: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SCFConfig | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("Config 'scf' must be an object.")
        return cls(
            raw=dict(data),
            max_cycle=data.get("max_cycle"),
            conv_tol=data.get("conv_tol"),
            level_shift=data.get("level_shift"),
            damping=data.get("damping"),
            diis=data.get("diis"),
            force_restricted=data.get("force_restricted"),
            force_unrestricted=data.get("force_unrestricted"),
            extra=data.get("extra"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class OptimizerASEConfig:
    raw: dict[str, Any]
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
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("Config 'optimizer.ase' must be an object.")
        return cls(
            raw=dict(data),
            d3_backend=data.get("d3_backend"),
            dftd3_backend=data.get("dftd3_backend"),
            d3_params=data.get("d3_params"),
            dftd3_params=data.get("dftd3_params"),
            optimizer=data.get("optimizer"),
            fmax=data.get("fmax"),
            steps=data.get("steps"),
            trajectory=data.get("trajectory"),
            logfile=data.get("logfile"),
            sella=data.get("sella"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class OptimizerConfig:
    raw: dict[str, Any]
    output_xyz: str | None = None
    mode: str | None = None
    ase: OptimizerASEConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "OptimizerConfig | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("Config 'optimizer' must be an object.")
        return cls(
            raw=dict(data),
            output_xyz=data.get("output_xyz"),
            mode=data.get("mode"),
            ase=OptimizerASEConfig.from_dict(data.get("ase")),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class SinglePointConfig:
    raw: dict[str, Any]
    basis: str | None = None
    xc: str | None = None
    solvent: str | None = None
    solvent_model: str | None = None
    solvent_map: str | None = None
    dispersion: str | None = None
    scf: SCFConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SinglePointConfig | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("Config 'single_point' must be an object.")
        return cls(
            raw=dict(data),
            basis=data.get("basis"),
            xc=data.get("xc"),
            solvent=data.get("solvent"),
            solvent_model=data.get("solvent_model"),
            solvent_map=data.get("solvent_map"),
            dispersion=data.get("dispersion"),
            scf=SCFConfig.from_dict(data.get("scf")),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class FrequencyConfig:
    raw: dict[str, Any]
    dispersion: str | None = None
    dispersion_model: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "FrequencyConfig | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("Config 'frequency' must be an object.")
        return cls(
            raw=dict(data),
            dispersion=data.get("dispersion"),
            dispersion_model=data.get("dispersion_model"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class IrcConfig:
    raw: dict[str, Any]
    steps: int | None = None
    step_size: float | None = None
    force_threshold: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "IrcConfig | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("Config 'irc' must be an object.")
        return cls(
            raw=dict(data),
            steps=data.get("steps"),
            step_size=data.get("step_size"),
            force_threshold=data.get("force_threshold"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class TSQualityConfig:
    raw: dict[str, Any]
    expected_imaginary_count: int | None = None
    imaginary_frequency_min_abs: float | None = None
    imaginary_frequency_max_abs: float | None = None
    projection_step: float | None = None
    projection_min_abs: float | None = None
    internal_coordinates: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TSQualityConfig | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("Config 'ts_quality' must be an object.")
        return cls(
            raw=dict(data),
            expected_imaginary_count=data.get("expected_imaginary_count"),
            imaginary_frequency_min_abs=data.get("imaginary_frequency_min_abs"),
            imaginary_frequency_max_abs=data.get("imaginary_frequency_max_abs"),
            projection_step=data.get("projection_step"),
            projection_min_abs=data.get("projection_min_abs"),
            internal_coordinates=data.get("internal_coordinates"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class ThermoConfig:
    raw: dict[str, Any]
    temperature: float | None = None
    pressure: float | None = None
    unit: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ThermoConfig | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("Config 'thermo' must be an object.")
        return cls(
            raw=dict(data),
            temperature=data.get("T"),
            pressure=data.get("P"),
            unit=data.get("unit"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class RunConfig:
    raw: dict[str, Any]
    threads: int | None = None
    memory_gb: float | None = None
    basis: str | None = None
    xc: str | None = None
    solvent: str | None = None
    solvent_model: str | None = None
    dispersion: str | None = None
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
    frequency: FrequencyConfig | None = None
    irc: IrcConfig | None = None
    ts_quality: TSQualityConfig | None = None
    thermo: ThermoConfig | None = None
    constraints: dict[str, Any] | None = None
    scan: dict[str, Any] | None = None
    scan2d: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        if not isinstance(data, dict):
            raise ValueError("Config must be an object/mapping.")
        frequency_block = data.get("frequency")
        if not frequency_block:
            frequency_block = data.get("freq")
        return cls(
            raw=dict(data),
            threads=data.get("threads"),
            memory_gb=data.get("memory_gb"),
            basis=data.get("basis"),
            xc=data.get("xc"),
            solvent=data.get("solvent"),
            solvent_model=data.get("solvent_model"),
            dispersion=data.get("dispersion"),
            calculation_mode=data.get("calculation_mode"),
            enforce_os_memory_limit=data.get("enforce_os_memory_limit"),
            verbose=data.get("verbose"),
            single_point_enabled=data.get("single_point_enabled"),
            frequency_enabled=data.get("frequency_enabled"),
            irc_enabled=data.get("irc_enabled"),
            log_file=data.get("log_file"),
            event_log_file=data.get("event_log_file"),
            optimized_xyz_file=data.get("optimized_xyz_file"),
            run_metadata_file=data.get("run_metadata_file"),
            qcschema_output_file=data.get("qcschema_output_file"),
            frequency_file=data.get("frequency_file"),
            irc_file=data.get("irc_file"),
            irc_profile_csv_file=data.get("irc_profile_csv_file"),
            scan_result_csv_file=data.get("scan_result_csv_file"),
            solvent_map=data.get("solvent_map"),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer")),
            scf=SCFConfig.from_dict(data.get("scf")),
            single_point=SinglePointConfig.from_dict(data.get("single_point")),
            frequency=FrequencyConfig.from_dict(frequency_block),
            irc=IrcConfig.from_dict(data.get("irc")),
            ts_quality=TSQualityConfig.from_dict(data.get("ts_quality")),
            thermo=ThermoConfig.from_dict(data.get("thermo")),
            constraints=data.get("constraints"),
            scan=data.get("scan"),
            scan2d=data.get("scan2d"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


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
    try:
        resource = resources.files(__package__).joinpath(DEFAULT_SOLVENT_MAP_PATH)
    except (ModuleNotFoundError, AttributeError):
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
        raise ValueError("Config must be an object/mapping.")
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
            for backend_key in ("d3_backend", "dftd3_backend"):
                backend_value = ase_config.get(backend_key)
                if backend_value is None:
                    continue
                if not isinstance(backend_value, str):
                    raise ValueError(
                        f"Config 'optimizer.ase.{backend_key}' must be a string."
                    )
                if backend_value.strip().lower() != "dftd3":
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
            "chkfile": (is_str, "Config '{name}' must be a string path."),
            "force_restricted": (is_bool, "Config '{name}' must be a boolean."),
            "force_unrestricted": (is_bool, "Config '{name}' must be a boolean."),
            "extra": (is_dict, "Config '{name}' must be an object."),
        }
        _validate_fields(config["scf"], scf_validation_rules, prefix="scf.")
        _validate_scf_extra(config["scf"].get("extra"), "scf")
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
                "chkfile": (is_str, "Config '{name}' must be a string path."),
                "force_restricted": (is_bool, "Config '{name}' must be a boolean."),
                "force_unrestricted": (is_bool, "Config '{name}' must be a boolean."),
                "extra": (is_dict, "Config '{name}' must be an object."),
            }
            _validate_fields(config["single_point"]["scf"], scf_validation_rules, prefix="single_point.scf.")
            _validate_scf_extra(config["single_point"]["scf"].get("extra"), "single_point.scf")
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
                "dispersion_model": (is_str, "Config '{name}' must be a string."),
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
