"""Parse .inp files into InpConfig for pyscf_auto."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .route_line import RouteLineResult, parse_route_line
from .geometry import parse_geometry_block


@dataclass
class InpConfig:
    """Fully parsed .inp file contents, ready for conversion to RunConfig."""

    # From route line
    job_type: str  # "optimization", "single_point", "frequency", "irc", "scan"
    functional: str
    basis: str
    charge: int
    multiplicity: int
    atom_spec: str  # PySCF atom specification string
    xyz_source: str  # "inline" or path to .xyz file
    dispersion: str | None = None
    solvent_model: str | None = None
    solvent_name: str | None = None
    optimizer_mode: str | None = None  # "minimum", "transition_state"

    # From optional blocks
    scf: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    thermo: dict[str, Any] = field(default_factory=dict)
    frequency: dict[str, Any] = field(default_factory=dict)
    irc: dict[str, Any] = field(default_factory=dict)
    ts_quality: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    scan: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None

    # Extra stage flags from route line
    frequency_enabled: bool = False
    single_point_enabled: bool = False
    irc_enabled: bool = False

    # Provenance
    raw_text: str = ""
    source_path: str | None = None

    # Tag for molecule identification
    tag: str | None = None


def parse_inp_file(inp_path: str | Path) -> InpConfig:
    """Parse a .inp file and return a fully validated InpConfig.

    The .inp format supports:
    - Route line: ``! Opt B3LYP def2-SVP D3BJ PCM(water) +Freq``
    - Parameter blocks: ``%scf ... end``, ``%optimizer ... end``, etc.
    - Geometry: ``* xyz 0 1 ... *`` or ``* xyzfile 0 1 molecule.xyz``
    - Comments: lines starting with ``#``
    - Tag: ``# TAG: <value>`` for molecule identification

    Args:
        inp_path: Path to the .inp file.

    Returns:
        An ``InpConfig`` with all parsed contents.

    Raises:
        ValueError: If the file is malformed or missing required sections.
        FileNotFoundError: If the file does not exist.
    """
    inp_path = Path(inp_path).expanduser().resolve()
    if not inp_path.exists():
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    raw_text = inp_path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    inp_dir = str(inp_path.parent)

    # Extract components
    route_result = _extract_route_line(lines)
    blocks = _extract_parameter_blocks(lines)
    geometry = parse_geometry_block(lines, inp_dir)
    tag = _extract_tag(lines)

    # Determine extra stage flags
    freq_enabled = "freq" in route_result.extra_stages
    sp_enabled = "sp" in route_result.extra_stages
    irc_enabled_flag = "irc" in route_result.extra_stages

    # Determine xyz source
    if geometry.source_type == "xyzfile" and geometry.source_path:
        xyz_source = geometry.source_path
    else:
        xyz_source = "inline"

    return InpConfig(
        job_type=route_result.job_type,
        functional=route_result.functional,
        basis=route_result.basis,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        atom_spec=geometry.atom_spec,
        xyz_source=xyz_source,
        dispersion=route_result.dispersion,
        solvent_model=route_result.solvent_model,
        solvent_name=route_result.solvent_name,
        optimizer_mode=route_result.optimizer_mode,
        scf=blocks.get("scf", {}),
        optimizer=blocks.get("optimizer", {}),
        thermo=blocks.get("thermo", {}),
        frequency=blocks.get("frequency", {}),
        irc=blocks.get("irc", {}),
        ts_quality=blocks.get("ts_quality", {}),
        runtime=blocks.get("runtime", {}),
        scan=blocks.get("scan"),
        constraints=blocks.get("constraints"),
        frequency_enabled=freq_enabled,
        single_point_enabled=sp_enabled,
        irc_enabled=irc_enabled_flag,
        raw_text=raw_text,
        source_path=str(inp_path),
        tag=tag,
    )


def _extract_route_line(lines: list[str]) -> RouteLineResult:
    """Find and parse the route line (starts with '!')."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!"):
            return parse_route_line(stripped)
    raise ValueError("No route line found. Expected a line starting with '!'.")


def _extract_tag(lines: list[str]) -> str | None:
    """Extract TAG from comment: ``# TAG: <value>``."""
    tag_re = re.compile(r"^#\s*TAG\s*:\s*(.+)$", re.IGNORECASE)
    for line in lines:
        m = tag_re.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def _extract_parameter_blocks(lines: list[str]) -> dict[str, dict[str, Any]]:
    """Extract all %block ... end parameter blocks.

    Each block looks like::

        %block_name
          key value
          key value
        end

    Returns a dictionary mapping block names to key-value dictionaries.
    """
    blocks: dict[str, dict[str, Any]] = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("%"):
            block_name = stripped[1:].strip().lower()
            # Normalize aliases
            block_name = _BLOCK_ALIASES.get(block_name, block_name)
            block_lines = []
            i += 1
            while i < len(lines):
                bline = lines[i].strip()
                if bline.lower() == "end":
                    break
                if bline and not bline.startswith("#"):
                    block_lines.append(bline)
                i += 1
            blocks[block_name] = _parse_block_lines(block_name, block_lines)
        i += 1
    return blocks


_BLOCK_ALIASES = {
    "freq": "frequency",
}


def _parse_block_lines(block_name: str, block_lines: list[str]) -> dict[str, Any]:
    """Parse key-value pairs from block lines.

    Supports:
    - ``key value`` (space separated)
    - ``key=value``
    - Nested blocks for constraints (list items)
    """
    result: dict[str, Any] = {}

    for line in block_lines:
        # Handle key=value or key value format
        if "=" in line:
            key, _, value = line.partition("=")
        else:
            parts = line.split(None, 1)
            if len(parts) == 1:
                key, value = parts[0], "true"
            else:
                key, value = parts

        key = key.strip().lower()
        value = value.strip()

        # Type coercion
        result[key] = _coerce_value(value)

    return result


def _coerce_value(value: str) -> Any:
    """Coerce a string value to the appropriate Python type."""
    if not value:
        return None

    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none" or lower == "null":
        return None

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    return value


def inp_config_to_dict(inp: InpConfig) -> dict[str, Any]:
    """Convert an InpConfig to a configuration dictionary suitable for RunConfig.

    This is the bridge between the .inp format and the existing
    RunConfig Pydantic model. The returned dict can be passed to
    ``build_run_config()`` from ``run_opt_config.py``.
    """
    config: dict[str, Any] = {
        "basis": inp.basis,
        "xc": inp.functional,
        "calculation_mode": inp.job_type,
    }

    # Solvent (default to "vacuum" for gas-phase calculations)
    config["solvent"] = inp.solvent_name if inp.solvent_name else "vacuum"
    if inp.solvent_model:
        config["solvent_model"] = inp.solvent_model

    # Dispersion
    if inp.dispersion:
        config["dispersion"] = inp.dispersion

    # Runtime
    if inp.runtime.get("threads"):
        config["threads"] = inp.runtime["threads"]
    if inp.runtime.get("memory_gb"):
        config["memory_gb"] = inp.runtime["memory_gb"]

    # SCF config
    if inp.scf:
        config["scf"] = dict(inp.scf)

    # Optimizer config
    if inp.optimizer or inp.optimizer_mode:
        opt_config: dict[str, Any] = {}
        if inp.optimizer_mode:
            opt_config["mode"] = inp.optimizer_mode
        if inp.optimizer:
            # Map .inp optimizer keys to RunConfig optimizer keys
            ase_keys = {"fmax", "steps", "optimizer", "trajectory"}
            ase_config = {}
            for k, v in inp.optimizer.items():
                if k == "mode":
                    opt_config["mode"] = v
                elif k in ase_keys:
                    ase_config[k] = v
                else:
                    opt_config[k] = v
            if ase_config:
                opt_config["ase"] = ase_config
        config["optimizer"] = opt_config

    # Thermo config - needs T, P, unit to pass validation
    if inp.thermo:
        thermo_config: dict[str, Any] = {}
        for k, v in inp.thermo.items():
            if k == "temperature":
                thermo_config["T"] = v
            elif k == "pressure":
                thermo_config["P"] = v
            else:
                thermo_config[k] = v
        # Ensure defaults for required fields
        thermo_config.setdefault("T", 298.15)
        thermo_config.setdefault("P", 1.0)
        thermo_config.setdefault("unit", "atm")
        config["thermo"] = thermo_config

    # Frequency config
    if inp.frequency:
        config["frequency"] = dict(inp.frequency)

    # IRC config
    if inp.irc:
        config["irc"] = dict(inp.irc)

    # TS quality config
    if inp.ts_quality:
        config["ts_quality"] = dict(inp.ts_quality)

    # Scan config
    if inp.scan:
        config["scan"] = dict(inp.scan)

    # Constraints
    if inp.constraints:
        config["constraints"] = dict(inp.constraints)

    # Stage flags
    if inp.frequency_enabled:
        config["frequency_enabled"] = True
    if inp.single_point_enabled:
        config["single_point_enabled"] = True
    if inp.irc_enabled:
        config["irc_enabled"] = True

    # Spin mode (multiplicity is provided, so use auto)
    config["spin_mode"] = "auto"

    return config


def inp_config_to_xyz_content(inp: InpConfig) -> str:
    """Generate a temporary .xyz file content from an InpConfig.

    This is used when the geometry is specified inline in the .inp file.
    The generated content follows the standard XYZ format with charge/spin
    in the comment line.
    """
    atom_lines = inp.atom_spec.strip().split("\n")
    atom_count = len(atom_lines)
    spin = inp.multiplicity - 1
    comment = f"pyscf_auto generated. charge={inp.charge} spin={spin}"
    return f"{atom_count}\n{comment}\n{inp.atom_spec}\n"
