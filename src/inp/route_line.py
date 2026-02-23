"""Parse the route line (! keyword line) from a .inp file."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Known job type keywords (case-insensitive match)
_JOB_TYPES = {
    "opt": ("optimization", "minimum"),
    "optts": ("optimization", "transition_state"),
    "sp": ("single_point", None),
    "energy": ("single_point", None),
    "freq": ("frequency", None),
    "irc": ("irc", None),
    "scan": ("scan", None),
}

# Known basis sets (lowercased for matching)
_KNOWN_BASIS_SETS = {
    "sto-3g",
    "3-21g",
    "6-31g",
    "6-31g*",
    "6-31g**",
    "6-31+g",
    "6-31+g*",
    "6-31+g**",
    "6-31++g**",
    "6-311g",
    "6-311g*",
    "6-311g**",
    "6-311+g**",
    "6-311++g**",
    "def2-svp",
    "def2-svpp",
    "def2-tzvp",
    "def2-tzvpp",
    "def2-qzvp",
    "def2-qzvpp",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "cc-pv5z",
    "aug-cc-pvdz",
    "aug-cc-pvtz",
    "aug-cc-pvqz",
}

# Known dispersion models
_DISPERSION_MODELS = {
    "d3bj": "d3bj",
    "d3zero": "d3zero",
    "d4": "d4",
}

# Known functionals (lowercased) - representative set, not exhaustive
_KNOWN_FUNCTIONALS = {
    "b3lyp", "pbe0", "pbe", "bp86", "blyp", "tpss", "m06", "m06-2x",
    "m06-l", "m06-hf", "wb97x-d", "wb97x_d", "wb97x-d3", "wb97x_d3",
    "wb97m-v", "wb97m-d3bj", "cam-b3lyp", "b97-d", "b97-d3",
    "b97m-v", "b97m-d3bj", "revpbe", "rpbe", "hse06", "scan0",
    "r2scan", "r2scan0", "b3pw91", "x3lyp", "tpssh", "mn15",
    "mn15-l", "mn12-sx", "pw6b95", "pw6b95-d3bj",
    "lda", "svwn", "hf",
}

# Solvent pattern: PCM(name) or SMD(name)
_SOLVENT_RE = re.compile(r"^(pcm|smd)\(([^)]+)\)$", re.IGNORECASE)

# Extra stage flags
_EXTRA_STAGE_RE = re.compile(r"^\+(freq|sp|irc)$", re.IGNORECASE)


@dataclass
class RouteLineResult:
    """Result of parsing a route line."""

    job_type: str  # "optimization", "single_point", "frequency", "irc", "scan"
    optimizer_mode: str | None = None  # "minimum", "transition_state"
    functional: str | None = None
    basis: str | None = None
    dispersion: str | None = None
    solvent_model: str | None = None
    solvent_name: str | None = None
    extra_stages: list[str] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)


def _normalize_token(token: str) -> str:
    """Normalize a token for matching (strip, lowercase)."""
    return token.strip().lower()


def _match_basis(token_lower: str) -> str | None:
    """Try to match a token as a basis set, return original-case if matched."""
    if token_lower in _KNOWN_BASIS_SETS:
        return token_lower
    return None


def _match_functional(token_lower: str) -> bool:
    """Check if a token matches a known functional."""
    # Normalize hyphens and underscores for matching
    normalized = re.sub(r"[\s_\-]+", "", token_lower)
    for func in _KNOWN_FUNCTIONALS:
        if re.sub(r"[\s_\-]+", "", func) == normalized:
            return True
    return False


def parse_route_line(line: str) -> RouteLineResult:
    """Parse a route line starting with '!'.

    Format: ``! <JobType> <Functional> <Basis> [Dispersion] [Solvent] [+Freq] [+SP] [+IRC]``

    Returns a ``RouteLineResult`` with all parsed components.
    Raises ``ValueError`` if required components (job type, functional, basis) are missing.
    """
    text = line.strip()
    if text.startswith("!"):
        text = text[1:].strip()

    tokens = text.split()
    if not tokens:
        raise ValueError("Route line is empty.")

    result = RouteLineResult(job_type="", functional=None, basis=None)

    for token in tokens:
        token_lower = _normalize_token(token)

        # Job type
        if token_lower in _JOB_TYPES and not result.job_type:
            mode, opt_mode = _JOB_TYPES[token_lower]
            result.job_type = mode
            result.optimizer_mode = opt_mode
            continue

        # Dispersion
        if token_lower in _DISPERSION_MODELS and result.dispersion is None:
            result.dispersion = _DISPERSION_MODELS[token_lower]
            continue

        # Solvent: PCM(water) or SMD(DMSO)
        solvent_match = _SOLVENT_RE.match(token)
        if solvent_match and result.solvent_model is None:
            result.solvent_model = solvent_match.group(1).lower()
            result.solvent_name = solvent_match.group(2).strip()
            continue

        # Extra stage flags: +Freq, +SP, +IRC
        extra_match = _EXTRA_STAGE_RE.match(token)
        if extra_match:
            result.extra_stages.append(extra_match.group(1).lower())
            continue

        # Basis set
        basis = _match_basis(token_lower)
        if basis is not None and result.basis is None:
            result.basis = basis
            continue

        # Functional
        if _match_functional(token_lower) and result.functional is None:
            result.functional = token
            continue

        # Unrecognized - could be a functional or basis we don't know
        result.unrecognized.append(token)

    if not result.job_type:
        raise ValueError(
            "Route line must contain a job type keyword "
            "(Opt, OptTS, SP, Freq, IRC, Scan)."
        )

    # If we have unrecognized tokens and missing functional/basis, try to assign them
    remaining = list(result.unrecognized)
    result.unrecognized = []
    for token in remaining:
        if result.functional is None:
            # Assume first unrecognized is functional
            result.functional = token
        elif result.basis is None:
            # Assume second unrecognized is basis
            result.basis = token
        else:
            result.unrecognized.append(token)

    if not result.functional:
        raise ValueError("Route line must contain a DFT functional (e.g., B3LYP).")
    if not result.basis:
        raise ValueError("Route line must contain a basis set (e.g., def2-SVP).")

    return result
