"""PySCF-specific retry strategies for failed calculations.

Each strategy modifies the configuration dictionary to improve the
chances of convergence on the next attempt.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def retry_increase_max_cycle(
    config: dict[str, Any], failure_reason: str
) -> tuple[dict[str, Any], str]:
    """Increase SCF max_cycle from default to 500."""
    config = copy.deepcopy(config)
    scf = config.setdefault("scf", {})
    old = scf.get("max_cycle", 200)
    scf["max_cycle"] = max(old, 500)
    return config, f"increase_max_cycle({old}->{scf['max_cycle']})"


def retry_add_level_shift(
    config: dict[str, Any], failure_reason: str
) -> tuple[dict[str, Any], str]:
    """Add level shift to help SCF convergence."""
    config = copy.deepcopy(config)
    scf = config.setdefault("scf", {})
    scf["level_shift"] = 0.2
    return config, "add_level_shift(0.2)"


def retry_switch_diis_preset(
    config: dict[str, Any], failure_reason: str
) -> tuple[dict[str, Any], str]:
    """Switch to a more stable DIIS preset."""
    config = copy.deepcopy(config)
    scf = config.setdefault("scf", {})
    scf["diis_preset"] = "stable"
    return config, "diis_preset(stable)"


def retry_increase_damping(
    config: dict[str, Any], failure_reason: str
) -> tuple[dict[str, Any], str]:
    """Add SCF damping to slow down density mixing."""
    config = copy.deepcopy(config)
    scf = config.setdefault("scf", {})
    scf["damping"] = 0.3
    return config, "add_damping(0.3)"


def retry_change_init_guess(
    config: dict[str, Any], failure_reason: str
) -> tuple[dict[str, Any], str]:
    """Try a different initial guess method."""
    config = copy.deepcopy(config)
    scf = config.setdefault("scf", {})
    extra = scf.setdefault("extra", {})
    extra["init_guess"] = "atom"
    return config, "init_guess(atom)"


# Ordered list of retry strategies to apply
RETRY_STRATEGIES: list[
    Callable[[dict[str, Any], str], tuple[dict[str, Any], str]]
] = [
    retry_increase_max_cycle,
    retry_add_level_shift,
    retry_switch_diis_preset,
    retry_increase_damping,
    retry_change_init_guess,
]


def apply_retry_strategy(
    config_dict: dict[str, Any],
    attempt_index: int,
    failure_reason: str,
) -> tuple[dict[str, Any], list[str]]:
    """Apply the appropriate retry strategy based on attempt index.

    Args:
        config_dict: Current configuration dictionary.
        attempt_index: The 1-based attempt number (2 = first retry).
        failure_reason: The reason the previous attempt failed.

    Returns:
        Tuple of (modified_config, list_of_patch_action_descriptions).
    """
    retry_index = attempt_index - 2  # attempt 2 -> retry 0, attempt 3 -> retry 1
    if retry_index < 0:
        return config_dict, []

    patches: list[str] = []
    modified = config_dict

    # Apply all strategies up to and including the current retry level
    # This is cumulative - each retry adds to previous modifications
    for i in range(min(retry_index + 1, len(RETRY_STRATEGIES))):
        modified, description = RETRY_STRATEGIES[i](modified, failure_reason)
        patches.append(description)
        logger.info(
            "Retry strategy %d applied: %s", i + 1, description
        )

    return modified, patches
