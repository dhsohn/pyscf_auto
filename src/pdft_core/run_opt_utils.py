"""Shared helpers for run_opt workflows."""


def extract_step_count(*candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ("n_steps", "nsteps", "nstep", "steps", "step_count"):
            if not hasattr(candidate, attr):
                continue
            value = getattr(candidate, attr)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, int):
                return value
    return None
