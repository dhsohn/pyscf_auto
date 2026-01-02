import inspect
import logging
import re


# DFT-D3 damping parameters that can be tuned for a given functional.
#
# - simple-dftd3 (dftd3-python) exposes these through the `params_tweaks` dict.
# - ASE's dftd3 wrapper exposes these as top-level keywords.
_D3_TWEAK_KEYS = {
    # Common
    "s6",
    "s8",
    "s9",
    # BJ / rational
    "a1",
    "a2",
    # Zero damping / ATM
    "alp",
    "sr6",
    "sr8",
    "alpha6",
    "beta",
    "rs6",
    "rs8",
}


def load_d3_calculator(prefer_backend=None):
    """Return (DFTD3_class, backend_label) or (None, None).

    backend_label is one of:
      - "dftd3"  (simple-dftd3 / dftd3-python)
    """

    normalized_preference = str(prefer_backend or "").strip().lower()
    if normalized_preference and normalized_preference != "dftd3":
        return None, None

    # Auto: prefer the Python API if present.
    try:
        from dftd3.ase import DFTD3

        return DFTD3, "dftd3"
    except ImportError:
        return None, None


def _signature_info(callable_obj):
    """Return (param_names, accepts_kwargs)."""

    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return set(), True
    params = sig.parameters
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    return set(params.keys()), accepts_kwargs


def _coerce_float(value, key_path):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            "DFTD3 damping parameter '{name}' must be a number.".format(name=key_path)
        )
    return float(value)


def _split_d3_params(d3_params):
    """Split a user-provided d3_params dict into:

    - other_settings: keys intended to be passed as normal calculator keywords
    - tweak_params:   damping parameter overrides (s8, a1, a2, ...)

    This function also normalizes multiple accepted shapes:
      {"s8": 1.2, "a1": 0.3, "a2": 4.5}
      {"parameters": {"s8": ...}}
      {"params_tweaks": {"s8": ...}}
      {"damping": {"parameters": {...}}}
    """

    if d3_params is None:
        return {}, {}
    if not isinstance(d3_params, dict):
        raise ValueError("DFTD3 parameters must be provided as a JSON object.")

    other_settings = {}
    tweak_params = {}

    def _walk(obj, prefix=""):
        if obj is None:
            return
        if not isinstance(obj, dict):
            # Ignore non-dict containers at this level.
            return

        for key, value in obj.items():
            if value is None:
                continue
            key_str = str(key)
            path = f"{prefix}{key_str}" if prefix else key_str

            # Common nesting patterns.
            if key_str in ("damping", "variant") and isinstance(value, dict):
                _walk(value, prefix=path + ".")
                continue
            if key_str in ("parameters", "params_tweaks") and isinstance(value, dict):
                for pkey, pval in value.items():
                    if pval is None:
                        continue
                    pkey_str = str(pkey)
                    if pkey_str in _D3_TWEAK_KEYS:
                        tweak_params[pkey_str] = _coerce_float(pval, f"{path}.{pkey_str}")
                    else:
                        # Unknown keys inside a params dict are ignored to avoid
                        # passing invalid values to the calculator.
                        logging.debug(
                            "Ignoring unknown DFTD3 damping key '%s' under %s",
                            pkey_str,
                            path,
                        )
                continue

            # Direct tweak keys at top-level.
            if key_str in _D3_TWEAK_KEYS and not isinstance(value, dict):
                tweak_params[key_str] = _coerce_float(value, path)
                continue

            # Avoid passing nested dicts directly to calculators (they may
            # interpret some keys specially, e.g. `parameters`).
            if isinstance(value, dict):
                logging.debug(
                    "Ignoring nested DFTD3 setting '%s' (dict values are not supported)",
                    path,
                )
                continue

            other_settings[key_str] = value

    _walk(d3_params)
    return other_settings, tweak_params


def _select_xc_keyword(param_names, accepts_kwargs):
    # simple-dftd3 docs use `method=`
    for key in ("method", "xc", "functional", "func"):
        if key in param_names or accepts_kwargs:
            return key
    return "method"


def _select_damping_keyword(param_names, accepts_kwargs):
    for key in ("damping", "variant"):
        if key in param_names or accepts_kwargs:
            return key
    return "damping"


def parse_dispersion_settings(
    dispersion_model,
    xc,
    charge=None,
    spin=None,
    d3_params=None,
    prefer_d3_backend=None,
):
    """Parse dispersion settings for geometry optimization and SP/frequency stages.

    Returns a dict:
      {"backend": "d3"|"d4", "settings": {...}}
    """

    if not xc:
        raise ValueError("Dispersion correction requires an XC functional to be set.")

    normalized = re.sub(r"[\s_\-]+", "", dispersion_model or "").lower()
    if normalized == "d3":
        raise ValueError(
            "Dispersion model 'd3' is ambiguous. Use 'd3bj' or 'd3zero' explicitly."
        )

    # ------------------ D3 ------------------
    if normalized in ("d3bj", "d3(bj)"):
        base_variant = "bj"
    elif normalized in ("d3zero", "d30", "d3(0)"):
        base_variant = "zero"
    else:
        base_variant = None

    if base_variant is not None:
        d3_cls, _ = load_d3_calculator(prefer_d3_backend)
        if d3_cls is None:
            raise ImportError(
                "DFTD3 dispersion requested but no DFTD3 calculator is available. "
                "Install `dftd3` (recommended)."
            )

        damping_value = f"d3{base_variant}"

        param_names, accepts_kwargs = _signature_info(d3_cls)
        xc_key = _select_xc_keyword(param_names, accepts_kwargs)
        damping_key = _select_damping_keyword(param_names, accepts_kwargs)

        settings = {xc_key: xc, damping_key: damping_value}

        other_settings, tweak_params = _split_d3_params(d3_params)

        # Add supported non-tweak settings.
        for key, value in other_settings.items():
            # Never forward the reserved/unsafe `parameters` key.
            if key == "parameters":
                continue
            if accepts_kwargs or key in param_names:
                settings[key] = value
            else:
                logging.debug(
                    "Ignoring unsupported D3 setting '%s' for dftd3 backend",
                    key,
                )

        # Add damping tweaks (backend-specific).
        if tweak_params:
            # simple-dftd3 uses `params_tweaks`.
            if accepts_kwargs or "params_tweaks" in param_names:
                existing = settings.get("params_tweaks")
                merged = {}
                if isinstance(existing, dict):
                    merged.update(existing)
                merged.update(tweak_params)
                settings["params_tweaks"] = merged
            else:
                logging.debug(
                    "DFTD3 tweaks were provided but the calculator does not accept params_tweaks; ignoring."
                )

        return {"backend": "d3", "settings": settings}

    # ------------------ D4 ------------------
    if normalized in ("d4", "d4bj", "d4(bj)"):
        try:
            from dftd4.ase import DFTD4
        except ImportError as exc:
            raise ImportError(
                "DFTD4 dispersion requested but the dftd4 package is not installed. "
                "Install it with `conda install -c conda-forge dftd4` or `pip install dftd4`."
            ) from exc

        param_names, accepts_kwargs = _signature_info(DFTD4)
        # DFTD4 supports `method` in most builds.
        xc_key = None
        for key in ("method", "xc", "functional"):
            if key in param_names or accepts_kwargs:
                xc_key = key
                break
        if xc_key is None:
            xc_key = "method"
        settings = {xc_key: xc}

        if charge is not None:
            for key in ("charge", "q"):
                if key in param_names or accepts_kwargs:
                    settings[key] = charge
                    break
        if spin is not None:
            multiplicity = spin + 1
            for key in ("multiplicity", "mult"):
                if key in param_names or accepts_kwargs:
                    settings[key] = multiplicity
                    break
            if "spin" in param_names or accepts_kwargs:
                settings["spin"] = spin

        return {"backend": "d4", "settings": settings}

    raise ValueError(
        "Unsupported dispersion model '{model}'. Use 'd3bj', 'd3zero', or 'd4'.".format(
            model=dispersion_model
        )
    )
