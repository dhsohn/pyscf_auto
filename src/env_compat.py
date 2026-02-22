import os


def getenv_with_legacy(
    name: str,
    legacy_name: str | None = None,
    default: str | None = None,
) -> str | None:
    value = os.environ.get(name)
    if value is not None:
        return value
    if legacy_name:
        legacy_value = os.environ.get(legacy_name)
        if legacy_value is not None:
            return legacy_value
    return default


def env_truthy(name: str, legacy_name: str | None = None) -> bool:
    value = getenv_with_legacy(name, legacy_name)
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")
