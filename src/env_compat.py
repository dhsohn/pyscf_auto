import os


TRUTHY_VALUES = ("1", "true", "yes", "on")


def getenv_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None:
        return value
    return default


def env_truthy(name: str) -> bool:
    value = getenv_str(name)
    if value is None:
        return False
    return value.strip().lower() in TRUTHY_VALUES
