import os
from pathlib import Path


def get_app_base_dir() -> str:
    override = os.environ.get("DFTFLOW_BASE_DIR")
    if override:
        return override
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return os.getcwd()
    return os.path.join(Path.home(), "DFTFlow")


def get_runs_base_dir() -> str:
    return os.path.join(get_app_base_dir(), "runs")


def get_smoke_runs_base_dir() -> str:
    return os.path.join(get_runs_base_dir(), "smoke")
