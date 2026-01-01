import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _evaluate_openmp_availability(requested_threads, effective_threads):
    if not isinstance(effective_threads, int) or not requested_threads:
        return None
    if requested_threads > 1 and effective_threads == 1:
        return False
    return True


def _collect_thread_env():
    return {env_name: os.environ.get(env_name) for env_name in THREAD_ENV_VARS}


def _infer_requested_threads(requested_threads, environment):
    if requested_threads is not None:
        return requested_threads
    for env_value in environment.values():
        if not env_value:
            continue
        try:
            return int(env_value)
        except ValueError:
            continue
    return None


def _load_pyscf_lib():
    pyscf_lib_spec = importlib.util.find_spec("pyscf.lib")
    if pyscf_lib_spec is None:
        return None
    pyscf_lib = importlib.import_module("pyscf.lib")
    if not hasattr(pyscf_lib, "num_threads"):
        return None
    return pyscf_lib


def _get_effective_threads():
    pyscf_lib = _load_pyscf_lib()
    if pyscf_lib is None:
        return None
    return pyscf_lib.num_threads()


def _collect_threading_snapshot(requested_threads=None, include_env=False):
    environment = _collect_thread_env()
    inferred_request = _infer_requested_threads(requested_threads, environment)
    effective_threads = _get_effective_threads()
    status = {
        "requested": inferred_request,
        "effective_threads": effective_threads,
        "openmp_available": _evaluate_openmp_availability(
            inferred_request,
            effective_threads,
        ),
    }
    if include_env:
        status["env"] = environment
    return status


def inspect_thread_settings(requested_threads=None):
    return _collect_threading_snapshot(requested_threads, include_env=True)


def apply_thread_settings(thread_count):
    status = {
        "requested": thread_count,
        "effective_threads": None,
        "openmp_available": None,
    }
    if not thread_count:
        return status
    thread_value = str(thread_count)
    for env_name in THREAD_ENV_VARS:
        os.environ[env_name] = thread_value
    pyscf_lib = _load_pyscf_lib()
    if pyscf_lib is None:
        return status
    pyscf_lib.num_threads(thread_count)
    effective_threads = pyscf_lib.num_threads()
    status["effective_threads"] = effective_threads
    status["openmp_available"] = _evaluate_openmp_availability(
        thread_count,
        effective_threads,
    )
    return status


def apply_memory_limit(memory_gb, enforce):
    if not memory_gb:
        return None, None
    memory_mb = int(memory_gb * 1024)
    status = {"applied": False, "reason": None, "limit_bytes": None, "limit_name": None}
    if not enforce:
        status["reason"] = "disabled by config"
        return memory_mb, status
    resource_spec = importlib.util.find_spec("resource")
    if resource_spec is None:
        status["reason"] = "'resource' module unavailable"
        return memory_mb, status
    resource = importlib.import_module("resource")
    limit_bytes = int(memory_gb * 1024 ** 3)
    candidate_limits = ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS")
    attempted = []
    available = []
    for limit_name in candidate_limits:
        if not hasattr(resource, limit_name):
            attempted.append(f"{limit_name} unavailable")
            continue
        available.append(limit_name)
        limit_value = getattr(resource, limit_name)
        current_soft, current_hard = resource.getrlimit(limit_value)
        adjusted_limit = limit_bytes
        if current_hard != resource.RLIM_INFINITY:
            adjusted_limit = min(adjusted_limit, current_hard)
        try:
            resource.setrlimit(limit_value, (adjusted_limit, current_hard))
        except (OSError, ValueError) as exc:
            attempted.append(f"{limit_name} failed ({exc})")
            continue
        status["applied"] = True
        status["reason"] = "applied"
        status["limit_bytes"] = adjusted_limit
        status["limit_name"] = limit_name
        break
    if not status["applied"]:
        if not available:
            status["reason"] = "no supported rlimit available on this platform"
        else:
            status["reason"] = "unable to set memory limit (" + "; ".join(attempted) + ")"
        status["reason"] += (
            " (platform/permission limitations possible; "
            "set enforce_os_memory_limit=false to skip)"
        )
    return memory_mb, status


def create_run_directory(base_dir="runs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    candidate = os.path.join(base_dir, timestamp)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{timestamp}_{suffix}")
        suffix += 1
    os.makedirs(candidate, exist_ok=True)
    return candidate


def format_log_path(log_path):
    if "{timestamp}" in log_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return log_path.replace("{timestamp}", timestamp)
    return log_path


def resolve_run_path(run_dir, path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(run_dir, path)


def ensure_parent_dir(file_path):
    if not file_path:
        return
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _run_conda_command(args):
    try:
        completed = subprocess.run(
            ["conda", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _extract_conda_environment_name():
    default_env = os.environ.get("CONDA_DEFAULT_ENV")
    if default_env:
        return default_env
    info_output = _run_conda_command(["info", "--json"])
    if not info_output:
        return None
    try:
        info = json.loads(info_output)
    except json.JSONDecodeError:
        return None
    env_name = info.get("active_prefix_name")
    if env_name:
        return env_name
    active_prefix = info.get("active_prefix")
    if not active_prefix:
        return None
    basename = os.path.basename(active_prefix.rstrip(os.sep))
    return basename or None


def _calculate_conda_list_export_sha256():
    list_output = _run_conda_command(["list", "--export"])
    if not list_output:
        return None
    return hashlib.sha256(list_output.encode("utf-8")).hexdigest()


def collect_environment_snapshot(thread_count):
    conda_snapshot = {
        "name": _extract_conda_environment_name(),
        "list_export_sha256": _calculate_conda_list_export_sha256(),
    }
    threading_snapshot = _collect_threading_snapshot(thread_count, include_env=True)
    return {
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform_string": platform.platform(),
        },
        "cpu": {"count": os.cpu_count()},
        "threading": threading_snapshot,
        "conda": conda_snapshot,
    }
