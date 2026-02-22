import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from env_compat import getenv_with_legacy
from run_opt_paths import get_runs_base_dir
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

RUN_ARCHIVE_AFTER_DAYS = int(
    getenv_with_legacy(
        "PYSCF_AUTO_RUN_ARCHIVE_AFTER_DAYS",
        "DFTFLOW_RUN_ARCHIVE_AFTER_DAYS",
        "7",
    )
)
RUN_ARCHIVE_INTERVAL_HOURS = int(
    getenv_with_legacy(
        "PYSCF_AUTO_RUN_ARCHIVE_INTERVAL_HOURS",
        "DFTFLOW_RUN_ARCHIVE_INTERVAL_HOURS",
        "6",
    )
)
RUN_ARCHIVE_MAX_PER_PASS = int(
    getenv_with_legacy(
        "PYSCF_AUTO_RUN_ARCHIVE_MAX_PER_PASS",
        "DFTFLOW_RUN_ARCHIVE_MAX_PER_PASS",
        "5",
    )
)
RUN_ARCHIVE_DIRNAME = "archive"
RUN_ARCHIVE_STATE_FILENAME = ".archive_state.json"
RUN_ARCHIVE_LOCK_FILENAME = ".archive.lock"
RUN_ARCHIVE_STATUSES = ("completed", "failed", "timeout", "canceled", "skipped")
RUN_ARCHIVE_SKIP_DIRS = {
    RUN_ARCHIVE_DIRNAME,
    "configs",
    "inputs",
    "smoke",
}


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


def create_run_directory(base_dir=None):
    if base_dir is None:
        base_dir = get_runs_base_dir()
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


def _parse_iso_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_archive_state(base_dir: Path) -> dict:
    state_path = base_dir / RUN_ARCHIVE_STATE_FILENAME
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_archive_state(base_dir: Path, state: dict) -> None:
    state_path = base_dir / RUN_ARCHIVE_STATE_FILENAME
    ensure_parent_dir(str(state_path))
    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(base_dir),
        prefix=".archive_state.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with temp_handle as handle:
            json.dump(state, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_handle.name, state_path)
    finally:
        if os.path.exists(temp_handle.name):
            try:
                os.remove(temp_handle.name)
            except FileNotFoundError:
                pass


def _try_acquire_archive_lock(lock_path: Path, stale_seconds: int = 3600) -> bool:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            mtime = os.path.getmtime(lock_path)
        except OSError:
            return False
        if time.time() - mtime < stale_seconds:
            return False
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            return False
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()} {datetime.now().isoformat()}")
    return True


def _release_archive_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _load_runs_index(base_dir: Path) -> dict | None:
    index_path = base_dir / "index.json"
    if not index_path.exists():
        return None
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        return None
    return payload


def _write_runs_index(base_dir: Path, payload: dict) -> None:
    index_path = base_dir / "index.json"
    ensure_parent_dir(str(index_path))
    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(base_dir),
        prefix=".index.json.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with temp_handle as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_handle.name, index_path)
    finally:
        if os.path.exists(temp_handle.name):
            try:
                os.remove(temp_handle.name)
            except FileNotFoundError:
                pass


def _remove_runs_index_entries(base_dir: Path, run_dirs: set[str]) -> None:
    payload = _load_runs_index(base_dir)
    if not payload:
        return
    entries = payload.get("entries") or []
    if not entries:
        return
    normalized = {str(Path(path).resolve()) for path in run_dirs}
    filtered = [
        entry
        for entry in entries
        if str(Path(entry.get("run_dir", "")).resolve()) not in normalized
    ]
    if len(filtered) == len(entries):
        return
    payload["entries"] = filtered
    payload["updated_at"] = datetime.now().isoformat()
    _write_runs_index(base_dir, payload)


def _archive_run_dir(run_dir: Path, archive_dir: Path) -> Path | None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{run_dir.name}.tar.gz"
    archive_path = archive_dir / archive_name
    if archive_path.exists():
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = archive_dir / f"{run_dir.name}.{stamp}.tar.gz"
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(run_dir, arcname=run_dir.name)
    except (OSError, tarfile.TarError):
        return None
    try:
        shutil.rmtree(run_dir)
    except OSError:
        return None
    return archive_path


def _collect_archive_candidates(base_dir: Path, cutoff: datetime) -> list[Path]:
    candidates: list[Path] = []
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name in RUN_ARCHIVE_SKIP_DIRS:
            continue
        metadata_path = entry / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = metadata.get("status")
        if status not in RUN_ARCHIVE_STATUSES:
            continue
        ended_at = _parse_iso_timestamp(
            metadata.get("run_ended_at") or metadata.get("run_updated_at")
        )
        if ended_at is None:
            try:
                ended_at = datetime.fromtimestamp(metadata_path.stat().st_mtime)
            except OSError:
                continue
        if ended_at < cutoff:
            candidates.append(entry)
    candidates.sort(key=lambda path: path.name)
    return candidates


def auto_archive_runs(base_dir: str | None = None) -> int:
    if RUN_ARCHIVE_AFTER_DAYS <= 0:
        return 0
    base_path = Path(base_dir) if base_dir else Path(get_runs_base_dir())
    if not base_path.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=RUN_ARCHIVE_AFTER_DAYS)
    candidates = _collect_archive_candidates(base_path, cutoff)
    if not candidates:
        return 0
    archive_dir = base_path / RUN_ARCHIVE_DIRNAME
    archived = []
    for run_dir in candidates[:RUN_ARCHIVE_MAX_PER_PASS]:
        archive_path = _archive_run_dir(run_dir, archive_dir)
        if archive_path is None:
            continue
        archived.append(str(run_dir.resolve()))
    if archived:
        _remove_runs_index_entries(base_path, set(archived))
    return len(archived)


def maybe_auto_archive_runs(base_dir: str | None = None) -> int:
    if RUN_ARCHIVE_AFTER_DAYS <= 0:
        return 0
    base_path = Path(base_dir) if base_dir else Path(get_runs_base_dir())
    if not base_path.exists():
        return 0
    state = _load_archive_state(base_path)
    last_run = _parse_iso_timestamp(state.get("last_run_at"))
    if last_run:
        delta = datetime.now() - last_run
        if delta < timedelta(hours=RUN_ARCHIVE_INTERVAL_HOURS):
            return 0
    lock_path = base_path / RUN_ARCHIVE_LOCK_FILENAME
    if not _try_acquire_archive_lock(lock_path):
        return 0
    archived = 0
    try:
        archived = auto_archive_runs(str(base_path))
        state = {
            "last_run_at": datetime.now().isoformat(),
            "last_archived": archived,
        }
        _write_archive_state(base_path, state)
    finally:
        _release_archive_lock(lock_path)
    return archived
