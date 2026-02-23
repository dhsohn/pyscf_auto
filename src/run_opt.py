"""CLI entrypoint module for pyscf_auto."""

__all__ = ["main"]

import argparse
import importlib.util
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

try:
    import cli
except ImportError:
    cli = None  # Legacy CLI removed; using cli_new
try:
    import run_queue
except ImportError:
    run_queue = None  # Legacy queue removed
try:
    import run_opt_smoke
except ImportError:
    run_opt_smoke = None  # Legacy smoke test removed
import execution
from env_compat import getenv_with_legacy
from run_opt_config import (
    DEFAULT_QUEUE_LOCK_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_QUEUE_RUNNER_LOCK_PATH,
    DEFAULT_RUN_METADATA_PATH,
    DEFAULT_SOLVENT_MAP_PATH,
    SMD_UNSUPPORTED_SOLVENT_KEYS,
    build_run_config,
    load_run_config,
    load_solvent_map,
)
from run_opt_paths import get_runs_base_dir, get_smoke_runs_base_dir
from run_opt_resources import create_run_directory, maybe_auto_archive_runs
from run_opt_metadata import compute_text_hash, write_run_metadata
from run_opt_utils import normalize_solvent_key as _normalize_solvent_key

TERMINAL_RESUME_STATUSES = {"completed", "failed", "timeout", "canceled"}

SMOKE_TEST_XYZ = """3
pyscf_auto smoke-test water molecule. charge=0 spin=0
O     -0.1659811139    2.0308399200   -0.0000031757
H     -2.5444712639    1.0182403326    0.6584512591
H     -1.0147968531    2.4412472248   -2.0058431625
"""
SMOKE_TEST_MODES = ("single_point", "optimization", "frequency", "irc", "scan")
SMOKE_TEST_SOLVENT_MODELS = (None, "pcm", "smd")
SMOKE_TEST_DISPERSION_MODELS = (None, "d3bj", "d3zero", "d4")
SMOKE_TEST_PROGRESS_FILE = "smoke_progress.json"
SMOKE_TEST_HEARTBEAT_FILE = "smoke_heartbeat.txt"
SMOKE_TEST_HEARTBEAT_INTERVAL = 30
DEFAULT_BASIS_SET_OPTIONS = [
    "6-31g",
    "6-31g*",
    "6-31g**",
    "def2-svp",
    "def2-tzvp",
    "def2-tzvpp",
    "cc-pvdz",
    "cc-pvtz",
]
DEFAULT_XC_FUNCTIONAL_OPTIONS = [
    "b3lyp",
    "pbe0",
    "WB97X_D",
    "m06-2x",
    "pbe",
    "b97-d",
]
QUICK_BASIS_SET_OPTIONS = ["6-31g", "def2-svp"]
QUICK_XC_FUNCTIONAL_OPTIONS = ["b3lyp", "pbe0"]
QUICK_DISPERSION_MODELS = (None, "d3bj")
QUICK_SOLVENT_MODELS = (None, "smd")


def _read_json_file(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _index_entry_from_metadata(metadata_path: Path, metadata: dict) -> dict:
    run_dir = metadata.get("run_directory") or str(metadata_path.parent)
    return {
        "run_dir": str(Path(run_dir).resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "status": metadata.get("status"),
        "run_started_at": metadata.get("run_started_at"),
        "run_ended_at": metadata.get("run_ended_at"),
        "calculation_mode": metadata.get("calculation_mode"),
        "basis": metadata.get("basis"),
        "xc": metadata.get("xc"),
        "solvent": metadata.get("solvent"),
        "solvent_model": metadata.get("solvent_model"),
        "dispersion": metadata.get("dispersion"),
        "updated_at": datetime.now().isoformat(),
    }


def _load_run_entries(base_dir: Path, limit: int | None = None) -> list[dict]:
    entries: list[dict] = []
    if not base_dir.exists():
        return entries
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        metadata_path = entry / DEFAULT_RUN_METADATA_PATH
        if not metadata_path.exists():
            continue
        metadata = _read_json_file(metadata_path)
        if not metadata:
            continue
        entries.append(_index_entry_from_metadata(metadata_path, metadata))
    entries.sort(
        key=lambda item: os.path.getmtime(item["metadata_path"])
        if os.path.exists(item["metadata_path"])
        else 0,
        reverse=True,
    )
    if limit:
        entries = entries[:limit]
    return entries


def _build_smoke_test_config(base_config, mode, overrides):
    config = dict(base_config)
    config["calculation_mode"] = mode
    config["basis"] = overrides["basis"]
    config["xc"] = overrides["xc"]
    config["solvent_model"] = overrides["solvent_model"]
    config["solvent"] = overrides["solvent"]
    config["dispersion"] = overrides["dispersion"]
    if mode != "scan":
        config.pop("scan", None)
        config.pop("scan2d", None)
    scf_config = dict(config.get("scf") or {})
    scf_config["max_cycle"] = 1
    config["scf"] = scf_config
    single_point_config = dict(config.get("single_point") or {})
    single_point_config["basis"] = overrides["basis"]
    single_point_config["xc"] = overrides["xc"]
    single_point_config["solvent"] = overrides["solvent"]
    single_point_config["solvent_model"] = overrides["solvent_model"]
    single_point_config["dispersion"] = overrides["dispersion"]
    single_point_scf = dict(single_point_config.get("scf") or {})
    single_point_scf["max_cycle"] = 1
    single_point_config["scf"] = single_point_scf
    config["single_point"] = single_point_config
    optimizer_config = dict(config.get("optimizer") or {})
    optimizer_ase = dict(optimizer_config.get("ase") or {})
    optimizer_ase["steps"] = 1
    optimizer_config["ase"] = optimizer_ase
    config["optimizer"] = optimizer_config
    if mode == "optimization":
        config["frequency_enabled"] = False
        config["single_point_enabled"] = False
    if mode == "irc":
        config["irc"] = {
            "steps": 1,
            "step_size": 0.05,
            "force_threshold": 0.1,
        }
    if mode == "scan":
        config.pop("scan2d", None)
        config["scan"] = {
            "mode": "single_point",
            "dimensions": [
                {
                    "type": "bond",
                    "i": 0,
                    "j": 1,
                    "start": 0.9,
                    "end": 0.9,
                    "step": 0.1,
                }
            ],
        }
    return config


def _load_smoke_progress(base_run_dir):
    progress_path = Path(base_run_dir) / SMOKE_TEST_PROGRESS_FILE
    if not progress_path.exists():
        return {"cases": {}, "updated_at": None}
    try:
        with progress_path.open("r", encoding="utf-8") as progress_file:
            data = json.load(progress_file)
    except (OSError, json.JSONDecodeError):
        return {"cases": {}, "updated_at": None}
    if not isinstance(data, dict):
        return {"cases": {}, "updated_at": None}
    data.setdefault("cases", {})
    return data


def _write_smoke_progress(base_run_dir, data):
    progress_path = Path(base_run_dir) / SMOKE_TEST_PROGRESS_FILE
    data["updated_at"] = datetime.now().isoformat()
    tmp_path = progress_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as progress_file:
        json.dump(data, progress_file, indent=2)
        progress_file.flush()
        os.fsync(progress_file.fileno())
    os.replace(tmp_path, progress_path)


def _update_smoke_progress(base_run_dir, run_dir, status, error=None):
    data = _load_smoke_progress(base_run_dir)
    cases = data.setdefault("cases", {})
    entry = cases.get(str(run_dir), {})
    entry["status"] = status
    entry["updated_at"] = datetime.now().isoformat()
    if error:
        entry["error"] = error
    cases[str(run_dir)] = entry
    _write_smoke_progress(base_run_dir, data)


def _smoke_progress_status(base_run_dir, run_dir):
    data = _load_smoke_progress(base_run_dir)
    entry = data.get("cases", {}).get(str(run_dir))
    if not entry:
        return None
    return entry.get("status")


def _parse_smoke_status_file(run_dir):
    status_path = Path(run_dir) / "smoke_subprocess.status"
    if not status_path.exists():
        return None
    try:
        payload = status_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if "exit_code=" not in payload:
        return None
    exit_code = payload.split("exit_code=", 1)[-1].strip()
    try:
        return int(exit_code)
    except ValueError:
        return None


def _infer_smoke_case_status(run_dir):
    metadata_path = Path(run_dir) / DEFAULT_RUN_METADATA_PATH
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if metadata:
            status = metadata.get("status")
            if status in ("completed", "failed", "skipped"):
                return status
    exit_code = _parse_smoke_status_file(run_dir)
    if exit_code is None:
        return None
    return "completed" if exit_code == 0 else "failed"


def _write_smoke_heartbeat(path):
    payload = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
    heartbeat_path = Path(path)
    with heartbeat_path.open("w", encoding="utf-8") as heartbeat_file:
        heartbeat_file.write(payload)
        heartbeat_file.flush()
        os.fsync(heartbeat_file.fileno())


def _start_smoke_heartbeat(path, interval):
    stop_event = threading.Event()

    def _beat():
        _write_smoke_heartbeat(path)
        while not stop_event.wait(interval):
            _write_smoke_heartbeat(path)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return stop_event


def _unique_values(values):
    seen = set()
    ordered = []
    for value in values:
        key = value if value is not None else "__none__"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def _d3_damping_support_status(xc, dispersion_model):
    if not dispersion_model:
        return True, None
    normalized = str(dispersion_model).strip().lower()
    if normalized not in ("d3bj", "d3zero"):
        return True, None
    try:
        from dftd3 import ase as dftd3_ase
    except ImportError:
        return False, "dftd3 is not installed"
    damping_key = normalized
    damping_cls = dftd3_ase._damping_param.get(damping_key)
    if damping_cls is None:
        return False, f"no damping parameters for {damping_key}"
    try:
        damping_cls(method=xc)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _write_smoke_skip_metadata(run_dir, overrides, mode, reason):
    now = datetime.now().isoformat()
    metadata = {
        "status": "skipped",
        "run_directory": str(run_dir),
        "run_started_at": now,
        "run_ended_at": now,
        "skip_reason": reason,
        "basis": overrides["basis"],
        "xc": overrides["xc"],
        "solvent": overrides["solvent"],
        "solvent_model": overrides["solvent_model"],
        "dispersion": overrides["dispersion"],
        "calculation_mode": mode,
        "run_metadata_file": str(Path(run_dir) / DEFAULT_RUN_METADATA_PATH),
    }
    write_run_metadata(str(Path(run_dir) / DEFAULT_RUN_METADATA_PATH), metadata)


def _prepare_smoke_test_suite(args):
    config_path = Path(args.config).expanduser().resolve()
    base_config, _base_raw = load_run_config(config_path)
    if args.smoke_mode == "quick":
        basis_seed = QUICK_BASIS_SET_OPTIONS
        xc_seed = QUICK_XC_FUNCTIONAL_OPTIONS
        solvent_model_seed = QUICK_SOLVENT_MODELS
        dispersion_seed = QUICK_DISPERSION_MODELS
    else:
        basis_seed = DEFAULT_BASIS_SET_OPTIONS
        xc_seed = DEFAULT_XC_FUNCTIONAL_OPTIONS
        solvent_model_seed = SMOKE_TEST_SOLVENT_MODELS
        dispersion_seed = SMOKE_TEST_DISPERSION_MODELS
    basis_options = _unique_values([*basis_seed, base_config.get("basis")])
    basis_options = [basis for basis in basis_options if basis]
    xc_options = _unique_values([*xc_seed, base_config.get("xc")])
    xc_options = [xc for xc in xc_options if xc]
    try:
        from pyscf.scf import dispersion as pyscf_dispersion
    except ImportError:
        pass
    else:
        filtered_xc = []
        for xc in xc_options:
            try:
                pyscf_dispersion.parse_dft(str(xc))
            except NotImplementedError as exc:
                logging.warning(
                    "Skipping XC %s in smoke tests (%s).",
                    xc,
                    exc,
                )
                continue
            filtered_xc.append(xc)
        xc_options = filtered_xc
    solvent_model_options = _unique_values(
        [*solvent_model_seed, base_config.get("solvent_model")]
    )
    dispersion_options = _unique_values([*dispersion_seed, base_config.get("dispersion")])
    if "d4" in dispersion_options:
        try:
            d4_spec = importlib.util.find_spec("dftd4.ase")
        except ModuleNotFoundError:
            d4_spec = None
        if d4_spec is None:
            dispersion_options = [item for item in dispersion_options if item != "d4"]
            logging.warning(
                "dftd4 is not installed; skipping D4 dispersion in smoke tests."
            )
    solvent_map_path = base_config.get("solvent_map") or DEFAULT_SOLVENT_MAP_PATH
    solvent_options = sorted(load_solvent_map(solvent_map_path).keys())
    if args.smoke_mode == "quick":
        quick_solvents = ["water", "benzene", "acetonitrile"]
        preferred = [s for s in quick_solvents if s in solvent_options]
        if preferred:
            solvent_options = preferred
        else:
            solvent_options = solvent_options[:3]
    smd_supported_keys = None
    if "smd" in solvent_model_options:
        try:
            from pyscf.solvent import smd as pyscf_smd
            from run_opt_engine import _build_smd_supported_map

            if getattr(pyscf_smd, "libsolvent", None) is None:
                logging.warning(
                    "SMD is unavailable in this PySCF build; skipping SMD smoke tests."
                )
                smd_supported_keys = set()
            else:
                smd_supported_keys = set(_build_smd_supported_map().keys())
        except Exception as exc:
            logging.warning(
                "Unable to load SMD solvent list; skipping SMD smoke tests (%s).", exc
            )
            smd_supported_keys = set()
    modes = list(SMOKE_TEST_MODES)
    cases = []
    for basis, xc, solvent_model, dispersion in itertools.product(
        basis_options, xc_options, solvent_model_options, dispersion_options
    ):
        d3_supported, d3_reason = _d3_damping_support_status(xc, dispersion)
        if not d3_supported:
            logging.warning(
                "Skipping dispersion %s for XC %s in smoke tests (%s).",
                dispersion,
                xc,
                d3_reason,
            )
        if solvent_model:
            for solvent in solvent_options:
                if solvent_model == "smd":
                    normalized_solvent = _normalize_solvent_key(solvent)
                    if normalized_solvent in SMD_UNSUPPORTED_SOLVENT_KEYS:
                        continue
                    if smd_supported_keys is not None:
                        if normalized_solvent not in smd_supported_keys:
                            continue
                cases.append(
                    {
                        "basis": basis,
                        "xc": xc,
                        "solvent_model": solvent_model,
                        "solvent": solvent,
                        "dispersion": dispersion,
                        "skip": not d3_supported,
                        "skip_reason": d3_reason,
                    }
                )
        else:
            cases.append(
                {
                    "basis": basis,
                    "xc": xc,
                    "solvent_model": None,
                    "solvent": "vacuum",
                    "dispersion": dispersion,
                    "skip": not d3_supported,
                    "skip_reason": d3_reason,
                }
            )
    return base_config, config_path, modes, cases


def _prepare_smoke_test_run_dir(base_run_dir, mode, overrides, index):
    parts = [
        f"{index:04d}",
        _slugify(mode),
        _slugify(overrides["basis"]),
        _slugify(overrides["xc"]),
        _slugify(overrides["solvent_model"] or "vacuum"),
        _slugify(overrides["solvent"]),
        _slugify(overrides["dispersion"] or "none"),
    ]
    run_dir = Path(base_run_dir) / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_smoke_test_status(run_dir):
    metadata_path = Path(run_dir) / DEFAULT_RUN_METADATA_PATH
    try:
        if not metadata_path.exists():
            return None
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return None
    return metadata.get("status")


def _format_subprocess_returncode(returncode):
    if returncode is None:
        return "unknown"
    if returncode < 0:
        try:
            import signal

            signal_name = signal.Signals(-returncode).name
        except (ValueError, AttributeError):
            signal_name = f"SIG{-returncode}"
        return f"signal {signal_name} ({returncode})"
    return str(returncode)


def _write_smoke_status_file(run_dir, exit_code=None, overwrite=True):
    status_path = Path(run_dir) / "smoke_subprocess.status"
    if status_path.exists() and not overwrite:
        return
    _write_smoke_status_path(status_path, exit_code)


def _write_smoke_status_path(status_path, exit_code=None):
    status_message = _format_subprocess_returncode(exit_code)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    status_payload = f"{timestamp} exit_code={status_message}\n"
    status_path = Path(status_path)
    with status_path.open("w", encoding="utf-8") as status_file:
        status_file.write(status_payload)
        status_file.flush()
        os.fsync(status_file.fileno())


def _ensure_smoke_status_file(run_dir, exit_code=None):
    try:
        _write_smoke_status_file(run_dir, exit_code, overwrite=False)
    except OSError as exc:
        logging.warning("Failed to write smoke-test status in %s: %s", run_dir, exc)


def _coerce_smoke_status_from_metadata(run_dir):
    metadata_path = Path(run_dir) / DEFAULT_RUN_METADATA_PATH
    if not metadata_path.exists():
        return
    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return
    status = metadata.get("status")
    if status == "completed":
        _ensure_smoke_status_file(run_dir, exit_code=0)


def _coerce_smoke_statuses(base_run_dir):
    base = Path(base_run_dir)
    if not base.exists():
        logging.warning(
            "Smoke-test status coercion skipped; run dir missing: %s", base
        )
        return 0
    coerced = 0
    logging.warning("Scanning for missing smoke-test status files in %s", base)
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        status_path = run_dir / "smoke_subprocess.status"
        if status_path.exists():
            continue
        metadata_path = run_dir / DEFAULT_RUN_METADATA_PATH
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") == "completed":
            _ensure_smoke_status_file(run_dir, exit_code=0)
            coerced += 1
            logging.warning("Coerced missing smoke-test status: %s", run_dir)
    return coerced


def _run_smoke_test_case(
    *,
    args,
    run_dir,
    xyz_path,
    solvent_map_path,
    smoke_config_path,
    smoke_config_raw,
    smoke_config,
):
    skip_capability_check = "PYSCF_AUTO_SKIP_CAPABILITY_CHECK"
    legacy_skip_capability_check = "DFTFLOW_SKIP_CAPABILITY_CHECK"
    run_args = argparse.Namespace(
        xyz_file=str(xyz_path),
        solvent_map=solvent_map_path,
        config=str(smoke_config_path),
        background=False,
        no_background=True,
        run_dir=str(run_dir),
        resume=None,
        run_id=None,
        force_resume=False,
        queue_priority=0,
        queue_max_runtime=None,
        scan_dimension=None,
        scan_grid=None,
        scan_mode=None,
        scan_result_csv=None,
        profile=bool(getattr(args, "profile", False)),
        queue_runner=False,
    )
    if args.no_isolate:
        previous_skip = os.environ.get(skip_capability_check)
        previous_legacy_skip = os.environ.get(legacy_skip_capability_check)
        os.environ[skip_capability_check] = "1"
        os.environ[legacy_skip_capability_check] = "1"
        config = build_run_config(smoke_config)
        try:
            execution.run(
                run_args,
                config,
                smoke_config_raw,
                str(smoke_config_path),
                False,
            )
        finally:
            if previous_skip is None:
                os.environ.pop(skip_capability_check, None)
            else:
                os.environ[skip_capability_check] = previous_skip
            if previous_legacy_skip is None:
                os.environ.pop(legacy_skip_capability_check, None)
            else:
                os.environ[legacy_skip_capability_check] = previous_legacy_skip
        return 0
    env = os.environ.copy()
    env[skip_capability_check] = "1"
    env[legacy_skip_capability_check] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    stderr_path = Path(run_dir) / "smoke_subprocess.err"
    stdout_path = Path(run_dir) / "smoke_subprocess.out"
    env["PYSCF_AUTO_SMOKE_STATUS_PATH"] = str(
        Path(run_dir) / "smoke_subprocess.status"
    )
    env["DFTFLOW_SMOKE_STATUS_PATH"] = env["PYSCF_AUTO_SMOKE_STATUS_PATH"]
    env["PYSCF_AUTO_SMOKE_HEARTBEAT_PATH"] = str(
        Path(run_dir) / SMOKE_TEST_HEARTBEAT_FILE
    )
    env["DFTFLOW_SMOKE_HEARTBEAT_PATH"] = env["PYSCF_AUTO_SMOKE_HEARTBEAT_PATH"]
    env["PYSCF_AUTO_SMOKE_HEARTBEAT_INTERVAL"] = str(SMOKE_TEST_HEARTBEAT_INTERVAL)
    env["DFTFLOW_SMOKE_HEARTBEAT_INTERVAL"] = env["PYSCF_AUTO_SMOKE_HEARTBEAT_INTERVAL"]
    src_dir = str(Path(__file__).resolve().parent)
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([src_dir, existing_pythonpath])
    else:
        env["PYTHONPATH"] = src_dir
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "run_opt",
        "run",
        str(xyz_path),
        "--config",
        str(smoke_config_path),
        "--run-dir",
        str(run_dir),
    ]
    completed = None
    try:
        with open(stdout_path, "a", encoding="utf-8") as stdout_file, open(
            stderr_path, "a", encoding="utf-8"
        ) as stderr_file:
            completed = subprocess.run(
                command,
                env=env,
                cwd=os.getcwd(),
                check=False,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
    finally:
        returncode = completed.returncode if completed else None
        try:
            _write_smoke_status_file(run_dir, returncode, overwrite=True)
        except OSError as exc:
            logging.warning(
                "Failed to write smoke-test status in %s: %s", run_dir, exc
            )
    try:
        if completed and completed.returncode == 0:
            metadata_path = Path(run_dir) / DEFAULT_RUN_METADATA_PATH
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                metadata["status"] = "completed"
                with metadata_path.open("w", encoding="utf-8") as metadata_file:
                    json.dump(metadata, metadata_file, indent=2)
                    metadata_file.flush()
                    os.fsync(metadata_file.fileno())
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to refresh smoke-test metadata in %s: %s", run_dir, exc)
    return completed.returncode if completed else 1


def _find_latest_smoke_activity_mtime(base_run_dir):
    latest = None
    for root, _dirs, files in os.walk(base_run_dir):
        for filename in ("run.log", SMOKE_TEST_HEARTBEAT_FILE):
            if filename not in files:
                continue
            log_path = os.path.join(root, filename)
            try:
                mtime = os.path.getmtime(log_path)
            except OSError:
                continue
            if latest is None or mtime > latest:
                latest = mtime
    return latest


def _smoke_test_has_failures(base_run_dir):
    for root, _dirs, files in os.walk(base_run_dir):
        if DEFAULT_RUN_METADATA_PATH not in files:
            continue
        metadata_path = os.path.join(root, DEFAULT_RUN_METADATA_PATH)
        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") == "failed":
            return True
    return False


def _run_smoke_test_watch(args):
    if not args.run_dir:
        args.run_dir = create_run_directory()
    base_run_dir = str(Path(args.run_dir).expanduser().resolve())
    os.makedirs(base_run_dir, exist_ok=True)
    watch_log_path = os.path.join(base_run_dir, "smoke_watch.log")

    cmd = [
        sys.executable,
        "-m",
        "run_opt",
        "smoke-test",
        "--run-dir",
        base_run_dir,
        "--resume",
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.stop_on_error:
        cmd.append("--stop-on-error")
    if args.no_isolate:
        cmd.append("--no-isolate")

    def _log_watch(message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{timestamp} {message}"
        logging.info(line)
        try:
            with open(watch_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except OSError:
            pass

    restarts = 0
    while True:
        _log_watch("Starting smoke-test run.")
        process = subprocess.Popen(cmd)
        last_activity = time.time()
        latest = _find_latest_smoke_activity_mtime(base_run_dir)
        if latest:
            last_activity = max(last_activity, latest)

        while True:
            return_code = process.poll()
            if return_code is not None:
                if return_code == 0:
                    _log_watch("Smoke-test completed successfully.")
                    print(f"Smoke test completed: {base_run_dir}")
                    return
                if _smoke_test_has_failures(base_run_dir):
                    _log_watch(
                        f"Smoke-test exited with failures (code {return_code})."
                    )
                    raise SystemExit(return_code)
                _log_watch(
                    f"Smoke-test exited unexpectedly (code {return_code}); restarting."
                )
                restarts += 1
                if args.watch_max_restarts and restarts > args.watch_max_restarts:
                    raise SystemExit("Smoke-test watch exceeded max restarts.")
                break

            latest = _find_latest_smoke_activity_mtime(base_run_dir)
            now = time.time()
            if latest:
                last_activity = max(last_activity, latest)
            if now - last_activity > args.watch_timeout:
                _log_watch(
                    f"Smoke-test logs stalled for {args.watch_timeout} seconds; restarting."
                )
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                restarts += 1
                if args.watch_max_restarts and restarts > args.watch_max_restarts:
                    raise SystemExit("Smoke-test watch exceeded max restarts.")
                break
            time.sleep(args.watch_interval)


def _load_resume_checkpoint(resume_dir):
    resume_path = Path(resume_dir).expanduser().resolve()
    if not resume_path.exists() or not resume_path.is_dir():
        raise ValueError(f"--resume path is not a directory: {resume_path}")
    checkpoint_path = resume_path / "checkpoint.json"
    if not checkpoint_path.exists():
        raise ValueError(f"Missing checkpoint.json in resume directory: {resume_path}")
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint.json in {resume_path}: {exc}") from exc
    xyz_file = checkpoint.get("xyz_file")
    if not xyz_file:
        raise ValueError("checkpoint.json is missing required key: xyz_file")
    xyz_path = Path(xyz_file).expanduser()
    if not xyz_path.is_absolute():
        xyz_path = (resume_path / xyz_path).resolve()
    else:
        xyz_path = xyz_path.resolve()
    if not xyz_path.exists():
        raise ValueError(f"XYZ file from checkpoint.json not found: {xyz_path}")

    config_used_path = resume_path / "config_used.json"
    config_raw = None
    config_source_path = None
    if config_used_path.exists():
        config_raw = config_used_path.read_text(encoding="utf-8")
        config_source_path = config_used_path
    else:
        config_raw = checkpoint.get("config_raw")
        config_source_path = checkpoint.get("config_source_path")
        if config_source_path:
            config_source_path = Path(config_source_path).expanduser()
            if not config_source_path.is_absolute():
                config_source_path = (resume_path / config_source_path).resolve()
            else:
                config_source_path = config_source_path.resolve()
            if config_source_path.exists() and config_raw is None:
                config_raw = config_source_path.read_text(encoding="utf-8")

    if not config_raw:
        raise ValueError("Unable to reconstruct config_raw from checkpoint/config_used.json.")

    return {
        "resume_dir": resume_path,
        "checkpoint_path": checkpoint_path,
        "xyz_file": str(xyz_path),
        "config_source_path": config_source_path,
        "config_raw": config_raw,
    }


def _read_text_file(path):
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _check_resume_config_mismatch(resume_state, policy):
    if policy == "ignore":
        return
    resume_raw = resume_state.get("config_raw")
    resume_hash = compute_text_hash(resume_raw)
    if resume_hash is None:
        return
    mismatches = []
    checkpoint_raw = None
    checkpoint_path = resume_state.get("checkpoint_path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint_payload = json.loads(
                Path(checkpoint_path).read_text(encoding="utf-8")
            )
            checkpoint_raw = checkpoint_payload.get("config_raw")
        except (OSError, json.JSONDecodeError):
            checkpoint_raw = None
    if checkpoint_raw:
        checkpoint_hash = compute_text_hash(checkpoint_raw)
        if checkpoint_hash and checkpoint_hash != resume_hash:
            mismatches.append(
                "config_used.json differs from checkpoint.json "
                f"(resume={resume_hash[:8]}, checkpoint={checkpoint_hash[:8]})"
            )
    config_source_path = resume_state.get("config_source_path")
    if config_source_path:
        config_source_raw = _read_text_file(config_source_path)
        if config_source_raw:
            config_source_hash = compute_text_hash(config_source_raw)
            if config_source_hash and config_source_hash != resume_hash:
                mismatches.append(
                    "config source differs from resume config "
                    f"({config_source_path}, resume={resume_hash[:8]}, "
                    f"source={config_source_hash[:8]})"
                )
    if not mismatches:
        return
    message = "Resume config mismatch detected: " + "; ".join(mismatches)
    if policy == "error":
        raise ValueError(message)
    logging.warning(message)


def _load_resume_status(run_metadata_path):
    if not run_metadata_path:
        return None
    try:
        if not os.path.exists(run_metadata_path):
            return None
        with open(run_metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return None
    return metadata.get("status")


def _parse_scan_dimension(spec):
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError("Scan dimension spec must not be empty.")
    dim_type = parts[0].lower()
    if dim_type not in ("bond", "angle", "dihedral"):
        raise ValueError("Scan dimension type must be bond, angle, or dihedral.")
    index_count = {"bond": 2, "angle": 3, "dihedral": 4}[dim_type]
    expected_len = 1 + index_count + 3
    if len(parts) != expected_len:
        raise ValueError(
            "Scan dimension spec '{spec}' must have {expected} fields.".format(
                spec=spec, expected=expected_len
            )
        )
    try:
        indices = [int(value) for value in parts[1 : 1 + index_count]]
    except ValueError as exc:
        raise ValueError("Scan dimension indices must be integers.") from exc
    try:
        start, end, step = (float(value) for value in parts[1 + index_count :])
    except ValueError as exc:
        raise ValueError("Scan dimension start/end/step must be numbers.") from exc
    dimension = {"type": dim_type, "start": start, "end": end, "step": step}
    for key, value in zip(("i", "j", "k", "l"), indices, strict=False):
        dimension[key] = value
    return dimension


def _apply_scan_cli_overrides(config, args):
    if not (args.scan_dimension or args.scan_grid or args.scan_mode):
        return config
    if args.scan_dimension is None:
        raise ValueError("--scan-dimension is required when using scan options.")
    dimensions = [_parse_scan_dimension(spec) for spec in args.scan_dimension]
    if len(dimensions) not in (1, 2):
        raise ValueError("Scan mode currently supports 1D or 2D dimensions only.")
    scan_config = {}
    base_scan = config.get("scan")
    if isinstance(base_scan, dict):
        scan_config.update(base_scan)
    for key in ("type", "i", "j", "k", "l", "start", "end", "step", "dimensions"):
        scan_config.pop(key, None)
    scan_config["dimensions"] = dimensions
    if args.scan_grid:
        if len(args.scan_grid) != len(dimensions):
            raise ValueError("--scan-grid entries must match scan dimension count.")
        grid = []
        for entry in args.scan_grid:
            values = [value.strip() for value in entry.split(",") if value.strip()]
            if not values:
                raise ValueError("--scan-grid entries must contain values.")
            try:
                grid.append([float(value) for value in values])
            except ValueError as exc:
                raise ValueError("--scan-grid values must be numbers.") from exc
        scan_config["grid"] = grid
    else:
        existing_grid = base_scan.get("grid") if isinstance(base_scan, dict) else None
        if isinstance(existing_grid, list) and len(existing_grid) == len(dimensions):
            scan_config["grid"] = existing_grid
        else:
            scan_config.pop("grid", None)
    if args.scan_mode:
        scan_config["mode"] = args.scan_mode
    config = dict(config)
    config["scan"] = scan_config
    config.pop("scan2d", None)
    config["calculation_mode"] = "scan"
    return config


def _parse_cli_args():
    parser = cli.build_parser()
    normalized_argv = cli._normalize_cli_args(sys.argv[1:])
    return parser.parse_args(normalized_argv)


def _maybe_auto_archive(command):
    if command not in ("run", "smoke-test"):
        return
    try:
        maybe_auto_archive_runs()
    except Exception as exc:
        logging.warning("Auto-archive check failed: %s", exc)


def _build_run_config_or_raise(config):
    try:
        return build_run_config(config)
    except ValueError as error:
        message = str(error)
        print(message, file=sys.stderr)
        logging.error(message)
        raise


def _run_doctor_command(_args):
    execution.run_doctor()


def _run_scan_point_command(args):
    from execution.stage_scan import run_scan_point_from_manifest

    run_scan_point_from_manifest(args.manifest, args.index)


def _run_queue_command(args):
    run_queue.ensure_queue_file(DEFAULT_QUEUE_PATH)
    if args.queue_command == "status":
        run_queue.reconcile_queue_entries(DEFAULT_QUEUE_PATH, DEFAULT_QUEUE_LOCK_PATH)
        with run_queue.queue_lock(DEFAULT_QUEUE_LOCK_PATH):
            queue_state = run_queue.load_queue(DEFAULT_QUEUE_PATH)
        run_queue.format_queue_status(queue_state)
        return
    if args.queue_command == "cancel":
        canceled, error = run_queue.cancel_queue_entry(
            DEFAULT_QUEUE_PATH,
            DEFAULT_QUEUE_LOCK_PATH,
            args.run_id,
        )
        if not canceled:
            raise ValueError(error)
        print(f"Canceled queued run: {args.run_id}")
        return
    if args.queue_command == "retry":
        retried, error = run_queue.requeue_queue_entry(
            DEFAULT_QUEUE_PATH,
            DEFAULT_QUEUE_LOCK_PATH,
            args.run_id,
            reason="retry",
        )
        if not retried:
            raise ValueError(error)
        print(f"Re-queued run: {args.run_id}")
        return
    if args.queue_command == "requeue-failed":
        count = run_queue.requeue_failed_entries(
            DEFAULT_QUEUE_PATH, DEFAULT_QUEUE_LOCK_PATH
        )
        print(f"Re-queued failed runs: {count}")
        return
    if args.queue_command == "prune":
        run_queue.load_queue(DEFAULT_QUEUE_PATH)
        removed, remaining = run_queue.prune_queue_entries(
            DEFAULT_QUEUE_PATH,
            DEFAULT_QUEUE_LOCK_PATH,
            args.keep_days,
            {"completed", "failed", "timeout", "canceled"},
        )
        print(f"Pruned queue entries: {removed} removed, {remaining} remaining.")
        return
    if args.queue_command == "archive":
        archive_path = run_queue.archive_queue(
            DEFAULT_QUEUE_PATH,
            DEFAULT_QUEUE_LOCK_PATH,
            args.path,
        )
        print(f"Archived queue entries to: {archive_path}")
        return
    raise ValueError(f"Unknown queue command: {args.queue_command}")


def _run_status_command(args):
    if args.recent and args.run_path:
        raise ValueError("--recent cannot be used with a run path.")
    if args.recent:
        run_queue.print_recent_statuses(args.recent)
        return
    if args.run_path:
        run_queue.print_status(args.run_path, DEFAULT_RUN_METADATA_PATH)
        return
    raise ValueError("status requires a run path or --recent.")


def _run_list_runs_command(args):
    runs_dir = args.runs_dir or get_runs_base_dir()
    entries = _load_run_entries(Path(runs_dir), limit=args.limit)
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now().isoformat(),
        "entries": entries,
    }
    print(json.dumps(payload))


def _run_validate_config_command(args):
    config_path = Path(args.config_path or args.config).expanduser().resolve()
    config, _config_raw = load_run_config(config_path)
    _build_run_config_or_raise(config)
    print(f"Config validation passed: {config_path}")


def _run_smoke_test_command(args):
    deps = run_opt_smoke.SmokeCommandDeps(
        default_solvent_map_path=DEFAULT_SOLVENT_MAP_PATH,
        smoke_test_xyz=SMOKE_TEST_XYZ,
        get_smoke_runs_base_dir=get_smoke_runs_base_dir,
        create_run_directory=create_run_directory,
        prepare_smoke_test_suite=_prepare_smoke_test_suite,
        run_smoke_test_watch=_run_smoke_test_watch,
        coerce_smoke_statuses=_coerce_smoke_statuses,
        prepare_smoke_test_run_dir=_prepare_smoke_test_run_dir,
        infer_smoke_case_status=_infer_smoke_case_status,
        update_smoke_progress=_update_smoke_progress,
        smoke_progress_status=_smoke_progress_status,
        load_smoke_test_status=_load_smoke_test_status,
        coerce_smoke_status_from_metadata=_coerce_smoke_status_from_metadata,
        write_smoke_skip_metadata=_write_smoke_skip_metadata,
        build_smoke_test_config=_build_smoke_test_config,
        run_smoke_test_case=_run_smoke_test_case,
        ensure_smoke_status_file=_ensure_smoke_status_file,
        format_subprocess_returncode=_format_subprocess_returncode,
    )
    run_opt_smoke.run_smoke_test_command(args, deps)


def _dispatch_non_run_command(args):
    handlers = {
        "doctor": _run_doctor_command,
        "scan-point": _run_scan_point_command,
        "queue": _run_queue_command,
        "status": _run_status_command,
        "list-runs": _run_list_runs_command,
        "validate-config": _run_validate_config_command,
        "smoke-test": _run_smoke_test_command,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return False
    handler(args)
    return True


def _validate_run_cli_args(args):
    if args.resume and args.run_dir:
        raise ValueError("--resume and --run-dir cannot be used together.")
    if args.resume and args.scan_dimension:
        raise ValueError("--scan-dimension cannot be used with --resume.")
    if args.resume and args.scan_grid:
        raise ValueError("--scan-grid cannot be used with --resume.")
    if args.resume and args.scan_mode:
        raise ValueError("--scan-mode cannot be used with --resume.")
    if args.resume and args.xyz_file:
        raise ValueError("xyz_file cannot be provided when using --resume.")


def _load_run_command_config(args):
    config_source_path = None
    if args.resume:
        resume_state = _load_resume_checkpoint(args.resume)
        args.xyz_file = resume_state["xyz_file"]
        args.run_dir = str(resume_state["resume_dir"])
        config_raw = resume_state["config_raw"]
        config_source_path = resume_state["config_source_path"]
        _check_resume_config_mismatch(
            resume_state,
            getattr(args, "resume_config_mismatch", "warn"),
        )
        if config_source_path is not None:
            args.config = str(config_source_path)
        else:
            args.config = str(resume_state["checkpoint_path"])
        try:
            config = json.loads(config_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid config JSON in resume data: {exc}") from exc
    else:
        if not args.xyz_file:
            raise ValueError("xyz_file is required unless --resume is used.")
        config_path = Path(args.config).expanduser().resolve()
        config, config_raw = load_run_config(config_path)
        args.config = str(config_path)
        config_source_path = config_path
    return config, config_raw, config_source_path


def _apply_run_cli_config_overrides(config, config_raw, args):
    if args.scan_dimension or args.scan_grid or args.scan_mode:
        config = _apply_scan_cli_overrides(config, args)
        config_raw = json.dumps(config, indent=2, ensure_ascii=False)
    if args.scan_result_csv:
        config = dict(config)
        config["scan_result_csv_file"] = args.scan_result_csv
        config_raw = json.dumps(config, indent=2, ensure_ascii=False)
    return config, config_raw


def _apply_resume_status_guard(args, config):
    if not args.resume:
        return
    metadata_candidate = config.run_metadata_file or DEFAULT_RUN_METADATA_PATH
    if os.path.isabs(metadata_candidate):
        run_metadata_path = Path(metadata_candidate)
    else:
        run_metadata_path = Path(args.run_dir) / metadata_candidate
    previous_status = _load_resume_status(str(run_metadata_path))
    args.resume_previous_status = previous_status
    if previous_status in TERMINAL_RESUME_STATUSES and not args.force_resume:
        raise ValueError(
            "Refusing to resume a {status} run without --force-resume.".format(
                status=previous_status
            )
        )
    if previous_status in TERMINAL_RESUME_STATUSES and args.force_resume:
        print(
            "Warning: resuming a {status} run with --force-resume.".format(
                status=previous_status
            ),
            file=sys.stderr,
        )


def _run_execution_with_smoke_signals(
    args, config, config_raw, config_source_path, run_in_background
):
    smoke_status_path = getenv_with_legacy(
        "PYSCF_AUTO_SMOKE_STATUS_PATH",
        "DFTFLOW_SMOKE_STATUS_PATH",
    )
    smoke_heartbeat_path = getenv_with_legacy(
        "PYSCF_AUTO_SMOKE_HEARTBEAT_PATH",
        "DFTFLOW_SMOKE_HEARTBEAT_PATH",
    )
    heartbeat_interval = getenv_with_legacy(
        "PYSCF_AUTO_SMOKE_HEARTBEAT_INTERVAL",
        "DFTFLOW_SMOKE_HEARTBEAT_INTERVAL",
    )
    exit_code = 1
    stop_heartbeat = None
    if smoke_heartbeat_path:
        interval = SMOKE_TEST_HEARTBEAT_INTERVAL
        if heartbeat_interval:
            try:
                interval = max(1, int(heartbeat_interval))
            except ValueError:
                interval = SMOKE_TEST_HEARTBEAT_INTERVAL
        stop_heartbeat = _start_smoke_heartbeat(smoke_heartbeat_path, interval)
    try:
        execution.run(args, config, config_raw, config_source_path, run_in_background)
        exit_code = 0
    finally:
        if stop_heartbeat:
            stop_heartbeat.set()
        if smoke_status_path:
            try:
                _write_smoke_status_path(smoke_status_path, exit_code)
            except OSError as exc:
                logging.warning(
                    "Failed to write smoke-test status in %s: %s",
                    smoke_status_path,
                    exc,
                )


def _run_command(args):
    run_in_background = bool(args.background and not args.no_background)
    if args.queue_runner:
        run_queue.run_queue_worker(
            os.path.abspath(sys.argv[0]),
            DEFAULT_QUEUE_PATH,
            DEFAULT_QUEUE_LOCK_PATH,
            DEFAULT_QUEUE_RUNNER_LOCK_PATH,
        )
        return

    _validate_run_cli_args(args)
    config, config_raw, config_source_path = _load_run_command_config(args)
    config, config_raw = _apply_run_cli_config_overrides(config, config_raw, args)
    config = _build_run_config_or_raise(config)
    _apply_resume_status_guard(args, config)
    _run_execution_with_smoke_signals(
        args,
        config,
        config_raw,
        config_source_path,
        run_in_background,
    )


def main():
    """Main function — redirects to the new .inp-based CLI."""
    from cli_new import main as new_main
    new_main()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        message = str(error)
        if message:
            print(message, file=sys.stderr)
        sys.exit(1)
