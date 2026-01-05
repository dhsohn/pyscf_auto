import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis

from run_opt_config import SMD_UNSUPPORTED_SOLVENT_KEYS, validate_run_config
from run_queue import format_queue_status, load_queue
from run_opt_paths import get_app_base_dir, get_runs_base_dir


REFRESH_INTERVAL_MS = 2000
DISK_USAGE_INTERVAL_MS = 60000
LOG_TAIL_LINES = 2000
RUN_LIST_LIMIT = 20
RUNS_INDEX_SCHEMA_VERSION = 1
RUNS_INDEX_FILENAME = "index.json"


def _normalize_solvent_key(name):
    return "".join(char for char in str(name).lower() if char.isalnum())


@dataclass
class RunEntry:
    run_dir: str
    metadata_path: str
    status: str
    started_at: str | None
    calculation_mode: str | None
    basis: str | None
    xc: str | None


def _ensure_runs_dir() -> Path:
    runs_dir = Path(get_runs_base_dir())
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _format_bytes(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "n/a"
    value = float(size_bytes)
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def _walk_directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _dirs, files in os.walk(path, followlinks=False):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                continue
    return total


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    du_path = shutil.which("du")
    if du_path:
        try:
            completed = subprocess.run(
                [du_path, "-sk", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                value = completed.stdout.strip().split()[0]
                return int(value) * 1024
        except (OSError, ValueError, IndexError):
            pass
    return _walk_directory_size(path)


def _calculate_disk_usage(path: Path) -> dict:
    result: dict[str, int | str | None] = {
        "runs_bytes": None,
        "free_bytes": None,
        "total_bytes": None,
        "error": None,
    }
    try:
        usage = shutil.disk_usage(path)
        result["free_bytes"] = usage.free
        result["total_bytes"] = usage.total
    except OSError as exc:
        result["error"] = str(exc)
    try:
        result["runs_bytes"] = _directory_size_bytes(path)
    except OSError as exc:
        if not result.get("error"):
            result["error"] = str(exc)
    return result


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque(handle, maxlen=max_lines))
    except OSError:
        return []


def _axis_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        pad = 1.0 if vmin == 0 else abs(vmin) * 0.1
        return (vmin - pad, vmax + pad)
    pad = (vmax - vmin) * 0.05
    return (vmin - pad, vmax + pad)


def _safe_results(payload: dict | None) -> dict:
    if not payload:
        return {}
    results = payload.get("results")
    if isinstance(results, dict):
        return results
    return {}


def _iter_metadata_files(base_dir: Path):
    if not base_dir.exists():
        return
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        metadata_path = entry / "metadata.json"
        if metadata_path.exists():
            yield metadata_path


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


def _load_runs_index(base_dir: Path) -> dict | None:
    index_path = base_dir / RUNS_INDEX_FILENAME
    payload = _read_json(index_path)
    if not payload or not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("entries"), list):
        return None
    return payload


def _write_runs_index(base_dir: Path, entries: list[dict]):
    index_path = base_dir / RUNS_INDEX_FILENAME
    base_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RUNS_INDEX_SCHEMA_VERSION,
        "updated_at": datetime.now().isoformat(),
        "entries": entries,
    }
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


def _build_runs_index(base_dir: Path) -> dict:
    entries: list[dict] = []
    for metadata_path in _iter_metadata_files(base_dir):
        metadata = _read_json(metadata_path)
        if not metadata:
            continue
        entries.append(_index_entry_from_metadata(metadata_path, metadata))
    _write_runs_index(base_dir, entries)
    return {"entries": entries}


def _run_entry_from_index(item: dict, base_dir: Path) -> RunEntry | None:
    run_dir = item.get("run_dir") or item.get("run_directory")
    metadata_path = item.get("metadata_path")
    if not run_dir:
        return None
    run_dir_path = Path(run_dir)
    if not run_dir_path.is_absolute():
        run_dir_path = (base_dir / run_dir_path).resolve()
    if run_dir_path.parent != base_dir.resolve():
        return None
    if not metadata_path:
        metadata_path = str(run_dir_path / "metadata.json")
    metadata_path = str(Path(metadata_path).resolve())
    return RunEntry(
        run_dir=str(run_dir_path),
        metadata_path=metadata_path,
        status=item.get("status", "unknown"),
        started_at=item.get("run_started_at"),
        calculation_mode=item.get("calculation_mode"),
        basis=item.get("basis"),
        xc=item.get("xc"),
    )


def _run_entry_from_metadata(metadata_path: Path, metadata: dict) -> RunEntry:
    run_dir = metadata.get("run_directory") or str(metadata_path.parent)
    return RunEntry(
        run_dir=str(Path(run_dir).resolve()),
        metadata_path=str(metadata_path.resolve()),
        status=metadata.get("status", "unknown"),
        started_at=metadata.get("run_started_at"),
        calculation_mode=metadata.get("calculation_mode"),
        basis=metadata.get("basis"),
        xc=metadata.get("xc"),
    )


def _load_run_entries(base_dir: Path, limit: int | None = None) -> list[RunEntry]:
    entries: list[RunEntry] = []
    index_state = _load_runs_index(base_dir)
    if index_state is None:
        index_state = _build_runs_index(base_dir)
    for item in index_state.get("entries") or []:
        if not isinstance(item, dict):
            continue
        entry = _run_entry_from_index(item, base_dir)
        if entry:
            entries.append(entry)
    entries.sort(
        key=lambda item: os.path.getmtime(item.metadata_path)
        if os.path.exists(item.metadata_path)
        else 0,
        reverse=True,
    )
    if limit:
        entries = entries[:limit]
    return entries


class RunSubmitWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, on_submit=None):
        super().__init__(parent)
        self._on_submit = on_submit
        self.setMinimumWidth(520)

        self._atom_labels: list[str] = []

        form = QtWidgets.QGridLayout()
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        row = 0
        self.xyz_path = QtWidgets.QLineEdit()
        xyz_button = QtWidgets.QPushButton("Browse...")
        xyz_button.clicked.connect(self._pick_xyz)
        form.addWidget(QtWidgets.QLabel("XYZ file"), row, 0)
        form.addWidget(self._wrap_picker(self.xyz_path, xyz_button), row, 1, 1, 3)
        row += 1

        self.calc_mode = QtWidgets.QComboBox()
        self.calc_mode.addItems(
            [
                "Optimization",
                "Constrained relaxation",
                "Frequency",
                "Single point",
                "Scan",
            ]
        )
        self.calc_mode.currentIndexChanged.connect(self._update_mode_panel)
        form.addWidget(QtWidgets.QLabel("Simulation"), row, 0)
        form.addWidget(self.calc_mode, row, 1, 1, 3)
        row += 1

        self.basis_box = QtWidgets.QComboBox()
        self.basis_box.setEditable(True)
        basis_values = [
            "sto-3g",
            "3-21g",
            "6-31g",
            "6-31g*",
            "6-31g**",
            "6-31+g",
            "6-31+g*",
            "6-31+g**",
            "6-31g(d)",
            "6-31g(d,p)",
            "6-31+g(d)",
            "6-31+g(d,p)",
            "6-311g",
            "6-311g*",
            "6-311g**",
            "6-311+g",
            "6-311+g*",
            "6-311+g**",
            "6-311++g",
            "6-311++g**",
            "def2-svp",
            "def2-svpd",
            "def2-tzvp",
            "def2-tzvpd",
            "def2-tzvpp",
            "def2-qzvp",
            "def2-qzvpd",
            "def2-qzvpp",
            "cc-pvdz",
            "cc-pvtz",
            "cc-pvqz",
            "aug-cc-pvdz",
            "aug-cc-pvtz",
            "aug-cc-pvqz",
            "cc-pcvdz",
            "cc-pcvtz",
            "def2-svp-jkfit",
            "def2-tzvp-jkfit",
            "def2-svp-ri",
            "def2-tzvp-ri",
        ]
        self.basis_box.addItems(basis_values)
        basis_completer = QtWidgets.QCompleter(basis_values, self)
        basis_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        basis_completer.setFilterMode(QtCore.Qt.MatchContains)
        self.basis_box.setCompleter(basis_completer)
        form.addWidget(QtWidgets.QLabel("Basis"), row, 0)
        form.addWidget(self.basis_box, row, 1)

        self.xc_box = QtWidgets.QComboBox()
        self.xc_box.setEditable(True)
        xc_values = [
            "b3lyp",
            "pbe0",
            "pbe",
            "bp86",
            "blyp",
            "tpss",
            "rev-tpss",
            "m06",
            "m06-2x",
            "m06-l",
            "wb97x-d",
            "wb97m-v",
            "cam-b3lyp",
            "b97-d",
            "b97-3c",
            "b97x-d",
            "pbe-d3bj",
            "pbe0-d3bj",
            "b3lyp-d3bj",
            "scan",
            "r2scan",
            "r2scan-3c",
        ]
        self.xc_box.addItems(xc_values)
        xc_completer = QtWidgets.QCompleter(xc_values, self)
        xc_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        xc_completer.setFilterMode(QtCore.Qt.MatchContains)
        self.xc_box.setCompleter(xc_completer)
        form.addWidget(QtWidgets.QLabel("XC"), row, 2)
        form.addWidget(self.xc_box, row, 3)
        row += 1

        self.solvent_box = QtWidgets.QComboBox()
        self.solvent_box.setEditable(True)
        self.solvent_box.addItems(
            [
                "water",
                "acetonitrile",
                "methanol",
                "ethanol",
                "acetone",
                "dichloromethane",
            ]
        )
        form.addWidget(QtWidgets.QLabel("Solvent"), row, 0)
        form.addWidget(self.solvent_box, row, 1)

        self.solvent_model_box = QtWidgets.QComboBox()
        self.solvent_model_box.addItems(["none", "pcm", "smd"])
        form.addWidget(QtWidgets.QLabel("Solvent model"), row, 2)
        form.addWidget(self.solvent_model_box, row, 3)
        row += 1

        self.dispersion_box = QtWidgets.QComboBox()
        self.dispersion_box.addItems(["none", "d3bj", "d3zero", "d4"])
        form.addWidget(QtWidgets.QLabel("Dispersion"), row, 0)
        form.addWidget(self.dispersion_box, row, 1)

        self.charge_override = QtWidgets.QCheckBox("Override charge/spin/multiplicity")
        self.charge_override.toggled.connect(self._toggle_charge_fields)
        form.addWidget(QtWidgets.QLabel("Charge/spin override"), row, 2)
        form.addWidget(self.charge_override, row, 3)
        row += 1

        self.charge_value = QtWidgets.QSpinBox()
        self.charge_value.setRange(-20, 20)
        self.charge_value.setValue(0)
        form.addWidget(QtWidgets.QLabel("Charge"), row, 0)
        form.addWidget(self.charge_value, row, 1)

        self.spin_value = QtWidgets.QSpinBox()
        self.spin_value.setRange(0, 20)
        self.spin_value.setValue(0)
        form.addWidget(QtWidgets.QLabel("Spin (2S)"), row, 2)
        form.addWidget(self.spin_value, row, 3)
        row += 1

        self.multiplicity_value = QtWidgets.QSpinBox()
        self.multiplicity_value.setRange(1, 21)
        self.multiplicity_value.setValue(1)
        form.addWidget(QtWidgets.QLabel("Multiplicity"), row, 0)
        form.addWidget(self.multiplicity_value, row, 1)

        self.threads_value = QtWidgets.QSpinBox()
        self.threads_value.setRange(0, 128)
        self.threads_value.setValue(0)
        self.threads_value.setSpecialValueText("auto")
        form.addWidget(QtWidgets.QLabel("Threads"), row, 2)
        form.addWidget(self.threads_value, row, 3)
        row += 1

        self.memory_value = QtWidgets.QDoubleSpinBox()
        self.memory_value.setRange(0.0, 1024.0)
        self.memory_value.setDecimals(2)
        self.memory_value.setSingleStep(0.5)
        self.memory_value.setValue(0.0)
        self.memory_value.setSpecialValueText("auto")
        form.addWidget(QtWidgets.QLabel("Memory (GB)"), row, 0)
        form.addWidget(self.memory_value, row, 1)

        self.enforce_memory_limit = QtWidgets.QCheckBox("Enforce OS memory limit")
        form.addWidget(QtWidgets.QLabel("Memory limit"), row, 2)
        form.addWidget(self.enforce_memory_limit, row, 3)
        row += 1

        self.scf_max_cycle = QtWidgets.QSpinBox()
        self.scf_max_cycle.setRange(0, 2000)
        self.scf_max_cycle.setValue(0)
        self.scf_max_cycle.setSpecialValueText("default")
        form.addWidget(QtWidgets.QLabel("SCF max cycle"), row, 0)
        form.addWidget(self.scf_max_cycle, row, 1)

        self.scf_conv_tol = QtWidgets.QDoubleSpinBox()
        self.scf_conv_tol.setRange(0.0, 1.0)
        self.scf_conv_tol.setDecimals(10)
        self.scf_conv_tol.setSingleStep(0.000001)
        self.scf_conv_tol.setValue(0.0)
        self.scf_conv_tol.setSpecialValueText("default")
        form.addWidget(QtWidgets.QLabel("SCF conv tol"), row, 2)
        form.addWidget(self.scf_conv_tol, row, 3)
        row += 1

        self.scf_diis = QtWidgets.QComboBox()
        self.scf_diis.addItems(["default", "on", "off", "8", "12"])
        form.addWidget(QtWidgets.QLabel("SCF DIIS"), row, 0)
        form.addWidget(self.scf_diis, row, 1)

        self.scf_level_shift = QtWidgets.QDoubleSpinBox()
        self.scf_level_shift.setRange(0.0, 5.0)
        self.scf_level_shift.setDecimals(4)
        self.scf_level_shift.setSingleStep(0.05)
        self.scf_level_shift.setValue(0.0)
        self.scf_level_shift.setSpecialValueText("default")
        form.addWidget(QtWidgets.QLabel("SCF level shift"), row, 2)
        form.addWidget(self.scf_level_shift, row, 3)
        row += 1

        self.scf_damping = QtWidgets.QDoubleSpinBox()
        self.scf_damping.setRange(0.0, 1.0)
        self.scf_damping.setDecimals(4)
        self.scf_damping.setSingleStep(0.05)
        self.scf_damping.setValue(0.0)
        self.scf_damping.setSpecialValueText("default")
        form.addWidget(QtWidgets.QLabel("SCF damping"), row, 0)
        form.addWidget(self.scf_damping, row, 1)
        row += 1

        self.mode_panel = QtWidgets.QStackedWidget()
        self.mode_panel.addWidget(self._build_empty_panel())
        self.mode_panel.addWidget(self._build_optimization_panel())
        self.mode_panel.addWidget(self._build_constraint_panel())
        self.mode_panel.addWidget(self._build_scan_panel())
        form.addWidget(self.mode_panel, row, 0, 1, 4)

        self.submit_button = QtWidgets.QPushButton("Start Run")
        self.submit_button.clicked.connect(self._emit_submit)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.submit_button)
        self.setLayout(layout)
        self._update_mode_panel()
        self._toggle_charge_fields(False)

    def _wrap_picker(self, line_edit, button):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        container.setLayout(layout)
        return container

    def _pick_file(self, target: QtWidgets.QLineEdit):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file")
        if path:
            target.setText(path)

    def _emit_submit(self):
        if self._on_submit:
            self._on_submit()

    def _toggle_charge_fields(self, enabled):
        self.charge_value.setEnabled(enabled)
        self.spin_value.setEnabled(enabled)
        self.multiplicity_value.setEnabled(enabled)

    def _pick_xyz(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select XYZ file")
        if not path:
            return
        self.xyz_path.setText(path)
        self._load_xyz_atoms(path)

    def _load_xyz_atoms(self, path: str):
        labels: list[str] = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                first = handle.readline().strip()
                count = int(first)
                handle.readline()
                for idx in range(count):
                    line = handle.readline()
                    if not line:
                        break
                    parts = line.split()
                    symbol = parts[0] if parts else "X"
                    labels.append(f"{idx}: {symbol}")
        except (OSError, ValueError):
            labels = []
        self._atom_labels = labels
        self._refresh_atom_combos()

    def _refresh_atom_combos(self):
        combos = [
            self.constraint_i,
            self.constraint_j,
            self.constraint_k,
            self.constraint_l,
            self.scan_i,
            self.scan_j,
            self.scan_k,
            self.scan_l,
        ]
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._atom_labels)
            combo.blockSignals(False)

    def _build_empty_panel(self):
        panel = QtWidgets.QWidget()
        panel.setLayout(QtWidgets.QVBoxLayout())
        panel.layout().addWidget(QtWidgets.QLabel("No additional options."))
        return panel

    def _build_optimization_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        self.opt_optimizer = QtWidgets.QComboBox()
        self.opt_optimizer.addItems(["bfgs", "lbfgs", "fire", "gpmin", "mdmin", "sella"])
        self.opt_fmax = QtWidgets.QDoubleSpinBox()
        self.opt_fmax.setRange(0.0001, 10.0)
        self.opt_fmax.setDecimals(4)
        self.opt_fmax.setSingleStep(0.01)
        self.opt_fmax.setValue(0.05)
        self.opt_steps = QtWidgets.QSpinBox()
        self.opt_steps.setRange(1, 5000)
        self.opt_steps.setValue(200)
        layout.addWidget(QtWidgets.QLabel("Optimizer"), 0, 0)
        layout.addWidget(self.opt_optimizer, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Fmax"), 0, 2)
        layout.addWidget(self.opt_fmax, 0, 3)
        layout.addWidget(QtWidgets.QLabel("Max steps"), 1, 0)
        layout.addWidget(self.opt_steps, 1, 1)
        panel.setLayout(layout)
        return panel

    def _build_constraint_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        self.constraint_type = QtWidgets.QComboBox()
        self.constraint_type.addItems(["bond", "angle", "dihedral"])
        self.constraint_type.currentIndexChanged.connect(self._update_constraint_fields)
        self.constraint_i = QtWidgets.QComboBox()
        self.constraint_j = QtWidgets.QComboBox()
        self.constraint_k = QtWidgets.QComboBox()
        self.constraint_l = QtWidgets.QComboBox()
        self.constraint_value = QtWidgets.QDoubleSpinBox()
        self.constraint_value.setRange(-360.0, 360.0)
        self.constraint_value.setDecimals(4)
        self.constraint_value.setSingleStep(0.1)
        layout.addWidget(QtWidgets.QLabel("Constraint type"), 0, 0)
        layout.addWidget(self.constraint_type, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Target value"), 0, 2)
        layout.addWidget(self.constraint_value, 0, 3)
        layout.addWidget(QtWidgets.QLabel("Atom i"), 1, 0)
        layout.addWidget(self.constraint_i, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Atom j"), 1, 2)
        layout.addWidget(self.constraint_j, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Atom k"), 2, 0)
        layout.addWidget(self.constraint_k, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Atom l"), 2, 2)
        layout.addWidget(self.constraint_l, 2, 3)
        self.constraint_optimizer = QtWidgets.QComboBox()
        self.constraint_optimizer.addItems(["bfgs", "lbfgs", "fire", "gpmin", "mdmin", "sella"])
        self.constraint_fmax = QtWidgets.QDoubleSpinBox()
        self.constraint_fmax.setRange(0.0001, 10.0)
        self.constraint_fmax.setDecimals(4)
        self.constraint_fmax.setSingleStep(0.01)
        self.constraint_fmax.setValue(0.05)
        self.constraint_steps = QtWidgets.QSpinBox()
        self.constraint_steps.setRange(1, 5000)
        self.constraint_steps.setValue(200)
        layout.addWidget(QtWidgets.QLabel("Optimizer"), 3, 0)
        layout.addWidget(self.constraint_optimizer, 3, 1)
        layout.addWidget(QtWidgets.QLabel("Fmax"), 3, 2)
        layout.addWidget(self.constraint_fmax, 3, 3)
        layout.addWidget(QtWidgets.QLabel("Max steps"), 4, 0)
        layout.addWidget(self.constraint_steps, 4, 1)
        panel.setLayout(layout)
        self._update_constraint_fields()
        return panel

    def _build_scan_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        self.scan_type = QtWidgets.QComboBox()
        self.scan_type.addItems(["bond", "angle", "dihedral"])
        self.scan_type.currentIndexChanged.connect(self._update_scan_fields)
        self.scan_mode = QtWidgets.QComboBox()
        self.scan_mode.addItems(["optimization", "single_point"])
        self.scan_i = QtWidgets.QComboBox()
        self.scan_j = QtWidgets.QComboBox()
        self.scan_k = QtWidgets.QComboBox()
        self.scan_l = QtWidgets.QComboBox()
        self.scan_start = QtWidgets.QDoubleSpinBox()
        self.scan_start.setRange(-360.0, 360.0)
        self.scan_start.setDecimals(4)
        self.scan_start.setSingleStep(0.1)
        self.scan_end = QtWidgets.QDoubleSpinBox()
        self.scan_end.setRange(-360.0, 360.0)
        self.scan_end.setDecimals(4)
        self.scan_end.setSingleStep(0.1)
        self.scan_step = QtWidgets.QDoubleSpinBox()
        self.scan_step.setRange(0.0001, 360.0)
        self.scan_step.setDecimals(4)
        self.scan_step.setSingleStep(0.1)
        layout.addWidget(QtWidgets.QLabel("Scan type"), 0, 0)
        layout.addWidget(self.scan_type, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Scan mode"), 0, 2)
        layout.addWidget(self.scan_mode, 0, 3)
        layout.addWidget(QtWidgets.QLabel("Atom i"), 1, 0)
        layout.addWidget(self.scan_i, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Atom j"), 1, 2)
        layout.addWidget(self.scan_j, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Atom k"), 2, 0)
        layout.addWidget(self.scan_k, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Atom l"), 2, 2)
        layout.addWidget(self.scan_l, 2, 3)
        layout.addWidget(QtWidgets.QLabel("Start"), 3, 0)
        layout.addWidget(self.scan_start, 3, 1)
        layout.addWidget(QtWidgets.QLabel("End"), 3, 2)
        layout.addWidget(self.scan_end, 3, 3)
        layout.addWidget(QtWidgets.QLabel("Step"), 4, 0)
        layout.addWidget(self.scan_step, 4, 1)
        panel.setLayout(layout)
        self._update_scan_fields()
        return panel

    def _update_constraint_fields(self):
        constraint_type = self.constraint_type.currentText()
        needs_k = constraint_type in ("angle", "dihedral")
        needs_l = constraint_type == "dihedral"
        self.constraint_k.setEnabled(needs_k)
        self.constraint_l.setEnabled(needs_l)
        if constraint_type == "bond":
            self.constraint_value.setRange(0.0, 10.0)
            self.constraint_value.setSingleStep(0.01)
        elif constraint_type == "angle":
            self.constraint_value.setRange(0.0, 180.0)
            self.constraint_value.setSingleStep(1.0)
        else:
            self.constraint_value.setRange(-180.0, 180.0)
            self.constraint_value.setSingleStep(1.0)

    def _update_scan_fields(self):
        scan_type = self.scan_type.currentText()
        needs_k = scan_type in ("angle", "dihedral")
        needs_l = scan_type == "dihedral"
        self.scan_k.setEnabled(needs_k)
        self.scan_l.setEnabled(needs_l)

    def _update_mode_panel(self):
        mode = self.calc_mode.currentText()
        if mode == "Optimization":
            self.mode_panel.setCurrentIndex(1)
        elif mode == "Constrained relaxation":
            self.mode_panel.setCurrentIndex(2)
        elif mode == "Scan":
            self.mode_panel.setCurrentIndex(3)
        else:
            self.mode_panel.setCurrentIndex(0)

    def get_values(self):
        if not self.xyz_path.text().strip():
            return {"xyz": None}
        basis = self.basis_box.currentText().strip()
        xc = self.xc_box.currentText().strip()
        solvent = self.solvent_box.currentText().strip()
        solvent_model = self.solvent_model_box.currentText()
        if solvent_model == "none":
            solvent_model = None
        if solvent_model and solvent_model.lower() == "smd":
            if _normalize_solvent_key(solvent) in SMD_UNSUPPORTED_SOLVENT_KEYS:
                return {
                    "xyz": None,
                    "error": (
                        f"SMD solvent '{solvent}' is not supported by PySCF SMD. "
                        "Use PCM or choose another solvent."
                    ),
                }
        dispersion = self.dispersion_box.currentText()
        if dispersion == "none":
            dispersion = None
        mode_text = self.calc_mode.currentText()
        calculation_mode = {
            "Optimization": "optimization",
            "Constrained relaxation": "optimization",
            "Frequency": "frequency",
            "Single point": "single_point",
            "Scan": "scan",
        }[mode_text]

        if mode_text in ("Constrained relaxation", "Scan") and not self._atom_labels:
            return {"xyz": None, "error": "Unable to read atoms from XYZ file."}

        def _combo_index(combo):
            index = combo.currentIndex()
            return index if index >= 0 else None

        config = {
            "basis": basis,
            "xc": xc,
            "solvent": solvent,
            "solvent_model": solvent_model,
            "dispersion": dispersion,
            "calculation_mode": calculation_mode,
        }

        threads = self.threads_value.value()
        if threads > 0:
            config["threads"] = int(threads)
        memory_gb = float(self.memory_value.value())
        if memory_gb > 0:
            config["memory_gb"] = memory_gb
            if self.enforce_memory_limit.isChecked():
                config["enforce_os_memory_limit"] = True

        scf = {}
        if self.scf_max_cycle.value() > 0:
            scf["max_cycle"] = int(self.scf_max_cycle.value())
        if self.scf_conv_tol.value() > 0:
            scf["conv_tol"] = float(self.scf_conv_tol.value())
        diis_choice = self.scf_diis.currentText()
        if diis_choice == "on":
            scf["diis"] = True
        elif diis_choice == "off":
            scf["diis"] = False
        elif diis_choice.isdigit():
            scf["diis"] = int(diis_choice)
        if self.scf_level_shift.value() > 0:
            scf["level_shift"] = float(self.scf_level_shift.value())
        if self.scf_damping.value() > 0:
            scf["damping"] = float(self.scf_damping.value())
        if scf:
            config["scf"] = scf

        if mode_text == "Optimization":
            optimizer_block = {
                "optimizer": self.opt_optimizer.currentText().strip(),
                "fmax": float(self.opt_fmax.value()),
                "steps": int(self.opt_steps.value()),
            }
            config["optimizer"] = {"mode": "minimum", "ase": optimizer_block}

        if mode_text == "Constrained relaxation":
            optimizer_block = {
                "optimizer": self.constraint_optimizer.currentText().strip(),
                "fmax": float(self.constraint_fmax.value()),
                "steps": int(self.constraint_steps.value()),
            }
            config["optimizer"] = {"mode": "minimum", "ase": optimizer_block}
            constraint_type = self.constraint_type.currentText()
            constraint = {
                "i": _combo_index(self.constraint_i),
                "j": _combo_index(self.constraint_j),
            }
            if constraint_type in ("angle", "dihedral"):
                constraint["k"] = _combo_index(self.constraint_k)
            if constraint_type == "dihedral":
                constraint["l"] = _combo_index(self.constraint_l)
            if any(value is None for value in constraint.values()):
                return {"xyz": None, "error": "Select all constraint atom indices."}
            value = float(self.constraint_value.value())
            if constraint_type == "bond":
                constraint["length"] = value
                config["constraints"] = {"bonds": [constraint]}
            elif constraint_type == "angle":
                constraint["angle"] = value
                config["constraints"] = {"angles": [constraint]}
            else:
                constraint["dihedral"] = value
                config["constraints"] = {"dihedrals": [constraint]}

        if mode_text == "Scan":
            scan_type = self.scan_type.currentText()
            scan_config = {
                "type": scan_type,
                "i": _combo_index(self.scan_i),
                "j": _combo_index(self.scan_j),
                "start": float(self.scan_start.value()),
                "end": float(self.scan_end.value()),
                "step": float(self.scan_step.value()),
                "mode": self.scan_mode.currentText(),
            }
            if scan_type in ("angle", "dihedral"):
                scan_config["k"] = _combo_index(self.scan_k)
            if scan_type == "dihedral":
                scan_config["l"] = _combo_index(self.scan_l)
            if any(value is None for key, value in scan_config.items() if key in {"i", "j", "k", "l"}):
                return {"xyz": None, "error": "Select all scan atom indices."}
            config["scan"] = scan_config

        xyz_metadata = None
        if self.charge_override.isChecked():
            xyz_metadata = {
                "charge": int(self.charge_value.value()),
                "spin": int(self.spin_value.value()),
                "multiplicity": int(self.multiplicity_value.value()),
            }

        return {
            "xyz": self.xyz_path.text().strip() or None,
            "config": config,
            "xyz_metadata": xyz_metadata,
        }


class QueueDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, on_submit=None):
        super().__init__(parent)
        self.setWindowTitle("Queue")
        self.setMinimumSize(900, 600)
        self._on_submit = on_submit

        splitter = QtWidgets.QSplitter()
        self.run_form = RunSubmitWidget(self, on_submit=self._handle_submit)
        splitter.addWidget(self.run_form)

        queue_panel = QtWidgets.QWidget()
        queue_layout = QtWidgets.QVBoxLayout()
        queue_layout.setContentsMargins(0, 0, 0, 0)
        self.queue_view = QtWidgets.QPlainTextEdit()
        self.queue_view.setReadOnly(True)
        self.queue_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        queue_layout.addWidget(self.queue_view)
        queue_panel.setLayout(queue_layout)
        splitter.addWidget(queue_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

        self._refresh_queue()
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_queue)
        self.refresh_timer.start(2000)

    def _refresh_queue(self):
        try:
            from run_queue import reconcile_queue_entries
            reconcile_queue_entries()
            queue_state = load_queue()
            formatted = format_queue_status(queue_state, print_output=False)
            if isinstance(formatted, str):
                text = formatted
            else:
                text = "\n".join(formatted)
            self.queue_view.setPlainText(text)
        except Exception as exc:
            self.queue_view.setPlainText(f"Unable to load queue: {exc}")

    def _handle_submit(self):
        if self._on_submit:
            self._on_submit(self.run_form.get_values())


class DFTFlowWindow(QtWidgets.QMainWindow):
    disk_usage_ready = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DFTFlow")
        self.resize(1200, 800)

        self.runs_dir = _ensure_runs_dir()
        self.current_run: RunEntry | None = None
        self.manual_runs: dict[str, RunEntry] = {}
        self.current_log_lines: list[str] = []
        self.current_query = ""
        self._disk_usage_inflight = False
        self.disk_usage_ready.connect(self._apply_disk_usage)

        self._build_ui()
        self._refresh_runs()

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_runs)
        self.refresh_timer.start(REFRESH_INTERVAL_MS)

        self.disk_usage_timer = QtCore.QTimer(self)
        self.disk_usage_timer.timeout.connect(self._refresh_disk_usage)
        self.disk_usage_timer.start(DISK_USAGE_INTERVAL_MS)
        self._refresh_disk_usage()

    def _log_gui_error(self, label, exc):
        error_path = Path(get_app_base_dir()) / "gui_errors.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{label}] {exc}\n")
            handle.write(traceback.format_exc())
            handle.write("\n")

    def _build_ui(self):
        toolbar = QtWidgets.QToolBar()
        self.addToolBar(toolbar)

        queue_action = QtGui.QAction("Queue", self)
        queue_action.triggered.connect(self._open_queue_dialog)
        toolbar.addAction(queue_action)

        stop_action = QtGui.QAction("Stop Run", self)
        stop_action.triggered.connect(self._stop_run)
        toolbar.addAction(stop_action)

        open_runs_action = QtGui.QAction("Open Runs Folder", self)
        open_runs_action.triggered.connect(self._open_runs_folder)
        toolbar.addAction(open_runs_action)

        open_run_action = QtGui.QAction("Open Run...", self)
        open_run_action.triggered.connect(self._open_run_dialog)
        toolbar.addAction(open_run_action)

        splitter = QtWidgets.QSplitter()

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setContentsMargins(8, 8, 8, 8)
        self.run_list = QtWidgets.QListWidget()
        self.run_list.itemSelectionChanged.connect(self._select_run)
        left_layout.addWidget(self.run_list, 1)
        self.disk_usage_label = QtWidgets.QLabel("Disk: calculating...")
        self.disk_usage_label.setWordWrap(True)
        left_layout.addWidget(self.disk_usage_label)
        left_panel.setLayout(left_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.summary_view = QtWidgets.QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        log_tab = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        self.log_search = QtWidgets.QLineEdit()
        self.log_search.setPlaceholderText("Filter log...")
        self.log_search.textChanged.connect(self._update_log_view)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        log_layout.addWidget(self.log_search)
        log_layout.addWidget(self.log_view)
        log_tab.setLayout(log_layout)

        results_tab = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout()
        self.results_info = QtWidgets.QLabel("No results available.")
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QtGui.QPainter.Antialiasing)
        results_layout.addWidget(self.results_info)
        results_layout.addWidget(self.chart_view, 1)
        results_tab.setLayout(results_layout)

        self.tabs.addTab(self.summary_view, "Summary")
        self.tabs.addTab(log_tab, "Logs")
        self.tabs.addTab(results_tab, "Results")

        splitter.addWidget(left_panel)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def _submit_run(self, values):
        if values.get("error"):
            QtWidgets.QMessageBox.warning(self, "Invalid input", values["error"])
            return
        if not values.get("xyz") or not values.get("config"):
            QtWidgets.QMessageBox.warning(
                self, "Missing input", "XYZ and simulation settings are required."
            )
            return
        config = values["config"]
        try:
            validate_run_config(config)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Invalid settings", f"Config validation failed: {exc}"
            )
            return
        config_dir = Path(self.runs_dir) / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = config_dir / f"run_config_{timestamp}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        input_xyz = values["xyz"]
        if values.get("xyz_metadata"):
            xyz_dir = Path(self.runs_dir) / "inputs"
            xyz_dir.mkdir(parents=True, exist_ok=True)
            xyz_path = xyz_dir / f"input_{timestamp}.xyz"
            try:
                with open(input_xyz, "r", encoding="utf-8", errors="replace") as handle:
                    lines = handle.readlines()
                if not lines:
                    raise ValueError("XYZ file is empty.")
                atom_line = lines[0].rstrip("\n")
                comment = lines[1].rstrip("\n") if len(lines) > 1 else ""
                meta_parts = [
                    f"charge={values['xyz_metadata']['charge']}",
                    f"spin={values['xyz_metadata']['spin']}",
                    f"multiplicity={values['xyz_metadata']['multiplicity']}",
                ]
                if comment:
                    comment = f"{comment} {' '.join(meta_parts)}"
                else:
                    comment = " ".join(meta_parts)
                new_lines = [atom_line + "\n", comment + "\n"]
                new_lines.extend(lines[2:] if len(lines) > 2 else [])
                with xyz_path.open("w", encoding="utf-8") as handle:
                    handle.writelines(new_lines)
                input_xyz = str(xyz_path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "XYZ metadata", f"Failed to apply charge/spin: {exc}"
                )
                return
        command = [
            sys.executable,
            "-m",
            "run_opt",
            "run",
            input_xyz,
            "--config",
            str(config_path),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.Popen(
            command,
            cwd=str(Path(get_app_base_dir())),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

    def _open_queue_dialog(self):
        dialog = QueueDialog(self, on_submit=self._submit_run)
        dialog.exec()

    def _open_runs_folder(self):
        url = QtCore.QUrl.fromLocalFile(str(self.runs_dir))
        QtGui.QDesktopServices.openUrl(url)

    def _open_run_dialog(self):
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open Run Directory", str(self.runs_dir)
        )
        if not selected_dir:
            return
        run_dir = Path(selected_dir)
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Open Run",
                f"No metadata.json found in {run_dir}",
            )
            return
        metadata = _read_json(metadata_path)
        if not metadata:
            QtWidgets.QMessageBox.warning(
                self,
                "Open Run",
                f"Unable to read metadata.json in {run_dir}",
            )
            return
        entry = _run_entry_from_metadata(metadata_path, metadata)
        self.manual_runs[entry.run_dir] = entry
        self.current_run = entry
        self._refresh_runs()
        self._select_run_by_dir(entry.run_dir)

    def _stop_run(self):
        if not self.current_run:
            return
        metadata = _read_json(Path(self.current_run.metadata_path))
        if not metadata:
            return
        pid = metadata.get("pid")
        if not pid:
            QtWidgets.QMessageBox.information(
                self, "Stop Run", "No PID recorded for this run."
            )
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Stop Run", f"Unable to stop process {pid}: {exc}"
            )

    def _select_run_by_dir(self, run_dir):
        for index in range(self.run_list.count()):
            item = self.run_list.item(index)
            entry = item.data(QtCore.Qt.UserRole)
            if entry and entry.run_dir == run_dir:
                item.setSelected(True)
                self.run_list.scrollToItem(item)
                return

    def _refresh_runs(self):
        selected = self.current_run.run_dir if self.current_run else None
        entries = _load_run_entries(self.runs_dir, limit=RUN_LIST_LIMIT)
        if self.manual_runs:
            known = {entry.run_dir for entry in entries}
            for run_dir, entry in self.manual_runs.items():
                if run_dir not in known:
                    entries.insert(0, entry)
                    known.add(run_dir)
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for entry in entries:
            label_parts = [
                entry.status,
                entry.calculation_mode or "unknown",
                entry.basis or "-",
                entry.xc or "-",
                Path(entry.run_dir).name,
            ]
            item = QtWidgets.QListWidgetItem(" | ".join(label_parts))
            item.setData(QtCore.Qt.UserRole, entry)
            self.run_list.addItem(item)
            if selected and entry.run_dir == selected:
                item.setSelected(True)
        self.run_list.blockSignals(False)
        if selected:
            self._select_run()

    def _refresh_disk_usage(self):
        if self._disk_usage_inflight:
            return
        self._disk_usage_inflight = True
        runs_dir = Path(self.runs_dir)

        def worker():
            usage = _calculate_disk_usage(runs_dir)
            self.disk_usage_ready.emit(usage)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_disk_usage(self, usage):
        self._disk_usage_inflight = False
        if not hasattr(self, "disk_usage_label"):
            return
        if not isinstance(usage, dict):
            self.disk_usage_label.setText("Disk: unavailable")
            self.disk_usage_label.setToolTip("")
            return
        runs_text = _format_bytes(usage.get("runs_bytes"))
        free_text = _format_bytes(usage.get("free_bytes"))
        total_text = _format_bytes(usage.get("total_bytes"))
        label = f"Runs: {runs_text} | Free: {free_text}"
        if usage.get("total_bytes") is not None:
            label += f" / {total_text}"
        self.disk_usage_label.setText(label)
        self.disk_usage_label.setToolTip(usage.get("error") or "")

    def _select_run(self):
        selected_items = self.run_list.selectedItems()
        if not selected_items:
            self.current_run = None
            return
        entry = selected_items[0].data(QtCore.Qt.UserRole)
        self.current_run = entry
        self._update_summary_view()
        self._update_log_view()
        self._update_results_view()

    def _update_summary_view(self):
        if not self.current_run:
            self.summary_view.setPlainText("")
            return
        metadata = _read_json(Path(self.current_run.metadata_path))
        if not metadata:
            self.summary_view.setPlainText("Unable to load metadata.")
            return
        summary_lines = [
            f"Run dir: {metadata.get('run_directory')}",
            f"Status: {metadata.get('status')}",
            f"Started: {metadata.get('run_started_at')}",
            f"Ended: {metadata.get('run_ended_at')}",
            f"Mode: {metadata.get('calculation_mode')}",
            f"Basis: {metadata.get('basis')}",
            f"XC: {metadata.get('xc')}",
            f"Solvent: {metadata.get('solvent')}",
            f"Solvent model: {metadata.get('solvent_model')}",
            f"Dispersion: {metadata.get('dispersion')}",
            "",
            "Metadata JSON:",
            json.dumps(metadata, indent=2, ensure_ascii=False),
        ]
        self.summary_view.setPlainText("\n".join(summary_lines))

    def _update_log_view(self):
        if not self.current_run:
            self.log_view.setPlainText("")
            return
        metadata = _read_json(Path(self.current_run.metadata_path))
        if not metadata:
            return
        log_path = metadata.get("log_file") or str(Path(self.current_run.run_dir) / "run.log")
        lines = _tail_lines(Path(log_path), LOG_TAIL_LINES)
        query = self.log_search.text().strip().lower()
        if query:
            lines = [line for line in lines if query in line.lower()]
        self.log_view.setPlainText("".join(lines))

    def _update_results_view(self):
        try:
            if not self.current_run:
                self.results_info.setText("No run selected.")
                self.chart_view.setChart(QChart())
                return
            run_dir = Path(self.current_run.run_dir)
            opt_log = run_dir / "ase_opt.log"
            irc_csv = run_dir / "irc_profile.csv"
            scan_csv = run_dir / "scan_result.csv"
            freq_json = run_dir / "frequency_result.json"
            sp_json = run_dir / "qcschema_result.json"

            if opt_log.exists():
                step_index = None
                energy_index = None
                x_values = []
                y_values = []
                line_series = QLineSeries()
                with opt_log.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if parts[0] == "Step" and "Energy" in parts:
                            step_index = parts.index("Step")
                            energy_index = parts.index("Energy")
                            continue
                        if parts[0].endswith(":"):
                            parts = parts[1:]
                        if step_index is not None and energy_index is not None:
                            if len(parts) <= max(step_index, energy_index):
                                continue
                            step_token = parts[step_index]
                            energy_token = parts[energy_index]
                        else:
                            if len(parts) < 3:
                                continue
                            step_token = parts[0]
                            energy_token = parts[2]
                        try:
                            step = float(step_token)
                            energy = float(energy_token)
                        except ValueError:
                            continue
                        x_values.append(step)
                        y_values.append(energy)
                        line_series.append(step, energy)
                if x_values:
                    chart = QChart()
                    chart.addSeries(line_series)
                    axis_x = QValueAxis()
                    axis_x.setTitleText("Step")
                    axis_y = QValueAxis()
                    axis_y.setTitleText("Energy (Hartree)")
                    x_min, x_max = _axis_range(x_values)
                    y_min, y_max = _axis_range(y_values)
                    axis_x.setRange(x_min, x_max)
                    axis_y.setRange(y_min, y_max)
                    chart.addAxis(axis_x, QtCore.Qt.AlignBottom)
                    chart.addAxis(axis_y, QtCore.Qt.AlignLeft)
                    line_series.attachAxis(axis_x)
                    line_series.attachAxis(axis_y)
                    self.results_info.setText("Optimization energy profile.")
                    self.chart_view.setChart(chart)
                    return

            if irc_csv.exists():
                series = {}
                x_values = []
                y_values = []
                with irc_csv.open("r", encoding="utf-8", errors="replace") as handle:
                    header = handle.readline().strip().split(",")
                    try:
                        direction_index = header.index("direction")
                        step_index = header.index("step")
                        energy_index = header.index("energy_ev")
                    except ValueError:
                        self.results_info.setText("IRC profile format not recognized.")
                        self.chart_view.setChart(QChart())
                        return
                    for line in handle:
                        parts = [item.strip() for item in line.split(",")]
                        if len(parts) <= energy_index:
                            continue
                        direction = parts[direction_index]
                        try:
                            step = float(parts[step_index])
                            energy = float(parts[energy_index])
                        except ValueError:
                            continue
                        x_values.append(step)
                        y_values.append(energy)
                        series.setdefault(direction, []).append((step, energy))
                chart = QChart()
                for direction, points in series.items():
                    line_series = QLineSeries()
                    line_series.setName(direction)
                    for step, energy in points:
                        line_series.append(step, energy)
                    chart.addSeries(line_series)
                axis_x = QValueAxis()
                axis_x.setTitleText("Step")
                axis_y = QValueAxis()
                axis_y.setTitleText("Energy (eV)")
                x_min, x_max = _axis_range(x_values)
                y_min, y_max = _axis_range(y_values)
                axis_x.setRange(x_min, x_max)
                axis_y.setRange(y_min, y_max)
                chart.addAxis(axis_x, QtCore.Qt.AlignBottom)
                chart.addAxis(axis_y, QtCore.Qt.AlignLeft)
                for s in chart.series():
                    s.attachAxis(axis_x)
                    s.attachAxis(axis_y)
                chart.legend().setVisible(True)
                self.results_info.setText("IRC energy profile.")
                self.chart_view.setChart(chart)
                return

            if scan_csv.exists():
                x_values = []
                y_values = []
                with scan_csv.open("r", encoding="utf-8", errors="replace") as handle:
                    header = handle.readline().strip().split(",")
                    if len(header) < 3:
                        self.results_info.setText("Scan result format not recognized.")
                        self.chart_view.setChart(QChart())
                        return
                    energy_index = header.index("energy") if "energy" in header else None
                    coord_index = None
                    for idx, name in enumerate(header):
                        if name.startswith("bond_") or name.startswith("angle_") or name.startswith("dihedral_"):
                            coord_index = idx
                            break
                    if energy_index is None or coord_index is None:
                        self.results_info.setText("Scan result format not recognized.")
                        self.chart_view.setChart(QChart())
                        return
                    line_series = QLineSeries()
                    for line in handle:
                        parts = [item.strip() for item in line.split(",")]
                        if len(parts) <= energy_index:
                            continue
                        try:
                            coord = float(parts[coord_index])
                            energy = float(parts[energy_index])
                        except ValueError:
                            continue
                        x_values.append(coord)
                        y_values.append(energy)
                        line_series.append(coord, energy)
                chart = QChart()
                chart.addSeries(line_series)
                axis_x = QValueAxis()
                axis_x.setTitleText(header[coord_index])
                axis_y = QValueAxis()
                axis_y.setTitleText("Energy (Hartree)")
                x_min, x_max = _axis_range(x_values)
                y_min, y_max = _axis_range(y_values)
                axis_x.setRange(x_min, x_max)
                axis_y.setRange(y_min, y_max)
                chart.addAxis(axis_x, QtCore.Qt.AlignBottom)
                chart.addAxis(axis_y, QtCore.Qt.AlignLeft)
                line_series.attachAxis(axis_x)
                line_series.attachAxis(axis_y)
                self.results_info.setText("Scan energy profile.")
                self.chart_view.setChart(chart)
                return

            if freq_json.exists():
                payload = _read_json(freq_json)
                if not payload:
                    self.results_info.setText("Frequency result unavailable.")
                    self.chart_view.setChart(QChart())
                    return
                results = _safe_results(payload)
                frequencies = results.get("frequencies_wavenumber") or []
                line_series = QLineSeries()
                x_values = []
                y_values = []
                for idx, value in enumerate(frequencies, start=1):
                    try:
                        freq = float(value)
                    except (TypeError, ValueError):
                        continue
                    x_values.append(float(idx))
                    y_values.append(freq)
                    line_series.append(idx, freq)
                chart = QChart()
                chart.addSeries(line_series)
                axis_x = QValueAxis()
                axis_x.setTitleText("Mode index")
                axis_y = QValueAxis()
                axis_y.setTitleText("Frequency (cm^-1)")
                x_min, x_max = _axis_range(x_values)
                y_min, y_max = _axis_range(y_values)
                axis_x.setRange(x_min, x_max)
                axis_y.setRange(y_min, y_max)
                chart.addAxis(axis_x, QtCore.Qt.AlignBottom)
                chart.addAxis(axis_y, QtCore.Qt.AlignLeft)
                line_series.attachAxis(axis_x)
                line_series.attachAxis(axis_y)
                self.results_info.setText("Frequency spectrum.")
                self.chart_view.setChart(chart)
                return

            if sp_json.exists():
                payload = _read_json(sp_json)
                energy = None
                if payload:
                    energy = payload.get("return_result")
                    if energy is None:
                        energy = payload.get("properties", {}).get("return_energy")
                if energy is not None:
                    self.results_info.setText(f"Single-point energy: {energy:.6f} Hartree")
                    scatter = QScatterSeries()
                    scatter.append(1.0, float(energy))
                    chart = QChart()
                    chart.addSeries(scatter)
                    axis_x = QValueAxis()
                    axis_x.setTitleText("Point")
                    axis_y = QValueAxis()
                    axis_y.setTitleText("Energy (Hartree)")
                    x_min, x_max = _axis_range([1.0])
                    y_min, y_max = _axis_range([float(energy)])
                    axis_x.setRange(x_min, x_max)
                    axis_y.setRange(y_min, y_max)
                    chart.addAxis(axis_x, QtCore.Qt.AlignBottom)
                    chart.addAxis(axis_y, QtCore.Qt.AlignLeft)
                    scatter.attachAxis(axis_x)
                    scatter.attachAxis(axis_y)
                    self.chart_view.setChart(chart)
                else:
                    self.results_info.setText("Single-point result available.")
                    self.chart_view.setChart(QChart())
                return

            self.results_info.setText("No results available.")
            self.chart_view.setChart(QChart())
        except Exception as exc:
            self._log_gui_error("results_view", exc)
            self.results_info.setText(
                "Error loading results (see ~/DFTFlow/gui_errors.log)."
            )
            self.chart_view.setChart(QChart())


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = DFTFlowWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
