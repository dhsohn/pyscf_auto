import json
import os
import signal
import subprocess
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

from run_opt_paths import get_app_base_dir, get_runs_base_dir


REFRESH_INTERVAL_MS = 2000
LOG_TAIL_LINES = 2000


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


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque(handle, maxlen=max_lines))
    except OSError:
        return []


def _iter_metadata_files(base_dir: Path, max_depth: int = 2):
    base_depth = len(base_dir.parts)
    for root, dirs, files in os.walk(base_dir):
        depth = len(Path(root).parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if "metadata.json" in files:
            yield Path(root) / "metadata.json"


def _load_run_entries(base_dir: Path) -> list[RunEntry]:
    entries: list[RunEntry] = []
    for metadata_path in _iter_metadata_files(base_dir):
        metadata = _read_json(metadata_path)
        if not metadata:
            continue
        run_dir = metadata.get("run_directory") or str(metadata_path.parent)
        entries.append(
            RunEntry(
                run_dir=run_dir,
                metadata_path=str(metadata_path),
                status=metadata.get("status", "unknown"),
                started_at=metadata.get("run_started_at"),
                calculation_mode=metadata.get("calculation_mode"),
                basis=metadata.get("basis"),
                xc=metadata.get("xc"),
            )
        )
    entries.sort(
        key=lambda item: os.path.getmtime(item.metadata_path)
        if os.path.exists(item.metadata_path)
        else 0,
        reverse=True,
    )
    return entries


class RunSubmitDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Run")
        self.setModal(True)

        self.xyz_path = QtWidgets.QLineEdit()
        self.config_path = QtWidgets.QLineEdit()
        self.solvent_map_path = QtWidgets.QLineEdit()

        xyz_button = QtWidgets.QPushButton("Browse...")
        config_button = QtWidgets.QPushButton("Browse...")
        solvent_button = QtWidgets.QPushButton("Browse...")

        xyz_button.clicked.connect(lambda: self._pick_file(self.xyz_path))
        config_button.clicked.connect(lambda: self._pick_file(self.config_path))
        solvent_button.clicked.connect(lambda: self._pick_file(self.solvent_map_path))

        form = QtWidgets.QFormLayout()
        form.addRow("XYZ file", self._wrap_picker(self.xyz_path, xyz_button))
        form.addRow("Config file", self._wrap_picker(self.config_path, config_button))
        form.addRow("Solvent map (optional)", self._wrap_picker(self.solvent_map_path, solvent_button))

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

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

    def get_values(self):
        return {
            "xyz": self.xyz_path.text().strip() or None,
            "config": self.config_path.text().strip() or None,
            "solvent_map": self.solvent_map_path.text().strip() or None,
        }


class DFTFlowWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DFTFlow")
        self.resize(1200, 800)

        self.runs_dir = _ensure_runs_dir()
        self.current_run: RunEntry | None = None
        self.current_log_lines: list[str] = []
        self.current_query = ""

        self._build_ui()
        self._refresh_runs()

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_runs)
        self.refresh_timer.start(REFRESH_INTERVAL_MS)

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

        new_run_action = QtGui.QAction("New Run", self)
        new_run_action.triggered.connect(self._new_run)
        toolbar.addAction(new_run_action)

        stop_action = QtGui.QAction("Stop Run", self)
        stop_action.triggered.connect(self._stop_run)
        toolbar.addAction(stop_action)

        open_runs_action = QtGui.QAction("Open Runs Folder", self)
        open_runs_action.triggered.connect(self._open_runs_folder)
        toolbar.addAction(open_runs_action)

        splitter = QtWidgets.QSplitter()

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.addWidget(QtWidgets.QLabel("Runs"))
        self.run_list = QtWidgets.QListWidget()
        self.run_list.itemSelectionChanged.connect(self._select_run)
        left_layout.addWidget(self.run_list)
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

    def _new_run(self):
        dialog = RunSubmitDialog(self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        values = dialog.get_values()
        if not values["xyz"] or not values["config"]:
            QtWidgets.QMessageBox.warning(
                self, "Missing input", "XYZ and config files are required."
            )
            return
        command = [
            sys.executable,
            "-m",
            "run_opt",
            "run",
            values["xyz"],
            "--config",
            values["config"],
        ]
        if values["solvent_map"]:
            command.extend(["--solvent-map", values["solvent_map"]])
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

    def _open_runs_folder(self):
        url = QtCore.QUrl.fromLocalFile(str(self.runs_dir))
        QtGui.QDesktopServices.openUrl(url)

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

    def _refresh_runs(self):
        selected = self.current_run.run_dir if self.current_run else None
        entries = _load_run_entries(self.runs_dir)
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
            irc_csv = run_dir / "irc_profile.csv"
            scan_csv = run_dir / "scan_result.csv"
            freq_json = run_dir / "frequency_result.json"
            sp_json = run_dir / "qcschema_result.json"

            if irc_csv.exists():
                series = {}
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
                        line_series.append(coord, energy)
                chart = QChart()
                chart.addSeries(line_series)
                axis_x = QValueAxis()
                axis_x.setTitleText(header[coord_index])
                axis_y = QValueAxis()
                axis_y.setTitleText("Energy (Hartree)")
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
                frequencies = payload.get("results", {}).get("frequencies_wavenumber") or []
                line_series = QLineSeries()
                for idx, value in enumerate(frequencies, start=1):
                    try:
                        line_series.append(idx, float(value))
                    except (TypeError, ValueError):
                        continue
                chart = QChart()
                chart.addSeries(line_series)
                axis_x = QValueAxis()
                axis_x.setTitleText("Mode index")
                axis_y = QValueAxis()
                axis_y.setTitleText("Frequency (cm^-1)")
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
