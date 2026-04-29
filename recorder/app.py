from __future__ import annotations

import argparse
import queue
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    from .hardware import ImuHardware, RobotHardware, default_config_path, list_serial_ports
    from .recorder import SessionRecorder
    from .scenario import ScenarioError, ScenarioRunner, load_scenario_file
    from .sync_marker import SyncMarkerWindow, SyncState
except ImportError:
    from hardware import ImuHardware, RobotHardware, default_config_path, list_serial_ports  # type: ignore
    from recorder import SessionRecorder  # type: ignore
    from scenario import ScenarioError, ScenarioRunner, load_scenario_file  # type: ignore
    from sync_marker import SyncMarkerWindow, SyncState  # type: ignore


class RecordApp:
    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.root.title("Record")
        self.root.geometry("980x700")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.robot = RobotHardware(self._queue_log, config_path=config_path)
        self.imu = ImuHardware(self._queue_log)
        self.recorder = SessionRecorder(sample_hz=30.0)
        self.sync_window = None
        self.scenario_runner = ScenarioRunner(
            on_move=self._scenario_move,
            on_record=self._scenario_record,
            on_export=self._scenario_export,
            on_log=self._queue_log,
        )
        self.loaded_scenario_path: Path | None = None
        self.last_snapshot = None

        self.robot_port_var = tk.StringVar(value="")
        self.imu_port_var = tk.StringVar(value="")
        self.imu_baud_var = tk.StringVar(value="115200")
        self.roll_var = tk.IntVar(value=180)
        self.seg1_var = tk.IntVar(value=180)
        self.seg2_var = tk.IntVar(value=180)
        default_records_dir = Path.cwd() / "records"
        default_records_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path_var = tk.StringVar(value=str((default_records_dir / "record_session.csv").resolve()))
        self.status_var = tk.StringVar(value="Ready")
        self.record_state_var = tk.StringVar(value="idle")
        self.connection_state_var = tk.StringVar(value="Robot: disconnected | IMU: disconnected")
        self.load_state_var = tk.StringVar(value="Load1=n/a mA | Load2=n/a mA | Samples: 0")
        self.imu_value_var = tk.StringVar(value="IMU1/2: no samples")
        self.scenario_var = tk.StringVar(value="No scenario loaded")
        self.sample_count_var = tk.StringVar(value="0")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._init_sync_window)
        self._poll()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        conn = tk.LabelFrame(self.root, text="Connections")
        conn.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        for col in range(6):
            conn.columnconfigure(col, weight=1)

        tk.Label(conn, text="Robot Port").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        tk.Entry(conn, textvariable=self.robot_port_var).grid(row=0, column=1, sticky="ew", padx=5, pady=4)
        tk.Button(conn, text="Connect Robot", command=self.connect_robot).grid(row=0, column=2, sticky="ew", padx=5, pady=4)
        tk.Button(conn, text="Disconnect Robot", command=self.disconnect_robot).grid(row=0, column=3, sticky="ew", padx=5, pady=4)
        tk.Button(conn, text="Refresh Ports", command=self.refresh_ports).grid(row=0, column=4, sticky="ew", padx=5, pady=4)

        tk.Label(conn, text="IMU Port").grid(row=1, column=0, sticky="w", padx=5, pady=4)
        tk.Entry(conn, textvariable=self.imu_port_var).grid(row=1, column=1, sticky="ew", padx=5, pady=4)
        tk.Label(conn, text="Baud").grid(row=1, column=2, sticky="e", padx=5, pady=4)
        tk.Entry(conn, textvariable=self.imu_baud_var).grid(row=1, column=3, sticky="ew", padx=5, pady=4)
        tk.Button(conn, text="Connect IMU", command=self.connect_imu).grid(row=1, column=4, sticky="ew", padx=5, pady=4)
        tk.Button(conn, text="Disconnect IMU", command=self.disconnect_imu).grid(row=1, column=5, sticky="ew", padx=5, pady=4)
        port_list_frame = tk.Frame(conn)
        port_list_frame.grid(row=2, column=0, columnspan=6, sticky="ew", padx=5, pady=(0, 4))
        port_list_frame.columnconfigure(0, weight=1)
        self.port_list = tk.Listbox(port_list_frame, height=3)
        self.port_list.grid(row=0, column=0, sticky="ew")
        port_scroll = tk.Scrollbar(port_list_frame, orient="vertical", command=self.port_list.yview)
        port_scroll.grid(row=0, column=1, sticky="ns")
        self.port_list.configure(yscrollcommand=port_scroll.set)
        self.port_list.bind("<<ListboxSelect>>", self._on_select_port)

        status = tk.LabelFrame(self.root, text="Status")
        status.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        status.columnconfigure(0, weight=1)
        tk.Label(status, textvariable=self.connection_state_var).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Label(status, textvariable=self.load_state_var).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Label(status, textvariable=self.imu_value_var).grid(row=2, column=0, sticky="w", padx=5, pady=2)
        tk.Label(status, textvariable=self.record_state_var).grid(row=3, column=0, sticky="w", padx=5, pady=2)

        control = tk.LabelFrame(self.root, text="Manual Control")
        control.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        control.columnconfigure(1, weight=1)
        self._add_slider(control, 0, "Roll", self.roll_var)
        self._add_slider(control, 1, "Seg1", self.seg1_var)
        self._add_slider(control, 2, "Seg2", self.seg2_var)

        buttons = tk.Frame(control)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
        for col in range(6):
            buttons.columnconfigure(col, weight=1)
        tk.Button(buttons, text="Send Pose", command=self.send_pose).grid(row=0, column=0, sticky="ew", padx=3)
        tk.Button(buttons, text="Record Start", command=self.start_record).grid(row=0, column=1, sticky="ew", padx=3)
        tk.Button(buttons, text="Record Stop", command=self.stop_record).grid(row=0, column=2, sticky="ew", padx=3)
        tk.Button(buttons, text="Record Reset", command=self.reset_record).grid(row=0, column=3, sticky="ew", padx=3)
        tk.Button(buttons, text="Export CSV", command=self.export_csv).grid(row=0, column=4, sticky="ew", padx=3)
        tk.Entry(buttons, textvariable=self.csv_path_var).grid(row=1, column=0, columnspan=6, sticky="ew", padx=3, pady=(4, 0))

        scenario = tk.LabelFrame(self.root, text="Scenario")
        scenario.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        scenario.columnconfigure(0, weight=1)
        scenario.rowconfigure(1, weight=1)
        tk.Label(scenario, textvariable=self.scenario_var).grid(row=0, column=0, sticky="w", padx=5, pady=4)
        scenario_buttons = tk.Frame(scenario)
        scenario_buttons.grid(row=0, column=1, sticky="e", padx=5, pady=4)
        tk.Button(scenario_buttons, text="Scenario Load", command=self.load_scenario).grid(row=0, column=0, padx=3)
        tk.Button(scenario_buttons, text="Scenario Start", command=self.start_scenario).grid(row=0, column=1, padx=3)
        tk.Button(scenario_buttons, text="Scenario Stop", command=self.stop_scenario).grid(row=0, column=2, padx=3)

        log_frame = tk.Frame(scenario)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=4)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap="word", height=16)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set, state="disabled")

        footer = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        footer.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.refresh_ports()

    def _add_slider(self, parent: tk.Widget, row: int, label: str, variable: tk.IntVar) -> None:
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=4)
        scale = tk.Scale(
            parent,
            from_=0,
            to=360,
            orient="horizontal",
            resolution=1,
            variable=variable,
            showvalue=True,
            length=620,
        )
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=4)

    def connect_robot(self) -> None:
        try:
            self.robot.connect(self.robot_port_var.get())
            pose = self.robot.read_pose()
            self.roll_var.set(int(pose.roll))
            self.seg1_var.set(int(pose.seg1))
            self.seg2_var.set(int(pose.seg2))
            self.status_var.set("Robot connected")
        except Exception as exc:
            messagebox.showerror("Robot", str(exc))

    def disconnect_robot(self) -> None:
        self.robot.disconnect()
        self.status_var.set("Robot disconnected")

    def connect_imu(self) -> None:
        try:
            self.imu.connect(self.imu_port_var.get(), int(self.imu_baud_var.get()))
            self.status_var.set("IMU connected")
        except Exception as exc:
            messagebox.showerror("IMU", str(exc))

    def disconnect_imu(self) -> None:
        self.imu.disconnect()
        self.status_var.set("IMU disconnected")

    def refresh_ports(self) -> None:
        ports = list_serial_ports()
        self.port_list.delete(0, "end")
        if not ports:
            self.port_list.insert("end", "No serial ports found")
            return
        for port in ports:
            self.port_list.insert("end", port)
        if not self.robot_port_var.get().strip():
            self.robot_port_var.set(ports[0])
        if not self.imu_port_var.get().strip():
            fallback = ports[1] if len(ports) > 1 else ports[0]
            self.imu_port_var.set(fallback)

    def _on_select_port(self, _event=None) -> None:
        selection = self.port_list.curselection()
        if not selection:
            return
        value = str(self.port_list.get(selection[0])).strip()
        if not value or value == "No serial ports found":
            return
        current_robot = self.robot_port_var.get().strip()
        current_imu = self.imu_port_var.get().strip()
        if not current_robot or current_robot == value:
            self.robot_port_var.set(value)
            return
        if not current_imu or current_imu == value:
            self.imu_port_var.set(value)
            return
        self.imu_port_var.set(value)

    def send_pose(self) -> None:
        try:
            self.robot.send_pose(self.roll_var.get(), self.seg1_var.get(), self.seg2_var.get())
            self.status_var.set("Pose sent")
        except Exception as exc:
            messagebox.showerror("Move", str(exc))

    def start_record(self) -> None:
        self.recorder.start()
        if self.sync_window is not None:
            self.sync_window.set_state(SyncState.RECORDING)
        self.record_state_var.set("recording: active")
        self.status_var.set("Recording started")

    def stop_record(self) -> None:
        self.recorder.stop()
        if self.sync_window is not None:
            self.sync_window.set_state(SyncState.DONE)
        self.record_state_var.set("recording: stopped")
        self.status_var.set("Recording stopped")

    def reset_record(self) -> None:
        self.recorder.reset()
        if self.sync_window is not None:
            self.sync_window.set_state(SyncState.BEFORE)
        self.record_state_var.set("recording: reset")
        self.sample_count_var.set("samples: 0")
        self.status_var.set("Recording reset")
        self._queue_log("Recording buffer reset")

    def export_csv(self, path: str | None = None) -> None:
        target = path or self.csv_path_var.get().strip()
        if not target:
            messagebox.showerror("Export", "CSV path is empty.")
            return
        try:
            exported = self.recorder.export_csv(target)
            self.csv_path_var.set(str(exported))
            self.status_var.set(f"Exported {len(self.recorder.samples)} samples")
            self._queue_log(f"CSV exported -> {exported}")
        except Exception as exc:
            messagebox.showerror("Export", str(exc))

    def load_scenario(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Scenario",
            filetypes=[("Scenario files", "*.txt *.scenario"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            commands = load_scenario_file(path)
            self.scenario_runner.load(commands)
            self.loaded_scenario_path = Path(path)
            self.scenario_var.set(str(self.loaded_scenario_path))
            self.status_var.set("Scenario loaded")
        except Exception as exc:
            messagebox.showerror("Scenario", str(exc))

    def start_scenario(self) -> None:
        try:
            self.scenario_runner.start()
        except Exception as exc:
            messagebox.showerror("Scenario", str(exc))

    def stop_scenario(self) -> None:
        self.scenario_runner.stop()
        self.status_var.set("Scenario stopped")

    def _scenario_move(self, roll: int, seg1: int, seg2: int) -> None:
        self.roll_var.set(int(roll))
        self.seg1_var.set(int(seg1))
        self.seg2_var.set(int(seg2))
        self.robot.send_pose(int(roll), int(seg1), int(seg2))

    def _scenario_record(self, mode: str) -> None:
        if mode == "start":
            self.start_record()
            return
        if mode == "end":
            self.stop_record()
            return
        if mode == "reset":
            self.reset_record()
            return
        raise ScenarioError(f"unsupported record mode: {mode}")

    def _scenario_export(self, path: str) -> None:
        self.export_csv(str(self._resolve_scenario_export_path(path)))

    def _resolve_scenario_export_path(self, path: str) -> Path:
        scenario_path = self.loaded_scenario_path.resolve() if self.loaded_scenario_path is not None else None
        scenario_dir = scenario_path.parent if scenario_path is not None else Path.cwd()
        records_dir = (Path.cwd() / "records").resolve()
        template = path.strip()
        variables = {
            "cwd": str(Path.cwd().resolve()),
            "records_dir": str(records_dir),
            "scenario_dir": str(scenario_dir),
            "scenario_path": "" if scenario_path is None else str(scenario_path),
            "csv_path": self.csv_path_var.get().strip(),
            "csv_path_var": self.csv_path_var.get().strip(),
        }

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in variables:
                raise ScenarioError(f"unknown export path variable: {name}")
            value = variables[name].strip()
            if not value:
                raise ScenarioError(f"export path variable is empty: {name}")
            return value

        resolved = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, template)
        target = Path(resolved).expanduser()
        if not target.is_absolute():
            target = scenario_dir / target
        return target.resolve()

    def _poll(self) -> None:
        self._drain_logs()
        try:
            self.last_snapshot = self.robot.snapshot(self.imu)
            if self.recorder.capture_if_due(self.last_snapshot):
                self.sample_count_var.set(str(len(self.recorder.samples)))
            self._refresh_status()
        except Exception as exc:
            self.status_var.set(str(exc))
        try:
            self.scenario_runner.tick()
        except Exception as exc:
            self.scenario_runner.stop()
            messagebox.showerror("Scenario", str(exc))
        self.root.after(1, self._poll)

    def _refresh_status(self) -> None:
        self.connection_state_var.set(
            f"Robot: {'connected' if self.robot.connected else 'disconnected'}"
            f" | IMU: {'connected' if self.imu.connected else 'disconnected'}"
        )
        if self.last_snapshot is None:
            return
        snap = self.last_snapshot
        self.load_state_var.set(
            "Load1=%s mA | Load2=%s mA | Samples: %s"
            % (
                "n/a" if snap.load1_ma is None else f"{snap.load1_ma:.1f}",
                "n/a" if snap.load2_ma is None else f"{snap.load2_ma:.1f}",
                self.sample_count_var.get(),
            )
        )
        def _fmt_opt(value: float | None, digits: int = 3) -> str:
            if value is None:
                return "n/a"
            return f"{float(value):.{digits}f}"

        imu1 = snap.imu1
        imu2 = snap.imu2
        self.imu_value_var.set(
            "IMU1 t=%s Q=(%s,%s,%s,%s) G=(%s,%s,%s) W=(%s,%s,%s) | "
            "IMU2 t=%s Q=(%s,%s,%s,%s) G=(%s,%s,%s) W=(%s,%s,%s)"
            % (
                _fmt_opt(imu1.timestamp, 3),
                _fmt_opt(imu1.qw, 3),
                _fmt_opt(imu1.qx, 3),
                _fmt_opt(imu1.qy, 3),
                _fmt_opt(imu1.qz, 3),
                _fmt_opt(imu1.gvx, 3),
                _fmt_opt(imu1.gvy, 3),
                _fmt_opt(imu1.gvz, 3),
                _fmt_opt(imu1.gx, 3),
                _fmt_opt(imu1.gy, 3),
                _fmt_opt(imu1.gz, 3),
                _fmt_opt(imu2.timestamp, 3),
                _fmt_opt(imu2.qw, 3),
                _fmt_opt(imu2.qx, 3),
                _fmt_opt(imu2.qy, 3),
                _fmt_opt(imu2.qz, 3),
                _fmt_opt(imu2.gvx, 3),
                _fmt_opt(imu2.gvy, 3),
                _fmt_opt(imu2.gvz, 3),
                _fmt_opt(imu2.gx, 3),
                _fmt_opt(imu2.gy, 3),
                _fmt_opt(imu2.gz, 3),
            )
        )
        if (
            self.sync_window is not None
            and (not self.recorder.recording_active)
            and self.sync_window.state == SyncState.RECORDING
        ):
            self.sync_window.set_state(SyncState.BEFORE)
        if self.recorder.recording_active:
            self.record_state_var.set("recording: active")
        else:
            self.record_state_var.set("recording: idle")

    def _drain_logs(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

    def _queue_log(self, line: str) -> None:
        self.log_queue.put(str(line))

    def _init_sync_window(self) -> None:
        try:
            self.sync_window = SyncMarkerWindow(self.root)
        except Exception:
            self.sync_window = None

    def _on_close(self) -> None:
        self.scenario_runner.stop()
        self.robot.disconnect()
        self.imu.disconnect()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="error_collector record app")
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()
    root = tk.Tk()
    app = RecordApp(root, config_path=args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
