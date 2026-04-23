from __future__ import annotations

import argparse
import base64
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2

try:
    from .detector import ArucoDetector
    from .sync import SyncStateMachine
    from .writer import ArucoCsvWriter
except ImportError:
    from detector import ArucoDetector  # type: ignore
    from sync import SyncStateMachine  # type: ignore
    from writer import ArucoCsvWriter  # type: ignore


class ArucoReaderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Aruco Reader")
        self.root.geometry("1280x820")
        self.base_dir = Path(__file__).resolve().parent
        self.records_dir = self.base_dir / "records"
        self.videos_dir = self.base_dir / "videos"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        self.video_path_var = tk.StringVar(value="No video loaded")
        self.status_var = tk.StringVar(value="Ready")
        self.export_path_var = tk.StringVar(value=str((self.records_dir / "aruco_detection.csv").resolve()))
        self.playing = False
        self.detecting = False
        self.capture: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame_index = 0
        self.current_frame_bgr = None
        self.current_image = None
        self.detector = ArucoDetector()
        self.sync_machine = SyncStateMachine()
        self.writer = ArucoCsvWriter()
        self._detection_start_t: float | None = None
        self._last_log_line: str | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        top = tk.Frame(self.root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        for idx in range(8):
            top.columnconfigure(idx, weight=1 if idx in (1, 7) else 0)

        tk.Button(top, text="Load", command=self.load_video).grid(row=0, column=0, padx=4)
        tk.Label(top, textvariable=self.video_path_var, anchor="w").grid(row=0, column=1, columnspan=3, sticky="ew", padx=4)
        tk.Button(top, text="Play/Pause", command=self.toggle_play).grid(row=0, column=4, padx=4)
        tk.Button(top, text="Detect", command=self.start_detection).grid(row=0, column=5, padx=4)
        tk.Button(top, text="Export", command=self.export_csv).grid(row=0, column=6, padx=4)
        tk.Label(top, textvariable=self.status_var, anchor="e").grid(row=0, column=7, sticky="e", padx=4)
        tk.Label(top, text="Export Path").grid(row=1, column=0, sticky="w", padx=4, pady=(6, 0))
        tk.Entry(top, textvariable=self.export_path_var).grid(row=1, column=1, columnspan=6, sticky="ew", padx=4, pady=(6, 0))
        tk.Button(top, text="Browse", command=self.browse_export_path).grid(row=1, column=7, padx=4, pady=(6, 0))

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))

        right = tk.LabelFrame(self.root, text="Log")
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = tk.Scale(
            right,
            from_=0,
            to=1,
            orient="horizontal",
            resolution=1,
            variable=self.seek_var,
            command=self.on_seek,
            length=420,
        )
        self.seek_scale.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        log_frame = tk.Frame(right)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=24)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set, state="disabled")

    def load_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Video",
            initialdir=str(self.videos_dir),
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self._close_capture()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Video", "Failed to open video.")
            return
        self.capture = cap
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.current_frame_index = 0
        self.current_frame_bgr = None
        self.video_path_var.set(path)
        self.seek_scale.configure(to=max(0, self.frame_count - 1))
        self._log(f"Loaded the video from {path}")
        self._seek_and_show_frame(0)

    def toggle_play(self) -> None:
        if self.capture is None:
            return
        self.playing = not self.playing
        self.status_var.set("Playing" if self.playing else "Paused")
        if self.playing:
            self._play_loop()

    def start_detection(self) -> None:
        if self.capture is None:
            return
        if self.current_frame_bgr is None:
            if not self._seek_and_show_frame(self.current_frame_index):
                return
        self.detecting = True
        self.playing = True
        self.sync_machine = SyncStateMachine()
        self._detection_start_t = None
        self.writer.reset()
        self._log("Detection started")
        self._process_current_frame()
        if not self.playing:
            return
        self._play_loop()

    def export_csv(self) -> None:
        if not self.writer.rows:
            messagebox.showerror("Export", "No detection rows to export.")
            return
        path = self.export_path_var.get().strip()
        if not path:
            messagebox.showerror("Export", "Export path is empty.")
            return
        exported = self.writer.export_csv(Path(path))
        self.export_path_var.set(str(exported))
        self._log(f"CSV exported to {exported}")
        self.status_var.set("CSV exported")

    def browse_export_path(self) -> None:
        source_name = Path(self.video_path_var.get()).stem if self.video_path_var.get() != "No video loaded" else "aruco_detection"
        path = filedialog.asksaveasfilename(
            title="Export CSV",
            initialdir=str(self.records_dir),
            initialfile=f"{source_name}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.export_path_var.set(path)

    def _play_loop(self) -> None:
        if not self.playing or self.capture is None:
            return
        if not self._read_next_and_show_frame():
            self.playing = False
            self.detecting = False
            self.status_var.set("Playback complete")
            return
        delay_ms = max(1, int(round(1000.0 / max(1.0, self.fps))))
        self.root.after(delay_ms, self._play_loop)

    def _seek_and_show_frame(self, frame_index: int) -> bool:
        if self.capture is None:
            return False
        frame_index = max(0, min(frame_index, max(0, self.frame_count - 1)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False
        self.current_frame_index = int(frame_index)
        self.current_frame_bgr = frame.copy()
        self.seek_var.set(float(frame_index))
        display = frame
        if self.detecting:
            display = self._process_current_frame()
        else:
            self._show_frame(display)
        return True

    def _read_next_and_show_frame(self) -> bool:
        if self.capture is None:
            return False
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False
        self.current_frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        self.current_frame_bgr = frame.copy()
        self.seek_var.set(float(self.current_frame_index))
        if self.detecting:
            self._process_current_frame()
        else:
            self._show_frame(frame)
        return True

    def _process_current_frame(self):
        if self.current_frame_bgr is None:
            return None
        result = self.detector.process_frame(self.current_frame_bgr)
        display = result.display_frame
        if self.detecting:
            for line in self.sync_machine.update(result.sync_ids):
                self._log(line)
            if self.sync_machine.csv_started and self._detection_start_t is None:
                self._detection_start_t = float(self.current_frame_index) / max(1.0, self.fps)
            if self.sync_machine.phase.value == "recording" and self._detection_start_t is not None:
                t_sec = (float(self.current_frame_index) / max(1.0, self.fps)) - self._detection_start_t
                self.writer.write_row(t_sec, result.relative_poses)
            if self.sync_machine.csv_completed:
                self.detecting = False
                self.playing = False
        self._show_frame(display)
        return display

    def _show_frame(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        max_w, max_h = 760, 760
        h, w = frame_rgb.shape[:2]
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale < 1.0:
            frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", frame_rgb)
        if not ok:
            return
        image = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"), format="png")
        self.current_image = image
        self.video_label.configure(image=image)

    def on_seek(self, value: str) -> None:
        if self.capture is None:
            return
        if self.playing:
            return
        try:
            frame_index = int(float(value))
        except Exception:
            return
        self._seek_and_show_frame(frame_index)

    def _log(self, line: str) -> None:
        text = str(line)
        if self._last_log_line == text:
            return
        self._last_log_line = text
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _close_capture(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        self.capture = None
        self.current_frame_bgr = None

    def _on_close(self) -> None:
        self._close_capture()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="error_collector Aruco Reader")
    parser.parse_args()
    root = tk.Tk()
    app = ArucoReaderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
