from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    from .merge_core import merge_csv_files
except ImportError:
    from merge_core import merge_csv_files  # type: ignore


class MergerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Merger")
        self.root.geometry("980x620")

        self.base_dir = Path(__file__).resolve().parent
        self.recorder_dir = self.base_dir.parent / "record_refiner" / "results"
        self.aruco_dir = self.base_dir.parent / "aruco_refiner" / "results"
        self.output_dir = self.base_dir / "results"
        self.recorder_dir.mkdir(parents=True, exist_ok=True)
        self.aruco_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.recorder_path_var = tk.StringVar(value="")
        self.aruco_path_var = tk.StringVar(value="")
        self.output_path_var = tk.StringVar(value=str((self.output_dir / "merged.csv").resolve()))
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        top = tk.LabelFrame(self.root, text="CSV Selection")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        top.columnconfigure(1, weight=1)

        tk.Label(top, text="Refined Recorder CSV").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(top, textvariable=self.recorder_path_var).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        tk.Button(top, text="Browse", command=self.browse_recorder).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(top, text="Refined Aruco CSV").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(top, textvariable=self.aruco_path_var).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        tk.Button(top, text="Browse", command=self.browse_aruco).grid(row=1, column=2, padx=5, pady=5)

        tk.Label(top, text="Output CSV").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(top, textvariable=self.output_path_var).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        tk.Button(top, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=5, pady=5)

        action = tk.Frame(self.root)
        action.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        tk.Button(action, text="Merge", command=self.merge).pack(side="left", padx=(0, 8))
        tk.Label(action, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)

        info = tk.LabelFrame(self.root, text="Rule")
        info.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        tk.Label(
            info,
            justify="left",
            anchor="w",
            text=(
                f"Refined recorder browser starts at {self.recorder_dir}\n"
                f"Aruco browser starts at {self.aruco_dir}\n"
                "Refined recorder rows are the base timeline.\n"
                "Model-predicted marker columns from record_refiner are preserved as-is.\n"
                "Each refined recorder timestamp is matched to the nearest aruco timestamp\n"
                "within +/- 0.5 frame inferred from aruco CSV spacing."
            ),
        ).pack(fill="x", padx=8, pady=8)

        log_frame = tk.LabelFrame(self.root, text="Log")
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set, state="disabled")

    def browse_recorder(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Refined Recorder CSV",
            initialdir=str(self.recorder_dir),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.recorder_path_var.set(path)
            self._sync_output_name()

    def browse_aruco(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Refined Aruco CSV",
            initialdir=str(self.aruco_dir),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.aruco_path_var.set(path)
            self._sync_output_name()

    def browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select Output CSV",
            initialdir=str(self.output_dir),
            initialfile=Path(self.output_path_var.get()).name or "merged.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.output_path_var.set(path)

    def _sync_output_name(self) -> None:
        recorder_name = Path(self.recorder_path_var.get()).stem if self.recorder_path_var.get().strip() else "recorder"
        aruco_name = Path(self.aruco_path_var.get()).stem if self.aruco_path_var.get().strip() else "aruco"
        self.output_path_var.set(str((self.output_dir / f"{recorder_name}__{aruco_name}__merged.csv").resolve()))

    def merge(self) -> None:
        recorder_path = self.recorder_path_var.get().strip()
        aruco_path = self.aruco_path_var.get().strip()
        output_path = self.output_path_var.get().strip()
        if not recorder_path or not aruco_path or not output_path:
            messagebox.showerror("Merge", "Refined recorder CSV, aruco CSV, and output CSV are required.")
            return
        try:
            result = merge_csv_files(Path(recorder_path), Path(aruco_path), Path(output_path))
        except Exception as exc:
            messagebox.showerror("Merge", str(exc))
            return
        self.status_var.set("Merge complete")
        self._log(f"Merged -> {result.output_path}")
        self._log(f"Recorder rows: {result.recorder_rows}")
        self._log(f"Matched rows: {result.matched_rows}")
        self._log(f"Unmatched rows: {result.unmatched_rows}")
        self._log(f"Tolerance: +/- {result.tolerance_sec:.6f} sec")

    def _log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(line) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def main() -> None:
    parser = argparse.ArgumentParser(description="error_collector Merger")
    parser.parse_args()
    root = tk.Tk()
    app = MergerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
