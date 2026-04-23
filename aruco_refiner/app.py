from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]

try:
    from .refine_core import RefineConfig, refine_csv_file
except ImportError:
    from refine_core import RefineConfig, refine_csv_file  # type: ignore


class RefinerApp:
    def __init__(self, root: Any) -> None:
        if tk is None or filedialog is None or messagebox is None:
            raise RuntimeError("tkinter is not available. Install Python tkinter support to use the GUI.")
        self.root = root
        self.root.title("Aruco Refiner")
        self.root.geometry("820x560")

        self.base_dir = Path(__file__).resolve().parent
        self.input_dir = self.base_dir.parent / "aruco_reader" / "records"
        self.results_dir = self.base_dir / "results"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.input_path_var = tk.StringVar(value="")
        self.output_path_var = tk.StringVar(value=str((self.results_dir / "refined.csv").resolve()))
        self.max_angle_var = tk.StringVar(value="75")
        self.max_position_step_var = tk.StringVar(value="0.10")
        self.reset_gap_var = tk.StringVar(value="5")
        self.interpolate_gap_var = tk.StringVar(value="2")
        self.vote_min_support_var = tk.StringVar(value="2")
        self.vote_delta_step_var = tk.StringVar(value="0.04")
        self.vote_delta_angle_var = tk.StringVar(value="30")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        files = tk.LabelFrame(self.root, text="CSV Files")
        files.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        files.columnconfigure(1, weight=1)

        tk.Label(files, text="Input CSV").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(files, textvariable=self.input_path_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(files, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=6, pady=6)

        tk.Label(files, text="Output CSV").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(files, textvariable=self.output_path_var).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(files, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=6, pady=6)

        options = tk.LabelFrame(self.root, text="Filter Options")
        options.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        for col in range(4):
            options.columnconfigure(col, weight=1)

        self._add_option(options, 0, 0, "Max Angle Deg", self.max_angle_var)
        self._add_option(options, 0, 2, "Max Position Step M", self.max_position_step_var)
        self._add_option(options, 1, 0, "Reset Gap Rows", self.reset_gap_var)
        self._add_option(options, 1, 2, "Interpolate Gap Rows", self.interpolate_gap_var)
        self._add_option(options, 2, 0, "Vote Min Support", self.vote_min_support_var)
        self._add_option(options, 2, 2, "Vote Delta Step M", self.vote_delta_step_var)
        self._add_option(options, 3, 0, "Vote Delta Angle Deg", self.vote_delta_angle_var)

        actions = tk.Frame(self.root)
        actions.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        actions.columnconfigure(1, weight=1)
        tk.Button(actions, text="Run Refine", command=self.run_refine).grid(row=0, column=0, padx=(0, 8))
        tk.Label(actions, textvariable=self.status_var, anchor="w").grid(row=0, column=1, sticky="ew")

        log_frame = tk.LabelFrame(self.root, text="Log")
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set, state="disabled")

    def _add_option(self, parent: Any, row: int, col: int, label: str, variable: Any) -> None:
        tk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=6)
        tk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=col + 1, sticky="ew", padx=6, pady=6)

    def browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Aruco Reader CSV",
            initialdir=str(self.input_dir),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.input_path_var.set(path)
        stem = Path(path).stem
        self.output_path_var.set(str((self.results_dir / f"{stem}_refined.csv").resolve()))

    def browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Refined CSV",
            initialdir=str(self.results_dir),
            initialfile=Path(self.output_path_var.get()).name or "refined.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.output_path_var.set(path)

    def run_refine(self) -> None:
        input_path = self.input_path_var.get().strip()
        output_path = self.output_path_var.get().strip()
        if not input_path:
            messagebox.showerror("Refine", "Input CSV is required.")
            return
        if not output_path:
            messagebox.showerror("Refine", "Output CSV is required.")
            return
        try:
            config = RefineConfig(
                max_angle_deg=float(self.max_angle_var.get()),
                max_position_step_m=float(self.max_position_step_var.get()),
                reset_gap_rows=int(self.reset_gap_var.get()),
                interpolate_gap_rows=int(self.interpolate_gap_var.get()),
                vote_min_support=int(self.vote_min_support_var.get()),
                vote_max_delta_step_m=float(self.vote_delta_step_var.get()),
                vote_max_delta_angle_deg=float(self.vote_delta_angle_var.get()),
            )
            result = refine_csv_file(Path(input_path), Path(output_path), config)
        except Exception as exc:
            messagebox.showerror("Refine", str(exc))
            self.status_var.set("Failed")
            return

        self.status_var.set("Refine complete")
        self._log(f"Refined -> {result.output_path}")
        self._log(f"Rows: {result.rows}")
        for marker_id, stats in result.stats.items():
            self._log(
                f"m{marker_id}: raw={stats.raw} accepted={stats.accepted} "
                f"rejected={stats.rejected} interpolated={stats.interpolated}"
            )
        messagebox.showinfo("Refine", f"Refined CSV saved:\n{result.output_path}")

    def _log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(line) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Refine error_collector ArUco CSV p/q tracks.")
    parser.add_argument("input_csv", type=Path, help="Input ArUco CSV from aruco_reader.")
    parser.add_argument("output_csv", type=Path, help="Output refined ArUco CSV.")
    parser.add_argument("--max-angle-deg", type=float, default=75.0)
    parser.add_argument("--max-position-step-m", type=float, default=0.10)
    parser.add_argument("--reset-gap-rows", type=int, default=5)
    parser.add_argument("--interpolate-gap-rows", type=int, default=2)
    parser.add_argument("--disable-voting", action="store_true")
    parser.add_argument("--vote-min-support", type=int, default=2)
    parser.add_argument("--vote-max-delta-step-m", type=float, default=0.04)
    parser.add_argument("--vote-max-delta-angle-deg", type=float, default=30.0)
    args = parser.parse_args(argv)

    result = refine_csv_file(
        args.input_csv,
        args.output_csv,
        RefineConfig(
            max_angle_deg=args.max_angle_deg,
            max_position_step_m=args.max_position_step_m,
            reset_gap_rows=args.reset_gap_rows,
            interpolate_gap_rows=args.interpolate_gap_rows,
            enable_voting=not bool(args.disable_voting),
            vote_min_support=args.vote_min_support,
            vote_max_delta_step_m=args.vote_max_delta_step_m,
            vote_max_delta_angle_deg=args.vote_max_delta_angle_deg,
        ),
    )

    print(f"Refined -> {result.output_path}")
    print(f"Rows: {result.rows}")
    for marker_id, stats in result.stats.items():
        print(
            f"m{marker_id}: raw={stats.raw} accepted={stats.accepted} "
            f"rejected={stats.rejected} interpolated={stats.interpolated}"
        )


def main() -> None:
    if len(sys.argv) > 1:
        run_cli(sys.argv[1:])
        return
    if tk is None:
        raise RuntimeError("tkinter is not available. Install Python tkinter support to use the GUI.")
    root = tk.Tk()
    RefinerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
