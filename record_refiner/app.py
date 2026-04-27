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
    from .refine_core import refine_csv_file
except ImportError:
    from refine_core import refine_csv_file  # type: ignore


class RefinerApp:
    def __init__(self, root: Any) -> None:
        if tk is None or filedialog is None or messagebox is None:
            raise RuntimeError("tkinter is not available. Install Python tkinter support to use the GUI.")
        self.root = root
        self.root.title("Record Refiner")
        self.root.geometry("820x520")

        self.base_dir = Path(__file__).resolve().parent
        self.input_dir = self.base_dir.parent / "recorder" / "records"
        self.results_dir = self.base_dir / "results"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.input_path_var = tk.StringVar(value="")
        self.output_path_var = tk.StringVar(value=str((self.results_dir / "record_refined.csv").resolve()))
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

        info = tk.LabelFrame(self.root, text="Model")
        info.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        tk.Label(
            info,
            justify="left",
            anchor="w",
            text=(
                "Reads recorder CSV rows and appends model-predicted ArUco poses.\n"
                "Output columns use mp{id}x/y/z and mq{id}x/y/z/w for markers 1-12.\n"
                "Prediction uses the node9 marker layout and the same control-to-angle mapping as player."
            ),
        ).pack(fill="x", padx=8, pady=8)

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

    def browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Recorder CSV",
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
            initialfile=Path(self.output_path_var.get()).name or "record_refined.csv",
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
            result = refine_csv_file(Path(input_path), Path(output_path))
        except Exception as exc:
            messagebox.showerror("Refine", str(exc))
            self.status_var.set("Failed")
            return

        self.status_var.set("Refine complete")
        self._log(f"Refined -> {result.output_path}")
        self._log(f"Rows: {result.rows}")
        messagebox.showinfo("Refine", f"Refined CSV saved:\n{result.output_path}")

    def _log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(line) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Annotate recorder CSV with model-predicted ArUco poses.")
    parser.add_argument("input_csv", type=Path, help="Input recorder CSV.")
    parser.add_argument("output_csv", type=Path, help="Output refined CSV.")
    args = parser.parse_args(argv)

    result = refine_csv_file(args.input_csv, args.output_csv)
    print(f"Refined -> {result.output_path}")
    print(f"Rows: {result.rows}")


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
