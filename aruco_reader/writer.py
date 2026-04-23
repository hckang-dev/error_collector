from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:
    from .geometry import Pose
except ImportError:
    from geometry import Pose  # type: ignore


NODE_MARKER_IDS = tuple(range(1, 13))
CAMERA_COLUMNS = ["pCAMx", "pCAMy", "pCAMz", "qCAMx", "qCAMy", "qCAMz", "qCAMw"]
CAMERA_ROW_VALUES = {
    "pCAMx": "0.000000",
    "pCAMy": "0.000000",
    "pCAMz": "0.000000",
    "qCAMx": "0.000000",
    "qCAMy": "0.000000",
    "qCAMz": "0.000000",
    "qCAMw": "1.000000",
}


def csv_headers() -> list[str]:
    headers = ["t", *CAMERA_COLUMNS]
    for marker_id in NODE_MARKER_IDS:
        headers.extend(
            [
                f"p{marker_id}x",
                f"p{marker_id}y",
                f"p{marker_id}z",
                f"q{marker_id}x",
                f"q{marker_id}y",
                f"q{marker_id}z",
                f"q{marker_id}w",
            ]
        )
    return headers


@dataclass
class ArucoCsvWriter:
    rows: List[Dict[str, str]] = field(default_factory=list)

    def reset(self) -> None:
        self.rows = []

    def write_row(self, t_sec: float, relative_poses: Dict[int, Pose]) -> None:
        row: Dict[str, str] = {"t": f"{float(t_sec):.6f}", **CAMERA_ROW_VALUES}
        for marker_id in NODE_MARKER_IDS:
            pose = relative_poses.get(int(marker_id))
            if pose is None:
                for key in ("x", "y", "z"):
                    row[f"p{marker_id}{key}"] = ""
                    row[f"q{marker_id}{key}"] = ""
                row[f"q{marker_id}w"] = ""
                continue
            row[f"p{marker_id}x"] = f"{pose.p[0]:.6f}"
            row[f"p{marker_id}y"] = f"{pose.p[1]:.6f}"
            row[f"p{marker_id}z"] = f"{pose.p[2]:.6f}"
            row[f"q{marker_id}x"] = f"{pose.q[0]:.8f}"
            row[f"q{marker_id}y"] = f"{pose.q[1]:.8f}"
            row[f"q{marker_id}z"] = f"{pose.q[2]:.8f}"
            row[f"q{marker_id}w"] = f"{pose.q[3]:.8f}"
        self.rows.append(row)

    def export_csv(self, output_path: Path) -> Path:
        target = output_path.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_headers())
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        return target
