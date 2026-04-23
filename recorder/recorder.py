from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from .hardware import HardwareSnapshot
except ImportError:
    from hardware import HardwareSnapshot  # type: ignore


CSV_COLUMNS = [
    "t",
    "roll",
    "seg1",
    "seg2",
    "load1",
    "load2",
    "imu1_qw",
    "imu1_qx",
    "imu1_qy",
    "imu1_qz",
    "imu2_qw",
    "imu2_qx",
    "imu2_qy",
    "imu2_qz",
]


@dataclass
class RecordedSample:
    t: float
    roll: float
    seg1: float
    seg2: float
    load1: Optional[float]
    load2: Optional[float]
    imu1_qw: float
    imu1_qx: float
    imu1_qy: float
    imu1_qz: float
    imu2_qw: float
    imu2_qx: float
    imu2_qy: float
    imu2_qz: float

    def to_row(self) -> dict:
        return {
            "t": f"{self.t:.3f}",
            "roll": f"{self.roll:.2f}",
            "seg1": f"{self.seg1:.2f}",
            "seg2": f"{self.seg2:.2f}",
            "load1": "" if self.load1 is None else f"{self.load1:.3f}",
            "load2": "" if self.load2 is None else f"{self.load2:.3f}",
            "imu1_qw": f"{self.imu1_qw:.8f}",
            "imu1_qx": f"{self.imu1_qx:.8f}",
            "imu1_qy": f"{self.imu1_qy:.8f}",
            "imu1_qz": f"{self.imu1_qz:.8f}",
            "imu2_qw": f"{self.imu2_qw:.8f}",
            "imu2_qx": f"{self.imu2_qx:.8f}",
            "imu2_qy": f"{self.imu2_qy:.8f}",
            "imu2_qz": f"{self.imu2_qz:.8f}",
        }


class SessionRecorder:
    def __init__(self, sample_hz: float = 10.0) -> None:
        self.sample_period_s = 1.0 / max(0.1, float(sample_hz))
        self.samples: List[RecordedSample] = []
        self.recording_active = False
        self._recording_started_at: Optional[float] = None
        self._last_sample_wall: Optional[float] = None

    def start(self) -> None:
        self.samples = []
        self.recording_active = True
        self._recording_started_at = None
        self._last_sample_wall = None

    def stop(self) -> None:
        self.recording_active = False

    def reset(self) -> None:
        self.samples = []
        self.recording_active = False
        self._recording_started_at = None
        self._last_sample_wall = None

    def capture_if_due(self, snapshot: HardwareSnapshot) -> bool:
        if not self.recording_active:
            return False
        now = float(snapshot.captured_at)
        if self._recording_started_at is None:
            self._recording_started_at = now
        if self._last_sample_wall is not None and (now - self._last_sample_wall) < self.sample_period_s:
            return False
        self._last_sample_wall = now
        self.samples.append(
            RecordedSample(
                t=now - self._recording_started_at,
                roll=float(snapshot.roll),
                seg1=float(snapshot.seg1),
                seg2=float(snapshot.seg2),
                load1=snapshot.load1_ma,
                load2=snapshot.load2_ma,
                imu1_qw=float(snapshot.imu1.qw),
                imu1_qx=float(snapshot.imu1.qx),
                imu1_qy=float(snapshot.imu1.qy),
                imu1_qz=float(snapshot.imu1.qz),
                imu2_qw=float(snapshot.imu2.qw),
                imu2_qx=float(snapshot.imu2.qx),
                imu2_qy=float(snapshot.imu2.qy),
                imu2_qz=float(snapshot.imu2.qz),
            )
        )
        return True

    def export_csv(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(sample.to_row())
        return target
