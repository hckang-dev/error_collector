from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from .hardware import HardwareSnapshot, ImuSensorSample
except ImportError:
    from hardware import HardwareSnapshot, ImuSensorSample  # type: ignore


def _imu_field_names(imu_id: int) -> list[str]:
    return [
        f"imu{imu_id}_timestamp",
        f"imu{imu_id}_gvx",
        f"imu{imu_id}_gvy",
        f"imu{imu_id}_gvz",
        f"imu{imu_id}_gx",
        f"imu{imu_id}_gy",
        f"imu{imu_id}_gz",
        f"imu{imu_id}_qw",
        f"imu{imu_id}_qx",
        f"imu{imu_id}_qy",
        f"imu{imu_id}_qz",
        f"imu{imu_id}_quatAcc",
        f"imu{imu_id}_gravAcc",
        f"imu{imu_id}_gyroAcc",
        f"imu{imu_id}_lax",
        f"imu{imu_id}_lay",
        f"imu{imu_id}_laz",
        f"imu{imu_id}_ax",
        f"imu{imu_id}_ay",
        f"imu{imu_id}_az",
        f"imu{imu_id}_mx",
        f"imu{imu_id}_my",
        f"imu{imu_id}_mz",
    ]


CSV_COLUMNS = (
    ["t", "roll", "seg1", "seg2", "load1", "load2"]
    + _imu_field_names(1)
    + _imu_field_names(2)
)


def _fmt_opt(value: Optional[float], precision: int = 8) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (not math.isfinite(value)):
        return ""
    return f"{float(value):.{precision}f}"


def _imu_to_row(imu: ImuSensorSample, imu_id: int) -> dict[str, str]:
    p = imu_id
    return {
        f"imu{p}_timestamp": _fmt_opt(imu.timestamp, 6),
        f"imu{p}_gvx": _fmt_opt(imu.gvx),
        f"imu{p}_gvy": _fmt_opt(imu.gvy),
        f"imu{p}_gvz": _fmt_opt(imu.gvz),
        f"imu{p}_gx": _fmt_opt(imu.gx),
        f"imu{p}_gy": _fmt_opt(imu.gy),
        f"imu{p}_gz": _fmt_opt(imu.gz),
        f"imu{p}_qw": f"{imu.qw:.8f}",
        f"imu{p}_qx": f"{imu.qx:.8f}",
        f"imu{p}_qy": f"{imu.qy:.8f}",
        f"imu{p}_qz": f"{imu.qz:.8f}",
        f"imu{p}_quatAcc": _fmt_opt(imu.quatAcc),
        f"imu{p}_gravAcc": _fmt_opt(imu.gravAcc),
        f"imu{p}_gyroAcc": _fmt_opt(imu.gyroAcc),
        f"imu{p}_lax": _fmt_opt(imu.lax),
        f"imu{p}_lay": _fmt_opt(imu.lay),
        f"imu{p}_laz": _fmt_opt(imu.laz),
        f"imu{p}_ax": _fmt_opt(imu.ax),
        f"imu{p}_ay": _fmt_opt(imu.ay),
        f"imu{p}_az": _fmt_opt(imu.az),
        f"imu{p}_mx": _fmt_opt(imu.mx),
        f"imu{p}_my": _fmt_opt(imu.my),
        f"imu{p}_mz": _fmt_opt(imu.mz),
    }


@dataclass
class RecordedSample:
    t: float
    roll: float
    seg1: float
    seg2: float
    load1: Optional[float]
    load2: Optional[float]
    imu1: ImuSensorSample
    imu2: ImuSensorSample

    def to_row(self) -> dict:
        row = {
            "t": f"{self.t:.3f}",
            "roll": f"{self.roll:.2f}",
            "seg1": f"{self.seg1:.2f}",
            "seg2": f"{self.seg2:.2f}",
            "load1": "" if self.load1 is None else f"{self.load1:.3f}",
            "load2": "" if self.load2 is None else f"{self.load2:.3f}",
        }
        row.update(_imu_to_row(self.imu1, 1))
        row.update(_imu_to_row(self.imu2, 2))
        return row


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
                imu1=snapshot.imu1,
                imu2=snapshot.imu2,
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
