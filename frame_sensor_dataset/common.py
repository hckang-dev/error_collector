from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


def _float_or_none(raw: object) -> Optional[float]:
    try:
        value = float(str(raw).strip())
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.{digits}f}"


@dataclass(frozen=True)
class TimedSensorRow:
    t_mono: float
    row: dict[str, str]


MANIFEST_FIELDS: list[str] = [
    "frame_idx",
    "frame_path",
    "t_frame_wall",
    "t_frame_mono",
    "t_sensor_mono",
    "dt_sensor_sec",
    "sync_valid",
    "base_deg",
    "seg1_deg",
    "seg2_deg",
    "rev_u",
    "pris_u",
    "present_current_ma_0",
    "present_current_ma_1",
    "present_current_ma_2",
    "present_current_ma_3",
    "goal_current_ma_0",
    "goal_current_ma_1",
    "goal_current_ma_2",
    "goal_current_ma_3",
    "present_velocity_0",
    "present_velocity_1",
    "present_velocity_2",
    "present_velocity_3",
    "position_error_0",
    "position_error_1",
    "position_error_2",
    "position_error_3",
    "bus_voltage_v",
    "imu_timestamp",
    "gvx",
    "gvy",
    "gvz",
    "gx",
    "gy",
    "gz",
    "qw",
    "qx",
    "qy",
    "qz",
    "quatAcc",
    "gravAcc",
    "gyroAcc",
    "lax",
    "lay",
    "laz",
    "ax",
    "ay",
    "az",
    "mx",
    "my",
    "mz",
]


def load_sensor_csv_rows(path: Path) -> list[TimedSensorRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "t" not in fields:
            raise RuntimeError(f"{path.name}: missing t column")
        rows: list[TimedSensorRow] = []
        for row in reader:
            t = _float_or_none(row.get("t", ""))
            if t is None:
                continue
            rows.append(TimedSensorRow(t_mono=float(t), row=dict(row)))
    rows.sort(key=lambda x: x.t_mono)
    return rows


def nearest_sensor_row(rows: list[TimedSensorRow], t_frame_mono: float) -> tuple[Optional[TimedSensorRow], Optional[float]]:
    if not rows:
        return None, None
    lo = 0
    hi = len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid].t_mono < t_frame_mono:
            lo = mid + 1
        else:
            hi = mid
    candidates: list[TimedSensorRow] = []
    if lo < len(rows):
        candidates.append(rows[lo])
    if lo > 0:
        candidates.append(rows[lo - 1])
    if not candidates:
        return None, None
    best = min(candidates, key=lambda r: abs(r.t_mono - t_frame_mono))
    return best, float(best.t_mono - t_frame_mono)


def build_manifest_row(
    frame_idx: int,
    frame_path_rel: str,
    t_frame_wall: float,
    t_frame_mono: float,
    sensor: Optional[dict[str, str]],
    t_sensor_mono: Optional[float],
    dt_sensor: Optional[float],
    sync_valid: bool,
) -> dict[str, str]:
    row = {k: "nan" for k in MANIFEST_FIELDS}
    row["frame_idx"] = str(int(frame_idx))
    row["frame_path"] = frame_path_rel
    row["t_frame_wall"] = _fmt(t_frame_wall, 6)
    row["t_frame_mono"] = _fmt(t_frame_mono, 6)
    row["t_sensor_mono"] = _fmt(t_sensor_mono, 6) if t_sensor_mono is not None else "nan"
    row["dt_sensor_sec"] = _fmt(dt_sensor, 6) if dt_sensor is not None else "nan"
    row["sync_valid"] = "true" if sync_valid else "false"
    if sensor is None or not sync_valid:
        return row

    row["base_deg"] = sensor.get("roll", "nan") or "nan"
    row["seg1_deg"] = sensor.get("seg1", "nan") or "nan"
    row["seg2_deg"] = sensor.get("seg2", "nan") or "nan"
    row["rev_u"] = sensor.get("rev_u", "nan") or "nan"
    row["pris_u"] = sensor.get("pris_u", "nan") or "nan"

    load1 = sensor.get("load1", "nan") or "nan"
    load2 = sensor.get("load2", "nan") or "nan"
    row["present_current_ma_0"] = sensor.get("present_current_ma_0", load1) or "nan"
    row["present_current_ma_1"] = sensor.get("present_current_ma_1", load2) or "nan"

    for i in range(2, 4):
        row[f"present_current_ma_{i}"] = sensor.get(f"present_current_ma_{i}", "nan") or "nan"
    for i in range(4):
        row[f"goal_current_ma_{i}"] = sensor.get(f"goal_current_ma_{i}", "nan") or "nan"
        row[f"present_velocity_{i}"] = sensor.get(f"present_velocity_{i}", "nan") or "nan"
        row[f"position_error_{i}"] = sensor.get(f"position_error_{i}", "nan") or "nan"
    row["bus_voltage_v"] = sensor.get("bus_voltage_v", "nan") or "nan"

    row["imu_timestamp"] = sensor.get("imu1_timestamp", sensor.get("imu_timestamp", "nan")) or "nan"
    for k in ("gvx", "gvy", "gvz", "gx", "gy", "gz", "qw", "qx", "qy", "qz", "quatAcc", "gravAcc", "gyroAcc", "lax", "lay", "laz", "ax", "ay", "az", "mx", "my", "mz"):
        row[k] = sensor.get(f"imu1_{k}", sensor.get(k, "nan")) or "nan"
    return row


def write_manifest_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
