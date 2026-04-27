from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class TimedRow:
    t: float
    row: dict[str, str]


@dataclass(frozen=True)
class MergeResult:
    output_path: Path
    recorder_rows: int
    matched_rows: int
    unmatched_rows: int
    tolerance_sec: float


def load_timed_csv(path: Path) -> tuple[list[str], list[TimedRow]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if "t" not in fieldnames:
            raise RuntimeError(f"{path.name} does not contain a 't' column")
        rows: list[TimedRow] = []
        for row in reader:
            raw_t = str(row.get("t", "")).strip()
            if not raw_t:
                continue
            rows.append(TimedRow(t=float(raw_t), row=dict(row)))
    rows.sort(key=lambda item: item.t)
    return fieldnames, rows


def infer_half_frame_tolerance(aruco_rows: Iterable[TimedRow], fallback_fps: float = 30.0) -> float:
    values = [row.t for row in aruco_rows]
    if len(values) < 2:
        return 0.5 / float(fallback_fps)
    deltas = [values[idx + 1] - values[idx] for idx in range(len(values) - 1)]
    positive = sorted(delta for delta in deltas if delta > 0.0)
    if not positive:
        return 0.5 / float(fallback_fps)
    median = positive[len(positive) // 2]
    return 0.5 * float(median)


def _nearest_aruco_row(recorder_t: float, aruco_rows: list[TimedRow]) -> TimedRow | None:
    if not aruco_rows:
        return None
    times = [row.t for row in aruco_rows]
    idx = bisect_left(times, float(recorder_t))
    candidates: list[TimedRow] = []
    if idx < len(aruco_rows):
        candidates.append(aruco_rows[idx])
    if idx > 0:
        candidates.append(aruco_rows[idx - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item.t - float(recorder_t)))


def recorder_output_fields(recorder_fields: list[str]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    imu_sources = {
        "imu1_qw",
        "imu1_qx",
        "imu1_qy",
        "imu1_qz",
        "imu2_qw",
        "imu2_qx",
        "imu2_qy",
        "imu2_qz",
    }
    for field in recorder_fields:
        if field == "t" or field in imu_sources or field in seen:
            continue
        fields.append(field)
        seen.add(field)
    # Preserve any recorder-side derived columns, including record_refiner
    # model pose fields like mp{id}* and mq{id}*.
    for imu_id in (1, 2):
        source_fields = [f"imu{imu_id}_q{axis}" for axis in ("x", "y", "z", "w")]
        if all(field in recorder_fields for field in source_fields):
            for axis in ("x", "y", "z", "w"):
                fields.append(f"qIMU{imu_id}{axis}")
    return fields


def copy_recorder_values(source: dict[str, str], output: dict[str, str], output_fields: list[str]) -> None:
    for field in output_fields:
        output[field] = source.get(field, "")
    for imu_id in (1, 2):
        for axis in ("x", "y", "z", "w"):
            output_field = f"qIMU{imu_id}{axis}"
            if output_field in output:
                output[output_field] = source.get(f"imu{imu_id}_q{axis}", "")


def merge_csv_files(recorder_csv: Path, aruco_csv: Path, output_csv: Path) -> MergeResult:
    recorder_fields, recorder_rows = load_timed_csv(recorder_csv)
    aruco_fields, aruco_rows = load_timed_csv(aruco_csv)
    tolerance_sec = infer_half_frame_tolerance(aruco_rows)

    recorder_extra_fields = recorder_output_fields(recorder_fields)
    aruco_extra_fields = [field for field in aruco_fields if field != "t"]
    output_fields = ["t_aruco", "t_recorder", "dt"] + recorder_extra_fields + aruco_extra_fields

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    matched_rows = 0
    unmatched_rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for recorder_row in recorder_rows:
            output_row = {
                "t_aruco": "",
                "t_recorder": f"{recorder_row.t:.6f}",
                "dt": "",
            }
            copy_recorder_values(recorder_row.row, output_row, recorder_extra_fields)
            for field in aruco_extra_fields:
                output_row[field] = ""

            nearest = _nearest_aruco_row(recorder_row.t, aruco_rows)
            if nearest is not None:
                dt = float(nearest.t) - float(recorder_row.t)
                if abs(dt) <= tolerance_sec:
                    output_row["t_aruco"] = f"{nearest.t:.6f}"
                    output_row["dt"] = f"{dt:.6f}"
                    for field in aruco_extra_fields:
                        output_row[field] = nearest.row.get(field, "")
                    matched_rows += 1
                else:
                    unmatched_rows += 1
            else:
                unmatched_rows += 1

            writer.writerow(output_row)

    return MergeResult(
        output_path=output_csv,
        recorder_rows=len(recorder_rows),
        matched_rows=matched_rows,
        unmatched_rows=unmatched_rows,
        tolerance_sec=tolerance_sec,
    )
