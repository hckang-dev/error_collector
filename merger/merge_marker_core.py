from __future__ import annotations

import argparse
import csv
import math
import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from vision_shape_builder.shape_core import (
    nominal_chain_xyz_from_motors_deg,
    nominal_joint_deg_from_chain_xyz,
)


@dataclass(frozen=True)
class TimedRow:
    t: float
    row: dict[str, str]


@dataclass(frozen=True)
class ShapeMergeResult:
    output_path: Path
    recorder_rows: int
    matched_in_tol: int
    matched_out_tol: int
    sync_tol_sec: float


def load_recorder_timed(path: Path) -> tuple[list[str], list[TimedRow]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "t" not in fields:
            raise RuntimeError(f"{path.name} missing column t")
        rows: list[TimedRow] = []
        for row in reader:
            raw = str(row.get("t", "")).strip()
            if not raw:
                continue
            rows.append(TimedRow(t=float(raw), row=dict(row)))
    rows.sort(key=lambda r: r.t)
    return fields, rows


def load_vision_timed(path: Path) -> tuple[list[str], list[TimedRow]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "t_video_sec" not in fields:
            raise RuntimeError(f"{path.name} missing t_video_sec")
        rows: list[TimedRow] = []
        for row in reader:
            raw = str(row.get("t_video_sec", "")).strip()
            if not raw:
                continue
            rows.append(TimedRow(t=float(raw), row=dict(row)))
    rows.sort(key=lambda r: r.t)
    return fields, rows


def infer_half_frame_tolerance_vision(vision_rows: Iterable[TimedRow], fallback_fps: float = 30.0) -> float:
    values = [row.t for row in vision_rows]
    if len(values) < 2:
        return 0.5 / float(fallback_fps)
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    positive = sorted(d for d in deltas if d > 0.0)
    if not positive:
        return 0.5 / float(fallback_fps)
    median = positive[len(positive) // 2]
    return 0.5 * float(median)


def _nearest(rows: list[TimedRow], t_query: float) -> Optional[TimedRow]:
    if not rows:
        return None
    times = [r.t for r in rows]
    idx = bisect_left(times, float(t_query))
    candidates: list[TimedRow] = []
    if idx < len(rows):
        candidates.append(rows[idx])
    if idx > 0:
        candidates.append(rows[idx - 1])
    return min(candidates, key=lambda r: abs(r.t - float(t_query)))


def _float_or_nan(s: object) -> float:
    try:
        v = float(str(s).strip())
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan")


def _nan_str(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.6f}"


def merge_shape_csv_files(
    record_csv: Path,
    vision_csv: Path,
    output_csv: Path,
    max_nodes: int,
    sync_tol_sec: Optional[float],
) -> ShapeMergeResult:
    rec_fields, rec_rows = load_recorder_timed(record_csv)
    vis_fields, vis_rows = load_vision_timed(vision_csv)
    tol = float(sync_tol_sec) if sync_tol_sec is not None else infer_half_frame_tolerance_vision(vis_rows)

    rec_skip_t = [f for f in rec_fields if f != "t"]
    vision_cols = [f for f in vis_fields if f not in ("frame_idx",)]
    vision_prefixed = [f"vision_{f}" for f in vision_cols]

    nominal_node_fields: list[str] = []
    for i in range(max_nodes):
        nominal_node_fields.extend([f"nominal_node{i}_x_m", f"nominal_node{i}_y_m", f"nominal_node{i}_z_m"])
    nominal_joint_fields = [f"nominal_joint{j}_deg" for j in range(max(0, max_nodes - 1))]
    nominal_ee_fields = ["nominal_ee_x_m", "nominal_ee_y_m", "nominal_ee_z_m"]

    res_node_fields: list[str] = []
    for i in range(max_nodes):
        res_node_fields.extend([f"res_node{i}_x_m", f"res_node{i}_y_m", f"res_node{i}_z_m"])
    res_ee_fields = ["res_ee_x_m", "res_ee_y_m", "res_ee_z_m"]

    valid_node_fields = [f"valid_node{i}" for i in range(max_nodes)]

    head = [
        "t_recorder",
        "t_vision",
        "dt_vision",
        "sync_dt_sec",
        "sync_tol_sec",
        "valid_shape",
        "mean_conf",
        "confidence",
        "imu_fresh",
    ]
    out_fields = (
        head
        + rec_skip_t
        + nominal_node_fields
        + nominal_joint_fields
        + nominal_ee_fields
        + vision_prefixed
        + res_node_fields
        + res_ee_fields
        + ["valid_ee"] + valid_node_fields
    )

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    in_tol = 0
    out_tol = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        for rec in rec_rows:
            roll = _float_or_nan(rec.row.get("roll", ""))
            seg1 = _float_or_nan(rec.row.get("seg1", ""))
            seg2 = _float_or_nan(rec.row.get("seg2", ""))
            nominal_xyz = nominal_chain_xyz_from_motors_deg(roll, seg1, seg2, max_nodes)
            j_nom = nominal_joint_deg_from_chain_xyz(nominal_xyz)
            nominal_row: Dict[str, str] = {}
            for i in range(max_nodes):
                if i < nominal_xyz.shape[0]:
                    nominal_row[f"nominal_node{i}_x_m"] = f"{float(nominal_xyz[i, 0]):.6f}"
                    nominal_row[f"nominal_node{i}_y_m"] = f"{float(nominal_xyz[i, 1]):.6f}"
                    nominal_row[f"nominal_node{i}_z_m"] = f"{float(nominal_xyz[i, 2]):.6f}"
                else:
                    nominal_row[f"nominal_node{i}_x_m"] = ""
                    nominal_row[f"nominal_node{i}_y_m"] = ""
                    nominal_row[f"nominal_node{i}_z_m"] = ""
            for j in range(max(0, max_nodes - 1)):
                nominal_row[f"nominal_joint{j}_deg"] = f"{float(j_nom[j]):.4f}" if j < j_nom.shape[0] else ""
            nominal_row["nominal_ee_x_m"] = f"{float(nominal_xyz[-1, 0]):.6f}" if nominal_xyz.shape[0] else ""
            nominal_row["nominal_ee_y_m"] = f"{float(nominal_xyz[-1, 1]):.6f}" if nominal_xyz.shape[0] else ""
            nominal_row["nominal_ee_z_m"] = f"{float(nominal_xyz[-1, 2]):.6f}" if nominal_xyz.shape[0] else ""

            nearest = _nearest(vis_rows, rec.t)
            out: Dict[str, str] = {k: "" for k in out_fields}
            out["t_recorder"] = f"{rec.t:.6f}"
            out["sync_tol_sec"] = f"{tol:.8f}"
            for f in rec_skip_t:
                out[f] = rec.row.get(f, "")

            imu_fresh = "true"
            for imu_id in (1, 2):
                ts_key = f"imu{imu_id}_timestamp"
                if ts_key in rec.row and str(rec.row.get(ts_key, "")).strip():
                    try:
                        imu_t = float(rec.row[ts_key])
                        if math.isfinite(imu_t) and abs(imu_t - rec.t) > tol * 4.0:
                            imu_fresh = "false"
                    except Exception:
                        imu_fresh = "false"
            out["imu_fresh"] = imu_fresh

            for k, v in nominal_row.items():
                out[k] = v

            if nearest is None:
                dtv = float("nan")
                out["t_vision"] = ""
                out["dt_vision"] = "nan"
                out["sync_dt_sec"] = "nan"
                out["valid_shape"] = "false"
                out["mean_conf"] = ""
                out["confidence"] = ""
                for vf in vision_prefixed:
                    out[vf] = "nan"
                for rf in res_node_fields + res_ee_fields:
                    out[rf] = "nan"
                out["valid_ee"] = "false"
                for vf in valid_node_fields:
                    out[vf] = "false"
                out_tol += 1
                writer.writerow(out)
                continue

            dtv = float(nearest.t) - float(rec.t)
            out["t_vision"] = f"{nearest.t:.6f}"
            out["dt_vision"] = f"{dtv:.6f}"
            out["sync_dt_sec"] = f"{dtv:.6f}"

            valid_sync = abs(dtv) <= tol
            if valid_sync:
                in_tol += 1
            else:
                out_tol += 1

            vrow = nearest.row
            vs_raw = str(vrow.get("valid_shape", "")).lower() == "true"
            valid_shape = bool(valid_sync and vs_raw)
            out["valid_shape"] = "true" if valid_shape else "false"
            out["mean_conf"] = vrow.get("mean_conf", vrow.get("confidence", ""))
            out["confidence"] = vrow.get("confidence", vrow.get("mean_conf", ""))

            if not valid_shape:
                for col, src in zip(vision_prefixed, vision_cols):
                    out[col] = "nan"
                for rf in res_node_fields + res_ee_fields:
                    out[rf] = "nan"
                out["valid_ee"] = "false"
                for vf in valid_node_fields:
                    out[vf] = "false"
                writer.writerow(out)
                continue

            for col, src in zip(vision_prefixed, vision_cols):
                out[col] = vrow.get(src, "")

            vx = [_float_or_nan(vrow.get(f"node{i}_x_m", "")) for i in range(max_nodes)]
            vy = [_float_or_nan(vrow.get(f"node{i}_y_m", "")) for i in range(max_nodes)]
            vz = [_float_or_nan(vrow.get(f"node{i}_z_m", "")) for i in range(max_nodes)]

            ee_ok = True
            for i in range(max_nodes):
                nx = _float_or_nan(nominal_row.get(f"nominal_node{i}_x_m", "nan"))
                ny = _float_or_nan(nominal_row.get(f"nominal_node{i}_y_m", "nan"))
                nz = _float_or_nan(nominal_row.get(f"nominal_node{i}_z_m", "nan"))
                vi_ok = (
                    i < len(vx)
                    and math.isfinite(vx[i])
                    and math.isfinite(vy[i])
                    and math.isfinite(vz[i])
                    and math.isfinite(nx)
                    and math.isfinite(ny)
                    and math.isfinite(nz)
                )
                if vi_ok:
                    out[f"res_node{i}_x_m"] = _nan_str(vx[i] - nx)
                    out[f"res_node{i}_y_m"] = _nan_str(vy[i] - ny)
                    out[f"res_node{i}_z_m"] = _nan_str(vz[i] - nz)
                    out[f"valid_node{i}"] = "true"
                else:
                    out[f"res_node{i}_x_m"] = "nan"
                    out[f"res_node{i}_y_m"] = "nan"
                    out[f"res_node{i}_z_m"] = "nan"
                    out[f"valid_node{i}"] = "false"
                    ee_ok = False

            vex = _float_or_nan(vrow.get("ee_x_m", ""))
            vey = _float_or_nan(vrow.get("ee_y_m", ""))
            vez = _float_or_nan(vrow.get("ee_z_m", ""))
            nex = _float_or_nan(nominal_row.get("nominal_ee_x_m", ""))
            ney = _float_or_nan(nominal_row.get("nominal_ee_y_m", ""))
            nez = _float_or_nan(nominal_row.get("nominal_ee_z_m", ""))
            if all(math.isfinite(v) for v in (vex, vey, vez, nex, ney, nez)):
                out["res_ee_x_m"] = _nan_str(vex - nex)
                out["res_ee_y_m"] = _nan_str(vey - ney)
                out["res_ee_z_m"] = _nan_str(vez - nez)
                out["valid_ee"] = "true" if ee_ok else "false"
            else:
                out["res_ee_x_m"] = "nan"
                out["res_ee_y_m"] = "nan"
                out["res_ee_z_m"] = "nan"
                out["valid_ee"] = "false"

            writer.writerow(out)

    return ShapeMergeResult(
        output_path=output_csv,
        recorder_rows=len(rec_rows),
        matched_in_tol=in_tol,
        matched_out_tol=out_tol,
        sync_tol_sec=tol,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge record_refiner CSV with vision_shape_builder CSV.")
    p.add_argument("--record", type=str, required=True)
    p.add_argument("--vision", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-nodes", type=int, default=11)
    p.add_argument("--sync-tol-sec", type=float, default=None, help="Override inferred half-frame tolerance")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = merge_shape_csv_files(
        Path(args.record),
        Path(args.vision),
        Path(args.out),
        int(args.max_nodes),
        float(args.sync_tol_sec) if args.sync_tol_sec is not None else None,
    )
    print(
        f"wrote {res.output_path} rows={res.recorder_rows} in_tol={res.matched_in_tol} out_tol={res.matched_out_tol} tol={res.sync_tol_sec:.6f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
