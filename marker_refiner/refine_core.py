from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class RefineParams:
    max_nodes: int
    min_mean_conf: float
    min_node_conf: float
    jump_px_threshold: float
    max_gap_frames: int


def _float_field(raw: object) -> float:
    try:
        v = float(str(raw).strip())
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan")


def _read_marker_rows(path: Path, max_nodes: int) -> List[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "c0_x" not in fields:
            raise RuntimeError(f"{path.name}: missing c0_x (marker_reader CSV)")
        rows = [dict(r) for r in reader]
    return rows


def _extract_detections(row: dict[str, str], max_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    cfs: List[float] = []
    for i in range(max_nodes):
        sx = row.get(f"c{i}_x", "")
        sy = row.get(f"c{i}_y", "")
        sc = row.get(f"c{i}_conf", "")
        if str(sx).strip() == "" or str(sy).strip() == "":
            continue
        cf = _float_field(sc) if str(sc).strip() else 1.0
        xs.append(float(sx))
        ys.append(float(sy))
        cfs.append(cf)
    if not xs:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)]), np.asarray(cfs, dtype=np.float64)


def _extract_anchor(row: dict[str, str]) -> np.ndarray:
    sx = str(row.get("shoulder_x", "")).strip()
    sy = str(row.get("shoulder_y", "")).strip()
    if not sx or not sy:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray([float(sx), float(sy)], dtype=np.float64)


def _sort_base_to_tip(xy: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] == 0:
        return xy, conf
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    return xy[order].copy(), conf[order].copy()


def _sort_by_anchor(xy: np.ndarray, conf: np.ndarray, anchor_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] == 0 or anchor_xy.shape[0] != 2:
        return _sort_base_to_tip(xy, conf)
    d = np.linalg.norm(xy - anchor_xy.reshape(1, 2), axis=1)
    order = np.argsort(d)
    return xy[order].copy(), conf[order].copy()


def _assign_greedy_prev_to_curr(prev_xy: np.ndarray, curr_xy: np.ndarray) -> np.ndarray:
    if curr_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if prev_xy.shape[0] == 0:
        return np.argsort(np.lexsort((curr_xy[:, 0], curr_xy[:, 1])))
    unused = set(range(int(curr_xy.shape[0])))
    perm: List[int] = []
    for i in range(int(prev_xy.shape[0])):
        best_j = -1
        best_d = float("inf")
        for j in unused:
            d = float(np.linalg.norm(curr_xy[int(j)] - prev_xy[i]))
            if d < best_d:
                best_d = d
                best_j = int(j)
        if best_j >= 0:
            perm.append(best_j)
            unused.remove(best_j)
    perm.extend(sorted(unused))
    return np.asarray(perm, dtype=np.int64)


def _order_track(prev_xy: np.ndarray, curr_xy: np.ndarray, curr_conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if curr_xy.shape[0] == 0:
        return curr_xy, curr_conf
    perm = _assign_greedy_prev_to_curr(prev_xy, curr_xy)
    return curr_xy[perm].copy(), curr_conf[perm].copy()


def _jump_scale_conf(prev_xy: np.ndarray, curr_xy: np.ndarray, conf: np.ndarray, jump_px: float) -> np.ndarray:
    out = conf.copy()
    if prev_xy.shape[0] == 0 or curr_xy.shape[0] == 0:
        return out
    n = int(min(prev_xy.shape[0], curr_xy.shape[0], out.shape[0]))
    for i in range(n):
        d = float(np.linalg.norm(curr_xy[i] - prev_xy[i]))
        if d > jump_px:
            out[i] *= 0.35
    return out


def _interpolate_gaps(
    xy_seq: List[np.ndarray],
    cf_seq: List[np.ndarray],
    max_gap: int,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    n = len(xy_seq)
    if n == 0:
        return xy_seq, cf_seq
    out_xy = [x.copy() for x in xy_seq]
    out_cf = [c.copy() for c in cf_seq]
    i = 0
    while i < n:
        if out_xy[i].shape[0] > 0:
            i += 1
            continue
        a = i - 1
        j = i
        while j < n and out_xy[j].shape[0] == 0:
            j += 1
        b = j
        gap = b - i
        if a < 0 or b >= n or gap > max_gap:
            i = max(i + 1, j)
            continue
        xa, xb = out_xy[a], out_xy[b]
        if xa.shape[0] != xb.shape[0]:
            i = max(i + 1, j)
            continue
        ca, cb = out_cf[a], out_cf[b]
        for k in range(1, gap + 1):
            alpha = float(k) / float(gap + 1)
            idx = i + k - 1
            out_xy[idx] = (1.0 - alpha) * xa + alpha * xb
            out_cf[idx] = ((1.0 - alpha) * ca + alpha * cb) * 0.85
        i = b
    return out_xy, out_cf


def refine_marker_csv(input_csv: Path, output_csv: Path, params: RefineParams) -> int:
    rows = _read_marker_rows(input_csv, params.max_nodes)
    prev_xy = np.zeros((0, 2), dtype=np.float64)
    xy_seq: List[np.ndarray] = []
    cf_seq: List[np.ndarray] = []
    for row in rows:
        xy, cf = _extract_detections(row, params.max_nodes)
        anchor_xy = _extract_anchor(row)
        mask = np.isfinite(cf) & (cf >= params.min_node_conf)
        xy = xy[mask]
        cf = cf[mask]
        if prev_xy.shape[0] == 0:
            xy, cf = _sort_by_anchor(xy, cf, anchor_xy)
        else:
            xy, cf = _order_track(prev_xy, xy, cf)
            if anchor_xy.shape[0] == 2:
                xy, cf = _sort_by_anchor(xy, cf, anchor_xy)
        xy_seq.append(xy)
        cf_seq.append(cf)
        if xy.shape[0] > 0:
            prev_xy = xy.copy()

    xy_seq, cf_seq = _interpolate_gaps(xy_seq, cf_seq, params.max_gap_frames)

    prev_xy = np.zeros((0, 2), dtype=np.float64)
    out_rows: List[dict[str, str]] = []
    for frame_idx, (row, xy, cf) in enumerate(zip(rows, xy_seq, cf_seq)):
        cf2 = _jump_scale_conf(prev_xy, xy, cf, params.jump_px_threshold) if xy.shape[0] else cf
        mean_conf = float(np.mean(cf2)) if cf2.size else 0.0
        valid_shape = xy.shape[0] >= params.max_nodes and mean_conf >= params.min_mean_conf
        out: Dict[str, str] = {
            "frame_idx": row.get("frame_idx", str(frame_idx)),
            "t_video_sec": row.get("t_video_sec", ""),
            "valid_shape": "true" if valid_shape else "false",
            "n_nodes": str(int(xy.shape[0])),
            "mean_conf": f"{mean_conf:.6f}",
            "confidence": f"{mean_conf:.6f}",
            "validity": "true" if valid_shape else "false",
            "sync_dt_sec": row.get("sync_dt_sec", ""),
        }
        for i in range(params.max_nodes):
            if i < xy.shape[0]:
                out[f"node{i}_px_x"] = f"{float(xy[i, 0]):.4f}"
                out[f"node{i}_px_y"] = f"{float(xy[i, 1]):.4f}"
                out[f"node{i}_conf"] = f"{float(cf2[i]):.6f}"
            else:
                out[f"node{i}_px_x"] = ""
                out[f"node{i}_px_y"] = ""
                out[f"node{i}_conf"] = ""
        out_rows.append(out)
        if xy.shape[0] >= params.max_nodes:
            prev_xy = xy[: params.max_nodes].copy()

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_idx", "t_video_sec", "valid_shape", "n_nodes", "mean_conf", "confidence", "validity", "sync_dt_sec"]
    for i in range(params.max_nodes):
        fieldnames.extend([f"node{i}_px_x", f"node{i}_px_y", f"node{i}_conf"])
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    return len(out_rows)
