from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]

COMMAND_DIRECTION = (-1, -1, 1, -1)
ROLL_RANGE_RAD = (-math.pi / 2.0, math.pi / 2.0)
BEND_RANGE_RAD = (-math.radians(36.0), math.radians(36.0))
ROLL_JOINT_OFFSET_M = np.array([0.0, 0.0, 0.128], dtype=np.float64)
NODE_CHAIN_SEGMENTS_M: tuple[float, ...] = (0.028,) + (0.05,) * 9


def _map_control_to_axis(u_value: float, direction: int, q_min: float, q_max: float) -> float:
    u = float(min(max(float(u_value), 0.0), 360.0))
    if int(direction) < 0:
        u = 360.0 - u
    ratio = u / 360.0
    return float(q_min) + ratio * (float(q_max) - float(q_min))


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _transform(translation: Sequence[float], rotation: Optional[np.ndarray] = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def nominal_chain_xyz_from_motors_deg(roll_deg: float, seg1_deg: float, seg2_deg: float, n_nodes: int) -> np.ndarray:
    roll_rad = _map_control_to_axis(float(roll_deg), COMMAND_DIRECTION[1], *ROLL_RANGE_RAD)
    seg1_rad = _map_control_to_axis(float(seg1_deg), COMMAND_DIRECTION[2], *BEND_RANGE_RAD)
    seg2_rad = _map_control_to_axis(float(seg2_deg), COMMAND_DIRECTION[3], *BEND_RANGE_RAD)
    bend_angles = [seg1_rad] * 5 + [seg2_rad] * 5
    T = _transform(ROLL_JOINT_OFFSET_M, _rotation_x(roll_rad))
    points: List[np.ndarray] = [T[:3, 3].copy()]
    for length_m, bend_rad in zip(NODE_CHAIN_SEGMENTS_M, bend_angles):
        T = T @ _transform((float(length_m), 0.0, 0.0), _rotation_y(bend_rad))
        points.append(T[:3, 3].copy())
    pts = np.stack(points, axis=0)
    if int(n_nodes) <= int(pts.shape[0]):
        return pts[: int(n_nodes)].astype(np.float64)
    pad = int(n_nodes) - int(pts.shape[0])
    if pad > 0:
        tail = pts[-1:]
        return np.vstack([pts, np.repeat(tail, pad, axis=0)]).astype(np.float64)
    return pts.astype(np.float64)


def nominal_joint_deg_from_chain_xyz(nodes_xyz: np.ndarray) -> np.ndarray:
    if nodes_xyz.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    v = np.diff(nodes_xyz[:, :2], axis=0)
    lens = np.linalg.norm(v, axis=1)
    lens = np.maximum(lens, 1e-9)
    v = v / lens.reshape(-1, 1)
    angles: List[float] = []
    angles.append(math.degrees(math.atan2(float(v[0, 1]), float(v[0, 0]))))
    for i in range(1, v.shape[0]):
        a0 = math.atan2(float(v[i - 1, 1]), float(v[i - 1, 0]))
        a1 = math.atan2(float(v[i, 1]), float(v[i, 0]))
        da = a1 - a0
        while da > math.pi:
            da -= 2.0 * math.pi
        while da < -math.pi:
            da += 2.0 * math.pi
        angles.append(math.degrees(da))
    return np.asarray(angles, dtype=np.float64)


def load_homography_3x3(path: Path) -> np.ndarray:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "H" in raw:
        raw = raw["H"]
    arr = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    return arr


def pixels_to_plane_xy(
    px_x: float,
    px_y: float,
    meters_per_pixel: Optional[float],
    H: Optional[np.ndarray],
) -> tuple[float, float]:
    if H is not None:
        v = np.array([float(px_x), float(px_y), 1.0], dtype=np.float64)
        w = H @ v
        if abs(w[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(w[0] / w[2]), float(w[1] / w[2])
    if meters_per_pixel is None:
        return float("nan"), float("nan")
    s = float(meters_per_pixel)
    return float(px_x) * s, float(px_y) * s


def vision_joint_deg_from_nodes_xy(nodes_xy_m: np.ndarray) -> np.ndarray:
    if nodes_xy_m.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    v = np.diff(nodes_xy_m[:, :2], axis=0)
    lens = np.linalg.norm(v, axis=1)
    lens = np.maximum(lens, 1e-9)
    v = v / lens.reshape(-1, 1)
    angles: List[float] = []
    angles.append(math.degrees(math.atan2(float(v[0, 1]), float(v[0, 0]))))
    for i in range(1, v.shape[0]):
        a0 = math.atan2(float(v[i - 1, 1]), float(v[i - 1, 0]))
        a1 = math.atan2(float(v[i, 1]), float(v[i, 0]))
        da = a1 - a0
        while da > math.pi:
            da -= 2.0 * math.pi
        while da < -math.pi:
            da += 2.0 * math.pi
        angles.append(math.degrees(da))
    return np.asarray(angles, dtype=np.float64)


def build_shape_rows(
    refined_rows: Iterable[dict[str, str]],
    max_nodes: int,
    meters_per_pixel: Optional[float],
    homography: Optional[np.ndarray],
) -> List[dict[str, str]]:
    out: List[dict[str, str]] = []
    for row in refined_rows:
        valid_shape = str(row.get("valid_shape", "")).lower() == "true"
        mean_conf = row.get("mean_conf", row.get("confidence", ""))
        nodes_xy = []
        nodes_conf = []
        for i in range(max_nodes):
            sx = row.get(f"node{i}_px_x", "")
            sy = row.get(f"node{i}_px_y", "")
            sc = row.get(f"node{i}_conf", "")
            if str(sx).strip() == "" or str(sy).strip() == "":
                continue
            x_m, y_m = pixels_to_plane_xy(float(sx), float(sy), meters_per_pixel, homography)
            nodes_xy.append((x_m, y_m))
            try:
                nodes_conf.append(float(sc))
            except Exception:
                nodes_conf.append(0.0)
        n_nodes = len(nodes_xy)
        arr_xy = np.asarray(nodes_xy, dtype=np.float64) if nodes_xy else np.zeros((0, 2), dtype=np.float64)
        jdeg = vision_joint_deg_from_nodes_xy(arr_xy)
        ee_x = float(arr_xy[-1, 0]) if n_nodes else float("nan")
        ee_y = float(arr_xy[-1, 1]) if n_nodes else float("nan")
        ee_z = 0.0 if n_nodes else float("nan")
        out_row: Dict[str, str] = {
            "frame_idx": row.get("frame_idx", ""),
            "t_video_sec": row.get("t_video_sec", ""),
            "valid_shape": "true" if valid_shape and n_nodes >= max_nodes else "false",
            "n_nodes": str(int(n_nodes)),
            "mean_conf": str(mean_conf),
            "confidence": str(mean_conf),
            "validity": "true" if valid_shape and n_nodes >= max_nodes else "false",
            "sync_dt_sec": row.get("sync_dt_sec", ""),
        }
        for i in range(max_nodes):
            if i < n_nodes:
                x_m, y_m = nodes_xy[i]
                out_row[f"node{i}_x_m"] = f"{float(x_m):.6f}"
                out_row[f"node{i}_y_m"] = f"{float(y_m):.6f}"
                out_row[f"node{i}_z_m"] = "0.000000"
            else:
                out_row[f"node{i}_x_m"] = ""
                out_row[f"node{i}_y_m"] = ""
                out_row[f"node{i}_z_m"] = ""
        n_joints = max(0, max_nodes - 1)
        for j in range(n_joints):
            if j < jdeg.shape[0]:
                out_row[f"joint{j}_deg"] = f"{float(jdeg[j]):.4f}"
            else:
                out_row[f"joint{j}_deg"] = ""
        out_row["ee_x_m"] = f"{ee_x:.6f}" if math.isfinite(ee_x) else ""
        out_row["ee_y_m"] = f"{ee_y:.6f}" if math.isfinite(ee_y) else ""
        out_row["ee_z_m"] = f"{ee_z:.6f}" if math.isfinite(ee_z) else ""
        out.append(out_row)
    return out


def shape_fieldnames(max_nodes: int) -> List[str]:
    fields = [
        "frame_idx",
        "t_video_sec",
        "valid_shape",
        "n_nodes",
        "mean_conf",
        "confidence",
        "validity",
        "sync_dt_sec",
    ]
    for i in range(max_nodes):
        fields.extend([f"node{i}_x_m", f"node{i}_y_m", f"node{i}_z_m"])
    for j in range(max(0, max_nodes - 1)):
        fields.append(f"joint{j}_deg")
    fields.extend(["ee_x_m", "ee_y_m", "ee_z_m"])
    return fields


def build_shape_csv(input_csv: Path, output_csv: Path, max_nodes: int, meters_per_pixel: Optional[float], homography_json: Optional[Path]) -> int:
    H = load_homography_3x3(homography_json) if homography_json is not None else None
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(r) for r in reader]
    built = build_shape_rows(rows, max_nodes, meters_per_pixel, H)
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = shape_fieldnames(max_nodes)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in built:
            writer.writerow(r)
    return len(built)
