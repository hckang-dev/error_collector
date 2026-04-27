from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # x, y, z, w

MARKER_IDS: tuple[int, ...] = tuple(range(1, 13))
Q_AXES = ("x", "y", "z", "w")
AXES = ("x", "y", "z")

COMMAND_DIRECTION = (-1, -1, 1, -1)
ROLL_RANGE_RAD = (-math.pi / 2.0, math.pi / 2.0)
BEND_RANGE_RAD = (-math.radians(36.0), math.radians(36.0))

# Base markers are fixed on the housing and define the coordinate frame used by
# aruco_reader outputs. Marker columns 1/4/7/10 are relative to base 13,
# 2/5/8/11 to base 14, and 3/6/9/12 to base 15.
ARUCO_GROUPS_BY_BASE: Dict[int, tuple[int, ...]] = {
    13: (1, 4, 7, 10),
    14: (2, 5, 8, 11),
    15: (3, 6, 9, 12),
}
BASE_OFFSETS_M: Dict[int, Vec3] = {
    13: (-0.010, 0.065, 0.010),
    14: (-0.030, 0.065, 0.010),
    15: (-0.050, 0.065, 0.010),
}
BASE_MARKER_AXES_WORLD: Dict[int, tuple[str, str]] = {
    13: ("-x", "+y"),
    14: ("-x", "+y"),
    15: ("-x", "+y"),
}
WORLD_AXIS_VECTORS: Dict[str, Vec3] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

# Robot geometry from player/craft/robot.urdf.
ROLL_JOINT_OFFSET_M = np.array([0.0, 0.0, 0.128], dtype=np.float64)
NODE_CHAIN_SEGMENTS_M: tuple[float, ...] = (0.028,) + (0.05,) * 9

# Node9 shape inferred from node_end obj and the user's face layout notes.
# X columns are spaced by 20 mm with the second marker centered on each face.
MARKER_COLUMN_X_M: tuple[float, float, float] = (0.005, 0.025, 0.045)
MARKER_GROUP_SPACING_M = 0.020
NODE_END_TOP_Z_M = 0.035
NODE_END_BOTTOM_Z_M = -0.035
NODE_END_LEFT_Y_M = 0.044
NODE_END_RIGHT_Y_M = -0.044


@dataclass(frozen=True)
class Pose:
    p: Vec3
    q: Quat


@dataclass(frozen=True)
class RefineResult:
    output_path: Path
    rows: int


def model_pose_fields(marker_ids: Iterable[int] = MARKER_IDS) -> list[str]:
    fields: list[str] = []
    for marker_id in marker_ids:
        fields.extend([f"mp{marker_id}{axis}" for axis in AXES])
        fields.extend([f"mq{marker_id}{axis}" for axis in Q_AXES])
    return fields


def output_fields(input_fields: Sequence[str], marker_ids: Iterable[int] = MARKER_IDS) -> list[str]:
    fields = list(input_fields)
    seen = set(fields)
    for field in model_pose_fields(marker_ids):
        if field not in seen:
            fields.append(field)
            seen.add(field)
    return fields


def base_owner(marker_id: int) -> int:
    for base_id, children in ARUCO_GROUPS_BY_BASE.items():
        if int(marker_id) in children:
            return int(base_id)
    raise KeyError(f"unsupported marker id: {marker_id}")


def _world_axis(name: str) -> np.ndarray:
    return np.asarray(WORLD_AXIS_VECTORS[name.lower()], dtype=np.float64)


def _base_marker_rotation(base_id: int) -> np.ndarray:
    x_axis_name, z_axis_name = BASE_MARKER_AXES_WORLD[int(base_id)]
    x_axis = _world_axis(x_axis_name)
    z_axis = _world_axis(z_axis_name)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def fixed_base_world_pose(base_id: int) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _base_marker_rotation(base_id)
    T[:3, 3] = np.asarray(BASE_OFFSETS_M[int(base_id)], dtype=np.float64)
    return T


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _transform(translation: Sequence[float], rotation: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def _invert_pose(T: np.ndarray) -> np.ndarray:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _quat_xyzw_from_matrix(rot: np.ndarray) -> Quat:
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return (float(qx), float(qy), float(qz), float(qw))


def _map_control_to_axis(u_value: float, direction: int, q_min: float, q_max: float) -> float:
    u = float(min(max(float(u_value), 0.0), 360.0))
    if int(direction) < 0:
        u = 360.0 - u
    ratio = u / 360.0
    return float(q_min) + ratio * (float(q_max) - float(q_min))


def _float_or_none(raw: object) -> float | None:
    try:
        value = float(str(raw).strip())
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _marker_rotation_from_axes(x_axis: Vec3, z_axis: Vec3) -> np.ndarray:
    x = np.asarray(x_axis, dtype=np.float64)
    x /= max(float(np.linalg.norm(x)), 1e-12)
    z = np.asarray(z_axis, dtype=np.float64)
    z /= max(float(np.linalg.norm(z)), 1e-12)
    y = np.cross(z, x)
    y /= max(float(np.linalg.norm(y)), 1e-12)
    x = np.cross(y, z)
    x /= max(float(np.linalg.norm(x)), 1e-12)
    return np.column_stack([x, y, z])


def _rotation_about_axis(axis: Vec3, angle_rad: float) -> np.ndarray:
    axis_v = np.asarray(axis, dtype=np.float64)
    axis_v /= max(float(np.linalg.norm(axis_v)), 1e-12)
    x, y, z = axis_v.tolist()
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


MARKER_GROUP_CENTER_SPECS: Dict[int, tuple[Vec3, Vec3, Vec3]] = {
    2: ((MARKER_COLUMN_X_M[1], 0.0, NODE_END_TOP_Z_M), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    5: ((MARKER_COLUMN_X_M[1], NODE_END_RIGHT_Y_M, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    8: ((MARKER_COLUMN_X_M[1], NODE_END_LEFT_Y_M, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    11: ((MARKER_COLUMN_X_M[1], 0.0, NODE_END_BOTTOM_Z_M), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
}
MARKER_GROUP_MEMBERS: Dict[int, tuple[int, int, int]] = {
    2: (1, 2, 3),
    5: (4, 5, 6),
    8: (7, 8, 9),
    11: (10, 11, 12),
}


def _marker_local_transforms() -> Dict[int, np.ndarray]:
    transforms: Dict[int, np.ndarray] = {}
    for center_id, (center_pos, x_axis, z_axis) in MARKER_GROUP_CENTER_SPECS.items():
        # Rotate the whole 3-marker face group counter-clockwise in-plane around
        # the center marker, instead of rotating each square independently.
        group_spin = _rotation_about_axis(z_axis, math.pi / 2.0)
        row_axis = group_spin @ np.asarray(x_axis, dtype=np.float64)
        row_axis /= max(float(np.linalg.norm(row_axis)), 1e-12)
        group_rot = _marker_rotation_from_axes(tuple(row_axis.tolist()), z_axis)
        left_id, mid_id, right_id = MARKER_GROUP_MEMBERS[int(center_id)]
        center = np.asarray(center_pos, dtype=np.float64)
        # The face-group ordering in the real hardware is mirrored relative to
        # the original model assumption. Place the lower-numbered marker on the
        # opposite side so rows read 1-2-3 / 4-5-6 / 7-8-9 / 10-11-12 correctly.
        transforms[int(left_id)] = _transform(tuple((center + row_axis * MARKER_GROUP_SPACING_M).tolist()), group_rot)
        transforms[int(mid_id)] = _transform(tuple(center.tolist()), group_rot)
        transforms[int(right_id)] = _transform(tuple((center - row_axis * MARKER_GROUP_SPACING_M).tolist()), group_rot)
    return transforms


MARKER_LOCAL_TRANSFORMS = _marker_local_transforms()


def predicted_marker_poses(roll_deg: float, seg1_deg: float, seg2_deg: float) -> Dict[int, Pose]:
    roll_rad = _map_control_to_axis(float(roll_deg), COMMAND_DIRECTION[1], *ROLL_RANGE_RAD)
    seg1_rad = _map_control_to_axis(float(seg1_deg), COMMAND_DIRECTION[2], *BEND_RANGE_RAD)
    seg2_rad = _map_control_to_axis(float(seg2_deg), COMMAND_DIRECTION[3], *BEND_RANGE_RAD)
    bend_angles = [seg1_rad] * 5 + [seg2_rad] * 5

    T_world_node = _transform(ROLL_JOINT_OFFSET_M, _rotation_x(roll_rad))
    for length_m, bend_rad in zip(NODE_CHAIN_SEGMENTS_M, bend_angles):
        T_world_node = T_world_node @ _transform((length_m, 0.0, 0.0), _rotation_y(bend_rad))

    poses: Dict[int, Pose] = {}
    for marker_id, T_node_marker in MARKER_LOCAL_TRANSFORMS.items():
        T_world_marker = T_world_node @ T_node_marker
        T_world_base = fixed_base_world_pose(base_owner(marker_id))
        T_base_marker = _invert_pose(T_world_base) @ T_world_marker
        poses[int(marker_id)] = Pose(
            p=tuple(float(v) for v in T_base_marker[:3, 3]),
            q=_quat_xyzw_from_matrix(T_base_marker[:3, :3]),
        )
    return poses


def _write_model_pose(row: dict[str, str], marker_id: int, pose: Pose) -> None:
    row[f"mp{marker_id}x"] = f"{pose.p[0]:.6f}"
    row[f"mp{marker_id}y"] = f"{pose.p[1]:.6f}"
    row[f"mp{marker_id}z"] = f"{pose.p[2]:.6f}"
    row[f"mq{marker_id}x"] = f"{pose.q[0]:.8f}"
    row[f"mq{marker_id}y"] = f"{pose.q[1]:.8f}"
    row[f"mq{marker_id}z"] = f"{pose.q[2]:.8f}"
    row[f"mq{marker_id}w"] = f"{pose.q[3]:.8f}"


def refine_csv_file(input_csv: Path, output_csv: Path) -> RefineResult:
    input_csv = input_csv.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_out: List[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = list(reader.fieldnames or [])
        required = {"roll", "seg1", "seg2"}
        missing = sorted(field for field in required if field not in input_fields)
        if missing:
            raise RuntimeError(f"{input_csv.name} is missing required columns: {', '.join(missing)}")

        for source_row in reader:
            row = dict(source_row)
            roll = _float_or_none(row.get("roll"))
            seg1 = _float_or_none(row.get("seg1"))
            seg2 = _float_or_none(row.get("seg2"))
            if None not in (roll, seg1, seg2):
                poses = predicted_marker_poses(float(roll), float(seg1), float(seg2))
                for marker_id, pose in poses.items():
                    _write_model_pose(row, marker_id, pose)
            else:
                for field in model_pose_fields():
                    row[field] = ""
            rows_out.append(row)

    fields = output_fields(input_fields)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    return RefineResult(output_path=output_csv, rows=len(rows_out))
