from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # x, y, z, w


@dataclass(frozen=True)
class Pose:
    p: Vec3
    q: Quat


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

# Each entry is (marker local +X in world, marker local +Z normal in world).
# The +Z normal is the main measurement direction, but +X is also required to
# remove the rotation ambiguity around that normal when reconstructing poses.
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


def base_owner(marker_id: int) -> int | None:
    for base_id, children in ARUCO_GROUPS_BY_BASE.items():
        if int(marker_id) in children:
            return int(base_id)
    return int(marker_id) if int(marker_id) in BASE_OFFSETS_M else None


def quat_from_matrix(rot: np.ndarray) -> Quat:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    return (float(qx), float(qy), float(qz), float(qw))


def pose_to_matrix(pose: Pose) -> np.ndarray:
    x, y, z, w = pose.q
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = np.asarray(pose.p, dtype=np.float64)
    return T


def matrix_to_pose(T: np.ndarray) -> Pose:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return Pose(
        p=(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])),
        q=quat_from_matrix(T[:3, :3]),
    )


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def pose_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> Pose:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return Pose(
        p=tuple(float(v) for v in np.asarray(tvec, dtype=np.float64).reshape(3)),
        q=quat_from_matrix(rot),
    )


def direction_vector_from_pose(pose: Pose) -> Vec3:
    T = pose_to_matrix(pose)
    z_axis = T[:3, 2]
    return (float(z_axis[0]), float(z_axis[1]), float(z_axis[2]))


def _world_axis(name: str) -> np.ndarray:
    try:
        return np.asarray(WORLD_AXIS_VECTORS[name.lower()], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"unsupported world axis: {name}") from exc


def _base_marker_rotation(base_id: int) -> np.ndarray:
    try:
        x_axis_name, z_axis_name = BASE_MARKER_AXES_WORLD[int(base_id)]
    except KeyError as exc:
        raise ValueError(f"missing base marker orientation: {base_id}") from exc

    x_axis = _world_axis(x_axis_name)
    z_axis = _world_axis(z_axis_name)
    if abs(float(np.dot(x_axis, z_axis))) > 1e-9:
        raise ValueError(f"base marker axes must be perpendicular: {base_id}")
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def fixed_base_world_pose(base_id: int) -> np.ndarray:
    offset = np.asarray(BASE_OFFSETS_M[int(base_id)], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _base_marker_rotation(base_id)
    T[:3, 3] = offset
    return T


def reconstruct_base_relative_pose(marker_pose_camera: Pose, base_pose_camera: Pose) -> Pose:
    T_cam_marker = pose_to_matrix(marker_pose_camera)
    T_cam_base = pose_to_matrix(base_pose_camera)
    T_base_cam = invert_pose(T_cam_base)
    T_base_marker = T_base_cam @ T_cam_marker
    return matrix_to_pose(T_base_marker)


def reconstruct_world_pose(marker_id: int, marker_pose_camera: Pose, base_pose_camera: Pose) -> Pose:
    T_world_base = fixed_base_world_pose(base_owner(marker_id) or 13)
    T_base_marker = pose_to_matrix(reconstruct_base_relative_pose(marker_pose_camera, base_pose_camera))
    T_world_marker = T_world_base @ T_base_marker
    return matrix_to_pose(T_world_marker)
