from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError as exc:  # pragma: no cover - environment dependent
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    TK_IMPORT_ERROR = exc
else:
    TK_IMPORT_ERROR = None

try:
    import genesis as gs
except ImportError as exc:  # pragma: no cover - runtime dependency
    gs = None  # type: ignore[assignment]
    GENESIS_IMPORT_ERROR = exc
else:
    GENESIS_IMPORT_ERROR = None


Vec3 = Tuple[float, float, float]

ARUCO_MARKER_SIZE_M = 0.017
ARUCO_NORMAL_STICK_M = 0.010
ARUCO_MARKER_MESH_PATH = Path(__file__).resolve().parent / "assets" / "aruco_marker" / "marker_square.obj"
ARUCO_MARKER_IDS = tuple(range(1, 16))
ARUCO_BASE_IDS = (13, 14, 15)
ARUCO_NODE_MARKER_IDS = tuple(marker_id for marker_id in ARUCO_MARKER_IDS if marker_id not in ARUCO_BASE_IDS)
ARUCO_GROUPS_BY_BASE: Dict[int, tuple[int, ...]] = {
    13: (1, 4, 7, 10),
    14: (2, 5, 8, 11),
    15: (3, 6, 9, 12),
}
ARUCO_GROUP_COLOR: Dict[int, tuple[float, float, float, float]] = {
    13: (0.95, 0.35, 0.30, 0.95),
    14: (0.20, 0.70, 0.35, 0.95),
    15: (0.20, 0.45, 1.00, 0.95),
}
IDEAL_MARKER_GROUP_COLOR: Dict[int, tuple[float, float, float, float]] = {
    13: (0.78, 0.56, 0.54, 0.95),
    14: (0.55, 0.70, 0.58, 0.95),
    15: (0.56, 0.64, 0.82, 0.95),
}
IDEAL_MARKER_COLOR_DEFAULT = (0.60, 0.60, 0.60, 0.95)
IDEAL_MARKER_COLOR_BY_ID: Dict[int, tuple[float, float, float, float]] = {}
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
COMMAND_DIRECTION = (-1, -1, 1, -1)
LINEAR_RANGE_M = (-0.23, 0.01)
ROLL_RANGE_RAD = (-math.pi / 2.0, math.pi / 2.0)
BEND_RANGE_RAD = (-math.radians(36.0), math.radians(36.0))
WORLD_AXIS_VECTORS: Dict[str, Vec3] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
HIDE_POS = np.array([0.0, 0.0, -10.0], dtype=float)
ROBOT_SPAWN_POS = np.array([0.0, 0.0, 1.0], dtype=float)


@dataclass(frozen=True)
class PlaybackFrame:
    t: float
    roll: Optional[float]
    seg1: Optional[float]
    seg2: Optional[float]
    markers: Dict[int, tuple[np.ndarray, np.ndarray]]
    model_markers: Dict[int, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class PlaybackData:
    frames: List[PlaybackFrame]
    marker_ids: tuple[int, ...]


def default_urdf_path() -> Path:
    return (Path(__file__).resolve().parent / "craft" / "robot.urdf").resolve()


def _float_or_none(raw: object) -> Optional[float]:
    try:
        value = float(str(raw).strip())
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _normalize(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(vv))
    if n <= 1e-12:
        raise ValueError("zero-length vector")
    return vv / n


def _rotation_from_z(z_axis: np.ndarray, x_hint: Optional[np.ndarray] = None) -> np.ndarray:
    z_axis = _normalize(z_axis)
    x_seed = np.array([1.0, 0.0, 0.0], dtype=float) if x_hint is None else np.asarray(x_hint, dtype=float).reshape(3)
    x_proj = x_seed - z_axis * float(np.dot(x_seed, z_axis))
    if float(np.linalg.norm(x_proj)) <= 1e-9:
        fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        x_proj = fallback - z_axis * float(np.dot(fallback, z_axis))
    x_axis = _normalize(x_proj)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _quat_wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=float).reshape(3, 3)
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
    return np.array([qw, qx, qy, qz], dtype=float)


def _axis(name: str) -> np.ndarray:
    return np.asarray(WORLD_AXIS_VECTORS[name.lower()], dtype=float)


def _base_marker_rotation(base_id: int) -> np.ndarray:
    x_name, z_name = BASE_MARKER_AXES_WORLD[int(base_id)]
    x_axis = _axis(x_name)
    z_axis = _axis(z_name)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def fixed_base_markers() -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    markers: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for base_id in ARUCO_BASE_IDS:
        markers[int(base_id)] = (
            ROBOT_SPAWN_POS + np.asarray(BASE_OFFSETS_M[int(base_id)], dtype=float),
            _base_marker_rotation(int(base_id)),
        )
    return markers


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=float).reshape(4)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("zero-length quaternion")
    x, y, z, w = (q / norm).tolist()
    return _quat_wxyz_to_matrix(np.array([w, x, y, z], dtype=float))


def _to_numpy_1d(raw: object) -> np.ndarray:
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    return np.asarray(raw, dtype=float).reshape(-1)


def _map_control_to_axis(u_value: float, direction: int, q_min: float, q_max: float) -> float:
    u = float(min(max(float(u_value), 0.0), 360.0))
    if int(direction) < 0:
        u = 360.0 - u
    ratio = u / 360.0
    return float(q_min) + ratio * (float(q_max) - float(q_min))


def _compose_transform(
    a_pos: np.ndarray,
    a_rot: np.ndarray,
    b_pos: np.ndarray,
    b_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ar = np.asarray(a_rot, dtype=float).reshape(3, 3)
    ap = np.asarray(a_pos, dtype=float).reshape(3)
    br = np.asarray(b_rot, dtype=float).reshape(3, 3)
    bp = np.asarray(b_pos, dtype=float).reshape(3)
    return (ap + ar @ bp, ar @ br)


def marker_group(marker_id: int) -> int:
    for base_id, children in ARUCO_GROUPS_BY_BASE.items():
        if int(marker_id) == int(base_id) or int(marker_id) in children:
            return int(base_id)
    return 15


def load_playback_csv(path: Path) -> PlaybackData:
    frames: List[PlaybackFrame] = []
    seen_marker_ids: set[int] = set()
    with path.expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            t = _float_or_none(row.get("t"))
            if t is None:
                t = _float_or_none(row.get("t_aruco"))
            if t is None:
                t = _float_or_none(row.get("t_recorder"))
            if t is None:
                t = float(index) / 30.0
            markers: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
            model_markers: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for marker_id in ARUCO_MARKER_IDS:
                px = _float_or_none(row.get(f"p{marker_id}x"))
                py = _float_or_none(row.get(f"p{marker_id}y"))
                pz = _float_or_none(row.get(f"p{marker_id}z"))
                mx = _float_or_none(row.get(f"mp{marker_id}x"))
                my = _float_or_none(row.get(f"mp{marker_id}y"))
                mz = _float_or_none(row.get(f"mp{marker_id}z"))
                mqx = _float_or_none(row.get(f"mq{marker_id}x"))
                mqy = _float_or_none(row.get(f"mq{marker_id}y"))
                mqz = _float_or_none(row.get(f"mq{marker_id}z"))
                mqw = _float_or_none(row.get(f"mq{marker_id}w"))
                if None not in (mx, my, mz, mqx, mqy, mqz, mqw):
                    try:
                        model_rot = _quat_xyzw_to_matrix(np.array([mqx, mqy, mqz, mqw], dtype=float))
                    except ValueError:
                        model_rot = None
                    if model_rot is not None and np.all(np.isfinite(model_rot)):
                        model_markers[int(marker_id)] = (np.array([mx, my, mz], dtype=float), model_rot)
                if None in (px, py, pz):
                    continue

                qx = _float_or_none(row.get(f"q{marker_id}x"))
                qy = _float_or_none(row.get(f"q{marker_id}y"))
                qz = _float_or_none(row.get(f"q{marker_id}z"))
                qw = _float_or_none(row.get(f"q{marker_id}w"))
                if None not in (qx, qy, qz, qw):
                    try:
                        rot = _quat_xyzw_to_matrix(np.array([qx, qy, qz, qw], dtype=float))
                    except ValueError:
                        continue
                else:
                    vx = _float_or_none(row.get(f"v{marker_id}x"))
                    vy = _float_or_none(row.get(f"v{marker_id}y"))
                    vz = _float_or_none(row.get(f"v{marker_id}z"))
                    if None in (vx, vy, vz):
                        continue
                    normal = np.array([vx, vy, vz], dtype=float)
                    try:
                        rot = _rotation_from_z(normal)
                    except ValueError:
                        continue
                if not np.all(np.isfinite(rot)):
                    continue
                markers[int(marker_id)] = (np.array([px, py, pz], dtype=float), rot)
                seen_marker_ids.add(int(marker_id))
            frames.append(
                PlaybackFrame(
                    t=float(t),
                    roll=_float_or_none(row.get("roll")),
                    seg1=_float_or_none(row.get("seg1")),
                    seg2=_float_or_none(row.get("seg2")),
                    markers=markers,
                    model_markers=model_markers,
                )
            )
    return PlaybackData(frames=frames, marker_ids=tuple(sorted(seen_marker_ids)))


class GenesisPlayer:
    def __init__(
        self,
        urdf_path: Path,
        *,
        use_gpu: bool = False,
        status_queue: object | None = None,
        command_queue: object | None = None,
    ) -> None:
        self.urdf_path = urdf_path.expanduser().resolve()
        self.use_gpu = bool(use_gpu)
        self.frames: List[PlaybackFrame] = []
        self.marker_ids: tuple[int, ...] = ()
        self.frame_index = 0
        self.playing = False
        self._last_robot_frame_index: Optional[int] = None
        self.status_queue = status_queue if status_queue is not None else queue.Queue()
        self.command_queue = command_queue
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._scene = None
        self._robot = None
        self._marker_plate_entities: Dict[int, object] = {}
        self._marker_debug_objects: Dict[int, List[object]] = {}
        self._model_marker_plate_entities: Dict[int, object] = {}
        self._model_marker_debug_objects: Dict[int, List[object]] = {}
        self._joint_names: List[str] = []
        self._dofs_idx_local: List[int] = []
        self._last_wall = time.monotonic()
        self._last_applied_frame_index: Optional[int] = None
        self._last_applied_signature: tuple[int, int] | None = None
        self._last_reported_state: tuple[int, int, bool] | None = None

    def _log(self, message: str) -> None:
        text = f"[Player] {message}"
        print(text, flush=True)
        self.status_queue.put(message)

    def _emit_playback_state(self) -> None:
        with self._lock:
            total = len(self.frames)
            index = min(max(int(self.frame_index), 0), max(0, total - 1)) if total else 0
            playing = bool(self.playing)
            t_sec = float(self.frames[index].t) if total else 0.0
            duration_sec = float(self.frames[-1].t) if total else 0.0
        signature = (index, total, playing)
        if self._last_reported_state == signature:
            return
        self._last_reported_state = signature
        self.status_queue.put(("playback_state", index, total, t_sec, duration_sec, playing))

    def set_playback_data(self, data: PlaybackData) -> None:
        with self._lock:
            self.frames = list(data.frames)
            self.marker_ids = tuple(data.marker_ids)
            self.frame_index = 0
            self.playing = False
            self._last_robot_frame_index = None
            self._last_applied_frame_index = None
            self._last_applied_signature = None
            self._last_reported_state = None
        marker_frames = sum(1 for frame in data.frames if frame.markers)
        model_marker_frames = sum(1 for frame in data.frames if frame.model_markers)
        robot_frames = sum(1 for frame in data.frames if frame.roll is not None or frame.seg1 is not None or frame.seg2 is not None)
        self._log(
            f"Loaded {len(data.frames)} frames; robot intent frames={robot_frames}; "
            f"marker actual frames={marker_frames}; marker ideal frames={model_marker_frames}; "
            f"marker ids={list(data.marker_ids)}"
        )
        self._emit_playback_state()

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self.playing = bool(playing)
            self._last_wall = time.monotonic()
        self._emit_playback_state()

    def seek_frame(self, frame_index: int) -> None:
        with self._lock:
            if not self.frames:
                self.frame_index = 0
                self.playing = False
            else:
                self.frame_index = max(0, min(int(frame_index), len(self.frames) - 1))
            self._last_wall = time.monotonic()
            self._last_robot_frame_index = None
            self._last_applied_signature = None
        self._emit_playback_state()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self._log("Genesis thread starting")
        if GENESIS_IMPORT_ERROR is not None:
            self._log(f"Genesis import failed: {GENESIS_IMPORT_ERROR}")
            return
        if not self.urdf_path.is_file():
            self._log(f"URDF not found: {self.urdf_path}")
            return
        try:
            self._init_scene()
            self._log("Genesis ready; entering playback loop")
            while not self._stop.is_set():
                self._drain_commands()
                self._tick()
        except Exception as exc:
            self._log(f"Genesis error: {exc}")

    def _drain_commands(self) -> None:
        if self.command_queue is None:
            return
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return
            except Exception:
                return
            if not isinstance(command, tuple) or not command:
                continue
            name = str(command[0])
            if name == "stop":
                self.stop()
                return
            if name == "playing" and len(command) >= 2:
                self.set_playing(bool(command[1]))
                self._log("Playback playing" if bool(command[1]) else "Playback paused")
                continue
            if name == "seek" and len(command) >= 2:
                try:
                    self.seek_frame(int(command[1]))
                except Exception:
                    pass
                continue
            if name == "frames" and len(command) >= 2:
                data = command[1]
                if isinstance(data, PlaybackData):
                    self.set_playback_data(data)
                elif isinstance(data, list):
                    self.set_playback_data(PlaybackData(frames=data, marker_ids=()))
                continue

    def _init_scene(self) -> None:
        assert gs is not None
        self._log(f"Preparing Genesis URDF from {self.urdf_path}")
        urdf_path = self._prepare_genesis_urdf()
        self._log(f"Using Genesis URDF {urdf_path}")
        os.chdir(urdf_path.parent)
        self._log(f"Changed working directory to {urdf_path.parent}")
        backend = gs.gpu if self.use_gpu else gs.cpu
        self._log(f"Initializing Genesis backend={'gpu' if self.use_gpu else 'cpu'}")
        try:
            gs.init(backend=backend, logging_level="warning")
        except TypeError:
            gs.init(backend=backend)
        self._log("Genesis initialized")
        self._log("Creating scene")
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / 240.0, gravity=(0.0, 0.0, 0.0)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.85, -0.85, 1.55),
                camera_lookat=(-0.05, 0.00, 1.05),
                camera_fov=35,
                max_FPS=60,
            ),
            show_viewer=True,
        )
        self._log("Adding floor")
        self._scene.add_entity(gs.morphs.Plane())
        morph_kwargs = {
            "file": str(urdf_path),
            "pos": tuple(float(v) for v in ROBOT_SPAWN_POS),
            "euler": (0.0, 0.0, 0.0),
            "fixed": True,
            "prioritize_urdf_material": True,
            "merge_fixed_links": False,
        }
        self._log("Creating URDF morph")
        try:
            morph = gs.morphs.URDF(**morph_kwargs, requires_jac_and_IK=False)
        except TypeError:
            morph = gs.morphs.URDF(**morph_kwargs)
        self._log("Adding robot entity")
        self._robot = self._scene.add_entity(morph)
        self._joint_names = self._read_urdf_joint_names(urdf_path)
        self._log(f"Read {len(self._joint_names)} movable joints from URDF")
        self._log("Adding ArUco marker plate entities")
        self._init_marker_plate_entities()
        self._log("Scene build start")
        t0 = time.monotonic()
        self._scene.build()
        self._log(f"Scene build done in {time.monotonic() - t0:.2f}s")
        self._log("Initializing robot DOF handles")
        self._init_robot_dofs()
        self._log(f"Initialized {len(self._dofs_idx_local)} robot DOFs")
        self._log("Applying initial frame")
        self._apply_frame(None)

    def _init_marker_plate_entities(self) -> None:
        if self._scene is None:
            return
        self._marker_plate_entities = {}
        self._model_marker_plate_entities = {}
        for marker_id in ARUCO_MARKER_IDS:
            color = ARUCO_GROUP_COLOR[marker_group(int(marker_id))]
            try:
                self._marker_plate_entities[int(marker_id)] = self._make_marker_plate_entity(color)
            except Exception as exc:
                self._log(f"Failed to create marker {marker_id} plate: {exc}")
            if int(marker_id) in ARUCO_BASE_IDS:
                continue
            try:
                self._model_marker_plate_entities[int(marker_id)] = self._make_marker_plate_entity(
                    IDEAL_MARKER_COLOR_BY_ID.get(
                        int(marker_id),
                        IDEAL_MARKER_GROUP_COLOR.get(marker_group(int(marker_id)), IDEAL_MARKER_COLOR_DEFAULT),
                    )
                )
            except Exception as exc:
                self._log(f"Failed to create ideal marker {marker_id} plate: {exc}")

    def _make_marker_plate_entity(self, color: tuple[float, float, float, float]) -> object:
        assert gs is not None
        if self._scene is None:
            raise RuntimeError("scene is not initialized")

        color3 = tuple(float(c) for c in color[:3])
        surface = None
        try:
            surface = gs.surfaces.Plastic(color=color3)
        except Exception:
            pass

        morph = None
        last_exc: Exception | None = None
        # Prefer a thin box over the flat marker mesh. The mesh is effectively
        # planar and can trigger trimesh volume/center-of-mass warnings.
        for kwargs in (
            {"pos": tuple(float(v) for v in HIDE_POS), "size": (ARUCO_MARKER_SIZE_M, ARUCO_MARKER_SIZE_M, 0.0006), "fixed": True},
            {"pos": tuple(float(v) for v in HIDE_POS), "extents": (ARUCO_MARKER_SIZE_M, ARUCO_MARKER_SIZE_M, 0.0006), "fixed": True},
            {"pos": tuple(float(v) for v in HIDE_POS), "size": (ARUCO_MARKER_SIZE_M, ARUCO_MARKER_SIZE_M, 0.0006)},
            {"pos": tuple(float(v) for v in HIDE_POS), "extents": (ARUCO_MARKER_SIZE_M, ARUCO_MARKER_SIZE_M, 0.0006)},
        ):
            try:
                morph = gs.morphs.Box(**kwargs)
                break
            except Exception as exc:
                last_exc = exc

        if morph is None and ARUCO_MARKER_MESH_PATH.is_file() and hasattr(gs.morphs, "Mesh"):
            for kwargs in (
                {"file": str(ARUCO_MARKER_MESH_PATH), "pos": tuple(float(v) for v in HIDE_POS), "fixed": True},
                {"file": str(ARUCO_MARKER_MESH_PATH), "pos": tuple(float(v) for v in HIDE_POS)},
                {"filename": str(ARUCO_MARKER_MESH_PATH), "pos": tuple(float(v) for v in HIDE_POS), "fixed": True},
                {"filename": str(ARUCO_MARKER_MESH_PATH), "pos": tuple(float(v) for v in HIDE_POS)},
            ):
                try:
                    morph = gs.morphs.Mesh(**kwargs)
                    break
                except Exception as exc:
                    last_exc = exc

        if morph is None:
            raise RuntimeError(f"failed to create marker plate morph: {last_exc}")

        for kwargs in (
            {"morph": morph, "surface": surface} if surface is not None else {"morph": morph},
            {"morph": morph},
        ):
            try:
                return self._scene.add_entity(**kwargs)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"failed to add marker plate entity: {last_exc}")

    def _prepare_genesis_urdf(self) -> Path:
        import xml.etree.ElementTree as ET

        source = self.urdf_path
        target = source.with_name(f"{source.stem}.genesis.urdf")
        self._log("Sanitizing URDF for Genesis visual-only load")
        tree = ET.parse(source)
        root = tree.getroot()
        removed_mujoco = 0
        removed_collisions = 0
        relaxed_limits = 0
        removed_dynamics = 0
        for child in list(root):
            if child.tag == "mujoco":
                root.remove(child)
                removed_mujoco += 1
        for link in root.findall("link"):
            for collision in list(link.findall("collision")):
                link.remove(collision)
                removed_collisions += 1
        for joint in root.findall("joint"):
            joint_type = str(joint.attrib.get("type", "")).lower()
            if joint_type == "prismatic":
                limit = joint.find("limit")
                if limit is None:
                    limit = ET.SubElement(joint, "limit")
                limit.set("lower", "-1.0")
                limit.set("upper", "1.0")
                limit.set("effort", "1000000")
                limit.set("velocity", "1000000")
                relaxed_limits += 1
            elif joint_type in {"revolute", "continuous"}:
                limit = joint.find("limit")
                if limit is None:
                    limit = ET.SubElement(joint, "limit")
                limit.set("lower", f"{-math.pi:.12g}")
                limit.set("upper", f"{math.pi:.12g}")
                limit.set("effort", "1000000")
                limit.set("velocity", "1000000")
                relaxed_limits += 1
            for dynamics in list(joint.findall("dynamics")):
                joint.remove(dynamics)
                removed_dynamics += 1
        tree.write(target, encoding="utf-8", xml_declaration=True)
        self._log(
            f"Sanitized URDF wrote {target.name}; removed mujoco={removed_mujoco}, "
            f"collisions={removed_collisions}, relaxed_limits={relaxed_limits}, removed_dynamics={removed_dynamics}"
        )
        return target

    def _read_urdf_joint_names(self, path: Path) -> List[str]:
        import xml.etree.ElementTree as ET

        try:
            root = ET.parse(path).getroot()
        except Exception:
            return []
        names: List[str] = []
        for joint in root.findall("joint"):
            joint_type = str(joint.attrib.get("type", "")).lower()
            name = str(joint.attrib.get("name", "")).strip()
            if name and joint_type in {"prismatic", "revolute", "continuous"}:
                names.append(name)
        return names

    def _init_robot_dofs(self) -> None:
        if self._robot is None:
            return
        pairs: List[tuple[int, str]] = []
        for name in self._joint_names:
            try:
                joint = self._robot.get_joint(name)
                raw_idx = getattr(joint, "dofs_idx_local")
                if isinstance(raw_idx, (list, tuple, np.ndarray)):
                    idx = int(np.asarray(raw_idx).reshape(-1)[0])
                else:
                    idx = int(raw_idx)
                pairs.append((idx, name))
            except Exception:
                continue
        pairs.sort(key=lambda item: item[0])
        self._dofs_idx_local = [idx for idx, _name in pairs]
        self._joint_names = [name for _idx, name in pairs]

    def _tick(self) -> None:
        frame_index, frame = self._next_frame()
        should_apply = frame is None
        if frame is not None:
            signature = (frame_index, len(frame.markers), len(frame.model_markers))
            should_apply = self._last_applied_signature != signature
            self._last_applied_signature = signature
        if should_apply:
            self._apply_frame(frame)
        self._emit_playback_state()
        if self._scene is not None:
            self._scene.step()
        time.sleep(1.0 / 60.0)

    def _next_frame(self) -> tuple[int, Optional[PlaybackFrame]]:
        with self._lock:
            if not self.frames:
                return (-1, None)
            if not self.playing:
                index = min(self.frame_index, len(self.frames) - 1)
                return (index, self.frames[index])
            now = time.monotonic()
            current = self.frames[min(self.frame_index, len(self.frames) - 1)]
            next_index = min(self.frame_index + 1, len(self.frames) - 1)
            next_frame = self.frames[next_index]
            dt = max(0.0, float(next_frame.t) - float(current.t))
            if now - self._last_wall >= dt:
                self.frame_index = next_index
                self._last_wall = now
                if self.frame_index >= len(self.frames) - 1:
                    self.playing = False
            index = min(self.frame_index, len(self.frames) - 1)
            return (index, self.frames[index])

    def _apply_frame(self, frame: Optional[PlaybackFrame]) -> None:
        if frame is None and self._last_applied_frame_index == -1:
            return
        if frame is not None:
            self._apply_robot_pose(frame)
        markers = self._base_markers_from_housing()
        model_markers: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if frame is not None:
            markers.update(self._align_actual_markers_to_housing_bases(frame.markers, markers))
            model_markers = self._align_relative_markers_to_housing_bases(frame.model_markers, markers)
        self._apply_markers(markers, model_markers)
        self._last_applied_frame_index = -1 if frame is None else self.frame_index

    def _align_relative_markers_to_housing_bases(
        self,
        relative_markers: Dict[int, tuple[np.ndarray, np.ndarray]],
        housing_bases: Dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
        aligned: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for marker_id, marker_pose in relative_markers.items():
            marker_int = int(marker_id)
            if marker_int in ARUCO_BASE_IDS:
                continue
            base_id = marker_group(marker_int)
            housing_base = housing_bases.get(base_id)
            if housing_base is None:
                aligned[marker_int] = (
                    ROBOT_SPAWN_POS + np.asarray(marker_pose[0], dtype=float).reshape(3),
                    marker_pose[1],
                )
                continue

            aligned[marker_int] = _compose_transform(housing_base[0], housing_base[1], marker_pose[0], marker_pose[1])
        return aligned

    def _align_actual_markers_to_housing_bases(
        self,
        actual_markers: Dict[int, tuple[np.ndarray, np.ndarray]],
        housing_bases: Dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
        return self._align_relative_markers_to_housing_bases(actual_markers, housing_bases)

    def _base_markers_from_housing(self) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
        if self._robot is None:
            return fixed_base_markers()
        try:
            link = self._robot.get_link("housing")
            housing_pos = _to_numpy_1d(link.get_pos())[:3]
            housing_quat = _to_numpy_1d(link.get_quat())[:4]
            housing_rot = _quat_wxyz_to_matrix(housing_quat)
        except Exception:
            return fixed_base_markers()
        markers: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for base_id, offset in BASE_OFFSETS_M.items():
            marker_rot_housing = _base_marker_rotation(int(base_id))
            markers[int(base_id)] = (
                housing_pos + housing_rot @ np.asarray(offset, dtype=float).reshape(3),
                housing_rot @ marker_rot_housing,
            )
        return markers

    def _apply_robot_pose(self, frame: PlaybackFrame) -> None:
        if self._robot is None or not self._dofs_idx_local:
            return
        if self._last_robot_frame_index == self.frame_index:
            return
        values = self._joint_values_from_frame(frame)
        q_target = np.asarray(values, dtype=float).reshape(-1)
        try:
            self._robot.set_dofs_position(q_target, dofs_idx_local=self._dofs_idx_local)
        except Exception:
            try:
                self._robot.control_dofs_position(q_target, dofs_idx_local=self._dofs_idx_local)
            except Exception:
                return
        self._last_robot_frame_index = self.frame_index

    def _joint_values_from_frame(self, frame: PlaybackFrame) -> List[float]:
        roll_rad = 0.0 if frame.roll is None else _map_control_to_axis(float(frame.roll), COMMAND_DIRECTION[1], *ROLL_RANGE_RAD)
        seg1_rad = 0.0 if frame.seg1 is None else _map_control_to_axis(float(frame.seg1), COMMAND_DIRECTION[2], *BEND_RANGE_RAD)
        seg2_rad = 0.0 if frame.seg2 is None else _map_control_to_axis(float(frame.seg2), COMMAND_DIRECTION[3], *BEND_RANGE_RAD)
        revolute_count = max(0, len(self._joint_names) - 1)
        values = [0.0]
        if revolute_count >= 1:
            values.append(roll_rad)
        bends = max(0, revolute_count - 1)
        first = bends // 2
        second = bends - first
        values.extend([seg1_rad] * first)
        values.extend([seg2_rad] * second)
        return values[: len(self._joint_names)]

    def _apply_markers(
        self,
        markers: Dict[int, tuple[np.ndarray, np.ndarray]],
        model_markers: Dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if self._scene is None:
            return
        for marker_id in ARUCO_MARKER_IDS:
            payload = markers.get(int(marker_id))
            self._clear_marker_debug(int(marker_id))
            if payload is None:
                self._hide_marker_plate(int(marker_id))
                continue
            pos, rot = payload
            self._draw_marker_debug(int(marker_id), np.asarray(pos, dtype=float), np.asarray(rot, dtype=float))
        for marker_id in ARUCO_NODE_MARKER_IDS:
            payload = model_markers.get(int(marker_id))
            self._clear_model_marker_debug(int(marker_id))
            if payload is None:
                self._hide_model_marker_plate(int(marker_id))
                continue
            pos, rot = payload
            self._draw_model_marker_debug(int(marker_id), np.asarray(pos, dtype=float), np.asarray(rot, dtype=float))

    def _clear_model_marker_debug(self, marker_id: int) -> None:
        if self._scene is None:
            return
        objects = self._model_marker_debug_objects.pop(int(marker_id), [])
        for obj in objects:
            try:
                self._scene.clear_debug_object(obj)
            except Exception:
                pass

    def _clear_marker_debug(self, marker_id: int) -> None:
        if self._scene is None:
            return
        objects = self._marker_debug_objects.pop(int(marker_id), [])
        for obj in objects:
            try:
                self._scene.clear_debug_object(obj)
            except Exception:
                pass

    def _draw_marker_debug(self, marker_id: int, pos: np.ndarray, rot: np.ndarray) -> None:
        if self._scene is None:
            return
        center = np.asarray(pos, dtype=float).reshape(3)
        rot = np.asarray(rot, dtype=float).reshape(3, 3)
        normal = _normalize(rot[:, 2])
        color = ARUCO_GROUP_COLOR[marker_group(int(marker_id))]
        normal_color = (
            min(1.0, float(color[0]) + 0.20),
            min(1.0, float(color[1]) + 0.20),
            min(1.0, float(color[2]) + 0.20),
            float(color[3]),
        )
        self._set_marker_plate_pose(int(marker_id), center, rot)
        objects: List[object] = []
        try:
            objects.extend(
                self._draw_debug_segment(
                    center,
                    center + normal * float(ARUCO_NORMAL_STICK_M),
                    normal_color,
                    radius=0.0010,
                )
            )
            objects.append(
                self._scene.draw_debug_sphere(
                    pos=center + normal * float(ARUCO_NORMAL_STICK_M),
                    radius=0.0022,
                    color=normal_color,
                )
            )
        except Exception:
            pass
        self._marker_debug_objects[int(marker_id)] = objects

    def _draw_model_marker_debug(self, marker_id: int, pos: np.ndarray, rot: np.ndarray) -> None:
        if self._scene is None:
            return
        center = np.asarray(pos, dtype=float).reshape(3)
        rot = np.asarray(rot, dtype=float).reshape(3, 3)
        x_axis = _normalize(rot[:, 0])
        normal = _normalize(rot[:, 2])
        color = IDEAL_MARKER_COLOR_BY_ID.get(
            int(marker_id),
            IDEAL_MARKER_GROUP_COLOR.get(marker_group(int(marker_id)), IDEAL_MARKER_COLOR_DEFAULT),
        )
        normal_color = (
            min(1.0, float(color[0]) + 0.12),
            min(1.0, float(color[1]) + 0.12),
            min(1.0, float(color[2]) + 0.12),
            float(color[3]),
        )
        self._set_model_marker_plate_pose(int(marker_id), center, rot)
        objects: List[object] = []
        try:
            objects.extend(
                self._draw_debug_segment(
                    center,
                    center + normal * float(ARUCO_NORMAL_STICK_M) * 0.75,
                    normal_color,
                    radius=0.0008,
                )
            )
            objects.append(
                self._scene.draw_debug_sphere(
                    pos=center + normal * float(ARUCO_NORMAL_STICK_M) * 0.75,
                    radius=0.0018,
                    color=color,
                )
            )
            objects.extend(
                self._draw_debug_segment(
                    center,
                    center + x_axis * float(ARUCO_MARKER_SIZE_M) * 0.6,
                    color,
                    radius=0.0008,
                )
            )
        except Exception:
            pass
        self._model_marker_debug_objects[int(marker_id)] = objects

    def _hide_marker_plate(self, marker_id: int) -> None:
        entity = self._marker_plate_entities.get(int(marker_id))
        if entity is None:
            return
        self._set_entity_pose(entity, HIDE_POS, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    def _hide_model_marker_plate(self, marker_id: int) -> None:
        entity = self._model_marker_plate_entities.get(int(marker_id))
        if entity is None:
            return
        self._set_entity_pose(entity, HIDE_POS, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    def _set_marker_plate_pose(self, marker_id: int, pos: np.ndarray, rot: np.ndarray) -> None:
        entity = self._marker_plate_entities.get(int(marker_id))
        if entity is None:
            return
        self._set_entity_pose(entity, np.asarray(pos, dtype=float).reshape(3), _quat_wxyz_from_matrix(rot))

    def _set_model_marker_plate_pose(self, marker_id: int, pos: np.ndarray, rot: np.ndarray) -> None:
        entity = self._model_marker_plate_entities.get(int(marker_id))
        if entity is None:
            return
        self._set_entity_pose(entity, np.asarray(pos, dtype=float).reshape(3), _quat_wxyz_from_matrix(rot))

    def _set_entity_pose(self, entity: object, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=float).reshape(3)
        quat_wxyz = np.asarray(quat_wxyz, dtype=float).reshape(4)
        moved = False
        try:
            entity.set_pos(pos)
            moved = True
        except Exception:
            pass
        try:
            entity.set_quat(quat_wxyz)
            moved = True
        except Exception:
            pass
        if moved:
            return
        try:
            entity.set_qpos(
                np.array(
                    [pos[0], pos[1], pos[2], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]],
                    dtype=float,
                )
            )
        except Exception:
            pass

    def _draw_debug_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[float, float, float, float],
        *,
        radius: float,
    ) -> List[object]:
        if self._scene is None:
            return []
        a = np.asarray(start, dtype=float).reshape(3)
        b = np.asarray(end, dtype=float).reshape(3)
        try:
            return [self._scene.draw_debug_line(start=a, end=b, radius=float(radius), color=color)]
        except Exception:
            pass
        try:
            return [self._scene.draw_debug_line(a, b, float(radius), color)]
        except Exception:
            pass

        length = float(np.linalg.norm(b - a))
        steps = max(2, int(math.ceil(length / max(float(radius) * 2.5, 1e-4))))
        objects: List[object] = []
        for idx in range(steps + 1):
            alpha = idx / float(steps)
            pos = (1.0 - alpha) * a + alpha * b
            try:
                objects.append(self._scene.draw_debug_sphere(pos=pos, radius=float(radius), color=color))
            except Exception:
                break
        return objects

def run_genesis_process(
    urdf_path: str,
    use_gpu: bool,
    status_queue: object,
    command_queue: object,
) -> None:
    player = GenesisPlayer(
        Path(urdf_path),
        use_gpu=bool(use_gpu),
        status_queue=status_queue,
        command_queue=command_queue,
    )
    player.run()


class PlayerClient:
    def __init__(self, urdf_path: Path, *, use_gpu: bool = False) -> None:
        self.status_queue = mp.Queue()
        self._command_queue = mp.Queue()
        self._process = mp.Process(
            target=run_genesis_process,
            args=(str(urdf_path.expanduser().resolve()), bool(use_gpu), self.status_queue, self._command_queue),
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()

    def set_playback_data(self, data: PlaybackData) -> None:
        self._command_queue.put(("frames", data))

    def set_playing(self, playing: bool) -> None:
        self._command_queue.put(("playing", bool(playing)))

    def seek_frame(self, frame_index: int) -> None:
        self._command_queue.put(("seek", int(frame_index)))

    def stop(self) -> None:
        try:
            self._command_queue.put(("stop",))
        except Exception:
            pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


class PlayerApp:
    def __init__(self, root: Any, player: PlayerClient) -> None:
        self.root = root
        self.player = player
        self.root.title("Player")
        self.root.geometry("620x170")
        self.default_input_dir = Path(__file__).resolve().parent.parent / "merger" / "results"
        self.default_input_dir.mkdir(parents=True, exist_ok=True)
        self.path_var = tk.StringVar(value="No file loaded")
        self.status_var = tk.StringVar(value="Starting Genesis...")
        self.position_var = tk.StringVar(value="0 / 0   0.000s / 0.000s")
        self.seek_var = tk.DoubleVar(value=0.0)
        self._frame_count = 0
        self._current_frame_index = 0
        self._duration_sec = 0.0
        self._is_playing = False
        self._frame_times: List[float] = []
        self._dragging_seek = False
        self._updating_seek = False
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_status()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        buttons = tk.Frame(self.root)
        buttons.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        for col in range(3):
            buttons.columnconfigure(col, weight=1)
        tk.Button(buttons, text="Load File", command=self.load_file).grid(row=0, column=0, sticky="ew", padx=3)
        tk.Button(buttons, text="Play", command=lambda: self.player.set_playing(True)).grid(row=0, column=1, sticky="ew", padx=3)
        tk.Button(buttons, text="Pause", command=lambda: self.player.set_playing(False)).grid(row=0, column=2, sticky="ew", padx=3)
        self.seek_scale = tk.Scale(
            self.root,
            from_=0,
            to=0,
            orient="horizontal",
            resolution=1,
            showvalue=False,
            variable=self.seek_var,
            command=self._on_seek_move,
        )
        self.seek_scale.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 0))
        self.seek_scale.bind("<ButtonPress-1>", self._on_seek_press)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)
        tk.Label(self.root, textvariable=self.position_var, anchor="w").grid(row=2, column=0, sticky="ew", padx=12, pady=2)
        tk.Label(self.root, textvariable=self.path_var, anchor="w").grid(row=3, column=0, sticky="ew", padx=12, pady=2)
        tk.Label(self.root, textvariable=self.status_var, anchor="w").grid(row=4, column=0, sticky="ew", padx=12, pady=2)

    def load_file(self) -> None:
        assert filedialog is not None and messagebox is not None
        path = filedialog.askopenfilename(
            title="Load Playback CSV",
            initialdir=str(self.default_input_dir),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = load_playback_csv(Path(path))
        except Exception as exc:
            messagebox.showerror("Load File", str(exc))
            return
        self.player.set_playback_data(data)
        self.path_var.set(path)
        self._set_local_frames(data.frames)

    def _set_local_frames(self, frames: List[PlaybackFrame]) -> None:
        self._frame_times = [float(frame.t) for frame in frames]
        self._frame_count = len(self._frame_times)
        self._duration_sec = self._frame_times[-1] if self._frame_times else 0.0
        self._current_frame_index = 0
        self.seek_scale.configure(to=max(0, self._frame_count - 1))
        self._set_seek_value(0)
        self._update_position_label(0, self._frame_count, 0.0, self._duration_sec, False)

    def _set_seek_value(self, frame_index: int) -> None:
        self._updating_seek = True
        try:
            self.seek_var.set(float(max(0, int(frame_index))))
        finally:
            self._updating_seek = False

    def _on_seek_press(self, _event: object) -> None:
        self._dragging_seek = True

    def _on_seek_move(self, value: object) -> None:
        if self._updating_seek:
            return
        try:
            frame_index = int(round(float(value)))
        except Exception:
            return
        self._current_frame_index = frame_index
        self._update_position_label(
            frame_index,
            self._frame_count,
            self._time_for_frame(frame_index),
            self._duration_sec,
            self._is_playing,
        )

    def _on_seek_release(self, _event: object) -> None:
        self._dragging_seek = False
        frame_index = int(round(float(self.seek_var.get())))
        self.player.seek_frame(frame_index)

    def _handle_playback_state(self, payload: tuple[object, ...]) -> None:
        try:
            _name, index, total, t_sec, duration_sec, playing = payload
            frame_index = int(index)
            frame_count = int(total)
            t_value = float(t_sec)
            duration_value = float(duration_sec)
            is_playing = bool(playing)
        except Exception:
            return
        self._frame_count = max(0, frame_count)
        self._current_frame_index = max(0, frame_index)
        self._duration_sec = max(0.0, duration_value)
        self._is_playing = is_playing
        self.seek_scale.configure(to=max(0, self._frame_count - 1))
        if not self._dragging_seek:
            self._set_seek_value(self._current_frame_index)
        self._update_position_label(self._current_frame_index, self._frame_count, t_value, duration_value, is_playing)

    def _time_for_frame(self, frame_index: int) -> float:
        if self._frame_times:
            index = max(0, min(int(frame_index), len(self._frame_times) - 1))
            return float(self._frame_times[index])
        if self._frame_count <= 1:
            return 0.0
        ratio = max(0.0, min(float(frame_index) / float(self._frame_count - 1), 1.0))
        return ratio * float(self._duration_sec)

    def _update_position_label(
        self,
        frame_index: int,
        frame_count: int,
        t_sec: float,
        duration_sec: float,
        playing: bool,
    ) -> None:
        if frame_count <= 0:
            self.position_var.set("0 / 0   0.000s / 0.000s")
            return
        state = "Playing" if playing else "Paused"
        self.position_var.set(
            f"{frame_index + 1} / {frame_count}   {t_sec:.3f}s / {duration_sec:.3f}s   {state}"
        )

    def _poll_status(self) -> None:
        while True:
            try:
                message = self.player.status_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, tuple) and message and message[0] == "playback_state":
                self._handle_playback_state(message)
            else:
                self.status_var.set(str(message))
        self.root.after(100, self._poll_status)

    def _on_close(self) -> None:
        self.player.stop()
        self.root.destroy()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Genesis player for robot URDF and ArUco marker CSVs.")
    parser.add_argument("--urdf", default=str(default_urdf_path()), help="URDF path to load in Genesis.")
    parser.add_argument("--gpu", action="store_true", help="Use Genesis GPU backend.")
    parser.add_argument("--file", default="", help="Optional playback CSV to load on startup.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    if TK_IMPORT_ERROR is not None or tk is None:
        raise RuntimeError(f"tkinter is required for the player UI: {TK_IMPORT_ERROR}")
    args = parse_args(argv)
    player = PlayerClient(Path(args.urdf), use_gpu=bool(args.gpu))
    player.start()
    if args.file:
        try:
            player.set_playback_data(load_playback_csv(Path(args.file)))
        except Exception as exc:
            player.status_queue.put(f"Initial file load failed: {exc}")
    root = tk.Tk()
    app = PlayerApp(root, player)
    if args.file:
        app.path_var.set(str(Path(args.file).expanduser()))
    root.mainloop()


if __name__ == "__main__":
    main()
