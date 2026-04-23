from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    from .geometry import Pose, reconstruct_base_relative_pose, pose_from_rvec_tvec, base_owner
except ImportError:
    from geometry import Pose, reconstruct_base_relative_pose, pose_from_rvec_tvec, base_owner  # type: ignore


SYNC_DICT_NAME = cv2.aruco.DICT_5X5_50
MEASUREMENT_DICT_NAME = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M = 0.017


@dataclass
class DetectionResult:
    display_frame: np.ndarray
    sync_ids: set[int]
    relative_poses: Dict[int, Pose]


class ArucoDetector:
    def __init__(self) -> None:
        self.sync_dict = cv2.aruco.getPredefinedDictionary(SYNC_DICT_NAME)
        self.meas_dict = cv2.aruco.getPredefinedDictionary(MEASUREMENT_DICT_NAME)
        self.sync_detector = cv2.aruco.ArucoDetector(self.sync_dict, cv2.aruco.DetectorParameters())
        self.meas_detector = cv2.aruco.ArucoDetector(self.meas_dict, cv2.aruco.DetectorParameters())
        half = MARKER_SIZE_M * 0.5
        self.object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> DetectionResult:
        source = np.asarray(frame_bgr, dtype=np.uint8)
        display = source.copy()
        camera_matrix, dist_coeffs = self._camera_guess(source.shape[1], source.shape[0])
        sync_ids, sync_corners, sync_detected_ids = self._detect_sync(source)
        meas_relative_poses, meas_corners, meas_detected_ids, axes_payloads = self._detect_measurement(
            source,
            camera_matrix,
            dist_coeffs,
        )
        if sync_detected_ids is not None and len(sync_detected_ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, sync_corners, sync_detected_ids)
        if meas_detected_ids is not None and len(meas_detected_ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, meas_corners, meas_detected_ids)
        for rvec, tvec in axes_payloads:
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE_M * 0.75, 2)
        return DetectionResult(display_frame=display, sync_ids=sync_ids, relative_poses=meas_relative_poses)

    def _camera_guess(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        fx = float(width)
        fy = float(width)
        cx = float(width) * 0.5
        cy = float(height) * 0.5
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        return camera_matrix, dist_coeffs

    def _detect_sync(self, source: np.ndarray) -> tuple[set[int], list[np.ndarray], np.ndarray | None]:
        corners, ids, _ = self.sync_detector.detectMarkers(source)
        sync_ids: set[int] = set()
        if ids is None or len(ids) == 0:
            return sync_ids, corners, ids
        for marker_id in ids.reshape(-1):
            marker_int = int(marker_id)
            if marker_int in {0, 1, 2}:
                sync_ids.add(marker_int)
        return sync_ids, corners, ids

    def _detect_measurement(
        self,
        source: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[Dict[int, Pose], list[np.ndarray], np.ndarray | None, list[tuple[np.ndarray, np.ndarray]]]:
        corners, ids, _ = self.meas_detector.detectMarkers(source)
        if ids is None or len(ids) == 0:
            return {}, corners, ids, []
        camera_poses: Dict[int, Pose] = {}
        axes_payloads: list[tuple[np.ndarray, np.ndarray]] = []
        for idx, marker_id in enumerate(ids.reshape(-1)):
            marker_int = int(marker_id)
            ok, rvec, tvec = cv2.solvePnP(
                self.object_points,
                np.asarray(corners[idx], dtype=np.float64).reshape(4, 2),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue
            pose = pose_from_rvec_tvec(rvec, tvec)
            camera_poses[marker_int] = pose
            axes_payloads.append((rvec, tvec))

        relative_poses: Dict[int, Pose] = {}
        for marker_id, pose_camera_marker in camera_poses.items():
            owner = base_owner(marker_id)
            if owner is None or int(owner) == int(marker_id):
                continue
            pose_camera_base = camera_poses.get(int(owner))
            if pose_camera_base is None:
                continue
            relative_poses[int(marker_id)] = reconstruct_base_relative_pose(
                marker_pose_camera=pose_camera_marker,
                base_pose_camera=pose_camera_base,
            )
        return relative_poses, corners, ids, axes_payloads
