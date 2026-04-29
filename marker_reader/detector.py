from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class YellowBlob:
    cx: float
    cy: float
    area: float
    circularity: float
    conf: float


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 1e-9:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def _contour_center(contour: np.ndarray) -> tuple[float, float]:
    cx: float
    cy: float
    if contour.shape[0] >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            center = ellipse[0]
            cx, cy = float(center[0]), float(center[1])
            return cx, cy
        except Exception:
            pass
    (x, y), _ = cv2.minEnclosingCircle(contour)
    cx, cy = float(x), float(y)
    moments = cv2.moments(contour)
    if moments["m00"] > 1e-9:
        mx = float(moments["m10"] / moments["m00"])
        my = float(moments["m01"] / moments["m00"])
        if abs(mx - cx) <= 12.0 and abs(my - cy) <= 12.0:
            cx, cy = mx, my
    return cx, cy


def detect_yellow_features(
    frame_bgr: np.ndarray,
    hsv_lower: Sequence[int],
    hsv_upper: Sequence[int],
    area_min: float,
    area_max: float,
    marker_aspect_max: float,
    min_circularity: float,
    max_markers: int,
) -> List[YellowBlob]:
    hsv = cv2.cvtColor(np.asarray(frame_bgr, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    lower = np.array(list(hsv_lower), dtype=np.uint8)
    upper = np.array(list(hsv_upper), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_blobs: List[YellowBlob] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < area_min or area > area_max:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-9:
            continue
        circ = _circularity(area, perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = float(max(w, h)) / float(min(w, h))
        cx, cy = _contour_center(contour)
        score_area = min(1.0, area / max(area_min * 2.0, 1.0))
        if circ < min_circularity or aspect > marker_aspect_max:
            continue
        marker_conf = float(min(1.0, (circ + 0.15)) * score_area)
        marker_blobs.append(YellowBlob(cx=cx, cy=cy, area=area, circularity=circ, conf=marker_conf))
    marker_blobs.sort(key=lambda b: b.conf, reverse=True)
    return marker_blobs[: max(0, int(max_markers))]


def draw_preview(
    frame_bgr: np.ndarray,
    markers: Sequence[YellowBlob],
    hsv_lower: Sequence[int],
    hsv_upper: Sequence[int],
) -> np.ndarray:
    out = np.asarray(frame_bgr, dtype=np.uint8).copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    lower = np.array(list(hsv_lower), dtype=np.uint8)
    upper = np.array(list(hsv_upper), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    overlay = np.zeros_like(out)
    overlay[:, :, 1] = mask
    out = cv2.addWeighted(out, 0.85, overlay, 0.15, 0)
    for idx, blob in enumerate(markers):
        pt = (int(round(blob.cx)), int(round(blob.cy)))
        cv2.circle(out, pt, 6, (0, 255, 255), 2)
        cv2.putText(out, str(idx), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out
