from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class YellowBlob:
    cx: float
    cy: float
    area: float
    circularity: float
    conf: float


@dataclass(frozen=True)
class DetectionSplit:
    shoulder_anchor: Optional[YellowBlob]
    circular_markers: List[YellowBlob]


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
    anchor_aspect_min: float,
    min_circularity: float,
    anchor_max_circularity: float,
    max_markers: int,
) -> DetectionSplit:
    hsv = cv2.cvtColor(np.asarray(frame_bgr, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    lower = np.array(list(hsv_lower), dtype=np.uint8)
    upper = np.array(list(hsv_upper), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_blobs: List[YellowBlob] = []
    anchor_candidates: List[YellowBlob] = []
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
        if circ <= anchor_max_circularity and aspect >= anchor_aspect_min:
            anchor_conf = float(score_area * min(1.0, aspect / max(anchor_aspect_min, 1.0)))
            anchor_candidates.append(YellowBlob(cx=cx, cy=cy, area=area, circularity=circ, conf=anchor_conf))
            continue
        if circ < min_circularity or aspect > marker_aspect_max:
            continue
        marker_conf = float(min(1.0, (circ + 0.15)) * score_area)
        marker_blobs.append(YellowBlob(cx=cx, cy=cy, area=area, circularity=circ, conf=marker_conf))
    marker_blobs.sort(key=lambda b: b.conf, reverse=True)
    markers = marker_blobs[: max(0, int(max_markers))]
    shoulder_anchor = None
    if anchor_candidates:
        anchor_candidates.sort(key=lambda b: (b.conf, b.area), reverse=True)
        shoulder_anchor = anchor_candidates[0]
    return DetectionSplit(shoulder_anchor=shoulder_anchor, circular_markers=markers)


def draw_preview(
    frame_bgr: np.ndarray,
    shoulder_anchor: Optional[YellowBlob],
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
    if shoulder_anchor is not None:
        a = shoulder_anchor
        apt = (int(round(a.cx)), int(round(a.cy)))
        cv2.circle(out, apt, 8, (0, 140, 255), 2)
        cv2.putText(out, "A", (apt[0] + 8, apt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
    for idx, blob in enumerate(markers):
        pt = (int(round(blob.cx)), int(round(blob.cy)))
        cv2.circle(out, pt, 6, (0, 255, 255), 2)
        cv2.putText(out, str(idx + 1), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out
