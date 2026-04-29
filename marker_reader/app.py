from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import cv2

try:
    from .detector import YellowBlob, detect_yellow_features, draw_preview
except ImportError:
    from detector import YellowBlob, detect_yellow_features, draw_preview  # type: ignore


def _parse_hsv_pair(text: str) -> tuple[int, int, int]:
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected H,S,V comma triple")
    return (parts[0], parts[1], parts[2])


def _row_fields(max_nodes: int) -> List[str]:
    fields = [
        "frame_idx",
        "t_video_sec",
        "n_detected",
        "mean_conf",
        "validity",
        "sync_dt_sec",
    ]
    for i in range(max_nodes):
        fields.extend(
            [
                f"c{i}_x",
                f"c{i}_y",
                f"c{i}_area",
                f"c{i}_circularity",
                f"c{i}_conf",
            ]
        )
    return fields


def _blobs_to_row(
    frame_idx: int,
    t_video_sec: float,
    blobs: List[YellowBlob],
    max_nodes: int,
) -> dict[str, str]:
    mean_conf = sum(b.conf for b in blobs) / max(len(blobs), 1)
    row: dict[str, str] = {
        "frame_idx": str(int(frame_idx)),
        "t_video_sec": f"{float(t_video_sec):.6f}",
        "n_detected": str(len(blobs)),
        "mean_conf": f"{float(mean_conf):.6f}",
        "validity": "true" if len(blobs) >= max_nodes else "false",
        "sync_dt_sec": "",
    }
    for i in range(max_nodes):
        if i < len(blobs):
            b = blobs[i]
            row[f"c{i}_x"] = f"{b.cx:.4f}"
            row[f"c{i}_y"] = f"{b.cy:.4f}"
            row[f"c{i}_area"] = f"{b.area:.4f}"
            row[f"c{i}_circularity"] = f"{b.circularity:.6f}"
            row[f"c{i}_conf"] = f"{b.conf:.6f}"
        else:
            row[f"c{i}_x"] = ""
            row[f"c{i}_y"] = ""
            row[f"c{i}_area"] = ""
            row[f"c{i}_circularity"] = ""
            row[f"c{i}_conf"] = ""
    return row


def run_capture(args: argparse.Namespace) -> int:
    max_nodes = int(args.max_nodes)
    hsv_lo = _parse_hsv_pair(args.hsv_lower)
    hsv_hi = _parse_hsv_pair(args.hsv_upper)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = str(args.video) if args.video is not None else str(int(args.camera))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"failed to open video source: {source}", file=sys.stderr)
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    fields = _row_fields(max_nodes)
    frame_idx = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            blobs = detect_yellow_features(
                frame,
                hsv_lo,
                hsv_hi,
                area_min=float(args.area_min),
                area_max=float(args.area_max),
                marker_aspect_max=float(args.marker_aspect_max),
                min_circularity=float(args.min_circularity),
                max_markers=max_nodes * 4,
            )
            blobs = sorted(blobs, key=lambda b: b.cy)
            blobs = blobs[:max_nodes]
            t_video_sec = float(frame_idx) / fps
            writer.writerow(_blobs_to_row(frame_idx, t_video_sec, blobs, max_nodes))
            if args.preview:
                vis = draw_preview(frame, blobs, hsv_lo, hsv_hi)
                cv2.imshow("marker_reader", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1

    cap.release()
    if args.preview:
        cv2.destroyAllWindows()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Yellow circular sticker detector (HSV + contours).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, default=None, help="Video file path")
    src.add_argument("--camera", type=int, default=None, help="Camera index")
    p.add_argument("--out", type=str, required=True, help="Output CSV path")
    p.add_argument("--max-nodes", type=int, default=11)
    p.add_argument("--hsv-lower", type=str, default="20,100,100", help="H,S,V lower (comma)")
    p.add_argument("--hsv-upper", type=str, default="35,255,255", help="H,S,V upper (comma)")
    p.add_argument("--area-min", type=float, default=80.0)
    p.add_argument("--area-max", type=float, default=8000.0)
    p.add_argument("--marker-aspect-max", type=float, default=1.6)
    p.add_argument("--min-circularity", type=float, default=0.45)
    p.add_argument("--preview", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
