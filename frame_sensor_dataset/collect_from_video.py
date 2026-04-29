from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .common import (
    build_manifest_row,
    load_sensor_csv_rows,
    nearest_sensor_row,
    write_manifest_csv,
    write_manifest_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build raw frame+sensor synced dataset from video and sensor log.")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--sensor-log", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--sync-tol-sec", type=float, default=0.05)
    p.add_argument("--save-every-n", type=int, default=1)
    p.add_argument("--image-ext", type=str, default="jpg", choices=("jpg", "png"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    video_path = Path(args.video).expanduser().resolve()
    sensor_path = Path(args.sensor_log).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sensor_rows = load_sensor_csv_rows(sensor_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    run_wall_start = time.time()
    run_mono_start = time.monotonic()
    save_every_n = max(1, int(args.save_every_n))
    ext = str(args.image_ext).lower()

    rows: list[dict[str, str]] = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % save_every_n != 0:
            frame_idx += 1
            continue

        t_frame_mono = float(frame_idx) / fps
        t_frame_wall = run_wall_start + max(0.0, t_frame_mono - (time.monotonic() - run_mono_start))
        nearest, dt = nearest_sensor_row(sensor_rows, t_frame_mono)
        sync_valid = nearest is not None and dt is not None and abs(float(dt)) <= float(args.sync_tol_sec)

        frame_name = f"frame_{saved_idx:06d}.{ext}"
        frame_rel = f"frames/{frame_name}"
        frame_path = frames_dir / frame_name
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"failed to write frame: {frame_path}")

        row = build_manifest_row(
            frame_idx=saved_idx,
            frame_path_rel=frame_rel,
            t_frame_wall=t_frame_wall,
            t_frame_mono=t_frame_mono,
            sensor=nearest.row if nearest is not None else None,
            t_sensor_mono=nearest.t_mono if nearest is not None else None,
            dt_sensor=dt,
            sync_valid=bool(sync_valid),
        )
        rows.append(row)
        saved_idx += 1
        frame_idx += 1

    cap.release()
    write_manifest_csv(out_dir / "manifest.csv", rows)
    write_manifest_jsonl(out_dir / "manifest.jsonl", rows)
    print(f"saved frames={saved_idx} manifest={out_dir/'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
