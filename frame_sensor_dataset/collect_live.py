from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .common import build_manifest_row, write_manifest_csv, write_manifest_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build raw frame+sensor synced dataset from live camera stream.")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--sensor-zmq", type=str, required=True)
    p.add_argument("--imu-port", type=str, default="")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--sync-tol-sec", type=float, default=0.05)
    p.add_argument("--save-every-n", type=int, default=1)
    p.add_argument("--image-ext", type=str, default="jpg", choices=("jpg", "png"))
    p.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    return p


def _setup_subscriber(endpoint: str):
    try:
        import zmq
    except Exception as exc:
        raise RuntimeError("pyzmq is required for --sensor-zmq live mode") from exc
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 1)
    return ctx, sub


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(args.camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera index {args.camera_index}")
    ctx, sub = _setup_subscriber(str(args.sensor_zmq))

    latest_sensor: dict[str, str] | None = None
    latest_sensor_t: float | None = None
    save_every_n = max(1, int(args.save_every_n))
    ext = str(args.image_ext).lower()
    max_frames = int(args.max_frames)
    rows: list[dict[str, str]] = []

    frame_idx = 0
    saved_idx = 0
    try:
        while True:
            try:
                payload = sub.recv_string()
                msg = json.loads(payload)
                latest_sensor = {str(k): str(v) for k, v in msg.items()}
                latest_sensor_t = float(msg.get("t_sensor_mono", time.monotonic()))
            except Exception:
                pass

            ok, frame = cap.read()
            if not ok:
                continue
            if frame_idx % save_every_n != 0:
                frame_idx += 1
                continue
            t_frame_mono = time.monotonic()
            t_frame_wall = time.time()
            dt = None if latest_sensor_t is None else float(latest_sensor_t - t_frame_mono)
            sync_valid = dt is not None and abs(float(dt)) <= float(args.sync_tol_sec)

            frame_name = f"frame_{saved_idx:06d}.{ext}"
            frame_rel = f"frames/{frame_name}"
            frame_path = frames_dir / frame_name
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"failed to write frame: {frame_path}")

            rows.append(
                build_manifest_row(
                    frame_idx=saved_idx,
                    frame_path_rel=frame_rel,
                    t_frame_wall=t_frame_wall,
                    t_frame_mono=t_frame_mono,
                    sensor=latest_sensor,
                    t_sensor_mono=latest_sensor_t,
                    dt_sensor=dt,
                    sync_valid=bool(sync_valid),
                )
            )
            saved_idx += 1
            frame_idx += 1
            if max_frames > 0 and saved_idx >= max_frames:
                break
    finally:
        cap.release()
        try:
            sub.close()
            ctx.term()
        except Exception:
            pass

    write_manifest_csv(out_dir / "manifest.csv", rows)
    write_manifest_jsonl(out_dir / "manifest.jsonl", rows)
    print(f"saved frames={saved_idx} manifest={out_dir/'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
