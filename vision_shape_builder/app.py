from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .shape_core import build_shape_csv
except ImportError:
    from shape_core import build_shape_csv  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build metric shape CSV from refined marker pixels.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-nodes", type=int, default=11)
    p.add_argument("--meters-per-pixel", type=float, default=None, help="Pixel scale to meters (planar)")
    p.add_argument("--homography-json", type=str, default=None, help="3x3 homography JSON (pixel -> plane meters)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mpp = float(args.meters_per_pixel) if args.meters_per_pixel is not None else None
    hpath = Path(args.homography_json) if args.homography_json else None
    if mpp is None and hpath is None:
        print("need --meters-per-pixel or --homography-json", file=sys.stderr)
        return 2
    n = build_shape_csv(Path(args.input), Path(args.out), int(args.max_nodes), mpp, hpath)
    print(f"wrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
