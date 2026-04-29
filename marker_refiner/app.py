from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .refine_core import RefineParams, refine_marker_csv
except ImportError:
    from refine_core import RefineParams, refine_marker_csv  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refine yellow marker centroid CSV.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-nodes", type=int, default=10)
    p.add_argument("--min-mean-conf", type=float, default=0.25)
    p.add_argument("--min-node-conf", type=float, default=0.12)
    p.add_argument("--jump-px", type=float, default=40.0)
    p.add_argument("--max-gap-frames", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    params = RefineParams(
        max_nodes=int(args.max_nodes),
        min_mean_conf=float(args.min_mean_conf),
        min_node_conf=float(args.min_node_conf),
        jump_px_threshold=float(args.jump_px),
        max_gap_frames=int(args.max_gap_frames),
    )
    n = refine_marker_csv(Path(args.input), Path(args.out), params)
    print(f"wrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
