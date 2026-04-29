from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _float_or_nan(value: object) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("nan")


def _bool01(value: object) -> float:
    s = str(value).strip().lower()
    return 1.0 if s in {"1", "true", "yes", "y"} else 0.0


def _read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append({k: str(v) for k, v in json.loads(line).items()})
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def _detect_max_nodes(first_row: dict[str, str]) -> int:
    i = 0
    while f"res_node{i}_x_m" in first_row or f"nominal_node{i}_x_m" in first_row:
        i += 1
    return i


def _feature_columns(max_nodes: int, include_nominal_nodes: bool) -> list[str]:
    cols: list[str] = []
    cols += ["roll", "seg1", "seg2", "load1", "load2"]
    for imu_id in (1, 2):
        cols += [
            f"imu{imu_id}_timestamp",
            f"imu{imu_id}_gvx",
            f"imu{imu_id}_gvy",
            f"imu{imu_id}_gvz",
            f"imu{imu_id}_gx",
            f"imu{imu_id}_gy",
            f"imu{imu_id}_gz",
            f"imu{imu_id}_qw",
            f"imu{imu_id}_qx",
            f"imu{imu_id}_qy",
            f"imu{imu_id}_qz",
            f"imu{imu_id}_quatAcc",
            f"imu{imu_id}_gravAcc",
            f"imu{imu_id}_gyroAcc",
            f"imu{imu_id}_lax",
            f"imu{imu_id}_lay",
            f"imu{imu_id}_laz",
            f"imu{imu_id}_ax",
            f"imu{imu_id}_ay",
            f"imu{imu_id}_az",
            f"imu{imu_id}_mx",
            f"imu{imu_id}_my",
            f"imu{imu_id}_mz",
        ]
    for j in range(max(0, max_nodes - 1)):
        cols.append(f"nominal_joint{j}_deg")
    cols += ["nominal_ee_x_m", "nominal_ee_y_m", "nominal_ee_z_m"]
    if include_nominal_nodes:
        for i in range(max_nodes):
            cols += [f"nominal_node{i}_x_m", f"nominal_node{i}_y_m", f"nominal_node{i}_z_m"]
    return cols


def _target_columns(max_nodes: int) -> list[str]:
    cols: list[str] = []
    for i in range(max_nodes):
        cols += [f"res_node{i}_x_m", f"res_node{i}_y_m", f"res_node{i}_z_m"]
    cols += ["res_ee_x_m", "res_ee_y_m", "res_ee_z_m"]
    return cols


def _session_key(row: dict[str, str], default_key: str) -> str:
    for k in ("session_id", "episode_id", "run_id"):
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return default_key


def _split_sessions(session_ids: list[str], seed: int = 7) -> dict[str, str]:
    uniq = sorted(set(session_ids))
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(uniq)))
    uniq = [uniq[i] for i in perm]
    n = len(uniq)
    n_train = max(1, int(round(n * 0.7)))
    n_val = max(1, int(round(n * 0.15))) if n >= 3 else max(0, n - n_train)
    n_test = max(0, n - n_train - n_val)
    if n_test == 0 and n >= 3:
        n_test = 1
        n_train = max(1, n_train - 1)
    train = set(uniq[:n_train])
    val = set(uniq[n_train : n_train + n_val])
    test = set(uniq[n_train + n_val : n_train + n_val + n_test])
    out: dict[str, str] = {}
    for sid in session_ids:
        if sid in train:
            out[sid] = "train"
        elif sid in val:
            out[sid] = "val"
        elif sid in test:
            out[sid] = "test"
        else:
            out[sid] = "train"
    return out


def build_dataset(input_path: Path, output_path: Path, include_nominal_nodes: bool, seed: int) -> dict[str, int]:
    rows = _read_rows(input_path)
    if not rows:
        raise RuntimeError(f"no rows in {input_path}")
    max_nodes = _detect_max_nodes(rows[0])
    x_cols = _feature_columns(max_nodes, include_nominal_nodes)
    y_cols = _target_columns(max_nodes)

    session_default = input_path.stem
    session_ids = [_session_key(r, session_default) for r in rows]
    split_map = _split_sessions(session_ids, seed)

    X = np.asarray([[_float_or_nan(r.get(c, "")) for c in x_cols] for r in rows], dtype=np.float32)
    Y = np.asarray([[_float_or_nan(r.get(c, "")) for c in y_cols] for r in rows], dtype=np.float32)

    valid_shape = np.asarray([_bool01(r.get("valid_shape", r.get("validity", "false"))) for r in rows], dtype=np.float32)
    valid_ee = np.asarray([_bool01(r.get("valid_ee", "false")) for r in rows], dtype=np.float32)
    imu_fresh = np.asarray([_bool01(r.get("imu_fresh", r.get("imu_valid", "true"))) for r in rows], dtype=np.float32)
    valid_nodes = np.asarray(
        [[_bool01(r.get(f"valid_node{i}", "false")) for i in range(max_nodes)] for r in rows],
        dtype=np.float32,
    )
    split = np.asarray([split_map[sid] for sid in session_ids], dtype=object)

    metadata = {
        "input": str(input_path),
        "rows": len(rows),
        "max_nodes": max_nodes,
        "include_nominal_nodes": include_nominal_nodes,
        "split_counts": {
            "train": int(np.sum(split == "train")),
            "val": int(np.sum(split == "val")),
            "test": int(np.sum(split == "test")),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        Y=Y,
        feature_names=np.asarray(x_cols, dtype=object),
        target_names=np.asarray(y_cols, dtype=object),
        split=split,
        mask_valid_shape=valid_shape,
        mask_valid_ee=valid_ee,
        mask_valid_nodes=valid_nodes,
        mask_imu_fresh=imu_fresh,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object),
    )
    return {"rows": len(rows), "x_dim": X.shape[1], "y_dim": Y.shape[1], "max_nodes": max_nodes}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build shape residual NPZ dataset from merged CSV/JSONL.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--include-nominal-nodes", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = build_dataset(
        Path(args.input).expanduser().resolve(),
        Path(args.out).expanduser().resolve(),
        bool(args.include_nominal_nodes),
        int(args.seed),
    )
    print(
        f"wrote {args.out} rows={stats['rows']} x_dim={stats['x_dim']} y_dim={stats['y_dim']} max_nodes={stats['max_nodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
