from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


AXES = ("x", "y", "z")
Q_AXES = ("x", "y", "z", "w")
CAMERA_COLUMNS = ("pCAMx", "pCAMy", "pCAMz", "qCAMx", "qCAMy", "qCAMz", "qCAMw")
CAMERA_VALUES = {
    "pCAMx": "0.000000",
    "pCAMy": "0.000000",
    "pCAMz": "0.000000",
    "qCAMx": "0.000000",
    "qCAMy": "0.000000",
    "qCAMz": "0.000000",
    "qCAMw": "1.000000",
}
FRIEND_MARKER_GROUPS = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
)
FRIENDS_BY_MARKER = {
    marker_id: tuple(other for other in group if other != marker_id)
    for group in FRIEND_MARKER_GROUPS
    for marker_id in group
}


@dataclass(frozen=True)
class PoseSample:
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]


@dataclass
class MarkerState:
    accepted: PoseSample | None = None
    accepted_row: int | None = None


@dataclass(frozen=True)
class RefineConfig:
    max_angle_deg: float = 75.0
    max_position_step_m: float = 0.10
    reset_gap_rows: int = 5
    interpolate_gap_rows: int = 2
    marker_ids: tuple[int, ...] | None = None
    enable_voting: bool = True
    vote_min_support: int = 2
    vote_max_delta_step_m: float = 0.04
    vote_max_delta_angle_deg: float = 30.0


@dataclass(frozen=True)
class MarkerStats:
    raw: int = 0
    accepted: int = 0
    rejected: int = 0
    interpolated: int = 0


@dataclass(frozen=True)
class RefineResult:
    output_path: Path
    rows: int
    stats: dict[int, MarkerStats]


@dataclass
class _MutableStats:
    raw: int = 0
    accepted: int = 0
    rejected: int = 0
    interpolated: int = 0

    def frozen(self) -> MarkerStats:
        return MarkerStats(
            raw=self.raw,
            accepted=self.accepted,
            rejected=self.rejected,
            interpolated=self.interpolated,
        )


@dataclass(frozen=True)
class AcceptDecision:
    accepted: bool
    reason: str
    angle_jump: float | None = None
    pos_step: float | None = None


@dataclass(frozen=True)
class VoteDecision:
    applies: bool
    passed: bool
    support: int = 1
    available_friends: int = 0


def marker_columns(marker_id: int) -> list[str]:
    return [f"p{marker_id}{axis}" for axis in AXES] + [f"q{marker_id}{axis}" for axis in Q_AXES]


def diagnostic_columns(marker_id: int) -> list[str]:
    return [
        f"valid{marker_id}",
        f"quality{marker_id}",
        f"reason{marker_id}",
        f"angle_jump{marker_id}",
        f"pos_step{marker_id}",
    ]


def infer_marker_ids(fieldnames: Iterable[str]) -> tuple[int, ...]:
    fields = set(fieldnames)
    ids: list[int] = []
    for marker_id in range(1, 100):
        has_position = all(f"p{marker_id}{axis}" in fields for axis in AXES)
        has_quat = all(f"q{marker_id}{axis}" in fields for axis in Q_AXES)
        if has_position and has_quat:
            ids.append(marker_id)
    return tuple(ids)


def output_fields(input_fields: list[str], marker_ids: tuple[int, ...]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for field in input_fields:
        if field in seen:
            continue
        fields.append(field)
        seen.add(field)
        if field == "t":
            for camera_field in CAMERA_COLUMNS:
                if camera_field not in seen:
                    fields.append(camera_field)
                    seen.add(camera_field)
    if "t" not in seen:
        raise RuntimeError("input CSV does not contain a 't' column")
    for marker_id in marker_ids:
        for field in diagnostic_columns(marker_id):
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def refine_csv_file(input_csv: Path, output_csv: Path, config: RefineConfig | None = None) -> RefineResult:
    cfg = config or RefineConfig()
    input_csv = input_csv.expanduser().resolve()
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if "t" not in input_fields:
        raise RuntimeError(f"{input_csv.name} does not contain a 't' column")

    detected_marker_ids = infer_marker_ids(input_fields)
    requested_marker_ids = detected_marker_ids if cfg.marker_ids is None else cfg.marker_ids
    marker_ids = tuple(marker_id for marker_id in requested_marker_ids if marker_id in detected_marker_ids)
    states = {marker_id: MarkerState() for marker_id in marker_ids}
    stats = {marker_id: _MutableStats() for marker_id in marker_ids}

    refined = [dict(row) for row in rows]
    for row in refined:
        for key, value in CAMERA_VALUES.items():
            row[key] = value

    for row_index, row in enumerate(refined):
        current_by_id = {marker_id: _read_marker(row, marker_id) for marker_id in marker_ids}
        self_decisions: dict[int, AcceptDecision] = {}
        vote_decisions: dict[int, VoteDecision] = {}

        for marker_id, current in current_by_id.items():
            if current is None:
                self_decisions[marker_id] = AcceptDecision(False, "missing")
                continue
            stats[marker_id].raw += 1
            self_decisions[marker_id] = _accept_marker(states[marker_id], current, row_index, cfg)

        for marker_id, current in current_by_id.items():
            vote_decisions[marker_id] = _vote_marker(
                marker_id,
                current,
                current_by_id,
                states,
                cfg,
            )

        for marker_id in marker_ids:
            current = current_by_id[marker_id]
            self_decision = self_decisions[marker_id]
            vote_decision = vote_decisions[marker_id]
            final_decision = _combine_decisions(self_decision, vote_decision)
            if current is None:
                _write_marker_blank(row, marker_id, reason=final_decision.reason)
                continue
            if final_decision.accepted:
                states[marker_id].accepted = current
                states[marker_id].accepted_row = row_index
                stats[marker_id].accepted += 1
                _write_marker(
                    row,
                    marker_id,
                    current,
                    valid=True,
                    reason=final_decision.reason,
                    angle_jump=final_decision.angle_jump,
                    pos_step=final_decision.pos_step,
                )
            else:
                stats[marker_id].rejected += 1
                _write_marker_blank(
                    row,
                    marker_id,
                    reason=final_decision.reason,
                    angle_jump=final_decision.angle_jump,
                    pos_step=final_decision.pos_step,
                )

    _interpolate_short_gaps(refined, marker_ids, cfg, stats)

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = output_fields(input_fields, marker_ids)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(refined)

    return RefineResult(
        output_path=output_csv,
        rows=len(refined),
        stats={marker_id: marker_stats.frozen() for marker_id, marker_stats in stats.items()},
    )


def _accept_marker(
    state: MarkerState,
    current: PoseSample,
    row_index: int,
    cfg: RefineConfig,
) -> AcceptDecision:
    if state.accepted is None or state.accepted_row is None:
        return AcceptDecision(True, "first")
    row_gap = int(row_index) - int(state.accepted_row)
    if row_gap > int(cfg.reset_gap_rows):
        return AcceptDecision(True, "reset")

    angle_jump = _quat_angle_deg(state.accepted.q, current.q)
    pos_step = _distance(state.accepted.p, current.p)
    if angle_jump is not None and angle_jump > float(cfg.max_angle_deg):
        return AcceptDecision(False, "angle_jump", angle_jump, pos_step)
    if pos_step is not None and pos_step > float(cfg.max_position_step_m):
        return AcceptDecision(False, "position_jump", angle_jump, pos_step)
    return AcceptDecision(True, "accepted", angle_jump, pos_step)


def _vote_marker(
    marker_id: int,
    current: PoseSample | None,
    current_by_id: dict[int, PoseSample | None],
    states: dict[int, MarkerState],
    cfg: RefineConfig,
) -> VoteDecision:
    if not cfg.enable_voting or current is None:
        return VoteDecision(False, True)
    own_state = states.get(int(marker_id))
    if own_state is None or own_state.accepted is None:
        return VoteDecision(False, True)

    own_delta_p = _sub_vec(current.p, own_state.accepted.p)
    own_delta_q = _relative_quat(own_state.accepted.q, current.q)
    support = 1
    available = 0
    for friend_id in FRIENDS_BY_MARKER.get(int(marker_id), ()):
        friend_current = current_by_id.get(int(friend_id))
        friend_state = states.get(int(friend_id))
        if friend_current is None or friend_state is None or friend_state.accepted is None:
            continue
        available += 1
        friend_delta_p = _sub_vec(friend_current.p, friend_state.accepted.p)
        friend_delta_q = _relative_quat(friend_state.accepted.q, friend_current.q)
        delta_step = _distance(own_delta_p, friend_delta_p)
        delta_angle = _quat_angle_deg(own_delta_q, friend_delta_q)
        if delta_step <= float(cfg.vote_max_delta_step_m) and (
            delta_angle is None or delta_angle <= float(cfg.vote_max_delta_angle_deg)
        ):
            support += 1

    if available <= 0:
        return VoteDecision(False, True, support=support, available_friends=available)
    return VoteDecision(
        True,
        support >= max(1, int(cfg.vote_min_support)),
        support=support,
        available_friends=available,
    )


def _combine_decisions(self_decision: AcceptDecision, vote_decision: VoteDecision) -> AcceptDecision:
    if not vote_decision.applies:
        return self_decision
    if vote_decision.passed:
        if self_decision.accepted:
            return self_decision
        if self_decision.reason in {"position_jump", "angle_jump", "reset"}:
            return AcceptDecision(True, "voted", self_decision.angle_jump, self_decision.pos_step)
        return self_decision
    if self_decision.accepted:
        return AcceptDecision(False, "vote_mismatch", self_decision.angle_jump, self_decision.pos_step)
    return self_decision


def _interpolate_short_gaps(
    rows: list[dict[str, str]],
    marker_ids: tuple[int, ...],
    cfg: RefineConfig,
    stats: dict[int, _MutableStats],
) -> None:
    if cfg.interpolate_gap_rows <= 0:
        return
    for marker_id in marker_ids:
        accepted_indices = [idx for idx, row in enumerate(rows) if row.get(f"valid{marker_id}") == "1"]
        for left, right in zip(accepted_indices, accepted_indices[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > int(cfg.interpolate_gap_rows):
                continue
            left_pair = _read_marker(rows[left], marker_id)
            right_pair = _read_marker(rows[right], marker_id)
            if left_pair is None or right_pair is None:
                continue
            for offset, row_index in enumerate(range(left + 1, right), start=1):
                alpha = offset / float(gap + 1)
                sample = PoseSample(
                    p=_lerp_vec(left_pair.p, right_pair.p, alpha),
                    q=_slerp_quat(left_pair.q, right_pair.q, alpha),
                )
                _write_marker(rows[row_index], marker_id, sample, valid=True, reason="interpolated")
                stats[marker_id].interpolated += 1


def _read_marker(row: dict[str, str], marker_id: int) -> PoseSample | None:
    p_values = [_float_or_none(row.get(f"p{marker_id}{axis}")) for axis in AXES]
    q_values = [_float_or_none(row.get(f"q{marker_id}{axis}")) for axis in Q_AXES]
    if any(value is None for value in p_values + q_values):
        return None
    q = _normalize_quat(tuple(float(value) for value in q_values if value is not None))
    if q is None:
        return None
    return PoseSample(
        p=tuple(float(value) for value in p_values if value is not None),
        q=q,
    )


def _write_marker(
    row: dict[str, str],
    marker_id: int,
    sample: PoseSample,
    *,
    valid: bool,
    reason: str,
    angle_jump: float | None = None,
    pos_step: float | None = None,
) -> None:
    row[f"p{marker_id}x"] = f"{sample.p[0]:.6f}"
    row[f"p{marker_id}y"] = f"{sample.p[1]:.6f}"
    row[f"p{marker_id}z"] = f"{sample.p[2]:.6f}"
    row[f"q{marker_id}x"] = f"{sample.q[0]:.8f}"
    row[f"q{marker_id}y"] = f"{sample.q[1]:.8f}"
    row[f"q{marker_id}z"] = f"{sample.q[2]:.8f}"
    row[f"q{marker_id}w"] = f"{sample.q[3]:.8f}"
    row[f"valid{marker_id}"] = "1" if valid else "0"
    row[f"quality{marker_id}"] = "1.000" if valid else "0.000"
    row[f"reason{marker_id}"] = reason
    row[f"angle_jump{marker_id}"] = "" if angle_jump is None else f"{angle_jump:.3f}"
    row[f"pos_step{marker_id}"] = "" if pos_step is None else f"{pos_step:.6f}"


def _write_marker_blank(
    row: dict[str, str],
    marker_id: int,
    *,
    reason: str,
    angle_jump: float | None = None,
    pos_step: float | None = None,
) -> None:
    for field in marker_columns(marker_id):
        row[field] = ""
    row[f"valid{marker_id}"] = "0"
    row[f"quality{marker_id}"] = "0.000"
    row[f"reason{marker_id}"] = reason
    row[f"angle_jump{marker_id}"] = "" if angle_jump is None else f"{angle_jump:.3f}"
    row[f"pos_step{marker_id}"] = "" if pos_step is None else f"{pos_step:.6f}"


def _float_or_none(value: str | None) -> float | None:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    length = math.sqrt(sum(value * value for value in quat))
    if length <= 1e-12:
        return None
    return tuple(float(value) / length for value in quat)


def _quat_angle_deg(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float | None:
    na = _normalize_quat(a)
    nb = _normalize_quat(b)
    if na is None or nb is None:
        return None
    dot = abs(max(-1.0, min(1.0, sum(x * y for x, y in zip(na, nb)))))
    return math.degrees(2.0 * math.acos(dot))


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _sub_vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))


def _lerp_vec(a: tuple[float, float, float], b: tuple[float, float, float], alpha: float) -> tuple[float, float, float]:
    return tuple((1.0 - alpha) * x + alpha * y for x, y in zip(a, b))


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def _quat_multiply(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _relative_quat(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    prev = _normalize_quat(previous) or (0.0, 0.0, 0.0, 1.0)
    cur = _normalize_quat(current) or (0.0, 0.0, 0.0, 1.0)
    return _normalize_quat(_quat_multiply(cur, _quat_conjugate(prev))) or (0.0, 0.0, 0.0, 1.0)


def _slerp_quat(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    if qa is None:
        return qb or (0.0, 0.0, 0.0, 1.0)
    if qb is None:
        return qa

    dot = sum(x * y for x, y in zip(qa, qb))
    if dot < 0.0:
        qb = tuple(-value for value in qb)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        mixed = tuple((1.0 - alpha) * x + alpha * y for x, y in zip(qa, qb))
        return _normalize_quat(mixed) or qa

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    scale_a = math.sin(theta_0 - theta) / sin_theta_0
    scale_b = math.sin(theta) / sin_theta_0
    return tuple(scale_a * x + scale_b * y for x, y in zip(qa, qb))
