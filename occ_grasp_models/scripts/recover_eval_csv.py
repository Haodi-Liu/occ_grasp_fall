#!/usr/bin/env python3
"""Recover a missing eval_data.csv from TensorBoard events and eval logs.

This script is intended for evaluation directories where `eval_data.csv` was
not flushed to disk even though the evaluation run completed (or mostly
completed). It:

1. Reads the latest scalar summaries from a TensorBoard event file.
2. Optionally parses the textual eval log to recover phase/scheme metrics.
3. Writes a single-row `eval_data.csv`.

The implementation has a pure-stdlib fallback for `.tfevents` parsing, so it
can run even when `tensorboard` is unavailable.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import struct
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple


ScalarRow = OrderedDict[str, float]


def _read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Unexpected end of buffer while decoding varint.")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7


def _skip_field(data: bytes, offset: int, wire_type: int) -> int:
    if wire_type == 0:
        _, offset = _read_varint(data, offset)
        return offset
    if wire_type == 1:
        return offset + 8
    if wire_type == 2:
        length, offset = _read_varint(data, offset)
        return offset + length
    if wire_type == 5:
        return offset + 4
    raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def _parse_summary_value(message: bytes) -> Optional[Tuple[str, float]]:
    tag = None
    value = None
    offset = 0
    while offset < len(message):
        key, offset = _read_varint(message, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number == 1 and wire_type == 2:
            length, offset = _read_varint(message, offset)
            tag = message[offset: offset + length].decode("utf-8", errors="replace")
            offset += length
        elif field_number == 2 and wire_type == 5:
            value = struct.unpack("<f", message[offset: offset + 4])[0]
            offset += 4
        else:
            offset = _skip_field(message, offset, wire_type)
    if tag is None or value is None:
        return None
    return tag, float(value)


def _parse_summary(message: bytes) -> List[Tuple[str, float]]:
    scalars: List[Tuple[str, float]] = []
    offset = 0
    while offset < len(message):
        key, offset = _read_varint(message, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number == 1 and wire_type == 2:
            length, offset = _read_varint(message, offset)
            parsed = _parse_summary_value(message[offset: offset + length])
            if parsed is not None:
                scalars.append(parsed)
            offset += length
        else:
            offset = _skip_field(message, offset, wire_type)
    return scalars


def _parse_event(message: bytes) -> Tuple[int, List[Tuple[str, float]]]:
    step = 0
    scalars: List[Tuple[str, float]] = []
    offset = 0
    while offset < len(message):
        key, offset = _read_varint(message, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number == 2 and wire_type == 0:
            step, offset = _read_varint(message, offset)
        elif field_number == 5 and wire_type == 2:
            length, offset = _read_varint(message, offset)
            scalars.extend(_parse_summary(message[offset: offset + length]))
            offset += length
        else:
            offset = _skip_field(message, offset, wire_type)
    return step, scalars


def _iter_tfrecord_records(path: str) -> Iterable[bytes]:
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError(f"Incomplete TFRecord header in {path}.")
            (length,) = struct.unpack("<Q", header)
            crc_header = handle.read(4)
            if len(crc_header) != 4:
                raise ValueError(f"Incomplete TFRecord length CRC in {path}.")
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"Incomplete TFRecord payload in {path}.")
            crc_payload = handle.read(4)
            if len(crc_payload) != 4:
                raise ValueError(f"Incomplete TFRecord payload CRC in {path}.")
            yield payload


def _load_latest_event_row_minimal(path: str) -> ScalarRow:
    rows: "OrderedDict[int, ScalarRow]" = OrderedDict()
    for payload in _iter_tfrecord_records(path):
        step, scalars = _parse_event(payload)
        if not scalars:
            continue
        row = rows.setdefault(step, OrderedDict(step=int(step)))
        for tag, value in scalars:
            row[tag] = value
    if not rows:
        raise ValueError(f"No scalar summaries found in event file: {path}")
    last_step = next(reversed(rows))
    return rows[last_step]


def _load_latest_event_row(path: str) -> ScalarRow:
    try:
        from tensorboard.backend.event_processing import event_accumulator

        accumulator = event_accumulator.EventAccumulator(path)
        accumulator.Reload()
        rows: "OrderedDict[int, ScalarRow]" = OrderedDict()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                row = rows.setdefault(event.step, OrderedDict(step=int(event.step)))
                row[tag] = float(event.value)
        if rows:
            last_step = next(reversed(rows))
            return rows[last_step]
    except Exception:
        pass
    return _load_latest_event_row_minimal(path)


def _parse_eval_log(log_path: str) -> Dict[str, float]:
    episodes: Dict[int, Dict[str, object]] = {}
    pending_phase_progress: List[Tuple[Dict[int, bool], int]] = []

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = re.search(r"Episode (\d+) GT scheme: ([A-Za-z_]+)", line)
            if match:
                episode_id = int(match.group(1))
                episodes.setdefault(episode_id, {})["gt_scheme"] = match.group(2)
                continue

            match = re.search(r"Phase progress: (\{.*\}), Max phase: (\d+)", line)
            if match:
                phase_status = ast.literal_eval(match.group(1))
                max_phase = int(match.group(2))
                pending_phase_progress.append((phase_status, max_phase))
                continue

            match = re.search(
                r"Episode (\d+) (SUCCESS|FAILED) with scheme '([A-Za-z_]+)'(?:, steps: (\d+))?",
                line,
            )
            if match:
                episode_id = int(match.group(1))
                episode = episodes.setdefault(episode_id, {})
                episode["outcome"] = match.group(2)
                episode["scheme"] = match.group(3)
                if match.group(4) is not None:
                    episode["steps"] = int(match.group(4))
                if pending_phase_progress:
                    phase_status, max_phase = pending_phase_progress.pop(0)
                    episode["phase_status"] = phase_status
                    episode["max_phase"] = max_phase

    if not episodes:
        raise ValueError(f"No episode summaries found in eval log: {log_path}")

    success_count = 0
    failed_count = 0
    phase_success_counts = {phase_id: 0 for phase_id in range(1, 5)}
    max_phases_reached: List[int] = []
    scheme_stats = {
        "left_grasper": {"success": 0, "total": 0, "success_steps": []},
        "right_grasper": {"success": 0, "total": 0, "success_steps": []},
        "unknown": {"success": 0, "total": 0, "success_steps": []},
    }

    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        gt_scheme = str(episode.get("gt_scheme", "unknown"))
        scheme_stats.setdefault(gt_scheme, {"success": 0, "total": 0, "success_steps": []})
        scheme_stats[gt_scheme]["total"] += 1

        outcome = episode.get("outcome")
        if outcome == "SUCCESS":
            success_count += 1
            if "steps" in episode:
                scheme_stats[gt_scheme]["success_steps"].append(int(episode["steps"]))
            scheme_stats[gt_scheme]["success"] += 1
        else:
            failed_count += 1

        phase_status = episode.get("phase_status", {})
        for phase_id in range(1, 5):
            if phase_status.get(phase_id, False):
                phase_success_counts[phase_id] += 1

        if "max_phase" in episode:
            max_phases_reached.append(int(episode["max_phase"]))

    total_episodes = success_count + failed_count
    if total_episodes == 0:
        raise ValueError(f"No terminal episode outcomes found in eval log: {log_path}")

    recovered: Dict[str, float] = {
        "eval_envs/success_count": float(success_count),
        "eval_envs/failed_count": float(failed_count),
        "eval_envs/success_rate": float(success_count / total_episodes),
        "eval_envs/avg_max_phase": (
            float(sum(max_phases_reached) / len(max_phases_reached))
            if max_phases_reached
            else 0.0
        ),
    }
    for phase_id in range(1, 5):
        recovered[f"eval_envs/phase_{phase_id}_success_rate"] = float(
            phase_success_counts[phase_id] / total_episodes
        )

    for scheme in ("left_grasper", "right_grasper"):
        stats = scheme_stats[scheme]
        total = int(stats["total"])
        success = int(stats["success"])
        success_steps = list(stats["success_steps"])
        recovered[f"eval_envs/success_rate_{scheme}_scenes"] = float(success / total) if total else 0.0
        recovered[f"eval_envs/total_{scheme}_episodes"] = float(total)
        if success_steps:
            recovered[f"eval_envs/avg_steps_{scheme}_scenes"] = float(
                sum(success_steps) / len(success_steps)
            )

    recovered["eval_envs/scheme_balance_gap"] = abs(
        recovered.get("eval_envs/success_rate_left_grasper_scenes", 0.0)
        - recovered.get("eval_envs/success_rate_right_grasper_scenes", 0.0)
    )
    return recovered


def _ordered_row(row: Dict[str, float]) -> ScalarRow:
    preferred_order = [
        "step",
        "eval_envs/return",
        "eval_envs/length",
        "eval_envs/total_transitions",
        "eval_envs/success_count",
        "eval_envs/failed_count",
        "eval_envs/success_rate",
        "eval_envs/phase_1_success_rate",
        "eval_envs/phase_2_success_rate",
        "eval_envs/phase_3_success_rate",
        "eval_envs/phase_4_success_rate",
        "eval_envs/avg_max_phase",
        "eval_envs/success_rate_left_grasper_scenes",
        "eval_envs/total_left_grasper_episodes",
        "eval_envs/avg_steps_left_grasper_scenes",
        "eval_envs/success_rate_right_grasper_scenes",
        "eval_envs/total_right_grasper_episodes",
        "eval_envs/avg_steps_right_grasper_scenes",
        "eval_envs/scheme_balance_gap",
    ]
    ordered = OrderedDict()
    for key in preferred_order:
        if key in row:
            ordered[key] = row[key]
    for key in sorted(row):
        if key not in ordered:
            ordered[key] = row[key]
    return ordered


def _write_csv(output_path: str, row: ScalarRow) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="Path to events.out.tfevents.*")
    parser.add_argument("--log", help="Optional eval log used to recover missing metrics.")
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to <events_dir>/eval_data.csv.",
    )
    args = parser.parse_args()

    row = _load_latest_event_row(args.events)
    if args.log:
        row.update(_parse_eval_log(args.log))

    output_path = args.output or os.path.join(os.path.dirname(args.events), "eval_data.csv")
    ordered_row = _ordered_row(row)
    _write_csv(output_path, ordered_row)

    recovered_keys = [key for key in ordered_row.keys() if key != "step"]
    print(f"Wrote {output_path}")
    print(f"Recovered step={ordered_row.get('step', 'unknown')} with {len(recovered_keys)} metrics.")
    if args.log:
        print(f"Merged log-derived metrics from {args.log}")


if __name__ == "__main__":
    main()
