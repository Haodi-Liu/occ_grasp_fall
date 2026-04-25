import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np  # Added for uniform phase sampling across GT phase segments.

try:
    from natsort import natsorted
except Exception:
    natsorted = None

from rlbench.demo import Demo

from helpers.demo_loading_utils import keypoint_discovery


def _load_demo(low_dim_path: Path) -> Demo:
    with low_dim_path.open("rb") as f:
        data = pickle.load(f)
    if hasattr(data, "_observations"):
        return data
    return Demo(data)


def _collect_keypoint_offsets(demo: Demo, offsets: List[int], keypoint_method: str) -> List[int]:
    # Preserve original behavior: expand keypoints by offsets and clamp to valid indices.
    keypoints = keypoint_discovery(demo, method=keypoint_method)
    if len(keypoints) == 0:
        return []
    last = len(demo) - 1
    frame_ids = set()
    for kp in keypoints:
        for off in offsets:
            idx = kp + off
            if idx < 0:
                idx = 0
            elif idx > last:
                idx = last
            frame_ids.add(idx)
    return sorted(frame_ids)


def _collect_phase_frames(demo: Demo, phase_sample_counts: List[int], phase_key: str = "phase_type") -> List[int]:
    # Uniformly sample frames per GT phase from obs.misc (phase_type is 1-based in demos).
    phase_to_frames = {pid: [] for pid in range(len(phase_sample_counts))}
    max_phase = len(phase_sample_counts) - 1
    for i, obs in enumerate(demo):
        misc = getattr(obs, "misc", {}) or {}
        phase_1b = int(misc.get(phase_key, 1))  # Default to phase 1 if missing.
        phase = max(0, min(phase_1b - 1, max_phase))  # Clamp to valid range.
        phase_to_frames[phase].append(i)

    sampled = set()
    for phase_id, frames in phase_to_frames.items():
        k = int(phase_sample_counts[phase_id])
        if k <= 0:
            continue  # Allow zero to disable a phase without changing code paths.
        if len(frames) <= k:
            sampled.update(frames)  # Not enough frames → take all.
        else:
            # Evenly spaced indices for uniform coverage within each phase segment.
            idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
            sampled.update([frames[j] for j in idx])
    return sorted(sampled)


def _collect_frames(demo: Demo,
                    offsets: List[int],
                    keypoint_method: str,
                    phase_sample_counts: Optional[List[int]] = None) -> List[int]:
    # Combine keypoint-offset frames with phase-uniform frames, then de-duplicate.
    frame_ids = set(_collect_keypoint_offsets(demo, offsets, keypoint_method))
    if phase_sample_counts is not None:
        frame_ids.update(_collect_phase_frames(demo, phase_sample_counts))
    return sorted(frame_ids)


def build_index(data_root: str,
                tasks: List[str],
                split: str,
                offsets: List[int],
                phase_sample_counts: Optional[List[int]],  # Phase-uniform sampling counts per phase.
                keypoint_method: str,
                max_episodes: Optional[int],
                out_path: str):
    items = []
    suffix = ".train" if split == "train" else ""

    for task in tasks:
        task_name = task if (split == "eval" or task.endswith(".train")) else f"{task}{suffix}"
        episodes_root = Path(data_root) / task_name / "all_variations" / "episodes"
        if not episodes_root.exists():
            raise FileNotFoundError(f"episodes_root not found: {episodes_root}")

        episode_dirs = [p for p in episodes_root.iterdir() if p.is_dir()]
        if natsorted is not None:
            episode_dirs = natsorted(episode_dirs, key=lambda p: p.name)
        else:
            episode_dirs = sorted(episode_dirs, key=lambda p: p.name)
        if max_episodes is not None:
            episode_dirs = episode_dirs[:max_episodes]

        for ep_dir in episode_dirs:
            low_dim_path = ep_dir / "low_dim_obs.pkl"
            if not low_dim_path.exists():
                continue
            demo = _load_demo(low_dim_path)
            # Include phase-uniform frames in addition to keypoint-offset frames.
            frame_ids = _collect_frames(demo, offsets, keypoint_method, phase_sample_counts)
            for frame_id in frame_ids:
                items.append({
                    "task": task_name,
                    "episode": ep_dir.name,
                    "frame": int(frame_id),
                })

    payload = {
        "meta": {
            "data_root": os.path.abspath(data_root),
            "tasks": tasks,
            "split": split,
            "offsets": offsets,
            "phase_sample_counts": phase_sample_counts,  # Record phase sampling for reproducibility.
            "keypoint_method": keypoint_method,
            "max_episodes": max_episodes,
        },
        "items": items,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        out_path.write_text(json.dumps(payload, indent=2))
    else:
        with out_path.open("wb") as f:
            pickle.dump(payload, f)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--split", choices=["train", "eval"], required=True)
    p.add_argument("--offsets", nargs="+", type=int, default=[-3, -2, -1, 0, 1, 2, 3])
    p.add_argument("--phase_sample_counts", nargs="+", type=int,
                   default=[80, 100, 10, 20],
                   help="Per-phase uniform sample counts for phase 1..4 (1-based).")
    p.add_argument("--keypoint_method", default="heuristic")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--out_path", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_index(
        data_root=args.data_root,
        tasks=args.tasks,
        split=args.split,
        offsets=args.offsets,
        phase_sample_counts=args.phase_sample_counts,  # Phase-uniform sampling counts per phase.
        keypoint_method=args.keypoint_method,
        max_episodes=args.max_episodes,
        out_path=args.out_path,
    )
