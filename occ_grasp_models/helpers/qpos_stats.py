import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from rlbench.demo import Demo

try:
    from natsort import natsorted
except Exception:
    natsorted = None


def _load_demo(low_dim_path: Path) -> Demo:
    with low_dim_path.open("rb") as f:
        data = pickle.load(f)
    if hasattr(data, "_observations"):
        return data
    return Demo(data)


def _as_list(tasks: Iterable[str]) -> list:
    return list(tasks) if tasks is not None else []


def compute_qpos_stats(data_root: str,
                       train_tasks: Iterable[str],
                       out_path: Optional[str] = None,
                       eps: float = 1e-6,
                       device: Optional[object] = None) -> Dict[str, np.ndarray]:
    """
    Compute qpos normalization statistics (right/left position + gripper open).

    Args:
        data_root: RLBench dataset root.
        train_tasks: list of task names (without .train suffix).
        out_path: optional cache path (.pkl). If exists, load from it.
        eps: min std clamp for numerical stability.
        device: if provided, returns torch tensors on this device.
    """
    if out_path is not None and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            stats = pickle.load(f)
        if device is not None:
            import torch
            return {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in stats.items()}
        return stats

    right_gripper_poses = []
    left_gripper_poses = []
    right_gripper_open = []
    left_gripper_open = []

    tasks = _as_list(train_tasks)
    for task in tasks:
        task_name = task if task.endswith(".train") else f"{task}.train"
        episodes_root = Path(data_root) / task_name / "all_variations" / "episodes"
        if not episodes_root.exists():
            raise FileNotFoundError(f"episodes_root not found: {episodes_root}")

        episode_dirs = [p for p in episodes_root.iterdir() if p.is_dir()]
        if natsorted is not None:
            episode_dirs = natsorted(episode_dirs, key=lambda p: p.name)
        else:
            episode_dirs = sorted(episode_dirs, key=lambda p: p.name)

        for ep_dir in episode_dirs:
            low_dim_path = ep_dir / "low_dim_obs.pkl"
            if not low_dim_path.exists():
                continue
            demo = _load_demo(low_dim_path)
            for obs in demo:
                right_gripper_poses.append(obs.right.gripper_pose)
                left_gripper_poses.append(obs.left.gripper_pose)
                right_gripper_open.append([obs.right.gripper_open])
                left_gripper_open.append([obs.left.gripper_open])

    right_gripper_poses = np.asarray(right_gripper_poses, dtype=np.float32)
    left_gripper_poses = np.asarray(left_gripper_poses, dtype=np.float32)
    right_gripper_open = np.asarray(right_gripper_open, dtype=np.float32)
    left_gripper_open = np.asarray(left_gripper_open, dtype=np.float32)

    stats = {
        "right_pos_mean": right_gripper_poses[:, :3].mean(axis=0),
        "right_pos_std": right_gripper_poses[:, :3].std(axis=0),
        "left_pos_mean": left_gripper_poses[:, :3].mean(axis=0),
        "left_pos_std": left_gripper_poses[:, :3].std(axis=0),
        "right_gripper_open_mean": right_gripper_open.mean(axis=0),
        "right_gripper_open_std": right_gripper_open.std(axis=0),
        "left_gripper_open_mean": left_gripper_open.mean(axis=0),
        "left_gripper_open_std": left_gripper_open.std(axis=0),
    }

    if eps is not None and eps > 0:
        stats["right_pos_std"] = np.maximum(stats["right_pos_std"], eps)
        stats["left_pos_std"] = np.maximum(stats["left_pos_std"], eps)
        stats["right_gripper_open_std"] = np.maximum(stats["right_gripper_open_std"], eps)
        stats["left_gripper_open_std"] = np.maximum(stats["left_gripper_open_std"], eps)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(stats, f)

    if device is not None:
        import torch
        return {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in stats.items()}

    return stats
