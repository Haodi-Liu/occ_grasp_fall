"""Index replay and dynamic sequence loading for DIFFUSION_POLICY."""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from natsort import natsorted
except Exception:
    natsorted = None

from helpers import observation_utils, utils


LOW_DIM_PICKLE = "low_dim_obs.pkl"


@dataclass
class EpisodeInfo:
    episode_id: int
    task: str
    path: str
    length: int


def iter_logical_starts(
    length: int, horizon: int, pad_before: int = 0, pad_after: int = 0
) -> Iterable[int]:
    """Yield sequence start indices following diffusion_policy create_indices semantics."""
    horizon = max(1, int(horizon))
    length = int(length)

    # Match SequenceSampler/create_indices clamping behavior.
    pad_before = min(max(int(pad_before), 0), horizon - 1)
    pad_after = min(max(int(pad_after), 0), horizon - 1)

    min_start = -pad_before
    max_start = length - horizon + pad_after
    for start in range(min_start, max_start + 1):
        yield start


class LRUCache:
    """Minimal LRU cache backed by OrderedDict."""

    def __init__(self, maxsize: int):
        self.maxsize = max(0, int(maxsize))
        self._store: "OrderedDict[str, object]" = OrderedDict()

    def get(self, key, default=None):
        if self.maxsize <= 0 or key not in self._store:
            return default
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value) -> None:
        if self.maxsize <= 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def __contains__(self, key):
        return key in self._store


def build_episode_index(
    dataset_root: str,
    tasks: Sequence[str],
    num_demos: int,
    index_seed: Optional[int] = None,
) -> List[EpisodeInfo]:
    episode_index: List[EpisodeInfo] = []
    episode_id = 0
    num_demos = int(num_demos)
    rng = np.random.default_rng(int(index_seed)) if index_seed is not None else None

    for task in tasks:
        task_name = task if str(task).endswith(".train") else f"{task}.train"
        episodes_root = os.path.join(
            dataset_root, task_name, "all_variations", "episodes"
        )
        if not os.path.isdir(episodes_root):
            raise FileNotFoundError(f"episodes_root not found: {episodes_root}")

        episode_names = [
            x
            for x in os.listdir(episodes_root)
            if os.path.isdir(os.path.join(episodes_root, x))
        ]
        if natsorted is not None:
            episode_names = list(natsorted(episode_names))
        else:
            episode_names = sorted(episode_names)

        available = len(episode_names)
        if num_demos > available:
            raise ValueError(f"{task}: num_demos={num_demos} > available={available}")

        selected = episode_names[:num_demos]
        if rng is not None and num_demos < available:
            subset_idx = np.sort(rng.choice(available, size=num_demos, replace=False))
            selected = [episode_names[int(i)] for i in subset_idx]

        for ep_name in selected:
            ep_path = os.path.join(episodes_root, ep_name)
            length = _get_demo_length(ep_path)
            episode_index.append(
                EpisodeInfo(
                    episode_id=episode_id,
                    task=str(task),
                    path=ep_path,
                    length=length,
                )
            )
            episode_id += 1

    return episode_index


def save_episode_index(
    episode_index: Sequence[EpisodeInfo],
    replay_dir: Optional[str],
    fallback_dir: str,
) -> str:
    save_dir = replay_dir if replay_dir is not None else fallback_dir
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "episode_index.json")
    payload = [asdict(info) for info in episode_index]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_episode_index(cfg=None, episode_index_path: Optional[str] = None) -> List[EpisodeInfo]:
    candidates: List[str] = []
    if episode_index_path is not None:
        candidates.append(episode_index_path)

    if cfg is not None:
        method_path = getattr(getattr(cfg, "method", None), "episode_index_path", None)
        if method_path is not None:
            candidates.append(method_path)

        framework_logdir = getattr(getattr(cfg, "framework", None), "logdir", None)
        if framework_logdir is not None:
            candidates.append(os.path.join(framework_logdir, "episode_index.json"))

        replay_path = getattr(getattr(cfg, "replay", None), "path", None)
        if replay_path is not None:
            task_folder = "multi"
            tasks = list(getattr(getattr(cfg, "rlbench", None), "tasks", []))
            if len(tasks) == 1:
                task_folder = tasks[0]
            method_name = getattr(getattr(cfg, "method", None), "name", None)
            if method_name is not None:
                seed = int(getattr(cfg, "seed", 0))
                candidates.append(
                    os.path.join(
                        replay_path,
                        task_folder,
                        method_name,
                        f"seed{seed}",
                        "episode_index.json",
                    )
                )

    seen = set()
    for path in candidates:
        if path is None or path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info(
                "DIFFUSION_POLICY loaded episode index | path=%s | episodes=%d",
                path,
                len(data),
            )
            return [EpisodeInfo(**item) for item in data]
    raise FileNotFoundError("episode_index.json not found in candidate paths.")


class IndexSequenceLoader:
    """Build obs/action sequences from (episode_id, timestep) index pairs."""

    def __init__(self, cfg, episode_index_path: Optional[str] = None):
        self.cfg = cfg
        self.camera_names = list(cfg.method.camera_names)
        self.robot_name = str(getattr(cfg.method, "robot_name", "bimanual"))
        self.image_size = tuple(int(x) for x in cfg.method.image_size)
        self.n_obs_steps = int(cfg.method.n_obs_steps)
        self.horizon = int(cfg.method.horizon)
        self.episode_length = int(cfg.rlbench.episode_length)

        self.episode_index = load_episode_index(cfg, episode_index_path=episode_index_path)
        self._episode_by_id = {int(info.episode_id): info for info in self.episode_index}
        self.lowdim_cache = LRUCache(maxsize=int(getattr(cfg.method, "index_cache_size", 0)))
        self.image_cache = LRUCache(maxsize=int(getattr(cfg.method, "image_cache_size", 0)))

    def _load_low_dim_demo_cached(self, episode_path: str):
        demo = self.lowdim_cache.get(episode_path)
        if demo is not None:
            return demo

        low_dim_path = os.path.join(episode_path, LOW_DIM_PICKLE)
        with open(low_dim_path, "rb") as f:
            demo = pickle.load(f)
        self.lowdim_cache.put(episode_path, demo)
        return demo

    def _resolve_episode(self, episode_id: int) -> EpisodeInfo:
        key = int(episode_id)
        if key not in self._episode_by_id:
            raise KeyError(f"episode_id={key} missing in episode index.")
        return self._episode_by_id[key]

    def build_obs_sequence(
        self,
        info: EpisodeInfo,
        timestep: int,
        n_obs_steps: Optional[int] = None,
        cameras: Optional[Sequence[str]] = None,
        image_size: Optional[Sequence[int]] = None,
        episode_length: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        n_obs_steps = int(self.n_obs_steps if n_obs_steps is None else n_obs_steps)
        cameras = list(self.camera_names if cameras is None else cameras)
        image_size = tuple(self.image_size if image_size is None else image_size)
        episode_length = int(
            self.episode_length if episode_length is None else episode_length
        )

        demo = self._load_low_dim_demo_cached(info.path)
        max_idx = int(info.length) - 1
        indices = [min(max(int(timestep) + i, 0), max_idx) for i in range(n_obs_steps)]

        obs_seq = {f"{cam}_rgb": [] for cam in cameras}
        lowdim_seq = []

        for idx in indices:
            obs = demo[idx]
            _attach_rgb_to_obs(
                obs=obs,
                episode_path=info.path,
                idx=idx,
                cameras=cameras,
                image_size=image_size,
                image_cache=self.image_cache,
            )
            frame = observation_utils.extract_obs(
                obs=obs,
                cameras=cameras,
                t=idx,
                channels_last=False,
                episode_length=episode_length,
                robot_name=self.robot_name,
            )
            frame["low_dim_state"] = _get_action_with_ignore_from_obs(obs)
            frame = _strip_ignore_collisions(frame)
            frame = _filter_obs_keys(frame, cameras)

            for cam in cameras:
                obs_seq[f"{cam}_rgb"].append(frame[f"{cam}_rgb"])
            lowdim_seq.append(frame["low_dim_state"])

        obs_seq = {
            k: np.stack(v, axis=0).astype(np.float32, copy=False)
            for k, v in obs_seq.items()
        }
        obs_seq["low_dim_state"] = np.stack(lowdim_seq, axis=0).astype(
            np.float32, copy=False
        )
        return obs_seq

    def build_action_sequence(
        self, info: EpisodeInfo, timestep: int, horizon: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        horizon = int(self.horizon if horizon is None else horizon)
        demo = self._load_low_dim_demo_cached(info.path)
        max_idx = int(info.length) - 1

        actions = []
        is_pad = []
        for k in range(horizon):
            target_idx = int(timestep) + k
            pad = (target_idx < 0) or (target_idx >= int(info.length))
            idx = min(max(target_idx, 0), max_idx)
            actions.append(_get_action_with_ignore_from_obs(demo[idx]))
            is_pad.append(1 if pad else 0)

        return (
            np.stack(actions, axis=0).astype(np.float32, copy=False),
            np.asarray(is_pad, dtype=np.int32),
        )

    def build_batch(
        self, episode_ids: Iterable[int], timesteps: Iterable[int]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        obs_list = []
        action_list = []
        is_pad_list = []

        for episode_id, timestep in zip(episode_ids, timesteps):
            info = self._resolve_episode(int(episode_id))
            t = int(timestep)
            obs_seq = self.build_obs_sequence(info, t)
            action_seq, is_pad = self.build_action_sequence(info, t)
            obs_list.append(obs_seq)
            action_list.append(action_seq)
            is_pad_list.append(is_pad)

        batch = collate_to_batch(obs_list, action_list)
        batch["is_pad"] = np.stack(is_pad_list, axis=0).astype(np.int32, copy=False)
        return batch

    def build_batch_from_replay_sample(
        self, replay_sample: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        episode_ids = _extract_index_vector(replay_sample["episode_id"])
        timesteps = _extract_index_vector(replay_sample["timestep"])
        return self.build_batch(episode_ids, timesteps)

    def sample_action_lowdim_stats(
        self, episode_ids: Iterable[int], timesteps: Iterable[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        actions = []
        lowdims = []
        for episode_id, timestep in zip(episode_ids, timesteps):
            info = self._resolve_episode(int(episode_id))
            t = int(timestep)
            action_seq, _ = self.build_action_sequence(info, t)
            obs_seq = self.build_obs_sequence(info, t)
            actions.append(_flatten_last_dim(action_seq))
            lowdims.append(_flatten_last_dim(obs_seq["low_dim_state"]))

        if len(actions) == 0:
            raise ValueError("No samples were provided for normalizer statistics.")
        return np.concatenate(actions, axis=0), np.concatenate(lowdims, axis=0)


class ReplaySampleToBatchTransform:
    def __init__(self, cfg, episode_index_path: Optional[str] = None):
        self._cfg = cfg
        self._episode_index_path = episode_index_path
        self._loader: Optional[IndexSequenceLoader] = None

    def __call__(self, replay_sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._loader is None:
            self._loader = IndexSequenceLoader(
                cfg=self._cfg, episode_index_path=self._episode_index_path
            )

        batch = self._loader.build_batch_from_replay_sample(replay_sample)
        flat_batch = {
            key: _add_single_task_dim(value) for key, value in batch["obs"].items()
        }
        flat_batch["action"] = _add_single_task_dim(batch["action"])
        flat_batch["is_pad"] = _add_single_task_dim(batch["is_pad"])
        for key in ("demo", "timeout", "sampling_probabilities"):
            if key in replay_sample:
                flat_batch[key] = replay_sample[key]
        return flat_batch


def _flatten_last_dim(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"Expected ndim>=2, got shape={arr.shape}")
    return arr.reshape(-1, arr.shape[-1])


def _extract_index_vector(index_array) -> np.ndarray:
    if torch.is_tensor(index_array):
        arr = index_array.detach().cpu().numpy()
    else:
        arr = np.asarray(index_array)

    if arr.ndim == 1:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 2:
        return arr[:, -1].astype(np.int64, copy=False)
    raise ValueError(f"Unexpected index tensor shape: {arr.shape}")


def _add_single_task_dim(array):
    if torch.is_tensor(array):
        return array.unsqueeze(1)
    return np.expand_dims(np.asarray(array), axis=1)


def _get_demo_length(episode_path: str) -> int:
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), "rb") as f:
        demo = pickle.load(f)
    return len(demo)


def _attach_rgb_to_obs(
    obs,
    episode_path: str,
    idx: int,
    cameras: Sequence[str],
    image_size: Sequence[int],
    image_cache: Optional[LRUCache] = None,
) -> None:
    if not hasattr(obs, "perception_data") or obs.perception_data is None:
        obs.perception_data = {}

    # image_size follows project-wide convention [W, H].
    width, height = int(image_size[0]), int(image_size[1])
    for cam in cameras:
        cache_key = f"{episode_path}::{cam}::{idx}::{width}x{height}"
        image = image_cache.get(cache_key) if image_cache is not None else None
        if image is None:
            image = _load_rgb_image(episode_path, cam, idx, (width, height))
            if image_cache is not None:
                image_cache.put(cache_key, image)
        obs.perception_data[f"{cam}_rgb"] = image


def _load_rgb_image(
    episode_path: str, camera_name: str, idx: int, image_size: Tuple[int, int]
) -> np.ndarray:
    rgb_folder = os.path.join(episode_path, f"{camera_name}_rgb")
    candidates = [
        os.path.join(rgb_folder, f"rgb_{idx:04d}.png"),
        os.path.join(rgb_folder, f"{idx}.png"),
    ]
    image_path = None
    for path in candidates:
        if os.path.exists(path):
            image_path = path
            break
    if image_path is None:
        raise FileNotFoundError(
            f"RGB frame not found for {camera_name} idx={idx} under {rgb_folder}"
        )

    image = Image.open(image_path).convert("RGB")
    if image.size != tuple(image_size):
        image = image.resize(tuple(image_size))
    return np.asarray(image, dtype=np.float32) / 255.0


def _get_action_from_obs(obs) -> np.ndarray:
    right_quat = utils.normalize_quaternion(obs.right.gripper_pose[3:])
    if right_quat[-1] < 0:
        right_quat = -right_quat
    left_quat = utils.normalize_quaternion(obs.left.gripper_pose[3:])
    if left_quat[-1] < 0:
        left_quat = -left_quat

    right = np.concatenate(
        [obs.right.gripper_pose[:3], right_quat, [float(obs.right.gripper_open)]],
        axis=0,
    )
    left = np.concatenate(
        [obs.left.gripper_pose[:3], left_quat, [float(obs.left.gripper_open)]],
        axis=0,
    )
    return np.concatenate([right, left], axis=0).astype(np.float32, copy=False)


def _get_action_with_ignore_from_obs(obs) -> np.ndarray:
    action16 = _get_action_from_obs(obs)
    right_ignore, left_ignore = _get_ignore_collisions_from_obs(obs)
    return _append_ignore_collisions(
        action16, right_ignore=right_ignore, left_ignore=left_ignore
    )


def _get_ignore_collisions_from_obs(obs) -> Tuple[float, float]:
    right_obs = getattr(obs, "right", None)
    left_obs = getattr(obs, "left", None)
    if right_obs is not None and left_obs is not None:
        right_ignore = getattr(right_obs, "ignore_collisions", None)
        left_ignore = getattr(left_obs, "ignore_collisions", None)
        if right_ignore is not None and left_ignore is not None:
            return float(right_ignore), float(left_ignore)

    ignore = getattr(obs, "ignore_collisions", None)
    if ignore is not None:
        value = float(ignore)
        return value, value
    return 1.0, 1.0


def _append_ignore_collisions(
    action16: np.ndarray, right_ignore: float = 1.0, left_ignore: float = 1.0
) -> np.ndarray:
    action16 = np.asarray(action16, dtype=np.float32).reshape(-1)
    if action16.shape[0] != 16:
        raise ValueError(f"Expected 16D action, got shape={action16.shape}")
    return np.concatenate(
        [action16[:8], [float(right_ignore)], action16[8:], [float(left_ignore)]], axis=0
    ).astype(np.float32, copy=False)


def _strip_ignore_collisions(frame: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    frame.pop("ignore_collisions", None)
    frame.pop("right_ignore_collisions", None)
    frame.pop("left_ignore_collisions", None)
    return frame


def _filter_obs_keys(frame: Dict[str, np.ndarray], cameras: Sequence[str]) -> Dict[str, np.ndarray]:
    keep = {f"{cam}_rgb" for cam in cameras}
    keep.add("low_dim_state")
    return {k: v for k, v in frame.items() if k in keep}


def collate_to_batch(
    obs_list: Sequence[Dict[str, np.ndarray]], action_list: Sequence[np.ndarray]
) -> Dict[str, Dict[str, np.ndarray]]:
    if len(obs_list) == 0:
        raise ValueError("obs_list is empty.")

    obs_batch = {}
    for key in obs_list[0].keys():
        obs_batch[key] = np.stack([obs[key] for obs in obs_list], axis=0).astype(
            np.float32, copy=False
        )
    action_batch = np.stack(action_list, axis=0).astype(np.float32, copy=False)
    return {"obs": obs_batch, "action": action_batch}
