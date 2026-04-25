"""Normalizer fitting utilities for DIFFUSION_POLICY."""

from __future__ import annotations

import numpy as np

from agents.diffusion_policy.common.normalize_util import (
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
)
from agents.diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from agents.diffusion_policy.replay_utils import IndexSequenceLoader


def fit_normalizer_from_index_replay(
    cfg,
    replay_buffer,
    sample_size: int = 10000,
    device: str = "cpu",
    episode_index_path: str = None,
) -> LinearNormalizer:
    del device
    sample_size = max(1, int(sample_size))
    loader = IndexSequenceLoader(cfg=cfg, episode_index_path=episode_index_path)

    action_chunks = []
    lowdim_chunks = []
    sampled = 0
    per_batch = _infer_batch_size(replay_buffer, fallback=128)

    while sampled < sample_size:
        this_batch = min(per_batch, sample_size - sampled)
        replay_sample = replay_buffer.sample_transition_batch(
            batch_size=this_batch, pack_in_dict=True
        )
        episode_ids = _extract_index_vector(replay_sample["episode_id"])
        timesteps = _extract_index_vector(replay_sample["timestep"])

        actions, lowdims = loader.sample_action_lowdim_stats(episode_ids, timesteps)
        action_chunks.append(actions.astype(np.float32, copy=False))
        lowdim_chunks.append(lowdims.astype(np.float32, copy=False))
        sampled += this_batch

    action_array = np.concatenate(action_chunks, axis=0)
    lowdim_array = np.concatenate(lowdim_chunks, axis=0)
    return _build_normalizer(cfg, action_array, lowdim_array)


def _build_normalizer(cfg, action_array: np.ndarray, lowdim_array: np.ndarray) -> LinearNormalizer:
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
        action_array, mode="limits", last_n_dims=1
    )
    normalizer["low_dim_state"] = SingleFieldLinearNormalizer.create_fit(
        lowdim_array, mode="limits", last_n_dims=1
    )

    image_norm_identity = bool(getattr(cfg.method, "imagenet_norm", True))
    image_stat = _unit_image_stats()
    for cam in cfg.method.camera_names:
        key = f"{cam}_rgb"
        if image_norm_identity:
            normalizer[key] = get_identity_normalizer_from_stat(image_stat)
        else:
            normalizer[key] = get_image_range_normalizer()
    return normalizer


def _extract_index_vector(index_array) -> np.ndarray:
    arr = np.asarray(index_array)
    if arr.ndim == 1:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 2:
        return arr[:, -1].astype(np.int64, copy=False)
    raise ValueError(f"Unexpected index tensor shape: {arr.shape}")


def _infer_batch_size(replay_buffer, fallback: int = 128) -> int:
    for attr in ("_batch_size", "batch_size"):
        value = getattr(replay_buffer, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value is not None:
            return max(1, int(value))
    return max(1, int(fallback))


def _unit_image_stats():
    return {
        "min": np.array([0.0], dtype=np.float32),
        "max": np.array([1.0], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1.0 / 12.0)], dtype=np.float32),
    }
