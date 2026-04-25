"""Diffusion policy agent implementation for indexed sequence replay."""

import copy
import os
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
from diffusers.optimization import get_scheduler as get_diffusers_scheduler

from agents.diffusion_policy.common.pytorch_util import optimizer_to
from agents.diffusion_policy.model.diffusion.ema_model import EMAModel
from agents.diffusion_policy.replay_utils import IndexSequenceLoader
from yarr.agents.agent import ActResult, Agent, ScalarSummary, Summary


class DiffusionPolicyAgent(Agent):
    def __init__(self, policy, cfg):
        self.policy = policy
        self.cfg = cfg
        self._device = torch.device("cpu")
        self._optimizer = None
        self._lr_scheduler = None
        self._ema_model = None
        self._ema = None
        self._use_ema = bool(getattr(cfg.method, "use_ema", False))
        self._use_amp = False
        self._amp_dtype = None
        self._grad_scaler = None
        self._update_step = 0
        self.seq_loader = None
        self._episode_index_path = None
        self._summaries = {}
        self._obs_keys = [f"{cam}_rgb" for cam in cfg.method.camera_names] + [
            "low_dim_state"
        ]
        self.reset()

    def set_episode_index_path(self, path: str) -> None:
        self._episode_index_path = path

    def set_aux_eval_cfg(self, cfg) -> None:
        # Explicit no-op for eval runner compatibility.
        del cfg

    def build(self, training: bool, device: torch.device = None) -> None:
        if device is None:
            device = torch.device("cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        self._device = device
        self._use_amp = _should_use_amp(self.cfg, self._device)
        self._amp_dtype = _resolve_amp_dtype(self.cfg) if self._use_amp else None
        _validate_amp_dtype_support(self._device, self._amp_dtype)
        self._grad_scaler = None
        if self._device.type == "cuda":
            _configure_tf32(self.cfg)
        self.policy = self.policy.to(self._device)
        self.policy.train(training)
        self._lr_scheduler = None
        self._optimizer = None
        self._ema = None
        self._ema_model = None
        self._update_step = 0

        if self._use_ema:
            self._ema_model = copy.deepcopy(self.policy).to(self._device)
            self._ema_model.eval()
            self._ema_model.requires_grad_(False)

        if training:
            self.seq_loader = IndexSequenceLoader(
                cfg=self.cfg, episode_index_path=self._episode_index_path
            )
            optimizer_betas = getattr(self.cfg.method, "optimizer_betas", [0.95, 0.999])
            if len(optimizer_betas) != 2:
                raise ValueError(
                    "method.optimizer_betas must contain exactly 2 values, "
                    f"got {optimizer_betas}."
                )
            optimizer_betas = (float(optimizer_betas[0]), float(optimizer_betas[1]))
            model_type = str(getattr(self.cfg.method, "model_type", "unet")).lower()

            if model_type == "transformer" and hasattr(self.policy, "get_optimizer"):
                self._optimizer = self.policy.get_optimizer(
                    transformer_weight_decay=float(
                        getattr(self.cfg.method, "transformer_weight_decay", 0.001)
                    ),
                    obs_encoder_weight_decay=float(
                        getattr(self.cfg.method, "obs_encoder_weight_decay", 0.000001)
                    ),
                    learning_rate=float(self.cfg.method.lr),
                    betas=optimizer_betas,
                )
            else:
                self._optimizer = torch.optim.AdamW(
                    self.policy.parameters(),
                    lr=float(self.cfg.method.lr),
                    weight_decay=float(self.cfg.method.weight_decay),
                    betas=optimizer_betas,
                )
            lr_scheduler_name = str(getattr(self.cfg.method, "lr_scheduler", "none")).lower()
            if lr_scheduler_name not in ("", "none", "null", "false"):
                self._lr_scheduler = get_diffusers_scheduler(
                    lr_scheduler_name,
                    optimizer=self._optimizer,
                    num_warmup_steps=int(getattr(self.cfg.method, "lr_warmup_steps", 0)),
                    num_training_steps=int(self.cfg.framework.training_iterations),
                )
            if self._use_amp and self._amp_dtype == torch.float16:
                self._grad_scaler = torch.amp.GradScaler(device="cuda", enabled=True)
            if self._use_ema:
                self._ema = EMAModel(
                    model=self._ema_model,
                    update_after_step=int(getattr(self.cfg.method, "ema_update_after_step", 0)),
                    inv_gamma=float(getattr(self.cfg.method, "ema_inv_gamma", 1.0)),
                    power=float(getattr(self.cfg.method, "ema_power", 0.75)),
                    min_value=float(getattr(self.cfg.method, "ema_min_value", 0.0)),
                    max_value=float(getattr(self.cfg.method, "ema_max_value", 0.9999)),
                )
        else:
            self.seq_loader = None
            if self._ema_model is not None:
                self._ema_model.load_state_dict(self.policy.state_dict())
        self.reset()

    def reset(self) -> None:
        self._obs_buffer = []
        self._action_cache = None
        self._action_cache_idx = 0

    def update(self, step: int, replay_sample: dict) -> dict:
        del step
        if self._optimizer is None:
            raise RuntimeError("Agent optimizer is not initialized. build(training=True) first.")
        if _is_prebuilt_flat_batch(replay_sample, self._obs_keys):
            batch = _build_training_batch_from_flat_sample(
                replay_sample, obs_keys=self._obs_keys
            )
        else:
            if self.seq_loader is None:
                self.seq_loader = IndexSequenceLoader(
                    cfg=self.cfg, episode_index_path=self._episode_index_path
                )
            episode_ids, timesteps = _extract_index_from_replay_sample(replay_sample)
            batch = self.seq_loader.build_batch(episode_ids, timesteps)
        batch = _to_device(batch, self._device)

        self._optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self._device, self._use_amp, self._amp_dtype):
            loss = self.policy.compute_loss(batch)

        grad_clip = float(getattr(self.cfg.method, "grad_clip", 0.0))
        optimizer_stepped = True
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            if grad_clip > 0:
                self._grad_scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            prev_scale = float(self._grad_scaler.get_scale())
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
            optimizer_stepped = float(self._grad_scaler.get_scale()) >= prev_scale
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            self._optimizer.step()
        if self._lr_scheduler is not None and optimizer_stepped:
            self._lr_scheduler.step()
        if self._ema is not None and optimizer_stepped:
            self._ema.step(self.policy)
        if optimizer_stepped:
            self._update_step += 1

        loss_v = float(loss.detach().float().cpu().item())
        if self._lr_scheduler is not None:
            lr_v = float(self._lr_scheduler.get_last_lr()[0])
        else:
            lr_v = float(self._optimizer.param_groups[0]["lr"])
        self._summaries = {"total_losses": loss_v, "loss": loss_v, "lr": lr_v}
        return dict(self._summaries)

    @torch.no_grad()
    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:
        del step, deterministic
        obs_1step = _extract_latest_obs_step(observation, expected_keys=self._obs_keys)
        self._obs_buffer = _push_obs(
            self._obs_buffer, obs_1step, n_obs_steps=int(self.cfg.method.n_obs_steps)
        )

        refresh_len = 0
        if self._action_cache is not None:
            refresh_len = min(
                int(self.cfg.method.n_action_steps), int(self._action_cache.shape[0])
            )
        need_refresh = self._action_cache is None or self._action_cache_idx >= max(
            1, refresh_len
        )

        if need_refresh:
            obs_batch = _stack_obs_buffer(
                self._obs_buffer, n_obs_steps=int(self.cfg.method.n_obs_steps)
            )
            obs_batch = _to_device(obs_batch, self._device)
            policy = self._ema_model if self._ema_model is not None else self.policy
            pred = policy.predict_action(obs_batch)
            action_seq = pred["action"]
            if not torch.is_tensor(action_seq):
                action_seq = torch.as_tensor(action_seq)
            if action_seq.ndim != 3:
                raise ValueError(
                    f"policy.predict_action()['action'] must be [B,Ta,Da], got {tuple(action_seq.shape)}"
                )
            self._action_cache = action_seq[0].detach()
            self._action_cache_idx = 0

        action = self._action_cache[self._action_cache_idx]
        self._action_cache_idx += 1
        action = _sanitize_bimanual_action(action)
        return ActResult(action.detach().cpu().numpy())

    def update_summaries(self) -> List[Summary]:
        return [
            ScalarSummary("DiffusionPolicyAgent/loss", self._summaries.get("loss", 0.0))
        ]

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str) -> None:
        path = os.path.join(savedir, "diffusion_policy.pt")
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.policy.load_state_dict(state_dict)
        # Keep all policy submodules (including custom normalizer params) on runtime device.
        self.policy = self.policy.to(self._device)
        if self._ema_model is not None:
            self._ema_model = self._ema_model.to(self._device)
            self._ema_model.eval()
            self._ema_model.requires_grad_(False)

        train_state_path = os.path.join(savedir, "diffusion_policy_train_state.pt")
        if os.path.isfile(train_state_path):
            train_state = torch.load(train_state_path, map_location=torch.device("cpu"))
            if self._optimizer is not None and "optimizer" in train_state:
                self._optimizer.load_state_dict(train_state["optimizer"])
                optimizer_to(self._optimizer, self._device)
            if self._grad_scaler is not None and "grad_scaler" in train_state:
                self._grad_scaler.load_state_dict(train_state["grad_scaler"])
            if self._lr_scheduler is not None and "lr_scheduler" in train_state:
                self._lr_scheduler.load_state_dict(train_state["lr_scheduler"])
            if self._ema_model is not None and "ema_model" in train_state:
                self._ema_model.load_state_dict(train_state["ema_model"])
            elif self._ema_model is not None:
                self._ema_model.load_state_dict(self.policy.state_dict())
            if self._ema is not None:
                ema_state = train_state.get("ema_helper", {})
                if "decay" in ema_state:
                    self._ema.decay = float(ema_state["decay"])
                if "optimization_step" in ema_state:
                    self._ema.optimization_step = int(ema_state["optimization_step"])
            self._update_step = int(train_state.get("update_step", 0))
        else:
            self._update_step = 0
            if self._ema_model is not None:
                self._ema_model.load_state_dict(self.policy.state_dict())

        # DictOfTensorMixin custom loading can recreate normalizer params on CPU.
        # Force all submodules (policy + EMA policy) back to runtime device after load.
        self.policy = self.policy.to(self._device)
        if self._ema_model is not None:
            self._ema_model = self._ema_model.to(self._device)
            self._ema_model.eval()
            self._ema_model.requires_grad_(False)

    def save_weights(self, savedir: str) -> None:
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, "diffusion_policy.pt")
        torch.save(self.policy.state_dict(), path)
        train_state = {"update_step": int(self._update_step)}
        if self._optimizer is not None:
            train_state["optimizer"] = self._optimizer.state_dict()
        if self._grad_scaler is not None:
            train_state["grad_scaler"] = self._grad_scaler.state_dict()
        if self._lr_scheduler is not None:
            train_state["lr_scheduler"] = self._lr_scheduler.state_dict()
        if self._ema_model is not None:
            train_state["ema_model"] = self._ema_model.state_dict()
        if self._ema is not None:
            train_state["ema_helper"] = {
                "decay": float(self._ema.decay),
                "optimization_step": int(self._ema.optimization_step),
            }
        train_state_path = os.path.join(savedir, "diffusion_policy_train_state.pt")
        torch.save(train_state, train_state_path)


def _extract_index_from_replay_sample(replay_sample: Dict[str, torch.Tensor]):
    episode_ids = _to_numpy_index(replay_sample["episode_id"])
    timesteps = _to_numpy_index(replay_sample["timestep"])
    return episode_ids, timesteps


def _is_prebuilt_flat_batch(replay_sample: Dict, obs_keys: List[str]) -> bool:
    return (
        isinstance(replay_sample, dict)
        and "action" in replay_sample
        and "is_pad" in replay_sample
        and all(key in replay_sample for key in obs_keys)
    )


def _build_training_batch_from_flat_sample(
    replay_sample: Dict[str, torch.Tensor], obs_keys: List[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    obs_batch = {
        key: _strip_optional_single_task_dim(
            replay_sample[key], expected_ndim=(5 if key.endswith("_rgb") else 3)
        )
        for key in obs_keys
    }
    return {
        "obs": obs_batch,
        "action": _strip_optional_single_task_dim(replay_sample["action"], expected_ndim=3),
        "is_pad": _strip_optional_single_task_dim(replay_sample["is_pad"], expected_ndim=2),
    }


def _strip_optional_single_task_dim(value, expected_ndim: int):
    if torch.is_tensor(value):
        if value.ndim == expected_ndim + 1 and value.shape[1] == 1:
            return value[:, 0]
        return value

    arr = np.asarray(value)
    if arr.ndim == expected_ndim + 1 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _to_numpy_index(index_tensor) -> np.ndarray:
    if torch.is_tensor(index_tensor):
        arr = index_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(index_tensor)

    if arr.ndim == 2:
        arr = arr[:, -1]
    elif arr.ndim != 1:
        raise ValueError(f"Expected index ndim 1 or 2, got shape={arr.shape}")
    return arr.astype(np.int64, copy=False)


def _extract_latest_obs_step(observation: Dict[str, torch.Tensor], expected_keys: List[str]):
    out = {}
    for key, value in observation.items():
        if key in ("lang_goal", "lang_goal_tokens"):
            continue

        x = value if torch.is_tensor(value) else torch.as_tensor(value)
        if key.endswith("_rgb"):
            if x.ndim == 5:
                x = x[:, -1]
            elif x.ndim == 3:
                x = x.unsqueeze(0)
            elif x.ndim != 4:
                raise ValueError(f"Unexpected rgb tensor shape for {key}: {tuple(x.shape)}")
        else:
            if x.ndim >= 3:
                x = x[:, -1]
            elif x.ndim == 1:
                x = x.unsqueeze(0)
            elif x.ndim == 0:
                x = x.reshape(1, 1)
        out[key] = x

    proprio = _build_action_equivalent_lowdim(out)
    if proprio is not None:
        out["low_dim_state"] = proprio
    elif "low_dim_state" not in out:
        raise KeyError(
            "Missing action-equivalent proprioception keys. "
            "Expected right/left gripper pose+open or low_dim_state."
        )

    if out["low_dim_state"].shape[-1] != 18:
        raise ValueError(
            f"DIFFUSION_POLICY expects 18D low_dim_state, got {tuple(out['low_dim_state'].shape)}."
        )

    out.pop("ignore_collisions", None)
    out.pop("right_ignore_collisions", None)
    out.pop("left_ignore_collisions", None)
    out.pop("right_low_dim_state", None)
    out.pop("left_low_dim_state", None)

    filtered = {}
    for key in expected_keys:
        if key not in out:
            raise KeyError(f"Missing observation key '{key}' for DIFFUSION_POLICY act().")
        x = out[key].to(dtype=torch.float32)
        if key.endswith("_rgb"):
            # Runtime rollout RGB is typically uint8 [0,255]; training loader is [0,1].
            if x.max().item() > 1.5:
                x = x / 255.0
        filtered[key] = x
    return filtered


def _build_action_equivalent_lowdim(out: Dict[str, torch.Tensor]):
    required = (
        "right_gripper_pose",
        "right_gripper_open",
        "left_gripper_pose",
        "left_gripper_open",
    )
    if any(key not in out for key in required):
        return None

    right_pos, right_quat = _split_pose_and_quat(
        out["right_gripper_pose"], key_name="right_gripper_pose"
    )
    left_pos, left_quat = _split_pose_and_quat(
        out["left_gripper_pose"], key_name="left_gripper_pose"
    )
    right_open = _extract_scalar_column(out["right_gripper_open"], key_name="right_gripper_open")
    left_open = _extract_scalar_column(out["left_gripper_open"], key_name="left_gripper_open")

    batch_size = int(right_pos.shape[0])
    device = right_pos.device
    dtype = right_pos.dtype
    right_ignore, left_ignore = _extract_ignore_collision_columns(
        out=out, batch_size=batch_size, device=device, dtype=dtype
    )

    right_open = right_open.to(device=device, dtype=dtype)
    left_open = left_open.to(device=device, dtype=dtype)
    left_pos = left_pos.to(device=device, dtype=dtype)
    left_quat = left_quat.to(device=device, dtype=dtype)

    return torch.cat(
        [
            right_pos,
            right_quat,
            right_open,
            right_ignore,
            left_pos,
            left_quat,
            left_open,
            left_ignore,
        ],
        dim=-1,
    )


def _split_pose_and_quat(pose_tensor: torch.Tensor, key_name: str):
    x = _flatten_batch_feature(pose_tensor, key_name=key_name)
    if x.shape[-1] < 7:
        raise ValueError(f"{key_name} requires at least 7 dims, got {tuple(x.shape)}")
    pos = x[:, :3]
    quat = _normalize_and_canonicalize_quat(x[:, 3:7], key_name=key_name)
    return pos, quat


def _normalize_and_canonicalize_quat(quat: torch.Tensor, key_name: str) -> torch.Tensor:
    quat = quat.to(dtype=torch.float32)
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError(f"{key_name} has invalid quaternion with near-zero norm.")
    quat = quat / torch.clamp(norm, min=1e-8)
    quat = torch.where(quat[:, -1:] < 0, -quat, quat)
    return quat


def _extract_ignore_collision_columns(
    out: Dict[str, torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype
):
    right_ignore = None
    left_ignore = None
    if "right_ignore_collisions" in out:
        right_ignore = _extract_scalar_column(
            out["right_ignore_collisions"], key_name="right_ignore_collisions"
        )
    if "left_ignore_collisions" in out:
        left_ignore = _extract_scalar_column(
            out["left_ignore_collisions"], key_name="left_ignore_collisions"
        )
    if right_ignore is not None and left_ignore is not None:
        return right_ignore.to(device=device, dtype=dtype), left_ignore.to(
            device=device, dtype=dtype
        )

    if "ignore_collisions" in out:
        shared_ignore = _extract_scalar_column(
            out["ignore_collisions"], key_name="ignore_collisions"
        ).to(device=device, dtype=dtype)
        return shared_ignore, shared_ignore

    ones = torch.ones((batch_size, 1), device=device, dtype=dtype)
    return ones, ones


def _extract_scalar_column(x: torch.Tensor, key_name: str) -> torch.Tensor:
    flattened = _flatten_batch_feature(x, key_name=key_name)
    if flattened.shape[-1] < 1:
        raise ValueError(f"{key_name} has empty feature dimension.")
    return flattened[:, :1]


def _flatten_batch_feature(x: torch.Tensor, key_name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    else:
        x = x.reshape(x.shape[0], -1)
    if x.ndim != 2:
        raise ValueError(f"{key_name} must be 2D after flatten, got {tuple(x.shape)}")
    return x.to(dtype=torch.float32)


def _push_obs(
    buffer: List[Dict[str, torch.Tensor]],
    obs_1step: Dict[str, torch.Tensor],
    n_obs_steps: int,
):
    buffer.append({k: v.detach().cpu() for k, v in obs_1step.items()})
    if len(buffer) > n_obs_steps:
        buffer = buffer[-n_obs_steps:]
    return buffer


def _stack_obs_buffer(
    buffer: List[Dict[str, torch.Tensor]], n_obs_steps: int
) -> Dict[str, torch.Tensor]:
    if len(buffer) == 0:
        raise RuntimeError("Observation buffer is empty.")

    if len(buffer) < n_obs_steps:
        pad = [buffer[0]] * (n_obs_steps - len(buffer))
        frames = pad + buffer
    else:
        frames = buffer[-n_obs_steps:]

    out = {}
    for key in frames[0].keys():
        out[key] = torch.stack([frame[key] for frame in frames], dim=1)
    return out


def _to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}

    if torch.is_tensor(batch):
        if batch.dtype.is_floating_point:
            return batch.to(device=device, dtype=torch.float32, non_blocking=True)
        return batch.to(device=device, non_blocking=True)

    arr = np.asarray(batch)
    tensor = torch.as_tensor(arr, device=device)
    if tensor.dtype.is_floating_point:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def _should_use_amp(cfg, device: torch.device) -> bool:
    return device.type == "cuda" and bool(getattr(cfg.method, "use_amp", False))


def _resolve_amp_dtype(cfg) -> torch.dtype:
    amp_dtype = str(getattr(cfg.method, "amp_dtype", "bf16")).lower()
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError(
        f"Unsupported method.amp_dtype='{amp_dtype}'. Expected 'bf16' or 'fp16'."
    )


def _validate_amp_dtype_support(device: torch.device, amp_dtype) -> None:
    if device.type != "cuda" or amp_dtype != torch.bfloat16:
        return
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "method.amp_dtype='bf16' requires CUDA bf16 support on the current device. "
            "Use method.amp_dtype='fp16' instead."
        )


def _configure_tf32(cfg) -> None:
    enable_tf32 = bool(getattr(cfg.method, "enable_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")


def _autocast_context(device: torch.device, enabled: bool, amp_dtype):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def _sanitize_bimanual_action(action: torch.Tensor) -> torch.Tensor:
    a = action.detach().to(dtype=torch.float32).clone().reshape(-1)
    if a.numel() != 18:
        return a

    a[3:7] = _normalize_quaternion_1d(a[3:7])
    a[12:16] = _normalize_quaternion_1d(a[12:16])

    # Match dataset/observation canonicalization (w >= 0).
    if a[6].item() < 0.0:
        a[3:7] = -a[3:7]
    if a[15].item() < 0.0:
        a[12:16] = -a[12:16]

    # RLBench expects binary gripper/ignore-collisions controls.
    for idx in (7, 8, 16, 17):
        v = a[idx]
        if not torch.isfinite(v):
            a[idx] = torch.tensor(1.0, device=a.device, dtype=a.dtype)
        else:
            a[idx] = torch.tensor(
                1.0 if float(v.item()) > 0.5 else 0.0, device=a.device, dtype=a.dtype
            )
    return a


def _normalize_quaternion_1d(q: torch.Tensor) -> torch.Tensor:
    q64 = q.to(dtype=torch.float64)
    norm = torch.linalg.norm(q64)
    if (not torch.isfinite(norm)) or norm.item() < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=q.device)
    qn = (q64 / norm).to(dtype=torch.float32)
    qn_norm = torch.linalg.norm(qn)
    if (not torch.isfinite(qn_norm)) or qn_norm.item() < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=q.device)
    return qn / torch.clamp(qn_norm, min=1e-8)
