from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from agents.diffusion_policy.model.common.normalizer import LinearNormalizer
from agents.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from agents.diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from agents.diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from agents.diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from agents.diffusion_policy.common.pytorch_util import dict_apply


class DiffusionTransformerImagePolicy(BaseImagePolicy):
    """MultiImageObsEncoder + TransformerForDiffusion (obs_as_cond=True only)."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_cond=True,
        pred_action_steps_only=False,
        n_layer=8,
        n_cond_layers=0,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
        time_as_cond=True,
        **kwargs,
    ):
        super().__init__()

        if not obs_as_cond:
            raise ValueError(
                "DiffusionTransformerImagePolicy supports only obs_as_cond=True."
            )
        if pred_action_steps_only and not obs_as_cond:
            raise ValueError("pred_action_steps_only=True requires obs_as_cond=True.")

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = int(action_shape[0])
        obs_feature_dim = int(obs_encoder.output_shape()[0])

        model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.obs_feature_dim = int(obs_feature_dim)
        self.action_dim = int(action_dim)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_cond = True
        self.pred_action_steps_only = bool(pred_action_steps_only)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = int(num_inference_steps)

    # ========= inference ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        cond=None,
        generator=None,
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, _ = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        cond = nobs_features.reshape(B, To, -1)

        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=float(transformer_weight_decay))
        optim_groups.append(
            {
                "params": self.obs_encoder.parameters(),
                "weight_decay": float(obs_encoder_weight_decay),
            }
        )
        optimizer = torch.optim.AdamW(
            optim_groups, lr=float(learning_rate), betas=tuple(betas)
        )
        return optimizer

    def compute_loss(self, batch):
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        To = self.n_obs_steps

        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        cond = nobs_features.reshape(batch_size, To, -1)

        trajectory = nactions
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:, start:end]

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
