from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from agents.diffusion_policy.model.common.normalizer import LinearNormalizer
from agents.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from agents.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from agents.diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from agents.diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from agents.diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_feature_projection_dim=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # ===== 解析动作维度与观测编码维度 =====
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]  # Da
        # obs_encoder 输出的单帧特征维度（Do）
        raw_obs_feature_dim = obs_encoder.output_shape()[0]
        if obs_feature_projection_dim is None:
            obs_feature_dim = raw_obs_feature_dim
            obs_feature_projector = nn.Identity()
        else:
            obs_feature_dim = int(obs_feature_projection_dim)
            if obs_feature_dim <= 0:
                raise ValueError(
                    "obs_feature_projection_dim must be positive or None, "
                    f"got {obs_feature_projection_dim}."
                )
            obs_feature_projector = nn.Sequential(
                nn.Linear(raw_obs_feature_dim, obs_feature_dim),
                nn.LayerNorm(obs_feature_dim),
                nn.Mish(),
            )

        # ===== 构建扩散模型（ConditionalUnet1D） =====
        # 扩散对象维度：
        # - obs_as_global_cond=True  -> 仅动作序列 (Da)
        # - obs_as_global_cond=False -> 动作 + obs特征序列 (Da + Do)
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            # global_cond = 将 n_obs_steps 帧特征拼接成一个长向量
            # 形状：(B, n_obs_steps * Do)
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.obs_feature_projector = obs_feature_projector
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.raw_obs_feature_dim = raw_obs_feature_dim
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _encode_obs_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.obs_feature_projector(self.obs_encoder(obs_dict))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        # condition_data 形状与扩散对象一致：
        # - obs_as_global_cond=True  -> (B, T, Da)
        # - obs_as_global_cond=False -> (B, T, Da + Do)
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input（对每个 obs key 做归一化）
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            # 将 (B, To, ...) 拉平成 (B*To, ...)，逐帧编码
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self._encode_obs_features(this_nobs)
            # (B*To, Do) -> (B, To*Do)
            global_cond = nobs_features.reshape(B, -1)
            # 扩散对象仅为动作：cond_data = (B, T, Da)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            # 逐帧编码后再 reshape 成时序特征
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self._encode_obs_features(this_nobs)
            # (B*To, Do) -> (B, To, Do)
            nobs_features = nobs_features.reshape(B, To, -1)
            # 扩散对象包含动作 + obs特征
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            # 把前 To 步的 obs 特征写入轨迹并固定（inpainting）
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        # 注意：这里从 To-1 开始取动作窗口（与 video 策略不同）
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        # trajectory 是扩散对象：
        # - obs_as_global_cond=True  -> 仅动作序列 (B, T, Da)
        # - obs_as_global_cond=False -> 动作 + obs特征序列 (B, T, Da+Do)
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self._encode_obs_features(this_nobs)
            # (B*To, Do) -> (B, To*Do) 作为 global_cond
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self._encode_obs_features(this_nobs)
            # (B*T, Do) -> (B, T, Do)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            # 将 obs 特征拼入轨迹作为 inpainting 约束
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
