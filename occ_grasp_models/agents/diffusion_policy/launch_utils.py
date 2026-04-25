"""Launch utilities for DIFFUSION_POLICY."""

import logging

import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import DictConfig
from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from agents.diffusion_policy.agent import DiffusionPolicyAgent
from agents.diffusion_policy.configs import shape_meta_utils
from agents.diffusion_policy.model.vision import model_getter
from agents.diffusion_policy.model.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)
from agents.diffusion_policy.policy.diffusion_transformer_image_policy import (
    DiffusionTransformerImagePolicy,
)
from agents.diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy,
)
from agents.diffusion_policy import replay_utils
from helpers.preprocess_agent import PreprocessAgent


def create_agent(cfg: DictConfig):
    """Create DIFFUSION_POLICY agent with UNet/Transformer backbone switch."""
    action_dim = int(cfg.method.action_dim)
    low_dim_size = int(cfg.method.low_dim_size)
    robot_name = str(getattr(cfg.method, "robot_name", "bimanual"))
    if low_dim_size != action_dim:
        raise ValueError(
            "DIFFUSION_POLICY requires low_dim_size == action_dim "
            f"(got low_dim_size={low_dim_size}, action_dim={action_dim})."
        )
    if robot_name != "bimanual":
        raise ValueError(
            "DIFFUSION_POLICY currently supports only robot_name='bimanual' "
            f"(got '{robot_name}')."
        )

    shape_meta = shape_meta_utils.build_shape_meta(
        camera_names=cfg.method.camera_names,
        image_size=cfg.method.image_size,
        low_dim_size=low_dim_size,
        action_dim=action_dim,
    )

    model_type = str(getattr(cfg.method, "model_type", "unet")).lower()
    obs_encoder = _build_obs_encoder(cfg, shape_meta)
    noise_scheduler = _build_ddpm_scheduler(cfg)
    if model_type == "unet":
        policy = DiffusionUnetImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=cfg.method.horizon,
            n_action_steps=cfg.method.n_action_steps,
            n_obs_steps=cfg.method.n_obs_steps,
            obs_as_global_cond=cfg.method.obs_as_global_cond,
            num_inference_steps=cfg.method.num_inference_steps,
            diffusion_step_embed_dim=int(
                getattr(cfg.method, "diffusion_step_embed_dim", 128)
            ),
            down_dims=tuple(getattr(cfg.method, "down_dims", [512, 1024, 2048])),
            kernel_size=int(getattr(cfg.method, "kernel_size", 5)),
            n_groups=int(getattr(cfg.method, "n_groups", 8)),
            cond_predict_scale=bool(getattr(cfg.method, "cond_predict_scale", True)),
        )
    elif model_type == "transformer":
        obs_as_cond = bool(getattr(cfg.method, "obs_as_cond", True))
        if not obs_as_cond:
            raise ValueError(
                "DIFFUSION_POLICY transformer route supports only obs_as_cond=True."
            )
        time_as_cond = bool(getattr(cfg.method, "time_as_cond", True))
        if not time_as_cond:
            raise ValueError(
                "DIFFUSION_POLICY transformer route requires time_as_cond=True."
            )
        policy = DiffusionTransformerImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=cfg.method.horizon,
            n_action_steps=cfg.method.n_action_steps,
            n_obs_steps=cfg.method.n_obs_steps,
            num_inference_steps=cfg.method.num_inference_steps,
            obs_as_cond=obs_as_cond,
            pred_action_steps_only=bool(
                getattr(cfg.method, "pred_action_steps_only", False)
            ),
            n_layer=int(getattr(cfg.method, "n_layer", 8)),
            n_cond_layers=int(getattr(cfg.method, "n_cond_layers", 0)),
            n_head=int(getattr(cfg.method, "n_head", 4)),
            n_emb=int(getattr(cfg.method, "n_emb", 256)),
            p_drop_emb=float(getattr(cfg.method, "p_drop_emb", 0.0)),
            p_drop_attn=float(getattr(cfg.method, "p_drop_attn", 0.3)),
            causal_attn=bool(getattr(cfg.method, "causal_attn", True)),
            time_as_cond=time_as_cond,
        )
    else:
        raise NotImplementedError(
            f"Unsupported model_type '{cfg.method.model_type}'. "
            "Expected one of ['unet', 'transformer']."
        )
    # Keep diffusion policy's internal RGB normalization behavior unchanged.
    return PreprocessAgent(
        pose_agent=DiffusionPolicyAgent(policy=policy, cfg=cfg),
        norm_rgb=False,
    )


def _build_ddpm_scheduler(cfg: DictConfig):
    del cfg
    return DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )


def _build_obs_encoder(cfg: DictConfig, shape_meta: dict):
    rgb_model = model_getter.get_resnet(name=cfg.method.rgb_backbone, weights=None)
    return MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=cfg.method.resize_shape,
        crop_shape=cfg.method.crop_shape,
        random_crop=cfg.method.random_crop,
        use_group_norm=cfg.method.use_group_norm,
        share_rgb_model=cfg.method.share_rgb_model,
        imagenet_norm=cfg.method.imagenet_norm,
    )


def create_replay(
    batch_size: int,
    timesteps: int,
    prioritisation: bool,
    task_uniform: bool,
    save_dir: str,
    cameras: list,
    image_size,
    replay_size=3e5,
):
    del prioritisation, task_uniform, cameras, image_size
    observation_elements = [
        ObservationElement("episode_id", (), np.int32),
        ObservationElement("timestep", (), np.int32),
    ]
    extra_replay_elements = [
        ReplayElement("task", (), str),
        ReplayElement("demo", (), bool),
    ]
    return TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(1,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements,
    )


def fill_multi_task_replay(
    cfg: DictConfig,
    obs_config,
    rank: int,
    replay,
    tasks,
    num_demos: int,
    demo_augmentation=None,
    demo_augmentation_every_n=None,
    cameras=None,
):
    del obs_config, demo_augmentation, demo_augmentation_every_n, cameras
    dataset_root = getattr(cfg.method, "train_demo_path", cfg.rlbench.demo_path)
    if rank == 0:
        logging.info(
            "DIFFUSION_POLICY building indexed replay | dataset_root=%s | tasks=%s | demos_per_task=%d",
            dataset_root,
            list(tasks),
            int(num_demos),
        )
    episode_index = replay_utils.build_episode_index(
        dataset_root=dataset_root,
        tasks=tasks,
        num_demos=int(num_demos),
        index_seed=getattr(cfg.method, "index_seed", None),
    )
    index_path = replay_utils.save_episode_index(
        episode_index=episode_index,
        replay_dir=getattr(replay, "_save_dir", None),
        fallback_dir=cfg.framework.logdir,
    )
    if rank == 0:
        logging.info("DIFFUSION_POLICY episode index saved to %s", index_path)
        logging.info("DIFFUSION_POLICY indexing %d episodes", len(episode_index))
        per_task_demo_idx = {}
        for info in episode_index:
            task_name = str(info.task)
            demo_idx = per_task_demo_idx.get(task_name, 0)
            per_task_demo_idx[task_name] = demo_idx + 1
            logging.info(
                "DIFFUSION_POLICY indexed demo(%d) | task=%s | episode_id=%d | frames=%d | path=%s",
                demo_idx,
                task_name,
                int(info.episode_id),
                int(info.length),
                info.path,
            )

    action_dummy = np.zeros((1,), dtype=np.float32)
    horizon = int(cfg.method.horizon)
    n_obs_steps = int(cfg.method.n_obs_steps)
    n_action_steps = int(cfg.method.n_action_steps)
    n_latency_steps = int(getattr(cfg.method, "n_latency_steps", 0))
    pad_before = n_obs_steps - 1 + n_latency_steps
    pad_after = n_action_steps - 1
    total_steps = 0
    for info in episode_index:
        starts = list(
            replay_utils.iter_logical_starts(
                length=int(info.length),
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
            )
        )
        if len(starts) == 0:
            if rank == 0:
                logging.warning(
                    "DIFFUSION_POLICY replay indexing skipped | episode_id=%d | task=%s | length=%d | no logical starts (horizon=%d, pad_before=%d, pad_after=%d)",
                    int(info.episode_id),
                    info.task,
                    int(info.length),
                    horizon,
                    pad_before,
                    pad_after,
                )
            continue
        if rank == 0:
            logging.info(
                "DIFFUSION_POLICY replay indexing | episode_id=%d | task=%s | indexed_steps=%d",
                int(info.episode_id),
                info.task,
                int(len(starts)),
            )
        for i, t in enumerate(starts):
            terminal = i == (len(starts) - 1)
            replay.add(
                action_dummy,
                0.0,
                terminal,
                False,
                episode_id=np.int32(info.episode_id),
                timestep=np.int32(int(t)),
                task=info.task,
                demo=True,
            )
            total_steps += 1
    if rank == 0:
        logging.info(
            "DIFFUSION_POLICY indexed replay ready | episodes=%d | transitions=%d",
            len(episode_index),
            total_steps,
        )
    return index_path
