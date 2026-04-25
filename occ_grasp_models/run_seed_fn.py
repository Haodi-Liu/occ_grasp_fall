import os
import pickle
import gc
from typing import List
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from agents import agent_factory
from agents import replay_utils

import peract_config
from functools import partial

def run_seed(
    rank,
    cfg: DictConfig,
    obs_config: ObservationConfig,
    seed,
    world_size,
) -> None:
    

    peract_config.config_logging()

    with open_dict(cfg):
        cfg.seed = int(seed)
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    tasks = cfg.rlbench.tasks
    cams = cfg.rlbench.cameras

    task_folder = "multi" if len(tasks) > 1 else tasks[0] 
    replay_path = os.path.join(
        cfg.replay.path, task_folder, cfg.method.name, "seed%d" % seed
    )

    if cfg.method.name == "PPI":
        raise NotImplementedError(
            "Use train_ppi_ddp.py instead of train.py method=PPI."
        )

    agent = agent_factory.create_agent(cfg)
    wrapped_replay_kwargs = {"num_workers": int(cfg.framework.num_workers)}

    if not agent:
        print("Unable to create agent")
        return

    if cfg.method.name == "ACT_BC_ENC_KEYPOINT_STRATEGY":
        # 注意：必须在其他 ACT_BC_ENC 变体之前匹配
        from agents import act_bc_enc_keypoint_strategy

        assert cfg.ddp.num_devices == 1, "ACT_BC_ENC_KEYPOINT_STRATEGY only supports single GPU training"
        replay_buffer = act_bc_enc_keypoint_strategy.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            prev_action_horizon=cfg.method.prev_action_horizon,
            next_action_horizon=cfg.method.next_action_horizon
        )

        act_bc_enc_keypoint_strategy.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name == "ACT_BC_ENC_STRATEGY":
        # 注意：必须在 ACT_BC_ENC 之前匹配，否则会被 startswith("ACT_BC_ENC") 错误捕获
        from agents import act_bc_enc_strategy

        assert cfg.ddp.num_devices == 1, "ACT_BC_ENC_STRATEGY only supports single GPU training"
        replay_buffer = act_bc_enc_strategy.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            prev_action_horizon=cfg.method.prev_action_horizon,
            next_action_horizon=cfg.method.next_action_horizon
        )

        act_bc_enc_strategy.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name == "ACT_BC_ENC_KEYPOINT":
        # 注意：必须在 ACT_BC_ENC 之前匹配，否则会被 startswith("ACT_BC_ENC") 错误捕获
        from agents import act_bc_enc_keypoint

        assert cfg.ddp.num_devices == 1, "ACT_BC_ENC_KEYPOINT only supports single GPU training"
        replay_buffer = act_bc_enc_keypoint.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            prev_action_horizon=cfg.method.prev_action_horizon,
            next_action_horizon=cfg.method.next_action_horizon
        )

        act_bc_enc_keypoint.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name == "ACT_BC_KEYPOINT_STRATEGY":
        from agents import act_bc_keypoint_strategy

        assert cfg.ddp.num_devices == 1, "ACT_BC_KEYPOINT_STRATEGY only supports single GPU training"
        replay_buffer = act_bc_keypoint_strategy.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            keyframe_seq_len=cfg.method.keyframe_seq_len,
        )

        act_bc_keypoint_strategy.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name == "ACT_BC_KEYPOINT":
        from agents import act_bc_keypoint

        assert cfg.ddp.num_devices == 1, "ACT_BC_KEYPOINT only supports single GPU training"
        replay_buffer = act_bc_keypoint.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            keyframe_seq_len=cfg.method.keyframe_seq_len,
        )

        act_bc_keypoint.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name == "ACT_BC_KEY":
        from agents import act_bc_key

        assert cfg.ddp.num_devices == 1, "ACT_BC_KEY only supports single GPU training"
        replay_buffer = act_bc_key.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            keyframe_seq_len=cfg.method.keyframe_seq_len
        )

        act_bc_key.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name.startswith("ACT_BC_ENC"):
        from agents import act_bc_enc

        assert cfg.ddp.num_devices == 1, "ACT_BC_ENC only supports single GPU training"
        replay_buffer = act_bc_enc.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            prev_action_horizon=cfg.method.prev_action_horizon,
            next_action_horizon=cfg.method.next_action_horizon
        )

        act_bc_enc.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name.startswith("ACT_BC_VISION"):
        from agents import act_bc_vision

        assert cfg.ddp.num_devices == 1, "ACT_BC_VISION only supports single GPU training"
        replay_buffer = act_bc_vision.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            prev_action_horizon=cfg.method.prev_action_horizon,
            next_action_horizon=cfg.method.next_action_horizon
        )

        act_bc_vision.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name == "DIFFUSION_POLICY":
        from agents import diffusion_policy
        from agents.diffusion_policy.normalizer_utils import (
            fit_normalizer_from_index_replay,
        )
        from agents.diffusion_policy.replay_utils import ReplaySampleToBatchTransform

        replay_buffer = diffusion_policy.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
        )

        episode_index_path = diffusion_policy.launch_utils.fill_multi_task_replay(
            cfg=cfg,
            obs_config=obs_config,
            rank=rank,
            replay=replay_buffer,
            tasks=tasks,
            num_demos=cfg.rlbench.demos,
        )

        if hasattr(agent, "set_episode_index_path"):
            agent.set_episode_index_path(episode_index_path)

        fitted_normalizer = fit_normalizer_from_index_replay(
            cfg=cfg,
            replay_buffer=replay_buffer,
            sample_size=int(getattr(cfg.method, "normalizer_sample_size", 10000)),
            device=f"cuda:{rank}",
            episode_index_path=episode_index_path,
        )
        if hasattr(agent, "policy"):
            agent.policy.set_normalizer(fitted_normalizer)
        elif hasattr(agent, "_pose_agent") and hasattr(agent._pose_agent, "policy"):
            agent._pose_agent.policy.set_normalizer(fitted_normalizer)

        if bool(getattr(cfg.method, "move_batch_build_to_worker", False)):
            wrapped_replay_kwargs = {
                "num_workers": int(
                    getattr(cfg.method, "batch_builder_workers", cfg.framework.num_workers)
                ),
                "sample_transform": ReplaySampleToBatchTransform(
                    cfg=cfg,
                    episode_index_path=episode_index_path,
                ),
                "pin_memory": bool(getattr(cfg.method, "batch_builder_pin_memory", True)),
                "persistent_workers": bool(
                    getattr(cfg.method, "batch_builder_persistent_workers", True)
                ),
                "prefetch_factor": int(
                    getattr(cfg.method, "batch_builder_prefetch_factor", 2)
                ),
            }

    elif cfg.method.name.startswith("BIMANUAL_PERACT"):

        replay_buffer = replay_utils.create_replay(cfg, replay_path)

        replay_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks
        )

    else:
        raise ValueError("Method %s does not exists." % cfg.method.name)

    wrapped_replay = PyTorchReplayBuffer(replay_buffer, **wrapped_replay_kwargs)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, "seed%d" % seed, "weights")
    logdir = os.path.join(cwd, "seed%d" % seed)

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
    )

    train_runner._on_thread_start = partial(peract_config.config_logging, cfg.framework.logging_level)
    
    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()
