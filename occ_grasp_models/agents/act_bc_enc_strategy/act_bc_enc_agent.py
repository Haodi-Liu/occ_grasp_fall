"""
ACT BC Enc Agent with Strategy and Phase Conditioning

This agent extends ActBCEncAgent with:
1. Strategy and phase type handling in update() for training
2. Strategy and phase type handling in act() for inference
3. Extended summaries for classification metrics
"""
import copy
import logging
from functools import lru_cache
import pickle
import os
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary

from helpers import utils
from helpers.utils import stack_on_channel

NAME = 'ActBCEncStrategyAgent'


class ActBCEncAgent(Agent):

    def __init__(self,
                 actor_network: nn.Module,
                 camera_names: List[str],
                 lr: float = 0.01,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 20.0,
                 episode_length: int = 400, train_demo_path=None, task_name=None):
        self._camera_names = camera_names
        self._actor = actor_network
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._episode_length = episode_length
        self.train_demo_path = train_demo_path
        self.task_name = task_name
        self.visual_targets = []

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self._actor = self._actor.to(device).train(training)
        self._actor_optimizer = self._actor.configure_optimizers()

        self._device = device

    def reset(self):
        super(ActBCEncAgent, self).reset()

        self._timestep = 0
        # .. input_dim = input_dim * 2 for bimanual
        self._all_time_actions = torch.zeros([self._episode_length,
                                              self._episode_length+self._actor.model.num_queries,
                                              self._actor.model.input_dim]).to(self._device)
        self._all_actions = None

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()



    @lru_cache()
    def train_stats(self):

        right_gripper_poses = []  # 7D: xyz(3) + quat(4)
        left_gripper_poses = []

        right_gripper_open = []   # 1D: gripper open state
        left_gripper_open = []

        episodes_dir = f"{self.train_demo_path}/{self.task_name}.train/all_variations/episodes/"

        for episode in os.listdir(episodes_dir):
            with open(os.path.join(episodes_dir, episode, "low_dim_obs.pkl"), "br") as f:
                d = pickle.load(f)

            for o in d:
                right_gripper_poses.append(o.right.gripper_pose)
                left_gripper_poses.append(o.left.gripper_pose)

                right_gripper_open.append([o.right.gripper_open])
                left_gripper_open.append([o.left.gripper_open])

        right_gripper_poses = np.asarray(right_gripper_poses, dtype=np.float32)
        left_gripper_poses = np.asarray(left_gripper_poses, dtype=np.float32)

        right_gripper_open = np.asarray(right_gripper_open, dtype=np.float32)
        left_gripper_open = np.asarray(left_gripper_open, dtype=np.float32)

        # Compute statistics for position (xyz) only, not quaternion
        # Quaternions are unit vectors and should not be normalized with z-score
        stats = {
            "right_pos_mean": right_gripper_poses[:, :3].mean(axis=0),
            "right_pos_std": right_gripper_poses[:, :3].std(axis=0),

            "left_pos_mean": left_gripper_poses[:, :3].mean(axis=0),
            "left_pos_std": left_gripper_poses[:, :3].std(axis=0),

            "right_gripper_open_mean": right_gripper_open.mean(axis=0),
            "right_gripper_open_std": right_gripper_open.std(axis=0),

            "left_gripper_open_mean":  left_gripper_open.mean(axis=0),
            "left_gripper_open_std": left_gripper_open.std(axis=0)
        }

        return {k: torch.from_numpy(v).to(self._device) for k,v in stats.items()}



    def normalize_z(self, data, mean, std):
        return (data - mean) / std

    def unnormalize_z(self, data, mean, std):
        return data * std + mean


    def preprocess_qpos(self, observation: dict):

        stats = self.train_stats()

        # Right gripper pose: normalize position (xyz), keep quaternion as is
        # Handle both 3D (B, timesteps, 7) from training and 2D (B, 7) from evaluation
        right_pose = observation['right_gripper_pose']
        if right_pose.dim() == 3:
            right_pose = right_pose[:, -1]  # (B, T, 7) -> (B, 7), take last timestep (current state)
        # else: already (B, 7), use directly
        right_pos_norm = self.normalize_z(right_pose[:, :3], stats["right_pos_mean"], stats["right_pos_std"])
        right_quat = right_pose[:, 3:7]  # Keep quaternion as is

        # Right gripper open state (already 0/1, keep as is)
        # Handle both 3D (B, timesteps, 1) from training and 2D (B, 1) from evaluation
        right_gripper = observation['right_gripper_open']
        if right_gripper.dim() == 3:
            right_gripper = right_gripper[:, -1]  # (B, T, 1) -> (B, 1), take last timestep
        # else: already (B, 1), use directly

        # Left gripper pose
        left_pose = observation['left_gripper_pose']
        if left_pose.dim() == 3:
            left_pose = left_pose[:, -1]  # (B, T, 7) -> (B, 7), take last timestep
        left_pos_norm = self.normalize_z(left_pose[:, :3], stats["left_pos_mean"], stats["left_pos_std"])
        left_quat = left_pose[:, 3:7]  # Keep quaternion as is

        # Left gripper open state
        left_gripper = observation['left_gripper_open']
        if left_gripper.dim() == 3:
            left_gripper = left_gripper[:, -1]  # (B, T, 1) -> (B, 1), take last timestep

        # Concatenate: [right_pos(3), right_quat(4), right_gripper(1), left_pos(3), left_quat(4), left_gripper(1)]
        qpos = torch.cat([right_pos_norm, right_quat, right_gripper,
                          left_pos_norm, left_quat, left_gripper], dim=-1)

        return qpos



    def preprocess_action(self, replay_sample: dict):

        stats = self.train_stats()

        # Process previous (current) state: [right_pose(7), right_gripper(1), left_pose(7), left_gripper(1)]
        # Use [:, -1] to get the last (current) timestep, works for any prev_action_horizon
        right_prev_pose = replay_sample['right_prev_gripper_pose'][:, -1]  # (B, T, 7) -> (B, 7)
        right_prev_pos_norm = self.normalize_z(right_prev_pose[:, :3], stats["right_pos_mean"], stats["right_pos_std"])
        right_prev_quat = right_prev_pose[:, 3:7]
        right_prev_gripper = replay_sample['right_prev_gripper_open'][:, -1]  # (B, T, 1) -> (B, 1)

        left_prev_pose = replay_sample['left_prev_gripper_pose'][:, -1]  # (B, T, 7) -> (B, 7)
        left_prev_pos_norm = self.normalize_z(left_prev_pose[:, :3], stats["left_pos_mean"], stats["left_pos_std"])
        left_prev_quat = left_prev_pose[:, 3:7]
        left_prev_gripper = replay_sample['left_prev_gripper_open'][:, -1]  # (B, T, 1) -> (B, 1)

        qpos = torch.cat([right_prev_pos_norm, right_prev_quat, right_prev_gripper,
                          left_prev_pos_norm, left_prev_quat, left_prev_gripper], dim=-1)

        # Process action sequence: (B, T, 16)
        right_next_pose = replay_sample['right_next_gripper_pose']  # (B, T, 7)
        right_next_pos_norm = self.normalize_z(right_next_pose[:, :, :3], stats["right_pos_mean"], stats["right_pos_std"])
        right_next_quat = right_next_pose[:, :, 3:7]
        right_next_gripper = replay_sample['right_next_gripper_open']  # (B, T, 1)

        left_next_pose = replay_sample['left_next_gripper_pose']  # (B, T, 7)
        left_next_pos_norm = self.normalize_z(left_next_pose[:, :, :3], stats["left_pos_mean"], stats["left_pos_std"])
        left_next_quat = left_next_pose[:, :, 3:7]
        left_next_gripper = replay_sample['left_next_gripper_open']  # (B, T, 1)

        action_seq = torch.cat([right_next_pos_norm, right_next_quat, right_next_gripper,
                                left_next_pos_norm, left_next_quat, left_next_gripper], dim=-1)

        return qpos, action_seq

    def preprocess_images(self, replay_sample: dict):
        stacked_rgb = []
        stacked_point_cloud = []

        for camera in self._camera_names:
            rgb = replay_sample['%s_rgb' % camera]
            rgb = rgb if rgb.dim() == 4 else rgb[:,0]
            stacked_rgb.append(rgb)

            point_cloud = replay_sample['%s_point_cloud' % camera]
            point_cloud = point_cloud if point_cloud.dim() == 4 else point_cloud[:,0]
            stacked_point_cloud.append(point_cloud)

        stacked_rgb = torch.stack(stacked_rgb, dim=1)
        stacked_point_cloud = torch.stack(stacked_point_cloud, dim=1)

        return stacked_rgb, stacked_point_cloud

    def update(self, step: int, replay_sample: dict) -> dict:
        # NO LANGUAGE - removed lang_goal_emb handling
        robot_state = replay_sample['low_dim_state']

        # preprocess input
        qpos, action_seq = self.preprocess_action(replay_sample)
        stacked_rgb, stacked_point_cloud = self.preprocess_images(replay_sample)
        is_pad = replay_sample['is_pad'].bool()

        # === Strategy and Phase labels (new for ACT_BC_ENC_STRATEGY) ===
        # Extract from replay sample (already 0-based from launch_utils conversion)
        # Note: YARR replay buffer adds timesteps dimension, so shape is (B, timesteps) = (B, 1)
        # Need to squeeze to get (B,) for cross_entropy
        strategy_type = replay_sample['strategy_type'].squeeze(-1).long()  # (B, 1) -> (B,) int64
        phase_type = replay_sample['phase_type'].squeeze(-1).long()        # (B, 1) -> (B,) int64

        # forward pass with strategy and phase
        loss_dict = self._actor(
            qpos, stacked_rgb, action_seq, is_pad,
            strategy_type=strategy_type, phase_type=phase_type
        )

        # gradient step
        loss = loss_dict['total_losses']
        loss.backward()
        self._actor_optimizer.step()
        self._actor_optimizer.zero_grad()

        # Extended summaries with strategy/phase metrics
        self._summaries = {
            'loss': loss_dict['total_losses'],
            'l1': loss_dict['l1'],
            'right_l1': loss_dict['right_l1'],
            'left_l1': loss_dict['left_l1'],
            'kl': loss_dict['kl'],
            'strategy_loss': loss_dict.get('strategy_loss', 0.0),
            'phase_loss': loss_dict.get('phase_loss', 0.0),
            'strategy_acc': loss_dict.get('strategy_acc', 0.0),
            'phase_acc': loss_dict.get('phase_acc', 0.0),
        }

        return loss_dict

    def _normalize_quat(self, x):
        """Normalize quaternion to unit length"""
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)




    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        """
        Act with strategy and phase conditioning.

        *** END-TO-END INFERENCE ***
        Model uses StrategyPhasePredictor internally to predict strategy/phase.
        No external strategy_type/phase_type needed - model is fully autonomous.

        Args:
            step: Current timestep
            observation: Observation dict
            deterministic: Whether to act deterministically

        Returns:
            ActResult with action and visual targets
        """
        # NOTE: strategy_type and phase_type are NOT needed for inference
        # The model internally uses StrategyPhasePredictor to predict them

        action_horizon = self._actor.model.num_queries
        query_freq = 1

        stats = self.train_stats()

        if self._timestep % query_freq == 0:
            with torch.no_grad():
                # preprocess input
                qpos = self.preprocess_qpos(observation)
                stacked_rgb, stacked_point_cloud = self.preprocess_images(observation)

                # forward pass - model internally predicts strategy/phase using StrategyPhasePredictor
                self._all_actions = self._actor(
                    qpos, stacked_rgb, actions=None, is_pad=None
                    # No strategy_type/phase_type - model predicts them internally
                )

        # temporal aggregation
        t = self._timestep

        self._all_time_actions[[t], t:t + action_horizon] = self._all_actions
        actions_for_curr_step = self._all_time_actions[:, t]
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).to(self._device).unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        raw_action = raw_action[0]


        # raw_action: [right_pos_norm(3), right_quat(4), right_gripper(1),
        #               left_pos_norm(3), left_quat(4), left_gripper(1)] = 16D

        # Right arm: unnormalize position, normalize quaternion, discretize gripper
        right_pos = self.unnormalize_z(raw_action[0:3], stats["right_pos_mean"], stats["right_pos_std"])
        right_quat = raw_action[3:7]
        right_quat_normalized = right_quat / torch.norm(right_quat)  # Ensure unit quaternion
        right_gripper = torch.tensor([1.0 if raw_action[7] > 0.5 else 0.0], device=self._device)  # Discretize
        right_ignore_collision = torch.tensor([1.0], device=self._device)  # Hard-coded

        # Left arm
        left_pos = self.unnormalize_z(raw_action[8:11], stats["left_pos_mean"], stats["left_pos_std"])
        left_quat = raw_action[11:15]
        left_quat_normalized = left_quat / torch.norm(left_quat)
        left_gripper = torch.tensor([1.0 if raw_action[15] > 0.5 else 0.0], device=self._device)
        left_ignore_collision = torch.tensor([1.0], device=self._device)

        # Output format: [right_pos(3), right_quat(4), right_gripper(1), right_ignore(1),
        #                 left_pos(3), left_quat(4), left_gripper(1), left_ignore(1)] = 18D
        raw_action = torch.cat([right_pos, right_quat_normalized, right_gripper, right_ignore_collision,
                                left_pos, left_quat_normalized, left_gripper, left_ignore_collision], dim=-1)

        self._timestep += 1

        return ActResult(raw_action.detach().cpu().numpy(), visual_targets=self.visual_targets)

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        self._actor.load_state_dict(
            torch.load(os.path.join(savedir, 'bc_actor.pt'),
                       map_location=torch.device('cpu')))
        print('Loaded weights from %s' % savedir)

    def save_weights(self, savedir: str):
        torch.save(self._actor.state_dict(),
                   os.path.join(savedir, 'bc_actor.pt'))
