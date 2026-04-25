import os
import pickle
from functools import lru_cache
from typing import List

import numpy as np
import torch
import torch.nn as nn

from yarr.agents.agent import Agent, Summary, ActResult, ScalarSummary

NAME = "ActBCKeyAgent"


class ActBCKeyAgent(Agent):
    """ACT_BC_KEY agent trained on keyframes with temporal ensembling."""

    def __init__(self, actor_network, camera_names, lr=0.01, weight_decay=1e-5,
                 grad_clip=20.0, episode_length=400, train_demo_path=None,
                 task_name=None, temporal_ensemble_k=0.01,
                 keyframe_pos_thresh=0.01, keyframe_rot_thresh=0.10,
                 keyframe_max_steps=20):
        self._camera_names = camera_names
        self._actor = actor_network
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._episode_length = episode_length
        self.train_demo_path = train_demo_path
        self.task_name = task_name
        self.visual_targets = []

        self._temporal_ensemble_k = temporal_ensemble_k
        self._keyframe_pos_thresh = keyframe_pos_thresh
        self._keyframe_rot_thresh = keyframe_rot_thresh
        self._keyframe_max_steps = keyframe_max_steps

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self._actor = self._actor.to(device).train(training)
        self._actor_optimizer = self._actor.configure_optimizers()
        self._device = device

    def reset(self):
        self._timestep = 0
        self._all_time_actions = torch.zeros([
            self._episode_length,
            self._episode_length + self._actor.model.num_queries,
            self._actor.model.input_dim
        ]).to(self._device)
        self._all_actions = None

        self._keyframe_step = 0
        self._keyframe_step_start_t = 0
        self._all_time_actions_mask = torch.zeros(
            self._all_time_actions.shape[:2], dtype=torch.bool, device=self._device)

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    @lru_cache()
    def train_stats(self):
        right_gripper_poses = []
        left_gripper_poses = []

        right_gripper_open = []
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

        stats = {
            "right_pos_mean": right_gripper_poses[:, :3].mean(axis=0),
            "right_pos_std": right_gripper_poses[:, :3].std(axis=0),
            "left_pos_mean": left_gripper_poses[:, :3].mean(axis=0),
            "left_pos_std": left_gripper_poses[:, :3].std(axis=0),
            "right_gripper_open_mean": right_gripper_open.mean(axis=0),
            "right_gripper_open_std": right_gripper_open.std(axis=0),
            "left_gripper_open_mean": left_gripper_open.mean(axis=0),
            "left_gripper_open_std": left_gripper_open.std(axis=0)
        }

        return {k: torch.from_numpy(v).to(self._device) for k, v in stats.items()}

    def normalize_z(self, data, mean, std):
        return (data - mean) / std

    def unnormalize_z(self, data, mean, std):
        return data * std + mean

    def preprocess_qpos(self, observation: dict):
        stats = self.train_stats()

        right_pose = observation['right_gripper_pose']
        if right_pose.dim() == 3:
            right_pose = right_pose[:, -1]
        right_pos_norm = self.normalize_z(right_pose[:, :3],
                                          stats["right_pos_mean"], stats["right_pos_std"])
        right_quat = right_pose[:, 3:7]

        right_gripper = observation['right_gripper_open']
        if right_gripper.dim() == 3:
            right_gripper = right_gripper[:, -1]

        left_pose = observation['left_gripper_pose']
        if left_pose.dim() == 3:
            left_pose = left_pose[:, -1]
        left_pos_norm = self.normalize_z(left_pose[:, :3],
                                         stats["left_pos_mean"], stats["left_pos_std"])
        left_quat = left_pose[:, 3:7]

        left_gripper = observation['left_gripper_open']
        if left_gripper.dim() == 3:
            left_gripper = left_gripper[:, -1]

        qpos = torch.cat([right_pos_norm, right_quat, right_gripper,
                          left_pos_norm, left_quat, left_gripper], dim=-1)

        return qpos

    def preprocess_images(self, replay_sample: dict):
        stacked_rgb = []
        stacked_point_cloud = []

        for camera in self._camera_names:
            rgb = replay_sample['%s_rgb' % camera]
            rgb = rgb if rgb.dim() == 4 else rgb[:, 0]
            stacked_rgb.append(rgb)

            point_cloud = replay_sample['%s_point_cloud' % camera]
            point_cloud = point_cloud if point_cloud.dim() == 4 else point_cloud[:, 0]
            stacked_point_cloud.append(point_cloud)

        stacked_rgb = torch.stack(stacked_rgb, dim=1)
        stacked_point_cloud = torch.stack(stacked_point_cloud, dim=1)

        return stacked_rgb, stacked_point_cloud

    @staticmethod
    def _last_step(x):
        return x[:, -1] if x.dim() == 3 else x

    def _build_keyframe_batch(self, replay_sample: dict):
        stats = self.train_stats()

        right_curr_pose = self._last_step(replay_sample['right_curr_gripper_pose'])
        right_curr_pos = self.normalize_z(right_curr_pose[:, :3],
                                          stats["right_pos_mean"], stats["right_pos_std"])
        right_curr_quat = right_curr_pose[:, 3:7]
        right_curr_gripper = self._last_step(replay_sample['right_curr_gripper_open'])

        left_curr_pose = self._last_step(replay_sample['left_curr_gripper_pose'])
        left_curr_pos = self.normalize_z(left_curr_pose[:, :3],
                                         stats["left_pos_mean"], stats["left_pos_std"])
        left_curr_quat = left_curr_pose[:, 3:7]
        left_curr_gripper = self._last_step(replay_sample['left_curr_gripper_open'])

        qpos = torch.cat([right_curr_pos, right_curr_quat, right_curr_gripper,
                          left_curr_pos, left_curr_quat, left_curr_gripper], dim=-1)

        right_next_pose = replay_sample['right_next_gripper_pose_seq']
        if right_next_pose.dim() == 4:
            right_next_pose = right_next_pose[:, -1]
        right_next_pos = self.normalize_z(right_next_pose[:, :, :3],
                                          stats["right_pos_mean"], stats["right_pos_std"])
        right_next_quat = right_next_pose[:, :, 3:7]
        right_next_gripper = replay_sample['right_next_gripper_open_seq']
        if right_next_gripper.dim() == 4:
            right_next_gripper = right_next_gripper[:, -1]

        left_next_pose = replay_sample['left_next_gripper_pose_seq']
        if left_next_pose.dim() == 4:
            left_next_pose = left_next_pose[:, -1]
        left_next_pos = self.normalize_z(left_next_pose[:, :, :3],
                                         stats["left_pos_mean"], stats["left_pos_std"])
        left_next_quat = left_next_pose[:, :, 3:7]
        left_next_gripper = replay_sample['left_next_gripper_open_seq']
        if left_next_gripper.dim() == 4:
            left_next_gripper = left_next_gripper[:, -1]

        action_seq = torch.cat([right_next_pos, right_next_quat, right_next_gripper,
                                left_next_pos, left_next_quat, left_next_gripper], dim=-1)
        is_pad = replay_sample['is_pad']
        if is_pad.dim() == 3:
            is_pad = is_pad[:, -1]
        is_pad = is_pad.bool()

        return qpos, action_seq, is_pad

    def update(self, step: int, replay_sample: dict) -> dict:
        qpos, action_seq, is_pad = self._build_keyframe_batch(replay_sample)
        stacked_rgb, _ = self.preprocess_images(replay_sample)
        loss_dict = self._actor(qpos, stacked_rgb, action_seq, is_pad)

        loss = loss_dict['total_losses']
        loss.backward()
        self._actor_optimizer.step()
        self._actor_optimizer.zero_grad()

        self._summaries = {
            'loss': loss_dict['total_losses'],
            'l1': loss_dict['l1'],
            'right_l1': loss_dict['right_l1'],
            'left_l1': loss_dict['left_l1'],
            'kl': loss_dict['kl'],
        }
        return loss_dict

    def _keyframe_reached(self, observation: dict, target_action: torch.Tensor) -> bool:
        right_pose = observation['right_gripper_pose']
        if right_pose.dim() == 3:
            right_pose = right_pose[:, -1]
        left_pose = observation['left_gripper_pose']
        if left_pose.dim() == 3:
            left_pose = left_pose[:, -1]

        right_pos = right_pose[:, :3]
        right_quat = right_pose[:, 3:7]
        left_pos = left_pose[:, :3]
        left_quat = left_pose[:, 3:7]

        tgt_r_pos = target_action[0:3].unsqueeze(0)
        tgt_r_quat = target_action[3:7].unsqueeze(0)
        tgt_r_grip = target_action[7]
        tgt_l_pos = target_action[9:12].unsqueeze(0)
        tgt_l_quat = target_action[12:16].unsqueeze(0)
        tgt_l_grip = target_action[16]

        r_pos_ok = torch.norm(right_pos - tgt_r_pos, dim=1) < self._keyframe_pos_thresh
        l_pos_ok = torch.norm(left_pos - tgt_l_pos, dim=1) < self._keyframe_pos_thresh

        r_dot = torch.abs(torch.sum(right_quat * tgt_r_quat, dim=1)).clamp(max=1.0)
        l_dot = torch.abs(torch.sum(left_quat * tgt_l_quat, dim=1)).clamp(max=1.0)
        r_ang = 2.0 * torch.acos(r_dot)
        l_ang = 2.0 * torch.acos(l_dot)
        r_rot_ok = r_ang < self._keyframe_rot_thresh
        l_rot_ok = l_ang < self._keyframe_rot_thresh

        right_grip = observation['right_gripper_open']
        if right_grip.dim() == 3:
            right_grip = right_grip[:, -1]
        left_grip = observation['left_gripper_open']
        if left_grip.dim() == 3:
            left_grip = left_grip[:, -1]

        grip_ok = ((right_grip > 0.5) == (tgt_r_grip > 0.5)) & \
                  ((left_grip > 0.5) == (tgt_l_grip > 0.5))

        return bool((r_pos_ok & l_pos_ok & r_rot_ok & l_rot_ok & grip_ok).all())

    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        stats = self.train_stats()
        with torch.no_grad():
            qpos = self.preprocess_qpos(observation)
            stacked_rgb, _ = self.preprocess_images(observation)
            pred_seq = self._actor(qpos, stacked_rgb, actions=None, is_pad=None)

        K = self._actor.model.num_queries
        t = self._timestep
        k = min(self._keyframe_step, self._all_time_actions.shape[1] - 1)

        k_end = min(k + K, self._all_time_actions.shape[1])
        pred_len = k_end - k
        if pred_len > 0:
            self._all_time_actions[[t], k:k_end] = pred_seq[:, :pred_len]
            self._all_time_actions_mask[t, k:k_end] = True

        valid_rows = self._all_time_actions_mask[:, k]
        if valid_rows.any():
            actions_for_curr = self._all_time_actions[valid_rows, k]
            times = torch.nonzero(valid_rows, as_tuple=False).squeeze(1)
            ages = (t - times).float()
            weights = torch.exp(-self._temporal_ensemble_k * ages)
            weights = weights / weights.sum()
            raw_action = (actions_for_curr * weights.unsqueeze(1)).sum(dim=0)
        else:
            raw_action = pred_seq[0, 0]

        right_pos = self.unnormalize_z(raw_action[0:3], stats["right_pos_mean"], stats["right_pos_std"])
        right_quat = raw_action[3:7]
        right_quat_normalized = right_quat / torch.norm(right_quat)
        right_gripper = torch.tensor([1.0 if raw_action[7] > 0.5 else 0.0], device=self._device)
        right_ignore_collision = torch.tensor([1.0], device=self._device)

        left_pos = self.unnormalize_z(raw_action[8:11], stats["left_pos_mean"], stats["left_pos_std"])
        left_quat = raw_action[11:15]
        left_quat_normalized = left_quat / torch.norm(left_quat)
        left_gripper = torch.tensor([1.0 if raw_action[15] > 0.5 else 0.0], device=self._device)
        left_ignore_collision = torch.tensor([1.0], device=self._device)

        action = torch.cat([right_pos, right_quat_normalized, right_gripper, right_ignore_collision,
                            left_pos, left_quat_normalized, left_gripper, left_ignore_collision], dim=-1)

        if self._keyframe_reached(observation, action) or \
                (self._timestep - self._keyframe_step_start_t) >= self._keyframe_max_steps:
            self._keyframe_step += 1
            self._keyframe_step_start_t = self._timestep + 1

        self._timestep += 1
        return ActResult(action.detach().cpu().numpy(), visual_targets=self.visual_targets)

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary("%s/%s" % (NAME, n), v))
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
