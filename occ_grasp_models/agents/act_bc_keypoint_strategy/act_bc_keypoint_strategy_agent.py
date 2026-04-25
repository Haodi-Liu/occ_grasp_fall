import os
from functools import lru_cache
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from yarr.agents.agent import Agent, Summary, ActResult, ScalarSummary

from helpers.qpos_stats import compute_qpos_stats

NAME = "ActBCKeypointStrategyAgent"

STRATEGY_NAMES = ["EdgeHang", "WallLever", "PressTilt"]
PHASE_NAMES = ["PreManipulation", "Grasp", "ClearPath", "Lift"]
KEYPOINT_NAMES = ["contact", "grasp", "affordance"]


class ActBCKeypointStrategyAgent(Agent):
    """Standalone keyframe agent with keypoint + strategy/phase conditioning."""

    def __init__(self, actor_network, camera_names, lr=0.01, weight_decay=1e-5,
                 grad_clip=20.0, episode_length=400, train_demo_path=None,
                 task_name=None, temporal_ensemble_k=0.01,
                 keyframe_pos_thresh=0.01, keyframe_rot_thresh=0.10,
                 keyframe_max_steps=10):
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
        self._aux_eval_cfg = None
        self._aux_eval_last = {}
        self._aux_eval_step = 0

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
        self._aux_eval_step = 0
        self._aux_eval_last = {}

    def set_aux_eval_cfg(self, cfg):
        self._aux_eval_cfg = cfg
        self._aux_eval_step = 0
        self._aux_eval_last = {}

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    @lru_cache()
    def train_stats(self):
        return compute_qpos_stats(
            data_root=self.train_demo_path,
            train_tasks=[self.task_name],
            out_path=None,
            device=self._device,
        )

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
            rgb = rgb if rgb.dim() == 4 else rgb[:, -1]
            stacked_rgb.append(rgb)

            point_cloud = replay_sample['%s_point_cloud' % camera]
            point_cloud = point_cloud if point_cloud.dim() == 4 else point_cloud[:, -1]
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

    def _extract_keypoint_gt(self, replay_sample: dict):
        has_affordance = replay_sample['has_affordance']
        if has_affordance.dim() == 2:
            has_affordance = has_affordance[:, -1]
        has_affordance = has_affordance.bool()

        keypoint_2d_gt = {}
        keypoint_2d_visible = {}

        for cam_name in self._camera_names:
            contact_2d = replay_sample[f'{cam_name}_contact_2d']
            grasp_2d = replay_sample[f'{cam_name}_grasp_2d']
            affordance_2d = replay_sample[f'{cam_name}_affordance_2d']

            if contact_2d.dim() == 3:
                contact_2d = contact_2d[:, -1]
            if grasp_2d.dim() == 3:
                grasp_2d = grasp_2d[:, -1]
            if affordance_2d.dim() == 3:
                affordance_2d = affordance_2d[:, -1]

            keypoint_2d_gt[cam_name] = torch.stack(
                [contact_2d, grasp_2d, affordance_2d], dim=1
            )

            contact_vis = replay_sample[f'{cam_name}_contact_visible']
            grasp_vis = replay_sample[f'{cam_name}_grasp_visible']
            affordance_vis = replay_sample[f'{cam_name}_affordance_visible']

            if contact_vis.dim() == 2:
                contact_vis = contact_vis[:, -1]
            if grasp_vis.dim() == 2:
                grasp_vis = grasp_vis[:, -1]
            if affordance_vis.dim() == 2:
                affordance_vis = affordance_vis[:, -1]

            keypoint_2d_visible[cam_name] = torch.stack(
                [contact_vis.bool(), grasp_vis.bool(), affordance_vis.bool()], dim=1
            )

        return has_affordance, keypoint_2d_gt, keypoint_2d_visible

    def _extract_strategy_phase(self, replay_sample: dict):
        strategy_type = replay_sample['strategy_type']
        if strategy_type.dim() == 2:
            strategy_type = strategy_type[:, -1]
        strategy_type = strategy_type.long()

        phase_type = replay_sample['phase_type']
        if phase_type.dim() == 2:
            phase_type = phase_type[:, -1]
        phase_type = phase_type.long()

        return strategy_type, phase_type

    def update(self, step: int, replay_sample: dict) -> dict:
        qpos, action_seq, is_pad = self._build_keyframe_batch(replay_sample)
        stacked_rgb, _ = self.preprocess_images(replay_sample)

        has_affordance, keypoint_2d_gt, keypoint_2d_visible = \
            self._extract_keypoint_gt(replay_sample)
        strategy_type, phase_type = self._extract_strategy_phase(replay_sample)

        cam_extr = {c: replay_sample[f"{c}_camera_extrinsics"] for c in self._camera_names}
        cam_intr = {c: replay_sample[f"{c}_camera_intrinsics"] for c in self._camera_names}

        loss_dict = self._actor(
            qpos, stacked_rgb, action_seq, is_pad,
            has_affordance=has_affordance,
            keypoint_2d_gt=keypoint_2d_gt,
            keypoint_2d_visible=keypoint_2d_visible,
            strategy_type=strategy_type,
            phase_type=phase_type,
            camera_extrinsics=cam_extr,
            camera_intrinsics=cam_intr,
            gripper_pose_world={
                "right": replay_sample["right_curr_gripper_pose"],
                "left": replay_sample["left_curr_gripper_pose"],
            },
        )

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
            # [KEYPOINT]
            'proj_2d': loss_dict.get('proj_2d', 0.0),
            'aff_vis': loss_dict.get('aff_vis', 0.0),
            'kp_total': loss_dict.get('kp_total', 0.0),
            # [STRATEGY]
            'strategy_loss': loss_dict.get('strategy_loss', 0.0),
            'phase_loss': loss_dict.get('phase_loss', 0.0),
            'strategy_acc': loss_dict.get('strategy_acc', 0.0),
            'phase_acc': loss_dict.get('phase_acc', 0.0),
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
        pred_info = None
        with torch.no_grad():
            qpos = self.preprocess_qpos(observation)
            stacked_rgb, _ = self.preprocess_images(observation)
            cam_extr = {c: observation[f"{c}_camera_extrinsics"] for c in self._camera_names}
            cam_intr = {c: observation[f"{c}_camera_intrinsics"] for c in self._camera_names}
            aux = None
            if hasattr(self._actor, "predict_with_aux"):
                pred_seq, aux = self._actor.predict_with_aux(
                    qpos, stacked_rgb,
                    camera_extrinsics=cam_extr,
                    camera_intrinsics=cam_intr,
                    gripper_pose_world={
                        "right": observation["right_gripper_pose"],
                        "left": observation["left_gripper_pose"],
                    },
                )
                pred_info = self._build_pred_info(aux)
            else:
                pred_seq = self._actor(
                    qpos, stacked_rgb, actions=None, is_pad=None,
                    camera_extrinsics=cam_extr,
                    camera_intrinsics=cam_intr,
                    gripper_pose_world={
                        "right": observation["right_gripper_pose"],
                        "left": observation["left_gripper_pose"],
                    },
                )

            if self._aux_eval_cfg is not None and bool(getattr(self._aux_eval_cfg, "enabled", False)):
                if "has_affordance" in observation:
                    has_aff, kp_gt, kp_vis = self._extract_keypoint_gt(observation)
                    strat_gt, phase_gt = self._extract_strategy_phase(observation)
                    device = qpos.device

                    kp_pred = aux.get("keypoint_2d_pred") if aux else None
                    strat_logits = aux.get("strategy_logits") if aux else None
                    phase_logits = aux.get("phase_logits") if aux else None

                    kp_loss = self._actor._compute_2d_loss(kp_pred, kp_gt, kp_vis, has_aff) if kp_pred is not None \
                        else torch.tensor(0.0, device=device)
                    strat_loss = F.cross_entropy(strat_logits, strat_gt) if strat_logits is not None \
                        else torch.tensor(0.0, device=device)
                    phase_loss = F.cross_entropy(phase_logits, phase_gt) if phase_logits is not None \
                        else torch.tensor(0.0, device=device)

                    self._aux_eval_last = {
                        "kp_2d": kp_loss,
                        "strategy_ce": strat_loss,
                        "phase_ce": phase_loss,
                    }
                self._aux_eval_step += 1

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
        info = {"pred_info": pred_info} if pred_info is not None else None
        return ActResult(action.detach().cpu().numpy(), visual_targets=self.visual_targets, info=info)

    def _build_pred_info(self, aux: dict):
        if not aux:
            return None

        pred_info = {}
        img_size = getattr(self._actor.model, "img_size", None)
        if img_size is not None:
            pred_info["img_size"] = int(img_size)

        strategy_logits = aux.get("strategy_logits")
        if strategy_logits is not None:
            strat_id = int(strategy_logits.argmax(dim=-1)[0].item())
            pred_info["strategy_id"] = strat_id
            pred_info["strategy_name"] = (
                STRATEGY_NAMES[strat_id] if strat_id < len(STRATEGY_NAMES) else str(strat_id)
            )

        phase_logits = aux.get("phase_logits")
        if phase_logits is not None:
            phase_id = int(phase_logits.argmax(dim=-1)[0].item())
            pred_info["phase_id"] = phase_id
            pred_info["phase_name"] = (
                PHASE_NAMES[phase_id] if phase_id < len(PHASE_NAMES) else str(phase_id)
            )

        keypoint_2d_pred = aux.get("keypoint_2d_pred")
        if keypoint_2d_pred is not None:
            kp_by_cam = {}
            for cam_idx, cam_name in enumerate(self._camera_names):
                cam_kps = {}
                for kp_name in KEYPOINT_NAMES:
                    kp_tensor = keypoint_2d_pred.get(kp_name)
                    if kp_tensor is None:
                        continue
                    coords = kp_tensor[0, cam_idx].detach().cpu().tolist()
                    cam_kps[kp_name] = coords
                kp_by_cam[cam_name] = cam_kps
            pred_info["keypoints_2d"] = kp_by_cam

        affordance_visible = aux.get("affordance_visible")
        if affordance_visible is not None:
            pred_info["affordance_visible"] = float(affordance_visible[0].item())

        return pred_info

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary("%s/%s" % (NAME, n), v))
        return summaries

    def act_summaries(self) -> List[Summary]:
        summaries = []
        cfg = self._aux_eval_cfg
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return summaries
        log_every_n_steps = int(getattr(cfg, "log_every_n_steps", 1))
        if log_every_n_steps > 0 and (self._aux_eval_step % log_every_n_steps) == 0:
            def _as_float(val):
                if torch.is_tensor(val):
                    return float(val.detach().cpu().item())
                return float(val)

            summaries.append(ScalarSummary("aux_eval/kp_2d", _as_float(self._aux_eval_last.get("kp_2d", 0.0))))
            summaries.append(ScalarSummary("aux_eval/strategy_ce", _as_float(self._aux_eval_last.get("strategy_ce", 0.0))))
            summaries.append(ScalarSummary("aux_eval/phase_ce", _as_float(self._aux_eval_last.get("phase_ce", 0.0))))
        return summaries

    def load_weights(self, savedir: str):
        self._actor.load_state_dict(
            torch.load(os.path.join(savedir, 'bc_actor.pt'),
                       map_location=torch.device('cpu')),
            strict=False)
        print('Loaded weights from %s' % savedir)

    def save_weights(self, savedir: str):
        torch.save(self._actor.state_dict(),
                   os.path.join(savedir, 'bc_actor.pt'))
