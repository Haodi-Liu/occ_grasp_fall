from typing import Type, List, Dict, Tuple

import math
import numpy as np
from rlbench import ObservationConfig, ActionMode
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import (
    BimanualObservation,
    Observation,
    UnimanualObservation,
)
from rlbench.backend.task import Task
from yarr.agents.agent import ActResult, VideoSummary, TextSummary
from yarr.envs.rlbench_env import RLBenchEnv, MultiTaskRLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case

from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy

import logging
import os
from pdb import set_trace
from helpers.aux_eval_visualizer import render_aux_eval_like


class CustomRLBenchEnv(RLBenchEnv):
    def __init__(
        self,
        task_class: Type[Task],
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        episode_length: int,
        dataset_root: str = "",
        channels_last: bool = False,
        reward_scale=100.0,
        headless: bool = True,
        time_in_state: bool = False,
        include_lang_goal_in_obs: bool = False,
        # lang_path: str = None,
        record_every_n: int = 20,
        aux_eval_cfg=None,
        dagger_collect_cfg=None,
    ):
        super(CustomRLBenchEnv, self).__init__(
            task_class,
            observation_config,
            action_mode,
            dataset_root,
            channels_last,
            headless=headless,
            include_lang_goal_in_obs=include_lang_goal_in_obs,
            # lang_path=lang_path,
        )
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self._last_exception = None
        # ===== 新增：用于 scheme 分层评估的 episode 编号追踪 =====
        self._current_episode_number = -1
        # ===== 用于可视化：保存最新预测信息 =====
        self._last_pred_info = None
        self._last_pred_obs_dict = None
        # ===== AUX_EVAL =====
        self._aux_eval_cfg = aux_eval_cfg
        self._dagger_collect_cfg = dagger_collect_cfg
        self._aux_eval_step = 0
        self._aux_eval_sample_count = 0
        self._aux_eval_phase_counts = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if "low_dim_state" in oe.name:
                oe.shape = (
                    oe.shape[0] - 7 * 3 + int(self._time_in_state),
                )  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]

        return obs_elems

    def _active_label_cfg(self):
        if self._aux_eval_cfg is not None and bool(getattr(self._aux_eval_cfg, "enabled", False)):
            return self._aux_eval_cfg
        if self._dagger_collect_cfg is not None and bool(getattr(self._dagger_collect_cfg, "enabled", False)):
            return self._dagger_collect_cfg
        return None

    def _append_aux_gt(self, obs, obs_dict):
        cfg = self._active_label_cfg()
        if cfg is None:
            return obs_dict

        misc = getattr(obs, "misc", {}) or {}

        if "has_affordance" in misc:
            obs_dict["has_affordance"] = np.bool_(misc.get("has_affordance", False))

        camera_configs = getattr(self._observation_config, "camera_configs", {}) or {}
        if camera_configs:
            camera_names = list(camera_configs.keys())
        else:
            camera_names = sorted({k[:-4] for k in obs_dict.keys() if k.endswith("_rgb")})
        for cam_name in camera_names:
            obs_dict[f"{cam_name}_contact_2d"] = np.asarray(
                misc.get(f"{cam_name}_contact_2d", [-1.0, -1.0]), dtype=np.float32
            )
            obs_dict[f"{cam_name}_grasp_2d"] = np.asarray(
                misc.get(f"{cam_name}_grasp_2d", [-1.0, -1.0]), dtype=np.float32
            )
            obs_dict[f"{cam_name}_affordance_2d"] = np.asarray(
                misc.get(f"{cam_name}_affordance_2d", [-1.0, -1.0]), dtype=np.float32
            )
            obs_dict[f"{cam_name}_contact_visible"] = np.bool_(misc.get(f"{cam_name}_contact_visible", False))
            obs_dict[f"{cam_name}_grasp_visible"] = np.bool_(misc.get(f"{cam_name}_grasp_visible", False))
            obs_dict[f"{cam_name}_affordance_visible"] = np.bool_(misc.get(f"{cam_name}_affordance_visible", False))

        if "strategy_type" in misc:
            strategy = int(misc.get("strategy_type", 1)) - 1
            num_strategies = int(getattr(cfg, "num_strategies", 3))
            strategy = max(0, min(strategy, num_strategies - 1))
            obs_dict["strategy_type"] = np.int32(strategy)
        if "phase_type" in misc:
            phase = int(misc.get("phase_type", 1)) - 1
            num_phases = int(getattr(cfg, "num_phases", 4))
            phase = max(0, min(phase, num_phases - 1))
            obs_dict["phase_type"] = np.int32(phase)

        return obs_dict

    def _maybe_save_aux_sample(self, obs_dict, pred_info):
        cfg = self._aux_eval_cfg
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return
        if pred_info is None:
            return

        if (self._aux_eval_step % int(getattr(cfg, "sample_every_n_steps", 10))) != 0:
            return
        max_s = int(getattr(cfg, "max_samples_per_episode", 5))
        if self._aux_eval_sample_count >= max_s:
            return
        num_phases = int(getattr(cfg, "num_phases", 4))
        if self._aux_eval_phase_counts is None or len(self._aux_eval_phase_counts) != num_phases:
            self._aux_eval_phase_counts = [0] * num_phases
        phase_id = obs_dict.get("phase_type")
        if phase_id is not None:
            phase_id = int(phase_id)
            if phase_id < 0 or phase_id >= num_phases:
                phase_id = None
        if phase_id is not None:
            per_phase_limit = max(1, int(math.ceil(max_s / float(num_phases))))
            if self._aux_eval_phase_counts[phase_id] >= per_phase_limit:
                return
        save_dir = getattr(cfg, "save_path", "/tmp/aux_eval_samples")
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(
            save_dir, f"ep{self._current_episode_number}_step{self._aux_eval_step:04d}.png"
        )
        render_aux_eval_like(obs_dict, pred_info, cfg, out_file)
        self._aux_eval_sample_count += 1
        if phase_id is not None:
            self._aux_eval_phase_counts[phase_id] += 1

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        if obs.is_bimanual:
            return self.extract_obs_bimanual(obs, t, prev_action)
        else:
            return self.extract_obs_unimanual(obs, t, prev_action)

    def extract_obs_bimanual(self, obs: BimanualObservation, t=None, prev_action=None):
        obs.right.joint_velocities = None
        right_grip_mat = obs.right.gripper_matrix
        right_grip_pose = obs.right.gripper_pose
        right_joint_pos = obs.right.joint_positions
        obs.right.gripper_pose = None
        obs.right.gripper_matrix = None
        obs.right.joint_positions = None

        obs.left.joint_velocities = None
        left_grip_mat = obs.left.gripper_matrix
        left_grip_pose = obs.left.gripper_pose
        left_joint_pos = obs.left.joint_positions
        obs.left.gripper_pose = None
        obs.left.gripper_matrix = None
        obs.left.joint_positions = None

        if obs.right.gripper_joint_positions is not None:
            obs.right.gripper_joint_positions = np.clip(
                obs.right.gripper_joint_positions, 0.0, 0.04
            )
            obs.left.gripper_joint_positions = np.clip(
                obs.left.gripper_joint_positions, 0.0, 0.04
            )

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0

            if "low_dim_state" in obs_dict:
                obs_dict["low_dim_state"] = np.concatenate(
                    [obs_dict["low_dim_state"], [time]]
                ).astype(np.float32)
            else:
                obs_dict["right_low_dim_state"] = np.concatenate(
                    [obs_dict["right_low_dim_state"], [time]]
                ).astype(np.float32)
                obs_dict["left_low_dim_state"] = np.concatenate(
                    [obs_dict["left_low_dim_state"], [time]]
                ).astype(np.float32)

        obs.right.gripper_matrix = right_grip_mat
        obs.right.joint_positions = right_joint_pos
        obs.right.gripper_pose = right_grip_pose
        obs.left.gripper_matrix = left_grip_mat
        obs.left.joint_positions = left_joint_pos
        obs.left.gripper_pose = left_grip_pose

        obs_dict['left_joint_positions'] = obs.left.joint_positions
        obs_dict['left_gripper_joint_positions'] = obs.left.gripper_joint_positions
        obs_dict['left_gripper_pose'] = obs.left.gripper_pose
        obs_dict['left_gripper_open'] = np.array([obs.left.gripper_open])
        obs_dict['right_joint_positions'] = obs.right.joint_positions
        obs_dict['right_gripper_joint_positions'] = obs.right.gripper_joint_positions
        obs_dict['right_gripper_pose'] = obs.right.gripper_pose
        obs_dict['right_gripper_open'] = np.array([obs.right.gripper_open])
        obs_dict = self._append_aux_gt(obs, obs_dict)
        obs_dict['task_id'] = np.int32(max(self.active_task_id, 0))
        return obs_dict

    def extract_obs_unimanual(self, obs: UnimanualObservation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0.0, 0.04
            )

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0
            obs_dict["low_dim_state"] = np.concatenate(
                [obs_dict["low_dim_state"], [time]]
            ).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose

        obs_dict['joint_positions'] = obs.joint_positions
        obs_dict['gripper_joint_positions'] = obs.gripper_joint_positions

        obs_dict = self._append_aux_gt(obs, obs_dict)
        obs_dict['task_id'] = np.int32(max(self.active_task_id, 0))
        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            cam_base = Dummy("cam_cinematic_base")
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomRLBenchEnv, self).reset()
        self._record_current_episode = (
            self.eval and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()
        self._last_pred_info = None
        self._last_pred_obs_dict = None
        self._aux_eval_step = 0
        self._aux_eval_sample_count = 0
        self._aux_eval_phase_counts = None
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10,) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    # def step(self, act_result: ActResult) -> Transition:
    #     action = act_result.action
    #     success = False
    #     obs = self._previous_obs_dict  # in case action fails.

    #     try:
    #         obs, reward, terminal = self._task.step(action)
    #         if reward >= 1:
    #             success = True
    #             reward *= self._reward_scale
    #         else:
    #             reward = 0.0
    #         obs = self.extract_obs(obs)
    #         self._previous_obs_dict = obs
    #     except (IKError, ConfigurationPathError, InvalidActionError) as e:
    #         terminal = True
    #         reward = 0.0

    #         if isinstance(e, IKError):
    #             self._error_type_counts["IKError"] += 1
    #         elif isinstance(e, ConfigurationPathError):
    #             self._error_type_counts["ConfigurationPathError"] += 1
    #         elif isinstance(e, InvalidActionError):
    #             self._error_type_counts["InvalidActionError"] += 1

    #         self._last_exception = e

    #     summaries = []
    #     self._i += 1
    #     if (
    #         terminal or self._i == self._episode_length
    #     ) and self._record_current_episode:
    #         self._append_final_frame(success)
    #         vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
    #         summaries.append(
    #             VideoSummary(
    #                 "episode_rollout_" + ("success" if success else "fail"), vid, fps=30
    #             )
    #         )

    #         # error summary
    #         error_str = (
    #             f"Errors - IK : {self._error_type_counts['IKError']}, "
    #             f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, "
    #             f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
    #         )
    #         if not success and self._last_exception is not None:
    #             error_str += f"\n Last Exception: {self._last_exception}"
    #             self._last_exception = None

    #         summaries.append(
    #             TextSummary("errors", f"Success: {success} | " + error_str)
    #         )
    #     return Transition(obs, reward, terminal, summaries=summaries)
    
    
    
    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        visual_targets = act_result.visual_targets

        if act_result is not None and act_result.info:
            self._last_pred_info = act_result.info.get("pred_info")

        # Cache the observation used for prediction to align overlay timing.
        self._last_pred_obs_dict = self._previous_obs_dict

        if self._previous_obs_dict is not None:
            dagger_on = (
                self._dagger_collect_cfg is not None
                and bool(getattr(self._dagger_collect_cfg, "enabled", False))
            )
            if not dagger_on:
                self._maybe_save_aux_sample(self._previous_obs_dict, self._last_pred_info)
                self._aux_eval_step += 1

        success = False
        obs = self._previous_obs_dict  # in case action fails.
        # set_trace()
        try:
            obs, reward, terminal = self._task.step(action, visual_targets)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts["IKError"] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts["ConfigurationPathError"] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts["InvalidActionError"] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if (
            terminal or self._i == self._episode_length
        ) and self._record_current_episode:
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(
                VideoSummary(
                    "episode_rollout_" + ("success" if success else "fail"), vid, fps=30
                )
            )

            # error summary
            error_str = (
                f"Errors - IK : {self._error_type_counts['IKError']}, "
                f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, "
                f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            )
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(
                TextSummary("errors", f"Success: {success} | " + error_str)
            )

        info = {}
        if self._last_pred_info is not None:
            info["pred_info"] = self._last_pred_info

        return Transition(obs, reward, terminal, info=info, summaries=summaries)

    # ===== 新增：阶段评估接口（第8部分） =====

    def get_phase_progress(self) -> Dict:
        """
        获取当前任务的阶段进度信息

        Returns:
            Dict: 包含阶段完成状态的字典，若任务不支持阶段评估则返回None

        注意：self._task 是 TaskEnvironment 实例，self._task._task 是实际的任务类实例
        """
        if self._task is None or self._task._task is None:
            return None

        task = self._task._task
        if hasattr(task, 'get_phase_progress'):
            return task.get_phase_progress()
        return None

    def evaluate_current_phase(self) -> Tuple[bool, int]:
        """
        评估当前阶段是否完成

        Returns:
            (phase_completed, completed_phase): 阶段是否刚完成及刚完成的阶段ID

        注意：当 phase_completed=True 时，返回的第二个值是刚完成的阶段编号，不需要再 -1
        """
        if self._task is None or self._task._task is None:
            return False, 1

        task = self._task._task
        if hasattr(task, 'phased_evaluator') and task.phased_evaluator is not None:
            return task.phased_evaluator.evaluate_current_phase()
        return False, 1

    def get_strategy_and_phase(self) -> Tuple[int, int]:
        """
        获取当前策略类型和阶段类型

        Returns:
            (strategy_type, phase_type)
        """
        if self._task is None or self._task._task is None:
            return 1, 1

        task = self._task._task
        if hasattr(task, 'evaluate_phase_and_get_labels'):
            return task.evaluate_phase_and_get_labels()

        # 默认值
        strategy = getattr(task, 'STRATEGY_TYPE', 1)
        return strategy, 1

    # ============================================

    def reset_to_demo(self, i, max_attempts=3):
        self._i = 0
        # ===== 新增：记录当前 episode 编号用于 scheme 查询 =====
        self._current_episode_number = i
        # super(CustomRLBenchEnv, self).reset()
        self._aux_eval_step = 0
        self._aux_eval_sample_count = 0
        self._aux_eval_phase_counts = None
        self._last_pred_info = None
        self._last_pred_obs_dict = None

        for attempt in range(max_attempts):
            try:
                self._task.set_variation(-1)
                (d,) = self._task.get_demos(
                    1, live_demos=False, random_selection=False, from_episode_number=i
                )

                self._task.set_variation(d.variation_number)

                # On retry, do a full scene reset to clear any stuck state
                if attempt > 0:
                    import time
                    time.sleep(0.5)  # Brief pause to let CoppeliaSim stabilize
                    self._task._scene.reset()

                _, obs = self._task.reset_to_demo(d)

                self._lang_goal = self._task.get_task_descriptions()[0]
                self._previous_obs_dict = self.extract_obs(obs)
                break  # Success, exit loop

            except RuntimeError as e:
                if attempt < max_attempts - 1:
                    logging.warning(f"Scene reset failed for episode {i}, attempt {attempt+1}/{max_attempts}: {e}. Retrying...")
                else:
                    # Last attempt failed, re-raise
                    logging.error(f"Scene reset failed after {max_attempts} attempts for episode {i}")
                    raise
        self._record_current_episode = (
            self.eval and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict

    # ===== 新增：scheme 分层评估接口 =====
    def get_current_episode_number(self) -> int:
        """
        获取当前评估的 episode 编号。

        Returns:
            int: episode 编号，用于从 episode_scheme_map 中查询对应的 GT scheme。
                 返回 -1 表示尚未调用 reset_to_demo。

        Usage:
            ep_num = env.get_current_episode_number()
            # 例如返回 5，表示当前正在评估 episode5
            # 可从 episode_scheme_map[5] 获取该 episode 的 GT scheme
        """
        return self._current_episode_number
    # =====================================


class CustomMultiTaskRLBenchEnv(MultiTaskRLBenchEnv):
    def __init__(
        self,
        task_classes: List[Type[Task]],
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        episode_length: int,
        dataset_root: str = "",
        channels_last: bool = False,
        reward_scale=100.0,
        headless: bool = True,
        swap_task_every: int = 1,
        time_in_state: bool = False,
        include_lang_goal_in_obs: bool = False,
        record_every_n: int = 20,
        aux_eval_cfg=None,
        dagger_collect_cfg=None,
    ):
        super(CustomMultiTaskRLBenchEnv, self).__init__(
            task_classes,
            observation_config,
            action_mode,
            dataset_root,
            channels_last,
            headless=headless,
            swap_task_every=swap_task_every,
            include_lang_goal_in_obs=include_lang_goal_in_obs,
        )
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self._last_exception = None
        # ===== 新增：用于 scheme 分层评估的 episode 编号追踪 =====
        self._current_episode_number = -1
        # ===== 用于可视化：保存最新预测信息 =====
        self._last_pred_info = None
        self._last_pred_obs_dict = None
        # ===== AUX_EVAL =====
        self._aux_eval_cfg = aux_eval_cfg
        self._dagger_collect_cfg = dagger_collect_cfg
        self._aux_eval_step = 0
        self._aux_eval_sample_count = 0
        self._aux_eval_phase_counts = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomMultiTaskRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if "low_dim_state" in oe.name:
                # ..todo:: since we have the low_dimensional state separate for both robots this will also work
                oe.shape = (
                    oe.shape[0] - 7 * 3 + int(self._time_in_state),
                )  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def _active_label_cfg(self):
        if self._aux_eval_cfg is not None and bool(getattr(self._aux_eval_cfg, "enabled", False)):
            return self._aux_eval_cfg
        if self._dagger_collect_cfg is not None and bool(getattr(self._dagger_collect_cfg, "enabled", False)):
            return self._dagger_collect_cfg
        return None

    def _append_aux_gt(self, obs, obs_dict):
        cfg = self._active_label_cfg()
        if cfg is None:
            return obs_dict

        misc = getattr(obs, "misc", {}) or {}

        if "has_affordance" in misc:
            obs_dict["has_affordance"] = np.bool_(misc.get("has_affordance", False))

        camera_configs = getattr(self._observation_config, "camera_configs", {}) or {}
        if camera_configs:
            camera_names = list(camera_configs.keys())
        else:
            camera_names = sorted({k[:-4] for k in obs_dict.keys() if k.endswith("_rgb")})
        for cam_name in camera_names:
            obs_dict[f"{cam_name}_contact_2d"] = np.asarray(
                misc.get(f"{cam_name}_contact_2d", [-1.0, -1.0]), dtype=np.float32
            )
            obs_dict[f"{cam_name}_grasp_2d"] = np.asarray(
                misc.get(f"{cam_name}_grasp_2d", [-1.0, -1.0]), dtype=np.float32
            )
            obs_dict[f"{cam_name}_affordance_2d"] = np.asarray(
                misc.get(f"{cam_name}_affordance_2d", [-1.0, -1.0]), dtype=np.float32
            )
            obs_dict[f"{cam_name}_contact_visible"] = np.bool_(misc.get(f"{cam_name}_contact_visible", False))
            obs_dict[f"{cam_name}_grasp_visible"] = np.bool_(misc.get(f"{cam_name}_grasp_visible", False))
            obs_dict[f"{cam_name}_affordance_visible"] = np.bool_(misc.get(f"{cam_name}_affordance_visible", False))

        if "strategy_type" in misc:
            strategy = int(misc.get("strategy_type", 1)) - 1
            num_strategies = int(getattr(cfg, "num_strategies", 3))
            strategy = max(0, min(strategy, num_strategies - 1))
            obs_dict["strategy_type"] = np.int32(strategy)
        if "phase_type" in misc:
            phase = int(misc.get("phase_type", 1)) - 1
            num_phases = int(getattr(cfg, "num_phases", 4))
            phase = max(0, min(phase, num_phases - 1))
            obs_dict["phase_type"] = np.int32(phase)

        return obs_dict

    def _maybe_save_aux_sample(self, obs_dict, pred_info):
        cfg = self._aux_eval_cfg
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return
        if pred_info is None:
            return

        if (self._aux_eval_step % int(getattr(cfg, "sample_every_n_steps", 10))) != 0:
            return
        max_s = int(getattr(cfg, "max_samples_per_episode", 5))
        if self._aux_eval_sample_count >= max_s:
            return
        num_phases = int(getattr(cfg, "num_phases", 4))
        if self._aux_eval_phase_counts is None or len(self._aux_eval_phase_counts) != num_phases:
            self._aux_eval_phase_counts = [0] * num_phases
        phase_id = obs_dict.get("phase_type")
        if phase_id is not None:
            phase_id = int(phase_id)
            if phase_id < 0 or phase_id >= num_phases:
                phase_id = None
        if phase_id is not None:
            per_phase_limit = max(1, int(math.ceil(max_s / float(num_phases))))
            if self._aux_eval_phase_counts[phase_id] >= per_phase_limit:
                return
        save_dir = getattr(cfg, "save_path", "/tmp/aux_eval_samples")
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(
            save_dir, f"ep{self._current_episode_number}_step{self._aux_eval_step:04d}.png"
        )
        render_aux_eval_like(obs_dict, pred_info, cfg, out_file)
        self._aux_eval_sample_count += 1
        if phase_id is not None:
            self._aux_eval_phase_counts[phase_id] += 1

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        if obs.is_bimanual:
            return self.extract_obs_bimanual(obs, t, prev_action)
        else:
            return self.extract_obs_unimanual(obs, t, prev_action)

    def extract_obs_bimanual(self, obs: BimanualObservation, t=None, prev_action=None):
        obs.right.joint_velocities = None
        right_gripper_mat = obs.right.gripper_matrix
        right_gripper_pose = obs.right.gripper_pose
        right_joint_pos = obs.right.joint_positions
        obs.right.gripper_pose = None
        obs.right.gripper_matrix = None
        obs.right.joint_positions = None

        obs.left.joint_velocities = None
        left_gripper_mat = obs.left.gripper_matrix
        left_gripper_pose = obs.left.gripper_pose
        left_joint_pos = obs.left.joint_positions
        obs.left.gripper_pose = None
        obs.left.gripper_matrix = None
        obs.left.joint_positions = None

        if obs.right.gripper_joint_positions is not None:
            obs.right.gripper_joint_positions = np.clip(
                obs.right.gripper_joint_positions, 0.0, 0.04
            )
            obs.left.gripper_joint_positions = np.clip(
                obs.left.gripper_joint_positions, 0.0, 0.04
            )

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0
            obs_dict["right_low_dim_state"] = np.concatenate(
                [obs_dict["right_low_dim_state"], [time]]
            ).astype(np.float32)
            obs_dict["left_low_dim_state"] = np.concatenate(
                [obs_dict["left_low_dim_state"], [time]]
            ).astype(np.float32)

        obs.right.gripper_matrix = right_gripper_mat
        obs.right.joint_positions = right_joint_pos
        obs.right.gripper_pose = right_gripper_pose
        obs.left.gripper_matrix = left_gripper_mat
        obs.left.joint_positions = left_joint_pos
        obs.left.gripper_pose = left_gripper_pose

        obs_dict['left_joint_positions'] = obs.left.joint_positions
        obs_dict['left_gripper_joint_positions'] = obs.left.gripper_joint_positions
        obs_dict['left_gripper_pose'] = obs.left.gripper_pose
        obs_dict['left_gripper_open'] = np.array([obs.left.gripper_open])
        obs_dict['right_joint_positions'] = obs.right.joint_positions
        obs_dict['right_gripper_joint_positions'] = obs.right.gripper_joint_positions
        obs_dict['right_gripper_pose'] = obs.right.gripper_pose
        obs_dict['right_gripper_open'] = np.array([obs.right.gripper_open])

        obs_dict = self._append_aux_gt(obs, obs_dict)
        obs_dict['task_id'] = np.int32(self.active_task_id)
        return obs_dict

    def extract_obs_unimanual(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0.0, 0.04
            )

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0
            obs_dict["low_dim_state"] = np.concatenate(
                [obs_dict["low_dim_state"], [time]]
            ).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose

        obs_dict['joint_positions'] = obs.joint_positions
        obs_dict['gripper_joint_positions'] = obs.gripper_joint_positions

        obs_dict = self._append_aux_gt(obs, obs_dict)
        obs_dict['task_id'] = np.int32(self.active_task_id)
        return obs_dict

    def launch(self):
        super(CustomMultiTaskRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            cam_base = Dummy("cam_cinematic_base")
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomMultiTaskRLBenchEnv, self).reset()
        self._record_current_episode = (
            self.eval and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()
        self._last_pred_info = None
        self._last_pred_obs_dict = None
        self._aux_eval_step = 0
        self._aux_eval_sample_count = 0
        self._aux_eval_phase_counts = None
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10,) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        if act_result is not None and act_result.info:
            self._last_pred_info = act_result.info.get("pred_info")

        # Cache the observation used for prediction to align overlay timing.
        self._last_pred_obs_dict = self._previous_obs_dict

        if self._previous_obs_dict is not None:
            dagger_on = (
                self._dagger_collect_cfg is not None
                and bool(getattr(self._dagger_collect_cfg, "enabled", False))
            )
            if not dagger_on:
                self._maybe_save_aux_sample(self._previous_obs_dict, self._last_pred_info)
                self._aux_eval_step += 1

        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts["IKError"] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts["ConfigurationPathError"] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts["InvalidActionError"] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if (
            terminal or self._i == self._episode_length
        ) and self._record_current_episode:
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            task_name = change_case(self._task._task.__class__.__name__)
            summaries.append(
                VideoSummary(
                    "episode_rollout_"
                    + ("success" if success else "fail")
                    + f"/{task_name}",
                    vid,
                    fps=30,
                )
            )

            # error summary
            error_str = (
                f"Errors - IK : {self._error_type_counts['IKError']}, "
                f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, "
                f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            )
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(
                TextSummary("errors", f"Success: {success} | " + error_str)
            )

        info = {}
        if self._last_pred_info is not None:
            info["pred_info"] = self._last_pred_info

        return Transition(obs, reward, terminal, info=info, summaries=summaries)

    # ===== 新增：阶段评估接口（与 CustomRLBenchEnv 保持一致） =====

    def get_phase_progress(self) -> Dict:
        if self._task is None or self._task._task is None:
            return None

        task = self._task._task
        if hasattr(task, 'get_phase_progress'):
            return task.get_phase_progress()
        return None

    def evaluate_current_phase(self) -> Tuple[bool, int]:
        if self._task is None or self._task._task is None:
            return False, 1

        task = self._task._task
        if hasattr(task, 'phased_evaluator') and task.phased_evaluator is not None:
            return task.phased_evaluator.evaluate_current_phase()
        return False, 1

    def get_strategy_and_phase(self) -> Tuple[int, int]:
        if self._task is None or self._task._task is None:
            return 1, 1

        task = self._task._task
        if hasattr(task, 'evaluate_phase_and_get_labels'):
            return task.evaluate_phase_and_get_labels()

        strategy = getattr(task, 'STRATEGY_TYPE', 1)
        return strategy, 1

    # ============================================

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        # ===== 新增：记录当前 episode 编号用于 scheme 查询 =====
        self._current_episode_number = i
        # super(CustomMultiTaskRLBenchEnv, self).reset()
        self._aux_eval_step = 0
        self._aux_eval_sample_count = 0
        self._aux_eval_phase_counts = None
        self._last_pred_info = None
        self._last_pred_obs_dict = None

        # if variation_number == -1:
        #     self._task.sample_variation()
        # else:
        #     self._task.set_variation(variation_number)

        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict

    # ===== 新增：scheme 分层评估接口 =====
    def get_current_episode_number(self) -> int:
        """
        获取当前评估的 episode 编号。

        Returns:
            int: episode 编号，用于从 episode_scheme_map 中查询对应的 GT scheme。
                 返回 -1 表示尚未调用 reset_to_demo。

        Usage:
            ep_num = env.get_current_episode_number()
            # 例如返回 5，表示当前正在评估 episode5
            # 可从 episode_scheme_map[5] 获取该 episode 的 GT scheme
        """
        return self._current_episode_number
    # =====================================
