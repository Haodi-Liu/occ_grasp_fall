import copy
import logging
import os
import time
import pandas as pd

from multiprocessing import Process, Manager
from multiprocessing import get_start_method, set_start_method
from typing import Any

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.log_writer import LogWriter
from yarr.utils.process_str import change_case
from yarr.utils.video_utils import CircleCameraMotion, TaskRecorder

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from yarr.runners._env_runner import _EnvRunner
import os
from helpers.dagger_data_collector import DaggerDataCollector

# ===== 新增：scheme 分层评估工具 =====
try:
    from helpers.scheme_utils import build_episode_scheme_map, get_scheme_stats_summary
    SCHEME_UTILS_AVAILABLE = True
except ImportError:
    # 如果 scheme_utils 不可用，禁用 scheme 分层评估功能
    SCHEME_UTILS_AVAILABLE = False
    logging.warning("scheme_utils not available, scheme-stratified evaluation disabled")
# ====================================

class _IndependentEnvRunner(_EnvRunner):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 num_eval_episodes_signal: Any,
                 eval_epochs_signal: Any,
                 eval_report_signal: Any,
                 log_freq: int,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 weightsdir: str = None,
                 logdir: str = None,
                 env_device: torch.device = None,
                 previous_loaded_weight_folder: str = '',
                 num_eval_runs: int = 1,
                 ):

            super().__init__(train_env, eval_env, agent, timesteps,
                             train_envs, eval_envs, rollout_episodes, eval_episodes,
                             training_iterations, eval_from_eps_number, episode_length,
                             kill_signal, step_signal, num_eval_episodes_signal,
                             eval_epochs_signal, eval_report_signal, log_freq,
                             rollout_generator, save_load_lock, current_replay_ratio,
                             target_replay_ratio, weightsdir, logdir, env_device,
                             previous_loaded_weight_folder, num_eval_runs)

    def _load_save(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                        self._new_weights = True
                    else:
                        self._new_weights = False
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _get_task_name(self):
        if hasattr(self._eval_env, '_task_class'):
            eval_task_name = change_case(self._eval_env._task_class.__name__)
            multi_task = False
        elif hasattr(self._eval_env, '_task_classes'):
            if self._eval_env.active_task_id != -1:
                task_id = (self._eval_env.active_task_id) % len(self._eval_env._task_classes)
                eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
            else:
                eval_task_name = ''
            multi_task = True
        else:
            raise Exception('Neither task_class nor task_classes found in eval env')
        return eval_task_name, multi_task

    def _get_task_name_from_env_state(self, env: Env) -> str:
        if hasattr(env, '_task_class'):
            return change_case(env._task_class.__name__)
        if hasattr(env, '_task_classes') and env.active_task_id != -1:
            task_id = env.active_task_id % len(env._task_classes)
            return change_case(env._task_classes[task_id].__name__)
        return ''

    def _get_expected_eval_task_name(self, env: Env, eval_run_idx: int) -> str:
        if hasattr(env, '_task_classes') and len(env._task_classes) > 0:
            if self._num_eval_runs == len(env._task_classes):
                task_id = eval_run_idx % len(env._task_classes)
                return change_case(env._task_classes[task_id].__name__)
        if hasattr(env, '_task_class'):
            return change_case(env._task_class.__name__)
        return ''

    def _resolve_current_gt_scheme(self, env: Env, episode_scheme_maps, eval_demo_seed: int,
                                   fallback_task_name: str = ''):
        task_name = self._get_task_name_from_env_state(env) or fallback_task_name
        episode_number = eval_demo_seed
        if hasattr(env, 'get_current_episode_number'):
            try:
                current_episode_number = int(env.get_current_episode_number())
                if current_episode_number >= 0:
                    episode_number = current_episode_number
            except Exception:
                pass

        if not task_name:
            return 'unknown', episode_number, ''

        task_scheme_map = episode_scheme_maps.get(task_name, {})
        return task_scheme_map.get(episode_number, 'unknown'), episode_number, task_name

    def _run_eval_independent(self, name: str,
                              stats_accumulator,
                              weight,
                              writer_lock,
                              eval=True,
                              device_idx=0,
                              save_metrics=True,
                              cinematic_recorder_cfg=None,
                              aux_eval_cfg=None,
                              dagger_collect_cfg=None):
        self._name = name
        self._save_metrics = save_metrics
        self._is_test_set = type(weight) == dict

        self._agent = copy.deepcopy(self._agent)

        # Respect configured eval device (framework.gpu -> env_device) instead of forcing CUDA.
        if isinstance(self._env_device, torch.device):
            if self._env_device.type == "cpu":
                device = torch.device("cpu")
            elif torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    device = torch.device(f"cuda:{device_idx}")
                elif self._env_device.index is not None:
                    device = torch.device(f"cuda:{self._env_device.index}")
                else:
                    device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                device = (
                    torch.device(f"cuda:{device_idx}")
                    if torch.cuda.device_count() > 1
                    else torch.device("cuda:0")
                )
            else:
                device = torch.device("cpu")
        logging.info(
            "Eval runner device selection | env_device=%s | device_idx=%s | build_device=%s",
            self._env_device,
            device_idx,
            device,
        )
        with writer_lock: # hack to prevent multiple CLIP downloads ... argh should use a separate lock
            self._agent.build(training=False, device=device)
        if aux_eval_cfg is not None and hasattr(self._agent, "set_aux_eval_cfg"):
            self._agent.set_aux_eval_cfg(aux_eval_cfg)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env
        env.eval = eval
        env.launch()

        # C1 数据采集器（关闭时保持旧行为）
        collector = None
        if dagger_collect_cfg is not None and bool(getattr(dagger_collect_cfg, "enabled", False)):
            collector = DaggerDataCollector(dagger_collect_cfg)

        # ===== 新增：初始化 scheme 分层评估所需的数据 =====
        # 获取 dataset_root（评估数据目录，不带 .train 后缀）
        # 示例: '/mnt/rlbench_data'
        # 注意: _dataset_root 存储在 env._rlbench_env 中，而非 env 本身
        if hasattr(env, '_rlbench_env') and hasattr(env._rlbench_env, '_dataset_root'):
            dataset_root = env._rlbench_env._dataset_root
        else:
            dataset_root = ''
            logging.warning("Cannot access dataset_root from environment, scheme evaluation disabled")

        # episode_scheme_maps: 每个任务的 episode → scheme 映射
        # 结构: {'bimanual_edge_phone': {0: 'left_grasper', 1: 'right_grasper', ...}, ...}
        episode_scheme_maps = {}

        if SCHEME_UTILS_AVAILABLE and dataset_root:
            # 获取所有任务名称列表
            if hasattr(env, '_task_classes'):
                # 多任务模式: ['BimanualEdgePhone', 'BimanualPivotPhone', ...]
                task_names = [change_case(tc.__name__) for tc in env._task_classes]
            elif hasattr(env, '_task_class'):
                # 单任务模式
                task_names = [change_case(env._task_class.__name__)]
            else:
                task_names = []

            # 为每个任务构建 episode → scheme 映射
            for task_name in task_names:
                episode_scheme_maps[task_name] = build_episode_scheme_map(dataset_root, task_name)
                logging.info(f"Built scheme map for {task_name}: {len(episode_scheme_maps[task_name])} episodes")
        # ===================================================

        # initialize cinematic recorder if specified
        rec_cfg = cinematic_recorder_cfg
        if rec_cfg.enabled:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam = VisionSensor.create(rec_cfg.camera_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), rec_cfg.rotate_speed)
            tr = TaskRecorder(env, cam_motion, fps=rec_cfg.fps, overlay_cfg=rec_cfg)

            env.env._action_mode.arm_action_mode.set_callable_each_step(tr.take_snap)
            # env.env._action_mode.arm_action_mode.set_callable_each_prediction(tr.take_snap_prediction)

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')

        # to save or not to save evaluation metrics (set as False for recording videos)
        if self._save_metrics:
            csv_file = 'eval_data.csv' if not self._is_test_set else 'test_data.csv'
            writer = LogWriter(self._logdir, True, True,
                               env_csv=csv_file)

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self._weightsdir, str(weight))
            seed_path = self._weightsdir.replace('/weights', '')
            self._agent.load_weights(weight_path)
            weight_name = str(weight)

        new_transitions = {'train_envs': 0, 'eval_envs': 0}
        total_transitions = {'train_envs': 0, 'eval_envs': 0}
        current_task_id = -1

        for n_eval in range(self._num_eval_runs):
            # ===== MODIFICATION: Add success/failed episode counters =====
            success_count = 0
            failed_count = 0
            failed_episodes = []  # List of (ep_idx, seed, error_msg)
            # =============================================================

            # ===== 新增：阶段级别评估统计 =====
            phase_success_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # 各阶段成功次数
            max_phases_reached = []  # 每个episode达到的最大阶段
            phase_completion_frames = {1: [], 2: [], 3: [], 4: []}  # 各阶段完成帧数
            # =================================

            # ===== Video recording quota counters =====
            success_videos_saved = 0
            fail_videos_saved = 0
            max_success_videos = rec_cfg.get('max_success_videos', 5) if rec_cfg.enabled else 0
            max_fail_videos = rec_cfg.get('max_fail_videos', 5) if rec_cfg.enabled else 0
            logging.info(f"Video recording quota: max {max_success_videos} success, {max_fail_videos} fail videos")
            # ===========================================

            # ===== 新增：scheme 分层统计 =====
            # scheme_stats: 按 scheme 类型分层的统计
            # 结构: {
            #     'left_grasper': {'success': 0, 'total': 0, 'success_steps': []},
            #     'right_grasper': {'success': 0, 'total': 0, 'success_steps': []},
            #     'unknown': {'success': 0, 'total': 0, 'success_steps': []}
            # }
            scheme_stats = {
                'left_grasper': {'success': 0, 'total': 0, 'success_steps': []},
                'right_grasper': {'success': 0, 'total': 0, 'success_steps': []},
                'unknown': {'success': 0, 'total': 0, 'success_steps': []}
            }
            # ================================

            if rec_cfg.enabled:
                tr._cam_motion.save_pose()

            expected_task_name = self._get_expected_eval_task_name(env, n_eval)

            # best weight for each task (used for test evaluation)
            if type(weight) == dict:
                task_name = list(weight.keys())[n_eval]
                task_weight = weight[task_name]
                weight_path = os.path.join(self._weightsdir, str(task_weight))
                seed_path = self._weightsdir.replace('/weights', '')
                self._agent.load_weights(weight_path)
                weight_name = str(task_weight)
                print('Evaluating weight %s for %s' % (weight_name, task_name))

            # evaluate on N tasks * M episodes per task = total eval episodes
            for ep in range(self._eval_episodes):
                eval_demo_seed = ep + self._eval_from_eps_number
                logging.info('%s: Starting episode %d, number %d.' % (name, eval_demo_seed, ep))

                current_gt_scheme = 'unknown'
                scheme_total_counted = False

                # the current task gets reset after every M episodes
                episode_rollout = []
                generator = self._rollout_generator.generator(
                    self._step_signal, env, self._agent,
                    self._episode_length, self._timesteps,
                    eval, eval_demo_seed=eval_demo_seed,
                    record_enabled=rec_cfg.enabled,device=device)

                collector_started = False
                collector_task_name = ""
                episode_success = False
                episode_aborted = True

                # ===== MODIFICATION: Episode-level exception handling =====
                try:
                    for step_id, replay_transition in enumerate(generator):
                        if not scheme_total_counted and SCHEME_UTILS_AVAILABLE:
                            current_gt_scheme, scheme_episode_number, resolved_task_name = (
                                self._resolve_current_gt_scheme(
                                    env, episode_scheme_maps, eval_demo_seed, expected_task_name
                                )
                            )
                            if current_gt_scheme not in scheme_stats:
                                scheme_stats[current_gt_scheme] = {
                                    'success': 0, 'total': 0, 'success_steps': []
                                }
                            scheme_stats[current_gt_scheme]['total'] += 1
                            scheme_total_counted = True
                            logging.info(
                                "Episode %s resolved GT scheme: %s (task=%s, lookup_episode=%s)",
                                eval_demo_seed,
                                current_gt_scheme,
                                resolved_task_name or 'unknown',
                                scheme_episode_number,
                            )

                        if collector is not None and not collector_started:
                            # 多任务时在首帧到来后再读 task_name，避免切换时机错位。
                            collector_task_name, _ = self._get_task_name()
                            collector.start_episode(task=collector_task_name, episode_seed=eval_demo_seed)
                            collector_started = True

                        while True:
                            if self._kill_signal.value:
                                if collector is not None:
                                    collector.close()
                                env.shutdown()
                                return
                            if (eval or self._target_replay_ratio is None or
                                    self._step_signal.value <= 0 or (
                                            self._current_replay_ratio.value >
                                            self._target_replay_ratio)):
                                break
                            time.sleep(1)
                            logging.debug(
                                'Agent. Waiting for replay_ratio %f to be more than %f' %
                                (self._current_replay_ratio.value, self._target_replay_ratio))

                        with self.write_lock:
                            if len(self.agent_summaries) == 0:
                                # Only store new summaries if the previous ones
                                # have been popped by the main env runner.
                                for s in self._agent.act_summaries():
                                    self.agent_summaries.append(s)
                        episode_rollout.append(replay_transition)

                        if collector is not None and collector_started:
                            collector.maybe_add_step(
                                task=collector_task_name,
                                episode_seed=eval_demo_seed,
                                step_id=step_id,
                                obs=replay_transition.observation,
                                reward=replay_transition.reward,
                                terminal=replay_transition.terminal,
                                pred_info=(replay_transition.info or {}).get("pred_info"),
                            )

                        # ===== 新增：每帧评估阶段状态 =====
                        if hasattr(env, 'evaluate_current_phase'):
                            phase_completed, completed_phase = env.evaluate_current_phase()
                            # 记录阶段完成事件（注意：返回值已经是 completed_phase，不需要 -1）
                            if phase_completed:
                                # 检查该阶段是否已记录过（避免重复记录）
                                already_recorded = completed_phase in [r.info.get('phase_completed') for r in episode_rollout]
                                if not already_recorded:
                                    # 标记该阶段刚完成
                                    replay_transition.info['phase_completed'] = completed_phase
                                    replay_transition.info['completion_frame'] = len(episode_rollout)
                        # ============================================

                    # ===== Only process data if episode completed successfully =====
                    if len(episode_rollout) > 0:
                        try:
                            with self.write_lock:
                                for transition in episode_rollout:
                                    self.stored_transitions.append((name, transition, eval))

                                    new_transitions['eval_envs'] += 1
                                    total_transitions['eval_envs'] += 1
                                    stats_accumulator.step(transition, eval)
                                    current_task_id = transition.info['active_task_id']

                            self._num_eval_episodes_signal.value += 1

                            task_name, _ = self._get_task_name()
                            reward = episode_rollout[-1].reward
                            lang_goal = env._lang_goal
                            print(f"✓ Evaluating {task_name} | Episode {ep} | Score: {reward} | Lang Goal: {lang_goal}")

                            # ===== 新增：统计该episode的阶段完成情况 =====
                            if hasattr(env, 'get_phase_progress'):
                                phase_progress = env.get_phase_progress()
                                if phase_progress is not None:
                                    phase_status = phase_progress.get('phase_status', {})
                                    current_max_phase = 0
                                    for phase_id in range(1, 5):
                                        if phase_status.get(phase_id, False):
                                            phase_success_counts[phase_id] += 1
                                            current_max_phase = phase_id
                                    max_phases_reached.append(current_max_phase)
                                    logging.info(f"Phase progress: {phase_status}, Max phase: {current_max_phase}")
                            # ===========================================

                            # save recording with quota control
                            if rec_cfg.enabled:
                                success = reward > 0.99

                                # Check if we should save this video based on quota
                                should_save = False
                                if success and success_videos_saved < max_success_videos:
                                    should_save = True
                                    success_videos_saved += 1
                                    logging.info(f"Recording success video {success_videos_saved}/{max_success_videos}")
                                elif not success and fail_videos_saved < max_fail_videos:
                                    should_save = True
                                    fail_videos_saved += 1
                                    logging.info(f"Recording fail video {fail_videos_saved}/{max_fail_videos}")

                                if should_save:
                                    # Save video to disk
                                    # 按 checkpoint 分目录存储: {save_path}/videos/{weight_name}/{task_name}_s{seed}_{succ/fail}.mp4
                                    video_dir = os.path.join(rec_cfg.save_path, 'videos', weight_name)
                                    os.makedirs(video_dir, exist_ok=True)
                                    record_file = os.path.join(video_dir,
                                                               '%s_s%s_%s.mp4' % (task_name,
                                                                                      eval_demo_seed,
                                                                                      'succ' if success else 'fail'))
                                    lang_goal = self._eval_env._lang_goal
                                    tr.save(record_file, lang_goal, reward)
                                    logging.info(f"✓ Saved video: {record_file}")
                                else:
                                    # Don't save, just clear memory
                                    tr.clear_current_snaps()
                                    logging.info(f"✗ Skipped video for episode {ep} ({'success' if success else 'fail'}), quota reached")

                                tr._cam_motion.restore_pose()

                            # Episode completed successfully
                            success_count += 1

                            # ===== 新增：更新 scheme 分层统计 =====
                            if reward > 0.99:  # 成功的 episode
                                scheme_stats[current_gt_scheme]['success'] += 1
                                # 记录成功 episode 的步数（用于计算平均步数）
                                episode_steps = len(episode_rollout)
                                scheme_stats[current_gt_scheme]['success_steps'].append(episode_steps)
                                logging.info(f"Episode {eval_demo_seed} SUCCESS with scheme '{current_gt_scheme}', steps: {episode_steps}")
                            else:
                                logging.info(f"Episode {eval_demo_seed} FAILED with scheme '{current_gt_scheme}'")
                            # =====================================

                            # ===== Memory cleanup after each episode =====
                            self.stored_transitions[:] = []
                            episode_rollout.clear()
                            # ==============================================

                            episode_success = bool(reward > 0.99)
                            episode_aborted = False

                        except (ConnectionResetError, BrokenPipeError, EOFError) as comm_error:
                            # ===== Multiprocessing communication error =====
                            logging.error(f"✗ Episode {ep} (seed {eval_demo_seed}) FAILED with communication error: {comm_error}")
                            logging.error("This usually indicates the main process or manager crashed. Skipping this episode.")
                            failed_count += 1
                            failed_episodes.append((ep, eval_demo_seed, f"CommError: {comm_error}"))
                            episode_rollout.clear()
                            # Continue to next episode
                            continue

                except StopIteration:
                    if not scheme_total_counted and SCHEME_UTILS_AVAILABLE:
                        current_gt_scheme, scheme_episode_number, resolved_task_name = (
                            self._resolve_current_gt_scheme(
                                env, episode_scheme_maps, eval_demo_seed, expected_task_name
                            )
                        )
                        if current_gt_scheme not in scheme_stats:
                            scheme_stats[current_gt_scheme] = {
                                'success': 0, 'total': 0, 'success_steps': []
                            }
                        scheme_stats[current_gt_scheme]['total'] += 1
                        scheme_total_counted = True
                        logging.info(
                            "Episode %s resolved GT scheme during StopIteration: %s (task=%s, lookup_episode=%s)",
                            eval_demo_seed,
                            current_gt_scheme,
                            resolved_task_name or 'unknown',
                            scheme_episode_number,
                        )
                    logging.warning(f"Episode {ep} (seed {eval_demo_seed}) stopped iteration, skipping...")
                    failed_count += 1
                    failed_episodes.append((ep, eval_demo_seed, "StopIteration"))
                    continue

                except Exception as e:
                    if not scheme_total_counted and SCHEME_UTILS_AVAILABLE:
                        current_gt_scheme, scheme_episode_number, resolved_task_name = (
                            self._resolve_current_gt_scheme(
                                env, episode_scheme_maps, eval_demo_seed, expected_task_name
                            )
                        )
                        if current_gt_scheme not in scheme_stats:
                            scheme_stats[current_gt_scheme] = {
                                'success': 0, 'total': 0, 'success_steps': []
                            }
                        scheme_stats[current_gt_scheme]['total'] += 1
                        scheme_total_counted = True
                        logging.info(
                            "Episode %s resolved GT scheme during failure: %s (task=%s, lookup_episode=%s)",
                            eval_demo_seed,
                            current_gt_scheme,
                            resolved_task_name or 'unknown',
                            scheme_episode_number,
                        )
                    # ===== CRITICAL CHANGE: Log error but continue to next episode =====
                    logging.exception(
                        f"✗ Episode {ep} (seed {eval_demo_seed}) FAILED with error: {e}"
                    )
                    failed_count += 1
                    failed_episodes.append((ep, eval_demo_seed, str(e)))
                    # Clear failed episode data
                    episode_rollout.clear()
                    # Continue to next episode instead of crashing
                    continue
                # =============================================================
                finally:
                    if collector is not None and collector_started:
                        collector.end_episode(success=episode_success, aborted=episode_aborted)

            # ===== MODIFICATION: Print failed episodes summary =====
            if failed_episodes:
                logging.warning(f"\n{'='*70}")
                logging.warning(f"Failed Episodes Summary ({failed_count}/{self._eval_episodes}):")
                for ep_idx, seed, error in failed_episodes:
                    logging.warning(f"  - Episode {ep_idx} (seed {seed}): {error}")
                logging.warning(f"{'='*70}\n")
            # =============================================================

            # ===== Video recording summary =====
            if rec_cfg.enabled:
                logging.info(f"\n{'='*70}")
                logging.info(f"Video Recording Summary:")
                logging.info(f"  - Success videos saved: {success_videos_saved}/{max_success_videos}")
                logging.info(f"  - Fail videos saved: {fail_videos_saved}/{max_fail_videos}")
                logging.info(f"  - Total videos saved: {success_videos_saved + fail_videos_saved}")
                logging.info(f"{'='*70}\n")
            # ===================================

            # report summaries
            summaries = []
            summaries.extend(stats_accumulator.pop())
            if len(self.agent_summaries) > 0:
                summaries.extend(self.agent_summaries)

            # ===== MODIFICATION: Add success/failed count to summaries =====
            summaries.append(ScalarSummary('eval_envs/success_count', success_count))
            summaries.append(ScalarSummary('eval_envs/failed_count', failed_count))
            # =============================================================

            # ===== 新增：阶段级别评估指标 =====
            total_episodes = success_count + failed_count
            if total_episodes > 0:
                success_rate = success_count / total_episodes
                summaries.append(ScalarSummary('eval_envs/success_rate', success_rate))
                logging.info(f"Eval Success Rate: {success_rate:.2%} ({success_count}/{total_episodes})")
                for phase_id in range(1, 5):
                    phase_rate = phase_success_counts[phase_id] / total_episodes
                    summaries.append(ScalarSummary(f'eval_envs/phase_{phase_id}_success_rate', phase_rate))

                if len(max_phases_reached) > 0:
                    avg_max_phase = sum(max_phases_reached) / len(max_phases_reached)
                    summaries.append(ScalarSummary('eval_envs/avg_max_phase', avg_max_phase))
            # ==================================

            # ===== 新增：scheme 分层评估指标 =====
            if SCHEME_UTILS_AVAILABLE:
                # 使用 get_scheme_stats_summary 计算汇总指标
                scheme_summary = get_scheme_stats_summary(scheme_stats)

                # 添加 scheme 相关指标到 summaries
                # 示例指标:
                #   - success_rate_left_grasper_scenes: left_grasper 场景的成功率
                #   - success_rate_right_grasper_scenes: right_grasper 场景的成功率
                #   - scheme_balance_gap: 两种场景成功率的差距
                #   - total_left_grasper_episodes: left_grasper 场景的总数
                #   - total_right_grasper_episodes: right_grasper 场景的总数
                #   - avg_steps_left_grasper_scenes: left_grasper 成功案例的平均步数
                #   - avg_steps_right_grasper_scenes: right_grasper 成功案例的平均步数
                for metric_name, metric_value in scheme_summary.items():
                    summaries.append(ScalarSummary(f'eval_envs/{metric_name}', metric_value))

                # 打印 scheme 分层统计结果
                logging.info(f"\n{'='*70}")
                logging.info("Scheme-Stratified Evaluation Results:")
                for scheme in ['left_grasper', 'right_grasper', 'unknown']:
                    stats = scheme_stats.get(scheme, {'success': 0, 'total': 0, 'success_steps': []})
                    if stats['total'] > 0:
                        rate = stats['success'] / stats['total']
                        avg_steps = sum(stats['success_steps']) / len(stats['success_steps']) if stats['success_steps'] else 0
                        logging.info(f"  {scheme}: {stats['success']}/{stats['total']} = {rate:.2%}, avg_steps: {avg_steps:.1f}")
                logging.info(f"  Balance Gap: {scheme_summary.get('scheme_balance_gap', 0):.4f}")
                logging.info(f"{'='*70}\n")
            # ====================================

            eval_task_name, multi_task = self._get_task_name()

            if eval_task_name and multi_task:
                for s in summaries:
                    if 'eval' in s.name:
                        s.name = '%s/%s' % (s.name, eval_task_name)

            if len(summaries) > 0:
                if multi_task:
                    task_scores = [s.value for s in summaries if f'eval_envs/return/{eval_task_name}' in s.name]
                else:
                    task_scores = [s.value for s in summaries if f'eval_envs/return' in s.name]
                task_score = task_scores[0] if task_scores else "unknown"
            else:
                task_score = "unknown"

            print(f"Finished {eval_task_name} | Final Score: {task_score}")
            print(f"  Success: {success_count}/{success_count+failed_count} episodes")
            if failed_count > 0:
                print(f"  Failed: {failed_count} episodes (see logs for details)\n")
            else:
                print(f"  All episodes completed successfully!\n")

            if self._save_metrics:
                with writer_lock:
                    writer.add_summaries(weight_name, summaries)

            self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
            self.agent_summaries[:] = []
            self.stored_transitions[:] = []

        if self._save_metrics:
            with writer_lock:
                writer.end_iteration()

        if collector is not None:
            collector.close()

        logging.info('Finished evaluation.')
        env.shutdown()

    def kill(self):
        self._kill_signal.value = True
