# Adapted from ACT_BC_ENC to use keyframe-based replay.

import logging
from typing import List

import numpy as np
from omegaconf import DictConfig
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import observation_utils
from helpers import demo_loading_utils
from helpers import utils
from helpers.preprocess_agent import PreprocessAgent
from agents.act_bc_keypoint.act_bc_keypoint_agent import ActBCKeypointAgent
from agents.act_bc_keypoint.act_policy import ACTPolicy

import torch
from torch.multiprocessing import Process, Value, Manager

LOW_DIM_SIZE = 8

def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  image_size=[256, 256],
                  replay_size=3e5,
                  keyframe_seq_len: int = 1):

    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))

    K = keyframe_seq_len
    observation_elements.extend([
        ObservationElement('right_curr_gripper_pose', (7,), np.float32),
        ObservationElement('right_curr_gripper_open', (1,), np.float32),
        ObservationElement('left_curr_gripper_pose', (7,), np.float32),
        ObservationElement('left_curr_gripper_open', (1,), np.float32),
        ObservationElement('right_next_gripper_pose_seq', (K, 7), np.float32),
        ObservationElement('right_next_gripper_open_seq', (K, 1), np.float32),
        ObservationElement('left_next_gripper_pose_seq', (K, 7), np.float32),
        ObservationElement('left_next_gripper_open_seq', (K, 1), np.float32),
        ObservationElement('is_pad', (K,), np.int32),
    ])

    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # === Keypoint 2D fields ===
    observation_elements.append(ObservationElement('has_affordance', (), np.bool_))
    for cname in cameras:
        observation_elements.extend([
            ObservationElement(f'{cname}_contact_2d', (2,), np.float32),
            ObservationElement(f'{cname}_grasp_2d', (2,), np.float32),
            ObservationElement(f'{cname}_affordance_2d', (2,), np.float32),
            ObservationElement(f'{cname}_contact_visible', (), np.bool_),
            ObservationElement(f'{cname}_grasp_visible', (), np.bool_),
            ObservationElement(f'{cname}_affordance_visible', (), np.bool_),
        ])

    observation_elements.extend([
        ReplayElement('task', (), str),
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8 * 2,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(obs_tp1: Observation):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate([obs_tp1.gripper_pose[:3], quat,
                           [float(obs_tp1.gripper_open)]])


def _add_keyframes_to_replay(keypoint_idx: int,
                             keypoints: List[int],
                             cfg: DictConfig,
                             task: str,
                             replay: ReplayBuffer,
                             demo: Demo):

    cameras = cfg.rlbench.cameras
    robot_name = cfg.method.robot_name

    k = keypoints[keypoint_idx]
    obs = demo[k]
    next_kp_idx = min(keypoint_idx + 1, len(keypoints) - 1)
    obs_tp1 = demo[keypoints[next_kp_idx]]
    K = cfg.method.keyframe_seq_len

    right_pose_seq = np.zeros((K, 7), dtype=np.float32)
    right_open_seq = np.zeros((K, 1), dtype=np.float32)
    left_pose_seq = np.zeros((K, 7), dtype=np.float32)
    left_open_seq = np.zeros((K, 1), dtype=np.float32)
    is_pad = np.zeros((K,), dtype=np.int32)

    future_available = len(keypoints) - (keypoint_idx + 1)
    pad_start = max(future_available, 1)
    for i in range(K):
        kp_i = keypoint_idx + 1 + i
        if kp_i >= len(keypoints):
            kp_i = len(keypoints) - 1
        if i >= pad_start:
            is_pad[i] = 1

        obs_i = demo[keypoints[kp_i]]
        right_pose_seq[i] = obs_i.right.gripper_pose
        right_open_seq[i] = [obs_i.right.gripper_open]
        left_pose_seq[i] = obs_i.left.gripper_pose
        left_open_seq[i] = [obs_i.left.gripper_open]

    if obs_tp1.is_bimanual and robot_name == "bimanual":
        right_action = _get_action(obs_tp1.right)
        left_action = _get_action(obs_tp1.left)
        action = np.append(right_action, left_action)
    else:
        logging.error("ACT_BC_KEY expects bimanual observations.")
        raise Exception("Invalid robot name or observation type.")

    terminal = (keypoint_idx == len(keypoints) - 1)
    reward = float(terminal) if terminal else 0

    obs_dict = observation_utils.extract_obs(
        obs, t=k, prev_action=None, cameras=cameras,
        episode_length=cfg.rlbench.episode_length, robot_name=robot_name)

    keys_to_remove = [k for k in obs_dict.keys() if '_depth' in k]
    for key in keys_to_remove:
        del obs_dict[key]

    obs_dict['low_dim_state'] = np.concatenate(
        [obs_dict['right_low_dim_state'], obs_dict['left_low_dim_state']])
    del obs_dict['right_low_dim_state']
    del obs_dict['left_low_dim_state']
    if 'right_ignore_collisions' in obs_dict:
        del obs_dict['right_ignore_collisions']
    if 'left_ignore_collisions' in obs_dict:
        del obs_dict['left_ignore_collisions']

    obs_dict['right_curr_gripper_pose'] = obs.right.gripper_pose
    obs_dict['right_curr_gripper_open'] = np.array([obs.right.gripper_open], dtype=np.float32)
    obs_dict['left_curr_gripper_pose'] = obs.left.gripper_pose
    obs_dict['left_curr_gripper_open'] = np.array([obs.left.gripper_open], dtype=np.float32)
    obs_dict['right_next_gripper_pose_seq'] = right_pose_seq
    obs_dict['right_next_gripper_open_seq'] = right_open_seq
    obs_dict['left_next_gripper_pose_seq'] = left_pose_seq
    obs_dict['left_next_gripper_open_seq'] = left_open_seq
    obs_dict['is_pad'] = is_pad

    # ===== Keypoint 2D from misc (current keyframe) =====
    misc = obs.misc if hasattr(obs, 'misc') else {}
    obs_dict['has_affordance'] = misc.get('has_affordance', False)
    for cam_name in cameras:
        obs_dict[f'{cam_name}_contact_2d'] = misc.get(
            f'{cam_name}_contact_2d', np.array([-1.0, -1.0], dtype=np.float32)
        ).astype(np.float32)
        obs_dict[f'{cam_name}_grasp_2d'] = misc.get(
            f'{cam_name}_grasp_2d', np.array([-1.0, -1.0], dtype=np.float32)
        ).astype(np.float32)
        obs_dict[f'{cam_name}_affordance_2d'] = misc.get(
            f'{cam_name}_affordance_2d', np.array([-1.0, -1.0], dtype=np.float32)
        ).astype(np.float32)
        obs_dict[f'{cam_name}_contact_visible'] = misc.get(f'{cam_name}_contact_visible', False)
        obs_dict[f'{cam_name}_grasp_visible'] = misc.get(f'{cam_name}_grasp_visible', False)
        obs_dict[f'{cam_name}_affordance_visible'] = misc.get(f'{cam_name}_affordance_visible', False)

    others = {'demo': True, 'task': task}
    others.update(obs_dict)
    timeout = False
    replay.add(action, reward, terminal, timeout, **others)


def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str]):

    logging.debug('Filling %s replay ...' % task)
    for d_idx in range(num_demos):
        demo = rlbench_utils.get_stored_demos(
            amount=1, image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1, task_name=f"{task}.train",
            obs_config=obs_config, random_selection=False,
            from_episode_number=d_idx)[0]

        keypoints = demo_loading_utils.keypoint_discovery(
            demo, method=cfg.method.keypoint_method)
        if len(keypoints) == 0:
            continue

        if rank == 0:
            logging.info(
                f"Loading Demo({d_idx}) - found {len(keypoints)} keyframes - {task}"
            )

        for kp_idx in range(len(keypoints)):
            _add_keyframes_to_replay(kp_idx, keypoints, cfg, task, replay, demo)

    logging.debug('Replay filled with keyframe demos.')


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig,
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str]):
    manager = Manager()
    store = manager.dict()
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device(
                'cuda:%s' % (e_idx % torch.cuda.device_count())
                if torch.cuda.is_available() else 'cpu')

            fill_replay(cfg, obs_config, rank, replay, task,
                        num_demos, demo_augmentation,
                        demo_augmentation_every_n, cameras)

    logging.debug('Replay filled with multi demos.')


def create_agent(cfg: DictConfig):
    actor_net = ACTPolicy(cfg.method)

    bc_agent = ActBCKeypointAgent(
        actor_network=actor_net,
        camera_names=cfg.rlbench.cameras,
        lr=cfg.method.lr,
        weight_decay=cfg.method.weight_decay,
        grad_clip=cfg.method.grad_clip,
        episode_length=cfg.rlbench.episode_length,
        train_demo_path=cfg.method.train_demo_path,
        task_name=cfg.rlbench.tasks[0],
        temporal_ensemble_k=getattr(cfg.method, "temporal_ensemble_k", 0.01),
        keyframe_pos_thresh=getattr(cfg.method, "keyframe_pos_thresh", 0.01),
        keyframe_rot_thresh=getattr(cfg.method, "keyframe_rot_thresh", 0.10),
        keyframe_max_steps=getattr(cfg.method, "keyframe_max_steps", 20))

    return PreprocessAgent(pose_agent=bc_agent, norm_type='imagenet')
