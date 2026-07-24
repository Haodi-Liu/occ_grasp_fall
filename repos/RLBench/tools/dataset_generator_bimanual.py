#!/usr/bin/env python3

import sys
import os
import logging
from functools import partial
import multiprocessing as mp
import pickle
import numpy as np
import imageio
import cv2

from rlbench import ObservationConfig
from rlbench.observation_config import CameraConfig

from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualJointVelocity
from rlbench.action_modes.arm_action_modes import BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

from rlbench.backend.exceptions import BoundaryError, InvalidActionError, TaskEnvironmentError, WaypointError, DemoError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.backend.task import BIMANUAL_TASKS_PATH
from pyrep.objects.vision_sensor import VisionSensor


import rich_click as click
from rich.logging import RichHandler
from click_prompt import choice_option
from click_prompt import filepath_option


camera_names = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"]
video_camera_names = ["overview"] + camera_names

HQ_VIDEO_RESOLUTION = (1920, 1080)
HQ_VIDEO_OUTPUT_SIZE = (1280, 1080)
HQ_VIDEO_BITRATE = "12000k"
HQ_OVERVIEW_FOV = 66.0
HQ_OVERVIEW_DISTANCE = 2.35
HQ_OVERVIEW_HEIGHT = 1.0

STRATEGY_NAMES = {
    1: "EdgeHang",
    2: "WallLever",
    3: "PressTilt",
}

PHASE_NAMES = {
    1: "PreManipulation",
    2: "Grasp",
    3: "ClearPath",
    4: "Lift",
}

SCHEME_ROLE_ASSIGNMENTS = {
    "right_grasper": {"grasper": "right", "pusher": "left"},
    "left_grasper": {"grasper": "left", "pusher": "right"},
}

MARKER_STYLES = {
    "contact": ((0, 0, 255), "contact"),
    "grasp": ((0, 255, 0), "grasp"),
    "affordance": ((255, 0, 0), "afford"),
    "left_tip": ((0, 255, 255), "left"),
    "right_tip": ((255, 0, 255), "right"),
}


class DemoVideoError(RuntimeError):
    """Raised when an annotated demo video cannot be recorded safely."""


def _normalize_vector(vector, field_name):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise DemoVideoError(f"Invalid {field_name}: {vector!r}.")
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise DemoVideoError(f"Cannot normalize zero-length {field_name}.")
    return vector / norm


def _look_at_matrix(position, target):
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = _normalize_vector(target - position, "camera forward vector")
    left = _normalize_vector(
        np.cross([0.0, 0.0, 1.0], forward), "camera left vector")
    up = _normalize_vector(
        np.cross(forward, left), "camera up vector")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = left
    matrix[:3, 1] = up
    matrix[:3, 2] = forward
    matrix[:3, 3] = position
    return matrix


def _task_centered_overview_matrix(task_instance):
    target_object = getattr(task_instance, "target_object", None)
    if target_object is None or not target_object.still_exists():
        raise DemoVideoError(
            "Cannot configure overview camera without a valid target_object.")
    object_position = np.asarray(
        target_object.get_position(), dtype=np.float64)
    if object_position.shape != (3,) or not np.all(np.isfinite(object_position)):
        raise DemoVideoError(
            f"Invalid target object position: {object_position!r}.")

    look_target = object_position.copy()
    look_target[2] = max(float(look_target[2]), 0.85)
    camera_position = look_target + np.array(
        [0.0, -HQ_OVERVIEW_DISTANCE, HQ_OVERVIEW_HEIGHT],
        dtype=np.float64)
    return _look_at_matrix(camera_position, look_target)


class HQAnnotatedDemoRecorder:
    """Stream an annotated HQ video from the live data-collection scene."""

    def __init__(self, source_camera: VisionSensor, task_instance,
                 temporary_path: str, fps: int, use_overview: bool = False):
        if fps <= 0:
            raise ValueError(f"Video fps must be positive, got {fps}.")
        if not source_camera.still_exists():
            raise DemoVideoError("The selected source camera no longer exists.")

        self._task = task_instance
        self._temporary_path = os.path.abspath(temporary_path)
        self._fps = int(fps)
        self._writer = None
        self._writer_closed = False
        self._camera = None
        self._frame_count = 0
        self._overlay_signature = None

        parent_dir = os.path.dirname(self._temporary_path)
        os.makedirs(parent_dir, exist_ok=True)
        if os.path.exists(self._temporary_path):
            os.unlink(self._temporary_path)

        try:
            # Copying the RLBench sensor preserves its exact optical/rendering
            # properties. Only the copy is made high resolution, so dataset
            # observations keep their configured image_size.
            self._camera = source_camera.copy()
            self._camera.set_resolution(list(HQ_VIDEO_RESOLUTION))
            self._camera.set_explicit_handling(True)
            if use_overview:
                # A static oblique view is needed to keep both arms, the object,
                # and all spatial markers in frame.
                self._camera.set_parent(None, keep_in_place=True)
                self._camera.set_matrix(
                    _task_centered_overview_matrix(self._task))
                self._camera.set_perspective_angle(HQ_OVERVIEW_FOV)
                self._camera.set_near_clipping_plane(0.01)
                self._camera.set_far_clipping_plane(10.0)
            else:
                # CoppeliaSim may paste a copied object with a positional
                # offset. Restore the exact source pose before following it.
                self._camera.set_pose(source_camera.get_pose())
                self._camera.set_parent(source_camera, keep_in_place=True)

            self._writer = imageio.get_writer(
                self._temporary_path,
                fps=self._fps,
                codec="libx264",
                bitrate=HQ_VIDEO_BITRATE,
                macro_block_size=None,
                ffmpeg_params=[
                    "-pix_fmt", "yuv420p",
                    "-preset", "slow",
                    "-movflags", "+faststart",
                ],
            )
        except Exception:
            self.discard()
            self.remove_camera()
            raise

    @staticmethod
    def _as_finite_xyz(value, field_name):
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,):
            raise DemoVideoError(
                f"{field_name} must have shape (3,), got {point.shape}.")
        if not np.all(np.isfinite(point)):
            raise DemoVideoError(f"{field_name} contains non-finite values.")
        return point

    @staticmethod
    def _gripper_xyz(obs, arm_name):
        arm_obs = getattr(obs, arm_name, None)
        pose = getattr(arm_obs, "gripper_pose", None)
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise DemoVideoError(
                f"{arm_name}.gripper_pose must have shape (7,), got {pose.shape}.")
        if not np.all(np.isfinite(pose)):
            raise DemoVideoError(
                f"{arm_name}.gripper_pose contains non-finite values.")
        return pose[:3]

    def _get_overlay_state(self, obs):
        if not hasattr(self._task, "get_phase_progress"):
            raise DemoVideoError(
                f"{self._task.__class__.__name__} has no get_phase_progress().")
        progress = self._task.get_phase_progress()
        if not isinstance(progress, dict):
            raise DemoVideoError("get_phase_progress() did not return a dict.")

        raw_phase = progress.get("current_phase")
        try:
            raw_phase = int(raw_phase)
        except (TypeError, ValueError) as exc:
            raise DemoVideoError(
                f"Invalid online current_phase: {raw_phase!r}.") from exc
        if raw_phase not in (1, 2, 3, 4, 5):
            raise DemoVideoError(
                f"Online current_phase must be in 1..5, got {raw_phase}.")

        # State 5 means "all four phases completed"; it is not a fifth label.
        phase_label = min(raw_phase, 4)
        stored_phase = obs.misc.get("phase_type")
        if stored_phase is not None and int(stored_phase) != phase_label:
            raise DemoVideoError(
                "Video phase disagrees with the observation label: "
                f"online_state={raw_phase}, video_label={phase_label}, "
                f"observation_phase_type={stored_phase}.")

        strategy_type = getattr(
            self._task, "STRATEGY_TYPE", obs.misc.get("strategy_type"))
        try:
            strategy_type = int(strategy_type)
        except (TypeError, ValueError) as exc:
            raise DemoVideoError(
                f"Invalid strategy type: {strategy_type!r}.") from exc
        if strategy_type not in STRATEGY_NAMES:
            raise DemoVideoError(
                f"Unsupported strategy type: {strategy_type}.")

        if not hasattr(self._task, "get_active_scheme"):
            raise DemoVideoError(
                f"{self._task.__class__.__name__} has no get_active_scheme().")
        if not hasattr(self._task, "get_role_assignment"):
            raise DemoVideoError(
                f"{self._task.__class__.__name__} has no get_role_assignment().")

        scheme = self._task.get_active_scheme()
        roles = self._task.get_role_assignment()
        expected_roles = SCHEME_ROLE_ASSIGNMENTS.get(scheme)
        if expected_roles is None:
            raise DemoVideoError(f"Unsupported GT arm scheme: {scheme!r}.")
        if not isinstance(roles, dict):
            raise DemoVideoError("get_role_assignment() did not return a dict.")
        actual_roles = {
            "grasper": roles.get("grasper"),
            "pusher": roles.get("pusher"),
        }
        if actual_roles != expected_roles:
            raise DemoVideoError(
                f"GT scheme/role mismatch: {scheme} expects {expected_roles}, "
                f"got {actual_roles}.")

        signature = (
            strategy_type, scheme, actual_roles["grasper"],
            actual_roles["pusher"])
        if self._overlay_signature is None:
            self._overlay_signature = signature
        elif signature != self._overlay_signature:
            raise DemoVideoError(
                "Strategy or GT arm scheme changed during one demo: "
                f"{self._overlay_signature} -> {signature}.")

        points = {
            "contact": self._as_finite_xyz(
                obs.misc.get("contact_position"), "contact_position"),
            "grasp": self._as_finite_xyz(
                obs.misc.get("grasp_position"), "grasp_position"),
            "left_tip": self._gripper_xyz(obs, "left"),
            "right_tip": self._gripper_xyz(obs, "right"),
        }
        if bool(obs.misc.get("has_affordance", False)):
            points["affordance"] = self._as_finite_xyz(
                obs.misc.get("affordance_position"), "affordance_position")

        return {
            "strategy_name": STRATEGY_NAMES[strategy_type],
            "phase": phase_label,
            "phase_name": PHASE_NAMES[phase_label],
            "scheme": scheme,
            "roles": actual_roles,
            "points": points,
        }

    def _project_world_to_video(self, point_xyz):
        extrinsic = np.asarray(self._camera.get_matrix(), dtype=np.float64)
        intrinsic = np.asarray(
            self._camera.get_intrinsic_matrix(), dtype=np.float64)
        if extrinsic.shape != (4, 4):
            raise DemoVideoError(
                f"HQ camera extrinsic has shape {extrinsic.shape}, expected (4, 4).")
        if intrinsic.shape != (3, 3):
            raise DemoVideoError(
                f"HQ camera intrinsic has shape {intrinsic.shape}, expected (3, 3).")
        if not np.all(np.isfinite(extrinsic)) or not np.all(np.isfinite(intrinsic)):
            raise DemoVideoError("HQ camera calibration contains non-finite values.")

        rotation = extrinsic[:3, :3]
        camera_position = extrinsic[:3, 3:4]
        rotation_inv = rotation.T
        world_to_camera = np.concatenate(
            [rotation_inv, -(rotation_inv @ camera_position)], axis=1)
        projection = intrinsic @ world_to_camera
        point_h = np.append(point_xyz, 1.0)
        projected = projection @ point_h
        if projected[2] <= 1e-6:
            return None

        u = projected[0] / projected[2]
        v = projected[1] / projected[2]
        width, height = HQ_VIDEO_RESOLUTION
        if not (0 <= u < width and 0 <= v < height):
            return None
        return int(round(float(u))), int(round(float(v)))

    @staticmethod
    def _draw_text_with_shadow(frame_bgr, text, origin, font_scale,
                               thickness):
        cv2.putText(
            frame_bgr, text, (origin[0] + 1, origin[1] + 1),
            cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0),
            thickness + 2, cv2.LINE_AA)
        cv2.putText(
            frame_bgr, text, origin, cv2.FONT_HERSHEY_DUPLEX,
            font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_markers(self, frame_bgr, points):
        for name, point in points.items():
            pixel = self._project_world_to_video(point)
            if pixel is None:
                continue
            color, label = MARKER_STYLES[name]
            cv2.circle(frame_bgr, pixel, 7, color, -1)
            cv2.circle(frame_bgr, pixel, 9, (255, 255, 255), 1)
            self._draw_text_with_shadow(
                frame_bgr, label, (pixel[0] + 10, pixel[1] - 8),
                font_scale=0.48, thickness=1)

    @staticmethod
    def _center_crop(frame_bgr):
        output_width, output_height = HQ_VIDEO_OUTPUT_SIZE
        height, width = frame_bgr.shape[:2]
        if (width, height) != HQ_VIDEO_RESOLUTION:
            raise DemoVideoError(
                f"HQ frame has size {width}x{height}, expected "
                f"{HQ_VIDEO_RESOLUTION[0]}x{HQ_VIDEO_RESOLUTION[1]}.")
        x = (width - output_width) // 2
        y = (height - output_height) // 2
        return frame_bgr[y:y + output_height, x:x + output_width]

    def capture(self, obs):
        if self._writer is None or self._writer_closed:
            raise DemoVideoError("Attempted to capture with a closed video writer.")
        if self._camera is None or not self._camera.still_exists():
            raise DemoVideoError("The HQ video camera no longer exists.")

        state = self._get_overlay_state(obs)
        self._camera.handle_explicitly()
        frame_rgb = np.clip(
            self._camera.capture_rgb() * 255.0, 0, 255).astype(np.uint8)
        if frame_rgb.shape != (
                HQ_VIDEO_RESOLUTION[1], HQ_VIDEO_RESOLUTION[0], 3):
            raise DemoVideoError(
                f"HQ capture has shape {frame_rgb.shape}, expected "
                f"({HQ_VIDEO_RESOLUTION[1]}, {HQ_VIDEO_RESOLUTION[0]}, 3).")

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._draw_markers(frame_bgr, state["points"])
        frame_bgr = self._center_crop(frame_bgr)

        roles = state["roles"]
        mechanism_and_phase = (
            f"Mechanism: {state['strategy_name']} | "
            f"Phase: {state['phase']} {state['phase_name']}")
        gt_scheme = (
            f"GT scheme: {state['scheme']} | "
            f"{roles['grasper'][0].upper()} grasp / "
            f"{roles['pusher'][0].upper()} push")
        self._draw_text_with_shadow(
            frame_bgr, mechanism_and_phase, (20, 36),
            font_scale=0.72, thickness=2)
        self._draw_text_with_shadow(
            frame_bgr, gt_scheme, (20, 70),
            font_scale=0.72, thickness=2)

        self._writer.append_data(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._frame_count += 1

    def finish(self):
        if self._writer is not None and not self._writer_closed:
            self._writer.close()
            self._writer_closed = True
        if self._frame_count <= 0:
            raise DemoVideoError("No frames were captured for the HQ demo video.")
        if (not os.path.isfile(self._temporary_path)
                or os.path.getsize(self._temporary_path) <= 0):
            raise DemoVideoError(
                f"Video encoder did not produce {self._temporary_path}.")

    @property
    def recorded_scheme(self):
        if self._overlay_signature is None:
            return None
        return self._overlay_signature[1]

    def commit(self, final_path: str):
        if not self._writer_closed:
            raise DemoVideoError("Video must be finished before it is committed.")
        final_path = os.path.abspath(final_path)
        os.replace(self._temporary_path, final_path)
        logging.info(
            "Saved annotated HQ video (%d frames) to %s",
            self._frame_count, final_path)

    def discard(self):
        if self._writer is not None and not self._writer_closed:
            try:
                self._writer.close()
            except Exception as exc:
                logging.debug("Failed to close discarded video writer: %s", exc)
            self._writer_closed = True
        if os.path.isfile(self._temporary_path):
            try:
                os.unlink(self._temporary_path)
            except OSError as exc:
                logging.warning(
                    "Failed to remove temporary video %s: %s",
                    self._temporary_path, exc)

    def remove_camera(self):
        if self._camera is not None and self._camera.still_exists():
            try:
                self._camera.remove()
            except Exception as exc:
                logging.warning("Failed to remove temporary HQ camera: %s", exc)
        self._camera = None


def save_demo(demo, example_path, variation, save_video=False, video_camera="front", video_fps=30):
    data_types = ["rgb", "depth", "point_cloud", "mask"]
    #full_camera_names = list(map(lambda x: ('_'.join(x), x[-1]), product(camera_names, data_types)))

    # Collect video frames if needed
    video_frames = [] if save_video else None

    # Save image data first, and then None the image data, and pickle
    for i, obs in enumerate(demo):
        for camera_name in camera_names:
            for dtype in data_types:

                camera_full_name = f"{camera_name}_{dtype}"
                data_path = os.path.join(example_path, camera_full_name)
                # ..todo:: actually I prefer to abort if this one exists
                os.makedirs(data_path, exist_ok=True)

                data = obs.perception_data.get(camera_full_name, None)

                if data is not None:
                    # Collect video frames from specified camera
                    if save_video and camera_name == video_camera and dtype == 'rgb':
                        video_frames.append(data.copy())

                    if dtype == 'rgb':
                        data = Image.fromarray(data)
                    elif dtype == 'depth':
                        data = utils.float_array_to_rgb_image(data, scale_factor=DEPTH_SCALE)
                    elif dtype == 'point_cloud':
                        continue
                    elif dtype == 'mask':
                        data = Image.fromarray((data * 255).astype(np.uint8))
                    else:
                        raise Exception('Invalid data type')
                    logging.debug("saving %s", camera_full_name)
                    data.save(os.path.join(data_path, f"{dtype}_{i:04d}.png"))

        # ..why don't we put everything into a pickle file?
        obs.perception_data.clear()

    # Save video if enabled
    if save_video and video_frames:
        video_path = os.path.join(example_path, f"demo_{video_camera}.mp4")
        imageio.mimsave(video_path, video_frames, fps=video_fps)
        logging.info("Saved video to %s", video_path)

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)


def run_all_variations(task_name, headless, save_path, episodes_per_task,
                       image_size, save_video=False, video_camera="front",
                       video_fps=30, video_profile="hq-annotated", ttt=None):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    from rich.logging import RichHandler
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    logging.root.name = task_name

    logging.info("Collecting data for %s", task_name)
    if save_video:
        logging.info(
            "Demo video profile: %s (camera=%s, fps=%d)",
            video_profile, video_camera, video_fps)

    # ===== Scheme统计 =====
    scheme_stats = {'right_grasper': [], 'left_grasper': [], 'unknown': []}

    tasks = [task_file_to_task_class(task_name, True)]

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    default_config_params = {"image_size": image_size, "depth_in_meters": False, "masks_as_one_channel": False}
    camera_configs = {camera_name: CameraConfig(**default_config_params) for camera_name in camera_names}
    obs_config.camera_configs = camera_configs


    # ..record date with BimanualJointPosition
    robot_setup = 'dual_panda'
    rlbench_env = Environment(
        action_mode=BimanualMoveArmThenGripper(BimanualJointPosition(), BimanualDiscrete()),
        obs_config=obs_config,
        robot_setup=robot_setup,
        headless=headless,
        ttt_file=ttt)

    rlbench_env.launch()

    tasks_with_problems = ""

    for task in tasks:
        
        task_env = rlbench_env.get_task(task)
        possible_variations = task_env.variation_count()

        logging.info("Task has %s possible variations", possible_variations)

        variation_path = os.path.join(save_path, task_env.get_name(), VARIATIONS_ALL_FOLDER)
        os.makedirs(variation_path, exist_ok=True)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        os.makedirs(episodes_path, exist_ok=True)


        abort_variation = False
        for ex_idx in range(episodes_per_task):
        # for ex_idx in range(140, 150):  # 从episode100到episode130，共30个
            attempts = 20           # 真正错误的重试次数
            scheme_skips = 0        # DemoError（方案过滤）计数
            max_scheme_skips = 20   # 方案过滤的最大重试次数（自由选择scheme时此值无影响）
            while attempts > 0 and scheme_skips < max_scheme_skips:
                video_recorder = None
                video_committed = False
                episode_path = os.path.join(
                    episodes_path, EPISODE_FOLDER % ex_idx)
                try:
                    variation = np.random.randint(possible_variations)

                    task_env = rlbench_env.get_task(task)

                    task_env.set_variation(variation)

                    descriptions, obs = task_env.reset()

                    logging.info("// Task: %s Variation %s Demo %s", task_env.get_name(), variation, ex_idx)

                    if save_video and video_profile == "hq-annotated":
                        source_camera_name = (
                            "front" if video_camera == "overview"
                            else video_camera)
                        source_camera = task_env._scene.camera_sensors.get(
                            source_camera_name)
                        if source_camera is None:
                            raise DemoVideoError(
                                f"Camera {source_camera_name!r} is unavailable.")
                        temporary_video_path = os.path.join(
                            episode_path,
                            f".demo_{video_camera}_hq.tmp.mp4")
                        video_recorder = HQAnnotatedDemoRecorder(
                            source_camera=source_camera,
                            task_instance=task_env._task,
                            temporary_path=temporary_video_path,
                            fps=video_fps,
                            use_overview=(video_camera == "overview"),
                        )

                    # TODO: for now we do the explicit looping.
                    demo_kwargs = {}
                    if video_recorder is not None:
                        # One attempt per writer prevents failed-demo frames
                        # from being mixed with the following retry.
                        demo_kwargs.update({
                            "callable_each_step": video_recorder.capture,
                            "max_attempts": 1,
                        })
                    demo, = task_env.get_demos(
                        amount=1, live_demos=True, **demo_kwargs)

                    # ===== 获取并记录scheme信息 =====
                    active_scheme = 'unknown'
                    if hasattr(task_env._task, 'get_active_scheme'):
                        active_scheme = task_env._task.get_active_scheme()
                    elif hasattr(task_env._task, 'active_waypoint_mode'):
                        active_scheme = task_env._task.active_waypoint_mode

                    if video_recorder is not None:
                        video_recorder.finish()
                        if video_recorder.recorded_scheme != active_scheme:
                            raise DemoVideoError(
                                "Recorded GT scheme disagrees with the "
                                f"completed demo: video="
                                f"{video_recorder.recorded_scheme}, "
                                f"demo={active_scheme}.")

                except DemoError as e:
                    if video_recorder is not None:
                        video_recorder.discard()
                        video_recorder.remove_camera()
                    # DemoError 是方案过滤，不计入真正错误，单独计数
                    scheme_skips += 1
                    if scheme_skips % 10 == 0:
                        logging.info(f"Scheme skip #{scheme_skips}: {e}")
                    continue

                #  NoWaypointsError
                except (BoundaryError, WaypointError, InvalidActionError,
                        TaskEnvironmentError, RuntimeError) as e:
                    if video_recorder is not None:
                        video_recorder.discard()
                        video_recorder.remove_camera()
                    logging.warning("Exception %s", e)
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            task_env.get_name(), variation, ex_idx,str(e))
                    )
                    logging.error(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                except Exception:
                    if video_recorder is not None:
                        video_recorder.discard()
                        video_recorder.remove_camera()
                    raise

                try:
                    legacy_video = (
                        save_video and video_profile == "legacy")
                    save_demo(
                        demo, episode_path, variation, legacy_video,
                        video_camera, video_fps)

                    with open(os.path.join(
                            episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                        pickle.dump(descriptions, f)

                    # 保存scheme信息到episode目录
                    scheme_info = {
                        'active_scheme': active_scheme,
                        'role_assignment': task_env._task.get_role_assignment()
                        if hasattr(task_env._task, 'get_role_assignment') else {}
                    }
                    with open(os.path.join(
                            episode_path,
                            f'scheme_info_{active_scheme}.pkl'), 'wb') as f:
                        pickle.dump(scheme_info, f)

                    if video_recorder is not None:
                        final_video_path = os.path.join(
                            episode_path, f"demo_{video_camera}.mp4")
                        video_recorder.commit(final_video_path)
                        video_committed = True

                    # ===== 记录scheme统计 =====
                    if active_scheme in scheme_stats:
                        scheme_stats[active_scheme].append(ex_idx)
                    else:
                        scheme_stats['unknown'].append(ex_idx)
                    logging.info(
                        f"Episode {ex_idx}: scheme={active_scheme}")
                    break
                finally:
                    if video_recorder is not None:
                        if not video_committed:
                            video_recorder.discard()
                        video_recorder.remove_camera()

            # 检查while循环退出原因
            if scheme_skips >= max_scheme_skips:
                logging.warning(f"Episode {ex_idx}: Skipped after {scheme_skips} scheme rejects. "
                                f"Target scheme may be rarely feasible for this task.")

            if abort_variation:
                break

        # ===== 打印scheme统计汇总 =====
        logging.info("=" * 60)
        logging.info(f"SCHEME STATISTICS for {task_env.get_name()}")
        logging.info("=" * 60)
        total_episodes = sum(len(v) for v in scheme_stats.values())
        for scheme_name, episodes in scheme_stats.items():
            if episodes:
                pct = len(episodes) / total_episodes * 100 if total_episodes > 0 else 0
                logging.info(f"  {scheme_name}: {len(episodes)} episodes ({pct:.1f}%)")
                logging.info(f"    Episode IDs: {episodes}")
        logging.info(f"  Total: {total_episodes} episodes")
        logging.info("=" * 60)

        # 保存汇总统计到文件
        stats_path = os.path.join(variation_path, 'scheme_statistics.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(scheme_stats, f)
        logging.info(f"Scheme statistics saved to {stats_path}")


    rlbench_env.shutdown()

    return tasks_with_problems



def get_bimanual_tasks():
    tasks =  [t.replace('.py', '') for t in
    os.listdir(BIMANUAL_TASKS_PATH) if t != '__init__.py' and t.endswith('.py')]
    return sorted(tasks)


@click.command()
@filepath_option("--save_path", default="/tmp/rlbench_data/",  help="Where to save the demos.")
@choice_option('--tasks', type=click.Choice(get_bimanual_tasks()), multiple=True, help='The tasks to collect. If empty, all tasks are collected.')
@click.option("--episodes_per_task", default=10, help="The number of episodes to collect per task.", prompt="Number of episodes")
@click.option("--all_variations", is_flag=True, default=True, help="Include all variations when sampling epsiodes")
#@click.option("--variations", default=-1, help="Number of variations to collect per task. -1 for all.")
@click.option("--headless/--no-headless", default=True, is_flag=True, help='Hide the simulator window')
#@click.option("--color-robot/--no-color-robot", default=False, is_flag=True, help='Colorize')
@choice_option('--image-size', type=click.Choice(["128x128", "256x256", "640x480"]), multiple=False, help='Select the image_size (width, height)')
@click.option("--save-video/--no-save-video", default=False, is_flag=True, help='Save demo video for each episode')
@click.option(
    "--video-camera", default="overview",
    type=click.Choice(video_camera_names), show_default=True,
    help="Camera to use; overview is the static two-arm presentation view.")
@click.option("--video-fps", default=30, type=click.IntRange(min=1), help='Video frame rate')
@click.option(
    "--video-profile",
    default="hq-annotated",
    type=click.Choice(["hq-annotated", "legacy"]),
    show_default=True,
    help="Use direct live HQ annotated recording or the original observation-frame video.")
@click.option("--ttt", default=None, type=str, help='Custom TTT file (e.g., left_task_design_bimanual.ttt)')
def main(save_path, tasks, episodes_per_task, all_variations, headless,
         image_size, save_video, video_camera, video_fps, video_profile, ttt):

    # ..todo check if already exits

    mp.set_start_method("spawn")

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    np.random.seed(None)

    ctx = mp.get_context('spawn')

    if save_video and video_profile == "legacy" and video_camera == "overview":
        logging.info(
            "Legacy videos use stored observation frames; selecting front "
            "because overview is an HQ-only virtual camera.")
        video_camera = "front"

    if not tasks:
        logging.error("No tasks selected!")


    logging.info("Generating %s episodes for each tasks %s with image size %s", episodes_per_task, tasks, image_size)


    image_size = list(map(int, image_size.split("x")))

    os.makedirs(save_path, exist_ok=True)

    if not all_variations:
        logging.error("Variations not supported")
        sys.exit(-1)

    logging.debug("Selected tasks %s", tasks)

    fn = partial(
        run_all_variations, headless=headless, save_path=save_path,
        episodes_per_task=episodes_per_task, image_size=image_size,
        save_video=save_video, video_camera=video_camera,
        video_fps=video_fps, video_profile=video_profile, ttt=ttt)

    # A single HQ renderer/encoder is the safe default. It avoids several
    # CoppeliaSim workers simultaneously allocating 1080p render targets.
    worker_count = 1 if (
        save_video and video_profile == "hq-annotated"
    ) else min(8, max(1, len(tasks)))
    logging.info("Using %d data collection worker(s)", worker_count)
    with ctx.Pool(processes=worker_count) as pool:
        pool.map(fn, tasks)


if __name__ == '__main__':
  main()
