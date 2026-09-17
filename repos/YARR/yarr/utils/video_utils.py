import os
import logging
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench import Environment
from rlbench.backend.observation import Observation
import imageio
import cv2

# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = <YOUR_PATH_TO_COPPELIASIM>

class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(180)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        self.origin.rotate([0, 0, self.speed])


class TaskRecorder(object):

    def __init__(self, env: Environment, cam_motion: CameraMotion, fps=30, overlay_cfg=None):
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._snaps = []
        self._current_snaps = []
        self._current_frame_indices = []
        self._frame_count = 0
        self._overlay_cfg = overlay_cfg

    def take_snap(self, obs: Observation):
        self._cam_motion.step()
        frame = (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8)
        frame = self._apply_overlay(frame, obs)
        frame_idx = self._frame_count
        self._frame_count += 1

        if self._should_keep_frame(frame_idx):
            self._current_snaps.append(frame)
            self._current_frame_indices.append(frame_idx)
    
    def save(self, path, lang_goal, reward):
        print(f"Saving recording ... {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        image_size = self._cam_motion.cam.get_resolution()
        save_video = self._save_video_enabled()
        save_frame_dir = self._save_frame_dir()

        if save_frame_dir:
            self._save_png_frames(path, save_frame_dir)

        if save_video:
            # 创建一个空的列表来存储每一帧图像
            frames = []

            for image in self._current_snaps:
                # 将图像从 RGB 转换为 BGR（OpenCV 使用 BGR）
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = (0.45 * image_size[0]) / 640
                font_thickness = 2

                if lang_goal:
                    lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
                    lang_textX = (image_size[0] - lang_textsize[0]) // 2

                    # 在图像上添加文本
                    frame = cv2.putText(frame, lang_goal, org=(lang_textX, image_size[1] - 35),
                                        fontScale=font_scale, fontFace=font, color=(0, 0, 0),
                                        thickness=font_thickness, lineType=cv2.LINE_AA)

                # 将每一帧添加到 frames 列表中
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            # 使用 imageio.mimsave 保存视频
            imageio.mimsave(path, frames, fps=self._fps)

        # 清空当前的快照列表
        self._current_snaps = []
        self._current_frame_indices = []
        self._frame_count = 0

    def clear_current_snaps(self):
        """Clear current snapshots without saving to free memory"""
        num_snaps = len(self._current_snaps)
        self._current_snaps = []
        self._current_frame_indices = []
        self._frame_count = 0
        logging.info(f"Cleared {num_snaps} snapshots to free memory")

    def _save_video_enabled(self):
        cfg = self._overlay_cfg
        if cfg is None:
            return True
        return bool(getattr(cfg, "save_video", True))

    def _save_frame_dir(self):
        cfg = self._overlay_cfg
        if cfg is None:
            return None
        save_dir = getattr(cfg, "save_frame_dir", None)
        if save_dir is None:
            return None
        save_dir = str(save_dir).strip()
        return save_dir or None

    def _save_every(self):
        cfg = self._overlay_cfg
        if cfg is None:
            return 1
        return max(1, int(getattr(cfg, "save_every", 1)))

    def _save_start_frame(self):
        cfg = self._overlay_cfg
        if cfg is None:
            return 0
        return max(0, int(getattr(cfg, "save_start_frame", 0)))

    def _should_keep_frame(self, frame_idx):
        if self._save_video_enabled():
            return True
        if not self._save_frame_dir():
            return False
        if frame_idx < self._save_start_frame():
            return False
        return (frame_idx - self._save_start_frame()) % self._save_every() == 0

    def _save_png_frames(self, video_path, save_frame_dir):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        episode_frame_dir = os.path.join(save_frame_dir, stem)
        os.makedirs(episode_frame_dir, exist_ok=True)

        saved = 0
        for frame_idx, image in zip(self._current_frame_indices, self._current_snaps):
            if frame_idx < self._save_start_frame():
                continue
            if (frame_idx - self._save_start_frame()) % self._save_every() != 0:
                continue

            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame_bgr, crop_suffix = self._maybe_crop_saved_frame(frame_bgr)
            frame_path = os.path.join(
                episode_frame_dir, f"{stem}_f{frame_idx:05d}{crop_suffix}.png"
            )
            cv2.imwrite(frame_path, frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            saved += 1
        logging.info("Saved %d PNG frames to %s", saved, episode_frame_dir)

    def _maybe_crop_saved_frame(self, frame_bgr):
        cfg = self._overlay_cfg
        if cfg is None or not bool(getattr(cfg, "save_square_crop", False)):
            return frame_bgr, ""

        height, width = frame_bgr.shape[:2]
        crop = getattr(cfg, "frame_crop", None)
        if crop is not None:
            x, y, size = [int(v) for v in crop]
        else:
            crop_scale = float(getattr(cfg, "frame_crop_scale", 5.0 / 6.0))
            x_bias = float(getattr(cfg, "frame_x_bias", 0.44))
            y_bias = float(getattr(cfg, "frame_y_bias", 0.50))
            size = int(round(min(width, height) * crop_scale))
            size = max(1, min(size, width, height))
            x = int(round((width - size) * x_bias))
            y = int(round((height - size) * y_bias))

        size = max(1, min(size, width, height))
        x = max(0, min(x, width - size))
        y = max(0, min(y, height - size))
        return frame_bgr[y:y + size, x:x + size], f"_x{x}_y{y}_s{size}"

    def _get_env_overlay_state(self, include_phase=True):
        if self._env is None:
            return {}
        if hasattr(self._env, "get_env_overlay_state"):
            if include_phase:
                return self._env.get_env_overlay_state()
            return self._env.get_env_overlay_state(include_phase=False)
        inner_env = getattr(self._env, "env", None)
        if inner_env is not None and hasattr(inner_env, "get_env_overlay_state"):
            if include_phase:
                return inner_env.get_env_overlay_state()
            return inner_env.get_env_overlay_state(include_phase=False)
        return {}

    def _apply_overlay(self, frame_rgb, obs: Observation):
        cfg = self._overlay_cfg
        if cfg is None or not bool(getattr(cfg, "overlay_enabled", False)):
            return frame_rgb
        if getattr(cfg, "overlay_source", "env") != "env":
            return frame_rgb

        draw_phase = bool(getattr(cfg, "overlay_draw_phase", True))
        state = self._get_env_overlay_state(include_phase=draw_phase)
        if not state:
            return frame_rgb

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        self._draw_state_text(frame_bgr, state)

        points_3d = state.get("points_3d", {}) or {}
        draw_keypoints = bool(getattr(cfg, "overlay_draw_keypoints", True))
        draw_grippers = bool(getattr(cfg, "overlay_draw_grippers", True))
        for name, point in points_3d.items():
            if name in ("contact", "grasp", "affordance") and not draw_keypoints:
                continue
            if name in ("left_tip", "right_tip") and not draw_grippers:
                continue
            uv, visible = self._project_world_to_camera(point)
            if visible:
                self._draw_marker(frame_bgr, name, uv)

        if bool(getattr(cfg, "overlay_draw_xyz_table", False)):
            self._draw_xyz_table(frame_bgr, points_3d, draw_keypoints, draw_grippers)

        self._draw_subtask_prompt(frame_bgr, state)

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _draw_state_text(self, frame_bgr, state):
        strategy_name = state.get("strategy_name", "Unknown")
        text_parts = [strategy_name]
        if bool(getattr(self._overlay_cfg, "overlay_draw_phase", True)):
            phase_progress = state.get("phase_progress") or {}
            phase = int(phase_progress.get("current_phase", 1))
            phase_name = phase_progress.get("current_phase_name", "PreManipulation")
            text_parts.append(f"{phase} {phase_name}")
        if bool(getattr(self._overlay_cfg, "overlay_draw_gt_arm_scheme", True)):
            gt_label = self._format_gt_arm_scheme_label(state)
            if gt_label:
                text_parts.append(gt_label)
        text = " | ".join(text_parts)
        scale = max(1.0, frame_bgr.shape[1] / 1280.0)
        self._draw_text_with_shadow(
            frame_bgr, text, (int(round(20 * scale)), int(round(40 * scale))),
            cv2.FONT_HERSHEY_DUPLEX, 0.8 * scale,
            (255, 255, 255), max(2, int(round(2 * scale)))
        )

    def _draw_subtask_prompt(self, frame_bgr, state):
        prompt = state.get("active_subtask_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return

        subtask_type = state.get("active_subtask_type")
        label = "Subtask: " if subtask_type is None else f"Subtask {subtask_type}: "
        text = label + prompt.strip()

        scale = max(1.0, frame_bgr.shape[1] / 1280.0)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55 * scale
        thickness = max(1, int(round(scale)))
        max_width = frame_bgr.shape[1] - int(round(40 * scale))
        lines = self._wrap_text(text, font, font_scale, thickness, max_width)

        x = int(round(20 * scale))
        line_height = int(round(27 * scale))
        bottom_y = frame_bgr.shape[0] - int(round(72 * scale))
        first_y = bottom_y - line_height * (len(lines) - 1)
        for index, line in enumerate(lines):
            self._draw_text_with_shadow(
                frame_bgr, line, (x, first_y + index * line_height),
                font, font_scale, (255, 255, 255), thickness
            )

    @staticmethod
    def _wrap_text(text, font, font_scale, thickness, max_width):
        lines = []
        current = ""
        for word in text.split():
            candidate = word if not current else current + " " + word
            width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
            if current and width > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    @staticmethod
    def _format_gt_arm_scheme_label(state):
        scheme = state.get("gt_arm_scheme", "unknown")
        roles = state.get("gt_arm_roles", {}) or {}
        grasper = roles.get("grasper")
        if isinstance(grasper, str):
            grasper = grasper.lower()

        if grasper is None:
            if scheme == "right_grasper":
                grasper = "right"
            elif scheme == "left_grasper":
                grasper = "left"

        if grasper == "right":
            return "GT: R grasp"
        if grasper == "left":
            return "GT: L grasp"
        return ""

    def _project_world_to_camera(self, point_3d):
        try:
            cam = self._cam_motion.cam
            extrinsic = self._coerce_extrinsic(cam.get_matrix())
            intrinsic = self._coerce_intrinsic(cam.get_intrinsic_matrix())
            width, height = [int(v) for v in cam.get_resolution()]

            R = extrinsic[:3, :3]
            C = extrinsic[:3, 3:4]
            R_inv = R.T
            extrinsics_w2c = np.concatenate([R_inv, -(R_inv @ C)], axis=-1)
            proj = intrinsic @ extrinsics_w2c

            p = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float64)
            q = proj @ p
            if q[2] <= 1e-6:
                return np.array([-1.0, -1.0]), False

            u = q[0] / q[2]
            v = q[1] / q[2]
            visible = 0 <= u < width and 0 <= v < height
            return np.array([u, v]), visible
        except Exception as exc:
            logging.debug("Failed to project overlay point: %s", exc)
            return np.array([-1.0, -1.0]), False

    @staticmethod
    def _coerce_extrinsic(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix
        if matrix.shape == (3, 4):
            return np.vstack([matrix, np.array([0.0, 0.0, 0.0, 1.0])])
        if matrix.size == 16:
            return matrix.reshape(4, 4)
        if matrix.size == 12:
            matrix = matrix.reshape(3, 4)
            return np.vstack([matrix, np.array([0.0, 0.0, 0.0, 1.0])])
        raise ValueError(f"Unexpected camera extrinsic shape: {matrix.shape}")

    @staticmethod
    def _coerce_intrinsic(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape == (3, 3):
            return matrix
        if matrix.size == 9:
            return matrix.reshape(3, 3)
        raise ValueError(f"Unexpected camera intrinsic shape: {matrix.shape}")

    def _draw_marker(self, frame_bgr, name, uv):
        styles = {
            "contact": ((0, 0, 255), "contact"),
            "grasp": ((0, 255, 0), "grasp"),
            "affordance": ((255, 0, 0), "afford"),
            "left_tip": ((0, 255, 255), "left"),
            "right_tip": ((255, 0, 255), "right"),
        }
        color, label = styles.get(name, ((255, 255, 255), name))
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))
        scale = max(1.0, frame_bgr.shape[1] / 1280.0)
        radius = max(5, int(round(7 * scale)))
        outline_radius = max(radius + 2, int(round(9 * scale)))
        thickness = max(1, int(round(scale)))
        cv2.circle(frame_bgr, (u, v), radius, color, -1)
        cv2.circle(frame_bgr, (u, v), outline_radius, (255, 255, 255), thickness)
        self._draw_text_with_shadow(
            frame_bgr, label,
            (u + int(round(10 * scale)), v - int(round(8 * scale))),
            cv2.FONT_HERSHEY_DUPLEX, 0.48 * scale,
            (255, 255, 255), thickness
        )

    def _draw_xyz_table(self, frame_bgr, points_3d, draw_keypoints, draw_grippers):
        rows = []
        for name in ("contact", "grasp", "affordance", "left_tip", "right_tip"):
            if name in ("contact", "grasp", "affordance") and not draw_keypoints:
                continue
            if name in ("left_tip", "right_tip") and not draw_grippers:
                continue
            point = points_3d.get(name)
            if point is None:
                continue
            rows.append(f"{name}: {point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")

        if not rows:
            return

        margin = 20
        line_height = 22
        x = max(margin, frame_bgr.shape[1] - 360)
        y = margin + 20
        for i, row in enumerate(rows):
            self._draw_text_with_shadow(
                frame_bgr, row, (x, y + i * line_height),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
            )

    @staticmethod
    def _draw_text_with_shadow(frame_bgr, text, org, font, scale, color, thickness):
        x, y = org
        cv2.putText(
            frame_bgr, text, (x + 1, y + 1), font, scale, (0, 0, 0),
            thickness + 2, cv2.LINE_AA
        )
        cv2.putText(
            frame_bgr, text, org, font, scale, color, thickness, cv2.LINE_AA
        )
