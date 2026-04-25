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
        self._overlay_cfg = overlay_cfg
        # self._prediction_snaps = []

    def take_snap(self, obs: Observation):
        self._cam_motion.step()
        frame = (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8)
        frame = self._apply_overlay(frame, obs)
        self._current_snaps.append(frame)
        
    # def take_snap_prediction(self, obs: Observation):
    #     self._prediction_snaps.append(
    #         (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))

    # def save(self, path, lang_goal, reward):
    #     print(f"Converting to video ... {path}")
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     # OpenCV QT version can conflict with PyRep, so import here
    #     import cv2
    #     image_size = self._cam_motion.cam.get_resolution()
    #     video = cv2.VideoWriter(
    #             path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self._fps,
    #             tuple(image_size))
        
    #     for image in self._current_snaps:
    #         frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         font_scale = (0.45 * image_size[0]) / 640
    #         font_thickness = 2


    #         if lang_goal:

    #             lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
    #             lang_textX = (image_size[0] - lang_textsize[0]) // 2

    #             frame = cv2.putText(frame, lang_goal, org=(lang_textX, image_size[1] - 35),
    #                                 fontScale=font_scale, fontFace=font, color=(0, 0, 0),
    #                                 thickness=font_thickness, lineType=cv2.LINE_AA)
                

    #         video.write(frame)
    #     video.release()
    #     self._current_snaps = []
    
    def save(self, path, lang_goal, reward):
        print(f"Converting to video ... {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        image_size = self._cam_motion.cam.get_resolution()

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

    def clear_current_snaps(self):
        """Clear current snapshots without saving to free memory"""
        num_snaps = len(self._current_snaps)
        self._current_snaps = []
        logging.info(f"Cleared {num_snaps} snapshots to free memory")

    def _apply_overlay(self, frame_rgb, obs: Observation):
        cfg = self._overlay_cfg
        if cfg is None or not bool(getattr(cfg, "overlay_enabled", False)):
            return frame_rgb
        if self._env is None or not hasattr(self._env, "_last_pred_info"):
            return frame_rgb
        pred_info = getattr(self._env, "_last_pred_info")
        if not pred_info:
            return frame_rgb

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        strategy_name = pred_info.get("strategy_name")
        phase_name = pred_info.get("phase_name")
        if strategy_name or phase_name:
            text = "Strategy: %s | Phase: %s" % (
                strategy_name or "N/A", phase_name or "N/A"
            )
            cv2.putText(
                frame_bgr, text, (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )

        pip_cam = getattr(cfg, "overlay_pip_camera", None)
        if pip_cam:
            cam_key = f"{pip_cam}_rgb"
            cam_img = None
            pred_obs = getattr(self._env, "_last_pred_obs_dict", None)
            if pred_obs is not None and cam_key in pred_obs:
                cam_img = pred_obs.get(cam_key)
                if isinstance(cam_img, np.ndarray) and cam_img.ndim == 3 and cam_img.shape[0] == 3 and cam_img.shape[-1] != 3:
                    cam_img = np.transpose(cam_img, (1, 2, 0))
            else:
                cam_img = getattr(obs, "perception_data", {}).get(cam_key)
            if cam_img is not None:
                if cam_img.dtype != np.uint8:
                    cam_img = (cam_img * 255.).astype(np.uint8)
                cam_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)

                kp_by_cam = pred_info.get("keypoints_2d", {})
                kp_cam = kp_by_cam.get(pip_cam, {})
                pred_img_size = pred_info.get("img_size")
                h_cam, w_cam = cam_bgr.shape[:2]
                if pred_img_size:
                    sx = w_cam / float(pred_img_size)
                    sy = h_cam / float(pred_img_size)
                else:
                    sx = sy = 1.0

                aff_vis = pred_info.get("affordance_visible")
                kp_colors = {
                    "contact": (0, 0, 255),     # red
                    "grasp": (0, 255, 0),       # green
                    "affordance": (255, 0, 0),  # blue
                }
                for kp_name, color in kp_colors.items():
                    if kp_name == "affordance" and aff_vis is not None and aff_vis < 0.5:
                        continue
                    coords = kp_cam.get(kp_name)
                    if coords is None or len(coords) != 2:
                        continue
                    u = int(max(0, min(w_cam - 1, coords[0] * sx)))
                    v = int(max(0, min(h_cam - 1, coords[1] * sy)))
                    cv2.circle(cam_bgr, (u, v), 4, color, -1)

                scale = float(getattr(cfg, "overlay_pip_scale", 0.35))
                margin = 16
                pip_w = int(frame_bgr.shape[1] * scale)
                pip_h = int(pip_w * (h_cam / float(w_cam)))
                pip_w = max(1, min(pip_w, frame_bgr.shape[1] - 2 * margin))
                pip_h = max(1, min(pip_h, frame_bgr.shape[0] - 2 * margin))
                pip_img = cv2.resize(cam_bgr, (pip_w, pip_h))

                x = frame_bgr.shape[1] - pip_w - margin
                y = frame_bgr.shape[0] - pip_h - margin
                frame_bgr[y:y + pip_h, x:x + pip_w] = pip_img
                cv2.rectangle(
                    frame_bgr, (x - 2, y - 2), (x + pip_w + 2, y + pip_h + 2),
                    (255, 255, 255), 2
                )

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


    # def save_prediction(self, path, lang_goal, reward):
    #     print(f"Converting to video ... {path}")
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     # OpenCV QT version can conflict with PyRep, so import here
    #     import cv2
    #     image_size = self._cam_motion.cam.get_resolution()
    #     video = cv2.VideoWriter(
    #             path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 2,
    #             tuple(image_size))
        
    #     for image in self._prediction_snaps:
    #         frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         font_scale = (0.45 * image_size[0]) / 640
    #         font_thickness = 2


    #         if lang_goal:

    #             lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
    #             lang_textX = (image_size[0] - lang_textsize[0]) // 2

    #             frame = cv2.putText(frame, lang_goal, org=(lang_textX, image_size[1] - 35),
    #                                 fontScale=font_scale, fontFace=font, color=(0, 0, 0),
    #                                 thickness=font_thickness, lineType=cv2.LINE_AA)
                

    #         video.write(frame)
    #     video.release()
    #     self._prediction_snaps = []
