#!/usr/bin/env python3
"""Re-render an RLBench demo from its saved random seed."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    imageio = None

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel_path in ("repos/RLBench", "repos/YARR", "repos/PyRep", "occ_grasp_models"):
    path = REPO_ROOT / rel_path
    if path.exists():
        sys.path.insert(0, str(path))


CAMERA_NAMES = (
    "front",
    "over_shoulder_left",
    "over_shoulder_right",
    "overhead",
    "wrist_left",
    "wrist_right",
)

DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_BITRATE = "12000k"

EXPECTED_BIMANUAL_SUCCESS_TASKS = (
    "bimanual_edge_phone",
    "bimanual_pivot_phone",
    "bimanual_pick_plate",
    "bimanual_pick_fork",
)

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
    "left": ((0, 255, 255), "left"),
    "right": ((255, 0, 255), "right"),
    "contact": ((0, 0, 255), "contact"),
    "grasp": ((0, 255, 0), "grasp"),
    "affordance": ((255, 0, 0), "afford"),
}


class StopRendering(Exception):
    pass


def parse_resolution(value):
    try:
        width, height = value.lower().split("x", 1)
        width = int(width)
        height = int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Resolution must look like 1920x1080 or 3840x2160."
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive.")
    return width, height


def parse_crop(value):
    presets = {
        "left-square",
        "center-square",
        "right-square",
        "center-1280x1080",
    }
    if value in presets:
        return value
    parts = value.replace(";", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Crop must be one of left-square, center-square, right-square, "
            "center-1280x1080, or x,y,width,height."
        )
    try:
        x, y, width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Custom crop values must be integers, e.g. 0,0,1080,1080."
        ) from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            "Crop x/y must be non-negative and crop width/height must be positive."
        )
    return x, y, width, height


def parse_phase_text_fields(value):
    aliases = {
        "strategy": "strategy",
        "mechanism": "strategy",
        "phase": "phase",
        "stage": "phase",
        "arm": "arm",
        "role": "arm",
        "armrole": "arm",
        "gt": "arm",
    }
    fields = []
    for raw_part in value.replace(";", ",").split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part in ("none", "off", "false", "0"):
            return []
        if part not in aliases:
            raise argparse.ArgumentTypeError(
                "Phase text fields must be comma-separated values from "
                "strategy/mechanism, phase/stage, arm/role/armrole/gt, or none."
            )
        field = aliases[part]
        if field not in fields:
            fields.append(field)
    return fields


def success_demo_features_requested(args):
    return args.draw_keypoints or args.draw_phase_text


def validate_success_demo_features(args, task_name):
    if success_demo_features_requested(args) and task_name not in EXPECTED_BIMANUAL_SUCCESS_TASKS:
        raise ValueError(
            "Phase text and keypoint overlay are only supported for bimanual "
            "success demo tasks: "
            + ", ".join(EXPECTED_BIMANUAL_SUCCESS_TASKS)
            + f". Got task {task_name!r}."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch RLBench/CoppeliaSim, restore a saved demo random seed, "
            "and re-render the demo as an annotated video."
        )
    )
    parser.add_argument(
        "--episode-dir",
        required=True,
        type=Path,
        help="Path to an RLBench episode directory containing low_dim_obs.pkl.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Task file name, e.g. bimanual_edge_phone. If omitted, infer it "
            "from a path segment like bimanual_edge_phone.train."
        ),
    )
    parser.add_argument(
        "--camera",
        default="front",
        choices=CAMERA_NAMES,
        help="Camera to render.",
    )
    parser.add_argument(
        "--resolution",
        default=DEFAULT_RESOLUTION,
        type=parse_resolution,
        help=(
            "Output render resolution, e.g. 1920x1080 or 3840x2160. "
            f"Default: {DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output MP4 path.",
    )
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument(
        "--bitrate",
        default=DEFAULT_BITRATE,
        help="Target video bitrate, e.g. 12000k for 1080p or 40000k for 4K.",
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        default=None,
        help=(
            "Optional crop applied before writing video/PNG. For front-camera "
            "replay videos, center-1280x1080 keeps the interaction centered "
            "while matching the common blocked-video output size. Custom "
            "x,y,width,height values such as 320,0,1280,1080 are also supported."
        ),
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="Run CoppeliaSim headlessly.",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Show the CoppeliaSim window for debugging.",
    )
    parser.add_argument(
        "--ttt-file",
        default=None,
        help=(
            "Optional TTT file relative to repos/RLBench/rlbench, matching "
            "RLBench Environment(ttt_file=...)."
        ),
    )
    parser.add_argument(
        "--render-mode",
        default="OPENGL",
        choices=("OPENGL", "OPENGL3"),
        help="PyRep render mode for RGB cameras.",
    )
    parser.add_argument(
        "--record-gripper-closing",
        action="store_true",
        default=True,
        help="Record intermediate frames while grippers open/close.",
    )
    parser.add_argument(
        "--no-record-gripper-closing",
        dest="record_gripper_closing",
        action="store_false",
        help="Match older stored-demo behavior more closely by recording fewer gripper frames.",
    )
    parser.add_argument(
        "--no-gripper-overlay",
        dest="draw_grippers",
        action="store_false",
        default=True,
        help="Render raw video without left/right gripper markers.",
    )
    parser.add_argument(
        "--draw-keypoint-overlay",
        dest="draw_keypoints",
        action="store_true",
        default=False,
        help=(
            "Overlay projected contact/grasp/affordance keypoints. "
            "Only supported for the four bimanual_* success demo tasks."
        ),
    )
    parser.add_argument(
        "--no-keypoint-overlay",
        dest="draw_keypoints",
        action="store_false",
        help="Do not draw keypoint markers.",
    )
    parser.add_argument(
        "--draw-phase-text",
        dest="draw_phase_text",
        action="store_true",
        default=False,
        help=(
            "Overlay the current waypoint-execution phase at the top-left. "
            "Only supported for the four bimanual_* success demo tasks."
        ),
    )
    parser.add_argument(
        "--no-phase-text",
        dest="draw_phase_text",
        action="store_false",
        help="Do not draw phase text.",
    )
    parser.add_argument(
        "--phase-text-fields",
        type=parse_phase_text_fields,
        default=("phase",),
        help=(
            "Comma-separated fields to show in the top-left phase text. "
            "Allowed: strategy/mechanism, phase/stage, arm/role/armrole/gt, none. "
            "Default: phase."
        ),
    )
    parser.add_argument(
        "--save-frame-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG frames, useful for paper figures.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="When --save-frame-dir is set, save every Nth rendered frame.",
    )
    parser.add_argument(
        "--sample-frame",
        type=Path,
        default=None,
        help="Optional PNG path for the first rendered frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap for smoke tests. 0 means render the full demo.",
    )
    parser.add_argument(
        "--force-scheme",
        default="auto",
        choices=("auto", "opposite", "right_grasper", "left_grasper"),
        help=(
            "Temporarily override the task's selected waypoint scheme after "
            "reset. 'opposite' flips right_grasper <-> left_grasper for the "
            "same restored scene. Default: auto, no override."
        ),
    )
    parser.add_argument(
        "--allow-demo-error",
        action="store_true",
        help=(
            "Keep the rendered frames if RLBench raises DemoError. Useful for "
            "counterfactual failed demos such as forcing the opposite arm role."
        ),
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Fail if fewer than this many frames were captured. Default: 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, saved demo metadata, task import, and obs config without launching CoppeliaSim.",
    )
    return parser.parse_args()


def infer_task_name(episode_dir):
    for part in episode_dir.parts:
        if part.endswith(".train"):
            return part[: -len(".train")]
    raise ValueError(
        "Could not infer task name from episode path. Pass --task explicitly."
    )


def load_seed_demo(episode_dir):
    demo_path = episode_dir / "low_dim_obs.pkl"
    with demo_path.open("rb") as f:
        demo = pickle.load(f)
    if getattr(demo, "random_seed", None) is None:
        raise ValueError(
            f"{demo_path} does not contain demo.random_seed, so it cannot be "
            "re-rendered deterministically."
        )
    return demo


def build_obs_config(camera, resolution, render_mode_name, record_gripper_closing):
    from pyrep.const import RenderMode
    from rlbench.observation_config import CameraConfig, ObservationConfig

    render_mode = getattr(RenderMode, render_mode_name)
    cam_config = CameraConfig(
        rgb=True,
        depth=False,
        point_cloud=False,
        mask=False,
        image_size=resolution,
        render_mode=render_mode,
        depth_in_meters=True,
    )
    return ObservationConfig(
        camera_configs={camera: cam_config},
        joint_velocities=False,
        joint_positions=True,
        joint_forces=False,
        gripper_open=True,
        gripper_pose=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
        gripper_touch_forces=False,
        task_low_dim_state=False,
        record_gripper_closing=record_gripper_closing,
        robot_name="bimanual",
    )


def project_world_to_camera(point_xyz, extrinsic, intrinsic, image_size):
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    point_xyz = np.asarray(point_xyz, dtype=np.float64)

    r = extrinsic[:3, :3]
    c = extrinsic[:3, 3:4]
    r_inv = r.T
    extrinsic_w2c = np.concatenate([r_inv, -(r_inv @ c)], axis=1)
    proj = intrinsic @ extrinsic_w2c

    point_h = np.concatenate([point_xyz[:3], np.ones(1, dtype=np.float64)])
    q = proj @ point_h
    if q[2] <= 1e-6:
        return None, False

    u = q[0] / q[2]
    v = q[1] / q[2]
    width, height = image_size
    visible = 0 <= u < width and 0 <= v < height
    return np.array([u, v], dtype=np.float32), visible


def build_phase_specs(task):
    execution_phases = list(getattr(task, "execution_phases", []) or [])
    if len(execution_phases) != 4:
        raise RuntimeError(
            "Success demo phase text expects exactly four execution phases, "
            f"but {task.__class__.__name__} returned {len(execution_phases)}."
        )

    phase_specs = []
    for index, phase in enumerate(execution_phases, start=1):
        phase_specs.append(
            {
                "index": index,
                "name": PHASE_NAMES.get(index, f"Phase{index}"),
                "arm": phase.get("arm", ""),
                "waypoints": list(phase.get("waypoints", [])),
            }
        )
    return phase_specs


def format_gt_arm_label(task):
    roles = {}
    if hasattr(task, "get_role_assignment"):
        roles = task.get_role_assignment() or {}
    else:
        roles = getattr(task, "current_role_assignment", {}) or {}

    grasper = roles.get("grasper")
    if isinstance(grasper, str):
        grasper = grasper.lower()

    if grasper is None:
        scheme = getattr(task, "active_waypoint_mode", "")
        if scheme == "right_grasper":
            grasper = "right"
        elif scheme == "left_grasper":
            grasper = "left"

    if grasper == "right":
        return "GT: R grasp"
    if grasper == "left":
        return "GT: L grasp"
    return ""


def configure_success_demo_metadata(args, task):
    args.phase_specs = build_phase_specs(task)
    strategy_type = getattr(task, "STRATEGY_TYPE", None)
    try:
        strategy_type = int(strategy_type)
    except (TypeError, ValueError):
        strategy_type = None
    args.strategy_name = STRATEGY_NAMES.get(strategy_type, task.__class__.__name__)
    args.gt_arm_label = format_gt_arm_label(task)


def get_active_scheme(task):
    if hasattr(task, "get_active_scheme"):
        scheme = task.get_active_scheme()
        if scheme:
            return scheme
    return getattr(task, "active_waypoint_mode", None)


def opposite_scheme(scheme):
    if scheme == "right_grasper":
        return "left_grasper"
    if scheme == "left_grasper":
        return "right_grasper"
    raise ValueError(f"Cannot infer opposite scheme from {scheme!r}.")


def resolve_force_scheme(requested_scheme, selected_scheme):
    if requested_scheme == "auto":
        return None
    if requested_scheme == "opposite":
        return opposite_scheme(selected_scheme)
    return requested_scheme


def force_task_scheme(task, scheme):
    if scheme not in SCHEME_ROLE_ASSIGNMENTS:
        raise ValueError(f"Unsupported scheme override: {scheme!r}.")
    waypoint_sets = getattr(task, "waypoint_sets", None)
    if not isinstance(waypoint_sets, dict) or scheme not in waypoint_sets:
        raise RuntimeError(
            f"{task.__class__.__name__} does not define waypoint set {scheme!r}."
        )
    if not hasattr(task, "_setup_waypoint_mapping"):
        raise RuntimeError(
            f"{task.__class__.__name__} cannot rebuild waypoint mapping."
        )

    task.active_waypoint_mode = scheme
    task.current_role_assignment = SCHEME_ROLE_ASSIGNMENTS[scheme].copy()
    if hasattr(task, "_waypoints"):
        task._waypoints = None
    task._setup_waypoint_mapping()

    if hasattr(task, "_setup_phased_evaluator"):
        task._setup_phased_evaluator()
    phased_evaluator = getattr(task, "phased_evaluator", None)
    if phased_evaluator is not None and hasattr(phased_evaluator, "reset"):
        phased_evaluator.reset()
    return task.current_role_assignment.copy()


def get_phase_index(obs, args):
    if not getattr(args, "phase_specs", None):
        return None
    value = obs.misc.get("execution_phase_index")
    if value is None:
        return 1
    try:
        phase_index = int(value)
    except (TypeError, ValueError):
        return 1
    if phase_index < 1 or phase_index > len(args.phase_specs):
        return 1
    return phase_index


def get_phase_spec(args, phase_index):
    if phase_index is None:
        return None
    for spec in getattr(args, "phase_specs", []):
        if spec["index"] == phase_index:
            return spec
    return None


def format_phase_text(args, phase_index):
    spec = get_phase_spec(args, phase_index)
    if spec is None:
        return ""
    requested_fields = list(getattr(args, "phase_text_fields", ()))
    if not requested_fields:
        return ""

    parts = []
    if "strategy" in requested_fields:
        parts.append(getattr(args, "strategy_name", "Unknown"))
    if "phase" in requested_fields:
        parts.append(f"{spec['index']} {spec['name']}")
    if "arm" in requested_fields:
        gt_arm_label = getattr(args, "gt_arm_label", "")
        if gt_arm_label:
            parts.append(gt_arm_label)
    return " | ".join(parts)


def draw_text_with_shadow(frame_bgr, text, org, scale, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    x, y = org
    cv2.putText(
        frame_bgr,
        text,
        (x + 1, y + 1),
        font,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_marker(frame_bgr, name, uv):
    color, label = MARKER_STYLES.get(name, ((255, 255, 255), name))
    height, width = frame_bgr.shape[:2]
    scale = max(1.0, width / 1280.0)
    radius = max(3, int(round(4 * scale)))
    outline_radius = radius + max(1, int(round(0.75 * scale)))
    text_scale = 0.38 * scale

    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))
    if not (0 <= u < width and 0 <= v < height):
        return

    cv2.circle(frame_bgr, (u, v), outline_radius, (255, 255, 255), -1)
    cv2.circle(frame_bgr, (u, v), radius, color, -1)

    text_x = min(max(u + int(round(7 * scale)), 0), max(0, width - int(70 * scale)))
    text_y = min(max(v - int(round(6 * scale)), int(18 * scale)), height - 6)
    draw_text_with_shadow(
        frame_bgr, label, (text_x, text_y), text_scale, max(1, int(round(scale)))
    )


def valid_uv(uv):
    if uv is None:
        return False
    uv = np.asarray(uv, dtype=np.float64)
    return uv.size >= 2 and np.all(np.isfinite(uv[:2])) and np.all(uv[:2] >= 0)


def draw_gripper_markers(frame_bgr, obs, camera):
    height, width = frame_bgr.shape[:2]
    extrinsic = obs.misc[f"{camera}_camera_extrinsics"]
    intrinsic = obs.misc[f"{camera}_camera_intrinsics"]
    for name, pose in (
        ("right", obs.right.gripper_pose),
        ("left", obs.left.gripper_pose),
    ):
        uv, visible = project_world_to_camera(
            pose[:3], extrinsic, intrinsic, (width, height)
        )
        if visible:
            draw_marker(frame_bgr, name, uv)


def draw_keypoint_markers(frame_bgr, obs, camera):
    height, width = frame_bgr.shape[:2]
    extrinsic = obs.misc.get(f"{camera}_camera_extrinsics")
    intrinsic = obs.misc.get(f"{camera}_camera_intrinsics")
    for name in ("contact", "grasp", "affordance"):
        visible = bool(obs.misc.get(f"{camera}_{name}_visible", False))
        uv = obs.misc.get(f"{camera}_{name}_2d")
        if visible and valid_uv(uv):
            draw_marker(frame_bgr, name, uv)
            continue

        point = obs.misc.get(f"{name}_position")
        if point is None or extrinsic is None or intrinsic is None:
            continue
        uv, visible = project_world_to_camera(
            point, extrinsic, intrinsic, (width, height)
        )
        if visible:
            draw_marker(frame_bgr, name, uv)


def draw_phase_text(frame_bgr, args, phase_index):
    text = format_phase_text(args, phase_index)
    if not text:
        return
    scale = max(1.0, frame_bgr.shape[1] / 1280.0)
    draw_text_with_shadow(
        frame_bgr,
        text,
        (int(round(20 * scale)), int(round(40 * scale))),
        0.8 * scale,
        max(2, int(round(2 * scale))),
    )


def render_observation(obs, camera, args):
    frame_rgb = obs.perception_data.get(f"{camera}_rgb")
    if frame_rgb is None:
        raise RuntimeError(f"Observation does not contain {camera}_rgb.")
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if args.draw_keypoints:
        draw_keypoint_markers(frame_bgr, obs, camera)
    if args.draw_grippers:
        draw_gripper_markers(frame_bgr, obs, camera)
    return frame_bgr


def crop_frame(frame_bgr, crop_spec):
    if crop_spec is None:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    if isinstance(crop_spec, str):
        if crop_spec == "center-1280x1080":
            crop_w, crop_h = 1280, 1080
            x = (width - crop_w) // 2
            y = (height - crop_h) // 2
        else:
            side = min(width, height)
            if crop_spec == "left-square":
                x = 0
            elif crop_spec == "center-square":
                x = (width - side) // 2
            elif crop_spec == "right-square":
                x = width - side
            else:
                raise ValueError(f"Unsupported crop preset: {crop_spec}")
            y = (height - side) // 2
            crop_w = crop_h = side
    else:
        x, y, crop_w, crop_h = crop_spec

    if x < 0 or y < 0:
        raise ValueError(
            f"Crop preset {crop_spec} is larger than rendered frame {width}x{height}."
        )

    if x + crop_w > width or y + crop_h > height:
        raise ValueError(
            f"Crop {x},{y},{crop_w},{crop_h} exceeds rendered frame "
            f"{width}x{height}."
        )
    return frame_bgr[y : y + crop_h, x : x + crop_w]


def crop_output_size(render_size, crop_spec):
    width, height = render_size
    if crop_spec is None:
        return width, height
    if crop_spec == "center-1280x1080":
        crop_w, crop_h = 1280, 1080
    elif isinstance(crop_spec, str):
        side = min(width, height)
        crop_w = crop_h = side
    else:
        _, _, crop_w, crop_h = crop_spec
    if crop_w > width or crop_h > height:
        raise ValueError(
            f"Crop {crop_spec} exceeds rendered frame {width}x{height}."
        )
    return crop_w, crop_h


def open_video_writer(path, args):
    return imageio.get_writer(
        str(path),
        fps=args.fps,
        codec="libx264",
        bitrate=args.bitrate,
        macro_block_size=None,
        ffmpeg_params=[
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
        ],
    )


class FrameSink:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if args.save_frame_dir is not None:
            args.save_frame_dir.mkdir(parents=True, exist_ok=True)
        if args.sample_frame is not None:
            args.sample_frame.parent.mkdir(parents=True, exist_ok=True)

        self.writer = open_video_writer(args.output, args)
        self.count = 0
        self.output_size = None
        self.closed = False

    def append(self, obs):
        if self.args.max_frames and self.count >= self.args.max_frames:
            raise StopRendering()

        phase_index = get_phase_index(obs, self.args)
        frame_bgr = render_observation(obs, self.args.camera, self.args)
        frame_bgr = crop_frame(frame_bgr, self.args.crop)
        if self.args.draw_phase_text:
            draw_phase_text(frame_bgr, self.args, phase_index)

        if self.output_size is None:
            out_h, out_w = frame_bgr.shape[:2]
            self.output_size = (out_w, out_h)
        if self.count == 0 and self.args.sample_frame is not None:
            cv2.imwrite(str(self.args.sample_frame), frame_bgr)

        if self.args.save_frame_dir is not None and self.count % self.args.save_every == 0:
            frame_path = self.args.save_frame_dir / f"frame_{self.count:05d}.png"
            cv2.imwrite(str(frame_path), frame_bgr)

        self.writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.count += 1
        if self.args.max_frames and self.count >= self.args.max_frames:
            raise StopRendering()

    def close(self):
        if not self.closed:
            self.writer.close()
            self.closed = True


def main():
    args = parse_args()
    if args.headless and not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    args.episode_dir = args.episode_dir.resolve()
    args.output = args.output.resolve()
    args.phase_specs = []
    args.strategy_name = "Unknown"
    args.gt_arm_label = ""
    width, height = args.resolution
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative.")
    if args.min_frames < 0:
        raise ValueError("--min-frames cannot be negative.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    task_name = args.task or infer_task_name(args.episode_dir)
    validate_success_demo_features(args, task_name)
    seed_demo = load_seed_demo(args.episode_dir)
    variation = int(getattr(seed_demo, "variation_number", 0))

    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import BimanualJointPosition
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
    from rlbench.backend.exceptions import DemoError
    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.environment import Environment

    task_class = task_file_to_task_class(task_name, bimanual=True)
    obs_config = build_obs_config(
        args.camera,
        (width, height),
        args.render_mode,
        args.record_gripper_closing,
    )
    if args.dry_run:
        print("Dry run OK")
        print(f"  episode_dir: {args.episode_dir}")
        print(f"  task: {task_name} ({task_class.__name__})")
        print(f"  variation: {variation}")
        print(f"  demo_frames: {len(seed_demo)}")
        print(f"  random_seed_present: {getattr(seed_demo, 'random_seed', None) is not None}")
        print(f"  camera: {args.camera}")
        print(f"  resolution: {width}x{height}")
        print(f"  crop: {args.crop}")
        out_w, out_h = crop_output_size((width, height), args.crop)
        print(f"  output_size: {out_w}x{out_h}")
        print(f"  render_mode: {args.render_mode}")
        print(f"  record_gripper_closing: {args.record_gripper_closing}")
        print(f"  draw_gripper_overlay: {args.draw_grippers}")
        print(f"  draw_keypoint_overlay: {args.draw_keypoints}")
        print(f"  draw_phase_text: {args.draw_phase_text}")
        print(f"  phase_text_fields: {','.join(args.phase_text_fields)}")
        print(f"  force_scheme: {args.force_scheme}")
        print(f"  allow_demo_error: {args.allow_demo_error}")
        print(f"  min_frames: {args.min_frames}")
        print(f"  output: {args.output}")
        return

    missing_video_deps = []
    if cv2 is None:
        missing_video_deps.append("cv2")
    if imageio is None:
        missing_video_deps.append("imageio")
    if missing_video_deps:
        raise RuntimeError(
            "Video rendering requires "
            + ", ".join(missing_video_deps)
            + ". Activate the same Python environment you use for the "
            "existing capture script."
        )

    action_mode = BimanualMoveArmThenGripper(
        BimanualJointPosition(), BimanualDiscrete()
    )
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        robot_setup="dual_panda",
        headless=args.headless,
        ttt_file=args.ttt_file,
    )

    sink = FrameSink(args)
    selected_scheme = None
    forced_scheme = None
    forced_roles = None
    stopped_by_frame_cap = False
    demo_error = None
    try:
        env.launch()
        task_env = env.get_task(task_class)
        task_env.set_variation(variation)

        try:
            seed_demo.restore_state()
            _, _ = task_env.reset()
            selected_scheme = get_active_scheme(task_env._scene.task)
            forced_scheme = resolve_force_scheme(args.force_scheme, selected_scheme)
            if forced_scheme is not None:
                forced_roles = force_task_scheme(task_env._scene.task, forced_scheme)
            if success_demo_features_requested(args):
                configure_success_demo_metadata(args, task_env._scene.task)
            first_obs = task_env.get_observation()
            sink.append(first_obs)

            task_env._scene.get_demo(
                record=False,
                callable_each_step=sink.append,
                randomly_place=False,
            )
        except StopRendering:
            stopped_by_frame_cap = True
        except DemoError as exc:
            demo_error = exc
            if not args.allow_demo_error:
                raise
    finally:
        sink.close()
        if env is not None:
            env.shutdown()

    if sink.count < args.min_frames:
        raise RuntimeError(
            f"Only captured {sink.count} frames, below --min-frames={args.min_frames}."
        )

    output_size = sink.output_size or (width, height)
    status = "frame cap reached" if stopped_by_frame_cap else "recorded"
    if demo_error is not None:
        status = "recorded expected failure"
    print(
        f"Wrote {args.output} from {task_name} episode {args.episode_dir.name}: "
        f"{sink.count} frames, {width}x{height} render, "
        f"{output_size[0]}x{output_size[1]} output, "
        f"{args.fps} fps, camera={args.camera}, "
        f"variation={variation}, bitrate={args.bitrate}, status={status}"
    )
    if selected_scheme is not None:
        print(f"Selected scheme after reset: {selected_scheme}")
    if forced_scheme is not None:
        print(f"Forced scheme: {forced_scheme}, roles={forced_roles}")
    if demo_error is not None:
        print(f"Captured DemoError after rendering frames: {demo_error}")
    if args.sample_frame is not None:
        print(f"Sample frame: {args.sample_frame}")
    if args.save_frame_dir is not None:
        print(f"PNG frames: {args.save_frame_dir}")


if __name__ == "__main__":
    main()
