#!/usr/bin/env python3
"""Capture high-quality live videos from bimanual RLBench tasks."""

import argparse
import math
import os
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
VIRTUAL_CAMERA_NAMES = ("overview",)
ALL_CAMERA_NAMES = VIRTUAL_CAMERA_NAMES + CAMERA_NAMES

DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_BITRATE = "12000k"
DEFAULT_OVERVIEW_POSITION = (2.35, 0.0, 1.85)
DEFAULT_OVERVIEW_TARGET = (-0.05, 0.0, 0.72)
DEFAULT_OVERVIEW_FOV = 66.0

EXPECTED_BLOCKED_TASKS = (
    "blocked_edge_phone",
    "blocked_pivot_phone",
    "blocked_pick_plate",
    "blocked_pick_fork",
)
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
    presets = {"left-square", "center-square", "right-square"}
    if value in presets:
        return value
    parts = value.replace(";", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Crop must be one of left-square, center-square, right-square, "
            "or x,y,width,height."
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


def parse_vec3(value):
    parts = value.replace(";", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values.")
    try:
        return tuple(float(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Vector values must be numbers, e.g. 1.65,-0.75,1.45."
        ) from exc


def parse_phase_text_fields(value):
    aliases = {
        "strategy": "strategy",
        "mechanism": "strategy",
        "phase": "phase",
        "stage": "phase",
        "arm": "arm",
        "role": "arm",
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
                "strategy/mechanism, phase/stage, arm/role/gt, or none."
            )
        field = aliases[part]
        if field not in fields:
            fields.append(field)
    return fields


def sensor_camera_name(camera_name):
    return "front" if camera_name == "overview" else camera_name


def success_demo_features_requested(args):
    return (
        args.draw_keypoints
        or args.draw_phase_text
        or args.phase_output_dir is not None
    )


def validate_success_demo_features(args):
    if success_demo_features_requested(args) and args.task not in EXPECTED_BIMANUAL_SUCCESS_TASKS:
        raise ValueError(
            "Phase text, keypoint overlay, and per-phase video output are only "
            "supported for bimanual success demo tasks: "
            + ", ".join(EXPECTED_BIMANUAL_SUCCESS_TASKS)
            + f". Got --task {args.task!r}."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch RLBench/CoppeliaSim and record one live bimanual demo. "
            "The default behavior writes a clean video. Optional phase text, "
            "phase clips, and keypoint overlays are intended only for the four "
            "bimanual_* success demo tasks."
        )
    )
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Task file name, e.g. blocked_edge_phone or bimanual_pick_plate. "
            "Success overlay tasks: "
            + ", ".join(EXPECTED_BIMANUAL_SUCCESS_TASKS)
            + ". Blocked failure tasks: "
            + ", ".join(EXPECTED_BLOCKED_TASKS)
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--camera",
        default="overview",
        choices=ALL_CAMERA_NAMES,
        help=(
            "Camera to render. 'overview' reuses cam_front but moves it to a "
            "clean oblique view similar to the CoppeliaSim editor viewport."
        ),
    )
    parser.add_argument(
        "--resolution",
        default=DEFAULT_RESOLUTION,
        type=parse_resolution,
        help=(
            "Output render resolution. Default: "
            f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}."
        ),
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
            "Optional crop applied before writing video/PNG. Use left-square "
            "for the left 1080x1080 area of a 1920x1080 render, or custom "
            "x,y,width,height such as 0,0,1080,1080."
        ),
    )
    parser.add_argument(
        "--variation",
        default=0,
        type=int,
        help="RLBench variation index. The blocked tasks currently use 0.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Numpy random seed for any remaining stochastic setup.",
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
        help="Show the CoppeliaSim window while recording.",
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
        help="Skip intermediate gripper open/close frames.",
    )
    parser.add_argument(
        "--draw-gripper-overlay",
        dest="draw_grippers",
        action="store_true",
        default=False,
        help="Overlay projected left/right gripper markers on the video.",
    )
    parser.add_argument(
        "--no-gripper-overlay",
        dest="draw_grippers",
        action="store_false",
        help="Write clean video without gripper markers.",
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
            "Allowed: strategy/mechanism, phase/stage, arm/role/gt, none. "
            "Default: phase."
        ),
    )
    parser.add_argument(
        "--phase-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for per-phase MP4 clips. Only supported for "
            "the four bimanual_* success demo tasks."
        ),
    )
    parser.add_argument(
        "--overview-position",
        type=parse_vec3,
        default=DEFAULT_OVERVIEW_POSITION,
        help=(
            "World xyz for the overview camera. Default: "
            + ",".join(str(v) for v in DEFAULT_OVERVIEW_POSITION)
        ),
    )
    parser.add_argument(
        "--overview-target",
        type=parse_vec3,
        default=DEFAULT_OVERVIEW_TARGET,
        help=(
            "World xyz that the overview camera looks at. Default: "
            + ",".join(str(v) for v in DEFAULT_OVERVIEW_TARGET)
        ),
    )
    parser.add_argument(
        "--overview-fov",
        type=float,
        default=DEFAULT_OVERVIEW_FOV,
        help=f"Perspective angle in degrees for --camera overview. Default: {DEFAULT_OVERVIEW_FOV}.",
    )
    parser.add_argument(
        "--overview-from-object",
        default=None,
        help=(
            "Name of a saved CoppeliaSim camera or vision sensor object whose "
            "matrix/FOV should drive --camera overview, e.g. "
            "blocked_video_view_cam. This overrides --overview-position, "
            "--overview-target, and --overview-fov."
        ),
    )
    parser.add_argument(
        "--sample-frame",
        type=Path,
        default=None,
        help="Optional PNG path for the first rendered frame.",
    )
    parser.add_argument(
        "--save-frame-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG frames.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="When --save-frame-dir is set, save every Nth rendered frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame cap for previews. 0 means record the full attempt.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=30,
        help="Fail if fewer than this many frames were captured.",
    )
    parser.add_argument(
        "--strict-success",
        action="store_true",
        help=(
            "Treat RLBench DemoError as fatal. Recommended for bimanual_* "
            "success videos; leave off for blocked_* failure videos."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate task import and arguments without launching CoppeliaSim.",
    )
    return parser.parse_args()


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


def normalize(vector, label):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize zero-length vector: {label}")
    return vector / norm


def look_at_matrix(position, target):
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = normalize(target - position, "camera forward")
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    left = np.cross(up_hint, forward)
    if np.linalg.norm(left) < 1e-6:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        left = np.cross(up_hint, forward)
    left = normalize(left, "camera left")
    up = normalize(np.cross(forward, left), "camera up")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = left
    matrix[:3, 1] = up
    matrix[:3, 2] = forward
    matrix[:3, 3] = position
    return matrix


def configure_overview_camera(scene, args):
    camera = scene.camera_sensors[args.sensor_camera]
    if args.overview_from_object:
        matrix, fov, source_type = read_view_object_matrix_and_fov(
            args.overview_from_object
        )
        camera.set_matrix(matrix)
        camera.set_perspective_angle(fov)
        position = matrix[:3, 3]
        target = position + matrix[:3, 2]
        args.resolved_overview_position = tuple(float(v) for v in position)
        args.resolved_overview_target = tuple(float(v) for v in target)
        args.resolved_overview_fov = float(fov)
        args.resolved_overview_source = f"{args.overview_from_object} ({source_type})"
    else:
        camera.set_matrix(
            look_at_matrix(args.overview_position, args.overview_target)
        )
        camera.set_perspective_angle(args.overview_fov)
        args.resolved_overview_position = args.overview_position
        args.resolved_overview_target = args.overview_target
        args.resolved_overview_fov = args.overview_fov
        args.resolved_overview_source = "manual args"
    camera.set_near_clipping_plane(0.01)
    camera.set_far_clipping_plane(10.0)


def read_view_object_matrix_and_fov(object_name):
    from pyrep.backend import sim
    from pyrep.const import ObjectType
    from pyrep.objects.camera import Camera
    from pyrep.objects.object import Object
    from pyrep.objects.vision_sensor import VisionSensor

    if not Object.exists(object_name):
        raise RuntimeError(
            f"Overview source object '{object_name}' does not exist in this task scene. "
            "Create it in CoppeliaSim and save the corresponding blocked_*.ttm first."
        )

    object_type = Object.get_object_type(object_name)
    if object_type == ObjectType.CAMERA:
        view_object = Camera(object_name)
        fov = math.degrees(
            sim.simGetObjectFloatParameter(
                view_object.get_handle(),
                sim.sim_camerafloatparam_perspective_angle,
            )
        )
    elif object_type == ObjectType.VISION_SENSOR:
        view_object = VisionSensor(object_name)
        fov = view_object.get_perspective_angle()
    else:
        raise RuntimeError(
            f"Overview source object '{object_name}' is a {object_type.name}; "
            "expected a CoppeliaSim Camera or VisionSensor."
        )

    return view_object.get_matrix(), fov, object_type.name


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


def sanitize_filename_part(value):
    text = str(value)
    chars = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text]
    return "".join(chars).strip("_") or "phase"


def build_phase_specs(task):
    execution_phases = list(getattr(task, "execution_phases", []) or [])
    if len(execution_phases) != 4:
        raise RuntimeError(
            "Success demo phase output expects exactly four execution phases, "
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

    if x + crop_w > width or y + crop_h > height:
        raise ValueError(
            f"Crop {x},{y},{crop_w},{crop_h} exceeds rendered frame "
            f"{width}x{height}."
        )
    return frame_bgr[y : y + crop_h, x : x + crop_w]


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
        if args.phase_output_dir is not None:
            args.phase_output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_frame_dir is not None:
            args.save_frame_dir.mkdir(parents=True, exist_ok=True)
        if args.sample_frame is not None:
            args.sample_frame.parent.mkdir(parents=True, exist_ok=True)

        self.writer = open_video_writer(args.output, args)
        self.phase_writers = {}
        self.phase_paths = {}
        self.phase_counts = {}
        self.count = 0
        self.output_size = None
        self.closed = False

    def _phase_path(self, phase_index):
        spec = get_phase_spec(self.args, phase_index)
        if spec is None:
            return None
        name = sanitize_filename_part(spec["name"])
        return (
            self.args.phase_output_dir
            / f"{self.output.stem}_phase{phase_index:02d}_{name}.mp4"
        )

    def _write_phase_frame(self, phase_index, frame_bgr):
        if self.args.phase_output_dir is None or phase_index is None:
            return
        writer = self.phase_writers.get(phase_index)
        if writer is None:
            path = self._phase_path(phase_index)
            if path is None:
                return
            writer = open_video_writer(path, self.args)
            self.phase_writers[phase_index] = writer
            self.phase_paths[phase_index] = path
            self.phase_counts[phase_index] = 0
        writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.phase_counts[phase_index] += 1

    def append(self, obs):
        if self.args.max_frames and self.count >= self.args.max_frames:
            raise StopRendering()

        phase_index = get_phase_index(obs, self.args)
        frame_bgr = render_observation(obs, self.args.sensor_camera, self.args)
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
        self._write_phase_frame(phase_index, frame_bgr)
        self.count += 1
        if self.args.max_frames and self.count >= self.args.max_frames:
            raise StopRendering()

    def close(self):
        if not self.closed:
            self.writer.close()
            for writer in self.phase_writers.values():
                writer.close()
            self.closed = True


def main():
    args = parse_args()
    args.output = args.output.resolve()
    if args.phase_output_dir is not None:
        args.phase_output_dir = args.phase_output_dir.resolve()
    args.sensor_camera = sensor_camera_name(args.camera)
    args.phase_specs = []
    args.strategy_name = "Unknown"
    args.gt_arm_label = ""
    width, height = args.resolution

    validate_success_demo_features(args)
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative.")
    if args.min_frames < 0:
        raise ValueError("--min-frames cannot be negative.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    if args.headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    np.random.seed(args.seed)

    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import BimanualJointPosition
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
    from rlbench.backend.exceptions import DemoError
    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.environment import Environment

    task_class = task_file_to_task_class(args.task, bimanual=True)
    obs_config = build_obs_config(
        args.sensor_camera,
        (width, height),
        args.render_mode,
        args.record_gripper_closing,
    )

    if args.dry_run:
        print("Dry run OK")
        print(f"  task: {args.task} ({task_class.__name__})")
        print(f"  variation: {args.variation}")
        print(f"  camera: {args.camera}")
        print(f"  sensor_camera: {args.sensor_camera}")
        print(f"  resolution: {width}x{height}")
        print(f"  crop: {args.crop}")
        print(f"  render_mode: {args.render_mode}")
        print(f"  output: {args.output}")
        print(f"  success_demo_features: {success_demo_features_requested(args)}")
        print(f"  draw_phase_text: {args.draw_phase_text}")
        print(f"  phase_text_fields: {','.join(args.phase_text_fields)}")
        print(f"  draw_keypoint_overlay: {args.draw_keypoints}")
        print(f"  phase_output_dir: {args.phase_output_dir}")
        if args.camera == "overview":
            print(f"  overview_from_object: {args.overview_from_object}")
            print(f"  overview_position: {args.overview_position}")
            print(f"  overview_target: {args.overview_target}")
            print(f"  overview_fov: {args.overview_fov}")
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
        static_positions=True,
        ttt_file=args.ttt_file,
    )

    sink = FrameSink(args)
    stopped_by_frame_cap = False
    demo_error = None
    try:
        env.launch()
        task_env = env.get_task(task_class)
        task_env.set_variation(args.variation)
        descriptions, _ = task_env.reset()

        if hasattr(task_env._scene.task, "post_placement_setup"):
            task_env._scene.task.post_placement_setup()
        if success_demo_features_requested(args):
            configure_success_demo_metadata(args, task_env._scene.task)
        if args.camera == "overview":
            configure_overview_camera(task_env._scene, args)

        try:
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
            if args.strict_success:
                raise
    finally:
        sink.close()
        if env is not None:
            env.shutdown()

    if sink.count < args.min_frames:
        raise RuntimeError(
            f"Only captured {sink.count} frames, below --min-frames={args.min_frames}."
        )

    status = "frame cap reached" if stopped_by_frame_cap else "recorded"
    if demo_error is not None:
        status = "recorded expected failure"

    print(
        f"Wrote {args.output} from {args.task}: {sink.count} frames, "
        f"{width}x{height} render, {sink.output_size[0]}x{sink.output_size[1]} output, "
        f"{args.fps} fps, camera={args.camera}, "
        f"sensor={args.sensor_camera}, bitrate={args.bitrate}, status={status}"
    )
    if descriptions:
        print(f"Description: {descriptions[0]}")
    if demo_error is not None:
        print(f"Captured DemoError after rendering frames: {demo_error}")
    if args.camera == "overview":
        print(f"Overview source: {getattr(args, 'resolved_overview_source', 'manual args')}")
        print(f"Overview position: {getattr(args, 'resolved_overview_position', args.overview_position)}")
        print(f"Overview target: {getattr(args, 'resolved_overview_target', args.overview_target)}")
        print(f"Overview fov: {getattr(args, 'resolved_overview_fov', args.overview_fov)}")
    if args.sample_frame is not None:
        print(f"Sample frame: {args.sample_frame}")
    if args.save_frame_dir is not None:
        print(f"PNG frames: {args.save_frame_dir}")
    if args.phase_output_dir is not None:
        print(f"Phase clips: {args.phase_output_dir}")
        for phase_index in sorted(sink.phase_paths):
            spec = get_phase_spec(args, phase_index)
            name = spec["name"] if spec is not None else f"Phase{phase_index}"
            path = sink.phase_paths[phase_index]
            count = sink.phase_counts.get(phase_index, 0)
            print(f"  phase {phase_index} {name}: {path} ({count} frames)")


if __name__ == "__main__":
    main()
