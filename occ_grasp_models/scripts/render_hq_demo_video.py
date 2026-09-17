#!/usr/bin/env python3
"""Render a high-bitrate demo video from an RLBench episode."""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel_path in ("repos/RLBench", "repos/YARR", "occ_grasp_models"):
    path = REPO_ROOT / rel_path
    if path.exists():
        sys.path.insert(0, str(path))


MARKER_STYLES = {
    "left": ((0, 255, 255), "left"),
    "right": ((255, 0, 255), "right"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a high-quality annotated video from an RLBench episode."
    )
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", default="6000k")
    parser.add_argument(
        "--fit",
        choices=("contain", "stretch"),
        default="contain",
        help="contain preserves the source aspect ratio with padding; stretch fills the output.",
    )
    parser.add_argument("--no-sharpen", action="store_true")
    parser.add_argument("--sample-frame", type=Path)
    return parser.parse_args()


def load_demo(episode_dir):
    low_dim_path = episode_dir / "low_dim_obs.pkl"
    with low_dim_path.open("rb") as f:
        return pickle.load(f)


def list_rgb_frames(episode_dir, camera):
    rgb_dir = episode_dir / f"{camera}_rgb"
    frames = sorted(rgb_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found under {rgb_dir}")
    return frames


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


def resize_frame(frame_bgr, width, height, fit):
    src_h, src_w = frame_bgr.shape[:2]
    if fit == "stretch":
        resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)
        transform = (width / float(src_w), height / float(src_h), 0.0, 0.0)
        return resized, transform

    scale = min(width / float(src_w), height / float(src_h))
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - out_w) // 2
    y0 = (height - out_h) // 2
    canvas[y0 : y0 + out_h, x0 : x0 + out_w] = resized
    transform = (scale, scale, float(x0), float(y0))
    return canvas, transform


def sharpen(frame_bgr):
    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), 1.0)
    return cv2.addWeighted(frame_bgr, 1.35, blurred, -0.35, 0)


def draw_text_with_shadow(frame_bgr, text, org, scale=0.48, thickness=1):
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
    color, label = MARKER_STYLES[name]
    h, w = frame_bgr.shape[:2]
    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))
    if not (0 <= u < w and 0 <= v < h):
        return

    cv2.circle(frame_bgr, (u, v), 7, color, -1)
    cv2.circle(frame_bgr, (u, v), 9, (255, 255, 255), 1)

    text_x = min(max(u + 10, 0), max(0, w - 60))
    text_y = min(max(v - 8, 18), max(18, h - 8))
    draw_text_with_shadow(frame_bgr, label, (text_x, text_y))


def render_frame(frame_bgr, obs, camera, width, height, fit, do_sharpen):
    src_h, src_w = frame_bgr.shape[:2]
    frame_bgr, transform = resize_frame(frame_bgr, width, height, fit)
    if do_sharpen:
        frame_bgr = sharpen(frame_bgr)

    sx, sy, ox, oy = transform
    extrinsic = obs.misc[f"{camera}_camera_extrinsics"]
    intrinsic = obs.misc[f"{camera}_camera_intrinsics"]
    for name, pose in (
        ("right", obs.right.gripper_pose),
        ("left", obs.left.gripper_pose),
    ):
        uv, visible = project_world_to_camera(
            pose[:3], extrinsic, intrinsic, (src_w, src_h)
        )
        if not visible:
            continue
        uv_out = np.array([uv[0] * sx + ox, uv[1] * sy + oy], dtype=np.float32)
        draw_marker(frame_bgr, name, uv_out)
    return frame_bgr


def main():
    args = parse_args()
    episode_dir = args.episode_dir
    output = args.output

    demo = load_demo(episode_dir)
    frame_paths = list_rgb_frames(episode_dir, args.camera)
    num_frames = min(len(demo), len(frame_paths))
    if num_frames == 0:
        raise RuntimeError("No frames to render.")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output),
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

    first_frame = None
    try:
        for idx, (obs, frame_path) in enumerate(zip(demo[:num_frames], frame_paths[:num_frames])):
            frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            rendered = render_frame(
                frame_bgr,
                obs,
                args.camera,
                args.width,
                args.height,
                args.fit,
                not args.no_sharpen,
            )
            if first_frame is None:
                first_frame = rendered.copy()
            writer.append_data(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()

    if args.sample_frame and first_frame is not None:
        args.sample_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.sample_frame), first_frame)

    print(
        f"Wrote {output} ({num_frames} frames, {args.width}x{args.height}, "
        f"{args.fps} fps, bitrate={args.bitrate}, fit={args.fit})"
    )


if __name__ == "__main__":
    main()
