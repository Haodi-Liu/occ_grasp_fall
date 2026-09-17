#!/usr/bin/env python3
"""Extract near-square, high-quality crops from evaluation videos."""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2


def parse_crop(value):
    try:
        x, y, size = (int(part) for part in value.split(",", 2))
    except Exception as exc:
        raise argparse.ArgumentTypeError("Crop must be formatted as x,y,size.") from exc
    if size <= 0:
        raise argparse.ArgumentTypeError("Crop size must be positive.")
    return x, y, size


def parse_crop_rect(value):
    try:
        x, y, crop_width, crop_height = (int(part) for part in value.split(",", 3))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Crop rectangle must be formatted as x,y,width,height."
        ) from exc
    if crop_width <= 0 or crop_height <= 0:
        raise argparse.ArgumentTypeError("Crop width and height must be positive.")
    return x, y, crop_width, crop_height


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract near-square PNG crops from one or more MP4 files. The "
            "default crop removes the top/bottom text overlays by cropping "
            "only as much vertical space as needed, while keeping extra "
            "horizontal context around the robot arms and task scene."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input MP4 files or directories containing MP4 files.",
    )
    parser.add_argument("--glob", default="*.mp4", help="Glob used for directory inputs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--output-video-dir",
        type=Path,
        default=None,
        help="Optional directory for full-length cropped MP4 videos.",
    )
    parser.add_argument(
        "--save-every",
        "--every-frames",
        dest="save_every",
        type=int,
        default=150,
        help="Save one frame every N frames. Ignored when --every-seconds is set.",
    )
    parser.add_argument(
        "--every-seconds",
        type=float,
        default=None,
        help="Save one frame every N seconds, based on the video FPS.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-per-video",
        type=int,
        default=6,
        help="Maximum saved frames per video. Use 0 for no limit.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Maximum number of videos to process after sorting. Use 0 for no limit.",
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        default=None,
        help="Explicit square crop as x,y,size in source-video pixels.",
    )
    parser.add_argument(
        "--crop-rect",
        type=parse_crop_rect,
        default=None,
        help="Explicit rectangular crop as x,y,width,height in source-video pixels.",
    )
    parser.add_argument(
        "--crop-height-scale",
        "--crop-scale",
        dest="crop_height_scale",
        type=float,
        default=29.0 / 36.0,
        help="Default crop height as a fraction of the source-video height.",
    )
    parser.add_argument(
        "--crop-aspect",
        type=float,
        default=46.0 / 29.0,
        help="Default crop width / height ratio.",
    )
    parser.add_argument(
        "--trim-top",
        type=int,
        default=None,
        help=(
            "Pixels to crop from the top before applying horizontal crop. "
            "Use with --trim-bottom for videos whose text overlays are only "
            "on selected edges."
        ),
    )
    parser.add_argument(
        "--trim-bottom",
        type=int,
        default=None,
        help=(
            "Pixels to crop from the bottom before applying horizontal crop. "
            "Use --trim-top 0 --trim-bottom N to keep the original top edge."
        ),
    )
    parser.add_argument(
        "--x-bias",
        type=float,
        default=0.50,
        help="Default horizontal crop position between left=0 and right=1.",
    )
    parser.add_argument(
        "--y-bias",
        type=float,
        default=3.0 / 7.0,
        help="Default vertical crop position between top=0 and bottom=1.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        choices=range(10),
        help="PNG compression level. Quality is lossless for all levels.",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=10,
        help=(
            "H.264 CRF for optional MP4 output. Lower is higher quality; 0 is "
            "lossless, 10 is visually near-lossless for these evaluation videos."
        ),
    )
    parser.add_argument(
        "--video-preset",
        default="slow",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        help="H.264 preset for optional MP4 output.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=Path,
        default=None,
        help=(
            "FFmpeg executable for optional MP4 output. By default the script "
            "prefers an FFmpeg build with libx264 support."
        ),
    )
    return parser.parse_args()


def iter_videos(inputs, pattern):
    videos = []
    for item in inputs:
        if item.is_dir():
            videos.extend(sorted(item.glob(pattern)))
        elif item.is_file():
            videos.append(item)
        else:
            raise FileNotFoundError(item)
    return sorted(set(videos))


def clamp(value, low, high):
    return max(low, min(value, high))


def clamp_crop(x, y, crop_width, crop_height, width, height):
    crop_width = clamp(crop_width, 1, width)
    crop_height = clamp(crop_height, 1, height)
    return (
        clamp(x, 0, width - crop_width),
        clamp(y, 0, height - crop_height),
        crop_width,
        crop_height,
    )


def default_crop(width, height, crop_height_scale, crop_aspect, x_bias, y_bias):
    crop_height = int(round(height * crop_height_scale))
    crop_height = clamp(crop_height, 1, height)
    crop_width = int(round(crop_height * crop_aspect))
    crop_width = clamp(crop_width, 1, width)
    x = int(round((width - crop_width) * x_bias))
    y = int(round((height - crop_height) * y_bias))
    return clamp_crop(x, y, crop_width, crop_height, width, height)


def edge_trim_crop(width, height, trim_top, trim_bottom, crop_aspect, x_bias):
    trim_top = 0 if trim_top is None else trim_top
    trim_bottom = 0 if trim_bottom is None else trim_bottom
    if trim_top < 0 or trim_bottom < 0:
        raise ValueError("--trim-top and --trim-bottom must be non-negative.")
    if trim_top + trim_bottom >= height:
        raise ValueError("--trim-top + --trim-bottom must be less than video height.")

    crop_height = height - trim_top - trim_bottom
    crop_width = int(round(crop_height * crop_aspect))
    crop_width = clamp(crop_width, 1, width)
    x = int(round((width - crop_width) * x_bias))
    return clamp_crop(x, trim_top, crop_width, crop_height, width, height)


def crop_suffix(x, y, crop_width, crop_height):
    if crop_width == crop_height:
        return f"x{x}_y{y}_s{crop_width}"
    return f"x{x}_y{y}_w{crop_width}_h{crop_height}"


def ffmpeg_has_encoder(ffmpeg, encoder):
    completed = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and encoder in completed.stdout


def resolve_ffmpeg(args):
    candidates = []
    if args.ffmpeg_bin is not None:
        candidates.append(args.ffmpeg_bin)
    candidates.append(Path("/usr/bin/ffmpeg"))

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg is not None:
        candidates.append(Path(path_ffmpeg))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        if ffmpeg_has_encoder(candidate, "libx264"):
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "Could not find an ffmpeg executable with libx264 support for MP4 "
        f"output. Checked: {checked}. Pass --ffmpeg-bin /path/to/ffmpeg if needed."
    )


def export_crop_video(video_path, output_video_dir, x, y, crop_width, crop_height, args):
    if not 0 <= args.video_crf <= 51:
        raise ValueError("--video-crf must be between 0 and 51.")
    if crop_width % 2 != 0 or crop_height % 2 != 0:
        raise ValueError(
            "MP4 H.264 yuv420p output requires even crop width and height. "
            "Pass --crop-rect or --crop with even dimensions for video export."
        )

    ffmpeg = resolve_ffmpeg(args)

    output_video_dir.mkdir(parents=True, exist_ok=True)
    suffix = crop_suffix(x, y, crop_width, crop_height)
    out_path = output_video_dir / f"{video_path.stem}_{suffix}.mp4"
    crop_filter = (
        f"crop=out_w={crop_width}:out_h={crop_height}:x={x}:y={y}:exact=1"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map_metadata",
        "0",
        "-vf",
        crop_filter,
        "-c:v",
        "libx264",
        "-preset",
        args.video_preset,
        "-crf",
        str(args.video_crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "copy",
        str(out_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"Could not write cropped video: {out_path}"
            + (f"\nffmpeg stderr:\n{stderr}" if stderr else "")
        )
    return out_path


def extract_video(video_path, output_dir, args):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video size: {video_path}")

    if args.crop is not None and args.crop_rect is not None:
        raise ValueError("Use either --crop or --crop-rect, not both.")

    if args.crop_rect is not None:
        x, y, crop_width, crop_height = args.crop_rect
        x, y, crop_width, crop_height = clamp_crop(
            x, y, crop_width, crop_height, width, height
        )
    elif args.crop is not None:
        x, y, size = args.crop
        x, y, crop_width, crop_height = clamp_crop(x, y, size, size, width, height)
    elif args.trim_top is not None or args.trim_bottom is not None:
        x, y, crop_width, crop_height = edge_trim_crop(
            width,
            height,
            args.trim_top,
            args.trim_bottom,
            args.crop_aspect,
            args.x_bias,
        )
    else:
        x, y, crop_width, crop_height = default_crop(
            width,
            height,
            args.crop_height_scale,
            args.crop_aspect,
            args.x_bias,
            args.y_bias,
        )

    step = args.save_every
    if args.every_seconds is not None:
        step = max(1, int(round(args.every_seconds * fps)))
    if step <= 0:
        raise ValueError("--save-every must be positive.")

    saved = 0
    frame_idx = max(0, args.start_frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    video_out_path = None

    while frame_count <= 0 or frame_idx < frame_count:
        if args.max_per_video and saved >= args.max_per_video:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        crop = frame[y : y + crop_height, x : x + crop_width]
        suffix = crop_suffix(x, y, crop_width, crop_height)
        out_path = output_dir / f"{video_path.stem}_f{frame_idx:05d}_{suffix}.png"
        if not cv2.imwrite(str(out_path), crop, png_params):
            raise RuntimeError(f"Could not write frame: {out_path}")

        saved += 1
        frame_idx += step

    cap.release()

    if args.output_video_dir is not None:
        video_out_path = export_crop_video(
            video_path, args.output_video_dir, x, y, crop_width, crop_height, args
        )

    message = (
        f"{video_path.name}: saved={saved}, source={width}x{height}, "
        f"crop=x{x},y{y},w{crop_width},h{crop_height}, "
        f"fps={fps:.3f}, frames={frame_count}"
    )
    if video_out_path is not None:
        message += f", video={video_out_path}"
    print(message)
    return saved


def main():
    args = parse_args()
    videos = iter_videos(args.inputs, args.glob)
    if args.limit_videos:
        videos = videos[: args.limit_videos]
    if not videos:
        raise SystemExit("No videos found.")

    total = 0
    for video_path in videos:
        total += extract_video(video_path, args.output_dir, args)
    print(f"Wrote {total} PNG frames to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
