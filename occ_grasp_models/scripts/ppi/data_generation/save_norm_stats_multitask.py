import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


OCC_GRASP_ROOT = Path(__file__).resolve().parents[3]
if str(OCC_GRASP_ROOT) not in sys.path:
    sys.path.insert(0, str(OCC_GRASP_ROOT))

from ppi.model.common.normalizer import LinearNormalizer
from scripts.ppi.data_generation.save_norm_stats_generic import (
    DATA_ROOT,
    LANG_EMB_PATH,
    RunningStats,
    infer_feature_dim,
    load_episode_index,
    make_single_field,
    update_feature_stats,
)


DEFAULT_TASKS = [
    "bimanual_edge_phone",
    "bimanual_pivot_phone",
    "bimanual_pick_plate",
    "bimanual_pick_fork",
]
DEFAULT_OUTPUT_NAME = "bimanual_four_tasks"
DEFAULT_PREDICTION_TYPE = "keyframe_continuous"
DEFAULT_EP_START = 0
DEFAULT_EP_END = 149
DEFAULT_KP_NUM = 10
DEFAULT_PCD_TYPE = "rgb_pcd_rps6144"
DEFAULT_POINT_FLOW_TYPE = "world_ordered_rps200"
DEFAULT_SKIP_EP = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a multitask PPI normalization stats file with streaming updates."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task names to aggregate, for example: bimanual_edge_phone bimanual_pivot_phone",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output task alias used in the saved norm-stats filename.",
    )
    parser.add_argument(
        "--prediction-type",
        default=DEFAULT_PREDICTION_TYPE,
        choices=["continuous", "keyframe", "keyframe_continuous"],
    )
    parser.add_argument("--ep-start", type=int, default=DEFAULT_EP_START)
    parser.add_argument("--ep-end", type=int, default=DEFAULT_EP_END)
    parser.add_argument("--kp-num", type=int, default=DEFAULT_KP_NUM)
    parser.add_argument("--pcd-type", default=DEFAULT_PCD_TYPE)
    parser.add_argument("--point-flow-type", default=DEFAULT_POINT_FLOW_TYPE)
    parser.add_argument(
        "--skip-ep",
        type=int,
        nargs="*",
        default=DEFAULT_SKIP_EP,
        help="Episodes to skip for every task, for example: --skip-ep 3 17 42",
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--lang-emb-path", default=str(LANG_EMB_PATH))
    return parser.parse_args()


def build_task_paths(task, data_root):
    data_root = Path(data_root)
    return {
        "data_path": data_root / "training_raw" / task / "all_variations" / "episodes",
        "pcd_root": data_root
        / "training_processed"
        / "point_cloud"
        / task
        / "all_variations"
        / "episodes",
        "dino_root": data_root
        / "training_processed"
        / "dino_feature"
        / task
        / "all_variations"
        / "episodes",
        "point_flow_root": data_root
        / "training_processed"
        / "point_flow"
        / task
        / "all_variations"
        / "episodes",
        "norm_stats_root": data_root / "training_processed" / "norm_stats",
    }


def validate_args(args):
    if not args.tasks:
        raise ValueError("At least one task must be provided via --tasks.")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError(f"Duplicate tasks are not allowed: {args.tasks}")
    if args.ep_end < args.ep_start:
        raise ValueError("ep_end must be greater than or equal to ep_start.")
    if not Path(args.lang_emb_path).exists():
        raise FileNotFoundError(f"Missing language embedding file: {args.lang_emb_path}")


def validate_task_paths(task, task_paths, prediction_type):
    if not task_paths["data_path"].exists():
        raise FileNotFoundError(f"[{task}] Missing task data path: {task_paths['data_path']}")
    if not task_paths["pcd_root"].exists():
        raise FileNotFoundError(f"[{task}] Missing point-cloud path: {task_paths['pcd_root']}")
    if not task_paths["dino_root"].exists():
        raise FileNotFoundError(f"[{task}] Missing dino-feature path: {task_paths['dino_root']}")
    if prediction_type == "keyframe_continuous" and not task_paths["point_flow_root"].exists():
        raise FileNotFoundError(
            f"[{task}] Missing point-flow path: {task_paths['point_flow_root']}"
        )


def ensure_tracker(stats, key, dim, task):
    if key not in stats:
        stats[key] = RunningStats(dim=dim)
        return

    existing_dim = stats[key].sum.shape[0]
    if existing_dim != dim:
        raise ValueError(
            f"[{task}] Feature '{key}' has dim {dim}, expected {existing_dim}."
        )


def make_task_loader_args(args, task):
    return SimpleNamespace(
        task=task,
        prediction_type=args.prediction_type,
        ep_start=args.ep_start,
        ep_end=args.ep_end,
        kp_num=args.kp_num,
        pcd_type=args.pcd_type,
        point_flow_type=args.point_flow_type,
        skip_ep=args.skip_ep,
        data_root=args.data_root,
        lang_emb_path=args.lang_emb_path,
    )


def update_task_stats(stats, args, task):
    task_paths = build_task_paths(task, args.data_root)
    validate_task_paths(task, task_paths, args.prediction_type)

    task_args = make_task_loader_args(args, task)
    root = load_episode_index(task_args, task_paths["data_path"], Path(args.lang_emb_path))
    data = root["data"]

    if len(data["action"]) == 0:
        raise ValueError(f"[{task}] No samples were loaded. Check episode range and skip list.")

    ensure_tracker(stats, "action", data["action"].shape[-1], task)
    ensure_tracker(stats, "agent_pos", data["state"].shape[-1], task)
    ensure_tracker(stats, "lang", data["lang"].shape[-1], task)
    ensure_tracker(
        stats,
        "point_cloud",
        infer_feature_dim(data["point_cloud"], task_paths["pcd_root"], args.pcd_type),
        task,
    )
    ensure_tracker(
        stats,
        "dino_feature",
        infer_feature_dim(data["dino_feature"], task_paths["dino_root"], args.pcd_type),
        task,
    )

    if args.prediction_type == "keyframe_continuous":
        ensure_tracker(
            stats,
            "point_flow",
            infer_feature_dim(
                data["point_flow"], task_paths["point_flow_root"], args.point_flow_type
            ),
            task,
        )
        ensure_tracker(
            stats,
            "initial_point_flow",
            infer_feature_dim(
                data["initial_point_flow"],
                task_paths["point_flow_root"],
                args.point_flow_type,
            ),
            task,
        )

    print(
        f"[{task}] loaded samples={len(data['action'])}, "
        f"point_cloud_refs={len(data['point_cloud'])}, "
        f"dino_refs={len(data['dino_feature'])}"
    )

    stats["action"].update(data["action"])
    stats["agent_pos"].update(data["state"])
    stats["lang"].update(data["lang"])

    update_feature_stats(
        stats, data["point_cloud"], task_paths["pcd_root"], args.pcd_type, "point_cloud"
    )
    update_feature_stats(
        stats, data["dino_feature"], task_paths["dino_root"], args.pcd_type, "dino_feature"
    )

    if args.prediction_type == "keyframe_continuous":
        update_feature_stats(
            stats,
            data["point_flow"],
            task_paths["point_flow_root"],
            args.point_flow_type,
            "point_flow",
        )
        update_feature_stats(
            stats,
            data["initial_point_flow"],
            task_paths["point_flow_root"],
            args.point_flow_type,
            "initial_point_flow",
        )

    return len(data["action"])


def build_output_path(args):
    norm_stats_root = Path(args.data_root) / "training_processed" / "norm_stats"
    filename = (
        f"norm_stats_{args.output_name}_{args.pcd_type}_{args.prediction_type}"
        f"_{args.point_flow_type}.pth"
    )
    return norm_stats_root / filename


def main():
    args = parse_args()
    validate_args(args)

    stats = dict()
    total_samples = 0
    print(f"Aggregating tasks: {', '.join(args.tasks)}")
    for task in args.tasks:
        total_samples += update_task_stats(stats, args, task)

    normalizer = LinearNormalizer()
    for key, tracker in stats.items():
        normalizer[key] = make_single_field(tracker.finalize())

    out_path = build_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalizer.state_dict(), out_path)

    try:
        display_path = out_path.relative_to(OCC_GRASP_ROOT)
    except ValueError:
        display_path = out_path

    print(f"total_samples={total_samples}")
    print(f"saved to {display_path}")


if __name__ == "__main__":
    main()
