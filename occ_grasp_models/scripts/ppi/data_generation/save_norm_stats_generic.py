import argparse
import sys
from pathlib import Path

import numpy as np
import torch


OCC_GRASP_ROOT = Path(__file__).resolve().parents[3]
if str(OCC_GRASP_ROOT) not in sys.path:
    sys.path.insert(0, str(OCC_GRASP_ROOT))

from ppi.common.get_data_continuous import GetDataContinuous
from ppi.common.get_data_keyframe import GetDataKeyframe
from ppi.common.get_data_keyframe_continuous import GetDataKeyframeContinuous
from ppi.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


# Default config. If other task settings differ, edit these values or override them
# with command-line arguments when running the script.
TASK = "bimanual_pivot_phone"
PREDICTION_TYPE = "keyframe_continuous"
EP_START = 0
EP_END = 149
KP_NUM = 10
PCD_TYPE = "rgb_pcd_rps6144"
POINT_FLOW_TYPE = "world_ordered_rps200"
SKIP_EP = []

DATA_ROOT = OCC_GRASP_ROOT / "data"
LANG_EMB_PATH = DATA_ROOT / "training_processed" / "instruction_embeddings.pkl"


class RunningStats:
    def __init__(self, dim):
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        else:
            array = array.reshape(-1, array.shape[-1])
        self.count += array.shape[0]
        self.sum += array.sum(axis=0, dtype=np.float64)
        self.sumsq += np.square(array, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.min = np.minimum(self.min, array.min(axis=0))
        self.max = np.maximum(self.max, array.max(axis=0))

    def finalize(self):
        if self.count == 0:
            raise ValueError("RunningStats received no data.")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)
        return {
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }


def make_single_field(stats_dict, output_min=-1.0, output_max=1.0, range_eps=1e-4):
    input_min = torch.from_numpy(stats_dict["min"])
    input_max = torch.from_numpy(stats_dict["max"])
    input_mean = torch.from_numpy(stats_dict["mean"])
    input_std = torch.from_numpy(stats_dict["std"])

    input_range = input_max - input_min
    ignore = input_range < range_eps
    input_range = input_range.clone()
    input_range[ignore] = output_max - output_min

    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore] = (output_max + output_min) / 2 - input_min[ignore]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict={
            "min": input_min,
            "max": input_max,
            "mean": input_mean,
            "std": input_std,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PPI normalization stats with streaming updates."
    )
    parser.add_argument("--task", default=TASK)
    parser.add_argument(
        "--prediction-type",
        default=PREDICTION_TYPE,
        choices=["continuous", "keyframe", "keyframe_continuous"],
    )
    parser.add_argument("--ep-start", type=int, default=EP_START)
    parser.add_argument("--ep-end", type=int, default=EP_END)
    parser.add_argument("--kp-num", type=int, default=KP_NUM)
    parser.add_argument("--pcd-type", default=PCD_TYPE)
    parser.add_argument("--point-flow-type", default=POINT_FLOW_TYPE)
    parser.add_argument(
        "--skip-ep",
        type=int,
        nargs="*",
        default=SKIP_EP,
        help="Episodes to skip, for example: --skip-ep 3 17 42",
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--lang-emb-path", default=str(LANG_EMB_PATH))
    return parser.parse_args()


def build_task_paths(args):
    data_root = Path(args.data_root)
    task = args.task
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


def load_episode_index(args, data_path, lang_emb_path):
    if args.prediction_type == "continuous":
        loader = GetDataContinuous(data_path=str(data_path), lang_emb_path=str(lang_emb_path))
        return loader.process_episodes(args.ep_start, args.ep_end, args.skip_ep)
    if args.prediction_type == "keyframe":
        loader = GetDataKeyframe(data_path=str(data_path), lang_emb_path=str(lang_emb_path))
        return loader.process_episodes(args.ep_start, args.ep_end, args.skip_ep, args.kp_num)
    loader = GetDataKeyframeContinuous(
        data_path=str(data_path), lang_emb_path=str(lang_emb_path)
    )
    return loader.process_episodes(args.ep_start, args.ep_end, args.skip_ep, args.kp_num)


def infer_feature_dim(index_array, root_path, value_subdir):
    if len(index_array) == 0:
        raise ValueError(f"No samples found under {root_path}.")
    episode, step = index_array[0]
    feature_path = root_path / f"episode{episode}" / value_subdir / f"step{step:03d}.npy"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    sample = np.load(feature_path)
    return int(sample.shape[-1])


def update_feature_stats(stats, index_array, root_path, value_subdir, key):
    for episode, step in index_array:
        feature_path = root_path / f"episode{episode}" / value_subdir / f"step{step:03d}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
        stats[key].update(np.load(feature_path))


def build_output_path(args, norm_stats_root):
    name = (
        f"norm_stats_{args.task}_{args.pcd_type}_{args.prediction_type}"
        f"_{args.point_flow_type}"
    )
    return norm_stats_root / f"{name}.pth"


def main():
    args = parse_args()
    task_paths = build_task_paths(args)

    if args.ep_end < args.ep_start:
        raise ValueError("ep_end must be greater than or equal to ep_start.")

    if not task_paths["data_path"].exists():
        raise FileNotFoundError(f"Missing task data path: {task_paths['data_path']}")
    if not task_paths["pcd_root"].exists():
        raise FileNotFoundError(f"Missing point-cloud path: {task_paths['pcd_root']}")
    if not task_paths["dino_root"].exists():
        raise FileNotFoundError(f"Missing dino-feature path: {task_paths['dino_root']}")
    if args.prediction_type == "keyframe_continuous" and not task_paths["point_flow_root"].exists():
        raise FileNotFoundError(f"Missing point-flow path: {task_paths['point_flow_root']}")
    if not Path(args.lang_emb_path).exists():
        raise FileNotFoundError(f"Missing language embedding file: {args.lang_emb_path}")

    root = load_episode_index(args, task_paths["data_path"], Path(args.lang_emb_path))
    data = root["data"]

    if len(data["action"]) == 0:
        raise ValueError("No samples were loaded. Check episode range and skip list.")

    stats = {
        "action": RunningStats(dim=data["action"].shape[-1]),
        "agent_pos": RunningStats(dim=data["state"].shape[-1]),
        "lang": RunningStats(dim=data["lang"].shape[-1]),
        "point_cloud": RunningStats(
            dim=infer_feature_dim(data["point_cloud"], task_paths["pcd_root"], args.pcd_type)
        ),
        "dino_feature": RunningStats(
            dim=infer_feature_dim(data["dino_feature"], task_paths["dino_root"], args.pcd_type)
        ),
    }

    if args.prediction_type == "keyframe_continuous":
        stats["point_flow"] = RunningStats(
            dim=infer_feature_dim(
                data["point_flow"], task_paths["point_flow_root"], args.point_flow_type
            )
        )
        stats["initial_point_flow"] = RunningStats(
            dim=infer_feature_dim(
                data["initial_point_flow"],
                task_paths["point_flow_root"],
                args.point_flow_type,
            )
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

    normalizer = LinearNormalizer()
    for key, tracker in stats.items():
        normalizer[key] = make_single_field(tracker.finalize())

    out_path = build_output_path(args, task_paths["norm_stats_root"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalizer.state_dict(), out_path)

    try:
        display_path = out_path.relative_to(OCC_GRASP_ROOT)
    except ValueError:
        display_path = out_path

    print(f"saved to {display_path}")


if __name__ == "__main__":
    main()
