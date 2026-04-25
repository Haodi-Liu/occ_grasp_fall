import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _parse_episode_seed(episode_name: str, fallback: int = -1) -> int:
    # Support episode_000012 / episode12 naming styles.
    digits = "".join(ch for ch in episode_name if ch.isdigit())
    if not digits:
        return fallback
    return int(digits)


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _cap_episode_records(records: List[Dict], max_frames_per_episode: int) -> List[Dict]:
    # Sort by step_id then cap to preserve prefix semantics through C2.
    records = sorted(records, key=lambda x: int(x.get("step_id", 0)))
    if max_frames_per_episode is not None and max_frames_per_episode > 0:
        records = records[: int(max_frames_per_episode)]
    return records


def _collect_records(
    dagger_root: Path,
    round_ids: List[int],
    tasks: List[str],
    max_frames_per_episode: int,
) -> Dict[Tuple[str, int], List[Dict]]:
    grouped = defaultdict(list)

    for rid in round_ids:
        round_root = dagger_root / f"round_{int(rid):03d}"
        for task in tasks:
            task_root = round_root / task
            if not task_root.exists():
                continue

            episode_dirs = sorted([p for p in task_root.iterdir() if p.is_dir() and p.name.startswith("episode_")])
            for ep_dir in episode_dirs:
                samples_path = ep_dir / "samples.jsonl"
                if not samples_path.exists():
                    continue

                records = _load_jsonl(samples_path)
                if not records:
                    continue

                records = _cap_episode_records(records, max_frames_per_episode)

                # Prefer seed from record, fallback to dirname parsing.
                ep_seed = int(records[0].get("episode_seed", _parse_episode_seed(ep_dir.name, fallback=-1)))
                group_key = (task, ep_seed)

                for rec in records:
                    item = dict(rec)
                    item["task"] = task
                    item["episode_seed"] = int(rec.get("episode_seed", ep_seed))
                    item["round_id"] = int(rid)
                    item["episode_dir"] = str(ep_dir)
                    grouped[group_key].append(item)

    # Keep deterministic order inside each episode group.
    for key in list(grouped.keys()):
        grouped[key] = sorted(grouped[key], key=lambda x: (int(x.get("round_id", 0)), int(x.get("step_id", 0))))

    return grouped


def _split_episode_keys(
    grouped: Dict[Tuple[str, int], List[Dict]],
    eval_ratio: float,
    split_seed: int,
) -> Tuple[set, set]:
    keys = sorted(grouped.keys())
    if len(keys) == 0:
        return set(), set()

    rng = np.random.RandomState(split_seed)
    perm = list(keys)
    rng.shuffle(perm)

    eval_ratio = float(eval_ratio)
    eval_ratio = max(0.0, min(1.0, eval_ratio))

    n_total = len(perm)
    n_eval = int(round(n_total * eval_ratio))

    # Keep split stable: when ratio in (0,1), keep at least one train/eval group if possible.
    if 0.0 < eval_ratio < 1.0 and n_total >= 2:
        n_eval = max(1, min(n_total - 1, n_eval))
    else:
        n_eval = min(n_total, max(0, n_eval))

    eval_keys = set(perm[:n_eval])
    train_keys = set(perm[n_eval:])
    return train_keys, eval_keys


def _flatten_items(grouped: Dict[Tuple[str, int], List[Dict]], keys: set) -> List[Dict]:
    items = []
    for key in sorted(keys):
        items.extend(grouped[key])
    return items


def _write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build_dagger_index(
    dagger_root: str,
    round_ids: List[int],
    tasks: List[str],
    max_frames_per_episode: int,
    eval_ratio: float,
    split_seed: int,
    train_out_path: str,
    eval_out_path: str,
):
    dagger_root = Path(dagger_root)
    grouped = _collect_records(
        dagger_root=dagger_root,
        round_ids=round_ids,
        tasks=tasks,
        max_frames_per_episode=max_frames_per_episode,
    )

    if len(grouped) == 0:
        raise RuntimeError(f"No valid samples found under {dagger_root} for tasks={tasks}, rounds={round_ids}")

    train_keys, eval_keys = _split_episode_keys(grouped, eval_ratio=eval_ratio, split_seed=split_seed)

    train_items = _flatten_items(grouped, train_keys)
    eval_items = _flatten_items(grouped, eval_keys)

    common_meta = {
        "dagger_root": str(dagger_root.resolve()),
        "round_ids": [int(x) for x in round_ids],
        "tasks": list(tasks),
        "max_frames_per_episode": int(max_frames_per_episode) if max_frames_per_episode is not None else None,
        "eval_ratio": float(eval_ratio),
        "split_seed": int(split_seed),
        "split_unit": "episode_seed_per_task",
    }

    train_payload = {
        "meta": {
            **common_meta,
            "split": "online_train",
            "num_episode_groups": len(train_keys),
            "num_items": len(train_items),
        },
        "items": train_items,
    }
    eval_payload = {
        "meta": {
            **common_meta,
            "split": "online_eval",
            "num_episode_groups": len(eval_keys),
            "num_items": len(eval_items),
        },
        "items": eval_items,
    }

    _write_json(Path(train_out_path), train_payload)
    _write_json(Path(eval_out_path), eval_payload)

    print(f"[build_dagger_index] train episodes={len(train_keys)}, items={len(train_items)} -> {train_out_path}")
    print(f"[build_dagger_index] eval  episodes={len(eval_keys)}, items={len(eval_items)} -> {eval_out_path}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger_root", required=True)
    parser.add_argument("--round_ids", nargs="+", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--max_frames_per_episode", type=int, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--train_out_path", required=True)
    parser.add_argument("--eval_out_path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dagger_index(
        dagger_root=args.dagger_root,
        round_ids=args.round_ids,
        tasks=args.tasks,
        max_frames_per_episode=args.max_frames_per_episode,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        train_out_path=args.train_out_path,
        eval_out_path=args.eval_out_path,
    )
