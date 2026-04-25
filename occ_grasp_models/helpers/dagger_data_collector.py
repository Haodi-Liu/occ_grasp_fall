import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from helpers.aux_eval_visualizer import render_aux_eval_like


def _to_bool(value) -> bool:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(value.reshape(-1)[-1])
    return bool(value)


def _to_xy(value) -> List[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return [-1.0, -1.0]
    return [float(arr[0]), float(arr[1])]


def _to_uint8_rgb(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected RGB image with ndim=3, got shape={arr.shape}")

    # Support CHW and HWC image layouts.
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)

    return arr


def _uniform_pick_indices(total: int, want: int) -> List[int]:
    if total <= 0 or want <= 0:
        return []
    k = min(total, want)
    if k == total:
        return list(range(total))
    picked = np.linspace(0, total - 1, k).round().astype(int).tolist()

    # De-duplicate while preserving order.
    out = []
    seen = set()
    for idx in picked:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


class DaggerDataCollector:
    """
    C1 online replay data collector.

    Design goals:
    1) Save clean RGB + structured keypoint GT for predictor pretraining;
    2) Sample aux visualizations from collected records only;
    3) Favor prefix behavior with step interval + window + cap.
    """

    def __init__(self, cfg):
        self._cfg = cfg

        self.enabled = bool(getattr(cfg, "enabled", False))
        self.round_id = int(getattr(cfg, "round_id", 0))
        self.save_root = Path(str(getattr(cfg, "save_root", "/tmp/dagger_data")))
        self.round_root = self.save_root / f"round_{self.round_id:03d}"
        self.round_root.mkdir(parents=True, exist_ok=True)

        self.camera_names = list(getattr(cfg, "camera_names", []))

        self.collect_every_n_steps = max(1, int(getattr(cfg, "collect_every_n_steps", 1)))
        self.collect_window_steps = int(getattr(cfg, "collect_window_steps", 1 << 30))
        self.max_frames_per_episode = max(1, int(getattr(cfg, "max_frames_per_episode", 40)))

        self.collect_outcome = str(getattr(cfg, "collect_outcome", "both"))
        if self.collect_outcome not in ("both", "success_only", "fail_only"):
            raise ValueError(
                f"Unsupported collect_outcome={self.collect_outcome}, expected both/success_only/fail_only"
            )

        self.strict_required_keys = bool(getattr(cfg, "strict_required_keys", True))
        self.required_keys = list(getattr(cfg, "required_keys", []))

        aux_vis_cfg = getattr(cfg, "aux_vis", None)
        self.aux_vis_enabled = bool(getattr(aux_vis_cfg, "enabled", False)) if aux_vis_cfg is not None else False
        self.aux_vis_num_samples = (
            int(getattr(aux_vis_cfg, "num_samples_per_episode", 0)) if aux_vis_cfg is not None else 0
        )
        self.aux_vis_sample_camera = (
            str(getattr(aux_vis_cfg, "sample_camera", "over_shoulder_right"))
            if aux_vis_cfg is not None
            else "over_shoulder_right"
        )
        self.aux_vis_save_root = str(getattr(aux_vis_cfg, "save_path", "")).strip() if aux_vis_cfg is not None else ""

        # render_aux_eval_like accesses fields through getattr; keep a minimal cfg object.
        self.aux_vis_render_cfg = SimpleNamespace(
            sample_camera=self.aux_vis_sample_camera,
            sample_min_width=int(getattr(aux_vis_cfg, "sample_min_width", 300)) if aux_vis_cfg is not None else 300,
            sample_output_scale=float(getattr(aux_vis_cfg, "sample_output_scale", 0.0))
            if aux_vis_cfg is not None
            else 0.0,
        )

        self.summary_path = self.round_root / "summary.csv"
        self._ensure_summary_header()

        self._state: Optional[Dict] = None

    def _ensure_summary_header(self):
        if self.summary_path.exists():
            return
        with self.summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task",
                    "episode_seed",
                    "success",
                    "aborted",
                    "kept",
                    "num_saved_steps",
                    "num_aux_vis",
                    "missing_key_steps",
                ],
            )
            writer.writeheader()

    def _append_summary(self, row: Dict):
        with self.summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task",
                    "episode_seed",
                    "success",
                    "aborted",
                    "kept",
                    "num_saved_steps",
                    "num_aux_vis",
                    "missing_key_steps",
                ],
            )
            writer.writerow(row)

    def _infer_camera_names(self, obs: Dict) -> List[str]:
        return sorted({k[:-4] for k in obs.keys() if k.endswith("_rgb")})

    def _resolve_episode_cameras(self, obs: Dict) -> List[str]:
        if self._state["camera_names"] is not None:
            return self._state["camera_names"]

        if self.camera_names:
            cams = list(self.camera_names)
        else:
            cams = self._infer_camera_names(obs)

        self._state["camera_names"] = cams

        # Create image directories after cameras are finalized.
        episode_dir: Path = self._state["episode_dir"]
        for cam in cams:
            (episode_dir / "images" / cam).mkdir(parents=True, exist_ok=True)

        return cams

    def _required_keys_for_step(self, cameras: List[str]) -> List[str]:
        req = list(self.required_keys)
        req.append("has_affordance")
        for cam in cameras:
            req.extend(
                [
                    f"{cam}_rgb",
                    f"{cam}_contact_2d",
                    f"{cam}_grasp_2d",
                    f"{cam}_affordance_2d",
                    f"{cam}_contact_visible",
                    f"{cam}_grasp_visible",
                    f"{cam}_affordance_visible",
                ]
            )
        return req

    def _find_missing_keys(self, obs: Dict, cameras: List[str]) -> List[str]:
        required = self._required_keys_for_step(cameras)
        return [k for k in required if k not in obs]

    def _should_collect_step(self, step_id: int) -> bool:
        if step_id < 0:
            return False
        if step_id >= self.collect_window_steps:
            return False
        if (step_id % self.collect_every_n_steps) != 0:
            return False
        if len(self._state["records"]) >= self.max_frames_per_episode:
            return False
        return True

    def _save_images(self, obs: Dict, cameras: List[str], step_id: int) -> Dict[str, str]:
        episode_dir: Path = self._state["episode_dir"]
        image_paths = {}
        for cam in cameras:
            rgb = _to_uint8_rgb(obs[f"{cam}_rgb"])
            rel_path = f"images/{cam}/rgb_{step_id:04d}.png"
            abs_path = episode_dir / rel_path
            Image.fromarray(rgb).save(abs_path)
            image_paths[cam] = rel_path
        return image_paths

    def _build_sample(
        self,
        task: str,
        episode_seed: int,
        step_id: int,
        image_paths: Dict[str, str],
        cameras: List[str],
        obs: Dict,
    ) -> Dict:
        sample = {
            "task": task,
            "episode_seed": int(episode_seed),
            "step_id": int(step_id),
            "image_paths": image_paths,
            "has_affordance": _to_bool(obs.get("has_affordance", False)),
        }

        for cam in cameras:
            sample[f"{cam}_contact_2d"] = _to_xy(obs.get(f"{cam}_contact_2d", [-1.0, -1.0]))
            sample[f"{cam}_grasp_2d"] = _to_xy(obs.get(f"{cam}_grasp_2d", [-1.0, -1.0]))
            sample[f"{cam}_affordance_2d"] = _to_xy(obs.get(f"{cam}_affordance_2d", [-1.0, -1.0]))
            sample[f"{cam}_contact_visible"] = _to_bool(obs.get(f"{cam}_contact_visible", False))
            sample[f"{cam}_grasp_visible"] = _to_bool(obs.get(f"{cam}_grasp_visible", False))
            sample[f"{cam}_affordance_visible"] = _to_bool(obs.get(f"{cam}_affordance_visible", False))

        return sample

    def _build_vis_obs(self, obs: Dict, cameras: List[str]) -> Optional[Dict]:
        if not self.aux_vis_enabled:
            return None

        if not cameras:
            return None

        cam = self.aux_vis_sample_camera
        if cam not in cameras:
            cam = cameras[0]

        if f"{cam}_rgb" not in obs:
            return None

        vis_obs = {
            f"{cam}_rgb": np.array(obs[f"{cam}_rgb"], copy=True),
            f"{cam}_contact_2d": np.array(obs.get(f"{cam}_contact_2d", [-1.0, -1.0]), dtype=np.float32),
            f"{cam}_grasp_2d": np.array(obs.get(f"{cam}_grasp_2d", [-1.0, -1.0]), dtype=np.float32),
            f"{cam}_affordance_2d": np.array(obs.get(f"{cam}_affordance_2d", [-1.0, -1.0]), dtype=np.float32),
            f"{cam}_contact_visible": _to_bool(obs.get(f"{cam}_contact_visible", False)),
            f"{cam}_grasp_visible": _to_bool(obs.get(f"{cam}_grasp_visible", False)),
            f"{cam}_affordance_visible": _to_bool(obs.get(f"{cam}_affordance_visible", False)),
            "has_affordance": _to_bool(obs.get("has_affordance", False)),
        }

        # Gripper projection overlay depends on these optional fields.
        if "right_gripper_pose" in obs:
            vis_obs["right_gripper_pose"] = np.array(obs["right_gripper_pose"], copy=True)
        if "left_gripper_pose" in obs:
            vis_obs["left_gripper_pose"] = np.array(obs["left_gripper_pose"], copy=True)
        if f"{cam}_camera_extrinsics" in obs:
            vis_obs[f"{cam}_camera_extrinsics"] = np.array(obs[f"{cam}_camera_extrinsics"], copy=True)
        if f"{cam}_camera_intrinsics" in obs:
            vis_obs[f"{cam}_camera_intrinsics"] = np.array(obs[f"{cam}_camera_intrinsics"], copy=True)

        return vis_obs

    def start_episode(self, task: str, episode_seed: int):
        # Safety fallback in case previous episode was not closed.
        if self._state is not None:
            self.end_episode(success=False, aborted=True)

        episode_dir = self.round_root / task / f"episode_{int(episode_seed):06d}"

        # Re-run on same round/seed should overwrite stale content.
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)

        self._state = {
            "task": task,
            "episode_seed": int(episode_seed),
            "episode_dir": episode_dir,
            "camera_names": list(self.camera_names) if self.camera_names else None,
            "records": [],
            "missing_key_steps": 0,
        }

        # Pre-create image dirs if camera names are pre-configured.
        if self._state["camera_names"] is not None:
            for cam in self._state["camera_names"]:
                (episode_dir / "images" / cam).mkdir(parents=True, exist_ok=True)

    def maybe_add_step(self, task, episode_seed, step_id, obs, reward, terminal, pred_info=None):
        del reward, terminal  # Not used at step-time; kept for interface compatibility.

        if self._state is None:
            return

        if int(episode_seed) != self._state["episode_seed"] or str(task) != self._state["task"]:
            # Defensive check: ignore mismatched episode data.
            return

        if not self._should_collect_step(int(step_id)):
            return

        cameras = self._resolve_episode_cameras(obs)
        if not cameras:
            self._state["missing_key_steps"] += 1
            return

        missing = self._find_missing_keys(obs, cameras)
        if missing and self.strict_required_keys:
            self._state["missing_key_steps"] += 1
            return

        image_paths = self._save_images(obs, cameras, int(step_id))
        sample = self._build_sample(task, int(episode_seed), int(step_id), image_paths, cameras, obs)
        vis_obs = self._build_vis_obs(obs, cameras)

        self._state["records"].append(
            {
                "step_id": int(step_id),
                "sample": sample,
                "pred_info": pred_info,
                "vis_obs": vis_obs,
            }
        )

    def _should_keep_episode(self, success: bool, aborted: bool) -> bool:
        # Aborted episodes are dropped to avoid polluted partial data.
        if aborted:
            return False
        if self.collect_outcome == "both":
            return True
        if self.collect_outcome == "success_only":
            return bool(success)
        if self.collect_outcome == "fail_only":
            return not bool(success)
        return True

    def _resolve_aux_vis_dir(self, task: str, episode_seed: int, episode_dir: Path) -> Path:
        if self.aux_vis_save_root:
            return Path(self.aux_vis_save_root) / f"round_{self.round_id:03d}" / task / f"episode_{int(episode_seed):06d}"
        return episode_dir / "aux_vis"

    def end_episode(self, success: bool, aborted: bool = False):
        if self._state is None:
            return

        task = self._state["task"]
        episode_seed = self._state["episode_seed"]
        episode_dir: Path = self._state["episode_dir"]
        records: List[Dict] = self._state["records"]

        keep = self._should_keep_episode(success=success, aborted=aborted)
        aux_vis_count = 0

        if keep and records:
            # 1) Dump training samples.
            samples_path = episode_dir / "samples.jsonl"
            with samples_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec["sample"], ensure_ascii=False) + "\n")

            # 2) Uniformly sample aux visualizations from saved records.
            if self.aux_vis_enabled and self.aux_vis_num_samples > 0:
                vis_indices = _uniform_pick_indices(len(records), self.aux_vis_num_samples)
                vis_dir = self._resolve_aux_vis_dir(task, episode_seed, episode_dir)
                vis_dir.mkdir(parents=True, exist_ok=True)

                manifest_path = episode_dir / "aux_vis_manifest.jsonl"
                with manifest_path.open("w", encoding="utf-8") as mf:
                    for vis_i, rec_idx in enumerate(vis_indices):
                        rec = records[rec_idx]
                        pred_info = rec.get("pred_info")
                        vis_obs = rec.get("vis_obs")
                        if pred_info is None or vis_obs is None:
                            continue

                        vis_name = f"ep{int(episode_seed):06d}_u{vis_i:03d}.png"
                        vis_path = vis_dir / vis_name
                        render_aux_eval_like(vis_obs, pred_info, self.aux_vis_render_cfg, str(vis_path))

                        manifest = {
                            "task": task,
                            "episode_seed": int(episode_seed),
                            "step_id": int(rec["step_id"]),
                            "sample_index_in_episode": int(vis_i),
                            "sample_camera": self.aux_vis_sample_camera,
                            "vis_path": str(vis_path),
                        }
                        mf.write(json.dumps(manifest, ensure_ascii=False) + "\n")
                        aux_vis_count += 1

        else:
            # Drop entire directory for episodes not kept.
            if episode_dir.exists():
                shutil.rmtree(episode_dir)

        self._append_summary(
            {
                "task": task,
                "episode_seed": int(episode_seed),
                "success": int(bool(success)),
                "aborted": int(bool(aborted)),
                "kept": int(bool(keep and len(records) > 0)),
                "num_saved_steps": int(len(records) if keep else 0),
                "num_aux_vis": int(aux_vis_count),
                "missing_key_steps": int(self._state["missing_key_steps"]),
            }
        )

        self._state = None

    def close(self):
        if self._state is not None:
            self.end_episode(success=False, aborted=True)
