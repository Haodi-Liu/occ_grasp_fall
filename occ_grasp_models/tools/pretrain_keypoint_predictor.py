import argparse
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from agents.act_bc_keypoint.detr.models.backbone import build_backbone
from agents.act_bc_keypoint.keypoint_predictor import KeypointPosePredictor
from datasets.predictor_dataset import PredictorDataset
from datasets.predictor_online_dataset import PredictorOnlineDataset
from helpers.loss_utils import compute_2d_loss
from helpers.qpos_stats import compute_qpos_stats
from helpers.aux_eval_visualizer import render_aux_eval_like, build_pred_info_from_outputs
from yarr.utils.log_writer import LogWriter


class KeypointPredictorBundle(nn.Module):
    """
    Predictor bundle with visual modules only:
    - pred_backbones + pred_input_proj
    - keypoint_pose_predictor (visual-only)

    Note: pred_encoder_proprio_proj / pred_temporal_qpos_proj are omitted.
    """

    def __init__(self, args):
        super().__init__()
        self.camera_names = list(args.camera_names)
        self.hidden_dim = int(args.hidden_dim)
        self.img_size = int(args.img_size)

        backbones = [build_backbone(args) for _ in self.camera_names]
        self.pred_backbones = nn.ModuleList(backbones)
        self.pred_input_proj = nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1)

        # Temporal fusion (kept aligned with the main model)
        self.temporal_fuse_alpha = float(getattr(args, "temporal_fuse_alpha", 0.7))
        self.temporal_img_proj = nn.Linear(self.hidden_dim, 1)

        kp_cfg = getattr(args, "keypoint_predictor", {})
        self.keypoint_pose_predictor = KeypointPosePredictor(
            hidden_dim=self.hidden_dim,
            num_cameras=len(self.camera_names),
            img_size=self.img_size,
            num_heads=kp_cfg.get("num_heads", 8),
            num_layers=kp_cfg.get("num_layers", 2),
            dropout=kp_cfg.get("dropout", 0.1),
        )

    def _encode_visual_features(self, image):
        """Encode visual features only for keypoint prediction."""
        all_img_features = []
        all_img_pooled = []

        for cam_id in range(len(self.camera_names)):
            cam_img = image[:, cam_id]
            if cam_img.dim() == 5:
                b, t, c, h, w = cam_img.shape
                cam_img = cam_img.reshape(b * t, c, h, w)
                features, _ = self.pred_backbones[cam_id](cam_img)
                features = features[0]
                features = self.pred_input_proj(features)
                _, c2, h2, w2 = features.shape
                features = features.view(b, t, c2, h2, w2)

                pooled_t = features.mean(dim=[3, 4])
                weights = self.temporal_img_proj(pooled_t).squeeze(-1)
                weights = torch.softmax(weights, dim=1)
                fused = (features * weights.view(b, t, 1, 1, 1)).sum(dim=1)

                last = features[:, -1]
                features = self.temporal_fuse_alpha * last + (1.0 - self.temporal_fuse_alpha) * fused
            else:
                features, _ = self.pred_backbones[cam_id](cam_img)
                features = features[0]
                features = self.pred_input_proj(features)

            pooled = features.mean(dim=[2, 3])
            all_img_pooled.append(pooled)

            features_flat = features.flatten(2).permute(2, 0, 1)
            all_img_features.append(features_flat)

        img_embed = torch.cat(all_img_features, dim=0)
        img_pooled = torch.stack(all_img_pooled, dim=0).mean(dim=0)
        img_tokens = img_embed.permute(1, 0, 2)
        return img_pooled, img_tokens

    def forward(self, image):
        img_pooled, img_tokens = self._encode_visual_features(image)
        _, keypoint_2d_pred, _, aff_visible_logits = self.keypoint_pose_predictor(
            visual_feat=img_pooled,
            img_tokens=img_tokens,
        )
        return {
            "keypoint_2d_pred": keypoint_2d_pred,
            "aff_visible_logits": aff_visible_logits,
        }


class _DomainDataset(Dataset):
    """Attach domain_id for per-domain meters in mixed sampling."""

    def __init__(self, base_dataset: Dataset, domain_id: int):
        self._base = base_dataset
        self._domain_id = int(domain_id)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        sample = dict(self._base[idx])
        sample["domain_id"] = torch.tensor(self._domain_id, dtype=torch.long)
        return sample


def _save_ckpt(model: KeypointPredictorBundle, save_dir: str, step: int, save_prefix: str = "predictor_kp"):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "pred_backbone": model.pred_backbones.state_dict(),
        "pred_input_proj": model.pred_input_proj.state_dict(),
        "pred_temporal_img_proj": model.temporal_img_proj.state_dict(),
        "keypoint_pose_predictor": model.keypoint_pose_predictor.state_dict(),
    }
    path = os.path.join(save_dir, f"{save_prefix}_{step:06d}.pt")
    torch.save(ckpt, path)


def _load_init_ckpt_if_needed(model: KeypointPredictorBundle, cfg):
    path = str(getattr(cfg, "init_ckpt_path", "")).strip()
    if not path:
        return

    ckpt = torch.load(path, map_location="cpu")
    if "pred_backbone" in ckpt:
        model.pred_backbones.load_state_dict(ckpt["pred_backbone"], strict=False)
    if "pred_input_proj" in ckpt:
        model.pred_input_proj.load_state_dict(ckpt["pred_input_proj"], strict=False)
    if "pred_temporal_img_proj" in ckpt:
        model.temporal_img_proj.load_state_dict(ckpt["pred_temporal_img_proj"], strict=False)
    if "keypoint_pose_predictor" in ckpt:
        model.keypoint_pose_predictor.load_state_dict(ckpt["keypoint_pose_predictor"], strict=False)

    print(f"[warm-start] loaded init checkpoint: {path}")


def _build_obs_dict_for_vis(batch: Dict, cam_names, sample_idx: int):
    obs_dict = {}
    images = batch["image"][sample_idx]
    for cam_idx, cam in enumerate(cam_names):
        obs_dict[f"{cam}_rgb"] = images[cam_idx].detach().cpu().numpy()
        kp_gt = batch["keypoint_2d_gt"][cam][sample_idx].detach().cpu().numpy()
        kp_vis = batch["keypoint_2d_visible"][cam][sample_idx].detach().cpu().numpy()
        obs_dict[f"{cam}_contact_2d"] = kp_gt[0]
        obs_dict[f"{cam}_grasp_2d"] = kp_gt[1]
        obs_dict[f"{cam}_affordance_2d"] = kp_gt[2]
        obs_dict[f"{cam}_contact_visible"] = bool(kp_vis[0])
        obs_dict[f"{cam}_grasp_visible"] = bool(kp_vis[1])
        obs_dict[f"{cam}_affordance_visible"] = bool(kp_vis[2])
    obs_dict["has_affordance"] = bool(batch["has_affordance"][sample_idx].item())
    return obs_dict


def _init_meters() -> Dict[str, float]:
    return {
        "proj_2d": 0.0,
        "aff_vis": 0.0,
        "kp_total": 0.0,
        "total": 0.0,
        "count": 0,
    }


def _accumulate_meters(
    meters: Dict[str, float],
    proj_2d: torch.Tensor,
    aff_vis: torch.Tensor,
    kp_total: torch.Tensor,
    total: torch.Tensor,
    n: int,
):
    if n <= 0:
        return

    meters["proj_2d"] += float(proj_2d.detach().cpu().item()) * n
    meters["aff_vis"] += float(aff_vis.detach().cpu().item()) * n
    meters["kp_total"] += float(kp_total.detach().cpu().item()) * n
    meters["total"] += float(total.detach().cpu().item()) * n
    meters["count"] += int(n)


def _avg_meters(meters: Dict[str, float]) -> Dict[str, float]:
    count = max(1, int(meters["count"]))
    return {
        "proj_2d": meters["proj_2d"] / count,
        "aff_vis": meters["aff_vis"] / count,
        "kp_total": meters["kp_total"] / count,
        "total": meters["total"] / count,
    }


def _format_meters(meters: Dict[str, float]) -> str:
    avg = _avg_meters(meters)
    keys = ["proj_2d", "aff_vis", "kp_total", "total"]
    return " ".join([f"{k}={avg[k]:.4f}" for k in keys])


def _compute_losses(
    out: Dict,
    kp_gt: Dict[str, torch.Tensor],
    kp_vis: Dict[str, torch.Tensor],
    has_aff: torch.Tensor,
    cfg,
):
    kp_2d_loss = compute_2d_loss(out["keypoint_2d_pred"], kp_gt, kp_vis, has_aff, cfg.img_size)
    aff_loss = F.binary_cross_entropy_with_logits(
        out["aff_visible_logits"].squeeze(-1),
        has_aff.float(),
    )
    kp_total = kp_2d_loss * cfg.proj_2d_weight + aff_loss * cfg.aff_vis_weight
    total = kp_total
    return kp_2d_loss, aff_loss, kp_total, total


def _masked_out(out: Dict, mask: torch.Tensor) -> Dict:
    return {
        "keypoint_2d_pred": {k: v[mask] for k, v in out["keypoint_2d_pred"].items()},
        "aff_visible_logits": out["aff_visible_logits"][mask],
    }


def _masked_gt(gt: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[mask] for k, v in gt.items()}


def _update_domain_meters(
    domain_meters: Dict[str, float],
    out: Dict,
    kp_gt: Dict[str, torch.Tensor],
    kp_vis: Dict[str, torch.Tensor],
    has_aff: torch.Tensor,
    domain_ids: torch.Tensor,
    domain_id: int,
    cfg,
):
    mask = (domain_ids == int(domain_id))
    n = int(mask.sum().item())
    if n <= 0:
        return

    sub_out = _masked_out(out, mask)
    sub_gt = _masked_gt(kp_gt, mask)
    sub_vis = _masked_gt(kp_vis, mask)
    sub_has_aff = has_aff[mask]

    sub_kp_2d_loss, sub_aff_loss, sub_kp_total, sub_total = _compute_losses(
        sub_out,
        sub_gt,
        sub_vis,
        sub_has_aff,
        cfg,
    )
    _accumulate_meters(domain_meters, sub_kp_2d_loss, sub_aff_loss, sub_kp_total, sub_total, n)


def _build_source_weights(
    n_offline: int,
    n_online: int,
    offline_w: float,
    online_w: float,
) -> torch.Tensor:
    if n_offline <= 0 or n_online <= 0:
        raise ValueError("_build_source_weights expects both n_offline and n_online > 0")

    if offline_w <= 0.0 and online_w <= 0.0:
        raise ValueError("offline_sample_weight and online_sample_weight cannot both be <= 0")

    offline_w = max(0.0, float(offline_w))
    online_w = max(0.0, float(online_w))
    total = offline_w + online_w
    if total <= 0.0:
        offline_w, online_w = 0.5, 0.5
    else:
        offline_w /= total
        online_w /= total

    off_each = offline_w / float(n_offline)
    on_each = online_w / float(n_online)
    weights = [off_each] * n_offline + [on_each] * n_online
    return torch.tensor(weights, dtype=torch.double)


def _resolve_samples_per_epoch(cfg, default_value: int) -> int:
    val = int(getattr(cfg, "samples_per_epoch", 0))
    return val if val > 0 else int(default_value)


def _build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    sampler=None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _run_eval_loader(
    model,
    loader,
    device,
    cfg,
    domain_tag: str,
    aux_cfg=None,
    epoch: int = 0,
    log_dir: str = "",
) -> Dict[str, float]:
    meters = _init_meters()

    aux_count_by_task = {}
    aux_save_root = None
    if aux_cfg is not None:
        aux_save_root = getattr(aux_cfg, "save_path", None) or os.path.join(log_dir, "aux_eval_samples")

    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            image = batch["image"].to(device)
            has_aff = batch["has_affordance"].to(device)
            kp_gt = {k: v.to(device) for k, v in batch["keypoint_2d_gt"].items()}
            kp_vis = {k: v.to(device) for k, v in batch["keypoint_2d_visible"].items()}

            out = model(image)
            kp_2d_loss, aff_loss, kp_total, total = _compute_losses(out, kp_gt, kp_vis, has_aff, cfg)
            _accumulate_meters(meters, kp_2d_loss, aff_loss, kp_total, total, int(image.shape[0]))

            # Keep old aux visualization only for offline eval.
            if aux_cfg is not None and domain_tag == "offline":
                if not bool(getattr(aux_cfg, "enabled", False)):
                    continue
                if (batch_i % int(getattr(aux_cfg, "sample_every_n_steps", 10))) != 0:
                    continue

                max_s = int(getattr(aux_cfg, "max_samples_per_epoch", 5))
                tasks = batch.get("task")
                if tasks is None:
                    tasks = ["unknown"] * batch["image"].shape[0]

                for s in range(batch["image"].shape[0]):
                    task_name = str(tasks[s])
                    total_count = aux_count_by_task.get(task_name, 0)
                    if total_count >= max_s:
                        continue

                    pred_info = build_pred_info_from_outputs(out, cfg.camera_names, cfg.img_size, s)
                    obs_dict = _build_obs_dict_for_vis(batch, cfg.camera_names, s)

                    task_dir = os.path.join(aux_save_root, domain_tag, task_name)
                    os.makedirs(task_dir, exist_ok=True)
                    out_file = os.path.join(task_dir, f"ep{epoch:03d}_step{batch_i:04d}_s{s:02d}.png")
                    render_aux_eval_like(obs_dict, pred_info, aux_cfg, out_file)
                    aux_count_by_task[task_name] = total_count + 1

    return meters


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = compute_qpos_stats(cfg.data_root, cfg.train_tasks, cfg.train_stats_path)

    offline_train_base = PredictorDataset(
        index_path=cfg.train_index_path,
        data_root=cfg.data_root,
        camera_names=cfg.camera_names,
        img_size=cfg.img_size,
        stats=stats,
        num_strategies=cfg.num_strategies,
        num_phases=cfg.num_phases,
    )
    offline_eval_base = PredictorDataset(
        index_path=cfg.eval_index_path,
        data_root=cfg.data_root,
        camera_names=cfg.camera_names,
        img_size=cfg.img_size,
        stats=stats,
        num_strategies=cfg.num_strategies,
        num_phases=cfg.num_phases,
    )

    offline_train_ds = _DomainDataset(offline_train_base, domain_id=0)
    offline_eval_ds = _DomainDataset(offline_eval_base, domain_id=0)

    online_train_index = str(getattr(cfg, "online_train_index_path", "")).strip()
    online_eval_index = str(getattr(cfg, "online_eval_index_path", "")).strip()

    online_train_ds: Optional[_DomainDataset] = None
    online_eval_ds: Optional[_DomainDataset] = None

    if online_train_index:
        online_train_base = PredictorOnlineDataset(
            index_path=online_train_index,
            camera_names=cfg.camera_names,
            img_size=cfg.img_size,
            source_domain="online",
        )
        online_train_ds = _DomainDataset(online_train_base, domain_id=1)

    if online_eval_index:
        online_eval_base = PredictorOnlineDataset(
            index_path=online_eval_index,
            camera_names=cfg.camera_names,
            img_size=cfg.img_size,
            source_domain="online",
        )
        online_eval_ds = _DomainDataset(online_eval_base, domain_id=1)

    num_workers = int(getattr(cfg, "num_workers", 4))
    pin_memory = bool(getattr(cfg, "pin_memory", True))

    if online_train_ds is None:
        train_loader = _build_loader(
            offline_train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            sampler=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        concat_train_ds = ConcatDataset([offline_train_ds, online_train_ds])
        weights = _build_source_weights(
            n_offline=len(offline_train_ds),
            n_online=len(online_train_ds),
            offline_w=float(getattr(cfg, "offline_sample_weight", 0.8)),
            online_w=float(getattr(cfg, "online_sample_weight", 0.2)),
        )
        num_samples = _resolve_samples_per_epoch(cfg, len(concat_train_ds))
        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
        train_loader = _build_loader(
            concat_train_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    offline_eval_loader = _build_loader(
        offline_eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    online_eval_loader = None
    if online_eval_ds is not None:
        online_eval_loader = _build_loader(
            online_eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    model = KeypointPredictorBundle(cfg).to(device)
    _load_init_ckpt_if_needed(model, cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_dir = getattr(cfg, "log_dir", None) or cfg.save_dir
    writer = LogWriter(
        logdir=log_dir,
        tensorboard_logging=bool(getattr(cfg, "tensorboard_logging", False)),
        csv_logging=True,
        train_csv="pretrain_train.csv",
        env_csv="pretrain_eval.csv",
    )

    for epoch in range(cfg.num_epochs):
        model.train()

        train_all_meters = _init_meters()
        train_offline_meters = _init_meters()
        train_online_meters = _init_meters()

        for batch in train_loader:
            image = batch["image"].to(device)
            has_aff = batch["has_affordance"].to(device)
            kp_gt = {k: v.to(device) for k, v in batch["keypoint_2d_gt"].items()}
            kp_vis = {k: v.to(device) for k, v in batch["keypoint_2d_visible"].items()}

            out = model(image)
            kp_2d_loss, aff_loss, kp_total, total = _compute_losses(out, kp_gt, kp_vis, has_aff, cfg)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            batch_size = int(image.shape[0])
            _accumulate_meters(train_all_meters, kp_2d_loss, aff_loss, kp_total, total, batch_size)

            domain_ids = batch.get("domain_id")
            if domain_ids is None:
                domain_ids = torch.zeros((batch_size,), dtype=torch.long, device=device)
            else:
                domain_ids = domain_ids.to(device)

            _update_domain_meters(
                train_offline_meters,
                out=out,
                kp_gt=kp_gt,
                kp_vis=kp_vis,
                has_aff=has_aff,
                domain_ids=domain_ids,
                domain_id=0,
                cfg=cfg,
            )
            _update_domain_meters(
                train_online_meters,
                out=out,
                kp_gt=kp_gt,
                kp_vis=kp_vis,
                has_aff=has_aff,
                domain_ids=domain_ids,
                domain_id=1,
                cfg=cfg,
            )

        aux_cfg = getattr(cfg, "aux_eval", None)
        eval_offline_meters = _run_eval_loader(
            model=model,
            loader=offline_eval_loader,
            device=device,
            cfg=cfg,
            domain_tag="offline",
            aux_cfg=aux_cfg,
            epoch=epoch,
            log_dir=log_dir,
        )

        eval_online_meters = _init_meters()
        if online_eval_loader is not None:
            eval_online_meters = _run_eval_loader(
                model=model,
                loader=online_eval_loader,
                device=device,
                cfg=cfg,
                domain_tag="online",
                aux_cfg=None,
                epoch=epoch,
                log_dir=log_dir,
            )

        print(f"[epoch {epoch}] train_all     {_format_meters(train_all_meters)}")
        print(f"[epoch {epoch}] train_offline {_format_meters(train_offline_meters)}")
        if train_online_meters["count"] > 0:
            print(f"[epoch {epoch}] train_online  {_format_meters(train_online_meters)}")

        print(f"[epoch {epoch}] eval_offline  {_format_meters(eval_offline_meters)}")
        if eval_online_meters["count"] > 0:
            print(f"[epoch {epoch}] eval_online   {_format_meters(eval_online_meters)}")

        # Keep compatibility with old metric namespaces.
        train_all_avg = _avg_meters(train_all_meters)
        eval_offline_avg = _avg_meters(eval_offline_meters)
        for k, v in train_all_avg.items():
            writer.add_scalar(epoch, f"pretrain/train/{k}", v)
        for k, v in eval_offline_avg.items():
            writer.add_scalar(epoch, f"pretrain/eval/{k}", v)

        # New per-domain namespaces.
        train_offline_avg = _avg_meters(train_offline_meters)
        for k, v in train_offline_avg.items():
            writer.add_scalar(epoch, f"pretrain/train_offline/{k}", v)

        if train_online_meters["count"] > 0:
            train_online_avg = _avg_meters(train_online_meters)
            for k, v in train_online_avg.items():
                writer.add_scalar(epoch, f"pretrain/train_online/{k}", v)

        for k, v in eval_offline_avg.items():
            writer.add_scalar(epoch, f"pretrain/eval_offline/{k}", v)

        if eval_online_meters["count"] > 0:
            eval_online_avg = _avg_meters(eval_online_meters)
            for k, v in eval_online_avg.items():
                writer.add_scalar(epoch, f"pretrain/eval_online/{k}", v)

        writer.end_iteration()

        save_prefix = getattr(cfg, "save_prefix", "predictor_kp")
        _save_ckpt(model, cfg.save_dir, epoch, save_prefix=save_prefix)

    writer.close()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
