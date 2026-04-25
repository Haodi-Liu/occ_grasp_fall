import argparse
import math
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from agents.act_bc_keypoint_strategy.detr.models.backbone import build_backbone
from agents.act_bc_keypoint_strategy.keypoint_predictor import KeypointPosePredictor
from agents.act_bc_keypoint_strategy.strategy_phase_predictor import StrategyPhasePredictor
from datasets.predictor_dataset import PredictorDataset
from helpers.loss_utils import compute_2d_loss
from helpers.qpos_stats import compute_qpos_stats
from helpers.aux_eval_visualizer import render_aux_eval_like, build_pred_info_from_outputs
from yarr.utils.log_writer import LogWriter


class PredictorBundle(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.camera_names = list(args.camera_names)
        self.hidden_dim = int(args.hidden_dim)
        self.img_size = int(args.img_size)

        backbones = [build_backbone(args) for _ in self.camera_names]
        self.pred_backbones = nn.ModuleList(backbones)
        self.pred_input_proj = nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1)
        self.pred_encoder_proprio_proj = nn.Linear(args.input_dim, self.hidden_dim)

        self.temporal_fuse_alpha = float(getattr(args, "temporal_fuse_alpha", 0.7))
        self.temporal_qpos_proj = nn.Linear(args.input_dim, 1)
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
        # Strategy predictor dropout is configurable for stronger regularization
        sp_cfg = getattr(args, "strategy_predictor", {})
        self.strategy_phase_predictor = StrategyPhasePredictor(
            hidden_dim=self.hidden_dim,
            num_strategies=int(args.num_strategies),
            num_phases=int(args.num_phases),
            dropout=sp_cfg.get("dropout", 0.1),
        )

    def _freeze_modules(self, modules):
        for module in modules:
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def freeze_backbone(self):
        self._freeze_modules([
            self.pred_backbones,
            self.pred_input_proj,
            self.pred_encoder_proprio_proj,
            self.temporal_qpos_proj,
            self.temporal_img_proj,
        ])

    def freeze_keypoint(self):
        self._freeze_modules([self.keypoint_pose_predictor])

    def freeze_strategy(self):
        self._freeze_modules([self.strategy_phase_predictor])

    def _temporal_softmax_pool(self, seq, proj):
        weights = proj(seq).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return (seq * weights.unsqueeze(-1)).sum(dim=1)

    def _encode_visual_features(self, qpos, image):
        if qpos.dim() == 3:
            qpos_last = qpos[:, -1]
            qpos_pool = self._temporal_softmax_pool(qpos, self.temporal_qpos_proj)
            qpos = self.temporal_fuse_alpha * qpos_last + (1.0 - self.temporal_fuse_alpha) * qpos_pool

        proprio_embed = self.pred_encoder_proprio_proj(qpos)

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
        return proprio_embed, img_embed, img_pooled, img_tokens

    def forward(self, qpos, image):
        proprio_embed, _, img_pooled, img_tokens = self._encode_visual_features(qpos, image)
        strategy_logits, phase_logits = self.strategy_phase_predictor(img_pooled, proprio_embed)
        _, keypoint_2d_pred, _, aff_visible_logits = self.keypoint_pose_predictor(
            visual_feat=img_pooled,
            proprio_feat=proprio_embed,
            img_tokens=img_tokens,
        )
        return {
            "strategy_logits": strategy_logits,
            "phase_logits": phase_logits,
            "keypoint_2d_pred": keypoint_2d_pred,
            "aff_visible_logits": aff_visible_logits,
        }


def _load_ckpt(model: PredictorBundle, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "pred_backbone" in ckpt:
        model.pred_backbones.load_state_dict(ckpt["pred_backbone"], strict=False)
    if "pred_input_proj" in ckpt:
        model.pred_input_proj.load_state_dict(ckpt["pred_input_proj"], strict=False)
    if "pred_encoder_proprio_proj" in ckpt:
        model.pred_encoder_proprio_proj.load_state_dict(ckpt["pred_encoder_proprio_proj"], strict=False)
    if "pred_temporal_qpos_proj" in ckpt:
        model.temporal_qpos_proj.load_state_dict(ckpt["pred_temporal_qpos_proj"], strict=False)
    if "pred_temporal_img_proj" in ckpt:
        model.temporal_img_proj.load_state_dict(ckpt["pred_temporal_img_proj"], strict=False)
    if "keypoint_pose_predictor" in ckpt:
        model.keypoint_pose_predictor.load_state_dict(ckpt["keypoint_pose_predictor"], strict=False)
    if "strategy_phase_predictor" in ckpt:
        model.strategy_phase_predictor.load_state_dict(ckpt["strategy_phase_predictor"], strict=False)


def _save_ckpt(model: PredictorBundle, save_dir: str, step: int,
               save_prefix: str = "predictor", save_modules: dict = None):
    os.makedirs(save_dir, exist_ok=True)
    save_modules = save_modules or {}
    ckpt = {}
    if save_modules.get("backbone", True):
        ckpt.update({
            "pred_backbone": model.pred_backbones.state_dict(),
            "pred_input_proj": model.pred_input_proj.state_dict(),
            "pred_encoder_proprio_proj": model.pred_encoder_proprio_proj.state_dict(),
            "pred_temporal_qpos_proj": model.temporal_qpos_proj.state_dict(),
            "pred_temporal_img_proj": model.temporal_img_proj.state_dict(),
        })
    if save_modules.get("keypoint", True):
        ckpt["keypoint_pose_predictor"] = model.keypoint_pose_predictor.state_dict()
    if save_modules.get("strategy", True):
        ckpt["strategy_phase_predictor"] = model.strategy_phase_predictor.state_dict()
    path = os.path.join(save_dir, f"{save_prefix}_{step:06d}.pt")
    torch.save(ckpt, path)


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


def _init_meters():
    return {
        "proj_2d": 0.0,
        "aff_vis": 0.0,
        "strategy_ce": 0.0,
        "phase_ce": 0.0,
        "kp_total": 0.0,
        "total": 0.0,
        "count": 0,
    }


def _update_meters(meters, proj_2d, aff_vis, strat_ce, phase_ce, kp_total, total):
    meters["proj_2d"] += float(proj_2d)
    meters["aff_vis"] += float(aff_vis)
    meters["strategy_ce"] += float(strat_ce)
    meters["phase_ce"] += float(phase_ce)
    meters["kp_total"] += float(kp_total)
    meters["total"] += float(total)
    meters["count"] += 1


def _format_meters(meters):
    count = max(1, meters["count"])
    keys = ["proj_2d", "aff_vis", "strategy_ce", "phase_ce", "kp_total", "total"]
    return " ".join([f"{k}={meters[k] / count:.4f}" for k in keys])

def _avg_meters(meters):
    count = max(1, meters["count"])
    keys = ["proj_2d", "aff_vis", "strategy_ce", "phase_ce", "kp_total", "total"]
    return {k: meters[k] / count for k in keys}


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_mode = getattr(cfg, "train_mode", None)
    if train_mode not in ("keypoint", "strategy"):
        raise ValueError(
            "predictor pretrain only supports train_mode: keypoint|strategy. "
            "Joint training is disabled by design."
        )

    init_ckpt_path = getattr(cfg, "init_ckpt_path", "")
    if train_mode == "strategy" and not init_ckpt_path:
        raise ValueError("train_mode=strategy requires init_ckpt_path (keypoint ckpt).")

    stats = compute_qpos_stats(cfg.data_root, cfg.train_tasks, cfg.train_stats_path)

    train_ds = PredictorDataset(
        index_path=cfg.train_index_path,
        data_root=cfg.data_root,
        camera_names=cfg.camera_names,
        img_size=cfg.img_size,
        stats=stats,
        num_strategies=cfg.num_strategies,
        num_phases=cfg.num_phases,
    )
    eval_ds = PredictorDataset(
        index_path=cfg.eval_index_path,
        data_root=cfg.data_root,
        camera_names=cfg.camera_names,
        img_size=cfg.img_size,
        stats=stats,
        num_strategies=cfg.num_strategies,
        num_phases=cfg.num_phases,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PredictorBundle(cfg).to(device)
    if init_ckpt_path:
        _load_ckpt(model, init_ckpt_path)

    def _apply_freeze():
        if train_mode == "keypoint":
            model.freeze_strategy()
        elif train_mode == "strategy":
            model.freeze_backbone()
            model.freeze_keypoint()

        if bool(getattr(cfg, "freeze_backbone", False)):
            model.freeze_backbone()
        if bool(getattr(cfg, "freeze_keypoint", False)):
            model.freeze_keypoint()
        if bool(getattr(cfg, "freeze_strategy", False)):
            model.freeze_strategy()

    _apply_freeze()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after freezing.")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
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
        _apply_freeze()
        train_meters = _init_meters()
        for batch in train_loader:
            qpos = batch["qpos"].to(device)
            image = batch["image"].to(device)
            has_aff = batch["has_affordance"].to(device)
            kp_gt = {k: v.to(device) for k, v in batch["keypoint_2d_gt"].items()}
            kp_vis = {k: v.to(device) for k, v in batch["keypoint_2d_visible"].items()}
            strategy = batch["strategy_type"].to(device)
            phase = batch["phase_type"].to(device)

            out = model(qpos, image)
            if train_mode == "keypoint":
                kp_2d_loss = compute_2d_loss(out["keypoint_2d_pred"], kp_gt, kp_vis, has_aff, cfg.img_size)
                aff_loss = F.binary_cross_entropy_with_logits(
                    out["aff_visible_logits"].squeeze(-1), has_aff.float()
                )
                strat_loss = torch.tensor(0.0, device=device)
                phase_loss = torch.tensor(0.0, device=device)
            else:
                kp_2d_loss = torch.tensor(0.0, device=device)
                aff_loss = torch.tensor(0.0, device=device)
                strat_loss = F.cross_entropy(out["strategy_logits"], strategy)
                phase_loss = F.cross_entropy(out["phase_logits"], phase)

            kp_total = kp_2d_loss * cfg.proj_2d_weight + aff_loss * cfg.aff_vis_weight
            total = (kp_total +
                     strat_loss * cfg.strategy_weight +
                     phase_loss * cfg.phase_weight)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            _update_meters(train_meters, kp_2d_loss, aff_loss, strat_loss, phase_loss, kp_total, total)

        model.eval()
        eval_meters = _init_meters()
        aux_cfg = getattr(cfg, "aux_eval", None)
        aux_count_by_task = {}
        aux_phase_counts_by_task = {}
        aux_save_root = None
        if aux_cfg is not None:
            aux_save_root = getattr(aux_cfg, "save_path", None)
            if not aux_save_root:
                aux_save_root = os.path.join(log_dir, "aux_eval_samples")
        num_phases = int(getattr(cfg, "num_phases", 4))
        with torch.no_grad():
            for batch_i, batch in enumerate(eval_loader):
                qpos = batch["qpos"].to(device)
                image = batch["image"].to(device)
                has_aff = batch["has_affordance"].to(device)
                kp_gt = {k: v.to(device) for k, v in batch["keypoint_2d_gt"].items()}
                kp_vis = {k: v.to(device) for k, v in batch["keypoint_2d_visible"].items()}
                strategy = batch["strategy_type"].to(device)
                phase = batch["phase_type"].to(device)

                out = model(qpos, image)
                if train_mode == "keypoint":
                    kp_2d_loss = compute_2d_loss(out["keypoint_2d_pred"], kp_gt, kp_vis, has_aff, cfg.img_size)
                    aff_loss = F.binary_cross_entropy_with_logits(
                        out["aff_visible_logits"].squeeze(-1), has_aff.float()
                    )
                    strat_loss = torch.tensor(0.0, device=device)
                    phase_loss = torch.tensor(0.0, device=device)
                else:
                    kp_2d_loss = torch.tensor(0.0, device=device)
                    aff_loss = torch.tensor(0.0, device=device)
                    strat_loss = F.cross_entropy(out["strategy_logits"], strategy)
                    phase_loss = F.cross_entropy(out["phase_logits"], phase)

                kp_total = kp_2d_loss * cfg.proj_2d_weight + aff_loss * cfg.aff_vis_weight
                total = (kp_total +
                         strat_loss * cfg.strategy_weight +
                         phase_loss * cfg.phase_weight)

                _update_meters(eval_meters, kp_2d_loss, aff_loss, strat_loss, phase_loss, kp_total, total)

                if aux_cfg is not None and bool(getattr(aux_cfg, "enabled", False)):
                    if (batch_i % int(getattr(aux_cfg, "sample_every_n_steps", 10))) != 0:
                        continue
                    max_s = int(getattr(aux_cfg, "max_samples_per_epoch", 5))
                    per_phase_limit = max(1, int(math.ceil(max_s / float(num_phases))))
                    tasks = batch.get("task")
                    if tasks is None:
                        tasks = ["unknown"] * batch["image"].shape[0]
                    phases = batch.get("phase_type")
                    if phases is not None:
                        phases = phases.detach().cpu().tolist()
                    for s in range(batch["image"].shape[0]):
                        task_name = str(tasks[s])
                        total_count = aux_count_by_task.get(task_name, 0)
                        if total_count >= max_s:
                            continue
                        phase_id = None
                        if phases is not None:
                            phase_id = int(phases[s])
                            if phase_id < 0 or phase_id >= num_phases:
                                phase_id = None
                        if phase_id is not None:
                            phase_counts = aux_phase_counts_by_task.get(task_name)
                            if phase_counts is None:
                                phase_counts = [0] * num_phases
                                aux_phase_counts_by_task[task_name] = phase_counts
                            if phase_counts[phase_id] >= per_phase_limit:
                                continue
                        pred_info = build_pred_info_from_outputs(out, cfg.camera_names, cfg.img_size, s)
                        obs_dict = _build_obs_dict_for_vis(batch, cfg.camera_names, s)
                        task_dir = os.path.join(aux_save_root, task_name)
                        os.makedirs(task_dir, exist_ok=True)
                        out_file = os.path.join(
                            task_dir,
                            f"ep{epoch:03d}_step{batch_i:04d}_s{s:02d}.png",
                        )
                        render_aux_eval_like(obs_dict, pred_info, aux_cfg, out_file)
                        aux_count_by_task[task_name] = total_count + 1
                        if phase_id is not None:
                            aux_phase_counts_by_task[task_name][phase_id] += 1

        print(f"[epoch {epoch}] train {_format_meters(train_meters)}")
        print(f"[epoch {epoch}] eval  {_format_meters(eval_meters)}")
        train_avg = _avg_meters(train_meters)
        eval_avg = _avg_meters(eval_meters)
        for k, v in train_avg.items():
            writer.add_scalar(epoch, f"pretrain/train/{k}", v)
        for k, v in eval_avg.items():
            writer.add_scalar(epoch, f"pretrain/eval/{k}", v)
        writer.end_iteration()

        save_prefix = getattr(cfg, "save_prefix", "predictor")
        save_modules = getattr(cfg, "save_modules", {})
        _save_ckpt(model, cfg.save_dir, epoch, save_prefix=save_prefix, save_modules=save_modules)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
