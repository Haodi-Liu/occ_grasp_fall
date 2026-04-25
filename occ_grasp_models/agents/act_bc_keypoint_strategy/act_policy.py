# ACT Policy for Keypoint + Strategy Fusion Agent
#
# This policy merges ACT_BC_ENC_KEYPOINT and ACT_BC_ENC_STRATEGY:
# 1. Action L1 loss (bimanual with position/quat/gripper weighting)
# 2. KL divergence loss (conditional CVAE)
# 3. 2D keypoint projection loss (from KEYPOINT)
# 4. Affordance visibility loss (from KEYPOINT)
# 5. Strategy classification loss (from STRATEGY)
# 6. Phase classification loss (from STRATEGY)

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np

from agents.act_bc_keypoint_strategy.detr.build import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from helpers.loss_utils import compute_2d_loss


class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args.kl_weight
        self.condition_encoder = getattr(args, 'condition_encoder', True)

        # === Keypoint loss weights (from KEYPOINT) ===
        self.aff_vis_weight = getattr(args, 'aff_vis_weight', 2.0)
        self.proj_2d_weight = getattr(args, 'proj_2d_weight', 5.0)

        # === Strategy/Phase loss weights (from STRATEGY) ===
        self.strategy_weight = getattr(args, 'strategy_weight', 1.0)
        self.phase_weight = getattr(args, 'phase_weight', 1.0)
        self.num_strategies = getattr(args, 'num_strategies', 3)
        self.num_phases = getattr(args, 'num_phases', 4)
        self._lr = getattr(args, 'lr', 3e-5)
        self._weight_decay = getattr(args, 'weight_decay', 1e-6)
        self._lr_backbone = getattr(args, 'lr_backbone', 3e-6)

        # Camera names and image size
        self.camera_names = getattr(args, 'camera_names', ['front', 'wrist'])
        self.img_size = getattr(args, 'img_size', 256)

        kp_ckpt = getattr(args, "predictor_keypoint_ckpt_path", "")
        sp_ckpt = getattr(args, "predictor_strategy_ckpt_path", "")
        self._predictor_ckpt_loaded = False
        if kp_ckpt and sp_ckpt:
            self.model.load_predictor_state(kp_ckpt, strict=False)
            self.model.load_predictor_state(sp_ckpt, strict=False)
            self.model.freeze_predictor_modules()
            self.optimizer = self._build_optimizer()
            self._predictor_ckpt_loaded = True
        elif getattr(args, "predictor_ckpt_path", ""):
            raise RuntimeError("predictor_ckpt_path is deprecated. Provide keypoint/strategy ckpt paths.")
        self.predictor_loss_in_cvae = False

        print(f'KL Weight: {self.kl_weight}')
        print(f'Condition Encoder: {self.condition_encoder}')
        print(f'[KEYPOINT] 2D Projection Weight: {self.proj_2d_weight}')
        print(f'[KEYPOINT] Affordance Visibility Weight: {self.aff_vis_weight}')
        print(f'[KEYPOINT] Image Size: {self.img_size}')
        print(f'[STRATEGY] Strategy Weight: {self.strategy_weight}')
        print(f'[STRATEGY] Phase Weight: {self.phase_weight}')

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep predictor modules frozen/eval during CVAE training.
        if self.condition_encoder and getattr(self, "_predictor_ckpt_loaded", False):
            self.model.freeze_predictor_modules()
        return self

    def forward(self, qpos, image, actions=None, is_pad=None,
                has_affordance=None,
                keypoint_2d_gt=None, keypoint_2d_visible=None,
                strategy_type=None, phase_type=None,
                camera_extrinsics=None, camera_intrinsics=None,
                gripper_pose_world=None):
        """
        Forward pass with all loss components.

        Args:
            qpos: (B, input_dim) robot proprioception
            image: (B, num_cam, C, H, W) images
            actions: (B, seq, input_dim) action sequence (None during inference)
            is_pad: (B, seq) padding mask
            has_affordance: (B,) GT affordance existence (for visibility loss)
            keypoint_2d_gt: Dict[cam_name -> Tensor[B, 3, 2]] - 2D GT coords per camera
            keypoint_2d_visible: Dict[cam_name -> Tensor[B, 3]] - visibility flags per camera
            strategy_type: (B,) GT strategy labels (0-based)
            phase_type: (B,) GT phase labels (0-based)

        Returns:
            Training: loss_dict with all loss components
            Inference: predicted actions
        """
        env_state = None

        if actions is not None:  # training time
            if self.condition_encoder and not self._predictor_ckpt_loaded:
                raise RuntimeError(
                    "predictor_keypoint_ckpt_path + predictor_strategy_ckpt_path are required "
                    "for ACT_BC_KEYPOINT_STRATEGY training."
                )
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Forward pass: returns all outputs from dual injection model
            (a_hat, is_pad_hat, (mu, logvar),
             keypoint_2d_pred, affordance_visible, affordance_visible_logits,
             strategy_logits, phase_logits) = self.model(
                qpos, image, env_state, actions, is_pad,
                has_affordance=has_affordance,
                strategy_type=strategy_type, phase_type=phase_type,
                camera_extrinsics=camera_extrinsics,
                camera_intrinsics=camera_intrinsics,
                gripper_pose_world=gripper_pose_world
            )

            # Compute KL divergence
            if self.condition_encoder:
                mu_prior, logvar_prior = self.model.get_prior_params()
                if mu_prior is not None:
                    total_kld, dim_wise_kld, mean_kld = kl_divergence_gaussian(
                        mu, logvar, mu_prior, logvar_prior
                    )
                else:
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            loss_dict = dict()
            device = qpos.device

            # ====== Action L1 Loss (bimanual) ======
            right_pos_gt, right_pos_pred = actions[:, :, 0:3], a_hat[:, :, 0:3]
            right_quat_gt, right_quat_pred = actions[:, :, 3:7], a_hat[:, :, 3:7]
            right_gripper_gt, right_gripper_pred = actions[:, :, 7], a_hat[:, :, 7]

            left_pos_gt, left_pos_pred = actions[:, :, 8:11], a_hat[:, :, 8:11]
            left_quat_gt, left_quat_pred = actions[:, :, 11:15], a_hat[:, :, 11:15]
            left_gripper_gt, left_gripper_pred = actions[:, :, 15], a_hat[:, :, 15]

            pos_weight = 3.0
            quat_weight = 1.0
            gripper_weight = 3.0

            # Right arm losses
            right_pos_loss = F.l1_loss(right_pos_pred, right_pos_gt, reduction='none')
            right_quat_loss = F.l1_loss(right_quat_pred, right_quat_gt, reduction='none')
            right_gripper_loss = F.l1_loss(right_gripper_pred, right_gripper_gt, reduction='none')

            right_pos_l1 = (right_pos_loss * ~is_pad.unsqueeze(-1)).mean() * pos_weight
            right_quat_l1 = (right_quat_loss * ~is_pad.unsqueeze(-1)).mean() * quat_weight
            right_gripper_l1 = (right_gripper_loss * ~is_pad).mean() * gripper_weight
            right_l1 = right_pos_l1 + right_quat_l1 + right_gripper_l1

            # Left arm losses
            left_pos_loss = F.l1_loss(left_pos_pred, left_pos_gt, reduction='none')
            left_quat_loss = F.l1_loss(left_quat_pred, left_quat_gt, reduction='none')
            left_gripper_loss = F.l1_loss(left_gripper_pred, left_gripper_gt, reduction='none')

            left_pos_l1 = (left_pos_loss * ~is_pad.unsqueeze(-1)).mean() * pos_weight
            left_quat_l1 = (left_quat_loss * ~is_pad.unsqueeze(-1)).mean() * quat_weight
            left_gripper_l1 = (left_gripper_loss * ~is_pad).mean() * gripper_weight
            left_l1 = left_pos_l1 + left_quat_l1 + left_gripper_l1

            l1 = right_l1 + left_l1

            loss_dict['right_l1'] = right_l1
            loss_dict['left_l1'] = left_l1
            loss_dict['right_pos_l1'] = right_pos_l1
            loss_dict['right_quat_l1'] = right_quat_l1
            loss_dict['right_gripper_l1'] = right_gripper_l1
            loss_dict['left_pos_l1'] = left_pos_l1
            loss_dict['left_quat_l1'] = left_quat_l1
            loss_dict['left_gripper_l1'] = left_gripper_l1
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]

            # ====== [KEYPOINT] 2D Projection Loss ======
            if keypoint_2d_pred is not None and keypoint_2d_gt is not None:
                proj_2d_loss = self._compute_2d_loss(
                    keypoint_2d_pred, keypoint_2d_gt, keypoint_2d_visible, has_affordance
                )
            else:
                proj_2d_loss = torch.tensor(0.0, device=device)
            loss_dict['proj_2d'] = proj_2d_loss

            # ====== [KEYPOINT] Affordance Visibility Loss ======
            if has_affordance is not None and affordance_visible_logits is not None:
                vis_loss = F.binary_cross_entropy_with_logits(
                    affordance_visible_logits.squeeze(-1),
                    has_affordance.float()
                )
            else:
                vis_loss = torch.tensor(0.0, device=device)
            loss_dict['aff_vis'] = vis_loss

            # Keypoint total loss
            kp_loss = proj_2d_loss * self.proj_2d_weight + vis_loss * self.aff_vis_weight
            loss_dict['kp_total'] = kp_loss

            # ====== [STRATEGY] Strategy Classification Loss ======
            if strategy_logits is not None and strategy_type is not None:
                strategy_loss = F.cross_entropy(strategy_logits, strategy_type.long())
                strategy_pred = strategy_logits.argmax(dim=-1)
                strategy_acc = (strategy_pred == strategy_type).float().mean()
                loss_dict['strategy_loss'] = strategy_loss
                loss_dict['strategy_acc'] = strategy_acc
            else:
                loss_dict['strategy_loss'] = torch.tensor(0.0, device=device)
                loss_dict['strategy_acc'] = torch.tensor(0.0, device=device)

            # ====== [STRATEGY] Phase Classification Loss ======
            if phase_logits is not None and phase_type is not None:
                phase_loss = F.cross_entropy(phase_logits, phase_type.long())
                phase_pred = phase_logits.argmax(dim=-1)
                phase_acc = (phase_pred == phase_type).float().mean()
                loss_dict['phase_loss'] = phase_loss
                loss_dict['phase_acc'] = phase_acc
            else:
                loss_dict['phase_loss'] = torch.tensor(0.0, device=device)
                loss_dict['phase_acc'] = torch.tensor(0.0, device=device)

            # ====== Total Loss ======
            loss_dict['total_losses'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict

        else:  # inference time
            # Model internally uses both predictors for condition injection
            (a_hat, _, (_, _), _, _, _, _, _) = self.model(
                qpos, image, env_state,
                camera_extrinsics=camera_extrinsics,
                camera_intrinsics=camera_intrinsics,
                gripper_pose_world=gripper_pose_world
            )
            return a_hat

    def predict_with_aux(self, qpos, image,
                         camera_extrinsics=None, camera_intrinsics=None,
                         gripper_pose_world=None):
        """
        Inference with auxiliary predictions for visualization.

        Returns:
            a_hat: (B, num_queries, input_dim)
            aux: dict with keypoints and strategy/phase logits
        """
        env_state = None
        (a_hat, _, (_, _),
         keypoint_2d_pred, affordance_visible, _, strategy_logits, phase_logits) = self.model(
            qpos, image, env_state,
            camera_extrinsics=camera_extrinsics,
            camera_intrinsics=camera_intrinsics,
            gripper_pose_world=gripper_pose_world
        )
        aux = {
            "keypoint_2d_pred": keypoint_2d_pred,
            "affordance_visible": affordance_visible,
            "strategy_logits": strategy_logits,
            "phase_logits": phase_logits,
        }
        return a_hat, aux

    def _compute_2d_loss(self, pred, gt, visible, has_affordance):
        return compute_2d_loss(pred, gt, visible, has_affordance, self.img_size)

    def _build_optimizer(self):
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters()
                        if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters()
                        if "backbone" in n and p.requires_grad],
             "lr": self._lr_backbone},
        ]
        return torch.optim.AdamW(param_dicts, lr=self._lr, weight_decay=self._weight_decay)

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args)
        self.model = model  # decoder
        self.optimizer = optimizer

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None

        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """
    KL divergence between q(z) = N(mu, sigma^2) and p(z) = N(0, I)
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    # Clamp logvar to prevent exp() overflow
    logvar = torch.clamp(logvar, min=-20, max=20)

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL divergence between two Gaussian distributions:
    q(z) = N(mu_q, sigma_q^2) and p(z) = N(mu_p, sigma_p^2)
    """
    batch_size = mu_q.size(0)
    assert batch_size != 0

    # Reshape if needed
    if mu_q.data.ndimension() == 4:
        mu_q = mu_q.view(mu_q.size(0), mu_q.size(1))
    if logvar_q.data.ndimension() == 4:
        logvar_q = logvar_q.view(logvar_q.size(0), logvar_q.size(1))
    if mu_p.data.ndimension() == 4:
        mu_p = mu_p.view(mu_p.size(0), mu_p.size(1))
    if logvar_p.data.ndimension() == 4:
        logvar_p = logvar_p.view(logvar_p.size(0), logvar_p.size(1))

    # Clamp logvar to prevent exp() overflow
    logvar_q = torch.clamp(logvar_q, min=-20, max=20)
    logvar_p = torch.clamp(logvar_p, min=-20, max=20)

    # KL divergence formula for two Gaussians
    var_ratio = (logvar_q - logvar_p).exp()
    mu_diff_sq = (mu_q - mu_p).pow(2)
    inv_var_p = (-logvar_p).exp()

    klds = 0.5 * (logvar_p - logvar_q + var_ratio + mu_diff_sq * inv_var_p - 1)

    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
