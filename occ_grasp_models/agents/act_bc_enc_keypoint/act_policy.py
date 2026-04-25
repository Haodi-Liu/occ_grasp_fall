# ACT Policy with 2D Keypoint Supervision - Path B Version
# Reference: KEYPOINT_POSE_INJECTION_PLAN.md Section 7
#
# *** Path B Design ***
# - REMOVED: 3D keypoint loss (position L1, rotation geodesic)
# - NEW: 2D projection loss only
# - KEEP: Affordance visibility loss (BCE)

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np

from agents.act_bc_enc_keypoint.detr.build import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer


class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args.kl_weight
        self.condition_encoder = getattr(args, 'condition_encoder', False)

        # [Path B] Loss weights
        self.aff_vis_weight = getattr(args, 'aff_vis_weight', 2.0)
        self.proj_2d_weight = getattr(args, 'proj_2d_weight', 5.0)

        # Camera names for organizing 2D GT data
        self.camera_names = getattr(args, 'camera_names', ['front', 'wrist'])

        # Image size for 2D loss normalization (must match data collection resolution)
        self.img_size = getattr(args, 'img_size', 256)

        print(f'KL Weight {self.kl_weight}')
        print(f'Condition Encoder: {self.condition_encoder}')
        print(f'[Path B] 2D Projection Weight: {self.proj_2d_weight}')
        print(f'[Path B] Affordance Visibility Weight: {self.aff_vis_weight}')
        print(f'[Path B] Image Size: {self.img_size}')

    def forward(self, qpos, image, actions=None, is_pad=None,
                has_affordance=None,
                keypoint_2d_gt=None, keypoint_2d_visible=None):
        """
        Forward pass - Path B Version

        Args:
            qpos: (B, input_dim) robot proprioception
            image: (B, num_cam, C, H, W) images
            actions: (B, seq, input_dim) action sequence (None during inference)
            is_pad: (B, seq) padding mask
            has_affordance: Tensor[B] - GT affordance existence (for visibility loss)
            keypoint_2d_gt: Dict[cam_name -> Tensor[B, 3, 2]] - 2D GT coordinates per camera
            keypoint_2d_visible: Dict[cam_name -> Tensor[B, 3]] - visibility flags per camera

        Returns:
            Training: loss_dict with all loss components
            Inference: predicted actions
        """
        env_state = None

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Forward pass - [Path B] model returns keypoint_2d instead of keypoint_poses
            a_hat, is_pad_hat, (mu, logvar), keypoint_2d_pred, \
                affordance_visible, affordance_visible_logits = \
                self.model(qpos, image, env_state, actions, is_pad,
                           has_affordance=has_affordance)

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

            # Action L1 loss (same as before)
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

            # ====== [Path B] 2D Projection Loss ======
            device = keypoint_2d_pred['contact'].device

            proj_2d_loss = self._compute_2d_loss(
                keypoint_2d_pred, keypoint_2d_gt, keypoint_2d_visible, has_affordance
            )
            loss_dict['proj_2d'] = proj_2d_loss

            # ====== Affordance visibility loss ======
            if has_affordance is not None and affordance_visible_logits is not None:
                vis_loss = F.binary_cross_entropy_with_logits(
                    affordance_visible_logits.squeeze(-1),  # Raw logits (B,)
                    has_affordance.float()                   # GT (B,)
                )
            else:
                vis_loss = torch.tensor(0.0, device=device)
            loss_dict['aff_vis'] = vis_loss

            # Total keypoint loss = 2D projection + visibility
            kp_loss = proj_2d_loss * self.proj_2d_weight + vis_loss * self.aff_vis_weight
            loss_dict['kp_total'] = kp_loss

            # Total loss
            loss_dict['total_losses'] = (
                loss_dict['l1'] +
                loss_dict['kl'] * self.kl_weight +
                loss_dict['kp_total']
            )
            return loss_dict

        else:  # inference time
            # Model internally uses Predictor to predict keypoints, no external input needed
            a_hat, _, (_, _), _, _, _ = self.model(qpos, image, env_state)
            return a_hat

    def _compute_2d_loss(self, pred, gt, visible, has_affordance):
        """
        Compute 2D projection loss

        Args:
            pred: Dict[kp_name -> Tensor[B, num_cameras, 2]] - predicted 2D coords
            gt: Dict[cam_name -> Tensor[B, 3, 2]] - GT 2D coords (3 keypoints per camera)
            visible: Dict[cam_name -> Tensor[B, 3]] - visibility flags (3 keypoints per camera)
            has_affordance: Tensor[B] - whether affordance exists

        Returns:
            loss: scalar, normalized 2D distance loss
        """
        device = pred['contact'].device
        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        kp_names = ['contact', 'grasp', 'affordance']
        camera_names = list(gt.keys())

        for cam_idx, cam_name in enumerate(camera_names):
            for kp_idx, kp_name in enumerate(kp_names):
                pred_2d = pred[kp_name][:, cam_idx, :]      # [B, 2]
                gt_2d = gt[cam_name][:, kp_idx, :]          # [B, 2]
                vis_mask = visible[cam_name][:, kp_idx]     # [B]

                # affordance additionally requires has_affordance=True
                if kp_name == 'affordance' and has_affordance is not None:
                    vis_mask = vis_mask & has_affordance

                # Only compute loss for visible keypoints
                if vis_mask.any():
                    diff = pred_2d[vis_mask] - gt_2d[vis_mask]
                    # Normalized L2 distance (divide by image size)
                    loss = (diff ** 2).sum(dim=-1).sqrt().mean() / float(self.img_size)
                    total_loss = total_loss + loss
                    valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count

        return total_loss

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
    KL(q||p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    # Clamp logvar to prevent exp() overflow (exp(88) ≈ 1e38, exp(89) = inf)
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

    KL(q||p) = 0.5 * sum(
        log(sigma_p^2 / sigma_q^2) + (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2 - 1
    )

    In terms of logvar:
    KL(q||p) = 0.5 * sum(
        logvar_p - logvar_q + exp(logvar_q - logvar_p) + (mu_q - mu_p)^2 * exp(-logvar_p) - 1
    )
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
