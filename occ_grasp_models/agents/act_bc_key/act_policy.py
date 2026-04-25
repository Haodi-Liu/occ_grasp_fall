import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from agents.act_bc_key.detr.build import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer


class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args.kl_weight
        self.condition_encoder = getattr(args, 'condition_encoder', False)
        print(f'KL Weight {self.kl_weight}')
        print(f'Condition Encoder: {self.condition_encoder}')

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)

            # Compute KL divergence
            if self.condition_encoder:
                # Get prior parameters from model
                mu_prior, logvar_prior = self.model.get_prior_params()
                if mu_prior is not None:
                    # KL divergence between q(z|a,c) and p(z|c)
                    total_kld, dim_wise_kld, mean_kld = kl_divergence_gaussian(
                        mu, logvar, mu_prior, logvar_prior
                    )
                else:
                    # Fallback to standard KL if prior not available
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            else:
                # Original: KL divergence between q(z|a) and N(0,I)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            loss_dict = dict()

            # Action format: [right_pos(3), right_quat(4), right_gripper(1),
            #                 left_pos(3), left_quat(4), left_gripper(1)] = 16D

            # Right arm: position, quaternion, gripper
            right_pos_gt, right_pos_pred = actions[:, :, 0:3], a_hat[:, :, 0:3]
            right_quat_gt, right_quat_pred = actions[:, :, 3:7], a_hat[:, :, 3:7]
            right_gripper_gt, right_gripper_pred = actions[:, :, 7], a_hat[:, :, 7]

            # Left arm
            left_pos_gt, left_pos_pred = actions[:, :, 8:11], a_hat[:, :, 8:11]
            left_quat_gt, left_quat_pred = actions[:, :, 11:15], a_hat[:, :, 11:15]
            left_gripper_gt, left_gripper_pred = actions[:, :, 15], a_hat[:, :, 15]

            # Weighted L1 loss: higher weight for position (following PPI and common practice)
            pos_weight = 3.0  # Position is more critical
            quat_weight = 1.0  # Quaternion
            gripper_weight = 3.0  # Gripper is important for manipulation

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
            loss_dict['total_losses'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # sample from prior p(z|c)
            return a_hat

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

    # KL divergence formula for two Gaussians
    var_ratio = (logvar_q - logvar_p).exp()
    mu_diff_sq = (mu_q - mu_p).pow(2)
    inv_var_p = (-logvar_p).exp()

    klds = 0.5 * (logvar_p - logvar_q + var_ratio + mu_diff_sq * inv_var_p - 1)

    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
