# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR-VAE model for ACT_BC_ENC_KEYPOINT_STRATEGY.

This fusion version integrates:
1. KeypointPosePredictor for contact/grasp/affordance 2D localization (Memory injection)
2. StrategyPhasePredictor for strategy/phase classification (Query injection)
3. 9 type embeddings for the conditional encoder
4. Dual decoder injection: condition_tokens + query_condition
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from ...keypoint_predictor import KeypointPosePredictor
from ...strategy_phase_predictor import StrategyPhasePredictor

import numpy as np


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """DETR-VAE with dual injection: KeypointPose (Memory) + Strategy/Phase (Query)"""

    def __init__(self, backbones, transformer, encoder, state_dim,
                 num_queries, camera_names, input_dim,
                 condition_encoder=True,
                 encoder_use_type_embed=True,
                 img_size=256,
                 num_strategies=3,
                 num_phases=4,
                 keypoint_predictor_cfg=None,
                 strategy_predictor_cfg=None,
                 temporal_fuse_alpha=0.7,
                 use_gripper_token=True,
                 gripper_token_fuse="mean",
                 pred_backbones=None):
        """
        Parameters:
            backbones: torch module of the backbone to be used
            transformer: torch module of the transformer architecture (supports dual injection)
            encoder: transformer encoder for CVAE
            state_dim: robot state dimension (action output dimension)
            num_queries: number of action queries (chunk size)
            camera_names: list of camera names
            input_dim: action dimension
            condition_encoder: whether to inject conditions into encoder
            encoder_use_type_embed: whether to use type embeddings in encoder
            img_size: image resolution for 2D keypoint prediction
            num_strategies: number of strategy types (EdgeHang/WallLever/PressTilt)
            num_phases: number of phase types (PreManip/Grasp/ClearPath/Lift)
            keypoint_predictor_cfg: config for KeypointPosePredictor
            pred_backbones: predictor backbone list (dual-backbone mandatory)
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.condition_encoder = condition_encoder
        self.encoder_use_type_embed = encoder_use_type_embed
        self.img_size = img_size
        self.num_strategies = num_strategies
        self.num_phases = num_phases
        self.temporal_fuse_alpha = float(temporal_fuse_alpha)
        self.temporal_qpos_proj = nn.Linear(self.input_dim, 1)
        self.use_gripper_token = bool(use_gripper_token)
        self.gripper_token_fuse = gripper_token_fuse

        # Decoder heads
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(self.input_dim, hidden_dim)
        else:
            raise ValueError("ACT_BC_KEYPOINT_STRATEGY requires visual backbones for action branch.")

        if pred_backbones is not None:
            self.pred_backbones = nn.ModuleList(pred_backbones)
            self.pred_input_proj = nn.Conv2d(pred_backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.pred_encoder_proprio_proj = nn.Linear(self.input_dim, hidden_dim)
            self.pred_temporal_qpos_proj = nn.Linear(self.input_dim, 1)
        else:
            raise ValueError("Dual-backbone is mandatory; pred_backbones must be provided.")

        # CVAE encoder parameters
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(self.input_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # mu and logvar

        # Decoder: project latent to hidden dim
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        if condition_encoder and backbones is not None:
            # === Conditional encoder components ===
            self.encoder_proprio_proj = nn.Linear(self.input_dim, hidden_dim)

            # === Strategy and Phase Embeddings (from STRATEGY) ===
            self.strategy_embed = nn.Embedding(num_strategies, hidden_dim)
            self.phase_embed = nn.Embedding(num_phases, hidden_dim)

            # === Affordance Invisible Embedding (from KEYPOINT) ===
            # When affordance is not visible, use this learned embedding instead
            self.affordance_invisible_embed = nn.Parameter(torch.zeros(1, hidden_dim))

            # === Type embeddings: 11 types ===
            # 0=CLS, 1=qpos, 2=strategy, 3=phase, 4=kp_contact, 5=kp_grasp, 6=kp_affordance,
            # 7=gripper_r, 8=gripper_l, 9=image, 10=action
            if encoder_use_type_embed:
                self.encoder_type_embed = nn.Embedding(11, hidden_dim)

            # Position table for encoder sequence
            max_img_tokens = len(camera_names) * 64 + 100
            # 1(CLS) + 1(qpos) + 1(strategy) + 1(phase) + 3(keypoints) + 2(grippers) + img + action
            extra_gripper = 2 if self.use_gripper_token else 0
            encoder_pos_table_size = 1 + 1 + 1 + 1 + 3 + extra_gripper + max_img_tokens + num_queries + 50
            self.register_buffer(
                'encoder_pos_table',
                get_sinusoid_encoding_table(encoder_pos_table_size, hidden_dim)
            )

            # === Conditional Prior p(z|c) ===
            # Input: qpos + img + strategy + phase + contact + grasp + affordance (+ gripper_r + gripper_l)
            # Extended from KEYPOINT (5: qpos+img+3keypoints) and STRATEGY (4: qpos+img+strategy+phase)
            prior_in_dim = hidden_dim * (9 if self.use_gripper_token else 7)
            self.prior_proj = nn.Sequential(
                nn.Linear(prior_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.latent_dim * 2)
            )

            # === Strategy/Phase Predictor (from STRATEGY) ===
            sp_cfg = strategy_predictor_cfg or {}
            self.strategy_phase_predictor = StrategyPhasePredictor(
                hidden_dim=hidden_dim,
                num_strategies=num_strategies,
                num_phases=num_phases,
                dropout=sp_cfg.get('dropout', 0.1)
            )

            # === Keypoint Pose Predictor (from KEYPOINT) ===
            kp_cfg = keypoint_predictor_cfg or {}
            self.keypoint_pose_predictor = KeypointPosePredictor(
                hidden_dim=hidden_dim,
                num_cameras=len(camera_names),
                img_size=img_size,
                num_heads=kp_cfg.get('num_heads', 8),
                num_layers=kp_cfg.get('num_layers', 2),
                dropout=kp_cfg.get('dropout', 0.1)
            )

            # === Query Condition Projection (for STRATEGY injection) ===
            self.query_condition_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            # === Condition Token Position Embedding (for KEYPOINT/GRIPPER memory injection) ===
            cond_len = 5 if self.use_gripper_token else 3
            self.condition_pos_embed = nn.Embedding(cond_len, hidden_dim)
        else:
            self.register_buffer('pos_table', get_sinusoid_encoding_table(num_queries + 1, hidden_dim))

    def _temporal_softmax_pool(self, seq: torch.Tensor, proj: nn.Module):
        weights = proj(seq).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        pooled = (seq * weights.unsqueeze(-1)).sum(dim=1)
        return pooled

    def _encode_visual_features(self, qpos, image, use_pred_backbone=False):
        """
        Encode visual and proprioceptive features.

        Returns:
            proprio_embed: (B, hidden_dim) proprioception embedding
            img_embed: (N_img, B, hidden_dim) image feature embeddings for encoder
            img_pooled: (B, hidden_dim) pooled image features
            img_tokens: (B, N_img, hidden_dim) image tokens for KeypointPosePredictor
        """
        if use_pred_backbone:
            if self.pred_backbones is None:
                raise ValueError("predictor backbone is required in dual-backbone mode.")
            backbones = self.pred_backbones
            input_proj = self.pred_input_proj
            proprio_proj = self.pred_encoder_proprio_proj
            temporal_qpos_proj = self.pred_temporal_qpos_proj
        else:
            backbones = self.backbones
            input_proj = self.input_proj
            proprio_proj = self.encoder_proprio_proj
            temporal_qpos_proj = self.temporal_qpos_proj

        if qpos.dim() == 3:
            qpos_last = qpos[:, -1]
            qpos_pool = self._temporal_softmax_pool(qpos, temporal_qpos_proj)
            qpos = self.temporal_fuse_alpha * qpos_last + (1.0 - self.temporal_fuse_alpha) * qpos_pool

        proprio_embed = proprio_proj(qpos)

        all_img_features = []
        all_img_pooled = []

        for cam_id, cam_name in enumerate(self.camera_names):
            features, _ = backbones[cam_id](image[:, cam_id])
            features = features[0]
            features = input_proj(features)

            pooled = features.mean(dim=[2, 3])
            all_img_pooled.append(pooled)

            features_flat = features.flatten(2).permute(2, 0, 1)
            all_img_features.append(features_flat)

        img_embed = torch.cat(all_img_features, dim=0)  # (N_img, B, hidden_dim)
        img_pooled = torch.stack(all_img_pooled, dim=0).mean(dim=0)  # (B, hidden_dim)

        # img_tokens for KeypointPosePredictor: (B, N_img, hidden_dim)
        img_tokens = img_embed.permute(1, 0, 2)

        return proprio_embed, img_embed, img_pooled, img_tokens

    def _build_gripper_tokens(self, gripper_pose_world, image, camera_extrinsics, camera_intrinsics,
                              use_pred_backbone=False):
        """
        Build gripper tokens by projecting world coordinates to each camera view
        and sampling backbone feature maps at the projected 2D locations.

        Args:
            gripper_pose_world: dict {"right": Tensor, "left": Tensor}, each (B,7) or (B,T,7)
            image: (B, num_cam, C, H, W)
            camera_extrinsics/intrinsics: dict cam_name -> (B,4,4)/(4,4) and (B,3,3)/(3,3)
        Returns:
            gripper_tokens: (B, 2, D) for [right, left]
        """
        if gripper_pose_world is None or camera_extrinsics is None or camera_intrinsics is None:
            raise ValueError("gripper_pose_world/camera_extrinsics/camera_intrinsics are required for gripper token.")

        device = image.device
        dtype = image.dtype

        def _as_tensor(x):
            if torch.is_tensor(x):
                return x.to(device=device, dtype=dtype)
            return torch.as_tensor(x, device=device, dtype=dtype)

        def _last_step(pose):
            pose = _as_tensor(pose)
            return pose[:, -1] if pose.dim() == 3 else pose

        # world xyz
        right_pose = _last_step(gripper_pose_world["right"])
        left_pose = _last_step(gripper_pose_world["left"])
        right_xyz = right_pose[:, :3]
        left_xyz = left_pose[:, :3]

        def _project_3d_to_2d_torch(points_3d, extr, intr, image_size):
            # extr: camera-to-world, intr: 3x3, image_size=(W,H)
            points_3d = _as_tensor(points_3d)
            extr = _as_tensor(extr)
            intr = _as_tensor(intr)
            # If time dimension exists, use the latest step
            if extr.dim() == 4:
                extr = extr[:, -1]
            if intr.dim() == 4:
                intr = intr[:, -1]
            if extr.dim() == 2:
                extr = extr.unsqueeze(0).repeat(points_3d.shape[0], 1, 1)
            if intr.dim() == 2:
                intr = intr.unsqueeze(0).repeat(points_3d.shape[0], 1, 1)

            R = extr[:, :3, :3]
            C = extr[:, :3, 3:4]
            R_inv = R.transpose(1, 2)
            extr_w2c = torch.cat([R_inv, -torch.bmm(R_inv, C)], dim=-1)
            cam_proj = torch.bmm(intr, extr_w2c)

            ones = torch.ones(points_3d.shape[0], 1, device=device, dtype=dtype)
            p_h = torch.cat([points_3d, ones], dim=-1).unsqueeze(-1)
            p_img = torch.bmm(cam_proj, p_h).squeeze(-1)
            z = p_img[:, 2]
            u = p_img[:, 0] / z
            v = p_img[:, 1] / z

            W, H = image_size
            visible = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            uv = torch.stack([u, v], dim=-1)
            return uv, visible, z

        if use_pred_backbone:
            if self.pred_backbones is None or self.pred_input_proj is None:
                raise ValueError("predictor backbone is required for gripper tokens.")
            backbones = self.pred_backbones
            input_proj = self.pred_input_proj
        else:
            backbones = self.backbones
            input_proj = self.input_proj

        right_tokens = []
        left_tokens = []
        right_vis = []
        left_vis = []
        right_depths = []
        left_depths = []

        for cam_id, cam_name in enumerate(self.camera_names):
            cam_img = image[:, cam_id]

            features, _ = backbones[cam_id](cam_img)
            feats = input_proj(features[0])
            B, D, Hf, Wf = feats.shape

            W_img = cam_img.shape[-1]
            H_img = cam_img.shape[-2]
            right_uv, right_visible, right_depth = _project_3d_to_2d_torch(
                right_xyz, camera_extrinsics[cam_name], camera_intrinsics[cam_name], (W_img, H_img)
            )
            left_uv, left_visible, left_depth = _project_3d_to_2d_torch(
                left_xyz, camera_extrinsics[cam_name], camera_intrinsics[cam_name], (W_img, H_img)
            )

            right_grid = torch.stack([
                right_uv[:, 0] / (W_img - 1) * 2 - 1,
                right_uv[:, 1] / (H_img - 1) * 2 - 1
            ], dim=-1).view(B, 1, 1, 2)
            left_grid = torch.stack([
                left_uv[:, 0] / (W_img - 1) * 2 - 1,
                left_uv[:, 1] / (H_img - 1) * 2 - 1
            ], dim=-1).view(B, 1, 1, 2)

            r_token = torch.nn.functional.grid_sample(
                feats, right_grid, align_corners=True
            ).squeeze(-1).squeeze(-1)
            l_token = torch.nn.functional.grid_sample(
                feats, left_grid, align_corners=True
            ).squeeze(-1).squeeze(-1)

            invisible = self.affordance_invisible_embed.expand(B, -1)
            r_token = torch.where(right_visible.unsqueeze(-1), r_token, invisible)
            l_token = torch.where(left_visible.unsqueeze(-1), l_token, invisible)

            right_tokens.append(r_token)
            left_tokens.append(l_token)
            right_vis.append(right_visible)
            left_vis.append(left_visible)
            right_depths.append(right_depth)
            left_depths.append(left_depth)

        right_tokens = torch.stack(right_tokens, dim=1)
        left_tokens = torch.stack(left_tokens, dim=1)
        right_vis = torch.stack(right_vis, dim=1).float()
        left_vis = torch.stack(left_vis, dim=1).float()
        right_depths = torch.stack(right_depths, dim=1).float()
        left_depths = torch.stack(left_depths, dim=1).float()

        if self.gripper_token_fuse == "weighted":
            r_w_raw = right_vis / right_depths.clamp(min=1e-6)
            l_w_raw = left_vis / left_depths.clamp(min=1e-6)
            r_w = r_w_raw / r_w_raw.sum(dim=1, keepdim=True).clamp(min=1.0)
            l_w = l_w_raw / l_w_raw.sum(dim=1, keepdim=True).clamp(min=1.0)
            right_fused = (right_tokens * r_w.unsqueeze(-1)).sum(dim=1)
            left_fused = (left_tokens * l_w.unsqueeze(-1)).sum(dim=1)
        else:
            r_w = right_vis.unsqueeze(-1)
            l_w = left_vis.unsqueeze(-1)
            right_fused = (right_tokens * r_w).sum(dim=1) / r_w.sum(dim=1).clamp(min=1.0)
            left_fused = (left_tokens * l_w).sum(dim=1) / l_w.sum(dim=1).clamp(min=1.0)

        r_any = right_vis.sum(dim=1) > 0
        l_any = left_vis.sum(dim=1) > 0
        invisible = self.affordance_invisible_embed.expand(right_fused.shape[0], -1)
        right_fused = torch.where(r_any.unsqueeze(-1), right_fused, invisible)
        left_fused = torch.where(l_any.unsqueeze(-1), left_fused, invisible)

        return torch.stack([right_fused, left_fused], dim=1)

    def forward(self, qpos, image, env_state, actions=None, is_pad=None,
                has_affordance=None, strategy_type=None, phase_type=None,
                camera_extrinsics=None, camera_intrinsics=None,
                gripper_pose_world=None):
        """
        Forward pass with dual injection (keypoint + strategy/phase).

        Args:
            qpos: (B, input_dim) robot proprioception
            image: (B, num_cam, C, H, W) images
            env_state: unused
            actions: (B, seq, input_dim) action sequence (None during inference)
            is_pad: (B, seq) padding mask
            has_affordance: (B,) GT affordance existence (for visibility loss, training only)
            strategy_type: (B,) GT strategy labels (for classification loss, training only)
            phase_type: (B,) GT phase labels (for classification loss, training only)
            camera_extrinsics/intrinsics: dict cam_name -> extr/intr (for gripper token)
            gripper_pose_world: dict {"right","left"} world poses (for gripper token)

        Returns:
            a_hat: predicted actions
            is_pad_hat: predicted padding
            [mu, logvar]: latent distribution parameters
            keypoint_2d_pred: Dict[kp_name -> (B, num_cameras, 2)] predicted 2D coords
            affordance_visible: (B,) predicted affordance visibility (bool)
            affordance_visible_logits: (B,) raw logits for visibility
            strategy_logits: (B, num_strategies) strategy classification logits
            phase_logits: (B, num_phases) phase classification logits
        """
        is_training = actions is not None
        bs = qpos.shape[0]

        # Default outputs
        keypoint_2d_pred = None
        affordance_visible = None
        affordance_visible_logits = None
        strategy_logits = None
        phase_logits = None

        if self.condition_encoder and self.backbones is not None:
            # === Step 1: Encode visual and proprio features ===
            proprio_embed, img_embed, img_pooled, img_tokens = self._encode_visual_features(
                qpos, image, use_pred_backbone=True
            )
            N_img = img_embed.shape[0]

            # === Step 2: Strategy/Phase Prediction ===
            strategy_logits, phase_logits = self.strategy_phase_predictor(img_pooled, proprio_embed)
            strategy_pred = strategy_logits.argmax(dim=-1)
            phase_pred = phase_logits.argmax(dim=-1)

            # Strategy/Phase embeddings (using predicted values for condition injection)
            strategy_feat = self.strategy_embed(strategy_pred)  # (B, hidden_dim)
            phase_feat = self.phase_embed(phase_pred)            # (B, hidden_dim)
            strategy_token = strategy_feat.unsqueeze(0)          # (1, B, hidden_dim)
            phase_token = phase_feat.unsqueeze(0)                # (1, B, hidden_dim)

            # === Step 3: Keypoint Prediction (matches KEYPOINT's interface) ===
            # KeypointPosePredictor expects: visual_feat, proprio_feat, img_tokens
            kp_features, keypoint_2d_pred, affordance_visible, affordance_visible_logits = \
                self.keypoint_pose_predictor(
                    visual_feat=img_pooled,
                    proprio_feat=proprio_embed,
                    img_tokens=img_tokens
                )
            # kp_features: (B, 3, hidden_dim) - tensor, NOT dict!
            # keypoint_2d_pred: Dict[kp_name -> (B, num_cameras, 2)]
            # affordance_visible: (B, 1) probability
            # affordance_visible_logits: (B, 1) raw logits

            # === Step 4: Build keypoint tokens (matches KEYPOINT's logic) ===
            # Extract individual keypoint features from tensor
            contact_feat = kp_features[:, 0, :]        # (B, hidden_dim)
            grasp_feat = kp_features[:, 1, :]          # (B, hidden_dim)
            affordance_feat_raw = kp_features[:, 2, :] # (B, hidden_dim)

            # Hard threshold selection for affordance visibility (from KEYPOINT)
            # Use invisible embedding when affordance_visible < 0.5
            affordance_invisible = self.affordance_invisible_embed.expand(bs, -1)  # (B, hidden_dim)
            affordance_feat = torch.where(
                affordance_visible > 0.5,  # (B, 1) -> broadcasts
                affordance_feat_raw,
                affordance_invisible
            )

            # Build tokens for encoder sequence
            contact_token = contact_feat.unsqueeze(0)     # (1, B, hidden_dim)
            grasp_token = grasp_feat.unsqueeze(0)         # (1, B, hidden_dim)
            affordance_token = affordance_feat.unsqueeze(0)  # (1, B, hidden_dim)

            # === Step 5: Build condition_pooled for prior ===
            # === Step 5: Build gripper tokens (optional) ===
            gripper_r_token = None
            gripper_l_token = None
            if self.use_gripper_token:
                gripper_tokens = self._build_gripper_tokens(
                    gripper_pose_world, image, camera_extrinsics, camera_intrinsics,
                    use_pred_backbone=True
                )  # (B, 2, D)
                gripper_r_token = gripper_tokens[:, 0, :].unsqueeze(0)  # (1, B, D)
                gripper_l_token = gripper_tokens[:, 1, :].unsqueeze(0)

            # === Step 6: Build condition_pooled for prior ===
            if self.use_gripper_token:
                condition_pooled = torch.cat([
                    proprio_embed,    # (B, hidden_dim)
                    img_pooled,       # (B, hidden_dim)
                    strategy_feat,    # (B, hidden_dim)
                    phase_feat,       # (B, hidden_dim)
                    contact_feat,     # (B, hidden_dim)
                    grasp_feat,       # (B, hidden_dim)
                    affordance_feat,  # (B, hidden_dim)
                    gripper_tokens[:, 0, :],
                    gripper_tokens[:, 1, :]
                ], dim=-1)  # (B, hidden_dim * 9)
            else:
                condition_pooled = torch.cat([
                    proprio_embed,    # (B, hidden_dim)
                    img_pooled,       # (B, hidden_dim)
                    strategy_feat,    # (B, hidden_dim)
                    phase_feat,       # (B, hidden_dim)
                    contact_feat,     # (B, hidden_dim)
                    grasp_feat,       # (B, hidden_dim)
                    affordance_feat   # (B, hidden_dim)
                ], dim=-1)  # (B, hidden_dim * 7)

            if is_training:
                # --- Training: Encode full sequence ---
                action_embed = self.encoder_action_proj(actions)
                action_embed = action_embed.permute(1, 0, 2)  # (seq, B, hidden_dim)
                N_act = action_embed.shape[0]

                cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(1, bs, 1)
                proprio_embed_seq = proprio_embed.unsqueeze(0)

                # Encoder input: [CLS, qpos, strategy, phase, kp_contact, kp_grasp, kp_aff, (gripper_r, gripper_l), img, actions]
                encoder_chunks = [
                    cls_embed,          # (1, B, D) - type 0
                    proprio_embed_seq,  # (1, B, D) - type 1
                    strategy_token,     # (1, B, D) - type 2
                    phase_token,        # (1, B, D) - type 3
                    contact_token,      # (1, B, D) - type 4
                    grasp_token,        # (1, B, D) - type 5
                    affordance_token,   # (1, B, D) - type 6
                ]
                if self.use_gripper_token:
                    encoder_chunks.extend([
                        gripper_r_token,  # (1, B, D) - type 7
                        gripper_l_token,  # (1, B, D) - type 8
                    ])
                encoder_chunks.extend([
                    img_embed,          # (N_img, B, D) - type 9
                    action_embed        # (N_act, B, D) - type 10
                ])
                encoder_input = torch.cat(encoder_chunks, dim=0)

                total_len = encoder_input.shape[0]

                # Position encoding
                pos_embed = self.encoder_pos_table[:, :total_len, :].clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)

                # Type embeddings
                if self.encoder_use_type_embed:
                    type_blocks = [
                        torch.zeros(1, bs, dtype=torch.long, device=qpos.device),           # CLS
                        torch.ones(1, bs, dtype=torch.long, device=qpos.device),            # qpos
                        torch.full((1, bs), 2, dtype=torch.long, device=qpos.device),       # strategy
                        torch.full((1, bs), 3, dtype=torch.long, device=qpos.device),       # phase
                        torch.full((1, bs), 4, dtype=torch.long, device=qpos.device),       # kp_contact
                        torch.full((1, bs), 5, dtype=torch.long, device=qpos.device),       # kp_grasp
                        torch.full((1, bs), 6, dtype=torch.long, device=qpos.device),       # kp_affordance
                    ]
                    if self.use_gripper_token:
                        type_blocks.extend([
                            torch.full((1, bs), 7, dtype=torch.long, device=qpos.device),   # gripper_r
                            torch.full((1, bs), 8, dtype=torch.long, device=qpos.device),   # gripper_l
                        ])
                        img_type_id = 9
                        act_type_id = 10
                    else:
                        img_type_id = 7
                        act_type_id = 8

                    type_blocks.extend([
                        torch.full((N_img, bs), img_type_id, dtype=torch.long, device=qpos.device),   # image
                        torch.full((N_act, bs), act_type_id, dtype=torch.long, device=qpos.device),   # action
                    ])
                    type_ids = torch.cat(type_blocks, dim=0)
                    type_embed = self.encoder_type_embed(type_ids)
                    encoder_input = encoder_input + type_embed

                # Padding mask: everything before actions is not padded
                # Fixed tokens: CLS+qpos+strategy+phase+keypoints(+grippers)+img
                n_fixed = (9 if self.use_gripper_token else 7) + N_img
                condition_mask = torch.full((bs, n_fixed), False, device=qpos.device)
                encoder_is_pad = torch.cat([condition_mask, is_pad], dim=1)

                # Encode
                encoder_output = self.encoder(
                    encoder_input, pos=pos_embed, src_key_padding_mask=encoder_is_pad
                )
                cls_output = encoder_output[0]

                # Latent space
                latent_info = self.latent_proj(cls_output)
                mu = latent_info[:, :self.latent_dim]
                logvar = latent_info[:, self.latent_dim:]

                # Prior
                prior_info = self.prior_proj(condition_pooled)
                mu_prior = prior_info[:, :self.latent_dim]
                logvar_prior = prior_info[:, self.latent_dim:]

                latent_sample = reparametrize(mu, logvar)
                latent_input = self.latent_out_proj(latent_sample)

                self._mu_prior = mu_prior
                self._logvar_prior = logvar_prior

            else:
                # --- Inference: Sample from prior ---
                prior_info = self.prior_proj(condition_pooled)
                mu_prior = prior_info[:, :self.latent_dim]
                logvar_prior = prior_info[:, self.latent_dim:]

                latent_sample = reparametrize(mu_prior, logvar_prior)
                latent_input = self.latent_out_proj(latent_sample)

                mu = mu_prior
                logvar = logvar_prior

            if qpos.dim() == 3:
                qpos_last = qpos[:, -1]
                qpos_pool = self._temporal_softmax_pool(qpos, self.temporal_qpos_proj)
                qpos_action = self.temporal_fuse_alpha * qpos_last + (1.0 - self.temporal_fuse_alpha) * qpos_pool
            else:
                qpos_action = qpos

            # === Decoder with dual injection ===
            all_cam_features_dec = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features_dec.append(self.input_proj(features))
                all_cam_pos.append(pos)

            proprio_input = self.input_proj_robot_state(qpos_action)
            src = torch.cat(all_cam_features_dec, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # Memory injection: keypoint (+ gripper) condition tokens
            if self.use_gripper_token:
                condition_tokens = torch.cat(
                    [contact_token, grasp_token, affordance_token, gripper_r_token, gripper_l_token], dim=0
                )  # (5, B, D)
            else:
                condition_tokens = torch.cat([contact_token, grasp_token, affordance_token], dim=0)  # (3, B, D)
            condition_pos = self.condition_pos_embed.weight

            # Query injection: strategy/phase condition
            query_condition = self.query_condition_proj(
                torch.cat([strategy_feat, phase_feat], dim=-1)
            )

            hs = self.transformer(
                src, None, self.query_embed.weight, pos,
                latent_input, proprio_input, self.additional_pos_embed.weight,
                condition_tokens=condition_tokens,
                condition_pos=condition_pos,
                query_condition=query_condition
            )[0]

        else:
            # === Original ACT Path (no condition) ===
            if is_training:
                action_embed = self.encoder_action_proj(actions)
                cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
                encoder_input = torch.cat([cls_embed, action_embed], axis=1)
                encoder_input = encoder_input.permute(1, 0, 2)

                cls_is_pad = torch.full((bs, 1), False).to(qpos.device)
                is_pad_full = torch.cat([cls_is_pad, is_pad], axis=1)

                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)

                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_full)
                encoder_output = encoder_output[0]

                latent_info = self.latent_proj(encoder_output)
                mu = latent_info[:, :self.latent_dim]
                logvar = latent_info[:, self.latent_dim:]
                latent_sample = reparametrize(mu, logvar)
                latent_input = self.latent_out_proj(latent_sample)
            else:
                mu = logvar = None
                latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                latent_input = self.latent_out_proj(latent_sample)

            if self.backbones is not None:
                all_cam_features_dec = []
                all_cam_pos = []
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, pos = self.backbones[cam_id](image[:, cam_id])
                    features = features[0]
                    pos = pos[0]
                    all_cam_features_dec.append(self.input_proj(features))
                    all_cam_pos.append(pos)

                if qpos.dim() == 3:
                    qpos_last = qpos[:, -1]
                    qpos_pool = self._temporal_softmax_pool(qpos, self.temporal_qpos_proj)
                    qpos_action = self.temporal_fuse_alpha * qpos_last + (1.0 - self.temporal_fuse_alpha) * qpos_pool
                else:
                    qpos_action = qpos

                proprio_input = self.input_proj_robot_state(qpos_action)
                src = torch.cat(all_cam_features_dec, axis=3)
                pos = torch.cat(all_cam_pos, axis=3)

                hs = self.transformer(src, None, self.query_embed.weight, pos,
                                      latent_input, proprio_input, self.additional_pos_embed.weight)[0]
            else:
                qpos_proj = self.input_proj_robot_state(qpos)
                env_state_proj = self.input_proj_env_state(env_state)
                transformer_input = torch.cat([qpos_proj, env_state_proj], axis=1)
                hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return (a_hat, is_pad_hat, [mu, logvar],
                keypoint_2d_pred, affordance_visible, affordance_visible_logits,
                strategy_logits, phase_logits)

    def load_predictor_state(self, ckpt_path: str, strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if self.pred_backbones is not None and "pred_backbone" in ckpt:
            self.pred_backbones.load_state_dict(ckpt["pred_backbone"], strict=strict)
        if self.pred_input_proj is not None and "pred_input_proj" in ckpt:
            self.pred_input_proj.load_state_dict(ckpt["pred_input_proj"], strict=strict)
        if self.pred_encoder_proprio_proj is not None and "pred_encoder_proprio_proj" in ckpt:
            self.pred_encoder_proprio_proj.load_state_dict(ckpt["pred_encoder_proprio_proj"], strict=strict)
        if hasattr(self, "pred_temporal_qpos_proj") and "pred_temporal_qpos_proj" in ckpt:
            self.pred_temporal_qpos_proj.load_state_dict(ckpt["pred_temporal_qpos_proj"], strict=strict)
        if "keypoint_pose_predictor" in ckpt:
            self.keypoint_pose_predictor.load_state_dict(ckpt["keypoint_pose_predictor"], strict=strict)
        if "strategy_phase_predictor" in ckpt:
            self.strategy_phase_predictor.load_state_dict(ckpt["strategy_phase_predictor"], strict=strict)

    def freeze_predictor_modules(self):
        modules = [
            self.pred_backbones,
            self.pred_input_proj,
            self.pred_encoder_proprio_proj,
            getattr(self, "pred_temporal_qpos_proj", None),
            self.keypoint_pose_predictor,
            self.strategy_phase_predictor,
        ]
        for m in modules:
            if m is None:
                continue
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def get_prior_params(self):
        """Get the prior parameters computed during training forward pass."""
        if hasattr(self, '_mu_prior') and hasattr(self, '_logvar_prior'):
            return self._mu_prior, self._logvar_prior
        return None, None


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        is_training = actions is not None
        bs, _ = qpos.shape
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            pos = pos[0]
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)
        features = torch.cat([flattened_features, qpos], axis=1)
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.input_dim

    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    transformer = build_transformer(args)
    encoder = build_encoder(args)

    # Get settings
    condition_encoder = getattr(args, 'condition_encoder', True)
    encoder_use_type_embed = getattr(args, 'encoder_use_type_embed', True)
    img_size = getattr(args, 'img_size', 256)
    num_strategies = getattr(args, 'num_strategies', 3)
    num_phases = getattr(args, 'num_phases', 4)
    temporal_fuse_alpha = getattr(args, 'temporal_fuse_alpha', 0.7)
    use_gripper_token = getattr(args, 'use_gripper_token', True)
    gripper_token_fuse = getattr(args, 'gripper_token_fuse', 'mean')

    # Predictor backbone (dual-backbone mandatory)
    pred_backbones = [build_backbone(args) for _ in args.camera_names]

    # Keypoint predictor config
    kp_cfg = getattr(args, 'keypoint_predictor', None)
    if kp_cfg is not None:
        keypoint_predictor_cfg = dict(kp_cfg)
    else:
        keypoint_predictor_cfg = {
            'num_heads': 8,
            'num_layers': 2,
            'dropout': 0.1
        }

    # Strategy predictor config
    sp_cfg = getattr(args, 'strategy_predictor', None)
    if sp_cfg is not None:
        strategy_predictor_cfg = dict(sp_cfg)
    else:
        strategy_predictor_cfg = {
            'dropout': 0.1
        }

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        input_dim=args.input_dim,
        condition_encoder=condition_encoder,
        encoder_use_type_embed=encoder_use_type_embed,
        img_size=img_size,
        num_strategies=num_strategies,
        num_phases=num_phases,
        keypoint_predictor_cfg=keypoint_predictor_cfg,
        strategy_predictor_cfg=strategy_predictor_cfg,
        temporal_fuse_alpha=temporal_fuse_alpha,
        use_gripper_token=use_gripper_token,
        gripper_token_fuse=gripper_token_fuse,
        pred_backbones=pred_backbones,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_cnnmlp(args):
    state_dim = args.input_dim

    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
