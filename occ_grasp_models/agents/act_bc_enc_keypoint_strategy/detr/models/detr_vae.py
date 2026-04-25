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
                 keypoint_predictor_cfg=None):
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

        # Decoder heads
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(self.input_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(self.input_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(self.input_dim, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

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

            # === Type embeddings: 9 types ===
            # 0=CLS, 1=qpos, 2=strategy, 3=phase, 4=kp_contact, 5=kp_grasp, 6=kp_affordance, 7=image, 8=action
            if encoder_use_type_embed:
                self.encoder_type_embed = nn.Embedding(9, hidden_dim)

            # Position table for encoder sequence
            max_img_tokens = len(camera_names) * 64 + 100
            # 1(CLS) + 1(qpos) + 1(strategy) + 1(phase) + 3(keypoints) + img + action
            encoder_pos_table_size = 1 + 1 + 1 + 1 + 3 + max_img_tokens + num_queries + 50
            self.register_buffer(
                'encoder_pos_table',
                get_sinusoid_encoding_table(encoder_pos_table_size, hidden_dim)
            )

            # === Conditional Prior p(z|c) ===
            # Input: qpos + img + strategy + phase + contact + grasp + affordance = hidden_dim * 7
            # Extended from KEYPOINT (5: qpos+img+3keypoints) and STRATEGY (4: qpos+img+strategy+phase)
            self.prior_proj = nn.Sequential(
                nn.Linear(hidden_dim * 7, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.latent_dim * 2)
            )

            # === Strategy/Phase Predictor (from STRATEGY) ===
            self.strategy_phase_predictor = StrategyPhasePredictor(
                hidden_dim=hidden_dim,
                num_strategies=num_strategies,
                num_phases=num_phases,
                dropout=0.1
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

            # === Condition Token Position Embedding (for KEYPOINT memory injection) ===
            # 3 positions for contact, grasp, affordance keypoints
            self.condition_pos_embed = nn.Embedding(3, hidden_dim)
        else:
            self.register_buffer('pos_table', get_sinusoid_encoding_table(num_queries + 1, hidden_dim))

    def _encode_visual_features(self, qpos, image):
        """
        Encode visual and proprioceptive features (shared for both predictors).

        Returns:
            proprio_embed: (B, hidden_dim) proprioception embedding
            img_embed: (N_img, B, hidden_dim) image feature embeddings for encoder
            img_pooled: (B, hidden_dim) pooled image features
            img_tokens: (B, N_img, hidden_dim) image tokens for KeypointPosePredictor
        """
        proprio_embed = self.encoder_proprio_proj(qpos)

        all_img_features = []
        all_img_pooled = []

        for cam_id, cam_name in enumerate(self.camera_names):
            features, _ = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            features = self.input_proj(features)

            pooled = features.mean(dim=[2, 3])
            all_img_pooled.append(pooled)

            features_flat = features.flatten(2).permute(2, 0, 1)
            all_img_features.append(features_flat)

        img_embed = torch.cat(all_img_features, dim=0)  # (N_img, B, hidden_dim)
        img_pooled = torch.stack(all_img_pooled, dim=0).mean(dim=0)  # (B, hidden_dim)

        # img_tokens for KeypointPosePredictor: (B, N_img, hidden_dim)
        img_tokens = img_embed.permute(1, 0, 2)

        return proprio_embed, img_embed, img_pooled, img_tokens

    def forward(self, qpos, image, env_state, actions=None, is_pad=None,
                has_affordance=None, strategy_type=None, phase_type=None):
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
            proprio_embed, img_embed, img_pooled, img_tokens = self._encode_visual_features(qpos, image)
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
            # Uses all 7 components: qpos + img + strategy + phase + contact + grasp + affordance
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

                # Encoder input: [CLS, qpos, strategy, phase, kp_contact, kp_grasp, kp_aff, img, actions]
                encoder_input = torch.cat([
                    cls_embed,          # (1, B, D) - type 0
                    proprio_embed_seq,  # (1, B, D) - type 1
                    strategy_token,     # (1, B, D) - type 2
                    phase_token,        # (1, B, D) - type 3
                    contact_token,      # (1, B, D) - type 4
                    grasp_token,        # (1, B, D) - type 5
                    affordance_token,   # (1, B, D) - type 6
                    img_embed,          # (N_img, B, D) - type 7
                    action_embed        # (N_act, B, D) - type 8
                ], dim=0)

                total_len = encoder_input.shape[0]

                # Position encoding
                pos_embed = self.encoder_pos_table[:, :total_len, :].clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)

                # Type embeddings
                if self.encoder_use_type_embed:
                    type_ids = torch.cat([
                        torch.zeros(1, bs, dtype=torch.long, device=qpos.device),           # CLS
                        torch.ones(1, bs, dtype=torch.long, device=qpos.device),            # qpos
                        torch.full((1, bs), 2, dtype=torch.long, device=qpos.device),       # strategy
                        torch.full((1, bs), 3, dtype=torch.long, device=qpos.device),       # phase
                        torch.full((1, bs), 4, dtype=torch.long, device=qpos.device),       # kp_contact
                        torch.full((1, bs), 5, dtype=torch.long, device=qpos.device),       # kp_grasp
                        torch.full((1, bs), 6, dtype=torch.long, device=qpos.device),       # kp_affordance
                        torch.full((N_img, bs), 7, dtype=torch.long, device=qpos.device),   # image
                        torch.full((N_act, bs), 8, dtype=torch.long, device=qpos.device),   # action
                    ], dim=0)
                    type_embed = self.encoder_type_embed(type_ids)
                    encoder_input = encoder_input + type_embed

                # Padding mask: everything before actions is not padded
                # Fixed tokens: CLS(1) + qpos(1) + strategy(1) + phase(1) + keypoints(3) + img(N_img) = 7 + N_img
                n_fixed = 7 + N_img
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

            # === Decoder with dual injection ===
            all_cam_features_dec = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features_dec.append(self.input_proj(features))
                all_cam_pos.append(pos)

            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features_dec, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # Memory injection: keypoint condition tokens
            condition_tokens = torch.cat([contact_token, grasp_token, affordance_token], dim=0)  # (3, B, D)
            condition_pos = self.condition_pos_embed.weight  # (3, D)

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

                proprio_input = self.input_proj_robot_state(qpos)
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
