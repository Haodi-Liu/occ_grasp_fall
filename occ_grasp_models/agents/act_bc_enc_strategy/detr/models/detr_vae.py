# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
Modified for ACT_BC_ENC_STRATEGY: Conditional encoder with strategy/phase injection.

This version extends ACT_BC_ENC by:
1. Adding strategy and phase embeddings to the encoder input sequence
2. Expanding type embeddings from 4 to 6 types (CLS, qpos, strategy, phase, image, action)
3. Conditioning the prior network on strategy and phase features
4. Integrating StrategyPhasePredictor for auxiliary classification
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
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
    """ DETR-VAE module with conditional encoder and strategy/phase injection """
    def __init__(self, backbones, transformer, encoder, state_dim,
                 num_queries, camera_names, input_dim,
                 condition_encoder=False,
                 encoder_use_type_embed=True,
                 num_strategies=3,
                 num_phases=4):
        """
        Parameters:
            backbones: torch module of the backbone to be used
            transformer: torch module of the transformer architecture
            encoder: transformer encoder for CVAE
            state_dim: robot state dimension
            num_queries: number of action queries (chunk size)
            camera_names: list of camera names
            input_dim: action dimension
            condition_encoder: whether to inject conditions into encoder
            encoder_use_type_embed: whether to use type embeddings in encoder
            num_strategies: number of strategy types (default: 3)
                - 0: EdgeHang (edge overhang grasp)
                - 1: WallLever (wall lever grasp)
                - 2: PressTilt (press tilt grasp)
            num_phases: number of execution phases (default: 4)
                - 0: PreManipulation (create grasping space)
                - 1: Grasp (grasp the object)
                - 2: ClearPath (move auxiliary arm away)
                - 3: Lift (lift the object)
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
            # Proprio projection for encoder (shared projection with decoder)
            self.encoder_proprio_proj = nn.Linear(self.input_dim, hidden_dim)

            # === Strategy and Phase Embeddings ===
            # Learnable embeddings for strategy type and execution phase
            # These are injected into the encoder input sequence
            self.strategy_embed = nn.Embedding(num_strategies, hidden_dim)
            self.phase_embed = nn.Embedding(num_phases, hidden_dim)

            # Type embeddings: 0=CLS, 1=qpos, 2=strategy, 3=phase, 4=image, 5=action
            # Expanded from 4 to 6 types to accommodate strategy and phase tokens
            if encoder_use_type_embed:
                self.encoder_type_embed = nn.Embedding(6, hidden_dim)

            # Calculate encoder sequence length for position table
            # Assuming 8x8=64 tokens per camera after backbone
            # Total = 1(CLS) + 1(qpos) + 1(strategy) + 1(phase) + N_cameras*64(img) + num_queries(action)
            max_img_tokens = len(camera_names) * 64 + 100  # buffer
            encoder_pos_table_size = 1 + 1 + 1 + 1 + max_img_tokens + num_queries + 50
            self.register_buffer(
                'encoder_pos_table',
                get_sinusoid_encoding_table(encoder_pos_table_size, hidden_dim)
            )

            # === Conditional Prior Network p(z|c) ===
            # Extended to include strategy and phase features
            # Input: qpos_feat + img_feat + strategy_feat + phase_feat = hidden_dim * 4
            self.prior_proj = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),  # qpos + img + strategy + phase
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.latent_dim * 2)  # mu_prior, logvar_prior
            )

            # === Strategy and Phase Predictor (Auxiliary Task) ===
            # Predicts strategy type and phase from visual + proprioceptive features
            self.strategy_phase_predictor = StrategyPhasePredictor(
                hidden_dim=hidden_dim,
                num_strategies=num_strategies,
                num_phases=num_phases,
                dropout=0.1
            )

            # === [NEW] Query Condition Projection for Decoder Injection ===
            # Projects strategy + phase embeddings to a single condition vector
            # This is added to the decoder target (action queries) for direct conditioning
            self.query_condition_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # strategy_feat + phase_feat
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            # Original position table for action-only encoder
            self.register_buffer('pos_table', get_sinusoid_encoding_table(num_queries + 1, hidden_dim))

    def _encode_visual_features(self, qpos, image):
        """
        Encode visual and proprioceptive features (without strategy/phase).
        This is called first to get features for the StrategyPhasePredictor.

        Args:
            qpos: (B, input_dim) robot proprioception
            image: (B, num_cam, C, H, W) images

        Returns:
            proprio_embed: (B, hidden_dim) proprioception embedding
            img_embed: (N_img, B, hidden_dim) image feature embeddings
            img_pooled: (B, hidden_dim) pooled image features for predictor
        """
        # 1. Proprio embedding
        proprio_embed = self.encoder_proprio_proj(qpos)  # (B, hidden_dim)

        # 2. Image features (using shared backbone with decoder)
        all_img_features = []
        all_img_pooled = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, _ = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # (B, C, H, W)
            features = self.input_proj(features)  # (B, hidden_dim, H, W)

            # Pool for prior network
            pooled = features.mean(dim=[2, 3])  # (B, hidden_dim)
            all_img_pooled.append(pooled)

            # Flatten for encoder sequence
            features = features.flatten(2).permute(2, 0, 1)  # (H*W, B, hidden_dim)
            all_img_features.append(features)

        img_embed = torch.cat(all_img_features, dim=0)  # (N_img, B, hidden_dim)
        img_pooled = torch.stack(all_img_pooled, dim=0).mean(dim=0)  # (B, hidden_dim)

        return proprio_embed, img_embed, img_pooled

    def _build_condition_tokens(self, proprio_embed, img_pooled, strategy_type, phase_type):
        """
        Build strategy/phase tokens and condition_pooled using given strategy/phase indices.
        These indices can be either GT labels or Predictor predictions.

        Args:
            proprio_embed: (B, hidden_dim) proprioception embedding
            img_pooled: (B, hidden_dim) pooled image features
            strategy_type: (B,) strategy type indices (0-based) - from GT or Predictor
            phase_type: (B,) phase type indices (0-based) - from GT or Predictor

        Returns:
            strategy_token: (1, B, hidden_dim) strategy embedding
            phase_token: (1, B, hidden_dim) phase embedding
            condition_pooled: (B, hidden_dim*4) pooled condition for prior network
        """
        # Strategy and Phase embeddings (using provided indices)
        strategy_feat = self.strategy_embed(strategy_type)  # (B, hidden_dim)
        phase_feat = self.phase_embed(phase_type)          # (B, hidden_dim)
        strategy_token = strategy_feat.unsqueeze(0)        # (1, B, hidden_dim)
        phase_token = phase_feat.unsqueeze(0)              # (1, B, hidden_dim)

        # Combined condition for prior network
        # Shape: (B, hidden_dim*4)
        condition_pooled = torch.cat([
            proprio_embed,   # (B, hidden_dim)
            img_pooled,      # (B, hidden_dim)
            strategy_feat,   # (B, hidden_dim)
            phase_feat       # (B, hidden_dim)
        ], dim=-1)

        return strategy_token, phase_token, condition_pooled

    def _encode_conditions(self, qpos, image, strategy_type, phase_type):
        """
        Encode condition information (qpos, images, strategy, phase).
        Returns encoded features and pooled representation for prior network.

        NOTE: This method now uses the provided strategy_type/phase_type directly,
        which should be PREDICTOR PREDICTIONS (not GT labels) for end-to-end inference.

        Args:
            qpos: (B, input_dim) robot proprioception
            image: (B, num_cam, C, H, W) images
            strategy_type: (B,) strategy type indices (from Predictor, 0-based)
            phase_type: (B,) phase type indices (from Predictor, 0-based)

        Returns:
            proprio_embed: (1, B, hidden_dim) proprioception embedding
            strategy_token: (1, B, hidden_dim) strategy embedding
            phase_token: (1, B, hidden_dim) phase embedding
            img_embed: (N_img, B, hidden_dim) image feature embeddings
            condition_pooled: (B, hidden_dim*4) pooled condition for prior network
            img_pooled: (B, hidden_dim) pooled image features for predictor
        """
        # 1. Encode visual features
        proprio_embed, img_embed, img_pooled = self._encode_visual_features(qpos, image)
        proprio_embed_seq = proprio_embed.unsqueeze(0)  # (1, B, hidden_dim)

        # 2. Build condition tokens using provided strategy/phase (from Predictor)
        strategy_token, phase_token, condition_pooled = self._build_condition_tokens(
            proprio_embed, img_pooled, strategy_type, phase_type
        )

        return proprio_embed_seq, strategy_token, phase_token, img_embed, condition_pooled, img_pooled

    def forward(self, qpos, image, env_state, actions=None, is_pad=None,
                strategy_type=None, phase_type=None):
        """
        Forward pass with strategy and phase conditioning.

        *** END-TO-END DESIGN ***
        - Both training and inference use StrategyPhasePredictor to predict strategy/phase
        - Predictor predictions are used for condition injection (NOT GT labels)
        - GT labels (strategy_type, phase_type) are ONLY used for computing classification loss

        Args:
            qpos: (B, input_dim) robot proprioception
            image: (B, num_cam, C, H, W) images
            env_state: unused
            actions: (B, seq, input_dim) action sequence (None during inference)
            is_pad: (B, seq) padding mask for actions
            strategy_type: (B,) GT strategy labels (0-based), ONLY for classification loss during training
            phase_type: (B,) GT phase labels (0-based), ONLY for classification loss during training

        Returns:
            a_hat: predicted actions
            is_pad_hat: predicted padding
            [mu, logvar]: latent distribution parameters
            strategy_logits: (B, num_strategies) strategy classification logits
            phase_logits: (B, num_phases) phase classification logits
        """
        is_training = actions is not None
        bs = qpos.shape[0]

        # Initialize auxiliary outputs
        strategy_logits = None
        phase_logits = None

        if self.condition_encoder and self.backbones is not None:
            # === Conditional Encoder Path with Strategy/Phase ===

            # Step 1: Encode visual and proprio features (shared for Predictor and condition injection)
            proprio_embed, img_embed, img_pooled = self._encode_visual_features(qpos, image)
            N_img = img_embed.shape[0]

            # Step 2: Use StrategyPhasePredictor to predict strategy/phase (both training and inference)
            # This is the END-TO-END approach: model learns to predict conditions, then uses them
            strategy_logits, phase_logits = self.strategy_phase_predictor(img_pooled, proprio_embed)

            # Step 3: Get predicted strategy/phase indices (used for condition injection)
            # Note: We use argmax predictions, NOT GT labels
            strategy_pred = strategy_logits.argmax(dim=-1)  # (B,)
            phase_pred = phase_logits.argmax(dim=-1)        # (B,)

            # Step 4: Build condition tokens using PREDICTED values
            proprio_embed_seq = proprio_embed.unsqueeze(0)  # (1, B, hidden_dim)
            strategy_token, phase_token, condition_pooled = self._build_condition_tokens(
                proprio_embed, img_pooled, strategy_pred, phase_pred
            )

            if is_training:
                # --- Training: Encode [CLS, qpos, strategy, phase, img, actions] -> q(z|a,c) ---
                # NOTE: strategy_token and phase_token are from PREDICTOR predictions (Step 3-4 above)
                # GT labels (strategy_type, phase_type) are only used in act_policy.py for classification loss

                # Action embeddings
                action_embed = self.encoder_action_proj(actions)  # (B, seq, hidden_dim)
                action_embed = action_embed.permute(1, 0, 2)  # (seq, B, hidden_dim)
                N_act = action_embed.shape[0]

                # CLS token
                cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(1, bs, 1)  # (1, B, hidden_dim)

                # Concatenate: [CLS, qpos, strategy, phase, img, actions]
                # strategy_token and phase_token are from Predictor predictions
                encoder_input = torch.cat([
                    cls_embed,          # (1, B, hidden_dim)   - position 0
                    proprio_embed_seq,  # (1, B, hidden_dim)   - position 1
                    strategy_token,     # (1, B, hidden_dim)   - position 2 (from Predictor)
                    phase_token,        # (1, B, hidden_dim)   - position 3 (from Predictor)
                    img_embed,          # (N_img, B, hidden_dim) - positions 4 to 4+N_img-1
                    action_embed        # (N_act, B, hidden_dim) - positions 4+N_img to end
                ], dim=0)

                total_len = encoder_input.shape[0]

                # Position encoding
                pos_embed = self.encoder_pos_table[:, :total_len, :].clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (total_len, 1, hidden_dim)

                # Type embeddings (optional)
                # Type IDs: 0=CLS, 1=qpos, 2=strategy, 3=phase, 4=image, 5=action
                if self.encoder_use_type_embed:
                    type_ids = torch.cat([
                        torch.zeros(1, bs, dtype=torch.long, device=qpos.device),           # CLS: type 0
                        torch.ones(1, bs, dtype=torch.long, device=qpos.device),            # qpos: type 1
                        torch.full((1, bs), 2, dtype=torch.long, device=qpos.device),       # strategy: type 2 [NEW]
                        torch.full((1, bs), 3, dtype=torch.long, device=qpos.device),       # phase: type 3 [NEW]
                        torch.full((N_img, bs), 4, dtype=torch.long, device=qpos.device),   # image: type 4
                        torch.full((N_act, bs), 5, dtype=torch.long, device=qpos.device),   # action: type 5
                    ], dim=0)
                    type_embed = self.encoder_type_embed(type_ids)
                    encoder_input = encoder_input + type_embed

                # Padding mask: CLS, qpos, strategy, phase, img are not padded; only actions can be padded
                # Extended from 1+1+N_img to 1+1+1+1+N_img (added strategy and phase)
                condition_mask = torch.full((bs, 1 + 1 + 1 + 1 + N_img), False, device=qpos.device)
                encoder_is_pad = torch.cat([condition_mask, is_pad], dim=1)

                # Encode
                encoder_output = self.encoder(
                    encoder_input,
                    pos=pos_embed,
                    src_key_padding_mask=encoder_is_pad
                )
                cls_output = encoder_output[0]  # (B, hidden_dim) - CLS token output

                # Project to latent space -> q(z|a,c)
                latent_info = self.latent_proj(cls_output)
                mu = latent_info[:, :self.latent_dim]
                logvar = latent_info[:, self.latent_dim:]

                # Also compute prior p(z|c) for KL divergence
                prior_info = self.prior_proj(condition_pooled)
                mu_prior = prior_info[:, :self.latent_dim]
                logvar_prior = prior_info[:, self.latent_dim:]

                # Sample from posterior q(z|a,c)
                latent_sample = reparametrize(mu, logvar)
                latent_input = self.latent_out_proj(latent_sample)

                # Store prior for KL computation (will be used in loss)
                self._mu_prior = mu_prior
                self._logvar_prior = logvar_prior

            else:
                # --- Inference: Sample from prior p(z|c) ---

                # Compute conditional prior
                prior_info = self.prior_proj(condition_pooled)
                mu_prior = prior_info[:, :self.latent_dim]
                logvar_prior = prior_info[:, self.latent_dim:]

                # Sample from prior (or use mean for deterministic)
                latent_sample = reparametrize(mu_prior, logvar_prior)
                latent_input = self.latent_out_proj(latent_sample)

                # Return prior parameters
                mu = mu_prior
                logvar = logvar_prior

        else:
            # === Original ACT Path (action-only encoder) ===

            if is_training:
                # Encode action sequence
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

        # === Decoder (same for both paths) ===
        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # [NEW] Prepare query condition for decoder injection
            # Inject strategy/phase information directly into action queries
            if self.condition_encoder:
                # Get strategy and phase features from predicted indices
                strategy_feat = self.strategy_embed(strategy_pred)  # (B, hidden_dim)
                phase_feat = self.phase_embed(phase_pred)          # (B, hidden_dim)
                # Project combined features to query condition
                query_condition = self.query_condition_proj(
                    torch.cat([strategy_feat, phase_feat], dim=-1)
                )  # (B, hidden_dim)
            else:
                query_condition = None

            hs = self.transformer(src, None, self.query_embed.weight, pos,
                                  latent_input, proprio_input, self.additional_pos_embed.weight,
                                  query_condition=query_condition)[0]
        else:
            qpos_proj = self.input_proj_robot_state(qpos)
            env_state_proj = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos_proj, env_state_proj], axis=1)
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        # Return with auxiliary classification outputs
        return a_hat, is_pad_hat, [mu, logvar], strategy_logits, phase_logits

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

    # Get conditional encoder settings
    condition_encoder = getattr(args, 'condition_encoder', False)
    encoder_use_type_embed = getattr(args, 'encoder_use_type_embed', True)

    # Get strategy/phase settings (new for ACT_BC_ENC_STRATEGY)
    num_strategies = getattr(args, 'num_strategies', 3)
    num_phases = getattr(args, 'num_phases', 4)

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
        num_strategies=num_strategies,
        num_phases=num_phases,
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
