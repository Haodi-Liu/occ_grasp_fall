# KeypointPosePredictor Module - Path B Version
# Reference: KEYPOINT_POSE_INJECTION_PLAN.md Section 7
#
# *** Path B Design ***
# - REMOVED: 3D pose prediction heads (position_head, rotation_head)
# - NEW: 2D coordinate prediction head (coord_2d_head)
# - NEW: Returns kp_queries features for condition injection
#
# Uses Transformer Decoder architecture with:
# - Self-Attention: keypoint interaction (contact-grasp-affordance relationships)
# - Cross-Attention: visual localization from img_tokens
# - Pre-LN architecture for stable training

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KeypointPosePredictor(nn.Module):
    """
    Keypoint Localization Module - Path B Version

    *** Key Design Changes (Path B) ***
    - Removed 3D pose prediction (position_head, rotation_head)
    - Only predicts 2D pixel coordinates for supervision
    - Returns kp_queries features directly for condition injection
    - 2D supervision guides attention to localize keypoints in image space

    Input:
        visual_feat: [B, D] - pooled visual feature (fallback)
        proprio_feat: [B, D] - proprioception feature (projected to hidden_dim)
        img_tokens: [B, N, D] - image patch tokens (for Cross-Attention)

    Output:
        kp_features: [B, 3, D] - keypoint features for condition injection
        keypoint_2d: Dict[str, Tensor[B, num_cameras, 2]] - 2D pixel coordinates
        affordance_visible_prob: [B, 1] - affordance visibility probability
        affordance_visible_logits: [B, 1] - raw logits for BCE loss
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_keypoints: int = 3,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 num_cameras: int = 2,
                 img_size: int = 256):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_cameras = num_cameras
        self.img_size = img_size

        # ====== Learnable keypoint queries ======
        # Each keypoint has an independent query vector, learning its semantic representation
        self.keypoint_queries = nn.Parameter(torch.randn(num_keypoints, hidden_dim))

        # ====== Proprioception fusion ======
        # Project proprio info and add to each query, providing current state context
        self.proprio_proj = nn.Linear(hidden_dim, hidden_dim)

        # ====== Decoder layers ======
        # Self-Attention: keypoint interaction
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        # Cross-Attention: localize from visual features
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        # FFN: feature transformation
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # ====== LayerNorm (Pre-LN architecture) ======
        self.norm_self = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        # Pre-LN required: LayerNorm before final output
        self.final_norm = nn.LayerNorm(hidden_dim)

        # ====== [REMOVED] 3D prediction heads ======
        # self.position_head = ...  # REMOVED
        # self.rotation_head = ...  # REMOVED

        # ====== [NEW] 2D coordinate prediction head ======
        # Predicts normalized [0, 1] 2D coordinates for each camera
        self.coord_2d_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_cameras * 2),  # (u, v) for each camera
            nn.Sigmoid()  # Normalize to [0, 1]
        )

        # ====== [KEEP] Visibility prediction (for affordance) ======
        self.visibility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Weight initialization"""
        # Xavier initialize keypoint queries to ensure initial diversity
        nn.init.xavier_uniform_(self.keypoint_queries.unsqueeze(0))

    def forward(self, visual_feat, proprio_feat, img_tokens=None):
        """
        Forward pass - Path B Version

        Args:
            visual_feat: [B, hidden_dim] - pooled visual feature (fallback)
            proprio_feat: [B, hidden_dim] - proprioception feature (projected to hidden_dim)
            img_tokens: [B, N, hidden_dim] - image patch tokens (for Cross-Attention)
                        If provided, use full spatial features; otherwise use pooled features

        Returns:
            kp_features: [B, 3, hidden_dim] - keypoint features for condition injection
            keypoint_2d: Dict[str, Tensor[B, num_cameras, 2]] - 2D pixel coordinates
            affordance_visible_prob: [B, 1] - affordance visibility probability
            affordance_visible_logits: [B, 1] - raw logits for BCE loss
        """
        B = visual_feat.shape[0]

        # ====== 1. Initialize keypoint queries ======
        # [num_keypoints, D] -> [B, num_keypoints, D]
        kp_queries = self.keypoint_queries.unsqueeze(0).expand(B, -1, -1)

        # ====== 2. Fuse proprioception info ======
        # Add proprio info to each keypoint query, providing current state context
        proprio_cond = self.proprio_proj(proprio_feat).unsqueeze(1)  # [B, 1, D]
        kp_queries = kp_queries + proprio_cond  # broadcast to all queries

        # ====== 3. Prepare visual KV ======
        # Prefer img_tokens (full spatial info), otherwise use pooled features
        if img_tokens is not None:
            visual_kv = img_tokens  # [B, N, D]
        else:
            visual_kv = visual_feat.unsqueeze(1)  # [B, 1, D]

        # ====== 4. Multi-layer Decoder processing ======
        for i in range(self.num_layers):
            # 4.1 Self-Attention: keypoint interaction (Pre-LN)
            residual = kp_queries
            kp_queries = self.norm_self[i](kp_queries)
            sa_out, _ = self.self_attn_layers[i](kp_queries, kp_queries, kp_queries)
            kp_queries = residual + sa_out

            # 4.2 Cross-Attention: localize from visual features (Pre-LN)
            residual = kp_queries
            kp_queries = self.norm_cross[i](kp_queries)
            ca_out, _ = self.cross_attn_layers[i](kp_queries, visual_kv, visual_kv)
            kp_queries = residual + ca_out

            # 4.3 FFN: feature transformation (Pre-LN)
            residual = kp_queries
            kp_queries = self.norm_ffn[i](kp_queries)
            kp_queries = residual + self.ffn_layers[i](kp_queries)

        # ====== 5. Final LayerNorm (Pre-LN required) ======
        kp_queries = self.final_norm(kp_queries)  # [B, 3, D]

        # ====== 6. [NEW] Predict 2D coordinates ======
        keypoint_2d = {}
        keypoint_names = ['contact', 'grasp', 'affordance']

        for idx, kp_name in enumerate(keypoint_names):
            kp_feat = kp_queries[:, idx, :]  # [B, D]

            # Predict normalized 2D coordinates [0, 1]
            coords_norm = self.coord_2d_head(kp_feat)  # [B, num_cameras * 2]
            coords_norm = coords_norm.view(B, self.num_cameras, 2)  # [B, num_cameras, 2]

            # Scale to pixel coordinates
            coords_pixel = coords_norm * self.img_size
            keypoint_2d[kp_name] = coords_pixel

        # ====== 7. [KEEP] Visibility prediction (affordance) ======
        affordance_feat = kp_queries[:, 2, :]  # affordance is the 3rd
        # Return raw logits for numerically stable BCE_with_logits loss computation
        affordance_visible_logits = self.visibility_head(affordance_feat)  # [B, 1]
        # Also compute probability for condition injection (clamped for safety)
        affordance_visible_prob = torch.sigmoid(affordance_visible_logits).clamp(1e-6, 1 - 1e-6)

        # ====== 8. [NEW] Return kp_queries as features for condition injection ======
        # kp_queries already contains spatial localization info from Cross-Attention
        # This is the core of Path B: use features directly instead of 3D pose projection
        return kp_queries, keypoint_2d, affordance_visible_prob, affordance_visible_logits
