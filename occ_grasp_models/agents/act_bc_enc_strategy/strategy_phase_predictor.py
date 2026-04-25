"""
Strategy and Phase Predictor Module

This module predicts the strategy type and execution phase from visual and
proprioceptive features. Used as an auxiliary task to help the model understand
the current manipulation context.
"""

import torch
import torch.nn as nn


class StrategyPhasePredictor(nn.Module):
    """
    Strategy and Phase Predictor

    Predicts strategy type (EdgeHang/WallLever/PressTilt) and execution phase
    (PreManip/Grasp/ClearPath/Lift) from pooled visual and proprioceptive features.

    Input:
        visual_feat: [B, hidden_dim] - Pooled image features (averaged across cameras)
        proprio_feat: [B, hidden_dim] - Projected qpos features

    Output:
        strategy_logits: [B, num_strategies] - Strategy classification logits
        phase_logits: [B, num_phases] - Phase classification logits
    """

    def __init__(self,
                 hidden_dim: int = 512,
                 num_strategies: int = 3,
                 num_phases: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            hidden_dim: Hidden dimension size (should match model's hidden_dim)
            num_strategies: Number of strategy types (default: 3)
                - 0: EdgeHang (edge overhang grasp)
                - 1: WallLever (wall lever grasp)
                - 2: PressTilt (press tilt grasp)
            num_phases: Number of execution phases (default: 4)
                - 0: PreManipulation (create grasping space)
                - 1: Grasp (grasp the object)
                - 2: ClearPath (move auxiliary arm away)
                - 3: Lift (lift the object)
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        self.num_phases = num_phases

        # Input: visual_feat + proprio_feat
        input_dim = hidden_dim * 2

        # Shared feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Strategy classification head
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_strategies)
        )

        # Phase classification head
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_phases)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, visual_feat: torch.Tensor, proprio_feat: torch.Tensor):
        """
        Forward pass

        Args:
            visual_feat: [B, hidden_dim] - Pooled image features
            proprio_feat: [B, hidden_dim] - Proprioception features

        Returns:
            strategy_logits: [B, num_strategies] - Strategy prediction logits
            phase_logits: [B, num_phases] - Phase prediction logits
        """
        # Concatenate features
        fused = torch.cat([visual_feat, proprio_feat], dim=-1)  # [B, hidden_dim*2]

        # Shared fusion
        fused = self.fusion(fused)  # [B, hidden_dim]

        # Predict strategy and phase
        strategy_logits = self.strategy_head(fused)  # [B, num_strategies]
        phase_logits = self.phase_head(fused)        # [B, num_phases]

        return strategy_logits, phase_logits

    def predict(self, visual_feat: torch.Tensor, proprio_feat: torch.Tensor):
        """
        Predict strategy and phase IDs (argmax of logits)

        Args:
            visual_feat: [B, hidden_dim] - Pooled image features
            proprio_feat: [B, hidden_dim] - Proprioception features

        Returns:
            strategy_id: [B] - Predicted strategy IDs (0, 1, or 2)
            phase_id: [B] - Predicted phase IDs (0, 1, 2, or 3)
        """
        strategy_logits, phase_logits = self.forward(visual_feat, proprio_feat)
        strategy_id = strategy_logits.argmax(dim=-1)
        phase_id = phase_logits.argmax(dim=-1)
        return strategy_id, phase_id
