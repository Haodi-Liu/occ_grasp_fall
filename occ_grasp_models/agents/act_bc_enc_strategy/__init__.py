"""
ACT_BC_ENC_STRATEGY: Strategy and Phase Conditioned Action Chunking Transformer

This module extends ACT_BC_ENC with:
1. Strategy type conditioning (EdgeHang/WallLever/PressTilt)
2. Phase type conditioning (PreManipulation/Grasp/ClearPath/Lift)
3. Auxiliary classification losses for strategy and phase prediction
4. StrategyPhasePredictor module for context understanding
"""

import agents.act_bc_enc_strategy.launch_utils
