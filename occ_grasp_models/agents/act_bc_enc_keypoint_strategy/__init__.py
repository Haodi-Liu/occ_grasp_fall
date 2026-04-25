"""
ACT_BC_ENC_KEYPOINT_STRATEGY: Fusion Agent with Keypoint + Strategy Conditioning

This module fuses ACT_BC_ENC_KEYPOINT and ACT_BC_ENC_STRATEGY:
1. KeypointPosePredictor for contact/grasp/affordance 2D localization (Memory injection)
2. StrategyPhasePredictor for strategy/phase classification (Query injection)
3. 9 type embeddings in conditional encoder
4. Dual decoder injection: condition_tokens + query_condition
5. All loss components: action L1, KL, 2D projection, visibility, strategy, phase
"""

import agents.act_bc_enc_keypoint_strategy.launch_utils
