"""
ACT_BC_ENC_KEYPOINT: ACT with Conditional Encoder and Keypoint Pose Injection

This module extends ACT_BC_ENC with:
1. KeypointPosePredictor for contact/grasp/affordance pose prediction
2. End-to-end design: predictor outputs used for condition injection
3. Affordance visibility handling with learnable embeddings
4. Extended encoder sequence with 7 type IDs
5. Geodesic rotation loss for quaternion supervision
"""

import agents.act_bc_enc_keypoint.launch_utils
