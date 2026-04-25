# occ_grasp_fall

Monorepo for occupancy-based grasp research code and locally modified upstream dependencies.

## Main directories

- `docs/`: project documentation and research notes intended to be included on GitHub.
- `occ_grasp_models/`: main research and training code.
- `repos/`: locally modified upstream codebases used by this project.
- `data_sample/`: lightweight sample dataset for code reading and format inspection.
- `LOCAL_ASSETS.md`: local-only large assets and excluded directories.

## Sample data policy

`data_sample/` is intentionally compressed for GitHub:

- per task: one `episode0`
- image sequences: only the first 5 PNG frames are kept in each image folder
- `low_dim_obs.pkl`, other `pkl` metadata, and `demo_front.mp4`: kept in original form

This keeps the repository readable for GPT-style code inspection while avoiding multi-GB data upload.

## Excluded local assets

The following directories stay on the local machine and are excluded from Git:

- `pretrained_models/`
- `occ_grasp_models/data/`
- `occ_grasp_models/exp_logs/`
- `occ_grasp_models/eval_logs/`
- `occ_grasp_models/eval_videos/`
- `occ_grasp_models/eval_weights/`
