# Local-only Assets

These paths exist on the development machine but are intentionally excluded from GitHub.

## Repo root on current machine

- `/home/hdliu/occ_grasp_fall`

## Excluded local directories

- `/home/hdliu/occ_grasp_fall/pretrained_models`
- `/home/hdliu/occ_grasp_fall/occ_grasp_models/data`
- `/home/hdliu/occ_grasp_fall/occ_grasp_models/exp_logs`
- `/home/hdliu/occ_grasp_fall/occ_grasp_models/eval_logs`
- `/home/hdliu/occ_grasp_fall/occ_grasp_models/eval_videos`
- `/home/hdliu/occ_grasp_fall/occ_grasp_models/eval_weights`

## Key pretrained assets

- `pretrained_models/groundingdino_swinb_cogcoor.pth`
- `pretrained_models/sam_vit_b_01ec64.pth`
- `pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth`
- `pretrained_models/bert-base-uncased/`

## Included GitHub sample data

`data_sample/` is included in the repository as a lightweight format example:

- four tasks are included
- each task keeps one `episode0`
- each PNG image folder keeps the first 5 frames
- `demo_front.mp4` and all `pkl` files remain unchanged

Use `data_sample/README.md` for the exact sample layout.
