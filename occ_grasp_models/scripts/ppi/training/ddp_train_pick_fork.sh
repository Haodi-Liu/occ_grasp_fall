#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

ngpus=2
export WANDB__SERVICE_WAIT=600
export WANDB_API_KEY="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"
if [ "${WANDB_API_KEY}" = "REPLACE_WITH_YOUR_WANDB_API_KEY" ]; then
    echo "Please set WANDB_API_KEY before running this script." >&2
    exit 1
fi
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1

# train_ppi_ddp.py divides these global batch sizes by visible GPU count.
torchrun --nnodes 1 --nproc_per_node $ngpus --master_port 10004 train_ppi_ddp.py \
    task='pick_fork' \
    name='train_ppi_ddp' \
    addition_info='20260405_baseline' \
    wandb_name='ppi_pick_fork' \
    logging.mode=online \
    n_obs_steps=1 \
    n_action_steps=54 \
    policy.use_lang=true \
    policy.what_condition='ppi' \
    policy.predict_point_flow=true \
    task.dataset.pcd_fps=6144 \
    task.dataset.pcd_type='rgb_pcd_rps6144' \
    task.dataset.point_flow_type='world_ordered_rps200' \
    task.dataset.kp_num=10 \
    task.dataset.prediction_type='keyframe_continuous' \
    task.dataset.stats_filepath='data/training_processed/norm_stats/norm_stats_bimanual_pick_fork_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth' \
    horizon_keyframe=4 \
    horizon_continuous=50 \
    dataloader.batch_size=32 \
    val_dataloader.batch_size=32 \
    training.num_epochs=350 \
    seed=0 \
    training.resume=true \
    training.checkpoint_every=20
