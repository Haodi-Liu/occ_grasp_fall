# data_sample

This directory is a lightweight demonstration subset intended for repository readers and LLM-based code inspection.

## Included tasks

- `bimanual_edge_phone`
- `bimanual_pick_fork`
- `bimanual_pick_plate`
- `bimanual_pivot_phone`

Each task currently includes:

- `all_variations/episodes/episode0/`
- 18 image subdirectories with PNG frames
- original `low_dim_obs.pkl`
- original metadata `pkl` files
- original `demo_front.mp4`

## Compression policy

To reduce repository size while preserving the data schema:

- every PNG image directory is truncated to the first 5 frames
- `demo_front.mp4` is kept unchanged
- all `pkl` files are kept unchanged
- empty point-cloud directories are kept as-is

## Purpose

This sample is meant to show:

- episode directory layout
- camera naming conventions
- RGB, depth, and mask file naming
- the presence of original low-dimensional and metadata files

It is not intended to be a training dataset or a full evaluation dataset.
