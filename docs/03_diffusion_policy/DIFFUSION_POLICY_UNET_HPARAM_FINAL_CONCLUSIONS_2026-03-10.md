# DIFFUSION_POLICY（UNet / Transformer）超参数最终结论（按 2026-03-10 最新代码复核）

> 复核时间：2026-03-10  
> 适用范围：`occ_grasp_models` 中 `DIFFUSION_POLICY`（`model_type: unet` 与 `model_type: transformer`）  
> 复核依据：当前代码已落地 `docs/research/DIFFUSION_POLICY_TRAINING_SPEEDUP_PLAN.md` 中 P1-2（AMP/TF32）与 P2-1（worker 侧 build_batch）

## 1. 本次复核后的结论总览

1. 原文档中 UNet 主结构与大部分训练超参结论仍成立。  
2. 两条路线都必须补齐两组“已生效”性能配置：
   - `use_amp / amp_dtype / enable_tf32`
   - `move_batch_build_to_worker / batch_builder_*`
3. 当 `move_batch_build_to_worker=True` 时，DIFFUSION_POLICY 分支并行度主要由 `batch_builder_workers` 控制，不再主要看 `framework.num_workers`。  
4. `index_cache_size` 推荐值更新为 `32`（与当前方法默认一致）。  
5. Transformer 需单独强调两点：
   - 必须 `obs_as_cond=True` 且 `time_as_cond=True`
   - `model_type=transformer` 时优化器走 `policy.get_optimizer(...)`，`method.weight_decay` 不作为主 weight decay 旋钮

## 2. 已最终确认的代码事实

1. Policy 本体不直接消费 `use_ema/lr_scheduler/lr_warmup_steps/ema_*`；这些训练态键由 `agent.py` 生效。  
2. `agent.py` 训练路径已接入 AMP、fp16 GradScaler、bf16 设备检查与 TF32。  
3. `run_seed_fn.py + ReplaySampleToBatchTransform + PyTorchReplayBuffer` 已接入 worker-side batch build。  
4. DDPM scheduler 在 `launch_utils.py` 仍为固定配置：
   - `num_train_timesteps=100`
   - `beta_schedule=squaredcos_cap_v2`
   - `prediction_type=epsilon`
5. Transformer 路线在 `launch_utils.py` 中有强约束：
   - `obs_as_cond=True`
   - `time_as_cond=True`

## 3. UNet 路线推荐（`model_type: unet`）

### 3.1 通用参数

| 参数 | 推荐值 |
|---|---|
| `action_dim` | `18` |
| `horizon` | `16` |
| `n_obs_steps` | `2` |
| `n_action_steps` | `8` |
| `obs_as_global_cond` | `True` |
| `lr` | `1e-4` |
| `weight_decay` | `1e-6` |
| `optimizer_betas` | `[0.95, 0.999]` |
| `grad_clip` | `0` |
| `lr_scheduler` | `cosine` |
| `lr_warmup_steps` | `500` |
| `use_ema` | `True` |
| `ema_update_after_step` | `0` |
| `ema_inv_gamma` | `1.0` |
| `ema_power` | `0.75` |
| `ema_min_value` | `0.0` |
| `ema_max_value` | `0.9999` |
| `num_inference_steps` | `100` |
| `diffusion_step_embed_dim` | `128` |
| `down_dims` | `[512, 1024, 2048]` |
| `kernel_size` | `5` |
| `n_groups` | `8` |
| `cond_predict_scale` | `True` |
| `share_rgb_model` | `False` |
| `use_group_norm` | `True` |
| `resize_shape` | `null` |
| `crop_shape` | `[76, 76]` |
| `random_crop` | `True` |
| `imagenet_norm` | `True` |
| `low_dim_size` | `18` |
| `index_cache_size` | `32` |
| `image_cache_size` | `0` |
| `index_seed` | `null` |

### 3.2 性能配置（UNet 与 Transformer 共用）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `use_amp` | `True` | CUDA 下启用 autocast |
| `amp_dtype` | `bf16` | A800 优先；不支持 bf16 时切到 `fp16` |
| `enable_tf32` | `True` | 提升 matmul/cuDNN 吞吐 |
| `move_batch_build_to_worker` | `True` | worker 内做 `build_batch` |
| `batch_builder_workers` | `4` | DIFFUSION_POLICY 主并行度旋钮 |
| `batch_builder_pin_memory` | `True` | host->device 更快 |
| `batch_builder_persistent_workers` | `True` | 减少 worker 反复重建 |
| `batch_builder_prefetch_factor` | `2` | 预取深度 |

### 3.3 完整配置 A：UNet + ResNet18

`conf/method/DIFFUSION_POLICY.yaml`（关键段）：

```yaml
model_type: 'unet'
action_dim: 18
horizon: 16
n_obs_steps: 2
n_action_steps: 8
obs_as_global_cond: True

lr: 1e-4
weight_decay: 0.000001
optimizer_betas: [0.95, 0.999]
grad_clip: 0
num_inference_steps: 100

lr_scheduler: cosine
lr_warmup_steps: 500
use_ema: True
ema_update_after_step: 0
ema_inv_gamma: 1.0
ema_power: 0.75
ema_min_value: 0.0
ema_max_value: 0.9999

use_amp: True
amp_dtype: bf16
enable_tf32: True

move_batch_build_to_worker: True
batch_builder_workers: 4
batch_builder_pin_memory: True
batch_builder_persistent_workers: True
batch_builder_prefetch_factor: 2

rgb_backbone: resnet18
share_rgb_model: False
use_group_norm: True
resize_shape: null
crop_shape: [76, 76]
random_crop: True
imagenet_norm: True

diffusion_step_embed_dim: 128
down_dims: [512, 1024, 2048]
kernel_size: 5
n_groups: 8
cond_predict_scale: True

low_dim_size: 18
index_cache_size: 32
image_cache_size: 0
index_seed: null
```

`conf/config.yaml` 联动建议：

```yaml
replay:
  batch_size: 64
framework:
  num_workers: 0
  training_iterations: 5496000
```

### 3.4 完整配置 B：UNet + ResNet50

仅 backbone 差异：

```yaml
rgb_backbone: resnet50
```

`conf/config.yaml` 联动建议：

```yaml
replay:
  batch_size: 32
framework:
  num_workers: 0
  training_iterations: 10992000
```

若 `batch_size=32` 仍 OOM，可退到：

```yaml
replay:
  batch_size: 16
framework:
  training_iterations: 21976000
```

## 4. Transformer 专项复核（`model_type: transformer`）

## 4.1 必须满足的约束

| 参数 | 要求 |
|---|---|
| `obs_as_cond` | 必须 `True` |
| `time_as_cond` | 必须 `True` |
| `pred_action_steps_only` | 推荐 `False`（与当前序列训练口径一致） |

## 4.2 生效/忽略关系（关键，跑实验前请先对齐）

| 参数 | transformer 下状态 | 说明 |
|---|---|---|
| `weight_decay` | 非主旋钮 | 该路线优化器走 `policy.get_optimizer(...)` 分组，主衰减由下面两项控制 |
| `transformer_weight_decay` | 生效 | 控制 transformer 参数组 decay（建议 `1e-3`） |
| `obs_encoder_weight_decay` | 生效 | 控制视觉编码器参数组 decay（建议 `1e-6`） |
| `optimizer_betas` | 生效 | 会传入 transformer 优化器；建议覆盖为 `[0.9, 0.95]` |
| `lr_scheduler/lr_warmup_steps` | 生效 | 在 `agent.py` 中统一由 diffusers scheduler 构建 |
| `use_ema` | 生效 | 与 UNet 同逻辑 |
| `use_amp/amp_dtype/enable_tf32` | 生效 | 与 UNet 同逻辑 |
| `move_batch_build_to_worker/batch_builder_*` | 生效 | 与 UNet 同逻辑 |
| `obs_as_global_cond` | 不使用 | UNet 专用键 |

## 4.3 Transformer 推荐参数

| 参数 | 推荐值 |
|---|---|
| `model_type` | `transformer` |
| `action_dim` | `18` |
| `horizon` | `16` |
| `n_obs_steps` | `2` |
| `n_action_steps` | `8` |
| `obs_as_cond` | `True` |
| `time_as_cond` | `True` |
| `pred_action_steps_only` | `False` |
| `lr` | `1e-4` |
| `optimizer_betas` | `[0.9, 0.95]` |
| `transformer_weight_decay` | `1e-3` |
| `obs_encoder_weight_decay` | `1e-6` |
| `grad_clip` | `0` |
| `lr_scheduler` | `cosine` |
| `lr_warmup_steps` | `1000` |
| `use_ema` | `True` |
| `num_inference_steps` | `100` |
| `n_layer` | `8` |
| `n_cond_layers` | `0` |
| `n_head` | `4` |
| `n_emb` | `256` |
| `p_drop_emb` | `0.0` |
| `p_drop_attn` | `0.3` |
| `causal_attn` | `True` |
| `share_rgb_model` | `False` |
| `use_group_norm` | `True` |
| `resize_shape` | `null` |
| `crop_shape` | `[76, 76]` |
| `random_crop` | `True` |
| `imagenet_norm` | `True` |
| `low_dim_size` | `18` |
| `index_cache_size` | `32` |
| `image_cache_size` | `0` |
| `index_seed` | `null` |

> 注：`lr_warmup_steps=1000` 来自上游 transformer 训练配置口径；当前仓库代码同样支持该值。

## 4.4 完整配置 C：Transformer + ResNet18

`conf/method/DIFFUSION_POLICY.yaml`（关键段）：

```yaml
model_type: 'transformer'
action_dim: 18
horizon: 16
n_obs_steps: 2
n_action_steps: 8

obs_as_cond: True
time_as_cond: True
pred_action_steps_only: False

lr: 1e-4
optimizer_betas: [0.9, 0.95]
transformer_weight_decay: 0.001
obs_encoder_weight_decay: 0.000001
grad_clip: 0
num_inference_steps: 100

lr_scheduler: cosine
lr_warmup_steps: 1000
use_ema: True
ema_update_after_step: 0
ema_inv_gamma: 1.0
ema_power: 0.75
ema_min_value: 0.0
ema_max_value: 0.9999

use_amp: True
amp_dtype: bf16
enable_tf32: True

move_batch_build_to_worker: True
batch_builder_workers: 4
batch_builder_pin_memory: True
batch_builder_persistent_workers: True
batch_builder_prefetch_factor: 2

rgb_backbone: resnet18
share_rgb_model: False
use_group_norm: True
resize_shape: null
crop_shape: [76, 76]
random_crop: True
imagenet_norm: True

n_layer: 8
n_cond_layers: 0
n_head: 4
n_emb: 256
p_drop_emb: 0.0
p_drop_attn: 0.3
causal_attn: True

low_dim_size: 18
index_cache_size: 32
image_cache_size: 0
index_seed: null
```

`conf/config.yaml` 联动建议：

```yaml
replay:
  batch_size: 64
framework:
  num_workers: 0
  training_iterations: 5496000
```

## 4.5 完整配置 D：Transformer + ResNet50

仅 backbone 差异：

```yaml
rgb_backbone: resnet50
```

`conf/config.yaml` 联动建议：

```yaml
replay:
  batch_size: 32
framework:
  num_workers: 0
  training_iterations: 10992000
```

若 `batch_size=32` 仍 OOM，可退到：

```yaml
replay:
  batch_size: 16
framework:
  training_iterations: 21976000
```

## 5. DDPM 固定参数（两路线一致）

| 参数 | 值 |
|---|---|
| `num_train_timesteps` | `100` |
| `beta_start` | `0.0001` |
| `beta_end` | `0.02` |
| `beta_schedule` | `squaredcos_cap_v2` |
| `variance_type` | `fixed_small` |
| `clip_sample` | `True` |
| `prediction_type` | `epsilon` |

## 6. `training_iterations` 换算口径

该训练框架中 `framework.training_iterations` 表示“参数更新步数”（不是 epoch）。

沿用换算：

`training_iterations = num_epochs * ceil(total_indexed_transitions / replay.batch_size)`

其中 `total_indexed_transitions` 来自 index replay logical starts 数量。

## 7. 相比旧版文档的关键修订点

1. 补齐 AMP/TF32 参数，并设为默认推荐。  
2. 补齐 worker-side batch build 参数，并明确并行度主旋钮改为 `batch_builder_workers`。  
3. `index_cache_size` 推荐从 `8` 更新为 `32`。  
4. 新增 Transformer 专项“生效/忽略关系”与完整配置，避免直接套用 UNet 口径。  
5. Transformer 路线将 `lr_warmup_steps` 推荐为 `1000`，与上游 transformer 配置口径对齐。

