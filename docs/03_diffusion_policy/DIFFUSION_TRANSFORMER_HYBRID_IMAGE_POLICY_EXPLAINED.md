# DiffusionTransformerHybridImagePolicy 详解（框架结构 + 机理 + 训练算法）

> 说明：本文基于 `diffusion_policy/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py` 进行逐层解析，并在关键位置加入**带行号的代码块**（来源脚本与行号已标注）。内容覆盖从整体逻辑到细节技巧（如 receding horizon、条件注入、inpainting 等）。

---

## 1. 整体框架概览（逻辑脉络）

这个策略可以概括为三层：

1. **观测编码层**：使用 robomimic 的 `ObservationEncoder` 把多相机 RGB（+低维）编码成特征。
2. **扩散建模层**：在“动作序列维度”上做扩散，主干为 `TransformerForDiffusion`。
3. **控制执行层**：推理时生成一段动作序列，只执行其中一小段（receding horizon），然后滚动更新。

关键特点：
- **obs_as_cond=True** 时，观测特征作为 Transformer 的条件输入，不进入轨迹；
- **obs_as_cond=False** 时，观测特征拼入轨迹，并通过 inpainting 固定；
- 训练/推理均使用 DDPM 的加噪/去噪机制。

---

## 2. 观测编码与 robomimic 依赖

### 2.1 shape_meta -> robomimic obs_config

模型首先读取 `shape_meta`，构建 robomimic 的观测配置（决定哪些键是 RGB、哪些是 low_dim）：

```python
# diffusion_transformer_hybrid_image_policy.py:58-80
action_shape = shape_meta['action']['shape']
assert len(action_shape) == 1
action_dim = action_shape[0]  # Da
obs_shape_meta = shape_meta['obs']
obs_config = {
    'low_dim': [],
    'rgb': [],
    'depth': [],
    'scan': []
}
...
for key, attr in obs_shape_meta.items():
    type = attr.get('type', 'low_dim')
    if type == 'rgb':
        obs_config['rgb'].append(key)
    elif type == 'low_dim':
        obs_config['low_dim'].append(key)
    else:
        raise RuntimeError(...)
```

**直观理解：**
- 这里把 `shape_meta` 转换成 robomimic 能识别的配置格式。
- 目标只是**复用 robomimic 的图像编码器**。

### 2.2 robomimic 模板配置 + 裁剪控制

```python
# diffusion_transformer_hybrid_image_policy.py:82-104
config = get_robomimic_config(
    algo_name='bc_rnn',
    hdf5_type='image',
    task_name='square',
    dataset_type='ph')

with config.unlocked():
    config.observation.modalities.obs = obs_config
    if crop_shape is None:
        ... # 关闭随机裁剪
    else:
        ... # 设置 CropRandomizer 的 crop_height / crop_width
```

**关键点：**
- 这里**不依赖真实数据集**，只是借用 robomimic 的默认配置模板。
- `crop_shape` 用于控制随机裁剪大小，提升泛化。

### 2.3 复用 robomimic 的 ObservationEncoder

```python
# diffusion_transformer_hybrid_image_policy.py:108-118
policy: PolicyAlgo = algo_factory(
    algo_name=config.algo_name,
    config=config,
    obs_key_shapes=obs_key_shapes,
    ac_dim=action_dim,
    device='cpu',
)
obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
```

**直观理解：**
- 这里用 robomimic 的 `algo_factory` 实例化一个算法对象；
- 只取出其中的 `obs_encoder` 来使用（不会训练 robomimic 自身的 policy head）。

### 2.4 小技巧：GroupNorm 与固定裁剪

```python
# diffusion_transformer_hybrid_image_policy.py:119-143
if obs_encoder_group_norm:
    replace_submodules(... BatchNorm2d -> GroupNorm ...)

if eval_fixed_crop:
    replace_submodules(... CropRandomizer -> Fixed CropRandomizer ...)
```

**意义：**
- **GroupNorm**：小 batch 下比 BatchNorm 更稳定。
- **eval_fixed_crop**：评估时固定裁剪位置，减少随机性带来的波动。

---

## 3. Transformer 扩散主干

### 3.1 输入/输出/条件维度

```python
# diffusion_transformer_hybrid_image_policy.py:145-166
obs_feature_dim = obs_encoder.output_shape()[0]
input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
output_dim = input_dim
cond_dim = obs_feature_dim if obs_as_cond else 0

model = TransformerForDiffusion(...)
```

**解释：**
- **obs_as_cond=True**：
  - Transformer 只建模动作序列；
  - 观测特征作为条件序列输入（cond_dim > 0）。
- **obs_as_cond=False**：
  - 动作与观测特征拼在一起做扩散；
  - cond_dim = 0，观测通过 inpainting 固定。

### 3.2 保存关键配置

```python
# diffusion_transformer_hybrid_image_policy.py:168-186
self.obs_encoder = obs_encoder
self.model = model
self.noise_scheduler = noise_scheduler
self.mask_generator = LowdimMaskGenerator(...)
self.normalizer = LinearNormalizer()
...
```

**要点：**
- `mask_generator` 用于 inpainting 的掩码生成。
- `normalizer` 用于训练/推理一致的尺度。

---

## 4. 采样机制（conditional_sample）

```python
# diffusion_transformer_hybrid_image_policy.py:193-230
trajectory = torch.randn(size=condition_data.shape, ...)
scheduler.set_timesteps(self.num_inference_steps)

for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]
    model_output = model(trajectory, t, cond)
    trajectory = scheduler.step(model_output, t, trajectory, ...).prev_sample

trajectory[condition_mask] = condition_data[condition_mask]
```

**关键机制：**
- 起点是纯噪声动作序列；
- 每一步先把条件维度固定（inpainting），再预测噪声残差；
- 最终得到完整动作序列。

---

## 5. 推理流程（predict_action）详解

### 5.1 归一化与维度准备

```python
# diffusion_transformer_hybrid_image_policy.py:238-246
nobs = self.normalizer.normalize(obs_dict)
value = next(iter(nobs.values()))
B, To = value.shape[:2]
T = self.horizon
Da = self.action_dim
Do = self.obs_feature_dim
To = self.n_obs_steps
```

### 5.2 条件构造（obs_as_cond 分支）

#### 方案 A：观测特征作为条件输入（obs_as_cond=True）

```python
# diffusion_transformer_hybrid_image_policy.py:256-266
this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
nobs_features = self.obs_encoder(this_nobs)
cond = nobs_features.reshape(B, To, -1)
shape = (B, T, Da)  # or (B, n_action_steps, Da)
cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
```

**理解：**
- 观测特征被当成条件序列输入 Transformer；
- 扩散对象只包含动作序列；
- cond_mask 全 False，表示没有固定维度。

#### 方案 B：观测特征写入轨迹（obs_as_cond=False）

```python
# diffusion_transformer_hybrid_image_policy.py:268-277
nobs_features = self.obs_encoder(...).reshape(B, To, -1)
shape = (B, T, Da+Do)
cond_data = torch.zeros(size=shape, ...)
cond_data[:,:To,Da:] = nobs_features
cond_mask[:,:To,Da:] = True
```

**理解：**
- obs 特征写入轨迹，并通过 mask 固定；
- Transformer 只看到“动作+观测拼接序列”，但观测部分不被噪声破坏。

### 5.3 采样与 receding horizon

```python
# diffusion_transformer_hybrid_image_policy.py:279-296
nsample = self.conditional_sample(...)
action_pred = self.normalizer['action'].unnormalize(nsample[...,:Da])

if self.pred_action_steps_only:
    action = action_pred
else:
    start = To - 1
    end = start + self.n_action_steps
    action = action_pred[:,start:end]
```

**receding horizon 核心点：**
- 生成完整动作序列，但只执行一段窗口。
- 这里窗口从 `To-1` 开始（比 UNet 版本偏移 1），通常表示“当前时刻动作”。

---

## 6. 训练算法（compute_loss）

### 6.1 归一化

```python
# diffusion_transformer_hybrid_image_policy.py:326-334
nobs = self.normalizer.normalize(batch['obs'])
nactions = self.normalizer['action'].normalize(batch['action'])
...
```

### 6.2 条件与轨迹构造

```python
# diffusion_transformer_hybrid_image_policy.py:335-357
if self.obs_as_cond:
    cond = obs_features.reshape(batch_size, To, -1)
    if self.pred_action_steps_only:
        trajectory = nactions[:,start:end]
else:
    obs_features = obs_encoder(...).reshape(batch_size, horizon, -1)
    trajectory = torch.cat([nactions, obs_features], dim=-1).detach()
```

**说明：**
- obs_as_cond=True：观测作为 cond；轨迹为动作序列。
- obs_as_cond=False：观测写入轨迹并 detach，避免梯度反传到编码器。

### 6.3 Inpainting mask

```python
# diffusion_transformer_hybrid_image_policy.py:358-363
if self.pred_action_steps_only:
    condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
else:
    condition_mask = self.mask_generator(trajectory.shape)
```

**说明：**
- 如果只预测动作窗口，则无需 inpainting；
- 否则使用 mask 固定观测部分。

### 6.4 扩散加噪与损失

```python
# diffusion_transformer_hybrid_image_policy.py:364-397
noise = torch.randn(trajectory.shape, device=trajectory.device)
timesteps = torch.randint(0, num_train_timesteps, (bsz,))
noisy_trajectory = scheduler.add_noise(trajectory, noise, timesteps)

noisy_trajectory[condition_mask] = trajectory[condition_mask]
pred = self.model(noisy_trajectory, timesteps, cond)

if prediction_type == 'epsilon':
    target = noise
elif prediction_type == 'sample':
    target = trajectory

loss = F.mse_loss(pred, target, reduction='none')
loss = (loss * (~condition_mask)).mean()
```

**核心训练目标：**
- 预测噪声残差（epsilon）或直接预测 x0（sample）。
- 只在非条件位置计算损失。

---

## 7. 优化器设置（分组权重衰减）

```python
# diffusion_transformer_hybrid_image_policy.py:308-324
optim_groups = self.model.get_optim_groups(weight_decay=...)
optim_groups.append({
    "params": self.obs_encoder.parameters(),
    "weight_decay": obs_encoder_weight_decay
})
optimizer = torch.optim.AdamW(...)
```

**意义：**
- Transformer 主干和 obs_encoder 可使用不同权重衰减；
- 更灵活地控制视觉编码器的正则化强度。

---

## 8. 细节与技巧汇总

- **receding horizon control**：
  - `predict_action()` 中只执行动作序列的一小段（窗口）而非全部。
- **全局条件注入**：
  - obs_as_cond=True 时，观测序列作为条件输入 Transformer。
- **inpainting**：
  - obs_as_cond=False 时，将观测写入轨迹并固定，避免被噪声破坏。
- **稳定性技巧**：
  - BatchNorm -> GroupNorm
  - eval_fixed_crop 固定裁剪
- **prediction_type 灵活性**：
  - 支持 epsilon 或 sample 两种目标。

---

## 9. 总结（一句话版）

DiffusionTransformerHybridImagePolicy 是一种**“观测序列编码 + Transformer 扩散生成动作序列”**的策略：
> 用 robomimic 编码器提取观测特征，在动作序列维度做扩散，并通过 receding horizon 实时滚动执行。

---

如果你需要我再补充：
- `TransformerForDiffusion` 的内部注意力结构
- `LowdimMaskGenerator` 的 mask 生成策略
- obs_as_cond / time_as_cond / causal_attn 的更细节机制

告诉我即可。
