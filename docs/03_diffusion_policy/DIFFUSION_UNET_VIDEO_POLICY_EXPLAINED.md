# DiffusionUnetVideoPolicy 详解（框架结构 + 机理 + 训练算法）

> 说明：本文完全基于仓库中的 `diffusion_policy/diffusion_policy/policy/diffusion_unet_video_policy.py` 进行解读，并结合该文件中引用的核心模块来解释机制与细节。文中代码块都标明**来源脚本与行号**。

---

## 1. 整体框架概览（先看大脉络）

这套策略可以理解为：

1. **多模态观测编码**（多相机 RGB + 低维状态序列）
2. **以动作序列为建模对象的扩散模型**（UNet1D + DDPM 反向采样）
3. **receding horizon control**（只执行预测动作序列的“前几步”）

它的核心思想是：
- **观测只作为条件**，不直接作为扩散对象；
- **扩散对象是动作序列**（长度 `T=horizon`）；
- **动作序列在时间维度上做扩散/去噪**；
- 推理时采样出全序列动作，但只执行其中一小段（`n_action_steps`），然后滚动更新。

在代码层面的入口是一个 `DiffusionUnetVideoPolicy` 类，主要包含：
- 初始化：构建视觉编码器、低维聚合器、UNet、噪声调度器；
- 推理：`predict_action()`；
- 训练：`compute_loss()`。

---

## 2. 模型结构详解

### 2.1 输入与 shape_meta
- `shape_meta` 描述动作和观测的形状与类型。
- 观测分为：
  - **rgb**：每个相机一条视频序列 `(B, T, C, H, W)`
  - **lowdim**：低维状态序列 `(B, T, D)`
- 动作为序列 `(B, T, Da)`。

下面代码展示了 `shape_meta` 解析逻辑：

```python
# diffusion_unet_video_policy.py:48-79
# 解析 action / obs 的维度与类型
action_shape = shape_meta['action']['shape']
assert len(action_shape) == 1
action_dim = action_shape[0]  # Da
obs_shape_meta = shape_meta['obs']

for key, attr in obs_shape_meta.items():
    shape = tuple(attr['shape'])
    type = attr.get('type', 'lowdim')
    if type == 'rgb':
        ...
    elif type == 'lowdim':
        ...
```

**直观理解：**
- `Da` 是单步动作维度（比如 16 维双臂动作）。
- 每个相机都有自己的编码器（可共享权重或复制）。
- 低维观测会统一拼接。

---

### 2.2 视觉编码器（Video -> 全局特征）

每个 RGB 输入走独立编码器（`rgb_net`），最后拼接。

```python
# diffusion_unet_video_policy.py:59-75
if type == 'rgb':
    if len(rgb_nets_map) == 0:
        net = rgb_net
    else:
        net = copy.deepcopy(rgb_net)
    rgb_nets_map[key] = net

    # video input with n_obs_steps timesteps
    shape = (n_obs_steps,) + shape
    output_shape = get_output_shape(shape, net)
    rgb_feature_dims.append(output_shape[0])
```

**要点：**
- 输入是 `(B, To, C, H, W)`
- 输出是 `(B, Do)`（每个相机一个向量）
- 多相机特征拼接为 `rgb_feature`。

---

### 2.3 低维观测的两种用法（关键设计）

这里有两个策略分支，控制参数 `lowdim_as_global_cond`：

1. **作为全局条件（global_cond）**
   - 低维序列通过 `TemporalAggregator` 压缩成全局向量
   - 与视觉特征拼成 `global_cond`

2. **作为轨迹的一部分（inpainting）**
   - 将低维序列拼进轨迹，并固定这些维度不被扩散污染

对应代码：

```python
# diffusion_unet_video_policy.py:88-107
rgb_feature_dim = sum(rgb_feature_dims)
lowdim_input_dim = sum(lowdim_input_dims)
global_cond_dim = rgb_feature_dim
input_dim = action_dim
if lowdim_as_global_cond:
    lowdim_net = TemporalAggregator(...)
    lowdim_feature_shape = get_output_shape((n_obs_steps, lowdim_input_dim), lowdim_net)
    global_cond_dim += lowdim_feature_shape[0]
else:
    input_dim += lowdim_input_dim
```

**直观理解：**
- 低维作为全局条件：扩散轨迹仅包含动作 `(B, T, Da)`
- 低维拼入轨迹：扩散轨迹变成 `(B, T, Da + Dlow)`，并在条件 mask 中锁定观测部分

---

### 2.4 动作扩散 UNet1D

核心扩散模型是 `ConditionalUnet1D`：

```python
# diffusion_unet_video_policy.py:109-118
model = ConditionalUnet1D(
    input_dim=input_dim,
    local_cond_dim=None,
    global_cond_dim=global_cond_dim,
    diffusion_step_embed_dim=diffusion_step_embed_dim,
    down_dims=down_dims,
    kernel_size=kernel_size,
    n_groups=n_groups,
    cond_predict_scale=cond_predict_scale
)
```

**关键点：**
- 输入是一个动作序列 `(B, T, D)`，通过 UNet 在时间维度上做卷积。
- `global_cond` 通过 FiLM 或类似机制注入到 UNet 中（见 `ConditionalUnet1D` 内部逻辑）。
- `diffusion_step_embed_dim` 负责注入时间步编码（扩散 t）。

---

### 2.5 噪声调度器（DDPM）

```python
# diffusion_unet_video_policy.py:121-128
self.noise_scheduler = noise_scheduler
self.mask_generator = LowdimMaskGenerator(...)
```

**意义：**
- `noise_scheduler` 负责加噪与去噪步骤（DDPM 标准流程）。
- `mask_generator` 用于**inpainting**：决定哪些维度被固定不动。

---

## 3. 推理流程（predict_action）详解

### 3.1 总体逻辑
推理流程分成三段：

1. **归一化观测**
2. **构造条件**（视觉 + 低维）
3. **扩散采样动作序列**，再取前 `n_action_steps`

### 3.2 代码拆解（含 receding horizon）

#### (1) 归一化与准备

```python
# diffusion_unet_video_policy.py:191-198
nobs = self.normalizer.normalize(obs_dict)
value = next(iter(nobs.values()))
B, To = value.shape[:2]
T = self.horizon
Da = self.action_dim
To = self.n_obs_steps
```

**直观解释：**
- `normalizer` 保证训练/推理一致尺度。
- `To` 是用作条件的观测长度。
- `T` 是完整预测序列长度。

#### (2) RGB 编码 + 低维拼接

```python
# diffusion_unet_video_policy.py:203-212
rgb_features_map = dict()
for key, net in self.rgb_nets_map.items():
    rgb_features_map[key] = net(nobs[key][:,:self.n_obs_steps])
rgb_feature = torch.cat(list(rgb_features_map.values()), dim=-1)

lowdim_input = torch.cat([nobs[k] for k in self.lowdim_keys], dim=-1)
```

**要点：**
- 每个相机单独编码，再拼接。
- `lowdim_input` 保持为 `(B, To, Dlow)`。

#### (3) 条件构造（两种模式）

```python
# diffusion_unet_video_policy.py:217-231
if self.lowdim_as_global_cond:
    lowdim_feature = self.lowdim_net(lowdim_input[:,:To])
    global_cond = torch.cat([rgb_feature, lowdim_feature], dim=-1)
    cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
else:
    global_cond = rgb_feature
    cond_data = torch.zeros(size=(B, T, Da+self.lowdim_input_dim), device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    cond_data[:,:To,Da:] = lowdim_input[:,:To]
    cond_mask[:,:To,Da:] = True
```

**解释：**
- **全局条件模式**：低维序列经过 `TemporalAggregator` 压缩后拼接到视觉特征中，作为 `global_cond`。
- **inpainting 模式**：低维观测直接写入轨迹，并在 mask 中锁定不被噪声破坏。

#### (4) 采样 + receding horizon control

```python
# diffusion_unet_video_policy.py:232-248
nsample = self.conditional_sample(...)
naction_pred = nsample[...,:Da]
action_pred = self.normalizer['action'].unnormalize(naction_pred)

start = To
end = start + self.n_action_steps
action = action_pred[:,start:end]
```

**Receding horizon 的体现：**
- 采样会生成长度 `T` 的完整动作序列。
- 只取 `[To, To + n_action_steps)` 这一段作为当前执行窗口。
- 下一时刻重新采样（或重用），实现滚动控制。

---

## 4. 采样过程细节（conditional_sample）

```python
# diffusion_unet_video_policy.py:143-182
trajectory = torch.randn(size=condition_data.shape, ...)
scheduler.set_timesteps(self.num_inference_steps)

for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]
    model_output = model(trajectory, t, local_cond=..., global_cond=...)
    trajectory = scheduler.step(...).prev_sample

trajectory[condition_mask] = condition_data[condition_mask]
```

**核心机制：**
- 起点是随机噪声序列。
- 每个扩散步都先**把条件维固定**（inpainting）。
- UNet 预测噪声残差（或 x0），调度器完成去噪。
- 最终得到动作序列。

**细节技巧：**
- **“每步都固定条件”** 是关键稳定性技巧，可避免条件被逐步污染。

---

## 5. 训练算法（compute_loss）详解

训练目标与扩散模型一致：
- 对动作序列加噪，
- 让 UNet 预测噪声（epsilon），
- 用 MSE 作为损失。

### 5.1 数据归一化

```python
# diffusion_unet_video_policy.py:260-264
nobs = self.normalizer.normalize(batch['obs'])
nactions = self.normalizer['action'].normalize(batch['action'])
```

### 5.2 条件构造

```python
# diffusion_unet_video_policy.py:275-287
if self.lowdim_as_global_cond:
    lowdim_feature = self.lowdim_net(lowdim_input[:,:self.n_obs_steps])
    global_cond = torch.cat([rgb_feature, lowdim_feature], dim=-1)
    trajectory = nactions
    cond_data = nactions
else:
    global_cond = rgb_feature
    trajectory = torch.cat([nactions, lowdim_input], dim=-1)
    cond_data = trajectory
```

训练时 `cond_data` 用于**inpainting**：保证条件维不被噪声破坏。

### 5.3 扩散加噪与损失

```python
# diffusion_unet_video_policy.py:292-324
noise = torch.randn(trajectory.shape, device=trajectory.device)
bsz = trajectory.shape[0]
timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,))
noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

loss_mask = ~condition_mask
noisy_trajectory[condition_mask] = cond_data[condition_mask]

pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

if self.kwargs.get('predict_epsilon', True):
    target = noise
else:
    target = trajectory

loss = F.mse_loss(pred, target, reduction='none')
loss = loss * loss_mask
loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
```

**解释：**
- 每个 batch 随机采样扩散时间步 `t`。
- 把条件维固定成真实值（inpainting）。
- 模型预测噪声残差（默认 epsilon）。
- MSE 损失只在非条件位置计算。

---

## 6. 关键技巧总结（工程细节）

### 6.1 Receding Horizon Control
- **只执行预测序列的前几步**，其余作为“未来计划”。
- 好处：
  - 可在下一时刻用新观测重新规划；
  - 减少长期误差积累。
- 代码位置：`predict_action()` 最后 5 行。

### 6.2 全局条件注入
- 视觉特征和低维聚合特征拼接成 `global_cond`。
- 作用：
  - 控制 UNet 中的 FiLM 或类似调制机制；
  - 在每层卷积中注入观测信息。

### 6.3 Inpainting
- 条件维在每个扩散步都被重置为真实值。
- 对于 `lowdim_as_global_cond=False` 的模式非常关键。

### 6.4 归一化一致性
- 训练与推理统一使用 `LinearNormalizer`。
- 如果在外部 pipeline 中已有归一化（如 PreprocessAgent），需避免重复归一化。

---

## 7. 依赖说明（务必注意）

该文件中直接依赖以下模块（部分当前仓库缺失）：
- `TemporalAggregator`（缺失实现）
- `video_core` / `GlobalAvgpool`（在 config 中引用，但缺失）

**迁移时必须补齐，否则无法运行。**

---

## 8. 小结（一句话版）

DiffusionUnetVideoPolicy 本质上是：
> “用视频和低维状态作为条件，对动作序列做扩散建模，并在推理时以 receding horizon 方式滚动执行。”

它的优势在于：
- 能充分利用历史时序信息；
- 通过扩散采样获得多样化动作序列；
- 在执行端用短窗口实现稳定控制。

---

如需进一步深挖（比如 `ConditionalUnet1D` 内部的 FiLM 结构、mask 生成策略、TemporalAggregator 的具体实现），告诉我，我可以继续扩展文档。
