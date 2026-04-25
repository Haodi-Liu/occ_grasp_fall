# DIFFUSION_POLICY Replaybuffer 向 ACT_BC_VISION 模式对齐改造计划（校验完善版）

## 1. 结论

结论是：

- 方向上，让 `occ_grasp_models/agents/diffusion_policy` 的 replay 构造、读取和训练消费方式向 `occ_grasp_models/agents/act_bc_vision` 的“train-ready sample”模式对齐，是正确的。
- 但当前文档原版对“与原始 `diffusion_policy` image policy 的数据契约和超参数效果对齐”考虑还不够充分，不能直接按原文落地。

真正需要对齐的是：

1. replay fill 阶段就物化完整训练样本
2. `agent.update()` 不再回磁盘重建 sequence
3. policy 接收到的 `batch` 与当前本地实现、以及上游原始 image policy 的输入契约保持一致
4. normalizer 统计口径保持一致
5. UNet / Transformer 的关键训练超参数分别对齐

不应该机械照搬的是：

1. `act_bc_vision` 的 point cloud / intrinsics / extrinsics
2. `act_bc_vision` 把 RGB 以 `float32` 存 replay 的做法
3. 所有与 DIFFUSION_POLICY 当前训练目标无关的字段

一句话总结：

- 这次要学的是 `ACT_BC_VISION` 的 replay 运作模式
- 不是把 `ACT_BC_VISION` 的字段设计整套抄过来


## 2. 这次校验实际对照了什么

这份完善版结论，不是只基于文档直觉，而是对照了以下代码路径：

### 2.1 上游原始 diffusion_policy

- `diffusion_policy/diffusion_policy/policy/diffusion_unet_image_policy.py`
- `diffusion_policy/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py`
- `diffusion_policy/diffusion_policy/dataset/robomimic_replay_image_dataset.py`
- `diffusion_policy/diffusion_policy/dataset/real_pusht_image_dataset.py`
- `diffusion_policy/diffusion_policy/common/sampler.py`
- `diffusion_policy/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`
- `diffusion_policy/diffusion_policy/config/train_diffusion_transformer_hybrid_workspace.yaml`

### 2.2 当前本地 DIFFUSION_POLICY 实现

- `occ_grasp_models/agents/diffusion_policy/launch_utils.py`
- `occ_grasp_models/agents/diffusion_policy/replay_utils.py`
- `occ_grasp_models/agents/diffusion_policy/agent.py`
- `occ_grasp_models/agents/diffusion_policy/normalizer_utils.py`
- `occ_grasp_models/agents/diffusion_policy/policy/diffusion_unet_image_policy.py`
- `occ_grasp_models/agents/diffusion_policy/policy/diffusion_transformer_image_policy.py`
- `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml`
- `occ_grasp_models/run_seed_fn.py`

### 2.3 用来参考 replay 模式的 ACT_BC_VISION

- `occ_grasp_models/agents/act_bc_vision/launch_utils.py`
- `occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py`


## 3. 原文档遗漏但必须明确的硬约束

## 3.1 这次对齐只应承诺覆盖“当前本地真实使用的安全子集”

当前本地代码真实支持的安全子集是：

1. UNet：`obs_as_global_cond=True`
2. Transformer：`obs_as_cond=True`
3. Transformer：`time_as_cond=True`
4. 当前默认：`n_latency_steps=0`
5. 当前本地假设：`robot_name='bimanual'`，`low_dim_size == action_dim == 18`

这点必须写进文档，因为：

1. 上游原始 UNet 支持 `obs_as_global_cond=False`
2. 上游原始 Transformer 支持 `obs_as_cond=False`
3. 本地 Transformer 路径并没有完整复刻这些分支

因此，文档应该明确说：

- 第一阶段做的是“当前本地活跃子集”的严格对齐
- 不是“上游原始 image policy 全部分支”的完全覆盖

## 3.2 原始 policy 真正要求对齐的是输入契约，不只是字段名

改 replay 之后，喂给 policy 的最终 `batch` 仍必须满足：

1. `batch['obs'][key]` 是 `(B, To, ...)`
2. `batch['action']` 是 `(B, T, Da)`
3. `To == n_obs_steps`
4. `T == horizon`

其中：

- UNet `obs_as_global_cond=True` 时，只会消费前 `n_obs_steps` 的 obs
- Transformer `obs_as_cond=True` 时，也只会消费前 `n_obs_steps` 的 obs

但如果走这些分支：

1. UNet `obs_as_global_cond=False`
2. 原始 Transformer `obs_as_cond=False`

则 policy 在 `compute_loss()` 中会按 horizon 全长消费 obs，而不是只消费 `n_obs_steps`。

这意味着：

- 当前文档原先默认“replay 只存 `n_obs_steps` obs”这个设计，只对当前安全子集成立
- 如果将来要开放其它条件模式，replay 中 obs 的存储长度必须扩成 `horizon`，或者预存全长 obs feature

## 3.3 YARR replay sample 有一层原文档没写到的额外时间轴

当前训练不是直接从普通 `Dataset.__getitem__()` 返回单样本，而是从 YARR 的 `TaskUniformReplayBuffer.sample_transition_batch()` 返回 batch。

对 `timesteps=1` 的当前配置，原始 replay sample 形状实际上会是：

- observation: `(B, 1, ...)`
- action: `(B, 1, ...)`
- extra replay elements（如 `demo`）通常没有这层 observation 时间轴

因此，新文档必须明确区分两种 shape：

### replay sample 原始 shape

- `front_rgb`: `(B, 1, n_obs_steps, 3, H, W)`
- `low_dim_state`: `(B, 1, n_obs_steps, 18)`
- `action`: `(B, 1, horizon, 18)`
- `is_pad`: `(B, 1, horizon)`

### 喂给 policy 前的最终 batch shape

- `front_rgb`: `(B, n_obs_steps, 3, H, W)`
- `low_dim_state`: `(B, n_obs_steps, 18)`
- `action`: `(B, horizon, 18)`
- `is_pad`: `(B, horizon)`

也就是说，`agent.update()` 必须先显式做一次 `[:, 0]` / squeeze，再组装 policy batch。

## 3.4 `is_pad` 对当前 diffusion loss 不起作用

这是原文档里最大的一个潜在误导点。

当前本地和上游原始 diffusion image policy 的 `compute_loss()`：

1. 都不会读取 `batch['is_pad']`
2. 都不会用 `is_pad` 去构造 loss mask
3. loss mask 只来自原始 policy 内部的 `condition_mask`

所以现阶段正确理解应当是：

1. `is_pad` 可以保留在 replay 中
2. 它主要用于边界检查和新旧路径一致性校验
3. 第一版不要把 `is_pad` 接到 loss 里

否则就已经不是“与原始 diffusion_policy 行为对齐”，而是在改变训练目标。

## 3.5 low_dim 与 action 的 18D 语义必须保持完全一致

当前本地 DIFFUSION_POLICY 的 low_dim 不是一般意义上的 proprio，而是“action-equivalent low_dim”。

其语义是：

- 18D = 右臂 8D + `right_ignore_collisions` + 左臂 8D + `left_ignore_collisions`

并且依赖以下 canonicalization：

1. quaternion 归一化
2. quaternion canonical sign 到 `w >= 0`
3. `ignore_collisions` 拼入 low_dim / action

因此，新 replay fill 阶段必须继续复用当前 `replay_utils.py` 中与 `_get_action_with_ignore_from_obs(...)` 等价的语义，确保：

1. `low_dim_state` 来自这套逻辑
2. `action` 也来自这套逻辑
3. 两者的 18D 排布完全一致

不能让：

1. `low_dim_state` 走一套构造逻辑
2. `action` 走另一套未 canonicalize 的原始 pose 逻辑

否则 normalizer、训练分布和 `act()` 后处理都会漂移。

## 3.6 `n_latency_steps > 0` 时，当前原文档的 logical-start 表述并不充分

原文档写“继续保留 `pad_before = n_obs_steps - 1 + n_latency_steps`、`pad_after = n_action_steps - 1` 就够了”，这在 `n_latency_steps=0` 时成立，但对 latency-aware 语义并不完整。

从上游 `real_pusht_image_dataset.py` 看，若 `n_latency_steps > 0`，正确做法是：

1. 先按 `sequence_length = horizon + n_latency_steps` 采样
2. obs 仍只取前 `n_obs_steps`
3. action 再丢掉最前面的 `n_latency_steps`
4. 最终喂给 policy 的 action 仍是 `(horizon, action_dim)`

因此文档必须写成：

- 当前本地配置下，`n_latency_steps=0`，可直接保留现有 logical-start 语义
- 如果未来启用 latency，replay builder 和 logical-start 范围必须一起改，不能只保留现有 `iter_logical_starts(length, horizon, ...)`


## 4. 目标架构

改造后的目标架构如下。

## 4.1 replay 内存储的内容

对每个 logical start，存一条完整训练样本：

- `obs.low_dim_state`: `(n_obs_steps, 18)`
- `obs.{camera}_rgb`: `(n_obs_steps, 3, H, W)`
- `action`: `(horizon, 18)`
- `is_pad`: `(horizon,)`
- `task`
- `demo`

这里的“存 `n_obs_steps` obs”是第一阶段的刻意约束，不是普适真理。

## 4.2 训练时的数据流

训练时不再做这些动作：

- 不再用 `episode_id/timestep` 回查 episode
- 不再在 `agent.update()` 内逐张读图
- 不再在 `agent.update()` 内动态构造 sequence batch

训练时改为：

1. `PyTorchReplayBuffer.dataset()` 直接 sample 完整样本
2. worker 把 sample 交给主线程
3. `agent.update()` 先去掉 YARR 的 replay 时间轴
4. 再把 sample 组织成 policy 需要的 `batch`
5. 然后执行 `compute_loss -> backward -> optimizer.step`

## 4.3 normalizer 的来源

normalizer 不再基于 `IndexSequenceLoader` 二次回放统计，而是直接基于 replay 中已经物化好的：

- `action`
- `low_dim_state`

来做统计。

但要注意：

1. image normalizer 不是拟合出来的
2. 当 `imagenet_norm=True` 时，image normalizer 应保持 identity
3. RGB 只做 `uint8 -> float32 / 255.0`
4. 当 `imagenet_norm=False` 时，image normalizer 才走固定 `[0,1] -> [-1,1]`


## 5. 与 ACT_BC_VISION 的相同点和不同点

## 5.1 需要对齐的点

需要对齐的是：

1. replay fill 阶段直接写完整训练样本
2. replay sample 直接可喂训练
3. `agent.update()` 不再做磁盘 I/O 和样本重组
4. 数据准备职责从训练主线程前移

## 5.2 不需要照搬的点

不需要照搬的是：

1. point cloud
2. camera intrinsics / extrinsics
3. ACT 风格动作字段拆分方式
4. `float32` RGB replay 存储

DIFFUSION_POLICY 只保留它真实训练所需的最小字段集合。


## 6. 关键设计决策

## 6.1 replay 存储粒度

建议按“每个 logical start 一条样本”存。

即对每个 logical start `t`，直接生成一条包含：

- `n_obs_steps` 观测序列
- `horizon` 动作序列
- `is_pad`

的完整训练样本。

这可以完全替代当前运行期的 `build_obs_sequence()` 与 `build_action_sequence()`。

但必须附加一句：

- 这个设计在当前 `n_latency_steps=0` 时严格成立；未来若启用 latency，action builder 要按 `horizon + n_latency_steps` 先构造再裁掉前缀。

## 6.2 RGB 存储类型

建议 replay 中存 `np.uint8`，训练时再转 `float32 / 255.0`。

原因：

1. 多相机、多步序列下，`float32` 图像过于膨胀
2. 当前 encoder 输入本来也只需要 `[0,1]` float
3. 与现有 loader 的实际数值语义更容易保持一致

## 6.3 replay action 字段的使用方式

建议直接把 `(horizon, action_dim)` 的动作序列存到 replay 的 `action` 字段：

- `action_shape = (horizon, action_dim)`
- 当前即 `(16, 18)`

`is_pad` 仍作为 observation element 存储，但第一版不进入 loss。

## 6.4 观测字段命名

建议继续使用 policy 当前习惯：

- `{cam}_rgb`
- `low_dim_state`

这样 `agent.update()` 只需要做 shape 和 dtype 处理，不需要改 policy key 解析逻辑。

## 6.5 RGB 数值语义必须与现有 loader 一致

新的 replay fill 必须保持与 `IndexSequenceLoader.build_obs_sequence()` 相同的语义：

1. 从 episode 对应 PNG 读图
2. 仍按当前路径 resize 到 `cfg.method.image_size`
3. 仍保持 `channels_first`
4. replay 中建议存 `uint8`
5. 训练时转为 `float32 / 255.0`

这里如果换了另一套 resize / 通道 / 色彩流程，即使 shape 相同，也不算与当前训练行为对齐。


## 7. 详细代码修改计划

## 7.1 `occ_grasp_models/agents/diffusion_policy/launch_utils.py`

### 目标

把当前“索引 replay”改成“完整样本 replay”。

### 需要改的点

#### 1. 重写 `create_replay(...)`

当前只存：

- `episode_id`
- `timestep`

需要改成存：

- `low_dim_state`: `(n_obs_steps, 18)`
- `is_pad`: `(horizon,)`
- 每个相机的 RGB 序列：`(n_obs_steps, 3, H, W)`
- `task`
- `demo`

并把 replay action shape 改为：

```python
action_shape=(cfg.method.horizon, cfg.method.action_dim)
```

### 推荐 schema

对 `n_obs_steps=2`, `horizon=16`, 3 个相机：

- `ObservationElement("low_dim_state", (2, 18), np.float32)`
- `ObservationElement("is_pad", (16,), np.int32)`
- `ObservationElement("front_rgb", (2, 3, H, W), np.uint8)`
- `ObservationElement("wrist_right_rgb", (2, 3, H, W), np.uint8)`
- `ObservationElement("wrist_left_rgb", (2, 3, H, W), np.uint8)`

`replay.add(...)` 时：

- `action` 直接传 `(16, 18)` 的动作序列

#### 2. 重写 `fill_multi_task_replay(...)`

当前 fill 逻辑只构建 episode index 和 logical start index。

需要改成：

1. 逐 task 读 demo
2. 逐 episode 遍历 logical start
3. 对每个 logical start 直接生成完整训练样本
4. 立刻 `replay.add(...)`

#### 3. logical-start 语义要写成“当前配置下保持不变”

当前配置下可保留：

- `pad_before = n_obs_steps - 1 + n_latency_steps`
- `pad_after = n_action_steps - 1`

但文档必须补充：

- 当前这句话只在 `n_latency_steps=0` 的现配置下成立
- 如果以后支持 latency，logical-start 的范围和 action builder 都要一起改

## 7.2 `occ_grasp_models/agents/diffusion_policy/replay_utils.py`

### 目标

把这个文件从“训练期动态组 batch 工具”改成“fill replay 阶段样本构造工具”。

### 需要改的点

#### 1. 弱化或移除 `IndexSequenceLoader`

处理建议：

- 第一阶段：保留类定义作为一致性对照工具
- 第二阶段：把可复用 helper 拆出来给 replay fill 使用
- 第三阶段：确认无训练路径调用后，删除或保留为 legacy

#### 2. 抽出 fill replay 阶段专用 helper

建议新增或重构成这些 helper：

- `build_obs_sequence_from_demo(demo, timestep, cfg, cameras)`
- `build_action_sequence_from_demo(demo, timestep, horizon)`
- `build_replay_sample_from_demo(demo, timestep, cfg, task)`

其中：

`build_obs_sequence_from_demo(...)` 负责：

1. 根据 logical start 和 `n_obs_steps` 取观测索引
2. 走与当前 loader 相同的 RGB 读取与 resize 路径
3. 生成 `{cam}_rgb`
4. 生成 `low_dim_state`
5. 保持 RGB 输出为 `uint8`

`build_action_sequence_from_demo(...)` 负责：

1. 生成 `(horizon, 18)` 动作序列
2. 生成 `(horizon,)` 的 `is_pad`
3. 保持当前 quaternion / ignore-collision canonicalization 不变

#### 3. 明确 RGB 输出格式

replay 中 RGB shape 应为：

```python
(n_obs_steps, 3, H, W)
```

而不是：

```python
(n_obs_steps, H, W, 3)
```

#### 4. 明确 raw replay shape 与 policy batch shape 的区别

文档和实现里都要显式区分：

- replay sample `front_rgb`: `(B, 1, n_obs_steps, 3, H, W)`
- policy batch `front_rgb`: `(B, n_obs_steps, 3, H, W)`

- replay sample `action`: `(B, 1, horizon, 18)`
- policy batch `action`: `(B, horizon, 18)`

## 7.3 `occ_grasp_models/agents/diffusion_policy/agent.py`

### 目标

让 agent 从 index-based 动态组 batch 模式切换到直接消费 replay sample 模式。

### 需要改的点

#### 1. 删除 index-based 状态

这些成员不再需要：

- `self.seq_loader`
- `self._episode_index_path`
- `set_episode_index_path(...)`
- `_extract_index_from_replay_sample(...)`

#### 2. 重写 `update(...)`

新逻辑应改为：

1. 直接从 `replay_sample` 读取 `{cam}_rgb`, `low_dim_state`, `action`, `is_pad`
2. 先去掉 YARR 的 replay 时间轴 `[:, 0]`
3. RGB 若为 `uint8`，转 `float32 / 255.0`
4. 组装成 policy 需要的 `batch`
5. 执行 `compute_loss -> backward -> step`

### 推荐伪代码

```python
obs = {}
for cam in self.cfg.method.camera_names:
    rgb = replay_sample[f"{cam}_rgb"][:, 0]
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    else:
        rgb = rgb.float()
    obs[f"{cam}_rgb"] = rgb

obs["low_dim_state"] = replay_sample["low_dim_state"][:, 0].float()

batch = {
    "obs": obs,
    "action": replay_sample["action"][:, 0].float(),
    "is_pad": replay_sample["is_pad"][:, 0].to(dtype=torch.int32),
}
```

#### 3. `_to_device(...)` 仍保留

这个 helper 仍有价值，但会明显简化，因为不再需要处理 index-based 二次构造。

## 7.4 `occ_grasp_models/run_seed_fn.py`

### 需要改的点

#### 1. 删除 episode index 相关流程

这些逻辑不再需要：

- `episode_index_path = diffusion_policy.launch_utils.fill_multi_task_replay(...)`
- `agent.set_episode_index_path(...)`

#### 2. 替换 normalizer 拟合入口

当前是：

- `fit_normalizer_from_index_replay(...)`

需要改成：

- `fit_normalizer_from_replay_samples(...)`

直接基于 replay 中已存好的 `action` 和 `low_dim_state` 做统计。

## 7.5 `occ_grasp_models/agents/diffusion_policy/normalizer_utils.py`

### 目标

normalizer 统计直接建立在 replay 存储的数据上，但统计口径必须与当前实现一致。

### 新接口建议

```python
def fit_normalizer_from_replay_samples(cfg, replay_buffer, sample_size=10000):
    ...
```

### 统计数据来源

- `action`: replay 的 `action` 字段
- `low_dim_state`: replay 的 observation element

### 统计口径要求

1. 仍按当前实现，把 sequence flatten 到最后一维后统计
2. 不要只取最后一个 obs step
3. 不要只取未 pad 的 action
4. 否则就不再与当前 `fit_normalizer_from_index_replay(...)` 等价

### 实现建议

方式 A：随机 sample 若干 batch 后 flatten 统计

- 优点：好实现
- 缺点：口径要非常小心地与新 replay 形状对齐

方式 B：直接扫描 replay 中已经物化好的完整序列，再 flatten 统计

- 优点：更容易做到与当前统计口径严格等价
- 缺点：要更直接接触 replay 存储布局

如果目标是“先保等价再提速”，建议优先做方式 B。


## 8. 配置与超参数的效果对齐

## 8.1 需要调整的配置项

以下配置会变成 legacy：

- `index_cache_size`
- `image_cache_size`
- `index_seed`

建议保留一轮兼容期，但明确标注为 legacy。

## 8.2 必须新增的配置项

```yaml
replay_rgb_dtype: uint8
replay_store_full_sequences: true
legacy_index_replay: false
transformer_optimizer_betas: [0.9, 0.95]
transformer_lr_warmup_steps: 1000
```

其中：

- `replay_rgb_dtype`：replay 内 RGB 存储类型
- `replay_store_full_sequences`：明确启用 train-ready replay
- `legacy_index_replay`：过渡期回退开关
- `transformer_optimizer_betas`：对齐原始 transformer image workspace
- `transformer_lr_warmup_steps`：对齐原始 transformer image workspace

## 8.3 UNet 与 Transformer 不能共用同一套 optimizer beta / warmup

这是原文档漏掉、但对“效果对齐”影响最大的点之一。

### UNet 建议保持

- `lr=1e-4`
- `weight_decay=1e-6`
- `optimizer_betas=[0.95, 0.999]`
- `lr_scheduler=cosine`
- `lr_warmup_steps=500`
- `use_ema=True`
- `num_train_timesteps=100`
- `beta_schedule=squaredcos_cap_v2`
- `prediction_type=epsilon`

### Transformer 建议保持

- `learning_rate=1e-4`
- `transformer_weight_decay=1e-3`
- `obs_encoder_weight_decay=1e-6`
- `optimizer_betas=[0.9, 0.95]`
- `lr_scheduler=cosine`
- `lr_warmup_steps=1000`
- `use_ema=True`
- `num_train_timesteps=100`
- `beta_schedule=squaredcos_cap_v2`
- `prediction_type=epsilon`

必须明确指出：

- 当前本地 `DIFFUSION_POLICY.yaml` 中把 `optimizer_betas` 统一设为 `[0.95, 0.999]`
- 这对 UNet 是对齐的
- 但对 Transformer 并不对齐上游 `train_diffusion_transformer_hybrid_workspace.yaml`

如果文档目标包含“超参数在效果上也对齐”，这点必须写死，不应只写成“相近”。

## 8.4 关于 Transformer 的一个现实边界

当前本地 Transformer 路径虽然在条件注入语义和训练流程上可对齐，但它并不是上游原始 `DiffusionTransformerHybridImagePolicy` 的完整复刻：

1. 本地用的是 `MultiImageObsEncoder`
2. 上游原始实现用的是 robomimic `ObservationEncoder`

因此应当这样表述：

- 本次改造追求的是“数据处理契约、训练路径和关键优化器设置”的效果对齐
- 不是声称本地 transformer 分支已经与上游实现做到结构级完全同构


## 9. 分阶段实施计划

建议按 4 个阶段推进。

## 阶段 1：只改 replay fill，不动 policy

目标：

- replay 内成功写入完整样本
- 不再只写 `episode_id/timestep`

重点验证：

1. replay schema 正确
2. raw replay sample shape 正确
3. 单个 batch 可视化和内容检查通过

## 阶段 2：改 agent.update 直接消费 replay sample

目标：

- 完全绕过 `IndexSequenceLoader`
- 单步训练能够跑通

重点验证：

1. replay sample 去掉 YARR 时间轴后，`batch['obs']` shape 是否与 policy 预期一致
2. `batch['action']` 是否为 `(B, horizon, 18)`
3. loss 是否与旧路径同量级

## 阶段 3：改 normalizer 流程

目标：

- 去掉 `fit_normalizer_from_index_replay`
- 改为新 replay 直接统计

重点验证：

1. 新旧 normalizer 统计量是否近似一致
2. 初始 loss 是否在合理范围

## 阶段 4：清理 legacy 路径

目标：

- 删除或弃用 `IndexSequenceLoader` 的训练主路径调用
- 删除无用配置项


## 10. 验证清单

每个阶段都建议跑以下检查。

## 10.1 shape 检查

先确认 raw replay sample 中：

- `front_rgb.shape == (B, 1, n_obs_steps, 3, H, W)`
- `wrist_right_rgb.shape == (B, 1, n_obs_steps, 3, H, W)`
- `wrist_left_rgb.shape == (B, 1, n_obs_steps, 3, H, W)`
- `low_dim_state.shape == (B, 1, n_obs_steps, 18)`
- `action.shape == (B, 1, horizon, 18)`
- `is_pad.shape == (B, 1, horizon)`

再确认喂给 policy 前的最终 batch 中：

- `front_rgb.shape == (B, n_obs_steps, 3, H, W)`
- `wrist_right_rgb.shape == (B, n_obs_steps, 3, H, W)`
- `wrist_left_rgb.shape == (B, n_obs_steps, 3, H, W)`
- `low_dim_state.shape == (B, n_obs_steps, 18)`
- `action.shape == (B, horizon, 18)`
- `is_pad.shape == (B, horizon)`

## 10.2 数值检查

确认：

- `uint8` RGB 转 float 后范围为 `[0,1]`
- quaternion 已归一化且满足 canonical sign（`w >= 0`）
- `low_dim_state` 与 action 的 18D 排布完全一致
- `is_pad` 与边界样本一致

## 10.3 语义一致性检查

随机抽若干样本，比较：

- 旧 `IndexSequenceLoader.build_batch(...)`
- 新 replay sample 去掉 YARR 时间轴后组出来的最终 `batch`

在同一 logical start 下是否一致。

至少比较：

- obs RGB
- low_dim_state
- action sequence
- is_pad

## 10.4 normalizer 一致性检查

比较：

- 旧 `fit_normalizer_from_index_replay(...)`
- 新 `fit_normalizer_from_replay_samples(...)`

至少确认：

1. `normalizer['action']` 的输入统计量量级一致
2. `normalizer['low_dim_state']` 的输入统计量量级一致
3. 新旧初始 loss 在同一数量级

## 10.5 loss 语义检查

确认新路径下：

1. `compute_loss()` 仍然没有把 `is_pad` 接进 loss mask
2. `condition_mask` 仍然只来自 policy 内部原始逻辑
3. `obs_as_global_cond=True` / `obs_as_cond=True` 分支下，policy 实际只消费前 `n_obs_steps` 的 obs

## 10.6 配置对齐检查

至少检查：

1. `model_type=unet` 时，optimizer beta / warmup 仍是 UNet 对齐值
2. `model_type=transformer` 时，optimizer beta / warmup 已切到 transformer 对齐值
3. 本地 transformer 路径仍明确限制在 `obs_as_cond=True`、`time_as_cond=True`

## 10.7 性能检查

对比以下指标：

1. replay fill 时间
2. 训练 step time
3. GPU 利用率
4. CPU 利用率
5. 磁盘占用


## 11. 风险与约束

## 11.1 最大风险：replay 体积膨胀

这是这套方案最大的代价。

因为每个 logical start 都会复制一份：

- `n_obs_steps` 图像序列
- `horizon` 动作序列

所以这是用更多存储换更快训练。

缓解方式：

1. RGB 用 `uint8`
2. 不存无关字段
3. 只存 policy 训练真实需要的最小集合

## 11.2 若不明确条件分支边界，会出现“伪对齐”

如果文档继续默认“所有模式都只存 `n_obs_steps` obs”，那它只对当前默认配置成立，对原始 policy 的其它分支并不成立。

因此必须在文档里明确：

1. 这是当前本地活跃子集的严格对齐
2. 不是原始 image policy 全部分支的完全覆盖

## 11.3 若把 `is_pad` 接进 diffusion loss，会改变训练目标

保留 `is_pad` 没问题。

但第一版如果把它接入 `compute_loss()`，那就已经不是“对齐原始 diffusion_policy”，而是在引入一个新的 pad-aware objective。

## 11.4 若不拆分 UNet / Transformer 优化器配置，会产生效果漂移

上游原始 workspace 对 UNet 和 Transformer 的 optimizer beta / warmup 设置并不相同。

如果不显式分流：

- UNet 用 `[0.95, 0.999]`, warmup `500`
- Transformer 用 `[0.9, 0.95]`, warmup `1000`

那即使 replay 改对了，也不应声称“效果上已经与原始实现对齐”。

## 11.5 第一版不建议同时做 AMP、compile、模型瘦身

这份方案只聚焦 replay 模式对齐。

第一版改造时，不建议同时再加：

- AMP
- `torch.compile`
- backbone / UNet 宽度调整

否则一旦效果异常，很难判断是 replay 改造，还是其它变量导致。


## 12. 推荐落地顺序

建议严格按下面顺序来：

1. 新建 replay schema
2. 新写 fill replay 路径
3. 保留旧 `IndexSequenceLoader` 作为对照 oracle
4. 写 shape / sample 一致性检查
5. 改 `agent.update()` 直接消费 replay sample
6. 改 normalizer
7. 校正 UNet / Transformer 分流配置
8. 做小规模训练 benchmark
9. 验证稳定后再删除 legacy 路径


## 13. 涉及文件总表

高概率会改到这些文件：

- `occ_grasp_models/agents/diffusion_policy/launch_utils.py`
- `occ_grasp_models/agents/diffusion_policy/replay_utils.py`
- `occ_grasp_models/agents/diffusion_policy/agent.py`
- `occ_grasp_models/agents/diffusion_policy/normalizer_utils.py`
- `occ_grasp_models/run_seed_fn.py`
- `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml`

可选新增：

- `occ_grasp_models/agents/diffusion_policy/replay_schema.py`
- `occ_grasp_models/agents/diffusion_policy/replay_migration_checks.py`


## 14. 最终建议

最关键的一句话是：

应该让 `occ_grasp_models/agents/diffusion_policy` 向 `act_bc_vision` 的 replay 运作模式对齐，但不能逐字段、逐数据类型机械照抄。

正确做法是：

- 学它“fill replay 阶段就生成完整训练样本”
- 不学它“把所有能存的东西都存进去”

并且要再补上一句原文档最该明确却没有明确的话：

- replay 改造只能前移数据准备成本，不能改变原始 diffusion policy 的条件注入语义、padding 语义、normalizer 口径，以及模型专属优化器配置。

如果后续真的实施，这份文档对应的第一个代码目标应是：

`让 diffusion_policy 的 update() 完全不再调用 IndexSequenceLoader.build_batch()`。
