# Diffusion Policy (Image UNet / Transformer) -> occ_grasp_models 迁移评估与方案

> 目标：把 `/home/hdliu/occ_grasp_fall/diffusion_policy` 中的**组件实现与训练逻辑**迁移到 `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents`，
> 在 occ_grasp_fall 框架下构建**可复现的 diffusion policy 基线**，并在此基础上扩展**带条件注入的增强策略**（策略/阶段/关键点等）。

## 0. 文档状态与导航（2026-03-08）

为避免“迁移计划”与“代码现状”混淆，本节先给出当前代码事实：

- **已落地（基线）**
  - `DIFFUSION_POLICY` 已支持 `model_type=unet|transformer` 双分支。
  - Transformer 路线（`DiffusionTransformerImagePolicy + TransformerForDiffusion`）已接入训练/推理主链路。
  - 索引式 replay、动态序列加载、normalizer 拟合、YARR `Agent.update/act` 闭环均已接通。
  - `agent.py` 已按 `model_type=transformer` 调用 `policy.get_optimizer(...)`，不再是统一 AdamW 一刀切。

- **未落地（后续增强）**
  - 条件注入模块（`cond_wrappers.py`）仍为占位。
  - 条件预测预训练与联合训练流程仍未实现。

- **阅读顺序建议**
  1) 第 1-8 节：总体约束、背景与风险（策略层面）。  
  2) 第 9 节：UNet baseline 与索引 replay 的实现细节（实现层面）。  
  3) 第 10 节：Transformer 路线落地复盘与当前边界（对照层面）。  

### 0.1 本次同步核查结论（面向实施）

- **配置与入口**：`DIFFUSION_POLICY.yaml`、`agent_factory.py`、`run_seed_fn.py` 已对齐；文档中补齐了 `index_seed` 与双主干优化器分支细节。  
- **训练/推理主链路**：`DiffusionPolicyAgent` 现已支持“UNet 默认 AdamW + Transformer 参数分组优化器 + 可配置 LR scheduler + EMA（用于推理）”，并补充 `act()` 的动作后处理语义。  
- **索引 replay 语义**：当前实现仍是 `t in [0, L-1]`；文档中的 `E3.1` 已改为“可执行的完整改造方案”（不仅改 replay 写入，还同步改 action pad 语义）。  
- **边界与后续**：条件注入仍未落地，保持后续阶段；本次已完成“最小改动版”训练能力补齐（scheduler+EMA），见 0.2 节。

### 0.2 可引用变更：最小改动版 Scheduler + EMA（2026-03-08）

> 本节为“可直接引用”摘要，用于回答“当前 DIFFUSION_POLICY 是否已支持 cosine warmup 与 EMA”。

- **目标**：在不重构 YARR runner 的前提下，为 `DIFFUSION_POLICY` 补齐与原始 diffusion_policy workspace 更一致的最小训练能力：`LR scheduler` 与 `EMA`。
- **代码改动（最小集合）**
  - `occ_grasp_models/agents/diffusion_policy/agent.py`
    - `build()` 新增：scheduler 初始化（`method.lr_scheduler/lr_warmup_steps`）与 EMA 初始化（`method.use_ema/*ema*`）。
    - `update()` 新增：`optimizer.step()` 后执行 `lr_scheduler.step()` 与 `ema.step(policy)`。
    - `act()` 新增：有 EMA 时优先使用 EMA policy 推理。
    - `save_weights/load_weights()` 新增：读写 `diffusion_policy_train_state.pt`（optimizer/scheduler/ema/update_step）；保留 `diffusion_policy.pt` 兼容旧权重。
  - `occ_grasp_models/agents/diffusion_policy/model/diffusion/ema_model.py`
    - 新增本地 `EMAModel` 实现，避免跨仓导入依赖。
  - `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml`
    - 新增配置键：`lr_scheduler`、`lr_warmup_steps`、`use_ema`、`ema_update_after_step`、`ema_inv_gamma`、`ema_power`、`ema_min_value`、`ema_max_value`。
- **默认行为**
  - 当前方法配置默认启用：`lr_scheduler: cosine`、`lr_warmup_steps: 500`、`use_ema: True`。
- **兼容性说明**
  - 旧 checkpoint（仅 `diffusion_policy.pt`）可继续加载；
  - 新 checkpoint 额外包含 `diffusion_policy_train_state.pt` 以支持训练态恢复。


## 1. 背景与目标

本迁移旨在将 diffusion_policy 的**可复现基线**带入 occ_grasp_fall，并在统一的 YARR 训练/评估框架下运行。
当前代码已覆盖 **UNet + Transformer 双主干 baseline**；条件注入与增强策略仍放在后续阶段。

### 1.1 基线目标（已落地）
- **组件层“原模原样”复刻**：优先从 diffusion_policy 直接复制实现（模型模块、工具函数、训练逻辑、归一化等）。
- **结构层可重组**：允许自由组合这些组件，形成我们需要的 diffusion policy 架构。
  - **已落地**：**MultiImageObsEncoder + UNet DDPM**（DiffusionUnetImagePolicy）。
  - **已落地**：**同一 obs encoder + Transformer DDPM**（去 robomimic）。
- **训练流程尽量对齐原实现**：扩散训练/采样流程沿用原脚本逻辑，在 YARR 框架下复刻。
- **控制方式为连续控制**（receding horizon），**不是关键帧控制**。

### 1.2 增强目标（后续阶段）
- 在基线扩散模型上**注入策略/阶段/关键点等条件信息**（来自 `DATA_COLLECTION_GUIDE.md` 中的标签）。
- **条件注入思路**参考 `ACT_BC_KEYPOINT_STRATEGY_KEYFRAME_PLAN.md`。
- **“先预训练条件预测模块，再用预测条件辅助动作生成”**的流程来自
  `ACT_BC_KEYPOINT_STRATEGY_MODEL_DIAGRAM.md`：
  - 先预训练条件预测模块（策略/阶段/关键点等）；
  - 再将其预测结果作为条件，辅助扩散策略训练与推理。
- **执行顺序**：先在新框架下跑通基线训练/评估（目标 A），再进入条件注入与预训练扩展（目标 B）。


## 2. 关键约束与一致性要求

- **不依赖 robomimic**：UNet/Transformer 两条路线都使用 `MultiImageObsEncoder`，不引入 robomimic 依赖。
- **不再依赖视频模块**：`VideoCore/TemporalAggregator/GlobalAvgpool` 在 image 基线中不再需要，避免缺失实现；若未来恢复 video 版本，再单独补齐。
- **观测编码器与 diffusion_unet_image_policy 对齐**：沿用 `MultiImageObsEncoder` + `model_getter.get_resnet` + `CropRandomizer` 等原实现与接口。
- **条件数据已包含**：策略/阶段/关键点等标签在演示数据中已具备，无需额外补标。
- **DIFFUSION_POLICY 低维口径专项约束**：`low_dim_state` 采用“当前帧动作等价本体”，与动作向量同内容同形状（18D）。
- **动作维度以 occ 规范为准**：采用 **18D（含 ignore_collisions）**，并同步修改扩散模型的 action_dim / normalizer。
- **归一化沿用 diffusion_policy**：使用 `LinearNormalizer` 作为唯一归一化体系，避免与 `PreprocessAgent` 重复处理。
- **图像归一化只走一条路径**：若启用 `imagenet_norm`，则 image normalizer 设为 identity（输入保持 [0,1]）；若使用 `get_image_range_normalizer`（0~1→-1~1），则关闭 `imagenet_norm`。二者不要叠加。
- **基线默认不使用 inpainting**：当前基线配置使用 `obs_as_global_cond=True`（扩散对象仅动作序列）；UNet 代码保留 `obs_as_global_cond=False` 分支但默认不启用。


## 3. 基线架构与扩展方向

### 3.1 基线架构候选（UNet Image 优先）

**方案 A：Image Encoder + UNet DDPM**
- 视觉编码：`MultiImageObsEncoder`（多相机单帧 2D 编码；可 share 或 per-cam ResNet；可选 resize/crop/imagenet_norm）。
- 条件建模：`obs_as_global_cond=True`，将 `To` 帧特征展平拼接为 `global_cond`。
- 扩散主干：`ConditionalUnet1D`（仅在动作序列上扩散）。

**方案 B：Image Encoder + Transformer DDPM**
- 视觉编码：同上（`MultiImageObsEncoder` 输出 per-frame 特征）。
- 条件建模：`obs_as_cond=True`，将 `B,To,Do` 作为 `cond` token 供 `TransformerForDiffusion` 使用。
- 扩散主干：`TransformerForDiffusion`（去 robomimic，保持 diffusion_policy 的实现）。
- 当前状态：`occ_grasp_models/agents/diffusion_policy/policy/diffusion_transformer_image_policy.py` 已接入训练/推理主链路。

> 两套方案共享同一观测编码器与数据组织，差异仅在扩散主干与条件注入接口。

### 3.2 增强架构（条件注入，后续阶段）
- 条件来源：`strategy_type / phase_type / keypoints (3D/2D) / affordance` 等。
- 条件处理：
  - 方案 1：编码为 **global_cond**，与视觉/低维特征拼接；
  - 方案 2 暂不考虑（不使用 inpainting）。
- 训练流程：
  1) 预训练条件预测模块；
  2) 冻结或半冻结条件模块；
  3) 扩散策略训练；
  4) 联合微调（可选）。


## 4. 目录组织与模块边界（为代码落地做准备）

> 约定：`occ_grasp_models/agents/diffusion_policy` 作为 **baseline agent**，
> 当前已覆盖 UNet + Transformer 双主干；条件注入仍保留为后续扩展。
> 配置文件保持与 agent 同名的单一入口，符合 occ 训练/评估流程（`run_seed_fn.py` / `agent_factory.py`）。

```
occ_grasp_models/agents/diffusion_policy/
├── __init__.py
├── launch_utils.py              # create_agent + create_replay + fill_replay
├── agent.py                     # YARR Agent: update/act/load/save
├── replay_utils.py              # 索引式 replay + 动态序列采样
├── normalizer_utils.py          # LinearNormalizer 统计/注入
├── common/                      # pytorch_util / normalize_util（已迁移）
├── policy/
│   ├── diffusion_unet_image_policy.py          # 当前主用
│   ├── diffusion_transformer_image_policy.py   # 已启用（obs_as_cond=True）
│   └── cond_wrappers.py                         # 条件注入占位 hook
├── model/
│   ├── common/
│   ├── diffusion/
│   └── vision/
└── configs/
    └── shape_meta_utils.py
```

仅新增一个方法配置（与 agent 同名）：
```
occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml
# 当前支持：
# model_type: unet | transformer
# 条件注入模块仍为后续扩展
```


## 5. 数据与序列设计：索引式 replay

### 5.1 序列对齐定义（t 为序列起点）
- **n_obs_steps**：历史观测长度（多视角 RGB + 低维 + 条件字段）。
- **horizon**：动作序列长度（扩散对象）。
- **n_action_steps**：执行窗口长度（receding horizon）。

当前实现的对齐规则（`t` 为 replay 存储的样本起点）：
- replay 当前仅写入真实索引：`t in [0, L-1]`（`L` 为 episode 长度）。
- `obs_seq = [obs[t], ..., obs[t+n_obs_steps-1]]`：越界按 clamp 到 `[0, L-1]`。
- `action_seq[k]` 语义仍是“在 `obs[t+k]` 时刻应执行的动作”；监督值取同索引动作（无额外 `+1` 偏移）。
- `is_pad` 当前只在 `t+k >= L` 时置 1（前缀 `t<0` 情形尚未进入当前 replay 采样范围）。
- 推理窗口保持 `start = n_obs_steps - 1`：第一步执行动作对应观测窗口末帧的同索引动作位。
- 备注：与原始 `SequenceSampler(create_indices)` 的 `pad_before` 前缀覆盖并非严格等价，详见第 K 节“当前语义与差异”。

### 5.2 选定方案：索引式 replay
- **原因**：多相机 + 多帧序列直接写 replay 内存/磁盘代价过高；索引式 replay 能在不损失语义的情况下大幅降负。
- **做法**：
  - replay 只保存 `(episode_id, timestep)`，不存序列数据；
  - 训练时按索引动态加载图像与低维序列；
  - 可选缓存低维/少量 RGB 以降低 IO。


## 6. YARR 接入与执行语义（连续控制）

### 6.1 Agent 核心职责
- `update()`：构造序列 batch → `policy.compute_loss()` → 反向传播 → `optimizer.step()`；若启用则继续 `lr_scheduler.step()` 与 `ema.step()`。
- `act()`：维护 obs history → `predict_action()`（启用 EMA 时优先用 EMA policy）→ 缓存 action 序列并执行前 `n_action_steps`。
- `build()`：设置 device / optimizer，并按配置初始化 `LR scheduler` 与 `EMA`；训练模式下初始化 `IndexSequenceLoader`；normalizer 由 `run_seed_fn.py` 先拟合再注入。

### 6.2 评估执行（receding horizon）
- 在 RLBench env 中维护 `n_obs_steps` 历史缓冲。
- 每次生成长度为 `horizon` 的动作序列，执行窗口起点为 `n_obs_steps-1`（与 DiffusionUnetImagePolicy 对齐），执行 `n_action_steps`。
- 动作维度需扩展为 18D（附加 `ignore_collisions`）。

### 6.3 归一化一致性
- 使用 diffusion_policy 的 `LinearNormalizer` 作为唯一归一化机制。
- 禁用 `PreprocessAgent` 的 RGB 归一化，避免重复处理。
- RGB 归一化需与 `imagenet_norm` 配置一致（identity ↔ range 二选一）。


## 7. 阶段化落地路线（先基线后扩展）

### 阶段 0：组件迁移与补齐
- 复制 diffusion_policy 的 `model/common`, `model/diffusion`, `model/vision`（含 `multi_image_obs_encoder`, `crop_randomizer`, `model_getter`）。
- 复制 `common/normalize_util.py`, `common/pytorch_util.py`, `model/common/tensor_util.py` 等依赖。
- 核对 `MultiImageObsEncoder` 依赖（torchvision/GroupNorm/RandomCrop）与输入格式（CHW + [0,1]）。

### 阶段 1：基线模型落地
- 复刻 UNet Image baseline（`DiffusionUnetImagePolicy + MultiImageObsEncoder`，`obs_as_global_cond=True`）。
- 接入 Transformer Image baseline（`diffusion_transformer_image_policy.py` + `TransformerForDiffusion`，`obs_as_cond=True`）。
- 对齐 `shape_meta`（`type: low_dim`）与 18D action，并验证 `start=n_obs_steps-1` 的执行窗口。

### 阶段 2：索引式 replay + 内存策略
- 实现索引式 replay（只存 episode_id + timestep）。
- 动态加载序列，保证与“序列直接存储”语义等效。
- 保证可在 YARR OfflineTrainRunner 下稳定训练。

### 阶段 3：训练/评估对齐
- 训练：loss 曲线可下降，输出动作维度正确。
- 评估：receding horizon 控制能在 RLBench 环境跑通。

### 阶段 4：条件注入预训练
- 构建策略/阶段/关键点预测模块。
- 预训练并验证条件预测准确性。

### 阶段 5：条件扩散策略训练
- 将条件模块注入扩散策略（仅 **global_cond**）。
- 联合训练或冻结微调，完成增强版策略。


## 8. 风险与对策

- **内存/IO 压力过大** → 索引式 replay + 压缩存储 + 分阶段缓存。
- **时序对齐错误** → 明确 `t` 对齐规则，统一 `To/horizon/n_action_steps` 与 `start=n_obs_steps-1` 语义。
- **图像归一化错配** → `imagenet_norm` 与 image normalizer 二选一（避免双重归一化）。
- **MultiImageObsEncoder 裁剪路径异常** → 已在迁移版中修正 `random_crop=False` 分支（`CenterCrop` 不再被覆盖）；后续改动需避免回归。
- **显存/速度瓶颈** → 多相机独立 ResNet 可能偏重，可启用 `share_rgb_model` 或缩小 backbone。
- **动作维度不一致** → 统一 18D（含 ignore_collisions），训练/推理同步修改。
- **条件注入影响收敛** → 先预训练条件模块，再逐步引入扩散训练。

### 8.1 环境兼容性专项结论（robodiff -> ppi）

本方案现在明确采用如下策略：
- **不建议**先整体改造 `ppi` 环境（风险是先破坏现有 PPI/ACT 训练链）。
- **不建议**“全部代码迁完后一次性补装”（风险是排错面过大）。
- **建议**“第9节里程碑触发的最小增量补装”：A-C 先迁移、D-M 按触发点补依赖，并按 N 节分批执行，每一步都做 smoke test。

依据调查：
- `diffusion_policy/conda_environment.yaml` 与 `diffusion_policy/conda_environment_real.yaml`（`robodiff`）核心固定为 `python=3.9`、`pytorch=1.12.1`、`torchvision=0.13.1`、`diffusers=0.11.1`、`einops=0.4.1`、`zarr=2.12.0`、`numba=0.56.4`、`dill=0.3.5.1`、`hydra-core=1.2.0`。
- `/home/hdliu/claude_repo/PPI/docs/INSTALLATION.md` 与 `/home/hdliu/claude_repo/PPI/requirements.txt`（`ppi`）已包含同一批核心包，并推荐 `python=3.8` + `pytorch-cuda=11.8`。
- **2026-02-25 本机快照（`/home/hdliu/miniconda3/envs/ppi`）**：`python=3.8.20`、`torch=2.4.1`、`torchvision=0.20.0`、`diffusers=0.11.1`、`einops=0.4.1`、`zarr=2.12.0`、`numba=0.56.4`、`dill=0.3.5.1`、`hydra-core=1.2.0`、`omegaconf=2.3.0`、`natsort=8.4.0`、`Pillow=9.5.0`（import smoke 通过）。
- 因此迁移主线应是“保留 `ppi` 现有 torch/cuda 锚点 + 仅补缺失核心包”，而不是把 `robodiff` 全量平移进来。

### 8.2 ppi 环境调整方案（最小侵入、分层增量）

原则：
- 只补“迁移基线运行必需”的依赖，避免一次性引入整套 `robodiff` 组件。
- 以 `ppi` 已稳定运行版本为锚点，不主动改动其核心 torch/cuda 组合。
- 每次新增依赖后先做 smoke test，再推进训练任务（详见第9节 9.0 表格与 N 节执行蓝图）。

依赖分层：
- **A0（应保持冻结）**：`python + torch + torchvision + cuda` 的当前 `ppi` 可运行组合。
- **A1（UNet Image baseline 必需）**：`diffusers==0.11.1`、`einops==0.4.1`、`zarr==2.12.0`、`numba==0.56.4`、`dill==0.3.5.1`、`hydra-core==1.2.0`、`omegaconf`、`natsort`、`pillow`、`cffi`、`numpy`。
- **B 层（后续扩展按需启用）**：`robomimic`、`robosuite`、`free-mujoco-py`、`ray[default,tune]`、`pymunk`、`pybullet-svl`、`dm-control`（原仓库 benchmark / robomimic 路线 / 分布式多 seed）。
- **C 层（当前阶段建议不引入）**：`pyrealsense2`、`ur-rtde`、`spnav`、`pynput` 等真机/遥操作依赖（系统耦合高，优先隔离）。

执行口径：
- 进入第9节某个代码块前，先跑该块对应的 import smoke；仅当失败时补装对应包。
- 已满足的包不重复改版本，避免无效扰动。

### 8.3 兼容风险与验证门槛

重点风险：
- **Python 主版本差异**：`robodiff` 偏向 3.9，`ppi` 文档为 3.8；需避免为迁移新方法而破坏既有 PPI 代码链。
- **Torch/CUDA 组合漂移**：优先沿用 `ppi` 当前可运行组合，避免为了对齐 `robodiff` 固定版本而引发二进制兼容问题。
- **旧版 diffusers 联动风险**：`diffusers==0.11.1` 与 `huggingface_hub/accelerate` 的组合需在本项目内实测后再冻结。
- **zarr 生态联动风险**：`zarr/numcodecs` 版本需与现有环境一致，否则会在 normalizer/replay 链路出现导入或读写异常。
- **入口回归风险**：`DIFFUSION_POLICY` 分支已在 `occ_grasp_models/agents/agent_factory.py` 与 `occ_grasp_models/run_seed_fn.py` 打通；后续改动需持续做路由回归，避免分支被重构破坏。

验证门槛（逐步通过）：
1. **A-C 关口**：配置与路由 import 成功（不应触发新增依赖）。
2. **D-I 关口**：policy/replay/normalizer 相关 import 成功（按缺失最小补装）。
3. **G/L 关口**：单 batch `compute_loss + backward`、YARR 1 个 update step 跑通。
4. **回归关口**：现有 PPI/ACT 训练入口可启动，确认新增依赖未破坏旧流程。

> 备注（状态同步，2026-02-25）：本节最初用于调查与方案设计；目前 A-M 已按 N2 分批完成，且未对 `ppi` 环境做 `conda/pip` 版本变更。

## 9. 代码实现方案（Baseline: UNet/Transformer + 索引式 Replay）

> **目标**：在 occ_grasp_fall 中稳定运行 **UNet/Transformer 双主干 baseline**，并使用**索引式 replay**（不落盘序列数据）。
> - 不破坏现有流程：保持 YARR 训练/评估框架不变。
> - 最小改动：新增 `diffusion_policy` agent 目录 + 少量接入点修改。
> - 归一化：**仅使用 LinearNormalizer**，不依赖 `PreprocessAgent` 的 RGB 归一化。
> - 语义说明：当前 replay 已满足主链路训练需求，但与原始 `pad_before/pad_after` 采样口径存在已知差异（第 K 节记录）。

---

### 9.0 A-M 代码块对应的 conda 调整（触发式执行）

执行原则：
- 默认沿用当前可运行 `ppi` 环境，先做 A-C 代码迁移，不预先“大补环境”。
- 进入下一代码块前做对应 smoke test，失败再补装该块依赖。
- 同一依赖只在第一次触发时补装一次，后续块复用。

| 代码块 | 实现内容 | 触发依赖 | conda 调整建议 |
| --- | --- | --- | --- |
| A | 新建目录与文件骨架 | 无新增三方包 | 不调整环境 |
| B | 新增 `DIFFUSION_POLICY.yaml` | `hydra-core`、`omegaconf` | 通常 `ppi` 已有；仅在配置解析失败时补装 |
| C | `agent_factory.py` 新方法入口 | 无新增三方包 | 不调整环境 |
| D | `run_seed_fn.py` 接入 index replay + normalizer | `numpy`、`zarr` | 若 `zarr` 缺失或版本不匹配，补 `zarr==2.12.0`（并检查 `numcodecs` 兼容） |
| E1 | `create_agent`（UNet/Transformer + DDPM + obs encoder） | `torch`、`torchvision`、`diffusers`、`einops` | 保持 `ppi` 的 torch/cuda，不升级大版本；缺失时补 `diffusers==0.11.1`、`einops==0.4.1` |
| E2/E3 | `create_replay/fill_multi_task_replay` | YARR 链路、`natsort`（由 F2 触发） | 先确保 YARR/RLBench 链路可导入；`natsort` 缺失再补 |
| F | 动态序列加载（low_dim + PNG） | `Pillow`、`natsort`、RLBench 数据读取链路 | `from PIL import Image` 失败时补 `pillow`；`natsort` 同理按需补 |
| G | `DiffusionPolicyAgent` 训练/推理 | `torch` | 不新增依赖 |
| H1/H2 | 迁移 UNet policy + encoder/common 模块 | 复用 E/D 依赖（含 `zarr`） | 一般无需新增；若 `normalizer.py` 导入报错，回查 `zarr` |
| H3 | Transformer 路线实现与对齐 | 复用 `torch/einops/diffusers` | 一般无需额外补包 |
| I | `normalizer_utils.py` | `numpy`、`zarr` | 与 D/H 共用，不重复补装 |
| J | `shape_meta_utils.py`（obs/action 规格） | 无新增三方包 | 不调整环境 |
| K | 索引式 replay 对齐规则固化 | 无新增三方包 | 不调整环境 |
| L | 自检清单执行 | 依赖上述所有已触发包 | 仅做验证，不新增“重型可选包” |
| M | Transformer/条件注入预留点 | 复用已有依赖 | 基线阶段不额外补包 |

推荐补装节奏（仅在对应 smoke fail 时执行）：
- **批次 1（进入 E/H 前）**：`diffusers==0.11.1`、`einops==0.4.1`。
- **批次 2（进入 D/F/I 前）**：`zarr==2.12.0`、`natsort`、`pillow`（按需）。
- **批次 3（后续扩展）**：`robomimic/robosuite/ray/...` 等 B 层依赖，仅在启用对应功能时引入。

### A. 新增 Agent 目录与文件清单

**新增目录**：`occ_grasp_models/agents/diffusion_policy/`

```
occ_grasp_models/agents/diffusion_policy/
├── __init__.py
├── launch_utils.py              # create_agent + create_replay + fill_replay
├── agent.py                     # YARR Agent: update/act/load/save
├── replay_utils.py              # 索引式 replay + 序列采样/加载
├── normalizer_utils.py          # LinearNormalizer 统计/保存/加载
├── common/                      # pytorch_util / normalize_util
├── policy/
│   ├── diffusion_unet_image_policy.py    # 迁移并适配（当前主用）
│   ├── diffusion_transformer_image_policy.py   # 已落地（obs_as_cond=True）
│   └── cond_wrappers.py                  # 预留（后续条件注入）
├── model/
│   ├── common/                  # normalizer / module mixin / tensor_util
│   ├── diffusion/               # conditional_unet1d / mask_generator / etc.
│   └── vision/                  # multi_image_obs_encoder / crop_randomizer / model_getter
└── configs/
    └── shape_meta_utils.py      # 构建 shape_meta（obs/action 规格）
```

> **说明**：UNet/Transformer 两条路线均依赖 `model/common|diffusion|vision` 与 `common/pytorch_util.py`；条件注入模块仍在后续阶段。

---

### B. 新增方法配置（DIFFUSION_POLICY.yaml）

**新增文件** `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml`

```yaml
1| # @package _group_
2| name: 'DIFFUSION_POLICY'
3|
4| # Agent
5| agent_type: 'bimanual'
6| robot_name: 'bimanual'
7| train_demo_path: ${rlbench.demo_path}
8|
9| # Diffusion backbone switch
10| # - unet:        DiffusionUnetImagePolicy + ConditionalUnet1D
11| # - transformer: DiffusionTransformerImagePolicy + TransformerForDiffusion
12| model_type: 'unet'
13| action_dim: 18
12| horizon: 16
13| n_obs_steps: 4
14| n_action_steps: 8
15| obs_as_global_cond: True
16|
17| # Transformer conditioning (方案B only)
18| obs_as_cond: True
19| pred_action_steps_only: False
20|
21| # 训练超参
18| lr: 1e-4
19| weight_decay: 1e-6
20| optimizer_betas: [0.95, 0.999]
21| transformer_weight_decay: 0.001
22| obs_encoder_weight_decay: 0.000001
23| grad_clip: 1.0
24| num_inference_steps: 100
22|
23| # 视觉输入
24| camera_names: ${rlbench.cameras}
25| image_size: ${rlbench.camera_resolution}
26|
27| # 视觉编码器（MultiImageObsEncoder）
28| rgb_backbone: "resnet50"
29| share_rgb_model: False
30| use_group_norm: True
31| resize_shape: null
32| crop_shape: [76, 76]
33| random_crop: True
34| imagenet_norm: True   # 若为 True，则 image normalizer 设为 identity
35|
36| # UNet architecture
37| diffusion_step_embed_dim: 128
38| down_dims: [512, 1024, 2048]
39| kernel_size: 5
40| n_groups: 8
41| cond_predict_scale: True
42|
43| # Transformer architecture
44| n_layer: 8
45| n_cond_layers: 0
46| n_head: 4
47| n_emb: 256
48| p_drop_emb: 0.0
49| p_drop_attn: 0.3
50| causal_attn: True
51| time_as_cond: True
52|
53| # low-dim 规格（与 action_dim 严格一致：当前帧动作等价 18D）
54| low_dim_size: 18
55|
56| # replay 索引设置
57| index_cache_size: 8        # 缓存若干 episode 的 low_dim（减轻 IO）
58| image_cache_size: 0        # 0 = 不缓存 RGB（可选）
59| index_seed: null           # null=固定前缀；int=确定性随机子集
```

**参数辨析：`n_obs_steps` vs `cfg.replay.timesteps`（不重叠）**

- `n_obs_steps`（方法配置，例：`4`）  
  定义 diffusion policy 每个样本要看的**历史观测窗口长度**（`To`）。  
  它直接决定：
  - `obs_seq = [obs[t], ..., obs[t+n_obs_steps-1]]`
  - `global_cond` 的拼接长度（UNet 分支）
  - 推理窗口起点 `start = n_obs_steps - 1`

- `cfg.replay.timesteps`（全局 replay 配置，`occ_grasp_models/conf/config.yaml` 中默认 `1`）  
  定义 YARR replay 在 `sample_transition_batch()` 返回时的**时间堆叠维度**。  
  它影响的是 `replay_sample` 的张量形状（如 `(B, timesteps, ...)`），不是 diffusion 的 `To/horizon` 语义。

- 推荐口径（本方案）  
  `replay.timesteps` 保持 `1`，由 `IndexSequenceLoader` 用 `(episode_id, timestep=t)` 动态构造 `n_obs_steps/horizon` 序列。  
  即：replay 负责“给索引”，policy/loader 负责“组时序”。

**例子**

- 例 1：`n_obs_steps=4, horizon=16, replay.timesteps=1`  
  replay 采到一个索引 `(ep_id, t=20)`；  
  loader 构造 `obs[20..23]` 和 `action[20..35]`（越界 clamp + `is_pad`）。

- 例 2：把 `replay.timesteps` 改成 `3`，但 `n_obs_steps` 仍是 `4`  
  replay 会返回 shape `(B,3,...)` 的堆叠样本；  
  diffusion 仍然需要 4 帧历史，`n_obs_steps` 不会被 `replay.timesteps` 替代。

**意义**：单一方法配置可稳定控制 UNet/Transformer 双主干切换；主链路已接通。

**补充**：`obs_as_global_cond` 仅服务于 UNet；Transformer 当前固定 `obs_as_cond=True` 且 `time_as_cond=True`。

---

### C. agent_factory.py 接入 DIFFUSION_POLICY

**文件** `occ_grasp_models/agents/agent_factory.py`

#### C1. supported_agents 增加新方法

**修改位置**：`supported_agents["bimanual"]`（当前文件约 L11-L21）

**旧代码**
```python
11| supported_agents = {
12|     "leader_follower": (),
13|     "independent": (),
14|     "bimanual": (
15|         "BIMANUAL_PERACT", "ACT_BC_VISION", "ACT_BC_ENC",
16|         "ACT_BC_KEY", "ACT_BC_KEYPOINT",
17|         "ACT_BC_KEYPOINT_STRATEGY", "ACT_BC_ENC_STRATEGY",
18|         "ACT_BC_ENC_KEYPOINT", "ACT_BC_ENC_KEYPOINT_STRATEGY",
19|     ),
20|     "unimanual": (),
21| }
```

**新代码（新增 DIFFUSION_POLICY）**
```python
14|     "bimanual": (
15|         "BIMANUAL_PERACT", "ACT_BC_VISION", "ACT_BC_ENC",
16|         "ACT_BC_KEY", "ACT_BC_KEYPOINT",
17|         "ACT_BC_KEYPOINT_STRATEGY", "ACT_BC_ENC_STRATEGY",
18|         "ACT_BC_ENC_KEYPOINT", "ACT_BC_ENC_KEYPOINT_STRATEGY",
19|         "DIFFUSION_POLICY",  # 新增
20|     ),
```

**含义**：允许 YARR 框架识别新方法名。

#### C2. agent_fn_by_name 增加创建入口

**修改位置**：`agent_fn_by_name` 末尾 `else` 之前（当前文件约 L112-L117）

**新代码**
```python
113|     elif method_name == "DIFFUSION_POLICY":
114|         from agents import diffusion_policy
115|         return diffusion_policy.launch_utils.create_agent
```

---

### D. run_seed_fn.py 接入 DIFFUSION_POLICY（索引式 replay + normalizer）

**文件** `occ_grasp_models/run_seed_fn.py`

**新增分支位置**：建议插入在 `ACT_BC_VISION` 分支之后、`BIMANUAL_PERACT` 分支之前（当前文件约 L283 之前）

**新代码（示意）**
```python
433|     elif cfg.method.name == "DIFFUSION_POLICY":
434|         from agents import diffusion_policy
435|         from agents.diffusion_policy.normalizer_utils import fit_normalizer_from_index_replay
436|
437|         # 1) 创建索引式 replay（只存 episode_id + timestep）
438|         replay_buffer = diffusion_policy.launch_utils.create_replay(
439|             cfg.replay.batch_size,
440|             cfg.replay.timesteps,
441|             cfg.replay.prioritisation,
442|             cfg.replay.task_uniform,
443|             replay_path if cfg.replay.use_disk else None,
444|             cams,
445|             cfg.rlbench.camera_resolution,
446|         )
447|
448|         # 2) 写入索引并返回 episode_index.json 路径
449|         episode_index_path = diffusion_policy.launch_utils.fill_multi_task_replay(
450|             cfg,
451|             obs_config,
452|             rank,
453|             replay_buffer,
454|             tasks,
455|             cfg.rlbench.demos,
456|         )
457|
458|         # 2.5) 显式告知 agent episode_index.json 路径（避免 build 时找不到索引）
459|         if hasattr(agent, 'set_episode_index_path'):
460|             agent.set_episode_index_path(episode_index_path)
461|
462|         # 3) 统计 normalizer（只用 low_dim + action；RGB 归一化按配置选择 identity 或 range）
465|         fitted_normalizer = fit_normalizer_from_index_replay(
466|             cfg=cfg,
467|             replay_buffer=replay_buffer,
468|             sample_size=int(getattr(cfg.method, "normalizer_sample_size", 10000)),
469|             device=f'cuda:{rank}',
470|             episode_index_path=episode_index_path
471|         )
471|
472|         # 4) 注入 normalizer（DiffusionPolicyAgent 内部持有 policy）
473|         if hasattr(agent, 'policy'):
474|             agent.policy.set_normalizer(fitted_normalizer)
475|         elif hasattr(agent, '_pose_agent') and hasattr(agent._pose_agent, 'policy'):
476|             agent._pose_agent.policy.set_normalizer(fitted_normalizer)
```

**含义**：在不改变训练主流程的前提下完成索引式 replay 构建与归一化绑定。

---

### E. launch_utils.py（索引式 replay 构建）

**新增文件** `occ_grasp_models/agents/diffusion_policy/launch_utils.py`

#### E0. __init__.py 注册（与现有 agent 对齐）

**新增文件** `occ_grasp_models/agents/diffusion_policy/__init__.py`

```python
1| """DIFFUSION_POLICY agent package."""
2| import agents.diffusion_policy.launch_utils
```

**含义**：保持与其他 agent 相同的包注册习惯。

#### E1. create_agent：构建 policy + agent

```python
 1| def create_agent(cfg: DictConfig):
 2|     action_dim = int(cfg.method.action_dim)
 3|     low_dim_size = int(cfg.method.low_dim_size)
 4|     robot_name = str(getattr(cfg.method, "robot_name", "bimanual"))
 5|     if low_dim_size != action_dim:
 6|         raise ValueError("DIFFUSION_POLICY requires low_dim_size == action_dim")
 7|     if robot_name != "bimanual":
 8|         raise ValueError("DIFFUSION_POLICY currently supports only robot_name='bimanual'")
 9|
10|     shape_meta = shape_meta_utils.build_shape_meta(
11|         camera_names=cfg.method.camera_names,
12|         image_size=cfg.method.image_size,
13|         low_dim_size=low_dim_size,
14|         action_dim=action_dim,
15|     )
16|
17|     model_type = str(getattr(cfg.method, "model_type", "unet")).lower()
18|     obs_encoder = _build_obs_encoder(cfg, shape_meta)
19|     noise_scheduler = _build_ddpm_scheduler(cfg)
20|
21|     if model_type == "unet":
22|         policy = DiffusionUnetImagePolicy(...)
23|     elif model_type == "transformer":
24|         if not bool(getattr(cfg.method, "obs_as_cond", True)):
25|             raise ValueError("transformer route supports only obs_as_cond=True")
26|         if not bool(getattr(cfg.method, "time_as_cond", True)):
27|             raise ValueError("transformer route requires time_as_cond=True")
28|         policy = DiffusionTransformerImagePolicy(...)
29|     else:
30|         raise NotImplementedError(
31|             f"Unsupported model_type '{cfg.method.model_type}'. Expected one of ['unet', 'transformer']."
32|         )
33|
34|     return PreprocessAgent(
35|         pose_agent=DiffusionPolicyAgent(policy=policy, cfg=cfg),
36|         norm_rgb=False,
37|     )
```

**含义**：基于同一入口完成 UNet/Transformer 双主干实例化，并通过 `PreprocessAgent(norm_rgb=False)` 避免与策略内部 normalizer 重复归一化。

**补充：builder helpers（放在同文件内）**
```python
25| def _build_ddpm_scheduler(cfg):
26|     return DDPMScheduler(
27|         num_train_timesteps=100,
28|         beta_start=0.0001,
29|         beta_end=0.02,
30|         beta_schedule="squaredcos_cap_v2",
31|         variance_type="fixed_small",
32|         clip_sample=True,
33|         prediction_type="epsilon",
34|     )
```

```python
38| def _build_obs_encoder(cfg, shape_meta):
39|     rgb_model = model_getter.get_resnet(name=cfg.method.rgb_backbone, weights=None)
40|     return MultiImageObsEncoder(
41|         shape_meta=shape_meta,
42|         rgb_model=rgb_model,
43|         resize_shape=cfg.method.resize_shape,
44|         crop_shape=cfg.method.crop_shape,
45|         random_crop=cfg.method.random_crop,
46|         use_group_norm=cfg.method.use_group_norm,
47|         share_rgb_model=cfg.method.share_rgb_model,
48|         imagenet_norm=cfg.method.imagenet_norm,
49|     )
```

**含义**：将双主干共享的 obs encoder 构造封装在 `launch_utils` 中，避免分散配置依赖。

**必要 import（同文件）**
- `from diffusers.schedulers.scheduling_ddpm import DDPMScheduler`
- `from agents.diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy`
- `from agents.diffusion_policy.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy`
- `from agents.diffusion_policy.agent import DiffusionPolicyAgent`
- `from helpers.preprocess_agent import PreprocessAgent`
- `from agents.diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder`
- `from agents.diffusion_policy.model.vision import model_getter`
- `from agents.diffusion_policy.configs import shape_meta_utils`

#### E2. create_replay：只存索引

```python
 1| def create_replay(batch_size, timesteps, prioritisation, task_uniform,
 2|                   save_dir, cameras, image_size):
 3|     # 仅存索引（episode_id + timestep），避免保存序列数据
 4|     observation_elements = [
 5|         ObservationElement('episode_id', (), np.int32),
 6|         ObservationElement('timestep', (), np.int32),
 7|     ]
 8|
 9|     extra_replay_elements = [
10|         ReplayElement('task', (), str),  # 仅用于 task-uniform 采样（不会进入 batch）
11|         ReplayElement('demo', (), bool),
12|     ]
13|
14|     # action/reward/terminal/timeout 只是占位（训练时不使用）
15|     replay_buffer = TaskUniformReplayBuffer(
16|         save_dir=save_dir,
17|         batch_size=batch_size,
18|         timesteps=timesteps,
19|         replay_capacity=int(3e5),
20|         action_shape=(1,),               # 占位即可
21|         action_dtype=np.float32,
22|         reward_shape=(),
23|         reward_dtype=np.float32,
24|         update_horizon=1,
25|         observation_elements=observation_elements,
26|         extra_replay_elements=extra_replay_elements,
27|     )
28|     return replay_buffer
```

**含义**：replay 仅存索引元信息，数据加载在训练时动态完成。

#### E3. fill_multi_task_replay：按 episode 生成索引

```python
30| def fill_multi_task_replay(cfg, obs_config, rank, replay, tasks, num_demos):
31|     # 1) 构建 episode 索引（确定 episode_id -> path 映射）
32|     dataset_root = getattr(cfg.method, "train_demo_path", cfg.rlbench.demo_path)
33|     episode_index = replay_utils.build_episode_index(
34|         dataset_root=dataset_root,
35|         tasks=tasks,
36|         num_demos=num_demos,
37|         index_seed=getattr(cfg.method, "index_seed", None),
37|     )
39|
40|     # 2) 保存 index 到磁盘（保证 agent 与 replay 一致）
41|     # 注意：save_dir 为空时，fallback 到 cfg.framework.logdir
42|     replay_utils.save_episode_index(episode_index, replay._save_dir, cfg.framework.logdir)
43|
44|     # 3) 为每个 episode 的每个 timestep 写入索引
45|     # demo_augmentation 在索引式 replay 下暂不生效（后续可扩展）
46|     for info in episode_index:
47|         for t in range(info.length):
48|             action_dummy = np.zeros((1,), dtype=np.float32)
49|             reward = 0.0
50|             terminal = (t == info.length - 1)
51|             timeout = False
52|             replay.add(action_dummy, reward, terminal, timeout,
53|                        episode_id=np.int32(info.episode_id),
54|                        timestep=np.int32(t),
55|                        task=info.task,
56|                        demo=True)
```

**含义**：当前实现会为每个 episode 的每个真实 timestep（`t in [0, L-1]`）写入索引；训练样本由索引 + 动态加载构造。

#### E3.1 候选修订：起点采样语义对齐原 diffusion_policy（未落地）

> 下述修订目前还未在代码中启用，保留为后续增量方案。  
> 该修订会同时作用于 **UNet** 与 **Transformer**，因为两者共享同一条 index replay -> `IndexSequenceLoader.build_batch()` 数据链路。

要点：仅改 `fill_multi_task_replay` 还不够。若允许 `t<0`，必须同步修改 action 序列的 pad 规则，才能与 `SequenceSampler` 语义一致。

```python
57| # A) replay_utils.py 新增 helper：复刻 create_indices 的 start 范围规则
58| def iter_logical_starts(length: int, horizon: int, pad_before: int = 0, pad_after: int = 0):
59|     horizon = max(1, int(horizon))
60|     length = int(length)
61|     pad_before = min(max(int(pad_before), 0), horizon - 1)
62|     pad_after = min(max(int(pad_after), 0), horizon - 1)
63|     min_start = -pad_before
64|     max_start = length - horizon + pad_after
65|     for t in range(min_start, max_start + 1):
66|         yield t
```

```python
67| # B) launch_utils.fill_multi_task_replay() 改为写入“逻辑起点”
68| pad_before = int(cfg.method.n_obs_steps) - 1 + int(getattr(cfg.method, "n_latency_steps", 0))
69| pad_after = int(cfg.method.n_action_steps) - 1
70| horizon = int(cfg.method.horizon)
71|
72| for info in episode_index:
73|     starts = list(replay_utils.iter_logical_starts(
74|         length=int(info.length), horizon=horizon,
75|         pad_before=pad_before, pad_after=pad_after,
76|     ))
77|     if len(starts) == 0:
78|         # 与原 SequenceSampler 行为一致：无可用窗口则跳过该 episode
79|         continue
80|     for i, t in enumerate(starts):
81|         replay.add(
82|             action_dummy, 0.0, (i == len(starts) - 1), False,
83|             episode_id=np.int32(info.episode_id),
84|             timestep=np.int32(t),  # 允许负值
85|             task=info.task, demo=True)
```

```python
86| # C) replay_utils.build_action_sequence() 同步修正 pad 判定
87| for k in range(horizon):
88|     target_idx = int(timestep) + k
89|     pad = (target_idx < 0) or (target_idx >= int(info.length))
90|     idx = min(max(target_idx, 0), int(info.length) - 1)
91|     actions.append(_get_action_with_ignore_from_obs(demo[idx]))
92|     is_pad.append(1 if pad else 0)
```

**落地后的预期效果**
- 起点分布与原始 `create_indices` 对齐：`t in [-pad_before, L-horizon+pad_after]`。  
- 前缀样本（`t<0`）与后缀样本（`t+horizon>L`）都被显式覆盖；`is_pad` 同步反映前后缀补齐。  
- UNet/Transformer 无需分别改动，二者共享同一 replay->loader 数据链路。  
- 不改模型结构与主训练框架，仍保持“只存索引”设计。

**安全边界（建议同时写入实施单）**
- `ObservationElement("timestep", (), np.int32)` 已支持负值，无需改 dtype。  
- 保持 `cfg.replay.timesteps=1`，避免引入额外时间堆叠干扰。  
- `policy.compute_loss()` 当前不消费 `is_pad`；该字段先作为语义正确性与调试信息保留。

---

### F. replay_utils.py（索引 + 动态序列加载）

**新增文件** `occ_grasp_models/agents/diffusion_policy/replay_utils.py`

#### F1. Episode 索引结构

```python
 1| @dataclass
 2| class EpisodeInfo:
 3|     episode_id: int         # 索引 id（数值）
 4|     task: str               # 任务名（用于 task-uniform）
 5|     path: str               # episode 绝对路径
 6|     length: int             # demo 长度
```

#### F2. build_episode_index（确定性索引 + 可选随机子集）

```python
 8| def build_episode_index(dataset_root, tasks, num_demos, index_seed=None):
 9|     episode_index = []
10|     episode_id = 0
11|     rng = np.random.default_rng(int(index_seed)) if index_seed is not None else None
11|     for task in tasks:
12|         task_name = task if str(task).endswith(".train") else f"{task}.train"
13|         episodes_root = os.path.join(dataset_root, task_name, "all_variations", "episodes")
14|         if not os.path.isdir(episodes_root):
15|             raise FileNotFoundError(f"episodes_root not found: {episodes_root}")
16|         episode_names = [x for x in os.listdir(episodes_root) if os.path.isdir(os.path.join(episodes_root, x))]
17|         episode_names = list(natsorted(episode_names)) if natsorted is not None else sorted(episode_names)
18|         if num_demos > len(episode_names):
19|             raise ValueError(f"{task}: num_demos={num_demos} > available={len(episode_names)}")
20|         selected = episode_names[:num_demos]  # 默认固定前缀（可复现）
21|         if rng is not None and num_demos < len(episode_names):
22|             subset_idx = np.sort(rng.choice(len(episode_names), size=num_demos, replace=False))
23|             selected = [episode_names[int(i)] for i in subset_idx]
24|         for ep in selected:
21|             ep_path = os.path.join(episodes_root, ep)
22|             length = _get_demo_length(ep_path)
23|             episode_index.append(EpisodeInfo(
24|                 episode_id=episode_id, task=str(task), path=ep_path, length=length))
25|             episode_id += 1
26|     return episode_index
```

#### F3. 动态序列加载（obs_seq / action_seq）

```python
30| def build_obs_sequence(self, info, t, n_obs_steps, cameras, image_size, episode_length):
31|     # t 为序列起点：obs 使用 [t ... t+n_obs_steps-1]，越界则 clamp
32|     demo = self._load_low_dim_demo_cached(info.path)
33|     indices = [min(max(t + i, 0), info.length - 1) for i in range(n_obs_steps)]
34|     obs_seq = {f"{cam}_rgb": [] for cam in cameras}
35|     lowdim_seq = []
36|     for idx in indices:
37|         obs = demo[idx]
38|         _attach_rgb_to_obs(obs, info.path, idx, cameras, image_size)
39|         frame = observation_utils.extract_obs(
40|             obs, cameras, t=idx, channels_last=False,
41|             episode_length=episode_length, robot_name=self.robot_name)
42|         # low_dim_state 使用“当前帧动作等价”18D 本体（与 action 同形状）
43|         frame["low_dim_state"] = _get_action_with_ignore_from_obs(obs)
44|         _strip_ignore_collisions(frame)
45|         frame = _filter_obs_keys(frame, cameras)
46|         for cam in cameras:
47|             obs_seq[f"{cam}_rgb"].append(frame[f"{cam}_rgb"])
48|         lowdim_seq.append(frame["low_dim_state"])
48|     # stack -> (T,C,H,W) / (T,D)
49|     obs_seq = {k: np.stack(v, axis=0).astype(np.float32) for k, v in obs_seq.items()}
50|     obs_seq["low_dim_state"] = np.stack(lowdim_seq, axis=0).astype(np.float32)
51|     return obs_seq
```

```python
55| def build_action_sequence(self, info, t, horizon):
56|     # action 对齐规则：action[k] 与 obs[t+k] 同索引对齐（无 +1 偏移）
57|     demo = self._load_low_dim_demo_cached(info.path)
58|     actions = []
59|     is_pad = []
60|     for k in range(horizon):
61|         idx = t + k
62|         pad = (idx >= info.length)  # 当前实现只对尾部越界打 pad
63|         idx = min(max(idx, 0), info.length - 1)
64|         a18 = _get_action_with_ignore_from_obs(demo[idx])  # 右9 + 左9 = 18D
65|         actions.append(a18)
67|         is_pad.append(1 if pad else 0)
68|     return np.stack(actions, axis=0).astype(np.float32), np.array(is_pad, dtype=np.int32)
```

说明：当前 replay 写入的 `timestep` 本身是非负区间（`0..L-1`），因此 `idx<0` 在现有链路中不会出现。

**含义**：索引式 replay 的样本在训练时动态构造，等价于“按当前索引定义”在线还原序列。

---

#### F4. Episode index 保存/加载 + 缓存策略

```python
70| def save_episode_index(episode_index, replay_dir, fallback_dir):
71|     # 优先写入 replay 保存目录；若为空，写入日志目录
72|     save_dir = replay_dir if replay_dir is not None else fallback_dir
73|     os.makedirs(save_dir, exist_ok=True)
74|     with open(os.path.join(save_dir, "episode_index.json"), "w") as f:
75|         json.dump([asdict(e) for e in episode_index], f, indent=2)
```

```python
80| def load_episode_index(cfg, episode_index_path=None):
81|     # 优先使用显式路径（run_seed_fn 在 fill_replay 后注入）
82|     candidates = []
83|     if episode_index_path is not None:
84|         candidates.append(episode_index_path)
85|     method_path = getattr(cfg.method, "episode_index_path", None)
86|     if method_path is not None:
87|         candidates.append(method_path)
88|     candidates.append(os.path.join(cfg.framework.logdir, "episode_index.json"))
89|     task_folder = "multi" if len(cfg.rlbench.tasks) > 1 else cfg.rlbench.tasks[0]
90|     method_name = cfg.method.name
91|     seed = int(getattr(cfg, "seed", 0))
92|     candidates.append(os.path.join(cfg.replay.path, task_folder, method_name, f"seed{seed}", "episode_index.json"))
93|     for path in candidates:
94|         if path is not None and os.path.exists(path):
95|             with open(path, "r") as f:
96|                 return [EpisodeInfo(**x) for x in json.load(f)]
97|     raise FileNotFoundError("episode_index.json not found in candidate paths.")
```

```python
92| class IndexSequenceLoader:
93|     def __init__(self, cfg, episode_index_path=None):
94|         self.cfg = cfg
95|         self.episode_index = load_episode_index(cfg, episode_index_path=episode_index_path)
96|         self._episode_by_id = {int(info.episode_id): info for info in self.episode_index}
96|         self.lowdim_cache = LRUCache(maxsize=cfg.method.index_cache_size)
97|         self.image_cache = LRUCache(maxsize=cfg.method.image_cache_size)
98|
99|     def build_batch(self, episode_ids, timesteps):
100|         # 组装 batch: {'obs': {...}, 'action': ...}
101|         obs_batch, action_batch, is_pad_batch = [], [], []
102|         for ep_id, t in zip(episode_ids, timesteps):
103|             info = self._episode_by_id[int(ep_id)]
104|             obs_seq = self.build_obs_sequence(info, t, ...)
105|             action_seq, is_pad = self.build_action_sequence(info, t, ...)
106|             obs_batch.append(obs_seq)
107|             action_batch.append(action_seq)
108|             is_pad_batch.append(is_pad)
109|         batch = collate_to_batch(obs_batch, action_batch)
110|         batch["is_pad"] = np.stack(is_pad_batch, axis=0).astype(np.int32)
111|         return batch
109|
110|     def sample_action_lowdim_stats(self, episode_ids, timesteps):
111|         # 仅用于 normalizer 统计，返回展平后的 action/lowdim
112|         actions, lowdims = [], []
113|         for ep_id, t in zip(episode_ids, timesteps):
114|             info = self._episode_by_id[int(ep_id)]
115|             action_seq, _ = self.build_action_sequence(info, t, ...)
116|             obs_seq = self.build_obs_sequence(info, t, ...)
117|             actions.append(action_seq.reshape(-1, action_seq.shape[-1]))
118|             lowdims.append(obs_seq['low_dim_state'].reshape(-1, obs_seq['low_dim_state'].shape[-1]))
119|         return np.concatenate(actions, axis=0), np.concatenate(lowdims, axis=0)
```

**参数辨析：F4 两个函数里的 `timesteps` 是“样本索引值”，不是配置项 `replay.timesteps`**

- `build_batch(self, episode_ids, timesteps)` 与  
  `sample_action_lowdim_stats(self, episode_ids, timesteps)`  
  这里的 `timesteps` 指的是每个样本对应的**真实帧索引 `t`**（序列起点），通常来自 replay 里存的 `timestep` 字段值。

- 语义是“数据内容”而不是“超参数”  
  - `cfg.replay.timesteps`：控制 replay 输出是否带时间堆叠维（shape 维度规则）。  
  - 函数参数 `timesteps`：一组具体的起点值，如 `[10, 25, 44]`，逐个喂给 `build_obs_sequence(info, t, ...)`。

- 典型流程（当前方案）  
  1. replay 采样得到 `replay_sample['episode_id']` 与 `replay_sample['timestep']`。  
  2. 因 `cfg.replay.timesteps=1`，通常取 `[:, -1]`（与当前实现一致）得到一维索引数组。  
  3. 作为 `episode_ids/timesteps` 传入上述函数，动态构造 obs/action 序列。

**例子**

- 假设 batch 中 3 个样本索引是  
  `(ep=2, t=10), (ep=7, t=25), (ep=7, t=44)`，  
  则传入参数是 `episode_ids=[2,7,7]`、`timesteps=[10,25,44]`。  
  函数会分别从各自 episode 的这 3 个起点构造三条训练序列。

**实现细节说明**
- `LOW_DIM_PICKLE` 文件名固定为 `low_dim_obs.pkl`（见 `repos/RLBench/rlbench/backend/const.py`）。
- RGB 图片优先读取 `rgb_{i:04d}.png`；若不存在则回退尝试 `{i}.png`（与当前 loader 实现一致）。
- `_attach_rgb_to_obs()` 只需读取 RGB 并写入 `obs.perception_data`（float32 且范围 [0,1]），避免加载 depth/point_cloud。
- `image_cache_size=0` 表示不缓存 RGB，最小化内存占用。
- `LRUCache` 可用 `collections.OrderedDict` 实现，避免引入额外依赖。

#### F5. replay_utils 辅助函数（IndexSequenceLoader 内部，最小实现）

```python
107| # 当前代码中尚未启用 iter_logical_starts（即未写入负 timestep）
108| # 相关“逻辑起点”方案保留在 E3.1 作为候选修订，便于后续 replay 语义对齐。
```

```python
110| def _get_demo_length(episode_path):
111|     # 只读取 low_dim_obs.pkl 的长度，避免加载 RGB
112|     with open(os.path.join(episode_path, "low_dim_obs.pkl"), "rb") as f:
113|         demo = pickle.load(f)
114|     return len(demo)
```

```python
118| def _load_low_dim_demo_cached(self, episode_path):
119|     # 低维 demo 可缓存，减少重复 pickle 读取
120|     demo = self.lowdim_cache.get(episode_path)
121|     if demo is not None:
122|         return demo
123|     with open(os.path.join(episode_path, "low_dim_obs.pkl"), "rb") as f:
124|         demo = pickle.load(f)
125|     self.lowdim_cache.put(episode_path, demo)
126|     return demo
```

```python
129| def _attach_rgb_to_obs(obs, episode_path, idx, cameras, image_size):
130|     # 仅加载 RGB，输出 float32 [0,1]；文件名支持 rgb_{idx:04d}.png / {idx}.png
131|     for cam in cameras:
132|         img = _load_rgb_image(episode_path, cam, idx, image_size)
134|         obs.perception_data[f"{cam}_rgb"] = img
```

```python
138| def _get_action_from_obs(obs):
139|     # 与现有 agent 对齐：action 取当前帧 gripper pose，并统一四元数规范
140|     r_quat = utils.normalize_quaternion(obs.right.gripper_pose[3:])
141|     if r_quat[-1] < 0: r_quat = -r_quat
142|     l_quat = utils.normalize_quaternion(obs.left.gripper_pose[3:])
143|     if l_quat[-1] < 0: l_quat = -l_quat
144|     right = np.concatenate([obs.right.gripper_pose[:3], r_quat, [obs.right.gripper_open]])
145|     left = np.concatenate([obs.left.gripper_pose[:3], l_quat, [obs.left.gripper_open]])
146|     return np.concatenate([right, left])  # 16D
```

```python
146| def _get_action_with_ignore_from_obs(obs):
147|     a16 = _get_action_from_obs(obs)
148|     r_ignore, l_ignore = _get_ignore_collisions_from_obs(obs)
149|     return _append_ignore_collisions(a16, r_ignore, l_ignore)
```

```python
152| def _get_ignore_collisions_from_obs(obs):
153|     if hasattr(obs, "right") and hasattr(obs, "left"):
154|         if hasattr(obs.right, "ignore_collisions") and hasattr(obs.left, "ignore_collisions"):
155|             return float(obs.right.ignore_collisions), float(obs.left.ignore_collisions)
156|     v = float(getattr(obs, "ignore_collisions", 1.0))
157|     return v, v
```

```python
158| def _append_ignore_collisions(a16, right_ignore=1.0, left_ignore=1.0):
159|     return np.concatenate([a16[:8], [right_ignore], a16[8:], [left_ignore]], axis=0)
```

```python
162| def _strip_ignore_collisions(frame):
163|     # 防止 obs_dict 多出不需要的键
164|     frame.pop('ignore_collisions', None)
164|     frame.pop('right_ignore_collisions', None)
165|     frame.pop('left_ignore_collisions', None)
166|     return frame
```

```python
170| def _filter_obs_keys(frame, cameras):
171|     # 只保留 RGB 与 low_dim_state，避免无用键占用内存
172|     keep = {f"{cam}_rgb" for cam in cameras}
173|     keep.add("low_dim_state")
174|     return {k: v for k, v in frame.items() if k in keep}
```

```python
178| def collate_to_batch(obs_list, action_list):
179|     # obs_list: List[Dict[str, np.ndarray]] -> Dict[str, np.ndarray]
180|     obs_batch = {}
181|     for key in obs_list[0].keys():
182|         obs_batch[key] = np.stack([o[key] for o in obs_list], axis=0)
183|     action_batch = np.stack(action_list, axis=0)
184|     return {'obs': obs_batch, 'action': action_batch}
```

### G. agent.py（DiffusionPolicyAgent）

**新增文件** `occ_grasp_models/agents/diffusion_policy/agent.py`

> 本节给出与当前代码一致的关键实现片段：聚焦 `build/update/act` 主链路与关键辅助函数行为，便于后续审查与回归核对。

#### G0. 接口契约与协同约束

- `DiffusionPolicyAgent` 必须完整实现 YARR `Agent` 抽象接口：  
  `build/update/act/reset/update_summaries/act_summaries/load_weights/save_weights`。
- `IndexSequenceLoader` **不能在 `__init__` 里创建**：  
  因为 `episode_index.json` 是在 `run_seed_fn.py -> fill_multi_task_replay()` 之后才生成。  
  正确做法是：`run_seed_fn.py` 先调用 `agent.set_episode_index_path(...)`，再由 `build()` 创建 `seq_loader`。
- `replay.timesteps` 建议固定为 `1`；若外部误配为 `>1`，`update()` 用“最后一个时间片”提取索引（`[:, -1]`），保证兼容不崩溃。

#### G1. Agent 初始化 / build / reset（完整）

```python
 1| class DiffusionPolicyAgent(Agent):
 2|     def __init__(self, policy, cfg):
 3|         self.policy = policy
 4|         self.cfg = cfg
 5|         self._device = torch.device("cpu")
 6|         self._optimizer = None
 7|         self.seq_loader = None
 8|         self._episode_index_path = None
 9|         self._summaries = {}
10|         self._obs_keys = [f"{cam}_rgb" for cam in cfg.method.camera_names] + ["low_dim_state"]
11|         self.reset()
12|
13|     def set_episode_index_path(self, path: str):
14|         self._episode_index_path = path
15|
16|     def set_aux_eval_cfg(self, cfg):
17|         # 与 eval runner 接口对齐；当前无额外行为
18|         del cfg
19|
20|     def build(self, training: bool, device: torch.device = None):
21|         if device is None:
22|             device = torch.device("cpu")
23|         elif not isinstance(device, torch.device):
24|             device = torch.device(device)
25|
26|         self._device = device
27|         self.policy = self.policy.to(self._device)
28|         self.policy.train(training)
29|
30|         if training:
31|             self.seq_loader = IndexSequenceLoader(
32|                 cfg=self.cfg, episode_index_path=self._episode_index_path
33|             )
34|             optimizer_betas = getattr(self.cfg.method, "optimizer_betas", [0.95, 0.999])
35|             if len(optimizer_betas) != 2:
36|                 raise ValueError("method.optimizer_betas must contain exactly 2 values.")
37|             optimizer_betas = (float(optimizer_betas[0]), float(optimizer_betas[1]))
38|             model_type = str(getattr(self.cfg.method, "model_type", "unet")).lower()
39|
40|             if model_type == "transformer" and hasattr(self.policy, "get_optimizer"):
41|                 self._optimizer = self.policy.get_optimizer(
42|                     transformer_weight_decay=float(getattr(self.cfg.method, "transformer_weight_decay", 0.001)),
43|                     obs_encoder_weight_decay=float(getattr(self.cfg.method, "obs_encoder_weight_decay", 0.000001)),
44|                     learning_rate=float(self.cfg.method.lr),
45|                     betas=optimizer_betas,
46|                 )
47|             else:
48|                 self._optimizer = torch.optim.AdamW(
49|                     self.policy.parameters(),
50|                     lr=float(self.cfg.method.lr),
51|                     weight_decay=float(self.cfg.method.weight_decay),
52|                     betas=optimizer_betas,
53|                 )
54|         else:
55|             self.seq_loader = None
56|             self._optimizer = None
57|         self.reset()
58|
59|     def reset(self):
60|         self._obs_buffer = []
61|         self._action_cache = None
62|         self._action_cache_idx = 0
```

#### G2. update（索引提取 + 动态序列构造 + 训练）

```python
35|     def update(self, step, replay_sample):
36|         del step
37|         if self._optimizer is None:
38|             raise RuntimeError("Agent optimizer is not initialized. build(training=True) first.")
39|         if self.seq_loader is None:
40|             self.seq_loader = IndexSequenceLoader(
41|                 cfg=self.cfg, episode_index_path=self._episode_index_path
42|             )
43|
44|         episode_ids, timesteps = _extract_index_from_replay_sample(replay_sample)
45|         batch_np = self.seq_loader.build_batch(episode_ids, timesteps)
46|         batch = _to_device(batch_np, self._device)
47|
48|         self._optimizer.zero_grad(set_to_none=True)
49|         loss = self.policy.compute_loss(batch)
50|         loss.backward()
51|
52|         grad_clip = float(getattr(self.cfg.method, "grad_clip", 0.0))
53|         if grad_clip > 0:
54|             torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
55|         self._optimizer.step()
56|
57|         loss_v = float(loss.detach().cpu().item())
58|         self._summaries = {"total_losses": loss_v, "loss": loss_v}
59|         return dict(self._summaries)
```

#### G3. act（receding horizon）+ 缺失辅助函数补齐

```python
52|     @torch.no_grad()
53|     def act(self, step, observation, deterministic=False):
54|         del step, deterministic
55|         obs_1step = _extract_latest_obs_step(observation, expected_keys=self._obs_keys)
56|         self._obs_buffer = _push_obs(
57|             self._obs_buffer, obs_1step, n_obs_steps=int(self.cfg.method.n_obs_steps)
58|         )
59|
60|         refresh_len = 0
61|         if self._action_cache is not None:
62|             refresh_len = min(int(self.cfg.method.n_action_steps), int(self._action_cache.shape[0]))
63|         need_refresh = self._action_cache is None or self._action_cache_idx >= max(1, refresh_len)
64|
65|         if need_refresh:
66|             obs_batch = _stack_obs_buffer(self._obs_buffer, n_obs_steps=int(self.cfg.method.n_obs_steps))
67|             obs_batch = _to_device(obs_batch, self._device)
68|             pred = self.policy.predict_action(obs_batch)
69|             action_seq = pred["action"]
70|             if not torch.is_tensor(action_seq):
71|                 action_seq = torch.as_tensor(action_seq)
72|             if action_seq.ndim != 3:
73|                 raise ValueError("policy.predict_action()['action'] must be [B,Ta,Da].")
74|             self._action_cache = action_seq[0].detach()
75|             self._action_cache_idx = 0
76|
77|         action = self._action_cache[self._action_cache_idx]
78|         self._action_cache_idx += 1
79|         action = _sanitize_bimanual_action(action)
80|         return ActResult(action.detach().cpu().numpy())
```

```python
75| def _extract_index_from_replay_sample(replay_sample):
76|     episode_ids = _to_numpy_index(replay_sample["episode_id"])
77|     timesteps = _to_numpy_index(replay_sample["timestep"])
78|     return episode_ids, timesteps
79|
80| def _to_numpy_index(index_tensor):
81|     arr = index_tensor.detach().cpu().numpy() if torch.is_tensor(index_tensor) else np.asarray(index_tensor)
82|     if arr.ndim == 2:
83|         arr = arr[:, -1]
84|     elif arr.ndim != 1:
85|         raise ValueError(f"Expected index ndim 1 or 2, got shape={arr.shape}")
86|     return arr.astype(np.int64, copy=False)
87|
88| def _extract_latest_obs_step(observation, expected_keys):
89|     # 兼容 (B,Ts,...) / (B,...) 输入；并构造 action-equivalent 18D low_dim_state
90|     ...
91|     proprio = _build_action_equivalent_lowdim(out)
92|     if proprio is not None:
93|         out["low_dim_state"] = proprio
94|     elif "low_dim_state" not in out:
95|         raise KeyError("Missing action-equivalent proprioception keys.")
96|     if out["low_dim_state"].shape[-1] != 18:
97|         raise ValueError("DIFFUSION_POLICY expects 18D low_dim_state.")
98|     ...
99|     return filtered
100|
101| def _push_obs(buffer, obs_1step, n_obs_steps):
106|     buffer.append({k: v.detach().cpu() for k, v in obs_1step.items()})
107|     if len(buffer) > n_obs_steps:
108|         buffer = buffer[-n_obs_steps:]
109|     return buffer
110|
111| def _stack_obs_buffer(buffer, n_obs_steps):
112|     assert len(buffer) > 0, "obs buffer is empty"
113|     if len(buffer) < n_obs_steps:
114|         pad = [buffer[0]] * (n_obs_steps - len(buffer))
115|         frames = pad + buffer
116|     else:
117|         frames = buffer[-n_obs_steps:]
118|     keys = frames[0].keys()
119|     out = {}
120|     for k in keys:
121|         # 每帧是 (1, ...)，stack 后变成 (1, To, ...)
122|         out[k] = torch.stack([f[k] for f in frames], dim=1)
123|     return out
124|
125| def _to_device(batch, device):
126|     if isinstance(batch, dict):
127|         return {k: _to_device(v, device) for k, v in batch.items()}
128|     if torch.is_tensor(batch):
129|         if batch.dtype.is_floating_point:
130|             return batch.to(device=device, dtype=torch.float32, non_blocking=True)
131|         return batch.to(device=device, non_blocking=True)
132|     arr = np.asarray(batch)
133|     tensor = torch.as_tensor(arr, device=device)
134|     if tensor.dtype.is_floating_point:
135|         tensor = tensor.to(dtype=torch.float32)
136|     return tensor
137|
138| def _sanitize_bimanual_action(action):
139|     # 18D 动作后处理：四元数归一化 + w>=0 规范化 + gripper/ignore 二值化
140|     ...
```

#### G3.1 运行语义与形状约定（补充说明）

为避免后续实现歧义，G3 的 `act()` 语义在此明确为“**chunked receding horizon**”：
每步都更新观测，但不是每步都重新跑一次扩散采样。

**关键变量（类型与形状）**

- `self._obs_buffer`：`list[dict[str, torch.Tensor]]`  
  - 每个元素表示“一帧观测字典”；每个键值通常是 `(1, ...)` 的 tensor。  
  - 在 `_push_obs` 中被转为 CPU tensor（`detach().cpu()`）。  
  - 列表长度始终 `<= To`，其中 `To = n_obs_steps`。

- `self._action_cache`：`torch.Tensor | None`  
  - 刷新后由 `pred["action"][0]` 得到，形状 `(Ta, Da)`。  
  - `Ta` 是本次可执行动作段长度，`Da` 是动作维度（本方案为 18）。  
  - 推理时每次 `act()` 从中取一条 `(Da,)` 返回。

**符号约定（与代码一致）**

- `Ts`：YARR rollout 传入 `observation` 的历史堆叠长度（外部输入时间维）。  
- `To`：policy 需要的观测窗口长度，固定等于 `n_obs_steps`。  
- `Ta`：`predict_action()` 这次返回的动作段长度（通常等于 `n_action_steps`）。  
- `Da`：动作维度（本方案 18D）。

**逐步执行流程（每次 act 调用）**

1. 从 `observation (1, Ts, ...)` 抽取最新一帧，得到 `curr_obs (1, ...)`；  
2. 将 `curr_obs` 压入 `self._obs_buffer`，只保留最近 `To` 帧；  
3. 当 `need_refresh=True` 时，使用 `_stack_obs_buffer` 得到 `(1, To, ...)` 输入并调用 `policy.predict_action(...)`，刷新 `self._action_cache`；  
4. 从 `self._action_cache` 取当前一条动作 `(Da,)` 输出，`_action_cache_idx += 1`。

**重规划频率如何被控制**

- 重规划触发条件：
  - `self._action_cache is None`（回合起点），或  
  - `self._action_cache_idx >= min(n_action_steps, self._action_cache.shape[0])`。
- 因此有效重规划周期是 `k = min(n_action_steps, Ta)`。  
- 常见情况下 `Ta == n_action_steps`，即“每执行 `n_action_steps` 步后重规划一次”。  
- `min(...)` 保留为防御式写法，处理模型返回长度与配置不一致的情况。

**回合起点观测不足 To 时如何处理**

- 本方案在 `_stack_obs_buffer` 中用首帧左侧补齐：`pad = [buffer[0]] * (To-len(buffer))`。  
- 这与 `diffusion_policy` 的原生做法一致：  
  - 评估侧 `MultiStepWrapper.stack_last_n_obs()` 不足时重复最早帧；  
  - 训练侧 `SequenceSampler` 的 `pad_before` 也用首样本补齐序列前段。  
- 结论：不会在回合起点临时切换为 `n_action_steps=1`；`n_action_steps` 是固定配置。

**与原始 diffusion_policy 执行接口的关系**

- 原始 `diffusion_policy`：`predict_action()` 返回一段动作，`MultiStepWrapper.step(action_seq)` 在一次 env.step 内循环执行整段。  
- 本方案（YARR）：`Agent.act()` 接口每次只能返回一个动作，因此采用 `action_cache` 逐步吐出。  
- 两者本质都可实现“按段重规划”的 receding horizon；差异在环境接口形态，不在策略语义本身。

**实现注意**

- `step` 与 `deterministic` 在当前草案里未参与分支逻辑，后续若引入“每步重规划/温度控制”等策略，应显式在此处扩展。

#### G4. 其余接口（权重 / summaries）

```python
124|     def update_summaries(self):
125|         return [ScalarSummary("DiffusionPolicyAgent/loss", self._summaries.get("loss", 0.0))]
126|
127|     def act_summaries(self):
128|         return []
129|
130|     def load_weights(self, savedir: str):
131|         path = os.path.join(savedir, "diffusion_policy.pt")
132|         state_dict = torch.load(path, map_location=torch.device("cpu"))
133|         self.policy.load_state_dict(state_dict)
134|         self.policy = self.policy.to(self._device)
133|
134|     def save_weights(self, savedir: str):
135|         os.makedirs(savedir, exist_ok=True)
136|         path = os.path.join(savedir, "diffusion_policy.pt")
137|         torch.save(self.policy.state_dict(), path)
```

**含义**：训练与推理都在 agent 内完成序列构造、归一化和执行窗口管理；同时满足 YARR 接口契约，并与 D/F/I 部分无缝衔接。

---

### H. policy/model 迁移与最小适配（UNet Image）

**目标（修订）**：确保 UNet Image baseline 所需组件“原模原样复制”且依赖闭包完整，不因漏迁文件导致隐性运行时错误。

#### H0. 完整迁移清单（UNet Image baseline 必需，按依赖闭包）

**必须复制（不是可选）**

- `policy/base_image_policy.py`  
- `policy/diffusion_unet_image_policy.py`
- `model/diffusion/conditional_unet1d.py`
- `model/diffusion/conv1d_components.py`
- `model/diffusion/positional_embedding.py`
- `model/diffusion/mask_generator.py`
- `model/vision/multi_image_obs_encoder.py`
- `model/vision/crop_randomizer.py`
- `model/vision/model_getter.py`
- `model/common/normalizer.py`
- `model/common/module_attr_mixin.py`
- `model/common/dict_of_tensor_mixin.py`
- `model/common/tensor_util.py`
- `common/pytorch_util.py`
- `common/normalize_util.py`（I 节会复用 image normalizer 逻辑）

**必须补齐包文件**

- `agents/diffusion_policy/__init__.py`
- `agents/diffusion_policy/policy/__init__.py`
- `agents/diffusion_policy/model/__init__.py`
- `agents/diffusion_policy/model/common/__init__.py`
- `agents/diffusion_policy/model/diffusion/__init__.py`
- `agents/diffusion_policy/model/vision/__init__.py`
- `agents/diffusion_policy/common/__init__.py`

> 说明：`conditional_unet1d.py` 依赖 `conv1d_components.py` 与 `positional_embedding.py`；这两个文件若漏迁，代码能 import 到 policy 但在首次 forward 才报错，属于高风险遗漏点。

#### H1. 适配边界（仅做必要调整，不改核心算法）

- **只改 import 根路径**：`diffusion_policy.*` → `agents.diffusion_policy.*`。  
- **不改类名/函数签名**：保持与原仓接口一致，降低回归风险。  
- **不改扩散主流程**：`compute_loss()` / `predict_action()` / `conditional_sample()` 的行为语义保持一致。

#### H2. 行为不变式（迁移后必须保持）

- `obs_as_global_cond=True`：UNet 走 action-only 扩散，不启用 inpainting。  
- `predict_action()` 取动作窗口：`start = n_obs_steps - 1`。  
- `global_cond` 形状：`(B, n_obs_steps * Do)`。  
- 动作维度来自 `shape_meta['action']['shape'][0]`，本方案应为 18。  
- normalizer key 与 obs key 一致：`action`、`low_dim_state`、`{cam}_rgb`。

#### H3. 关键注意与必要修正

- `MultiImageObsEncoder` 的 `random_crop=False` 分支在迁移版已修正：  
  `CenterCrop` 现在正确写入 `this_randomizer`，不会再被后续 `this_normalizer` 逻辑覆盖。  
  后续若继续修改该文件，需保留这一路径语义，避免回归。
- GroupNorm 替换策略需保留：  
  `replace_submodules(BatchNorm2d -> GroupNorm)`；如果通道数不满足整分组条件，需提供兜底分组数（如 `max(1, num_features//16)` 并保证可整除）。
- 图像值域约束需保持：  
  输入必须是 `float32` 且 `[0,1]`，再由 `imagenet_norm` 或 image normalizer 接管。

#### H4. Transformer 已落地（当前边界）

- `model/diffusion/transformer_for_diffusion.py` 与 `policy/diffusion_transformer_image_policy.py` 已落地并接入主链路。
- 迁移实现基于原始 `diffusion_transformer_hybrid_image_policy.py` 做“去 robomimic”重写，保持 `MultiImageObsEncoder` 口径。
- 当前约束：仅支持 `obs_as_cond=True` + `time_as_cond=True`；不启用 inpainting 路径。

#### H5. 迁移后 smoke 检查（建议写入实施清单）

- `import` smoke：逐个 import `policy/model/common/vision` 模块。  
- 构图 smoke：`create_agent()` 能完成 `policy` 实例化。  
- 前向 smoke：随机 batch 执行 `compute_loss()` 一次，确认维度与 device 正常。

---

### I. normalizer_utils.py（LinearNormalizer 统计）

**新增文件** `occ_grasp_models/agents/diffusion_policy/normalizer_utils.py`

```python
 1| def fit_normalizer_from_index_replay(cfg, replay_buffer, sample_size=10000, device="cpu", episode_index_path=None):
 2|     del device
 3|     sample_size = max(1, int(sample_size))
 4|     loader = IndexSequenceLoader(cfg=cfg, episode_index_path=episode_index_path)
 5|
 6|     action_chunks, lowdim_chunks = [], []
 7|     sampled = 0
 8|     per_batch = _infer_batch_size(replay_buffer, fallback=128)
 9|     while sampled < sample_size:
10|         this_batch = min(per_batch, sample_size - sampled)
11|         replay_sample = replay_buffer.sample_transition_batch(batch_size=this_batch, pack_in_dict=True)
12|         episode_ids = _extract_index_vector(replay_sample["episode_id"])
13|         timesteps = _extract_index_vector(replay_sample["timestep"])
14|         actions, lowdims = loader.sample_action_lowdim_stats(episode_ids, timesteps)
15|         action_chunks.append(actions.astype(np.float32, copy=False))
16|         lowdim_chunks.append(lowdims.astype(np.float32, copy=False))
17|         sampled += this_batch
18|
19|     action_array = np.concatenate(action_chunks, axis=0)
20|     lowdim_array = np.concatenate(lowdim_chunks, axis=0)
21|     return _build_normalizer(cfg, action_array, lowdim_array)
```

```python
22| def _extract_index_vector(index_array):
23|     arr = np.asarray(index_array)
24|     if arr.ndim == 1:
25|         return arr.astype(np.int64, copy=False)
26|     if arr.ndim == 2:
27|         return arr[:, -1].astype(np.int64, copy=False)
28|     raise ValueError(f"Unexpected index tensor shape: {arr.shape}")
```

```python
30| def _build_normalizer(cfg, action_array, lowdim_array):
31|     normalizer = LinearNormalizer()
32|     normalizer["action"] = SingleFieldLinearNormalizer.create_fit(action_array, mode="limits", last_n_dims=1)
33|     normalizer["low_dim_state"] = SingleFieldLinearNormalizer.create_fit(lowdim_array, mode="limits", last_n_dims=1)
34|
35|     image_norm_identity = bool(getattr(cfg.method, "imagenet_norm", True))
36|     image_stat = _unit_image_stats()
37|     for cam in cfg.method.camera_names:
38|         key = f"{cam}_rgb"
39|         if image_norm_identity:
40|             normalizer[key] = get_identity_normalizer_from_stat(image_stat)
41|         else:
42|             normalizer[key] = get_image_range_normalizer()
43|     return normalizer
```

```python
45| def _infer_batch_size(replay_buffer, fallback=128):
46|     for attr in ("_batch_size", "batch_size"):
47|         value = getattr(replay_buffer, attr, None)
48|         if callable(value):
49|             try:
50|                 value = value()
51|             except Exception:
52|                 value = None
53|         if value is not None:
54|             return max(1, int(value))
55|     return max(1, int(fallback))
56|
57| def _unit_image_stats():
58|     return {
59|         "min": np.array([0.0], dtype=np.float32),
60|         "max": np.array([1.0], dtype=np.float32),
61|         "mean": np.array([0.5], dtype=np.float32),
62|         "std": np.array([np.sqrt(1.0 / 12.0)], dtype=np.float32),
63|     }
```

**含义**：RGB 归一化与 obs_encoder 配置对齐（imagenet_norm → identity；否则 range），同时避免读取大量 RGB。

**必要 import**
- `from agents.diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer`
- `from agents.diffusion_policy.common.normalize_util import get_identity_normalizer_from_stat, get_image_range_normalizer`
- `import numpy as np`

#### I1. 起源与为何单独实现（修订说明）

I 部分不是“凭空新增算法”，而是把 diffusion_policy 里分散在多处的 normalizer 逻辑，按 occ 的索引式 replay 训练链路重组为一个适配层：

- **统计器来源**：`diffusion_policy/diffusion_policy/model/common/normalizer.py`  
  (`LinearNormalizer` / `SingleFieldLinearNormalizer`)
- **图像 normalizer 规则来源**：`diffusion_policy/diffusion_policy/common/normalize_util.py`  
  (`get_image_range_normalizer` 的同等实现思路)
- **训练接入模式来源**：`diffusion_policy/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`  
  （`dataset.get_normalizer()` -> `policy.set_normalizer(...)`）

#### I2. 在本方案中单独抽成 `normalizer_utils.py` 的理由

- 原仓依赖 `dataset.get_normalizer()`，但本方案使用“索引式 replay + 动态序列加载”，没有 robomimic 那套 dataset 对象。  
- 因此必须在 `run_seed_fn.py` 里显式拟合 normalizer，再注入 policy。  
- 将该流程独立到 `normalizer_utils.py`，可以避免把 normalizer 逻辑散落在 `run_seed_fn.py`/`agent.py`，降低耦合并方便单测。

#### I3. 利用路径（与 D/F/G 的协同）

1. D 节 `run_seed_fn.py` 采样索引并调用 `fit_normalizer_from_index_replay(..., episode_index_path=episode_index_path)`。  
2. I 节函数通过 F 节 `IndexSequenceLoader.sample_action_lowdim_stats(...)` 动态取样统计。  
3. 结果经 `policy.set_normalizer(...)` 注入，随后 G 节 `update()/act()` 共用同一 normalizer。

---

### J. shape_meta_utils.py（obs/action 规格）

**新增文件** `occ_grasp_models/agents/diffusion_policy/configs/shape_meta_utils.py`

```python
 1| def build_shape_meta(camera_names, image_size, low_dim_size, action_dim):
 2|     obs = {}
 3|     for cam in camera_names:
 4|         obs[f'{cam}_rgb'] = {'shape': [3, image_size[1], image_size[0]], 'type': 'rgb'}
 5|     obs['low_dim_state'] = {'shape': [low_dim_size], 'type': 'low_dim'}
 6|     return {
 7|         'obs': obs,
 8|         'action': {'shape': [action_dim]},
 9|     }
```

**含义**：让 UNet Image policy 能正确识别 obs/action 维度结构。

#### J1. 起源与为何单独实现（修订说明）

J 部分的来源不是某个单独 `.py` 脚本，而是 diffusion_policy 配置体系里的 `shape_meta` 约定：

- 主要来源：`diffusion_policy/diffusion_policy/config/task/*_image*.yaml` 中的 `shape_meta` 字段  
- 使用入口：`diffusion_policy/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py` 中 `cfg.policy.shape_meta`

在原仓里，`shape_meta` 常由 task yaml 静态给出；  
在本方案里，camera 列表、分辨率、动作维度由 occ 配置动态决定，因此需要一个显式构造器 `shape_meta_utils.py` 做运行时拼装。

#### J2. 在本方案中这样实现的道理

- 避免手写/复制多份 `shape_meta` yaml，减少配置漂移。  
- 让 `create_agent()` 在单入口里基于 `cfg.method.*` 生成一致的 `shape_meta`。  
- 当前 `create_agent()` 已支持 `model_type in {unet, transformer}`；两条路线复用同一份 obs/action 规格描述。

#### J3. 会被怎么利用（与 F/G/I 协同）

1. E 节 `create_agent()` 调用 `shape_meta_utils.build_shape_meta(...)`。  
2. 该 `shape_meta` 同时驱动：
   - `MultiImageObsEncoder` 的键类型解析（`rgb` / `low_dim`）；
   - `DiffusionUnetImagePolicy` 的 `action_dim/obs_feature_dim` 构建；
   - I 节 normalizer 的键集（`{cam}_rgb`、`low_dim_state`、`action`）。
3. F 节 `_filter_obs_keys()` 与 J 节输出键名需严格一致；若键名不一致，训练会在 normalizer 或 encoder 阶段报 KeyError。

#### J4. 一致性检查建议（落地前）

- 检查 `shape_meta['obs']` 的键集合是否与 F 节 `collate_to_batch()` 输出完全一致。  
- 检查 `shape_meta['action']['shape'][0] == cfg.method.action_dim == 18`。  
- 检查 `low_dim_state` 的长度是否与 `cfg.method.action_dim` 严格一致（本方案为 18），且来源是“当前帧动作等价”构造。

---

### K. 当前索引 replay 语义与已知差异（状态同步）

当前代码行为（`L` 为 episode 长度）：

1. **起点范围**：`t in [0, L-1]`（`fill_multi_task_replay` 逐帧写入）。
2. **obs 序列**：`[t, ..., t+n_obs_steps-1]`，越界 clamp 到 `[0, L-1]`。
3. **action 序列**：`action_seq[k]` 与 `obs[t+k]` 同索引对齐；越界 clamp 到 `[0, L-1]`。
4. **is_pad**：当前仅在 `t+k >= L` 时记为 `is_pad=1`（前缀 `t<0` 未进入当前采样范围）。
5. **推理窗口**：执行窗口起点为 `n_obs_steps-1`（与 `DiffusionUnetImagePolicy`/`DiffusionTransformerImagePolicy` 对齐）。

与原始 `SequenceSampler(create_indices)` 的关系：

- 当前实现与原始 `pad_before/pad_after` 口径**并非严格等价**，主要差异在“episode 起点前缀样本（`t<0`）”未覆盖。
- 该差异是 **replay 语义层** 问题，不是 UNet/Transformer 主干差异；会同时影响两条路线的数据分布一致性。
- 对齐修订方案保留在第 `E3.1`（当前未落地）。

若执行 `E3.1`，需同时满足以下“三点同改”才算完整：

1. `launch_utils.fill_multi_task_replay()`：起点改为 `iter_logical_starts(...)`，允许负 `timestep`。  
2. `replay_utils.build_action_sequence()`：`is_pad` 判定改为前后缀双侧（`target_idx < 0` 或 `>= L`）。  
3. `replay_utils` 新增 `iter_logical_starts()`：对 `pad_before/pad_after` 做 `<= horizon-1` 裁剪，范围与 `create_indices` 一致。

---

### L. 自检清单（调用链完整性）

- **配置入口**：`conf/method/DIFFUSION_POLICY.yaml` → `agent_factory.py` → `run_seed_fn.py`
- **replay 链路**：`create_replay` → `fill_multi_task_replay` → `episode_index.json`
- **训练链路**：`PyTorchReplayBuffer` 采样索引 → `DiffusionPolicyAgent.update` → `IndexSequenceLoader.build_batch` → `policy.compute_loss`
- **推理链路**：`DiffusionPolicyAgent.act` → `policy.predict_action` → receding horizon 执行
- **归一化链路**：`fit_normalizer_from_index_replay` → `policy.set_normalizer` → 训练/推理一致

### M. 备注（后续条件注入/增强预留点）

- `policy/diffusion_transformer_image_policy.py` 与 `TransformerForDiffusion` 已完成落地；不再是占位。
- `cond_wrappers.py` 仍为条件注入预留（策略/阶段/关键点），当前默认不启用。
- `agents/diffusion_policy/condition/` 目录当前尚未创建；条件预测模块仍属于后续增量任务。

---

### N. 第9部分分批实现 / 验证 / conda更新执行蓝图（新增）

**N0. 总原则（执行时必须遵守）**

- **冻结 A0**：不改 `python + torch + torchvision + cuda` 主锚点，不做大版本升级/降级。
- **分批推进**：每一批只实现对应代码块；先实现、再 smoke、缺包再补、补完重测。
- **触发式补装**：只在 smoke 报缺失时补装；已满足版本不重复改动。
- **回归守门**：每一批结束都要做一次现有 ACT/PERACT 路由 import 回归。
- **最终闭环**：所有批次完成后，必须逐条执行 L 节自检。

**N1. 当前环境基线（2026-02-25 快照）**

- `ppi` 环境已实测可 import：`diffusers==0.11.1`、`einops==0.4.1`、`zarr==2.12.0`、`numba==0.56.4`、`dill==0.3.5.1`、`hydra-core==1.2.0`、`omegaconf==2.3.0`、`natsort==8.4.0`、`Pillow==9.5.0`。
- A1 核心依赖当前已齐备，因此第 9 部分落地时默认先“**不改环境**”，以实现与验证为先。
- 若后续实际运行出现缺包或版本冲突，再按下表“触发式 conda 内补装”执行。

**N2. 分批执行表（执行顺序与验收口径）**

| 批次 | 覆盖代码块 | 代码实施目标 | 必做验证（smoke） | 触发式 conda 环境动作（仅失败时） | 批次通过标准 |
| --- | --- | --- | --- | --- | --- |
| 批次 1 | A + B + C | 建目录骨架、加 `DIFFUSION_POLICY.yaml`、打通 `agent_factory.py` 路由 | 1) `from agents import agent_factory` 2) `agent_factory.agent_fn_by_name("DIFFUSION_POLICY")` 可解析 | 默认无动作；若 `omegaconf/hydra` 缺失，再在 `ppi` 环境内补装 | 配置可加载，方法名可被识别 |
| 批次 2 | H + J + E1 | 迁移 UNet policy/model 依赖闭包，完成 `create_agent()` 最小可实例化 | 1) `import agents.diffusion_policy.policy.diffusion_unet_image_policy` 2) `create_agent(cfg)` 可构图 | 若 `diffusers/einops` 导入失败：补 `diffusers==0.11.1`、`einops==0.4.1` | policy 实例化成功，shape_meta 与 action_dim 对齐 |
| 批次 3 | E2 + E3 + F + I + D | 打通索引式 replay、episode index、normalizer 拟合、`run_seed_fn` 分支 | 1) 生成并读取 `episode_index.json` 2) `fit_normalizer_from_index_replay()` 可返回 normalizer 3) replay 采样索引可被 loader 还原为序列 | 若 `zarr/numcodecs/natsort/Pillow` 报缺失，再按最小集合补装（优先 `zarr==2.12.0`） | 索引链路与 normalizer 链路贯通，无 KeyError/导入错误 |
| 批次 4 | G + K + M | 完成 `DiffusionPolicyAgent` 训练/推理语义，并打通 Transformer 路由 | 1) 单 batch `compute_loss + backward` 2) `act()` 连续调用满足 receding horizon | 默认无新增依赖；如有报错优先修代码，不改 A0 | 训练与推理最小闭环跑通 |
| 批次 5 | L（全量） | 按 L 做端到端调用链自检 + 旧方法回归 | 1) L 五条链路逐项通过 2) 现有 ACT_BC_KEY / ACT_BC_ENC 入口 import 回归通过 | 禁止引入 B/C 层重型依赖（`robomimic/ray/real`） | 第 9 节基线迁移验收完成 |

**N3. 触发式补装命令口径（执行模板）**

- 环境内检查模板：`/home/hdliu/miniconda3/envs/ppi/bin/python -c "import diffusers,einops,zarr,numba,dill,hydra,omegaconf,natsort,PIL"`
- 环境内补装模板：`/home/hdliu/miniconda3/envs/ppi/bin/python -m pip install <pkg>==<ver>`
- 禁止动作：`conda install/upgrade` 触发 `torch/torchvision/cuda/python` 大版本漂移。

**N4. 与 L 的闭环关系（实施要求）**

- 每完成一个批次，都先执行该批次 smoke，再进入下一批。
- 批次 5 已完成，已按 L 的五条链路执行最终自检并记录结果。
- 后续实际代码实现必须严格遵循本节（N）顺序推进，避免“先大改环境再排错”。

**N5. 实施状态同步（2026-02-25）**

- **批次完成情况**：批次 1/2/3/4/5 已按 N2 顺序落地。
- **环境约束执行**：未执行 `conda install/upgrade`、未执行 `pip install`，A0 锚点保持不变。
- **L 自检结果**：
  1. 配置入口链路通过：`DIFFUSION_POLICY.yaml -> agent_factory -> run_seed_fn`。
  2. replay 链路通过：`create_replay -> fill_multi_task_replay -> episode_index.json`。
  3. 训练链路通过：索引采样 -> `DiffusionPolicyAgent.update` -> `IndexSequenceLoader.build_batch` -> `policy.compute_loss/backward`。
  4. 推理链路通过：`act()` 连续调用满足 receding horizon（示例：7 次 `act` 触发 3 次 `predict_action`，`n_action_steps=3`）。
  5. 归一化链路通过：`fit_normalizer_from_index_replay -> policy.set_normalizer` 后训练/推理均可用。
- **旧方法回归**：`ACT_BC_KEY` 与 `ACT_BC_ENC` 路由 import 回归通过。
- **执行环境说明**：在当前沙箱中，YARR 原生 `TaskUniformReplayBuffer` 受 `dist/mp` 资源限制，文档内 smoke 使用等价 `FakeReplay` 驱动链路验证；正式代码路径仍保持 `TaskUniformReplayBuffer` 不变。

**N6. 18D 动作等价本体对齐专项说明（2026-02-27）**

- **问题背景**：早期实现中，DIFFUSION_POLICY 的 `low_dim_state` 存在旧 8D 路径（来自 right/left low_dim 合并），而动作监督是 18D（含双臂 `ignore_collisions`），语义和维度不一致，容易导致训练/推理口径漂移与排查混淆。
- **本次调整的统一目标**：将 DIFFUSION_POLICY 的本体感知统一为“当前观测帧对应的动作等价向量”，即 `right(3+4+1+1) + left(3+4+1+1) = 18D`，并保证训练与推理严格使用同一语义。
- **代码改动集合及目的**：
  1. `replay_utils.py`：训练侧 `build_obs_sequence()` 与 `build_action_sequence()` 都改为 `_get_action_with_ignore_from_obs(...)`，使 `obs.low_dim_state` 与 `action` 天然同构，消除旧 8D 合并路径的歧义。
  2. `agent.py`：推理侧在 `act()` 前从 `right/left_gripper_pose + right/left_gripper_open (+ ignore)` 动态构造 18D `low_dim_state`，并做四元数归一化与符号规范化；若最终不是 18D 直接报错，防止错误静默传播。
  3. `launch_utils.py` 与 `DIFFUSION_POLICY.yaml`：增加 `low_dim_size == action_dim` 的一致性约束，并将配置明确为 `low_dim_size: 18`，防止配置层再次引入维度偏差。
  4. 本文档（本 md）同步替换旧描述：将“_merge_bimanual_lowdim 8D”相关口径改为“当前帧动作等价 18D”口径，保证文档与代码行为一致。
- **这些改动合在一起实现的核心目的**：让 DIFFUSION_POLICY 的“观测本体 token（low_dim_state）”与“预测动作 token（action）”在内容、维度、时序语义上完全对齐，从而减少迁移偏差、提高训练稳定性，并降低后续调参与故障定位成本。

---

**阶段性结论（已更新）**：  
1) UNet + Transformer 双主干 baseline 已在 occ_grasp_fall 中落地；  
2) 低维与动作格式按 18D（含 ignore_collisions）对齐；  
3) 索引式 replay 链路已跑通（当前语义与原始 `pad_before/pad_after` 非严格等价，见 K 节）；  
4) 仅用 LinearNormalizer 的归一化策略已打通并通过自检。  
后续保留条件注入扩展（M/B 层）作为增量阶段，不影响当前基线可用性。


## 附录 B. 迁移后 Diffusion Policy 训练与评估全流程指南（occ_grasp_fall）

本附录基于当前已落地代码实现，目标是让你在同一套训练框架下稳定切换 `UNet` 与 `Transformer DDPM` 主干并完成训练/评估闭环。  
默认前提：已激活 `ppi` 环境，工作目录为 `/home/hdliu/occ_grasp_fall/occ_grasp_models`。

### B1. 当前实现边界（先确认）

1. 主干切换入口是 `method.model_type`：
- `unet`：`DiffusionUnetImagePolicy + ConditionalUnet1D`
- `transformer`：`DiffusionTransformerImagePolicy + TransformerForDiffusion`

2. Transformer 路线固定约束（与代码一致）：
- `method.obs_as_cond=True`
- `method.time_as_cond=True`
- 不支持 inpainting 路径（即不支持 `obs_as_cond=False`）

3. 优化器行为（已对齐）：
- `unet`：统一 `AdamW(policy.parameters())`，使用 `method.lr / method.weight_decay / method.optimizer_betas`
- `transformer`：优先走 `policy.get_optimizer(...)` 参数分组，使用 `method.transformer_weight_decay / method.obs_encoder_weight_decay / method.lr / method.optimizer_betas`

4. 环境要求：
- 本次双主干切换不需要新增依赖，`ppi` 无需额外升级。

### B2. 主干切换参数速查

| 场景 | 必填参数 | 建议参数 |
| --- | --- | --- |
| UNet | `method.model_type=unet` | `framework.logdir` 独立于 transformer |
| Transformer | `method.model_type=transformer` `method.obs_as_cond=True` `method.time_as_cond=True` | `framework.logdir` 独立；按需调 `n_layer/n_head/n_emb` |

最小切换原则：
- 只改 `method.model_type` 和 transformer 必要开关，不同时改动数据口径与序列超参。
- UNet 与 Transformer 必须使用不同 `framework.logdir`，防止 checkpoint 覆盖。

### B3. 训练流程（单任务，先跑通）

UNet：

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
python train.py \
  method=DIFFUSION_POLICY \
  method.model_type=unet \
  rlbench.task_name=multi \
  rlbench.tasks='[bimanual_pick_plate]' \
  rlbench.demo_path=/mnt/rlbench_data \
  rlbench.demos=10 \
  rlbench.episode_length=30 \
  rlbench.cameras='[front]' \
  rlbench.camera_resolution='[256,256]' \
  framework.logdir=/home/hdliu/arm_test/dp_unet_runs \
  framework.start_seed=0 \
  framework.seeds=1 \
  ddp.num_devices=1
```

Transformer DDPM：

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
python train.py \
  method=DIFFUSION_POLICY \
  method.model_type=transformer \
  method.obs_as_cond=True \
  method.time_as_cond=True \
  rlbench.task_name=multi \
  rlbench.tasks='[bimanual_pick_plate]' \
  rlbench.demo_path=/mnt/rlbench_data \
  rlbench.demos=10 \
  rlbench.episode_length=30 \
  rlbench.cameras='[front]' \
  rlbench.camera_resolution='[256,256]' \
  framework.logdir=/home/hdliu/arm_test/dp_transformer_runs \
  framework.start_seed=0 \
  framework.seeds=1 \
  ddp.num_devices=1
```

训练输出与自动过程：
- 输出目录：`${framework.logdir}/${rlbench.task_name}/${method.name}/seed{N}`
- 自动执行：indexed replay 构建、`episode_index.json` 写入、normalizer 拟合

### B4. 训练流程（多任务模板）

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
python train.py \
  method=DIFFUSION_POLICY \
  method.model_type=transformer \
  method.obs_as_cond=True \
  method.time_as_cond=True \
  rlbench.task_name=multi \
  rlbench.tasks='[bimanual_pick_plate,bimanual_pick_fork,bimanual_edge_phone]' \
  rlbench.demo_path=/mnt/rlbench_data \
  rlbench.demos=10 \
  framework.logdir=/home/hdliu/arm_test/dp_transformer_multi_runs \
  framework.start_seed=0 \
  framework.seeds=1
```

切回 UNet 只需：
- 改 `method.model_type=unet`
- 去掉 transformer 必要开关覆盖（或保持默认）
- 使用新的 `framework.logdir`

### B5. 评估流程（两种主干通用）

评估核心原则：
- `eval.py` 以 `${logdir}/seed{N}/config.yaml` 作为训练配置基准
- 真正决定主干的是该配置里的 `method.model_type`
- 所以评估阶段最关键是 `framework.logdir + framework.start_seed` 对齐训练

评估命令示例（latest checkpoint）：

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
python eval.py \
  method=DIFFUSION_POLICY \
  rlbench.task_name=multi \
  rlbench.tasks='[bimanual_pick_plate]' \
  rlbench.demo_path=/mnt/rlbench_data \
  rlbench.episode_length=30 \
  rlbench.cameras='[front]' \
  rlbench.camera_resolution='[256,256]' \
  framework.logdir=/home/hdliu/arm_test/dp_transformer_runs \
  framework.start_seed=0 \
  framework.eval_type=last \
  framework.eval_episodes=50 \
  framework.eval_envs=1
```

`framework.eval_type` 支持：
- `last`、`best`、`missing`、`all`
- 或具体步数（如 `15800`）

### B6. 训练与评估必须对齐的参数

1. 方法与机器人口径：
- `method=DIFFUSION_POLICY`
- `method.agent_type='bimanual'`
- `method.robot_name='bimanual'`

2. 主干口径：
- UNet：`method.model_type=unet`
- Transformer：`method.model_type=transformer` 且 `method.obs_as_cond=True`、`method.time_as_cond=True`

3. 数据口径：
- `rlbench.demo_path`、`rlbench.tasks`、`rlbench.demos` 训练评估一致
- 若训练使用了 `method.train_demo_path`，需确认评估不误切数据根目录

4. 视觉与时序口径：
- `rlbench.cameras` 与 `method.camera_names` 对齐
- `rlbench.camera_resolution` 与 `method.image_size` 对齐
- `rlbench.episode_length`、`method.n_obs_steps`、`method.horizon`、`method.n_action_steps` 保持一致

5. 动作/低维口径：
- `method.action_dim=18`
- `method.low_dim_size=18`

6. 运行资源口径：
- 单卡起步建议 `ddp.num_devices=1`
- 评估并发建议 `framework.eval_envs=1`

### B7. 常见问题与快速排查

1. `eval.py` 报找不到 `seed0/config.yaml`：
- 检查 `framework.logdir`、`framework.start_seed` 是否与训练完全一致。

2. 报 `Unsupported model_type`：
- 检查 `method.model_type` 仅为 `unet` 或 `transformer`。

3. 报 `transformer route supports only obs_as_cond=True`：
- 设置 `method.obs_as_cond=True`。

4. 报 `transformer route requires time_as_cond=True`：
- 设置 `method.time_as_cond=True`。

5. 训练时动作/低维维度错误：
- 检查 `method.low_dim_size == method.action_dim == 18`。
- 检查数据中 action-equivalent proprioception 是否与 18D 口径一致。

6. UNet 与 Transformer 评估结果串线：
- 典型原因是共用 `framework.logdir`；改为独立目录并重新训练/评估。

7. 回放加载慢或内存高：
- 先保持 `method.image_cache_size=0`，按需调小 `method.index_cache_size`。

## 附录 A. 迁移 QA 记录

### Q1：多视角 RGB 视觉编码器是否同一种模型？（ACT_BC_KEY vs Diffusion UNet Image）

**结论：编码器类型相近（2D ResNet），但数据组织与聚合方式不同。**

**ACT_BC_KEY（occ_grasp_models）**
- **输入形态**：单帧多相机 RGB，先堆叠为 `(B, N_cam, 3, H, W)`。
  - 入口见 `occ_grasp_models/agents/act_bc_key/act_bc_key_agent.py` 的 `preprocess_images()`。
- **视觉编码方式**：每个相机 **独立 2D ResNet backbone（torchvision）**，得到特征后拼接进入 DETR/Transformer。
  - backbone 见 `occ_grasp_models/agents/act_bc_key/detr/models/backbone.py`。
  - 多相机特征在 `occ_grasp_models/agents/act_bc_key/detr/models/detr_vae.py` 中被逐相机编码并拼接。
- **时序建模**：**不显式建模视频时间维**（只有单帧、多视角）。

**Diffusion UNet Image（diffusion_policy）**
- **输入形态**：多相机 RGB 序列，实际按帧编码（`(B, T, 3, H, W)` 先展平为 `B*T`）。
  - 入口见 `diffusion_policy/diffusion_policy/policy/diffusion_unet_image_policy.py` 中对 `MultiImageObsEncoder` 的调用。
- **视觉编码方式**：`MultiImageObsEncoder` 对每个相机使用 **2D ResNet** 编码，得到每帧特征后拼接；
  - 视觉与低维特征在时间维上被展平为 `global_cond`（`obs_as_global_cond=True`）。
- **时序建模**：**不使用 3D 视频编码**，时间信息通过 `To` 帧特征拼接进入条件。

**两者流程结构图（简化）**

**(A) ACT_BC_KEY：单帧多视角 2D 编码**
```
cam1_rgb  cam2_rgb  ... camN_rgb
    |         |             |
    +---- stack -> (B, N, C, H, W)
                |
      per-cam 2D ResNet backbone
                |
        feature maps / tokens
                |
         concat across cams
                |
      DETR/Transformer (ACT)
                |
         action sequence
```

**(B) Diffusion UNet Image：多视角逐帧 2D 编码**
```
cam1_rgb (B,T,C,H,W)   cam2_rgb (B,T,C,H,W)   ...  camN_rgb
        |                    |                         |
        |-- per-frame 2D ResNet (MultiImageObsEncoder) -|
                         | (per-cam feature)
                         +----------- concat -----------+
                                      |
                     flatten To frames -> global_cond
                                      |
                         Conditional UNet1D (diffusion)
                                      |
                             action sequence (diffusion)
```

**一句话总结**：ACT_BC_KEY 与 Diffusion UNet Image 都是 **2D ResNet 单帧编码**，但前者进入 DETR/Transformer，后者把多帧特征拼成全局条件并用 UNet 扩散动作序列。

## 10. Transformer 路线落地复盘（对照原实现，2026-03-07）

本节以**当前代码事实**为准，回答“Transformer DDPM 迁移是否完善、与 UNet 的差异是否仅替换主干、replay/条件注入是否一致”。

### 10.1 结论先行

- `DIFFUSION_POLICY` 的 Transformer 路线已落地，不是占位实现。
- 差异**不止**“替换主干”：还包括条件接口约束、优化器分组策略、配置字段与创建路由。
- replay 与条件注入链路是 UNet/Transformer 共用的；当前 replay 语义存在已知差异（第 K 节），这不是主干差异。
- 条件注入模块（`cond_wrappers.py`）仍未落地，当前条件仅来自原始观测键（`{cam}_rgb` + `low_dim_state`）。

### 10.2 文件级事实对照（occ 迁移版 vs 原始实现）

| 文件 | 当前迁移版状态 | 与原始实现对照结论 |
| --- | --- | --- |
| `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml` | 已包含 `unet|transformer` 开关、Transformer 结构参数、`obs_as_cond/time_as_cond`、优化器分组参数 | 与原始 UNet/Transformer workspace 的配置语义对齐（按 occ 口径重组） |
| `occ_grasp_models/agents/diffusion_policy/launch_utils.py` | `create_agent` 已支持双分支；Transformer 分支强约束 `obs_as_cond=True` 与 `time_as_cond=True`；返回 `PreprocessAgent(norm_rgb=False)` | 与“去 robomimic + 保留 diffusion policy 归一化口径”目标一致 |
| `occ_grasp_models/agents/diffusion_policy/policy/diffusion_transformer_image_policy.py` | 完整实现：`conditional_sample/predict_action/compute_loss/set_normalizer/get_optimizer` | 对标原 `diffusion_transformer_hybrid_image_policy.py` 的核心训练/采样语义，去除了 robomimic 依赖 |
| `occ_grasp_models/agents/diffusion_policy/model/diffusion/transformer_for_diffusion.py` | 文件已存在且可用 | 主干实现已迁入，不再缺失 |
| `occ_grasp_models/agents/diffusion_policy/model/diffusion/__init__.py` | 已导出 `ConditionalUnet1D/LowdimMaskGenerator/TransformerForDiffusion` | 模块导出已补齐 |
| `occ_grasp_models/agents/diffusion_policy/agent.py` | `model_type=transformer` 时优先调用 `policy.get_optimizer(...)`，按分组 `weight_decay` | 已对齐原 Transformer 的关键优化器语义（非统一 AdamW） |

### 10.3 UNet 与 Transformer 的真实差异（不是仅替换主干）

共有部分（两条路线一致）：

- 观测编码器：`MultiImageObsEncoder`
- 数据入口：`run_seed_fn.py -> index replay -> IndexSequenceLoader`
- normalizer 注入：`fit_normalizer_from_index_replay -> policy.set_normalizer`
- YARR 执行接口：`DiffusionPolicyAgent.update/act`

主干相关差异（当前保留且必要）：

1. 条件接口不同  
   - UNet：默认 `obs_as_global_cond=True`，条件以展平 `global_cond` 注入。  
   - Transformer：固定 `obs_as_cond=True`，观测作为 cond token；当前不支持 `obs_as_cond=False`。
2. 时间条件开关约束不同  
   - Transformer 当前要求 `time_as_cond=True`（在 `launch_utils` 显式检查）。
3. 优化器构造不同  
   - UNet：统一 `AdamW(policy.parameters())`。  
   - Transformer：优先 `policy.get_optimizer(...)`，分开设置 `transformer_weight_decay` 与 `obs_encoder_weight_decay`。

### 10.4 replay 与条件注入口径（对标当前代码）

replay（当前实现）：

- 采样起点仅写入 `t in [0, L-1]`。
- `obs_seq` 与 `action_seq` 都按 clamp 构造；`is_pad` 当前仅标注尾部越界。
- 该实现可稳定训练，但与原始 `pad_before/pad_after` 语义并非严格等价（首段分布差异）。
- 此差异同时作用于 UNet/Transformer，共享同一 replay 数据链路。

条件注入（当前实现）：

- 当前条件来源只有观测本体与图像（`low_dim_state` + RGB）。
- `cond_wrappers.py` 与 condition 子目录仍是预留位；未进入训练/推理主链路。
- 因此，当前“条件注入差异”主要来自 UNet/Transformer 自身条件接口，而非外加策略/阶段/关键点模块。

### 10.5 环境核验与简单测验结论（状态同步）

2026-03-07 本机快速核验：

- `ppi`：`python 3.8.20`、`torch 2.4.1`、`torchvision 0.20.0`、`diffusers 0.11.1`、`einops 0.4.1`、`numba 0.56.4`、`Pillow 9.5.0`。
- `robodiff`：`python 3.9.18`、`torch 1.12.1`、`torchvision 0.13.1`、`diffusers 0.21.4`、`einops 0.4.1`。
- 结论：双主干迁移本身不要求新增三方依赖；按当前 `ppi` 可运行组合继续即可。

### 10.6 当前边界与后续项（与代码一致）

- 已落地：UNet/Transformer 双主干、双分支路由、Transformer 主干与策略实现、分组优化器。
- 未落地：策略/阶段/关键点条件注入模块。
- 已知差异：replay 起点语义尚未对齐原始 `pad_before/pad_after`；修订方案见第 `E3.1`（仍为候选，未实施）。

### 10.7 第10节同步结论

第10节已从“实施计划”改为“落地复盘”。  
凡“Transformer 占位/NotImplemented/文件不存在/审阅后再改”等表述，均已按当前代码状态清除并替换为可核验事实。
