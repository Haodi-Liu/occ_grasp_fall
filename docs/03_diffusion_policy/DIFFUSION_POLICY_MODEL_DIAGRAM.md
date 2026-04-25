# DIFFUSION_POLICY 模型架构梗概图（UNet / Transformer）

> 说明：图中只使用专业术语与抽象概念，不出现代码变量名。  
> 每张图下方提供“概念 ↔ 代码”对照，方便从设计回溯实现。
>
> 代码状态快照（2026-03-16）：
> - 已落地：同一 agent 内支持两种扩散主干切换（UNet / Transformer）。
> - 当前默认配置：Transformer 主干（`model_type: transformer`）。
> - 当前任务形态：双臂 18 维动作、观测窗 5 步、规划窗 52 步、每次执行 26 步。

---

## 1) 总览：整体架构与训练/推理管线

```
                   DIFFUSION_POLICY 架构总览
    ═══════════════════════════════════════════════════════════════════════

    输入（训练）                                      输入（推理）
    ───────────────────────────────────────────────────────────────────────
    多视角图像序列 + 本体状态序列 + 目标动作序列        多视角图像序列 + 本体状态序列

                                  │
                                  ▼
                       ┌──────────────────────────┐
                       │ 多视角观测编码器          │
                       │ (视觉 + 本体融合特征)     │
                       └───────────┬──────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
                  ▼                                 ▼
        ┌──────────────────────┐          ┌──────────────────────┐
        │ 主干A：条件 UNet      │          │ 主干B：条件 Transformer│
        │ 条件调制 / 轨迹锚定    │          │ 条件记忆 / 跨注意力    │
        └───────────┬──────────┘          └───────────┬──────────┘
                    │                                 │
                    └───────────────┬─────────────────┘
                                    ▼
                         ┌──────────────────────────┐
                         │ 动作轨迹逆扩散生成器      │
                         │ (迭代去噪)               │
                         └───────────┬──────────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │ 多步动作规划结果          │
                         └───────────┬──────────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │ 滚动执行与周期性重规划    │
                         └──────────────────────────┘
```

### 概念 ↔ 代码（总览）

| 概念 | 代码位置/组件 | 备注 |
|------|---------------|------|
| 主干切换（UNet / Transformer） | `launch_utils.create_agent()` | 根据 `model_type` 选择策略实现 |
| 统一观测编码器 | `MultiImageObsEncoder` | 多相机视觉特征与低维本体特征拼接 |
| 扩散调度器 | `_build_ddpm_scheduler()` | `DDPMScheduler`，训练步数 100 |
| 训练入口（损失反传） | `DiffusionPolicyAgent.update()` | 调用 `policy.compute_loss()` |
| 推理入口（动作缓存与滚动执行） | `DiffusionPolicyAgent.act()` | 周期性刷新动作序列并逐步执行 |

---

## 2) 详细展开：各模块架构与数据流

### 2.1 监督对象构造（索引回放 → 时序样本）

```
输入：离线演示轨迹库 + 逻辑起点索引
   │
   ▼
┌────────────────────────────────────────────────────┐
│ 观测窗口构造（长度=观测步数）                      │
│ • 多视角图像序列                                   │
│ • 动作等价本体状态序列                             │
└──────────────────┬─────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────┐
│ 动作窗口构造（长度=规划时域）                      │
│ • 双臂 18 维动作序列                               │
│ • 越界位置采用边界复制，并记录 padding 标记         │
└────────────────────────────────────────────────────┘
```

**概念 ↔ 代码**

| 概念 | 代码位置 |
|------|----------|
| 观测窗口与动作窗口切片 | `IndexSequenceLoader.build_obs_sequence()` / `build_action_sequence()` |
| 观测中的本体状态定义为“动作等价状态” | `replay_utils._get_action_with_ignore_from_obs()` |
| 18 维双臂动作构成（位姿、开合、碰撞忽略） | `_get_action_from_obs()` + `_append_ignore_collisions()` |
| 逻辑起点与 padding 语义 | `iter_logical_starts()` + `build_action_sequence()` |

---

### 2.2 观测编码器（视觉与本体融合）

```
每个观测时刻：
   多相机图像 ──► 视觉骨干网络（每相机独立或共享） ──► 视觉特征
   本体状态向量 ───────────────────────────────────────► 本体特征
                                 │
                                 ▼
                     特征拼接得到单时刻融合表示

多时刻堆叠后形成“条件序列”
```

**概念 ↔ 代码**

| 概念 | 代码位置 |
|------|----------|
| 观测字段定义（多相机 + 低维本体） | `configs/shape_meta_utils.build_shape_meta()` |
| 图像特征提取（ResNet 去分类头） | `model/vision/model_getter.get_resnet()` |
| 图像预处理（resize/crop/归一化） | `MultiImageObsEncoder.__init__()` |
| 视觉特征与低维特征拼接 | `MultiImageObsEncoder.forward()` |

---

### 2.3 主干A：UNet 条件扩散

```
扩散轨迹：时间轴上的动作轨迹（默认）

条件注入路径A（默认）
  条件序列 ─► 全局上下文向量 ─► 逐层调制去噪残差块

条件注入路径B（可选）
  条件序列 ─► 与轨迹通道拼接 ─► 前部观测时刻通道锚定

二者共同流程：
  随机扩散时刻加噪 ─► UNet 预测噪声 ─► 调度器迭代逆扩散
```

**概念 ↔ 代码**

| 概念 | 代码位置 |
|------|----------|
| UNet 策略封装 | `policy/diffusion_unet_image_policy.py` |
| 1D 条件 UNet 主体 | `model/diffusion/conditional_unet1d.py` |
| 全局条件调制（默认路径） | `DiffusionUnetImagePolicy.predict_action()/compute_loss()` 中 `obs_as_global_cond=True` 路径 |
| 观测锚定（inpainting）路径 | `obs_as_global_cond=False` 时拼接与 `condition_mask` 约束 |
| 条件掩码生成 | `LowdimMaskGenerator` |

---

### 2.4 主干B：Transformer 条件扩散

```
扩散轨迹：时间轴上的动作轨迹

条件侧：
  扩散时刻语义 + 观测条件序列 ─► 条件编码 ─► 条件记忆

轨迹侧：
  带噪动作轨迹 ─► 因果解码器
                 └► 通过跨注意力读取条件记忆
                      输出每个时刻的噪声估计

调度器迭代逆扩散后得到动作规划结果
```

**概念 ↔ 代码**

| 概念 | 代码位置 |
|------|----------|
| Transformer 策略封装 | `policy/diffusion_transformer_image_policy.py` |
| 条件 Transformer 主体 | `model/diffusion/transformer_for_diffusion.py` |
| 条件序列编码（观测窗） | `DiffusionTransformerImagePolicy.predict_action()/compute_loss()` |
| 条件记忆与轨迹解码交互 | `TransformerForDiffusion.forward()` 中 encoder + decoder + memory |
| 因果约束与条件可见性掩码 | `TransformerForDiffusion.__init__()` 中 `mask` / `memory_mask` |

---

### 2.5 训练与在线执行

```
训练：
  真实动作轨迹 ─► 随机时刻加噪 ─► 预测噪声 ─► 均方误差优化

推理：
  观测缓存达到窗口长度后进行一次完整规划
  执行规划中的前半段动作，再基于最新观测重规划
```

**概念 ↔ 代码**

| 概念 | 代码位置 |
|------|----------|
| 训练损失计算入口 | `DiffusionUnetImagePolicy.compute_loss()` / `DiffusionTransformerImagePolicy.compute_loss()` |
| 推理动作切片（从最近观测末端对齐） | 两个 policy 的 `predict_action()` |
| 动作缓存与滚动重规划 | `DiffusionPolicyAgent.act()` |
| 推理动作后处理（四元数归一化、二值控制） | `_sanitize_bimanual_action()` |

---

## 3) 关键问题的代码结论（客观口径）

### 3.1 扩散的对象是什么？

- 主口径（当前配置与常用路径）：**18 维双臂动作序列**，时域长度为 `horizon`（默认 52）。
- UNet 在可选路径下（非默认）可把“动作 + 观测特征”拼为联合轨迹做扩散，但最终控制输出仍取动作通道。
- Transformer 在可选设置下可只扩散执行窗口长度的动作（`pred_action_steps_only=True`）；默认仍是整段规划时域动作。

### 3.2 注入的条件是什么？

- 条件由两部分构成：**多视角视觉观测** + **动作等价本体状态**。
- 其中本体状态并非任意传感拼接，而是与动作同构的双臂状态（位姿/夹爪开合/忽略碰撞位）。
- 训练时由离线 demo 直接构造；推理时优先从在线观测中的双臂位姿等字段重建该本体状态。

### 3.3 条件如何注入，并如何与扩散对象交互？

- UNet：
  - 默认通过“全局条件调制”影响整条去噪链路（时间嵌入 + 观测全局向量共同调制残差块）。
  - 可选通过“轨迹锚定”把观测特征写入联合轨迹的前部时刻并在每个扩散步强制保持。
- Transformer：
  - 条件序列先编码为“条件记忆”，带噪动作轨迹在解码阶段通过跨注意力读取条件记忆完成去噪。
  - 因果掩码约束轨迹内部时序可见性；条件可见性由记忆掩码控制。

### 3.4 与数据管线相关的客观补充

- 回放样本会生成 `is_pad`（边界补齐标记），但当前两个 `compute_loss()` 实现未使用该标记做额外损失屏蔽；当前损失屏蔽仅由条件掩码控制。

---

## 4) 依据来源（脚本与大致行号）

| 主题 | 依据文件与行号（大致） |
|------|------------------------|
| 主干切换、配置约束（UNet/Transformer） | `occ_grasp_models/agents/diffusion_policy/launch_utils.py:28-113` |
| 方法默认超参与时域设定 | `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml:10-24, 74-93` |
| 观测字段定义（多相机 + 低维） | `occ_grasp_models/agents/diffusion_policy/configs/shape_meta_utils.py:4-25` |
| 观测编码与融合 | `occ_grasp_models/agents/diffusion_policy/model/vision/multi_image_obs_encoder.py:19-203` |
| UNet 扩散对象与条件注入路径 | `occ_grasp_models/agents/diffusion_policy/policy/diffusion_unet_image_policy.py:34-277` |
| UNet 条件调制机制（FiLM） | `occ_grasp_models/agents/diffusion_policy/model/diffusion/conditional_unet1d.py:14-241` |
| Transformer 条件注入与去噪交互 | `occ_grasp_models/agents/diffusion_policy/policy/diffusion_transformer_image_policy.py:18-249` |
| Transformer 记忆-解码结构与掩码 | `occ_grasp_models/agents/diffusion_policy/model/diffusion/transformer_for_diffusion.py:13-294` |
| 训练样本构造（动作序列/观测序列/padding） | `occ_grasp_models/agents/diffusion_policy/replay_utils.py:246-307, 457-508` |
| 在线推理本体重建与滚动执行 | `occ_grasp_models/agents/diffusion_policy/agent.py:186-223, 356-455, 610-633` |

