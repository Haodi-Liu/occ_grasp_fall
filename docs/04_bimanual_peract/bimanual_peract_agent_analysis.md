# Bimanual PerAct Agent 解析与迁移说明（中文）

## 范围与来源

本文是对以下目录中 bimanual PerAct agent 的系统性解析：

- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/

主要源码文件：

- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/qattention_peract_bc_agent.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/perceiver_lang_io.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/qattention_stack_agent.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/launch_utils.py

该 agent 依赖的支持模块：

- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/voxel/voxel_grid.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/voxel/augmentation.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/helpers/optim/lamb.py

在目标仓库（occ_grasp_fall）中的相关上下文：

- /home/hdliu/occ_grasp_fall/occ_grasp_models/agents/agent_factory.py
- /home/hdliu/occ_grasp_fall/occ_grasp_models/agents/replay_utils.py
- /home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/observation_utils.py
- /home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/utils.py
- /home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/preprocess_agent.py

## 高层架构

bimanual PerAct agent 的主线是“多视角感知 -> 体素化 -> Perceiver Q 评估 -> 双臂动作合成”。系统以 PreprocessAgent 作为输入入口，内部由 QAttentionStackAgent 组织多个单层 QAttentionPerActBCAgent，每个单层 agent 通过 QFunction 体素化并调用 PerceiverVoxelLangEncoder 预测动作 Q 值，最后将离散动作整合成连续双臂动作。

为了便于理解，下面按数据流顺序介绍主要模块与相互关系。

### PreprocessAgent 是什么？它做什么

PreprocessAgent 是输入适配层，它包裹真正的控制器（QAttentionStackAgent），在 update/act 之前完成 RGB 归一化与类型转换。它不改变动作逻辑，但决定了视觉输入的尺度与分布，直接影响后续体素化与 Q 网络的稳定性。

### QAttentionStackAgent 是什么？它做什么

QAttentionStackAgent 是“多层控制器”，内部持有一个或多个 QAttentionPerActBCAgent。它按层依次调用每个单层 agent：上一层预测出的注意力坐标会被写回 observation，作为下一层的聚焦区域输入。最终它把各层输出的离散动作拼接、转换为连续双臂动作（xyz + quaternion + gripper + ignore_collisions），并可生成像素坐标用于可视化或调试。

### QAttentionPerActBCAgent 是什么？它做什么

QAttentionPerActBCAgent 是单层 PerAct 的训练与推理核心。它负责：

- 构建并持有 QFunction 和 PerceiverVoxelLangEncoder。
- 训练时读取 replay 中的离散动作、低维状态与语言/目标图像特征，计算平移/旋转/夹爪/碰撞的分类损失，并更新优化器。
- 推理时对 Q 分布做 softmax 与 argmax，输出该层的离散动作索引。
- 统一管理语言开关（no_language）、目标图像开关（use_goal_image），以及日志与可视化数据。

### QFunction 是什么？它做什么

QFunction 负责把多视角 RGB+点云转为体素网格，并调用 PerceiverVoxelLangEncoder 输出右/左手的 Q 值。它是 QAttentionPerActBCAgent 与 Perceiver 主干之间的桥梁，封装了体素化与网络前向过程。

### VoxelGrid 体素化模块是什么？它做什么

VoxelGrid 将点云坐标映射到规则 3D 网格，并把 RGB 特征与占用信息融合为体素特征。QFunction 直接依赖该模块，体素化的质量决定了 Perceiver 的输入表示。

### Perceiver-based Q network 是什么？它做什么

Perceiver-based Q network 指的是 PerceiverVoxelLangEncoder 这条“Q 网络”主干，它的职责是：

- 输入：体素化后的 3D 场景（RGB+点云）+ 本体状态（proprio），可选拼接语言特征或目标图像特征。
- 编码：通过 Perceiver 的 cross-attention/self-attention 把大体素序列压缩到一组 latent 表示。
- 解码：从 latent 还原回体素网格并输出动作 Q 值：
  - 右/左手的平移 Q（每个 voxel 一个分数）
  - 右/左手的旋转/夹爪/碰撞 Q（离散分类分数）
- 决策：对 Q 取 argmax 得到离散动作索引（位置、姿态、夹爪、是否忽略碰撞）。

一句话：它把“当前场景”映射成“每个候选动作的打分”，再选最高分动作。

### 目标图像编码器（Goal Image Encoder）是什么？它做什么

目标图像编码器是 PerceiverVoxelLangEncoder 内部的轻量 CNN，用于把目标图像压缩成与体素特征同维度的向量，并在 3D 网格上广播后与体素特征拼接。它只在 use_goal_image=True 时启用，为 Q 网络提供“视觉目标条件”。

### 体素卷积编码/上采样模块（Conv3DBlock / Conv3DUpsampleBlock）是什么？它做什么

这些 3D 卷积模块负责体素特征的尺度变换：前端卷积与 patchify 将高分辨率体素压缩成 Perceiver 更易处理的 token 序列；后端上采样把 latent 表征还原到体素网格，为平移 Q 头输出高分辨率 Q 值。

### 3D SpatialSoftmax 与池化模块（SpatialSoftmax3D / MaxPool3d / AdaptiveMaxPool3d）是什么？它做什么

这些模块把 3D 特征聚合成全局向量，用于旋转/夹爪/碰撞的分类头。它们与体素卷积特征共同构成“全局语义特征”，供后续 MLP 头使用。

### 旋转/夹爪/碰撞 MLP 头（DenseBlock Heads）是什么？它做什么

这是 Perceiver 输出后的离散动作分类头，左右手各一套。它把聚合后的全局特征映射为旋转角度、夹爪开合、碰撞忽略等离散类别的 logits，用于最终离散决策。

### CLIP 文本编码器（可选）是什么？它做什么

语言模式下会使用 CLIP 的文本编码器把语言指令转换为向量（句子级与 token 级嵌入）。这些向量被 PerceiverVoxelLangEncoder 融合到 Q 网络中，作为语言条件。若 no_language=True，则该路径被完全跳过或置零。

### SE(3) 增强模块是什么？它做什么

SE(3) 增强在训练时对点云与动作标签进行一致性扰动，使模型对小幅平移与旋转更鲁棒。它由 QAttentionPerActBCAgent 调用，作用在体素化之前，是数据增强的重要一环。

### launch_utils.create_agent 是什么？它做什么

launch_utils.create_agent 是装配入口：它根据配置实例化 PerceiverVoxelLangEncoder 与 QAttentionPerActBCAgent（可多层），再包装成 QAttentionStackAgent，并最终套上 PreprocessAgent 作为对外接口。也就是说，训练与评估脚本看到的 agent 就是这个封装后的入口。

## 框架流程图（训练与评估）

训练流程（update）：

```
[Replay Batch: rgb/pcd/low_dim/actions/lang_emb]
            |
            v
     [PreprocessAgent]
            |
            v
   [QAttentionStackAgent]  (depth loop)
            |
            v
 [QAttentionPerActBCAgent]
     |-- (可选) SE(3) 增强
     |-- [QFunction]
           |-- [VoxelGrid 体素化]
           |-- [PerceiverVoxelLangEncoder]
                 |-- (可选) Goal Image Encoder
                 |-- (可选) Language Embeddings (来自 replay)
           |-- Q 值 (right/left)
     |-- 损失计算 + 优化器更新
     |-- prev_layer_voxel_grid/bounds (供下一层使用)
            |
            v
输出: total_losses + summaries
```

评估流程（act）：

```
[Observation: rgb/pcd/low_dim/lang_goal_tokens]
            |
            v
     [PreprocessAgent]
            |
            v
   [QAttentionStackAgent]  (depth loop)
            |
            v
 [QAttentionPerActBCAgent]
     |-- (可选) CLIP 编码 lang_goal_tokens
     |-- [QFunction]
           |-- [VoxelGrid 体素化]
           |-- [PerceiverVoxelLangEncoder]
                 |-- (可选) Goal Image Encoder
           |-- Q 值 (right/left)
     |-- argmax 得到离散动作 + attention 坐标
     |-- prev_layer_voxel_grid/bounds (供下一层使用)
            |
            v
[QAttentionStackAgent 合并层结果]
            |
            v
输出: 连续双臂动作 + 观测辅助信息
```

## 数据流（训练与推理）

训练（update）：

1. PreprocessAgent 对 replay batch 中的 RGB 做归一化，并统一类型。
2. QAttentionPerActBCAgent 读取离散动作、低维状态与语言/目标图像特征，必要时执行 SE(3) 增强。
3. QFunction 体素化多视角输入并调用 PerceiverVoxelLangEncoder，得到右/左手的 Q 分布。
4. QAttentionPerActBCAgent 用交叉熵分别监督平移、旋转、夹爪、碰撞分支，更新优化器并生成训练摘要。
5. 若是多层配置，上一层输出的注意力信息会被写回 replay 样本供下一层使用。

推理（act）：

1. PreprocessAgent 归一化观测输入。
2. QAttentionStackAgent 逐层调用单层 QAttentionPerActBCAgent，获取每层注意力坐标与离散动作。
3. QFunction 体素化并由 PerceiverVoxelLangEncoder 产生 Q 值，softmax 后 argmax 得到离散动作。
4. QAttentionStackAgent 将离散动作整合为连续双臂动作，并补充像素坐标等辅助信息。

## 迁移到 occ_grasp_fall 的文件清单与修改建议

### 必需复制的文件

创建新目录：

- /home/hdliu/occ_grasp_fall/occ_grasp_models/agents/bimanual_peract/

从源仓库复制：

- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/__init__.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/launch_utils.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/perceiver_lang_io.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/qattention_peract_bc_agent.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/qattention_stack_agent.py

复制 voxel 模块（必需）：

- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/voxel/__init__.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/voxel/voxel_grid.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/voxel/augmentation.py

目标放置位置：

- /home/hdliu/occ_grasp_fall/occ_grasp_models/voxel/（需包含 __init__.py）

复制 LAMB 优化器（若配置使用 lamb）：

- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/helpers/optim/__init__.py
- /home/hdliu/zeyu_haodi_grasp/peract_bimanual/helpers/optim/lamb.py

目标放置位置：

- /home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/optim/（需包含 __init__.py）

### 可选复制

- *_backup.py 备份文件仅用于参考，可不复制。

### 复制后需修改的代码（含行号与理由）

以下行号以源仓库文件为准（/home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract）。复制到 occ_grasp_fall 后请做对应修改（提供可直接落地的 diff）：

1. /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/qattention_peract_bc_agent.py:55-56
   - 修改：DDP 包装时的 device_ids 需兼容 torch.device，避免传入非 int。
   - 理由：occ_grasp_fall 的训练/评估流程可能传 torch.device，DDP 期望 GPU index（int）。
   - 建议 diff：

```diff
-        if training:
-            self._qnet = DDP(self._qnet, device_ids=[device])
+        if training:
+            ddp_device_ids = None
+            if isinstance(device, torch.device):
+                if device.type == "cuda":
+                    ddp_device_ids = [device.index]
+            elif isinstance(device, int):
+                ddp_device_ids = [device]
+            if ddp_device_ids is not None:
+                self._qnet = DDP(self._qnet, device_ids=ddp_device_ids)
```

2. /home/hdliu/zeyu_haodi_grasp/peract_bimanual/agents/bimanual_peract/qattention_peract_bc_agent.py:885-889
   - 修改：当 `lang_goal_tokens` 缺失且 no_language=False 时回退零向量，并记录警告。
   - 理由：occ_grasp_fall 的配置可能将 `include_lang_goal_in_obs=False`，评估时无 tokens 会直接断言失败。
   - 建议 diff：

```diff
-            with torch.no_grad():
-                assert self._clip_rn50 is not None, "CLIP not loaded but no_language=False"
-                assert lang_goal_tokens is not None, "Missing lang_goal_tokens in observation"
-                lang_goal_tokens = lang_goal_tokens.to(device=self._device).long()
-                lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])
+            with torch.no_grad():
+                assert self._clip_rn50 is not None, "CLIP not loaded but no_language=False"
+                if lang_goal_tokens is None:
+                    logging.warning("Missing lang_goal_tokens; using zero language embeddings.")
+                    lang_goal_emb = torch.zeros((1, 512), device=self._device)
+                    lang_token_embs = torch.zeros((1, 77, 512), device=self._device)
+                else:
+                    lang_goal_tokens = lang_goal_tokens.to(device=self._device).long()
+                    lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])
```

### 配置与集成更新

1. 方法配置：
   - 复制 /home/hdliu/zeyu_haodi_grasp/peract_bimanual/conf/method/BIMANUAL_PERACT.yaml
     到 /home/hdliu/occ_grasp_fall/occ_grasp_models/conf/method/BIMANUAL_PERACT.yaml
   - 更新配置内路径（如 goal_image_path）以匹配 occ_grasp_fall。

2. 若需要设置为默认 method：
   - /home/hdliu/occ_grasp_fall/occ_grasp_models/conf/config.yaml
   - /home/hdliu/occ_grasp_fall/occ_grasp_models/conf/eval.yaml

3. agent_factory 已包含 BIMANUAL_PERACT 分支，确保
   agents.bimanual_peract import 可用即可。

### 兼容性修改清单（必要）

1. 模块缺失：
   - occ_grasp_fall 目前无 voxel 与 helpers/optim，必须复制以满足 import：
     - from voxel.voxel_grid import VoxelGrid
     - from helpers.optim.lamb import Lamb

2. 依赖包：
   - perceiver_pytorch（PerceiverVoxelLangEncoder 使用）
   - pytorch3d（voxel/augmentation 使用）
   - transformers, einops, torchvision, PIL
   - ipdb 仅用于调试，若未安装可去除或包裹。
   - 注意：qattention_peract_bc_agent.py 顶部直接 import torchvision，因此即使不启用目标图像也需要 torchvision。

3. 语言输入：
   - 训练：replay_utils 生成 lang_goal_emb 与 lang_token_embs。
   - 评估：RLBench env 需要 include_lang_goal_in_obs=True 才有 lang_goal_tokens。

4. low_dim_size：
   - QAttentionPerActBCAgent 会拼接右/左 low_dim_state。
   - cfg.method.low_dim_size 应与拼接后的维度一致（如 4+4=8）。

5. 动作格式：
   - QAttentionStackAgent 输出 xyz+quat+gripper+ignore_collisions（双臂）。
   - 确认 RLBench action mode 与动作向量维度匹配。

6. DDP 设备：
   - QFunction 在 training 时包装 DDP，device_ids=[device]。
   - 若 device 为 torch.device，需确保 device_ids 使用 GPU index。

7. 目标图像路径：
   - 若启用 use_goal_image，需要在 occ_grasp_fall 中更新 goal_image_path。

8. 运行路径：
   - 需确保运行时工作目录为 /home/hdliu/occ_grasp_fall/occ_grasp_models，或将该目录加入 PYTHONPATH，保证 import voxel/helpers 生效。

### 集成检查清单（occ_grasp_fall）

- /home/hdliu/occ_grasp_fall/occ_grasp_models/agents/bimanual_peract 已存在且含 __init__.py。
- /home/hdliu/occ_grasp_fall/occ_grasp_models/voxel 已存在且可 import。
- /home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/optim 已存在并导出 Lamb。
- /home/hdliu/occ_grasp_fall/occ_grasp_models/conf/method/BIMANUAL_PERACT.yaml 已加入。
- agents.bimanual_peract 在 agent_factory 中可成功导入。
- 依赖包安装完成。
- 运行时 cwd 或 PYTHONPATH 已覆盖 /home/hdliu/occ_grasp_fall/occ_grasp_models。

## 额外观察与潜在改进点

- occ_grasp_fall 当前没有 agents/bimanual_peract 或 agents/peract_bc 目录，但 agent_factory 已引用，迁移后才能正常使用。
- qattention_peract_bc_agent.update 中计算了 right_bounds/left_bounds 但未在 voxelization 中使用；若需要多层裁剪，需补传入。
- replay_utils 始终计算 CLIP embedding；若 no_language=True 且 use_goal_image=True，可考虑跳过以节省时间。
