# 用 DINO / CLIP 替换 ResNet 的修改方案（仅 `ACT_BC_VISION` 与 `DIFFUSION_POLICY`）

本文档是“先设计后改代码”的实施方案，目标是把当前默认 ResNet backbone 替换为：

1. `DINO`（参考：`/home/hdliu/claude_repo/PPI/ppi/model/vision/semantic_feature_extractor.py`）
2. `CLIP`（参考：`/home/hdliu/claude_repo/3d_diffuser_actor/diffuser_actor/utils/clip.py`）

并覆盖两个 agent：

1. `occ_grasp_models/agents/act_bc_vision`
2. `occ_grasp_models/agents/diffusion_policy`

---

## 1. 先说结论（关键约束）

1. `ACT_BC_VISION` 下游默认吃 **2D feature map**（`B,C,H,W`），并做相机维拼接（沿宽度维）。
2. `DIFFUSION_POLICY` 下游默认吃 **1D feature vector**（`B,D`），由 `MultiImageObsEncoder` 拼接。
3. 你给的 DINO 代码不是直接可插拔 backbone（它是点云语义融合流程的一部分），要抽取其中“DINO特征提取”逻辑，封装成标准 `nn.Module`。
4. 你给的 CLIP 代码返回的是多尺度 dict（`res1~res5`），接入时要明确用哪一层（建议默认 `res5`）。
5. 归一化策略必须和 backbone 对齐：
   - DINOv2: ImageNet mean/std
   - CLIP RN50: CLIP mean/std (`[0.48145466, 0.4578275, 0.40821073] / [0.26862954, 0.26130258, 0.27577711]`)

---

## 2. 总体改造思路

统一做一个“视觉 backbone 工厂 + 适配层”，把不同模型输出统一成每个 agent 期望的格式：

1. 对 `ACT_BC_VISION`：统一输出 `B,C,H,W`。
2. 对 `DIFFUSION_POLICY`：统一输出 `B,D`（若拿到 map，则自适应池化再 flatten）。

推荐新增一组共享模块（避免两套重复实现）：

1. `agents/common/vision_backbones/dino_backbone.py`
2. `agents/common/vision_backbones/clip_backbone.py`
3. `agents/common/vision_backbones/factory.py`
4. `agents/common/vision_backbones/normalization.py`

说明：
- 不建议直接跨仓库 import `/home/hdliu/claude_repo/...`，因为路径硬编码会影响复现与部署。
- 建议把必要逻辑迁移到当前仓库，并在配置里保留 `dino_repo_path / dino_ckpt_path` 等可配置项。

---

## 3. `ACT_BC_VISION` 详细修改方案

## 3.1 配置层（`conf/method/ACT_BC_VISION.yaml`）

在保持现有字段兼容的前提下，新增：

1. `visual_backbone_type`: `resnet` | `dinov2` | `clip_rn50`
2. `visual_norm`: `imagenet` | `clip`
3. `clip_feature_key`: `res5`（可选 `res4/res3`）
4. `share_visual_backbone`: `false`（建议 DINO/CLIP 时设为 `true` 以降显存）
5. `dino_repo_path`: `/home/hdliu/occ_grasp_fall/repos/dinov2`
6. `dino_ckpt_path`: `/home/hdliu/occ_grasp_fall/pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth`
7. `clip_model_name`: `RN50`

兼容建议：
- `visual_backbone_type` 缺省时按旧逻辑走 `resnet`（读取原 `backbone` 字段）。

## 3.2 backbone 构建（`agents/act_bc_vision/detr/models/backbone.py`）

把当前仅支持 torchvision ResNet 的 `Backbone` 扩展为三分支：

1. `resnet` 分支：保持现有实现。
2. `dinov2` 分支：
   - 加载 `dinov2_vits14`；
   - 前向调用 `forward_features`，取 `x_norm_patchtokens`；
   - reshape 为 `B,384,Hp,Wp`（`Hp*Wp = token数`）；
   - 返回与现有 `Joiner` 兼容的结构（list/dict 均可，保持 `features[0]` 可用）。
3. `clip_rn50` 分支：
   - 采用你给的 `ModifiedResNetFeatures` 逻辑；
   - 从输出 dict 取 `clip_feature_key`（默认 `res5`）；
   - 输出 `B,C,H,W`。

同时处理：

1. `num_channels` 动态设置：
   - DINOv2 ViT-S/14: `384`
   - CLIP RN50 `res5`: `2048`
2. 冻结策略复用 `train_backbone = args.lr_backbone > 0`。
3. 对非 ResNet 分支忽略 `replace_stride_with_dilation`。

## 3.3 相机共享（`agents/act_bc_vision/detr/models/detr_vae.py`）

当前每个相机会实例化一套 backbone。DINO/CLIP 显存占用较高，建议可选共享：

1. 在 `build(args)` 中，当 `share_visual_backbone=true`：
   - 仅构建一个 backbone；
   - 多相机前向时复用同一模块。
2. 默认保持旧行为（每相机独立），避免破坏已有实验。

## 3.4 归一化（`agents/act_bc_vision/act_policy.py`）

当前是固定 ImageNet 归一化。改为按 `args.visual_norm` 选择：

1. `imagenet`: 现有均值方差
2. `clip`: CLIP 均值方差

注意：
- `PreprocessAgent(norm_type='imagenet')` 目前只做 `/255`，这部分可保持不变。
- 真正 mean/std 放在 `act_policy.py` 做，和 backbone 配置保持一致。

## 3.5 优化器与权重兼容

1. `agents/act_bc_vision/detr/build.py` 的参数分组规则（按名字含 `backbone`）保留即可。
2. 旧 ResNet checkpoint 与新 backbone 结构不兼容，加载时需：
   - 允许 `strict=False` 或
   - 仅加载非 backbone 参数。

---

## 4. `DIFFUSION_POLICY` 详细修改方案

## 4.1 配置层（`conf/method/DIFFUSION_POLICY.yaml`）

新增并兼容旧字段：

1. `rgb_backbone_type`: `resnet` | `dinov2` | `clip_rn50`
2. `rgb_norm`: `none` | `imagenet` | `clip`
3. `clip_feature_key`: `res5`
4. `freeze_rgb_backbone`: `false`
5. `dino_repo_path` / `dino_ckpt_path` / `clip_model_name`

建议默认：

1. DINO/CLIP 时设 `share_rgb_model: true`（避免每相机复制大模型）
2. DINO/CLIP 时设 `use_group_norm: false`

## 4.2 backbone 工厂（`agents/diffusion_policy/model/vision/model_getter.py`）

把 `get_resnet(...)` 扩展为通用 `get_rgb_backbone(...)`：

1. `resnet`：保持 `fc=Identity`，输出 `B,D`。
2. `dinov2`：封装成 `nn.Module`，输出 `B,D`（建议 patch tokens 全局平均得到 `B,384`）。
3. `clip_rn50`：封装成 `nn.Module`，取 `res5` 后 GAP，输出 `B,2048`。

这样 `MultiImageObsEncoder` 主逻辑可以基本不动。

## 4.3 启动入口（`agents/diffusion_policy/launch_utils.py`）

1. `_build_obs_encoder` 改为调用新 `get_rgb_backbone(cfg.method, ...)`。
2. 把 `rgb_norm` 传给 `MultiImageObsEncoder`（替代当前 `imagenet_norm` 的二值开关）。

## 4.4 编码器适配（`agents/diffusion_policy/model/vision/multi_image_obs_encoder.py`）

建议两点增强：

1. 归一化从 bool 扩展为 enum：`none/imagenet/clip`。
2. 对 backbone 输出做兜底：
   - 若输出是 `B,C,H,W`，自动 `adaptive_avg_pool2d -> flatten` 变成 `B,D`；
   - 若已是 `B,D`，直接拼接。

这样即便后续换成别的视觉模型，`DiffusionUnetImagePolicy` 输入维度推导仍稳定。

## 4.5 训练行为影响

1. `obs_encoder.output_shape()` 会自动变化，UNet 条件维度会随之变化（这是预期行为）。
2. 新 backbone 下无法直接加载旧 ResNet 训练权重（尤其 obs_encoder 部分）。

---

## 5. DINO 与 CLIP 的具体接法建议

## 5.1 DINO（参考 `semantic_feature_extractor.py`）

保留其中核心思想：

1. `torch.hub.load(repo_path, 'dinov2_vits14', source='local')`
2. 加载 `dinov2_vits14_pretrain.pth`
3. 用 `forward_features(...)["x_norm_patchtokens"]` 取 patch token 特征

但不要直接搬 `Fusion` 类（其包含点云投影/深度融合，不是本任务 backbone 接口）。

## 5.2 CLIP（参考 `clip.py`）

保留其中核心思想：

1. `load_clip()` 构造 `ModifiedResNetFeatures`
2. `forward` 得到 `res1~res5` 多尺度特征

接入策略：

1. `ACT_BC_VISION`：优先用 `res5` map（保留空间结构）
2. `DIFFUSION_POLICY`：用 `res5` + GAP 得到向量

---

## 6. 推荐默认配置（初次实验）

## 6.1 ACT + DINO

1. `visual_backbone_type: dinov2`
2. `visual_norm: imagenet`
3. `lr_backbone: 1e-5`（或 `0` 先冻结）
4. `share_visual_backbone: true`

## 6.2 ACT + CLIP

1. `visual_backbone_type: clip_rn50`
2. `clip_feature_key: res5`
3. `visual_norm: clip`
4. `share_visual_backbone: true`

## 6.3 DIFFUSION + DINO

1. `rgb_backbone_type: dinov2`
2. `rgb_norm: imagenet`
3. `share_rgb_model: true`
4. `use_group_norm: false`

## 6.4 DIFFUSION + CLIP

1. `rgb_backbone_type: clip_rn50`
2. `rgb_norm: clip`
3. `clip_feature_key: res5`
4. `share_rgb_model: true`
5. `use_group_norm: false`

---

## 7. 实施顺序（建议）

1. 先做配置扩展与默认兼容（不改默认行为）。
2. 再接入 `DIFFUSION_POLICY`（先跑通 `B,D` 路径，改动最小）。
3. 再接入 `ACT_BC_VISION`（处理 2D map + 多相机拼接）。
4. 最后统一清理归一化与 checkpoint 加载策略。

---

## 8. 验证清单（改代码后执行）

1. 模型构建 smoke test：两 agent、四组合（ACT/DP × DINO/CLIP）都能初始化。
2. 前向 smoke test：
   - ACT：`image -> backbone -> transformer` 一次前向成功。
   - DP：`obs_encoder.output_shape()` 与 UNet 构造一致。
3. 1~2 step 训练更新：无 shape mismatch / NaN / 归一化异常。
4. 参数统计：确认 `lr_backbone` 分组仍生效（ACT）。
5. 显存检查：多相机下 DINO/CLIP 建议共享 backbone。

---

## 9. 可能风险与规避

1. 风险：DINO token 网格尺寸与输入分辨率关系导致时序/显存波动。
   - 规避：固定输入分辨率，必要时在 wrapper 内固定 resize。
2. 风险：CLIP 归一化错配导致性能明显下降。
   - 规避：把 `rgb_norm=clip` 作为 CLIP 默认。
3. 风险：旧 checkpoint 误加载造成 silent mismatch。
   - 规避：显式打印 missing/unexpected keys，并区分“仅加载非视觉层”。

