# 视觉 Backbone 指南（仅 `ACT_BC_VISION` 与 `DIFFUSION_POLICY`）

本文档只覆盖以下两个 agent：

1. `occ_grasp_models/agents/act_bc_vision`
2. `occ_grasp_models/agents/diffusion_policy`

目标：

1. 说清楚这两个 agent 的视觉 backbone 是如何由配置驱动并构建出来的。
2. 说清楚如果替换视觉 backbone，应该改哪些文件、怎么改、哪些地方容易漏。

---

## 1. 总控链路（这两个 agent 共用）

### 1.1 method 配置入口

1. 训练默认 method：`conf/config.yaml:47-48`
2. 评测默认 method：`conf/eval.yaml:1-2`

### 1.2 method 分发入口

1. `agent_factory` 将 method 名映射到具体 `launch_utils.create_agent`：`agents/agent_factory.py:113-120`
2. `run_seed_fn` 将 method 分发到对应 replay + 训练流程：  
   `ACT_BC_VISION` 分支在 `run_seed_fn.py:257-285`  
   `DIFFUSION_POLICY` 分支在 `run_seed_fn.py:286-324`

---

## 2. `ACT_BC_VISION`：backbone 配置与构建

## 2.1 配置文件（控制项）

文件：`conf/method/ACT_BC_VISION.yaml`

核心视觉相关字段：

1. `backbone`（默认 `resnet18`）：`ACT_BC_VISION.yaml:24`
2. `lr_backbone`：`ACT_BC_VISION.yaml:23`
3. `dilation`：`ACT_BC_VISION.yaml:25`
4. `masks`：`ACT_BC_VISION.yaml:42`
5. `camera_names`：`ACT_BC_VISION.yaml:45`

---

## 2.2 代码构建链路（从 cfg 到 backbone 实例）

1. `launch_utils.create_agent` 创建 `ACTPolicy(cfg.method)`：  
   `agents/act_bc_vision/launch_utils.py:365-379`
2. `ACTPolicy.__init__` 调 `build_ACT_model_and_optimizer(args)`：  
   `agents/act_bc_vision/act_policy.py:33-39`
3. `detr/build.py` 构建模型并设置优化器参数组：  
   `agents/act_bc_vision/detr/build.py:11-24`
4. `models/__init__.py` 中 `build_ACT_model -> build_vae(args)`：  
   `agents/act_bc_vision/detr/models/__init__.py:5-6`
5. `detr_vae.build(args)` 为每个相机创建一个 backbone：  
   `for _ in args.camera_names: backbone = build_backbone(args)`  
   见 `agents/act_bc_vision/detr/models/detr_vae.py:234-237`
6. `build_backbone(args)` 真正构造视觉 backbone：  
   `agents/act_bc_vision/detr/models/backbone.py:113-120`

---

## 2.3 backbone 具体实现细节

文件：`agents/act_bc_vision/detr/models/backbone.py`

1. `Backbone` 通过 `getattr(torchvision.models, name)` 构造：`backbone.py:90`
2. 传入 `replace_stride_with_dilation=[False, False, dilation]`：`backbone.py:91`
3. 使用 `FrozenBatchNorm2d`：`backbone.py:92`
4. `num_channels` 当前按 ResNet 规则硬编码：
   - `resnet18/34 -> 512`
   - 其他 -> `2048`
   见 `backbone.py:93`
5. `train_backbone = args.lr_backbone > 0`：`backbone.py:115`
6. `return_interm_layers = args.masks`：`backbone.py:116`

---

## 2.4 前向里视觉特征如何使用

文件：`agents/act_bc_vision/detr/models/detr_vae.py`

1. 每个相机分别走自己的 backbone：`detr_vae.py:119-124`
2. 多相机视觉特征沿宽度维拼接：`src = torch.cat(all_cam_features, axis=3)`（`detr_vae.py:128`）
3. 位置编码同样拼接：`detr_vae.py:129`

这意味着当前实现默认视觉特征是 2D feature map（`B,C,H,W` 风格），不是 token-only 表示。

---

## 2.5 输入归一化（容易漏）

`ACT_BC_VISION` 有两层归一化相关处理：

1. `PreprocessAgent(norm_type='imagenet')`（实际上做的是 `/255`，不是 mean/std）：  
   `agents/act_bc_vision/launch_utils.py:378-379`  
   `helpers/preprocess_agent.py:32-37`
2. `ACTPolicy.forward` 里又执行一次真正 ImageNet mean/std 标准化：  
   `agents/act_bc_vision/act_policy.py:8-30, 42-45`

替换 backbone 时必须确认新 backbone 期望输入分布，避免重复或错误归一化。

---

## 2.6 替换 `ACT_BC_VISION` backbone：该改哪里

### 最小改动（仍用 torchvision 的 ResNet 家族）

1. 改 `conf/method/ACT_BC_VISION.yaml` 的 `backbone`。
2. 视需要调整 `lr_backbone`、`dilation`、`masks`。

通常无需改 Python 代码。

### 中等改动（换成 torchvision 的非 ResNet CNN，例如 convnext）

至少检查/修改：

1. `agents/act_bc_vision/detr/models/backbone.py`
   - `Backbone.__init__` 构造参数是否适配新模型（`replace_stride_with_dilation` 可能不支持）
   - `num_channels` 推断不能再用当前硬编码规则（`backbone.py:93`）
2. `agents/act_bc_vision/detr/build.py`
   - 确认 backbone 参数仍能被 `"backbone"` 名称分组到 `lr_backbone`（`detr/build.py:15-19`）

### 大改动（换成 ViT / token backbone）

必须重点处理：

1. `agents/act_bc_vision/detr/models/backbone.py`
   - 输出接口要么适配为 `B,C,H,W`，要么下游一起改
2. `agents/act_bc_vision/detr/models/detr_vae.py`
   - 当前 `torch.cat(all_cam_features, axis=3)` 假设是 2D map（`detr_vae.py:128`）
   - token 表示需要重写拼接与 transformer 输入组织方式
3. `agents/act_bc_vision/act_policy.py`
   - 校准输入归一化策略与新 backbone 预训练统计

---

## 3. `DIFFUSION_POLICY`：backbone 配置与构建

## 3.1 配置文件（控制项）

文件：`conf/method/DIFFUSION_POLICY.yaml`

核心视觉相关字段：

1. `rgb_backbone`（默认 `resnet50`）：`DIFFUSION_POLICY.yaml:30`
2. `share_rgb_model`：`DIFFUSION_POLICY.yaml:31`
3. `use_group_norm`：`DIFFUSION_POLICY.yaml:32`
4. `resize_shape`：`DIFFUSION_POLICY.yaml:33`
5. `crop_shape`：`DIFFUSION_POLICY.yaml:34`
6. `random_crop`：`DIFFUSION_POLICY.yaml:35`
7. `imagenet_norm`：`DIFFUSION_POLICY.yaml:36`
8. `camera_names`、`image_size`：`DIFFUSION_POLICY.yaml:25-27`

---

## 3.2 代码构建链路（从 cfg 到视觉编码器）

1. `create_agent` 先构 `shape_meta`：`launch_utils.py:41-46`
   - `shape_meta_utils.build_shape_meta` 定义每个相机 RGB shape：`configs/shape_meta_utils.py:4-25`
2. `_build_obs_encoder(cfg, shape_meta)`：
   - `rgb_model = model_getter.get_resnet(name=cfg.method.rgb_backbone, weights=None)`：`launch_utils.py:93-95`
   - 再构建 `MultiImageObsEncoder(...)`：`launch_utils.py:95-104`
3. `DiffusionUnetImagePolicy` 读取 `obs_encoder.output_shape()` 决定后续 UNet 输入维度：  
   `policy/diffusion_unet_image_policy.py:38-52`

---

## 3.3 `DIFFUSION_POLICY` 视觉 backbone 的关键实现点

### `model_getter.py`

1. `get_resnet(name, weights=None)` 用 `torchvision.models.<name>`：`model_getter.py:13-15`
2. `fc` 被替换为 `Identity`：`model_getter.py:15`

### `multi_image_obs_encoder.py`

1. 支持共享或每相机独立 backbone：`multi_image_obs_encoder.py:46-68`
2. 可选把 backbone 内 `BatchNorm2d` 替换为 `GroupNorm`：`multi_image_obs_encoder.py:70-77`
3. 支持 resize/crop/imagenet_norm：`multi_image_obs_encoder.py:83-119`
4. 前向中把每个相机特征拼接成最终观测特征：`multi_image_obs_encoder.py:135-187`
5. `output_shape()` 通过 dummy forward 推导最终特征维度：`multi_image_obs_encoder.py:189-203`

---

## 3.4 预处理注意点

`DIFFUSION_POLICY` 在 `PreprocessAgent` 中明确设置 `norm_rgb=False`：`launch_utils.py:73-77`。  
RGB 归一化主要由 `MultiImageObsEncoder` 的 `imagenet_norm` 配置控制（`multi_image_obs_encoder.py:113-117`）。

---

## 3.5 替换 `DIFFUSION_POLICY` backbone：该改哪里

### 最小改动（仍是 torchvision resnet18/34/50）

1. 改 `conf/method/DIFFUSION_POLICY.yaml` 的 `rgb_backbone`。
2. 按需调 `share_rgb_model/use_group_norm/crop/imagenet_norm`。

### 中等改动（换成其它 torchvision 模型）

至少检查/修改：

1. `agents/diffusion_policy/model/vision/model_getter.py`
   - 新模型构造方式
   - 最终输出形状（是否仍能被 `MultiImageObsEncoder` 直接拼接）
2. `agents/diffusion_policy/model/vision/multi_image_obs_encoder.py`
   - `use_group_norm` 替换逻辑只处理 `BatchNorm2d`，新模型可能没有 BN

### 大改动（换成 timm/ViT 或输出 4D 特征图的模型）

必须确认：

1. `MultiImageObsEncoder.forward` 当前默认可直接把 backbone 输出作为 feature 拼接（`multi_image_obs_encoder.py:154-173`）
2. 若输出是 `B,C,H,W`，通常需要先池化成 `B,D` 或另写 flatten 逻辑
3. `output_shape()` 与 `DiffusionUnetImagePolicy` 的维度推导必须一致，否则 UNet 输入维度会错（`diffusion_unet_image_policy.py:38-52`）

---

## 4. 只针对这两个 agent 的不遗漏清单

1. method 配置是否切到目标 agent
2. `conf/method/ACT_BC_VISION.yaml` 或 `DIFFUSION_POLICY.yaml` 的 backbone 字段是否更新
3. `agent_factory.py` / `run_seed_fn.py` 是否命中正确分支
4. backbone 构建代码是否支持新模型初始化参数
5. 输出通道或输出 shape 是否与下游一致
6. optimizer 是否仍正确应用 `lr_backbone`（ACT_BC_VISION）
7. 图像归一化路径是否与新 backbone 匹配
8. checkpoint 加载是否兼容（旧权重常会不兼容）
9. 训练与推理都做过最小 smoke test（至少 1 次前向 + 1-2 step update）

---

## 5. 推荐实施顺序（仅这两个 agent）

1. 先改配置，确认 baseline 可跑（不改代码）。
2. 再改 backbone 构建代码（`backbone.py` 或 `model_getter.py`）。
3. 再改特征 shape 适配（`detr_vae.py` 或 `multi_image_obs_encoder.py`）。
4. 最后统一处理归一化与 checkpoint 兼容。

---

## 6. 快速定位命令（仅这两个 agent）

```bash
# ACT_BC_VISION: 配置与构建
rg -n "backbone:|lr_backbone:|dilation:|masks:|camera_names:" conf/method/ACT_BC_VISION.yaml
rg -n "create_agent|ACTPolicy|build_ACT_model_and_optimizer" agents/act_bc_vision -S
rg -n "def build_backbone|class Backbone|torchvision.models" agents/act_bc_vision/detr/models/backbone.py
rg -n "self.backbones\\[cam_id\\]|torch.cat\\(all_cam_features" agents/act_bc_vision/detr/models/detr_vae.py

# DIFFUSION_POLICY: 配置与构建
rg -n "rgb_backbone:|share_rgb_model:|use_group_norm:|crop_shape:|imagenet_norm:" conf/method/DIFFUSION_POLICY.yaml
rg -n "_build_obs_encoder|get_resnet|MultiImageObsEncoder" agents/diffusion_policy -S
rg -n "replace_submodules|GroupNorm|output_shape|forward" agents/diffusion_policy/model/vision/multi_image_obs_encoder.py -S
rg -n "obs_feature_dim = obs_encoder.output_shape" agents/diffusion_policy/policy/diffusion_unet_image_policy.py
```

