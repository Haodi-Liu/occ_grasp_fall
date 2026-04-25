# 视觉 Backbone 统一说明（仅 `act_bc_vision` 与 `diffusion_policy`）

更新时间：2026-03-10  
适用环境：`conda activate ppi`

---

## 1. 范围与结论

本文件只覆盖两个目录：

1. `occ_grasp_models/agents/act_bc_vision`
2. `occ_grasp_models/agents/diffusion_policy`

当前代码结论（以仓库现状为准）：

1. 这两条主线都仍然是 **ResNet 主路径**，没有在这两个目录内落地 DINO/CLIP backbone 工厂。
2. `DIFFUSION_POLICY` 已支持 `model_type: unet | transformer` 双扩散骨干（不是视觉 backbone，视觉仍走 ResNet 编码器）。
3. 旧的两份文档里关于 DINO/CLIP 的内容大多是“计划项”，不是这两个 agent 已实现能力。

---

## 2. 总控链路（只看这两个 agent）

### 2.1 method 默认入口

1. 训练默认 method：`conf/config.yaml:47-48`（当前默认 `DIFFUSION_POLICY`）
2. 评测默认 method：`conf/eval.yaml:1-2`（当前默认 `ACT_BC_VISION`）

### 2.2 method 分发入口

1. `agents/agent_factory.py:113-120`  
   `ACT_BC_VISION` -> `agents.act_bc_vision.launch_utils.create_agent`  
   `DIFFUSION_POLICY` -> `agents.diffusion_policy.launch_utils.create_agent`
2. `run_seed_fn.py` 训练分支  
   `ACT_BC_VISION`：`run_seed_fn.py:258-285`  
   `DIFFUSION_POLICY`：`run_seed_fn.py:287-345`

---

## 3. `ACT_BC_VISION`（当前真实实现）

### 3.1 配置项（视觉相关）

文件：`conf/method/ACT_BC_VISION.yaml`

1. `lr_backbone`：`23`
2. `backbone`：`24`（默认 `resnet18`）
3. `dilation`：`25`
4. `masks`：`42`
5. `camera_names`：`45`

### 3.2 backbone 构建链路

1. `create_agent` 创建 `ACTPolicy(cfg.method)`：`agents/act_bc_vision/launch_utils.py:365-379`
2. `ACTPolicy` 构建模型和优化器：`agents/act_bc_vision/act_policy.py:33-39`
3. `build_ACT_model_and_optimizer` 参数分组（`"backbone"` 名字匹配）：`agents/act_bc_vision/detr/build.py:11-24`
4. `build_ACT_model -> build_vae`：`agents/act_bc_vision/detr/models/__init__.py:5-6`
5. 每个相机构造一套 backbone：`agents/act_bc_vision/detr/models/detr_vae.py:234-237`
6. 真正视觉 backbone 构造：`agents/act_bc_vision/detr/models/backbone.py:113-120`

### 3.3 backbone 细节（`backbone.py`）

1. 通过 `getattr(torchvision.models, name)` 构造模型：`90`
2. 传入 `replace_stride_with_dilation=[False, False, dilation]`：`91`
3. 使用 `FrozenBatchNorm2d`：`19-55, 92`
4. `num_channels` 写死规则：`93`  
   `resnet18/34 -> 512`，其他 -> `2048`
5. `train_backbone = args.lr_backbone > 0`：`115`  
   注意：`BackboneBase` 内冻结参数代码被注释（`62-64`），所以“冻结”主要依赖优化器学习率分组（`lr_backbone`）。

### 3.4 前向特征形状假设

文件：`agents/act_bc_vision/detr/models/detr_vae.py`

1. 每相机独立 backbone 前向：`119-124`
2. 特征按宽度维拼接：`src = torch.cat(all_cam_features, axis=3)`（`128`）
3. 位置编码同维度拼接：`129`

这意味着当前实现依赖 **2D feature map (`B,C,H,W`)**，不是 token-only 表达。

### 3.5 归一化路径（当前是两段）

1. 外层 `PreprocessAgent(norm_type='imagenet')`：`agents/act_bc_vision/launch_utils.py:378-379`  
   对 RGB 做 `/255`：`helpers/preprocess_agent.py:32-37`
2. `ACTPolicy` 再做 ImageNet mean/std：`agents/act_bc_vision/act_policy.py:8-30, 42-45`

备注：这里不是重复 `/255`。`_imagenet_normalize` 只在输入看起来是 `[0,255]` 时才额外除 255（`act_policy.py:15-17`）。

### 3.6 替换 backbone 时的影响面

1. 最小改动（ResNet 家族切换）  
   仅改 `conf/method/ACT_BC_VISION.yaml` 的 `backbone/lr_backbone/dilation/masks`。
2. 中等改动（其他 CNN）  
   重点改 `agents/act_bc_vision/detr/models/backbone.py` 的构造参数与 `num_channels` 推断。
3. 大改动（ViT、DINO token、CLIP token）  
   必改 `agents/act_bc_vision/detr/models/detr_vae.py:128-130` 的 2D 特征拼接与 transformer 输入组织。

---

## 4. `DIFFUSION_POLICY`（当前真实实现）

### 4.1 配置项（视觉相关）

文件：`conf/method/DIFFUSION_POLICY.yaml`

1. `model_type`：`13`（`unet` / `transformer`）
2. `camera_names` / `image_size`：`61-63`
3. `rgb_backbone`：`66`（默认 `resnet18`）
4. `share_rgb_model`：`67`
5. `use_group_norm`：`68`
6. `resize_shape` / `crop_shape` / `random_crop`：`69-71`
7. `imagenet_norm`：`72`

### 4.2 构建链路

1. `create_agent`：`agents/diffusion_policy/launch_utils.py:28-113`
2. 先构 `shape_meta`：`launch_utils.py:44-49`  
   具体在 `agents/diffusion_policy/configs/shape_meta_utils.py:4-25`
3. `_build_obs_encoder`：`launch_utils.py:129-140`
4. `rgb_model = model_getter.get_resnet(...)`：`launch_utils.py:130`  
   `model_getter.py:4-16`（`fc=Identity`）
5. 依据 `model_type` 走：  
   `unet`：`launch_utils.py:54-71`  
   `transformer`：`launch_utils.py:72-103`

### 4.3 `MultiImageObsEncoder` 关键点

文件：`agents/diffusion_policy/model/vision/multi_image_obs_encoder.py`

1. `share_rgb_model` 单模型共享或每相机复制：`47-68`
2. `use_group_norm` 时替换 `BatchNorm2d`：`70-77`
3. 输入变换链：resize/crop/imagenet_norm：`83-119`
4. 前向把所有视觉特征与低维状态拼接：`135-187`
5. `output_shape()` 用 dummy forward 推导维度：`189-203`

### 4.4 输出形状约束（很关键）

当前 `MultiImageObsEncoder.forward` 最终 `torch.cat(features, dim=-1)`（`186`）默认要求视觉 backbone 输出可直接拼成末维特征，主路径是 `B,D`。

如果换成输出 `B,C,H,W` 的 backbone，需先池化/flatten，否则和低维特征拼接会形状不匹配。

### 4.5 归一化路径（与 ACT 不同）

1. 外层 `PreprocessAgent` 关闭 RGB 归一化：`agents/diffusion_policy/launch_utils.py:109-113`（`norm_rgb=False`）
2. 视觉归一化由 `MultiImageObsEncoder(imagenet_norm=...)` 控制：`launch_utils.py:139`, `multi_image_obs_encoder.py:114-117`
3. `normalizer_utils` 中图像字段在 `imagenet_norm=True` 时用 identity normalizer：`agents/diffusion_policy/normalizer_utils.py:61-68`

### 4.6 与训练流程相关的新增点

`run_seed_fn.py:287-345` 的 `DIFFUSION_POLICY` 分支包含：

1. 基于 episode index 的 replay 构建
2. normalizer 拟合并注入 policy：`316-326`
3. 可选把 batch 构建移到 worker：`328-344`

---

## 5. DINO / CLIP：在这两个 agent 里的当前状态

基于当前目录代码，结论是：

1. `act_bc_vision` 未看到 DINO/CLIP backbone 分支或对应配置字段。
2. `diffusion_policy` 的视觉入口仍是 `model_getter.get_resnet(...)`（`launch_utils.py:130`）。
3. 旧文档中的 DINO/CLIP 方案属于设计/计划，不是这两条主线的已合入能力。

补充：`launch_utils.py:124` 的 `clip_sample=True` 属于扩散调度器参数，不是 CLIP 模型接入。

---

## 6. 若现在要在这两个 agent 上做 DINO/CLIP，最小改动面

### 6.1 对 `act_bc_vision`

1. 在 `agents/act_bc_vision/detr/models/backbone.py` 增加多类型 backbone 工厂（resnet/dino/clip）。
2. 在 `agents/act_bc_vision/detr/models/detr_vae.py` 适配非 `B,C,H,W` 输出。
3. 在 `agents/act_bc_vision/act_policy.py` 把归一化从固定 ImageNet 改为按 backbone 类型选择。

### 6.2 对 `diffusion_policy`

1. 把 `agents/diffusion_policy/model/vision/model_getter.py` 从 `get_resnet` 扩展为通用 `get_rgb_backbone`。
2. 保证 `MultiImageObsEncoder` 输入到 `torch.cat` 前是 `B,D`。
3. 把 `imagenet_norm` 从布尔扩展为可选枚举（如 `none/imagenet/clip`），并保持 normalizer 逻辑一致。

---

## 7. 验证清单（只针对这两个 agent）

1. `conda activate ppi`
2. 构建检查：两条链路都能初始化模型
3. 单步前向：输入一批数据，不报 shape mismatch
4. 训练 1-2 step：loss 有限值、无 NaN
5. 评估一回合：动作维度正确（18D bimanual）
6. 若改 backbone：旧 checkpoint 用 `strict=False` 或显式过滤视觉层参数

---

## 8. 快速定位命令（仅这两个 agent）

```bash
conda activate ppi
cd /home/hdliu/occ_grasp_fall/occ_grasp_models

# ACT_BC_VISION
rg -n "backbone:|lr_backbone:|dilation:|masks:|camera_names:" conf/method/ACT_BC_VISION.yaml
rg -n "create_agent|ACTPolicy|build_ACT_model_and_optimizer|build_backbone|torch.cat\\(all_cam_features" agents/act_bc_vision -S
rg -n "_imagenet_normalize|norm_type='imagenet'|norm_rgb" agents/act_bc_vision helpers/preprocess_agent.py -S

# DIFFUSION_POLICY
rg -n "model_type:|rgb_backbone:|share_rgb_model:|use_group_norm:|crop_shape:|imagenet_norm:" conf/method/DIFFUSION_POLICY.yaml
rg -n "create_agent|_build_obs_encoder|get_resnet|MultiImageObsEncoder|DiffusionUnetImagePolicy|DiffusionTransformerImagePolicy" agents/diffusion_policy -S
rg -n "output_shape|torch.cat\\(features, dim=-1\\)|imagenet_norm" agents/diffusion_policy/model/vision/multi_image_obs_encoder.py -S
```

