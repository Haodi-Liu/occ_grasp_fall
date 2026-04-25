# PPI (Point flow with Pose Inference) 方法详解

---

## 0. `occ_grasp_models/ppi/` 目录结构与文件说明

本节对 `occ_grasp_models/ppi` 目录下的组织结构进行系统介绍，帮助快速理解各模块职责并区分命名相似的脚本。

### 0.1 目录树概览

```
occ_grasp_models/
└── ppi/
    ├── policy/          # 策略封装层 (训练/推理接口)
    ├── model/           # 模型核心实现
    │   ├── common/      # 通用模块 (归一化、工具函数等)
    │   ├── diffusion/   # 扩散模型 & 各种变体
    │   └── vision/      # 视觉编码器 (PointNet++, DINOv2等)
    ├── dataset/         # 数据集加载器
    ├── common/          # 数据处理 (预处理、采样、回放缓冲)
    ├── config/          # Hydra配置文件
    └── utils/           # 通用工具 (分布式训练等)
```

---

### 0.2 各子目录详解

#### `policy/` — 策略封装层

| 文件 | 功能 |
|------|------|
| `base_policy.py` | 策略基类，定义 `forward()` 和 `predict_action()` 接口 |
| **`ppi.py`** | **PPI主策略类** (仿真环境)，封装扩散采样、损失计算、训练/推理逻辑 |
| `ppi_real.py` | 真实机器人版本，针对真机部署调整 (如输入输出格式) |

> **区分**: `ppi.py` 用于仿真 (RLBench)，`ppi_real.py` 用于真实机器人

---

#### `model/diffusion/` — 扩散模型变体 (重点区分)

| 文件 | 类名 | 功能说明 |
|------|------|----------|
| **`diffuser_actor_ppi.py`** | `DiffusionHeadPPI` | **核心模型**: 完整PPI (关键帧 + 连续动作 + 点流，层次化生成) |
| `diffuser_actor_ppi_real.py` | `DiffusionHeadPPIReal` | 真机版PPI (适配真实机器人输入输出) |
| `diffuser_actor_ppi_real_simple.py` | `DiffusionHeadPPIRealSimple` | 简化版真机PPI (减少计算开销) |
| `diffuser_actor_pure.py` | `DiffusionHeadPure` | **消融实验**: 仅预测纯连续动作 或 纯关键帧 |
| `diffuser_actor_keypose_continuous.py` | `DiffusionHeadKeyposeContinuous` | **消融实验**: 连续动作以关键帧为条件 (无点流) |
| `diffuser_actor_pointflow_continuous.py` | `DiffusionHeadPointflowContinuous` | **消融实验**: 连续动作以点流为条件 (无关键帧) |
| `ema_model.py` | `EMAModel` | 指数移动平均模型 (训练稳定性) |
| `positional_embedding.py` | - | 位置编码实现 |
| `mask_generator.py` | - | 掩码生成工具 |

> **命名规律**:
> - `*_ppi.py`: 完整PPI方法
> - `*_ppi_real*.py`: 真机部署版本
> - `*_pure.py`, `*_keypose_continuous.py`, `*_pointflow_continuous.py`: **消融实验变体**

**`diffuser_actor_utils/` 子目录**:
| 文件 | 功能 |
|------|------|
| `layers.py` | 自注意力/交叉注意力模块 (FFWRelative*AttentionModule) |
| `position_encodings.py` | 旋转位置编码 (RotaryPositionEncoding3D)、正弦编码 |
| `multihead_custom_attention.py` | 自定义多头注意力实现 |

---

#### `model/vision/` — 视觉编码器

| 文件 | 功能 |
|------|------|
| `pointnet2.py` | **PointNet++** 点云编码器 (FPS采样 + 局部特征聚合) |
| `semantic_feature_extractor.py` | **DINOv2特征提取** + 多视角3D投影融合 (Fusion类) |
| `observation_encoder.py` | **观测编码器**: 整合点云、语义特征、状态、语言、点流位置编码 |

---

#### `model/common/` — 模型通用模块

| 文件 | 功能 |
|------|------|
| `normalizer.py` | 数据归一化器 (训练时统计均值/方差) |
| `lr_scheduler.py` | 学习率调度器 |
| `tensor_util.py` / `shape_util.py` | 张量操作工具 |
| `module_attr_mixin.py` | 模块属性混入类 |
| `dict_of_tensor_mixin.py` | 字典张量处理混入类 |

---

#### `common/` — 数据处理 (重点区分)

**数据预处理类 (`get_data_*.py`)**:

| 文件 | 类名 | 预测目标 | 适用场景 |
|------|------|----------|----------|
| `get_data_keyframe.py` | `GetDataKeyframe` | **仅关键帧** | 纯关键帧预测 |
| `get_data_continuous.py` | `GetDataContinuous` | **仅连续动作** | 纯连续动作预测 |
| **`get_data_keyframe_continuous.py`** | `GetDataKeyframeContinuous` | **关键帧 + 连续动作 + 点流** | **PPI完整方法** |
| `get_data_keyframe_continuous_real.py` | `GetDataKeyframeContinuousReal` | 同上 (真机数据格式) | 真实机器人 |

> **选择指南**:
> - 完整PPI → `get_data_keyframe_continuous.py`
> - 消融 (纯关键帧) → `get_data_keyframe.py`
> - 消融 (纯连续) → `get_data_continuous.py`

**采样器 (`sampler_*.py`)**:

| 文件 | 功能 |
|------|------|
| `sampler_keyframe.py` | 关键帧采样 (仅采样关键帧时刻) |
| `sampler_continuous.py` | 连续采样 (固定间隔采样所有帧) |
| **`sampler_keyframe_continuous.py`** | **关键帧+连续混合采样** (PPI使用): 输出 `[连续序列, 关键帧序列]` |
| `sampler_keyframe_continuous_real.py` | 同上 (真机版本) |

> **采样逻辑**: `sampler_keyframe_continuous.py` 在每个时间步生成 `[cont_0, cont_1, ..., kf_0, kf_1, ...]` 格式的索引

**经验回放缓冲**:

| 文件 | 功能 |
|------|------|
| `replay_buffer.py` | 仿真环境回放缓冲 (RLBench) |
| `replay_buffer_real.py` | 真实机器人回放缓冲 |

**其他工具**:

| 文件 | 功能 |
|------|------|
| `checkpoint_util.py` | 检查点保存/加载 |
| `pytorch_util.py` | PyTorch工具函数 |
| `logger_util.py` | 日志工具 |
| `model_util.py` | 模型工具函数 |

---

#### `dataset/` — 数据集加载器

| 文件 | 功能 |
|------|------|
| `base_dataset.py` | 数据集基类 |
| `rlbench2_dataset.py` | RLBench2双臂仿真数据集 |
| `real_dataset.py` | 真实机器人数据集 |

---

#### `utils/` — 通用工具

| 文件 | 功能 |
|------|------|
| `distributed.py` | 分布式训练工具 (DDP相关) |

---

### 0.3 快速索引: 命名相似脚本对比

| 对比组 | 区别说明 |
|--------|----------|
| `ppi.py` vs `ppi_real.py` | 仿真策略 vs 真机策略 |
| `diffuser_actor_ppi.py` vs `*_real.py` vs `*_real_simple.py` | 完整版 vs 真机版 vs 简化真机版 |
| `diffuser_actor_ppi.py` vs `*_pure.py` | 完整PPI vs 消融 (无层次化) |
| `diffuser_actor_ppi.py` vs `*_keypose_continuous.py` | 完整PPI vs 消融 (无点流) |
| `diffuser_actor_ppi.py` vs `*_pointflow_continuous.py` | 完整PPI vs 消融 (无关键帧) |
| `get_data_keyframe.py` vs `*_continuous.py` vs `*_keyframe_continuous.py` | 纯关键帧 vs 纯连续 vs 混合 (PPI) |
| `replay_buffer.py` vs `*_real.py` | 仿真回放 vs 真机回放 |

---

## 文件概述

**代码路径**: `occ_grasp_models`

**主要功能**: PPI是一种基于扩散模型的双臂机器人策略学习方法，结合了**关键帧预测**和**连续控制**，通过预测物体**点流(Point Flow)**和末端执行器**位姿(Pose)**来实现高精度操作。该方法使用3D点云和DINOv2语义特征作为场景表示，能够有效处理复杂的双臂协调任务。

---

## 1. 整体架构概览

### 1.1 核心设计理念

PPI方法的核心创新在于将**关键帧预测**与**连续控制**相结合，并引入**物体点流**作为额外的监督信号：

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PPI 整体架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入层                                                              │
│  ├── 3D点云 + DINOv2语义特征                                         │
│  ├── 语言指令 (CLIP编码)                                             │
│  ├── 机器人状态 (双臂末端位姿)                                        │
│  └── 初始点流位置                                                    │
│                                                                      │
│  编码层                                                              │
│  ├── PointNet++ (点云特征提取)                                       │
│  ├── DINOv2特征投影                                                  │
│  ├── 状态MLP编码                                                     │
│  └── 点流位置编码                                                    │
│                                                                      │
│  扩散模型层                                                          │
│  ├── 点流预测头 (预测物体运动)                                        │
│  ├── 关键帧预测头 (预测关键位姿)                                      │
│  └── 连续动作预测头 (预测平滑轨迹)                                    │
│                                                                      │
│  输出层                                                              │
│  ├── 左臂: 位置(3) + 四元数(4) + 夹爪开合(1) = 8D                    │
│  ├── 右臂: 位置(3) + 四元数(4) + 夹爪开合(1) = 8D                    │
│  └── 点流: 物体关键点位置变化                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 关键文件结构

| 文件路径 | 功能描述 |
|---------|---------|
| `occ_grasp_models/ppi/policy/ppi.py` | PPI策略主类，封装训练和推理逻辑 |
| `occ_grasp_models/ppi/model/diffusion/diffuser_actor_ppi.py` | 扩散头核心实现，关键帧+连续+点流预测 |
| `occ_grasp_models/ppi/model/vision/observation_encoder.py` | 观测编码器，处理点云和语义特征 |
| `occ_grasp_models/ppi/model/vision/semantic_feature_extractor.py` | DINOv2特征提取和3D投影 |
| `occ_grasp_models/ppi/model/vision/pointnet2.py` | PointNet++点云编码器 |
| `occ_grasp_models/ppi/dataset/rlbench2_dataset.py` | RLBench数据集加载 |
| `occ_grasp_models/ppi/common/get_data_keyframe_continuous.py` | 关键帧发现和数据处理 |
| `occ_grasp_models/train_ppi_ddp.py` | 分布式训练脚本 |
| `occ_grasp_models/agents/ppi/ppi_agent.py` | RLBench评估Agent封装 |

---

## 2. 3D场景处理与表示

### 2.1 点云获取与融合

PPI使用多视角RGB-D相机获取场景信息，并将其融合为统一的3D点云表示。

**文件**: `occ_grasp_models/ppi/model/vision/semantic_feature_extractor.py`

```python
class Fusion():
    def __init__(self, num_cam, feat_backbone='dinov2', device='cuda:0', dtype=torch.float32):
        self.device = device
        self.mu = 0.02  # 深度融合阈值

        # 加载DINOv2特征提取器
        self.dinov2_feat_extractor = torch.hub.load(
            repo_path, 'dinov2_vits14', source='local', skip_validation=True
        )
```

**点云融合流程**:

1. **深度投影**: 将3D点投影到各相机图像平面
2. **深度验证**: 比较投影深度与真实深度，过滤遮挡点
3. **特征插值**: 对有效点从DINOv2特征图中双线性插值获取语义特征
4. **距离加权融合**: 多视角特征按距离权重融合

```python
def eval(self, pts, return_names=['dino_feats'], return_inter=False):
    # 投影3D点到相机坐标
    pts_2d, valid_mask, pts_depth = project_points_coords(
        pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K']
    )

    # 深度验证
    inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1), pts_2d, ...)
    dist = inter_depth - pts_depth
    dist_valid = (inter_depth > 0.0) & valid_mask & (dist > -self.mu)

    # 距离加权
    dist_weight = torch.exp(torch.clamp(self.mu - torch.abs(dist), max=0) / self.mu)

    # 特征融合
    for k in return_names:
        inter_k = interpolate_feats(self.curr_obs_torch[k].permute(0,3,1,2), pts_2d, ...)
        val = (inter_k * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) \
              / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6)
```

### 2.2 DINOv2语义特征提取

**文件**: `occ_grasp_models/ppi/model/vision/semantic_feature_extractor.py` (行166-192)

```python
def extract_dinov2_features(self, imgs, params):
    K, H, W, _ = imgs.shape
    patch_h = params['patch_h']  # 64
    patch_w = params['patch_w']  # 64
    feat_dim = 384  # vits14

    transform = T.Compose([
        T.Resize((patch_h * 14, patch_w * 14)),  # 896x896
        T.CenterCrop((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        features_dict = self.dinov2_feat_extractor.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']
        features = features.reshape((K, patch_h, patch_w, feat_dim))
    return features
```

**特征维度**: 每个相机产生 `(64, 64, 384)` 的特征图

### 2.3 PointNet++点云编码

**文件**: `occ_grasp_models/ppi/model/vision/pointnet2.py` (行253-273)

```python
class PointNet2DenseEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, use_bn=True,
                 npoint1=3072, npoint2=1024):
        super(PointNet2DenseEncoder, self).__init__()

        # 两层Set Abstraction
        self.sa1 = PointNetSetAbstraction(
            npoint=npoint1,      # 第一层采样3072点
            radius=0.04,         # 局部半径4cm
            nsample=32,          # 每个球内采样32点
            in_channel=in_channels,
            mlp=[64, 64, 128],
            group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=npoint2,      # 第二层采样1024点
            radius=0.08,         # 局部半径8cm
            nsample=64,          # 每个球内采样64点
            in_channel=128+3,
            mlp=[128, 128, 288],
            group_all=False
        )

    def forward(self, xyz):
        # xyz: (B, D, N) where D = 3 + 384 (坐标 + DINOv2特征)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        return (l2_xyz, l2_points)  # (B, 3, 1024), (B, 288, 1024)
```

**处理流程**:
1. 输入: 6144个点，每点3D坐标 + 384维DINOv2特征
2. 第一层SA: 采样到3072点，特征维度128
3. 第二层SA: 采样到1024点，特征维度288
4. 输出: 1024个采样点及其288维特征

### 2.4 观测编码器

**文件**: `occ_grasp_models/ppi/model/vision/observation_encoder.py` (行51-124)

```python
class ObservationEncoder(nn.Module):
    def __init__(self,
                 out_channel=288,
                 state_mlp_size=(128, 288),
                 lang_mlp_size=(288, 288),
                 pcd_mlp_size=(288, 288),
                 pointcloud_encoder_cfg=None,
                 use_lang=False,
                 use_initial_pointflow=True,
                 scene_pcd_num=6144):
        super().__init__()

        self.state_shape = 16   # 双臂末端位姿 (7+1)*2
        self.lang_shape = 1024  # CLIP语言特征维度

        # PointNet++编码器
        self.pointnet_encoder = PointNet2DenseEncoder(**pointcloud_encoder_cfg)

        # MLP编码器
        self.state_mlp = nn.Sequential(...)  # 状态编码: 16 -> 288
        self.pcd_mlp = nn.Sequential(...)     # 点云特征投影: 384 -> 288
        self.lang_mlp = nn.Sequential(...)    # 语言特征投影: 1024 -> 288
        self.point_flow_mlp = nn.Sequential(...)  # 点流位置编码: 3 -> 288
```

**输出结构**:
```python
def forward(self, observations: Dict) -> Tuple:
    # 1. 点云 + DINOv2特征处理
    ptc_wth_feature = torch.cat((points, dino_feature), dim=2)  # (B, N, 3+384)
    (sampled_pcd_coord, sampled_pcd_feat) = self.pointnet_encoder(ptc_wth_feature)

    # 2. 全局点云DINOv2特征投影
    pcd_feat = self.pcd_mlp(dino_feature)  # (B, 6144, 288)

    # 3. 状态编码
    state_feat = self.state_mlp(state)  # (B, 288)

    # 4. 语言编码
    lang_feat = self.lang_mlp(lang)  # (B, 288)

    # 5. 初始点流位置编码
    point_flow_feat = self.point_flow_mlp(initial_point_flow)  # (B, N_flow, 288)

    return (points, pcd_feat, lang_feat, state_feat,
            sampled_pcd_coord, sampled_pcd_feat,
            point_flow_feat, initial_point_flow)
```

#### 2.4.1 与 2.1 的衔接关系

`2.1` 和 `2.4` 是严格的上下游关系:

1. `2.1` 的 `Fusion` 模块先对每个3D点做多视角投影
2. 在每个相机视角上做深度一致性检查，过滤遮挡或无效投影
3. 从各相机的DINOv2特征图中，对该点做双线性插值
4. 将多视角特征按距离权重融合，得到该3D点最终的语义特征

因此，`2.1` 的直接产物就是与点云逐点对齐的 `dino_feature`:

- 输入点云: `point_cloud`，形状 `(B, 6144, 3)`
- 融合输出: `dino_feature`，形状 `(B, 6144, 384)`

也就是说，`point_cloud[:, i, :]` 和 `dino_feature[:, i, :]` 描述的是**同一个第 `i` 个3D点**:

- 前者提供几何位置
- 后者提供该点从多视角图像中融合得到的语义信息

随后在 `2.4` 中，`ObservationEncoder` 直接消费这两个张量:

- `points` = 场景点云坐标
- `dino_feature` = `2.1` 生成的逐点语义特征

所以可以把两节的关系概括为:

- `2.1` 负责给3D点云中的每个点“贴上语义标签”
- `2.4` 负责把“带语义标签的3D点云”编码成策略网络真正使用的场景特征

#### 2.4.2 `N` 是否等于 `6144`

在本文默认的 PPI 配置与分析语境下，答案是 **是的**。

因此下面这句:

```python
ptc_wth_feature = torch.cat((points, dino_feature), dim=2)  # (B, N, 3+384)
```

这里的 `N` 对应场景点数，默认就是 `6144`，所以该张量实际可写为:

```python
(B, 6144, 387)
```

原因如下:

- `point_cloud` 在本文维度表中定义为 `(B, 6144, 3)`
- `dino_feature` 在本文维度表中定义为 `(B, 6144, 384)`
- `ObservationEncoder` 默认参数 `scene_pcd_num=6144`
- 评估配置中 `fps_num=6144`

需要注意的是，`N` 本质上表示“输入到观测编码器的场景点数”。如果将来修改了 `fps_num` 或 `scene_pcd_num`，那么 `N` 也会随之改变。本文当前分析针对的是默认设置 `N=6144` 的情况。

#### 2.4.3 `pcd_feat` 和 `sampled_pcd_feat` 的关系

这两部分输出不是互相独立的两套场景表示，而是对**同一份输入点云**做出的两种不同粒度的编码。

| 分支 | 输入 | 输出 | 与原始点的对应关系 | 主要含义 |
|------|------|------|-------------------|----------|
| `pcd_feat` | `dino_feature` | `(B, 6144, 288)` | 与6144个原始点一一对应 | 稠密逐点语义特征 |
| `sampled_pcd_feat` | `cat(points, dino_feature)` | `(B, 1024, 288)` | 对应1024个采样点及其局部邻域 | 稀疏局部聚合特征 |

两者的具体区别如下:

1. `pcd_feat` 是逐点投影
   - 计算方式是 `self.pcd_mlp(dino_feature)`
   - 本质上是把每个点自己的 `384` 维 DINOv2 语义向量独立映射到 `288` 维
   - 不做邻域聚合，因此保留了“每个点自己的语义表示”
   - 所以 `pcd_feat[:, i, :]` 直接对应原始第 `i` 个点

2. `sampled_pcd_feat` 是稀疏采样后的局部聚合
   - 计算方式是先拼接 `points` 与 `dino_feature`，再送入 PointNet++
   - PointNet++ 先从 `6144` 个点采样到 `3072` 个点，再采样到 `1024` 个点
   - 每个采样点的特征不是单个点特征的拷贝，而是该采样点邻域内一组点经过局部聚合后的结果
   - 因此 `sampled_pcd_feat[:, j, :]` 对应的是“第 `j` 个采样锚点及其周围局部区域”

可以把它们理解为:

- `pcd_feat`: 给场景中每个点都保留一份语义描述，强调“全量、稠密、逐点”
- `sampled_pcd_feat`: 从场景中选出少量代表点，并总结局部几何与语义，强调“稀疏、压缩、局部结构”

有一个容易混淆但很重要的点:

`sampled_pcd_feat[:, j, :]` 一般**不等于**某个 `pcd_feat[:, i, :]`。

原因是:

- `pcd_feat` 是单点MLP投影结果
- `sampled_pcd_feat` 是PointNet++对邻域进行聚合后的结果

所以，二者共享同一输入来源，但表达层次不同:

- `pcd_feat` 更像“逐点语义记忆库”
- `sampled_pcd_feat` 更像“局部区域摘要特征”

#### 2.4.4 两路特征在后续扩散头中的分工

这两路特征在后续网络中的用途也不同:

1. `pcd_feat` 作为完整场景点云特征
   - 保留 `6144` 个点
   - 在扩散头中作为 `cross-attention` 的 `value`
   - 供轨迹token或点流token从全场景中检索相关语义与空间信息

2. `sampled_pcd_feat` 作为稀疏场景锚点特征
   - 只保留 `1024` 个采样点
   - 会与机械臂特征或点流特征拼接
   - 进入后续 `self-attention`
   - 用于提供压缩后的局部结构上下文

因此，一句话总结这两路特征的关系:

- `pcd_feat` 负责“看全场，每个点都保留”
- `sampled_pcd_feat` 负责“抓重点，把局部结构浓缩出来”

---

## 3. 关键帧预测与连续控制的结合

### 3.1 关键帧发现算法

**文件**: `occ_grasp_models/ppi/common/get_data_keyframe_continuous.py` (行117-159)

PPI使用基于速度和夹爪状态变化的关键帧发现算法：

```python
def keypoint_discovery_bimanual(self, low_dim_obs, episode, stopping_delta=0.1, total_kp=10):
    episode_keypoints = []
    openess_keypoints = []
    right_prev_gripper_open = low_dim_obs[0].right.gripper_open
    left_prev_gripper_open = low_dim_obs[0].left.gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(low_dim_obs):
        # 检测双臂是否停止
        right_stopped = self._is_stopped_right(low_dim_obs, i, obs.right, stopping_delta)
        left_stopped = self._is_stopped_left(low_dim_obs, i, obs.left, stopping_delta)
        stopped = (stopped_buffer <= 0) and right_stopped and left_stopped
        stopped_buffer = 10 if stopped else stopped_buffer - 1

        # 检测夹爪状态变化
        last = i == (len(low_dim_obs) - 1)
        right_state_changed = obs.right.gripper_open != right_prev_gripper_open
        left_state_changed = obs.left.gripper_open != left_prev_gripper_open
        state_changed = right_state_changed or left_state_changed

        # 关键帧条件: 状态变化 或 停止 或 最后一帧
        if i != 0 and (state_changed or last or stopped):
            episode_keypoints.append(i)

        # 记录夹爪状态变化点用于额外采样
        if i != 0 and state_changed:
            openess_keypoints.append(i)

    # 补充关键帧数量到total_kp
    if total_kp > len(episode_keypoints):
        remaining_indices = [i for i in range(len(low_dim_obs)) if i not in episode_keypoints]
        indices_to_sample = np.linspace(0, len(remaining_indices)-1,
                                         num=total_kp-len(episode_keypoints), dtype=int)
        extra_episode_keypoints = remaining_indices[indices_to_sample]
        episode_keypoints.extend(extra_episode_keypoints)

    episode_keypoints.sort()
    return episode_keypoints, openess_keypoints
```

**速度检测**:
```python
def _is_stopped_right(self, demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].right.gripper_open
        and obs.gripper_open == demo[i - 1].right.gripper_open
        and demo[i - 2].right.gripper_open == demo[i - 1].right.gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change
```

### 3.2 采样策略

**文件**: `occ_grasp_models/ppi/common/sampler_keyframe_continuous.py` (行9-83)

```python
def create_indices(keyframe_indices, episode_ends,
                   sequence_length_keyframe, sequence_length_continuous,
                   episode_mask, pad_before=0, pad_after=0,
                   openess_indices=None, add_openess_sampling=False):
    """
    为每个时间步创建训练样本索引

    采样结构:
    [continuous_0, continuous_1, ..., continuous_k-1, keyframe_0, keyframe_1, ...]
    └───────────── 连续动作序列 ─────────────┘  └─────── 关键帧序列 ───────┘
    """
    sequence_length = sequence_length_keyframe + sequence_length_continuous

    for idx in range(min_start, max_start + 1):
        # 获取当前时间步之后的关键帧
        keyframe_indices_mask = (keyframe_indices > idx + start_idx) & (keyframe_indices < end_idx)
        keyframe_filtered_indices = keyframe_indices[keyframe_indices_mask]

        # 获取连续动作索引
        continuous_indices = np.arange(idx + start_idx, idx + start_idx + sequence_length_continuous)

        # 组合索引: [连续序列, 关键帧序列]
        precent_indices = np.concatenate((
            continuous_padded_filtered_indices[:sequence_length_continuous],
            keyframe_padded_filtered_indices[:sequence_length_keyframe]
        ))

        # 夹爪状态变化点附近增加采样权重
        distance_to_openess = openess_filtered_indices[0] - continuous_padded_filtered_indices[0]
        if add_openess_sampling:
            if 0 < distance_to_openess <= 10:
                for _ in range(2):
                    indices.append(precent_indices)
            if 0 < distance_to_openess <= 5:
                for _ in range(3):
                    indices.append(precent_indices)

        indices.append(precent_indices)
```

**采样可视化**:
```
Episode Timeline:
0───────────────────────────────────────────────────────────────→ T
    ▲           ▲                    ▲              ▲
    K1          K2                   K3             K4 (keyframes)

Sample at t=5:
├─ Continuous: [5, 6, 7] (horizon_continuous=3)
└─ Keyframe: [K2, K3] (horizon_keyframe=2, 取t之后的关键帧)

输出动作序列: [act_5, act_6, act_7, act_K2, act_K3]
```

#### 3.2.1 样本时间轴是如何构造的

设:

- 当前采样起点为 `t0 = idx + start_idx`
- 连续段长度为 `Tc = sequence_length_continuous`
- 关键帧段长度为 `Tk = sequence_length_keyframe`
- 总长度为 `T = Tc + Tk`

则 `create_indices()` 为单个训练样本构造的时间索引遵循下面的规则:

1. **连续段索引**
   - 直接取从当前时刻开始的连续时间步
   - `I_cont = [t0, t0+1, ..., t0+Tc-1]`
   - 如果到达 episode 末尾，就用最后一个有效时间步重复填充，保证长度始终为 `Tc`

2. **关键帧段索引**
   - 从当前时刻之后的关键帧集合中选取
   - `I_kf = [k1, k2, ..., kTk]`
   - 满足 `k1 > t0`
   - 如果后续关键帧不足 `Tk` 个，就重复最后一个有效关键帧做 padding
   - 如果后续一个关键帧都没有，代码会退化为使用当前时刻 `t0` 作为占位索引

3. **最终样本时间顺序**
   - `I = [I_cont, I_kf]`
   - 即先放连续动作，再放关键帧动作
   - 后续训练、扩散头解耦预测、loss 对齐都严格遵循这个顺序

在默认配置下:

- `Tc = 50`
- `Tk = 4`
- `T = 54`

所以一个样本的动作时间布局可以写成:

```text
[t0, t0+1, ..., t0+49, K1, K2, K3, K4]
```

其中 `K1..K4` 是 `t0` 之后的未来关键帧时刻。

#### 3.2.2 夹爪状态变化点为什么会被重复采样

`create_indices()` 中还有一层时间重加权逻辑:

- 若当前起点 `t0` 到最近一次未来夹爪开合变化点的距离 `distance_to_openess` 满足 `0 < d <= 10`，则额外复制该样本 `2` 次
- 若进一步满足 `0 < d <= 5`，则再额外复制 `3` 次

因此:

- 普通样本出现 `1` 次
- 离开合变化点较近的样本可能出现 `3` 次
- 非常靠近开合变化点的样本可能出现 `6` 次

其目的不是改变时间顺序，而是**提高临近夹爪状态切换时段的训练权重**，因为这类时刻通常最关键，也最难预测。

#### 3.2.3 一个样本里每种数据来自哪些时间步

下面表格描述的是 `SequenceSamplerKeyframeContinuous.sample_sequence()` 返回的**单个样本**在时序上的来源:

| 字段 | 读取的时间步 | 单样本形状 | 含义 |
|------|--------------|------------|------|
| `action` | 全部 `I = [I_cont, I_kf]` | `(T, 16)` | 双臂动作监督序列 |
| `state` | 全部 `I = [I_cont, I_kf]` | `(T, 16)` | 双臂状态序列 |
| `lang` | 全部 `I = [I_cont, I_kf]` | `(T, 1024)` | 语言嵌入，通常在一个 episode 内重复 |
| `point_cloud` | 仅 `I[0] = t0` | `(1, 6144, 3)` | 当前观测时刻的场景点云 |
| `dino_feature` | 仅 `I[0] = t0` | `(1, 6144, 384)` | 当前观测时刻的逐点语义特征 |
| `point_flow` | 仅 `I_kf = [K1, ..., KTk]` | `(Tk, N_flow, 3)` | 关键帧时刻的物体点流监督 |
| `initial_point_flow` | 固定 episode 第 `0` 帧 | `(1, N_flow, 3)` | 同一组初始物体点 |

需要特别注意:

1. `point_cloud` 和 `dino_feature` 不是按 `T` 个时间步加载的
   - 它们只取样本起点 `t0` 的视觉观测
   - 所以当前 PPI 本质上是“用当前观测预测未来动作与关键帧”

2. `point_flow` 只在关键帧时刻监督
   - 它不对应连续段 `I_cont`
   - 只对应关键帧段 `I_kf`

3. `initial_point_flow` 总是 episode 第 `0` 帧
   - 它表示同一组物体点在初始状态下的位置
   - 后续模型预测的是这组点在未来关键帧时刻的位置

#### 3.2.4 从采样样本到模型输入时，哪些时间步真正进入编码器

虽然 `state` 和 `lang` 在采样阶段是按 `(T, ·)` 返回的，但在 `PPI.forward()` / `predict_action()` 中，模型只取前 `n_obs_steps` 个观测时间步:

```python
this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
```

当前默认配置中:

- `n_obs_steps = 1`

所以真正进入 `ObservationEncoder` 的观测只来自 **`t0` 这个时刻**。带 batch 维后，各主要输入形状为:

| 编码器输入字段 | 实际使用时间步 | 形状 |
|---------------|----------------|------|
| `obs.point_cloud` | `t0` | `(B, 6144, 3)` |
| `obs.dino_feature` | `t0` | `(B, 6144, 384)` |
| `obs.agent_pos` | `t0` | `(B, 16)` |
| `obs.lang` | `t0` | `(B, 1024)` |
| `obs.initial_point_flow` | episode 第 `0` 帧 | `(B, N_flow, 3)` |

因此，`3.2` 的采样策略在时序上的真正含义是:

- 用 `t0` 的场景观测作为条件
- 预测从 `t0` 开始的一段连续动作
- 同时预测 `t0` 之后若干未来关键帧动作
- 并对这些关键帧时刻的物体点流进行监督

### 3.3 关键帧与连续动作的解耦预测

**文件**: `occ_grasp_models/ppi/model/diffusion/diffuser_actor_ppi.py` (行285-436)

PPI的核心创新是在扩散模型中**分离处理**关键帧动作和连续动作：

```python
def prediction_head(self,
                    trajectory_coord_left, traj_feats_left,
                    trajectory_coord_right, traj_feats_right,
                    pcd_coord, pcd_features,
                    timesteps, state_feat,
                    sampled_pcd_coord, sampled_pcd_features,
                    lang_feat,
                    pointflow_feat, pointflow_coords):

    # 索引定义
    continuous_start = 0
    keyframe_start = self.horizon_continuous
    context_start = self.horizon_continuous + self.horizon_keyframe

    # ============ 1. 点流预测 ============
    features_point_flow = torch.cat([point_flow_features, sampled_pcd_features], 0)
    features_point_flow = self.self_attn_point_flow(
        query=features_point_flow,
        query_pos=rel_pos_point_flow,
        diff_ts=time_embs,
        context=lang_feat
    )[-1]
    position_point_flow = self.predict_pos_point_flow(features_point_flow, ...)

    # ============ 2. 关键帧预测 ============
    # 提取关键帧特征
    features_keyframe = torch.cat([
        gripper_features_left[keyframe_start:context_start],
        gripper_features_right[keyframe_start:context_start],
        features_point_flow
    ], 0)

    features_keyframe = self.self_attn_keyframe(
        query=features_keyframe,
        query_pos=rel_pos_keyframe,
        diff_ts=time_embs,
        context=lang_feat
    )[-1]

    # 预测关键帧位置、旋转、夹爪
    position_left_keyframe = self.predict_pos_left_kf(features_keyframe, ...)
    rotation_left_keyframe = self.predict_rot_left_kf(features_keyframe, ...)
    openess_left_keyframe = self.openess_predictor_left_kf(position_features_left_keyframe)

    # ============ 3. 连续动作预测 (条件于关键帧) ============
    # 关键: detach关键帧特征，作为连续动作的条件
    features_keyframe_detach = features_keyframe.detach()

    features_continuous = torch.cat([
        gripper_features_left[continuous_start:keyframe_start],
        gripper_features_right[continuous_start:keyframe_start],
        features_keyframe_detach  # 将关键帧作为条件
    ], 0)

    features_continuous = self.self_attn_continuous(
        query=features_continuous,
        query_pos=rel_pos_continuous,
        diff_ts=time_embs,
        context=lang_feat
    )[-1]

    # 预测连续动作
    position_left_continuous = self.predict_pos_left_cn(features_continuous, ...)
    rotation_left_continuous = self.predict_rot_left_cn(features_continuous, ...)
    openess_left_continuous = self.openess_predictor_left_cn(...)

    # ============ 4. 拼接输出 ============
    position_left = torch.cat((position_left_continuous, position_left_keyframe), 1)
    rotation_left = torch.cat((rotation_left_continuous, rotation_left_keyframe), 1)
    openess_left = torch.cat((openess_left_continuous, openess_left_keyframe), 1)
```

**设计原理**:
1. **点流先行**: 首先预测物体点流，获得物体运动信息
2. **关键帧为锚**: 将点流信息注入关键帧预测，确定任务目标
3. **连续动作条件于关键帧**: `detach()`后的关键帧特征作为连续动作的条件，形成层次化生成

#### 3.3.1 扩散头输入序列在时间维上的布局

扩散头接收的动作目标序列，时间顺序与 `3.2` 采样器构造的顺序完全一致:

```text
[连续段, 关键帧段] = [t0, t0+1, ..., t0+Tc-1, K1, K2, ..., KTk]
```

其中:

- 连续段长度 `Tc = horizon_continuous`
- 关键帧段长度 `Tk = horizon_keyframe`
- 总长度 `T = Tc + Tk`

在默认配置下:

- `Tc = 50`
- `Tk = 4`
- `T = 54`

动作张量在 `policy/ppi.py` 中的形状为:

- `trajectory`: `(B, T, 16)`

其最后一维内容为:

```text
[left_pos(3), left_quat(4), right_pos(3), right_quat(4), left_open(1), right_open(1)]
```

但传入 `DiffusionHeadPPI` 的并不是完整16维动作，而是拆分后的左右臂位姿 token:

- `trajectory_left`: `(B, T, 7)`，对应左臂 `position(3) + quaternion(4)`
- `trajectory_right`: `(B, T, 7)`，对应右臂 `position(3) + quaternion(4)`

夹爪开合 `openess` 不作为扩散输入 token，而是从位置特征分支上单独预测出来，所以解耦预测实际上是:

- 位姿序列通过扩散过程建模
- 开合状态作为每个时间步的附加预测头输出

#### 3.3.2 第一步: 点流预测对应哪些时间步

点流分支的输入不是动作时间序列，而是:

- `pointflow_coords = initial_point_flow`: `(B, N_flow, 3)`，来自 episode 第 `0` 帧
- `pointflow_feat`: `(B, N_flow, 288)`，由 `initial_point_flow` 编码得到
- 当前观测时刻 `t0` 的场景特征:
  - `pcd_coord`: `(B, 6144, 3)`
  - `pcd_feat`: `(B, 6144, 288)`
  - `sampled_pcd_coord`: `(B, 1024, 3)`
  - `sampled_pcd_feat`: `(B, 1024, 288)`

它的输出是:

- `position_point_flow`: `(B, Tk * N_flow, 3)`

如果按时间维重新整理，可以写成:

```text
(B, Tk, N_flow, 3)
```

这意味着:

- 同一组初始物体点
- 在 `Tk` 个未来关键帧时刻上的位置
- 被一次性预测出来

所以点流分支在时间上**只对应关键帧时刻**:

```text
[K1, K2, ..., KTk]
```

这也与训练监督完全对齐，因为 `policy/ppi.py` 中的点流监督目标来自:

```python
point_flow_trajectory = npoint_flow[:, -self.horizon_keyframe:, :, :]
```

即只取样本最后 `Tk` 个时间步对应的关键帧点流。

#### 3.3.3 第二步: 关键帧预测对应哪些时间步

关键帧分支从整条动作轨迹中只取关键帧段:

```python
continuous_start = 0
keyframe_start = self.horizon_continuous
context_start = self.horizon_continuous + self.horizon_keyframe
```

因此:

- 左臂关键帧 query: `trajectory_left[:, keyframe_start:context_start]`
- 右臂关键帧 query: `trajectory_right[:, keyframe_start:context_start]`
- 对应的时间步正是:

```text
[K1, K2, ..., KTk]
```

在经过对场景特征的 `cross-attention` 后，每个机械臂得到:

- `gripper_features_left[keyframe_start:context_start]`: `(Tk, B, 288)`
- `gripper_features_right[keyframe_start:context_start]`: `(Tk, B, 288)`

随后再与 `features_point_flow` 拼接，形成关键帧预测的联合上下文:

```text
features_keyframe
= [left_keyframe_tokens, right_keyframe_tokens, point_flow_tokens, sampled_pcd_tokens]
```

其 token 级形状为:

```text
(2*Tk + N_flow + 1024, B, 288)
```

关键帧分支最终输出:

- 左臂位置: `(B, Tk, 3)`
- 左臂旋转: `(B, Tk, 4)`
- 左臂开合: `(B, Tk, 1)`
- 右臂位置: `(B, Tk, 3)`
- 右臂旋转: `(B, Tk, 4)`
- 右臂开合: `(B, Tk, 1)`

这些输出严格对应关键帧时间步:

```text
[K1, K2, ..., KTk]
```

其作用是先确定任务中的“高层锚点状态”，例如到达、抓取、放置等关键姿态。

#### 3.3.4 第三步: 连续动作预测如何依赖关键帧

连续动作分支只取连续段:

```python
features_continuous = torch.cat([
    gripper_features_left[continuous_start:keyframe_start],
    gripper_features_right[continuous_start:keyframe_start],
    features_keyframe_detach
], 0)
```

这说明连续分支的直接时间 query 对应的是:

```text
[t0, t0+1, ..., t0+Tc-1]
```

其中:

- 左臂连续段 token 形状为 `(Tc, B, 288)`
- 右臂连续段 token 形状为 `(Tc, B, 288)`

但它不是孤立预测，而是把 `features_keyframe_detach` 一并拼接进来作为条件。这里的 `detach()` 非常关键，它的作用是:

1. **保留关键帧信息作为条件**
   - 连续段在预测时可以“看到”关键帧目标
   - 因而知道整条动作轨迹最终要到哪些关键姿态

2. **阻断连续损失反向影响关键帧分支**
   - 连续动作 loss 不会再回传去修改关键帧特征
   - 从而实现“关键帧先定锚，连续段再跟随”的解耦训练

3. **形成层次化生成**
   - 先预测物体点流
   - 再预测关键帧动作
   - 最后在关键帧约束下预测连续轨迹

需要特别说明的是:

- 代码里虽然也定义了 `features_point_flow_detach`
- 但当前 continuous 分支真正拼接进去的是 `features_keyframe_detach`
- 因此 continuous 分支对点流的依赖是**间接的**
- 即点流信息先影响关键帧特征，再通过关键帧特征传递给连续动作分支

连续动作分支最终输出:

- 左臂位置: `(B, Tc, 3)`
- 左臂旋转: `(B, Tc, 4)`
- 左臂开合: `(B, Tc, 1)`
- 右臂位置: `(B, Tc, 3)`
- 右臂旋转: `(B, Tc, 4)`
- 右臂开合: `(B, Tc, 1)`

这些输出严格对应连续时间步:

```text
[t0, t0+1, ..., t0+Tc-1]
```

#### 3.3.5 最终输出如何在时间上重新拼回去

最后一步，模型会把连续段输出和关键帧段输出按时间顺序重新拼回去:

```python
position_left = torch.cat((position_left_continuous, position_left_keyframe), 1)
rotation_left = torch.cat((rotation_left_continuous, rotation_left_keyframe), 1)
openess_left  = torch.cat((openess_left_continuous,  openess_left_keyframe), 1)
```

右臂同理，因此每个机械臂最终得到:

- `position_*`: `(B, T, 3)`
- `rotation_*`: `(B, T, 4)`
- `openess_*`: `(B, T, 1)`

按单臂拼接后是:

- 左臂输出: `(B, T, 8)`
- 右臂输出: `(B, T, 8)`

再在策略层组合成完整双臂动作序列:

- `action_pred`: `(B, T, 16)`

而这个 `T` 维时间顺序仍然是:

```text
[t0, t0+1, ..., t0+Tc-1, K1, K2, ..., KTk]
```

因此，`3.3` 的“解耦预测”不是把两种动作拆成两条互不相关的支路，而是在**同一个统一时间轴**上做三阶段层次化预测:

1. 先预测关键帧时刻的物体点流
2. 再预测关键帧时刻的双臂动作
3. 最后预测连续时间步的稠密动作，并以关键帧特征为条件

#### 3.3.6 一张表看清 3.2 和 3.3 的时序对应关系

| 模块 | 时间步 | 输入内容 | 输出内容 | 形状 |
|------|--------|----------|----------|------|
| 场景观测编码 | `t0` | `point_cloud`, `dino_feature`, `agent_pos`, `lang` | 场景特征 | `(B,6144,288)` / `(B,1024,288)` 等 |
| 点流预测 | `K1..KTk` | `initial_point_flow` + `t0` 场景特征 | 关键帧点流 | `(B, Tk*N_flow, 3)` |
| 关键帧动作预测 | `K1..KTk` | 关键帧段动作 token + 点流特征 | 关键帧动作 | 每臂 `(B, Tk, 8)` |
| 连续动作预测 | `t0..t0+Tc-1` | 连续段动作 token + `detach` 后关键帧特征 | 连续动作 | 每臂 `(B, Tc, 8)` |
| 最终拼接 | `[连续段, 关键帧段]` | 连续输出 + 关键帧输出 | 双臂完整动作序列 | `(B, T, 16)` |

---

## 4. 物体点流(Point Flow)预测

### 4.1 点流概念

点流是追踪物体表面关键点在任务执行过程中位置变化的表示方法：

```
初始状态:                    目标状态:
   ●  ●  ●                     ●  ●  ●
     物体                        物体
   ●  ●  ●                     ●  ●  ●
   ↓  ↓  ↓
Point Flow = 同一组 keypoint 在各时刻的世界坐标
# 如 p_world(t) = R(t) @ p_object + t(t)
```

### 4.2 点流数据准备

**文件**: `occ_grasp_models/ppi/common/get_data_keyframe_continuous.py` (行69-91)

```python
for i in range(len(low_dim_obs)):
    gripper_pose = np.concatenate([
        low_dim_obs[i].left.gripper_pose,
        low_dim_obs[i].right.gripper_pose
    ])

    # 提取物体6D位姿
    current_object_pose = np.concatenate([
        low_dim_obs[i].object_6d_pose['position'],
        low_dim_obs[i].object_6d_pose['quaternion']
    ])

    object_pose.append(current_object_pose)

    # 初始点流位置 - 每个episode使用第0帧的物体点
    initial_point_flow.append(np.array([episode, 0]))
```

### 4.3 点流采样与加载

**文件**: `occ_grasp_models/ppi/common/sampler_keyframe_continuous.py` (行195-212)

```python
elif key == 'point_flow':
    point_flow = []
    indice_keyframe = indices[-self.sequence_length_keyframe:]  # 只在关键帧处预测点流
    for pointflow_path_idx in indice_keyframe:
        episode, step = input_arr[pointflow_path_idx]
        point_flow_path = os.path.join(
            self.point_flow_path,
            f'episode{episode}/{self.point_flow_type}/step{step:03d}.npy'
        )
        point_flow_data = np.load(point_flow_path)
        point_flow.append(point_flow_data)
    data = np.array(point_flow)

elif key == 'initial_point_flow':
    # 初始点流位置 - 使用当前时间步第0帧的物体点
    initial_point_flow_path = os.path.join(
        self.point_flow_path,
        f'episode{episode}/{self.point_flow_type}/step{step:03d}.npy'
    )
```

### 4.4 点流预测头

**文件**: `occ_grasp_models/ppi/model/diffusion/diffuser_actor_ppi.py` (行459-480)

```python
def predict_pos_point_flow(self, features, rel_pos, time_embs, start_id, end_id, instr_feats):
    # 自注意力处理
    position_features = self.position_self_attn_point_flow(
        query=features,
        query_pos=rel_pos,
        diff_ts=time_embs,
        context=instr_feats,
        context_pos=None
    )[-1]

    # 特征投影
    position_features = self.position_proj_point_flow(position_features)  # (B, N, 288)

    # 预测12D输出: 4个关键帧 × 3D位置
    position = self.position_predictor_point_flow(position_features)  # (B, N, 12)

    # 重塑为 (B, 4, N, 3)
    B, N, _ = position.shape
    position = position.reshape(B, N, 4, 3)
    position = einops.rearrange(position, "b n t x -> b t n x")
    position = position.reshape(B, -1, 3)  # (B, 4*N, 3)

    return position
```

### 4.5 点流损失函数

**文件**: `occ_grasp_models/ppi/policy/ppi.py` (行452-459)

```python
if self.predict_point_flow:
    for layer_pred_point_flow in pred_point_flow:
        trans_point_flow = layer_pred_point_flow[..., :3]
        loss_point_flow = (
            600 * F.l1_loss(trans_point_flow, point_flow_trajectory[..., :3], reduction='mean')
        )
        total_loss = total_loss + loss_point_flow
    loss_dict["point_flow_loss"] = loss_point_flow.item()
```

**注意**: 点流损失权重(600)远高于位置损失(30)和旋转损失(10)，强调物体运动预测的重要性。

---

## 5. 基于扩散的策略

### 5.1 DDPM调度器配置

**文件**: `occ_grasp_models/ppi/policy/ppi.py` (行54-63)

```python
# 位置使用scaled_linear调度
self.position_noise_scheduler = DDPMScheduler(
    num_train_timesteps=noise_scheduler_cfg.num_train_timesteps,
    beta_schedule="scaled_linear",
    prediction_type=noise_scheduler_cfg.prediction_type  # "epsilon"
)

# 旋转使用squaredcos_cap_v2调度 (更适合球面分布)
self.rotation_noise_scheduler = DDPMScheduler(
    num_train_timesteps=noise_scheduler_cfg.num_train_timesteps,
    beta_schedule="squaredcos_cap_v2",
    prediction_type=noise_scheduler_cfg.prediction_type
)
```

### 5.2 前向过程(训练)

**文件**: `occ_grasp_models/ppi/policy/ppi.py` (行320-394)

```python
def forward(self, batch):
    # 1. 归一化输入
    nobs = self.normalizer.normalize(batch['obs'])
    nactions = self.normalizer['action'].normalize(batch['action'])
    npoint_flow = self.normalizer['point_flow'].normalize(batch['point_flow'])

    # 2. 采样随机时间步
    timesteps = torch.randint(
        0, self.noise_scheduler_cfg.num_train_timesteps,
        (bsz,), device=trajectory.device
    ).long()

    # 3. 分别对位置和旋转添加噪声
    noise = torch.randn(trajectory.shape, device=trajectory.device)

    pos_left = self.position_noise_scheduler.add_noise(
        trajectory[..., :3], noise[..., :3], timesteps
    )
    rot_left = self.rotation_noise_scheduler.add_noise(
        trajectory[..., 3:7], noise[..., 3:7], timesteps
    )
    pos_right = self.position_noise_scheduler.add_noise(
        trajectory[..., 7:10], noise[..., 7:10], timesteps
    )
    rot_right = self.rotation_noise_scheduler.add_noise(
        trajectory[..., 10:14], noise[..., 10:14], timesteps
    )

    noisy_trajectory = torch.cat(
        (pos_left, rot_left, pos_right, rot_right, gt_openess_left, gt_openess_right), -1
    )

    # 4. 模型预测噪声
    pred_left, pred_right, pred_point_flow = self.model(
        noisy_trajectory_left, noisy_trajectory_right,
        timesteps, fixed_inputs
    )

    # 5. 计算损失
    loss_left = (
        30 * F.l1_loss(trans_left, noise[..., :3], reduction='mean')      # 位置
        + 10 * F.l1_loss(rot_left, noise[..., 3:7], reduction='mean')     # 旋转
        + 30 * F.l1_loss(openess_left, gt_openess_left, reduction='mean') # 夹爪(直接预测)
    )
```

### 5.3 逆向过程(推理)

**文件**: `occ_grasp_models/ppi/policy/ppi.py` (行143-228)

```python
def conditional_sample_diffuser_actor(self, condition_data, condition_mask, fixed_inputs, ...):
    # 设置推理步数
    self.position_noise_scheduler.set_timesteps(self.num_inference_steps)
    self.rotation_noise_scheduler.set_timesteps(self.num_inference_steps)

    # 初始化纯噪声
    noise = torch.randn(size=condition_data.shape, ...)

    # 添加初始噪声
    noise_t = torch.ones((len(condition_data),), ...).long() \
              .mul(self.position_noise_scheduler.timesteps[0])

    noise_pos_left = self.position_noise_scheduler.add_noise(
        condition_data[..., :3], noise[..., :3], noise_t
    )
    noise_rot_left = self.rotation_noise_scheduler.add_noise(
        condition_data[..., 3:7], noise[..., 3:7], noise_t
    )

    # 迭代去噪
    timesteps = self.position_noise_scheduler.timesteps
    for t in timesteps:
        # 模型预测
        out_left, out_right, out_point_flow = model(
            trajectory_left, trajectory_right,
            t * torch.ones(len(trajectory_left)).to(device).long(),
            fixed_inputs
        )
        out_left = out_left[-1]  # 只取最后一层

        # DDPM step - 位置
        pos_left = self.position_noise_scheduler.step(
            out_left[..., :3], t, trajectory_left[..., :3]
        ).prev_sample

        # DDPM step - 旋转
        rot_left = self.rotation_noise_scheduler.step(
            out_left[..., 3:7], t, trajectory_left[..., 3:7]
        ).prev_sample

        trajectory_left = torch.cat((pos_left, rot_left), -1)

    # 组合最终输出
    trajectory = torch.cat((
        pos_left, rot_left,
        pos_right, rot_right,
        out_left[..., 7:8],    # 左臂夹爪
        out_right[..., 7:8]    # 右臂夹爪
    ), -1)

    return trajectory, out_point_flow[-1]
```

### 5.4 注意力机制

**文件**: `occ_grasp_models/ppi/model/diffusion/diffuser_actor_ppi.py` (行97-134)

```python
# 交叉注意力: 轨迹特征 → 点云特征
self.cross_attn = FFWRelativeCrossAttentionModule(
    embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
)

# 点流交叉注意力
self.cross_attn_pointflow = FFWRelativeCrossAttentionModule(
    embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
)

# 关键帧自注意力 (4层)
self.self_attn_keyframe = FFWRelativeSelfAttentionModule(
    embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
)

# 连续动作自注意力 (4层)
self.self_attn_continuous = FFWRelativeSelfAttentionModule(
    embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
)

# 点流自注意力 (4层)
self.self_attn_point_flow = FFWRelativeSelfAttentionModule(
    embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
)
```

**AdaLN (Adaptive Layer Normalization)**: 根据扩散时间步调整归一化参数

```python
def encode_denoising_timestep(self, timestep, curr_gripper_features):
    time_feats = self.time_emb(timestep)  # Sinusoidal编码
    curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features.flatten(1))
    return time_feats + curr_gripper_feats  # 时间+状态条件
```

---

## 6. 训练流程

### 6.1 分布式训练配置

**文件**: `occ_grasp_models/train_ppi_ddp.py` (行38-79)

```python
class TrainPPIWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir=None):
        # 设置分布式环境
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.ngpus_per_node = torch.cuda.device_count()

        # 设置随机种子
        seed = cfg.training.seed + self.local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 初始化模型
        self.model: PPI = hydra.utils.instantiate(cfg.policy)

        # EMA模型
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
```

### 6.2 训练循环

**文件**: `occ_grasp_models/train_ppi_ddp.py` (行81-273)

```python
def run(self):
    # DDP包装
    if self.distributed:
        self.model = DDP(self.model, find_unused_parameters=True)

    for local_epoch_idx in range(cfg.training.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            batch = dict_apply(batch, lambda x: x.to(device))

            # 前向传播和损失计算
            raw_loss, loss_dict = self.model(batch)
            loss = raw_loss / cfg.training.gradient_accumulate_every
            loss.backward()

            # 梯度累积
            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

            # 更新EMA
            if cfg.training.use_ema:
                ema.step(self.model.module)
```

### 6.3 损失函数权重

```python
# 位置损失权重
position_weight = 30

# 旋转损失权重
rotation_weight = 10

# 夹爪损失权重
openess_weight = 30

# 点流损失权重
point_flow_weight = 600
```

---

## 7. 推理与评估

### 7.1 RLBench Agent封装

**文件**: `occ_grasp_models/agents/ppi/ppi_agent.py`、`occ_grasp_models/agents/ppi/launch_utils.py`

```python
class PPIAgent:
    def __init__(self, actor_network, ...):
        self._actor = actor_network
        # 读取离线语言嵌入
        with open(instruction_embeddings_path, "rb") as f:
            self.text_embedding_list = pickle.load(f)

    def build(self, training=False, device=None):
        self._actor = self._actor.to(device).train(training)
        self.fusion = Fusion(num_cam=6, feat_backbone='dinov2', device=device)

    def act(self, obs):
        useful_obs = {
            'point_cloud': point_cloud,               # (B, T, fps_num, C)
            'agent_pos': agent_pos,                   # (B, T, 16)
            'dino_feature': dino_feature,             # (B, T, fps_num, 384)
            'initial_point_flow': self.initial_pointflow,
            'lang': torch.from_numpy(self.text_embedding_list[self.lang_goal]).unsqueeze(0).unsqueeze(0)
        }
        result_dict = self._actor.predict_action(useful_obs)
        action = result_dict['action']  # (B, horizon, 16)
```

### 7.2 动作格式转换

```python
# PPI输出格式 (16D):
# [left_pos(3), left_quat(4), left_gripper(1), right_pos(3), right_quat(4), right_gripper(1)]

# 转换为RLBench BimanualMoveArmThenGripper格式 (18D):
# [right_pos(3), right_quat(4), right_gripper(1), ignore_collision(1),
#  left_pos(3), left_quat(4), left_gripper(1), ignore_collision(1)]

raw_action = torch.cat([
    result[7:10],           # 右臂位置
    quat1_normalized,       # 右臂四元数
    gripper_right,          # 右臂夹爪
    ignore_collisions,      # 忽略碰撞标志
    result[0:3],            # 左臂位置
    quat2_normalized,       # 左臂四元数
    gripper_left,           # 左臂夹爪
    ignore_collisions       # 忽略碰撞标志
], dim=-1)
```

---

## 8. 数据流与维度总结

### 8.1 输入数据维度

| 数据类型 | 维度 | 说明 |
|---------|------|------|
| point_cloud | (B, 6144, 3) | 场景点云坐标 |
| dino_feature | (B, 6144, 384) | 每点DINOv2语义特征 |
| agent_pos | (B, 16) | 双臂末端位姿 |
| lang | (B, 1024) | CLIP语言特征 |
| initial_point_flow | (B, N_flow, 3) | 初始物体关键点位置 |

### 8.2 中间特征维度

| 特征类型 | 维度 | 来源 |
|---------|------|------|
| pcd_feat | (B, 6144, 288) | DINOv2特征投影 |
| sampled_pcd_coord | (B, 1024, 3) | PointNet++ FPS采样点坐标 |
| sampled_pcd_feat | (B, 1024, 288) | PointNet++ 采样点特征 |
| state_feat | (B, 288) | 状态MLP编码 |
| lang_feat | (B, 288) | 语言MLP编码 |
| point_flow_feat | (B, N_flow, 288) | 点流位置编码 |

### 8.3 输出数据维度

| 输出类型 | 维度 | 说明 |
|---------|------|------|
| action_pred | (B, T, 16) | 预测动作序列 |
| point_flow_pred | (B, K*N_flow, 3) | 预测点流位置 |

其中 T = horizon_continuous + horizon_keyframe

### 8.4 动作空间分解

```
模型输出: [continuous(horizon_continuous × 16), keyframe(horizon_keyframe × 16)]

每步动作 (16D):
├── left_pos (3): 左臂末端位置
├── left_quat (4): 左臂末端四元数
├── left_gripper (1): 左臂夹爪开合
├── right_pos (3): 右臂末端位置
├── right_quat (4): 右臂末端四元数
└── right_gripper (1): 右臂夹爪开合
```

---

## 9. 与RLBench的兼容性

### 9.1 Action Mode配置

**文件**: `occ_grasp_models/conf/eval_ppi.yaml`

```yaml
arm_action_mode: 'BimanualEndEffectorPoseViaPlanning'
action_mode: 'BimanualMoveArmThenGripper'
```

### 9.2 兼容性机制

PPI与RLBench waypoint系统兼容的关键在于：

1. **输出末端执行器位姿**: PPI预测的是末端位姿而非关节角度
2. **使用相同的运动规划器**: `BimanualEndEffectorPoseViaPlanning`调用`get_path()`进行IK规划
3. **与Demo生成一致**: Demo生成和评估都使用相同的规划机制

```
┌───────────────────────────────────────────────────────────────────┐
│                     Demo 生成流程                                  │
│  Waypoint位姿 → get_path() → IK规划 → 关节轨迹 → 执行            │
└───────────────────────────────────────────────────────────────────┘
                              ↕ 使用相同的规划器
┌───────────────────────────────────────────────────────────────────┐
│                     PPI 评估流程                                   │
│  预测位姿 → get_path() → IK规划 → 关节轨迹 → 执行                 │
└───────────────────────────────────────────────────────────────────┘
```

---

## 10. 关键设计特点总结

### 10.1 层次化动作生成

1. **点流优先**: 首先预测物体运动轨迹
2. **关键帧定锚**: 基于点流确定关键位姿
3. **连续动作条件生成**: 在关键帧约束下生成平滑轨迹

### 10.2 3D场景理解

1. **多视角融合**: 多相机RGB-D融合为统一点云
2. **语义增强**: DINOv2特征提供丰富语义信息
3. **层次化编码**: PointNet++提取多尺度几何特征

### 10.3 分离的扩散过程

1. **位置与旋转分离**: 使用不同的噪声调度
2. **夹爪直接预测**: 二值夹爪状态不经过扩散

### 10.4 训练优化

1. **关键帧梯度截断**: `detach()`防止梯度从连续动作回传到关键帧
2. **重要性采样**: 夹爪状态变化点附近增加采样权重
3. **高点流损失权重**: 强调物体运动预测的重要性

---

## 11. 代码质量评价

### 优点
- 清晰的层次化架构设计
- 完善的3D特征提取pipeline
- 灵活的配置系统(Hydra)
- 支持分布式训练(DDP)
- 详细的日志记录(WandB)

### 可改进之处
- 代码中有较多`set_trace()`调试语句
- 部分函数缺少docstring
- 硬编码的维度数值可提取为配置参数

---

## 12. 总结

PPI是一种先进的双臂机器人操作学习方法，其核心创新包括：

1. **关键帧与连续控制的解耦**: 通过层次化生成，先预测关键帧位姿作为锚点，再条件生成连续轨迹
2. **物体点流监督**: 引入物体运动信息作为额外监督信号，增强对物体状态变化的理解
3. **3D语义场景表示**: 融合PointNet++几何特征和DINOv2语义特征，实现丰富的场景理解
4. **分离的扩散过程**: 针对位置、旋转、夹爪使用不同的处理策略

该方法通过输出末端执行器位姿与RLBench的waypoint系统天然兼容，避免了直接关节控制带来的waypoint验证问题。
