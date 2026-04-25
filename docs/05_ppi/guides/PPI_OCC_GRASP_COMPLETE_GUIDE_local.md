# occ_grasp_fall 中 PPI 完整操作指南（基于官方文档修订版）

本指南整合了 PPI 官方文档（`/home/hdliu/claude_repo/PPI/docs`）和 occ_grasp_fall 框架的实际情况，提供从原始数据准备到 RLBench 仿真评估的完整流程。

## 参考文档

- `/home/hdliu/claude_repo/PPI/docs/INSTALLATION.md` - 环境安装
- `/home/hdliu/claude_repo/PPI/docs/DATA_PREPROCESSION.md` - 数据预处理
- `/home/hdliu/claude_repo/PPI/docs/TRAINING_DETAIL.md` - 训练细节
- `/home/hdliu/claude_repo/PPI/docs/INFERENCE.md` - 推理评估
- `/home/hdliu/occ_grasp_fall/PPI_data_structure_detailed_explanation.md` - 数据结构
- `/home/hdliu/occ_grasp_fall/PPI_EVAL_PIPELINE_ANALYSIS.md` - 评估流程
- `/home/hdliu/occ_grasp_fall/ppi_method_analysis.md` - 方法分析
- `/home/hdliu/claude_repo/PPI/PPI_OCC_GRASP_MIGRATION_MANIFEST.md` - 迁移清单

## 目标链路

```
/mnt/rlbench_data/*.train (训练数据)
/mnt/rlbench_data/*.test (测试数据)
  → 预处理（point cloud + dino + point flow + norm stats）
  → PPI 训练（train_ppi_ddp.py）
  → RLBench 仿真评估（eval_ppi.py）
```

本指南仅覆盖**仿真 PPI**，不涉及真机部署。

---

## 0. 核心要点

1. **训练入口**：`occ_grasp_models/train_ppi_ddp.py`（不是 `train.py method=PPI`）
2. **预处理脚本位置**：`occ_grasp_models/scripts/ppi/data_generation/`
3. **工作目录**：始终在 `/home/hdliu/occ_grasp_fall/occ_grasp_models` 下执行
4. **数据组织**：
   - 训练需要：原始训练 demo（`/mnt/rlbench_data/*.train`）+ 预处理产物（point_cloud/dino/point_flow/norm_stats）
   - 评估需要：原始测试 demo（`/mnt/rlbench_data/*.test`）+ 语言嵌入 + 训练权重（评估时在线重建视觉特征）
5. **目录适配**：训练和测试数据需通过软链接映射为无后缀目录
6. **验证集来源**：从 `.train` 内部按 `val_ratio` 划分，不自动使用 `.val` 目录
7. **本指南默认工作方式**：直接手动修改配置文件/脚本，再执行对应命令；不再依赖一组临时 `export` 环境变量

---

## 1. 当前任务与数据现状

### 1.1 四个新任务

| 任务目录 | 简称 | 训练数据 | 测试数据 |
|---------|------|---------|---------|
| `bimanual_edge_phone` | `edge_phone` | 150 episodes (`/mnt/rlbench_data/*.train`) | 30 episodes (`/mnt/rlbench_data/*.test`) |
| `bimanual_pivot_phone` | `pivot_phone` | 150 episodes (`/mnt/rlbench_data/*.train`) | 30 episodes (`/mnt/rlbench_data/*.test`) |
| `bimanual_pick_plate` | `pick_plate` | 150 episodes (`/mnt/rlbench_data/*.train`) | 30 episodes (`/mnt/rlbench_data/*.test`) |
| `bimanual_pick_fork` | `pick_fork` | 150 episodes (`/mnt/rlbench_data/*.train`) | 30 episodes (`/mnt/rlbench_data/*.test`) |

**说明**：
- 训练数据位于：`/mnt/rlbench_data/<task>.train/all_variations/episodes`
- 测试数据位于：`/mnt/rlbench_data/<task>.test/all_variations/episodes`
- 验证集（`/mnt/rlbench_data/<task>.val`，50 episodes）暂不使用，训练时从训练集按 `val_ratio=0.2` 划分

### 1.2 命名约定

- **TASK_KEY**：Hydra 配置短名（如 `edge_phone`），用于 yaml 文件名和日志标签
- **TASK_DIR**：真实 RLBench 任务目录名（如 `bimanual_edge_phone`），用于数据路径和任务加载

### 1.3 推荐起始参数

基于官方任务配置模式和当前任务特点：

| 任务 | TASK_KEY | TASK_DIR | PF_TEXT_PROMPT | SAM_CAMERAS | PF_PROMPT_TYPE | EPISODE_LENGTH | QUERY_FREQ |
|------|----------|----------|----------------|-------------|----------------|----------------|------------|
| edge phone | `edge_phone` | `bimanual_edge_phone` | `a black phone` | `["front"]` | `box` | 400 | 10 |
| pivot phone | `pivot_phone` | `bimanual_pivot_phone` | `a black phone` | `["front"]` | `box` | 400 | 10 |
| pick plate | `pick_plate` | `bimanual_pick_plate` | `a white plate` | `["front"]` | `box` | 400 | 10 |
| pick fork | `pick_fork` | `bimanual_pick_fork` | `a white fork` | `["front"]` | `box` | 400 | 10 |

**说明**：
- `prompt_type=box`：对小物体和细长物体通常比 `point` 更稳定
- `SAM_CAMERAS=["front"]`：本文档后续统一按你的决定采用 `front` 视角
- `BOUNDING_BOX=[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]`：统一起始值，若点云采样报错再调整

### 1.4 手动修改基线（推荐）

本指南下文默认采用”直接修改配置文件 / 脚本再运行”的方式，不再依赖临时 `export` 变量。以 `edge_phone` 为例，后续需要统一代入的参数如下：

| 参数 | 值 |
|------|----|
| `TASK_KEY` | `edge_phone` |
| `TASK_DIR` | `bimanual_edge_phone` |
| `EP_START` | `0` |
| `EP_END` | `149` |
| `TEST_EPISODES` | `30` |
| `BOUNDING_BOX` | `[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]` |
| `PF_TEXT_PROMPT` | `a black phone` |
| `SAM_CAMERAS` | `["front"]` |
| `PF_PROMPT_TYPE` | `box` |
| `EPISODE_LENGTH` | `400` |
| `QUERY_FREQ` | `10` |

推荐的修改落点如下：

- `ppi/config/task/<TASK_KEY>.yaml`：任务名、数据路径、episode 范围、`prediction_type`、`point_flow_type`
- `scripts/ppi/data_generation/save_ptc.py`：底部 `__main__` 中的 `task`、路径、`bounding_box`、episode 切分范围
- `scripts/ppi/data_generation/save_dino.py`：底部 `__main__` 中的 `device`、`task`、路径、episode 切分范围
- `scripts/ppi/data_generation/save_point_flow.py`：底部 `__main__` 中的 `task`、`text_prompt`、`cameras=["front"]`、`prompt_type`、`process_episodes(...)`
- `scripts/ppi/data_generation/save_norm_stats.py`：不建议直接沿用原 `main()`，建议复制为任务专用脚本后再改常量
- `scripts/ppi/training/ddp_train_<TASK_KEY>.sh`：建议从已有模板复制，改 `task`、`wandb_name`、`stats_filepath`、batch size、epochs
- `scripts/ppi/inference/evaluate_ppi_<TASK_KEY>.sh`：建议从已有模板复制，改 `task_name`、`tasks`、`episode_length=400`、`query_freq=10`、`sam_cameras=["front"]`

---

## 2. 环境准备（一次性）

### 2.1 进入工作目录

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
source ~/.bashrc
conda activate ppi
```

若 `conda activate` 报错，改用：

```bash
conda run --no-capture-output -n ppi python -c "import rlbench, pyrep; print('ppi env ok')"
```

### 2.2 CoppeliaSim 环境变量

评估阶段必需（参考官方 `INSTALLATION.md`）：

```bash
export COPPELIASIM_ROOT=/path/to/your/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

设置后重新 source：

```bash
source ~/.bashrc
conda activate ppi
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
```

### 2.3 预训练权重检查

当前工作区已包含所需权重，无需重复下载：

```bash
ls -lh \
  ../pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth \
  ../pretrained_models/sam_vit_b_01ec64.pth \
  ../pretrained_models/groundingdino_swinb_cogcoor.pth
```

同时确认依赖仓库：

```bash
ls -d \
  ../repos/dinov2 \
  ../repos/segment-anything \
  ../repos/GroundingDINO \
  ../repos/RLBench \
  ../repos/PyRep \
  ../repos/YARR
```

### 2.4 无桌面服务器显示环境

参考官方 `INFERENCE.md`，无显示器服务器需启动 Xvfb：

```bash
Xvfb :99 -screen 0 1024x768x16 >/tmp/xvfb_ppi.log 2>&1 &
export DISPLAY=:99
```

---

## 3. 数据目录映射

训练和评估代码期望无后缀的任务目录，需建立软链接适配。

### 3.1 创建目录结构

```bash
mkdir -p data/training_raw data/eval_raw
mkdir -p data/training_processed/point_cloud
mkdir -p data/training_processed/dino_feature
mkdir -p data/training_processed/point_flow
mkdir -p data/training_processed/norm_stats
```

### 3.2 建立软链接

```bash
for task in \
  bimanual_edge_phone \
  bimanual_pivot_phone \
  bimanual_pick_plate \
  bimanual_pick_fork
do
  ln -sfn "/mnt/rlbench_data/${task}.train" "data/training_raw/${task}"
  ln -sfn "/mnt/rlbench_data/${task}.test"  "data/eval_raw/${task}"
done
```

### 3.3 验证映射

```bash
readlink -f "data/training_raw/bimanual_edge_phone"
readlink -f "data/eval_raw/bimanual_edge_phone"
```

**原因**：
- 训练 yaml 默认路径：`data/training_raw/<task>/all_variations/episodes`
- 评估代码拼接路径：`demo_path/<task_name>/all_variations/episodes`

### 3.4 必须先补齐 `object_6d_pose` 元数据

这一点是本指南原先遗漏、但对**完整 PPI** 至关重要的前置条件。

先给结论：

- 如果你按本文当前配置走**完整 PPI**（`prediction_type=keyframe_continuous` + `policy.predict_point_flow=true`），那么**训练源 `/mnt/rlbench_data/*.train` 中缺少 `object_6d_pose` 会是实打实的问题**，不能忽略。
- 如果你明确改成**无 point flow 的消融/基线**，这个问题可以绕开；但那已经不是本文档这条“完整 PPI”流程。

#### 为什么这是硬阻塞

当前代码链路里，`object_6d_pose` 不是可有可无的信息，而是被直接读取：

- `scripts/ppi/data_generation/save_point_flow.py`
  - 用 `low_dim_obs[0].object_6d_pose['position']` 获取目标物体初始 3D 位置
  - 用 `low_dim_obs[0].object_6d_pose["matrix"]` / `low_dim_obs[step].object_6d_pose["matrix"]` 把初始采样点从世界坐标变到物体坐标系，再传播回每一帧
- `ppi/common/get_data_keyframe_continuous.py`
  - 直接读取 `low_dim_obs[i].object_6d_pose['position']` 和 `['quaternion']`
  - `save_norm_stats.py`、`RLBench2Dataset(prediction_type=keyframe_continuous)` 最终都会经过这里

我已实际核验当前数据与代码：

- `/mnt/rlbench_data/bimanual_edge_phone.train/.../low_dim_obs.pkl` 等四个新任务 demo 中，`BimanualObservation` **没有** `object_6d_pose`
- 直接调用 `GetDataKeyframeContinuous.process_episodes(0, 0, [], 10)` 会报：

```python
AttributeError: 'BimanualObservation' object has no attribute 'object_6d_pose'
```

同时，你当前 RLBench 分叉里这部分采集逻辑也确实没有启用：

- `../repos/RLBench/rlbench/backend/observation.py` 里 `object_6d_pose` 字段被注释掉了
- `../repos/RLBench/rlbench/backend/scene.py` 里采集 `object_6d_pose` 的代码块也被注释掉了

#### 这是否能用现有 `misc` 字段替代

不能直接替代。

虽然你当前 demo 的 `misc` 里有：

- `contact_position/contact_quaternion`
- `grasp_position/grasp_quaternion`
- `affordance_position/affordance_quaternion`

但这些是**任务语义点/接触点标签**，不是每一帧目标物体的完整刚体 6D 位姿。它们不能无损替代 `save_point_flow.py` 当前使用的“物体坐标系下点集传播”逻辑。

#### 推荐解决方案：先补采 `object_6d_pose`，再整批重跑预处理

这是唯一和当前完整 PPI 代码链路严格一致、风险最低的方案。

##### 第 1 步：在 RLBench 采集链里重新启用 `object_6d_pose`

先修改 `../repos/RLBench/rlbench/backend/observation.py`，给 `Observation` 恢复该字段：

```python
@dataclass
class Observation:
    perception_data: Dict[str, np.ndarray]
    task_low_dim_state: np.ndarray
    misc: Dict[str, Any]
    object_6d_pose: Dict[str, np.ndarray]
```

再修改 `../repos/RLBench/rlbench/backend/scene.py`，在 `Scene` 类里加入一个目标物体解析函数，并在 `get_observation()` 中写入 `object_6d_pose`。

推荐写法不要像原始 PPI 的 `scene_gen.py` 那样写死 `Shape('ball')`，而是优先复用任务里已经定义好的 `self.target_object`：

```python
def _get_target_object_for_pose(self):
    target = getattr(self.task, "target_object", None)
    if target is not None:
        return target

    fallback = {
        "bimanual_edge_phone": "Phone",
        "bimanual_pivot_phone": "Phone",
        "bimanual_pick_plate": "plate",
        "bimanual_pick_fork": "Fork_phy",
    }
    task_name = self.task.get_name()
    if task_name not in fallback:
        raise RuntimeError(
            f"Task {task_name} does not expose target_object; "
            "please add self.target_object in init_task() or extend fallback."
        )
    return Shape(fallback[task_name])
```

然后在 `get_observation()` 里、`observation_data.update({... "misc": self._get_misc()})` 之后，补上：

```python
target_obj = self._get_target_object_for_pose()
object_6d_pose = {
    "position": np.array(target_obj.get_position(), dtype=np.float32),
    "orientation": np.array(target_obj.get_orientation(), dtype=np.float32),
    "quaternion": np.array(target_obj.get_quaternion(), dtype=np.float32),
    "matrix": np.array(target_obj.get_matrix(), dtype=np.float32),
}

observation_data.update({
    "task_low_dim_state": task_low_dim_state,
    "perception_data": perception_data,
    "misc": self._get_misc(),
    "object_6d_pose": object_6d_pose,
})
```

#### 第 2 步：确认这四个任务的目标物体句柄

我已核对当前任务实现，这四个任务都已经在 `init_task()` 中显式设置了 `self.target_object`：

- `bimanual_edge_phone` → `Shape('Phone')`
- `bimanual_pivot_phone` → `Shape('Phone')`
- `bimanual_pick_plate` → `Shape('plate')`
- `bimanual_pick_fork` → `Shape('Fork_phy')`

因此，按上面的“优先读 `self.target_object`”方案改，适配这四个任务是成立的。

#### 第 3 步：重新采集原始 demo，不要混用旧数据

补上 `object_6d_pose` 之后，必须重新生成原始 `.train/.val/.test` demo。

注意：

- **不要**把“旧版无 `object_6d_pose` 的 episode”和“新版有 `object_6d_pose` 的 episode”混在同一个任务目录里
- 最稳妥的做法是先输出到新的根目录，例如 `/mnt/rlbench_data_pose`，内部继续保留 `<task>.train / <task>.val / <task>.test` 目录命名
- 验证完成后，再决定是整体替换 `/mnt/rlbench_data` 下对应任务目录，还是把本指南里的软链接改到新的 pose 根目录

如果你还要求和原始训练/测试场景配置严格一致，那么请沿用你自己的可复现采集设置（同样的 `PYTHONHASHSEED`、相同的 train/test seed 约定）。

#### 第 4 步：在开始预处理前，先做字段验收

至少抽查一个 episode：

```bash
conda run --no-capture-output -n ppi python - <<'PY'
import pickle
path = "/mnt/rlbench_data/bimanual_edge_phone.train/all_variations/episodes/episode0/low_dim_obs.pkl"
with open(path, "rb") as f:
    demo = pickle.load(f)
pose = demo[0].object_6d_pose
print(sorted(pose.keys()))
print("position:", pose["position"])
print("matrix shape:", pose["matrix"].shape)
PY
```

预期至少要满足：

- 能成功访问 `demo[0].object_6d_pose`
- 包含 `position / quaternion / matrix`
- `matrix.shape == (4, 4)`

#### 第 5 步：raw demo 一旦替换，四个预处理步骤都要重跑

只要你换了原始 demo，就不要继续复用旧的预处理产物。建议删除或备份以下目录后，重新执行本文第 6 节全部步骤：

- `data/training_processed/point_cloud/<task>/...`
- `data/training_processed/dino_feature/<task>/...`
- `data/training_processed/point_flow/<task>/...`
- `data/training_processed/norm_stats/norm_stats_<...>.pth`

原因：

- `point_flow` 直接依赖新的 `object_6d_pose`
- `norm_stats` 依赖新的 `point_flow`
- 若你重新采集了 raw demo，point cloud / dino 也应与新 demo 保持一一对应

#### 第 6 步：如果暂时不想重采，只能改走无 point flow 基线

这只是临时退路，不是本文这条“完整 PPI”流程。

若你短期内无法补采 `object_6d_pose`，则必须同步放弃 point flow 相关链路，例如：

- 不运行 `save_point_flow.py`
- 不使用 `prediction_type=keyframe_continuous + predict_point_flow=true`
- 改成当前代码里真正不依赖 `object_6d_pose` 的纯 `continuous` 路线

否则，后续流程会在 Step 3、Step 4 或 dataset 构建阶段直接失败。

### 3.5 替代方案：从现有 demo 数据中补充 `object_6d_pose`（无需重新采集）

这一节修正前文旧版 `3.5` 的几个问题，并给出当前更推荐的“**无需重采 raw demo，但精确对齐到物体本体坐标系**”方案。更完整的推导、常量矩阵来源和仿真验证过程，见：

- `/home/hdliu/occ_grasp_fall/PPI_contact_vs_object_pose_analysis.md`

#### 修正后的核心结论

- 当前 demo 的 `misc['contact_position']` / `misc['contact_quaternion']` 记录的是 contact dummy（`push_pt` / `press_pt`）的**世界位姿**，不是目标物体本体坐标系的 pose。
- 原始 PPI 的 `object_6d_pose` 来自 `environment_gen.py -> scene_gen.py` 的采集链：在 `scene_gen.py` 里直接通过 `Shape(...)` 读取目标物体句柄的真实 pose。这也是 `3.4` 想恢复的 ground-truth 方案。
- 对当前四个任务，contact dummy 与目标物体刚性固连，两者间存在**任务相关但时间上恒定的 SE(3) 偏移**。因此可以从 contact 世界位姿**精确恢复**目标物体本体世界位姿，无需重采 raw demo。
- 若只看 `save_point_flow.py` 中的世界系↔物体系点传播，直接把 contact frame 当 surrogate object frame 也能得到与 object frame 相同的世界点轨迹，因为常量偏移会在 `T_t T_0^{-1}` 中抵消。
- 但若直接把 contact pose 填进 `object_6d_pose['position'] / ['quaternion']`，语义上并不等于原始 PPI 的 object pose。**本指南不再推荐“直接把 contact pose 当 object pose”作为主方案，而是推荐先做精确恢复。**
- 当前代码里真正能完全绕开 `object_6d_pose` 的只有 **pure continuous** 路线。`keyframe`、`keyframe_continuous`，以及 `what_condition='keypose_continuous'` 的 ablation 仍会读取该字段。

#### 为什么旧版 `3.5` 需要修正

旧版 `3.5` 的核心方向是对的，但有几处表述和实现细节不够准确：

1. `contact_position/contact_quaternion` 不能直接表述为“就是物体坐标系在世界系下的 pose”。
   - 当前 `../repos/RLBench/rlbench/backend/scene.py` 里收集的是语义 dummy：
     - `contact` 来自 `push_pt` / `press_pt`
     - `grasp` 来自 `grasp_pt`
     - `affordance` 来自 `box_edge` / `wall_pivot`
   - 这些是刚性附着在物体上的语义 frame，不是目标物体句柄本身的 canonical frame。

2. `save_point_flow.py` 的当前活跃路径主要依赖 `matrix`，不是旧版 `3.5` 所强调的 `object_6d_pose['position']` 投影提示。
   - 当前 `predict_tracks()` 走的是 `get_sam_prompt_point_from_foundationdino(...)`
   - 基于 `object_6d_pose['position']` 投影的 `get_sam_prompt_point(...)` 在当前配置下不是主路径

3. 旧版“直接改 `read_pkl()`”的示例代码不能原样照抄。
   - `save_point_flow.py` 里的 `read_pkl()` 不只读 `low_dim_obs.pkl`，还读 `variation_descriptions.pkl`
   - `get_data_keyframe_continuous.py` 里的 `read_pkl()` 不只读 `low_dim_obs.pkl`，还读 `variation_descriptions.pkl` 和 `instruction_embeddings.pkl`
   - 如果无差别地对返回值执行 `for o in obs: o.misc[...]`，会把语言/embedding pkl 一起弄坏

4. 旧版验证脚本只能证明“补出了一个字段”，不能证明“它与原始 PPI 的 object pose 语义一致”。要做到语义一致，必须引入任务相关的常量逆偏移矩阵。

#### 原始 PPI 是怎么拿到 `object_6d_pose` 的

原始 PPI 不是从 `misc` 推 object pose，而是在采集 raw demo 时直接从仿真目标物体句柄读取：

- `/home/hdliu/claude_repo/PPI/repos/RLBench/rlbench/environment_gen.py` 导入的是 `rlbench.backend.scene_gen.Scene`
- `/home/hdliu/claude_repo/PPI/repos/RLBench/rlbench/backend/scene_gen.py` 中，`get_observation()` 每帧直接执行：
  - `object_nh = Shape('ball')`（按任务手动改成对应 object name）
  - `get_position() / get_orientation() / get_quaternion() / get_matrix()`
  - 写入 `object_6d_pose`
- `/home/hdliu/claude_repo/PPI/docs/DATA_GENERATION.md` 也明确要求：对每个任务在 `scene_gen.py` 中指定 object name

因此，原始 PPI 的 `object_6d_pose` 本质上是“**仿真里目标 object handle 的真实 pose**”。这与“直接拿 contact dummy pose 代替”不是同一个语义。

#### 为什么无需重采也能做精确对齐

根据 `/home/hdliu/occ_grasp_fall/PPI_contact_vs_object_pose_analysis.md` 的仿真核验结果，当前四个任务中的 contact dummy 都与目标物体刚性固连。设：

- $T_{\text{obj}}^{(t)}$：目标物体本体在时刻 $t$ 的世界位姿
- $T_{\text{contact}}^{(t)}$：contact dummy 在时刻 $t$ 的世界位姿
- $T_{\text{offset}}$：contact dummy 相对于目标物体本体的**常量**位姿偏移

则有：

$$
T_{\text{contact}}^{(t)} = T_{\text{obj}}^{(t)} \cdot T_{\text{offset}}
$$

所以可精确恢复：

$$
T_{\text{obj}}^{(t)} = T_{\text{contact}}^{(t)} \cdot T_{\text{offset}}^{-1}
$$

对 point flow 而言，如果只用 contact frame 做刚体传播，也有：

$$
T_{\text{contact}}^{(t)} \left(T_{\text{contact}}^{(0)}\right)^{-1}
= T_{\text{obj}}^{(t)} T_{\text{offset}} \left(T_{\text{offset}}\right)^{-1} \left(T_{\text{obj}}^{(0)}\right)^{-1}
= T_{\text{obj}}^{(t)} \left(T_{\text{obj}}^{(0)}\right)^{-1}
$$

这解释了为什么：

- **直接使用 contact frame**：足以支撑 point flow 的刚体传播
- **使用逆偏移恢复 object frame**：既保留 point flow 等价性，又恢复原始 PPI 语义

#### 当前四个任务的常量偏移概览

| 任务 | Contact Dummy | 目标物体 | 平移偏移模长 | 旋转偏移 |
|------|---------------|----------|--------------|----------|
| `bimanual_edge_phone` | `push_pt` | `Phone` | 10.66 cm | 120.00° |
| `bimanual_pivot_phone` | `push_pt` | `Phone` | 10.86 cm | 120.00° |
| `bimanual_pick_plate` | `press_pt` | `plate` | 6.87 cm | 177.62° |
| `bimanual_pick_fork` | `press_pt` | `Fork_phy` | 9.40 cm | 119.92° |

这些偏移不可忽略，所以若你希望 `object_6d_pose` 与原始 PPI 的 object frame 语义对齐，应当乘上任务相关的 `T_OFFSET_INV`。

#### 推荐实现：精确恢复 object 本体 pose

下面给出当前四个任务可直接使用的恢复代码。这里的四元数顺序与 PyRep / scipy 保持一致，均为 **`[x, y, z, w]`**。

```python
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

# 用法: T_obj_world = T_contact_world @ T_OFFSET_INV[task_name]
T_OFFSET_INV = {
    "bimanual_edge_phone": np.array([
        [ 0.000000357627798, -0.999999999999758, -0.000000596046490, -0.004460703107312],
        [ 0.000000238418686, -0.000000596046405,  0.999999999999794, -0.106504077672761],
        [-0.999999999999908, -0.000000357627940,  0.000000238418473, -0.000902679728312],
        [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
    ], dtype=np.float64),
    "bimanual_pivot_phone": np.array([
        [ 0.000000178813465, -0.999999999999471, -0.000001013278951, -0.004495898648837],
        [ 0.000000894069682, -0.000001013278791,  0.999999999999087, -0.108443408182149],
        [-0.999999999999584, -0.000000178814371,  0.000000894069501,  0.002094409457517],
        [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
    ], dtype=np.float64),
    "bimanual_pick_plate": np.array([
        [ 0.002598125880784, -0.058969175473169, -0.998256423012606, -0.067807748975981],
        [-0.000314574836860, -0.998259791077145,  0.058968555699510, -0.008845579165724],
        [-0.999996575386426,  0.000160818620698, -0.002612154817698,  0.006806277136513],
        [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
    ], dtype=np.float64),
    "bimanual_pick_fork": np.array([
        [-0.008017196545178, -0.999967469322108, -0.000885921607486, -0.000882002254333],
        [ 0.011323440070544, -0.000976678403156,  0.999935410816251, -0.093989696356541],
        [-0.999903747499990,  0.008006647040768,  0.011330901934082, -0.000689878880687],
        [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
    ], dtype=np.float64),
}


def recover_object_6d_pose(obs, task_name):
    """
    从 contact dummy 的世界位姿恢复目标物体本体的 object_6d_pose。

    注意：
    - task_name 必须是 RLBench 真实任务目录名，如 "bimanual_edge_phone"
    - contact_quaternion 采用 PyRep / scipy 兼容顺序: [x, y, z, w]
    """
    contact_pos = np.asarray(obs.misc["contact_position"], dtype=np.float64)
    contact_quat = np.asarray(obs.misc["contact_quaternion"], dtype=np.float64)

    T_contact = np.eye(4, dtype=np.float64)
    T_contact[:3, :3] = Rotation.from_quat(contact_quat).as_matrix()
    T_contact[:3, 3] = contact_pos

    T_obj = T_contact @ T_OFFSET_INV[task_name]
    R_obj = Rotation.from_matrix(T_obj[:3, :3])

    return {
        "position": T_obj[:3, 3].astype(np.float32),
        "quaternion": R_obj.as_quat().astype(np.float32),
        "orientation": R_obj.as_euler("xyz").astype(np.float32),
        "matrix": T_obj.astype(np.float32),
    }


def patch_demo_with_object_6d_pose(demo, task_name):
    """
    demo: 从 low_dim_obs.pkl 读出的 list[BimanualObservation]
    """
    for obs in demo:
        if not hasattr(obs, "object_6d_pose"):
            obs.object_6d_pose = recover_object_6d_pose(obs, task_name)
    return demo


def maybe_patch_low_dim_obs(file_path, obj, task_name):
    """
    仅对 low_dim_obs.pkl 的返回值打补丁，避免误伤语言/embedding pkl。
    """
    if os.path.basename(file_path) != "low_dim_obs.pkl":
        return obj
    return patch_demo_with_object_6d_pose(obj, task_name)
```

#### 安全落地方式

不要直接把旧版 `3.5` 里那段“无差别改 `read_pkl()`”原样贴到当前代码里。当前更安全的做法是：

1. 在公共位置放一个恢复 helper。
   - 例如新建 `ppi/common/object_pose_from_contact.py`

2. 只在读取 `low_dim_obs.pkl` 后补丁，不要动 `variation_descriptions.pkl` / `instruction_embeddings.pkl`。

3. 对当前 full-PPI 主链，至少需要覆盖以下读取点：
   - `scripts/ppi/data_generation/save_point_flow.py`
   - `ppi/common/get_data_keyframe_continuous.py`

4. 如果你后续还会跑 `prediction_type=keyframe` 的 ablation，也要同步覆盖：
   - `ppi/common/get_data_keyframe.py`

参考写法如下：

```python
def read_pkl(self, file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    if os.path.basename(file_path) == "low_dim_obs.pkl":
        # save_point_flow.py 里可直接使用 self.task
        obj = patch_demo_with_object_6d_pose(obj, self.task)

    return obj
```

对于 `GetDataKeyframeContinuous` / `GetDataKeyframe` 这类没有 `self.task` 的类，可以从 `self.data_path` 推出真实任务名：

```python
task_name = os.path.basename(os.path.dirname(os.path.dirname(self.data_path)))
```

因为 `self.data_path` 形如：

```python
data/training_raw/bimanual_edge_phone/all_variations/episodes
```

所以两层 `dirname(...)` 后的 basename 正好就是 `bimanual_edge_phone`。

#### 为什么这个版本比“直接拿 contact pose 填进去”更好

1. 对 `save_point_flow.py` 的刚体点传播，二者都能成立。
   - 常量偏移会在 `T_t T_0^{-1}` 中抵消
   - 实际数值核验中，world-point propagation 的差异可以到浮点精度级别

2. 对 `get_data_keyframe_continuous.py` / `get_data_keyframe.py` 中的 7D `object_pose` 字段，这个版本语义更正确。
   - 直接使用 contact pose：得到的是“contact dummy 的 pose”
   - 使用本节恢复版：得到的是“目标物体本体的 pose”

3. 当前 sim policy 虽然基本不直接消费 `object_pose`，但恢复成 object 本体 frame 仍然更接近原始 PPI，也更利于后续维护和复用。

#### 验证步骤

应用补丁后，至少先抽查一个 episode：

```bash
conda run --no-capture-output -n ppi python - <<'PY'
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

T_OFFSET_INV_EDGE_PHONE = np.array([
    [ 0.000000357627798, -0.999999999999758, -0.000000596046490, -0.004460703107312],
    [ 0.000000238418686, -0.000000596046405,  0.999999999999794, -0.106504077672761],
    [-0.999999999999908, -0.000000357627940,  0.000000238418473, -0.000902679728312],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
], dtype=np.float64)

path = "/mnt/rlbench_data/bimanual_edge_phone.train/all_variations/episodes/episode0/low_dim_obs.pkl"
with open(path, "rb") as f:
    demo = pickle.load(f)

obs0 = demo[0]
contact_pos = np.asarray(obs0.misc["contact_position"], dtype=np.float64)
contact_quat = np.asarray(obs0.misc["contact_quaternion"], dtype=np.float64)

T_contact = np.eye(4, dtype=np.float64)
T_contact[:3, :3] = Rotation.from_quat(contact_quat).as_matrix()
T_contact[:3, 3] = contact_pos
T_obj = T_contact @ T_OFFSET_INV_EDGE_PHONE
obj_quat = Rotation.from_matrix(T_obj[:3, :3]).as_quat()

print("contact_position:", contact_pos)
print("object_position :", T_obj[:3, 3])
print("delta_norm      :", np.linalg.norm(T_obj[:3, 3] - contact_pos))
print("matrix_shape    :", T_obj.shape)
print("quat_norm       :", np.linalg.norm(obj_quat))
PY
```

建议至少检查以下几项：

- `delta_norm` 应明显大于 0，说明恢复出的 object pose 不是简单抄写 contact pose
- `matrix_shape == (4, 4)`
- 恢复出的旋转矩阵是正交矩阵，四元数范数接近 1

#### 本节方案与 `3.4` 的关系

- `3.4`：最标准，直接恢复原始 PPI 那种“采集时读取 object handle pose”的 ground-truth 流程，但代价是重采 raw demo
- `3.5`：不重采 raw demo，而是在读取现有 `low_dim_obs.pkl` 时用 task-specific 常量偏移把 contact pose 精确恢复成 object pose

如果你的目标是：

- **严格贴近原始 PPI 语义**：优先 `3.4`
- **在不重采 demo 的前提下，让当前 full-PPI 主链尽量精确且安全地跑通**：优先采用本节的“精确恢复版 `3.5`”

#### 后续步骤

采用本节的精确恢复版后，可以继续执行第 6 节预处理流程，无需重新采集 raw demo。

---

## 4. 生成语言嵌入

参考官方 `DATA_PREPROCESSION.md` 的 `instruction_embeddings.pkl` 要求。

输出路径：`data/training_processed/instruction_embeddings.pkl`

```bash
python - <<'PY'
import glob
import os
import pickle
import torch

from helpers.clip.core.clip import build_model, load_clip, tokenize

data_root = "/mnt/rlbench_data"
instructions = set()

# 从训练数据收集指令
for task_dir in sorted(glob.glob(os.path.join(data_root, "*.train"))):
    episodes_root = os.path.join(task_dir, "all_variations", "episodes")
    for episode_dir in sorted(glob.glob(os.path.join(episodes_root, "episode*"))):
        path = os.path.join(episode_dir, "variation_descriptions.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            descriptions = pickle.load(f)
        for text in descriptions:
            instructions.add(text)

# 从测试数据收集指令
for task_dir in sorted(glob.glob(os.path.join(data_root, "*.test"))):
    episodes_root = os.path.join(task_dir, "all_variations", "episodes")
    for episode_dir in sorted(glob.glob(os.path.join(episodes_root, "episode*"))):
        path = os.path.join(episode_dir, "variation_descriptions.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            descriptions = pickle.load(f)
        for text in descriptions:
            instructions.add(text)

print(f"found {len(instructions)} unique instructions")

model, _ = load_clip("RN50", jit=False, device="cpu")
clip_model = build_model(model.state_dict()).to("cpu").eval()

embedding_dict = {}
with torch.no_grad():
    for text in sorted(instructions):
        tokens = tokenize([text])
        lang_feats, _ = clip_model.encode_text_with_embeddings(tokens)
        embedding_dict[text] = lang_feats[0].float().cpu().numpy()
        print(text)

out_path = "data/training_processed/instruction_embeddings.pkl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(embedding_dict, f)

print(f"saved to {out_path}")
PY
```

---

## 5. 创建任务配置

官方任务 yaml 不包含新任务，需手动创建。推荐直接新建 `ppi/config/task/edge_phone.yaml`，内容如下：

```yaml
name: edge_phone

task_name: edge_phone
dataset_task_name: bimanual_edge_phone

dataset:
  _target_: ppi.dataset.rlbench2_dataset.RLBench2Dataset
  data_path: data/training_raw/bimanual_edge_phone/all_variations/episodes
  pcd_path: data/training_processed/point_cloud/bimanual_edge_phone/all_variations/episodes
  dino_path: data/training_processed/dino_feature/bimanual_edge_phone/all_variations/episodes
  lang_emb_path: data/training_processed/instruction_embeddings.pkl
  stats_filepath: data/training_processed/norm_stats/norm_stats_bimanual_edge_phone_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth
  point_flow_path: data/training_processed/point_flow/bimanual_edge_phone/all_variations/episodes
  horizon_keyframe: ${horizon_keyframe}
  horizon_continuous: ${horizon_continuous}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  start: 0
  end: 149
  pcd_fps: 6144
  skip_ep: []
  kp_num: 10
  val_ratio: 0.2
  max_train_episodes: 150
  pcd_type: rgb_pcd_rps6144
  prediction_type: keyframe_continuous
  point_flow_type: world_ordered_rps200
  add_openess_sampling: true
```

对另外三个任务，复制这份 yaml 后仅需同步替换：

- `name`
- `task_name`
- `dataset_task_name`
- `data_path / pcd_path / dino_path / point_flow_path / stats_filepath` 中的任务目录名

**说明**：
- `val_ratio=0.2`：从 150 个 `.train` episode 中分出 20% 做验证
- `add_openess_sampling=true`：抓取类任务推荐开启

---

## 6. 数据预处理（四步骤）

参考官方 `DATA_PREPROCESSION.md` 的四步流程。

### Smoke Test 建议

首次运行建议先把四个预处理脚本里的 episode 上限改成 `2`，只处理 `episode0-2`；确认无误后，再统一改回 `149`。


### 6.1 Step 1: Point Cloud

参考官方 `scripts/data_generation/save_ptc.py`。

编辑 `scripts/ppi/data_generation/save_ptc.py` 底部 `if __name__ == "__main__":`，至少把 `task`、路径、`bounding_box` 和固定的 `0-99` episode 切分改成下面这种形式：

```python
if __name__ == "__main__":
    task = "bimanual_edge_phone"
    data_path = f"data/training_raw/{task}/all_variations/episodes"
    target_path = f"data/training_processed/point_cloud/{task}/all_variations/episodes"
    getpcd = pcd(data_path, target_path)
    cameras = ["over_shoulder_left", "over_shoulder_right", "overhead", "front", "wrist_left", "wrist_right"]
    pcd_type = "rgb_pcd_rps6144"
    bounding_box = np.array([[-0.5, -0.55, 0.77], [1.1, 0.55, 1.98]])

    EP_START = 0
    EP_END = 149  # smoke test 时先改成 2
    CHUNK_SIZE = 10
    MAX_THREADS = (EP_END - EP_START + CHUNK_SIZE) // CHUNK_SIZE + 1

    worker_threads = []
    for i in range(MAX_THREADS):
        start = EP_START + i * CHUNK_SIZE
        if start > EP_END:
            break
        end = min(start + CHUNK_SIZE - 1, EP_END)
        thread = threading.Thread(
            target=getpcd.process_episodes,
            args=(start, end, cameras, pcd_type, bounding_box),
        )
        worker_threads.append(thread)
        thread.start()

    for thread in worker_threads:
        thread.join()
```

运行命令：

```bash
python scripts/ppi/data_generation/save_ptc.py
```

检查输出：

```bash
find "data/training_processed/point_cloud/bimanual_edge_phone/all_variations/episodes" -name 'step*.npy' | wc -l
```

### 6.2 Step 2: Dino Feature

参考官方 `scripts/data_generation/save_dino.py`。

编辑 `scripts/ppi/data_generation/save_dino.py` 底部 `if __name__ == "__main__":`，把 `task`、路径和 episode 切分改成：

```python
if __name__ == "__main__":
    device = "cuda:0"
    task = "bimanual_edge_phone"
    pcd_path = f"data/training_processed/point_cloud/{task}/all_variations/episodes"
    data_path = f"data/training_raw/{task}/all_variations/episodes"
    target_path = f"data/training_processed/dino_feature/{task}/all_variations/episodes"
    ptc_type = "rgb_pcd_rps6144"

    EP_START = 0
    EP_END = 149  # smoke test 时先改成 2
    CHUNK_SIZE = 10
    NUM_WORKERS = (EP_END - EP_START + CHUNK_SIZE) // CHUNK_SIZE + 1

    worker_threads = []
    for i in range(NUM_WORKERS):
        fusion = Fusion(num_cam=6, feat_backbone="dinov2", device=device)
        start = EP_START + i * CHUNK_SIZE
        if start > EP_END:
            break
        end = min(start + CHUNK_SIZE - 1, EP_END)
        thread = threading.Thread(
            target=process_episodes,
            args=(start, end, task, fusion, pcd_path, data_path, target_path, ptc_type, device),
        )
        worker_threads.append(thread)
        thread.start()

    for thread in worker_threads:
        thread.join()
```

运行命令：

```bash
python scripts/ppi/data_generation/save_dino.py
```

检查输出：

```bash
find "data/training_processed/dino_feature/bimanual_edge_phone/all_variations/episodes" -name 'step*.npy' | wc -l
```


### 6.3 Step 3: Point Flow

参考官方 `scripts/data_generation/save_point_flow.py`。此步骤依赖 GroundingDINO 和 SAM。

编辑 `scripts/ppi/data_generation/save_point_flow.py` 底部 `if __name__ == "__main__":`，至少修改这些位置：

- `task = "bimanual_pivot_phone"`
- `data_path / target_path`
- `pcd_type = "world_ordered_rps200"`
- `text_prompt = "a black phone"`
- `cameras = ["front"]`
- `prompt_type = "box"`
- `get_point_flow.process_episodes(0, 149, cameras, pcd_type)`，smoke test 时先写成 `0, 2`

修改完成后运行：

```bash
python scripts/ppi/data_generation/save_point_flow.py
```

检查输出：

```bash
find "data/training_processed/point_flow/bimanual_edge_phone/all_variations/episodes" -name 'step*.npy' | wc -l
```


### 6.4 Step 4: Norm Stats

`scripts/ppi/data_generation/save_norm_stats.py` 原始版本不推荐直接沿用，因为它硬编码了：

- `0-99` episode 范围
- `cuda:5 / cuda:6 / cuda:7`
- 输出文件名后缀 `_new`

建议复制成任务专用脚本，例如 `scripts/ppi/data_generation/save_norm_stats_edge_phone.py`。

**重要说明**：此步骤会加载所有 episode 的 point_cloud 和 dino_feature 到内存中计算统计量，内存占用较大（约 100GB+）。如果内存不足，可以考虑：
1. 分批计算后合并统计量
2. 使用流式统计（已在下面脚本中实现）

脚本内容如下：

```python
import os
import numpy as np
import torch

from ppi.common.get_data_keyframe_continuous import GetDataKeyframeContinuous
from ppi.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

TASK = "bimanual_edge_phone"
EP_START = 0
EP_END = 149  # smoke test 时先改成 2
KP_NUM = 10
PCD_TYPE = "rgb_pcd_rps6144"
POINT_FLOW_TYPE = "world_ordered_rps200"

class RunningStats:
    def __init__(self, dim):
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, array):
        array = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])
        self.count += array.shape[0]
        self.sum += array.sum(axis=0, dtype=np.float64)
        self.sumsq += np.square(array, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.min = np.minimum(self.min, array.min(axis=0))
        self.max = np.maximum(self.max, array.max(axis=0))

    def finalize(self):
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)
        return {
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }

def make_single_field(stats_dict, output_min=-1.0, output_max=1.0, range_eps=1e-4):
    input_min = torch.from_numpy(stats_dict["min"])
    input_max = torch.from_numpy(stats_dict["max"])
    input_mean = torch.from_numpy(stats_dict["mean"])
    input_std = torch.from_numpy(stats_dict["std"])

    input_range = input_max - input_min
    ignore = input_range < range_eps
    input_range = input_range.clone()
    input_range[ignore] = output_max - output_min

    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore] = (output_max + output_min) / 2 - input_min[ignore]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict={
            "min": input_min,
            "max": input_max,
            "mean": input_mean,
            "std": input_std,
        },
    )

data_path = f"data/training_raw/{TASK}/all_variations/episodes"
pcd_root = f"data/training_processed/point_cloud/{TASK}/all_variations/episodes"
dino_root = f"data/training_processed/dino_feature/{TASK}/all_variations/episodes"
point_flow_root = f"data/training_processed/point_flow/{TASK}/all_variations/episodes"
lang_emb_path = "data/training_processed/instruction_embeddings.pkl"

gd = GetDataKeyframeContinuous(data_path=data_path, lang_emb_path=lang_emb_path)
root = gd.process_episodes(EP_START, EP_END, [], KP_NUM)
data = root["data"]

stats = {
    "action": RunningStats(dim=data["action"].shape[-1]),
    "agent_pos": RunningStats(dim=data["state"].shape[-1]),
    "lang": RunningStats(dim=data["lang"].shape[-1]),
    "point_cloud": RunningStats(dim=6),
    "dino_feature": RunningStats(dim=384),
    "point_flow": RunningStats(dim=3),
    "initial_point_flow": RunningStats(dim=3),
}

stats["action"].update(data["action"])
stats["agent_pos"].update(data["state"])
stats["lang"].update(data["lang"])

for episode, step in data["point_cloud"]:
    path = os.path.join(pcd_root, f"episode{episode}/{PCD_TYPE}/step{step:03d}.npy")
    stats["point_cloud"].update(np.load(path))

for episode, step in data["dino_feature"]:
    path = os.path.join(dino_root, f"episode{episode}/{PCD_TYPE}/step{step:03d}.npy")
    stats["dino_feature"].update(np.load(path))

for episode, step in data["point_flow"]:
    path = os.path.join(point_flow_root, f"episode{episode}/{POINT_FLOW_TYPE}/step{step:03d}.npy")
    stats["point_flow"].update(np.load(path))

for episode, step in data["initial_point_flow"]:
    path = os.path.join(point_flow_root, f"episode{episode}/{POINT_FLOW_TYPE}/step{step:03d}.npy")
    stats["initial_point_flow"].update(np.load(path))

normalizer = LinearNormalizer()
for key, tracker in stats.items():
    normalizer[key] = make_single_field(tracker.finalize())

out_path = (
    f"data/training_processed/norm_stats/"
    f"norm_stats_{TASK}_{PCD_TYPE}_keyframe_continuous_{POINT_FLOW_TYPE}.pth"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(normalizer.state_dict(), out_path)
print(f"saved to {out_path}")
```

运行命令：

```bash
python scripts/ppi/data_generation/save_norm_stats_pivot_phone.py
```

目标输出：

```
data/training_processed/norm_stats/norm_stats_bimanual_pivot_phone_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth
```

---


## 7. 训练 PPI

参考官方 `TRAINING_DETAIL.md` 和 `scripts/training/ddp_train_*.sh`。

### 7.1 推荐做法：复制模板再改

```bash
cp scripts/ppi/training/ddp_train_box.sh scripts/ppi/training/ddp_train_edge_phone.sh
```

然后把 `scripts/ppi/training/ddp_train_edge_phone.sh` 改成：

```bash
source ~/.bashrc
conda activate ppi
export PYTHONUNBUFFERED=1
ngpus=1
export WANDB__SERVICE_WAIT=600
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes 1 --nproc_per_node $ngpus --master_port 10004 train_ppi_ddp.py \
    task='edge_phone' \
    name='train_ppi_ddp' \
    addition_info='20260329_baseline' \
    wandb_name='ppi_edge_phone' \
    logging.mode=offline \
    n_obs_steps=1 \
    n_action_steps=54 \
    policy.use_lang=true \
    policy.what_condition='ppi' \
    policy.predict_point_flow=true \
    task.dataset.pcd_fps=6144 \
    task.dataset.pcd_type='rgb_pcd_rps6144' \
    task.dataset.point_flow_type='world_ordered_rps200' \
    task.dataset.kp_num=10 \
    task.dataset.prediction_type='keyframe_continuous' \
    task.dataset.stats_filepath='data/training_processed/norm_stats/norm_stats_bimanual_edge_phone_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth' \
    horizon_keyframe=4 \
    horizon_continuous=50 \
    dataloader.batch_size=16 \
    val_dataloader.batch_size=16 \
    training.num_epochs=500
```

运行命令：

```bash
bash scripts/ppi/training/ddp_train_edge_phone.sh
```

若改成 8 卡训练，仅需把 `ngpus=8`、`CUDA_VISIBLE_DEVICES` 和 batch size 改成官方同量级设置，例如 `64 / 64`。

### 7.2 训练输出

训练日志和权重保存在：

```
exp_logs/ckpt/bimanual_edge_phone/train_ppi_ddp_edge_phone_ppi_<ADDITION_INFO>_seed0/
├── checkpoints/
│   ├── latest_model.pth.tar
│   ├── epoch50_model.pth.tar
│   ├── epoch100_model.pth.tar
│   └── ...
└── logs/
```

### 7.3 训练建议

参考官方 `TRAINING_DETAIL.md`：
- 严格遵循官方超参数设置以复现论文结果
- 推荐训练更多 steps 和 epochs
- 官方任务的推荐 checkpoint epoch 数：ball(400-500), box(250-350), drawer(350-450)
- 如果服务器不支持 8-GPU 并行训练，调整超参数后成功率可能略有波动
- 增加 batch size 可能减少相同 epoch 下的 steps，可能降低模型成功率

---


## 8. 本地评测准备

这一节只讨论**本地机器上的仿真评测**。训练已经在别的远程机器完成，本地只负责：

1. 读取 `exp_logs/ckpt` 下已有训练结果
2. 读取 `/mnt/rlbench_data/<task>.test`
3. 手动建立评测所需软链接
4. 运行 `scripts/ppi/inference/evaluate_ppi_<task>.sh`

### 8.1 本机评测前置条件

先在本机确认下面这些前提都满足：

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
source ~/.bashrc
conda activate ppi
```

- `cd .../occ_grasp_models`：后续所有相对路径都基于这里
- `source ~/.bashrc`：让 `conda activate ppi` 生效
- `conda activate ppi`：切到评测用环境

本地这台机器有显示屏，所以默认按**非 headless**方式评测。仍然要先设置 CoppeliaSim：

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export DISPLAY=:0
```

- `COPPELIASIM_ROOT`：CoppeliaSim 根目录
- `LD_LIBRARY_PATH`：让 PyRep / CoppeliaSim 动态库可见
- `QT_QPA_PLATFORM_PLUGIN_PATH`：让 Qt 插件路径正确
- `DISPLAY=:0`：把 RLBench GUI 指向本机主显示器

然后核对评测依赖文件：

```bash
ls -lh data/training_processed/instruction_embeddings.pkl
ls -lh \
  ../pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth \
  ../pretrained_models/sam_vit_b_01ec64.pth \
  ../pretrained_models/groundingdino_swinb_cogcoor.pth
test -f ../repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py && echo ok
```

- 第 1 条：确认语言嵌入在本机已补齐；PPI 评测会直接加载它
- 第 2 条：确认 DINO / SAM / GroundingDINO 权重都在本机
- 第 3 条：确认 GroundingDINO 配置文件存在

另外要补一句当前代码层面的硬约束：

- PPI 仿真评测实际上要求 **CUDA 可用**
- `PPIAgent` 里会直接把 SAM 放到 `cuda`，并多次调用 `torch.cuda.synchronize()`
- 所以这条链路不能按 CPU-only 方式运行

再确认测试集目录：

```bash
find /mnt/rlbench_data -maxdepth 2 -type d | rg 'bimanual_edge_phone|bimanual_pivot_phone|bimanual_pick_plate|bimanual_pick_fork'
```

### 8.2 当前本地可直接评测的四个训练输出

截至当前工作区，`exp_logs/ckpt` 下可直接用于评测的四个 run 如下：

| 任务 | `TASK_DIR` | `RUN_NAME` | 当前推荐 `CKPT_NAME` |
|------|------------|------------|----------------------|
| edge phone | `bimanual_edge_phone` | `train_ppi_ddp_edge_phone_ppi_20260405_baseline_seed0` | `epoch=0100-val_loss=18.5103054_model` |
| pivot phone | `bimanual_pivot_phone` | `train_ppi_ddp_pivot_phone_ppi_20260405_baseline_seed0` | `epoch=0020-val_loss=21.9991779_model` |
| pick plate | `bimanual_pick_plate` | `train_ppi_ddp_pick_plate_ppi_20260405_baseline_seed0` | `epoch=0200-val_loss=16.1768589_model` |
| pick fork | `bimanual_pick_fork` | `train_ppi_ddp_pick_fork_ppi_20260405_baseline_seed0` | `latest_model` |

可用下面的命令再核对一次：

```bash
for d in exp_logs/ckpt/bimanual_*/*/checkpoints; do
  echo "===== ${d} ====="
  ls -1 "${d}"
done
```

### 8.3 手动建立 `data/eval_raw` 软链接

评测时 `rlbench.demo_path` 必须指向 `data/eval_raw` 根目录，而不是具体任务目录。因此要先手动建立下面这些软链接：

```bash
mkdir -p data/eval_raw

ln -sfn /mnt/rlbench_data/bimanual_edge_phone.test  data/eval_raw/bimanual_edge_phone
ln -sfn /mnt/rlbench_data/bimanual_pivot_phone.test data/eval_raw/bimanual_pivot_phone
ln -sfn /mnt/rlbench_data/bimanual_pick_plate.test  data/eval_raw/bimanual_pick_plate
ln -sfn /mnt/rlbench_data/bimanual_pick_fork.test   data/eval_raw/bimanual_pick_fork
```

校验命令：

```bash
readlink -f data/eval_raw/bimanual_edge_phone
readlink -f data/eval_raw/bimanual_pivot_phone
readlink -f data/eval_raw/bimanual_pick_plate
readlink -f data/eval_raw/bimanual_pick_fork
```

### 8.4 手动建立 `eval_weights` 软链接

当前 `eval_ppi.py` 的权重装载链是：

```text
framework.weightsdir/<eval_type>/<weight_name>/checkpoints/<ckpt_name>.pth.tar
```

所以即便真实训练结果在：

```text
exp_logs/ckpt/<TASK_DIR>/<RUN_NAME>/checkpoints/<CKPT_NAME>.pth.tar
```

评测前也仍然要手动建出一个“评测视角”的目录：

```text
eval_weights/<TASK_DIR>/0/<RUN_NAME> -> exp_logs/ckpt/<TASK_DIR>/<RUN_NAME>
```

直接执行：

```bash
mkdir -p eval_weights/bimanual_edge_phone/0
mkdir -p eval_weights/bimanual_pivot_phone/0
mkdir -p eval_weights/bimanual_pick_plate/0
mkdir -p eval_weights/bimanual_pick_fork/0

ln -sfn "$(readlink -f exp_logs/ckpt/bimanual_edge_phone/train_ppi_ddp_edge_phone_ppi_20260405_baseline_seed0)" \
  "eval_weights/bimanual_edge_phone/0/train_ppi_ddp_edge_phone_ppi_20260405_baseline_seed0"

ln -sfn "$(readlink -f exp_logs/ckpt/bimanual_pivot_phone/train_ppi_ddp_pivot_phone_ppi_20260405_baseline_seed0)" \
  "eval_weights/bimanual_pivot_phone/0/train_ppi_ddp_pivot_phone_ppi_20260405_baseline_seed0"

ln -sfn "$(readlink -f exp_logs/ckpt/bimanual_pick_plate/train_ppi_ddp_pick_plate_ppi_20260405_baseline_seed0)" \
  "eval_weights/bimanual_pick_plate/0/train_ppi_ddp_pick_plate_ppi_20260405_baseline_seed0"

ln -sfn "$(readlink -f exp_logs/ckpt/bimanual_pick_fork/train_ppi_ddp_pick_fork_ppi_20260405_baseline_seed0)" \
  "eval_weights/bimanual_pick_fork/0/train_ppi_ddp_pick_fork_ppi_20260405_baseline_seed0"
```

校验命令：

```bash
readlink -f eval_weights/bimanual_edge_phone/0/train_ppi_ddp_edge_phone_ppi_20260405_baseline_seed0
readlink -f eval_weights/bimanual_pivot_phone/0/train_ppi_ddp_pivot_phone_ppi_20260405_baseline_seed0
readlink -f eval_weights/bimanual_pick_plate/0/train_ppi_ddp_pick_plate_ppi_20260405_baseline_seed0
readlink -f eval_weights/bimanual_pick_fork/0/train_ppi_ddp_pick_fork_ppi_20260405_baseline_seed0
```

### 8.5 checkpoint 选择规则

这里的 `CKPT_NAME` 指 checkpoint 文件名去掉 `.pth.tar` 之后的部分。例如：

- `latest_model`
- `epoch100_model`
- `epoch=0100-val_loss=18.5103054_model`

注意：

1. `CKPT_NAME` 不要带 `.pth.tar`
2. `framework.eval_type=0` 只是评测槽位目录名，不是 epoch
3. `framework.weight_name` 必须和 `eval_weights/<TASK_DIR>/0/<RUN_NAME>` 这一级目录名完全一致
4. 如果你换了新的 `RUN_NAME`，要同时改脚本里的 `framework.weight_name`，并重建对应软链接

---

## 9. RLBench 仿真评测

这一节恢复成与原有任务相同的脚本风格：每个任务一个独立的 `evaluate_ppi_<task>.sh`，脚本内部直接写完整 `python eval_ppi.py \` 命令。

### 9.1 `evaluate_ppi_edge_phone.sh` 完整内容

文件：`occ_grasp_models/scripts/ppi/inference/evaluate_ppi_edge_phone.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate ppi
export DISPLAY=:0
CUDA_VISIBLE_DEVICES=0 python eval_ppi.py \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=30 \
    framework.csv_logging=true \
    framework.tensorboard_logging=false \
    framework.eval_save_metrics=true \
    framework.eval_type=0 \
    framework.weight_name="train_ppi_ddp_edge_phone_ppi_20260405_baseline_seed0" \
    framework.ckpt_name="epoch=0100-val_loss=18.5103054_model" \
    framework.jump_step=1 \
    framework.weightsdir="$(pwd)/eval_weights/bimanual_edge_phone" \
    framework.logdir="$(pwd)/eval_logs" \
    framework.eval_envs=1 \
    framework.eval_processes=1 \
    rlbench.headless=false \
    rlbench.episode_length=400 \
    rlbench.task_name="edge_phone" \
    rlbench.tasks=[bimanual_edge_phone] \
    rlbench.demo_path="$(pwd)/data/eval_raw" \
    rlbench.cameras=["over_shoulder_left","over_shoulder_right","overhead","wrist_right","wrist_left","front"] \
    rlbench.cameras_pcd=["over_shoulder_left","over_shoulder_right","overhead","wrist_right","wrist_left","front"] \
    rlbench.camera_resolution=[256,256] \
    rlbench.include_lang_goal_in_obs=true \
    rlbench.gripper_mode='BimanualDiscrete' \
    rlbench.arm_action_mode='BimanualEndEffectorPoseViaPlanning' \
    rlbench.action_mode='BimanualMoveArmThenGripper' \
    rlbench.query_freq=10 \
    method.policy.horizon_keyframe=4 \
    method.policy.horizon_continuous=50 \
    method.policy.n_obs_steps=1 \
    method.policy.n_action_steps=54 \
    method.policy.bounding_box=[[-0.5,-0.55,0.77],[1.1,0.55,1.98]] \
    method.policy.fps_num=6144 \
    method.policy.prediction_type=keyframe_continuous \
    method.policy.what_condition=ppi \
    method.policy.predict_point_flow=true \
    method.policy.use_lang=true \
    method.policy.pointflow_num=200 \
    method.policy.text_prompt="a black phone" \
    method.policy.prompt_type="box" \
    method.policy.sample_type="rps" \
    method.policy.num_inference_steps=1000 \
    method.policy.sam_cameras=["front"] \
    method.policy.sam_checkpoint_path="$(pwd)/../pretrained_models/sam_vit_b_01ec64.pth" \
    method.policy.gdino_config_path="$(pwd)/../repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    method.policy.gdino_checkpoint_path="$(pwd)/../pretrained_models/groundingdino_swinb_cogcoor.pth" \
    method.policy.instruction_embeddings_path="$(pwd)/data/training_processed/instruction_embeddings.pkl" \
    cinematic_recorder.enabled=false \
    cinematic_recorder.save_path="$(pwd)/eval_videos/bimanual_edge_phone/train_ppi_ddp_edge_phone_ppi_20260405_baseline_seed0"
```

### 9.2 `evaluate_ppi_pivot_phone.sh` 完整内容

文件：`occ_grasp_models/scripts/ppi/inference/evaluate_ppi_pivot_phone.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate ppi
export DISPLAY=:0
CUDA_VISIBLE_DEVICES=0 python eval_ppi.py \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=30 \
    framework.csv_logging=true \
    framework.tensorboard_logging=false \
    framework.eval_save_metrics=true \
    framework.eval_type=0 \
    framework.weight_name="train_ppi_ddp_pivot_phone_ppi_20260405_baseline_seed0" \
    framework.ckpt_name="epoch=0020-val_loss=21.9991779_model" \
    framework.jump_step=1 \
    framework.weightsdir="$(pwd)/eval_weights/bimanual_pivot_phone" \
    framework.logdir="$(pwd)/eval_logs" \
    framework.eval_envs=1 \
    framework.eval_processes=1 \
    rlbench.headless=false \
    rlbench.episode_length=400 \
    rlbench.task_name="pivot_phone" \
    rlbench.tasks=[bimanual_pivot_phone] \
    rlbench.demo_path="$(pwd)/data/eval_raw" \
    rlbench.cameras=["over_shoulder_left","over_shoulder_right","overhead","wrist_right","wrist_left","front"] \
    rlbench.cameras_pcd=["over_shoulder_left","over_shoulder_right","overhead","wrist_right","wrist_left","front"] \
    rlbench.camera_resolution=[256,256] \
    rlbench.include_lang_goal_in_obs=true \
    rlbench.gripper_mode='BimanualDiscrete' \
    rlbench.arm_action_mode='BimanualEndEffectorPoseViaPlanning' \
    rlbench.action_mode='BimanualMoveArmThenGripper' \
    rlbench.query_freq=10 \
    method.policy.horizon_keyframe=4 \
    method.policy.horizon_continuous=50 \
    method.policy.n_obs_steps=1 \
    method.policy.n_action_steps=54 \
    method.policy.bounding_box=[[-0.5,-0.55,0.77],[1.1,0.55,1.98]] \
    method.policy.fps_num=6144 \
    method.policy.prediction_type=keyframe_continuous \
    method.policy.what_condition=ppi \
    method.policy.predict_point_flow=true \
    method.policy.use_lang=true \
    method.policy.pointflow_num=200 \
    method.policy.text_prompt="a black phone" \
    method.policy.prompt_type="box" \
    method.policy.sample_type="rps" \
    method.policy.num_inference_steps=1000 \
    method.policy.sam_cameras=["front"] \
    method.policy.sam_checkpoint_path="$(pwd)/../pretrained_models/sam_vit_b_01ec64.pth" \
    method.policy.gdino_config_path="$(pwd)/../repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    method.policy.gdino_checkpoint_path="$(pwd)/../pretrained_models/groundingdino_swinb_cogcoor.pth" \
    method.policy.instruction_embeddings_path="$(pwd)/data/training_processed/instruction_embeddings.pkl" \
    cinematic_recorder.enabled=false \
    cinematic_recorder.save_path="$(pwd)/eval_videos/bimanual_pivot_phone/train_ppi_ddp_pivot_phone_ppi_20260405_baseline_seed0"
```

### 9.3 `evaluate_ppi_pick_plate.sh` 完整内容

文件：`occ_grasp_models/scripts/ppi/inference/evaluate_ppi_pick_plate.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate ppi
export DISPLAY=:0
CUDA_VISIBLE_DEVICES=0 python eval_ppi.py \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=30 \
    framework.csv_logging=true \
    framework.tensorboard_logging=false \
    framework.eval_save_metrics=true \
    framework.eval_type=0 \
    framework.weight_name="train_ppi_ddp_pick_plate_ppi_20260405_baseline_seed0" \
    framework.ckpt_name="epoch=0200-val_loss=16.1768589_model" \
    framework.jump_step=1 \
    framework.weightsdir="$(pwd)/eval_weights/bimanual_pick_plate" \
    framework.logdir="$(pwd)/eval_logs" \
    framework.eval_envs=1 \
    framework.eval_processes=1 \
    rlbench.headless=false \
    rlbench.episode_length=400 \
    rlbench.task_name="pick_plate" \
    rlbench.tasks=[bimanual_pick_plate] \
    rlbench.demo_path="$(pwd)/data/eval_raw" \
    rlbench.cameras=["over_shoulder_left","over_shoulder_right","overhead","wrist_right","wrist_left","front"] \
    rlbench.cameras_pcd=["over_shoulder_left","over_shoulder_right","overhead","wrist_right","wrist_left","front"] \
    rlbench.camera_resolution=[256,256] \
    rlbench.include_lang_goal_in_obs=true \
    rlbench.gripper_mode='BimanualDiscrete' \
    rlbench.arm_action_mode='BimanualEndEffectorPoseViaPlanning' \
    rlbench.action_mode='BimanualMoveArmThenGripper' \
    rlbench.query_freq=10 \
    method.policy.horizon_keyframe=4 \
    method.policy.horizon_continuous=50 \
    method.policy.n_obs_steps=1 \
    method.policy.n_action_steps=54 \
    method.policy.bounding_box=[[-0.5,-0.55,0.77],[1.1,0.55,1.98]] \
    method.policy.fps_num=6144 \
    method.policy.prediction_type=keyframe_continuous \
    method.policy.what_condition=ppi \
    method.policy.predict_point_flow=true \
    method.policy.use_lang=true \
    method.policy.pointflow_num=200 \
    method.policy.text_prompt="a white plate" \
    method.policy.prompt_type="box" \
    method.policy.sample_type="rps" \
    method.policy.num_inference_steps=1000 \
    method.policy.sam_cameras=["front"] \
    method.policy.sam_checkpoint_path="$(pwd)/../pretrained_models/sam_vit_b_01ec64.pth" \
    method.policy.gdino_config_path="$(pwd)/../repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    method.policy.gdino_checkpoint_path="$(pwd)/../pretrained_models/groundingdino_swinb_cogcoor.pth" \
    method.policy.instruction_embeddings_path="$(pwd)/data/training_processed/instruction_embeddings.pkl" \
    cinematic_recorder.enabled=false \
    cinematic_recorder.save_path="$(pwd)/eval_videos/bimanual_pick_plate/train_ppi_ddp_pick_plate_ppi_20260405_baseline_seed0"
```

### 9.4 `evaluate_ppi_pick_fork.sh` 完整内容

文件：`occ_grasp_models/scripts/ppi/inference/evaluate_ppi_pick_fork.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate ppi
export DISPLAY=:0
CUDA_VISIBLE_DEVICES=0 python eval_ppi.py \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=30 \
    framework.csv_logging=true \
    framework.tensorboard_logging=false \
    framework.eval_save_metrics=true \
    framework.eval_type=0 \
    framework.weight_name="train_ppi_ddp_pick_fork_ppi_20260405_baseline_seed0" \
    framework.ckpt_name="latest_model" \
    framework.jump_step=1 \
    framework.weightsdir="$(pwd)/eval_weights/bimanual_pick_fork" \
    framework.logdir="$(pwd)/eval_logs" \
    framework.eval_envs=1 \
    framework.eval_processes=1 \
    rlbench.headless=false \
    rlbench.episode_length=400 \
    rlbench.task_name="pick_fork" \
    rlbench.tasks=[bimanual_pick_fork] \
    rlbench.demo_path="$(pwd)/data/eval_raw" \
    rlbench.cameras=["over_shoulder_left","over_shoulder_right","overhead","wrist_right","wrist_left","front"] \
    rlbench.cameras_pcd=["over_shoulder_left","over_shoulder_right","overhead","wrist_right","wrist_left","front"] \
    rlbench.camera_resolution=[256,256] \
    rlbench.include_lang_goal_in_obs=true \
    rlbench.gripper_mode='BimanualDiscrete' \
    rlbench.arm_action_mode='BimanualEndEffectorPoseViaPlanning' \
    rlbench.action_mode='BimanualMoveArmThenGripper' \
    rlbench.query_freq=10 \
    method.policy.horizon_keyframe=4 \
    method.policy.horizon_continuous=50 \
    method.policy.n_obs_steps=1 \
    method.policy.n_action_steps=54 \
    method.policy.bounding_box=[[-0.5,-0.55,0.77],[1.1,0.55,1.98]] \
    method.policy.fps_num=6144 \
    method.policy.prediction_type=keyframe_continuous \
    method.policy.what_condition=ppi \
    method.policy.predict_point_flow=true \
    method.policy.use_lang=true \
    method.policy.pointflow_num=200 \
    method.policy.text_prompt="a white fork" \
    method.policy.prompt_type="box" \
    method.policy.sample_type="rps" \
    method.policy.num_inference_steps=1000 \
    method.policy.sam_cameras=["front"] \
    method.policy.sam_checkpoint_path="$(pwd)/../pretrained_models/sam_vit_b_01ec64.pth" \
    method.policy.gdino_config_path="$(pwd)/../repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    method.policy.gdino_checkpoint_path="$(pwd)/../pretrained_models/groundingdino_swinb_cogcoor.pth" \
    method.policy.instruction_embeddings_path="$(pwd)/data/training_processed/instruction_embeddings.pkl" \
    cinematic_recorder.enabled=false \
    cinematic_recorder.save_path="$(pwd)/eval_videos/bimanual_pick_fork/train_ppi_ddp_pick_fork_ppi_20260405_baseline_seed0"
```

### 9.5 你接下来要执行的命令

先进入工作目录并准备环境：

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
source ~/.bashrc
conda activate ppi
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export DISPLAY=:0
```

然后按第 8 节手动建立：

1. `data/eval_raw/<TASK_DIR> -> /mnt/rlbench_data/<TASK_DIR>.test`
2. `eval_weights/<TASK_DIR>/0/<RUN_NAME> -> exp_logs/ckpt/<TASK_DIR>/<RUN_NAME>`

完成后直接运行：

```bash
bash scripts/ppi/inference/evaluate_ppi_edge_phone.sh
bash scripts/ppi/inference/evaluate_ppi_pivot_phone.sh
bash scripts/ppi/inference/evaluate_ppi_pick_plate.sh
bash scripts/ppi/inference/evaluate_ppi_pick_fork.sh
```

如果只想先做 smoke test，可先把脚本中的：

```bash
framework.eval_episodes=30
```

临时改成：

```bash
framework.eval_episodes=5
```

如果想评训练末尾权重，就只改脚本里的：

```bash
framework.ckpt_name="latest_model"
```

### 9.6 评测输出在哪里

默认评测结果会写到：

```text
eval_logs/<TASK_KEY>/PPI/seed0/
```

最重要的文件通常是：

```text
eval_logs/<TASK_KEY>/PPI/seed0/eval_data.csv
```

如果把脚本里的：

```bash
cinematic_recorder.enabled=false
```

改成：

```bash
cinematic_recorder.enabled=true
```

则视频会写到：

```text
eval_videos/<TASK_DIR>/<RUN_NAME>/videos/0/
```

这里的 `0` 是 `framework.eval_type=0` 对应的槽位目录名，不是 epoch。

### 9.7 评测部分的关键说明

1. `.test` 数据不需要提前生成 `point_cloud / dino / point_flow / norm_stats`；评测时在线重建视觉特征
2. 评测时真正必须存在的是：`.test` 数据、`instruction_embeddings.pkl`、预训练视觉权重、训练好的 `ckpt`
3. `rlbench.demo_path` 必须指向 `data/eval_raw` 根目录，而不是具体某个任务目录
4. `framework.eval_type` 在当前 PPI 评测实现里只接受整数；这里统一固定为 `0`
5. 当前 PPI 评测里，真正控制 `eval_data.csv` 是否落盘的关键开关是 `framework.eval_save_metrics=true`
6. `framework.weight_name`、`framework.ckpt_name` 和你手动建出来的 `eval_weights/<TASK_DIR>/0/<RUN_NAME>` 必须三者对齐
7. 当前脚本把 `rlbench.cameras`、`rlbench.cameras_pcd`、`camera_resolution`、`include_lang_goal_in_obs`、action mode 三件套，以及 `predict_point_flow/use_lang` 都显式写死，避免继续依赖模板默认值

---


## 10. 完整执行流程总结

### 10.1 最简执行顺序

1. 进入工作目录并激活环境
2. 手动创建 `ppi/config/task/<TASK_KEY>.yaml`
3. 建立数据目录软链接
4. 生成语言嵌入
5. 依次手动修改 `save_ptc.py / save_dino.py / save_point_flow.py / save_norm_stats_<TASK>.py`
6. 执行四步预处理（先 smoke test，再全量）
7. 手动创建并运行 `ddp_train_<TASK_KEY>.sh`
8. 建立评估权重软链接
9. 手动创建并运行 `evaluate_ppi_<TASK_KEY>.sh`

### 10.2 直接调用命令

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
conda activate ppi

python scripts/ppi/data_generation/save_ptc.py
python scripts/ppi/data_generation/save_dino.py
python scripts/ppi/data_generation/save_point_flow.py
python scripts/ppi/data_generation/save_norm_stats_edge_phone.py
bash scripts/ppi/training/ddp_train_edge_phone.sh
bash scripts/ppi/inference/evaluate_ppi_edge_phone.sh
```

---

## 11. 常见问题与解决方案

### 11.1 工作目录错误

**问题**：脚本找不到 `../pretrained_models` 或 `../repos`

**解决**：确保在 `/home/hdliu/occ_grasp_fall/occ_grasp_models` 下执行

### 11.2 点云采样失败

**问题**：`Not enough points inside the bounding box`

**解决**：调整 `BOUNDING_BOX` 的 z_min，从 `0.77` 降到 `0.75`

### 11.3 GroundingDINO 检测失败

**问题**：point flow 全为零

**解决**：
1. 调整 `PF_TEXT_PROMPT`（更具体的描述）
2. 更换 `SAM_CAMERAS`（选择目标更清晰的视角）

### 11.4 训练入口错误

**问题**：尝试运行 `train.py method=PPI`

**解决**：使用 `train_ppi_ddp.py`，不是 `train.py`

### 11.5 评估路径错误

**问题**：找不到 demo 数据

**解决**：
- `rlbench.demo_path` 应指向 `data/eval_raw`（根目录）
- 确保软链接正确：`data/eval_raw/${TASK_DIR}` → `/mnt/rlbench_data/${TASK_DIR}.test`

### 11.6 point_flow_type 不匹配

**问题**：训练时找不到 point flow 文件

**解决**：确保以下一致：
- 预处理生成：`world_ordered_rps200`
- task yaml：`point_flow_type: world_ordered_rps200`
- 训练命令：`task.dataset.point_flow_type=world_ordered_rps200`

---


## 12. 官方文档与当前仓库的对应关系

| 官方文档/脚本 | 原仓位置 | 当前仓位置 |
|--------------|---------|-----------|
| 环境安装 | `docs/INSTALLATION.md` | 仍然适用 |
| 数据预处理 | `docs/DATA_PREPROCESSION.md` | 仍然适用 |
| 训练细节 | `docs/TRAINING_DETAIL.md` | 仍然适用 |
| 推理评估 | `docs/INFERENCE.md` | 仍然适用 |
| 预处理脚本 | `scripts/data_generation/*` | `occ_grasp_models/scripts/ppi/data_generation/*` |
| 训练脚本模板 | `scripts/training/ddp_train_*.sh` | `occ_grasp_models/scripts/ppi/training/ddp_train_*.sh` |
| 训练入口 | `ddp_train.py` | `occ_grasp_models/train_ppi_ddp.py` |
| 评估脚本模板 | `scripts/inference/evaluate_ppi_*.sh` | `occ_grasp_models/scripts/ppi/inference/evaluate_ppi_*.sh` |
| 评估入口 | `inference-for-rlbench2/eval_ppi.py` | `occ_grasp_models/eval_ppi.py` |

---

## 13. 数据结构参考

详细的数据结构说明请参考：

- **原始数据**：`PPI_data_structure_detailed_explanation.md` 第3节
- **预处理数据**：`PPI_data_structure_detailed_explanation.md` 第4节
- **ReplayBuffer**：`PPI_data_structure_detailed_explanation.md` 第5节
- **评估流程**：`PPI_EVAL_PIPELINE_ANALYSIS.md`

### 关键数据维度

- **state/action**：(16,) = 左臂(8) + 右臂(8)
  - 每臂：gripper_pose(7) + gripper_open(1)
- **point_cloud**：(6144, 6) = XYZ(3) + RGB(3)
- **dino_feature**：(6144, 384)
- **point_flow**：(200, 3)
- **lang**：(1024,)

---

## 14. 与原版指南的主要改进

本指南相比 `PPI_OCC_GRASP_END_TO_END_GUIDE.md` 的改进：

1. **整合官方文档**：直接引用官方 `docs/` 中的具体脚本和命令
2. **明确脚本位置**：指出官方四个预处理脚本在当前仓的对应位置
3. **保留官方流程**：遵循官方 `DATA_PREPROCESSION.md` 的四步顺序
4. **补充官方细节**：
   - 训练超参数参考 `TRAINING_DETAIL.md`
   - 评估配置参考 `INFERENCE.md`
   - 权重目录结构遵循官方约定
5. **改为手动改文件方式**：明确哪些配置落在 yaml、哪些落在预处理脚本、哪些落在训练/评估脚本

---

## 15. 参考资源

### 官方资源

- **GitHub**：https://github.com/OpenRobotLab/PPI
- **数据集**：https://huggingface.co/datasets/yuyinyang3y/Open-PPI
- **权重**：https://huggingface.co/datasets/yuyinyang3y/Open-PPI/tree/main/ckpt
- **RLBench2**：https://bimanual.github.io/

### 本地文档

- `/home/hdliu/claude_repo/PPI/docs/` - 官方完整文档
- `/home/hdliu/occ_grasp_fall/PPI_data_structure_detailed_explanation.md` - 数据结构详解
- `/home/hdliu/occ_grasp_fall/PPI_EVAL_PIPELINE_ANALYSIS.md` - 评估流程分析
- `/home/hdliu/occ_grasp_fall/ppi_method_analysis.md` - 方法分析
- `/home/hdliu/claude_repo/PPI/PPI_OCC_GRASP_MIGRATION_MANIFEST.md` - 迁移清单

---

## 附录：四个任务的手工修改参数速查表

| 任务 | `TASK_KEY` | `TASK_DIR` | `PF_TEXT_PROMPT` | `SAM_CAMERAS` | `PF_PROMPT_TYPE` | `BOUNDING_BOX` | `EPISODE_LENGTH` | `QUERY_FREQ` |
|------|------------|------------|------------------|---------------|------------------|----------------|------------------|--------------|
| edge phone | `edge_phone` | `bimanual_edge_phone` | `a black phone` | `["front"]` | `box` | `[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]` | `400` | `10` |
| pivot phone | `pivot_phone` | `bimanual_pivot_phone` | `a black phone` | `["front"]` | `box` | `[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]` | `400` | `10` |
| pick plate | `pick_plate` | `bimanual_pick_plate` | `a white plate` | `["front"]` | `box` | `[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]` | `400` | `10` |
| pick fork | `pick_fork` | `bimanual_pick_fork` | `a white fork` | `["front"]` | `box` | `[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]` | `400` | `10` |

---

**文档完成时间**：2026-03-29

**适用版本**：occ_grasp_fall 框架 + PPI 仿真分支

## 16. 关键脚本修改位置速查

### 16.1 预处理脚本修改要点

**save_ptc.py**：
- 修改位置：`if __name__ == "__main__":` 块
- 必改项：`task`、`data_path`、`target_path`、`bounding_box`、`EP_START`、`EP_END`

**save_dino.py**：
- 修改位置：`if __name__ == "__main__":` 块
- 必改项：`device`、`task`、`pcd_path`、`data_path`、`target_path`、`EP_START`、`EP_END`

**save_point_flow.py**：
- 修改位置：`if __name__ == "__main__":` 块
- 必改项：`task`、`text_prompt`、`cameras`、`prompt_type`、`process_episodes(start, end, ...)`
- 注意：官方脚本中有多个任务的注释示例，需要取消注释或新增你的任务配置

**save_norm_stats_<task>.py**（建议新建）：
- 修改位置：脚本顶部常量定义
- 必改项：`TASK`、`EP_START`、`EP_END`、`PCD_TYPE`、`POINT_FLOW_TYPE`

### 16.2 训练脚本修改要点

**ddp_train_<task>.sh**（建议从模板复制）：
- 必改项：
  - `task='<TASK_KEY>'`
  - `wandb_name='ppi_<TASK_KEY>'`
  - `task.dataset.stats_filepath='...'`（确保路径与 norm_stats 输出一致）
  - `ngpus` 和 `CUDA_VISIBLE_DEVICES`（根据可用 GPU 数量）
  - `dataloader.batch_size` 和 `val_dataloader.batch_size`（根据 GPU 内存）

### 16.3 评估脚本修改要点

**evaluate_ppi_<task>.sh**（建议从模板复制）：
- 必改项：
  - `framework.weight_name="<RUN_NAME>"`（从训练输出目录获取）
  - `framework.weightsdir="$(pwd)/eval_weights/<TASK_DIR>"`
  - `rlbench.episode_length=<EPISODE_LENGTH>`
  - `rlbench.task_name="<TASK_KEY>"`
  - `rlbench.tasks=[<TASK_DIR>]`
  - `rlbench.query_freq=<QUERY_FREQ>`
  - `method.policy.text_prompt="<PF_TEXT_PROMPT>"`
  - `method.policy.sam_cameras=<SAM_CAMERAS>`
  - `method.policy.bounding_box=<BOUNDING_BOX>`

---
