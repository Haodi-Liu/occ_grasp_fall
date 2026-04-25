# occ_grasp_fall 中 PPI 端到端操作指南

本文针对当前 `occ_grasp_fall` 工作区里的 PPI 代码路径，而不是原始 PPI 仓库的旧目录结构。目标是把下面这条链路一次讲清楚：

`/mnt/rlbench_data` 原始 RLBench 数据 -> `.train` 预处理 -> PPI 训练 -> `.test` 上的 RLBench 仿真评估

## 0. 先说结论

这套代码当前有 5 个非常关键的约束：

1. 训练入口是 `occ_grasp_models/train_ppi_ddp.py`，不是别的训练脚本。
2. PPI 训练真正需要的是两部分：
   - 原始 demo：`data/training_raw/<task>/all_variations/episodes`
   - 预处理产物：`data/training_processed/{point_cloud,dino_feature,point_flow,norm_stats}`
3. `.test` 上的 PPI 评估不会去读 `.test` 的离线 point cloud / dino / point flow；评估阶段会在线重建这些信息，所以 `.test` 只需要原始 demo、语言嵌入、模型权重和基础预训练权重。
4. 当前训练代码不会直接使用 `/mnt/rlbench_data/*.val`；它会在 `.train` 内部按 `val_ratio` 再切一份验证集。
5. 当前 `/mnt/rlbench_data` 只有 `.train/.val/.test` 后缀目录，而 RLBench 评估代码要求 `demo_path/<task_name>/all_variations/episodes` 这种无后缀目录，所以训练和评估都必须先做软链接适配。

## 1. 当前数据现状

当前 `/mnt/rlbench_data` 里可直接看到 4 个任务，每个任务的 episode 数量是：

| 任务目录 | episode 数 |
|---|---:|
| `bimanual_edge_phone.train` | 150 |
| `bimanual_edge_phone.val` | 50 |
| `bimanual_edge_phone.test` | 30 |
| `bimanual_pivot_phone.train` | 150 |
| `bimanual_pivot_phone.val` | 50 |
| `bimanual_pivot_phone.test` | 30 |
| `bimanual_pick_plate.train` | 150 |
| `bimanual_pick_plate.val` | 50 |
| `bimanual_pick_plate.test` | 30 |
| `bimanual_pick_fork.train` | 150 |
| `bimanual_pick_fork.val` | 50 |
| `bimanual_pick_fork.test` | 30 |

下面所有命令默认基于这套现状。

## 2. 一次性环境准备

### 2.1 进入正确目录

很多 PPI 预处理脚本内部写死了相对路径 `../pretrained_models` 和 `../repos`，所以最安全的做法是始终从 `occ_grasp_models` 目录运行：

```bash
source ~/.bashrc
conda activate ppi
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
```

### 2.2 CoppeliaSim / PyRep 环境变量

评估阶段要用 RLBench + PyRep + CoppeliaSim。若你还没有在 `~/.bashrc` 里配置，至少要有：

```bash
export COPPELIASIM_ROOT=/path/to/your/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

配置后重新执行：

```bash
source ~/.bashrc
conda activate ppi
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
```

### 2.3 预训练权重检查

下面 3 个路径是这套代码实际会访问的路径：

```bash
ls ../pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth
ls ../pretrained_models/sam_vit_b_01ec64.pth
ls ../pretrained_models/groundingdino_swinb_cogcoor.pth
```

当前工作区里我确认已经存在的是：

```text
../pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth
```

当前工作区里缺的是：

```text
../pretrained_models/sam_vit_b_01ec64.pth
../pretrained_models/groundingdino_swinb_cogcoor.pth
```

如果后两者不存在，下面两步会失败：

- `save_point_flow.py`
- `eval_ppi.py`

直接下载到代码实际会读取的位置：

```bash
cd /home/hdliu/occ_grasp_fall
mkdir -p pretrained_models/hub/checkpoints

# 当前工作区里这个文件已经存在；只有缺失时才需要重新下载
wget -c https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
  -O pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth

wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -O pretrained_models/sam_vit_b_01ec64.pth

wget -c https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth \
  -O pretrained_models/groundingdino_swinb_cogcoor.pth
```

下载完成后检查：

```bash
ls -lh \
  /home/hdliu/occ_grasp_fall/pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth \
  /home/hdliu/occ_grasp_fall/pretrained_models/sam_vit_b_01ec64.pth \
  /home/hdliu/occ_grasp_fall/pretrained_models/groundingdino_swinb_cogcoor.pth
```

如果 GitHub release 下载 GroundingDINO 很慢，也可以用官方 README 里的 Hugging Face 备用链接：

```bash
wget -c https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth \
  -O /home/hdliu/occ_grasp_fall/pretrained_models/groundingdino_swinb_cogcoor.pth
```

### 2.4 无桌面服务器时的显示环境

如果你在没有物理显示器的服务器上评估 RLBench，建议先起一个 `Xvfb`：

```bash
Xvfb :99 -screen 0 1024x768x16 >/tmp/xvfb_ppi.log 2>&1 &
export DISPLAY=:99
```

## 3. 先选一个任务

这一节里的 `TASK_KEY`、`TASK_DIR`、`PF_TEXT_PROMPT` 等，并不是原始 PPI 代码里逐字同名的一组环境变量；它们是为了把当前工作区里分散在 task yaml、预处理脚本、训练脚本和评估脚本里的关键参数统一起来，专门整理出的“操作层包装变量”。

如果你脑中参考的是 `pick_laptop` 那个打包示例，可以把它等价理解成当前工作区里的这组结构：

- `data/training_raw/bimanual_pick_laptop/...`
- `data/training_processed/point_cloud/bimanual_pick_laptop/...`
- `data/training_processed/dino_feature/bimanual_pick_laptop/...`
- `data/training_processed/point_flow/bimanual_pick_laptop/...`
- `data/training_processed/norm_stats/...`

为避免路径混淆，下面统一只用相对于仓库根 `/home/hdliu/occ_grasp_fall` 的路径说明。`pick_laptop` 在当前代码里的对应链条是：

- `occ_grasp_models/ppi/config/task/laptop.yaml`
- `occ_grasp_models/ppi/config/ppi.yaml`
- `occ_grasp_models/scripts/ppi/training/ddp_train_laptop.sh`
- `occ_grasp_models/scripts/ppi/inference/evaluate_ppi_laptop.sh`
- `occ_grasp_models/scripts/ppi/data_generation/save_ptc.py`
- `occ_grasp_models/scripts/ppi/data_generation/save_point_flow.py`
- `occ_grasp_models/agents/ppi/launch_utils.py`
- `occ_grasp_models/agents/ppi/ppi_agent.py`
- `occ_grasp_models/eval_ppi.py`
- `occ_grasp_models/helpers/custom_rlbench_env.py`
- `occ_grasp_models/helpers/observation_utils.py`

有两个容易混淆的事实要先记住：

- `TASK_KEY` / `TASK_DIR` 是本文为了操作方便而起的统一变量名。原始 PPI 里真正对应的是 `task=...`、`task.name`、`dataset_task_name`、`rlbench.task_name`、`rlbench.tasks` 这些分散字段。
- 真正生效的配置经常是“shell 脚本 override 后的结果”，不能只看 task yaml 默认值。比如 `occ_grasp_models/ppi/config/task/laptop.yaml` 默认写的是 `prediction_type: continuous` 和 `point_flow_type: rps200`，但 `occ_grasp_models/scripts/ppi/training/ddp_train_laptop.sh` 实际覆盖成了 `keyframe_continuous` 和 `world_ordered_rps200`。后面为手机任务写 task yaml 和训练命令时，要以这种“最终生效组合”为准。

下面先用 `pick_laptop` 这个现成例子解释这组变量在原始 PPI 设计中的含义，再给出四个新任务的起始值。

### 3.1 这组变量在原始 PPI 里分别是什么

- `TASK_KEY`
  原始 `pick_laptop` 的值是 `laptop`。它是 Hydra 里的短任务名，用来选 `occ_grasp_models/ppi/config/task/laptop.yaml`，并参与组成实验名与日志目录。对应位置包括：
  `occ_grasp_models/ppi/config/task/laptop.yaml` 里的 `name` / `task_name`，
  `occ_grasp_models/scripts/ppi/training/ddp_train_laptop.sh` 里的 `task='laptop'`，
  `occ_grasp_models/scripts/ppi/inference/evaluate_ppi_laptop.sh` 里的 `rlbench.task_name="laptop"`。

- `TASK_DIR`
  原始 `pick_laptop` 的值是 `bimanual_pick_laptop`。它是真正的 RLBench 数据目录名，也是离线预处理目录名。对应位置包括：
  `occ_grasp_models/ppi/config/task/laptop.yaml` 里的 `dataset_task_name: bimanual_pick_laptop`，
  `occ_grasp_models/scripts/ppi/inference/evaluate_ppi_laptop.sh` 里的 `rlbench.tasks=[bimanual_pick_laptop]`。

- `EP_START` / `EP_END`
  这两个控制 `.train` 里哪些 episode 会被离线预处理和训练数据构造读取。原始 `pick_laptop` 的 task yaml 用的是 `start: 0`、`end: 99`。
  它们会被 `occ_grasp_models/ppi/config/task/laptop.yaml` 的 `dataset.start/end` 读取，再继续传给：
  `occ_grasp_models/scripts/ppi/data_generation/save_ptc.py`、
  `occ_grasp_models/scripts/ppi/data_generation/save_point_flow.py`、
  `occ_grasp_models/ppi/dataset/rlbench2_dataset.py`、
  `occ_grasp_models/ppi/common/get_data_keyframe_continuous.py`。

- `TEST_EPISODES`
  这不是训练参数，而是评估时要跑多少个 demo seed。原始 `pick_laptop` 的评估脚本写的是 `framework.eval_episodes=100`。
  它会经由 `occ_grasp_models/eval_ppi.py` 传给 YARR runner，runner 再按 `eval_demo_seed = ep + eval_from_eps_number` 逐个 episode 跑。

- `BOUNDING_BOX`
  这是世界坐标系里的 3D 工作空间裁剪框，不是 SAM 的 2D 检测框。原始 `pick_laptop` 用的是：
  `[[ -0.5, -0.55, 0.77 ], [ 1.1, 0.55, 1.98 ]]`。
  它在离线点云预处理和在线评估里都生效：
  `occ_grasp_models/scripts/ppi/data_generation/save_ptc.py` 用它裁点云并采样，
  `occ_grasp_models/agents/ppi/ppi_agent.py` 在在线 `preprocess_pcd()` 里也会用同一个 box 过滤无关点。

- `PF_TEXT_PROMPT`
  这是给 GroundingDINO 的文本提示，用来先检测目标物体。原始 `pick_laptop` 用的是 `a black rectangle laptop`。
  它在离线和在线两边都生效：
  `occ_grasp_models/scripts/ppi/data_generation/save_point_flow.py` 会用它先找目标，
  `occ_grasp_models/agents/ppi/ppi_agent.py` 的 `get_point_from_mask()` 在线评估时也会用同样的 prompt。

- `SAM_CAMERAS`
  这是用哪些相机视角去生成初始 point-flow 种子点。原始 `pick_laptop` 用的是 `["front"]`。
  离线 point flow 生成会按这组 camera 做目标检测和 mask 采样；在线评估时，`occ_grasp_models/agents/ppi/ppi_agent.py` 的 `get_initial_pointflow()` 也会按这组 camera 逐个取图并重建初始点集。

- `PF_PROMPT_TYPE`
  这是 SAM 的提示方式，不是 `BOUNDING_BOX`。原始 `pick_laptop` 用的是 `point`，即先用 GroundingDINO 找框，再取框中心点去提示 SAM。
  在 `occ_grasp_models/scripts/ppi/data_generation/save_point_flow.py` 和 `occ_grasp_models/agents/ppi/ppi_agent.py` 里都只有两种模式：
  `point` 表示给 SAM 一个中心点；
  `box` 表示直接把检测框传给 SAM。
  `box` 的具体链路是：先用 GroundingDINO 在 `SAM_CAMERAS` 指定视角的首帧 RGB 上预测 2D 检测框；代码当前默认取第一条通过阈值的框，把它从 DINO 的归一化 `cxcywh` 还原成图像坐标 `xyxy`，再经 `apply_boxes_torch()` 映射到 SAM 输入坐标系，最后调用 `predict_torch(..., boxes=...)` 生成整块目标 mask。离线预处理会把 mask 内像素用深度反投影成首帧 3D 点，再通过 `object_6d_pose` 变换传播到整段 episode；在线评估则在当前观测上重复这一步，得到初始 point-flow 种子点。相对 `point` 模式，`box` 不依赖单个中心点击中目标内部，对细长、薄片、边缘易遮挡的物体通常更稳；但它仍然依赖 GroundingDINO 首先给出正确检测框。

- `EPISODE_LENGTH`
  这是 RLBench 评估时一个 episode 允许执行的最大步数。原始 `pick_laptop` 用的是 `300`。
  它会通过 `occ_grasp_models/eval_ppi.py` 传进环境 runner，`occ_grasp_models/helpers/custom_rlbench_env.py` 会在步数达到上限时截断 episode。
  此外，`occ_grasp_models/helpers/observation_utils.py` 还会用它把 timestep 归一化后拼进 `low_dim_state`，所以它不仅是“超时上限”，也影响 time-in-state 特征。

- `QUERY_FREQ`
  这是每隔多少个环境 step 才重新跑一次 PPI 网络。原始 `pick_laptop` 用的是 `15`。
  `occ_grasp_models/agents/ppi/ppi_agent.py` 里只有在 `self._timestep % query_freq == 0` 或 `self._timestep == 1` 时才重新预测；其余 step 会复用上一段 action chunk。
  所以它本质上是“推理频率 / 动作重用频率”的折中参数。

### 3.2 从 `pick_laptop` 迁移到这 4 个新任务时，哪些值是直接继承，哪些值是推荐起点

下面这些是直接从当前数据现状或现有 PPI 实现继承出来的，不是拍脑袋设的：

- `TASK_KEY`：你为新任务定义的 Hydra 短名，例如 `pivot_phone`
- `TASK_DIR`：真实数据目录名，例如 `bimanual_pivot_phone`
- `EP_START=0`
- `EP_END=149`
- `TEST_EPISODES=30`
- `BOUNDING_BOX='[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]'`

下面这些对 `edge_phone / pivot_phone / pick_plate / pick_fork` 在仓库里没有现成官方 PPI 配置，因此是“推荐起始值”，不是唯一正确答案：

- `PF_TEXT_PROMPT`
- `SAM_CAMERAS`
- `PF_PROMPT_TYPE`
- `EPISODE_LENGTH`
- `QUERY_FREQ`

其中有两个迁移理由尤其重要：

- `PF_PROMPT_TYPE`
  原始 `pick_laptop` 用的是 `point`，但手机和餐具类目标更小、更细，框中心点更容易偏到背景区域，所以第一轮更建议从 `box` 起步。

- `EPISODE_LENGTH` / `QUERY_FREQ`
  原始 `pick_laptop` 是 `300 / 15`。对 `pivot_phone` 这类需要更长调整过程的任务，先用 `400 / 10` 更保守：给 agent 更多总步数，同时更频繁地重算动作。

### 3.3 四个新任务的推荐起始值

说明：

- `bounding_box` 建议统一先用 `[[ -0.5, -0.55, 0.77 ], [ 1.1, 0.55, 1.98 ]]`。如果后面点云采样报 `Not enough points inside the bounding box`，优先把 `z_min` 从 `0.77` 降到 `0.75`。
- 下表是“推荐起始值”，不是代码仓库里已经固化验证过的唯一正确参数；尤其 `PF_TEXT_PROMPT`、`SAM_CAMERAS`、`QUERY_FREQ` 需要你根据检测效果和评估稳定性继续调。
- 基于当前这轮数据样本复核，第一轮统一推荐把 `SAM_CAMERAS` 设为 `["front"]`，`EPISODE_LENGTH` 设为 `400`。

| 任务 | `TASK_KEY` | `TASK_DIR` | `PF_TEXT_PROMPT` | `SAM_CAMERAS` | `PF_PROMPT_TYPE` | `EPISODE_LENGTH` | `QUERY_FREQ` |
|---|---|---|---|---|---|---:|---:|
| edge phone | `edge_phone` | `bimanual_edge_phone` | `a phone` | `["front"]` | `box` | 400 | 10 |
| pivot phone | `pivot_phone` | `bimanual_pivot_phone` | `a phone` | `["front"]` | `box` | 400 | 10 |
| pick plate | `pick_plate` | `bimanual_pick_plate` | `a plate` | `["front"]` | `box` | 400 | 10 |
| pick fork | `pick_fork` | `bimanual_pick_fork` | `a fork` | `["front"]` | `box` | 400 | 10 |

下面给一个 `pivot_phone` 示例。要换任务，只改这些变量即可：

```bash
export TASK_KEY=pivot_phone
export TASK_DIR=bimanual_pivot_phone
export EP_START=0
export EP_END=149
export TEST_EPISODES=30
export BOUNDING_BOX='[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]'
export PF_TEXT_PROMPT='a phone'
export SAM_CAMERAS='["front"]'
export PF_PROMPT_TYPE=box
export EPISODE_LENGTH=400
export QUERY_FREQ=10
```

## 4. 把 `.train` / `.test` 数据映射成代码能识别的目录

当前代码最稳妥的做法是建立两套软链接根目录：

- `data/training_raw/<task>` 指向 `/mnt/rlbench_data/<task>.train`
- `data/eval_raw/<task>` 指向 `/mnt/rlbench_data/<task>.test`

命令如下：

```bash
mkdir -p data/training_raw data/eval_raw
mkdir -p data/training_processed/point_cloud
mkdir -p data/training_processed/dino_feature
mkdir -p data/training_processed/point_flow
mkdir -p data/training_processed/norm_stats

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

检查一下：

```bash
readlink -f "data/training_raw/${TASK_DIR}"
readlink -f "data/eval_raw/${TASK_DIR}"
```

这一步必须做。否则：

- 训练阶段会找不到 `data/training_raw/...`
- 评估阶段会因为 `demo_path` 下没有无后缀任务目录而直接失败
- 如果你启用了 scheme 分层评估，它也会因为找不到 `scheme_info_*.pkl` 而失效

## 5. 生成共享的 `instruction_embeddings.pkl`

PPI 数据读取和评估都要用语言嵌入字典。这个文件建议做成全任务共享的一个文件：

```text
data/training_processed/instruction_embeddings.pkl
```

下面的脚本会扫描 `/mnt/rlbench_data/*.train`、`*.val`、`*.test` 中所有 `variation_descriptions.pkl`，去重后生成 CLIP RN50 文本特征。

注意：

- 第一次运行时，如果本机还没有 `RN50.pt`，`helpers/clip/core/clip.py` 会自动下载到 `~/.cache/clip/`。
- 如果机器离线，请提前把 CLIP RN50 权重放到 `~/.cache/clip/RN50.pt`。

```bash
python - <<'PY'
import glob
import os
import pickle
import torch

from helpers.clip.core.clip import build_model, load_clip, tokenize

data_root = "/mnt/rlbench_data"
instructions = set()

for pattern in ("*.train", "*.val", "*.test"):
    for task_dir in sorted(glob.glob(os.path.join(data_root, pattern))):
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

## 6. 为新任务创建 Hydra task 配置

仓库自带的 `ppi/config/task/*.yaml` 只覆盖旧任务，没有 `edge_phone / pivot_phone / pick_plate / pick_fork`。因此必须为当前任务补一份 task yaml。

下面的模板是当前代码下较稳妥的起点：

```bash
cat <<'EOF' > "ppi/config/task/${TASK_KEY}.yaml"
name: TASK_KEY_PLACEHOLDER

task_name: TASK_KEY_PLACEHOLDER
dataset_task_name: TASK_DIR_PLACEHOLDER

dataset:
  _target_: ppi.dataset.rlbench2_dataset.RLBench2Dataset
  data_path: data/training_raw/${task.dataset_task_name}/all_variations/episodes
  pcd_path: data/training_processed/point_cloud/${task.dataset_task_name}/all_variations/episodes
  dino_path: data/training_processed/dino_feature/${task.dataset_task_name}/all_variations/episodes
  lang_emb_path: data/training_processed/instruction_embeddings.pkl
  stats_filepath: data/training_processed/norm_stats/norm_stats_${task.dataset_task_name}_${task.dataset.pcd_type}_${task.dataset.prediction_type}_${task.dataset.point_flow_type}.pth
  point_flow_path: data/training_processed/point_flow/${task.dataset_task_name}/all_variations/episodes
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
  add_openess_sampling: false
EOF
```

把占位符替换掉：

```bash
sed -i "s/TASK_KEY_PLACEHOLDER/${TASK_KEY}/g" "ppi/config/task/${TASK_KEY}.yaml"
sed -i "s/TASK_DIR_PLACEHOLDER/${TASK_DIR}/g" "ppi/config/task/${TASK_KEY}.yaml"
```

检查：

```bash
sed -n '1,200p' "ppi/config/task/${TASK_KEY}.yaml"
```

说明：

- `prediction_type` 这里建议直接用 `keyframe_continuous`，因为当前训练 shell 脚本和策略代码都按这一路径在写。
- `point_flow_type` 必须和离线 point flow 目录保持一致，这里统一用 `world_ordered_rps200`。
- `val_ratio=0.2` 表示从 `.train` 的 150 个 episode 中内部切 20% 做验证。当前 `.val` 分片不会被训练代码自动使用。
- `max_train_episodes=150` 这里只是“训练 episode 上限”，等价于不额外下采样；在 `val_ratio=0.2` 时，实际参与训练的 episode 数大约还是 120 左右。
- `kp_num` 和 `add_openess_sampling` 对这 4 个任务没有现成官方配置，上面是保守起点。若后面发现抓手开合转折学得差，再考虑把 `add_openess_sampling` 改成 `true` 重新训练。

## 7. 在 `.train` 上做离线预处理

### 7.1 离线 point cloud

注意：原始数据里即使已经有 `*_point_cloud/` 目录，当前 PPI 脚本也不会直接用它；它会根据 depth、intrinsics、extrinsics 重新构建点云。

```bash
export CUDA_VISIBLE_DEVICES=0

python - <<'PY'
import ast
import os
import numpy as np

from scripts.ppi.data_generation.save_ptc import pcd

task = os.environ["TASK_DIR"]
ep_start = int(os.environ["EP_START"])
ep_end = int(os.environ["EP_END"])
bounding_box = np.array(ast.literal_eval(os.environ["BOUNDING_BOX"]), dtype=float)
cameras = [
    "over_shoulder_left",
    "over_shoulder_right",
    "overhead",
    "wrist_right",
    "wrist_left",
    "front",
]

runner = pcd(
    data_path=f"data/training_raw/{task}/all_variations/episodes",
    target_path=f"data/training_processed/point_cloud/{task}/all_variations/episodes",
)
runner.process_episodes(ep_start, ep_end, cameras, "rgb_pcd_rps6144", bounding_box)
PY
```

检查输出数量：

```bash
find "data/training_processed/point_cloud/${TASK_DIR}/all_variations/episodes" -name 'step*.npy' | wc -l
```

### 7.2 离线 DINO feature

```bash
export CUDA_VISIBLE_DEVICES=0
export DINO_DEVICE=cuda:0

python - <<'PY'
import os

from scripts.ppi.data_generation.save_dino import Fusion, process_episodes

task = os.environ["TASK_DIR"]
ep_start = int(os.environ["EP_START"])
ep_end = int(os.environ["EP_END"])
device = os.environ.get("DINO_DEVICE", "cuda:0")

fusion = Fusion(num_cam=6, feat_backbone="dinov2", device=device)
process_episodes(
    ep_start,
    ep_end,
    task,
    fusion,
    f"data/training_processed/point_cloud/{task}/all_variations/episodes",
    f"data/training_raw/{task}/all_variations/episodes",
    f"data/training_processed/dino_feature/{task}/all_variations/episodes",
    "rgb_pcd_rps6144",
    device,
)
PY
```

检查：

```bash
find "data/training_processed/dino_feature/${TASK_DIR}/all_variations/episodes" -name 'step*.npy' | wc -l
```

### 7.3 离线 point flow

`save_point_flow.py` 依赖 GroundingDINO + SAM，所以这一步对 `text_prompt` 和 `sam_cameras` 很敏感。建议先只抽一两个 episode 试跑一下，确认目标框对了，再全量跑。

全量命令如下：

```bash
export CUDA_VISIBLE_DEVICES=0
export PF_DEVICE_NUM=0

python - <<'PY'
import ast
import os

from scripts.ppi.data_generation.save_point_flow import GetPointFlow

task = os.environ["TASK_DIR"]
ep_start = int(os.environ["EP_START"])
ep_end = int(os.environ["EP_END"])
text_prompt = os.environ["PF_TEXT_PROMPT"]
prompt_type = os.environ["PF_PROMPT_TYPE"]
cameras = ast.literal_eval(os.environ["SAM_CAMERAS"])
device_num = int(os.environ.get("PF_DEVICE_NUM", "0"))

runner = GetPointFlow(
    data_path=f"data/training_raw/{task}/all_variations/episodes",
    target_path=f"data/training_processed/point_flow/{task}/all_variations/episodes",
    task=task,
    device_num=device_num,
    text_prompt=text_prompt,
    prompt_type=prompt_type,
)
runner.process_episodes(ep_start, ep_end, cameras, "world_ordered_rps200")
PY
```

检查：

```bash
find "data/training_processed/point_flow/${TASK_DIR}/all_variations/episodes" -name 'step*.npy' | wc -l
```

### 7.4 生成 `norm_stats`

仓库自带的 `scripts/ppi/data_generation/save_norm_stats.py` 有 3 个问题：

1. 任务名是硬编码的旧任务。
2. episode 范围硬编码成 `0..99`。
3. 输出文件名默认带 `_new.pth`，但训练脚本和 task yaml 期待的是不带 `_new` 的标准文件名。

因此更稳妥的做法是直接用下面这个流式脚本，按当前任务在线统计并保存成训练代码真正会读取的文件名。

注意：如果你把 task yaml 里的 `kp_num` 改成了别的值，这里脚本里的 `kp_num = 10` 也要同步改掉。

```bash
python - <<'PY'
import os
import numpy as np
import torch

from ppi.common.get_data_keyframe_continuous import GetDataKeyframeContinuous
from ppi.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class RunningStats:
    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, array):
        array = np.asarray(array, dtype=np.float32)
        array = array.reshape(-1, array.shape[-1])
        self.count += array.shape[0]
        self.sum += array.sum(axis=0, dtype=np.float64)
        self.sumsq += np.square(array, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.min = np.minimum(self.min, array.min(axis=0))
        self.max = np.maximum(self.max, array.max(axis=0))

    def finalize(self):
        mean = self.sum / self.count
        var = self.sumsq / self.count - np.square(mean)
        var = np.maximum(var, 0.0)
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


task = os.environ["TASK_DIR"]
ep_start = int(os.environ["EP_START"])
ep_end = int(os.environ["EP_END"])
kp_num = 10
pcd_type = "rgb_pcd_rps6144"
point_flow_type = "world_ordered_rps200"

data_path = f"data/training_raw/{task}/all_variations/episodes"
pcd_root = f"data/training_processed/point_cloud/{task}/all_variations/episodes"
dino_root = f"data/training_processed/dino_feature/{task}/all_variations/episodes"
point_flow_root = f"data/training_processed/point_flow/{task}/all_variations/episodes"
lang_emb_path = "data/training_processed/instruction_embeddings.pkl"

gd = GetDataKeyframeContinuous(data_path=data_path, lang_emb_path=lang_emb_path)
root = gd.process_episodes(ep_start, ep_end, [], kp_num)
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
    path = os.path.join(pcd_root, f"episode{episode}/{pcd_type}/step{step:03d}.npy")
    stats["point_cloud"].update(np.load(path))

for episode, step in data["dino_feature"]:
    path = os.path.join(dino_root, f"episode{episode}/{pcd_type}/step{step:03d}.npy")
    stats["dino_feature"].update(np.load(path))

for episode, step in data["point_flow"]:
    path = os.path.join(point_flow_root, f"episode{episode}/{point_flow_type}/step{step:03d}.npy")
    stats["point_flow"].update(np.load(path))

for episode, step in data["initial_point_flow"]:
    path = os.path.join(point_flow_root, f"episode{episode}/{point_flow_type}/step{step:03d}.npy")
    stats["initial_point_flow"].update(np.load(path))

normalizer = LinearNormalizer()
for key, tracker in stats.items():
    normalizer[key] = make_single_field(tracker.finalize())

out_path = (
    f"data/training_processed/norm_stats/"
    f"norm_stats_{task}_{pcd_type}_keyframe_continuous_{point_flow_type}.pth"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(normalizer.state_dict(), out_path)
print(f"saved to {out_path}")
PY
```

输出文件应该是：

```text
data/training_processed/norm_stats/norm_stats_${TASK_DIR}_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth
```

## 8. 在 `.train` 上训练 PPI

下面给的是单机单卡基线命令。若你有多卡，只需要把 `NGPUS` 和 `CUDA_VISIBLE_DEVICES` 改掉，并把 batch size 相应放大。

```bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
export ADDITION_INFO=$(date +%Y%m%d)_baseline
export WANDB__SERVICE_WAIT=600
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

torchrun --nnodes 1 --nproc_per_node ${NGPUS} --master_port 10004 train_ppi_ddp.py \
  task=${TASK_KEY} \
  name=train_ppi_ddp \
  addition_info=${ADDITION_INFO} \
  wandb_name=ppi_${TASK_KEY} \
  logging.mode=offline \
  n_obs_steps=1 \
  n_action_steps=54 \
  policy.use_lang=true \
  policy.what_condition=ppi \
  policy.predict_point_flow=true \
  task.dataset.pcd_fps=6144 \
  task.dataset.pcd_type=rgb_pcd_rps6144 \
  task.dataset.point_flow_type=world_ordered_rps200 \
  task.dataset.kp_num=10 \
  task.dataset.prediction_type=keyframe_continuous \
  task.dataset.stats_filepath=data/training_processed/norm_stats/norm_stats_${TASK_DIR}_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth \
  horizon_keyframe=4 \
  horizon_continuous=50 \
  dataloader.batch_size=16 \
  val_dataloader.batch_size=16 \
  training.num_epochs=500
```

说明：

- 训练输出目录是：

```text
exp_logs/ckpt/${TASK_DIR}/train_ppi_ddp_${TASK_KEY}_ppi_${ADDITION_INFO}_seed0
```

- 常见 checkpoint 名称有：
  - `checkpoints/latest.pth.tar`
  - `checkpoints/latest_model.pth.tar`
  - `checkpoints/epoch50_model.pth.tar` 这种周期性模型文件
- `dataloader.batch_size` 和 `val_dataloader.batch_size` 是全局 batch size；如果你把 `NGPUS` 改成多卡，请确保它们都能被 `NGPUS` 整除。

如果你只是想先验证流程是否通，再把 `training.num_epochs` 暂时改小到 `50` 或 `100`。

## 9. 把训练权重适配成 `eval_ppi.py` 能识别的结构

`eval_ppi.py` 的权重读取逻辑不是直接指向训练目录，而是会按下面这个模式拼路径：

```text
<framework.weightsdir>/<framework.eval_type>/<framework.weight_name>/checkpoints/<framework.ckpt_name>.pth.tar
```

因此最省事的办法是做一层软链接 shim。

### 9.1 选定训练输出目录

如果你刚刚训练完，通常可以直接取最新目录：

```bash
export EXP_DIR=$(ls -dt "exp_logs/ckpt/${TASK_DIR}"/* | head -1)
export RUN_NAME=$(basename "${EXP_DIR}")
echo "${EXP_DIR}"
echo "${RUN_NAME}"
```

### 9.2 建立评估权重目录

这里我们统一把 `eval_type` 固定成整数 `0`：

```bash
mkdir -p "eval_weights/${TASK_DIR}/0"
ln -sfn "$(readlink -f "${EXP_DIR}")" "eval_weights/${TASK_DIR}/0/${RUN_NAME}"
```

如果你想评估 `latest_model.pth.tar`，后面就令：

```bash
export CKPT_NAME=latest_model
```

如果你想评估某个周期快照，比如 `epoch50_model.pth.tar` 或 `epoch500_model.pth.tar`，则令：

```bash
export CKPT_NAME=epoch500_model
```

注意：`CKPT_NAME` 不要带 `.pth.tar` 后缀。

## 10. 在 `.test` 上做 RLBench 仿真评估

这里最重要的事实是：

- 评估使用的是 `data/eval_raw/${TASK_DIR}` 指向的 `.test` 原始 demo
- 不需要先给 `.test` 生成 point cloud / dino / point flow
- `framework.eval_type` 必须是整数，否则 `eval_ppi.py` 会直接抛 `Unknown eval type`

运行命令：

```bash
python eval_ppi.py \
  framework.eval_from_eps_number=0 \
  framework.eval_episodes=${TEST_EPISODES} \
  framework.csv_logging=true \
  framework.tensorboard_logging=false \
  framework.eval_type=0 \
  framework.weight_name="${RUN_NAME}" \
  framework.ckpt_name="${CKPT_NAME}" \
  framework.jump_step=1 \
  framework.weightsdir="$(pwd)/eval_weights/${TASK_DIR}" \
  framework.logdir="$(pwd)/eval_logs" \
  framework.eval_envs=1 \
  framework.eval_processes=1 \
  rlbench.headless=true \
  rlbench.episode_length=${EPISODE_LENGTH} \
  rlbench.task_name="${TASK_KEY}" \
  rlbench.tasks=[${TASK_DIR}] \
  rlbench.demo_path="$(pwd)/data/eval_raw" \
  rlbench.include_lang_goal_in_obs=true \
  rlbench.query_freq=${QUERY_FREQ} \
  method.policy.horizon_keyframe=4 \
  method.policy.horizon_continuous=50 \
  method.policy.n_obs_steps=1 \
  method.policy.n_action_steps=54 \
  method.policy.bounding_box=${BOUNDING_BOX} \
  method.policy.fps_num=6144 \
  method.policy.prediction_type=keyframe_continuous \
  method.policy.what_condition=ppi \
  method.policy.pointflow_num=200 \
  method.policy.text_prompt="${PF_TEXT_PROMPT}" \
  method.policy.prompt_type="${PF_PROMPT_TYPE}" \
  method.policy.sample_type=rps \
  method.policy.num_inference_steps=1000 \
  method.policy.sam_cameras=${SAM_CAMERAS} \
  method.policy.instruction_embeddings_path="$(pwd)/data/training_processed/instruction_embeddings.pkl" \
  cinematic_recorder.enabled=false
```

评估日志会落到：

```text
eval_logs/${TASK_KEY}/PPI/seed0
```

如果要录视频，把最后一项改成：

```bash
cinematic_recorder.enabled=true \
cinematic_recorder.save_path="$(pwd)/eval_videos/${TASK_DIR}/${RUN_NAME}"
```

## 11. 一条最简可执行流程

如果你已经理解上面每一步，真正执行时通常就是下面这条顺序：

1. `conda activate ppi`，然后 `cd /home/hdliu/occ_grasp_fall/occ_grasp_models`
2. 设好 `TASK_KEY`、`TASK_DIR`、`PF_TEXT_PROMPT`、`SAM_CAMERAS`
3. 建立 `data/training_raw` 和 `data/eval_raw` 软链接
4. 生成 `data/training_processed/instruction_embeddings.pkl`
5. 创建 `ppi/config/task/${TASK_KEY}.yaml`
6. 依次跑：
   - 离线 point cloud
   - 离线 dino
   - 离线 point flow
   - `norm_stats`
7. 用 `train_ppi_ddp.py` 训练
8. 给最新训练目录建一层 `eval_weights/${TASK_DIR}/0/${RUN_NAME}` 软链接
9. 用 `eval_ppi.py` 在 `.test` 上评估

## 12. 最常见的坑

### 12.1 在错误目录下运行脚本

很多路径是相对路径。最稳妥的做法始终是：

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
```

### 12.2 直接把 `rlbench.demo_path` 指向 `/mnt/rlbench_data`

当前 `/mnt/rlbench_data` 没有无后缀的 `bimanual_pivot_phone/` 目录，只有 `bimanual_pivot_phone.test/`。  
评估代码会按无后缀目录去找，所以直接指过去会失败。一定要先做：

```text
data/eval_raw/<task> -> /mnt/rlbench_data/<task>.test
```

### 12.3 以为 `.val` 会被训练代码自动使用

不会。当前 `RLBench2Dataset` 的验证集是从 `.train` 里按 `val_ratio` 再切出来的。`.val` 目前只是额外存在，但不会被自动消耗。

### 12.4 `save_norm_stats.py` 输出名和训练读取名不一致

仓库自带脚本默认会写出带 `_new.pth` 的文件，但训练 task yaml 期待的是标准名。所以推荐直接用上面第 7.4 节的流式脚本。

### 12.5 `point_flow_type` 配错

当前训练脚本和采样器读取路径时，真正应该统一的是：

```text
world_ordered_rps200
```

不要再用老 yaml 里的 `rps200`。

### 12.6 bounding box 过紧

若离线点云或在线评估时出现：

```text
Not enough points inside the bounding box
```

优先把：

```text
[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]
```

改成：

```text
[[-0.5,-0.55,0.75],[1.1,0.55,1.98]]
```

### 12.7 GroundingDINO 没有框到目标

这通常不是训练问题，而是 `PF_TEXT_PROMPT` / `SAM_CAMERAS` 没调好。建议先只在 `episode0` 上试跑 point flow，确认目标框稳定后再全量预处理和评估。

## 13. 推荐的第一轮实践策略

如果你第一次把这套流程跑通，建议按下面节奏来：

1. 先选 `pivot_phone` 或 `edge_phone`
2. 先只对 `episode0` 到 `episode2` 试跑 point cloud / dino / point flow
3. 确认 point flow 的目标框正常后，再全量预处理 `0..149`
4. 先训练较少 epoch 验证链路
5. 最后再用完整 epoch 和完整 `.test` 30 个 episode 做正式评估

这样最省时间，也最容易定位问题到底出在数据、预处理、训练还是评估。
