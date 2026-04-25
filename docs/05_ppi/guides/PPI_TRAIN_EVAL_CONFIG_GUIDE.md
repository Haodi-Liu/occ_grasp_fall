# occ_grasp_fall 中 PPI 训练与评估配置详解

本文面向当前仓库里的仿真 PPI 流程，重点解释以下几类文件的角色定位、相互关系，以及其中关键超参的实际含义和生效位置：

- `occ_grasp_models/ppi/config/ppi.yaml`
- `occ_grasp_models/ppi/config/task/*.yaml`
- `occ_grasp_models/conf/method/PPI.yaml`
- `occ_grasp_models/conf/eval_ppi.yaml`
- `occ_grasp_models/scripts/ppi/training/*.sh`
- `occ_grasp_models/scripts/ppi/inference/*.sh`
- 与这些配置强相关的数据预处理脚本

本文内容基于当前代码实际调用链整理，而不是只按 yaml 字面含义解释。

## 1. 先给结论：PPI 在这个仓库里有两套配置体系

PPI 在 `occ_grasp_fall` 里不是“一份配置跑到底”，而是训练和评估各有一套 Hydra 配置体系。

### 1.1 训练侧

训练真正的入口是：

- `occ_grasp_models/train_ppi_ddp.py`

它绑定的 Hydra 根配置是：

- `occ_grasp_models/ppi/config/ppi.yaml`

它再通过 `defaults` 引入任务配置：

- `occ_grasp_models/ppi/config/task/<task>.yaml`

实际运行时，通常还会被训练脚本继续覆盖：

- `occ_grasp_models/scripts/ppi/training/ddp_train_<task>.sh`

也就是说，训练侧的配置优先级可以理解成：

`ddp_train_<task>.sh` 命令行覆盖
-> `ppi/config/ppi.yaml`
-> `ppi/config/task/<task>.yaml`

### 1.2 评估侧

评估真正的入口是：

- `occ_grasp_models/eval_ppi.py`

它绑定的 Hydra 根配置是：

- `occ_grasp_models/conf/eval_ppi.yaml`

并通过 `defaults` 引入方法配置：

- `occ_grasp_models/conf/method/PPI.yaml`

实际运行时，通常还会被评估脚本继续覆盖：

- `occ_grasp_models/scripts/ppi/inference/evaluate_ppi_<task>.sh`

所以评估侧的配置优先级可以理解成：

`evaluate_ppi_<task>.sh` 命令行覆盖
-> `conf/eval_ppi.yaml`
-> `conf/method/PPI.yaml`

### 1.3 最重要的边界

- `ppi/config/ppi.yaml` 是 PPI 原生训练工作区配置，直接喂给 `train_ppi_ddp.py`
- `conf/method/PPI.yaml` 不是训练入口配置，它是评估框架里“PPI 方法”的描述文件，供 `eval_ppi.py -> agent_factory -> agents/ppi/launch_utils.py` 使用
- `ppi/config/task/*.yaml` 负责训练数据集和任务级别路径，不参与 `eval_ppi.py`
- `conf/eval_ppi.yaml` 负责 RLBench 评估环境、日志、权重装载和录像配置，不参与 `train_ppi_ddp.py`

这也是为什么当前仓库里训练 PPI 应该走 `train_ppi_ddp.py`，而不是 `train.py method=PPI`。

## 2. 调用链总览

### 2.1 训练链路

```text
scripts/ppi/training/ddp_train_<task>.sh
  -> torchrun train_ppi_ddp.py
     -> Hydra 读取 ppi/config/ppi.yaml
     -> defaults 引入 ppi/config/task/<task>.yaml
     -> 实例化 ppi.policy.ppi.PPI
     -> 实例化 ppi.dataset.rlbench2_dataset.RLBench2Dataset
     -> 训练 / 验证 / 保存 checkpoint
```

### 2.2 评估链路

```text
scripts/ppi/inference/evaluate_ppi_<task>.sh
  -> python eval_ppi.py
     -> Hydra 读取 conf/eval_ppi.yaml
     -> defaults 引入 conf/method/PPI.yaml
     -> agent_factory.create_agent(cfg)
     -> agents/ppi/launch_utils.py 创建 PPI + PPIAgent
     -> YARR / RLBench 环境运行评估
```

## 3. 训练侧文件的角色定位

## 3.1 `occ_grasp_models/ppi/config/ppi.yaml`

### 3.1.1 这个文件的角色

它是训练侧的“总控配置”。

它主要负责：

- 选择任务配置 `defaults: - task: ...`
- 定义模型本体 `policy`
- 定义优化器、EMA、dataloader、训练轮数、日志、checkpoint
- 定义 Hydra 输出目录结构

可以把它理解为：“一次 PPI 训练实验的全局模板”。

### 3.1.2 关键字段总览

#### `defaults`

- `task: "box"`

作用：

- 选择默认的任务配置文件，例如 `ppi/config/task/box.yaml`
- 训练脚本里常用 `task='pick_fork'` 这种命令行覆盖来切换任务

#### 运行命名相关

| 字段 | 含义 | 实际作用 |
|---|---|---|
| `name` | 实验主名 | 参与 `exp_name` 和 Hydra 输出目录命名 |
| `alg_name` | 算法名 | 一般固定为 `ppi` |
| `addition_info` | 附加标签 | 常用于写日期、版本、baseline 标识 |
| `seed` | 训练随机种子 | 模型、numpy、python random、cuda seed 都会用到 |
| `wandb_name` | W&B project 名 | 对应 `logging.project` |
| `config_name` | 配置名别名 | 一般等于 `alg_name` |
| `exp_name` | 实验名 | 由 `name/task_name/alg_name/addition_info` 拼出来 |
| `ckpt_dir` | checkpoint 根目录 | 默认为 `exp_logs/ckpt` |
| `run_dir` | 当前实验目录 | 真正的 Hydra 输出路径 |
| `task_name` | 任务短名 | 引用自 task yaml 里的 `task.name` |

当前输出目录的形成方式是：

`exp_logs/ckpt/<dataset_task_name>/<exp_name>_seed<seed>`

这里用的是 `task.dataset_task_name`，也就是 RLBench 真实任务目录名，例如 `bimanual_pick_fork`。

#### 时序相关核心参数

| 字段 | 含义 | 生效位置 | 说明 |
|---|---|---|---|
| `horizon_keyframe` | 关键帧段长度 | 模型结构、数据采样 | 完整 PPI 时通常为 4 |
| `horizon_continuous` | 连续段长度 | 模型结构、数据采样 | 当前常用 50 |
| `n_obs_steps` | 观测历史步数 | 模型输入、dataset pad_before、评估 history 长度 | 当前常用 1 |
| `n_action_steps` | 每次推理截取的动作步数 | 训练采样 pad_after、推理切片 | 当前常用 54 |
| `dataset_obs_steps` | 数据观测步数别名 | 基本只是引用 `n_obs_steps` | 当前配置里不是独立自由度 |

几个重要关系：

- 当前典型设置里，`horizon_keyframe + horizon_continuous = 4 + 50 = 54`
- 同时 `n_action_steps = 54`
- 这意味着模型输出的动作长度和训练/评估切片长度是对齐的

在当前代码里，最好保持这三者一致。如果你把 `n_action_steps` 改得和 `horizon_keyframe + horizon_continuous` 不一致，很容易引发训练采样、推理切片和评估执行之间的不对齐。

#### `policy`

这是最核心的一块，决定模型结构。

| 字段 | 含义 | 当前代码中的真实作用 |
|---|---|---|
| `_target_` | 实例化目标类 | 固定为 `ppi.policy.ppi.PPI` |
| `encoder_output_dim` | 编码器输出特征维度 | 传给 `ObservationEncoder` 和扩散头 |
| `horizon_keyframe` | 关键帧长度 | 决定扩散头结构与输出长度 |
| `horizon_continuous` | 连续长度 | 决定扩散头结构与输出长度 |
| `n_action_steps` | 动作切片长度 | `predict_action()` 里从全轨迹里截取多少步返回 |
| `n_obs_steps` | 观测步数 | 决定输入 history 使用几帧 |
| `noise_scheduler_cfg.num_train_timesteps` | 扩散训练步数 | 传给两个 DDPMScheduler |
| `noise_scheduler_cfg.prediction_type` | 扩散预测目标类型 | 当前实现应保持 `epsilon` |
| `num_inference_steps` | 推理扩散步数 | 越大越慢，通常质量更稳 |
| `use_lang` | 是否使用语言嵌入 | 决定是否启用语言分支 |
| `what_condition` | 选择哪种扩散头变体 | `ppi` 表示完整 PPI |
| `predict_point_flow` | 是否预测点流 | 打开点流分支和点流 loss |
| `pointcloud_encoder_cfg.in_channels` | PointNet 输入通道数 | 当前为 387 = XYZ 3 + DINO 384 |
| `pointcloud_encoder_cfg.out_channels` | PointNet 输出通道数 | 当前为 288 |
| `pointcloud_encoder_cfg.use_bn` | PointNet 是否使用 BN | 控制编码器内部 BN |

重点解释几个最容易误解的参数：

##### `what_condition`

这个参数不是普通标签，它直接决定 `ppi.policy.ppi.PPI` 里实例化哪种扩散头：

- `continuous` 或 `keyframe` -> `DiffusionHeadPure`
- `keypose_continuous` -> `DiffusionHeadKeyposeContinuous`
- `pointflow_continuous` -> `DiffusionHeadPointflowContinuous`
- `ppi` -> `DiffusionHeadPPI`

因此：

- `what_condition='ppi'` 才是完整 PPI 主方法
- 其他值本质上是消融或变体

##### `predict_point_flow`

这个布尔值会同时影响三件事：

1. 观测编码器是否接收 `initial_point_flow`
2. 前向训练时是否计算 `point_flow_loss`
3. 推理时是否输出 `point_flow_pred`

完整 PPI 的常见组合是：

- `what_condition='ppi'`
- `predict_point_flow=true`
- `task.dataset.prediction_type='keyframe_continuous'`

##### `noise_scheduler_cfg.prediction_type`

虽然底层 `DDPMScheduler` 支持多种 prediction type，但当前 `PPI.forward()` 的 loss 写法是按“预测噪声”来算的，代码里一直拿 `noise` 当监督目标。

因此当前实现里应保持：

- `prediction_type: epsilon`

如果把它改成 `sample` 或 `v_prediction`，scheduler 语义和训练 loss 就不再一致。

##### `pointcloud_encoder_cfg.in_channels=387`

这个数字不是随便来的，它对应：

- 点云坐标 `3`
- DINO 特征 `384`

合计 `387`

注意当前 sim PPI 主链路虽然常用 `pcd_type=rgb_pcd_rps6144`，但在 `ppi.policy.ppi.PPI` 里会把 `point_cloud` 截成前 3 维再送入编码器，所以当前网络真正消费的是：

- 点云 XYZ
- DINO 语义特征

而不是 RGB 颜色值本身。

#### `ema`

| 字段 | 含义 | 作用 |
|---|---|---|
| `_target_` | EMA 类 | `ppi.model.diffusion.ema_model.EMAModel` |
| `update_after_step` | 从第几步后开始进入 EMA 衰减 | 之前 decay=0 |
| `inv_gamma` | EMA warmup 形状参数 | 决定前期 decay 增长快慢 |
| `power` | EMA warmup 幂指数 | 越大越快逼近高 decay |
| `min_value` | EMA decay 下界 | 防止 decay 太小 |
| `max_value` | EMA decay 上界 | 限制最大 EMA 强度 |

当前实现里，EMA 用于评估和保存，训练期间每个优化步后都会 `ema.step(self.model.module)`。

#### `dataloader` / `val_dataloader`

| 字段 | 含义 | 实际行为 |
|---|---|---|
| `batch_size` | 全局 batch size | 进入 `train_ppi_ddp.py` 后会除以可见 GPU 数 |
| `num_workers` | DataLoader worker 数 | 常规 PyTorch 行为 |
| `shuffle` | 是否打乱 | 训练时会被代码强制改成 `False`，因为使用 `DistributedSampler` |
| `pin_memory` | 是否 pin memory | 常规 PyTorch 行为 |
| `persistent_workers` | worker 常驻 | 常规 PyTorch 行为 |

非常重要的一点：

- `train_ppi_ddp.py` 里会执行 `cfg.dataloader.batch_size = cfg.dataloader.batch_size // ngpus`
- 所以 yaml 或 shell 脚本里写的是“总 batch size”，不是单卡 batch size

例如：

- shell 里写 `dataloader.batch_size=32`
- 可见 GPU 数是 2
- 实际单卡 batch size 变成 16

#### `optimizer`

当前使用：

- `torch.optim.AdamW`

关键字段：

| 字段 | 含义 |
|---|---|
| `lr` | 学习率 |
| `betas` | Adam 一阶/二阶动量参数 |
| `eps` | 数值稳定项 |
| `weight_decay` | AdamW 权重衰减 |

#### `training`

| 字段 | 含义 | 当前代码中的真实作用 |
|---|---|---|
| `device` | 设备字符串 | 实际 DDP 训练还是按 `LOCAL_RANK` 选卡 |
| `seed` | 训练 seed | 参与随机初始化 |
| `debug` | debug 模式 | 会把 epoch、step、checkpoint 频率都缩小 |
| `resume` | 是否从 `latest.pth.tar` 恢复 | 只会从当前 `output_dir/checkpoints/latest.pth.tar` 续训 |
| `lr_scheduler` | 学习率调度器名 | 走 `diffusers.optimization` 的 scheduler |
| `lr_warmup_steps` | warmup 步数 | 用于 scheduler |
| `num_epochs` | 训练轮数 | 主训练循环上限 |
| `gradient_accumulate_every` | 梯度累积步数 | 影响等效 batch size |
| `use_ema` | 是否启用 EMA | 控制 EMA 模型和保存 |
| `rollout_every` | rollout 周期 | 当前训练脚本里基本未实际使用 |
| `checkpoint_every` | checkpoint 周期 | 每多少 epoch 存一次 |
| `val_every` | 验证周期 | 每多少 epoch 跑一次 val |
| `sample_every` | 采样可视化/误差统计周期 | 每多少 epoch 做一次训练 batch 采样评估 |
| `max_train_steps` | 限制每个 epoch 的训练 step 数 | debug/smoke test 常用 |
| `max_val_steps` | 限制每轮验证 step 数 | debug/smoke test 常用 |
| `tqdm_interval_sec` | 进度条刷新周期 | 纯显示相关 |

几个重点说明：

##### `resume`

当前 `resume=true` 的含义是：

- 从当前 Hydra `output_dir` 下的 `checkpoints/latest.pth.tar` 加载
- 同时恢复 model、ema_model、optimizer、lr_scheduler、epoch、global_step

因此它不是“自动找到最近一次任意实验”，而是“在当前 run_dir 里继续”。

##### `lr_scheduler`

这里调用的是 `ppi/model/common/lr_scheduler.py`，底层实际用的是 `diffusers.optimization.TYPE_TO_SCHEDULER_FUNCTION`。

因此合法值遵循 diffusers 的 scheduler 名字，例如：

- `constant`
- `constant_with_warmup`
- `cosine`
- 其他 diffusers 支持的类型

当前最常见是：

- `cosine`

##### `rollout_every`

这个字段在当前 `train_ppi_ddp.py` 里只是 debug 分支里会被改值，但正常训练主循环没有真正的 rollout 逻辑使用它。

因此它在当前训练脚本里基本是遗留/占位参数，不是有效调参手柄。

#### `logging`

对应 W&B 初始化参数。

| 字段 | 含义 |
|---|---|
| `group` | W&B group，当前等于 `exp_name` |
| `id` | run id，空则自动生成 |
| `mode` | `online/offline/disabled` 等 |
| `name` | run name，当前等于 `addition_info` |
| `project` | W&B project 名 |
| `resume` | W&B resume 标志 |
| `tags` | W&B tags |

#### `checkpoint`

| 字段 | 含义 | 备注 |
|---|---|---|
| `save_ckpt` | 是否保存 checkpoint | 总开关 |
| `topk.monitor_key` | 用哪个指标挑 top-k | 当前是 `val_loss` |
| `topk.mode` | `min/max` | `val_loss` 应该用 `min` |
| `topk.k` | 保留多少个 top checkpoint | 当前为 1 |
| `topk.format_str` | top checkpoint 命名格式 | 当前包含 epoch 和 val_loss |
| `save_last_ckpt` | 是否每次额外存一个 epoch 标签模型 | 当前代码里会生成 `epoch{n}_model.pth.tar` |
| `save_last_snapshot` | 是否存快照 | 当前代码里如果设成 `true` 会调用不存在的 `save_snapshot()`，不建议开启 |

这里有两个实践上很重要的细节：

1. 当前 `save_last_ckpt=true` 的行为不是“只保存 latest”，而是每次 checkpoint 周期额外保存一个 `epoch{n}_model.pth.tar`
2. 当前代码里没有实现 `save_snapshot()`，所以 `save_last_snapshot=true` 会有风险

#### `hydra`

这部分决定 Hydra 输出目录行为：

- `hydra.run.dir: ${run_dir}` 决定本次实验输出到哪里
- `hydra.job.override_dirname` / `hydra.sweep.*` 主要在 sweep 或多实验管理时使用

---

## 3.2 `occ_grasp_models/ppi/config/task/*.yaml`

### 3.2.1 这个目录的角色

这里的每个文件都是“任务级别训练配置”。

它主要负责：

- 真实任务名与短名映射
- 训练原始数据路径
- 预处理产物路径
- episode 范围
- sampler 类型
- 归一化 stats 文件名

对于新任务，当前最典型的是：

- `edge_phone.yaml`
- `pivot_phone.yaml`
- `pick_plate.yaml`
- `pick_fork.yaml`

这几个文件本质上是在 `box.yaml` 这一类旧模板上，换成了完整 PPI 需要的 `keyframe_continuous + world_ordered_rps200` 版本。

### 3.2.2 关键字段总览

#### 任务命名字段

| 字段 | 含义 | 例子 |
|---|---|---|
| `name` | 任务短名 | `pick_fork` |
| `task_name` | 同样是短名 | `pick_fork` |
| `dataset_task_name` | 真实 RLBench 任务目录名 | `bimanual_pick_fork` |

这里要分清：

- `task_name/name` 更偏配置短名和日志标签
- `dataset_task_name` 才是磁盘数据目录名

#### `dataset`

它决定 `RLBench2Dataset` 如何读取和采样训练数据。

| 字段 | 含义 | 生效位置 |
|---|---|---|
| `_target_` | 数据集类 | 固定为 `ppi.dataset.rlbench2_dataset.RLBench2Dataset` |
| `data_path` | 原始 demo 路径 | 读 `low_dim_obs.pkl` 等 |
| `pcd_path` | 预处理点云路径 | 采样器按 episode/step 读取 `.npy` |
| `dino_path` | DINO 特征路径 | 采样器按 episode/step 读取 `.npy` |
| `lang_emb_path` | 语言嵌入字典路径 | 用语言文本查 embedding |
| `stats_filepath` | 归一化参数路径 | `get_normalizer()` 读取 |
| `point_flow_path` | 点流路径 | `keyframe_continuous` 时必需 |
| `horizon_keyframe` | 关键帧长度 | 传给 sampler |
| `horizon_continuous` | 连续长度 | 传给 sampler |
| `pad_before` | 序列前补长度 | 当前通常等于 `n_obs_steps - 1` |
| `pad_after` | 序列后补长度 | 当前通常等于 `n_action_steps - 1` |
| `seed` | 数据集级随机种子 | 用于 val split 和下采样 |
| `start` | 起始 episode | 包含该索引 |
| `end` | 结束 episode | 包含该索引 |
| `pcd_fps` | 历史遗留参数 | 当前主链路里几乎没有真正控制点数 |
| `skip_ep` | 跳过的 episode 列表 | 数据清洗时用 |
| `kp_num` | 每个 episode 的目标关键帧数 | 影响 keypoint 补采样 |
| `val_ratio` | 验证集比例 | 从训练区间内部切分 |
| `max_train_episodes` | 最多使用多少训练 episode | 对 train split 再下采样 |
| `pcd_type` | 点云子目录名/格式标签 | 决定去哪个子目录读点云 |
| `prediction_type` | 数据组织模式 | 控制 ReplayBuffer 和 Sampler 选择 |
| `point_flow_type` | 点流子目录名/格式标签 | 决定去哪个子目录读点流 |
| `add_openess_sampling` | 是否上采样夹爪开合变化附近片段 | 影响训练样本分布 |

### 3.2.3 重要字段详解

#### `data_path / pcd_path / dino_path / point_flow_path / stats_filepath`

这五类路径分别对应五种不同的数据层：

1. `data_path`
   - 原始 RLBench demo
   - 包含 `low_dim_obs.pkl`、图像、深度、语言描述等

2. `pcd_path`
   - 预处理后的场景点云
   - 由 `scripts/ppi/data_generation/save_ptc.py` 生成

3. `dino_path`
   - 预处理后的点云语义特征
   - 由 `scripts/ppi/data_generation/save_dino.py` 生成

4. `point_flow_path`
   - 预处理后的目标点流
   - 由 `scripts/ppi/data_generation/save_point_flow.py` 生成

5. `stats_filepath`
   - 归一化统计文件
   - 由 `scripts/ppi/data_generation/save_norm_stats_generic.py` 生成

其中：

- `stats_filepath` 的命名规则默认是  
  `norm_stats_<task>_<pcd_type>_<prediction_type>_<point_flow_type>.pth`

这个命名规则很重要，因为它隐含了：

- 点云格式变了，stats 文件也应变
- `prediction_type` 变了，stats 文件也应变
- `point_flow_type` 变了，stats 文件也应变

#### `prediction_type`

这是训练侧最关键的 dataset 参数之一。

它决定 `RLBench2Dataset` 走哪种数据组织模式：

- `continuous`
  - 调用 `ReplayBuffer.getData_continuous()`
  - 使用 `SequenceSamplerContinuous`
  - 不读取 object_pose / point_flow / initial_point_flow

- `keyframe`
  - 调用 `ReplayBuffer.getData_keyframe()`
  - 使用 `SequenceSamplerKeyframe`
  - 只组织关键帧监督

- `keyframe_continuous`
  - 调用 `ReplayBuffer.getData_keyframe_continuous()`
  - 使用 `SequenceSamplerKeyframeContinuous`
  - 同时读取 action、object_pose、point_flow、initial_point_flow
  - 这是完整 PPI 的典型训练模式

新任务当前采用的是：

- `prediction_type: keyframe_continuous`

#### `kp_num`

这个参数不是“最多保留多少关键帧”，而是“目标关键帧数量”。

在 `GetDataKeyframeContinuous.keypoint_discovery_bimanual()` 里：

- 先根据夹爪开合变化、停止状态、episode 结尾去发现关键帧
- 如果发现的关键帧数量不足 `kp_num`
- 就会从剩余帧里均匀补采样，直到总数达到 `kp_num`

因此：

- `kp_num` 越大，关键帧监督越密
- 但也会让关键帧段更接近“稠密采样”

#### `val_ratio`

当前训练不会自动用外部 `.val` 数据目录。

`val_ratio` 的真实含义是：

- 在 `start ~ end` 这段训练 episode 内部，按比例随机切一部分做验证

实现位于 `sampler_keyframe_continuous.py` 里的 `get_val_mask()`。

也就是说：

- `val_ratio=0.2` 表示在当前训练区间内部切 20% episode 做 val
- 不是去读另一个 `.val` 数据源

#### `max_train_episodes`

它是在切完 train/val 之后，对 train split 再做一次随机下采样。

因此它的作用是：

- 控制真正参与训练的 episode 数
- 不影响 val split

如果它大于 train split 实际 episode 数，那么基本没有效果。

#### `add_openess_sampling`

这个参数只在 `SequenceSamplerKeyframeContinuous` 中生效。

作用是：

- 对接近夹爪开合变化时刻的样本窗口做重复采样
- 增大开合瞬间附近样本在训练集中的权重

当前实现里大致是：

- 距离开合变化点 10 步内，会额外重复采样
- 距离 5 步内，会再重复更多次

所以它的本质是：

- 一个针对夹爪状态切换的局部重加权开关

对于抓取/释放明显的双臂任务，这通常是有意义的。

#### `pcd_type`

这个字段不只是一个名字，它直接决定采样器去哪个子目录读取点云文件。

以当前新任务常见值为例：

- `rgb_pcd_rps6144`

它对应的真实含义可以按当前脚本理解为：

- `rgb_pcd`
  - `save_ptc.py` 里若 `pcd_type` 含 `rgb`，会把 RGB 一并拼到点云后面保存
- `rps6144`
  - 代码会从字符串里解析出 `ps_num=6144`
  - 再在 `bounding_box` 内做随机点采样保存

注意两点：

1. 当前点云预处理脚本用的是 bounding-box 内随机采样，不是 PointNet 内部的 FPS
2. 当前 sim PPI 主模型会把点云截成前 3 维再编码，所以训练主模型实际吃的是 XYZ，不直接吃 RGB

#### `point_flow_type`

它和 `pcd_type` 一样，首先是一个“子目录名”，采样器会按这个名字读文件。

新任务常见值：

- `world_ordered_rps200`

结合 `save_point_flow.py` 当前实现，它对应的实际语义可以理解为：

- `world`
  - 保存的是世界坐标系下的点位置
- `ordered`
  - 这些点来自同一批初始对象点，经物体坐标系传播到各时刻，因此点的对应关系是保持的
- `rps200`
  - 最终随机保留 200 个点

这个参数和下面几个量必须对齐：

- 训练用 `point_flow_type`
- `stats_filepath` 文件名
- 评估侧 `pointflow_num`

#### `pcd_fps`

这个字段名字很像“真实控制点数”的关键超参，但在当前 sim PPI 主链路里，它几乎没有真正参与后续文件读取逻辑。

更准确地说：

- 它会从 task yaml 传进 `RLBench2Dataset`
- 再传给 `ReplayBuffer.getData_*`
- 但当前这些 replay buffer loader 并没有继续用它决定文件内容

真正决定训练点数的是：

- `pcd_type` 对应的预处理文件内容
- 以及代码里写死/约定的维度，例如当前主链路默认 6144 scene points

因此当前代码下：

- `pcd_fps` 更像历史遗留参数或标签参数
- 真正要对齐的是 `pcd_type` 和预处理产物

### 3.2.4 新任务四个 task yaml 的共同特征

当前 `edge_phone/pivot_phone/pick_plate/pick_fork` 这四个任务文件有明显共性：

- `start: 0`
- `end: 149`
- `pcd_type: rgb_pcd_rps6144`
- `prediction_type: keyframe_continuous`
- `point_flow_type: world_ordered_rps200`
- `kp_num: 10`
- `val_ratio: 0.2`
- `add_openess_sampling: true`

这说明它们不是旧的“纯 continuous baseline”模板，而是完整 PPI 配置。

---

## 3.3 `occ_grasp_models/scripts/ppi/training/ddp_train_<task>.sh`

### 3.3.1 这个目录的角色

这些 shell 脚本是训练侧最后一层覆盖。

它们负责：

- 选择 GPU
- 设置 `torchrun` 参数
- 覆盖 task
- 覆盖 batch size / epoch / horizon
- 覆盖 `prediction_type`、`point_flow_type`、`stats_filepath`

因此它们虽然不是 yaml，但在实际实验里非常重要，因为最终生效值往往以 shell 覆盖为准。

### 3.3.2 以 `ddp_train_pick_fork.sh` 为例

它会覆盖：

- `task='pick_fork'`
- `name='train_ppi_ddp'`
- `addition_info='20260405_baseline'`
- `wandb_name='ppi_pick_fork'`
- `n_obs_steps=1`
- `n_action_steps=54`
- `policy.use_lang=true`
- `policy.what_condition='ppi'`
- `policy.predict_point_flow=true`
- `task.dataset.pcd_fps=6144`
- `task.dataset.pcd_type='rgb_pcd_rps6144'`
- `task.dataset.point_flow_type='world_ordered_rps200'`
- `task.dataset.kp_num=10`
- `task.dataset.prediction_type='keyframe_continuous'`
- `task.dataset.stats_filepath=...`
- `horizon_keyframe=4`
- `horizon_continuous=50`
- `dataloader.batch_size=32`
- `val_dataloader.batch_size=32`
- `training.num_epochs=500`

实践上可以把这些脚本理解成：

- “训练配置的实验版入口”

如果 task yaml 是任务静态模板，那么训练脚本就是：

- 当前机器、当前实验、当前 batch/GPU 资源下的最终运行版本

## 4. 与训练配置强相关的预处理文件

这些文件不是训练入口配置，但会直接决定训练配置里的路径、子目录名和超参是否成立。

## 4.1 `scripts/ppi/data_generation/save_ptc.py`

### 角色

- 生成 `pcd_path` 下的点云 `.npy`

### 它决定了哪些训练配置参数的真实含义

- `pcd_type`
- `bounding_box`
- 训练点云点数

### 关键事实

- 只要 `pcd_type` 含 `rgb`，就会把 RGB 拼到点云后面保存
- 会从 `pcd_type` 字符串里解析 `ps_num`
- 会在 `bounding_box` 内做随机点采样并保存

因此：

- `pcd_type=rgb_pcd_rps6144` 实际上对应“带 RGB 的点云文件目录，采样 6144 个点”

## 4.2 `scripts/ppi/data_generation/save_dino.py`

### 角色

- 读取 `pcd_path/<pcd_type>/stepXXX.npy`
- 为每个点生成 DINOv2 特征
- 保存到 `dino_path/<pcd_type>/stepXXX.npy`

### 关键事实

- DINO 特征维度当前固定是 384
- 这也是为什么 `pointcloud_encoder_cfg.in_channels = 3 + 384 = 387`

## 4.3 `scripts/ppi/data_generation/save_point_flow.py`

### 角色

- 生成 `point_flow_path/<point_flow_type>/stepXXX.npy`

### 它决定了哪些配置真正成立

- `point_flow_type`
- `text_prompt`
- `prompt_type`
- 预处理时使用的相机集合

### 关键事实

- 初始对象点来自 Grounding DINO + SAM 的分割结果
- 之后通过 `object_6d_pose` 把初始点从世界系变到物体系，再传播回每个时刻的世界系
- 最后按 `point_flow_type` 对应目录名保存
- 末尾 `ps200` 这一类数字来自字符串解析得到的采样点数

这意味着评估时的这些参数最好和训练前 point flow 预处理时保持一致：

- `method.policy.text_prompt`
- `method.policy.prompt_type`
- `method.policy.sam_cameras`
- `method.policy.pointflow_num`

## 4.4 `scripts/ppi/data_generation/save_norm_stats_generic.py`

### 角色

- 生成 `stats_filepath`

### 保存内容

当前脚本会统计并保存这些 normalizer 项：

- `action`
- `agent_pos`
- `lang`
- `point_cloud`
- `dino_feature`
- `point_flow`（仅 `keyframe_continuous`）
- `initial_point_flow`（仅 `keyframe_continuous`）

### 一个容易忽略的细节

训练时 `RLBench2Dataset.get_normalizer()` 会：

1. 先加载 `stats_filepath`
2. 再用当前 replay buffer 里的 `action` 重新 `fit`

所以：

- `stats_filepath` 主要承担的是观测项和点流项的归一化统计
- `action` 统计会在训练启动时再用当前数据重算一遍

## 4.5 `ppi/common/object_pose_from_contact.py`

### 角色

- 为缺少 `object_6d_pose` 的 demo 动态补出 `object_6d_pose`

### 为什么重要

`GetDataKeyframeContinuous` 和 `save_point_flow.py` 都依赖 `object_6d_pose`。

当前实现里，`maybe_patch_low_dim_obs()` 会在读取 `low_dim_obs.pkl` 时自动处理：

- 从 `misc.contact_position`
- 从 `misc.contact_quaternion`
- 结合 `T_OFFSET_INV[task_name]`

恢复出 `object_6d_pose`

当前仓库已经内置了四个新任务的常量：

- `bimanual_edge_phone`
- `bimanual_pivot_phone`
- `bimanual_pick_plate`
- `bimanual_pick_fork`

因此对于这四个任务，`keyframe_continuous` 链路现在是可跑通的。

但要注意：

- 如果换成不在 `T_OFFSET_INV` 里的新任务，这个补丁不会自动成立

## 5. 评估侧文件的角色定位

## 5.1 `occ_grasp_models/conf/eval_ppi.yaml`

### 5.1.1 这个文件的角色

它是评估侧的总控配置。

它主要负责：

- RLBench 环境参数
- YARR runner 参数
- 权重选择方式
- 日志目录
- 视频录制配置

它本身不定义 PPI 模型细节，模型细节主要来自：

- `conf/method/PPI.yaml`

### 5.1.2 关键字段总览

#### `defaults`

- `- method: PPI`

作用：

- 把 `conf/method/PPI.yaml` 组合进来

#### `rlbench`

| 字段 | 含义 | 实际作用 |
|---|---|---|
| `task_name` | 短任务名 | 主要用于日志目录和显示 |
| `tasks` | RLBench 真实任务列表 | 真正用于加载任务类 |
| `demo_path` | 评估 demo 根目录 | 环境会在其下按任务名找数据 |
| `lang_path` | 语言路径 | 当前 PPI 评估主链路基本不使用 |
| `episode_length` | 每个 episode 最长环境步数 | RLBench rollout 上限 |
| `cameras_pcd` | 用于场景点云融合的相机集合 | 传给 `PPIAgent.preprocess_pcd()` |
| `cameras` | 观测配置中的相机集合 | 决定环境会产出哪些 RGB/Depth/PointCloud |
| `camera_resolution` | 相机分辨率 | 传给 observation config |
| `scene_bounds` | 通用框架字段 | 当前 PPI 主链路不是核心参数 |
| `include_lang_goal_in_obs` | 是否在 obs 里带语言目标字符串 | PPIAgent 需要它来查 embedding |
| `time_in_state` | 是否把剩余时间拼进 low_dim_state | 对当前 PPIAgent 影响很小 |
| `headless` | 是否无界面运行 | RLBench / CoppeliaSim 常规参数 |
| `gripper_mode` | 夹爪动作模式 | 传给 RLBench action mode |
| `arm_action_mode` | 机械臂动作模式 | 当前应为末端位姿规划模式 |
| `action_mode` | 总动作模式 | 当前应为 `BimanualMoveArmThenGripper` |
| `query_freq` | 每隔多少环境步重新做一次模型推理 | PPIAgent 会缓存中间动作 |

重点解释：

##### `task_name` vs `tasks`

和训练侧 `task_name` vs `dataset_task_name` 很像，这里也有两层命名：

- `rlbench.task_name`
  - 更偏短标签
  - 用于日志目录、显示名

- `rlbench.tasks`
  - 必须是真正的 RLBench task 文件名列表
  - 例如 `bimanual_pick_fork`
  - `eval_ppi.py` 会据此加载 task class

##### `demo_path`

评估代码会把它当作“任务根目录的上一级”。

所以它应该指向：

- `data/eval_raw`

而不是：

- `data/eval_raw/bimanual_pick_fork`

##### `include_lang_goal_in_obs`

当前 PPIAgent 在第一步会读取：

- `observation['lang_goal']`

随后再用 `instruction_embeddings_path` 里的字典去查 embedding。

因此：

- 这个开关对当前 PPI 评估是实打实需要的

##### `cameras` 和 `cameras_pcd`

这两个字段不是完全独立的。

当前代码里：

- `rlbench.cameras` 决定环境真正会不会把对应相机的观测产出来
- `rlbench.cameras_pcd` 只是告诉 `PPIAgent` 去这些观测里取哪些相机来做场景点云融合

因此实践上应保证：

- `cameras_pcd` 是 `cameras` 的子集，或者至少两者一致

否则 `PPIAgent.preprocess_pcd()` 可能会去读一个环境根本没产出的相机字段。

##### `query_freq`

这个参数直接决定评估时的在线推理频率。

当前 `PPIAgent.act()` 的逻辑是：

- 当 `timestep % query_freq == 0` 或 `timestep == 1` 时重新跑一次模型
- 其余步复用上一次预测好的轨迹

因此：

- `query_freq` 越大，推理越省，但在线重规划越少
- `query_freq` 越小，推理更频繁，更接近逐步闭环

它要和下面这个参数一起理解：

- `framework.jump_step`

##### `time_in_state`

当前 `CustomRLBenchEnv` 会把一个时间标量拼到 `low_dim_state` 里。

但当前 PPIAgent 直接读取的是：

- `left_gripper_pose`
- `right_gripper_pose`
- `left_gripper_open`
- `right_gripper_open`

而不是 `low_dim_state`

因此在当前 sim PPI 主链路里：

- `time_in_state` 不是一个核心有效超参
- 它更多是通用 RLBench/YARR 框架的兼容项

#### `framework`

| 字段 | 含义 | 实际作用 |
|---|---|---|
| `tensorboard_logging` | 是否记 TensorBoard | 通用 runner 参数 |
| `csv_logging` | 是否输出 csv | 控制是否写 `eval_data.csv` |
| `gpu` | 评估设备 id | 传给 `utils.get_device()` |
| `logdir` | 评估日志根目录 | 最终会拼任务名、方法名、seed |
| `weightsdir` | 权重根目录 | 传入 YARR runner |
| `start_seed` | seed 标签 | 主要影响输出目录名，当前 eval 主链路里不是随机种子控制核心 |
| `record_every_n` | 录像相关频率 | 通用环境参数 |
| `eval_envs` | 评估环境数 | 传给 IndependentEnvRunner |
| `eval_from_eps_number` | 从第几个 demo episode 开始评估 | 直接影响评估 episode 编号 |
| `eval_episodes` | 总共评多少个 episode | 核心评估参数 |
| `eval_type` | 选择哪个权重子目录 | 在 `eval_ppi.py` 里当前只支持整数 |
| `eval_save_metrics` | 是否保存评估指标 | 控制 csv 输出 |
| `eval_processes` | 评估进程数 | 当前常设为 1 |
| `training_iterations` | YARR 兼容字段 | 对纯评估不是核心超参 |
| `weight_name` | 训练 run 名 | PPIAgent.load_weights 会用到 |
| `ckpt_name` | checkpoint 文件基名 | 例如 `latest_model` |
| `jump_step` | 从缓存轨迹里每次跳多少步 | 和 `query_freq` 一起决定动作推进速度 |

最重要的几项：

##### `eval_type`

这里要特别注意：

- 通用 `conf/eval.yaml` 里支持 `last/best/missing/all/int`
- 但 `eval_ppi.py` 当前只处理“整数”分支

也就是说，当前 PPI 评估入口里：

- `framework.eval_type` 应该设成整数

它不是 epoch 号，而是 YARR runner 会拼到 `weightsdir` 后面的“子目录名”。

这也是为什么当前指南推荐：

- `framework.eval_type=0`

并把 `0` 做成一个软链接目录。

##### `weightsdir + eval_type + weight_name + ckpt_name`

PPI 评估的权重装载路径不是两段，而是四段拼接。

拼接过程是：

1. `eval_ppi.py` 把 `framework.weightsdir` 交给 YARR runner
2. `_IndependentEnvRunner` 再拼出  
   `os.path.join(weightsdir, str(eval_type))`
3. `PPIAgent.load_weights(savedir)` 再继续拼  
   `os.path.join(savedir, weight_name, 'checkpoints', f'{ckpt_name}.pth.tar')`

所以最终路径是：

`framework.weightsdir/<eval_type>/<weight_name>/checkpoints/<ckpt_name>.pth.tar`

这就是下面三个字段为什么必须同时正确：

- `framework.weightsdir`
- `framework.weight_name`
- `framework.ckpt_name`

##### `jump_step`

当前 `PPIAgent.act()` 的动作推进逻辑是：

- 若 `action_id < jump_step * (query_freq - 1)`，则 `action_id += jump_step`
- 否则下一次重置回 0

因此：

- `jump_step=1` 表示缓存轨迹按连续步逐步取
- `jump_step=2` 表示每次从缓存轨迹里隔一步取一次

当前最稳妥的设置通常仍然是：

- `jump_step=1`

#### `cinematic_recorder`

| 字段 | 含义 |
|---|---|
| `enabled` | 是否录制视频 |
| `camera_resolution` | 视频分辨率 |
| `fps` | 视频帧率 |
| `rotate_speed` | 环绕相机速度 |
| `save_path` | 视频保存根目录 |

当前底层 runner 会把视频实际保存到：

- `${save_path}/videos/${weight_name}/...`

## 5.2 `occ_grasp_models/conf/method/PPI.yaml`

### 5.2.1 这个文件的角色

这是评估框架中的“PPI 方法定义文件”。

它的职责不是组织训练数据，而是告诉评估框架：

- 这是哪种 agent
- 用哪个 PPI 网络结构
- 在线评估时如何构造点云、点流、语言和视觉特征

所以它更像：

- “评估时的 PPI 方法说明书”

### 5.2.2 顶层字段

| 字段 | 含义 | 说明 |
|---|---|---|
| `name` | 方法名 | 必须是 `PPI`，因为 `agent_factory` 通过它路由到 `agents/ppi` |
| `agent_type` | agent 类型 | 当前必须是 `bimanual` |
| `robot_name` | 机器人类型 | 当前必须是 `bimanual` |
| `task.name` | 任务短名 | 从 `rlbench.task_name` 引用 |
| `task.task_name` | 同上 | 兼容字段 |
| `alg_name` | 算法名 | 更多是元数据 |
| `addition_info` | 附加信息 | 更多是元数据 |
| `seed` | seed | 评估主链路里不是核心控制项 |
| `horizon_keyframe` | 关键帧长度 | 被 `policy.*` 引用 |
| `horizon_continuous` | 连续长度 | 被 `policy.*` 引用 |
| `n_obs_steps` | 观测历史步数 | 被 `policy.*` 引用 |
| `n_action_steps` | 动作步数 | 被 `policy.*` 引用 |
| `task_name` | 任务短名 | 引用 `method.task.name` |

### 5.2.3 `policy`

这是评估侧最关键的一组字段。

它一部分用于创建 PPI 网络本身，一部分用于创建 `PPIAgent` 的在线感知逻辑。

#### 和训练模型结构直接相关的字段

| 字段 | 含义 | 与训练侧对应关系 |
|---|---|---|
| `encoder_output_dim` | 编码器输出维度 | 应与训练一致 |
| `horizon_keyframe` | 关键帧长度 | 应与训练一致 |
| `horizon_continuous` | 连续长度 | 应与训练一致 |
| `n_action_steps` | 动作步数 | 应与训练一致 |
| `n_obs_steps` | 观测步数 | 应与训练一致 |
| `noise_scheduler_cfg.num_train_timesteps` | 扩散总步数 | 应与训练一致 |
| `noise_scheduler_cfg.prediction_type` | 扩散预测目标 | 当前应保持 `epsilon` |
| `num_inference_steps` | 推理扩散步数 | 常可作为速度/质量权衡项 |
| `use_lang` | 是否使用语言 | 应与训练一致 |
| `what_condition` | 选哪种扩散头 | 完整 PPI 应是 `ppi` |
| `predict_point_flow` | 是否预测点流 | 完整 PPI 应为 `true` |
| `pointcloud_encoder_cfg.in_channels` | PointNet 输入通道 | 应与训练一致 |
| `pointcloud_encoder_cfg.out_channels` | PointNet 输出通道 | 应与训练一致 |
| `pointcloud_encoder_cfg.use_bn` | 是否用 BN | 应与训练一致 |

这些字段本质上是在评估时“重建与训练相同的网络结构”。

如果它们和训练配置不一致，最常见后果是：

- 权重 shape 对不上，加载失败
- 或能加载，但网络语义与训练时不一致

#### 只在评估在线感知路径中起作用的字段

| 字段 | 含义 | 生效位置 |
|---|---|---|
| `pointflow_num` | 在线初始化 point flow 采样点数 | `PPIAgent.get_initial_pointflow()` |
| `text_prompt` | Grounding DINO 文本提示 | `PPIAgent.get_point_from_mask()` |
| `prompt_type` | SAM 提示类型，`point/box` | `PPIAgent.get_point_from_mask()` |
| `sample_type` | 对初始目标点云的采样方式，`fps/rps` | `PPIAgent.get_initial_pointflow()` |
| `sam_cameras` | 用哪些相机做目标检测与 SAM | 只影响初始 point flow，不影响场景点云融合 |
| `bounding_box` | 场景点云裁剪框 | `PPIAgent.preprocess_pcd()` |
| `prediction_type` | 评估时如何解释输出轨迹 | `PPIAgent.act()` 里决定取哪一段动作 |
| `use_pc_color` | 是否把 RGB 拼进在线点云 | 当前主配置通常保持 `false` |
| `fps_num` | 在线场景点云采样点数 | `PPIAgent.preprocess_pcd()` |
| `sam_checkpoint_path` | SAM 权重路径 | 在线检测所需 |
| `gdino_config_path` | Grounding DINO 配置路径 | 在线检测所需 |
| `gdino_checkpoint_path` | Grounding DINO 权重路径 | 在线检测所需 |
| `instruction_embeddings_path` | 语言 embedding 字典路径 | 由 `lang_goal` 字符串查 embedding |

重点解释几个容易混淆的参数：

##### `prediction_type`

评估侧这个字段和训练侧 task yaml 里的 `prediction_type` 相关，但并不完全是同一层含义。

训练侧：

- 它决定 dataset / sampler 如何组织监督数据

评估侧：

- 它决定 `PPIAgent.act()` 如何从预测轨迹里取当前步动作
- 同时影响可视化颜色和轨迹解释

当前代码里：

- `continuous` -> 用 `result_action[0, action_id + 1]`
- `keyframe` -> 用 `result_action[0, action_id]`
- `keyframe_continuous` -> 也用 `result_action[0, action_id + 1]`

因此在完整 PPI 评估时，应与训练保持一致：

- `prediction_type=keyframe_continuous`

##### `fps_num`

这是评估时在线场景点云的采样点数。

它必须非常谨慎地与训练时保持一致，原因有两个：

1. `PPIAgent` 会显式创建形状为 `(B, T, fps_num, C)` 的点云和 `(B, T, fps_num, 384)` 的 DINO 特征
2. 当前 `ObservationEncoder` 里 `scene_pcd_num` 默认写死为 `6144`

所以在当前 sim PPI 主链路里，最稳妥的做法是：

- 保持 `fps_num=6144`
- 同时训练侧 `pcd_type` 也保持对应 `...rps6144`

##### `pointflow_num`

这是评估时在线初始化点流的点数。

它应该和训练的 `point_flow_type` 点数对齐。

例如当前新任务是：

- `point_flow_type=world_ordered_rps200`

那么评估时通常就应保持：

- `pointflow_num=200`

否则评估时点流分支输入/输出长度就和训练分布不一致。

##### `sam_cameras`

这个参数只影响：

- 初始 point flow 的目标定位与采样

它不等于场景点云使用的相机集合。

要区分三套相机：

1. `rlbench.cameras`
   - 环境实际输出哪些相机观测

2. `rlbench.cameras_pcd`
   - 场景点云融合用哪些相机

3. `method.policy.sam_cameras`
   - 初始目标点流由哪些相机做目标检测和掩膜

所以：

- `sam_cameras` 可以只用 `["front"]`
- 同时 `cameras_pcd` 仍然可以是 6 个相机

##### `bounding_box`

这个参数会在评估时裁掉场景点云中不在操作工作空间内的点，再从框内采样 `fps_num` 个点。

它的作用非常直接：

- 框太小 -> 有效点不够，采样可能报错
- 框太大 -> 点云里背景和无关物体太多

它和训练前 `save_ptc.py` 的 `bounding_box` 最好保持一致。

##### `instruction_embeddings_path`

评估时，PPIAgent 并不是用 `rlbench.lang_path` 去拿语言 embedding。

它真正做的是：

- 从环境观测里拿到 `lang_goal` 字符串
- 用这个字符串去索引 `instruction_embeddings_path` 对应的 pickle 字典

所以这个文件必须满足：

- key 是 RLBench 环境实际给出的语言指令字符串
- value 是对应的 1024 维 embedding

如果字符串对不上，评估会直接查不到。

## 5.3 `occ_grasp_models/scripts/ppi/inference/evaluate_ppi_<task>.sh`

### 5.3.1 这个目录的角色

和训练脚本一样，这些 shell 文件是评估侧最终的“实验运行版配置”。

它们负责覆盖：

- 任务名
- 权重路径
- checkpoint 名
- 评估 episode 数
- `query_freq`
- `text_prompt`
- `sam_cameras`
- `bounding_box`

### 5.3.2 这些脚本为什么比 `eval_ppi.yaml` 更关键

因为 `eval_ppi.yaml` 里的默认值明显只是模板值，例如：

- `task_name: "box"`
- `tasks: [bimanual_push_box]`
- `demo_path: your/test/demo/dir`
- `logdir: your/log/dir`
- `weight_name: ""`
- `ckpt_name: ""`

真正能跑的参数通常都在具体 `evaluate_ppi_<task>.sh` 里覆盖。

---

## 6. 参数对齐关系：哪些必须一起改

这一节是实际使用里最重要的“联动关系”。

## 6.1 训练结构和评估结构必须对齐

下面这些项训练和评估应该保持一致：

- `horizon_keyframe`
- `horizon_continuous`
- `n_obs_steps`
- `n_action_steps`
- `encoder_output_dim`
- `noise_scheduler_cfg.num_train_timesteps`
- `noise_scheduler_cfg.prediction_type`
- `what_condition`
- `predict_point_flow`
- `pointcloud_encoder_cfg.*`
- 评估侧 `prediction_type`

## 6.2 训练预处理和评估在线感知必须尽量对齐

下面这些项训练前预处理和评估在线感知最好一致：

- 训练 `pcd_type` <-> 评估 `fps_num`
- 训练 `point_flow_type` <-> 评估 `pointflow_num`
- `save_point_flow.py` 的 `text_prompt` <-> 评估 `method.policy.text_prompt`
- `save_point_flow.py` 的 `prompt_type` <-> 评估 `method.policy.prompt_type`
- `save_point_flow.py` 的相机集合 <-> 评估 `method.policy.sam_cameras`
- `save_ptc.py` 的 `bounding_box` <-> 评估 `method.policy.bounding_box`

## 6.3 路径相关三组必须一起对齐

### 训练侧

- `task.dataset.data_path`
- `task.dataset.pcd_path`
- `task.dataset.dino_path`
- `task.dataset.point_flow_path`
- `task.dataset.stats_filepath`

### 评估侧

- `rlbench.demo_path`
- `framework.weightsdir`
- `framework.weight_name`
- `framework.ckpt_name`
- `method.policy.instruction_embeddings_path`

## 7. 常见误区和当前代码里的特殊点

## 7.1 `train.py method=PPI` 不是当前主训练入口

当前仓库里，PPI 训练主入口是：

- `train_ppi_ddp.py`

而不是通用训练入口：

- `train.py`

## 7.2 `dataloader.shuffle=true` 在 DDP 训练里不会真正生效

因为 `train_ppi_ddp.py` 会把它改成 `False`，实际打乱由 `DistributedSampler` 完成。

## 7.3 `batch_size` 写的是全局 batch，不是单卡 batch

训练脚本会按可见 GPU 数再除一次。

## 7.4 `pcd_fps` 在当前 sim PPI 主链路里不是决定性有效参数

真正决定点云文件内容的是：

- `pcd_type`
- 预处理脚本生成的 `.npy`

## 7.5 `rollout_every` 当前基本未使用

它是遗留/占位超参，不是当前训练主循环的有效调节项。

## 7.6 `checkpoint.save_last_snapshot=true` 当前不建议开

因为训练脚本会调用不存在的 `save_snapshot()`。

## 7.7 `eval_ppi.py` 当前只接受整数型 `framework.eval_type`

不要直接照搬通用 `eval.yaml` 的：

- `last`
- `best`

到 `eval_ppi.py`

当前 PPI 评估入口只处理整数分支。

## 7.8 `rlbench.lang_path` 在当前 PPI 评估主链路里基本未使用

环境包装器里相关参数已经注释掉了，真正起作用的是：

- `include_lang_goal_in_obs`
- `method.policy.instruction_embeddings_path`

## 7.9 `use_pc_color` 不建议随便打开

当前训练和评估主配置默认都是围绕“XYZ + DINO 特征”设计的。

如果要让网络真的使用 RGB 点云颜色，通常还需要一起检查：

- 点云预处理文件格式
- `pointcloud_encoder_cfg.in_channels`
- `PPI.forward()/predict_action()` 里对 `point_cloud[..., :3]` 的截断

## 7.10 `prediction_type` 训练侧和评估侧虽然名字一样，但作用层级不同

- 训练侧：决定 dataset/sampler 组织方式
- 评估侧：决定如何解释预测轨迹

完整 PPI 时通常仍然应保持一致。

## 8. 对当前四个新任务的直接建议

如果你现在处理的是：

- `bimanual_edge_phone`
- `bimanual_pivot_phone`
- `bimanual_pick_plate`
- `bimanual_pick_fork`

那么当前代码下较合理的一致性组合是：

- 训练 task yaml 用 `prediction_type=keyframe_continuous`
- `pcd_type=rgb_pcd_rps6144`
- `point_flow_type=world_ordered_rps200`
- `kp_num=10`
- `horizon_keyframe=4`
- `horizon_continuous=50`
- `n_obs_steps=1`
- `n_action_steps=54`
- 评估时 `fps_num=6144`
- 评估时 `pointflow_num=200`
- 评估时 `what_condition=ppi`
- 评估时 `predict_point_flow=true`
- 评估时 `prediction_type=keyframe_continuous`

并让以下几项在训练预处理和评估之间尽量一致：

- `text_prompt`
- `prompt_type`
- `sam_cameras`
- `bounding_box`

## 9. 一句话概括每个关键文件

- `ppi/config/ppi.yaml`
  - 训练总控模板，管模型、优化、日志、checkpoint、Hydra 输出

- `ppi/config/task/*.yaml`
  - 训练任务模板，管数据路径、预处理产物、采样方式和任务级超参

- `scripts/ppi/training/ddp_train_<task>.sh`
  - 训练最终运行版覆盖层，管 GPU、batch、epoch 和任务选择

- `conf/eval_ppi.yaml`
  - 评估总控模板，管 RLBench 环境、runner、权重选择、日志和录像

- `conf/method/PPI.yaml`
  - 评估侧的 PPI 方法定义，管网络结构重建和在线感知超参

- `scripts/ppi/inference/evaluate_ppi_<task>.sh`
  - 评估最终运行版覆盖层，管具体任务、权重路径和在线感知参数

- `save_ptc.py / save_dino.py / save_point_flow.py / save_norm_stats_generic.py`
  - 训练配置能否成立的前置条件，决定路径命名、文件格式和 stats 内容

---

如果后续你希望，我可以继续在这份文档基础上再补一版：

- “面向实际调参”的版本

专门把哪些参数最值得调、哪些参数最好别动、以及针对四个新任务的推荐改法单独列出来。
