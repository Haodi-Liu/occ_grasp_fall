# ACT_BC_VISION 多任务按任务分别归一化的最小修改方案

## 结论

这个改动**不难**，而且可以做到比较“干净的小改”。

如果你的目标是：

1. 多任务 replay 继续按现在的方式混合采样；
2. 每个样本的 `qpos/action` 归一化都使用它**所属任务自己的统计量**；
3. 评估/推理时也按当前任务切换对应统计量；
4. 尽量少改代码，不引入新的复杂配置；

那么最小可行方案是：

1. 在训练 replay 中额外存一个数值型 `task_id`；
2. 在评估环境返回的 observation 中也额外带一个数值型 `task_id`；
3. 让 `ACT_BC_VISION` agent 在内部维护 `task_id -> 该任务统计量` 的映射，并在 `preprocess_action / preprocess_qpos / act` 三处按 `task_id` 选用对应统计量。

这样只需要改 3 个脚本：

1. [`occ_grasp_models/agents/act_bc_vision/launch_utils.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/launch_utils.py)
2. [`occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py)
3. [`occ_grasp_models/helpers/custom_rlbench_env.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py)

不需要改配置 schema，不需要改训练入口，也不需要改 replay 采样逻辑。

---

## 现状问题

当前 `ACT_BC_VISION` 的统计量来源是单任务：

- agent 初始化时只传了 `cfg.rlbench.tasks[0]`
  见 [`launch_utils.py:365-376`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/launch_utils.py#L365)
- `train_stats()` 只读取 `self.task_name.train` 这一份数据
  见 [`act_bc_vision_agent.py:69-113`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py#L69)

所以即使 replay 里已经混入了多任务样本，归一化依然只按第一个 task 的统计量做。

---

## 为什么这里要用 `task_id`，而不是直接传任务名字符串

训练 replay 里现在已经有字符串字段 `task`，它不能删，因为 `TaskUniformReplayBuffer` 就靠这个字段做“按任务均匀采样”：

- 见 [`task_uniform_replay_buffer.py:62-66`](/home/hdliu/occ_grasp_fall/repos/YARR/yarr/replay_buffer/task_uniform_replay_buffer.py#L62)

但是**评估 observation 不能直接塞字符串 task name**，因为 rollout generator 会把 observation 里的每个字段都转成 `torch.tensor(...)`：

- 见 [`repos/YARR/yarr/utils/rollout_generator.py:32-36`](/home/hdliu/occ_grasp_fall/repos/YARR/yarr/utils/rollout_generator.py#L32)

字符串会在这里出问题，所以评估路径里必须传**数值型 task_id**。

因此最稳妥的做法是：

- 保留已有的字符串 `task`，继续给 replay 采样逻辑使用；
- 额外新增一个整型 `task_id`，给 agent 做 task-wise 归一化使用。

---

## 已有工具可直接复用

仓库里已经有一个通用统计函数：

- [`helpers/qpos_stats.py:27-111`](/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/qpos_stats.py#L27)

它已经支持：

- 给定 `data_root`
- 给定一个或多个 `train_tasks`
- 返回 right/left gripper pose + gripper open 的统计量

所以 `ACT_BC_VISION` 不需要自己再手写遍历 `low_dim_obs.pkl` 的逻辑，直接复用这个 helper 即可。

---

## 修改总览

### 改动 1

文件：
[`occ_grasp_models/agents/act_bc_vision/launch_utils.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/launch_utils.py)

目的：

- replay 中新增 `task_id`
- 创建 agent 时把完整 `task_names` 列表传进去，而不是只传第一个任务名

### 改动 2

文件：
[`occ_grasp_models/helpers/custom_rlbench_env.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py)

目的：

- 单任务评估和多任务评估的 observation 都补上数值型 `task_id`
- 这样 `agent.act()` 才能知道当前 observation 属于哪个任务

### 改动 3

文件：
[`occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py)

目的：

- 为每个 task 单独计算一份 stats
- 训练时按 batch 内每个样本自己的 `task_id` 做归一化
- 评估时按当前 observation 的 `task_id` 做归一化和反归一化

---

## 详细改法

## 1. 修改 `launch_utils.py`

当前相关位置：

- replay schema：[`launch_utils.py:83-91`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/launch_utils.py#L83)
- 往 replay 写入字段：[`launch_utils.py:232-249`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/launch_utils.py#L232)
- create_agent：[`launch_utils.py:365-376`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/launch_utils.py#L365)

### 1.1 在 replay schema 里增加 `task_id`

把下面这段：

```python
    # NO LANGUAGE - removed lang_goal_emb and lang_goal
    observation_elements.extend([
        ReplayElement('task', (),
                      str),
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), bool),
    ]
```

改成：

```python
    # NO LANGUAGE - removed lang_goal_emb and lang_goal
    # Keep string task for TaskUniformReplayBuffer's task-balanced sampling.
    observation_elements.extend([
        ReplayElement('task', (), str),
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), bool),
        ReplayElement('task_id', (), np.int32),
    ]
```

### 注解

- `task` 字符串不能删，因为 `TaskUniformReplayBuffer` 还要用它做 task-uniform sampling。
- `task_id` 是新加的数值字段，只给 agent 做 task-wise stats 选择。
- 放在 `extra_replay_elements` 即可，因为它不是视觉观测，只是 meta 信息。

### 1.2 往 replay 里真正写入 `task_id`

把下面这段：

```python
    # NO LANGUAGE - removed language encoding
    final_obs = {
        'task': task,
    }
```

改成：

```python
    # NO LANGUAGE - removed language encoding
    final_obs = {
        'task': task,
        'task_id': np.int32(cfg.rlbench.tasks.index(task)),
    }
```

### 注解

- 这里直接用 `cfg.rlbench.tasks.index(task)` 就够了，逻辑最简单。
- 这个 `task_id` 的编号顺序与配置中的 `cfg.rlbench.tasks` 完全一致。
- 后面 agent 也按这同一顺序去建立 `task_id -> task_name -> stats` 映射。

### 1.3 创建 agent 时传完整任务列表

把当前这段：

```python
    bc_agent = ActBCVisionAgent(
        actor_network=actor_net,
        camera_names=cfg.rlbench.cameras,
        lr=cfg.method.lr,
        weight_decay=cfg.method.weight_decay,
        grad_clip=cfg.method.grad_clip,
        episode_length=cfg.rlbench.episode_length,
        train_demo_path=cfg.method.train_demo_path,
        task_name=cfg.rlbench.tasks[0])
```

改成：

```python
    bc_agent = ActBCVisionAgent(
        actor_network=actor_net,
        camera_names=cfg.rlbench.cameras,
        lr=cfg.method.lr,
        weight_decay=cfg.method.weight_decay,
        grad_clip=cfg.method.grad_clip,
        episode_length=cfg.rlbench.episode_length,
        train_demo_path=cfg.method.train_demo_path,
        task_name=cfg.rlbench.tasks[0],
        task_names=cfg.rlbench.tasks,
    )
```

### 注解

- `task_name=cfg.rlbench.tasks[0]` 可以保留，作为缺省 fallback。
- 新增 `task_names=cfg.rlbench.tasks` 后，agent 就知道这次训练涉及哪些任务。

---

## 2. 修改 `custom_rlbench_env.py`

当前相关位置：

- `CustomRLBenchEnv.extract_obs_bimanual()`：[`custom_rlbench_env.py:186-246`](/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py#L186)
- `CustomRLBenchEnv.extract_obs_unimanual()`：[`custom_rlbench_env.py:248`](/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py#L248)
- `CustomMultiTaskRLBenchEnv.extract_obs_bimanual()`：[`custom_rlbench_env.py:741-796`](/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py#L741)
- `CustomMultiTaskRLBenchEnv.extract_obs_unimanual()`：[`custom_rlbench_env.py:798-833`](/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py#L798)

### 2.1 在 `CustomRLBenchEnv` 的 observation 中增加 `task_id`

把 `CustomRLBenchEnv.extract_obs_bimanual()` 末尾从：

```python
        obs_dict['right_gripper_pose'] = obs.right.gripper_pose
        obs_dict['right_gripper_open'] = np.array([obs.right.gripper_open])
        obs_dict = self._append_aux_gt(obs, obs_dict)
        return obs_dict
```

改成：

```python
        obs_dict['right_gripper_pose'] = obs.right.gripper_pose
        obs_dict['right_gripper_open'] = np.array([obs.right.gripper_open])
        obs_dict = self._append_aux_gt(obs, obs_dict)
        obs_dict['task_id'] = np.int32(max(self.active_task_id, 0))
        return obs_dict
```

把 `CustomRLBenchEnv.extract_obs_unimanual()` 末尾从：

```python
        obs_dict['joint_positions'] = obs.joint_positions
        obs_dict['gripper_joint_positions'] = obs.gripper_joint_positions
        obs_dict = self._append_aux_gt(obs, obs_dict)
        return obs_dict
```

改成：

```python
        obs_dict['joint_positions'] = obs.joint_positions
        obs_dict['gripper_joint_positions'] = obs.gripper_joint_positions
        obs_dict = self._append_aux_gt(obs, obs_dict)
        obs_dict['task_id'] = np.int32(max(self.active_task_id, 0))
        return obs_dict
```

### 注解

- 单任务 env 下，`active_task_id` 默认就是 0。
- `max(self.active_task_id, 0)` 是保险写法，避免少数未初始化场景出现 `-1`。

### 2.2 在 `CustomMultiTaskRLBenchEnv` 的 observation 中增加 `task_id`

把 `CustomMultiTaskRLBenchEnv.extract_obs_bimanual()` 末尾从：

```python
        obs_dict['right_gripper_pose'] = obs.right.gripper_pose
        obs_dict['right_gripper_open'] = np.array([obs.right.gripper_open])

        obs_dict = self._append_aux_gt(obs, obs_dict)
        return obs_dict
```

改成：

```python
        obs_dict['right_gripper_pose'] = obs.right.gripper_pose
        obs_dict['right_gripper_open'] = np.array([obs.right.gripper_open])

        obs_dict = self._append_aux_gt(obs, obs_dict)
        obs_dict['task_id'] = np.int32(self.active_task_id)
        return obs_dict
```

把 `CustomMultiTaskRLBenchEnv.extract_obs_unimanual()` 末尾从：

```python
        obs_dict['joint_positions'] = obs.joint_positions
        obs_dict['gripper_joint_positions'] = obs.gripper_joint_positions

        obs_dict = self._append_aux_gt(obs, obs_dict)
        return obs_dict
```

改成：

```python
        obs_dict['joint_positions'] = obs.joint_positions
        obs_dict['gripper_joint_positions'] = obs.gripper_joint_positions

        obs_dict = self._append_aux_gt(obs, obs_dict)
        obs_dict['task_id'] = np.int32(self.active_task_id)
        return obs_dict
```

### 注解

- 多任务评估时，`self.active_task_id` 正好就是当前 task 在 `task_classes` / `cfg.rlbench.tasks` 中的编号。
- 这个编号会随着环境切换任务自动变化，不需要额外改 `eval.py`。

---

## 3. 修改 `act_bc_vision_agent.py`

这是核心改动。

当前相关位置：

- `__init__`：[`act_bc_vision_agent.py:25-40`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py#L25)
- `train_stats()`：[`act_bc_vision_agent.py:69-113`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py#L69)
- `preprocess_qpos()`：[`act_bc_vision_agent.py:124-160`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py#L124)
- `preprocess_action()`：[`act_bc_vision_agent.py:164-197`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py#L164)
- `act()`：[`act_bc_vision_agent.py:252-316`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py#L252)

### 3.1 先增加 import

在文件顶部 import 区增加：

```python
from helpers.qpos_stats import compute_qpos_stats
```

### 3.2 修改 `__init__`

把原来的构造函数：

```python
    def __init__(self,
                 actor_network: nn.Module,
                 camera_names: List[str],
                 lr: float = 0.01,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 20.0,
                 episode_length: int = 400, train_demo_path=None, task_name=None):
        self._camera_names = camera_names
        self._actor = actor_network
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._episode_length = episode_length
        self.train_demo_path = train_demo_path
        self.task_name = task_name
        self.visual_targets = []
```

替换成：

```python
    def __init__(self,
                 actor_network: nn.Module,
                 camera_names: List[str],
                 lr: float = 0.01,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 20.0,
                 episode_length: int = 400,
                 train_demo_path=None,
                 task_name=None,
                 task_names=None):
        self._camera_names = camera_names
        self._actor = actor_network
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._episode_length = episode_length
        self.train_demo_path = train_demo_path

        self.task_names = list(task_names) if task_names is not None else (
            [task_name] if task_name is not None else []
        )
        self.task_name = task_name if task_name is not None else (
            self.task_names[0] if len(self.task_names) > 0 else None
        )
        self._task_id_to_name = {
            idx: name for idx, name in enumerate(self.task_names)
        }

        self.visual_targets = []
```

### 注解

- `task_name` 保留，作为单任务和 fallback 兼容。
- `task_names` 新增，用来建立多任务的 `id -> name` 映射。
- 这里不需要新配置项，直接复用 `cfg.rlbench.tasks`。

### 3.3 用 `compute_qpos_stats()` 代替原来的单任务 `train_stats()`

把原来的 `train_stats()` 整段：

```python
    @lru_cache()
    def train_stats(self):

        right_gripper_poses = []  # 7D: xyz(3) + quat(4)
        left_gripper_poses = []

        right_gripper_open = []   # 1D: gripper open state
        left_gripper_open = []

        episodes_dir = f"{self.train_demo_path}/{self.task_name}.train/all_variations/episodes/"

        for episode in os.listdir(episodes_dir):
            with open(os.path.join(episodes_dir, episode, "low_dim_obs.pkl"), "br") as f:
                d = pickle.load(f)

            for o in d:
                right_gripper_poses.append(o.right.gripper_pose)
                left_gripper_poses.append(o.left.gripper_pose)

                right_gripper_open.append([o.right.gripper_open])
                left_gripper_open.append([o.left.gripper_open])

        right_gripper_poses = np.asarray(right_gripper_poses, dtype=np.float32)
        left_gripper_poses = np.asarray(left_gripper_poses, dtype=np.float32)

        right_gripper_open = np.asarray(right_gripper_open, dtype=np.float32)
        left_gripper_open = np.asarray(left_gripper_open, dtype=np.float32)

        # Compute statistics for position (xyz) only, not quaternion
        # Quaternions are unit vectors and should not be normalized with z-score
        stats = {
            "right_pos_mean": right_gripper_poses[:, :3].mean(axis=0),
            "right_pos_std": right_gripper_poses[:, :3].std(axis=0),

            "left_pos_mean": left_gripper_poses[:, :3].mean(axis=0),
            "left_pos_std": left_gripper_poses[:, :3].std(axis=0),

            "right_gripper_open_mean": right_gripper_open.mean(axis=0),
            "right_gripper_open_std": right_gripper_open.std(axis=0),

            "left_gripper_open_mean":  left_gripper_open.mean(axis=0),
            "left_gripper_open_std": left_gripper_open.std(axis=0)
        }

        return {k: torch.from_numpy(v).to(self._device) for k,v in stats.items()}
```

替换成下面这一组方法：

```python
    @lru_cache(maxsize=1)
    def train_stats(self):
        stats_by_task = {}
        for task_name in self.task_names:
            stats_by_task[task_name] = compute_qpos_stats(
                data_root=self.train_demo_path,
                train_tasks=[task_name],
                out_path=None,
                device=self._device,
            )
        return stats_by_task

    def _stats_for_task_id(self, task_id: int):
        task_name = self._task_id_to_name.get(int(task_id), self.task_name)
        return self.train_stats()[task_name]

    def _get_task_ids(self, task_source, batch_size: int) -> torch.Tensor:
        if task_source is None:
            return torch.zeros(batch_size, dtype=torch.long, device=self._device)

        if not torch.is_tensor(task_source):
            task_source = torch.as_tensor(task_source, device=self._device)

        task_ids = task_source.to(self._device).long()

        if task_ids.dim() == 0:
            return task_ids.unsqueeze(0).repeat(batch_size)

        if task_ids.dim() == 1:
            if task_ids.shape[0] == batch_size:
                return task_ids
            return task_ids[-1].repeat(batch_size)

        if task_ids.shape[0] == batch_size:
            return task_ids[:, -1]

        return task_ids.reshape(task_ids.shape[0], -1)[:, -1]

    def _normalize_pos_by_task(self,
                               data: torch.Tensor,
                               task_ids: torch.Tensor,
                               mean_key: str,
                               std_key: str) -> torch.Tensor:
        out = torch.empty_like(data)
        for row_idx, task_id in enumerate(task_ids.detach().cpu().tolist()):
            stats = self._stats_for_task_id(task_id)
            out[row_idx] = self.normalize_z(
                data[row_idx], stats[mean_key], stats[std_key]
            )
        return out
```

### 注解

- `train_stats()` 现在返回的是：

```python
{
    "bimanual_pick_plate": {...},
    "bimanual_pick_fork": {...},
    "bimanual_edge_phone": {...},
    "bimanual_pivot_phone": {...},
}
```

- `_get_task_ids()` 用来兼容训练和评估两种输入形状：
  - 训练 batch 时通常是 `(B,)` 或 `(B,1)`
  - 评估 observation history 经过 rollout generator 后通常是 `(1,T)`
- `_normalize_pos_by_task()` 是最简单、最稳的写法。
  - 它不是最极致高性能的写法；
  - 但 batch size 只有十几，这点循环开销相对模型前向是很小的；
  - 逻辑清晰，最适合先把功能改对。

### 3.4 修改 `preprocess_qpos()`

把当前版本：

```python
    def preprocess_qpos(self, observation: dict):

        stats = self.train_stats()

        # Right gripper pose: normalize position (xyz), keep quaternion as is
        # Handle both 3D (B, timesteps, 7) from training and 2D (B, 7) from evaluation
        right_pose = observation['right_gripper_pose']
        if right_pose.dim() == 3:
            right_pose = right_pose[:, -1]  # (B, T, 7) -> (B, 7), take last timestep (current state)
        # else: already (B, 7), use directly
        right_pos_norm = self.normalize_z(right_pose[:, :3], stats["right_pos_mean"], stats["right_pos_std"])
        right_quat = right_pose[:, 3:7]  # Keep quaternion as is

        # Right gripper open state (already 0/1, keep as is)
        # Handle both 3D (B, timesteps, 1) from training and 2D (B, 1) from evaluation
        right_gripper = observation['right_gripper_open']
        if right_gripper.dim() == 3:
            right_gripper = right_gripper[:, -1]  # (B, T, 1) -> (B, 1), take last timestep
        # else: already (B, 1), use directly

        # Left gripper pose
        left_pose = observation['left_gripper_pose']
        if left_pose.dim() == 3:
            left_pose = left_pose[:, -1]  # (B, T, 7) -> (B, 7), take last timestep
        left_pos_norm = self.normalize_z(left_pose[:, :3], stats["left_pos_mean"], stats["left_pos_std"])
        left_quat = left_pose[:, 3:7]  # Keep quaternion as is

        # Left gripper open state
        left_gripper = observation['left_gripper_open']
        if left_gripper.dim() == 3:
            left_gripper = left_gripper[:, -1]  # (B, T, 1) -> (B, 1), take last timestep

        # Concatenate: [right_pos(3), right_quat(4), right_gripper(1), left_pos(3), left_quat(4), left_gripper(1)]
        qpos = torch.cat([right_pos_norm, right_quat, right_gripper,
                          left_pos_norm, left_quat, left_gripper], dim=-1)

        return qpos
```

改成：

```python
    def preprocess_qpos(self, observation: dict):
        batch_size = observation['right_gripper_pose'].shape[0]
        task_ids = self._get_task_ids(observation.get('task_id'), batch_size)

        right_pose = observation['right_gripper_pose']
        if right_pose.dim() == 3:
            right_pose = right_pose[:, -1]
        right_pos_norm = self._normalize_pos_by_task(
            right_pose[:, :3], task_ids, "right_pos_mean", "right_pos_std"
        )
        right_quat = right_pose[:, 3:7]

        right_gripper = observation['right_gripper_open']
        if right_gripper.dim() == 3:
            right_gripper = right_gripper[:, -1]

        left_pose = observation['left_gripper_pose']
        if left_pose.dim() == 3:
            left_pose = left_pose[:, -1]
        left_pos_norm = self._normalize_pos_by_task(
            left_pose[:, :3], task_ids, "left_pos_mean", "left_pos_std"
        )
        left_quat = left_pose[:, 3:7]

        left_gripper = observation['left_gripper_open']
        if left_gripper.dim() == 3:
            left_gripper = left_gripper[:, -1]

        qpos = torch.cat([right_pos_norm, right_quat, right_gripper,
                          left_pos_norm, left_quat, left_gripper], dim=-1)

        return qpos, task_ids
```

### 注解

- 这里返回值从 `qpos` 变成了 `qpos, task_ids`，因为 `act()` 后面还要用当前任务的 stats 做反归一化。

### 3.5 修改 `preprocess_action()`

把当前版本：

```python
    def preprocess_action(self, replay_sample: dict):

        stats = self.train_stats()

        # Process previous (current) state: [right_pose(7), right_gripper(1), left_pose(7), left_gripper(1)]
        # Use [:, -1] to get the last (current) timestep, works for any prev_action_horizon
        right_prev_pose = replay_sample['right_prev_gripper_pose'][:, -1]  # (B, T, 7) -> (B, 7)
        right_prev_pos_norm = self.normalize_z(right_prev_pose[:, :3], stats["right_pos_mean"], stats["right_pos_std"])
        right_prev_quat = right_prev_pose[:, 3:7]
        right_prev_gripper = replay_sample['right_prev_gripper_open'][:, -1]  # (B, T, 1) -> (B, 1)

        left_prev_pose = replay_sample['left_prev_gripper_pose'][:, -1]  # (B, T, 7) -> (B, 7)
        left_prev_pos_norm = self.normalize_z(left_prev_pose[:, :3], stats["left_pos_mean"], stats["left_pos_std"])
        left_prev_quat = left_prev_pose[:, 3:7]
        left_prev_gripper = replay_sample['left_prev_gripper_open'][:, -1]  # (B, T, 1) -> (B, 1)

        qpos = torch.cat([right_prev_pos_norm, right_prev_quat, right_prev_gripper,
                          left_prev_pos_norm, left_prev_quat, left_prev_gripper], dim=-1)

        # Process action sequence: (B, T, 16)
        right_next_pose = replay_sample['right_next_gripper_pose']  # (B, T, 7)
        right_next_pos_norm = self.normalize_z(right_next_pose[:, :, :3], stats["right_pos_mean"], stats["right_pos_std"])
        right_next_quat = right_next_pose[:, :, 3:7]
        right_next_gripper = replay_sample['right_next_gripper_open']  # (B, T, 1)

        left_next_pose = replay_sample['left_next_gripper_pose']  # (B, T, 7)
        left_next_pos_norm = self.normalize_z(left_next_pose[:, :, :3], stats["left_pos_mean"], stats["left_pos_std"])
        left_next_quat = left_next_pose[:, :, 3:7]
        left_next_gripper = replay_sample['left_next_gripper_open']  # (B, T, 1)

        action_seq = torch.cat([right_next_pos_norm, right_next_quat, right_next_gripper,
                                left_next_pos_norm, left_next_quat, left_next_gripper], dim=-1)

        return qpos, action_seq
```

改成：

```python
    def preprocess_action(self, replay_sample: dict):
        batch_size = replay_sample['right_prev_gripper_pose'].shape[0]
        task_ids = self._get_task_ids(replay_sample.get('task_id'), batch_size)

        right_prev_pose = replay_sample['right_prev_gripper_pose'][:, -1]
        right_prev_pos_norm = self._normalize_pos_by_task(
            right_prev_pose[:, :3], task_ids, "right_pos_mean", "right_pos_std"
        )
        right_prev_quat = right_prev_pose[:, 3:7]
        right_prev_gripper = replay_sample['right_prev_gripper_open'][:, -1]

        left_prev_pose = replay_sample['left_prev_gripper_pose'][:, -1]
        left_prev_pos_norm = self._normalize_pos_by_task(
            left_prev_pose[:, :3], task_ids, "left_pos_mean", "left_pos_std"
        )
        left_prev_quat = left_prev_pose[:, 3:7]
        left_prev_gripper = replay_sample['left_prev_gripper_open'][:, -1]

        qpos = torch.cat([right_prev_pos_norm, right_prev_quat, right_prev_gripper,
                          left_prev_pos_norm, left_prev_quat, left_prev_gripper], dim=-1)

        right_next_pose = replay_sample['right_next_gripper_pose']
        right_next_pos_norm = self._normalize_pos_by_task(
            right_next_pose[:, :, :3], task_ids, "right_pos_mean", "right_pos_std"
        )
        right_next_quat = right_next_pose[:, :, 3:7]
        right_next_gripper = replay_sample['right_next_gripper_open']

        left_next_pose = replay_sample['left_next_gripper_pose']
        left_next_pos_norm = self._normalize_pos_by_task(
            left_next_pose[:, :, :3], task_ids, "left_pos_mean", "left_pos_std"
        )
        left_next_quat = left_next_pose[:, :, 3:7]
        left_next_gripper = replay_sample['left_next_gripper_open']

        action_seq = torch.cat([right_next_pos_norm, right_next_quat, right_next_gripper,
                                left_next_pos_norm, left_next_quat, left_next_gripper], dim=-1)

        return qpos, action_seq
```

### 注解

- 训练 batch 内可以混有多个任务。
- 所以这里不能只拿一份全局 stats；必须逐样本按 `task_id` 选 stats。

### 3.6 修改 `act()`

把当前版本里与 stats 和 `preprocess_qpos()` 相关的部分：

```python
        stats = self.train_stats()

        if self._timestep % query_freq == 0:
            with torch.no_grad():
                # preprocess input
                qpos = self.preprocess_qpos(observation)
                stacked_rgb, stacked_point_cloud = self.preprocess_images(observation)

                # forward pass
                self._all_actions = self._actor(qpos, stacked_rgb, actions=None, is_pad=None)
```

改成：

```python
        qpos, task_ids = self.preprocess_qpos(observation)
        stats = self._stats_for_task_id(int(task_ids[0].item()))

        if self._timestep % query_freq == 0:
            with torch.no_grad():
                stacked_rgb, stacked_point_cloud = self.preprocess_images(observation)
                self._all_actions = self._actor(
                    qpos, stacked_rgb, actions=None, is_pad=None
                )
```

后面的反归一化部分：

```python
        right_pos = self.unnormalize_z(raw_action[0:3], stats["right_pos_mean"], stats["right_pos_std"])
        ...
        left_pos = self.unnormalize_z(raw_action[8:11], stats["left_pos_mean"], stats["left_pos_std"])
```

保持不变即可，因为此时 `stats` 已经是“当前任务对应的 stats”。

### 注解

- 评估/推理时 batch size 是 1，所以这里直接取 `task_ids[0]` 即可。
- 现在 `act()` 输出动作时也会使用当前任务自己的反归一化统计量。

---

## 修改后的行为

改完之后，训练和评估会变成下面这样：

### 训练阶段

1. replay 仍然用字符串 `task` 做 task-uniform sampling；
2. 每个 sample 额外带一个整型 `task_id`；
3. `preprocess_action()` 对 batch 里的每条样本，按它自己的 `task_id` 找对应任务的 stats；
4. 因此同一个 batch 里混有 `pick_plate / pick_fork / edge_phone / pivot_phone` 也没问题。

### 评估阶段

1. env 返回的 observation 里新增数值型 `task_id`；
2. rollout generator 会把它安全地转成 tensor；
3. `preprocess_qpos()` 与 `act()` 用这个 `task_id` 选当前任务 stats；
4. 因此评估切换任务时，归一化也会自动切换。

---

## 这个方案为什么算“尽量简单”

它没有做下面这些更重的改法：

- 没有改 Hydra 配置结构；
- 没有新增额外 config 字段；
- 没有改 `train.py` / `run_seed_fn.py` / `eval.py`；
- 没有改 replay 采样策略；
- 没有把 task name 从字符串完全替换成整数；
- 没有重写 obs pipeline。

它只是：

1. 多带一个 `task_id`；
2. 让 agent 内部按 `task_id` 找 stats。

这是功能正确和改动成本之间最平衡的一种写法。

---

## 需要注意的兼容性问题

### 1. 旧 replay 目录最好不要复用

因为 replay schema 新增了 `task_id`，如果你还沿用旧的磁盘 replay 目录，可能会遇到旧样本字段不齐的问题。

建议二选一：

1. 换一个新的 `replay.path`；
2. 或删除这次实验对应的 replay 子目录后重跑。

结合当前代码，最安全的是清掉这次实验对应目录，例如：

```bash
/home/hdliu/arm/replay/multi/ACT_BC_VISION/seedX
```

### 2. 旧 checkpoint 不建议直接接着训

原因不是“能不能加载”，而是：

- 旧模型训练时用的是“第一个任务的全局 stats”；
- 新方案用的是“按任务分别 stats”；
- 二者数据分布假设不同。

因此更建议：

1. 新开一个 `logdir`；
2. `load_existing_weights: False`；
3. 从头训练。

### 3. 这个问题在别的 ACT 变体里也存在

同样的单任务 stats 写法也出现在：

- `ACT_BC_ENC`
- `ACT_BC_KEY`
- `ACT_BC_ENC_KEYPOINT`
- `ACT_BC_ENC_STRATEGY`
- `ACT_BC_ENC_KEYPOINT_STRATEGY`

如果你之后也想把这些方法做成真正的多任务 task-wise normalization，可以用同一套路改。

---

## 建议的落地顺序

推荐按这个顺序改：

1. 先改 [`launch_utils.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/launch_utils.py)
2. 再改 [`custom_rlbench_env.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py)
3. 最后改 [`act_bc_vision_agent.py`](/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py)

原因：

1. 先把 `task_id` 数据链路接通；
2. 再让 agent 消费这个 `task_id`；
3. 这样排查时最清楚。

---

## 修改后建议做的最小验证

在 `ppi` 环境里做：

```bash
conda activate ppi
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
python -m py_compile agents/act_bc_vision/launch_utils.py \
                    agents/act_bc_vision/act_bc_vision_agent.py \
                    helpers/custom_rlbench_env.py
```

然后建议先做一个小规模 smoke run：

1. `rlbench.tasks` 配成 2 个任务先试；
2. `framework.training_iterations` 暂时设小；
3. `load_existing_weights: False`；
4. 看训练是否能正常过 `update()`；
5. 看评估时切换任务是否还能正常 `act()`。

---

## 最后一句

如果只看“把多任务 replay 改成按各自任务 stats 归一化”这个目标，以上方案已经足够，而且是当前代码结构下最省事的一条路。

核心就是一句话：

**保留字符串 `task` 给 replay 采样，新增整数 `task_id` 给 agent 做 task-wise stats 选择。**
