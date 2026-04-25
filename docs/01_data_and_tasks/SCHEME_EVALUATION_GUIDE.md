# 双臂角色分配（Scheme）分层评估方案

## 1. 背景与目标

### 1.1 问题背景

在双臂协作任务中，`ArmRoleSelector` 根据场景配置（物体位置、手臂初始姿态等）智能选择最优的双臂角色分配方案（scheme）：
- `left_grasper`：左手执行抓取，右手执行推动
- `right_grasper`：右手执行抓取，左手执行推动

数据收集时，每个 episode 的 scheme 信息已保存在 `scheme_info_{scheme}.pkl` 文件中，如：
```
/mnt/rlbench_data/bimanual_edge_phone/all_variations/episodes/episode0/scheme_info_left_grasper.pkl
```

### 1.2 评估目标

在模型评估时，按 GT scheme 分层统计成功率，以凸显模型的**双臂角色分配灵活性**：

| 指标 | 含义 |
|------|------|
| `success_rate_left_grasper_scenes` | 在 GT=left_grasper 场景中的成功率 |
| `success_rate_right_grasper_scenes` | 在 GT=right_grasper 场景中的成功率 |
| `scheme_balance_gap` | 两者差值的绝对值（越小越"灵活"） |

**预期效果**：
- 您的模型：两种场景中成功率都高，balance_gap 小
- 固定角色基线：某一类场景成功率低，balance_gap 大

---

## 2. 关键发现：评估时的场景来源

### 2.1 数据目录结构

```
/mnt/rlbench_data/
├── bimanual_edge_phone/           # 【评估用】不带 .train 后缀
│   └── all_variations/episodes/
│       ├── episode0/
│       │   ├── low_dim_obs.pkl
│       │   ├── scheme_info_left_grasper.pkl   # ← GT scheme 在文件名中
│       │   └── ...
│       ├── episode1/
│       └── ...
├── bimanual_edge_phone.train/     # 【训练用】带 .train 后缀
│   └── all_variations/episodes/
│       └── ...
└── ...
```

**重要**：
- **评估时使用不带 `.train` 后缀的目录**
- 训练数据和评估数据是分开的
- 本文档所有代码只处理不带后缀的评估数据目录

### 2.2 评估时加载演示数据的初始场景

评估时**不是**随机生成新场景，而是从演示数据加载初始状态：

```python
# repos/YARR/yarr/utils/rollout_generator.py:23-24
if eval:
    obs = env.reset_to_demo(eval_demo_seed)  # eval_demo_seed 就是 episode 编号
```

```python
# occ_grasp_models/helpers/custom_rlbench_env.py:391-410
def reset_to_demo(self, i, max_attempts=3):
    """
    参数 i: episode 编号，如 0, 1, 2, ...
    """
    ...
    # 从演示数据目录加载第 i 个 episode
    (d,) = self._task.get_demos(
        1, live_demos=False, random_selection=False, from_episode_number=i
    )
    # 恢复到该 demo 的初始场景状态
    _, obs = self._task.reset_to_demo(d)
```

```python
# repos/RLBench/rlbench/utils.py:56
# 加载路径: {dataset_root}/{task_name}/all_variations/episodes/
# 例如: /mnt/rlbench_data/bimanual_edge_phone/all_variations/episodes/episode{i}/
task_root = join(dataset_root, task_name)  # 不加 .train 后缀
```

### 2.3 重要结论

| 问题 | 答案 |
|------|------|
| 评估时场景是否随机？ | **否**，加载演示数据的初始状态 |
| 同一 episode 多次评估相同？ | **是**，完全相同的初始场景 |
| 不同模型公平对比？ | **是**，只要用相同的 episode 范围 |
| GT scheme 来源？ | **直接从演示数据文件名获取**，不需要实时计算 |

---

## 3. 评估代码架构详解

### 3.1 评估流程概览

```
eval.py
  │
  ├── 创建 IndependentEnvRunner
  │     num_eval_runs = len(tasks)  # 任务数量，单任务=1，多任务=N
  │     eval_episodes = 10          # 每个任务评估的 episode 数
  │
  └── env_runner.start(weight, ...)
        │
        ├── 创建环境 (CustomRLBenchEnv 或 CustomMultiTaskRLBenchEnv)
        │
        └── _internal_env_runner._run_eval_independent(...)
              │
              ├── for n_eval in range(num_eval_runs):    # 遍历每个任务
              │     │
              │     └── for ep in range(eval_episodes):  # 遍历每个 episode
              │           │
              │           ├── eval_demo_seed = ep + eval_from_eps_number
              │           ├── env.reset_to_demo(eval_demo_seed)
              │           ├── 模型执行动作
              │           └── 统计成功/失败
              │
              └── 输出 summaries 到 eval_data.csv
```

### 3.2 单任务 vs 多任务模式

| 属性 | 单任务模式 | 多任务模式 |
|------|-----------|-----------|
| 环境类 | `CustomRLBenchEnv` | `CustomMultiTaskRLBenchEnv` |
| 任务属性 | `env._task_class` (单个类) | `env._task_classes` (类列表) |
| 当前任务索引 | 不适用 | `env.active_task_id` |
| `num_eval_runs` | 1 | `len(tasks)` |
| 任务切换 | 无 | 每 `swap_task_every` 个 episode 切换 |

### 3.3 关键变量详解

```python
# _independent_env_runner.py 中的关键变量

# 1. num_eval_runs: 外层循环次数（= 任务数量）
#    - 单任务: num_eval_runs = 1
#    - 多任务: num_eval_runs = len(tasks), 如 4 个任务则为 4
for n_eval in range(self._num_eval_runs):
    # n_eval = 0, 1, 2, 3 分别对应 4 个任务

    # 2. eval_episodes: 每个任务评估的 episode 数量
    for ep in range(self._eval_episodes):
        # ep = 0, 1, 2, ..., 9 (如果 eval_episodes=10)

        # 3. eval_demo_seed: 实际加载的 episode 编号
        #    eval_from_eps_number 是起始偏移，通常为 0
        eval_demo_seed = ep + self._eval_from_eps_number
        # 示例: 如果 eval_from_eps_number=0, ep=5, 则 eval_demo_seed=5
        #       表示加载 episode5/ 目录

        # 4. 加载对应 episode 的初始场景
        env.reset_to_demo(eval_demo_seed)
```

### 3.4 `_get_task_name()` 方法详解

```python
# repos/YARR/yarr/runners/_independent_env_runner.py:97-110

def _get_task_name(self):
    """
    获取当前正在评估的任务名称。

    Returns:
        (task_name, is_multi_task): 任务名和是否为多任务模式

    示例返回值:
        单任务: ('bimanual_edge_phone', False)
        多任务: ('bimanual_edge_phone', True) 或 ('bimanual_pivot_phone', True)
    """
    if hasattr(self._eval_env, '_task_class'):
        # 单任务模式: env 有 _task_class 属性
        # _task_class 是任务类，如 <class 'BimanualEdgePhone'>
        eval_task_name = change_case(self._eval_env._task_class.__name__)
        # change_case('BimanualEdgePhone') → 'bimanual_edge_phone'
        multi_task = False

    elif hasattr(self._eval_env, '_task_classes'):
        # 多任务模式: env 有 _task_classes 属性（列表）
        # _task_classes = [BimanualEdgePhone, BimanualPivotPhone, ...]
        if self._eval_env.active_task_id != -1:
            # active_task_id 是当前激活任务的索引
            # 在 reset_to_demo 时会调用 _set_new_task() 更新此值
            task_id = (self._eval_env.active_task_id) % len(self._eval_env._task_classes)
            eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
        else:
            eval_task_name = ''  # 尚未开始评估，active_task_id 为初始值 -1
        multi_task = True
    else:
        raise Exception('Neither task_class nor task_classes found in eval env')

    return eval_task_name, multi_task
```

### 3.5 多任务模式下的任务切换机制

```python
# repos/YARR/yarr/envs/rlbench_env.py:270-276

def _set_new_task(self, shuffle=False):
    """
    切换到下一个任务。

    self._task_classes 示例:
        [<class BimanualEdgePhone>, <class BimanualPivotPhone>,
         <class BimanualPickPlate>, <class BimanualPickFork>]

    self._active_task_id 变化: -1 → 0 → 1 → 2 → 3 → 0 → 1 → ...
    """
    if shuffle:
        self._active_task_id = np.random.randint(0, len(self._task_classes))
    else:
        # 顺序切换: 0 → 1 → 2 → 3 → 0 → ...
        self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)

    task = self._task_classes[self._active_task_id]
    self._task = self._rlbench_env.get_task(task)
```

```python
# occ_grasp_models/helpers/custom_rlbench_env.py:685-689

def reset_to_demo(self, i, variation_number=-1):
    # 检查是否需要切换任务（每 swap_task_every 个 episode 切换一次）
    if self._episodes_this_task == self._swap_task_every:
        self._set_new_task()  # 切换到下一个任务
        self._episodes_this_task = 0
    self._episodes_this_task += 1
    # ... 加载 demo ...
```

---

## 4. 修改脚本概览

| 脚本 | 相对路径 | 修改类型 | 修改内容 |
|------|---------|---------|---------|
| scheme 工具模块 | `occ_grasp_models/helpers/scheme_utils.py` | **新增文件** | episode→scheme 映射构建工具 |
| 环境封装 | `occ_grasp_models/helpers/custom_rlbench_env.py` | **小修改** | 存储当前 episode 编号 |
| 评估执行器 | `repos/YARR/yarr/runners/_independent_env_runner.py` | **核心修改** | 添加 scheme 分层统计逻辑 |

**不需要修改的文件**：
- `eval.py` - 无需修改，dataset_root 已通过 env_config 传递
- `independent_env_runner.py` - 无需修改，env_config 传递链路已完整

---

## 5. 具体修改方案

### 5.1 新增 scheme_utils.py

**文件**：`/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/scheme_utils.py`

**作用**：提供构建 episode→scheme 映射的工具函数

```python
"""
Scheme Utilities for Bimanual Task Evaluation

This module provides utilities for building episode-to-scheme mappings
from demonstration data, enabling scheme-stratified evaluation metrics.

重要：本模块只处理【不带 .train 后缀】的评估数据目录。
"""

import os
import logging
from typing import Dict
from natsort import natsorted


def build_episode_scheme_map(dataset_root: str, task_name: str) -> Dict[int, str]:
    """
    构建 episode 编号到 scheme 的映射。

    通过扫描演示数据目录中的 scheme_info_*.pkl 文件名来获取每个 episode 的 GT scheme。

    Args:
        dataset_root: 数据集根目录
            示例: '/mnt/rlbench_data'
        task_name: 任务名称（不带 .train 后缀）
            示例: 'bimanual_edge_phone'

    Returns:
        Dict[int, str]: episode 编号到 scheme 的映射
            示例: {
                0: 'left_grasper',
                1: 'right_grasper',
                2: 'left_grasper',
                3: 'right_grasper',
                ...
            }

    Note:
        - 只扫描不带 .train 后缀的目录（评估数据）
        - 文件名格式: scheme_info_left_grasper.pkl 或 scheme_info_right_grasper.pkl
        - 如果某个 episode 没有 scheme_info 文件，该 episode 将被跳过
    """
    # 构建评估数据目录路径（不带 .train 后缀）
    # 示例: /mnt/rlbench_data/bimanual_edge_phone/all_variations/episodes/
    episodes_path = os.path.join(dataset_root, task_name, "all_variations", "episodes")

    if not os.path.exists(episodes_path):
        logging.warning(f"Episodes directory not found: {episodes_path}")
        return {}

    episode_scheme_map = {}

    try:
        # 获取所有 episode 目录，按自然顺序排序
        # 示例: ['episode0', 'episode1', 'episode10', 'episode11', ...]
        # natsorted 会正确排序为: ['episode0', 'episode1', ..., 'episode10', 'episode11', ...]
        episode_dirs = natsorted([
            d for d in os.listdir(episodes_path)
            if d.startswith('episode') and os.path.isdir(os.path.join(episodes_path, d))
        ])
    except Exception as e:
        logging.warning(f"Error listing episodes directory: {e}")
        return {}

    for ep_dir in episode_dirs:
        ep_path = os.path.join(episodes_path, ep_dir)

        # 提取 episode 编号
        # 'episode5' → 5
        try:
            ep_idx = int(ep_dir.replace('episode', ''))
        except ValueError:
            continue

        # 查找 scheme_info_*.pkl 文件
        try:
            for f in os.listdir(ep_path):
                if f.startswith('scheme_info_') and f.endswith('.pkl'):
                    # 'scheme_info_left_grasper.pkl' → 'left_grasper'
                    scheme = f.replace('scheme_info_', '').replace('.pkl', '')
                    episode_scheme_map[ep_idx] = scheme
                    break
        except Exception as e:
            logging.warning(f"Error reading episode {ep_idx}: {e}")
            continue

    if episode_scheme_map:
        # 统计 scheme 分布
        # 示例: {'left_grasper': 45, 'right_grasper': 35}
        scheme_counts = {}
        for scheme in episode_scheme_map.values():
            scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1
        logging.info(f"Built scheme map for {task_name}: {len(episode_scheme_map)} episodes, "
                     f"distribution: {scheme_counts}")
    else:
        logging.warning(f"No scheme information found for task {task_name}")

    return episode_scheme_map


def get_scheme_stats_summary(scheme_stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """
    计算 scheme 分层统计的汇总指标。

    Args:
        scheme_stats: scheme 统计字典
            示例: {
                'left_grasper': {'success': 10, 'total': 15},
                'right_grasper': {'success': 8, 'total': 12},
                'unknown': {'success': 0, 'total': 3}
            }

    Returns:
        Dict[str, float]: 汇总指标
            示例: {
                'success_rate_left_grasper_scenes': 0.667,    # 10/15
                'success_rate_right_grasper_scenes': 0.667,   # 8/12
                'scheme_balance_gap': 0.0,                    # |0.667 - 0.667|
                'total_left_grasper_episodes': 15,
                'total_right_grasper_episodes': 12
            }
    """
    summary = {}

    rates = {}
    for scheme in ['left_grasper', 'right_grasper']:
        stats = scheme_stats.get(scheme, {'success': 0, 'total': 0})
        total = stats['total']
        success = stats['success']

        rate = success / total if total > 0 else 0.0
        rates[scheme] = rate

        summary[f'success_rate_{scheme}_scenes'] = rate
        summary[f'total_{scheme}_episodes'] = total

    # 计算 balance gap: 两种场景成功率的差距
    summary['scheme_balance_gap'] = abs(rates.get('left_grasper', 0) - rates.get('right_grasper', 0))

    return summary
```

---

### 5.2 修改 custom_rlbench_env.py

**文件**：`/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py`

**修改目的**：在 `reset_to_demo` 时记录当前 episode 编号，便于后续查询 scheme

#### 5.2.1 修改位置1：CustomRLBenchEnv.__init__ 添加实例变量

**当前代码**（约第60-67行）：
```python
        self._i = 0
        self._error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self._last_exception = None
```

**修改后**：
```python
        self._i = 0
        self._error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self._last_exception = None

        # ===== 新增：记录当前 episode 编号用于 scheme 查询 =====
        # 在 reset_to_demo(i) 调用时更新
        # 示例: _current_episode_number = 5 表示当前评估 episode5/
        self._current_episode_number = -1
        # ==================================================
```

#### 5.2.2 修改位置2：CustomRLBenchEnv.reset_to_demo 记录 episode 编号

**当前代码**（约第391-395行）：
```python
    def reset_to_demo(self, i, max_attempts=3):
        self._i = 0
        # super(CustomRLBenchEnv, self).reset()

        for attempt in range(max_attempts):
```

**修改后**：
```python
    def reset_to_demo(self, i, max_attempts=3):
        self._i = 0

        # ===== 新增：记录当前 episode 编号 =====
        # 参数 i 就是 episode 编号，如 i=5 表示加载 episode5/
        self._current_episode_number = i
        # =====================================

        for attempt in range(max_attempts):
```

#### 5.2.3 修改位置3：在阶段评估接口之后添加 getter 方法

**位置**：在 `get_strategy_and_phase()` 方法之后（约第389行后）

**添加内容**：
```python
    # ============================================

    def get_current_episode_number(self) -> int:
        """
        获取当前正在评估的 episode 编号。

        Returns:
            int: 当前 episode 编号
                示例: 5 表示当前评估 episode5/
                如果尚未调用 reset_to_demo 则返回 -1

        Note:
            此方法用于 scheme 分层评估，通过 episode 编号查询对应的 GT scheme
        """
        return self._current_episode_number

    def reset_to_demo(self, i, max_attempts=3):
        # ... 现有代码 ...
```

#### 5.2.4 CustomMultiTaskRLBenchEnv 的对应修改

对 `CustomMultiTaskRLBenchEnv` 类进行相同的修改：

**位置1**：`__init__` 方法末尾（约第473行后）添加：
```python
        self._last_exception = None

        # ===== 新增：记录当前 episode 编号用于 scheme 查询 =====
        self._current_episode_number = -1
        # ==================================================
```

**位置2**：`reset_to_demo` 方法开头（约第685行后）添加：
```python
    def reset_to_demo(self, i, variation_number=-1):
        # ===== 新增：记录当前 episode 编号 =====
        self._current_episode_number = i
        # =====================================

        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            # ...
```

**位置3**：添加 `get_current_episode_number()` 方法（与单任务版本相同）

---

### 5.3 修改 _independent_env_runner.py（核心修改）

**文件**：`/home/hdliu/occ_grasp_fall/repos/YARR/yarr/runners/_independent_env_runner.py`

**修改目的**：在评估循环中添加 scheme 分层统计逻辑

#### 5.3.1 修改位置1：添加导入语句

**当前代码**（约第25-26行）：
```python
from yarr.runners._env_runner import _EnvRunner
import os
```

**修改后**：
```python
from yarr.runners._env_runner import _EnvRunner
import os

# ===== 新增：scheme 评估工具 =====
from helpers.scheme_utils import build_episode_scheme_map, get_scheme_stats_summary
# ================================
```

#### 5.3.2 修改位置2：在环境启动后初始化 scheme 映射

**当前代码**（约第136-138行）：
```python
        env = self._eval_env
        env.eval = eval
        env.launch()
```

**修改后**（详细注释版）：
```python
        env = self._eval_env
        env.eval = eval
        env.launch()

        # ===== 新增：初始化 scheme 映射（单任务模式） =====
        # episode_scheme_map 结构示例:
        # {
        #     0: 'left_grasper',    # episode0/ 的 GT scheme
        #     1: 'right_grasper',   # episode1/ 的 GT scheme
        #     2: 'left_grasper',    # episode2/ 的 GT scheme
        #     ...
        # }
        episode_scheme_map = {}
        dataset_root = None  # 保存 dataset_root 供后续多任务模式使用

        try:
            # 从环境获取 dataset_root
            # 访问路径: env._rlbench_env._dataset_root
            # 示例值: '/mnt/rlbench_data'
            if hasattr(env, '_rlbench_env') and hasattr(env._rlbench_env, '_dataset_root'):
                dataset_root = env._rlbench_env._dataset_root
            else:
                logging.warning("Cannot access dataset_root from environment")

            # 获取 task_name（单任务模式）
            if hasattr(env, '_task_class'):
                # 单任务模式: env._task_class 是任务类
                # 示例: env._task_class = <class 'BimanualEdgePhone'>
                from yarr.utils.process_str import change_case
                task_name = change_case(env._task_class.__name__)
                # change_case('BimanualEdgePhone') → 'bimanual_edge_phone'

                if dataset_root:
                    episode_scheme_map = build_episode_scheme_map(dataset_root, task_name)
                    logging.info(f"Scheme evaluation enabled: {len(episode_scheme_map)} episodes mapped")

            elif hasattr(env, '_task_classes'):
                # 多任务模式: env._task_classes 是任务类列表
                # scheme 映射将在每个任务的 n_eval 循环开始时构建
                logging.info("Multi-task mode: scheme map will be built per task in n_eval loop")

        except Exception as e:
            logging.warning(f"Failed to initialize scheme map: {e}")
        # ==================================
```

#### 5.3.3 修改位置3：在 n_eval 循环内初始化 scheme 统计

**当前代码**（约第175-186行）：
```python
        for n_eval in range(self._num_eval_runs):
            # ===== MODIFICATION: Add success/failed episode counters =====
            success_count = 0
            failed_count = 0
            failed_episodes = []  # List of (ep_idx, seed, error_msg)
            # =============================================================

            # ===== 新增：阶段级别评估统计 =====
            phase_success_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # 各阶段成功次数
            max_phases_reached = []  # 每个episode达到的最大阶段
            phase_completion_frames = {1: [], 2: [], 3: [], 4: []}  # 各阶段完成帧数
            # =================================
```

**修改后**：
```python
        for n_eval in range(self._num_eval_runs):
            # n_eval: 当前评估的任务索引
            # - 单任务模式: n_eval 始终为 0
            # - 多任务模式: n_eval = 0, 1, 2, ... 分别对应不同任务

            # ===== MODIFICATION: Add success/failed episode counters =====
            success_count = 0
            failed_count = 0
            failed_episodes = []  # List of (ep_idx, seed, error_msg)
            # =============================================================

            # ===== 新增：阶段级别评估统计 =====
            phase_success_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            max_phases_reached = []
            phase_completion_frames = {1: [], 2: [], 3: [], 4: []}
            # =================================

            # ===== 新增：scheme 分层评估统计 =====
            # scheme_stats 结构:
            # {
            #     'left_grasper': {'success': 0, 'total': 0},   # GT=left_grasper 的统计
            #     'right_grasper': {'success': 0, 'total': 0},  # GT=right_grasper 的统计
            #     'unknown': {'success': 0, 'total': 0}         # 无 scheme 信息的统计
            # }
            scheme_stats = {
                'left_grasper': {'success': 0, 'total': 0},
                'right_grasper': {'success': 0, 'total': 0},
                'unknown': {'success': 0, 'total': 0}
            }

            # 多任务模式下，为当前任务构建 scheme 映射
            if hasattr(env, '_task_classes') and dataset_root:
                # 获取当前任务名
                # 注意: 此时 active_task_id 可能还是 -1，需要先触发一次任务切换
                # 但我们可以通过 n_eval 索引直接获取任务名
                current_task_class = env._task_classes[n_eval % len(env._task_classes)]
                current_task_name = change_case(current_task_class.__name__)
                # 示例: n_eval=1, _task_classes=[EdgePhone, PivotPhone, ...]
                #       → current_task_class = PivotPhone
                #       → current_task_name = 'bimanual_pivot_phone'

                episode_scheme_map = build_episode_scheme_map(dataset_root, current_task_name)
                logging.info(f"Task {n_eval}: {current_task_name}, "
                             f"scheme map: {len(episode_scheme_map)} episodes")
            # ====================================
```

#### 5.3.4 修改位置4：在 episode 循环内获取并记录 GT scheme

**当前代码**（约第210-214行）：
```python
            # evaluate on N tasks * M episodes per task = total eval episodes
            for ep in range(self._eval_episodes):
                eval_demo_seed = ep + self._eval_from_eps_number
                logging.info('%s: Starting episode %d, number %d.' % (name, eval_demo_seed, ep))
                # the current task gets reset after every M episodes
                episode_rollout = []
```

**修改后**：
```python
            # evaluate on N tasks * M episodes per task = total eval episodes
            for ep in range(self._eval_episodes):
                # eval_demo_seed: 实际加载的 episode 编号
                # 示例: eval_from_eps_number=0, ep=5 → eval_demo_seed=5
                #       表示加载 episode5/ 目录
                eval_demo_seed = ep + self._eval_from_eps_number
                logging.info('%s: Starting episode %d, number %d.' % (name, eval_demo_seed, ep))

                # ===== 新增：获取当前 episode 的 GT scheme =====
                # 从预构建的映射中查询
                # 示例: eval_demo_seed=5, episode_scheme_map={5: 'left_grasper', ...}
                #       → current_gt_scheme = 'left_grasper'
                current_gt_scheme = episode_scheme_map.get(eval_demo_seed, 'unknown')
                scheme_stats[current_gt_scheme]['total'] += 1
                logging.info(f'Episode {eval_demo_seed}: GT scheme = {current_gt_scheme}')
                # =============================================

                # the current task gets reset after every M episodes
                episode_rollout = []
```

#### 5.3.5 修改位置5：在 episode 成功时更新 scheme 统计

**当前代码**（约第323-329行）：
```python
                            # Episode completed successfully
                            success_count += 1

                            # ===== Memory cleanup after each episode =====
                            self.stored_transitions[:] = []
                            episode_rollout.clear()
                            # ==============================================
```

**修改后**：
```python
                            # Episode completed successfully
                            success_count += 1

                            # ===== 新增：更新 scheme 统计（成功） =====
                            # reward > 0.99 表示任务成功完成
                            if reward > 0.99:
                                scheme_stats[current_gt_scheme]['success'] += 1
                                logging.info(f'Episode {eval_demo_seed}: SUCCESS '
                                             f'(GT scheme={current_gt_scheme})')
                            # =========================================

                            # ===== Memory cleanup after each episode =====
                            self.stored_transitions[:] = []
                            episode_rollout.clear()
                            # ==============================================
```

#### 5.3.6 修改位置6：在 summaries 中添加 scheme 评估指标

**当前代码**（约第386-398行）：
```python
            # ===== 新增：阶段级别评估指标 =====
            total_episodes = success_count + failed_count
            if total_episodes > 0:
                for phase_id in range(1, 5):
                    phase_rate = phase_success_counts[phase_id] / total_episodes
                    summaries.append(ScalarSummary(f'eval_envs/phase_{phase_id}_success_rate', phase_rate))

                if len(max_phases_reached) > 0:
                    avg_max_phase = sum(max_phases_reached) / len(max_phases_reached)
                    summaries.append(ScalarSummary('eval_envs/avg_max_phase', avg_max_phase))
            # ==================================

            eval_task_name, multi_task = self._get_task_name()
```

**修改后**：
```python
            # ===== 新增：阶段级别评估指标 =====
            total_episodes = success_count + failed_count
            if total_episodes > 0:
                for phase_id in range(1, 5):
                    phase_rate = phase_success_counts[phase_id] / total_episodes
                    summaries.append(ScalarSummary(f'eval_envs/phase_{phase_id}_success_rate', phase_rate))

                if len(max_phases_reached) > 0:
                    avg_max_phase = sum(max_phases_reached) / len(max_phases_reached)
                    summaries.append(ScalarSummary('eval_envs/avg_max_phase', avg_max_phase))
            # ==================================

            # ===== 新增：scheme 分层评估指标 =====
            # 计算汇总指标
            scheme_summary = get_scheme_stats_summary(scheme_stats)
            # scheme_summary 示例:
            # {
            #     'success_rate_left_grasper_scenes': 0.8,
            #     'success_rate_right_grasper_scenes': 0.7,
            #     'scheme_balance_gap': 0.1,
            #     'total_left_grasper_episodes': 40,
            #     'total_right_grasper_episodes': 40
            # }

            # 添加到 summaries（会写入 eval_data.csv）
            for metric_name, metric_value in scheme_summary.items():
                summaries.append(ScalarSummary(f'eval_envs/{metric_name}', metric_value))

            # 打印 scheme 评估摘要到日志
            logging.info(f"\n{'='*60}")
            logging.info("Scheme-Stratified Evaluation Summary:")
            logging.info(f"  left_grasper scenes:  "
                         f"{scheme_stats['left_grasper']['success']}/{scheme_stats['left_grasper']['total']} "
                         f"({scheme_summary['success_rate_left_grasper_scenes']*100:.1f}%)")
            logging.info(f"  right_grasper scenes: "
                         f"{scheme_stats['right_grasper']['success']}/{scheme_stats['right_grasper']['total']} "
                         f"({scheme_summary['success_rate_right_grasper_scenes']*100:.1f}%)")
            logging.info(f"  scheme_balance_gap:   {scheme_summary['scheme_balance_gap']*100:.1f}%")
            if scheme_stats['unknown']['total'] > 0:
                logging.warning(f"  unknown scheme:       {scheme_stats['unknown']['total']} episodes "
                                f"(missing scheme_info files)")
            logging.info(f"{'='*60}\n")
            # ====================================

            eval_task_name, multi_task = self._get_task_name()
```

---

## 6. 数据流图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. 数据收集阶段（已完成）                                                      │
│    ├── ArmRoleSelector.select_scheme() 选择最优 scheme                        │
│    └── 保存 scheme_info_{scheme}.pkl 到 episode 目录                          │
│                                                                              │
│    目录结构:                                                                  │
│    /mnt/rlbench_data/bimanual_edge_phone/     # 评估数据（不带 .train）       │
│    └── all_variations/episodes/                                              │
│        ├── episode0/scheme_info_left_grasper.pkl                             │
│        ├── episode1/scheme_info_right_grasper.pkl                            │
│        └── ...                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. 评估初始化阶段                                                             │
│                                                                              │
│    build_episode_scheme_map('/mnt/rlbench_data', 'bimanual_edge_phone')      │
│                           ↓                                                  │
│    扫描 scheme_info_*.pkl 文件名                                              │
│                           ↓                                                  │
│    episode_scheme_map = {0: 'left_grasper', 1: 'right_grasper', ...}         │
│                                                                              │
│    scheme_stats = {                                                          │
│        'left_grasper': {'success': 0, 'total': 0},                           │
│        'right_grasper': {'success': 0, 'total': 0},                          │
│        'unknown': {'success': 0, 'total': 0}                                 │
│    }                                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. 评估循环阶段                                                               │
│                                                                              │
│    for ep in range(eval_episodes):          # ep = 0, 1, 2, ..., 9          │
│        eval_demo_seed = ep + eval_from_eps_number   # = 0, 1, 2, ..., 9     │
│                                                                              │
│        # 查询 GT scheme                                                      │
│        current_gt_scheme = episode_scheme_map[eval_demo_seed]                │
│        # 示例: eval_demo_seed=5 → current_gt_scheme='left_grasper'           │
│                                                                              │
│        scheme_stats[current_gt_scheme]['total'] += 1                         │
│                                                                              │
│        env.reset_to_demo(eval_demo_seed)    # 加载 episode5/ 初始场景        │
│        模型执行动作序列                                                       │
│                                                                              │
│        if success:                                                           │
│            scheme_stats[current_gt_scheme]['success'] += 1                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. 结果汇总阶段                                                               │
│                                                                              │
│    get_scheme_stats_summary(scheme_stats)                                    │
│                           ↓                                                  │
│    {                                                                         │
│        'success_rate_left_grasper_scenes': 0.8,     # 32/40                  │
│        'success_rate_right_grasper_scenes': 0.7,    # 28/40                  │
│        'scheme_balance_gap': 0.1,                   # |0.8 - 0.7|            │
│        'total_left_grasper_episodes': 40,                                    │
│        'total_right_grasper_episodes': 40                                    │
│    }                                                                         │
│                           ↓                                                  │
│    输出到 eval_data.csv 和日志                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 输出指标说明

### 7.1 新增评估指标

| 指标名 | 计算方式 | 含义 |
|--------|---------|------|
| `success_rate_left_grasper_scenes` | left_success / left_total | GT=left_grasper 场景中的成功率 |
| `success_rate_right_grasper_scenes` | right_success / right_total | GT=right_grasper 场景中的成功率 |
| `scheme_balance_gap` | \|left_rate - right_rate\| | 两种场景成功率的差距（越小越"灵活"） |
| `total_left_grasper_episodes` | left_total | left_grasper 场景总数 |
| `total_right_grasper_episodes` | right_total | right_grasper 场景总数 |

### 7.2 日志输出示例

```
============================================================
Scheme-Stratified Evaluation Summary:
  left_grasper scenes:  32/40 (80.0%)
  right_grasper scenes: 28/40 (70.0%)
  scheme_balance_gap:   10.0%
============================================================
```

### 7.3 论文表格示例

| Model | Overall | left_grasper | right_grasper | Balance Gap |
|-------|---------|--------------|---------------|-------------|
| **Ours** | 75% | 80% | 70% | **10%** |
| Baseline A (fixed right) | 60% | 35% | 85% | 50% |
| Baseline B (no coordination) | 55% | 50% | 60% | 10% |

**解读**：
- Baseline A 固定使用右手抓取，在 left_grasper 场景中失败率高
- Baseline B 虽然 balance gap 小，但整体成功率低（无章法）
- Ours 在两种场景中都保持较高成功率，且 balance gap 小

---

## 8. 为何不破坏原有流程

### 8.1 新增代码完全独立

- `scheme_utils.py` 是独立的新文件，不修改任何现有模块
- 所有 scheme 相关统计逻辑是**新增代码块**，不修改原有统计逻辑
- 原有的 `success_count`、`phase_success_counts` 等统计**完全保留**

### 8.2 条件执行，优雅降级

```python
# 如果无法构建 scheme 映射，评估仍正常进行
episode_scheme_map = {}  # 空映射

# 查询时返回 'unknown'
current_gt_scheme = episode_scheme_map.get(eval_demo_seed, 'unknown')
# scheme 统计全部归入 'unknown'，不影响其他统计
```

### 8.3 不影响现有配置

- 不需要修改 `eval.yaml` 或其他配置文件
- 不需要修改 `eval.py` 入口脚本
- 数据路径通过现有的 `env_config` 传递链路获取

---

## 9. 实现验证清单

### 9.1 新增文件

| 文件 | 状态 |
|------|------|
| `occ_grasp_models/helpers/scheme_utils.py` | 待实现 |

### 9.2 修改文件

| 文件 | 修改位置 | 内容 | 状态 |
|------|---------|------|------|
| `custom_rlbench_env.py` | `__init__` (约第67行后) | 添加 `_current_episode_number = -1` | 待实现 |
| `custom_rlbench_env.py` | `reset_to_demo` (约第392行后) | 记录 `self._current_episode_number = i` | 待实现 |
| `custom_rlbench_env.py` | 第389行后 | 添加 `get_current_episode_number()` 方法 | 待实现 |
| `_independent_env_runner.py` | 第26行后 | 添加 scheme_utils 导入 | 待实现 |
| `_independent_env_runner.py` | 第138行后 | 初始化 scheme 映射和 dataset_root | 待实现 |
| `_independent_env_runner.py` | 第186行后 | 初始化 scheme_stats，多任务时更新映射 | 待实现 |
| `_independent_env_runner.py` | 第214行后 | 查询并记录 current_gt_scheme | 待实现 |
| `_independent_env_runner.py` | 第324行后 | 更新 scheme_stats['success'] | 待实现 |
| `_independent_env_runner.py` | 第396行后 | 添加 scheme 评估指标到 summaries | 待实现 |

---

## 10. 附加考虑：运动效率指标（可选）

除了成功率，还可以统计**成功案例的平均步数**，按 scheme 分层：

```python
# 扩展 scheme_stats 结构
scheme_stats = {
    'left_grasper': {'success': 0, 'total': 0, 'success_steps': []},
    'right_grasper': {'success': 0, 'total': 0, 'success_steps': []},
    'unknown': {'success': 0, 'total': 0, 'success_steps': []}
}

# 在成功时记录步数
if reward > 0.99:
    scheme_stats[current_gt_scheme]['success'] += 1
    scheme_stats[current_gt_scheme]['success_steps'].append(len(episode_rollout))

# 汇总时计算平均步数
for scheme in ['left_grasper', 'right_grasper']:
    steps = scheme_stats[scheme]['success_steps']
    if steps:
        avg_steps = sum(steps) / len(steps)
        summaries.append(ScalarSummary(f'eval_envs/avg_steps_{scheme}_scenes', avg_steps))
```

**意义**：正确的角色分配应该带来更高效的执行（更少的步数）。

---

**文档版本**: 1.1
**创建日期**: 2026-01-17
**作者**: Claude Code Assistant

**更新日志**:
- v1.1: 增加详细的代码架构解释、多任务模式分析、变量示例注释
- v1.0: 初始版本

**相关文档**:
- [DATA_COLLECTION_GUIDE.md](./DATA_COLLECTION_GUIDE.md) - 数据收集指南（第8部分：评估时的阶段判断）
- [STRATEGY_MODEL_DESIGN.md](../06_misc/act_family/STRATEGY_MODEL_DESIGN.md) - 策略模型设计文档
