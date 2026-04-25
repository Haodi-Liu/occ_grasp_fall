# DIFFUSION_POLICY：P1-2 与 P2-1 精确改法

本文档只保留两件事：

- `P1-2. 启用 AMP 和 TF32`
- `P2-1. 把 build_batch() 从主线程移出，做异步预取或多 worker 构 batch`

其它与这两项代码修改无关的内容已删除，避免混淆。

当前源码已经按本文档落地；本文档同步记录最终实现，并补充实现过程中发现的遗漏修正。


## 1. 总结结论

### 1.1 P1-2 最终建议

需要改：

- `occ_grasp_models/agents/diffusion_policy/agent.py`
- `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml`

不需要改：

- `occ_grasp_models/run_seed_fn.py`
- `repos/YARR/yarr/runners/offline_train_runner.py`
- `repos/YARR/yarr/replay_buffer/wrappers/pytorch_replay_buffer.py`

从训练效果出发，第一轮不建议联动修改原有超参：

- `lr`
- `weight_decay`
- `optimizer_betas`
- `lr_scheduler`
- `lr_warmup_steps`
- `use_ema`
- `grad_clip`
- `batch_size`
- `horizon`
- `n_obs_steps`
- `n_action_steps`
- `share_rgb_model`
- `down_dims`

原因：

- `P1-2` 的目标是提速，不是改训练语义
- A800 上优先 `bf16`
- 第一轮应尽量只引入 `AMP + TF32` 这一个变量

### 1.2 P2-1 最终建议

需要改：

- `occ_grasp_models/agents/diffusion_policy/replay_utils.py`
- `repos/YARR/yarr/replay_buffer/wrappers/pytorch_replay_buffer.py`
- `occ_grasp_models/run_seed_fn.py`
- `occ_grasp_models/agents/diffusion_policy/agent.py`
- `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml`

不需要改：

- `repos/YARR/yarr/runners/offline_train_runner.py`
- `occ_grasp_models/agents/diffusion_policy/launch_utils.py`
- `occ_grasp_models/agents/diffusion_policy/normalizer_utils.py`
- `occ_grasp_models/conf/config.yaml`

从训练效果出发，第一轮也不建议联动修改原有“效果型”配置：

- `lr`
- `weight_decay`
- `optimizer_betas`
- `lr_scheduler`
- `use_ema`
- `batch_size`
- `horizon`
- `n_obs_steps`
- `n_action_steps`
- `share_rgb_model`
- `down_dims`
- `crop_shape`
- `random_crop`

原因：

- `P2-1` 应该只改变“谁来构 batch”，不改变 batch 语义
- 只要 `IndexSequenceLoader.build_batch(...)` 本身不变，训练样本内容就应保持一致


## 2. 共享层影响到底在哪

这是你刚才追问的重点，这里单独说清楚。

### 2.1 两个概念要分开

`零触达`：

- 指共享文件本身完全不改

`零运行时差异`：

- 指虽然共享文件被改了，但当前仓库里现有其它 agent 的实际运行行为不变

这两个不是一回事。

### 2.2 本次 P2-1 最终方案里，真正会触碰的共享文件只剩 1 个

是：

- `repos/YARR/yarr/replay_buffer/wrappers/pytorch_replay_buffer.py`

不会再改：

- `repos/YARR/yarr/runners/offline_train_runner.py`

这是我 double check 之后收紧出来的结论。

### 2.3 为什么不再建议改 `offline_train_runner.py`

原因不是“改不了”，而是“没必要冒这个共享风险”。

当前共享 replay 里确实存在非 tensor 顶层字段，例如：

- `task`
- `lang_goal`

现有 `OfflineTrainRunner` 的行为是：

- 只把顶层 tensor 搬到 device
- 非 tensor 顶层字段直接丢掉

如果把它改成递归搬运嵌套 tensor，虽然可以写成兼容实现，但这已经不属于我愿意推荐的“最稳妥最小改动”方案。

所以更保守的最终方案是：

- worker 返回“扁平预构 batch”
- 主线程继续沿用现有 `OfflineTrainRunner`
- `DiffusionPolicyAgent.update()` 自己再把扁平字段组回 `{"obs": ..., "action": ..., "is_pad": ...}`

这样 `OfflineTrainRunner` 完全不动。

### 2.4 `PyTorchReplayBuffer` 到底“改了什么”，对其他 agent 到底有没有差异

当前仓库里，`PyTorchReplayBuffer` 的构造调用点只有一个：

- `occ_grasp_models/run_seed_fn.py`

当前其它 agent 仍然会走旧调用：

```python
wrapped_replay = PyTorchReplayBuffer(
    replay_buffer, num_workers=cfg.framework.num_workers
)
```

而 `PyTorchReplayBuffer` 的拟议改法是：

- 在原有 `num_workers` 基础上新增几个可选参数
- 这些新增参数都有默认值
- 不传时，行为保持旧路径

也就是说，这里要区分两层：

第一层，源码层面：

- 共享文件 `pytorch_replay_buffer.py` 被改了
- 所以它不是“零触达”

第二层，当前仓库其它 agent 的运行时层面：

- 其它 agent 仍然按原来的参数构造 `PyTorchReplayBuffer`
- 不会传 `sample_transform`
- 因此当前仓库里其它 agent 的运行时行为，预期没有差异

所以最准确的说法是：

- `PyTorchReplayBuffer` 这处改动不是“零触达”
- 但对当前仓库里现有其它 agent，预期是“零运行时差异”

这就是“我说低风险，但不是 100% 零触达”的确切含义。


## 3. P1-2：启用 AMP 和 TF32

### 3.1 要不要改 `DIFFUSION_POLICY.yaml`

要改。

而且建议显式写进 YAML，不要在代码里偷偷默认打开。

建议新增：

```yaml
# Mixed precision / math mode
use_amp: true
amp_dtype: bf16         # bf16 / fp16，A800 优先 bf16
enable_tf32: true
```

这三个键的含义：

- `use_amp`: 是否启用 autocast 混合精度
- `amp_dtype`: `bf16` 更稳，`fp16` 更通用
- `enable_tf32`: 是否允许 float32 matmul / cuDNN 走 TF32

### 3.2 `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml` 改法

旧代码：

```yaml
# Training
lr: 1e-4
weight_decay: 0.000001
# Align with original diffusion_policy UNet AdamW defaults.
optimizer_betas: [0.95, 0.999]
# LR scheduler (minimal parity path with diffusion_policy workspaces)
# Set to "none" to disable scheduler.
lr_scheduler: cosine
lr_warmup_steps: 500
# EMA over policy weights for eval/inference.
use_ema: True
ema_update_after_step: 0
ema_inv_gamma: 1.0
ema_power: 0.75
ema_min_value: 0.0
ema_max_value: 0.9999
# Transformer optimizer alignment (used when model_type=transformer)
transformer_weight_decay: 0.001
obs_encoder_weight_decay: 0.000001
grad_clip: 0
num_inference_steps: 100
```

新代码：

```yaml
# Training
lr: 1e-4
weight_decay: 0.000001
# Align with original diffusion_policy UNet AdamW defaults.
optimizer_betas: [0.95, 0.999]
# LR scheduler (minimal parity path with diffusion_policy workspaces)
# Set to "none" to disable scheduler.
lr_scheduler: cosine
lr_warmup_steps: 500
# EMA over policy weights for eval/inference.
use_ema: True
ema_update_after_step: 0
ema_inv_gamma: 1.0
ema_power: 0.75
ema_min_value: 0.0
ema_max_value: 0.9999
# Transformer optimizer alignment (used when model_type=transformer)
transformer_weight_decay: 0.001
obs_encoder_weight_decay: 0.000001
grad_clip: 0
num_inference_steps: 100

# Mixed precision / math mode
use_amp: true
amp_dtype: bf16
enable_tf32: true
```

### 3.3 `occ_grasp_models/agents/diffusion_policy/agent.py` 改法

#### 3.3.1 import 段

旧代码：

```python
import copy
import os
from typing import Dict, List

import numpy as np
import torch
from diffusers.optimization import get_scheduler as get_diffusers_scheduler
```

新代码：

```python
import copy
import os
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
from diffusers.optimization import get_scheduler as get_diffusers_scheduler
```

#### 3.3.2 `__init__()` 增加 AMP 状态

旧代码：

```python
        self._ema_model = None
        self._ema = None
        self._use_ema = bool(getattr(cfg.method, "use_ema", False))
        self._update_step = 0
        self.seq_loader = None
```

新代码：

```python
        self._ema_model = None
        self._ema = None
        self._use_ema = bool(getattr(cfg.method, "use_ema", False))
        self._use_amp = False
        self._amp_dtype = None
        self._grad_scaler = None
        self._update_step = 0
        self.seq_loader = None
```

#### 3.3.3 `build()` 里初始化 TF32 / AMP

旧代码：

```python
        self._device = device
        self.policy = self.policy.to(self._device)
        self.policy.train(training)
        self._lr_scheduler = None
        self._optimizer = None
        self._ema = None
        self._ema_model = None
        self._update_step = 0
```

新代码：

```python
        self._device = device
        self._use_amp = _should_use_amp(self.cfg, self._device)
        self._amp_dtype = _resolve_amp_dtype(self.cfg) if self._use_amp else None
        self._grad_scaler = None

        if self._device.type == "cuda":
            _configure_tf32(self.cfg)

        self.policy = self.policy.to(self._device)
        self.policy.train(training)
        self._lr_scheduler = None
        self._optimizer = None
        self._ema = None
        self._ema_model = None
        self._update_step = 0
```

并且在 `if training:` 分支里，优化器构建完成后加：

```python
            if self._use_amp and self._amp_dtype == torch.float16:
                self._grad_scaler = torch.amp.GradScaler(
                    device="cuda",
                    enabled=True,
                )
```

#### 3.3.4 `update()` 用 autocast + scaler

旧代码：

```python
    def update(self, step: int, replay_sample: dict) -> dict:
        del step
        if self._optimizer is None:
            raise RuntimeError("Agent optimizer is not initialized. build(training=True) first.")
        if self.seq_loader is None:
            self.seq_loader = IndexSequenceLoader(
                cfg=self.cfg, episode_index_path=self._episode_index_path
            )

        episode_ids, timesteps = _extract_index_from_replay_sample(replay_sample)
        batch_np = self.seq_loader.build_batch(episode_ids, timesteps)
        batch = _to_device(batch_np, self._device)

        self._optimizer.zero_grad(set_to_none=True)
        loss = self.policy.compute_loss(batch)
        loss.backward()

        grad_clip = float(getattr(self.cfg.method, "grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
        self._optimizer.step()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        if self._ema is not None:
            self._ema.step(self.policy)
        self._update_step += 1

        loss_v = float(loss.detach().cpu().item())
        if self._lr_scheduler is not None:
            lr_v = float(self._lr_scheduler.get_last_lr()[0])
        else:
            lr_v = float(self._optimizer.param_groups[0]["lr"])
        self._summaries = {"total_losses": loss_v, "loss": loss_v, "lr": lr_v}
        return dict(self._summaries)
```

新代码：

```python
    def update(self, step: int, replay_sample: dict) -> dict:
        del step
        if self._optimizer is None:
            raise RuntimeError("Agent optimizer is not initialized. build(training=True) first.")
        if self.seq_loader is None:
            self.seq_loader = IndexSequenceLoader(
                cfg=self.cfg, episode_index_path=self._episode_index_path
            )

        episode_ids, timesteps = _extract_index_from_replay_sample(replay_sample)
        batch_np = self.seq_loader.build_batch(episode_ids, timesteps)
        batch = _to_device(batch_np, self._device)

        self._optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self._device, self._use_amp, self._amp_dtype):
            loss = self.policy.compute_loss(batch)

        grad_clip = float(getattr(self.cfg.method, "grad_clip", 0.0))
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            if grad_clip > 0:
                self._grad_scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            self._optimizer.step()

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        if self._ema is not None:
            self._ema.step(self.policy)
        self._update_step += 1

        loss_v = float(loss.detach().float().cpu().item())
        if self._lr_scheduler is not None:
            lr_v = float(self._lr_scheduler.get_last_lr()[0])
        else:
            lr_v = float(self._optimizer.param_groups[0]["lr"])
        self._summaries = {"total_losses": loss_v, "loss": loss_v, "lr": lr_v}
        return dict(self._summaries)
```

#### 3.3.5 文件末尾新增 helper

```python
def _should_use_amp(cfg, device: torch.device) -> bool:
    return device.type == "cuda" and bool(getattr(cfg.method, "use_amp", False))


def _resolve_amp_dtype(cfg) -> torch.dtype:
    amp_dtype = str(getattr(cfg.method, "amp_dtype", "bf16")).lower()
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError(
        f"Unsupported method.amp_dtype='{amp_dtype}'. Expected 'bf16' or 'fp16'."
    )


def _configure_tf32(cfg) -> None:
    enable_tf32 = bool(getattr(cfg.method, "enable_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")


def _autocast_context(device: torch.device, enabled: bool, amp_dtype):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
```

#### 3.3.6 实现时补充修正

文档初版漏掉了两个为了“安全”和“可恢复训练”必须补上的点：

- 如果 `amp_dtype=fp16`，需要把 `GradScaler` 状态写入 `diffusion_policy_train_state.pt`，并在恢复训练时一并加载
- 如果 `amp_dtype=bf16`，`build()` 里应显式检查当前 CUDA 设备是否支持 bf16；不支持时直接报错并提示改成 `fp16`


## 4. P2-1：把 `build_batch()` 从主线程移到 worker

### 4.1 最终数据链路

最终建议不是让 worker 返回嵌套 batch，而是返回“扁平预构 batch”。

改完后链路是：

1. worker 从 replay 里取 `episode_id/timestep`
2. worker 内调用 `IndexSequenceLoader.build_batch(...)`
3. worker 把结果摊平成顶层字段：
   - `front_rgb`
   - `wrist_right_rgb`
   - `wrist_left_rgb`
   - `low_dim_state`
   - `action`
   - `is_pad`
   - 以及 replay 侧原本还会被 summary 使用的 `demo` / `timeout`
4. 主线程继续沿用现有 `OfflineTrainRunner`
5. `DiffusionPolicyAgent.update()` 再把这些顶层字段组回：
   - `{"obs": ..., "action": ..., "is_pad": ...}`

这样就不需要改共享 `OfflineTrainRunner`。

实现时补了两个初版文档漏写的兼容点：

- 为兼容当前 `PreprocessAgent.update()` 的“单任务 squeeze”逻辑，worker 返回的 `obs/action/is_pad` 需要额外带一个 singleton task 维
- 下层 `DiffusionPolicyAgent.update()` 需要同时兼容“带这个 task 维”和“不带这个 task 维”两种输入

### 4.2 `occ_grasp_models/conf/method/DIFFUSION_POLICY.yaml` 改法

建议新增：

```yaml
# Worker-side batch build
move_batch_build_to_worker: true
batch_builder_workers: 4
batch_builder_pin_memory: true
batch_builder_persistent_workers: true
batch_builder_prefetch_factor: 2
```

这组键只属于性能调优，不属于训练效果超参。

### 4.3 `occ_grasp_models/agents/diffusion_policy/replay_utils.py` 改法

#### 4.3.1 import 段

旧代码：

```python
import numpy as np
from PIL import Image
```

新代码：

```python
import numpy as np
import torch
from PIL import Image
```

#### 4.3.2 给 `IndexSequenceLoader` 增加从 replay sample 直接构 batch 的入口

新增代码：

```python
    def build_batch_from_replay_sample(
        self, replay_sample: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        episode_ids = _extract_index_vector(replay_sample["episode_id"])
        timesteps = _extract_index_vector(replay_sample["timestep"])
        return self.build_batch(episode_ids, timesteps)
```

#### 4.3.3 增加 worker 侧 transform

新增代码：

```python
class ReplaySampleToBatchTransform:
    def __init__(self, cfg, episode_index_path: Optional[str] = None):
        self._cfg = cfg
        self._episode_index_path = episode_index_path
        self._loader: Optional[IndexSequenceLoader] = None

    def __call__(self, replay_sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._loader is None:
            self._loader = IndexSequenceLoader(
                cfg=self._cfg, episode_index_path=self._episode_index_path
            )
        batch = self._loader.build_batch_from_replay_sample(replay_sample)
        flat_batch = {
            key: _add_single_task_dim(value) for key, value in batch["obs"].items()
        }
        flat_batch["action"] = _add_single_task_dim(batch["action"])
        flat_batch["is_pad"] = _add_single_task_dim(batch["is_pad"])
        for key in ("demo", "timeout", "sampling_probabilities"):
            if key in replay_sample:
                flat_batch[key] = replay_sample[key]
        return flat_batch
```

#### 4.3.4 增加 index 提取 helper

新增代码：

```python
def _extract_index_vector(index_array) -> np.ndarray:
    if torch.is_tensor(index_array):
        arr = index_array.detach().cpu().numpy()
    else:
        arr = np.asarray(index_array)
    if arr.ndim == 1:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 2:
        return arr[:, -1].astype(np.int64, copy=False)
    raise ValueError(f"Unexpected index tensor shape: {arr.shape}")
```

并额外补一个 helper：

```python
def _add_single_task_dim(array):
    if torch.is_tensor(array):
        return array.unsqueeze(1)
    return np.expand_dims(np.asarray(array), axis=1)
```

### 4.4 `repos/YARR/yarr/replay_buffer/wrappers/pytorch_replay_buffer.py` 改法

#### 4.4.1 `PyTorchIterableReplayDataset`

旧代码：

```python
class PyTorchIterableReplayDataset(IterableDataset):

    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer

    def _generator(self):
        while True:
            yield self._replay_buffer.sample_transition_batch(pack_in_dict=True)

    def __iter__(self):
        return iter(self._generator())
```

新代码：

```python
class PyTorchIterableReplayDataset(IterableDataset):

    def __init__(self, replay_buffer: ReplayBuffer, sample_transform=None):
        self._replay_buffer = replay_buffer
        self._sample_transform = sample_transform

    def _generator(self):
        while True:
            sample = self._replay_buffer.sample_transition_batch(pack_in_dict=True)
            if self._sample_transform is not None:
                sample = self._sample_transform(sample)
            yield sample

    def __iter__(self):
        return iter(self._generator())
```

#### 4.4.2 `PyTorchReplayBuffer`

旧代码：

```python
class PyTorchReplayBuffer(WrappedReplayBuffer):
    def __init__(self, replay_buffer: ReplayBuffer, num_workers: int = 2):
        super(PyTorchReplayBuffer, self).__init__(replay_buffer)
        self._num_workers = num_workers

    def dataset(self, batch_size=None, drop_last=False) -> DataLoader:
        d = PyTorchIterableReplayDataset(self._replay_buffer)

        return DataLoader(d, batch_size=batch_size,
                          drop_last=drop_last,
                          num_workers=self._num_workers, pin_memory=True)
```

新代码：

```python
class PyTorchReplayBuffer(WrappedReplayBuffer):
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        num_workers: int = 2,
        sample_transform=None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
    ):
        super(PyTorchReplayBuffer, self).__init__(replay_buffer)
        self._num_workers = num_workers
        self._sample_transform = sample_transform
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._prefetch_factor = prefetch_factor

    def dataset(self, batch_size=None, drop_last=False) -> DataLoader:
        d = PyTorchIterableReplayDataset(
            self._replay_buffer,
            sample_transform=self._sample_transform,
        )

        dataloader_kwargs = dict(
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        if self._num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self._persistent_workers
            if self._prefetch_factor is not None and int(self._prefetch_factor) > 0:
                dataloader_kwargs["prefetch_factor"] = int(self._prefetch_factor)

        return DataLoader(d, **dataloader_kwargs)
```

这处共享改动的关键点：

- 新增的是可选参数
- 旧调用不传这些参数时，当前仓库其它 agent 的运行时行为不变

### 4.5 `occ_grasp_models/run_seed_fn.py` 改法

先在 `agent = agent_factory.create_agent(cfg)` 之后增加：

```python
    wrapped_replay_kwargs = {"num_workers": int(cfg.framework.num_workers)}
```

然后把 DIFFUSION_POLICY 分支改成下面这样。

旧代码：

```python
    elif cfg.method.name == "DIFFUSION_POLICY":
        from agents import diffusion_policy
        from agents.diffusion_policy.normalizer_utils import (
            fit_normalizer_from_index_replay,
        )

        replay_buffer = diffusion_policy.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
        )

        episode_index_path = diffusion_policy.launch_utils.fill_multi_task_replay(
            cfg=cfg,
            obs_config=obs_config,
            rank=rank,
            replay=replay_buffer,
            tasks=tasks,
            num_demos=cfg.rlbench.demos,
        )

        if hasattr(agent, "set_episode_index_path"):
            agent.set_episode_index_path(episode_index_path)

        fitted_normalizer = fit_normalizer_from_index_replay(
            cfg=cfg,
            replay_buffer=replay_buffer,
            sample_size=int(getattr(cfg.method, "normalizer_sample_size", 10000)),
            device=f"cuda:{rank}",
            episode_index_path=episode_index_path,
        )
        if hasattr(agent, "policy"):
            agent.policy.set_normalizer(fitted_normalizer)
        elif hasattr(agent, "_pose_agent") and hasattr(agent._pose_agent, "policy"):
            agent._pose_agent.policy.set_normalizer(fitted_normalizer)

    ...

    wrapped_replay = PyTorchReplayBuffer(
        replay_buffer, num_workers=cfg.framework.num_workers
    )
```

新代码：

```python
    elif cfg.method.name == "DIFFUSION_POLICY":
        from agents import diffusion_policy
        from agents.diffusion_policy.normalizer_utils import (
            fit_normalizer_from_index_replay,
        )
        from agents.diffusion_policy.replay_utils import ReplaySampleToBatchTransform

        replay_buffer = diffusion_policy.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
        )

        episode_index_path = diffusion_policy.launch_utils.fill_multi_task_replay(
            cfg=cfg,
            obs_config=obs_config,
            rank=rank,
            replay=replay_buffer,
            tasks=tasks,
            num_demos=cfg.rlbench.demos,
        )

        if hasattr(agent, "set_episode_index_path"):
            agent.set_episode_index_path(episode_index_path)

        fitted_normalizer = fit_normalizer_from_index_replay(
            cfg=cfg,
            replay_buffer=replay_buffer,
            sample_size=int(getattr(cfg.method, "normalizer_sample_size", 10000)),
            device=f"cuda:{rank}",
            episode_index_path=episode_index_path,
        )
        if hasattr(agent, "policy"):
            agent.policy.set_normalizer(fitted_normalizer)
        elif hasattr(agent, "_pose_agent") and hasattr(agent._pose_agent, "policy"):
            agent._pose_agent.policy.set_normalizer(fitted_normalizer)

        if bool(getattr(cfg.method, "move_batch_build_to_worker", False)):
            wrapped_replay_kwargs = {
                "num_workers": int(
                    getattr(cfg.method, "batch_builder_workers", cfg.framework.num_workers)
                ),
                "sample_transform": ReplaySampleToBatchTransform(
                    cfg=cfg,
                    episode_index_path=episode_index_path,
                ),
                "pin_memory": bool(getattr(cfg.method, "batch_builder_pin_memory", True)),
                "persistent_workers": bool(
                    getattr(cfg.method, "batch_builder_persistent_workers", True)
                ),
                "prefetch_factor": int(
                    getattr(cfg.method, "batch_builder_prefetch_factor", 2)
                ),
            }

    ...

    wrapped_replay = PyTorchReplayBuffer(replay_buffer, **wrapped_replay_kwargs)
```

### 4.6 `occ_grasp_models/agents/diffusion_policy/agent.py` 最终版改法

这里是 `P1-2 + P2-1` 合并后的最终 `update()`。

旧代码：

```python
    def update(self, step: int, replay_sample: dict) -> dict:
        del step
        if self._optimizer is None:
            raise RuntimeError("Agent optimizer is not initialized. build(training=True) first.")
        if self.seq_loader is None:
            self.seq_loader = IndexSequenceLoader(
                cfg=self.cfg, episode_index_path=self._episode_index_path
            )

        episode_ids, timesteps = _extract_index_from_replay_sample(replay_sample)
        batch_np = self.seq_loader.build_batch(episode_ids, timesteps)
        batch = _to_device(batch_np, self._device)

        self._optimizer.zero_grad(set_to_none=True)
        loss = self.policy.compute_loss(batch)
        loss.backward()

        grad_clip = float(getattr(self.cfg.method, "grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
        self._optimizer.step()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        if self._ema is not None:
            self._ema.step(self.policy)
        self._update_step += 1

        loss_v = float(loss.detach().cpu().item())
        if self._lr_scheduler is not None:
            lr_v = float(self._lr_scheduler.get_last_lr()[0])
        else:
            lr_v = float(self._optimizer.param_groups[0]["lr"])
        self._summaries = {"total_losses": loss_v, "loss": loss_v, "lr": lr_v}
        return dict(self._summaries)
```

新代码：

```python
    def update(self, step: int, replay_sample: dict) -> dict:
        del step
        if self._optimizer is None:
            raise RuntimeError("Agent optimizer is not initialized. build(training=True) first.")

        if _is_prebuilt_flat_batch(replay_sample, self._obs_keys):
            batch = _build_training_batch_from_flat_sample(
                replay_sample,
                obs_keys=self._obs_keys,
            )
        else:
            if self.seq_loader is None:
                self.seq_loader = IndexSequenceLoader(
                    cfg=self.cfg, episode_index_path=self._episode_index_path
                )
            episode_ids, timesteps = _extract_index_from_replay_sample(replay_sample)
            batch_np = self.seq_loader.build_batch(episode_ids, timesteps)
            batch = batch_np

        batch = _to_device(batch, self._device)

        self._optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self._device, self._use_amp, self._amp_dtype):
            loss = self.policy.compute_loss(batch)

        grad_clip = float(getattr(self.cfg.method, "grad_clip", 0.0))
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            if grad_clip > 0:
                self._grad_scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            self._optimizer.step()

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        if self._ema is not None:
            self._ema.step(self.policy)
        self._update_step += 1

        loss_v = float(loss.detach().float().cpu().item())
        if self._lr_scheduler is not None:
            lr_v = float(self._lr_scheduler.get_last_lr()[0])
        else:
            lr_v = float(self._optimizer.param_groups[0]["lr"])
        self._summaries = {"total_losses": loss_v, "loss": loss_v, "lr": lr_v}
        return dict(self._summaries)
```

新增 helper：

```python
def _is_prebuilt_flat_batch(replay_sample: Dict, obs_keys: List[str]) -> bool:
    return (
        isinstance(replay_sample, dict)
        and "action" in replay_sample
        and "is_pad" in replay_sample
        and all(key in replay_sample for key in obs_keys)
    )


def _build_training_batch_from_flat_sample(
    replay_sample: Dict[str, torch.Tensor], obs_keys: List[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    obs_batch = {
        key: _strip_optional_single_task_dim(
            replay_sample[key],
            expected_ndim=(5 if key.endswith("_rgb") else 3),
        )
        for key in obs_keys
    }
    return {
        "obs": obs_batch,
        "action": _strip_optional_single_task_dim(replay_sample["action"], expected_ndim=3),
        "is_pad": _strip_optional_single_task_dim(replay_sample["is_pad"], expected_ndim=2),
    }
```

新增 helper：

```python
def _strip_optional_single_task_dim(value, expected_ndim: int):
    if torch.is_tensor(value):
        if value.ndim == expected_ndim + 1 and value.shape[1] == 1:
            return value[:, 0]
        return value

    arr = np.asarray(value)
    if arr.ndim == expected_ndim + 1 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr
```


## 5. 最终落地口径

### 5.1 `DIFFUSION_POLICY.yaml` 该不该改原有参数

结论：

- 需要新增 `P1-2` 和 `P2-1` 的开关参数
- 不需要为了这两项，联动修改原有训练效果超参

第一轮建议只新增：

```yaml
use_amp: true
amp_dtype: bf16
enable_tf32: true

move_batch_build_to_worker: true
batch_builder_workers: 4
batch_builder_pin_memory: true
batch_builder_persistent_workers: true
batch_builder_prefetch_factor: 2
```

第一轮不建议顺手改：

- `lr`
- `weight_decay`
- `optimizer_betas`
- `grad_clip`
- `batch_size`
- `share_rgb_model`
- `down_dims`

### 5.2 对其他 agent 的最终影响判断

最终判断分两句：

第一句，源码层面：

- 不是“零触达”
- 因为共享文件 `repos/YARR/yarr/replay_buffer/wrappers/pytorch_replay_buffer.py` 会被修改

第二句，当前仓库其它 agent 的运行时层面：

- 预期是“零运行时差异”
- 因为当前其它 agent 仍然走旧调用
- 不会传 `sample_transform`
- `OfflineTrainRunner` 也不改

所以最准确的工程口径是：

- `P2-1` 最终方案对其它 agent 的预期运行影响很低
- 严格说不是 100% 零触达
- 但在当前仓库现有调用路径下，预期没有运行时差异
