# PPI 评估流程完整分析

## 调用链概览

```
eval_ppi.py:main()
  ├─> eval_seed()
  │     ├─> agent_factory.create_agent(cfg)  → PreprocessAgent(PPIAgent)
  │     └─> IndependentEnvRunner(agent, ...)
  │           └─> start(weight, ...)
  │                 ├─> CustomRLBenchEnv(action_mode, ...)  # 创建环境
  │                 ├─> env.launch()                         # 启动CoppeliaSim
  │                 ├─> agent.load_weights(weight_path)      # 加载模型权重
  │                 └─> _IndependentEnvRunner._run_eval_independent()
  │                       └─> RolloutGenerator.generator()   # 评估循环
  │                             ├─> env.reset_to_demo(seed)  # 重置到demo初始状态
  │                             └─> for step in range(episode_length):
  │                                   ├─> agent.act(obs)     # 模型推理
  │                                   └─> env.step(action)   # 环境执行
  └─> 保存评估结果到 eval_data.csv
```

---

## 1. eval_ppi.py:main() 详解

**文件**: `inference-for-rlbench2/eval_ppi.py`

```python
@hydra.main(config_name="eval_ppi", config_path="conf")
def main(eval_cfg: DictConfig) -> None:
    # 1. 设置路径
    logdir = os.path.join(
        eval_cfg.framework.logdir,
        eval_cfg.rlbench.task_name,
        eval_cfg.method.name,
        "seed%d" % start_seed,
    )
    weightsdir = eval_cfg.framework.weightsdir

    # 2. 配置 action mode
    gripper_mode = eval(eval_cfg.rlbench.gripper_mode)()      # e.g. BimanualDiscrete()
    arm_action_mode = eval(eval_cfg.rlbench.arm_action_mode)() # e.g. BimanualEndEffectorPoseViaPlanning()
    action_mode = eval(eval_cfg.rlbench.action_mode)(arm_action_mode, gripper_mode)

    # 3. 创建观测配置
    obs_config = observation_utils.create_obs_config(
        eval_cfg.rlbench.cameras,
        eval_cfg.rlbench.camera_resolution,
        eval_cfg.method.name,
        eval_cfg.method.robot_name
    )

    # 4. 准备环境配置 (tuple)
    env_config = (
        task_classes[0],
        obs_config,
        action_mode,
        eval_cfg.rlbench.demo_path,
        eval_cfg.rlbench.episode_length,
        eval_cfg.rlbench.headless,
        eval_cfg.rlbench.include_lang_goal_in_obs,
        eval_cfg.rlbench.time_in_state,
        eval_cfg.framework.record_every_n,
        eval_cfg.rlbench.lang_path,
    )

    # 5. 调用评估
    eval_seed(eval_cfg, weightsdir, logdir, env_device, multi_task, start_seed, env_config)
```

**关键配置项**:
- `eval_cfg.rlbench.action_mode`: 决定动作空间格式 (通常是 `BimanualMoveArmThenGripper`)
- `eval_cfg.method.policy.n_obs_steps`: 观测历史步数
- `eval_cfg.rlbench.query_freq`: 模型推理频率
- `eval_cfg.method.policy.horizon_keyframe`: 关键帧数量
- `eval_cfg.method.policy.horizon_continuous`: 连续步数

---

## 2. eval_seed() 详解

```python
def eval_seed(eval_cfg, weightsdir, logdir, env_device, multi_task, seed, env_config):
    # 1. 创建 RolloutGenerator
    rg = RolloutGenerator()

    # 2. 创建 Agent (通过 agent_factory)
    agent = agent_factory.create_agent(eval_cfg)
    # 返回: PreprocessAgent(PPIAgent)

    # 3. 创建 IndependentEnvRunner
    env_runner = IndependentEnvRunner(
        agent=agent,
        episode_length=eval_cfg.rlbench.episode_length,
        rollout_generator=rg,
        time_steps=eval_cfg.method.policy.n_obs_steps,  # 关键：观测历史步数
        ...
    )

    # 4. 启动评估进程
    for weight in weight_folders:
        env_runner.start(
            weight,
            save_load_lock,
            writer_lock,
            env_config,
            e_idx % torch.cuda.device_count(),
            ...
        )
```

---

## 3. agent_factory.create_agent() 详解

**文件**: `inference-for-rlbench2/agents/ppi/launch_utils.py`

```python
def create_agent(cfg: DictConfig):
    # 1. 创建 PPI 策略网络
    actor_net = PPI(
        noise_scheduler_cfg=cfg.method.policy.noise_scheduler_cfg,
        horizon_keyframe=cfg.method.policy.horizon_keyframe,      # 关键帧数
        horizon_continuous=cfg.method.policy.horizon_continuous,  # 连续步数
        n_action_steps=cfg.method.policy.n_action_steps,
        n_obs_steps=cfg.method.policy.n_obs_steps,
        num_inference_steps=cfg.method.policy.num_inference_steps,  # 扩散步数
        encoder_output_dim=cfg.method.policy.encoder_output_dim,
        use_lang=cfg.method.policy.use_lang,
        pointcloud_encoder_cfg=cfg.method.policy.pointcloud_encoder_cfg,
        what_condition=cfg.method.policy.what_condition,  # 'ppi' 或其他模式
        predict_point_flow=cfg.method.policy.predict_point_flow,
    )

    # 2. 创建 PPIAgent
    ppi_agent = PPIAgent(
        actor_network=actor_net,
        cameras=cfg.rlbench.cameras,
        task_name=cfg.rlbench.tasks[0],
        fps_num=cfg.method.policy.fps_num,              # 点云采样数量
        cameras_pcd=cfg.rlbench.cameras_pcd,
        use_pc_color=cfg.method.policy.use_pc_color,
        bounding_box=cfg.method.policy.bounding_box,
        prediction_type=cfg.method.policy.prediction_type,  # 'continuous', 'keyframe', 'keyframe_continuous'
        query_freq=cfg.rlbench.query_freq,
        horizon_continuous=cfg.method.policy.horizon_continuous,
        horizon_keyframe=cfg.method.policy.horizon_keyframe,
        predict_point_flow=cfg.method.policy.predict_point_flow,
        pointflow_num=cfg.method.policy.pointflow_num,
        text_prompt=cfg.method.policy.text_prompt,      # 用于 Grounding DINO
        prompt_type=cfg.method.policy.prompt_type,      # 'point' 或 'box'
        sample_type=cfg.method.policy.sample_type,      # 'fps' 或 'rps'
        sam_cameras=cfg.method.policy.sam_cameras,
        ...
    )

    # 3. 包装成 PreprocessAgent (不进行 RGB 归一化)
    return PreprocessAgent(pose_agent=ppi_agent, norm_rgb=False)
```

---

## 4. PreprocessAgent 详解

**文件**: `inference-for-rlbench2/helpers/preprocess_agent.py`

### 4.1 RGB 归一化

```python
def _norm_rgb_(self, x):
    if self._norm_type == 'zero_mean':
        return (x.float() / 255.0) * 2.0 - 1.0  # [-1, 1]
    elif self._norm_type == 'imagenet':
        return (x.float() / 255.0)  # [0, 1] (未使用 ImageNet 标准化)
```

**PPI 配置**: `norm_rgb=False`，即不在 PreprocessAgent 层进行归一化。

### 4.2 act() 方法

```python
def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
    for k, v in observation.items():
        if self._norm_rgb and "rgb" in k:
            observation[k] = self._norm_rgb_(v)  # PPI 中不执行此步
        elif k == 'lang_goal':
            observation[k] = v  # 保持不变
        else:
            observation[k] = v.float()  # 转为 float

    # 调用内部 PPIAgent
    act_res = self._pose_agent.act(step, observation, deterministic)
    return act_res
```

---

## 5. IndependentEnvRunner.start() 详解

**文件**: `repos/YARR/yarr/runners/independent_env_runner.py`

```python
def start(self, weight, save_load_lock, writer_lock, env_config, device_idx, ...):
    # 1. 创建环境
    eval_env = CustomRLBenchEnv(
        task_class=env_config[0],
        observation_config=env_config[1],
        action_mode=env_config[2],  # 关键：action_mode
        dataset_root=env_config[3],
        episode_length=env_config[4],
        ...
    )

    # 2. 创建内部 runner
    self._internal_env_runner = _IndependentEnvRunner(
        eval_env=eval_env,
        agent=self._agent,
        timesteps=self._timesteps,  # 传递 n_obs_steps
        ...
    )

    # 3. 运行评估
    self._internal_env_runner._run_eval_independent(...)
```

---

## 6. _IndependentEnvRunner._run_eval_independent() 详解

**文件**: `repos/YARR/yarr/runners/_independent_env_runner.py`

```python
def _run_eval_independent(self, name, stats_accumulator, weight, ...):
    # 1. 构建 agent (设置设备)
    self._agent.build(training=False, device=device)

    # 2. 启动环境
    env.launch()

    # 3. 加载权重
    self._agent.load_weights(weight_path)

    # 4. 评估循环
    for n_eval in range(self._num_eval_runs):
        for ep in range(self._eval_episodes):
            eval_demo_seed = ep + self._eval_from_eps_number

            # 使用 RolloutGenerator 生成轨迹
            generator = self._rollout_generator.generator(
                self._step_signal, env, self._agent,
                self._episode_length, self._timesteps,  # 传递 n_obs_steps
                eval=True, eval_demo_seed=eval_demo_seed
            )

            for replay_transition in generator:
                episode_rollout.append(replay_transition)
```

---

## 7. CustomRLBenchEnv 详解

**文件**: `inference-for-rlbench2/helpers/custom_rlbench_env.py`

### 7.1 extract_obs_bimanual()

```python
def extract_obs_bimanual(self, obs: BimanualObservation, t=None, prev_action=None):
    # 清除不需要的数据
    obs.right.joint_velocities = None
    obs.right.gripper_pose = None
    obs.right.gripper_matrix = None
    obs.right.joint_positions = None
    # 同样处理 left

    # clip gripper positions
    obs.right.gripper_joint_positions = np.clip(obs.right.gripper_joint_positions, 0.0, 0.04)
    obs.left.gripper_joint_positions = np.clip(obs.left.gripper_joint_positions, 0.0, 0.04)

    # 调用父类提取基本观测
    obs_dict = super().extract_obs(obs)

    # 恢复并添加关节位置和姿态信息
    obs_dict['right_joint_positions'] = obs.right.joint_positions        # shape: (7,)
    obs_dict['right_gripper_joint_positions'] = obs.right.gripper_joint_positions  # shape: (2,)
    obs_dict['right_gripper_pose'] = obs.right.gripper_pose              # shape: (7,) [xyz, quat]
    obs_dict['right_gripper_open'] = np.array([obs.right.gripper_open]) # shape: (1,)
    obs_dict['left_joint_positions'] = obs.left.joint_positions          # shape: (7,)
    obs_dict['left_gripper_joint_positions'] = obs.left.gripper_joint_positions    # shape: (2,)
    obs_dict['left_gripper_pose'] = obs.left.gripper_pose                # shape: (7,)
    obs_dict['left_gripper_open'] = np.array([obs.left.gripper_open])   # shape: (1,)

    return obs_dict
```

### 7.2 reset_to_demo()

```python
def reset_to_demo(self, i):
    self._i = 0

    # 获取指定 episode 的 demo
    (d,) = self._task.get_demos(1, live_demos=False, random_selection=False,
                                 from_episode_number=i)

    # 重置到 demo 初始状态
    _, obs = self._task.reset_to_demo(d)
    self._lang_goal = self._task.get_task_descriptions()[0]

    # 提取观测
    self._previous_obs_dict = self.extract_obs(obs)

    return self._previous_obs_dict
```

### 7.3 step()

```python
def step(self, act_result: ActResult) -> Transition:
    action = act_result.action          # 从 agent 获取动作
    visual_targets = act_result.visual_targets  # PPI 特有：可视化目标

    try:
        obs, reward, terminal = self._task.step(action, visual_targets)
        obs = self.extract_obs(obs)
    except (IKError, ConfigurationPathError, InvalidActionError):
        terminal = True
        reward = 0.0

    return Transition(obs, reward, terminal, ...)
```

---

## 8. RolloutGenerator.generator() 详解

**文件**: `repos/YARR/yarr/utils/rollout_generator.py`

```python
def generator(self, step_signal, env, agent, episode_length, timesteps, eval, eval_demo_seed):
    # 1. 重置环境
    if eval:
        obs = env.reset_to_demo(eval_demo_seed)  # 重置到 demo 初始状态
    else:
        obs = env.reset()

    # 2. 重置 agent
    agent.reset()

    # 3. 构建观测历史 (关键！)
    lang_goal = obs['lang_goal']
    obs.pop('lang_goal', None)
    obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
    # obs_history[k] 是一个长度为 timesteps 的 list

    # 4. 评估循环
    for step in range(episode_length):
        # 准备输入数据
        prepped_data = {k: torch.tensor(np.array(v)[None], device=device)
                        for k, v in obs_history.items()}
        prepped_data['lang_goal'] = lang_goal

        # 调用 agent.act()
        act_result = agent.act(step_signal.value, prepped_data, deterministic=eval)

        # 执行动作
        transition = env.step(act_result)

        # 更新观测历史 (滑动窗口)
        for k in obs_history.keys():
            obs_history[k].append(transition.observation[k])
            obs_history[k].pop(0)

        yield replay_transition
```

### 数据维度转换

```python
# 环境输出
obs['right_gripper_pose'] shape: (7,)  # [x, y, z, qw, qx, qy, qz]
obs['right_gripper_open'] shape: (1,)

# obs_history 构建
obs_history['right_gripper_pose'] = [np.array(obs['right_gripper_pose'])] * timesteps
# 结果: list of timesteps 个 (7,) arrays

# prepped_data 构建
np.array(obs_history['right_gripper_pose']) shape: (timesteps, 7)
torch.tensor(...)[None] shape: (1, timesteps, 7)  # 添加 batch 维度

# 最终
prepped_data['right_gripper_pose'] shape: (1, timesteps, 7)
```

---

## 9. PPIAgent.act() 详解

**文件**: `inference-for-rlbench2/agents/ppi/ppi_agent.py`

### 9.1 初始化和特征提取

```python
def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
    query_freq = self.query_freq  # 推理频率

    # 1. 提取当前状态
    left_gripper_pose = observation['left_gripper_pose']    # (1, T, 7)
    right_gripper_pose = observation['right_gripper_pose']  # (1, T, 7)
    left_gripper_open = observation['left_gripper_open']    # (1, T, 1)
    right_gripper_open = observation['right_gripper_open']  # (1, T, 1)

    agent_pos = torch.cat(
        [left_gripper_pose, right_gripper_pose,
         left_gripper_open, right_gripper_open],
        dim=-1
    )  # shape: (1, T, 16)

    # 2. 点云预处理
    point_cloud = self.preprocess_pcd(observation)
    # point_cloud shape: (1, T, fps_num, 3 or 6)
    # 使用 FPS 或 RPS 采样，限制在 bounding_box 内

    # 3. 语言目标
    if self.use_lang:
        if self._timestep == 0:
            self.lang_goal = observation.get('lang_goal', None)

    # 4. 初始化 point flow (用于 Grounding DINO + SAM)
    if self._timestep == 0 or self._timestep == 1:
        self.initial_pointflow = self.get_initial_pointflow(self.pointflow_num, observation)
        # 使用 Grounding DINO 检测目标 -> SAM 生成 mask -> 提取点云
```

### 9.2 DINOv2 特征提取

```python
    # 5. 提取深度、相机参数
    depth = self.preprocess_depth(observation)
    extrinsics, intrinsics = self.preprocess_extrinsics_intrinsics(observation)
    color = self.preprocess_images(observation)

    # 6. 使用 DINOv2 提取语义特征
    dino_feature = self.get_dino_feature(
        point_cloud, color, depth, extrinsics, intrinsics
    )
    # dino_feature shape: (B, T, fps_num, 384)
```

### 9.3 模型推理

```python
    # 7. 准备输入
    useful_obs = {
        'point_cloud': point_cloud,          # (1, T, fps_num, 3)
        'agent_pos': agent_pos,              # (1, T, 16)
        'dino_feature': dino_feature,        # (1, T, fps_num, 384)
        'initial_point_flow': self.initial_pointflow,  # (1, 1, pointflow_num, 3)
        'lang': text_embedding               # (1, 1, 512)
    }

    # 8. 模型推理 (按 query_freq 频率)
    if self._timestep % query_freq == 0 or self._timestep == 1:
        with torch.no_grad():
            result_dict = self._actor.predict_action(useful_obs)
            self.result_action = result_dict['action']  # (1, horizon, 16)
            if self.predict_point_flow:
                self.result_pointflow = result_dict['point_flow_pred']  # (1, keyframe*fps, 3)
```

### 9.4 动作选择和格式转换

```python
    # 9. 选择当前步的动作
    if self.prediction_type == "continuous":
        result = self.result_action[0, self.action_id + 1]
    elif self.prediction_type == "keyframe":
        result = self.result_action[0, self.action_id]
    elif self.prediction_type == "keyframe_continuous":
        result = self.result_action[0, self.action_id + 1]

    # 更新 action_id
    if self.action_id < self.jump_step * (query_freq - 1):
        self.action_id += self.jump_step
    else:
        self.action_id = 0

    # 10. 格式转换为环境所需格式
    # result 格式: [left_xyz(3), left_quat(4), right_xyz(3), right_quat(4), left_gripper(1), right_gripper(1)]

    # 归一化四元数
    quat1 = result[10:14] / result[10:14].norm(dim=-1, keepdim=True)  # right quat
    quat2 = result[3:7] / result[3:7].norm(dim=-1, keepdim=True)      # left quat

    # 转换为环境期望格式: BimanualMoveArmThenGripper 需要 18 维
    # [right_xyz(3), right_quat(4), right_gripper(1), right_ignore_collisions(1),
    #  left_xyz(3), left_quat(4), left_gripper(1), left_ignore_collisions(1)]
    raw_action = torch.cat([
        result[7:10],                                      # right_xyz
        quat1,                                             # right_quat
        torch.clamp(result[15].unsqueeze(-1), 0.0, 1.0),  # right_gripper
        torch.tensor([1,]).to(device=self._device),        # right_ignore_collisions
        result[0:3],                                       # left_xyz
        quat2,                                             # left_quat
        torch.clamp(result[14].unsqueeze(-1), 0.0, 1.0),  # left_gripper
        torch.tensor([1,]).to(device=self._device)         # left_ignore_collisions
    ], dim=-1)
    # raw_action shape: (18,)

    self._timestep += 1
    return ActResult(raw_action.detach().cpu().numpy(), visual_targets=self.visual_targets)
```

---

## 10. PPI 策略详解

**文件**: `ppi/policy/ppi.py`

### 10.1 模型架构

```python
class PPI(BasePolicy):
    def __init__(self, ...):
        # 1. 观测编码器
        obs_encoder = ObservationEncoder(
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_lang=use_lang,
            use_initial_pointflow=predict_point_flow
        )

        # 2. 噪声调度器
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )

        # 3. 扩散头 (根据 what_condition 选择)
        if what_condition == 'ppi':
            model = DiffusionHeadPPI(
                embedding_dim=encoder_output_dim,
                num_attn_heads=8,
                horizon_keyframe=horizon_keyframe,
                horizon_continuous=horizon_continuous
            )
```

### 10.2 推理过程

```python
def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # 1. 归一化输入
    nobs = self.normalizer.normalize(obs_dict)

    # 2. 编码观测
    if self.predict_point_flow:
        (context_coord, context_feat, lang_feat, state_feat,
         pn_coord, pn_feat, pointflow_feat, pointflow_coords) = self.obs_encoder(nobs)
        fixed_inputs = (context_coord, context_feat, lang_feat, state_feat,
                       pn_coord, pn_feat, pointflow_feat, pointflow_coords)

    # 3. 初始化条件
    cond_data = torch.zeros(size=(B, T, 16), device=device)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    cond_point_flow = torch.zeros(size=(B, keyframe*fps, 3), device=device)
    cond_mask_point_flow = torch.zeros_like(cond_point_flow, dtype=torch.bool)

    # 4. 扩散采样
    nsample, npoint_flow = self.conditional_sample_diffuser_actor(
        cond_data, cond_mask, fixed_inputs,
        condition_point_flow=cond_point_flow,
        condition_mask_point_flow=cond_mask_point_flow
    )

    # 5. 反归一化
    action_pred = self.normalizer['action'].unnormalize(nsample)
    point_flow_pred = self.normalizer['point_flow'].unnormalize(npoint_flow)

    return {
        'action': action_pred,
        'point_flow_pred': point_flow_pred
    }
```

### 10.3 扩散采样详解

```python
def conditional_sample_diffuser_actor(self, condition_data, condition_mask, fixed_inputs,
                                      condition_point_flow=None, condition_mask_point_flow=None):
    # 1. 设置时间步
    self.position_noise_scheduler.set_timesteps(self.num_inference_steps)
    self.rotation_noise_scheduler.set_timesteps(self.num_inference_steps)

    # 2. 初始化噪声
    noise = torch.randn(size=condition_data.shape, device=device)

    # 3. 添加噪声到初始轨迹
    trajectory_left = noise[..., :7]
    trajectory_right = noise[..., 7:14]

    # 4. 迭代去噪
    for t in self.position_noise_scheduler.timesteps:
        # 模型预测
        if self.predict_point_flow:
            out_left, out_right, out_point_flow = self.model(
                trajectory_left, trajectory_right, t, fixed_inputs
            )

        # 分别去噪 position 和 rotation
        pos_left = self.position_noise_scheduler.step(
            out_left[-1][..., :3], t, trajectory_left[..., :3]
        ).prev_sample
        rot_left = self.rotation_noise_scheduler.step(
            out_left[-1][..., 3:7], t, trajectory_left[..., 3:7]
        ).prev_sample

        pos_right = self.position_noise_scheduler.step(
            out_right[-1][..., :3], t, trajectory_right[..., :3]
        ).prev_sample
        rot_right = self.rotation_noise_scheduler.step(
            out_right[-1][..., 3:7], t, trajectory_right[..., 3:7]
        ).prev_sample

        # 更新轨迹
        trajectory_left = torch.cat((pos_left, rot_left), -1)
        trajectory_right = torch.cat((pos_right, rot_right), -1)

    # 5. 组合最终输出
    trajectory = torch.cat((
        pos_left, rot_left, pos_right, rot_right,
        out_left[-1][..., 7:8], out_right[-1][..., 7:8]
    ), -1)

    return trajectory, out_point_flow[-1]
```

---

## 11. Action Mode 详解

**文件**: `repos/RLBench/rlbench/action_modes/action_mode.py`

### BimanualMoveArmThenGripper

```python
class BimanualMoveArmThenGripper(ActionMode):
    def action(self, scene: Scene, action: np.ndarray):
        assert(len(action) == 18)  # 期望 18 维动作

        # 解析动作
        # action[0:3]   = right arm xyz
        # action[3:7]   = right arm quaternion
        # action[7]     = right gripper
        # action[8]     = right ignore_collisions
        # action[9:12]  = left arm xyz
        # action[12:16] = left arm quaternion
        # action[16]    = left gripper
        # action[17]    = left ignore_collisions

        right_arm_action = np.concatenate([action[0:3], action[3:7]], axis=0)  # 7维: xyz+quat
        left_arm_action = np.concatenate([action[9:12], action[12:16]], axis=0)  # 7维: xyz+quat
        arm_action = np.concatenate([right_arm_action, left_arm_action], axis=0)  # 14维

        ee_action = np.array([action[7], action[16]])  # 2维: gripper
        ignore_collisions = np.array([action[8], action[17]])  # 2维

        self.arm_action_mode.action_pre_step(scene, arm_action, ignore_collisions)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        ...
```

**PPI 模型输出格式与环境期望格式对比**:
```
PPI 模型输出: [left_xyz(3), left_quat(4), right_xyz(3), right_quat(4), left_gripper(1), right_gripper(1)]
               shape: (16,)

PPIAgent.act() 转换后: [right_xyz(3), right_quat(4), right_gripper(1), right_ignore_collisions(1),
                        left_xyz(3), left_quat(4), left_gripper(1), left_ignore_collisions(1)]
                       shape: (18,)

环境期望 (BimanualMoveArmThenGripper): 18 维
✓ 格式一致
```

---

## 12. 对比：训练时 vs 评估时数据格式

### 12.1 训练时

```python
# 数据加载 (来自 dataset)
batch['obs']['point_cloud']  # (B, T, fps_num, 3)
batch['obs']['dino_feature'] # (B, T, fps_num, 384)
batch['obs']['agent_pos']    # (B, T, 16)
batch['action']              # (B, horizon, 16)
batch['point_flow']          # (B, keyframe, fps, 3)
```

### 12.2 评估时

```python
# RolloutGenerator 处理后
prepped_data['point_cloud']   # (1, n_obs_steps, fps_num, 3)
prepped_data['dino_feature']  # (1, n_obs_steps, fps_num, 384)
prepped_data['agent_pos']     # (1, n_obs_steps, 16)

# PPIAgent 预处理后传入模型
useful_obs['point_cloud']     # (1, n_obs_steps, fps_num, 3)
useful_obs['dino_feature']    # (1, n_obs_steps, fps_num, 384)
useful_obs['agent_pos']       # (1, n_obs_steps, 16)
```

**格式对比**: ✓ 一致（都是 `(B, T, ...)` 格式）

---

## 13. PPI 特有的关键组件

### 13.1 Grounding DINO + SAM

```python
def get_initial_pointflow(self, fps, observation):
    # 1. 对每个相机
    for camera in self.sam_cameras:
        image = observation['%s_rgb' % camera][0, 0]

        # 2. 使用 Grounding DINO 检测目标
        point_coords = self.get_point_from_mask(image)

        # 3. 从深度图提取 3D 点云
        depth = observation[f'{camera}_depth'][0, 0, 0]
        extrinsics = observation[f'{camera}_camera_extrinsics'][0, 0]
        intrinsics = observation[f'{camera}_camera_intrinsics'][0, 0]
        pc_initial = self.pointflow_from_tracks(depth, extrinsics, intrinsics, point_coords)
        point_clouds.append(pc_initial)

    # 4. FPS 或 RPS 采样
    if self.sample_type == 'fps':
        sampled_pc, idx = sample_farthest_points(point_clouds, K=fps)
    elif self.sample_type == 'rps':
        rand_idx = np.random.choice(total_points, fps, replace=False)
        sampled_pc = point_clouds[:, rand_idx, :]

    return sampled_pc  # (1, 1, fps, 3)
```

### 13.2 DINOv2 特征提取

```python
def get_dino_feature(self, ptc, all_cam_images, depth_image_lst, cameras_extrinsics, cameras_intrinsics):
    for b in range(B):
        for t in range(T):
            pointcloud = ptc[b, t]  # (fps_num, 3)
            obs = {
                'color': all_cam_images[b, t],    # (num_cams, H, W, 3)
                'depth': depth_image_lst[b, t],   # (num_cams, H, W)
                'pose': cameras_extrinsics[b, t], # (num_cams, 3, 4)
                'K': cameras_intrinsics[b, t]     # (num_cams, 3, 3)
            }
            # 使用 Fusion 模块提取每个点的 DINOv2 特征
            dino_feature[b, t] = self.fusion.extract_semantic_feature_from_ptc(pointcloud, obs)
            # 输出: (fps_num, 384)

    return dino_feature  # (B, T, fps_num, 384)
```

### 13.3 点云采样（带 Bounding Box）

```python
def RPS_with_bounding_box(self, pc, num_samples, bounding_box):
    # 1. 提取边界框范围
    min_corner = bounding_box[0]
    max_corner = bounding_box[1]

    # 2. 过滤在边界框内的点
    pc_xyz = pc[:, :3]
    inside_mask = (pc_xyz >= min_corner) & (pc_xyz <= max_corner)
    inside_mask = inside_mask[:, 0] & inside_mask[:, 1] & inside_mask[:, 2]
    inside_pc = pc[inside_mask]

    # 3. 随机采样
    rand_idx = np.random.choice(inside_pc.shape[0], num_samples, replace=False)
    return inside_pc[rand_idx]
```

---

## 14. 可能的问题分析

### 14.1 点云采样数量不足 ⚠️

**问题**: 如果 bounding box 内的点数少于 `fps_num`，采样会失败。

**检查方法**:
```python
# 在 RPS_with_bounding_box 中添加
if inside_pc.shape[0] < num_samples:
    print(f"Warning: Only {inside_pc.shape[0]} points in bounding box, need {num_samples}")
```

### 14.2 Grounding DINO 检测失败 ⚠️

**问题**: 如果 `text_prompt` 无法检测到目标物体，返回 `[0, 0]`，导致 point flow 无效。

**影响**: 初始 point flow 为零，可能影响模型预测准确性。

**检查方法**:
```python
# 在 get_point_from_mask 中
if len(boxes) == 0:
    print(f"Warning: Grounding DINO failed to detect '{self.text_prompt}'")
```

### 14.3 DINOv2 特征提取时间 ⚠️

**问题**: DINOv2 特征提取需要对每个点进行多视图投影，计算量大。

**优化**: 可以考虑缓存特征或减少点云数量。

### 14.4 扩散步数与速度权衡

**当前设置**: `num_inference_steps` 通常为 100 步。

**影响**:
- 步数越多，生成质量越高，但推理时间越长
- 评估时可以适当减少步数（如 10-50 步）提高速度

### 14.5 Action Mode 配置问题 ⚠️

**重要**: 确保 eval 配置使用 `BimanualMoveArmThenGripper`，期望 18 维动作。

**检查方法**:
```python
# 查看 eval 配置
eval_cfg.rlbench.action_mode  # 应该是 "BimanualMoveArmThenGripper"
```

---

## 15. 调试建议

### 15.1 添加调试打印

在 `PPIAgent.act()` 中添加：
```python
if self._timestep == 0:
    print(f"observation keys: {observation.keys()}")
    print(f"point_cloud shape: {point_cloud.shape}")
    print(f"dino_feature shape: {dino_feature.shape}")
    print(f"initial_pointflow shape: {self.initial_pointflow.shape}")
```

### 15.2 检查动作范围

在 `PPIAgent.act()` 返回前添加：
```python
print(f"raw_action shape: {raw_action.shape}")
print(f"raw_action: {raw_action}")
print(f"right_xyz: {raw_action[0:3]}, left_xyz: {raw_action[9:12]}")
```

### 15.3 验证点云采样

在 `preprocess_pcd()` 中添加：
```python
print(f"Total points before sampling: {pc.shape}")
print(f"Points after bounding box filter: {inside_pc.shape}")
```

### 15.4 监控推理时间

```python
import time

# 在 PPIAgent.act() 中
torch.cuda.synchronize()
start_time = time.time()

# ... 模型推理 ...

torch.cuda.synchronize()
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.3f}s")
```

---

## 16. 总结

| 组件 | 状态 | 说明 |
|------|------|------|
| 数据维度转换 | ✓ | 评估时和训练时格式一致 |
| 动作格式 (18维) | ✓ | PPI 输出经过转换后与 BimanualMoveArmThenGripper 一致 |
| Action Mode 配置 | ⚠️ | 需验证 eval 配置使用 BimanualMoveArmThenGripper |
| 点云采样 | ⚠️ | 需确保 bounding box 内有足够点数 |
| Grounding DINO | ⚠️ | text_prompt 需要准确描述目标物体 |
| DINOv2 特征提取 | ⚠️ | 计算密集，可能影响推理速度 |
| 扩散采样步数 | ✓ | 可调节以平衡质量与速度 |

---

## 17. PPI 相比 ACT 的主要差异

| 特性 | ACT | PPI |
|------|-----|-----|
| **输入模态** | RGB 图像 | RGB + Point Cloud + DINOv2 特征 |
| **模型架构** | DETR Transformer + VAE | Diffusion Model |
| **动作表示** | 关节角度 (Joint Position) | 末端执行器姿态 (EE Pose) |
| **动作维度** | 16维 (BimanualJointPositionActionMode) | 18维 (BimanualMoveArmThenGripper) |
| **时序建模** | 时序聚合 (Temporal Aggregation) | 多步轨迹预测 (horizon_keyframe + horizon_continuous) |
| **目标检测** | 无 | Grounding DINO + SAM |
| **物体轨迹预测** | 无 | Point Flow 预测 |
| **归一化** | 基于训练统计的 Z-score | LinearNormalizer |
| **推理过程** | 单次前向传播 + 指数加权平均 | 扩散去噪迭代 (100 步) |

---

## 18. 推荐的调试步骤

1. **首先验证配置正确性**
   - 确认 `action_mode` 为 `BimanualMoveArmThenGripper`
   - 检查 `bounding_box` 设置合理
   - 验证 `text_prompt` 能正确检测目标

2. **检查数据流**
   - 打印观测维度
   - 验证点云采样数量
   - 确认 DINOv2 特征提取成功

3. **监控模型输出**
   - 检查动作范围是否合理
   - 验证 point flow 预测非零
   - 确认四元数归一化

4. **性能优化**
   - 减少扩散步数（如从 100 降至 50）
   - 调整点云采样数量
   - 优化特征提取频率

5. **错误处理**
   - 捕获 Grounding DINO 检测失败
   - 处理点云数量不足
   - 记录 IK/路径规划错误
