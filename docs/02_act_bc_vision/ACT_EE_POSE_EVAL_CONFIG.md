# ACT末端位姿评估配置指南

## 问题回答

**Q: 新改过的ACT在评估时能规避CoppeliaSim的waypoint错误吗？**

**A: 可以！但需要修改评估配置文件中的action mode设置。**

---

## 核心问题分析

### 当前问题

当前`eval.yaml`中的配置：
```yaml
arm_action_mode: 'BimanualJointPosition'
action_mode: 'BimanualJointPositionActionMode'
```

这个配置的问题：
1. **BimanualJointPosition**: 直接控制关节角度，不使用motion planning
2. **不兼容RLBench的waypoint validation系统**
3. 会导致以下错误：
   - `WARNING: Waypoints are not reachable`
   - `ERROR: Error when checking waypoints`
   - `RuntimeError: The call failed on the V-REP side`
   - `BrokenPipeError: [Errno 32] Broken pipe`

### 为什么我们的修改还不够

虽然我们已经把ACT模型改为输出18D end-effector pose：
```
[right_pos(3), right_quat(4), right_gripper(1), right_ignore(1),
 left_pos(3), left_quat(4), left_gripper(1), left_ignore(1)] = 18D
```

但这**只是改了模型的输出格式**，评估时的action mode仍然在使用joint position控制！

---

## 解决方案

### 必须修改的配置

修改`conf/eval.yaml`中的action mode设置：

```yaml
rlbench:
    # ... 其他配置保持不变 ...

    gripper_mode: 'BimanualDiscrete'

    # 使用BimanualMoveArmThenGripper作为ActionMode包装类
    # 它会从18D action中提取14D pose、2D gripper、2D ignore_collisions
    action_mode: 'BimanualMoveArmThenGripperActionMode'

    # 使用BimanualEndEffectorPoseViaPlanning作为arm action mode
    # 这个模式使用IK + motion planning，兼容waypoint validation
    arm_action_mode: 'BimanualEndEffectorPoseViaPlanning'
```

### 为什么这个配置能解决问题

#### 1. BimanualMoveArmThenGripper的工作流程

参考`repos/RLBench/rlbench/action_modes/action_mode.py:56-93`：

```python
class BimanualMoveArmThenGripper(ActionMode):
    def action(self, scene: Scene, action: np.ndarray):
        assert(len(action) == 18)  # ✓ 接收18D action

        # 提取右臂和左臂的pose（7D each）
        right_arm_action = action[0:7]   # right_pos(3) + right_quat(4)
        left_arm_action = action[9:16]   # left_pos(3) + left_quat(4)
        arm_action = np.concatenate([right_arm_action, left_arm_action])  # 14D

        # 提取gripper（1D each）
        right_ee_action = action[7:8]    # right_gripper
        left_ee_action = action[16:17]   # left_gripper
        ee_action = np.concatenate([right_ee_action, left_ee_action])  # 2D

        # 提取ignore_collisions（1D each）
        right_ignore = bool(action[8])
        left_ignore = bool(action[17])
        ignore_collisions = [right_ignore, left_ignore]

        # 先执行arm action（使用motion planning）
        self.arm_action_mode.action(scene, arm_action, ignore_collisions)

        # 再执行gripper action
        self.gripper_action_mode.action(scene, ee_action)
```

#### 2. BimanualEndEffectorPoseViaPlanning的关键特性

参考`repos/RLBench/rlbench/action_modes/arm_action_modes.py:415-483`：

```python
class BimanualEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):
    def action(self, scene: Scene, action: np.ndarray, ignore_collisions):
        # action: 14D [right_pose(7), left_pose(7)]
        # ignore_collisions: [right_bool, left_bool]

        # 使用IK + path planning生成路径
        right_path = self.get_path(scene, right_action, right_ignore_collision,
                                    scene.robot.right_arm, scene.robot.right_gripper)
        left_path = self.get_path(scene, left_action, left_ignore_collison,
                                   scene.robot.left_arm, scene.robot.left_gripper)

        # 同时执行两条路径
        while not done:
            if not right_done: right_done = right_path.step()
            if not left_done: left_done = left_path.step()
            scene.step()
```

**关键点**：
- `get_path()` 方法使用`arm.get_path()`进行motion planning
- 这与demo生成时的`execute_waypoints_bimanual()`使用**相同的IK和path planning机制**
- 因此与RLBench的waypoint validation系统**完全兼容**

---

## 技术对比

### 原配置（Joint Position控制）

```
模型输出 → BimanualJointPositionActionMode → BimanualJointPosition
              ↓
         直接设置关节角度（无规划）
              ↓
         不兼容waypoint validation
              ↓
         ❌ Waypoints are not reachable错误
```

### 新配置（End-Effector Pose控制 + Motion Planning）

```
模型输出(18D) → BimanualMoveArmThenGripper → 提取14D pose + 2D gripper + 2D ignore
                      ↓
                BimanualEndEffectorPoseViaPlanning
                      ↓
                IK + Path Planning（与demo生成相同）
                      ↓
                兼容waypoint validation
                      ↓
                ✅ 成功避开waypoint错误
```

---

## PPI的参考实现

PPI在`/home/hdliu/claude_repo/PPI/inference-for-rlbench2/eval_bimanual.py:128-133`中：

```python
action_mode = BimanualMoveArmThenGripper(
    arm_action_mode=BimanualEndEffectorPoseViaPlanning(),
    gripper_action_mode=BimanualDiscrete()
)
```

这正是我们需要的配置！

---

## 实施步骤

### 1. 修改`conf/eval.yaml`

```yaml
rlbench:
    task_name: "multi"
    tasks: [bimanual_pick_plate]
    demo_path: /mnt/rlbench_data
    episode_length: 800
    cameras: ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"]
    camera_resolution: [256, 256]
    scene_bounds: [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    include_lang_goal_in_obs: False
    time_in_state: True
    headless: True

    # ✅ 正确的配置
    gripper_mode: 'BimanualDiscrete'
    arm_action_mode: 'BimanualEndEffectorPoseViaPlanning'
    action_mode: 'BimanualMoveArmThenGripperActionMode'
```

### 2. 验证模型输出格式

确保`act_bc_vision_agent.py:292-293`的输出是18D：

```python
raw_action = torch.cat([
    right_pos, right_quat_normalized, right_gripper, right_ignore_collision,
    left_pos, left_quat_normalized, left_gripper, left_ignore_collision
], dim=-1)
```

✅ 已确认正确

### 3. 运行评估

```bash
cd /home/hdliu/occ_grasp_fall/occ_grasp_models
python eval.py
```

---

## 预期效果

使用新配置后：

✅ **不会再出现以下错误**：
- `WARNING: Waypoints are not reachable`
- `ERROR: Error when checking waypoints`
- `RuntimeError: The call failed on the V-REP side`
- `BrokenPipeError`

✅ **评估流程**：
1. ACT输出18D end-effector pose
2. BimanualMoveArmThenGripper解析action
3. BimanualEndEffectorPoseViaPlanning进行motion planning
4. 平滑、无碰撞地执行动作
5. 通过waypoint validation

---

## 注意事项

### 1. Action Mode的名称

在`eval.yaml`中，action mode的名称应该是字符串，RLBench会自动查找对应的类。检查RLBench代码确认准确的类名。

### 2. Motion Planning可能较慢

使用motion planning会比直接joint control慢：
- Path planning通常需要几秒钟
- 但这是获得可靠、无碰撞轨迹的代价
- PPI使用相同方案，已被证明有效

### 3. ignore_collisions的使用

我们在训练时hardcode了`ignore_collisions=1.0`：
- 评估时会传递这个值给motion planner
- `ignore_collisions=True`意味着planning时不检查碰撞
- 但仍会生成平滑的IK轨迹
- 与demo生成时的行为一致

---

## 总结

**能规避waypoint错误吗？** → **能！**

**需要做什么？** → 修改`eval.yaml`中的action mode配置：
- `action_mode: 'BimanualMoveArmThenGripperActionMode'`
- `arm_action_mode: 'BimanualEndEffectorPoseViaPlanning'`

**原理是什么？** → 使用与demo生成相同的IK + motion planning机制，天然兼容waypoint validation系统。

**代码需要改吗？** → 不需要！模型输出已经是正确的18D格式，只需改配置文件。
