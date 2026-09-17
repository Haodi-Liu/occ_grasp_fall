# Direct Blocked-Grasp Failure Video Collection Guide

**创建日期**: 2026-06-23

**目的**: 为 RAL 多媒体补充材料生成 Fig. 1 及相关说明中“blocked grasp”所需的失败素材。这里的失败素材不是只针对 `bimanual_edge_phone`，而是覆盖 Bi-GRACE 当前四个任务：在每个任务中跳过原本的预操作，直接让抓取臂尝试抓取，从而展示支撑面、墙面或物体几何导致的抓取位姿不可执行。

**当前文档状态**: 方案已进入实现阶段。四个 `blocked_*` 任务和 TTM 已创建并完成基础验证；高质量录制脚本已新增为 `occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py`。

---

## 1. 修正后的核心结论

需要为四个原任务各自派生一个新的失败素材任务，而不是只做 `bimanual_edge_phone`：

| 原任务 | 策略 | 原本的 grasp-enabling 操作 | 新素材任务建议名 | 直接抓取失败要展示的现象 |
|---|---|---|---|---|
| `bimanual_edge_phone` | EdgeHang | 推手机到边缘，使一部分悬空 | `blocked_edge_phone` | 手机未悬空时，夹爪无法从下方/侧下方进入抓取位姿 |
| `bimanual_pivot_phone` | WallLever | 推手机靠墙撬起，创造底部 clearance | `blocked_pivot_phone` | 手机未被墙面撬起时，底部 clearance 不足，直接抓取失败 |
| `bimanual_pick_plate` | PressTilt | 按压盘子一侧，使另一侧翘起 | `blocked_pick_plate` | 盘子平放时边缘高度不足，夹爪无法有效夹取 |
| `bimanual_pick_fork` | PressTilt | 按压叉子头部，使叉柄翘起 | `blocked_pick_fork` | 叉柄未翘起且几何细长，直接抓取空间和精度都不足 |

四个原任务和原 TTM 场景必须保持不动。后续实现时只新增四个派生任务脚本、四个派生 TTM，以及一个可复用的高质量失败素材录制脚本。

---

## 2. 对四个任务的共同研判

### 2.1 原任务共同结构

四个任务都采用 scheme-based waypoint 组织：

- `right_grasper`: 右臂是 grasper，左臂是 pusher/assistant。
- `left_grasper`: 左臂是 grasper，右臂是 pusher/assistant。

四个任务的 grasper waypoint 都是 4 个：

```text
right_grasper: waypoint0, waypoint2, waypoint4, waypoint6
left_grasper:  waypoint0_a, waypoint2_a, waypoint4_a, waypoint6_a
```

因此，四个直接抓取失败任务的共同原则是：

1. 不执行 pusher/assistant waypoints。
2. 只执行 grasper waypoints。
3. 保留物体初始 blocked 状态。
4. 不把目标物体注册为 RLBench graspable object，避免 `close_gripper()` 后被框架自动 attach 到夹爪。
5. 允许最终任务失败；视频素材的成功标准是“画面清楚展示失败原因”，不是 RLBench success condition 成功。

### 2.2 各任务差异

| 原任务 | target object | 原 pusher waypoint 数量 | 直接抓取执行 waypoint | 重点检查的支撑/约束 |
|---|---|---:|---|---|
| `bimanual_edge_phone` | `Phone` | 4 | grasper `0,2,4,6` | 盒面/桌面边缘，手机没有 overhang |
| `bimanual_pivot_phone` | `Phone` | 5 | grasper `0,2,4,6` | 墙面附近和桌面，手机没有 pivot 起 |
| `bimanual_pick_plate` | `plate` | 4 | grasper `0,2,4,6` | 盘子边缘仍贴近桌面 |
| `bimanual_pick_fork` | `Fork_phy` | 4 | grasper `0,2,4,6` | 叉柄仍贴近桌面，几何细长导致容错小 |

`bimanual_pivot_phone` 是唯一 pusher 有 5 个 waypoint 的任务，因为它有额外的清道撤退 waypoint。直接抓取失败任务不执行 pusher，所以这个差异只影响文档和后续检查，不影响 direct grasp 的执行结构。

### 2.3 为什么必须取消 graspable registration

当前原任务都会调用类似：

```python
self.register_graspable_objects([self.target_object])
```

而 `Scene._handle_extensions_strings()` 在处理 `close_gripper()` 后，会遍历 `task.get_graspable_objects()` 并调用 `robot.grasp(obj, name)`。这会触发 RLBench/PyRep 的稳定 grasp attachment。

对训练 demo 来说这是合理的；对直接抓取失败视频来说，这可能把本应失败的接触伪装成“被抓住并抬起”。因此四个派生素材任务都应在 `init_task()` 后清空 graspable objects：

```python
self.register_graspable_objects([])
```

这比删除 waypoint extension string 更稳妥，因为我们仍然希望画面里出现 close gripper 的动作，只是不希望物体被框架强行 attach。

---

## 3. 新增文件范围

以下文件是 blocked grasp 失败素材收集所需的新增范围。原四个任务脚本和原四个 TTM 不应改动。

### 3.1 四个新任务脚本

```text
repos/RLBench/rlbench/bimanual_tasks/blocked_edge_phone.py
repos/RLBench/rlbench/bimanual_tasks/blocked_pivot_phone.py
repos/RLBench/rlbench/bimanual_tasks/blocked_pick_plate.py
repos/RLBench/rlbench/bimanual_tasks/blocked_pick_fork.py
```

建议实现方式是分别继承原任务类，避免复制大量条件类和角色选择器代码。例如：

```python
from rlbench.bimanual_tasks.bimanual_edge_phone import BimanualEdgePhone


class BlockedEdgePhone(BimanualEdgePhone):
    DIRECT_GRASP_SCHEME = "right_grasper"

    def init_task(self):
        super().init_task()
        self.register_graspable_objects([])

    def init_episode(self, index):
        self._variation_index = index
        self._step_count = 0
        self.active_waypoint_mode = self.DIRECT_GRASP_SCHEME
        self.current_role_assignment = {"grasper": "right", "pusher": "left"}
        self._setup_waypoint_mapping()
        return ["directly attempt the blocked grasp without pre-manipulation"]

    def post_placement_setup(self):
        # 不使用原任务的 pusher reachability selector。
        self.active_waypoint_mode = self.DIRECT_GRASP_SCHEME
        self.current_role_assignment = {"grasper": "right", "pusher": "left"}
        self._setup_waypoint_mapping()

    @property
    def execution_phases(self):
        active_wps = self._get_active_waypoints()
        grasper_arm = self.current_role_assignment["grasper"]
        grasper_wps = active_wps["grasper"]
        return [
            {"arm": grasper_arm, "waypoints": grasper_wps[:3], "wait_after": 0.8},
            {"arm": grasper_arm, "waypoints": [grasper_wps[3]], "wait_after": 1.0},
        ]
```

四个派生类的结构一致，只是 import 的原任务类、类名、描述文本不同。这样做的好处：

- 原任务代码完全不动。
- 原任务中的 waypoint set 定义仍可复用。
- 原任务中的目标对象命名仍可复用。
- 新任务只改变“是否执行预操作”和“是否允许自动 grasp attachment”。

### 3.2 四个新 TTM 场景

```text
repos/RLBench/rlbench/task_ttms/blocked_edge_phone.ttm
repos/RLBench/rlbench/task_ttms/blocked_pivot_phone.ttm
repos/RLBench/rlbench/task_ttms/blocked_pick_plate.ttm
repos/RLBench/rlbench/task_ttms/blocked_pick_fork.ttm
```

每个新 TTM 从对应原 TTM 派生：

| 新 TTM | 来源 TTM |
|---|---|
| `blocked_edge_phone.ttm` | `bimanual_edge_phone.ttm` |
| `blocked_pivot_phone.ttm` | `bimanual_pivot_phone.ttm` |
| `blocked_pick_plate.ttm` | `bimanual_pick_plate.ttm` |
| `blocked_pick_fork.ttm` | `bimanual_pick_fork.ttm` |

不能只做普通文件复制。RLBench 的 `Task.load()` 会根据任务名找同名 TTM，之后 `Task.get_base()` 会找同名 root dummy。新 TTM 内部 root dummy 必须改成新任务名。

可靠方法：

1. 使用 `repos/RLBench/tools/task_builder_bimanual.py` 打开原任务。
2. 用 `u` duplicate/copy 成新任务名。
3. 保存新 TTM。
4. 后续只编辑新 TTM。

### 3.3 一个通用失败素材录制脚本

已新增一个通用脚本，而不是为四个任务各写一个：

```text
occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py
```

它应复用 `render_live_demo_hq.py` 中的高质量输出能力：

- `--camera`
- `--resolution`
- `--fps`
- `--bitrate`
- `--sample-frame`
- gripper overlay
- H.264 `yuv420p` 输出

但它不能原样调用成功 demo 流程。原因是 `Scene.get_demo()` 最后会检查 success，失败会抛 `DemoError`。这个脚本应捕获 `DemoError`，并在已经录到足够帧时正常保存视频。

建议 CLI：

```text
--task
--camera
--resolution
--output
--fps
--bitrate
--seed
--max-frames
--sample-frame
--overview-from-object
--crop
--headless / --no-headless
--no-gripper-overlay
```

---

## 4. 四个任务的具体设计

### 4.1 `blocked_edge_phone`

来源：

```text
bimanual_edge_phone.py
bimanual_edge_phone.ttm
```

原成功机制：

1. pusher 把手机推到盒子/桌面边缘。
2. 手机局部 overhang。
3. grasper 从下方/侧下方抓住悬空部分。

直接失败素材：

1. 手机保持初始位置，不 overhang。
2. 不执行 `waypoint1/3/5/7`。
3. 只执行 grasper `waypoint0/2/4/6`。
4. 画面应展示：夹爪试图进入原本需要悬空后才可达的位置，但盒面/桌面阻挡了夹爪。

TTM 调整重点：

- `waypoint0` 和 `waypoint2` 要保证手臂能进入画面。
- `waypoint4` 是关键 blocked grasp pose，应明显低于“安全直接抓取”所需高度，但不能穿模过重。
- `waypoint6` 用于展示 lift 后手机仍留在支撑面上。

### 4.2 `blocked_pivot_phone`

来源：

```text
bimanual_pivot_phone.py
bimanual_pivot_phone.ttm
```

原成功机制：

1. pusher 把手机推向墙面。
2. 借助墙面约束让手机 pivot 起。
3. grasper 抓取被撬起后产生 clearance 的部分。

直接失败素材：

1. 手机不被推墙、不 pivot。
2. 不执行 `waypoint1/3/5/7/8`。
3. 只执行 grasper `waypoint0/2/4/6`。
4. 画面应展示：手机仍平放，墙面/桌面约束没有被利用，底部 clearance 不足导致直接抓取失败。

TTM 调整重点：

- `bimanual_pivot_phone` 的随机放置和旋转限制比其他任务更敏感。视频素材建议优先使用固定初始布局，不做随机化。
- 如果 grasper path 完全规划失败，先微调 `waypoint0/2`，让夹爪能靠近手机；再保留 `waypoint4` 的 blocked 特征。
- 不建议让夹爪明显穿过墙面。WallLever 失败的重点应是“没有 pivot 就没有 clearance”，不是墙体穿模。

### 4.3 `blocked_pick_plate`

来源：

```text
bimanual_pick_plate.py
bimanual_pick_plate.ttm
```

原成功机制：

1. pusher 按压盘子一侧。
2. 盘子另一侧翘起。
3. grasper 抓取翘起边缘。

直接失败素材：

1. 盘子保持平放。
2. 不执行 `waypoint1/3/5/7`。
3. 只执行 grasper `waypoint0/2/4/6`。
4. 画面应展示：盘子边缘贴近桌面，夹爪无法从下方进入或夹取边缘，close 后无法稳定带起盘子。

TTM 调整重点：

- 盘子可能比手机更容易被碰滑。失败画面可以接受轻微滑动，但不应被夹爪稳定带起。
- `waypoint4` 应对准“原本翘起后可抓”的边缘，而不是盘子上表面；否则画面会像普通顶面误抓，不像 blocked grasp。
- 如果夹爪碰撞太剧烈导致盘子飞出，应略微抬高或外移 `waypoint4`。

### 4.4 `blocked_pick_fork`

来源：

```text
bimanual_pick_fork.py
bimanual_pick_fork.ttm
```

原成功机制：

1. pusher 按压叉子头部。
2. 叉柄翘起。
3. grasper 抓取翘起的细长叉柄。

直接失败素材：

1. 叉子保持平放。
2. 不执行 `waypoint1/3/5/7`。
3. 只执行 grasper `waypoint0/2/4/6`。
4. 画面应展示：叉柄未翘起，细长几何和桌面间隙太小，夹爪无法稳定夹住或拿起。

TTM 调整重点：

- 叉子几何细长，容错最小，容易出现“夹爪看似碰到了但没夹住”的好素材。
- 相机角度要能看清叉柄与桌面的间隙，否则审稿人可能看不出 blocked 原因。
- 如果 fork 被夹爪扫走，可以作为失败素材，但最好补录一条“明显无法进入下方抓取位姿”的版本。

---

## 5. 手动操作流程

以下流程在你批准实际改动后执行。

### 5.1 逐个创建四个派生 TTM

对四个源任务分别执行一次：

```bash
cd /home/hdliu/occ_grasp_fall
python repos/RLBench/tools/task_builder_bimanual.py
```

进入后：

1. 输入源任务名，例如 `bimanual_edge_phone`。
2. 确保仿真处于停止状态。
3. 按 `u` duplicate/copy。
4. 输入新任务名，例如 `blocked_edge_phone`。
5. 保存并退出。

四组映射：

```text
bimanual_edge_phone       -> blocked_edge_phone
bimanual_pivot_phone      -> blocked_pivot_phone
bimanual_pick_plate       -> blocked_pick_plate
bimanual_pick_fork        -> blocked_pick_fork
```

如果后续由我执行代码改动，我会先创建/调整四个新 Python 脚本，再让你在 GUI 中检查和保存四个 TTM。

### 5.2 逐个检查 direct grasp failure 动作

对每个新任务：

```bash
python repos/RLBench/tools/task_builder_bimanual.py
```

输入新任务名，然后在菜单中：

1. 按 `+` 启动仿真。
2. 按 `e` 初始化同一 variation。
3. 按 `d` 执行 demo。

检查标准：

- pusher/assistant 臂没有做预操作。
- grasper 臂直接尝试抓取。
- 目标物仍处于 blocked 初始状态。
- close gripper 后目标物没有稳定 attach 或被抬起。
- lift waypoint 后目标物仍留在支撑面上，或只发生滑动/扰动。

如果 `d` 后机器人没有明显动作，而终端报 waypoint/path failure，说明当前失败是“规划失败”，不是视频中可解释的“blocked grasp 物理失败”。此时要调 grasper waypoint。

### 5.3 调整 waypoint 的通用原则

只调整新 TTM，不调整原 TTM。

优先检查：

```text
waypoint0
waypoint2
waypoint4
waypoint6
```

如果使用 left scheme，则检查：

```text
waypoint0_a
waypoint2_a
waypoint4_a
waypoint6_a
```

调参顺序：

1. 先让 `waypoint0/2` 保证手臂进入画面并接近目标物。
2. 再让 `waypoint4` 保留 blocked grasp 的物理含义。
3. 最后确认 `waypoint6` 抬起时目标物没有跟随。

如果路径规划失败：

1. 先抬高或外移 approach waypoint。
2. 再增加/调整中间 waypoint。
3. 必要时在关键 waypoint extension string 加 `ignore_collision`。

使用 `ignore_collision` 时必须人工复查视频。如果画面出现明显穿过桌面、墙面或物体的穿模，应回退，改用更浅的 blocked pose。

---

## 6. 高质量视频录制流程

### 6.1 快速预览

在 `task_builder_bimanual.py` 中按 `r` 可以快速导出调试视频，路径通常类似：

```text
/tmp/rlbench_video_<task_name>.mp4
```

这适合调 waypoint，不建议直接作为最终提交素材。

### 6.2 最终录制命令

建议用通用脚本：

```text
occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py
```

该脚本不依赖已保存 demo，而是直接启动对应 `blocked_*` 任务并执行 live waypoint。任务失败是预期结果；脚本会捕获 `DemoError`，只要已经录到足够帧就正常保存 MP4。

默认相机是 `overview`。它不是录制 CoppeliaSim 编辑器 GUI viewport，而是复用 `cam_front` 视觉传感器作为最终渲染相机。

当前推荐做法是：在 CoppeliaSim 里把满意的 GUI 视角保存成一个 camera object，例如 `blocked_video_view_cam`，然后让录制脚本用 `--overview-from-object blocked_video_view_cam` 自动读取该 camera 的完整矩阵和 FOV。这样不需要手动运行 Lua 读取数值，也比手调 `--overview-position/target/fov` 更准确。

#### 6.2.1 保存截图同款视角 camera

对每个需要该视角的 `blocked_*` 任务，在 `task_builder_bimanual.py` 打开并调好画面后：

1. 在主视图里右键，选择 `Add -> Camera`。
2. 将新 camera 重命名为 `blocked_video_view_cam`。
3. 选中这个 camera，在同一个视图里右键，选择 `View -> Associate view with selected camera`。
4. 现在视图已经绑定到该 camera。用鼠标继续微调，直到画面和目标截图一致。
5. 将 camera 挂到当前任务 root dummy 下，否则 `task_builder_bimanual.py` 的 `s` 保存可能不会把它写进 `.ttm`。

对 `blocked_edge_phone`，在 CoppeliaSim Lua console 中执行：

```lua
local cam = sim.getObjectHandle('blocked_video_view_cam')
local root = sim.getObjectHandle('blocked_edge_phone')
sim.setObjectParent(cam, root, true)
print('blocked_video_view_cam parented to blocked_edge_phone, keeping world pose')
```

其它任务只改 root 名：

```text
blocked_pivot_phone
blocked_pick_plate
blocked_pick_fork
```

`sim.setObjectParent(cam, root, true)` 中的 `true` 表示保持当前世界位姿，不会破坏已经调好的视角。完成后，如果仿真正在运行，先在 task builder 里按 `+` 停止仿真，再按 `s` 保存 `.ttm`。

`blocked_video_view_cam` 在 GUI 中发灰或不方便点选通常不是问题；只要它存在于 scene hierarchy 中，并且已 parent 到当前任务 root 下，脚本就能读取。

如果想解除当前视图和 camera 的绑定，可以在该视图里右键 `View -> View selector...`，选回其它 camera/vision sensor 或空视图。是否解除绑定不影响录制脚本读取 `blocked_video_view_cam`。

#### 6.2.2 用保存的 camera 录制预览

先做单任务首帧预览：

```bash
python occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py \
  --task blocked_edge_phone \
  --camera overview \
  --overview-from-object blocked_video_view_cam \
  --resolution 1920x1080 \
  --crop 0,0,1280,1080 \
  --output /tmp/blocked_edge_phone_preview.mp4 \
  --fps 30 \
  --bitrate 12000k \
  --seed 0 \
  --no-headless \
  --max-frames 1 \
  --min-frames 1 \
  --sample-frame /tmp/blocked_edge_phone_preview.png
```

如果输出中出现类似：

```text
Overview source: blocked_video_view_cam (CAMERA)
```

说明脚本已经自动读到了保存在 `.ttm` 里的视角 camera。输出文件为：

```text
/tmp/blocked_edge_phone_preview.mp4
/tmp/blocked_edge_phone_preview.png
```

如果报 `Overview source object 'blocked_video_view_cam' does not exist`，说明该 camera 没有保存进当前任务的 `.ttm`，或没有挂到当前任务 root model 下。

#### 6.2.3 裁剪输出画面

`--resolution` 是 CoppeliaSim vision sensor 的渲染分辨率；`--crop` 是写出 MP4/PNG 前的后处理裁剪。对于当前 `blocked_edge_phone` 视角，主体偏左，推荐先用：

```text
--resolution 1920x1080 --crop 0,0,1280,1080
```

这样输出视频和 sample frame 会变为 `1280x1080`，能截掉右侧大面积空地，同时比严格正方形更完整地保留双臂、桌面和目标物。

如果必须严格正方形，可以改用：

```text
--crop left-square
```

在 `1920x1080` 渲染下，它等价于保留左侧 `1080x1080` 区域。若后续视角变化，也可以直接给自定义裁剪框：

```text
--crop x,y,width,height
```

例如 `--crop 0,0,1280,1080`。

#### 6.2.4 手写 overview 参数作为备用方案

如果不想在 `.ttm` 里保存 camera，也可以手动使用脚本内置 overview 参数。该备用视角会复用 `cam_front` 并移动到一个干净的斜上方总览位置，默认参数为：

```text
overview_position = 2.35,0.0,1.85
overview_target   = -0.05,0.0,0.72
overview_fov      = 66
```

如果首帧构图需要微调，可以只改命令行参数。注意负数向量建议使用等号写法，避免被 argparse 当成新选项：

```bash
python occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py \
  --task blocked_edge_phone \
  --camera overview \
  --overview-position=2.35,0.0,1.85 \
  --overview-target=-0.05,0.0,0.72 \
  --overview-fov 66 \
  --resolution 1920x1080 \
  --crop 0,0,1280,1080 \
  --output /tmp/blocked_edge_phone_preview.mp4 \
  --fps 30 \
  --bitrate 12000k \
  --seed 0 \
  --no-headless \
  --max-frames 1 \
  --min-frames 1 \
  --sample-frame /tmp/blocked_edge_phone_preview.png
```

四任务正式输出示例：

```bash
python occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py \
  --task blocked_edge_phone \
  --camera overview \
  --overview-from-object blocked_video_view_cam \
  --resolution 1920x1080 \
  --crop 0,0,1280,1080 \
  --output outputs/blocked_grasp_videos/blocked_edge_phone_overview.mp4 \
  --fps 30 \
  --bitrate 12000k \
  --seed 0 \
  --no-headless \
  --sample-frame outputs/blocked_grasp_videos/blocked_edge_phone_overview.png

python occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py \
  --task blocked_pivot_phone \
  --camera overview \
  --overview-from-object blocked_video_view_cam \
  --resolution 1920x1080 \
  --crop 0,0,1280,1080 \
  --output outputs/blocked_grasp_videos/blocked_pivot_phone_overview.mp4 \
  --fps 30 \
  --bitrate 12000k \
  --seed 0 \
  --no-headless \
  --sample-frame outputs/blocked_grasp_videos/blocked_pivot_phone_overview.png

python occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py \
  --task blocked_pick_plate \
  --camera overview \
  --overview-from-object blocked_video_view_cam \
  --resolution 1920x1080 \
  --crop 0,0,1280,1080 \
  --output outputs/blocked_grasp_videos/blocked_pick_plate_overview.mp4 \
  --fps 30 \
  --bitrate 12000k \
  --seed 0 \
  --no-headless \
  --sample-frame outputs/blocked_grasp_videos/blocked_pick_plate_overview.png

python occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py \
  --task blocked_pick_fork \
  --camera overview \
  --overview-from-object blocked_video_view_cam \
  --resolution 1920x1080 \
  --crop 0,0,1280,1080 \
  --output outputs/blocked_grasp_videos/blocked_pick_fork_overview.mp4 \
  --fps 30 \
  --bitrate 12000k \
  --seed 0 \
  --no-headless \
  --sample-frame outputs/blocked_grasp_videos/blocked_pick_fork_overview.png
```

推荐先用 `overview`。若局部遮挡关系不清楚，再补录内置相机：

- `front`
- `over_shoulder_left`
- `over_shoulder_right`
- `overhead`

脚本默认不画 gripper marker，适合最终提交素材。如果调试阶段需要确认左右夹爪轨迹，可加 `--draw-gripper-overlay`。

在当前机器上，CoppeliaSim 4.1 的视觉传感器渲染需要可用 OpenGL 上下文；如果默认 headless/offscreen 报 OpenGL/Qt 错误，使用 `--no-headless` 连接当前图形会话更可靠。

---

## 7. 验收标准

### 7.1 文件级验收

实际实现完成后，允许新增：

```text
repos/RLBench/rlbench/bimanual_tasks/blocked_edge_phone.py
repos/RLBench/rlbench/bimanual_tasks/blocked_pivot_phone.py
repos/RLBench/rlbench/bimanual_tasks/blocked_pick_plate.py
repos/RLBench/rlbench/bimanual_tasks/blocked_pick_fork.py

repos/RLBench/rlbench/task_ttms/blocked_edge_phone.ttm
repos/RLBench/rlbench/task_ttms/blocked_pivot_phone.ttm
repos/RLBench/rlbench/task_ttms/blocked_pick_plate.ttm
repos/RLBench/rlbench/task_ttms/blocked_pick_fork.ttm

occ_grasp_models/scripts/capture_blocked_grasp_failure_hq.py
```

不允许改动：

```text
repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py
repos/RLBench/rlbench/bimanual_tasks/bimanual_pivot_phone.py
repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_plate.py
repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_fork.py

repos/RLBench/rlbench/task_ttms/bimanual_edge_phone.ttm
repos/RLBench/rlbench/task_ttms/bimanual_pivot_phone.ttm
repos/RLBench/rlbench/task_ttms/bimanual_pick_plate.ttm
repos/RLBench/rlbench/task_ttms/bimanual_pick_fork.ttm
```

当前实现状态：上述四个 `blocked_*` 任务脚本、四个 `blocked_*` TTM 和通用录制脚本均已新增；原四个任务脚本和原四个 TTM 仍应保持不动。

### 7.2 行为级验收

四个任务的视频都应满足：

- 没有预操作。
- 只有 grasper 执行 direct grasp。
- 目标物保持对应任务的初始 blocked 状态。
- close gripper 后没有稳定抓住目标物。
- lift 后目标物没有被拿起。
- 失败原因能从画面直接理解。

各任务的额外验收：

- Edge phone: 能看出手机没有 overhang。
- Pivot phone: 能看出手机没有被墙面 pivot 起。
- Pick plate: 能看出盘子边缘仍贴近桌面。
- Pick fork: 能看出叉柄没有翘起，细长目标难以直接夹取。

### 7.3 画面级验收

首选画面：

- 同时看到目标物、支撑/约束结构、夹爪。
- 能看到夹爪尝试进入被遮挡的抓取区域。
- 能看到直接抓取失败后目标物仍留在原处。

应避免：

- 机器人完全不动，只是规划器报错。
- 夹爪明显穿过桌面、墙面或目标物。
- 目标物被自动 attach 到 gripper 并抬起。
- 相机角度无法说明 blocked grasp 的因果关系。

---

## 8. 推荐执行顺序

审阅通过后，建议按下面顺序推进：

1. 为四个源任务分别 duplicate 出四个新 TTM，并确保 root dummy 与新任务名一致。
2. 新增四个派生任务脚本，均只执行 grasper direct grasp，均清空 graspable objects。
3. 用 task builder 逐个播放 `d`，确认四个任务都没有预操作。
4. 对每个新 TTM 只调 grasper waypoints，让失败过程可视化。
5. 用 task builder 的 `r` 快速导出预览。
6. 运行通用 `capture_blocked_grasp_failure_hq.py`，逐个导出 1080p MP4 和 sample frame。
7. 从每个任务的 overview/front/shoulder/overhead 版本中选一条最能说明 blocked grasp 的视频片段。
