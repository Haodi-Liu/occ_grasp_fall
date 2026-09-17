# Bi-GRACE Baseline Taxonomy and Evaluation Protocol — Revised for RAL26 Draft

本文档用于将 Bi-GRACE 的 baseline taxonomy、训练/评估协议、指标解释和结果叙述方式统一到当前 RAL26 引言的立意上。当前论文的核心不应被表述为“又一个双臂任务集合”，而应被表述为一个围绕 **bimanual occluded grasping / graspability transformation** 的评测基准：策略是否能够利用物体几何和环境 affordance，将初始不可直接抓取的状态转化为可抓取状态，并进一步完成稳定抓取和双臂协调拿起。

因此，本文档中的评估逻辑统一服务于引言中的三项能力主张：

1. **Extrinsic Dexterity Leverage**：策略是否真正利用桌边、墙面、物体几何等外在结构来改变物体—环境接触关系，创造可抓取条件；
2. **Bimanual Role Coordination**：策略是否能根据随机物体位姿、可达性和运动代价，自适应选择哪只臂辅助、哪只臂抓取，而不是固定使用某一只手；
3. **Spatiotemporal Execution Precision**：策略是否能在正确空间位置、正确执行阶段、正确双臂时序下完成从预操作到抓取、清道、拿起的连续过程。

Bi-GRACE 面向四个 RLBench2-compatible 双臂遮挡抓取任务：`bimanual_edge_phone`、`bimanual_pivot_phone`、`bimanual_pick_plate` 和 `bimanual_pick_fork`。四个任务覆盖三类 grasp-enabling mechanisms：**EdgeHang**、**WallLever** 和 **PressTilt**，并统一拆分为 **PreManipulation、Grasp、ClearPath、Lift** 四个阶段。benchmark 额外提供 `strategy_type`、`phase_type`、双臂角色标签，以及 contact / grasp / affordance 三类空间关键点，用于训练后诊断、失败归因和可视化解释。

**当前 RAL 实验侧重点**：考虑篇幅，主实验建议以 **multi-task joint training & evaluation** 为核心，重点报告多任务联合训练后不同模型在四个任务上的最终成功率、策略一致性、阶段进展和角色适配能力；single-task 结果可压缩为最终成功率表或附录补充。附件三中的逐 episode 记录非常重要，但不需要完整搬入正文，它主要用于验证和支撑本文档定义的 coarse metrics 与 diagnostic claims。

**重要公平性说明**：当前所有 baseline 的训练均不使用 Bi-GRACE 的额外诊断标注，包括 `strategy_type`、`phase_type`、双臂角色标签，以及 contact / grasp / affordance 三类关键点。这些标注主要用于 evaluation-time diagnostics、failure localization 和 visualization。各 baseline 的训练流程、观测输入、动作输出和方法自身所需的中间信息应尽量保持与原方法一致。PPI 是需要单独说明的例外：PPI 原方法本身包含 object pointflow / keypose 等训练信号，因此 Bi-GRACE 适配中可保留 PPI-native pointflow 监督；这不等同于向 PPI 或其他 baseline 注入 Bi-GRACE 的 strategy、phase、role 或 diagnostic keypoint supervision。

---

## 1. Task, Strategy, Phase, and Annotation Overview

### 1.1 Four Tasks and Three Grasp-Enabling Mechanisms

| Task code | Paper-friendly name | Object type | Strategy type | Mechanism | Main BOG capability tested |
|---|---|---|---|---|---|
| `bimanual_edge_phone` | EdgePhone | flat phone | **EdgeHang** | 辅助臂将手机推向桌边 / 盒边，使一部分悬空，抓取臂从暴露边缘抓取 | 利用边缘 affordance 创造可抓取空间 |
| `bimanual_pivot_phone` | PivotPhone | flat phone | **WallLever** | 辅助臂将手机推向墙面 / 挡板，借助墙面约束产生 pivot / lever 效果，再由抓取臂抓取 | 利用外部支撑面改变接触关系 |
| `bimanual_pick_plate` | PressPlate | flat plate | **PressTilt** | 一只手按压盘子一侧，使另一侧翘起，另一只手抓取翘起边缘 | 利用物体几何产生局部抓取空隙 |
| `bimanual_pick_fork` | PressFork | slender fork | **PressTilt** | 通过按压 / 姿态改变使细长叉子产生可夹持空间 | 测试 PressTilt 对细长物体的空间精度和抓取稳定性 |

这四个任务不只是四个 object-level instances，而是三类外在灵巧机制的 controlled instantiations。`bimanual_edge_phone` 和 `bimanual_pivot_phone` 共享 phone 物体但使用不同环境 affordance；`bimanual_pick_plate` 和 `bimanual_pick_fork` 共享 PressTilt 机制但物体几何不同。这种设计使多任务评估能够区分：模型是在记忆对象 / 任务 ID，还是在理解“应当如何利用当前物体—环境关系创造抓取条件”。

### 1.2 Four Execution Phases

Bi-GRACE 将每个 rollout 统一拆为四个物理含义清晰的阶段：

| Phase | Name | Physical meaning | Diagnostic question |
|---|---|---|---|
| P1 | **PreManipulation** | 通过推、压、撬、悬空等预操作改变物体—环境接触状态 | 是否创造了可抓取条件？ |
| P2 | **Grasp** | 抓取臂在暴露出的 graspable region 建立稳定抓取 | 是否能从“可抓取状态”过渡到稳定抓取？ |
| P3 | **ClearPath** | 辅助臂撤离 / 清道，避免阻挡抓取臂抬升 | 双臂是否完成了 non-interfering coordination？ |
| P4 | **Lift** | 抬起目标物体并达到任务成功状态 | 抓取是否稳定，最终动作是否完成？ |

正文中应强调：phase metrics 是一条 **failure funnel**，不是四个互相独立的局部指标。P2 成功隐含 P1 已完成，P3 成功隐含 P1–P2 已完成，P4 成功通常与最终任务成功一致。附件三也采用“已完成阶段即记为 Y”的原则，即使后续动作随机失误导致状态回退，也只统计该阶段是否曾经被完成。因此，阶段指标更接近 episode-level progress record，而不是瞬时状态分类。

### 1.3 Strategy, Role, and Spatial Keypoint Annotations

Bi-GRACE 的额外标注包括：

- **`strategy_type`**：描述当前任务 / 演示所采用的 grasp-enabling mechanism，例如 EdgeHang、WallLever、PressTilt；
- **`phase_type`**：描述当前执行进展，配合在线 phase evaluator 形成 episode-level phase success record；
- **bimanual role scheme**：记录演示收集时根据可达性和运动代价选择的 `left_grasper` 或 `right_grasper` scheme；
- **contact / grasp / affordance keypoints**：分别描述预操作接触点、最终抓取点，以及环境中可被利用的边缘 / 墙面 / 支撑区域。对于 PressTilt 任务，affordance 更多来自物体自身几何，因此可以没有显式环境 affordance point。

这些标注在当前 baseline 评估中的定位是：**训练后解释模型行为，而非给 baseline 额外 supervision**。它们使同一次 rollout 能同时被解释为：策略选得对不对、执行卡在哪一阶段、双臂角色是否合理、机器人是否在正确物体—环境区域发生交互。

---

## 2. Baseline Taxonomy

本节保留原始 taxonomy，但对当前论文的实验状态作更清晰区分：**附件三中已有逐 episode 诊断记录的是 ACT 与 PPI 两类代表模型**；其他方法保留为 benchmark protocol 的 broader baseline pool，可在实验完成后纳入同一评估体系。正文写作时建议避免把尚未完成或未进入主表的模型写成已充分实验的结果。

### 2.1 Image-based Visuomotor Policies

这一类方法主要从多视角图像和机器人状态直接预测连续双臂动作序列，不显式构造体素、点云、物体 pointflow 或环境 affordance 等中间空间结构。它们适合作为 end-to-end imitation learning / visuomotor policy 的代表，用来检验模型是否能够仅依靠图像特征和动作序列建模，自动学到“预操作—抓取—清道—拿起”的完整动作链。

- **ACT / act_bc_vision**：ACT 通过 action chunking 一次预测未来一段动作，从而降低长时序模仿学习中的有效 horizon；temporal ensemble 通过融合重叠动作块提升执行平滑性和稳定性。在 Bi-GRACE 中，`act_bc_vision` 可视为纯视觉行为克隆基线，输入多视角 RGB 图像和双臂状态，输出双臂末端执行器动作。它不使用 Bi-GRACE 的 strategy / phase / role / keypoint 诊断标注。当前附件三中 ACT 的多任务逐 episode 记录可用于说明：纯图像动作克隆即使在部分任务中能产生接近预期的动作趋势，也容易在预操作完成、稳定抓取或阶段转换处掉点，因此 final success 不足以解释其失败来源。

- **Diffusion Policy**：Diffusion Policy 将动作序列建模为条件扩散生成过程，在视觉和机器人状态条件下迭代生成未来动作 chunk，并通过 receding horizon 方式闭环执行。与 ACT 相比，它不是直接回归动作块，而是用扩散模型表达多模态动作分布，因此适合处理演示中可能存在的多种可行轨迹。在 Bi-GRACE 中，它仍应沿用原始 image/state-conditioned action diffusion 范式，不使用 Bi-GRACE 额外诊断标注。若纳入实验，它主要用于检验生成式 action-sequence policy 是否能更好处理预操作和抓取阶段之间的长程依赖。

### 2.2 3D-aware Visuomotor Policies

这一类方法显式引入 3D 场景表示或空间交互接口，例如 voxel action、3D trajectory token、keypose 和 pointflow。它们与 Bi-GRACE 的任务结构高度契合，因为遮挡抓取的核心不是在当前图像中找到一个可见抓取框，而是理解物体、桌边、墙面、支撑面、接触区域和抓取区域之间的空间关系，并据此创造可抓取状态。这里的 “3D-aware” 应理解为方法自身的原始建模范式，而不是来自 Bi-GRACE 额外诊断标注的统一监督。

- **Bimanual PerAct / PerAct²**：PerAct 系列将多视角 RGB-D 观测融合为体素化 3D 场景，并通过 Perceiver-style 网络在体素空间中预测离散化 6-DoF 动作。双臂版本进一步为左右臂同时输出平移、旋转、夹爪和碰撞相关动作，使策略能够在共享 3D 场景表示中学习双臂协调。它在 Bi-GRACE 中代表 voxel-based 3D action policy，可用于测试体素化空间推理是否支持外在灵巧中的环境利用和角色分配。

- **3D Diffuser Actor**：3D Diffuser Actor 将扩散策略与 3D scene representation 结合，在三维空间中对末端执行器轨迹进行条件去噪。与标准 Diffusion Policy 相比，它的关键区别是动作生成直接发生在 3D 场景 token 和未来 end-effector pose token 的交互中，而不只是基于图像特征生成动作序列。它在 Bi-GRACE 中适合作为 3D trajectory diffusion baseline，检验三维轨迹生成是否有助于处理桌边、墙面、支撑面等几何约束。

- **PPI**：PPI 将关键帧预测、连续动作预测和物体 pointflow 预测结合起来，是一种更结构化的层次式扩散策略。它通过点云和语义特征建模场景，并使用原方法所需的 keypose / pointflow 训练信号来描述“物体将如何运动、机器人应在何处介入”。在 Bi-GRACE 中，PPI 的 pointflow 可由空间几何 / 关键点相关信息构造，但这是为了满足 PPI 原本的数据接口和训练目标，并不表示 Bi-GRACE 的 strategy、phase、role 或 diagnostic keypoint 标注被额外提供给模型。当前附件三中 PPI 的逐 episode 诊断记录尤其适合展示：结构化 3D 表示可能更容易学到正确的 grasp-enabling tendency 和 PreManipulation，但 phase funnel 仍会暴露从 graspable-state creation 到 stable grasp / lift 的瓶颈。

### 2.3 Vision-Language-Action Policies / VLAs

这一类方法以大规模视觉-语言-动作预训练模型为主体，利用 web-scale 或多机器人数据中的语义和动作先验，再通过连续动作头、flow matching 或 action expert 输出机器人控制信号。它与前两类的比较层级不同：前两类主要评估在 Bi-GRACE 数据上训练出的任务策略，而 VLA 类方法评估大规模预训练语义先验能否迁移到接触密集、几何约束强、需要双臂角色分配的遮挡抓取任务中。

- **π0.5 / openpi**：π0.5 / openpi 可作为 foundation VLA baseline，强调从大规模多模态数据中学习开放世界泛化能力，并通过连续动作生成模块连接到机器人控制。对 Bi-GRACE 而言，它的价值不只是比较最终成功率，而是观察语言 / 视觉语义先验是否能帮助模型理解任务目标、物体类别、环境结构和抓取策略。若当前 RAL 主实验尚未完成该模型的严格 rollout 评估，建议在正文中将其作为 planned / extensible baseline pool，而不是与 ACT、PPI 主结果混合汇报。

### 2.4 Suggested Baseline Taxonomy Table

| Family | Model | Main representation | Action interface | Diagnostic value |
|---|---|---|---|---|
| Image-based | ACT / act_bc_vision | RGB + robot state | action chunk | 测试纯视觉行为克隆是否能学到 BOG 长时序链条 |
| Image-based | Diffusion Policy | RGB/state-conditioned action diffusion | receding-horizon action chunk | 测试动作分布建模是否改善阶段转换 |
| 3D-aware | Bimanual PerAct / PerAct² | voxelized RGB-D scene | discrete 6-DoF voxel action | 测试 voxel 3D 推理对环境利用和角色协调的作用 |
| 3D-aware | 3D Diffuser Actor | 3D scene tokens + trajectory diffusion | 3D end-effector trajectory | 测试三维轨迹生成对几何约束任务的帮助 |
| 3D-aware | PPI | point cloud / keypose / pointflow | hierarchical diffusion / action prediction | 测试结构化 3D / object-motion modeling 是否改善 graspability transformation |
| VLA | π0.5 / openpi | pretrained VLA representation | continuous action head / action expert | 测试大规模语义先验能否迁移到接触密集 BOG |

**Supplement: actual information read from `/mnt/rlbench_data` by each baseline.** 下面按“模型实际消费的训练 / 评估输入”统计；若某些信息只在预处理阶段用于生成 point cloud、DINO feature 或 packaged dataset，但不是最终直接喂给模型的 tensor，则单独说明。

| Model | RGB | Language | Depth / point cloud | Low-dimensional state / action | Notes on `/mnt/rlbench_data` usage |
|---|---|---|---|---|---|
| ACT / `act_bc_vision` | Yes | No | Not used by the ACT forward pass | Yes | 通过 `train.py` 读取 `/mnt/rlbench_data/<task>.train`。replay 中可能保存由 depth 反投影得到的 point cloud，但 ACT actor 实际只接收 camera RGB 与 `qpos` / action chunk；point cloud 不进入模型前向。默认相机来自 `cfg.rlbench.cameras`，常见配置为 `overhead`。 |
| Diffusion Policy | Yes | No | No | Yes | 直接读取 `/mnt/rlbench_data/<task>.train/all_variations/episodes` 下的 RGB 与 `low_dim_obs.pkl`。不读取 depth、point cloud 或 `variation_descriptions.pkl`。低维状态 / 动作主要来自双臂 gripper pose、gripper open 和 ignore-collision 标志，形成 18D action/state 表示。 |
| Bimanual PerAct / PerAct² | Yes | Default no | Yes, as depth-derived point cloud | Yes | 使用多视角 RGB、由 depth + camera intrinsics/extrinsics 反投影得到的 point cloud、双臂低维状态和离散化 6-DoF 动作。显式 `*_depth` 通常会在 replay 样本中删除，模型消费的是 point cloud 而不是 raw depth image。当前默认 `no_language: True`；若某个旧 checkpoint 配置为 `no_language: False`，则会额外使用 CLIP text embedding。 |
| PPI | Yes, mainly through processed RGB point cloud / DINO features | Yes | Yes | Yes | 训练入口 `scripts/ppi/training/ddp_train_four_tasks.sh` 使用 processed 数据：`rgb_pcd_rps6144` point cloud、DINO feature、`world_ordered_rps200` point flow、language embedding 和低维状态。原始数据通过 `data/training_raw/<task>` symlink 指向 `/mnt/rlbench_data/<task>.train`；预处理会读六相机 RGB+depth 生成 point cloud / DINO，并读 front RGB+depth、`object_6d_pose` 等生成 PPI-native point flow。这里的 point flow 是 PPI 原方法所需监督，不等同于使用 Bi-GRACE strategy / phase / role / diagnostic keypoint labels。 |
| 3D Diffuser Actor | Yes | Yes | Yes, as packaged point cloud | Yes | `scripts/train_keypose_peract.sh` 训练阶段读取 `data/bimanual/packaged/train` 与 `val`，不直接读 `/mnt`；packaging 阶段从 `/mnt/rlbench_data/<task>.train` / `.val` 读取 RGB、depth、camera calibration 和 low-dim obs，打包为 RGB + PCD + action + instruction。当前脚本只训练 `bimanual_edge_phone`，使用 `over_shoulder_left`、`over_shoulder_right`、`overhead`、`wrist_right`、`wrist_left` 五个相机，不包含 `front`。 |
| π0.5 / openpi training | Yes | Yes | No | Yes | 按 `openpi/docs/rlbench_pi05_adaptation_guide_local.md`，先从 `/mnt` 导出 LeRobot 数据。只取 `front_rgb`、`wrist_left_rgb`、`wrist_right_rgb` 三路 RGB；语言来自 `variation_descriptions.pkl`；状态 / 动作是 left-first joint16，即左右 7D joint positions 加 gripper open。训练不使用 depth、point cloud、object pose 或 Bi-GRACE diagnostic labels。 |
| π0.5 / openpi evaluation | Yes | Yes | No | Yes | occ 侧 openpi eval agent 从 RLBench observation 中取 `front_rgb`、`wrist_left_rgb`、`wrist_right_rgb`、双臂 joint positions、gripper open 和 `lang_goal`，再通过 websocket 调 openpi policy。即使 RLBench obs config 可产生 depth / point cloud，openpi eval agent 也不会把它们传给模型。 |

从公平性角度，最重要的区分是：PerAct²、PPI 和 3D Diffuser Actor 使用 depth 主要是为了构造 3D 表示；ACT、Diffusion Policy 和 openpi 系列本质上是 RGB + robot-state 输入。除 PPI-native point flow 外，上表模型均不把 Bi-GRACE 的 `strategy_type`、`phase_type`、role label 或 diagnostic keypoints 作为训练监督输入。

---

## 3. Evaluation Settings and Metrics

### 3.1 Main Setting: Multi-task Joint Training & Evaluation

当前 RAL 正文建议将 multi-task joint training & evaluation 作为主设置：同一个模型在四个 Bi-GRACE 任务上联合训练，并分别在四个任务上评估。该设置最贴合引言中的问题，因为它不只是考察某个固定动作模板能否被模仿，而是考察模型是否能在不同物体、不同 affordance、不同 grasp-enabling mechanism 和不同双臂角色配置之间做出适配。

多任务主表建议围绕以下指标组织：

| Metric | Suggested name | Main capability | Explanation |
|---|---|---|---|
| 每任务最终成功率 | **Per-task Task Success Rate** | Overall task completion | 分别在四个任务上统计最终任务成功率，并报告四任务 macro average。macro average 比直接汇总所有 rollout 更适合做主结果，因为它避免某个任务的数据量或难度主导整体分数。 |
| 策略一致性 / 外在灵巧趋势 | **Extrinsic Dexterity Consistency (EDC)** 或 **Grasp-enabling Strategy Consistency** | Extrinsic Dexterity Leverage | 统计 policy 是否呈现出与当前任务相匹配的 grasp-enabling behavior，例如 EdgeHang、WallLever、PressTilt。附件三中的 `right_dex` 就是该指标的逐 episode 人工核验版本；它不要求任务成功，而是判断模型是否有“创造可抓取空间”的正确行为趋势。 |
| 四阶段累计成功率 | **Phase Success Rates** | Spatiotemporal Execution Precision | 分别统计 P1–P4 的累计成功率，形成 failure funnel。该指标显示模型是卡在预操作、抓取、清道还是最终抬升。 |
| 平均最大完成阶段 | **Average Maximum Completed Phase** | Execution progress | 统计每个 rollout 到达的最深阶段并求平均，可作为紧凑的 progress score。它适合在正文图表中补充 phase success rates。 |
| 角色分配准确率 | **Role Assignment Accuracy** | Bimanual Role Coordination | 在有明确左右臂角色变化的任务 / 子集上，比较 GT arm scheme 与 rollout exec scheme，判断模型是否选用正确抓取臂。 |
| 角色条件成功率 | **Role-conditioned Task Success Rate** | Bimanual Role Coordination | 在 GT-left 与 GT-right 子集上分别统计成功率，用于区分“选对角色”与“选对后是否执行成功”。 |

记号上，令四个任务集合为 $\mathcal{T}=\{\tau_1,\tau_2,\tau_3,\tau_4\}$，每个任务的测试 rollout 数为 $N_\tau$，则可报告：

- $SR_\tau$：任务 $\tau$ 的最终任务成功率；
- $EDC_\tau$：任务 $\tau$ 中执行出正确 grasp-enabling strategy tendency 的比例，对应附件三的 `right_dex`；
- $PSR_{\tau,k}$：第 $k$ 个阶段的累计成功率，$k \in \{\text{PreManipulation}, \text{Grasp}, \text{ClearPath}, \text{Lift}\}$；
- $AMP_\tau$：任务 $\tau$ 的 average maximum completed phase，取值范围 $[0,4]$；
- $RA_{\tau}^{L}$、$RA_{\tau}^{R}$：GT-left / GT-right scheme 下的角色分配准确率；
- $RSR_{\tau}^{L}$、$RSR_{\tau}^{R}$：GT-left / GT-right scheme 下的最终任务成功率。

跨任务主平均建议写为：

- $mSR = \frac{1}{|\mathcal{T}|}\sum_{\tau \in \mathcal{T}} SR_\tau$；
- $mEDC = \frac{1}{|\mathcal{T}|}\sum_{\tau \in \mathcal{T}} EDC_\tau$；
- $mPSR_k = \frac{1}{|\mathcal{T}|}\sum_{\tau \in \mathcal{T}} PSR_{\tau,k}$；
- $mAMP = \frac{1}{|\mathcal{T}|}\sum_{\tau \in \mathcal{T}} AMP_\tau$。

其中 $EDC_\tau$ 可由人工观察轨迹或基于规则的行为分类器判定。对于 `bimanual_edge_phone`，正确策略通常是 EdgeHang；对于 `bimanual_pivot_phone`，正确策略通常是 WallLever；对于 `bimanual_pick_plate` 和 `bimanual_pick_fork`，正确策略通常是 PressTilt。未来如果某个任务允许多种合理 grasp-enabling strategy，则可以将“正确策略”从单一标签扩展为 acceptable strategy set。

### 3.2 Compact Single-task Training & Evaluation

Single-task setting 仍应保留，但在 RAL 篇幅受限时可压缩为最终成功率表。它的作用是回答：当策略类型、物体和 affordance 机制基本固定时，模型是否能够拟合该任务的基本动作模式。

建议正文中只报告：

| Metric | Suggested name | Explanation |
|---|---|---|
| 单任务最终成功率 | **Single-task Task Success Rate** | 每个模型分别在单一任务数据上训练，并在同一任务测试集评估最终成功率。 |
| 可选附录诊断 | **Single-task Phase / Role Diagnostics** | 如果篇幅允许，可在 appendix 展开四阶段成功率和角色条件成功率；正文不强制。 |

这样可以避免实验部分过度分散，同时让 multi-task results 成为本文主要论证对象：Bi-GRACE 不是只测试“一个任务能不能学会”，而是测试模型能否在多个 graspability-transformation mechanisms 之间做出稳定适配。

### 3.3 Per-episode Diagnostic Record

附件三中的逐 episode 表格不建议完整搬入论文正文，但它定义了本文指标的底层观察单元。每个 rollout 至少包含：

- `task`：任务名；
- `right_dex` / `EDC`：是否呈现正确 grasp-enabling behavior tendency；
- `final`：最终是否成功；
- `p1`–`p4`：四阶段是否曾经完成；
- `max_done` / `max_completed_phase`：最深完成阶段；
- `GT arm scheme`：演示收集时的角色分配；
- `exec scheme`：rollout 中实际抓取臂；
- `video`：对应成功 / 失败视频。

该 per-episode record 的价值在于为结果表提供可复查证据：如果最终成功率低，我们可以进一步判断是模型完全没有创造 graspable state，还是已经完成预操作但没能稳定抓取，或者角色选择错误导致双臂干涉。

---

## 4. How the Metrics Support the Paper’s Claims

### 4.1 Diagnosing Extrinsic Dexterity Leverage

`strategy_type` 与 EDC / right_dex 指标用于回答：模型是否真的在利用外在灵巧，而不是只输出某段接近训练轨迹的动作。EdgeHang、WallLever 和 PressTilt 不是三个动作名字，而是三种创造可抓取条件的物理机制：桌边制造悬空，墙面提供撬动约束，按压使扁平或细长物体的一端翘起。

因此，EDC 应与 final success 分开报告。一个 rollout 即使最终失败，也可能已经表现出正确的 grasp-enabling tendency；反之，一个模型在某些简单 episode 中偶然成功，也不一定说明其学会了稳定的外在灵巧机制。附件三中 `right_dex` 的定义正适合作为这一诊断：它只判断 policy 是否按照预期策略产生“腾出抓取空间”的趋势，而不直接等同于任务成功。

### 4.2 Diagnosing Spatiotemporal Execution Precision

Phase success rates 是 Bi-GRACE 相比只报 final success 的关键优势。四阶段指标将 long-horizon BOG execution 拆成可解释的进度链：

- P1 低：模型没有学会创造可抓取条件；
- P1 高但 P2 低：模型能产生预操作趋势，但无法在新状态下建立稳定抓取；
- P2 高但 P3 低：问题集中在辅助臂撤离、清道或双臂避让；
- P3 高但 P4 低：更可能是抓取稳定性、抬升轨迹或夹爪控制问题；
- P4 高：通常对应最终任务完成。

从当前 ACT / PPI 逐 episode 记录的逻辑看，这类 phase funnel 能揭示 final success 隐藏的信息：例如，一个 3D-aware 或 pointflow-aware 模型可能更容易到达 PreManipulation，但仍可能在 Grasp / Lift 阶段掉点；一个 image-based 模型可能呈现某些正确动作趋势，但无法稳定完成阶段转换。正文中应将这些作为“Bi-GRACE 评估价值”的核心，而不是只比较哪个模型 final success 更高。

### 4.3 Diagnosing Bimanual Role Coordination

双臂遮挡抓取的成功不仅取决于“哪个点可以抓”，还取决于“哪只手应该抓、哪只手应该辅助、两只手如何避免互相阻挡”。因此，role assignment accuracy 和 role-conditioned success 应被解释为 embodiment-specific coordination metrics。

建议正文中特别强调 role metrics 与 final success 的互补性：

- 如果 role assignment accuracy 低，说明模型倾向固定用手或没有理解当前布局下的可达性约束；
- 如果 role assignment accuracy 高但 role-conditioned success 低，说明模型虽然选对抓取臂，但在该角色下的预操作、抓取或清道控制仍失败；
- 如果某一类 role scheme 成功率显著低，说明模型存在 left/right asymmetry、workspace bias 或训练数据覆盖不足。

附件三中 GT arm scheme 与 exec scheme 的逐 episode 记录尤其适合支撑这一点。正文中可将 role diagnostics 放在主结果之后，作为一张紧凑的 secondary table，重点报告具有左右角色变化的任务 / 子集。

### 4.4 Spatial Keypoints as Minimal Semantic Skeleton

contact / grasp / affordance 三类关键点可以被视为遮挡抓取的最小空间语义骨架：

- **contact point**：预操作应该作用在物体哪里；
- **grasp point**：最终应该在哪里建立稳定抓取；
- **affordance point**：环境中哪个位置提供边缘、墙面、支撑或约束。

在当前 baseline 中，它们不作为统一训练监督输入模型，而是用于可视化轨迹、解释模型动作落点是否靠近真正决定成功的物体—环境交互区域，以及为失败案例提供直观证据。对于 PPI，pointflow 属于方法自身需要的训练信号；即便 pointflow 可由空间几何 / 关键点相关信息构造，也应与 Bi-GRACE 诊断关键点的 evaluation usage 区分开。

---

## 5. Experimental Value and Result Narration

当前实验部分的价值不应只写成“我们测试了 ACT 和 PPI”，而应写成：Bi-GRACE 使同一组 rollouts 能够从 final success、extrinsic-dexterity strategy、phase progress、role coordination 和 spatial keypoint alignment 等多个方面被解释，从而揭示不同模型失败机制的差异。

建议正文结果叙述采用以下逻辑：

1. **先报 final success，但不止步于 final success**：给出每任务成功率和 macro average，作为 headline result。
2. **再报 EDC / strategy consistency**：判断模型是否产生正确 grasp-enabling behavior，即是否真的在尝试利用 EdgeHang、WallLever 或 PressTilt。
3. **用 phase funnel 定位瓶颈**：解释成功率差异来自 PreManipulation、Grasp、ClearPath 还是 Lift。
4. **用 role metrics 检查双臂协调**：说明模型是否存在固定用手偏置，是否在 left/right grasping schemes 下表现不同。
5. **用 keypoint/video visualization 给出机制解释**：展示成功/失败案例中接触点、抓取点、affordance point 与机器人动作的空间关系。

这种组织方式能让实验结论从“哪个模型更高”提升为“哪类模型更擅长利用外在结构、哪类模型更擅长空间接触推理、哪类模型更容易在角色分配或阶段转换中失败”。这正对应引言中的三项核心能力，也能解释为什么 Bi-GRACE 需要额外的结构化诊断标注。

---

## 6. Extensibility and Future Training Use

### 6.1 Extensibility Beyond the Current Four Tasks

Bi-GRACE 的核心可扩展性来自它没有绑定到具体物体，而是绑定到 BOG 的机制结构：

```text
ungraspable / occluded initial state
→ graspability transformation through extrinsic dexterity
→ stable grasp
→ bimanual clear path
→ lift / task completion
```

只要新任务仍包含“先改变物体—环境状态，再完成抓取”的过程，四阶段结构和三类关键点就可以基本复用。未来扩展可以从三方面进行：

- **New objects**：从手机、盘子、叉子扩展到卡片、薄片工具、包装袋、餐具、扁平盒子、柔性物体等；
- **New environmental affordances**：从桌边和墙面扩展到抽屉边缘、容器边界、夹具、橱柜边、凹槽、角落，甚至由另一只机械臂主动形成的支撑面；
- **New grasp-enabling mechanisms**：从 EdgeHang、WallLever、PressTilt 扩展到 slide-out、roll-over、pin-and-grasp、two-arm-scoop、push-to-corner、wedge-and-lift 等。

因此，Bi-GRACE 不只是一个固定四任务 benchmark，而是一套可复用的 BOG evaluation schema：用 strategy label 描述“采用什么机制”，用 phase label 描述“进展到哪里”，用 spatial keypoints 描述“在哪里与物体 / 环境交互”，用 role label 描述“双臂如何分工”。

### 6.2 Future Use of Annotations for Training

当前 baseline 不使用 Bi-GRACE 额外诊断标注，以保证公平评估。但这些标注可以自然转化为未来模型的 guidance signals。可能方向包括：

- **Strategy-conditioned policies**：将 `strategy_type` 作为条件输入，让模型显式选择或执行 EdgeHang / WallLever / PressTilt 等机制；
- **Auxiliary strategy prediction**：训练模型预测当前场景适合的 grasp-enabling mechanism，作为 representation learning 辅助任务；
- **Phase-conditioned control**：将 `phase_type` 注入 policy，使 action generation 根据 PreManipulation / Grasp / ClearPath / Lift 采用不同控制模式；
- **Progress-aware training**：用 phase transition label 训练进度预测器、阶段切换器或 termination classifier，减少长时序 imitation 中的阶段错位；
- **Keypoint prediction and pose injection**：预测 contact / grasp / affordance keypoints，并将其作为 pose token、attention anchor 或 action decoder 的空间条件；
- **Affordance-aware attention supervision**：利用 2D/3D keypoint projection 监督视觉模型关注真正决定成功的物体—环境交互区域；
- **Role-conditioned policy / role planner**：将 `left_grasper` / `right_grasper` scheme 用于训练角色选择器或双臂 motion-cost-aware planner；
- **Failure explanation and relabeling**：用 phase、role、keypoint diagnostics 对失败 rollout 进行自动归因，为 curriculum learning 或 targeted data collection 提供依据。

这部分应与引言第三条贡献呼应：Bi-GRACE 的标注不只是为了现在的评测，也为后续“条件注入、关键点注入、进度感知控制、角色选择学习”等新模型结构提供了可复用训练接口。

---

## 7. Suggested Reporting Layout for the Paper

考虑 RAL 篇幅，建议实验部分组织如下：

1. **Task and annotation overview table**：列出四个任务、物体、strategy type、环境 affordance、phase structure、是否存在动态角色变化。该表服务于 benchmark 介绍，不展开所有指标。
2. **Baseline taxonomy table**：列出 ACT、PPI 作为当前主实验模型，同时保留 Diffusion Policy、PerAct / PerAct²、3D Diffuser Actor、π0.5 / openpi 作为 broader protocol / future baseline pool；表中明确每个方法的输入表示、动作接口、是否显式建模 3D、是否使用 Bi-GRACE diagnostic labels 训练。
3. **Main multi-task result table**：每个模型一行，按任务报告 final success，并报告 macro average；同时可加入 EDC / strategy consistency 和 avg max phase。
4. **Phase funnel table or figure**：按模型和任务展示 P1–P4 成功率，突出 failure bottleneck。
5. **Role diagnostic table**：在有明确左右角色变化的任务 / 子集上报告 role assignment accuracy 和 role-conditioned success。
6. **Single-task compact table**：只报告 single-task final success；详细 phase / role diagnostics 可移到 appendix 或 supplementary。
7. **Qualitative visualization**：用视频帧或轨迹可视化展示 contact / grasp / affordance keypoints，与成功 / 失败 rollout 的动作落点对齐。

这种组织方式既保留原始 baseline taxonomy 和指标定义，又更贴近当前论文立意：Bi-GRACE 的关键贡献不是多报几个成功率，而是提供一个能够系统检验外在灵巧、双臂角色协调和时空执行精度的 bimanual occluded grasping benchmark。

---

## 8. Suggested Introduction Contributions

下面三条可直接替换当前引言最后的 placeholder：

1. **We introduce Bi-GRACE, a compact RLBench2-compatible benchmark for bimanual occluded grasping that evaluates graspability transformation through environment- and geometry-assisted pre-grasp manipulation, adaptive arm-role assignment, and stage/keypoint-aware execution diagnostics.**

2. **We benchmark representative image-based and 3D-aware visuomotor policies under a unified multi-task protocol, showing through success, grasp-enabling strategy, phase-progress, and role-conditioned metrics that final success alone hides distinct failures in extrinsic-dexterity use, phase transitions, and bimanual role coordination.**

3. **We design Bi-GRACE as an extensible annotation schema for broader objects, affordances, and grasp-enabling mechanisms, where strategy, phase, role, and spatial keypoint labels can also serve as future guidance signals for conditional policies, keypoint/pose injection, and progress-aware training.**
