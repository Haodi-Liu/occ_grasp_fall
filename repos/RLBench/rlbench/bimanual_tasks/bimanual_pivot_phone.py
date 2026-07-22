"""
BimanualPivotPhone - 靠墙撬起抓取任务

策略类型: 2 (WallLever)
四阶段执行：PreManipulation -> Grasp -> ClearPath -> Lift

预操作：推手机到墙壁，利用墙面作为支点撬起
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import logging
import numpy as np

from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.errors import ConfigurationPathError
from pyrep.const import ConfigurationPathAlgorithms as Algos
from rlbench.backend.conditions import JointCondition, Condition
from rlbench.backend.task import Task, BimanualTask
from rlbench.backend.robot import BimanualRobot


# ============================================================
# 常量定义
# ============================================================

STRATEGY_NAMES = {
    1: "EdgeHang",
    2: "WallLever",
    3: "PressTilt",
}

PHASE_NAMES = {
    1: "PreManipulation",  # 预操作：推手机到墙壁撬起
    2: "Grasp",            # 抓取：抓住翘起部分
    3: "ClearPath",        # 清道：辅助臂移开
    4: "Lift",             # 拿起：抬起手机
    5: "Complete",         # 四阶段全部完成
}


# ============================================================
# 阶段成功条件类
# ============================================================

class GraspPointHeightCondition(Condition):
    """
    抓取点高度条件：检查 grasp_pt 是否达到足够高度，表明物体已翘起。
    """

    def __init__(self,
                 grasp_pt_dummy: Dummy,
                 target_object: Shape,
                 min_height: float,
                 velocity_threshold: float = 0.02,
                 required_stable_frames: int = 5):
        self.grasp_pt_dummy = grasp_pt_dummy
        self.target_object = target_object
        self.min_height = min_height
        self.velocity_threshold = velocity_threshold
        self.required_stable_frames = required_stable_frames
        self.stable_count = 0

    def condition_met(self):
        grasp_pt_z = self.grasp_pt_dummy.get_position()[2]
        height_met = grasp_pt_z >= self.min_height

        if not height_met:
            self.stable_count = 0
            return False, False

        linear_vel, angular_vel = self.target_object.get_velocity()
        total_vel = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel) * 0.1

        if total_vel < self.velocity_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        is_stable = self.stable_count >= self.required_stable_frames

        return height_met and is_stable, False

    def reset(self):
        self.stable_count = 0


class StableGraspCondition(Condition):
    """
    稳定抓取条件：检查目标物体是否被夹爪稳定抓取。
    """

    def __init__(self,
                 gripper,
                 target_object: Shape,
                 velocity_threshold: float = 0.01,
                 required_stable_frames: int = 5):
        self.gripper = gripper
        self.target_object = target_object
        self.velocity_threshold = velocity_threshold
        self.required_stable_frames = required_stable_frames
        self.stable_count = 0
        self._target_handle = target_object.get_handle()

    def condition_met(self):
        grasped_objects = self.gripper.get_grasped_objects()
        is_grasped = any(
            obj.get_handle() == self._target_handle
            for obj in grasped_objects
        )

        if not is_grasped:
            self.stable_count = 0
            return False, False

        linear_vel, angular_vel = self.target_object.get_velocity()
        total_vel = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel) * 0.1

        if total_vel < self.velocity_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        is_stable = self.stable_count >= self.required_stable_frames

        return is_grasped and is_stable, False

    def reset(self):
        self.stable_count = 0


class ClearPathCondition(Condition):
    """
    清道条件：检查辅助臂是否已移开，与目标物体保持足够距离。
    """

    def __init__(self,
                 aux_gripper,
                 target_object: Shape,
                 aux_tip_dummy: Dummy,
                 lift_waypoints: List[Dummy] = None,
                 min_clearance: float = 0.15):
        self.aux_gripper = aux_gripper
        self.target_object = target_object
        self.aux_tip_dummy = aux_tip_dummy
        # 保留参数兼容旧调用；online 评估不再依赖预设 lift waypoint。
        self.lift_waypoints = lift_waypoints or []
        self.min_clearance = min_clearance

    def condition_met(self):
        if len(self.aux_gripper.get_grasped_objects()) > 0:
            return False, False

        aux_tip_pos = np.array(self.aux_tip_dummy.get_position())
        target_pos = np.array(self.target_object.get_position())
        distance_to_target = np.linalg.norm(aux_tip_pos - target_pos)

        if distance_to_target < self.min_clearance:
            return False, False

        return True, False

    def reset(self):
        pass


class LiftedCondition(Condition):
    """
    抬起条件：检查物体是否被抬起到目标高度。
    """

    def __init__(self, target_object: Shape, min_height: float = 0.9):
        self.target_object = target_object
        self.min_height = min_height

    def condition_met(self):
        pos = self.target_object.get_position()
        return pos[2] >= self.min_height, False

    def reset(self):
        pass


# ============================================================
# 双臂角色选择器
# ============================================================

class ArmRoleSelector:
    """
    双臂角色选择器：通过选择路径点方案来确定臂角色分配。

    核心职责：从两套路径点方案（right_grasper / left_grasper）中选择一个。
    方案确定后，臂角色自动确定：
    - right_grasper 方案：右臂=grasper，左臂=pusher
    - left_grasper 方案：左臂=grasper，右臂=pusher

    选择策略：
    1. 可行性优先：检查每套方案的 pusher 臂到关键 pusher 路径点是否可达
       （pusher 先执行，其路径点位置在物体移动前是稳定的）
    2. 成本次优：两套都可行时，选择总执行成本更低的方案
       （pusher 成本权重 1.5x，因为先执行且路径点位置稳定）
    """

    def __init__(self,
                 robot: BimanualRobot,
                 right_tip_name: str = "Panda_rightArm_tip",
                 left_tip_name: str = "Panda_leftArm_tip",
                 position_weight: float = 1.0,
                 orientation_weight: float = 0.5,
                 pusher_cost_weight: float = 1.5):
        self.robot = robot
        self.right_tip_name = right_tip_name
        self.left_tip_name = left_tip_name
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.pusher_cost_weight = pusher_cost_weight

    def select_scheme(self, waypoint_sets: Dict[str, Dict[str, List[str]]],
                      critical_pusher_indices: List[int] = [0]
                      ) -> Tuple[str, Dict[str, str]]:
        """
        选择最优的路径点方案。

        Args:
            critical_pusher_indices: 需要进行可行性验证的 pusher 路径点索引列表
                                     默认[0]表示只检查第一个 pusher 路径点

        Returns:
            Tuple[str, Dict[str, str]]: (选中的方案名称, 角色分配字典)
        """
        try:
            # ========== Step 1: 检查两套方案的可行性（检查 pusher 臂）==========
            right_feasible, right_reason = self._check_scheme_feasibility(
                'right_grasper', waypoint_sets, critical_pusher_indices
            )
            left_feasible, left_reason = self._check_scheme_feasibility(
                'left_grasper', waypoint_sets, critical_pusher_indices
            )

            logging.info(f"ArmRoleSelector: right_grasper feasible={right_feasible} ({right_reason}), "
                        f"left_grasper feasible={left_feasible} ({left_reason})")

            # ========== Step 2: 根据可行性筛选 ==========
            if right_feasible and not left_feasible:
                logging.info("ArmRoleSelector: Only right_grasper feasible, selecting it")
                return 'right_grasper', {'grasper': 'right', 'pusher': 'left'}

            if left_feasible and not right_feasible:
                logging.info("ArmRoleSelector: Only left_grasper feasible, selecting it")
                return 'left_grasper', {'grasper': 'left', 'pusher': 'right'}

            if not right_feasible and not left_feasible:
                logging.warning("ArmRoleSelector: Both schemes infeasible, move on")

            # ========== Step 3: 两套都可行，基于成本选择 ==========
            right_cost = self._compute_scheme_cost('right_grasper', waypoint_sets)
            left_cost = self._compute_scheme_cost('left_grasper', waypoint_sets)

            logging.info(f"ArmRoleSelector: Cost analysis - "
                        f"right_grasper={right_cost:.4f}, left_grasper={left_cost:.4f}")

            if right_cost <= left_cost:
                logging.info("ArmRoleSelector: Selecting right_grasper (lower cost)")
                return 'right_grasper', {'grasper': 'right', 'pusher': 'left'}
            else:
                logging.info("ArmRoleSelector: Selecting left_grasper (lower cost)")
                return 'left_grasper', {'grasper': 'left', 'pusher': 'right'}

        except Exception as e:
            logging.warning(f"ArmRoleSelector failed: {e}, using default (right_grasper)")
            return 'right_grasper', {'grasper': 'right', 'pusher': 'left'}

    def _check_scheme_feasibility(self, scheme_name: str,
                                   waypoint_sets: Dict,
                                   critical_pusher_indices: List[int]
                                   ) -> Tuple[bool, str]:
        """
        检查指定方案的可行性（检查 pusher 臂到 pusher 路径点）。
        """
        if scheme_name not in waypoint_sets:
            return False, f"Scheme '{scheme_name}' not defined"

        scheme = waypoint_sets[scheme_name]
        pusher_wps = scheme.get('pusher', [])

        # 确定该方案对应的 pusher 臂
        # right_grasper 方案：pusher = left
        # left_grasper 方案：pusher = right
        if scheme_name == 'right_grasper':
            pusher_arm = self.robot.left_arm
            pusher_arm_name = 'left'
        else:
            pusher_arm = self.robot.right_arm
            pusher_arm_name = 'right'

        for idx in critical_pusher_indices:
            if idx >= len(pusher_wps):
                continue
            wp_name = pusher_wps[idx]
            path_ok, reason = self._check_path_feasibility(pusher_arm, wp_name)
            if not path_ok:
                return False, f"{pusher_arm_name} arm (pusher) cannot reach {wp_name}: {reason}"

        return True, "All checks passed"

    def _check_path_feasibility(self, arm, waypoint_name: str) -> Tuple[bool, str]:
        try:
            wp_dummy = Dummy(waypoint_name)
            position = wp_dummy.get_position()
            euler = wp_dummy.get_orientation()

            arm.get_path(
                position,
                euler=euler,
                ignore_collisions=False,
                trials=100,
                max_configs=10,
                trials_per_goal=10,
                algorithm=Algos.RRTConnect
            )
            return True, "Path found"

        except ConfigurationPathError as e:
            return False, "No collision-free path"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def _compute_scheme_cost(self, scheme_name: str, waypoint_sets: Dict) -> float:
        """
        计算指定方案的总执行成本。
        成本 = grasper臂成本 + pusher_cost_weight × pusher臂成本
        """
        scheme = waypoint_sets[scheme_name]
        grasper_wps = scheme.get('grasper', [])
        pusher_wps = scheme.get('pusher', [])

        if scheme_name == 'right_grasper':
            grasper_tip = Dummy(self.right_tip_name)
            pusher_tip = Dummy(self.left_tip_name)
        else:
            grasper_tip = Dummy(self.left_tip_name)
            pusher_tip = Dummy(self.right_tip_name)

        grasper_cost = self._compute_waypoint_cost(grasper_tip, grasper_wps)
        pusher_cost = self._compute_waypoint_cost(pusher_tip, pusher_wps)

        # Pusher 成本乘以额外权重（默认1.5）
        total_cost = grasper_cost + self.pusher_cost_weight * pusher_cost

        logging.debug(f"Scheme {scheme_name}: grasper_cost={grasper_cost:.4f}, "
                     f"pusher_cost={pusher_cost:.4f} (×{self.pusher_cost_weight}), "
                     f"total={total_cost:.4f}")

        return total_cost

    def _compute_waypoint_cost(self, tip_dummy: Dummy, waypoint_names: List[str]) -> float:
        total_cost = 0.0
        tip_pos = np.array(tip_dummy.get_position())
        tip_quat = np.array(tip_dummy.get_quaternion())

        for i, wp_name in enumerate(waypoint_names):
            try:
                wp_dummy = Dummy(wp_name)
                wp_pos = np.array(wp_dummy.get_position())
                wp_quat = np.array(wp_dummy.get_quaternion())

                pos_dist = np.linalg.norm(tip_pos - wp_pos)
                quat_dot = np.abs(np.dot(tip_quat, wp_quat))
                quat_dist = 1.0 - min(quat_dot, 1.0)

                weight = 2.0 if i == 0 else 1.0
                total_cost += (self.position_weight * pos_dist +
                              self.orientation_weight * quat_dist) * weight

            except Exception as e:
                logging.debug(f"Failed to compute cost for {wp_name}: {e}")
                continue

        return total_cost


# ============================================================
# 分阶段成功评估器
# ============================================================

class PhasedSuccessEvaluator:
    """
    分阶段成功条件评估器。
    """

    def __init__(self, stage_conditions: Dict[int, Condition]):
        self.stage_conditions = stage_conditions
        self.num_phases = 4
        self.current_phase = 1
        self.max_current_phase_reached = 1
        self._phase_completion_status = {i: False for i in range(1, self.num_phases + 1)}
        self._last_condition_status = {i: False for i in range(1, self.num_phases + 1)}

    def reset(self):
        self.current_phase = 1
        self.max_current_phase_reached = 1
        self._phase_completion_status = {i: False for i in range(1, self.num_phases + 1)}
        self._last_condition_status = {i: False for i in range(1, self.num_phases + 1)}
        for cond in self.stage_conditions.values():
            if hasattr(cond, 'reset'):
                cond.reset()

    def _sample_conditions_once(self) -> Dict[int, bool]:
        status = {}
        for phase_id in range(1, self.num_phases + 1):
            cond = self.stage_conditions.get(phase_id)
            status[phase_id] = bool(cond.condition_met()[0]) if cond is not None else False
        self._last_condition_status = status
        return status

    def _maintenance_met(self, phase: int, cond: Dict[int, bool]) -> bool:
        if phase == 1:
            return True
        if phase == 2:
            return cond[1]
        if phase in (3, 4):
            return cond[2]
        return True

    def _transition_met(self, phase: int, cond: Dict[int, bool]) -> bool:
        if phase == 1:
            return cond[1]
        if phase == 2:
            return cond[1] and cond[2]
        if phase == 3:
            return cond[2] and cond[3]
        if phase == 4:
            return cond[2] and cond[4]
        return False

    def evaluate_current_phase(self) -> Tuple[bool, int]:
        old_phase = self.current_phase

        if self.current_phase >= 5:
            return False, self.num_phases

        cond = self._sample_conditions_once()

        if not self._maintenance_met(self.current_phase, cond):
            self.current_phase = 1

        while self.current_phase <= self.num_phases and self._transition_met(self.current_phase, cond):
            self.current_phase += 1

        self.max_current_phase_reached = max(
            self.max_current_phase_reached, self.current_phase
        )
        for phase_id in range(1, self.num_phases + 1):
            self._phase_completion_status[phase_id] = (
                self.max_current_phase_reached > phase_id
            )

        changed = self.current_phase != old_phase
        completed_phase = min(max(self.current_phase - 1, 0), self.num_phases)
        return changed, completed_phase

    def get_current_phase(self) -> int:
        return self.current_phase

    def is_phase_completed(self, phase_id: int) -> bool:
        return self._phase_completion_status.get(phase_id, False)

    def is_task_successful(self) -> bool:
        return self.current_phase >= 5

    def get_phase_progress(self) -> Dict:
        return {
            'current_phase': self.current_phase,
            'current_phase_name': PHASE_NAMES.get(self.current_phase, "Unknown"),
            'max_current_phase_reached': self.max_current_phase_reached,
            'max_completed_phase': max(self.max_current_phase_reached - 1, 0),
            'total_phases': self.num_phases,
            'completed': self.is_task_successful(),
            'phase_status': self._phase_completion_status.copy(),
            'condition_status': self._last_condition_status.copy(),
        }


# ============================================================
# 任务类
# ============================================================

class BimanualPivotPhone(BimanualTask):
    """
    靠墙撬起抓取任务：双臂协作将手机推向墙壁并撬起抓取。

    策略类型: 2 (WallLever)

    路径点方案:
    - right_grasper: 右臂抓取（默认）
      - grasper路径点: waypoint0, 2, 4, 6
      - pusher路径点: waypoint1, 3, 5, 7, 8
    - left_grasper: 左臂抓取（镜像）
      - grasper路径点: waypoint0_a, 2_a, 4_a, 6_a
      - pusher路径点: waypoint1_a, 3_a, 5_a, 7_a, 8_a
    """

    STRATEGY_TYPE = 2  # WallLever策略

    def init_task(self) -> None:
        """初始化任务"""
        # ===== 获取场景对象 =====
        self.target_object = Shape('Phone')

        self.grasp_pt = None
        if Object.exists('grasp_pt'):
            self.grasp_pt = Dummy('grasp_pt')

        self.register_graspable_objects([self.target_object])

        # ===== 保存默认位置（用于位置限制）=====
        # 场景放置时以此为中心，限制随机化范围
        self._default_base_position = np.array(self.get_base().get_position())

        # ===== 初始化角色选择器 =====
        self.role_selector = ArmRoleSelector(
            robot=self.robot,
            right_tip_name="Panda_rightArm_tip",
            left_tip_name="Panda_leftArm_tip",
        )

        # ===== 定义两套路径点方案 =====
        # BimanualPivotPhone: 9个waypoints (0-8)
        # grasper: 4个 (0,2,4,6)
        # pusher: 5个 (1,3,5,7,8) - waypoint8 用于清道撤退
        self.waypoint_sets = {
            'right_grasper': {
                'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
                'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7', 'waypoint8']
            },
            'left_grasper': {
                'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
                'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a', 'waypoint8_a']
            }
        }

        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()

        self.phased_evaluator = None

        self.register_success_conditions([
            LiftedCondition(self.target_object, min_height=0.9)
        ])

    def _get_active_waypoints(self) -> Dict[str, List[str]]:
        """获取当前激活方案的路径点配置"""
        return self.waypoint_sets[self.active_waypoint_mode]

    def _setup_waypoint_mapping(self):
        """根据当前激活方案和角色分配设置 waypoint_mapping"""
        active_wps = self._get_active_waypoints()
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for role, arm in self.current_role_assignment.items():
            for wp in active_wps.get(role, []):
                self.waypoint_mapping[wp] = arm

    def _setup_phased_evaluator(self):
        """设置分阶段成功条件评估器"""
        if self.grasp_pt is None:
            self.phased_evaluator = None
            return

        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']

        pusher_gripper = (self.robot.left_gripper if pusher_arm == 'left'
                         else self.robot.right_gripper)
        grasper_gripper = (self.robot.right_gripper if grasper_arm == 'right'
                          else self.robot.left_gripper)

        grasper_tip_name = "Panda_rightArm_tip" if grasper_arm == 'right' else "Panda_leftArm_tip"
        pusher_tip_name = "Panda_leftArm_tip" if pusher_arm == 'left' else "Panda_rightArm_tip"

        try:
            grasper_tip = Dummy(grasper_tip_name)
            pusher_tip = Dummy(pusher_tip_name)
        except Exception:
            self.phased_evaluator = None
            return

        # 获取抬起阶段的路径点
        active_wps = self._get_active_waypoints()
        lift_waypoints = []
        lift_wp_name = active_wps['grasper'][-1]  # waypoint6 或 waypoint6_a
        if Object.exists(lift_wp_name):
            lift_waypoints.append(Dummy(lift_wp_name))

        # ====== 单独阶段条件定义 ======
        con1 = GraspPointHeightCondition(
            self.grasp_pt, self.target_object,
            min_height=0.8,
            velocity_threshold=0.2, required_stable_frames=3
        )
        con2 = StableGraspCondition(
            grasper_gripper, self.target_object,
            velocity_threshold=0.1, required_stable_frames=3
        )
        con3 = ClearPathCondition(
            pusher_gripper, self.target_object, pusher_tip,
            lift_waypoints=lift_waypoints, min_clearance=0.2
        )
        con4 = LiftedCondition(self.target_object, min_height=0.9)

        stage_conditions = {
            1: con1,
            2: con2,
            3: con3,
            4: con4,
        }

        self.phased_evaluator = PhasedSuccessEvaluator(stage_conditions)
        logging.info("PhasedSuccessEvaluator initialized successfully for BimanualPivotPhone")

    def init_episode(self, index: int) -> List[str]:
        """初始化episode"""
        self._variation_index = index
        self._step_count = 0

        # 随机选择基础旋转偏移：0 (Pose A) 或 π (Pose B)
        self._base_rotation_offset = np.random.choice([0, np.pi])

        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()

        return ['push the phone against the wall and pivot it to grasp']

    def base_rotation_bounds(self):
        """
        限制场景旋转到两个离散区域：
        - Pose A: 0° ± 20°  (offset=0)
        - Pose B: 180° ± 20° (offset=π)

        覆盖父类默认的 [-π, +π] 全范围旋转。

        注意：此方法会在 init_episode() 之前被调用（如 task_builder 按 "+" 时），
        因此在这里自行初始化 _base_rotation_offset 以确保两种配置都能出现。
        """
        if not hasattr(self, '_base_rotation_offset'):
            self._base_rotation_offset = np.random.choice([0, np.pi], p=[1, 0])

        offset = self._base_rotation_offset
        delta = np.deg2rad(15)  # 15° ≈ 0.262 rad
        min_rot = (0.0, 0.0, offset - delta)
        max_rot = (0.0, 0.0, offset + delta)

        return min_rot, max_rot

    def boundary_root(self):
        """返回场景的边界根对象，用于 SpawnBoundary 放置"""
        return self.get_base()

    def base_position_bounds(self):
        """
        返回位置偏移限制 (delta_x, delta_y)，单位为米。
        用于在 post_placement_setup() 中限制场景位置随机化范围。
        """
        # 推荐值：±0.03m (3cm) 到 ±0.05m (5cm)
        # 较小的值使场景更接近默认位置，较大的值增加多样性
        return 0.05, 0.05  # x方向±5cm, y方向±5cm

    # def step(self) -> None:
    #     """每个仿真步骤都会被调用，用于追踪阶段指标"""
    #     self._step_count += 1
    #     if self._step_count % 5 != 0:
    #         return

    #     # 先评估阶段条件，再获取当前阶段
    #     if self.phased_evaluator is not None:
    #         self.phased_evaluator.evaluate_current_phase()
    #     phase = self.get_current_phase()
    #     target_pos = np.array(self.target_object.get_position())
    #     grasp_pt_z = self.grasp_pt.get_position()[2] if self.grasp_pt else -1.0
    #     eval_ok = "Y" if self.phased_evaluator else "N"

    #     # 获取pusher tip位置
    #     pusher_arm = self.current_role_assignment.get('pusher', 'left')
    #     pusher_tip_name = "Panda_leftArm_tip" if pusher_arm == 'left' else "Panda_rightArm_tip"
    #     pusher_tip_pos = np.array(Dummy(pusher_tip_name).get_position())

    #     dist_to_target = np.linalg.norm(pusher_tip_pos - target_pos)

    #     # 获取lift waypoint距离
    #     active_wps = self._get_active_waypoints()
    #     lift_wp_name = active_wps['grasper'][-1]
    #     if Object.exists(lift_wp_name):
    #         lift_wp_pos = np.array(Dummy(lift_wp_name).get_position())
    #         dist_to_wp = np.linalg.norm(pusher_tip_pos - lift_wp_pos)
    #         wp_str = f"{dist_to_wp:.3f}"
    #     else:
    #         wp_str = "N/A"

    #     print(f"[Step {self._step_count:4d}] phase={phase} eval={eval_ok} | grasp_pt_z={grasp_pt_z:.3f} target_z={target_pos[2]:.3f} | dist_to_target={dist_to_target:.3f} | dist_to_wp={wp_str}")

    def post_placement_setup(self) -> None:
        """在场景随机放置后选择方案并设置评估器"""
        # 位置限制：将场景位置钳制到默认位置附近
        self._clamp_position_to_bounds()

        # 根据可行性和成本选择最优方案
        self.active_waypoint_mode, role_assignment = self.role_selector.select_scheme(
            self.waypoint_sets,
            critical_pusher_indices=[0]  # 检查第一个 pusher 路径点
        )
        if role_assignment != self.current_role_assignment:
            self.current_role_assignment = role_assignment
            self._setup_waypoint_mapping()
            logging.info(f"Scheme selected: {self.active_waypoint_mode}, "
                        f"roles: {self.current_role_assignment}")

        # ===== [临时] 仅收集 left_grasper 方案，否则跳过 =====
        # 恢复正常收集：注释掉下面2行
        # from rlbench.backend.exceptions import DemoError
        # if self.active_waypoint_mode != 'left_grasper':
        #     raise DemoError(f"Skipping: scheme={self.active_waypoint_mode}, want left_grasper", self)

        self._setup_phased_evaluator()
        if self.phased_evaluator:
            self.phased_evaluator.reset()

    def _clamp_position_to_bounds(self) -> None:
        """
        将场景位置钳制到默认位置的指定邻域内。
        在 _place_task() 随机放置后调用，确保位置不会偏离太远。

        注意：仅在 _base_rotation_offset == 0 (Pose A) 时生效。
        """
        # 仅对 Pose A (offset=0) 生效，Pose B (offset=π) 不限制位置
        if getattr(self, '_base_rotation_offset', 0) != 0:
            return

        if not hasattr(self, '_default_base_position'):
            logging.warning("_default_base_position not set, skipping position clamp")
            return

        delta_x, delta_y = self.base_position_bounds()
        base = self.get_base()
        current_pos = np.array(base.get_position())
        default_pos = self._default_base_position

        # 计算钳制后的位置
        clamped_x = np.clip(current_pos[0],
                            default_pos[0] - delta_x,
                            default_pos[0] + delta_x)
        clamped_y = np.clip(current_pos[1],
                            default_pos[1] - delta_y,
                            default_pos[1] + delta_y)

        # 只在 x 或 y 超出范围时才调整
        if current_pos[0] != clamped_x or current_pos[1] != clamped_y:
            new_pos = [clamped_x, clamped_y, current_pos[2]]
            base.set_position(new_pos)
            logging.info(f"Position clamped: ({current_pos[0]:.4f}, {current_pos[1]:.4f}) "
                        f"-> ({clamped_x:.4f}, {clamped_y:.4f})")

    def variation_count(self) -> int:
        return 1

    @property
    def execution_phases(self):
        """动态生成四阶段执行计划，使用当前激活方案的路径点"""
        active_wps = self._get_active_waypoints()
        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']

        pusher_wps = active_wps['pusher']   # 5个路径点
        grasper_wps = active_wps['grasper'] # 4个路径点

        return [
            # Phase 1: 推向墙壁并撬动 (4个pusher路径点)
            {'arm': pusher_arm, 'waypoints': pusher_wps[:4], 'wait_after': 0.5},
            # Phase 2: 抓取翘起部分 (3个grasper路径点)
            {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 1},
            # Phase 3: 辅助臂清道撤退 (1个pusher路径点)
            {'arm': pusher_arm, 'waypoints': [pusher_wps[4]], 'wait_after': 0.5},
            # Phase 4: 抬起物体 (1个grasper路径点)
            {'arm': grasper_arm, 'waypoints': [grasper_wps[3]], 'wait_after': 0.5},
        ]

    # ========== 策略和阶段标签接口（scene.py调用）==========

    def get_strategy_type(self) -> int:
        """返回策略类型"""
        return self.STRATEGY_TYPE

    def get_current_phase(self) -> int:
        """返回当前执行阶段"""
        if self.phased_evaluator is None:
            return 1
        return self.phased_evaluator.get_current_phase()

    def evaluate_phase_and_get_labels(self) -> Tuple[int, int]:
        """评估当前状态并返回策略类型和阶段类型标签"""
        strategy_type = self.STRATEGY_TYPE

        if self.phased_evaluator is None:
            phase_type = 1
        else:
            self.phased_evaluator.evaluate_current_phase()
            phase_type = self.phased_evaluator.get_current_phase()

        return strategy_type, phase_type

    def get_phase_progress(self) -> Dict:
        """获取阶段进度信息"""
        if self.phased_evaluator is None:
            return {'current_phase': 1, 'current_phase_name': PHASE_NAMES[1],
                    'max_current_phase_reached': 1, 'max_completed_phase': 0,
                    'total_phases': 4, 'completed': False,
                    'phase_status': {1: False, 2: False, 3: False, 4: False},
                    'condition_status': {1: False, 2: False, 3: False, 4: False}}
        return self.phased_evaluator.get_phase_progress()

    def get_role_assignment(self) -> Dict[str, str]:
        """返回当前的角色分配"""
        return self.current_role_assignment.copy()

    def get_active_scheme(self) -> str:
        """返回当前激活的路径点方案"""
        return self.active_waypoint_mode
