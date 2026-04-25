"""
BimanualEdgePhone - 边缘悬空抓取任务

策略类型: 1 (EdgeHang)
四阶段执行：PreManipulation -> Grasp -> ClearPath -> Lift
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

# 策略类型名称映射（整数 → 字符串，供阅读参考和日志）
STRATEGY_NAMES = {
    1: "EdgeHang",      # 边缘悬空抓取
    2: "WallLever",     # 靠墙撬起
    3: "PressTilt",     # 按压翘起
}

# 阶段类型名称映射
PHASE_NAMES = {
    1: "PreManipulation",  # 预操作：推手机到边缘
    2: "Grasp",            # 抓取：抓住悬空部分
    3: "ClearPath",        # 清道：辅助臂移开
    4: "Lift",             # 拿起：抬起手机
}


# ============================================================
# 阶段成功条件类
# ============================================================

class EdgeOverhangCondition(Condition):
    """
    边缘悬空条件：检查手机是否悬空在盒子边缘，且处于稳定状态。

    判断逻辑（使用相对坐标系）：
    1. 获取 phone_edge 相对于 box_edge 坐标系的位置
    2. 悬空量 = -relative_y（相对y为负表示悬空）
    3. 条件满足：悬空量 > min_overhang 且手机稳定

    坐标系说明：
    - phone_edge 和 box_edge 姿态相同
    - 在 box_edge 坐标系下，phone_edge 初始时 y>0（手机在盒子上）
    - 推出后 phone_edge 相对 y 变为负值
    """

    def __init__(self,
                 target_object: Shape,
                 box_edge_dummy: Dummy,
                 phone_edge_dummy: Dummy,
                 min_overhang: float = 0.05,
                 velocity_threshold: float = 0.02,
                 required_stable_frames: int = 5):
        self.target_object = target_object
        self.box_edge_dummy = box_edge_dummy
        self.phone_edge_dummy = phone_edge_dummy
        self.min_overhang = min_overhang
        self.velocity_threshold = velocity_threshold
        self.required_stable_frames = required_stable_frames
        self.stable_count = 0

    def condition_met(self):
        # 使用相对坐标系计算悬空量
        relative_pos = self.phone_edge_dummy.get_position(relative_to=self.box_edge_dummy)
        relative_y = relative_pos[1]

        # 悬空量 = -relative_y（相对y为负表示手机边缘已超出盒子边缘）
        overhang = -relative_y

        overhang_met = overhang > self.min_overhang

        if not overhang_met:
            self.stable_count = 0
            return False, False

        # 检查手机稳定性
        linear_vel, angular_vel = self.target_object.get_velocity()
        total_vel = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel) * 0.1

        if total_vel < self.velocity_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        is_stable = self.stable_count >= self.required_stable_frames

        return overhang_met and is_stable, False

    def reset(self):
        self.stable_count = 0


class StableGraspCondition(Condition):
    """
    稳定抓取条件：检查目标手机是否被夹爪稳定抓取。

    成功条件：
    1. 目标手机被指定夹爪抓取
    2. 手机速度足够小（稳定）
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
        # 检查目标手机是否被抓取
        grasped_objects = self.gripper.get_grasped_objects()
        is_grasped = any(
            obj.get_handle() == self._target_handle
            for obj in grasped_objects
        )

        if not is_grasped:
            self.stable_count = 0
            return False, False

        # 检查手机稳定性
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
    清道条件：检查辅助臂是否已移开，不会阻碍抓取臂后续的抬起路径。

    成功条件：
    1. 辅助臂夹爪已松开物体
    2. 辅助臂夹爪tip与目标物体保持足够距离
    3. 辅助臂tip与抓取臂后续所有路径点保持足够距离
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
        self.lift_waypoints = lift_waypoints or []
        self.min_clearance = min_clearance

    def condition_met(self):
        # 检查辅助臂是否已松开物体
        if len(self.aux_gripper.get_grasped_objects()) > 0:
            return False, False

        aux_tip_pos = np.array(self.aux_tip_dummy.get_position())

        # 检查与目标物体的距离
        target_pos = np.array(self.target_object.get_position())
        distance_to_target = np.linalg.norm(aux_tip_pos - target_pos)

        if distance_to_target < self.min_clearance:
            return False, False

        # 检查与所有后续路径点的距离
        for wp_dummy in self.lift_waypoints:
            wp_pos = np.array(wp_dummy.get_position())
            distance_to_wp = np.linalg.norm(aux_tip_pos - wp_pos)
            if distance_to_wp < self.min_clearance:
                return False, False

        return True, False

    def reset(self):
        pass


class LiftedCondition(Condition):
    """
    抬起条件：检查手机是否被抬起到目标高度。
    """

    def __init__(self, target_object: Shape, min_height: float = 1.1):
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

    Args:
        robot: BimanualRobot实例
        right_tip_name: 右臂tip dummy名称
        left_tip_name: 左臂tip dummy名称
        position_weight: 位置成本权重（默认1.0）
        orientation_weight: 姿态成本权重（默认0.5）
        pusher_cost_weight: pusher成本额外权重（默认1.5）
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
            waypoint_sets: 两套路径点方案配置，格式如：
                {
                    'right_grasper': {
                        'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
                        'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
                    },
                    'left_grasper': {
                        'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
                        'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
                    }
                }
            critical_pusher_indices: 需要进行可行性验证的 pusher 路径点索引列表
                                     默认[0]表示只检查第一个 pusher 路径点
                                     （pusher 先执行，其初始路径点位置稳定可靠）

        Returns:
            Tuple[str, Dict[str, str]]: (选中的方案名称, 角色分配字典)
            例如: ('left_grasper', {'grasper': 'left', 'pusher': 'right'})
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
        检查指定方案的可行性。

        检查内容：该方案的 pusher 臂能否到达关键 pusher 路径点。

        为何检查 pusher 而非 grasper：
        - Pusher 在 Phase 1 先执行，此时物体处于初始位置
        - Grasper 路径点附着在物体上，Phase 1 后位置会改变
        - 检查 pusher 的初始可达性更能反映实际执行情况

        Args:
            scheme_name: 方案名称 ('right_grasper' 或 'left_grasper')
            waypoint_sets: 路径点配置
            critical_pusher_indices: 需要验证的 pusher 路径点索引

        Returns:
            Tuple[bool, str]: (是否可行, 原因说明)
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
        else:  # left_grasper
            pusher_arm = self.robot.right_arm
            pusher_arm_name = 'right'

        # 对关键 pusher 路径点进行可达性验证
        for idx in critical_pusher_indices:
            if idx >= len(pusher_wps):
                continue
            wp_name = pusher_wps[idx]
            path_ok, reason = self._check_path_feasibility(pusher_arm, wp_name)
            if not path_ok:
                return False, f"{pusher_arm_name} arm (pusher) cannot reach {wp_name}: {reason}"

        return True, "All checks passed"

    def _check_path_feasibility(self, arm, waypoint_name: str) -> Tuple[bool, str]:
        """
        使用路径规划验证臂到路径点的可达性。

        参数与实际执行一致（trials=100）以确保检查结果与实际执行一致。

        Args:
            arm: PyRep Arm对象
            waypoint_name: 路径点名称

        Returns:
            Tuple[bool, str]: (是否可达, 原因说明)
        """
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

    def _compute_scheme_cost(self, scheme_name: str,
                              waypoint_sets: Dict) -> float:
        """
        计算指定方案的总执行成本。

        成本 = grasper臂成本 + pusher_cost_weight × pusher臂成本

        Pusher 成本权重更高的原因：
        - Pusher 在 Phase 1 先执行，其路径点位置稳定
        - Pusher 成本对方案选择更具参考价值

        Args:
            scheme_name: 方案名称
            waypoint_sets: 路径点配置

        Returns:
            float: 方案总成本
        """
        scheme = waypoint_sets[scheme_name]
        grasper_wps = scheme.get('grasper', [])
        pusher_wps = scheme.get('pusher', [])

        # 确定该方案对应的臂tip
        if scheme_name == 'right_grasper':
            grasper_tip = Dummy(self.right_tip_name)
            pusher_tip = Dummy(self.left_tip_name)
        else:  # left_grasper
            grasper_tip = Dummy(self.left_tip_name)
            pusher_tip = Dummy(self.right_tip_name)

        # 计算grasper臂到grasper路径点的成本
        grasper_cost = self._compute_waypoint_cost(grasper_tip, grasper_wps)

        # 计算pusher臂到pusher路径点的成本（乘以权重）
        pusher_cost = self._compute_waypoint_cost(pusher_tip, pusher_wps)

        # Pusher 成本乘以额外权重（默认1.5）
        total_cost = grasper_cost + self.pusher_cost_weight * pusher_cost

        logging.debug(f"Scheme {scheme_name}: grasper_cost={grasper_cost:.4f}, "
                     f"pusher_cost={pusher_cost:.4f} (×{self.pusher_cost_weight}), "
                     f"total={total_cost:.4f}")

        return total_cost

    def _compute_waypoint_cost(self, tip_dummy: Dummy,
                                waypoint_names: List[str]) -> float:
        """
        计算tip到一组路径点的距离成本。

        成本 = Σ (position_weight * 位置距离 + orientation_weight * 姿态差异) * 路径点权重
        其中第一个路径点权重为2.0（更重要），其余为1.0

        Args:
            tip_dummy: 臂tip的Dummy对象
            waypoint_names: 路径点名称列表

        Returns:
            float: 距离成本
        """
        total_cost = 0.0
        tip_pos = np.array(tip_dummy.get_position())
        tip_quat = np.array(tip_dummy.get_quaternion())

        for i, wp_name in enumerate(waypoint_names):
            try:
                wp_dummy = Dummy(wp_name)
                wp_pos = np.array(wp_dummy.get_position())
                wp_quat = np.array(wp_dummy.get_quaternion())

                # 位置距离
                pos_dist = np.linalg.norm(tip_pos - wp_pos)

                # 姿态差异（使用四元数点积）
                quat_dot = np.abs(np.dot(tip_quat, wp_quat))
                quat_dist = 1.0 - min(quat_dot, 1.0)

                # 第一个路径点权重更高
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

    特点：
    - 每个阶段的成功条件累积包含之前阶段的条件
    - 一个阶段的圆满完成预示着下一阶段的开始
    - 使用整数标签表示阶段 (1, 2, 3, 4)
    """

    def __init__(self, phase_conditions: Dict[int, List[Condition]]):
        self.phase_conditions = phase_conditions
        self.num_phases = len(phase_conditions)
        self.current_phase = 1
        self._phase_completion_status = {i: False for i in range(1, self.num_phases + 1)}

    def reset(self):
        """重置评估器状态"""
        self.current_phase = 1
        self._phase_completion_status = {i: False for i in range(1, self.num_phases + 1)}
        for conditions in self.phase_conditions.values():
            for cond in conditions:
                if hasattr(cond, 'reset'):
                    cond.reset()

    def evaluate_current_phase(self) -> Tuple[bool, int]:
        """评估当前阶段是否完成

        修复：一次调用中评估所有可完成的阶段，避免因跳帧导致阶段漏记录
        """
        if self.current_phase > self.num_phases:
            return True, self.num_phases

        any_completed = False
        last_completed_phase = 0

        while self.current_phase <= self.num_phases:
            conditions = self.phase_conditions.get(self.current_phase, [])
            all_met = all(cond.condition_met()[0] for cond in conditions)

            if all_met:
                self._phase_completion_status[self.current_phase] = True
                last_completed_phase = self.current_phase
                self.current_phase += 1
                any_completed = True
            else:
                break

        if any_completed:
            return True, last_completed_phase
        return False, self.current_phase

    def get_current_phase(self) -> int:
        """获取当前阶段ID"""
        return min(self.current_phase, self.num_phases)

    def is_phase_completed(self, phase_id: int) -> bool:
        """检查指定阶段是否已完成"""
        return self._phase_completion_status.get(phase_id, False)

    def is_task_successful(self) -> bool:
        """检查任务是否整体成功"""
        return self.current_phase > self.num_phases

    def get_phase_progress(self) -> Dict:
        """获取阶段进度信息（用于评估时记录）"""
        return {
            'current_phase': self.get_current_phase(),
            'total_phases': self.num_phases,
            'completed': self.is_task_successful(),
            'phase_status': self._phase_completion_status.copy()
        }


# ============================================================
# 任务类
# ============================================================

class BimanualEdgePhone(BimanualTask):
    """
    边缘悬空抓取任务：双臂协作将手机从盒子边缘抓取。

    策略类型: 1 (EdgeHang)

    路径点方案:
    - right_grasper: 右臂抓取（默认）
      - grasper路径点: waypoint0, 2, 4, 6
      - pusher路径点: waypoint1, 3, 5, 7
    - left_grasper: 左臂抓取（镜像）
      - grasper路径点: waypoint0_a, 2_a, 4_a, 6_a
      - pusher路径点: waypoint1_a, 3_a, 5_a, 7_a
    """

    STRATEGY_TYPE = 1  # EdgeHang策略

    def init_task(self) -> None:
        """初始化任务"""
        # ===== 获取场景对象 =====
        self.target_object = Shape('Phone')

        # 尝试获取边缘标记dummies（用于条件检测）
        self.box_edge = None
        self.phone_edge = None
        if Object.exists('box_edge'):
            self.box_edge = Dummy('box_edge')
        if Object.exists('phone_edge'):
            self.phone_edge = Dummy('phone_edge')

        # 注册可抓取对象
        self.register_graspable_objects([self.target_object])

        # ===== 初始化角色选择器 =====
        self.role_selector = ArmRoleSelector(
            robot=self.robot,
            right_tip_name="Panda_rightArm_tip",
            left_tip_name="Panda_leftArm_tip",
        )

        # ===== 定义两套路径点方案 =====
        self.waypoint_sets = {
            'right_grasper': {
                'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
                'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
            },
            'left_grasper': {
                'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
                'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
            }
        }

        # 当前激活的方案（默认右臂抓取）
        self.active_waypoint_mode = 'right_grasper'

        # 当前角色分配（从方案推导）
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}

        # 设置waypoint映射
        self._setup_waypoint_mapping()

        # 分阶段评估器（在post_placement_setup中设置）
        self.phased_evaluator = None

        # 注册最终成功条件
        self.register_success_conditions([
            LiftedCondition(self.target_object, min_height=1.1)
        ])

    def _get_active_waypoints(self) -> Dict[str, List[str]]:
        """获取当前激活方案的路径点配置"""
        return self.waypoint_sets[self.active_waypoint_mode]

    def _setup_waypoint_mapping(self):
        """
        根据当前激活方案和角色分配设置 waypoint_mapping。

        waypoint_mapping 告诉 scene.py 的 _get_waypoints() 方法
        每个路径点应该由哪个臂执行。
        """
        active_wps = self._get_active_waypoints()
        self.waypoint_mapping = defaultdict(lambda: 'right')

        for role, arm in self.current_role_assignment.items():
            for wp in active_wps.get(role, []):
                self.waypoint_mapping[wp] = arm

    def _setup_phased_evaluator(self):
        """设置分阶段成功条件评估器"""
        # 检查必要的对象是否存在
        if self.box_edge is None or self.phone_edge is None:
            logging.warning("box_edge or phone_edge not found, phased evaluator disabled")
            self.phased_evaluator = None
            return

        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']

        pusher_gripper = (self.robot.left_gripper if pusher_arm == 'left'
                         else self.robot.right_gripper)
        grasper_gripper = (self.robot.right_gripper if grasper_arm == 'right'
                          else self.robot.left_gripper)

        # 获取tip dummies
        grasper_tip_name = "Panda_rightArm_tip" if grasper_arm == 'right' else "Panda_leftArm_tip"
        pusher_tip_name = "Panda_leftArm_tip" if pusher_arm == 'left' else "Panda_rightArm_tip"

        try:
            grasper_tip = Dummy(grasper_tip_name)
            pusher_tip = Dummy(pusher_tip_name)
        except Exception as e:
            logging.warning(f"Failed to get tip dummies: {e}, phased evaluator disabled")
            self.phased_evaluator = None
            return

        # 获取抬起阶段的路径点（根据激活方案选择）
        active_wps = self._get_active_waypoints()
        lift_waypoints = []
        lift_wp_name = active_wps['grasper'][-1]  # waypoint6 或 waypoint6_a
        if Object.exists(lift_wp_name):
            lift_waypoints.append(Dummy(lift_wp_name))

        # ====== 阶段条件定义 ======
        # Phase 1: 悬空条件
        phase1_conditions = [
            EdgeOverhangCondition(
                self.target_object, self.box_edge, self.phone_edge,
                min_overhang=0.05, velocity_threshold=0.2, required_stable_frames=3
            )
        ]

        # 共享的稳定抓取条件
        stable_grasp_condition = StableGraspCondition(
            grasper_gripper, self.target_object,
            velocity_threshold=0.1, required_stable_frames=3
        )

        # Phase 2: 稳定抓取
        phase2_conditions = [stable_grasp_condition]

        # Phase 3: 持续抓取 + 清道
        phase3_conditions = [
            stable_grasp_condition,
            ClearPathCondition(
                pusher_gripper, self.target_object, pusher_tip,
                lift_waypoints=lift_waypoints, min_clearance=0.34
            )
        ]

        # Phase 4: 抬起
        phase4_conditions = [
            LiftedCondition(self.target_object, min_height=1.1)
        ]

        phase_conditions = {
            1: phase1_conditions,
            2: phase2_conditions,
            3: phase3_conditions,
            4: phase4_conditions
        }

        self.phased_evaluator = PhasedSuccessEvaluator(phase_conditions)
        logging.info("PhasedSuccessEvaluator initialized successfully")

    def init_episode(self, index: int) -> List[str]:
        """
        初始化episode。

        注意：方案选择在 post_placement_setup() 中执行，
        因为需要在场景随机放置之后才能正确评估可行性和成本。
        """
        self._variation_index = index
        self._step_count = 0

        # 重置为默认方案（将在 post_placement_setup 中更新）
        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()

        return ['push the phone over the box edge and grasp it from below']

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

    #     print(f"[Step {self._step_count:4d}] phase={phase} | target_z={target_pos[2]:.3f} | dist_to_target={dist_to_target:.3f} | dist_to_wp={wp_str}")

    def post_placement_setup(self) -> None:
        """
        在场景随机放置后选择方案并设置评估器。

        此方法由 scene.py 在 _place_task() 之后、validate() 之前调用。
        此时场景已经被随机放置，可以正确评估可行性和成本。
        """
        # ===== [强制测试] 使用 left_grasper 方案 =====
        # 恢复自动选择：注释掉下面4行，取消注释自动选择部分
        # self.active_waypoint_mode = 'left_grasper'
        # self.current_role_assignment = {'grasper': 'left', 'pusher': 'right'}
        # self._setup_waypoint_mapping()
        # logging.info(f"[FORCE TEST] Using left_grasper scheme")

        # ===== [自动选择] 根据可行性和成本选择最优方案 =====
        self.active_waypoint_mode, role_assignment = self.role_selector.select_scheme(
            self.waypoint_sets,
            critical_pusher_indices=[0]  # 检查第一个 pusher 路径点（Phase 1 起始点）
        )
        # 更新角色分配
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

        # ===== 设置分阶段评估器 =====
        self._setup_phased_evaluator()
        if self.phased_evaluator:
            self.phased_evaluator.reset()

    def variation_count(self) -> int:
        return 1

    @property
    def execution_phases(self):
        """
        动态生成四阶段执行计划，使用当前激活方案的路径点。

        返回的结构供 scene.py 的 execute_waypoints_bimanual_phased() 使用。
        """
        active_wps = self._get_active_waypoints()
        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']

        pusher_wps = active_wps['pusher']   # 4个路径点
        grasper_wps = active_wps['grasper'] # 4个路径点

        return [
            # Phase 1: Pusher推动物体悬空 (3个路径点)
            {'arm': pusher_arm, 'waypoints': pusher_wps[:3], 'wait_after': 0.5},
            # Phase 2: Grasper接近并抓取 (3个路径点)
            {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 0.5},
            # Phase 3: Pusher撤离 (1个路径点)
            {'arm': pusher_arm, 'waypoints': [pusher_wps[3]], 'wait_after': 0.5},
            # Phase 4: Grasper抬起物体 (1个路径点)
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
        """
        评估当前状态并返回策略类型和阶段类型标签。
        用于演示数据收集时标注每一帧。

        Returns:
            (strategy_type, phase_type) 整数元组
        """
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
            return {'current_phase': 1, 'total_phases': 4, 'completed': False,
                    'phase_status': {1: False, 2: False, 3: False, 4: False}}
        return self.phased_evaluator.get_phase_progress()

    def get_role_assignment(self) -> Dict[str, str]:
        """返回当前的角色分配"""
        return self.current_role_assignment.copy()

    def get_active_scheme(self) -> str:
        """返回当前激活的路径点方案"""
        return self.active_waypoint_mode
