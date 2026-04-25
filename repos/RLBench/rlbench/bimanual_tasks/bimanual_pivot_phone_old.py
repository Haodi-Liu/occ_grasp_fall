from typing import List
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask
from collections import defaultdict

from pyrep.objects.shape import Shape
from rlbench.backend.conditions import Condition

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False



class BimanualPivotPhoneOld(BimanualTask):

    def init_task(self) -> None:
        phone = Shape('Phone')
        self.phone = phone  # 保存引用用于调试
        self.register_success_conditions([LiftedCondition(phone, 0.9)])
        # Register the object that can be grasped - required for gripper to attach object
        self.register_graspable_objects([phone])
        # Right arm (grasping): even waypoints 0, 2, 4, 6
        # Left arm (auxiliary): odd waypoints 1, 3, 5, 7, 8
        self.waypoint_mapping = defaultdict(lambda: 'left')
        for i in range(1, 9, 2):  # 1, 3, 5, 7
            self.waypoint_mapping.update({f'waypoint{i}': 'right'})
        self.waypoint_mapping.update({'waypoint8': 'right'})  # waypoint8 also uses right arm

    def init_episode(self, index: int) -> List[str]:
        self._step_count = 0  # 用于追踪步数
        return ['pick up the phone']

    def step(self) -> None:
        """每个仿真步骤都会被调用，用于追踪 waypoint4 和 phone 位置"""
        self._step_count += 1
        # 每2步打印一次
        if self._step_count % 2 == 0 and Object.exists('waypoint4'):
            wp4 = Dummy('waypoint4')
            wp4_z = wp4.get_position()[2]
            phone_z = self.phone.get_position()[2]
            print(f"[Step {self._step_count:4d}] waypoint4 z={wp4_z:.4f} | Phone z={phone_z:.4f}")

    def variation_count(self) -> int:
        return 1

    @property
    def execution_phases(self):
        """Define 4-phase sequential execution for pivot grasp strategy.

        9 waypoints total (0-8): even for right arm, odd + 8 for left arm.

        Phase 1: Left arm pushes object to wall and pivots (waypoint1 -> 3 -> 5)
        Phase 2: Right arm grasps pivoted part (waypoint0 -> 2 -> 4)
        Phase 3: Left arm clears path (waypoint7 -> 8)
        Phase 4: Right arm lifts object (waypoint6)

        Returns:
            List of phase dicts with 'arm', 'waypoints', and 'wait_after' keys.
        """
        return [
            {'arm': 'right', 'waypoints': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7'], 'wait_after': 0.5},
            {'arm': 'left', 'waypoints': ['waypoint0', 'waypoint2', 'waypoint4'], 'wait_after': 0.5},
            {'arm': 'right', 'waypoints': ['waypoint8'], 'wait_after': 0.5},
            {'arm': 'left', 'waypoints': ['waypoint6'], 'wait_after': 0.5},
        ]
