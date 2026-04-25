from typing import List
from pyrep.objects.joint import Joint
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



class BimanualEdgePhoneOld(BimanualTask):

    def init_task(self) -> None:
        phone = Shape('Phone')
        self.register_success_conditions([LiftedCondition(phone, 1.0)])
        # Register the object that can be grasped - required for gripper to attach object
        self.register_graspable_objects([phone])

        # ===== 镜像路径点方案 =====
        # 使用 _a 后缀的镜像路径点，实现左臂抓取
        # Right arm (pusher): waypoint1_a, waypoint3_a, waypoint5_a, waypoint7_a
        # Left arm (grasper): waypoint0_a, waypoint2_a, waypoint4_a, waypoint6_a
        self.waypoint_mapping = defaultdict(lambda: 'left')
        # Pusher waypoints -> right arm
        for wp_name in ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']:
            self.waypoint_mapping[wp_name] = 'right'
        # Grasper waypoints -> left arm
        for wp_name in ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a']:
            self.waypoint_mapping[wp_name] = 'left'

    def init_episode(self, index: int) -> List[str]:
        return ['pick up the phone']

    def variation_count(self) -> int:
        return 1

    @property
    def execution_phases(self):
        """Define 4-phase sequential execution for edge grasp strategy.

        8 waypoints total (0-7): even for right arm, odd for left arm.
        waypoint 0/6 and 1/7 are at start/end positions (overlapping).

        Phase 1: Left arm pushes object over edge (waypoint1 -> 3 -> 5)
        Phase 2: Right arm grasps overhanging part (waypoint0 -> 2 -> 4)
        Phase 3: Left arm clears path (waypoint5 -> 7)
        Phase 4: Right arm lifts object (waypoint4 -> 6)

        Returns:
            List of phase dicts with 'arm', 'waypoints', and 'wait_after' keys.
        """
        return [
            {'arm': 'right', 'waypoints': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a'], 'wait_after': 0.5},
            {'arm': 'left', 'waypoints': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a'], 'wait_after': 0.5},
            {'arm': 'right', 'waypoints': ['waypoint7_a'], 'wait_after': 0.5},
            {'arm': 'left', 'waypoints': ['waypoint6_a'], 'wait_after': 0.5},
        ]
