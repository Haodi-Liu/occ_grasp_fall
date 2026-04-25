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



class BimanualPivotLaptop(BimanualTask):

    def init_task(self) -> None:
        laptop = Shape('base')
        self.register_success_conditions([LiftedCondition(laptop, 0.8)])
        # Register the object that can be grasped - required for gripper to attach object
        self.register_graspable_objects([laptop])
        # Right arm (grasping): even waypoints 0, 2, 4, 6
        # Left arm (auxiliary): odd waypoints 1, 3, 5, 7, 8
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(1, 9, 2):  # 1, 3, 5, 7
            self.waypoint_mapping.update({f'waypoint{i}': 'left'})
        self.waypoint_mapping.update({'waypoint8': 'left'})  # waypoint8 also uses left arm

    def init_episode(self, index: int) -> List[str]:
        return ['pick up the laptop']

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
            {'arm': 'left', 'waypoints': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7'], 'wait_after': 1},
            {'arm': 'right', 'waypoints': ['waypoint0', 'waypoint2', 'waypoint4'], 'wait_after': 1},
            {'arm': 'left', 'waypoints': ['waypoint8'], 'wait_after': 1},
            {'arm': 'right', 'waypoints': ['waypoint6'], 'wait_after': 1},
        ]