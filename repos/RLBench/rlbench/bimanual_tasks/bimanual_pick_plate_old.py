from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.task import BimanualTask
from rlbench.backend.conditions import Condition
from collections import defaultdict


class LiftedCondition(Condition):
    """Check if plate is lifted above a minimum height."""

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False


class BimanualPickPlate(BimanualTask):

    def init_task(self) -> None:
        plate = Shape('plate')
        self.plate = plate  # 保存引用用于调试
        self.register_success_conditions([LiftedCondition(plate, 0.9)])
        # Register the plate as a graspable object - required for gripper to attach
        self.register_graspable_objects([plate])

        # Right arm (grasping): even waypoints 0, 2, 4, 6
        # Left arm (pressing): odd waypoints 1, 3, 5, 7
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(1, 9, 2):  # 1, 3, 5, 7
            self.waypoint_mapping.update({f'waypoint{i}': 'left'})

    def init_episode(self, index: int) -> List[str]:
        self._step_count = 0  # 用于追踪步数
        return ['pick up the plate']

    def step(self) -> None:
        """每个仿真步骤都会被调用，用于追踪 waypoint4 和 plate 位置"""
        self._step_count += 1
        # 每2步打印一次
        if self._step_count % 2 == 0 and Object.exists('waypoint4'):
            wp4 = Dummy('waypoint4')
            wp4_z = wp4.get_position()[2]
            plate_z = self.plate.get_position()[2]
            print(f"[Step {self._step_count:4d}] waypoint4 z={wp4_z:.4f} | Plate z={plate_z:.4f}")

    def variation_count(self) -> int:
        return 1

    @property
    def execution_phases(self):
        """Define 4-phase sequential execution for bimanual plate picking.

        8 waypoints total (0-7): even for right arm, odd for left arm.

        Phase 1: Left arm presses plate edge to tilt it (waypoint1 -> 3 -> 5)
        Phase 2: Right arm approaches and grasps lifted edge (waypoint0 -> 2 -> 4)
        Phase 3: Left arm clears path for lifting (waypoint5 -> 7)
        Phase 4: Right arm lifts plate (waypoint4 -> 6)

        Returns:
            List of phase dicts with 'arm', 'waypoints', and 'wait_after' keys.
        """
        return [
            {'arm': 'left', 'waypoints': ['waypoint1', 'waypoint3', 'waypoint5'], 'wait_after': 0.5},
            {'arm': 'right', 'waypoints': ['waypoint0', 'waypoint2', 'waypoint4'], 'wait_after': 0.5},
            {'arm': 'left', 'waypoints': ['waypoint7'], 'wait_after': 0.5},
            {'arm': 'right', 'waypoints': ['waypoint6'], 'wait_after': 0.5},
        ]
