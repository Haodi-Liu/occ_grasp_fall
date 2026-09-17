import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rlbench.backend.scene import Scene
from rlbench.observation_config import ObservationConfig


class _FakeFk(object):

    @staticmethod
    def expected_pose(joint_target):
        joint_target = np.asarray(joint_target, dtype=np.float32)
        return np.array([
            joint_target[0], joint_target[1], joint_target[2],
            0.0, 0.0, 0.0, 1.0,
        ], dtype=np.float32)

    def pose(self, joint_target):
        return self.expected_pose(joint_target)


class TestBimanualActionCommandRecording(unittest.TestCase):

    def test_recording_flag_defaults_to_false(self):
        self.assertFalse(
            ObservationConfig().record_bimanual_action_commands)

    @patch('rlbench.backend.scene.Object.exists', return_value=False)
    def test_misc_keeps_commands_across_observations(self, _):
        scene = Scene.__new__(Scene)
        scene.robot = SimpleNamespace(is_bimanual=True)
        scene._obs_config = SimpleNamespace(
            record_bimanual_action_commands=True)
        scene.camera_sensors = {}
        scene._variation_index = 0
        scene._current_strategy_type = None
        scene._right_execute_demo_joint_position_action = np.arange(7)
        scene._left_execute_demo_joint_position_action = np.arange(7) + 10
        scene._right_commanded_gripper_state = 0.0
        scene._left_commanded_gripper_state = 1.0
        scene._bimanual_action_fk = {
            'right': _FakeFk(),
            'left': _FakeFk(),
        }

        first = scene._get_misc()
        second = scene._get_misc()

        for key in (
                'right_executed_demo_joint_position_action',
                'left_executed_demo_joint_position_action',
                'right_joint_target_fk_pose',
                'left_joint_target_fk_pose',
                'right_commanded_gripper_state',
                'left_commanded_gripper_state'):
            self.assertIn(key, first)
            self.assertIn(key, second)

        np.testing.assert_array_equal(
            first['right_joint_target_fk_pose'],
            _FakeFk.expected_pose(
                scene._right_execute_demo_joint_position_action))
        np.testing.assert_array_equal(
            first['left_joint_target_fk_pose'],
            _FakeFk.expected_pose(
                scene._left_execute_demo_joint_position_action))
        np.testing.assert_array_equal(
            first['right_executed_demo_joint_position_action'],
            scene._right_execute_demo_joint_position_action)
        np.testing.assert_array_equal(
            first['left_executed_demo_joint_position_action'],
            scene._left_execute_demo_joint_position_action)


if __name__ == '__main__':
    unittest.main()
