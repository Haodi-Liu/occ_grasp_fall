import unittest

import numpy as np

from rlbench.action_modes.action_mode import BimanualJointPositionActionMode
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.backend.exceptions import InvalidActionError


class _ArmSpy:

    def __init__(self, events):
        self._events = events

    def action_shape(self, scene):
        return 14,

    def action_pre_step(self, scene, action):
        self._events.append(('arm_pre', action.copy()))

    def action_step(self, scene):
        self._events.append(('arm_step', None))

    def action_post_step(self, scene, action):
        self._events.append(('arm_post', action.copy()))


class _GripperSpy:

    def __init__(self, events):
        self._events = events

    def action_shape(self, scene):
        return 2,

    def action(self, scene, action):
        self._events.append(('gripper_action', action.copy()))

    def action_pre_step(self, scene, action):
        raise AssertionError('gripper action_pre_step must not be called')

    def action_post_step(self, scene, action):
        raise AssertionError('gripper action_post_step must not be called')


class _FakeGripper:

    def __init__(self, is_open):
        self._is_open = is_open
        self.grasped_objects = []
        self.release_calls = 0

    def get_open_amount(self):
        amount = 1.0 if self._is_open else 0.0
        return [amount, amount]

    def get_grasped_objects(self):
        return list(self.grasped_objects)

    def grasp(self, graspable_object):
        self.grasped_objects.append(graspable_object)

    def release(self):
        self.release_calls += 1
        self.grasped_objects = []


class _FakeRobot:

    def __init__(self, right_open=True, left_open=True):
        self.right_gripper = _FakeGripper(right_open)
        self.left_gripper = _FakeGripper(left_open)


class _StepCounter:

    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _FakeTask(_StepCounter):

    def get_graspable_objects(self):
        return []


class _FakeScene:

    def __init__(self, right_open=True, left_open=True):
        self.robot = _FakeRobot(right_open, left_open)
        self.pyrep = _StepCounter()
        self.task = _FakeTask()


class _RecordingBimanualDiscrete(BimanualDiscrete):

    def __init__(self):
        super().__init__()
        self.actuate_actions = []

    def _actuate(self, scene, action):
        self.actuate_actions.append(np.array(action, copy=True))


class TestBimanualJointPositionActionMode(unittest.TestCase):

    def test_action_order_and_payload(self):
        events = []
        action_mode = BimanualJointPositionActionMode(
            arm_action_mode=_ArmSpy(events),
            gripper_action_mode=_GripperSpy(events),
        )
        action = np.arange(16, dtype=np.float64)

        action_mode.action(object(), action)

        self.assertEqual(
            [name for name, _ in events],
            ['arm_pre', 'arm_step', 'arm_post', 'gripper_action'],
        )
        expected_arm_action = np.concatenate([action[0:7], action[8:15]])
        np.testing.assert_array_equal(events[0][1], expected_arm_action)
        np.testing.assert_array_equal(events[2][1], expected_arm_action)
        np.testing.assert_array_equal(events[3][1], action[[7, 15]])

    def test_invalid_action_shape_raises(self):
        action_mode = BimanualJointPositionActionMode()

        for size in (15, 17):
            with self.subTest(size=size):
                with self.assertRaises(InvalidActionError):
                    action_mode.action(object(), np.zeros(size))

    def test_invalid_gripper_values_raise(self):
        action_mode = BimanualDiscrete()
        scene = _FakeScene()
        invalid_actions = (
            [-0.01, 0.0],
            [0.0, 1.01],
            [np.nan, 0.0],
            [0.0, np.inf],
        )

        for action in invalid_actions:
            with self.subTest(action=action):
                with self.assertRaises(InvalidActionError):
                    action_mode.action(scene, action)

    def test_actuate_receives_binary_action(self):
        action_mode = _RecordingBimanualDiscrete()
        scene = _FakeScene()

        action_mode.action(scene, [0.2, 0.8])

        self.assertEqual(len(action_mode.actuate_actions), 1)
        np.testing.assert_array_equal(
            action_mode.actuate_actions[0],
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_unilateral_close_has_no_release_settling_steps(self):
        action_mode = _RecordingBimanualDiscrete()
        scene = _FakeScene(right_open=True, left_open=True)

        action_mode.action(scene, [0.0, 1.0])

        self.assertEqual(scene.pyrep.steps, 0)
        self.assertEqual(scene.task.steps, 0)

    def test_real_open_transition_keeps_release_settling_steps(self):
        action_mode = _RecordingBimanualDiscrete()
        scene = _FakeScene(right_open=False, left_open=False)

        action_mode.action(scene, [1.0, 0.0])

        self.assertEqual(scene.robot.right_gripper.release_calls, 1)
        self.assertEqual(scene.robot.left_gripper.release_calls, 0)
        self.assertEqual(scene.pyrep.steps, 10)
        self.assertEqual(scene.task.steps, 10)

    def test_action_bounds(self):
        lower, upper = BimanualDiscrete().action_bounds()

        self.assertEqual(lower.shape, (2,))
        self.assertEqual(upper.shape, (2,))
        np.testing.assert_array_equal(lower, [0.0, 0.0])
        np.testing.assert_array_equal(upper, [1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
