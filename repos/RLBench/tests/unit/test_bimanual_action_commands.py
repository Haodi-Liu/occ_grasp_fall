import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pyrep.const import JointType

from rlbench.backend.bimanual_action_commands import ArmForwardKinematics
from rlbench.backend.bimanual_action_commands import joint_target_fk_pose_key
from rlbench.backend.bimanual_action_commands import (
    validate_bimanual_action_command_sources)
from rlbench.backend.bimanual_action_commands import (
    validate_bimanual_action_commands)


def _translation_x(distance):
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = distance
    return matrix


class _FakeJoint(object):

    def __init__(self, handle, matrix, joint_type=JointType.REVOLUTE):
        self._handle = handle
        self._matrix = matrix
        self._joint_type = joint_type

    def get_handle(self):
        return self._handle

    def get_matrix(self):
        return self._matrix.copy()

    def get_joint_type(self):
        return self._joint_type

    def set_joint_position(self, _):
        raise AssertionError('FK must not mutate joint positions.')

    def set_joint_target_position(self, _):
        raise AssertionError('FK must not mutate joint targets.')


class _FakeTip(object):

    def get_matrix(self):
        return _translation_x(7.0)


class _FakeArm(object):

    def __init__(self, joint_type=JointType.REVOLUTE):
        self.joints = [
            _FakeJoint(index, _translation_x(float(index)), joint_type)
            for index in range(7)
        ]

    def get_tip(self):
        return _FakeTip()

    def set_joint_positions(self, _):
        raise AssertionError('FK must not mutate the arm.')

    def set_joint_target_positions(self, _):
        raise AssertionError('FK must not mutate arm targets.')


def _identity_joint_matrix(_):
    return np.eye(4, dtype=np.float64)[:3].reshape(-1).tolist()


def _complete_misc():
    misc = {}
    for index, side in enumerate(('right', 'left')):
        misc['%s_executed_demo_joint_position_action' % side] = (
            np.arange(7, dtype=np.float32) + index)
        misc['%s_commanded_gripper_state' % side] = float(index)
        misc[joint_target_fk_pose_key(side)] = np.array(
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32)
    return misc


class TestArmForwardKinematics(unittest.TestCase):

    @patch(
        'rlbench.backend.bimanual_action_commands.sim.simGetJointMatrix',
        side_effect=_identity_joint_matrix)
    def test_known_seven_link_chain(self, _):
        fk = ArmForwardKinematics(_FakeArm())

        zero_pose = fk.pose(np.zeros(7, dtype=np.float32))
        np.testing.assert_allclose(
            zero_pose,
            np.array([7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            rtol=0.0, atol=1e-6)

        quarter_turn = np.zeros(7, dtype=np.float32)
        quarter_turn[0] = np.pi / 2.0
        quarter_turn_pose = fk.pose(quarter_turn)
        np.testing.assert_allclose(
            quarter_turn_pose[:3], [0.0, 7.0, 0.0],
            rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            quarter_turn_pose[3:],
            [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            rtol=0.0, atol=1e-6)

    @patch(
        'rlbench.backend.bimanual_action_commands.sim.simGetJointMatrix',
        side_effect=_identity_joint_matrix)
    def test_pose_is_finite_float32_unit_quaternion(self, _):
        fk = ArmForwardKinematics(_FakeArm())
        target = np.zeros(7, dtype=np.float64)
        target[0] = 1.5 * np.pi

        pose = fk.pose(target)

        self.assertEqual(pose.shape, (7,))
        self.assertEqual(pose.dtype, np.float32)
        self.assertTrue(np.isfinite(pose).all())
        self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0, 6)
        self.assertGreaterEqual(float(pose[6]), 0.0)

    @patch(
        'rlbench.backend.bimanual_action_commands.sim.simGetJointMatrix',
        side_effect=_identity_joint_matrix)
    def test_invalid_joint_target_fails(self, _):
        fk = ArmForwardKinematics(_FakeArm())

        with self.assertRaisesRegex(ValueError, 'shape'):
            fk.pose(np.zeros(6))
        invalid = np.zeros(7)
        invalid[2] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            fk.pose(invalid)

    def test_invalid_arm_structure_fails(self):
        short_arm = _FakeArm()
        short_arm.joints.pop()
        with self.assertRaisesRegex(ValueError, '7 arm joints'):
            ArmForwardKinematics(short_arm)

        with self.assertRaisesRegex(ValueError, 'revolute'):
            ArmForwardKinematics(_FakeArm(JointType.PRISMATIC))


class TestBimanualActionCommandValidation(unittest.TestCase):

    def test_source_and_complete_schema(self):
        complete = SimpleNamespace(misc=_complete_misc())
        source_only_misc = {
            key: value for key, value in _complete_misc().items()
            if not key.endswith('_joint_target_fk_pose')
        }

        validate_bimanual_action_command_sources(
            [SimpleNamespace(misc=source_only_misc)])
        validate_bimanual_action_commands([complete])

    def test_missing_fk_pose_fails_complete_schema(self):
        misc = _complete_misc()
        del misc['left_joint_target_fk_pose']

        with self.assertRaisesRegex(ValueError, 'left_joint_target_fk_pose'):
            validate_bimanual_action_commands([SimpleNamespace(misc=misc)])

    def test_bad_pose_and_gripper_fail(self):
        bad_pose = _complete_misc()
        bad_pose['right_joint_target_fk_pose'][6] = 2.0
        with self.assertRaisesRegex(ValueError, 'right_joint_target_fk_pose'):
            validate_bimanual_action_commands(
                [SimpleNamespace(misc=bad_pose)])

        bad_gripper = _complete_misc()
        bad_gripper['left_commanded_gripper_state'] = 0.5
        with self.assertRaisesRegex(ValueError, 'commanded_gripper_state'):
            validate_bimanual_action_commands(
                [SimpleNamespace(misc=bad_gripper)])

    def test_non_float32_pose_fails(self):
        misc = _complete_misc()
        misc['right_joint_target_fk_pose'] = (
            misc['right_joint_target_fk_pose'].astype(np.float64))

        with self.assertRaisesRegex(ValueError, 'right_joint_target_fk_pose'):
            validate_bimanual_action_commands([SimpleNamespace(misc=misc)])

    def test_invalid_side_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unknown arm side'):
            joint_target_fk_pose_key('middle')


if __name__ == '__main__':
    unittest.main()
