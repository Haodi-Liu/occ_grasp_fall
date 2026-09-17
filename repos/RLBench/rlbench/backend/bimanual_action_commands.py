import numpy as np
from pyquaternion import Quaternion
from pyrep.backend import sim
from pyrep.const import JointType


SIDES = ('right', 'left')
SCHEMA_ID = 'rlbench_bimanual_action_commands/joint_target_fk_pose_v1'
JOINT_TARGET_FK_POSE = '{}_joint_target_fk_pose'


def joint_target_fk_pose_key(side):
    if side not in SIDES:
        raise ValueError('Unknown arm side: %r.' % (side,))
    return JOINT_TARGET_FK_POSE.format(side)


def _matrix4(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.size == 12:
        return np.vstack((
            matrix.reshape(3, 4),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        ))
    raise ValueError(
        'Expected a 4x4 or 12-value matrix, got %r.' %
        (matrix.shape,))


def _revolute_z(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([
        [cosine, -sine, 0.0, 0.0],
        [sine, cosine, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


class ArmForwardKinematics(object):
    """Fixed-base FK snapshot for one seven-joint PyRep arm.

    The snapshot stores only numpy matrices. Calling :meth:`pose` does not
    mutate robot state or step the simulator.
    """

    def __init__(self, arm):
        joints = tuple(arm.joints)
        if len(joints) != 7:
            raise ValueError(
                'Expected 7 arm joints, got %d.' % len(joints))
        if any(joint.get_joint_type() != JointType.REVOLUTE
               for joint in joints):
            raise ValueError('Only revolute arm joints are supported.')

        joint_bases = [_matrix4(joint.get_matrix()) for joint in joints]
        moving_frames = [
            joint_bases[index] @ _matrix4(
                sim.simGetJointMatrix(joints[index].get_handle()))
            for index in range(7)
        ]

        self._world_T_first_joint_base = joint_bases[0]
        self._links = tuple(
            np.linalg.solve(moving_frames[index], joint_bases[index + 1])
            for index in range(6)
        )
        self._last_joint_T_tip = np.linalg.solve(
            moving_frames[-1], _matrix4(arm.get_tip().get_matrix()))

    def matrix(self, joint_target):
        joint_target = np.asarray(joint_target, dtype=np.float64)
        if (joint_target.shape != (7,)
                or not np.isfinite(joint_target).all()):
            raise ValueError(
                'Expected a finite joint target with shape (7,), got %r.' %
                (joint_target,))

        transform = self._world_T_first_joint_base.copy()
        for index, angle in enumerate(joint_target):
            transform = transform @ _revolute_z(angle)
            transform = transform @ (
                self._links[index] if index < 6
                else self._last_joint_T_tip)
        return transform

    def pose(self, joint_target):
        transform = self.matrix(joint_target)
        # pyquaternion returns [qw, qx, qy, qz]. CoppeliaSim/RLBench uses
        # [qx, qy, qz, qw]. The slightly relaxed tolerance accommodates the
        # float32 matrices returned by the current CoppeliaSim API.
        wxyz = np.asarray(
            Quaternion(
                matrix=transform[:3, :3], atol=2e-6).elements,
            dtype=np.float64)
        xyzw = wxyz[[1, 2, 3, 0]]
        if xyzw[3] < 0.0:
            xyzw = -xyzw
        pose = np.concatenate((transform[:3, 3], xyzw))
        return pose.astype(np.float32)


def _validate_demo_not_empty(demo):
    if len(demo) == 0:
        raise ValueError('Demo is empty.')


def _source_keys(side):
    return (
        '%s_executed_demo_joint_position_action' % side,
        '%s_commanded_gripper_state' % side,
    )


def _validate_source_fields(obs, step_idx, side):
    target_key, gripper_key = _source_keys(side)
    missing = [
        key for key in (target_key, gripper_key) if key not in obs.misc
    ]
    if missing:
        raise ValueError(
            'Missing action command fields at step %d, side %s: %r.' %
            (step_idx, side, missing))

    target = np.asarray(obs.misc[target_key], dtype=np.float32)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError(
            'Invalid %s at step %d: %r.' %
            (target_key, step_idx, target))

    gripper = np.asarray(
        obs.misc[gripper_key], dtype=np.float32).reshape(-1)
    if (gripper.shape != (1,)
            or not np.isfinite(gripper).all()
            or float(gripper[0]) not in (0.0, 1.0)):
        raise ValueError(
            'Invalid %s at step %d: %r.' %
            (gripper_key, step_idx, gripper))


def validate_bimanual_action_command_sources(demo):
    """Validate the existing joint-target and gripper command fields."""
    _validate_demo_not_empty(demo)
    for step_idx, obs in enumerate(demo):
        for side in SIDES:
            _validate_source_fields(obs, step_idx, side)


def validate_bimanual_action_commands(demo):
    """Validate source commands and both derived joint-target FK poses."""
    _validate_demo_not_empty(demo)
    for step_idx, obs in enumerate(demo):
        for side in SIDES:
            _validate_source_fields(obs, step_idx, side)
            pose_key = joint_target_fk_pose_key(side)
            if pose_key not in obs.misc:
                raise ValueError(
                    'Missing %s at step %d, side %s.' %
                    (pose_key, step_idx, side))

            pose = np.asarray(obs.misc[pose_key])
            if (pose.shape != (7,)
                    or pose.dtype != np.dtype(np.float32)
                    or not np.isfinite(pose).all()):
                raise ValueError(
                    'Invalid %s at step %d: %r.' %
                    (pose_key, step_idx, pose))
            quaternion_norm = np.linalg.norm(pose[3:])
            if (not np.isclose(
                    quaternion_norm, 1.0, rtol=0.0, atol=1e-5)
                    or float(pose[6]) < -1e-7):
                raise ValueError(
                    'Invalid %s at step %d: %r.' %
                    (pose_key, step_idx, pose))
