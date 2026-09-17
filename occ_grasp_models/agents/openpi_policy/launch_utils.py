"""Launch utilities for OPENPI_POLICY."""

from omegaconf import DictConfig

from agents.openpi_policy.agent import OpenPIPolicyAgent


_REQUIRED_CAMERAS = ("front", "wrist_left", "wrist_right")
_EXPECTED_ACTION_MODE = {
    "gripper_mode": "BimanualDiscrete",
    "arm_action_mode": "BimanualJointPosition",
    "action_mode": "BimanualJointPositionActionMode",
}
_ORACLE_TASKS = {
    "bimanual_edge_phone",
    "bimanual_pivot_phone",
    "bimanual_pick_plate",
    "bimanual_pick_fork",
}


def create_agent(cfg: DictConfig):
    """Create an OPENPI_POLICY eval-only agent."""
    robot_name = str(getattr(cfg.method, "robot_name", ""))
    if robot_name != "bimanual":
        raise ValueError(
            "OPENPI_POLICY requires method.robot_name='bimanual' "
            "(got %r)." % robot_name
        )

    camera_names = list(getattr(cfg.rlbench, "cameras", []))
    missing_cameras = [name for name in _REQUIRED_CAMERAS if name not in camera_names]
    if missing_cameras:
        raise ValueError(
            "OPENPI_POLICY requires rlbench.cameras to include %s; missing %s."
            % (list(_REQUIRED_CAMERAS), missing_cameras)
        )

    for field_name, expected_value in _EXPECTED_ACTION_MODE.items():
        actual_value = str(getattr(cfg.rlbench, field_name, ""))
        if actual_value != expected_value:
            raise ValueError(
                "OPENPI_POLICY requires rlbench.%s=%r (got %r)."
                % (field_name, expected_value, actual_value)
            )

    oracle_enabled = bool(getattr(cfg.method, "oracle_phase_enabled", False))
    oracle_task = None
    if oracle_enabled:
        tasks = [str(task) for task in cfg.rlbench.tasks]
        if len(tasks) != 1:
            raise ValueError("oracle phase evaluation requires exactly one task")
        oracle_task = tasks[0]
        if oracle_task not in _ORACLE_TASKS:
            raise ValueError("unsupported oracle task: %s" % oracle_task)

    return OpenPIPolicyAgent(
        host=str(cfg.method.openpi_host),
        port=int(cfg.method.openpi_port),
        replan_steps=int(cfg.method.replan_steps),
        oracle_phase_enabled=oracle_enabled,
        oracle_phase_task=oracle_task,
    )
