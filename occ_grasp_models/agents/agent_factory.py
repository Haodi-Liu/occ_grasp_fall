import logging

from omegaconf import DictConfig


from yarr.agents.agent import BimanualAgent
from yarr.agents.agent import LeaderFollowerAgent
from yarr.agents.agent import Agent


supported_agents = {
    "leader_follower": (),
    "independent": (),
    "bimanual": (
        "BIMANUAL_PERACT", "ACT_BC_VISION", "ACT_BC_ENC",
        "ACT_BC_KEY", "ACT_BC_KEYPOINT",
        "ACT_BC_KEYPOINT_STRATEGY", "ACT_BC_ENC_STRATEGY",
        "ACT_BC_ENC_KEYPOINT", "ACT_BC_ENC_KEYPOINT_STRATEGY",
        "DIFFUSION_POLICY", "PPI", "OPENPI_POLICY",
    ),
    "unimanual": (),
}


def create_agent(cfg: DictConfig) -> Agent:

    method_name = cfg.method.name
    agent_type = cfg.method.agent_type

    logging.info("Using method %s with type %s", method_name, agent_type)

    if agent_type not in supported_agents:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    if method_name not in supported_agents[agent_type]:
        raise ValueError(
            f"Method {method_name} is not supported for agent_type {agent_type}. "
            f"Supported: {supported_agents[agent_type]}"
        )

    agent_fn = agent_fn_by_name(method_name)
    
    if agent_type == "leader_follower":
        checkpoint_name_prefix = cfg.framework.checkpoint_name_prefix
        cfg.method.robot_name = "right"
        cfg.framework.checkpoint_name_prefix = f"{checkpoint_name_prefix}_{method_name.lower()}_leader"
        leader_agent = agent_fn(cfg)

        cfg.method.robot_name = "left"
        cfg.framework.checkpoint_name_prefix = f"{checkpoint_name_prefix}_{method_name.lower()}_follower"
        cfg.method.low_dim_size = cfg.method.low_dim_size + 8 # also add the action size
        follower_agent = agent_fn(cfg)

        cfg.method.robot_name = "bimanual"

        return LeaderFollowerAgent(leader_agent, follower_agent)
    
    elif agent_type == "independent":
        checkpoint_name_prefix = cfg.framework.checkpoint_name_prefix
        cfg.method.robot_name = "right"
        cfg.framework.checkpoint_name_prefix = f"{checkpoint_name_prefix}_{method_name.lower()}_right"
        right_agent = agent_fn(cfg)

        cfg.method.robot_name = "left"
        cfg.framework.checkpoint_name_prefix = f"{checkpoint_name_prefix}_{method_name.lower()}_left"
        left_agent = agent_fn(cfg)

        cfg.method.robot_name = "bimanual"

        return BimanualAgent(right_agent, left_agent)
    elif agent_type == "bimanual" or agent_type == "unimanual":

        return agent_fn(cfg)
    else:
        raise Exception("invalid agent type")


def agent_fn_by_name(method_name: str) -> Agent:
    if method_name.startswith("BIMANUAL_PERACT"):
        from agents import bimanual_peract
        
        return bimanual_peract.launch_utils.create_agent
    elif method_name == "ACT_BC_ENC_KEYPOINT_STRATEGY":
        # 注意：必须在其他 ACT_BC_ENC 变体之前匹配
        from agents import act_bc_enc_keypoint_strategy

        return act_bc_enc_keypoint_strategy.launch_utils.create_agent
    elif method_name == "ACT_BC_ENC_STRATEGY":
        # 注意：必须在 ACT_BC_ENC 之前匹配，否则会被 startswith("ACT_BC_ENC") 错误捕获
        from agents import act_bc_enc_strategy

        return act_bc_enc_strategy.launch_utils.create_agent
    elif method_name == "ACT_BC_KEYPOINT_STRATEGY":
        from agents import act_bc_keypoint_strategy

        return act_bc_keypoint_strategy.launch_utils.create_agent
    elif method_name == "ACT_BC_KEYPOINT":
        from agents import act_bc_keypoint

        return act_bc_keypoint.launch_utils.create_agent
    elif method_name == "ACT_BC_KEY":
        from agents import act_bc_key

        return act_bc_key.launch_utils.create_agent
    elif method_name == "ACT_BC_ENC_KEYPOINT":
        # 注意：必须在 ACT_BC_ENC 之前匹配，否则会被 startswith("ACT_BC_ENC") 错误捕获
        from agents import act_bc_enc_keypoint

        return act_bc_enc_keypoint.launch_utils.create_agent
    elif method_name.startswith("ACT_BC_ENC"):
        from agents import act_bc_enc

        return act_bc_enc.launch_utils.create_agent
    elif method_name.startswith("ACT_BC_VISION"):
        from agents import act_bc_vision

        return act_bc_vision.launch_utils.create_agent
    elif method_name == "DIFFUSION_POLICY":
        from agents import diffusion_policy

        return diffusion_policy.launch_utils.create_agent
    elif method_name == "PPI":
        from agents import ppi

        return ppi.launch_utils.create_agent
    elif method_name == "OPENPI_POLICY":
        from agents import openpi_policy

        return openpi_policy.launch_utils.create_agent
    else:
        raise ValueError("Method %s does not exists." % method_name)
