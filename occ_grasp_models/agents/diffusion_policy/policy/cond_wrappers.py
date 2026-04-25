"""Condition-injection wrappers placeholder for future stages."""


def wrap_policy_with_condition_modules(policy, condition_modules):
    """Reserved API for strategy/phase/keypoint conditioning.

    Current baseline keeps plain UNet policy. This hook is added so later
    stages can inject condition modules without changing caller code.
    """

    del condition_modules
    return policy
