"""Shape-meta helper for DIFFUSION_POLICY."""


def build_shape_meta(camera_names, image_size, low_dim_size, action_dim):
    """Build diffusion policy shape_meta from method/rlbench config.

    Note: image_size uses project convention [W, H].
    """
    if len(image_size) != 2:
        raise ValueError(f"image_size must be [W, H], got: {image_size}")

    image_width = int(image_size[0])
    image_height = int(image_size[1])
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image_size must be positive [W, H], got: {image_size}")

    obs = {}
    for cam in camera_names:
        obs[f"{cam}_rgb"] = {"shape": [3, image_height, image_width], "type": "rgb"}
    obs["low_dim_state"] = {"shape": [int(low_dim_size)], "type": "low_dim"}

    return {
        "obs": obs,
        "action": {"shape": [int(action_dim)]},
    }
