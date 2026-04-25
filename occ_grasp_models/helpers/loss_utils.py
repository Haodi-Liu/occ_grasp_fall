import torch


KEYPOINT_NAMES = ["contact", "grasp", "affordance"]


def compute_2d_loss(pred, gt, visible, has_affordance, img_size):
    """
    Compute normalized 2D keypoint loss.

    Args:
        pred: Dict[kp_name -> Tensor[B, num_cameras, 2]]
        gt: Dict[cam_name -> Tensor[B, 3, 2]]
        visible: Dict[cam_name -> Tensor[B, 3]]
        has_affordance: Tensor[B] or None
        img_size: int
    """
    if pred is None or gt is None or visible is None:
        raise ValueError("compute_2d_loss expects pred/gt/visible to be non-None")

    device = pred[KEYPOINT_NAMES[0]].device
    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0

    camera_names = list(gt.keys())
    has_aff = None
    if has_affordance is not None:
        has_aff = has_affordance.to(torch.bool)

    for cam_idx, cam_name in enumerate(camera_names):
        gt_cam = gt[cam_name]
        vis_cam = visible[cam_name].to(torch.bool)
        for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
            pred_2d = pred[kp_name][:, cam_idx, :]
            gt_2d = gt_cam[:, kp_idx, :]
            vis_mask = vis_cam[:, kp_idx]
            if kp_name == "affordance" and has_aff is not None:
                vis_mask = vis_mask & has_aff

            if vis_mask.any():
                diff = pred_2d[vis_mask] - gt_2d[vis_mask]
                loss = (diff ** 2).sum(dim=-1).sqrt().mean() / float(img_size)
                total_loss = total_loss + loss
                valid_count += 1

    if valid_count > 0:
        total_loss = total_loss / valid_count

    return total_loss
