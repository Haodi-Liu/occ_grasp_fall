import os
import numpy as np


STRATEGY_NAMES = ["EdgeHang", "WallLever", "PressTilt"]
PHASE_NAMES = ["PreManipulation", "Grasp", "ClearPath", "Lift"]


def build_pred_info_from_outputs(out, cam_names, img_size, sample_idx):
    pred_info = {"img_size": int(img_size)}

    strat = out.get("strategy_logits")
    if strat is not None:
        strat_id = int(strat.argmax(dim=-1)[sample_idx].item())
        pred_info["strategy_id"] = strat_id
        pred_info["strategy_name"] = (
            STRATEGY_NAMES[strat_id] if strat_id < len(STRATEGY_NAMES) else str(strat_id)
        )

    phase = out.get("phase_logits")
    if phase is not None:
        phase_id = int(phase.argmax(dim=-1)[sample_idx].item())
        pred_info["phase_id"] = phase_id
        pred_info["phase_name"] = (
            PHASE_NAMES[phase_id] if phase_id < len(PHASE_NAMES) else str(phase_id)
        )

    kp = out.get("keypoint_2d_pred")
    if kp is not None:
        kp_by_cam = {}
        for cam_idx, cam in enumerate(cam_names):
            kp_by_cam[cam] = {
                "contact": kp["contact"][sample_idx, cam_idx].detach().cpu().tolist(),
                "grasp": kp["grasp"][sample_idx, cam_idx].detach().cpu().tolist(),
                "affordance": kp["affordance"][sample_idx, cam_idx].detach().cpu().tolist(),
            }
        pred_info["keypoints_2d"] = kp_by_cam

    aff = out.get("aff_visible_logits")
    if aff is not None:
        pred_info["affordance_visible"] = float((aff[sample_idx].sigmoid() > 0.5).item())

    return pred_info


def render_aux_eval_like(obs_dict, pred_info, cfg, out_path):
    if pred_info is None:
        return

    cam = getattr(cfg, "sample_camera", "over_shoulder_right")
    cam_key = f"{cam}_rgb"
    rgb = obs_dict.get(cam_key)
    if rgb is None:
        return
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255.).astype(np.uint8)

    import cv2
    h_src, w_src = rgb.shape[:2]
    scale = float(getattr(cfg, "sample_output_scale", 0.0))
    min_width = int(getattr(cfg, "sample_min_width", 512))
    if scale and scale > 0:
        out_w = max(1, int(round(w_src * scale)))
        out_h = max(1, int(round(h_src * scale)))
    elif min_width and w_src < min_width:
        out_w = min_width
        out_h = max(1, int(round(h_src * (min_width / float(w_src)))))
    else:
        out_w, out_h = w_src, h_src
    if out_w != w_src or out_h != h_src:
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    strategy_name = pred_info.get("strategy_name")
    phase_name = pred_info.get("phase_name")
    if strategy_name or phase_name:
        font = cv2.FONT_HERSHEY_DUPLEX
        h_cam, w_cam = frame_bgr.shape[:2]
        margin = max(8, min(32, int(round(16 * (w_cam / 640.0)))))
        base_scale = 0.6 * (w_cam / 640.0)
        base_scale = max(0.35, min(base_scale, 1.2))
        thickness = max(1, int(round(base_scale * 2)))

        lines = [
            "Strategy: %s" % (strategy_name or "N/A"),
            "Phase: %s" % (phase_name or "N/A"),
        ]
        sizes = [cv2.getTextSize(line, font, base_scale, thickness) for line in lines]
        max_line_w = max(tw for (tw, _), _ in sizes)
        if max_line_w > (w_cam - 2 * margin):
            scale = base_scale * (w_cam - 2 * margin) / float(max_line_w)
            thickness = max(1, int(round(scale * 2)))
            sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
        else:
            scale = base_scale

        spacing = max(2, int(round(4 * scale)))
        pad = max(3, int(round(4 * scale)))
        line_heights = [th + base for (tw, th), base in sizes]
        block_h = sum(line_heights) + spacing * (len(lines) - 1)
        block_w = max(tw for (tw, _), _ in sizes)
        x0, y0 = margin, margin
        cv2.rectangle(
            frame_bgr,
            (x0 - pad, y0 - pad),
            (x0 + block_w + pad, y0 + block_h + pad),
            (0, 0, 0),
            -1,
        )

        y = y0
        for line, ((tw, th), base) in zip(lines, sizes):
            y_text = y + th
            cv2.putText(
                frame_bgr, line, (x0, y_text),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
            y += (th + base + spacing)

    kp_by_cam = pred_info.get("keypoints_2d", {})
    kp_cam = kp_by_cam.get(cam, {})
    pred_img_size = pred_info.get("img_size")
    h_cam, w_cam = frame_bgr.shape[:2]
    gt_sx = w_cam / float(w_src) if w_src else 1.0
    gt_sy = h_cam / float(h_src) if h_src else 1.0
    if pred_img_size:
        sx = w_cam / float(pred_img_size)
        sy = h_cam / float(pred_img_size)
    else:
        sx = sy = 1.0

    aff_vis = pred_info.get("affordance_visible")
    kp_colors = {
        "contact": (0, 0, 255),
        "grasp": (0, 255, 0),
        "affordance": (255, 0, 0),
    }
    gt_kp_colors = {
        "contact": (0, 255, 255),    # yellow
        "grasp": (255, 0, 255),      # magenta
        "affordance": (0, 165, 255), # orange
    }
    dot_radius = max(2, int(round(4 * (w_cam / 640.0))))
    for kp_name, color in kp_colors.items():
        if kp_name == "affordance" and aff_vis is not None and aff_vis < 0.5:
            continue
        coords = kp_cam.get(kp_name)
        if coords is None or len(coords) != 2:
            continue
        u = int(max(0, min(w_cam - 1, coords[0] * sx)))
        v = int(max(0, min(h_cam - 1, coords[1] * sy)))
        cv2.circle(frame_bgr, (u, v), dot_radius, color, -1)
    has_aff = obs_dict.get("has_affordance")
    gt_radius = dot_radius
    gt_thickness = max(1, int(round(gt_radius / 3)))
    gt_coords = {
        "contact": obs_dict.get(f"{cam}_contact_2d"),
        "grasp": obs_dict.get(f"{cam}_grasp_2d"),
        "affordance": obs_dict.get(f"{cam}_affordance_2d"),
    }
    gt_vis = {
        "contact": obs_dict.get(f"{cam}_contact_visible"),
        "grasp": obs_dict.get(f"{cam}_grasp_visible"),
        "affordance": obs_dict.get(f"{cam}_affordance_visible"),
    }
    for kp_name, color in gt_kp_colors.items():
        vis = gt_vis.get(kp_name)
        if vis is not None and not bool(vis):
            continue
        if kp_name == "affordance" and has_aff is not None and not bool(has_aff):
            continue
        coords = gt_coords.get(kp_name)
        if coords is None or len(coords) != 2:
            continue
        if coords[0] < 0 or coords[1] < 0:
            continue
        u = int(max(0, min(w_cam - 1, coords[0] * gt_sx)))
        v = int(max(0, min(h_cam - 1, coords[1] * gt_sy)))
        cv2.circle(frame_bgr, (u, v), gt_radius, color, gt_thickness)
    cam_extr = obs_dict.get(f"{cam}_camera_extrinsics")
    cam_intr = obs_dict.get(f"{cam}_camera_intrinsics")
    if cam_extr is not None and cam_intr is not None:
        def _as_np(x):
            arr = np.asarray(x)
            if arr.ndim == 3:
                arr = arr[-1]
            if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] == 7:
                arr = arr[-1]
            return arr

        def _project_3d_to_2d_np(point_xyz, extr, intr, image_size):
            extr = _as_np(extr)
            intr = _as_np(intr)
            R = extr[:3, :3]
            C = extr[:3, 3:4]
            R_inv = R.T
            extr_w2c = np.concatenate([R_inv, -R_inv @ C], axis=1)
            cam_proj = intr @ extr_w2c
            p_h = np.concatenate([point_xyz, np.ones((1,), dtype=point_xyz.dtype)], axis=0)
            p_img = cam_proj @ p_h
            z = p_img[2]
            if z == 0:
                return None, False, z
            u = p_img[0] / z
            v = p_img[1] / z
            W, H = image_size
            visible = (z > 0) and (u >= 0) and (u < W) and (v >= 0) and (v < H)
            return np.array([u, v], dtype=np.float32), visible, z

        def _draw_cross(img, u, v, color, size, thickness, outline_color=(0, 0, 0), outline_thickness=2):
            u = int(round(u))
            v = int(round(v))
            if outline_color is not None:
                cv2.line(img, (u - size, v), (u + size, v), outline_color, outline_thickness)
                cv2.line(img, (u, v - size), (u, v + size), outline_color, outline_thickness)
            cv2.line(img, (u - size, v), (u + size, v), color, thickness)
            cv2.line(img, (u, v - size), (u, v + size), color, thickness)

        right_pose = obs_dict.get("right_gripper_pose")
        left_pose = obs_dict.get("left_gripper_pose")
        if right_pose is not None:
            right_xyz = _as_np(right_pose)[:3]
            uv, visible, _ = _project_3d_to_2d_np(right_xyz, cam_extr, cam_intr, (w_src, h_src))
            if uv is not None and visible:
                u = uv[0] * gt_sx
                v = uv[1] * gt_sy
                size = max(4, int(round(6 * (w_cam / 640.0))))
                thickness = max(1, int(round(size / 3)))
                _draw_cross(frame_bgr, u, v, (255, 255, 255), size, thickness)
        if left_pose is not None:
            left_xyz = _as_np(left_pose)[:3]
            uv, visible, _ = _project_3d_to_2d_np(left_xyz, cam_extr, cam_intr, (w_src, h_src))
            if uv is not None and visible:
                u = uv[0] * gt_sx
                v = uv[1] * gt_sy
                size = max(4, int(round(6 * (w_cam / 640.0))))
                thickness = max(1, int(round(size / 3)))
                _draw_cross(frame_bgr, u, v, (255, 255, 0), size, thickness)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, frame_bgr)
