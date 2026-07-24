import os

import numpy as np
from scipy.spatial.transform import Rotation


T_OFFSET_INV = {
    "bimanual_edge_phone": np.array([
        [0.000000357627798, -0.999999999999758, -0.000000596046490, -0.004460703107312],
        [0.000000238418686, -0.000000596046405, 0.999999999999794, -0.106504077672761],
        [-0.999999999999908, -0.000000357627940, 0.000000238418473, -0.000902679728312],
        [0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000],
    ], dtype=np.float64),
    "bimanual_pivot_phone": np.array([
        [0.000000178813465, -0.999999999999471, -0.000001013278951, -0.004495898648837],
        [0.000000894069682, -0.000001013278791, 0.999999999999087, -0.108443408182149],
        [-0.999999999999584, -0.000000178814371, 0.000000894069501, 0.002094409457517],
        [0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000],
    ], dtype=np.float64),
    "bimanual_pick_plate": np.array([
        [0.002598125880784, -0.058969175473169, -0.998256423012606, -0.067807748975981],
        [-0.000314574836860, -0.998259791077145, 0.058968555699510, -0.008845579165724],
        [-0.999996575386426, 0.000160818620698, -0.002612154817698, 0.006806277136513],
        [0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000],
    ], dtype=np.float64),
    "bimanual_pick_fork": np.array([
        [-0.008017196545178, -0.999967469322108, -0.000885921607486, -0.000882002254333],
        [0.011323440070544, -0.000976678403156, 0.999935410816251, -0.093989696356541],
        [-0.999903747499990, 0.008006647040768, 0.011330901934082, -0.000689878880687],
        [0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000],
    ], dtype=np.float64),
}


def validate_object_6d_pose(pose, context="object_6d_pose"):
    if not isinstance(pose, dict):
        raise ValueError(f"{context} must be a dictionary, got {type(pose).__name__}.")

    expected_shapes = {
        "position": (3,),
        "quaternion": (4,),
        "matrix": (4, 4),
    }
    if "orientation" in pose:
        expected_shapes["orientation"] = (3,)

    missing_keys = [key for key in expected_shapes if key not in pose]
    if missing_keys:
        raise ValueError(f"{context} is missing required keys: {missing_keys}.")

    arrays = {}
    for key, expected_shape in expected_shapes.items():
        value = pose[key]
        if value is None:
            raise ValueError(f"{context}[{key!r}] is None.")
        try:
            value_array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}[{key!r}] is not numeric.") from exc
        if value_array.shape != expected_shape:
            raise ValueError(
                f"{context}[{key!r}] has shape {value_array.shape}, "
                f"expected {expected_shape}.")
        if not np.all(np.isfinite(value_array)):
            raise ValueError(f"{context}[{key!r}] contains non-finite values.")
        arrays[key] = value_array

    quaternion_norm = np.linalg.norm(arrays["quaternion"])
    if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-4):
        raise ValueError(
            f"{context} quaternion norm is {quaternion_norm}, expected 1.")

    matrix = arrays["matrix"]
    if not np.allclose(
            matrix[3], np.array([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0, atol=1e-6):
        raise ValueError(f"{context} matrix is not a homogeneous transform.")
    if not np.allclose(
            arrays["position"], matrix[:3, 3],
            rtol=0.0, atol=1e-5):
        raise ValueError(
            f"{context} position does not match the matrix translation.")


def infer_task_name_from_data_path(data_path):
    if not data_path:
        raise ValueError("Cannot infer task name from an empty data_path.")
    normalized = os.path.normpath(data_path)
    task_name = os.path.basename(os.path.dirname(os.path.dirname(normalized)))
    for suffix in (".train", ".val", ".test"):
        if task_name.endswith(suffix):
            task_name = task_name[:-len(suffix)]
            break
    if not task_name:
        raise ValueError(f"Failed to infer task name from data_path: {data_path}")
    return task_name


def recover_object_6d_pose(obs, task_name):
    if task_name not in T_OFFSET_INV:
        raise KeyError(
            f"Task '{task_name}' does not have a contact-pose recovery constant. "
            "Extend T_OFFSET_INV or regenerate demos with object_6d_pose as described in guide section 3.4."
        )
    if not hasattr(obs, "misc") or obs.misc is None:
        raise AttributeError("Observation does not contain a misc dictionary.")

    missing_keys = [
        key for key in ("contact_position", "contact_quaternion")
        if key not in obs.misc
    ]
    if missing_keys:
        raise KeyError(
            f"Observation for task '{task_name}' is missing required misc keys: {missing_keys}"
        )

    contact_pos = np.asarray(obs.misc["contact_position"], dtype=np.float64).reshape(3)
    contact_quat = np.asarray(obs.misc["contact_quaternion"], dtype=np.float64).reshape(4)

    transform_contact = np.eye(4, dtype=np.float64)
    transform_contact[:3, :3] = Rotation.from_quat(contact_quat).as_matrix()
    transform_contact[:3, 3] = contact_pos

    transform_object = transform_contact @ T_OFFSET_INV[task_name]
    rotation_object = Rotation.from_matrix(transform_object[:3, :3])

    return {
        "position": transform_object[:3, 3].astype(np.float32),
        "quaternion": rotation_object.as_quat().astype(np.float32),
        "orientation": rotation_object.as_euler("xyz").astype(np.float32),
        "matrix": transform_object.astype(np.float32),
    }


def patch_demo_with_object_6d_pose(demo, task_name):
    for frame_index, obs in enumerate(demo):
        context = f"object_6d_pose for task '{task_name}', frame {frame_index}"
        pose = getattr(obs, "object_6d_pose", None)
        try:
            validate_object_6d_pose(pose, context=context)
        except ValueError as error:
            direct_pose_error = str(error)
        else:
            continue

        try:
            recovered_pose = recover_object_6d_pose(obs, task_name)
        except (AttributeError, KeyError, TypeError, ValueError) as recovery_error:
            raise ValueError(
                f"{context} is invalid ({direct_pose_error}); "
                f"contact-based recovery failed: {recovery_error}") from recovery_error

        try:
            validate_object_6d_pose(recovered_pose, context=context)
        except ValueError as recovered_pose_error:
            raise ValueError(
                f"{context} is invalid and contact-based recovery produced "
                f"another invalid pose: {recovered_pose_error}") from recovered_pose_error
        obs.object_6d_pose = recovered_pose
    return demo


def maybe_patch_low_dim_obs(file_path, obj, task_name):
    if os.path.basename(file_path) != "low_dim_obs.pkl":
        return obj
    return patch_demo_with_object_6d_pose(obj, task_name)
