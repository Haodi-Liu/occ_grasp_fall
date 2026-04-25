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
    for obs in demo:
        if not hasattr(obs, "object_6d_pose"):
            obs.object_6d_pose = recover_object_6d_pose(obs, task_name)
    return demo


def maybe_patch_low_dim_obs(file_path, obj, task_name):
    if os.path.basename(file_path) != "low_dim_obs.pkl":
        return obj
    return patch_demo_with_object_6d_pose(obj, task_name)
