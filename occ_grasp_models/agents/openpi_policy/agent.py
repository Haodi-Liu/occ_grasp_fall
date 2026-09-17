"""OpenPI websocket policy adapter for occ closed-loop evaluation (joint-control version).

This agent stays lightweight on the occ / RLBench side:
- it implements the YARR ``Agent`` interface expected by ``eval.py``
- it extracts raw RLBench observations without ``PreprocessAgent``
- it sends them to the openpi websocket server via ``openpi-client``
- it converts openpi's left-first 16D joint action into RLBench's right-first 16D joint action
"""

import hashlib
import json
import logging

import numpy as np

from yarr.agents.agent import ActResult, Agent

logger = logging.getLogger(__name__)


# Must match `RLBENCH_ACTION_LAYOUT` in openpi/src/openpi/policies/rlbench_policy.py.
# The server reports this through its policy metadata; the client refuses to
# decode actions unless the layout matches, so a future change to the openpi
# side's left/right convention can't silently swap the arms at evaluation time.
EXPECTED_ACTION_LAYOUT = "rlbench_bimanual_left_first_joint16"

# Hysteresis thresholds applied to the continuous gripper values returned by
# the openpi server. The model is trained to regress 0/1 but flow-matching
# leaves values around ~0.05-0.95 with some jitter near 0.5. Using a single
# 0.5 threshold causes chattering when a value drifts back and forth across
# the boundary. With two thresholds we only flip when the prediction clearly
# crosses the *other* side, which matches how a real bimanual gripper would
# be commanded.
_GRIPPER_OPEN_THRESHOLD = 0.6   # close -> open requires >= 0.6
_GRIPPER_CLOSE_THRESHOLD = 0.4  # open  -> close requires <= 0.4

_FORWARD_TYPES = {
    (1, 2): "1_to_2",
    (2, 3): "2_to_3",
    (3, 4): "3_to_4",
}
_ORACLE_TYPES = {
    "1",
    "1_to_2",
    "2",
    "2_to_3",
    "3",
    "3_to_4",
    "4",
}


def _select_oracle_subtask_type(
    current_phase,
    change_count,
    last_from,
    last_to,
    previous_anchor_change_count,
):
    values = {
        "current_phase": current_phase,
        "change_count": change_count,
        "last_from": last_from,
        "last_to": last_to,
    }
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("%s must be an integer, got %r." % (name, value))
    if current_phase not in (1, 2, 3, 4):
        raise ValueError("Cannot replan from phase %r." % current_phase)
    if change_count < 0:
        raise ValueError("Oracle phase change count must be non-negative.")
    if previous_anchor_change_count is None:
        return str(current_phase), "initial_pure"
    if (
        isinstance(previous_anchor_change_count, (bool, np.bool_))
        or not isinstance(previous_anchor_change_count, (int, np.integer))
        or previous_anchor_change_count < 0
    ):
        raise ValueError("previous_anchor_change_count must be a non-negative integer or None.")

    delta = change_count - previous_anchor_change_count
    if delta < 0:
        raise ValueError("Oracle phase change count moved backwards.")
    pair = (last_from, last_to)
    if delta == 1 and pair in _FORWARD_TYPES and current_phase == last_to:
        return _FORWARD_TYPES[pair], "single_adjacent_forward"
    return str(current_phase), "pure_current"


class OpenPIPolicyAgent(Agent):
    """Wrap an openpi websocket inference server as an occ-compatible joint agent."""

    def __init__(
        self,
        host="localhost",
        port=8000,
        replan_steps=1,
        oracle_phase_enabled=False,
        oracle_phase_task=None,
    ):
        self._host = host
        self._port = int(port)
        self._replan_steps = int(replan_steps)
        if self._replan_steps < 1:
            raise ValueError("replan_steps must be >= 1, got %r." % replan_steps)

        self._oracle_phase_enabled = bool(oracle_phase_enabled)
        if self._oracle_phase_enabled and (
            not isinstance(oracle_phase_task, str) or not oracle_phase_task
        ):
            raise ValueError("oracle_phase_task must be a non-empty string in oracle mode.")
        self._oracle_phase_task = oracle_phase_task

        self._client = None
        self._server_metadata = None
        self._action_cache = None
        self._action_cache_idx = 0
        self._episode_step = 0
        self._oracle_subtask_texts = None
        self._oracle_action_horizon = None
        self._previous_anchor_change_count = None
        # Last commanded gripper state per arm; used for the hysteresis decision
        # in `_binarize_gripper`. Defaults to "open" (1.0) which matches the
        # RLBench reset pose.
        self._last_gripper_left = 1.0
        self._last_gripper_right = 1.0

    def build(self, training, device=None):
        del device
        if training:
            raise NotImplementedError(
                "OpenPIPolicyAgent supports evaluation only. "
                "Use openpi's native training entrypoints for training."
            )
        logger.info(
            "OpenPIPolicyAgent.build(training=%s, host=%s, port=%s, replan_steps=%s)",
            training,
            self._host,
            self._port,
            self._replan_steps,
        )

    def reset(self):
        self._action_cache = None
        self._action_cache_idx = 0
        self._episode_step = 0
        self._previous_anchor_change_count = None
        # Re-prime the hysteresis state at episode start so the first decision
        # doesn't depend on whatever the previous episode ended with.
        self._last_gripper_left = 1.0
        self._last_gripper_right = 1.0
        if self._client is not None and hasattr(self._client, "reset"):
            self._client.reset()

    def set_aux_eval_cfg(self, cfg):
        del cfg

    def set_episode_index_path(self, path):
        del path

    def load_weights(self, savedir):
        """Connect to the openpi websocket server.

        ``savedir`` exists only to satisfy the occ runner contract; actual weights are
        loaded by ``scripts/serve_policy.py`` in the separate openpi process.
        """

        logger.info("OpenPIPolicyAgent.load_weights(savedir=%s)", savedir)
        ws_client = _import_ws_client()
        logger.info("Connecting to openpi server at %s:%s ...", self._host, self._port)
        self._client = ws_client.WebsocketClientPolicy(host=self._host, port=self._port)
        self._server_metadata = self._client.get_server_metadata()
        logger.info("Connected to openpi server. Metadata: %s", self._server_metadata)
        self._validate_server_metadata(self._server_metadata)
        self.reset()

    def _validate_server_metadata(self, metadata):
        """Fail loudly if the openpi server uses a different action convention."""
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "openpi server returned non-dict metadata (%r); cannot verify "
                "action_layout. Refusing to run to avoid silently mis-decoding "
                "left/right arms." % (metadata,)
            )
        layout = metadata.get("action_layout")
        if layout is None:
            raise RuntimeError(
                "openpi server metadata is missing 'action_layout'. The serve "
                "config likely predates the rlbench layout-id contract; either "
                "upgrade the openpi side to set `policy_metadata={'action_layout': "
                "'%s'}` or run a server you trust matches this client's decoder."
                % EXPECTED_ACTION_LAYOUT
            )
        if layout != EXPECTED_ACTION_LAYOUT:
            raise RuntimeError(
                "openpi server reports action_layout=%r but this client decodes "
                "actions as %r. Refusing to run; left/right arms would be "
                "swapped silently." % (layout, EXPECTED_ACTION_LAYOUT)
            )

        if self._oracle_phase_enabled:
            self._bind_oracle_metadata(metadata)

    def _bind_oracle_metadata(self, metadata):
        for field in ("language_condition", "subtask_protocol"):
            value = metadata.get(field)
            if not isinstance(value, str) or not value:
                raise RuntimeError("Oracle server metadata has invalid %s: %r." % (field, value))

        action_horizon = metadata.get("action_horizon")
        server_replan_steps = metadata.get("replan_steps")
        for field, value in (
            ("action_horizon", action_horizon),
            ("replan_steps", server_replan_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RuntimeError("Oracle server metadata has invalid %s: %r." % (field, value))
        if server_replan_steps != self._replan_steps:
            raise RuntimeError(
                "Server replan_steps=%r but OCC replan_steps=%r."
                % (server_replan_steps, self._replan_steps)
            )
        if server_replan_steps > action_horizon:
            raise RuntimeError(
                "Invalid oracle H/K metadata: replan_steps=%r, action_horizon=%r."
                % (server_replan_steps, action_horizon)
            )

        subtask_texts = metadata.get("subtask_texts")
        if not isinstance(subtask_texts, dict):
            raise RuntimeError("Oracle server metadata is missing a subtask_texts mapping.")
        expected_digest = metadata.get("subtask_texts_sha256")
        actual_digest = _canonical_sha256(subtask_texts)
        if not isinstance(expected_digest, str) or expected_digest != actual_digest:
            raise RuntimeError(
                "Oracle subtask_texts hash mismatch: metadata=%r, actual=%r."
                % (expected_digest, actual_digest)
            )

        task_texts = subtask_texts.get(self._oracle_phase_task)
        if not isinstance(task_texts, dict):
            raise RuntimeError(
                "Oracle subtask_texts is missing task %r." % self._oracle_phase_task
            )
        if set(task_texts) != _ORACLE_TYPES:
            raise RuntimeError(
                "Oracle task %r has invalid subtask types: %s."
                % (self._oracle_phase_task, sorted(task_texts))
            )
        invalid_types = [
            subtask_type
            for subtask_type, text in task_texts.items()
            if not isinstance(text, str) or not text
        ]
        if invalid_types:
            raise RuntimeError(
                "Oracle task %r has invalid text for types %s."
                % (self._oracle_phase_task, sorted(invalid_types))
            )

        self._oracle_action_horizon = action_horizon
        self._oracle_subtask_texts = dict(task_texts)

    def act(self, step, observation, deterministic):
        del step, deterministic
        if self._client is None:
            raise RuntimeError(
                "OpenPIPolicyAgent has no active websocket client. "
                "Did eval runner call load_weights() successfully?"
            )

        need_replan = (
            self._action_cache is None
            or self._action_cache_idx >= self._replan_steps
            or self._action_cache_idx >= len(self._action_cache)
        )

        selector_decision = None
        if need_replan:
            prompt_override = None
            if self._oracle_phase_enabled:
                phase, count, last_from, last_to = _extract_oracle_phase_state(observation)
                subtask_type, reason = _select_oracle_subtask_type(
                    phase,
                    count,
                    last_from,
                    last_to,
                    self._previous_anchor_change_count,
                )
                delta = (
                    None
                    if self._previous_anchor_change_count is None
                    else count - self._previous_anchor_change_count
                )
                self._previous_anchor_change_count = count
                prompt_override = self._oracle_subtask_texts[subtask_type]
                selector_decision = {
                    "episode_step": self._episode_step,
                    "task": self._oracle_phase_task,
                    "phase": phase,
                    "change_count": count,
                    "change_delta": delta,
                    "last_change": [last_from, last_to],
                    "type": subtask_type,
                    "reason": reason,
                    "prompt": prompt_override,
                }
                logger.info(
                    "oracle_selector_anchor=%s",
                    json.dumps(selector_decision, sort_keys=True),
                )
            obs_for_openpi = self._extract_openpi_obs(
                observation, prompt_override=prompt_override
            )
            result = self._client.infer(obs_for_openpi)
            if "actions" not in result:
                raise KeyError("openpi server response is missing 'actions': %r" % (result,))

            action_cache = np.asarray(result["actions"], dtype=np.float32)
            if action_cache.ndim == 1:
                action_cache = action_cache[None, :]
            if action_cache.ndim != 2 or action_cache.shape[-1] != 16:
                raise ValueError(
                    "Expected openpi joint action chunk with shape (T, 16), got %s."
                    % (tuple(action_cache.shape),)
                )
            if action_cache.shape[0] == 0:
                raise ValueError("openpi server returned an empty action chunk.")
            if self._oracle_phase_enabled and action_cache.shape != (
                self._oracle_action_horizon,
                16,
            ):
                raise ValueError(
                    "Oracle server action shape disagrees with metadata H: %s."
                    % (tuple(action_cache.shape),)
                )

            self._action_cache = action_cache
            self._action_cache_idx = 0

        openpi_action = np.array(self._action_cache[self._action_cache_idx], copy=True)
        self._action_cache_idx += 1

        if not np.isfinite(openpi_action).all():
            raise ValueError("openpi server returned non-finite action values: %r" % openpi_action)

        occ_action = self._openpi_joint16_to_occ_joint16(openpi_action)
        info = {"pred_info": selector_decision} if selector_decision is not None else None
        self._episode_step += 1
        return ActResult(occ_action, info=info)

    def _openpi_joint16_to_occ_joint16(self, action16: np.ndarray) -> np.ndarray:
        """Wrap the layout swap with stateful, per-arm hysteresis on the grippers."""
        left_gripper = _binarize_with_hysteresis(float(action16[7]), self._last_gripper_left)
        right_gripper = _binarize_with_hysteresis(float(action16[15]), self._last_gripper_right)
        self._last_gripper_left = left_gripper
        self._last_gripper_right = right_gripper
        return _openpi_joint16_to_occ_joint16(
            action16,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
        )

    def update(self, step, replay_sample):
        del step, replay_sample
        raise NotImplementedError("OpenPIPolicyAgent does not support training updates.")

    def update_summaries(self):
        return []

    def act_summaries(self):
        return []

    def save_weights(self, savedir):
        del savedir
        raise NotImplementedError("OpenPIPolicyAgent does not save local weights.")

    def _extract_openpi_obs(self, observation, prompt_override=None):
        required_keys = (
            "front_rgb",
            "wrist_left_rgb",
            "wrist_right_rgb",
            "left_joint_positions",
            "left_gripper_open",
            "right_joint_positions",
            "right_gripper_open",
        )
        missing = [key for key in required_keys if key not in observation]
        if missing:
            raise KeyError("Observation is missing keys required by OPENPI_POLICY: %s" % missing)

        front_rgb = _coerce_uint8_image(
            _extract_latest(observation["front_rgb"], is_image=True),
            "front_rgb",
        )
        wrist_left_rgb = _coerce_uint8_image(
            _extract_latest(observation["wrist_left_rgb"], is_image=True),
            "wrist_left_rgb",
        )
        wrist_right_rgb = _coerce_uint8_image(
            _extract_latest(observation["wrist_right_rgb"], is_image=True),
            "wrist_right_rgb",
        )

        left_joints = _extract_joint_positions(observation["left_joint_positions"], "left_joint_positions")
        left_gripper = _extract_gripper_scalar(observation["left_gripper_open"], "left_gripper_open")
        right_joints = _extract_joint_positions(observation["right_joint_positions"], "right_joint_positions")
        right_gripper = _extract_gripper_scalar(observation["right_gripper_open"], "right_gripper_open")

        state = np.concatenate(
            [
                left_joints,
                left_gripper,
                right_joints,
                right_gripper,
            ],
            axis=0,
        )
        if state.shape != (16,):
            raise ValueError(
                "Expected concatenated RLBench joint state to have shape (16,), got %s."
                % (tuple(state.shape),)
            )

        prompt = (
            _coerce_prompt(observation.get("lang_goal", ""))
            if prompt_override is None
            else str(prompt_override)
        )

        return {
            "observation/state": state.astype(np.float32, copy=False),
            "observation/front_rgb": front_rgb,
            "observation/wrist_left_rgb": wrist_left_rgb,
            "observation/wrist_right_rgb": wrist_right_rgb,
            "prompt": prompt,
        }


def _import_ws_client():
    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import openpi_client. Install websockets/msgpack in the ppi "
            "environment and add openpi/packages/openpi-client/src to PYTHONPATH "
            "before running OPENPI_POLICY evaluation."
        ) from exc
    return websocket_client_policy


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:%s" % hashlib.sha256(payload).hexdigest()


def _extract_oracle_phase_state(observation):
    if "oracle_phase_state" not in observation:
        raise KeyError("Observation is missing oracle_phase_state in oracle mode.")
    state = _extract_latest(observation["oracle_phase_state"])
    if state.shape != (4,):
        raise ValueError(
            "Expected oracle_phase_state to have shape (4,), got %s."
            % (tuple(state.shape),)
        )
    if np.issubdtype(state.dtype, np.bool_) or not np.issubdtype(
        state.dtype, np.integer
    ):
        raise ValueError(
            "oracle_phase_state must contain true integers, got dtype %s." % state.dtype
        )
    if not np.isfinite(state).all():
        raise ValueError("oracle_phase_state contains non-finite values.")

    current_phase, change_count, last_from, last_to = [int(value) for value in state]
    if current_phase not in (1, 2, 3, 4, 5):
        raise ValueError("oracle_phase_state has invalid current phase %r." % current_phase)
    if change_count < 0:
        raise ValueError("oracle_phase_state has a negative change count.")
    if change_count == 0:
        if (last_from, last_to) != (0, 0):
            raise ValueError("Zero oracle change count requires last pair (0, 0).")
    elif (
        last_from not in (1, 2, 3, 4, 5)
        or last_to not in (1, 2, 3, 4, 5)
        or last_from == last_to
    ):
        raise ValueError("Nonzero oracle change count requires a valid changed phase pair.")
    return current_phase, change_count, last_from, last_to


def _extract_latest(value, is_image=False):
    import torch

    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    while array.ndim > 0 and array.shape[0] == 1 and array.ndim > (3 if is_image else 1):
        array = array[0]

    if is_image:
        if array.ndim == 4:
            array = array[-1]
        if array.ndim != 3:
            raise ValueError("Expected image observation to have 3 dims, got %s." % (array.shape,))
    else:
        if array.ndim == 2:
            array = array[-1]
        if array.ndim > 1:
            raise ValueError(
                "Expected low-dimensional observation to have <=1 dims after slicing, got %s."
                % (array.shape,)
            )

    return np.asarray(array)


def _coerce_uint8_image(image, name):
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("Expected %s to have 3 dims, got %s." % (name, array.shape))

    if np.issubdtype(array.dtype, np.floating):
        min_value = float(np.nanmin(array))
        max_value = float(np.nanmax(array))
        if min_value >= -1.0 and max_value <= 1.0:
            if min_value < 0.0:
                array = (np.clip(array, -1.0, 1.0) + 1.0) * 127.5
            else:
                array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
        array = np.rint(array).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def _coerce_prompt(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Expected lang_goal array with exactly one element, got shape %s." % (value.shape,))
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _extract_joint_positions(value, name: str) -> np.ndarray:
    joints = _extract_latest(value).astype(np.float32).reshape(-1)
    if joints.shape != (7,):
        raise ValueError("Expected %s to contain 7 joint values, got shape %s." % (name, joints.shape))
    return joints.astype(np.float32, copy=False)


def _extract_gripper_scalar(value, name: str) -> np.ndarray:
    gripper = _extract_latest(value).astype(np.float32).reshape(-1)
    if gripper.shape != (1,):
        raise ValueError("Expected %s to contain exactly one value, got shape %s." % (name, gripper.shape))
    return gripper.astype(np.float32, copy=False)


def _binarize_with_hysteresis(value: float, last_command: float) -> float:
    """Discretize a continuous gripper prediction with hysteresis.

    ``value`` is the openpi server's continuous gripper prediction (already
    clipped to roughly [0, 1] on the server side). ``last_command`` is what
    we previously commanded for this gripper (0.0 = closed, 1.0 = open).
    A single 0.5 threshold causes chatter when the prediction sits near 0.5;
    the two-threshold scheme below only flips when the prediction clearly
    crosses to the opposite side, matching how a stateful gripper would be
    driven.
    """
    if not np.isfinite(value):
        return float(last_command)
    v = float(value)
    if last_command >= 0.5:
        # Currently commanding "open"; flip to closed only when clearly low.
        return 0.0 if v <= _GRIPPER_CLOSE_THRESHOLD else 1.0
    # Currently commanding "closed"; flip to open only when clearly high.
    return 1.0 if v >= _GRIPPER_OPEN_THRESHOLD else 0.0


def _openpi_joint16_to_occ_joint16(
    action16: np.ndarray,
    *,
    left_gripper: float,
    right_gripper: float,
) -> np.ndarray:
    action16 = np.asarray(action16, dtype=np.float32).reshape(-1)
    if action16.shape != (16,):
        raise ValueError("Expected 16D joint action, got %s." % (action16.shape,))

    left_joints = action16[0:7]
    left_gripper_arr = np.asarray([left_gripper], dtype=np.float32)
    right_joints = action16[8:15]
    right_gripper_arr = np.asarray([right_gripper], dtype=np.float32)

    occ_action = np.concatenate(
        [
            right_joints,
            right_gripper_arr,
            left_joints,
            left_gripper_arr,
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    if occ_action.shape != (16,):
        raise ValueError("Expected 16D occ joint action, got %s." % (occ_action.shape,))
    return occ_action
