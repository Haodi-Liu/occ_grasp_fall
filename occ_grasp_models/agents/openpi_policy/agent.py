"""OpenPI websocket policy adapter for occ closed-loop evaluation.

This agent stays lightweight on the occ / RLBench side:
- it implements the YARR ``Agent`` interface expected by ``eval.py``
- it extracts raw RLBench observations without ``PreprocessAgent``
- it sends them to the openpi websocket server via ``openpi-client``
- it reorders openpi's left-first 16D action into occ's right-first layout
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from yarr.agents.agent import ActResult, Agent, Summary

logger = logging.getLogger(__name__)


class OpenPIPolicyAgent(Agent):
    """Wrap an openpi websocket inference server as an occ-compatible agent."""

    def __init__(self, host="localhost", port=8000, replan_steps=1):
        self._host = host
        self._port = int(port)
        self._replan_steps = int(replan_steps)
        if self._replan_steps < 1:
            raise ValueError("replan_steps must be >= 1, got %r." % replan_steps)

        self._client = None
        self._server_metadata = None
        self._action_cache = None
        self._action_cache_idx = 0

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
        self.reset()

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

        if need_replan:
            obs_for_openpi = self._extract_openpi_obs(observation)
            result = self._client.infer(obs_for_openpi)
            if "actions" not in result:
                raise KeyError("openpi server response is missing 'actions': %r" % (result,))

            action_cache = np.asarray(result["actions"], dtype=np.float32)
            if action_cache.ndim == 1:
                action_cache = action_cache[None, :]
            if action_cache.ndim != 2 or action_cache.shape[-1] != 16:
                raise ValueError(
                    "Expected openpi action chunk with shape (T, 16), got %s."
                    % (tuple(action_cache.shape),)
                )
            if action_cache.shape[0] == 0:
                raise ValueError("openpi server returned an empty action chunk.")

            self._action_cache = action_cache
            self._action_cache_idx = 0

        openpi_action = np.array(self._action_cache[self._action_cache_idx], copy=True)
        self._action_cache_idx += 1

        if not np.isfinite(openpi_action).all():
            raise ValueError("openpi server returned non-finite action values: %r" % openpi_action)

        # occ's BimanualDiscrete gripper expects hard 0/1 commands.
        openpi_action[7] = 1.0 if openpi_action[7] > 0.5 else 0.0
        openpi_action[15] = 1.0 if openpi_action[15] > 0.5 else 0.0

        occ_action = np.concatenate(
            [
                openpi_action[8:15],   # right joints
                openpi_action[15:16],  # right gripper
                openpi_action[0:7],    # left joints
                openpi_action[7:8],    # left gripper
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        return ActResult(occ_action)

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

    def _extract_openpi_obs(self, observation):
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

        state = np.concatenate(
            [
                _extract_latest(observation["left_joint_positions"]).astype(np.float32).reshape(-1),
                _extract_latest(observation["left_gripper_open"]).astype(np.float32).reshape(-1),
                _extract_latest(observation["right_joint_positions"]).astype(np.float32).reshape(-1),
                _extract_latest(observation["right_gripper_open"]).astype(np.float32).reshape(-1),
            ],
            axis=0,
        )
        if state.shape != (16,):
            raise ValueError(
                "Expected concatenated RLBench state to have shape (16,), got %s."
                % (tuple(state.shape),)
            )

        prompt = _coerce_prompt(observation.get("lang_goal", ""))

        return {
            "observation/state": state,
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
