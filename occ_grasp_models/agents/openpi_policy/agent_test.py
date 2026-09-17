import copy
import unittest

import numpy as np

from agents.openpi_policy import agent as openpi_agent


_TASK = "bimanual_pick_plate"


def _task_texts(label):
    return {
        "1": "%s phase 1" % label,
        "1_to_2": "%s transition 1 to 2" % label,
        "2": "%s phase 2" % label,
        "2_to_3": "%s transition 2 to 3" % label,
        "3": "%s phase 3" % label,
        "3_to_4": "%s transition 3 to 4" % label,
        "4": "%s phase 4" % label,
    }


def _metadata(action_horizon=20, replan_steps=10, protocol="protocol_a"):
    subtask_texts = {_TASK: _task_texts(protocol)}
    return {
        "action_layout": openpi_agent.EXPECTED_ACTION_LAYOUT,
        "language_condition": "oracle_text_%s" % protocol,
        "subtask_protocol": protocol,
        "action_horizon": action_horizon,
        "replan_steps": replan_steps,
        "subtask_texts": subtask_texts,
        "subtask_texts_sha256": openpi_agent._canonical_sha256(subtask_texts),
    }


def _oracle_agent(replan_steps=10):
    return openpi_agent.OpenPIPolicyAgent(
        replan_steps=replan_steps,
        oracle_phase_enabled=True,
        oracle_phase_task=_TASK,
    )


def _observation(phase_state=None):
    if phase_state is None:
        phase_state = [1, 0, 0, 0]
    return {
        "front_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_left_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_right_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "left_joint_positions": np.zeros(7, dtype=np.float32),
        "left_gripper_open": np.ones(1, dtype=np.float32),
        "right_joint_positions": np.zeros(7, dtype=np.float32),
        "right_gripper_open": np.ones(1, dtype=np.float32),
        "lang_goal": "overall task prompt",
        "oracle_phase_state": np.asarray(phase_state, dtype=np.int64),
    }


class _FakeClient(object):
    def __init__(self, actions):
        self.actions = actions
        self.last_observation = None

    def infer(self, observation):
        self.last_observation = observation
        return {"actions": self.actions}


class OracleSelectorTest(unittest.TestCase):
    def test_initial_and_adjacent_forward_types(self):
        self.assertEqual(
            openpi_agent._select_oracle_subtask_type(2, 0, 0, 0, None),
            ("2", "initial_pure"),
        )
        for phase_pair, expected_type in (
            ((1, 2), "1_to_2"),
            ((2, 3), "2_to_3"),
            ((3, 4), "3_to_4"),
        ):
            old_phase, new_phase = phase_pair
            self.assertEqual(
                openpi_agent._select_oracle_subtask_type(
                    new_phase, 8, old_phase, new_phase, 7
                ),
                (expected_type, "single_adjacent_forward"),
            )

    def test_transition_is_one_shot_and_abnormal_changes_fall_back(self):
        fallback_cases = (
            (2, 8, 1, 2, 8),
            (2, 8, 3, 2, 7),
            (3, 8, 1, 3, 7),
            (3, 9, 2, 3, 7),
            (3, 8, 1, 2, 7),
        )
        for values in fallback_cases:
            with self.subTest(values=values):
                selected, reason = openpi_agent._select_oracle_subtask_type(*values)
                self.assertEqual(selected, str(values[0]))
                self.assertEqual(reason, "pure_current")

    def test_invalid_selector_state_fails(self):
        with self.assertRaisesRegex(ValueError, "Cannot replan from phase"):
            openpi_agent._select_oracle_subtask_type(5, 1, 4, 5, 0)
        with self.assertRaisesRegex(ValueError, "moved backwards"):
            openpi_agent._select_oracle_subtask_type(2, 3, 1, 2, 4)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            openpi_agent._select_oracle_subtask_type(2.0, 3, 1, 2, 2)


class OracleStateTest(unittest.TestCase):
    def test_extracts_latest_integer_state(self):
        observation = {
            "oracle_phase_state": np.asarray(
                [[[1, 0, 0, 0], [2, 1, 1, 2]]], dtype=np.int64
            )
        }
        self.assertEqual(
            openpi_agent._extract_oracle_phase_state(observation),
            (2, 1, 1, 2),
        )

    def test_rejects_malformed_or_inconsistent_state(self):
        invalid_states = (
            np.asarray([1, 0, 0], dtype=np.int64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([1, 0, 1, 2], dtype=np.int64),
            np.asarray([2, 1, 2, 2], dtype=np.int64),
            np.asarray([6, 1, 4, 5], dtype=np.int64),
        )
        for state in invalid_states:
            with self.subTest(state=state), self.assertRaises(ValueError):
                openpi_agent._extract_oracle_phase_state(
                    {"oracle_phase_state": state}
                )


class OracleMetadataTest(unittest.TestCase):
    def test_accepts_protocol_independent_h_and_k(self):
        for action_horizon, replan_steps, protocol in (
            (20, 10, "protocol_a"),
            (8, 3, "protocol_b"),
        ):
            with self.subTest(protocol=protocol):
                agent = _oracle_agent(replan_steps)
                agent._validate_server_metadata(
                    _metadata(action_horizon, replan_steps, protocol)
                )
                self.assertEqual(agent._oracle_action_horizon, action_horizon)
                self.assertEqual(
                    agent._oracle_subtask_texts,
                    _task_texts(protocol),
                )

    def test_rejects_invalid_contract_fields(self):
        invalid_metadata = []

        metadata = _metadata()
        metadata["subtask_protocol"] = ""
        invalid_metadata.append(metadata)

        metadata = _metadata()
        metadata["action_horizon"] = True
        invalid_metadata.append(metadata)

        metadata = _metadata()
        metadata["replan_steps"] = True
        invalid_metadata.append(metadata)

        metadata = _metadata()
        metadata["replan_steps"] = 9
        invalid_metadata.append(metadata)

        metadata = _metadata(action_horizon=8, replan_steps=10)
        invalid_metadata.append(metadata)

        metadata = _metadata()
        metadata["subtask_texts_sha256"] = "sha256:wrong"
        invalid_metadata.append(metadata)

        metadata = _metadata()
        del metadata["subtask_texts"][_TASK]
        metadata["subtask_texts_sha256"] = openpi_agent._canonical_sha256(
            metadata["subtask_texts"]
        )
        invalid_metadata.append(metadata)

        metadata = _metadata()
        metadata["subtask_texts"][_TASK]["4"] = 4
        metadata["subtask_texts_sha256"] = openpi_agent._canonical_sha256(
            metadata["subtask_texts"]
        )
        invalid_metadata.append(metadata)

        metadata = _metadata()
        del metadata["subtask_texts"][_TASK]["3_to_4"]
        metadata["subtask_texts_sha256"] = openpi_agent._canonical_sha256(
            metadata["subtask_texts"]
        )
        invalid_metadata.append(metadata)

        metadata = _metadata()
        metadata["subtask_texts"][_TASK]["4"] = ""
        metadata["subtask_texts_sha256"] = openpi_agent._canonical_sha256(
            metadata["subtask_texts"]
        )
        invalid_metadata.append(metadata)

        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), self.assertRaises(RuntimeError):
                _oracle_agent()._validate_server_metadata(copy.deepcopy(metadata))

    def test_old_generated_subtask_metadata_has_no_compatibility_path(self):
        old_metadata = {
            "action_layout": openpi_agent.EXPECTED_ACTION_LAYOUT,
            "language_condition": "generated_subtask",
            "subtask_replan_steps": 10,
            "action_horizon": 20,
        }
        with self.assertRaises(RuntimeError):
            _oracle_agent()._validate_server_metadata(old_metadata)

    def test_act_uses_metadata_text_and_requires_exact_h_by_16_shape(self):
        for action_horizon, replan_steps, protocol in (
            (20, 10, "protocol_a"),
            (8, 3, "protocol_b"),
        ):
            with self.subTest(protocol=protocol):
                agent = _oracle_agent(replan_steps)
                agent._validate_server_metadata(
                    _metadata(action_horizon, replan_steps, protocol)
                )
                client = _FakeClient(
                    np.zeros((action_horizon, 16), dtype=np.float32)
                )
                agent._client = client

                result = agent.act(999, _observation(), deterministic=True)

                self.assertEqual(
                    client.last_observation["prompt"],
                    _task_texts(protocol)["1"],
                )
                self.assertEqual(result.info["pred_info"]["type"], "1")
                self.assertEqual(result.info["pred_info"]["episode_step"], 0)

        agent = _oracle_agent(3)
        agent._validate_server_metadata(_metadata(8, 3, "shape_check"))
        agent._client = _FakeClient(np.zeros((7, 16), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "metadata H"):
            agent.act(0, _observation(), deterministic=True)


if __name__ == "__main__":
    unittest.main()
