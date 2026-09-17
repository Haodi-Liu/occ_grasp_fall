import types
import unittest

import numpy as np

from helpers.custom_rlbench_env import CustomRLBenchEnv


class _DummyEvaluator(object):
    def __init__(self, phase=1):
        self.phase = phase
        self.next_phase = phase
        self.evaluate_calls = 0

    def get_current_phase(self):
        return self.phase

    def evaluate_current_phase(self):
        self.evaluate_calls += 1
        old_phase = self.phase
        self.phase = self.next_phase
        return self.phase != old_phase, max(0, self.phase - 1)


def _environment(evaluator):
    env = CustomRLBenchEnv.__new__(CustomRLBenchEnv)
    task = types.SimpleNamespace(phased_evaluator=evaluator)
    env._task = types.SimpleNamespace(_task=task)
    env._reset_oracle_phase_state()
    return env


class OraclePhaseSummaryTest(unittest.TestCase):
    def test_counts_each_change_and_evaluates_once_per_callback(self):
        evaluator = _DummyEvaluator(phase=1)
        env = _environment(evaluator)

        expected_changes = 0
        for next_phase in (1, 2, 1, 4, 5):
            old_phase = evaluator.phase
            evaluator.next_phase = next_phase
            calls_before = evaluator.evaluate_calls

            env._update_phase_evaluation()

            self.assertEqual(evaluator.evaluate_calls, calls_before + 1)
            if next_phase != old_phase:
                expected_changes += 1
                self.assertEqual(env._oracle_phase_last_from, old_phase)
                self.assertEqual(env._oracle_phase_last_to, next_phase)
            self.assertEqual(env._oracle_phase_change_count, expected_changes)

        np.testing.assert_array_equal(
            env._get_oracle_phase_state(),
            np.asarray([5, 4, 4, 5], dtype=np.int64),
        )

    def test_reset_clears_change_summary_without_changing_evaluator(self):
        evaluator = _DummyEvaluator(phase=2)
        env = _environment(evaluator)
        env._oracle_phase_change_count = 3
        env._oracle_phase_last_from = 1
        env._oracle_phase_last_to = 4

        env._reset_oracle_phase_state()

        np.testing.assert_array_equal(
            env._get_oracle_phase_state(),
            np.asarray([2, 0, 0, 0], dtype=np.int64),
        )

    def test_missing_evaluator_is_ignored(self):
        env = _environment(None)
        self.assertIsNone(env._get_oracle_phase_state())
        self.assertEqual(env._update_phase_evaluation(), (False, 0))


if __name__ == "__main__":
    unittest.main()
