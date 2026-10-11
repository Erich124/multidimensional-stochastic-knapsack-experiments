import unittest

from experiments.evaluation import (
    exact_mixture_reward,
    exact_order_reward,
    monte_carlo_order_reward,
)
from experiments.models import Instance, make_item


class EvaluationTests(unittest.TestCase):
    def test_deterministic_instance(self):
        instance = Instance(
            "det",
            2,
            (
                make_item("a", 2.0, [(1.0, (0.4, 0.2))]),
                make_item("b", 3.0, [(1.0, (0.5, 0.7))]),
            ),
        )
        self.assertAlmostEqual(exact_order_reward(instance, (0, 1)), 5.0)

    def test_failure_terminates_and_failed_item_zero(self):
        instance = Instance(
            "fail",
            1,
            (
                make_item("a", 2.0, [(1.0, (0.6,))]),
                make_item("b", 100.0, [(1.0, (0.6,))]),
                make_item("c", 1000.0, [(1.0, (0.0,))]),
            ),
        )
        self.assertAlmostEqual(exact_order_reward(instance, (0, 1, 2)), 2.0)

    def test_tiny_exact_manual_enumeration(self):
        instance = Instance(
            "two_item",
            1,
            (
                make_item("a", 4.0, [(0.5, (0.4,)), (0.5, (0.8,))]),
                make_item("b", 6.0, [(1.0, (0.5,))]),
            ),
        )
        # If a=0.4, b fits and reward is 10. If a=0.8, b fails and reward is 4.
        self.assertAlmostEqual(exact_order_reward(instance, (0, 1)), 7.0)

    def test_mixture_exact(self):
        instance = Instance(
            "mix",
            1,
            (
                make_item("a", 1.0, [(1.0, (0.2,))]),
                make_item("b", 3.0, [(1.0, (0.2,))]),
            ),
        )
        self.assertAlmostEqual(
            exact_mixture_reward(instance, ((0.25, (0,)), (0.75, (1,)))),
            2.5,
        )

    def test_monte_carlo_close_to_exact(self):
        instance = Instance(
            "mc",
            1,
            (
                make_item("a", 4.0, [(0.5, (0.4,)), (0.5, (0.8,))]),
                make_item("b", 6.0, [(1.0, (0.5,))]),
            ),
        )
        estimate = monte_carlo_order_reward(instance, (0, 1), samples=30000, seed=123)
        self.assertLess(abs(estimate - 7.0), 0.08)

    def test_execution_awards_c_not_w(self):
        instance = Instance(
            "reward_sep",
            1,
            (make_item("risky", 10.0, [(0.5, (0.2,)), (0.5, (1.2,))]),),
        )
        self.assertAlmostEqual(exact_order_reward(instance, (0,)), 5.0)


if __name__ == "__main__":
    unittest.main()

