import unittest

from experiments.models import Instance, make_item
from experiments.surrogate import compute_surrogate_stats


class SurrogateTests(unittest.TestCase):
    def test_d1_identity_and_stats(self):
        instance = Instance(
            "d1",
            1,
            (
                make_item("a", 5.0, [(0.25, (0.2,)), (0.75, (1.4,))]),
            ),
        )
        stats = compute_surrogate_stats(instance)
        self.assertAlmostEqual(stats.items[0].epsilon, 0.25 * 0.2 + 0.75 * 1.0)
        self.assertAlmostEqual(stats.items[0].effective_value, 5.0 * 0.25)

    def test_multidim_epsilon_and_effective_reward(self):
        instance = Instance(
            "manual",
            2,
            (
                make_item(
                    "x",
                    8.0,
                    [(0.2, (0.1, 0.7)), (0.3, (1.2, 0.1)), (0.5, (0.4, 0.5))],
                ),
            ),
        )
        stats = compute_surrogate_stats(instance)
        expected_epsilon = 0.2 * 0.7 + 0.3 * 1.0 + 0.5 * 0.5
        expected_w = 8.0 * (0.2 + 0.5)
        self.assertAlmostEqual(stats.items[0].epsilon, expected_epsilon)
        self.assertAlmostEqual(stats.items[0].effective_value, expected_w)


if __name__ == "__main__":
    unittest.main()

