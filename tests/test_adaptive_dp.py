import itertools
import unittest

from experiments.adaptive_dp import exact_adaptive_optimum
from experiments.evaluation import exact_order_reward
from experiments.models import Instance, make_item


class AdaptiveDPTests(unittest.TestCase):
    def test_adaptive_branching_beats_fixed_orders(self):
        instance = Instance(
            "branching",
            1,
            (
                make_item("probe", 5.0, [(0.5, (0.1,)), (0.5, (0.6,))]),
                make_item("small", 10.0, [(1.0, (0.4,))]),
                make_item("large", 11.0, [(1.0, (0.9,))]),
            ),
        )
        result = exact_adaptive_optimum(instance, recover_policy=True)
        fixed_best = max(
            exact_order_reward(instance, order)
            for order in itertools.permutations(range(len(instance.items)))
        )
        self.assertAlmostEqual(result.value, 15.5)
        self.assertGreater(result.value, fixed_best)
        self.assertEqual(result.first_action, 0)


if __name__ == "__main__":
    unittest.main()

