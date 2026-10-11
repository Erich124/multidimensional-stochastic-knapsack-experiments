import unittest

from experiments.baselines import surrogate_density_order
from experiments.generators import random_instance
from experiments.validation import check_lifting


class LiftingTests(unittest.TestCase):
    def test_pathwise_lifting_random_realizations(self):
        for seed in range(20):
            instance = random_instance(3, 6, seed + 300, outcomes_per_item=3)
            order = surrogate_density_order(instance)
            self.assertTrue(check_lifting(instance, order, samples=200, seed=seed + 9000))


if __name__ == "__main__":
    unittest.main()

