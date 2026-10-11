import random
import unittest

from experiments.dgv_simple import build_simple_policy
from experiments.evaluation import exact_mixture_reward, monte_carlo_mixture_reward
from experiments.generators import random_instance
from experiments.surrogate import compute_surrogate_stats
from experiments.validation import simple_guarantee_value


class DGVSimpleTests(unittest.TestCase):
    def test_probability_normalization_random_instances(self):
        for seed in range(20):
            instance = random_instance(1, 5, seed, outcomes_per_item=2)
            policy = build_simple_policy(compute_surrogate_stats(instance))
            self.assertAlmostEqual(sum(prob for prob, _ in policy.mixture), 1.0)
            for prob, _ in policy.mixture:
                self.assertGreaterEqual(prob, -1e-10)

    def test_exact_mixture_matches_internal_randomization_mc(self):
        instance = random_instance(1, 5, 44, outcomes_per_item=2)
        policy = build_simple_policy(compute_surrogate_stats(instance))
        exact = exact_mixture_reward(instance, policy.mixture)
        estimate = monte_carlo_mixture_reward(instance, policy.mixture, samples=50000, seed=99)
        self.assertLess(abs(exact - estimate), 0.15)

    def test_one_dimensional_simple_guarantee_sanity(self):
        rng = random.Random(777)
        for _ in range(20):
            seed = rng.randrange(1_000_000)
            instance = random_instance(1, 5, seed, outcomes_per_item=2)
            alg, lower = simple_guarantee_value(instance)
            self.assertGreaterEqual(alg + 1e-8, lower)


if __name__ == "__main__":
    unittest.main()

