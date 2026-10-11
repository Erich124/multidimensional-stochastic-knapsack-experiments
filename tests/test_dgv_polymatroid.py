import random
import unittest

from experiments.dgv_polymatroid import (
    build_polymatroid_policy,
    psi_one_closed_form,
    psi_one_subset_lp_vertex_enumeration,
)
from experiments.generators import random_instance
from experiments.surrogate import compute_surrogate_stats
from experiments.validation import poly_guarantee_value


class DGVPolymatroidTests(unittest.TestCase):
    def test_closed_form_matches_subset_lp_tiny(self):
        for seed in range(10):
            instance = random_instance(1, 3, seed + 900, outcomes_per_item=2)
            stats = compute_surrogate_stats(instance)
            self.assertAlmostEqual(
                psi_one_closed_form(stats),
                psi_one_subset_lp_vertex_enumeration(stats),
                places=7,
            )

    def test_policy_modes_are_valid(self):
        instance = random_instance(2, 5, 2026, outcomes_per_item=2)
        policy = build_polymatroid_policy(compute_surrogate_stats(instance))
        self.assertIn(policy.mode, {"empty", "single_item", "greedy_order"})

    def test_one_dimensional_poly_guarantee_sanity(self):
        rng = random.Random(778)
        for _ in range(20):
            seed = rng.randrange(1_000_000)
            instance = random_instance(1, 5, seed, outcomes_per_item=2)
            alg, lower = poly_guarantee_value(instance)
            self.assertGreaterEqual(alg + 1e-8, lower)


if __name__ == "__main__":
    unittest.main()

