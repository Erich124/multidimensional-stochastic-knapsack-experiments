"""Tiny exact pilot runner. This is intentionally not the full study."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from .adaptive_dp import exact_adaptive_optimum
from .baselines import random_order, reward_density_order, surrogate_density_order
from .dgv_polymatroid import build_polymatroid_policy
from .dgv_simple import build_simple_policy
from .evaluation import exact_mixture_reward, exact_order_reward
from .generators import random_instance
from .surrogate import compute_surrogate_stats


def run_pilot(output_path: str = "results/pilot/pilot_results.csv") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    configs = [(2, 5), (2, 8), (3, 5), (3, 8)]
    for dims, n_items in configs:
        for offset in range(5):
            seed = 1000 + dims * 100 + n_items * 10 + offset
            instance = random_instance(dims, n_items, seed, outcomes_per_item=2 + (offset % 2))
            stats = compute_surrogate_stats(instance)

            t0 = time.perf_counter()
            simple_policy = build_simple_policy(stats)
            simple_build = time.perf_counter() - t0

            t0 = time.perf_counter()
            poly_policy = build_polymatroid_policy(stats)
            poly_build = time.perf_counter() - t0

            t0 = time.perf_counter()
            adaptive = exact_adaptive_optimum(instance)
            adaptive_time = time.perf_counter() - t0

            simple_reward = exact_mixture_reward(instance, simple_policy.mixture)
            poly_reward = exact_order_reward(instance, poly_policy.order)
            surrogate_reward = exact_order_reward(instance, surrogate_density_order(instance))
            reward_density = exact_order_reward(instance, reward_density_order(instance))
            random_reward = exact_order_reward(instance, random_order(instance, seed + 999))

            rows.append(
                {
                    "instance": instance.name,
                    "dims": dims,
                    "n_items": n_items,
                    "seed": seed,
                    "adaptive_exact": adaptive.value,
                    "adaptive_dp_states": adaptive.states_evaluated,
                    "adaptive_dp_runtime_sec": adaptive_time,
                    "simple_expected_reward": simple_reward,
                    "simple_ratio": simple_reward / adaptive.value if adaptive.value else "",
                    "simple_phi_2": simple_policy.phi_2,
                    "simple_build_runtime_sec": simple_build,
                    "polymatroid_expected_reward": poly_reward,
                    "polymatroid_ratio": poly_reward / adaptive.value if adaptive.value else "",
                    "polymatroid_psi_1": poly_policy.psi_1,
                    "polymatroid_build_runtime_sec": poly_build,
                    "surrogate_density_reward": surrogate_reward,
                    "reward_density_reward": reward_density,
                    "random_order_reward": random_reward,
                }
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    run_pilot()

