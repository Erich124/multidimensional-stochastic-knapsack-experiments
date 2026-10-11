"""Run large-instance scaling experiment with common random numbers."""

from __future__ import annotations

import time

from .baselines import random_order, reward_density_order, surrogate_density_order
from .dgv_polymatroid import build_polymatroid_policy
from .dgv_simple import build_simple_policy
from .generators import controlled_instance
from .study_utils import (
    REGIMES,
    append_csv,
    ensure_dirs,
    monte_carlo_common,
    reset_owned_outputs,
    summarize,
    surrogate_diagnostics,
    write_csv,
    write_json,
)
from .surrogate import compute_surrogate_stats


SCALING_CONFIG = {
    "instances_per_configuration_regime": 5,
    "dimensions": [2, 3, 4, 5],
    "n_items": [20, 50, 100],
    "regimes": REGIMES,
    "reward_variant": "independent",
    "outcomes_per_item": 2,
    "monte_carlo_samples": 10000,
    "convergence_check_samples": 20000,
    "base_seed": 610000,
}


def run_scaling() -> list[dict[str, object]]:
    ensure_dirs()
    write_json("results/configs/scaling_config.json", SCALING_CONFIG)
    reset_owned_outputs(
        [
            "results/scaling/raw_scaling.csv",
            "results/scaling/scaling_summary.csv",
            "results/scaling/scaling_summary_by_dim_n.csv",
            "results/scaling/mc_convergence_check.csv",
            "results/scaling/run_metadata.json",
        ]
    )
    rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    raw_path = "results/scaling/raw_scaling.csv"
    fieldnames = None
    counter = 0
    start_all = time.perf_counter()
    for regime in SCALING_CONFIG["regimes"]:
        for dims in SCALING_CONFIG["dimensions"]:
            for n_items in SCALING_CONFIG["n_items"]:
                for rep in range(SCALING_CONFIG["instances_per_configuration_regime"]):
                    seed = SCALING_CONFIG["base_seed"] + counter
                    counter += 1
                    instance = controlled_instance(
                        regime,
                        dims,
                        n_items,
                        seed,
                        reward_variant=SCALING_CONFIG["reward_variant"],
                        outcomes_per_item=SCALING_CONFIG["outcomes_per_item"],
                    )
                    stats = compute_surrogate_stats(instance)
                    t0 = time.perf_counter()
                    simple_policy = build_simple_policy(stats)
                    simple_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    poly_policy = build_polymatroid_policy(stats)
                    poly_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    surrogate_order = surrogate_density_order(instance)
                    reward_order = reward_density_order(instance)
                    random = random_order(instance, seed + 77)
                    baseline_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    mc = monte_carlo_common(
                        instance,
                        {
                            "poly": poly_policy.order,
                            "surrogate_density": surrogate_order,
                            "multidim_density": reward_order,
                            "random_order": random,
                        },
                        {"simple": simple_policy.mixture},
                        SCALING_CONFIG["monte_carlo_samples"],
                        seed + 101,
                    )
                    eval_time = time.perf_counter() - t0
                    best = max(stats_row["mean"] for stats_row in mc.values())
                    diag = surrogate_diagnostics(instance)
                    row: dict[str, object] = {
                        "instance": instance.name,
                        "seed": seed,
                        "regime": regime,
                        "d": dims,
                        "n": n_items,
                        "samples": SCALING_CONFIG["monte_carlo_samples"],
                        "BEST_TESTED": best,
                        "simple_mean": mc["simple"]["mean"],
                        "simple_se": mc["simple"]["se"],
                        "simple_ci95_low": mc["simple"]["ci95_low"],
                        "simple_ci95_high": mc["simple"]["ci95_high"],
                        "simple_over_best": mc["simple"]["mean"] / best if best else "",
                        "poly_mean": mc["poly"]["mean"],
                        "poly_se": mc["poly"]["se"],
                        "poly_over_best": mc["poly"]["mean"] / best if best else "",
                        "surrogate_density_mean": mc["surrogate_density"]["mean"],
                        "surrogate_density_over_best": mc["surrogate_density"]["mean"] / best if best else "",
                        "multidim_density_mean": mc["multidim_density"]["mean"],
                        "multidim_density_over_best": mc["multidim_density"]["mean"] / best if best else "",
                        "random_order_mean": mc["random_order"]["mean"],
                        "random_order_over_best": mc["random_order"]["mean"] / best if best else "",
                        "simple_construct_time": simple_time,
                        "poly_construct_time": poly_time,
                        "baseline_construct_time": baseline_time,
                        "evaluation_time": eval_time,
                        "Phi_Z_2": simple_policy.phi_2,
                        "Psi_Z_1": poly_policy.psi_1,
                    }
                    row.update(diag)
                    rows.append(row)
                    if fieldnames is None:
                        fieldnames = list(row.keys())
                    append_csv(raw_path, row, fieldnames)

                    if rep == 0 and dims in (2, 5) and n_items in (20, 100):
                        mc2 = monte_carlo_common(
                            instance,
                            {"poly": poly_policy.order, "surrogate_density": surrogate_order, "multidim_density": reward_order},
                            {"simple": simple_policy.mixture},
                            SCALING_CONFIG["convergence_check_samples"],
                            seed + 101,
                        )
                        convergence_rows.append(
                            {
                                "instance": instance.name,
                                "d": dims,
                                "n": n_items,
                                "regime": regime,
                                "samples_base": SCALING_CONFIG["monte_carlo_samples"],
                                "samples_double": SCALING_CONFIG["convergence_check_samples"],
                                "simple_base": mc["simple"]["mean"],
                                "simple_double": mc2["simple"]["mean"],
                                "poly_base": mc["poly"]["mean"],
                                "poly_double": mc2["poly"]["mean"],
                            }
                        )

    keys = [
        "simple_over_best",
        "poly_over_best",
        "surrogate_density_over_best",
        "multidim_density_over_best",
        "simple_construct_time",
        "poly_construct_time",
        "baseline_construct_time",
        "evaluation_time",
    ]
    write_csv("results/scaling/scaling_summary.csv", summarize(rows, ["regime", "d", "n"], keys))
    write_csv("results/scaling/scaling_summary_by_dim_n.csv", summarize(rows, ["d", "n"], keys))
    write_csv("results/scaling/mc_convergence_check.csv", convergence_rows)
    write_json("results/scaling/run_metadata.json", {"total_instances": len(rows), "total_runtime_sec": time.perf_counter() - start_all})
    return rows


if __name__ == "__main__":
    run_scaling()
