"""Run the exact small-instance Round 3 experiment."""

from __future__ import annotations

import time

from .generators import controlled_instance
from .study_utils import (
    REGIMES,
    append_csv,
    ensure_dirs,
    evaluate_exact_instance,
    instance_to_jsonable,
    low_poly_details,
    reset_owned_outputs,
    summarize,
    write_csv,
    write_json,
)


EXACT_CONFIG = {
    "instances_per_configuration_regime": 30,
    "reward_variants": ["independent", "size_correlated"],
    "outcomes_per_item": 2,
    "dimensions_and_sizes": {
        2: [6, 8, 10, 12],
        3: [6, 8, 10],
        4: [5, 6, 8],
        5: [5, 6],
    },
    "regimes": REGIMES,
    "base_seed": 310000,
}


RAW_PATH = "results/exact/raw_exact.csv"
LOW_POLY_PATH = "results/diagnostics/low_poly_cases.csv"
DIAG_PATH = "results/diagnostics/surrogate_diagnostics.csv"


def run_exact() -> list[dict[str, object]]:
    ensure_dirs()
    write_json("results/configs/exact_config.json", EXACT_CONFIG)
    reset_owned_outputs(
        [
            RAW_PATH,
            LOW_POLY_PATH,
            DIAG_PATH,
            "results/exact/exact_summary_by_regime_dim_n.csv",
            "results/exact/exact_summary_by_regime_dim.csv",
            "results/exact/exact_summary.csv",
            "results/exact/run_metadata.json",
        ]
    )
    rows: list[dict[str, object]] = []
    low_poly_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []

    fieldnames: list[str] | None = None
    low_fieldnames: list[str] | None = None
    diag_fieldnames: list[str] | None = None
    counter = 0
    start = time.perf_counter()
    for reward_variant in EXACT_CONFIG["reward_variants"]:
        for regime in EXACT_CONFIG["regimes"]:
            for dims, sizes in EXACT_CONFIG["dimensions_and_sizes"].items():
                for n_items in sizes:
                    for rep in range(EXACT_CONFIG["instances_per_configuration_regime"]):
                        seed = EXACT_CONFIG["base_seed"] + counter
                        counter += 1
                        instance = controlled_instance(
                            regime,
                            int(dims),
                            int(n_items),
                            seed,
                            reward_variant=reward_variant,
                            outcomes_per_item=EXACT_CONFIG["outcomes_per_item"],
                        )
                        row = evaluate_exact_instance(instance)
                        rows.append(row)
                        if fieldnames is None:
                            fieldnames = list(row.keys())
                        append_csv(RAW_PATH, row, fieldnames)

                        diag = {
                            key: row[key]
                            for key in [
                                "instance",
                                "seed",
                                "regime",
                                "reward_variant",
                                "d",
                                "n",
                                "simple_ratio",
                                "poly_ratio",
                                "mean_EZ",
                                "mean_sum_ES",
                                "mean_surrogate_geometry_ratio",
                                "avg_pairwise_coordinate_corr",
                            ]
                        }
                        diag_rows.append(diag)
                        if diag_fieldnames is None:
                            diag_fieldnames = list(diag.keys())
                        append_csv(DIAG_PATH, diag, diag_fieldnames)

                        if row["poly_ratio"] != "" and float(row["poly_ratio"]) < 0.60:
                            detail = low_poly_details(instance, row)
                            low_poly_rows.append(detail)
                            if low_fieldnames is None:
                                low_fieldnames = list(detail.keys())
                            append_csv(LOW_POLY_PATH, detail, low_fieldnames)

    summary_keys = [
        "simple_ratio",
        "poly_ratio",
        "surrogate_density_ratio",
        "multidim_density_ratio",
        "ADAPT_dp_time",
        "ADAPT_dp_states",
    ]
    write_csv("results/exact/exact_summary_by_regime_dim_n.csv", summarize(rows, ["regime", "reward_variant", "d", "n"], summary_keys))
    write_csv("results/exact/exact_summary_by_regime_dim.csv", summarize(rows, ["regime", "reward_variant", "d"], summary_keys))
    write_csv("results/exact/exact_summary.csv", summarize(rows, ["regime", "d"], summary_keys))
    write_json(
        "results/exact/run_metadata.json",
        {
            "total_instances": len(rows),
            "total_runtime_sec": time.perf_counter() - start,
            "total_dp_states": sum(int(row["ADAPT_dp_states"]) for row in rows),
            "low_poly_cases": len(low_poly_rows),
        },
    )
    return rows


if __name__ == "__main__":
    run_exact()
