"""Run the supplemental explicit-overflow exact benchmark."""

from __future__ import annotations

import time

from .generators import explicit_overflow_instance
from .study_utils import (
    append_csv,
    ensure_dirs,
    evaluate_exact_instance,
    reset_owned_outputs,
    summarize,
    write_csv,
    write_json,
)


OVERFLOW_CONFIG = {
    "instances_per_configuration": 30,
    "dimensions": [2, 3, 4, 5],
    "n_items": 6,
    "reward_variants": ["independent", "size_correlated"],
    "base_seed": 930000,
    "family": "explicit_overflow",
}


RAW_PATH = "results/overflow/raw_overflow.csv"
SUMMARY_PATH = "results/overflow/overflow_summary.csv"
METADATA_PATH = "results/overflow/run_metadata.json"


def run_overflow() -> list[dict[str, object]]:
    ensure_dirs()
    reset_owned_outputs([RAW_PATH, SUMMARY_PATH, METADATA_PATH])
    rows: list[dict[str, object]] = []
    fieldnames: list[str] | None = None
    counter = 0
    start = time.perf_counter()
    for reward_variant in OVERFLOW_CONFIG["reward_variants"]:
        for dims in OVERFLOW_CONFIG["dimensions"]:
            for _rep in range(OVERFLOW_CONFIG["instances_per_configuration"]):
                seed = OVERFLOW_CONFIG["base_seed"] + counter
                counter += 1
                instance = explicit_overflow_instance(
                    dims=int(dims),
                    n_items=int(OVERFLOW_CONFIG["n_items"]),
                    seed=seed,
                    reward_variant=str(reward_variant),
                )
                row = evaluate_exact_instance(instance)
                row["mean_item_overflow_prob"] = sum(instance.overflow_probs) / len(instance.overflow_probs)  # type: ignore[attr-defined]
                rows.append(row)
                if fieldnames is None:
                    fieldnames = list(row.keys())
                append_csv(RAW_PATH, row, fieldnames)

    ratio_keys = [
        "simple_ratio",
        "poly_ratio",
        "surrogate_density_ratio",
        "multidim_density_ratio",
        "mean_item_overflow_prob",
        "ADAPT_dp_time",
        "ADAPT_dp_states",
    ]
    summary = summarize(rows, ["reward_variant", "d"], ratio_keys)
    write_csv(SUMMARY_PATH, summary)
    write_json(
        METADATA_PATH,
        {
            "config": OVERFLOW_CONFIG,
            "total_instances": len(rows),
            "total_runtime_sec": time.perf_counter() - start,
            "mean_simple_ratio": _mean(rows, "simple_ratio"),
            "mean_poly_ratio": _mean(rows, "poly_ratio"),
            "mean_surrogate_density_ratio": _mean(rows, "surrogate_density_ratio"),
            "mean_multidim_density_ratio": _mean(rows, "multidim_density_ratio"),
            "mean_item_overflow_prob": _mean(rows, "mean_item_overflow_prob"),
        },
    )
    return rows


def _mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row[key] != ""]
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    run_overflow()
