"""Search small exact instances for low observed ratios."""

from __future__ import annotations

import heapq
import time

from .generators import controlled_instance
from .study_utils import (
    append_csv,
    ensure_dirs,
    evaluate_exact_instance,
    instance_to_jsonable,
    write_csv,
    write_json,
)


STRESS_CONFIG = {
    "candidates": 5000,
    "dimensions": [2, 3],
    "n_items": [6, 8],
    "regimes": ["high_variance", "resource_specialized", "near_capacity", "mixed_negative"],
    "reward_variants": ["independent", "size_correlated"],
    "outcomes_per_item": 2,
    "base_seed": 910000,
}


def run_stress_search() -> list[dict[str, object]]:
    ensure_dirs()
    write_json("results/configs/stress_config.json", STRESS_CONFIG)
    rows: list[dict[str, object]] = []
    simple_bottom: list[tuple[float, int, dict[str, object], object]] = []
    poly_bottom: list[tuple[float, int, dict[str, object], object]] = []
    fieldnames = None
    start = time.perf_counter()
    for idx in range(STRESS_CONFIG["candidates"]):
        dims = STRESS_CONFIG["dimensions"][idx % len(STRESS_CONFIG["dimensions"])]
        n_items = STRESS_CONFIG["n_items"][(idx // 2) % len(STRESS_CONFIG["n_items"])]
        regime = STRESS_CONFIG["regimes"][(idx // 4) % len(STRESS_CONFIG["regimes"])]
        reward_variant = STRESS_CONFIG["reward_variants"][(idx // 16) % len(STRESS_CONFIG["reward_variants"])]
        seed = STRESS_CONFIG["base_seed"] + idx
        instance = controlled_instance(regime, dims, n_items, seed, reward_variant, STRESS_CONFIG["outcomes_per_item"])
        row = evaluate_exact_instance(instance)
        rows.append(row)
        if fieldnames is None:
            fieldnames = list(row.keys())
        append_csv("results/stress_search/search_results.csv", row, fieldnames)
        _push_bottom(simple_bottom, float(row["simple_ratio"]), idx, row, instance)
        _push_bottom(poly_bottom, float(row["poly_ratio"]), idx, row, instance)

    simple_rows = [entry[2] for entry in sorted(simple_bottom, key=lambda x: -x[0])]
    poly_rows = [entry[2] for entry in sorted(poly_bottom, key=lambda x: -x[0])]
    write_csv("results/stress_search/simple_bottom20.csv", simple_rows)
    write_csv("results/stress_search/poly_bottom20.csv", poly_rows)
    for rank, (_, _, row, instance) in enumerate(sorted(simple_bottom, key=lambda x: -x[0]), start=1):
        write_json(f"results/stress_search/simple_bottom20/rank_{rank:02d}.json", instance_to_jsonable(instance, row))
    for rank, (_, _, row, instance) in enumerate(sorted(poly_bottom, key=lambda x: -x[0]), start=1):
        write_json(f"results/stress_search/poly_bottom20/rank_{rank:02d}.json", instance_to_jsonable(instance, row))
    write_json(
        "results/stress_search/run_metadata.json",
        {
            "candidates": len(rows),
            "total_runtime_sec": time.perf_counter() - start,
            "lowest_simple_ratio": min(float(r["simple_ratio"]) for r in rows),
            "lowest_poly_ratio": min(float(r["poly_ratio"]) for r in rows),
        },
    )
    return rows


def _push_bottom(heap: list, ratio: float, idx: int, row: dict[str, object], instance: object) -> None:
    entry = (-ratio, idx, row, instance)
    if len(heap) < 20:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


if __name__ == "__main__":
    run_stress_search()
