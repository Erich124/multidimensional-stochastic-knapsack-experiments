"""Shared utilities for Round 3 computational study scripts."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

from .adaptive_dp import exact_adaptive_optimum
from .baselines import random_order, reward_density_order, surrogate_density_order
from .config import TOL
from .dgv_polymatroid import build_polymatroid_policy
from .dgv_simple import build_simple_policy
from .evaluation import exact_mixture_reward, exact_order_reward
from .models import Instance, fits
from .surrogate import compute_surrogate_stats
from .validation import check_lifting


REGIMES = [
    "low_variance",
    "high_variance",
    "positive_correlation",
    "resource_specialized",
    "near_capacity",
]


def ensure_dirs() -> None:
    for path in [
        "results/exact",
        "results/scaling",
        "results/stress_search/simple_bottom20",
        "results/stress_search/poly_bottom20",
        "results/diagnostics",
        "results/tables",
        "results/figures",
        "results/configs",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, data: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_csv(path: str, row: dict[str, object], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_exact_instance(instance: Instance, check_bounds: bool = True) -> dict[str, object]:
    stats = compute_surrogate_stats(instance)

    t0 = time.perf_counter()
    simple_policy = build_simple_policy(stats)
    simple_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    poly_policy = build_polymatroid_policy(stats)
    poly_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    surrogate_order = surrogate_density_order(instance)
    surrogate_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    reward_order = reward_density_order(instance)
    reward_time = time.perf_counter() - t0

    random = random_order(instance, (instance.seed or 0) + 7919)

    simple_reward = exact_mixture_reward(instance, simple_policy.mixture)
    poly_reward = exact_order_reward(instance, poly_policy.order)
    surrogate_reward = exact_order_reward(instance, surrogate_order)
    reward_density_reward = exact_order_reward(instance, reward_order)
    random_reward = exact_order_reward(instance, random)

    t0 = time.perf_counter()
    adaptive = exact_adaptive_optimum(instance)
    adaptive_time = time.perf_counter() - t0

    if check_bounds:
        simple_lower = (7.0 / 32.0) * simple_policy.phi_2
        poly_lower = 0.5 * poly_policy.psi_1
        if simple_reward + 1e-8 < simple_lower:
            raise RuntimeError(f"Simple DGV bound violation on {instance.name}: {simple_reward} < {simple_lower}")
        if poly_reward + 1e-8 < poly_lower:
            raise RuntimeError(f"Polymatroid DGV bound violation on {instance.name}: {poly_reward} < {poly_lower}")
        order_for_lift = simple_policy.mixture[0][1] if simple_policy.mixture else poly_policy.order
        if not check_lifting(instance, order_for_lift, samples=25, seed=(instance.seed or 0) + 17):
            raise RuntimeError(f"Lifting violation on {instance.name}")

    diag = surrogate_diagnostics(instance)
    adapt_value = adaptive.value
    row: dict[str, object] = {
        "instance": instance.name,
        "seed": instance.seed,
        "regime": instance.family.split(":")[0],
        "reward_variant": instance.family.split(":")[1] if ":" in instance.family else "",
        "d": instance.dims,
        "n": len(instance.items),
        "ADAPT_exact": adapt_value,
        "simple_reward": simple_reward,
        "simple_ratio": _ratio(simple_reward, adapt_value),
        "poly_reward": poly_reward,
        "poly_ratio": _ratio(poly_reward, adapt_value),
        "surrogate_density_reward": surrogate_reward,
        "surrogate_density_ratio": _ratio(surrogate_reward, adapt_value),
        "multidim_density_reward": reward_density_reward,
        "multidim_density_ratio": _ratio(reward_density_reward, adapt_value),
        "random_order_reward": random_reward,
        "random_order_ratio": _ratio(random_reward, adapt_value),
        "Phi_Z_2": simple_policy.phi_2,
        "Psi_Z_1": poly_policy.psi_1,
        "simple_over_phi2": _ratio(simple_reward, simple_policy.phi_2),
        "poly_over_psi1": _ratio(poly_reward, poly_policy.psi_1),
        "simple_construct_time": simple_time,
        "poly_construct_time": poly_time,
        "surrogate_baseline_construct_time": surrogate_time,
        "multidim_baseline_construct_time": reward_time,
        "ADAPT_dp_time": adaptive_time,
        "ADAPT_dp_states": adaptive.states_evaluated,
        "poly_mode": poly_policy.mode,
        "poly_order": " ".join(str(i) for i in poly_policy.order),
        "simple_mixture_size": len(simple_policy.mixture),
    }
    row.update(diag)
    return row


def low_poly_details(instance: Instance, exact_row: dict[str, object]) -> dict[str, object]:
    stats = compute_surrogate_stats(instance)
    policy = build_polymatroid_policy(stats)
    full_order = policy.greedy_order
    chosen = policy.order[0] if policy.order else None
    full_density_reward = exact_order_reward(instance, full_order)
    single_reward = exact_order_reward(instance, (chosen,)) if chosen is not None else 0.0
    simple_policy = build_simple_policy(stats)
    detail = {
        "instance": instance.name,
        "seed": instance.seed,
        "regime": exact_row["regime"],
        "d": instance.dims,
        "n": len(instance.items),
        "poly_ratio": exact_row["poly_ratio"],
        "simple_reward": exact_row["simple_reward"],
        "ADAPT_exact": exact_row["ADAPT_exact"],
        "poly_mode": policy.mode,
        "Psi_Z_1": policy.psi_1,
        "threshold": 0.5 * policy.psi_1,
        "chosen_item_index": chosen if chosen is not None else "",
        "chosen_item_w": stats.items[chosen].effective_value if chosen is not None else "",
        "chosen_item_c": instance.items[chosen].reward if chosen is not None else "",
        "full_density_order_reward": full_density_reward,
        "single_item_policy_reward": single_reward,
        "simple_mixture_size": len(simple_policy.mixture),
    }
    return detail


def instance_to_jsonable(instance: Instance, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "name": instance.name,
        "dims": instance.dims,
        "seed": instance.seed,
        "family": instance.family,
        "metrics": metrics,
        "items": [
            {
                "name": item.name,
                "reward": item.reward,
                "outcomes": [
                    {"prob": outcome.prob, "size": list(outcome.size)}
                    for outcome in item.outcomes
                ],
            }
            for item in instance.items
        ],
    }


def surrogate_diagnostics(instance: Instance) -> dict[str, float]:
    ez_values = []
    sum_e_values = []
    ratios = []
    correlations = []
    for item in instance.items:
        ez = sum(outcome.prob * max(outcome.size) for outcome in item.outcomes)
        sums = [0.0 for _ in range(instance.dims)]
        for outcome in item.outcomes:
            for k, val in enumerate(outcome.size):
                sums[k] += outcome.prob * val
        total_sum = sum(sums)
        ez_values.append(ez)
        sum_e_values.append(total_sum)
        if total_sum > TOL:
            ratios.append(ez / total_sum)
        if instance.dims >= 2 and len(item.outcomes) >= 2:
            for a in range(instance.dims):
                for b in range(a + 1, instance.dims):
                    corr = _weighted_corr(
                        [outcome.size[a] for outcome in item.outcomes],
                        [outcome.size[b] for outcome in item.outcomes],
                        [outcome.prob for outcome in item.outcomes],
                    )
                    if corr is not None:
                        correlations.append(corr)
    return {
        "mean_EZ": statistics.mean(ez_values) if ez_values else 0.0,
        "mean_sum_ES": statistics.mean(sum_e_values) if sum_e_values else 0.0,
        "mean_surrogate_geometry_ratio": statistics.mean(ratios) if ratios else 0.0,
        "avg_pairwise_coordinate_corr": statistics.mean(correlations) if correlations else "",
    }


def _weighted_corr(xs: list[float], ys: list[float], ps: list[float]) -> float | None:
    mean_x = sum(p * x for p, x in zip(ps, xs))
    mean_y = sum(p * y for p, y in zip(ps, ys))
    var_x = sum(p * (x - mean_x) ** 2 for p, x in zip(ps, xs))
    var_y = sum(p * (y - mean_y) ** 2 for p, y in zip(ps, ys))
    if var_x <= TOL or var_y <= TOL:
        return None
    cov = sum(p * (x - mean_x) * (y - mean_y) for p, x, y in zip(ps, xs, ys))
    return cov / math.sqrt(var_x * var_y)


def summarize(rows: list[dict[str, object]], group_keys: list[str], value_keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    output: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items()):
        base = {group_keys[i]: key[i] for i in range(len(group_keys))}
        base["count"] = len(group_rows)
        for value_key in value_keys:
            vals = [float(r[value_key]) for r in group_rows if r[value_key] != ""]
            base.update(prefixed_stats(value_key, vals))
        output.append(base)
    return output


def prefixed_stats(prefix: str, vals: list[float]) -> dict[str, float | int | str]:
    if not vals:
        return {f"{prefix}_mean": ""}
    mean = statistics.mean(vals)
    median = statistics.median(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    se = stdev / math.sqrt(len(vals)) if vals else 0.0
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": median,
        f"{prefix}_stdev": stdev,
        f"{prefix}_se": se,
        f"{prefix}_ci95_low": mean - 1.96 * se,
        f"{prefix}_ci95_high": mean + 1.96 * se,
        f"{prefix}_min": min(vals),
        f"{prefix}_max": max(vals),
    }


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def environment_info() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "os": os.name,
        "numpy": _version("numpy"),
        "Pillow": _version("PIL"),
        "reportlab": _version("reportlab"),
    }


def _version(module: str) -> str:
    try:
        mod = __import__(module)
        return str(getattr(mod, "__version__", "installed"))
    except Exception:
        return "not installed"


def _ratio(num: float, denom: float) -> float | str:
    return num / denom if abs(denom) > TOL else ""


def monte_carlo_common(instance: Instance, orders: dict[str, tuple[int, ...]], mixtures: dict[str, tuple[tuple[float, tuple[int, ...]], ...]], samples: int, seed: int) -> dict[str, dict[str, float]]:
    import random

    rng = random.Random(seed)
    rewards: dict[str, list[float]] = {name: [] for name in orders}
    rewards.update({name: [] for name in mixtures})
    mixture_cdfs = {}
    for name, mixture in mixtures.items():
        cdf = []
        total = 0.0
        for prob, order in mixture:
            total += prob
            cdf.append((total, order))
        mixture_cdfs[name] = cdf

    for _ in range(samples):
        realized_indices = []
        for item in instance.items:
            draw = rng.random()
            cumulative = 0.0
            chosen = len(item.outcomes) - 1
            for idx, outcome in enumerate(item.outcomes):
                cumulative += outcome.prob
                if draw <= cumulative + TOL:
                    chosen = idx
                    break
            realized_indices.append(chosen)
        realized_sizes = {i: instance.items[i].outcomes[idx].size for i, idx in enumerate(realized_indices)}
        for name, order in orders.items():
            rewards[name].append(_execute_realized(instance, order, realized_sizes))
        for name, cdf in mixture_cdfs.items():
            draw = rng.random()
            order = cdf[-1][1]
            for threshold, candidate in cdf:
                if draw <= threshold + TOL:
                    order = candidate
                    break
            rewards[name].append(_execute_realized(instance, order, realized_sizes))

    results = {}
    for name, vals in rewards.items():
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        se = sd / math.sqrt(len(vals))
        results[name] = {
            "mean": mean,
            "sd": sd,
            "se": se,
            "ci95_low": mean - 1.96 * se,
            "ci95_high": mean + 1.96 * se,
        }
    return results


def _execute_realized(instance: Instance, order: tuple[int, ...], realized_sizes: dict[int, tuple[float, ...]]) -> float:
    capacity = [1.0] * instance.dims
    reward = 0.0
    for idx in order:
        size = realized_sizes[idx]
        if fits(size, capacity):
            reward += instance.items[idx].reward
            for k in range(instance.dims):
                capacity[k] -= size[k]
        else:
            break
    return reward
