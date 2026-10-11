"""Finite-support instance generators for validation, pilot, and full studies."""

from __future__ import annotations

import random

from .models import Instance, Item, make_item


def random_instance(dims: int, n_items: int, seed: int, outcomes_per_item: int = 3) -> Instance:
    rng = random.Random(seed)
    items: list[Item] = []
    for i in range(n_items):
        reward = round(rng.uniform(3.0, 12.0), 3)
        raw = []
        weights = [rng.uniform(0.2, 1.0) for _ in range(outcomes_per_item)]
        total = sum(weights)
        for weight in weights:
            mode = rng.choice(["light", "axis", "mixed", "overflow"])
            if mode == "light":
                size = tuple(rng.uniform(0.03, 0.22) for _ in range(dims))
            elif mode == "axis":
                axis = rng.randrange(dims)
                values = [rng.uniform(0.02, 0.12) for _ in range(dims)]
                values[axis] = rng.uniform(0.35, 0.85)
                size = tuple(values)
            elif mode == "overflow":
                axis = rng.randrange(dims)
                values = [rng.uniform(0.05, 0.30) for _ in range(dims)]
                values[axis] = rng.uniform(1.02, 1.25)
                size = tuple(values)
            else:
                size = tuple(rng.uniform(0.15, 0.45) for _ in range(dims))
            raw.append((weight / total, size))
        items.append(make_item(f"item_{i+1}", reward, raw))
    return Instance(name=f"random_d{dims}_n{n_items}_s{seed}", dims=dims, items=tuple(items), seed=seed, family="pilot_random")


def controlled_instance(
    regime: str,
    dims: int,
    n_items: int,
    seed: int,
    reward_variant: str = "independent",
    outcomes_per_item: int = 2,
) -> Instance:
    rng = random.Random(seed)
    items: list[Item] = []
    dominant_resources: list[int] = []
    for i in range(n_items):
        dom = i % dims
        dominant_resources.append(dom)
        raw_sizes = _regime_outcomes(regime, dims, dom, rng, outcomes_per_item)
        expected_sum = sum(prob * sum(size) for prob, size in raw_sizes)
        if reward_variant == "size_correlated":
            reward = round(1.0 + 8.0 * min(1.0, expected_sum / max(0.25, dims * 0.35)) + rng.uniform(0.0, 2.0), 3)
        else:
            reward = round(rng.uniform(1.0, 10.0), 3)
        items.append(make_item(f"item_{i+1}", reward, raw_sizes))
    instance = Instance(
        name=f"{regime}_{reward_variant}_d{dims}_n{n_items}_s{seed}",
        dims=dims,
        items=tuple(items),
        seed=seed,
        family=f"{regime}:{reward_variant}",
    )
    object.__setattr__(instance, "dominant_resources", tuple(dominant_resources))
    return instance


def explicit_overflow_instance(
    dims: int,
    n_items: int,
    seed: int,
    reward_variant: str = "independent",
) -> Instance:
    """Supplemental instances where every item can overflow its dominant coordinate."""
    rng = random.Random(seed)
    items: list[Item] = []
    dominant_resources: list[int] = []
    overflow_probs: list[float] = []
    for i in range(n_items):
        dom = i % dims
        dominant_resources.append(dom)
        p_over = rng.uniform(0.10, 0.30)
        overflow_probs.append(p_over)
        safe = tuple(rng.uniform(0.04, 0.24) for _ in range(dims))
        overflow_values = []
        for k in range(dims):
            if k == dom:
                overflow_values.append(rng.uniform(1.02, 1.30))
            else:
                overflow_values.append(rng.uniform(0.04, 0.24))
        raw_sizes = [(1.0 - p_over, safe), (p_over, tuple(overflow_values))]
        expected_sum = sum(prob * sum(size) for prob, size in raw_sizes)
        if reward_variant == "size_correlated":
            reward = round(1.0 + 8.0 * min(1.0, expected_sum / max(0.25, dims * 0.35)) + rng.uniform(0.0, 2.0), 3)
        else:
            reward = round(rng.uniform(1.0, 10.0), 3)
        items.append(make_item(f"item_{i+1}", reward, raw_sizes))
    instance = Instance(
        name=f"explicit_overflow_{reward_variant}_d{dims}_n{n_items}_s{seed}",
        dims=dims,
        items=tuple(items),
        seed=seed,
        family=f"explicit_overflow:{reward_variant}",
    )
    object.__setattr__(instance, "dominant_resources", tuple(dominant_resources))
    object.__setattr__(instance, "overflow_probs", tuple(overflow_probs))
    return instance


def _regime_outcomes(
    regime: str,
    dims: int,
    dominant: int,
    rng: random.Random,
    outcomes_per_item: int,
) -> list[tuple[float, tuple[float, ...]]]:
    if regime == "low_variance":
        base = [rng.uniform(0.05, 0.16) for _ in range(dims)]
        return _normalize_raw(
            [
                (0.45, tuple(x * rng.uniform(0.75, 0.95) for x in base)),
                (0.55, tuple(x * rng.uniform(1.05, 1.25) for x in base)),
            ][:outcomes_per_item]
        )
    if regime == "high_variance":
        small = tuple(rng.uniform(0.02, 0.12) for _ in range(dims))
        medium = tuple(rng.uniform(0.18, 0.42) for _ in range(dims))
        large_values = [rng.uniform(0.35, 0.85) for _ in range(dims)]
        if rng.random() < 0.45:
            large_values[rng.randrange(dims)] = rng.uniform(1.02, 1.35)
        return _normalize_raw([(0.45, small), (0.35, medium), (0.20, tuple(large_values))][:outcomes_per_item])
    if regime == "positive_correlation":
        direction = [rng.uniform(0.85, 1.15) for _ in range(dims)]
        low = tuple(rng.uniform(0.04, 0.11) * direction[k] for k in range(dims))
        high_scale = rng.uniform(0.35, 0.75)
        high = tuple(high_scale * direction[k] for k in range(dims))
        if rng.random() < 0.20:
            high = tuple(x * rng.uniform(1.25, 1.65) for x in high)
        mid = tuple(rng.uniform(0.16, 0.32) * direction[k] for k in range(dims))
        return _normalize_raw([(0.45, low), (0.35, mid), (0.20, high)][:outcomes_per_item])
    if regime == "resource_specialized":
        low = rng.uniform(0.02, 0.08)
        a = [rng.uniform(0.02, 0.08) for _ in range(dims)]
        b = [rng.uniform(0.02, 0.10) for _ in range(dims)]
        a[dominant] = rng.uniform(0.38, 0.78)
        b[dominant] = rng.uniform(0.55, 0.95)
        mixed = tuple(rng.uniform(low, 0.18) for _ in range(dims))
        return _normalize_raw([(0.48, tuple(a)), (0.34, tuple(b)), (0.18, mixed)][:outcomes_per_item])
    if regime == "near_capacity":
        axis = dominant
        a = [rng.uniform(0.04, 0.16) for _ in range(dims)]
        b = [rng.uniform(0.08, 0.24) for _ in range(dims)]
        c = [rng.uniform(0.12, 0.30) for _ in range(dims)]
        a[axis] = rng.uniform(0.44, 0.58)
        b[axis] = rng.uniform(0.62, 0.88)
        c[axis] = rng.uniform(0.92, 1.12)
        return _normalize_raw([(0.40, tuple(a)), (0.40, tuple(b)), (0.20, tuple(c))][:outcomes_per_item])
    if regime == "mixed_negative":
        axis_a = dominant
        axis_b = (dominant + 1) % dims
        a = [rng.uniform(0.02, 0.09) for _ in range(dims)]
        b = [rng.uniform(0.02, 0.09) for _ in range(dims)]
        a[axis_a] = rng.uniform(0.45, 0.90)
        b[axis_b] = rng.uniform(0.45, 0.90)
        c = tuple(rng.uniform(0.12, 0.28) for _ in range(dims))
        return _normalize_raw([(0.42, tuple(a)), (0.42, tuple(b)), (0.16, c)][:outcomes_per_item])
    raise ValueError(f"Unknown regime: {regime}")


def _normalize_raw(rows: list[tuple[float, tuple[float, ...]]]) -> list[tuple[float, tuple[float, ...]]]:
    total = sum(prob for prob, _ in rows)
    return [(prob / total, size) for prob, size in rows]


def tiny_adaptive_example() -> Instance:
    return Instance(
        name="tiny_adaptive_branching",
        dims=1,
        items=(
            make_item("probe", 1.0, [(0.5, (0.2,)), (0.5, (0.8,))]),
            make_item("small_followup", 2.0, [(1.0, (0.7,))]),
            make_item("large_followup", 3.0, [(1.0, (0.2,))]),
        ),
        family="validation",
    )
