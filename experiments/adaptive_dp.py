"""Exact adaptive optimum for small finite-support instances."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .models import Instance, capacity_tuple, fits, subtract_capacity


@dataclass(frozen=True)
class AdaptiveResult:
    value: float
    states_evaluated: int
    cache_hits: int
    first_action: int | None
    policy: dict[tuple[int, tuple[float, ...]], int] | None = None


def exact_adaptive_optimum(instance: Instance, recover_policy: bool = False) -> AdaptiveResult:
    items = instance.items
    policy: dict[tuple[int, tuple[float, ...]], int] = {}

    @lru_cache(maxsize=None)
    def value(mask: int, capacity: tuple[float, ...]) -> float:
        if mask == 0:
            return 0.0
        best = 0.0
        best_item: int | None = None
        for item_index, item in enumerate(items):
            if not (mask & (1 << item_index)):
                continue
            next_mask = mask ^ (1 << item_index)
            expected = 0.0
            for outcome in item.outcomes:
                if fits(outcome.size, capacity):
                    expected += outcome.prob * (
                        item.reward + value(next_mask, subtract_capacity(capacity, outcome.size))
                    )
            if expected > best + 1e-12:
                best = expected
                best_item = item_index
        if recover_policy and best_item is not None:
            policy[(mask, capacity)] = best_item
        return best

    full_mask = (1 << len(items)) - 1
    initial_capacity = capacity_tuple(instance.dims)
    optimum = value(full_mask, initial_capacity)
    info = value.cache_info()
    first_action = policy.get((full_mask, initial_capacity)) if recover_policy else None
    return AdaptiveResult(
        value=optimum,
        states_evaluated=info.misses,
        cache_hits=info.hits,
        first_action=first_action,
        policy=policy if recover_policy else None,
    )

