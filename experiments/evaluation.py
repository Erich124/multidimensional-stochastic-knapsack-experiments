"""Exact and Monte Carlo evaluation of fixed nonadaptive policies."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Sequence

from .config import TOL
from .models import Instance, capacity_tuple, fits, subtract_capacity
from .surrogate import z_value

Order = tuple[int, ...]
Mixture = tuple[tuple[float, Order], ...]


def exact_order_reward(instance: Instance, order: Sequence[int]) -> float:
    """Exactly evaluate a deterministic order in the multidimensional model."""

    order_tuple = tuple(order)

    @lru_cache(maxsize=None)
    def rec(pos: int, capacity: tuple[float, ...]) -> float:
        if pos >= len(order_tuple):
            return 0.0
        item = instance.items[order_tuple[pos]]
        total = 0.0
        for outcome in item.outcomes:
            if fits(outcome.size, capacity):
                total += outcome.prob * (
                    item.reward + rec(pos + 1, subtract_capacity(capacity, outcome.size))
                )
        return total

    return rec(0, capacity_tuple(instance.dims))


def exact_mixture_reward(instance: Instance, mixture: Mixture) -> float:
    return sum(prob * exact_order_reward(instance, order) for prob, order in mixture)


def execute_order_on_realization(
    instance: Instance,
    order: Sequence[int],
    realized_sizes: dict[int, tuple[float, ...]],
) -> float:
    capacity = list(capacity_tuple(instance.dims))
    reward = 0.0
    for item_index in order:
        size = realized_sizes[item_index]
        if fits(size, capacity):
            reward += instance.items[item_index].reward
            for k, amount in enumerate(size):
                capacity[k] -= amount
        else:
            break
    return reward


def execute_surrogate_order_on_realization(
    instance: Instance,
    order: Sequence[int],
    realized_sizes: dict[int, tuple[float, ...]],
) -> float:
    remaining = 1.0
    reward = 0.0
    for item_index in order:
        z = z_value(realized_sizes[item_index])
        if z <= remaining + TOL:
            reward += instance.items[item_index].reward
            remaining -= z
        else:
            break
    return reward


def sample_realization(instance: Instance, rng: random.Random) -> dict[int, tuple[float, ...]]:
    realized: dict[int, tuple[float, ...]] = {}
    for index, item in enumerate(instance.items):
        draw = rng.random()
        cumulative = 0.0
        chosen = item.outcomes[-1]
        for outcome in item.outcomes:
            cumulative += outcome.prob
            if draw <= cumulative + TOL:
                chosen = outcome
                break
        realized[index] = chosen.size
    return realized


def monte_carlo_order_reward(
    instance: Instance,
    order: Sequence[int],
    samples: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    total = 0.0
    for _ in range(samples):
        total += execute_order_on_realization(instance, order, sample_realization(instance, rng))
    return total / samples


def monte_carlo_mixture_reward(
    instance: Instance,
    mixture: Mixture,
    samples: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    cumulative: list[tuple[float, Order]] = []
    running = 0.0
    for prob, order in mixture:
        running += prob
        cumulative.append((running, order))
    total = 0.0
    for _ in range(samples):
        draw = rng.random()
        order = cumulative[-1][1]
        for threshold, candidate in cumulative:
            if draw <= threshold + TOL:
                order = candidate
                break
        total += execute_order_on_realization(instance, order, sample_realization(instance, rng))
    return total / samples

