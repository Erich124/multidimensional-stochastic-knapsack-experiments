"""Simple non-DGV baseline orderings."""

from __future__ import annotations

import random

from .models import Instance
from .surrogate import compute_surrogate_stats, expected_total_size


def reward_density_order(instance: Instance) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(instance.items)),
            key=lambda i: (
                instance.items[i].reward / expected_total_size(instance.items[i])
                if expected_total_size(instance.items[i]) > 0
                else float("inf")
            ),
            reverse=True,
        )
    )


def surrogate_density_order(instance: Instance) -> tuple[int, ...]:
    stats = compute_surrogate_stats(instance)
    return tuple(sorted(range(len(instance.items)), key=lambda i: stats.items[i].density, reverse=True))


def random_order(instance: Instance, seed: int) -> tuple[int, ...]:
    order = list(range(len(instance.items)))
    random.Random(seed).shuffle(order)
    return tuple(order)

