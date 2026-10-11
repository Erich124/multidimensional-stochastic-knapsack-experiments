"""Surrogate statistics and one-dimensional relaxation values."""

from __future__ import annotations

from dataclasses import dataclass

from .config import TOL
from .models import Instance, Item


@dataclass(frozen=True)
class SurrogateItemStats:
    index: int
    name: str
    reward: float
    epsilon: float
    effective_value: float
    density: float


@dataclass(frozen=True)
class SurrogateStats:
    items: tuple[SurrogateItemStats, ...]

    @property
    def epsilon(self) -> tuple[float, ...]:
        return tuple(item.epsilon for item in self.items)

    @property
    def effective_value(self) -> tuple[float, ...]:
        return tuple(item.effective_value for item in self.items)


def z_value(size: tuple[float, ...]) -> float:
    return max(size)


def item_surrogate_stats(index: int, item: Item) -> SurrogateItemStats:
    epsilon = 0.0
    fit_prob = 0.0
    for outcome in item.outcomes:
        z = z_value(outcome.size)
        epsilon += outcome.prob * min(z, 1.0)
        if z <= 1.0 + TOL:
            fit_prob += outcome.prob
    effective = item.reward * fit_prob
    density = float("inf") if epsilon <= TOL and effective > TOL else (
        effective / epsilon if epsilon > TOL else 0.0
    )
    return SurrogateItemStats(index, item.name, item.reward, epsilon, effective, density)


def compute_surrogate_stats(instance: Instance) -> SurrogateStats:
    return SurrogateStats(
        tuple(item_surrogate_stats(index, item) for index, item in enumerate(instance.items))
    )


def density_order(stats: SurrogateStats) -> list[int]:
    return sorted(
        range(len(stats.items)),
        key=lambda i: (stats.items[i].density, stats.items[i].effective_value, -i),
        reverse=True,
    )


def phi_value(stats: SurrogateStats, budget: float) -> float:
    """Value of the simple fractional knapsack LP Phi(t)."""
    value = 0.0
    remaining = budget
    for i in density_order(stats):
        item = stats.items[i]
        if item.epsilon <= TOL:
            if item.effective_value > TOL:
                value += item.effective_value
            continue
        if remaining <= TOL:
            break
        take = min(1.0, remaining / item.epsilon)
        value += item.effective_value * take
        remaining -= item.epsilon * take
    return value


def expected_total_size(item: Item) -> float:
    return sum(outcome.prob * sum(outcome.size) for outcome in item.outcomes)

