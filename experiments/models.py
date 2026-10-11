"""Finite-support multidimensional stochastic knapsack model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import TOL


@dataclass(frozen=True)
class Outcome:
    prob: float
    size: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.prob < -TOL:
            raise ValueError("Outcome probabilities must be nonnegative")
        if any(x < -TOL for x in self.size):
            raise ValueError("Outcome sizes must be nonnegative")


@dataclass(frozen=True)
class Item:
    name: str
    reward: float
    outcomes: tuple[Outcome, ...]

    def __post_init__(self) -> None:
        if self.reward < -TOL:
            raise ValueError("Rewards must be nonnegative")
        if not self.outcomes:
            raise ValueError("Each item needs at least one outcome")
        dims = len(self.outcomes[0].size)
        if dims == 0:
            raise ValueError("Outcome size vectors must be nonempty")
        if any(len(outcome.size) != dims for outcome in self.outcomes):
            raise ValueError("All outcomes of an item must have the same dimension")
        total = sum(outcome.prob for outcome in self.outcomes)
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Outcome probabilities for {self.name} sum to {total}")


@dataclass(frozen=True)
class Instance:
    name: str
    dims: int
    items: tuple[Item, ...]
    seed: int | None = None
    family: str = ""

    def __post_init__(self) -> None:
        if self.dims <= 0:
            raise ValueError("dims must be positive")
        for item in self.items:
            if any(len(outcome.size) != self.dims for outcome in item.outcomes):
                raise ValueError(f"Item {item.name} has wrong dimension")


def normalized_outcomes(raw: Iterable[tuple[float, Sequence[float]]]) -> tuple[Outcome, ...]:
    rows = [(float(prob), tuple(float(x) for x in size)) for prob, size in raw]
    total = sum(prob for prob, _ in rows)
    if total <= TOL:
        raise ValueError("Outcome probabilities must have positive total")
    return tuple(Outcome(prob / total, size) for prob, size in rows)


def make_item(name: str, reward: float, raw: Iterable[tuple[float, Sequence[float]]]) -> Item:
    return Item(name=name, reward=float(reward), outcomes=normalized_outcomes(raw))


def capacity_tuple(dims: int) -> tuple[float, ...]:
    return tuple([1.0] * dims)


def fits(size: Sequence[float], capacity: Sequence[float]) -> bool:
    return all(size[k] <= capacity[k] + TOL for k in range(len(capacity)))


def subtract_capacity(capacity: Sequence[float], size: Sequence[float]) -> tuple[float, ...]:
    return tuple(round(capacity[k] - size[k], 10) for k in range(len(capacity)))

