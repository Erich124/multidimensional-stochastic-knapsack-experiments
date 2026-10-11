"""DGV simple-LP randomized greedy policy."""

from __future__ import annotations

from dataclasses import dataclass

from .config import TOL
from .evaluation import Mixture, Order
from .surrogate import SurrogateStats, density_order, phi_value


@dataclass(frozen=True)
class SimplePolicy:
    mixture: Mixture
    greedy_order: tuple[int, ...]
    prefix: tuple[int, ...]
    critical_item: int | None
    critical_fraction: float
    phi_1: float
    phi_2: float


def _aggregate_duplicate_orders(rows: list[tuple[float, Order]]) -> Mixture:
    totals: dict[Order, float] = {}
    for prob, order in rows:
        if prob <= TOL:
            continue
        totals[order] = totals.get(order, 0.0) + prob
    return tuple((prob, order) for order, prob in totals.items())


def build_simple_policy(stats: SurrogateStats) -> SimplePolicy:
    """Build the exact finite mixture induced by DGV Section 4.

    The implementation uses the unscaled version of the DGV randomization:
    probabilities proportional to the fractional Phi(1) effective values.
    """

    order = density_order(stats)
    phi_1 = phi_value(stats, 1.0)
    phi_2 = phi_value(stats, 2.0)
    if phi_1 <= TOL:
        return SimplePolicy(tuple(), tuple(order), tuple(), None, 0.0, phi_1, phi_2)

    prefix: list[int] = []
    cumulative_mass = 0.0
    critical_item: int | None = None
    critical_fraction = 1.0
    fractional_values: dict[int, float] = {}

    for item_index in order:
        item = stats.items[item_index]
        if item.epsilon <= TOL:
            if item.effective_value > TOL:
                prefix.append(item_index)
                fractional_values[item_index] = item.effective_value
            continue
        if cumulative_mass + item.epsilon <= 1.0 + TOL:
            prefix.append(item_index)
            fractional_values[item_index] = item.effective_value
            cumulative_mass += item.epsilon
            if cumulative_mass >= 1.0 - TOL:
                critical_item = item_index
                critical_fraction = 1.0
                break
        else:
            critical_item = item_index
            remaining = max(0.0, 1.0 - cumulative_mass)
            critical_fraction = remaining / item.epsilon
            prefix.append(item_index)
            fractional_values[item_index] = critical_fraction * item.effective_value
            break

    rows: list[tuple[float, Order]] = []
    prefix_tuple = tuple(prefix)
    for selected in prefix_tuple:
        selection_prob = fractional_values[selected] / phi_1
        if selected == critical_item and critical_fraction < 1.0 - TOL:
            inserted_order = (selected,) + tuple(i for i in prefix_tuple if i != selected)
            omitted_order = tuple(i for i in prefix_tuple if i != selected)
            rows.append((selection_prob * critical_fraction, inserted_order))
            rows.append((selection_prob * (1.0 - critical_fraction), omitted_order))
        else:
            rows.append((selection_prob, (selected,) + tuple(i for i in prefix_tuple if i != selected)))

    mixture = _aggregate_duplicate_orders(rows)
    total_prob = sum(prob for prob, _ in mixture)
    if abs(total_prob - 1.0) > 1e-8:
        raise ValueError(f"DGV simple mixture probability is {total_prob}, not 1")
    return SimplePolicy(
        mixture=mixture,
        greedy_order=tuple(order),
        prefix=prefix_tuple,
        critical_item=critical_item,
        critical_fraction=critical_fraction,
        phi_1=phi_1,
        phi_2=phi_2,
    )

