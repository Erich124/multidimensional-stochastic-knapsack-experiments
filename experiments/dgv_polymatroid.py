"""DGV polymatroid-bound simplified greedy policy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .config import TOL
from .surrogate import SurrogateStats, density_order


@dataclass(frozen=True)
class PolymatroidPolicy:
    order: tuple[int, ...]
    psi_1: float
    mode: str
    greedy_order: tuple[int, ...]


def psi_one_closed_form(stats: SurrogateStats) -> float:
    value = 0.0
    survival = 1.0
    for item_index in density_order(stats):
        item = stats.items[item_index]
        value += item.effective_value * survival
        survival *= max(0.0, 1.0 - item.epsilon)
    return value


def build_polymatroid_policy(stats: SurrogateStats) -> PolymatroidPolicy:
    greedy = tuple(density_order(stats))
    psi_1 = psi_one_closed_form(stats)
    if not greedy or psi_1 <= TOL:
        return PolymatroidPolicy(tuple(), psi_1, "empty", greedy)
    best_single = max(greedy, key=lambda i: stats.items[i].effective_value)
    if stats.items[best_single].effective_value >= 0.5 * psi_1 - TOL:
        return PolymatroidPolicy((best_single,), psi_1, "single_item", greedy)
    return PolymatroidPolicy(greedy, psi_1, "greedy_order", greedy)


def psi_one_subset_lp_vertex_enumeration(stats: SurrogateStats) -> float:
    """Tiny-n LP check by enumerating active constraint vertices.

    This is intentionally for validation only.
    """
    n = len(stats.items)
    if n == 0:
        return 0.0
    constraints: list[tuple[tuple[float, ...], float]] = []
    eps = [item.epsilon for item in stats.items]
    for mask in range(1, 1 << n):
        row = tuple(eps[i] if mask & (1 << i) else 0.0 for i in range(n))
        rhs_prod = 1.0
        for i in range(n):
            if mask & (1 << i):
                rhs_prod *= 1.0 - eps[i]
        constraints.append((row, 1.0 - rhs_prod))
    for i in range(n):
        constraints.append((tuple(1.0 if j == i else 0.0 for j in range(n)), 1.0))
        constraints.append((tuple(-1.0 if j == i else 0.0 for j in range(n)), 0.0))

    best = 0.0
    for active in combinations(range(len(constraints)), n):
        matrix = [constraints[i][0] for i in active]
        rhs = [constraints[i][1] for i in active]
        point = _solve_linear_system(matrix, rhs)
        if point is None:
            continue
        if _feasible(point, constraints):
            value = sum(stats.items[i].effective_value * point[i] for i in range(n))
            best = max(best, value)
    return best


def _feasible(point: tuple[float, ...], constraints: list[tuple[tuple[float, ...], float]]) -> bool:
    for row, rhs in constraints:
        lhs = sum(row[i] * point[i] for i in range(len(point)))
        if lhs > rhs + 1e-8:
            return False
    return True


def _solve_linear_system(
    matrix: list[tuple[float, ...]], rhs: list[float]
) -> tuple[float, ...] | None:
    n = len(rhs)
    a = [[float(matrix[i][j]) for j in range(n)] + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return tuple(a[i][n] for i in range(n))

