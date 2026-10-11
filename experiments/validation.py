"""Validation helpers for implementation sanity checks."""

from __future__ import annotations

import random

from .config import TOL
from .dgv_polymatroid import build_polymatroid_policy
from .dgv_simple import build_simple_policy
from .evaluation import (
    exact_mixture_reward,
    exact_order_reward,
    execute_order_on_realization,
    execute_surrogate_order_on_realization,
    sample_realization,
)
from .models import Instance
from .surrogate import compute_surrogate_stats


def check_lifting(instance: Instance, order: tuple[int, ...], samples: int, seed: int) -> bool:
    rng = random.Random(seed)
    for _ in range(samples):
        realization = sample_realization(instance, rng)
        multidim = execute_order_on_realization(instance, order, realization)
        surrogate = execute_surrogate_order_on_realization(instance, order, realization)
        if multidim + TOL < surrogate:
            return False
    return True


def simple_guarantee_value(instance: Instance) -> tuple[float, float]:
    stats = compute_surrogate_stats(instance)
    policy = build_simple_policy(stats)
    alg = exact_mixture_reward(instance, policy.mixture)
    return alg, (7.0 / 32.0) * policy.phi_2


def poly_guarantee_value(instance: Instance) -> tuple[float, float]:
    stats = compute_surrogate_stats(instance)
    policy = build_polymatroid_policy(stats)
    alg = exact_order_reward(instance, policy.order)
    return alg, 0.5 * policy.psi_1

