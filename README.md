# Multidimensional Stochastic Knapsack Experiments

This repository contains the code and data used for the computational experiments accompanying the paper **Approximation Bounds for Multidimensional Stochastic Knapsack via One-Dimensional Surrogates**.

The experiments compare two lifted Dean--Goemans--Vondrák (DGV) policies with two nonadaptive density heuristics on finite-support multidimensional stochastic-knapsack instances. For small instances, the adaptive optimum is computed exactly by Bellman dynamic programming.

## Repository structure

- `experiments/` — model, surrogate construction, DGV policies, exact evaluators, adaptive DP, instance generators, full-study runners, tables, and figures.
- `tests/` — validation tests for the surrogate, evaluators, DGV implementations, adaptive DP, and lifting property.
- `results/exact/` — raw and summarized exact-benchmark results.
- `results/scaling/` — scaling and Monte Carlo results.
- `results/stress_search/` — difficult-instance search outputs.
- `results/diagnostics/` — surrogate-geometry and low-polymatroid diagnostics.
- `results/tables/` and `results/figures/` — generated publication outputs.
- `ALGORITHM_NOTES.md` — precise implementation notes for the two DGV policies.
- `COMPUTATIONAL_PLAN.md` — experiment design and validation plan.

## Main experiment

The exact benchmark contains 3,600 finite-support instances with dimensions `d in {2,3,4,5}`. The expected reward of fixed nonadaptive policies is evaluated exactly, and the optimal adaptive value is computed by memoized Bellman dynamic programming.

The larger scaling suite contains 300 instances and uses 10,000 common Monte Carlo realizations per instance.

A separate search evaluates 5,000 additional small exact instances to identify difficult examples. These are computationally observed instances only and are not claimed to be worst cases.

## Requirements

The core exact experiments use the Python standard library. Figure generation additionally uses:

```bash
pip install -r requirements.txt
```

Python 3.12 was used for the reported experiments.

## Validation

Run the complete validation suite with:

```bash
python -m unittest discover -s tests -v
```

## Reproducing experiments

Run stages individually:

```bash
python -m experiments.run_exact
python -m experiments.run_scaling
python -m experiments.run_stress_search
python -m experiments.make_tables
python -m experiments.make_figures
```

Or run the full pipeline:

```bash
python -m experiments.run_all
```

The exact benchmark is substantially more expensive than the validation suite because it solves millions of Bellman states.

## Algorithm notes

The one-dimensional DGV policies are implemented from the algorithms analyzed in:

Brian C. Dean, Michel X. Goemans, and Jan Vondrák, *Approximating the Stochastic Knapsack Problem: The Benefit of Adaptivity*, Mathematics of Operations Research 33(4), 2008.

The DGV paper itself is **not redistributed in this repository**. See `experiments/DGV_SIMPLE_IMPLEMENTATION.md` and `experiments/DGV_POLYMATROID_IMPLEMENTATION.md` for implementation details.

## Data and interpretation

All reported ratios to `ADAPT_exact` in the exact benchmark use the exact adaptive Bellman solution for the generated finite-support instance. Large-instance results that use `BEST_TESTED` are empirical comparisons and should not be interpreted as ratios to the true optimum.

## License

No license is asserted by default. Add an open-source license before inviting third-party reuse if desired.
