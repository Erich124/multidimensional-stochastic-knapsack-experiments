# Multidimensional Stochastic Knapsack Experiments

This repository contains the computational framework for the multidimensional
stochastic knapsack manuscript. It evaluates one-dimensional DGV policies after
lifting each multidimensional item through the surrogate
`Z_i = max_k S_ik`.

## Reproducibility

Run the unit and regression tests:

```bash
python -m unittest discover -s tests -v
```

Run the supplemental explicit-overflow suite:

```bash
python -m experiments.run_overflow
```

Regenerate publication tables from the committed raw data:

```bash
python -m experiments.make_tables
```

Run all stages:

```bash
python -m experiments.run_all
```

The experiment runners are idempotent: rerunning a runner rewrites only that
runner's owned outputs, so CSV rows are not duplicated.

## Benchmarks

The main exact benchmark is `results/exact/raw_exact.csv`. It contains 3,600
small exact instances from five regimes: `low_variance`, `high_variance`,
`positive_correlation`, `resource_specialized`, and `near_capacity`. Each item
has two size outcomes, and the benchmark pools independent and size-correlated
reward models.

The supplemental explicit-overflow benchmark is
`results/overflow/raw_overflow.csv`. It contains 240 instances with
`d in {2,3,4,5}`, `n=6`, two reward models, and 30 replications per
configuration. In this suite every item has positive probability that
`Z_i > 1`.

The main benchmark already tests cumulative capacity violations because an
accepted sequence may later fail when remaining capacity is insufficient. The
explicit-overflow suite additionally stresses the effective-value adjustment
`c_i P(Z_i <= 1)` and the truncation `min(Z_i, 1)` used in the one-dimensional
surrogate relaxation.

## Publication Tables

`python -m experiments.make_tables` writes the manuscript-facing tables to:

- `results/tables/publication_overall.csv`
- `results/tables/publication_overall.tex`
- `results/tables/publication_dimension_n6.csv`
- `results/tables/publication_dimension_n6.tex`

For compatibility, the same publication tables are also written as
`results/tables/table_exact.*` and `results/tables/table_dimension.*`.

The overall table uses all 3,600 exact instances and reports mean, median, and
minimum ratios for Simple LP, Polymatroid LP, surrogate-density greedy, and
multi-dimensional density greedy. The matched dimension table uses only `n=6`
instances, pooling all five original regimes and both reward models, yielding
exactly 300 observations for each dimension.
