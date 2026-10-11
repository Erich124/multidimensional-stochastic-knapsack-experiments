# Computational Plan

## Purpose

The computational section should evaluate the practical behavior of the manuscript's two lifted DGV algorithms without fabricating or pre-claiming results. The experiments should answer:

- How do the lifted simple-LP and polymatroid-LP policies perform relative to exact adaptive optimum on small finite instances?
- When does the maximum-coordinate surrogate `Z_i = max_k S_{ik}` lose value?
- How does performance change with dimension, coordinate correlation, resource specialization, and overflow probability?
- How expensive are the algorithms and the exact adaptive benchmark?
- Which generated instances are genuinely difficult?

## Algorithms To Compare

### Main algorithms

1. Lifted DGV simple-LP policy.

   Implement the actual DGV randomized greedy policy and/or its deterministic derandomization:

   - sort by `w_i / epsilon_i`;
   - find the `Phi(1)` critical prefix;
   - handle the fractional critical item;
   - evaluate the DGV candidate orders or include algorithmic randomization.

2. Lifted DGV polymatroid policy.

   Use DGV's simplified greedy policy:

   - compute `Psi(1) = sum_k w_k prod_{i<k}(1-epsilon_i)`;
   - if `max_i w_i >= Psi(1)/2`, run the best single item;
   - otherwise run all items by decreasing `w_i / epsilon_i`.

### Baselines

Include baselines that help interpret whether the surrogate is doing useful work:

- Best single item by exact expected reward.
- Greedy by `w_i / epsilon_i` over all items, without DGV simple randomization.
- Greedy by `c_i / E[min(sum_k S_{ik}, 1)]`.
- Greedy by `c_i / max_k E[min(S_{ik}, 1)]`.
- Greedy by `c_i / E[sum_k S_{ik}]` when finite.
- Greedy by expected reward in an empty knapsack, `c_i P(S_i <= 1 coordinatewise)`.
- Random order, averaged over many random permutations.
- Local-search order baseline, using adjacent swaps or insertion moves and a validation sample.
- Optional small-instance optimal nonadaptive order by exhaustive permutation only for very small `n`.
- Optional ordered-adaptive policy for a fixed order on small discretized cases.

## Benchmarks

### Exact adaptive optimum

For finite discrete instances, compute the exact adaptive optimum by Bellman recursion:

```text
V(R, C) = max_i E[ 1{S_i <= C} (c_i + V(R \ {i}, C - S_i)) ],
```

where `R` is the set of remaining items and `C` is the remaining capacity vector.

Use exact adaptive optimum only when computationally feasible. Report:

- `n`;
- `d`;
- outcomes per item;
- DP states evaluated;
- runtime;
- memory if available;
- timeout indicator.

Suggested exact regime:

- primary: `n <= 10`, outcomes per item `<= 3 or 4`, `d <= 5`;
- stretch: `n <= 12` only for low-outcome/easy state-space instances.

### Nonadaptive evaluation

For fixed orders:

- use exact enumeration when `n` and outcome count are small;
- use Monte Carlo with independent selection and evaluation samples otherwise;
- report mean, standard error, and 95% confidence intervals;
- use common random numbers across algorithms within the same instance for variance reduction;
- keep algorithm-selection samples separate from final evaluation samples.

## Instance Families

All instances should have capacity one in every coordinate and finite discrete item distributions. Continuous-looking families can be generated and then discretized into a few outcomes per item.

### 1. Dimension sweep

Purpose: measure the cost of dimension and the conservatism of `max_k S_{ik}`.

- `d in {1, 2, 3, 5, 8, 10, 20}`.
- Moderate `n`, such as `n in {8, 10, 20, 50}` depending on exact versus simulated evaluation.
- Hold expected total load roughly comparable across dimensions.

### 2. Coordinate correlation sweep

Purpose: isolate within-item dependence.

Families:

- positive correlation: all coordinates move together;
- independent coordinates;
- negative/anti-correlated coordinates: one coordinate large when another is small;
- one-spike coordinates: each outcome stresses one coordinate.

Metrics should include the ratio between `E[max_k S_{ik}]` and alternatives such as `max_k E[S_{ik}]` or `E[sum_k S_{ik}]`.

### 3. Resource-specialized instances

Purpose: test whether the max-coordinate surrogate punishes items that use different resources and could coexist well.

Construct item classes:

- class A mainly consumes coordinate 1;
- class B mainly consumes coordinate 2;
- through class `d`;
- mixed class consumes several coordinates moderately.

Vary reward premium for specialized versus mixed items.

### 4. Overflow-heavy instances

Purpose: test the role of `w_i = c_i P(Z_i <= 1)`.

Create items with:

- high reward and high probability of exceeding one coordinate even in an empty knapsack;
- moderate reward and low overflow probability;
- rare catastrophic large outcomes.

### 5. Small-items versus large-items

Purpose: connect to DGV behavior and identify cases where greedy ordering is strong or weak.

Vary `epsilon_i` distributions:

- all small, e.g. `epsilon_i <= 0.05`;
- mixed small and medium;
- many near-critical items with `epsilon_i` around `0.4` to `0.7`;
- deterministic sizes just above or below half capacity.

### 6. Reward-size correlation

Purpose: stress ordering decisions.

Cases:

- reward proportional to surrogate size;
- reward inversely proportional to surrogate size;
- noisy reward independent of size;
- high reward assigned to risky/overflow-prone items.

### 7. Difficult-instance search

Purpose: find and report hard examples honestly.

Use randomized instance generation plus filters/objectives such as:

- minimize lifted simple reward divided by exact adaptive optimum;
- minimize lifted polymatroid reward divided by exact adaptive optimum;
- maximize gap between exact adaptive optimum and best lifted algorithm;
- maximize difference between simple and polymatroid policies;
- maximize surrogate conservatism, e.g. adaptive optimum much larger than surrogate-based orders.

Use exact adaptive optimum for small instances during search. Keep a holdout set of difficult instances generated with different seeds to avoid overfitting conclusions.

## Experimental Matrix

Use two levels: exact small experiments and larger simulation experiments.

### Exact small suite

Suggested grid:

- `d in {2, 3, 5}`;
- `n in {6, 8, 10}`;
- outcomes per item in `{2, 3}`;
- 25 to 100 seeds per family, depending on DP cost;
- all main algorithms and baselines;
- exact adaptive optimum for every completed instance.

Report:

- `ALG / ADAPT*`;
- `ADAPT* / ALG`;
- rank of each algorithm;
- win counts;
- geometric mean ratios;
- 10th/50th/90th percentile ratios;
- worst observed ratios;
- DP states and runtime.

### Larger simulation suite

Suggested grid:

- `d in {2, 3, 5, 8, 10, 20}`;
- `n in {20, 50, 100, 250}`;
- 25 to 100 seeds per family;
- no exact adaptive optimum except for optional small subsets;
- fixed final evaluation sample size chosen by pilot variance, e.g. 100,000 or more when needed.

Report:

- expected reward with confidence intervals;
- relative performance against strongest baseline;
- algorithm runtime;
- LP/subroutine runtime;
- number of items attempted/succeeded;
- overflow coordinate statistics.

## Metrics

For each instance, record:

- `d`, `n`, family, seed;
- full item distributions for reproducibility;
- `epsilon_i`, `P(Z_i <= 1)`, `w_i`;
- `Phi(1)`, `Phi(2)`, `Psi(1)`;
- selected order/policy for each algorithm;
- expected reward;
- exact adaptive optimum when available;
- ratio to exact adaptive optimum;
- ratio to best baseline;
- theoretical lower bound values;
- runtime by component;
- DP states when exact adaptive optimum is computed;
- Monte Carlo standard error and confidence interval.

Aggregate by:

- dimension;
- family;
- coordinate-correlation level;
- resource-specialization level;
- item-size regime;
- reward-size correlation;
- exact versus simulated evaluation.

## Statistical Protocol

- Fix and publish random seeds.
- Store generated instances in JSON before evaluation.
- Use common random numbers across algorithms for final simulation.
- Use separate random streams for candidate-order selection and final evaluation.
- Report confidence intervals for simulation estimates.
- Avoid declaring small differences meaningful when confidence intervals overlap substantially.
- For difficult-instance search, separate search seeds from evaluation/reporting seeds.

## Implementation Plan

1. Create a clean experiment package in the requested workspace.
2. Port reusable components from the sibling script:

   - finite distribution classes;
   - surrogate statistic computation;
   - exact fixed-order evaluation;
   - Monte Carlo fixed-order evaluation;
   - exact adaptive DP;
   - JSON/CSV writers.

3. Implement exact DGV simple policy generation.
4. Implement DGV polymatroid policy using closed-form `Psi(1)`.
5. Add baselines.
6. Add unit tests and tiny deterministic examples.
7. Run smoke experiments only.
8. After validation, run exact small suite.
9. After reviewing small-suite behavior, run larger simulation suite.
10. Generate tables and plots for the manuscript.

## Proposed Tables and Figures

Tables:

- Algorithm definitions and implementation details.
- Exact-suite summary: median, mean, worst, and percentile ratios to `ADAPT*`.
- Runtime summary for exact DP and algorithms.
- Difficult-instance examples with full instance descriptors.

Figures:

- Ratio to `ADAPT*` by dimension for exact small suite.
- Reward relative to best baseline by dimension for larger suite.
- Performance versus coordinate-correlation parameter.
- Performance versus resource-specialization parameter.
- Runtime scaling in `n` and `d`.
- Scatter plot of surrogate conservatism versus algorithm ratio.

## Quality Gates Before Full Experiments

Do not run the full experiment suite until these checks pass:

- DGV simple candidate generation matches hand-computed critical-prefix examples.
- DGV polymatroid closed form matches full subset LP for small random instances.
- Exact fixed-order evaluator matches brute-force enumeration on tiny cases.
- Exact adaptive DP matches manually solved tiny instances.
- Monte Carlo evaluator converges to exact fixed-order values on small instances.
- The lifted multidimensional order earns at least the surrogate order pathwise in randomized spot checks.

## Immediate Next Request

The next implementation round should ask for:

```text
Implement a clean, tested experiment module in this workspace. Reuse the sibling script where appropriate, but first implement the exact DGV simple-LP randomized/derandomized policy and the DGV polymatroid simplified greedy policy. Add small correctness tests and only run smoke tests, not the full experiment suite.
```
