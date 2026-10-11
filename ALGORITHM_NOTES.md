# Algorithm Notes

## Scope and Sources Inspected

The initial workspace was empty at inspection time. The manuscript and DGV paper were available as local project files, and the manuscript source was also available in LaTeX form. A related legacy code directory contained the earlier experiment implementation used as a reference during the rebuild.

Human-authored files inspected in the related code folder:

- `stochastic_knapsack_experiments.py`
- `README.md`
- `EXPERIMENT_METHOD.md`
- `requirements.txt`
- `.gitignore`
- `results_summary.csv`
- `results.json`

Generated/binary/environment files observed but not treated as source code:

- `.DS_Store`
- `__pycache__/stochastic_knapsack_experiments.cpython-314.pyc`
- `.venv/`
- `.git/`

## Mathematical Model

The manuscript studies a normalized `d`-dimensional stochastic knapsack.

- Items: `N = {1,...,n}`.
- Reward: deterministic `c_i >= 0`.
- Size vector: `S_i = (S_{i1},...,S_{id}) in R_+^d`.
- Capacities: one in every coordinate.
- Independence: item vectors `S_1,...,S_n` are independent across items.
- Within-item dependence: coordinates of a single `S_i` may be arbitrarily correlated.
- Information: `S_i` is revealed only when item `i` is attempted.
- Success: item `i` succeeds if `S_{ik} <= R_k` for every coordinate `k`.
- Reward collection: on success, collect `c_i` and subtract the vector size from remaining capacities.
- Failure: if any coordinate violates remaining capacity, the item earns no reward and the process terminates.
- Adaptive policy: may choose the next item from prior revealed sizes and remaining items.
- Nonadaptive policy: fixes an item order in advance.

The surrogate one-dimensional size is

```text
Z_i = max_k S_{ik},       Ztilde_i = min{Z_i, 1},
epsilon_i = E[Ztilde_i],  w_i = c_i P(Z_i <= 1).
```

The key multidimensional pathwise bound is

```text
sum_{i attempted} min{Z_i, 1} <= d + 1.
```

The successful items contribute at most `d`, because `max_k S_{ik} <= sum_k S_{ik}` and each coordinate has total successful load at most one. The possible final failed item contributes at most one after truncation.

For any adaptive policy with attempt probabilities `x_i = P(i attempted)`, the manuscript obtains

```text
E[reward] <= sum_i w_i x_i,
sum_i epsilon_i x_i <= d + 1.
```

For any deterministic order, running the same order in the original multidimensional instance earns at least as much reward pathwise as running it in the one-dimensional surrogate instance with sizes `Z_i`, because `Z_i <= remaining surrogate capacity` implies `S_{ik} <= remaining coordinate capacity` for every `k`.

## DGV One-Dimensional Algorithms Used

The manuscript uses two one-dimensional nonadaptive policies from Dean, Goemans, and Vondrak (DGV).

### 1. Simple-LP randomized greedy algorithm

DGV define

```text
mu_i = E[min{s_i, 1}],
w_i = v_i P(s_i <= 1),
Phi(t) = max sum_i w_i x_i
         s.t. sum_i mu_i x_i <= t, 0 <= x_i <= 1.
```

For the surrogate instance, substitute `s_i = Z_i`, `v_i = c_i`, `mu_i = epsilon_i`, and `w_i = c_i P(Z_i <= 1)`.

The DGV simple algorithm is not merely "sort by density and run all items." Its structure is:

1. Sort items in nonincreasing density `w_i / mu_i`, handling zero `mu_i` items separately/free-first.
2. Let `r` be the critical index such that the cumulative `mu` first reaches one:

   ```text
   sum_{i < r} mu_i < 1 <= sum_{i <= r} mu_i.
   ```

3. Define the fractional critical mass

   ```text
   mu'_r = 1 - sum_{i < r} mu_i,
   p' = mu'_r / mu_r,
   w'_r = p' w_r,
   w'_i = w_i for i < r.
   ```

4. After scaling values so `Phi(1) = sum_{i=1}^r w'_i = 1`, choose an index `k` in `{1,...,r}` with probability `w'_k`.
5. If `k < r`, insert item `k` first.
6. If `k = r`, insert item `r` first only with probability `p'`; otherwise omit it.
7. Then insert the remaining items among `1,...,r` in greedy density order, excluding the selected first item.

DGV prove this randomized greedy policy obtains at least `(7/32) Phi(2)`, equivalently a `32/7` approximation to the one-dimensional adaptive optimum under their simple LP bound. They also state that the randomized policy can be derandomized by enumerating the deterministic nonadaptive policies obtainable from the random choices and choosing one with best certified/evaluated expected value.

For experiments, the exact DGV simple-LP policy should therefore be one of:

- a direct randomized implementation with independent algorithmic randomization included in the estimator; or
- a deterministic DGV derandomization over the candidate orders produced by the randomized policy.

The important implementation details are the density order, the prefix ending at the critical item for `Phi(1)`, the fractional critical item, and the possible omission of the critical item.

### 2. Polymatroid simplified greedy algorithm

DGV define the stronger relaxation

```text
Psi(t) = max sum_i w_i x_i
         s.t. sum_{i in J} mu_i x_i <= t (1 - prod_{i in J}(1 - mu_i))
              for all J subseteq N,
              0 <= x_i <= 1.
```

For `t = 1`, DGV give the closed form

```text
Psi(1) = sum_{k=1}^n w_k prod_{i < k} (1 - mu_i),
```

where items are sorted by nonincreasing `w_i / mu_i`.

The DGV nonadaptive policy is:

1. Compute `Psi(1)`.
2. If some item has `w_i >= Psi(1)/2`, insert only such an item.
3. Otherwise, insert all items in the greedy density order.

DGV prove this policy earns at least `Psi(1)/2`, and since the one-dimensional adaptive optimum is at most `2 Psi(1)`, it is a 4-approximation in one dimension.

For the manuscript's multidimensional lift, the one-dimensional reward guarantee `ALG_Z^poly >= Psi_Z(1)/2` combines with the manuscript's multidimensional subset domination lemma, yielding

```text
ALG_dD^poly >= (1 - exp(-(d+1))) / (2(d+1)) * ADAPT_dD.
```

## How the One-Dimensional Policies Are Lifted to d Dimensions

Both lifted algorithms use the same reduction:

1. For each item distribution over vectors `S_i`, compute the distribution/statistics of `Z_i = max_k S_{ik}`.
2. Compute `epsilon_i = E[min{Z_i,1}]`.
3. Compute `w_i = c_i P(Z_i <= 1)`.
4. Run the selected DGV one-dimensional policy using `(epsilon_i, w_i)` and the distribution of `Z_i` as needed.
5. Execute the resulting deterministic order, or randomized distribution over orders, in the original `d`-dimensional instance.

The lift is pathwise valid for each deterministic order. For a randomized DGV algorithm, condition on the algorithm's random choices and apply the same deterministic-order lifting pathwise.

## Existing Code That Can Be Reused

The sibling script `stochastic_knapsack_experiments.py` already implements several useful components correctly or nearly correctly:

- Data structures for finite discrete multidimensional item distributions: `Outcome`, `Item`, `Instance`.
- Surrogate statistic computation in `item_stats`: `epsilon_i`, `w_i`, density `w_i / epsilon_i`.
- Exact expected reward of a fixed nonadaptive order in the original multidimensional process: `exact_expected_reward`.
- Monte Carlo evaluator for fixed orders with common random samples: `MonteCarloEvaluator`.
- Exact adaptive optimum for small finite discrete instances via memoized Bellman recursion: `exact_adaptive_optimum`.
- Full subset-constraint polymatroid LP construction for small `n`: `polymatroid_constraints`.
- SciPy/HiGHS LP wrapper: `solve_lp_highs`.
- Fractional-knapsack solver for the simple LP value: `greedy_fractional_knapsack`.
- Synthetic finite-distribution instance generator covering several broad families.
- JSON/CSV output code for experiment summaries.

The polymatroid lifted policy is close to the DGV simplified greedy algorithm:

- It computes `Psi(1)` either by full subset LP or a closed-form fallback.
- It sorts by density.
- It selects a best single item if its effective value is at least `Psi(1)/2`; otherwise it uses the density order.

## What Still Needs To Be Implemented

Before running publishable experiments, I would implement or repair the following:

1. Faithful DGV simple-LP policy.

   The current `simple_lifted_algorithm` solves `Phi(2)` and then evaluates heuristic candidate orders based on density and prefix-first variants. This is not exactly the DGV randomized greedy/simple-LP algorithm. It does not explicitly compute the `Phi(1)` critical item, partial critical probability `p'`, randomized first-item distribution, or the critical-item omission case.

2. Explicit DGV simple derandomization.

   Implement candidate orders generated exactly by the DGV randomized policy:

   - `[k] + greedy_prefix_without_k` for each full prefix item `k < r`;
   - the appropriate critical-item-first order when `k = r` and the critical item is actually inserted;
   - the order omitting the critical item when the critical coin says not to insert it.

   Then evaluate these candidates exactly for small instances and by statistically clean simulation for larger instances.

3. Clear separation between algorithm selection and final evaluation.

   The existing code already uses independent samples for large-instance simple candidate selection and final evaluation. That should be formalized and applied consistently to any randomized/derandomized policy comparison.

4. Baselines.

   Existing code compares only the two lifted algorithms. Useful baselines are missing:

   - greedy by original expected total size or expected max size;
   - greedy by reward divided by `E[sum_k S_ik]`;
   - greedy by reward divided by coordinate bottleneck load;
   - deterministic mean-size multidimensional heuristic;
   - static order optimized by local search/swap on simulated reward;
   - single best item;
   - possibly an ordered-adaptive policy for a fixed order on small discrete instances.

5. Instance families designed around multidimensional structure.

   The current generator has a good starting set, but the next version should more directly control:

   - dimension `d`;
   - within-item coordinate correlation;
   - resource-specialized items;
   - sparse axis loads;
   - overflow probability;
   - reward-size correlation;
   - adversarial or difficult cases where `max_k S_ik` is conservative.

6. Exact adaptive optimum regime.

   The existing Bellman DP is suitable for small finite discrete instances, but publishable experiments need explicit limits and reporting:

   - max `n`;
   - number of outcomes per item;
   - number of states evaluated;
   - timeout/memory cap;
   - when exact optimum is omitted.

7. Test harness and correctness tests.

   Add unit tests for:

   - `epsilon_i` and `w_i` on hand-computed distributions;
   - exact fixed-order reward on small examples;
   - exact adaptive DP on tiny instances;
   - DGV simple candidate generation including critical item;
   - DGV polymatroid closed form against full subset LP for small `n`;
   - pathwise lift sanity checks.

## Ambiguities and Inconsistencies

1. Workspace mismatch.

   The requested folder was empty. The existing code was found in a sibling folder, not in the stated project folder. The new computational files are being created in the requested folder.

2. Manuscript says "run the DGV simple-LP nonadaptive algorithm to obtain an item order."

   DGV's simple-LP method is a randomized greedy policy with a critical item and possible omission. It does not directly specify one single deterministic order unless one applies the derandomization step. The manuscript should either describe the randomized policy or say that the DGV derandomized candidate-order version is used.

3. Current code's simple policy does not exactly match DGV.

   The code evaluates heuristic deterministic orders based on density order and prefix-first choices, using `Phi(2)` as an LP value. DGV's actual simple policy is based on the greedy prefix for `Phi(1)`, random first-item probabilities proportional to the prefix effective values, and a fractional critical item.

4. Current code may overstate "simple-LP algorithm" fidelity.

   Since the current simple policy selects the best candidate by exact reward/Monte Carlo in the multidimensional instance, it may perform well, but it is not exactly the policy analyzed by DGV and lifted in the manuscript.

5. Polymatroid implementation is closer but should be made explicit.

   The code's polymatroid policy matches DGV's simplified greedy structure, but the notes/results should say this is the DGV simplified greedy algorithm using the `Psi(1)` value, not a general LP-rounding algorithm.

6. Rewards in surrogate construction.

   The manuscript algorithm boxes say "construct the surrogate instance with sizes `Z_i`, rewards `c_i`." For computing the DGV ordering and LP values, the relevant effective value is `w_i = c_i P(Z_i <= 1)`. The actual execution reward remains `c_i`. This distinction should be explicit in the computational section.

7. DGV uses only limited distributional statistics for nonadaptive algorithms.

   The manuscript correctly notes that the lifted procedures need `epsilon_i` and `P(Z_i <= 1)`. Exact evaluation and exact adaptive optimum, however, require the full finite distribution of the original vector sizes.

8. General-distribution versus finite-distribution experiments.

   The theory allows arbitrary distributions, but exact adaptive optimum is only computationally feasible for finite discrete distributions. Continuous families should be discretized or evaluated by simulation, with exact optimum restricted to small discrete instances.

9. Dimension-one specialization.

   The manuscript's multidimensional polymatroid lift gives a weaker constant than DGV at `d = 1` because it uses the general `d+1` exponential domination argument instead of DGV's sharper one-dimensional subset lemma. This is not a contradiction, but it is worth noting if readers compare constants.

10. Full polymatroid LP scalability.

   The current full subset LP has `2^n - 1` constraints. This is fine for small `n` but not for larger sweeps. For `Psi(1)`, the DGV closed form should be the primary implementation.

## Recommended Next Implementation Round

The best next step is to build a clean experiment module in the requested workspace that reuses the good parts of the sibling script but first fixes algorithm fidelity:

1. Port the data structures, surrogate statistics, fixed-order evaluator, Monte Carlo evaluator, adaptive DP, and CSV/JSON writers.
2. Implement exact DGV simple randomized and derandomized policies.
3. Keep the DGV polymatroid simplified greedy policy, using the closed form for `Psi(1)`.
4. Add baselines and tests before running any large experiment.
5. Run only tiny smoke tests first, then design the full experiment suite once the implementation checks pass.
