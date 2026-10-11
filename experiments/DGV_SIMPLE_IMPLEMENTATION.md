# DGV Simple-LP Implementation

This implementation follows DGV Section 4, the randomized greedy algorithm for
the `32/7` one-dimensional approximation. For the manuscript's surrogate
instance, the one-dimensional item size is `Z_i = max_k S_ik`, the mean
truncated size is `epsilon_i = E[min(Z_i, 1)]`, and the effective LP value is
`w_i = c_i P(Z_i <= 1)`.

The policy construction uses `w_i` and `epsilon_i`; execution still awards the
original reward `c_i`.

Implementation steps:

1. Sort items by nonincreasing density `w_i / epsilon_i`.
2. Compute the capacity-one fractional greedy solution for `Phi(1)`.
3. Let `r` be the critical item where cumulative `epsilon` reaches one. Items
   before `r` are full prefix items.
4. If the critical item is fractional, define
   `p' = (1 - sum_{i<r} epsilon_i) / epsilon_r`.
5. Define fractional effective values:
   `w'_i = w_i` for full prefix items and `w'_r = p' w_r`.
6. Choose the first index `k` with probability `w'_k / Phi(1)`.
7. If `k` is not the fractional critical item, execute item `k` first and then
   the remaining prefix items in greedy order.
8. If `k` is the fractional critical item, flip the DGV extra coin:
   with probability `p'`, execute `r` first and then the rest of the prefix;
   with probability `1-p'`, omit `r` and execute the preceding prefix items.
9. Items beyond the capacity-one fractional prefix are not part of this DGV
   randomized policy.

The code represents the randomized policy as a finite mixture of deterministic
orders. This allows exact evaluation for finite-support multidimensional
instances by summing each order's exact expected reward weighted by its mixture
probability.

