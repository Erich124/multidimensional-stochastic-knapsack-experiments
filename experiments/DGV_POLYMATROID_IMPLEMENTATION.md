# DGV Polymatroid Implementation

This implementation follows the DGV simplified greedy policy supported by the
polymatroid relaxation. It is the one-dimensional algorithm corresponding to
the manuscript guarantee `ALG_Z^poly >= Psi_Z(1)/2`.

For the surrogate instance:

```text
epsilon_i = E[min(Z_i, 1)]
w_i = c_i P(Z_i <= 1)
```

Items are sorted by nonincreasing `w_i / epsilon_i`. In this order,

```text
Psi_Z(1) = sum_k w_k prod_{i<k} (1 - epsilon_i).
```

The DGV simplified greedy policy is:

1. Compute `Psi_Z(1)` using the closed form above.
2. If some item has `w_i >= Psi_Z(1)/2`, execute only a best such single item.
3. Otherwise, execute all items in greedy density order.

The resulting deterministic order is lifted to the multidimensional instance by
executing that same order on the original size vectors. Successful items earn
their original rewards `c_i`, not the effective values `w_i`.

