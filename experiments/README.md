# Round 2 Experiment Package

This package implements a finite-support computational framework for the
multidimensional stochastic knapsack manuscript. It separates:

- actual rewards `c_i`, awarded only when an item succeeds;
- effective relaxation values `w_i = c_i P(Z_i <= 1)`;
- surrogate masses `epsilon_i = E[min(Z_i, 1)]`;
- DGV simple-LP and polymatroid policy construction;
- exact fixed-order, exact mixture, Monte Carlo, and exact adaptive-DP evaluation.

The code is intentionally built for correctness checks and tiny pilots first.
It should not be used for the full computational study until the validation
suite passes and the Round 3 experiment design is confirmed.

