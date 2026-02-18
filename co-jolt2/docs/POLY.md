## Rep3OneHotPolynomial (RandOHV): terse design summary

`Rep3OneHotPolynomial` is a proving-optimized MPC representation of the sparse one-hot MLE
`ra(k, j)` where each cycle `j` has either one active address `k(j)` or is all-zero.

Key idea: avoid “indexing by secret `k(j)`” by sampling one RandOHV mask `r` once and opening only
masked indices:
- Sample secret-shared `r` and its secret-shared one-hot vector `E = e(r)` (length `K`).
- For each active cycle, open `c[j] = open(k(j) XOR r)` once (stored as public `Option<u8>`).
- Inject `E` into prime-field shares once: `E_field: Vec<Rep3PrimeFieldShare<F>>`.

With `c[j]` public and `E_field` shared, any lookup of a public table at secret index is done as a
prime-field inner product:
- `table[r XOR c] = Σ_i table[i] * E_field[i XOR c]` (no further opens).

This supports:
- **Evaluate** (true MLE): `Σ_j eq(j, r_cycle) * eq(k(j), r_address)` via `c[j]` + inner products.
- **Opening-reduction sumcheck**: address rounds use a shared `G[k]`; after the last address bind,
  build shared dense `H(j) = eq(k(j), u)` from `F.clone_values()` via the same selection; cycle
  rounds dot public `D` pieces with shared `H`.
- **Row commitments** (`commit_rows`): return additive group-element shares `Vec<G>` (no
  `Rep3PointShare`); each party uses only the `.a` limb of `E_field` scalars and the coordinator
  reconstructs row commitments by summing group shares (PST13-style linearity).

## Rep3OneHotPolynomial: EqCycleState design note (RandOHV)

In cycle rounds of the opening-reduction sumcheck, the RandOHV prover computes `q0` as a
**secret-shared** value (it depends on the shared dense `H(j)`).

Vanilla Jolt calls `GruenSplitEqPolynomial::gruen_evals_deg_2(q0, previous_claim_norm)`, but that
API requires `q0: F` (public), so we cannot use it directly in Rep3.

Instead, Rep3 re-implements the same degree-2 Gruen algebra with `q0` as
`Rep3PrimeFieldShare<F>`. This requires access to the “current” `w[i]` and the current scalar
tracked by `GruenSplitEqPolynomial`. Since `GruenSplitEqPolynomial`’s `w/current_index` are
`pub(crate)` in `jolt-core`, `EqCycleState` stores `w: Vec<F::Challenge>` plus
`num_variables_bound` to recover the needed `w[i]` during cycle rounds.
