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

## Rep3RaPolynomial (MPC RA helper)

`Rep3RaPolynomial<I, F>` is an MPC port of vanilla’s RA helper used during the last cycle-variable
rounds of opening reduction. It stores coefficients as `Rep3PrimeFieldShare<F>` and is designed to
avoid eager dense materialization when `T` is large.

**MPC assumptions (perf-driven):**
- `lookup_indices: Arc<Vec<Option<I>>>` are **opened** to the MPC parties (public to participants).
  In the one-hot use-case these are `masked_indices_c[j] = open(k(j) XOR r)`; this hides absolute
  `k(j)` under a secret mask `r`, but leaks `None` positions and XOR-relations across rows
  (because the same `r` is reused).
- If `lookup_indices` were secret, O(1) lookup would require an **oblivious access** primitive
  (MPC lookup / switching network / DPF-FSS / ORAM-style), which is out of scope.

**MPC-specific design choices:**
- `Rep3OneHotPolynomial.H` is stored as `Arc<RwLock<Rep3RaPolynomial<u8, F>>>` to mirror vanilla’s
  shared, mutable “round ladder” state without cloning large tables.
- One-hot optimization: compute a **secret-shared shifted EQ table**
  `F_shifted[c] = Σ_i E_field[i] * eq_u[i XOR c]` (depends on secret `E_field = e(r)`), so
  `get_bound_coeff(j)` is an O(1) lookup into `F_shifted` using the opened `c[j]`.
- Round ladder to reduce memory/allocation churn:
  - `Round1`: one table `F`
  - `Round2`: two scaled tables `F_0/F_1`
  - `Round3`: four scaled tables `F_00/F_01/F_10/F_11`
  - `RoundN`: materialize a dense `Rep3DensePolynomial<F>` once, then delegate binds to the dense
    implementation. Materialization allocates via `unsafe_allocate_zero_share_vec` to avoid
    initialization overhead and is filled in parallel (rayon).

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
