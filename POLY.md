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

