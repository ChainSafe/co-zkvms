# Dory in `co-jolt2` (design notes)

This document describes how the Dory polynomial commitment scheme is used in this repository, and how we adapt **commit** (and later **open/prove**) to Rep3 secret sharing.

## What Dory commits to

Dory treats the coefficient vector of a multilinear polynomial `f : {0,1}^n → F` as a matrix `A`:

- Fix a global **row length** `N_cols = 2^σ` (a power of two).
- Interpret the `2^n` coefficients as a length-`2^n` vector and lay them out row-major into a matrix with `N_cols` columns.
- If needed, **zero-pad** to a whole number of rows.

### Row commitments (practice)

Let `Γ₁[0..N_cols)` be public G1 generators from the URS.

For each matrix row `r`, Dory computes a “row commitment”

`T_r = ∑_{c=0..N_cols-1} A[r,c] · Γ₁[c]    ∈ G1`.

In code: this is just an MSM over the row’s scalars with the fixed column bases `Γ₁`.

These row commitments are *not* the final commitment object, but they are the key intermediate used by both committing and opening.

### Commitment output type

The (tier-2) Dory commitment in Jolt is a **public element of `GT`**:

`C = ∏_{r} e(T_r, Γ₂[r])    ∈ GT`,

where `Γ₂[r]` are public G2 generators from the URS and `e` is the pairing.

This is why the commitment type is `GT` (in arkworks for BN254: `Fq12` wrapped as a `PairingOutput`).

## Hiding vs binding

- As implemented in Jolt, Dory commitments are **not hiding by default**: they are deterministic functions of the polynomial and the URS generators.
- Dory supports **blinding** via extra generators `H₁ ∈ G1`, `H₂ ∈ G2` (they appear in the URS as `h1`, `h2`). A hiding variant adds random masks (Pedersen/AFGHO-style) into the commitment and into the opening proof.
- In our Rep3 commit path we currently use the same commitment formula as vanilla Jolt Dory; hiding behavior therefore matches vanilla.

## Setup + globals

Jolt’s Dory integration uses a global configuration `DoryGlobals`:

- `DoryGlobals::initialize(K, T)` fixes `NUM_COLUMNS` (i.e. `2^σ`) based on `K*T` and also fixes the maximum padded matrix size.
- Tests must not initialize these globals multiple times with conflicting parameters.

`DoryCommitmentScheme::setup_prover(max_log_n)` loads or generates a URS on disk:

- It uses `dory::setup_with_urs_file(..., Some("dory_urs_{max_log_n}_variables.urs"))`.
- On a cache miss it writes `dory_urs_{max_log_n}_variables.urs` in the current working directory.

### Sizing pitfall (`max_trace_length is too small`)

The `dory` crate’s URS generation for `max_log_n` creates `n = 2^ceil(max_log_n/2)` generators in each of `Γ₁` and `Γ₂` (roughly `sqrt(2^max_log_n)`).

Vanilla Jolt Dory commit requires `2^σ <= |Γ₁|`, so you must choose:

`max_log_n >= 2σ`.

This is why tests that call `setup_prover(...)` should use something like `setup_prover(2*sigma)` (not just `sigma`).

## Opening at a point (high level)

To open at a point `x ∈ F^n`, Dory builds vectors `(L, R)` derived from `x` and reduces a matrix-vector-matrix (VMV) relation via recursive folding rounds:

- Compute `L` / `R` from the evaluation point (note: Jolt vs Dory bit order differs; Jolt reverses the point for Dory).
- Use `T_r` row commitments and `L` to derive an `e1` witness-independent term.
- Exchange a sequence of reduction messages; each round yields Fiat–Shamir challenges (`β, α, γ, ...`) from a transcript and folds the state by a factor of 2.
- The final message is a scalar-product style relation involving one G1 element and one G2 element.

In this repo, the transcript used for challenges is `jolt_core::transcripts::Transcript` adapted into a Dory transcript interface via `JoltToDoryTranscriptRef`.

## Rep3 adaptation strategy (commit only, today)

We use the same linearity trick as PST13:

- The polynomial coefficients are Rep3-shared: each coefficient is a triple of field shares.
- Each party forms its **local committing view** by taking only the share component used for linear reconstruction (we call it `share.a` / “copy_share_a” semantics).
- Each party computes:
  - its additive share of each row commitment `T_r^{(i)} ∈ G1` by MSM over its local `share.a` scalars, and
  - its multiplicative share of `GT` commitment `C^{(i)} = ∏_r e(T_r^{(i)}, Γ₂[r])`.
- Reconstruction:
  - row commitments reconstruct additively in `G1`: `T_r = Σ_i T_r^{(i)}`,
  - commitments reconstruct multiplicatively in `GT`: `C = ∏_i C^{(i)}`.

This works because MSM is linear in scalars and the pairing is bilinear; combining shares matches committing to the reconstructed polynomial.

## Code map

- Vanilla Dory implementation (wrappers + scheme + globals):  
  `jolt_core::poly::commitment::dory` (vendored in this workspace under `examples/jolt/jolt-core/...`)
- `co-jolt2` Rep3 hooks + test live in:  
  `co-jolt2/src/poly/commitment/dory.rs`
  - re-exports vanilla types (`pub use jolt_core::poly::commitment::dory::*;`)
  - implements `Rep3CommitmentScheme` for `DoryCommitmentScheme`

