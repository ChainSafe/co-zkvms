# One-Hot Polynomial & RandOHV Masking

## Overview

In the MPC Jolt prover, each lookup cycle has a secret index `k(j) ∈ {0, ..., K-1}`
identifying which table entry is read. The prover commits to a one-hot polynomial
`p_j(x) = δ(x, k(j))` — evaluating to 1 at `x = k(j)` and 0 elsewhere.

Naively, building and committing this polynomial in MPC would require oblivious
scatter into a K-length vector, which is expensive. RandOHV (Random One-Hot Vector)
masking avoids this by opening a *masked* version of the index, then correcting
via a preprocessed one-hot mask vector.

## Current Approach: RandOHV with Rotation

### Core Idea

For each active cycle `j`:

1. **Preprocess** a secret random index `r` and its one-hot vector `E = e(r)` in
   `{0,1}^K`, so `E[i] = δ(i, r)`.
2. **Open** the masked index: `c[j] = open(k(j) ⊕ r)`. This is public.
3. **Reconstruct** the one-hot polynomial as: `p_j[i] = E[i ⊕ c[j]]`.
   Since `i ⊕ c[j] = i ⊕ k(j) ⊕ r`, this equals 1 when `i = k(j)` and
   the XOR-shift by `c[j]` is a public permutation applied to the secret vector `E`.

All cycles sharing the same mask `r` can be processed together: histogram
scatter uses the public `c[j]` values, and the secret XOR-shift is undone
in bulk via FWHT convolution with `Ê` (the Walsh-Hadamard transform of `E`).

### Rotation (Multiple Masks)

**Problem**: If all cycles share a single mask `r`, then for any two active cycles
`j₁, j₂`, the adversary sees `c[j₁] = k(j₁) ⊕ r` and `c[j₂] = k(j₂) ⊕ r`.
Computing `c[j₁] ⊕ c[j₂] = k(j₁) ⊕ k(j₂)` reveals the XOR relation between
secret indices. Equality (`k(j₁) = k(j₂)`) is also leaked as `c[j₁] = c[j₂]`.
Histogram frequency patterns over `c` values are identical to patterns over `k` values.

**Fix (current)**: Use `R` independent masks `r₀, ..., r_{R-1}`. Each active cycle
is assigned a slot `s(j) = active_ordinal(j) mod R`, and its masked index is
`c[j] = open(k(j) ⊕ r_{s(j)})`. Cross-slot XOR relations are meaningless (different
masks). Within a slot, the leakage persists but affects only `T/R` cycles per slot
(where T is the total active cycle count).

**Implementation** (see `one_hot_polynomial.rs:from_indices`):
- `rand_ohv_e_field_bank[slot]`: length-K vector of field-injected `E` shares for slot.
- `rotation_slot_by_cycle[j]`: `Some(s)` for active, `None` for padding.
- `masked_indices_c[j]`: public `c[j]` value for active cycles.
- Default `R = 16` (configurable via `RAND_OHV_ROTATIONS` env var).

### Performance Cost of Rotation

Each rotation slot introduces an independent mask that must be processed separately:

- **Commitment** (`commit_rows`): Per chunk, FWHT convolution is done once per slot.
  Cost: `O(R × K log K)` per chunk. Since K=16 (K_CHUNK), this is lightweight.
- **FWHT unmask** (`fwht_unmask_rep3_slots_to_additive`): Unmask each slot's
  histogram independently. Cost: `O(R × M log M)` where M = K^(chunks_per_phase).
  For 4-phase mode, M = 65536, so this is `O(R × 65536 × 16)` per suffix per phase.
- **Reshare** (`reshare_and_unmask_additive_hists_chunked`): Additive→Rep3 reshare
  of `R × M` field elements per histogram. This is the communication bottleneck.
- **Ehat16 tensor product** (`init_phase`): Builds `ehat16_by_slot[R][M]` via tree
  tensor product of per-chunk `e_field` vectors. Cost: `O(R × M)` interactive
  multiplications.

With M = 65536 (4-phase) and R = 16, the multiplicative overhead is 16× on all
M-proportional operations compared to R = 1.

## Security Issue: Residual Leakage Within a Slot

Even with R rotation slots, within each slot the adversary observes `T/R` masked
indices all using the same mask. The XOR of any two same-slot masked indices reveals
`k(j₁) ⊕ k(j₂)`. Equality of indices within a slot is directly visible.

For security-sensitive applications, this residual leakage may be unacceptable.
The following idea eliminates it entirely.

## Idea 3: Single-Party Reveal (t-Private Opening)

### Motivation

The core issue is that `c[j] = open(k(j) ⊕ r)` is revealed to ALL parties. Any
single corrupted party learns all masked indices and can compute cross-index XOR
relations. If instead we reveal `c[j]` to only ONE designated party, then in the
semi-honest honest-majority (2-of-3) model, the corrupted party either:
- IS the designated party → learns `c[j]` values (same as today, but only for their
  designated cycles), OR
- IS NOT the designated party → learns nothing about `c[j]`.

By rotating which party sees which cycles, the maximum leakage to any single
corrupted party is reduced to `T/3` cycles (not `T`).

### Protocol Specification

#### Setup

- Three parties P₀, P₁, P₂ in Rep3 secret sharing.
- Secret indices `k(j)` as Rep3RingShare<u8>.
- One preprocessed RandOHV mask `r` with one-hot vector `E = e(r)` (R = 1 suffices).

#### Phase 1: Private Opening

For each active cycle `j`, designate a "viewer" party `v(j) = j mod 3`.

To reveal `c[j] = k(j) ⊕ r` to party `v(j)` only:
- In Rep3, each party holds two of the three additive shares of `k(j) ⊕ r`.
- Party `v(j)` is missing one share. The party that holds the missing share sends
  it to `v(j)` (1 element of communication).
- The other two parties send nothing and learn nothing new.

Result: party `v(j)` knows plaintext `c[j]`. The other two parties know only their
existing Rep3 shares.

#### Phase 2: Local Histogram Scatter (at the Viewer)

Each party `Pᵥ` builds a histogram over cycles where `v(j) = v`:
```
H_local[v][i] = Σ_{j : v(j)=v} δ(i, c[j]) × coeff[j]
```
where `coeff[j]` is the appropriate coefficient (e.g., the Lagrange basis value
from the sumcheck). This histogram is computed locally by `Pᵥ` because `c[j]` is
known only to `Pᵥ` for those cycles.

The result `H_local[v]` is a **plaintext** vector known to party `Pᵥ` only. It must
be secret-shared (as a Rep3 share) before further processing.

#### Phase 3: Reshare Local Histograms

Party `Pᵥ` generates two random masks `m_a, m_b` (local randomness), computes
`m_c = H_local[v] - m_a - m_b`, then distributes:
- Send `(m_a, m_b)` to P₀ (or whichever party gets shares (a, b) in Rep3),
- Send `(m_b, m_c)` to P₁,
- Send `(m_c, m_a)` to P₂.

(Adapted to actual Rep3 share assignment.) Communication: 2M elements per viewer,
total 6M elements across all three viewers.

#### Phase 4: XOR-Unmask via FWHT

At this point, all parties hold Rep3 shares of `H_masked[i]` — the histogram indexed
by `c[j] = k(j) ⊕ r`, where `r` is a single shared mask (R = 1).

Apply standard FWHT unmask:
```
H_true = IFWHT( FWHT(H_masked) ⊙ Ê )
```
where `Ê` is the FWHT of the one-hot vector `e(r)`. Cost: `O(M log M)` — no R
multiplier.

### Communication Cost

| Step | Elements | Notes |
|------|----------|-------|
| Private open | T | 1 element per active cycle |
| Reshare histograms | 6M per histogram | 3 viewers × 2M shares each |
| FWHT unmask | 0 (local) | Rep3 × Rep3 → Additive is local |
| Reshare unmasked | 2M per histogram | Additive → Rep3 (if needed) |

Total per histogram: `O(T + M)` — compared to `O(R × M)` in the current approach.
For R = 16, this is a 16× reduction in the M-proportional communication, at the
cost of the `O(T)` private-open step.

### Security Analysis

**Semi-honest, honest-majority (t = 1 corruption out of n = 3)**:

- Each cycle's plaintext `c[j]` is revealed to exactly 1 party.
- A corrupted party `Pᵢ` sees `c[j]` for cycles where `v(j) = i` — approximately
  `T/3` cycles.
- For cycles where `v(j) ≠ i`, the corrupted party holds only Rep3 shares (no
  more information than standard MPC).
- Within the `T/3` visible cycles, the corrupted party can compute XOR relations
  and histogram frequencies — same leakage profile as current R = 1, but on 1/3
  of the data.

**Compared to current approach (R = 16)**:
- Current: ALL parties see ALL `c[j]` values. Within each slot (`T/16` cycles),
  full XOR and equality leakage. Cross-slot: no leakage.
- Single-party: ONE party sees `T/3` cycles with full leakage. Other parties: no
  leakage for those cycles.
- Quantitatively, R = 16 gives `T/16` vulnerable cycles per slot but to ALL parties;
  single-party gives `T/3` vulnerable cycles but to only ONE (unknown) party.

**Key advantage**: In semi-honest honest-majority, the adversary corrupts at most 1
party. With single-party reveal, there is a 1/3 chance the corrupted party is the
viewer for any given cycle. The expected leakage is `T/3` cycles, but the OTHER two
(honest) parties learn nothing about masked indices.

### Open Questions

1. **Shuffle integration**: The histogram scatter in Phase 2 requires the viewer to
   know which `c[j]` values correspond to which sumcheck coefficients. If coefficients
   are public (as in the current ReadRaf design), this is straightforward. If
   coefficients are secret, the viewer needs them revealed too (acceptable if the
   viewer already knows `c[j]`).

2. **Malicious security**: In the malicious model, the viewer could lie about their
   local histogram. Verifying correctness requires additional checks (e.g., MAC-based
   authentication of the histogram, or redundant computation by a second party).

3. **Compatibility with Dory commitment**: The Dory PCS commitment path also uses
   one-hot polynomials. The single-party reveal approach would need to be integrated
   into the commitment logic as well, not just the sumcheck/histogram path.

4. **Multiple polynomials**: In practice, there are D = 12 one-hot polynomials (one
   per dimension chunk). The viewer assignment could be shared across all D polynomials
   for the same cycle, keeping the private-open cost at `O(T)` not `O(D × T)`.
