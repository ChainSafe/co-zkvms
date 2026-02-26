# BarnOwl PCG EdaBits: PCF-Based Expansion + Standard B2A

## Context

The current `edabits.rs` uses a nonstandard Cheng 2023 correlated tuple `{gamma, alphas}` where `alpha_1 + alpha_2 = gamma_bit`. This is **incompatible** with BarnOwl which generates **standard** daBits and edaBits.

**BarnOwl definitions** (paper Eq. 1-2):
- **daBit**: `([b]_p, [b]_2)` — same bit `b` shared in both arithmetic (Fp) and boolean (Z_2)
- **edaBit**: `([r]_p, [r_0]_2, ..., [r_{l-1}]_2)` — random l-bit value with per-bit boolean shares + arithmetic field share

**Extended edaBit** (what we need): standard edaBit PLUS per-bit arithmetic field shares `{[r_i]_p}`. These per-bit field shares enable the constant-round B2A formula.

**Standard B2A formula** (1 online round):
```
1. Open c = x XOR r_packed  (1 round via binary open)
2. [x]_p = sum_i 2^i * XOR_p(c_i, [r_i]_p)  (local)
   where XOR_p(c, [r]) = c + [r] - 2c*[r]
```

## Architecture: PCF-Based Expansion

### Why PCF (Boyle22 Section 5)

Three expansion approaches were considered:
1. **ChaCha12 RNG**: O(1) seeds for P0, but O(N*K) stored elements for P1/P2 (non-derivable components)
2. **Offline-online PCG (Boyle22 Section 4)**: O(N) stored vector Y_sigma for ALL parties
3. **PCF (Boyle22 Section 5)**: O(lambda) keys for ALL parties, on-demand evaluation

PCF wins on storage: all parties store only short keys (~65-370 kB), and each correlation is computed on-demand with ~660-3700 AES-NI PRG evaluations.

### How PCF Maps to Our Setting

The Boyle22 PCF for subfield VOLE (Figure 12) produces **pairwise** correlations: party 0 gets `(r, y_0)`, party 1 gets `(Delta, y_1)` where `y_0 = y_1 + r * Delta`. This is convertible to OT correlations.

BarnOwl uses 3 pairwise PCG instances to feed F_3OT, which feeds Pi_{sh:daBit}.

**Our adaptation for 3-party Rep3 semi-honest with trusted dealer P0:**

P0 runs PCF.Gen for 3 pairwise channels (P0<->P1, P1<->P2, P2<->P0), distributing keys during a one-time setup. Each pair then uses PCF.Eval to non-interactively expand OT correlations that feed a simplified (semi-honest) Pi_{sh:daBit}.

Alternatively (simpler for semi-honest): P0 generates all PCF key pairs for a single "virtual OT" channel per pair, and each party stores only their PCF key. During expansion, each party evaluates their key at index x to get their share of the correlation.

### PCF Construction (Boyle22 Figure 12)

The PCF has two components:

**1. RDCF (Relaxed Distributed Comparison Function)** — Figure 11:
- Setup(1^lambda, alpha, beta): outputs keys (K_0, K_1)
  - K_0 = (alpha, {k_bar_j}, {B_bar_j}, y)  — size O(m * lambda) where m = log(N/t)
  - K_1 = k (single lambda-bit seed)
- Eval(sigma, k_sigma, x): traverses GGM-like tree, outputs y_sigma in group G
- Correctness: Eval(0, K_0, x) - Eval(1, K_1, x) = f^beta_{<alpha}(x) = beta if x < alpha, else 0
- Per-evaluation cost: ~m/2 PRG calls on average

**2. PCF for subfield VOLE** — Figure 12:
- PCF.Gen(1^lambda):
  1. Sample Delta from F_q
  2. For i in [t]: sample alpha_i from {0,...,N/t-1}, beta_i from F_p \ {0}
  3. For i in [t]: (K^dcf_{i,0}, K^dcf_{i,1}) <- RDCF.Setup(1^lambda, alpha_i, beta_i * Delta)
  4. k_0 = (K^dcf_{i,0}, alpha_i, beta_i)_{i in [t]}, k_1 = (Delta, (K^dcf_{i,1})_{i in [t]})

- PCF.Eval(sigma, k_sigma, x):
  1. Compute row B_x of sparse matrix B (l non-zero entries at positions i_1,...,i_l)
  2. For each j in [l]: decompose i_j = gamma_j * (N/t) + delta_j
  3. y_sigma = sum_j x_j * RDCF.Eval(K^dcf_{gamma_j, sigma}, delta_j)
  4. If sigma = 0: also compute correction r from beta values
  5. Output correlation share

**Parameters** (Boyle22 Section 5.4):
- Conservative: t=664, l=7, key ~370kB, ~1300 PRG calls/output, ~660 PRG with eval-time optimized
- Aggressive: t=68, l=62, key ~65kB, ~3700 PRG calls/output
- PRG instantiation: 2 calls to fixed-key AES-128 (~1.3 cycles/byte with AES-NI)
- Throughput estimate: ~1.4 * 10^5 evaluations/sec per core (conservative)

### From VOLE to daBits

Each PCF.Eval at index x produces a VOLE correlation share. To get daBits in Rep3:

1. Run 3 pairwise PCF instances (one per pair) to get OT correlations
2. Feed into simplified semi-honest F_3OT (Figure 3/4)
3. Use F_3OT in Pi_{sh:daBit} (Figure 5) to produce Rep3-shared daBits

For **extended edaBits**, we generate K daBits per edaBit and also produce per-bit field shares.

## Implementation Plan

### Step 0: Implement RDCF Primitive

Create `mpc-core/src/protocols/rep3_ring/pcg/rdcf.rs`:

```rust
/// PRG: {0,1}^lambda -> G x {0,1}^{2*lambda}
/// Instantiated as 2 calls to fixed-key AES-128
struct Prg { fixed_key: aes::Aes128 }

/// RDCF key for party 0 (knows alpha)
struct RdcfKey0 {
    alpha: u32,                    // comparison point in [0, 2^m)
    k_bar: Vec<[u8; LAMBDA]>,     // m GGM node keys
    b_bar: Vec<GroupElement>,      // m correction values
    y: GroupElement,               // final correction
}

/// RDCF key for party 1 (doesn't know alpha)
struct RdcfKey1 {
    k: [u8; LAMBDA],  // single seed
}

/// RDCF.Setup(1^lambda, alpha in {0,1}^m, beta in G)
fn rdcf_setup(alpha: u32, beta: GroupElement, rng: &mut impl RngCore) -> (RdcfKey0, RdcfKey1)

/// RDCF.Eval(sigma, key, x) -> GroupElement
fn rdcf_eval_0(key: &RdcfKey0, x: u32) -> GroupElement
fn rdcf_eval_1(key: &RdcfKey1, x: u32) -> GroupElement
// Correctness: eval_0(K0, x) - eval_1(K1, x) = beta if x < alpha, else 0
```

Key implementation detail: The PRG G maps lambda bits to (group_element, 2*lambda bits). We use AES-128 in fixed-key mode (Matyas-Meyer-Oseas construction) for ~1.3 cycles/byte with AES-NI.

### Step 1: Implement PCF for Subfield VOLE

Create `mpc-core/src/protocols/rep3_ring/pcg/pcf_vole.rs`:

```rust
/// Sparse matrix B row computation (EA-code structure)
/// B = expander * accumulator, row has l non-zero entries
fn compute_row(x: usize, params: &PcfParams) -> Vec<(usize, Fp)>

/// PCF key for party 0 (VOLE sender)
struct PcfKey0 {
    rdcf_keys: Vec<RdcfKey0>,  // t RDCF keys
    alphas: Vec<u32>,           // t noise positions
    betas: Vec<Fp>,             // t payloads
}

/// PCF key for party 1 (VOLE receiver)
struct PcfKey1 {
    delta: Fq,                  // global correlation
    rdcf_keys: Vec<RdcfKey1>,  // t RDCF keys
}

/// PCF.Gen: trusted dealer generates both keys
fn pcf_gen(params: &PcfParams, rng: &mut impl RngCore) -> (PcfKey0, PcfKey1)

/// PCF.Eval: evaluate at index x to get VOLE share
fn pcf_eval_0(key: &PcfKey0, params: &PcfParams, x: usize) -> (Fp, Fq)  // (r, y_0)
fn pcf_eval_1(key: &PcfKey1, params: &PcfParams, x: usize) -> (Fq,)     // (y_1)
// Correctness: y_0 = y_1 + r * Delta
```

### Step 2: Implement OT-from-VOLE Conversion

VOLE correlation `(r, y_0)` and `(Delta, y_1)` where `y_0 = y_1 + r * Delta` can be converted to random OT:
- Sender gets `(w_0, w_1)` random strings
- Receiver gets `(u, v)` where u is random bit, v = w_u

This conversion uses a correlation-robust hash function (Boyle22 Section 2.7).

### Step 3: Implement Pi_{sh:daBit} with PCF-Expanded OTs

Create `mpc-core/src/protocols/rep3_ring/pcg/dabit_gen.rs`:

Following BarnOwl Figure 5, adapted for PCF-expanded OT correlations:

1. **Generate Random Sharings** (local, using F_cr^1 and F_cr^2 from pairwise RNG):
   - 3-of-3 shares of 0 and 1 in Z_{2^l}
   - Replicated boolean share of random bit [b]_2

2. **Oblivious Transfer** (using PCF-expanded OT correlations):
   - Each party uses PCF.Eval to expand the next OT correlation
   - Run the 3OT protocol (Figure 4) using these OT correlations
   - Result: 2-of-2 arithmetic shares of value b

3. **Reshare** (1 round): convert 2-of-2 to Rep3 replicated sharing

Output: `DaBit<F>` with `bit: Rep3RingShare<Bit>` and `value: Rep3PrimeFieldShare<F>`

### Step 4: Implement PcgEdaBit and LazyPcgEdaBits

Create `mpc-core/src/protocols/rep3_ring/edabits_pcg.rs`:

```rust
/// Standard edaBit with per-bit field shares (BarnOwl-compatible)
pub struct PcgEdaBit<T: IntRing2k, F: PrimeField> {
    pub r_bits: Vec<Rep3RingShare<Bit>>,          // {[r_i]_2} per-bit boolean
    pub r_packed: Rep3RingShare<T>,                // [r]_2 = pack(r_bits)
    pub bit_values: Vec<Rep3PrimeFieldShare<F>>,   // {[r_i]_p} per-bit field shares
}

/// Lazy PCF-based edaBit source
/// All parties store only O(lambda) PCF keys, evaluate on demand
pub struct LazyPcgEdaBits<T: IntRing2k, F: PrimeField> {
    /// PCF keys for 3 pairwise channels (this party's keys)
    pcf_keys: PcfKeySet,          // O(lambda) storage
    /// Pairwise RNG for F_cr^1, F_cr^2 (local random sharings)
    cr_rand: Rep3Rand,            // O(1) — just seeds
    /// Parameters
    params: PcfParams,
    total: usize,
    cursor: usize,
    party_id: PartyID,
    _phantom: PhantomData<(T, F)>,
}
```

The `take(n)` method:
1. For each of n edaBits (K bits each):
   - Evaluate PCF at cursor positions to get OT correlations
   - Run local Pi_{sh:daBit} computation to get daBit shares
   - The OT expansion + daBit generation is entirely local (no communication)
2. Pack bits into r_packed via `binary::pack_bits`
3. Advance cursor

**Key property**: After one-time PCF.Gen (setup), ALL expansion is non-interactive. Each party independently evaluates their PCF keys to reconstruct the same correlated randomness.

### Step 5: Implement B2A Conversion

Same `ring_to_field_b2a_many` as in previous plan — this part doesn't change:

```rust
pub fn ring_to_field_b2a_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    eda: Vec<PcgEdaBit<T, F>>,
    io: &mut IoContext<N>,
) -> IoResult<Vec<Rep3PrimeFieldShare<F>>>
```

Protocol (1 round):
```
1. Local: c_shares[i] = x_binary[i] ^ eda[i].r_packed
2. Round 1: c_values = binary::open_vec(&c_shares, io)
3. Local: result[j] = sum_i 2^i * XOR_p(c_values[j].bit(i), eda[j].bit_values[i])
```

### Step 6: Pool + Integration

`PcgEdaBitsPool<F>` — same interface as `EdaBitsPool<F>`:
```rust
pub struct PcgEdaBitsPool<F: PrimeField> {
    edabits_u8: LazyPcgEdaBits<u8, F>,
    edabits_u16: LazyPcgEdaBits<u16, F>,
    edabits_u32: LazyPcgEdaBits<u32, F>,
    edabits_u64: LazyPcgEdaBits<u64, F>,
    edabits_u128: LazyPcgEdaBits<u128, F>,
    dabits: Vec<DaBit<F>>,
}
```

Update integration points:
- `co-jolt2/src/utils/future_ring.rs`: switch to `PcgEdaBitsPool` and `edabits_pcg::ring_to_field_b2a_many`
- `co-jolt2/tests/dag_correct.rs`: switch pool creation
- `co-jolt2/src/zkvm/mod.rs`: switch pool creation and type
- `co-jolt2/src/zkvm/dag/worker.rs`: update pool type
- `co-jolt2/src/zkvm/instruction_lookups/`: update pool references
- `co-jolt2/src/zkvm/instruction/suffixes/mod.rs`: update B2A calls
- `co-jolt2/examples/rep3_jolt.rs`: update pool creation

### Step 7: Tests

1. `rdcf_correctness` — verify Eval(0,K0,x) - Eval(1,K1,x) = beta if x < alpha, else 0
2. `pcf_vole_correctness` — verify y0 = y1 + r * Delta for random evaluations
3. `pcg_dabits_consistent` — verify combine(bit_shares) == combine(value_shares)
4. `pcg_edabits_consistent` — verify r_packed == pack(r_bits) and bit_values reconstruct correctly
5. `pcg_b2a_single` — B2A with one edaBit
6. `pcg_b2a_many_u64` — batched B2A
7. Integration: `cargo test -p co-jolt2 --test dag_correct`

## File Structure

```
mpc-core/src/protocols/rep3_ring/
  pcg/
    mod.rs            -- module root, re-exports
    rdcf.rs           -- RDCF primitive (Figure 11)
    pcf_vole.rs       -- PCF for subfield VOLE (Figure 12)
    ot_convert.rs     -- VOLE->OT conversion
    dabit_gen.rs      -- Pi_{sh:daBit} with PCF OTs
    params.rs         -- PCF parameter sets (conservative/aggressive)
  edabits_pcg.rs      -- PcgEdaBit, LazyPcgEdaBits, PcgEdaBitsPool, B2A
```

## Critical Files

| File | Action |
|------|--------|
| `mpc-core/src/protocols/rep3_ring/pcg/rdcf.rs` | **Create** — RDCF primitive |
| `mpc-core/src/protocols/rep3_ring/pcg/pcf_vole.rs` | **Create** — PCF for VOLE |
| `mpc-core/src/protocols/rep3_ring/pcg/ot_convert.rs` | **Create** — VOLE->OT |
| `mpc-core/src/protocols/rep3_ring/pcg/dabit_gen.rs` | **Create** — daBit generation |
| `mpc-core/src/protocols/rep3_ring/pcg/params.rs` | **Create** — parameter sets |
| `mpc-core/src/protocols/rep3_ring/pcg/mod.rs` | **Create** — module root |
| `mpc-core/src/protocols/rep3_ring/edabits_pcg.rs` | **Create** — edaBit types + B2A + pool |
| `mpc-core/src/protocols/rep3_ring.rs` | **Edit** — add `pub mod pcg; pub mod edabits_pcg;` |
| `co-jolt2/src/utils/future_ring.rs` | **Edit** — switch pool + B2A |
| `co-jolt2/tests/dag_correct.rs` | **Edit** — switch pool creation |
| `co-jolt2/src/zkvm/mod.rs` | **Edit** — switch pool creation |
| `co-jolt2/src/zkvm/dag/worker.rs` | **Edit** — update pool type |
| `co-jolt2/src/zkvm/instruction_lookups/` | **Edit** — update pool refs |
| `co-jolt2/src/zkvm/instruction/suffixes/mod.rs` | **Edit** — update B2A calls |
| `co-jolt2/examples/rep3_jolt.rs` | **Edit** — update pool creation |

## Key Reusable Functions

| Function | File | Purpose |
|----------|------|---------|
| `DaBit<F>` | `edabits.rs:58` | Reuse existing type |
| `binary::pack_bits::<T>()` | `binary.rs:375` | Pack K bit shares into ring share |
| `binary::open_vec()` | `binary.rs:206` | Open binary shares (1 round) |
| `rep3_arith::promote_to_trivial_share()` | `arithmetic.rs:341` | Public value -> Rep3 share |
| `rep3_arith::sub_public_by_shared()` | `arithmetic.rs:69` | Compute `public - [share]` |
| `Rep3Rand::fork()` | `rngs.rs:134` | Derive independent sub-RNG |
| `Rep3Rand::snapshot()` | `rngs.rs:122` | Capture seekable RNG state |

## Dependencies

- `aes` crate — for fixed-key AES-128 PRG (AES-NI hardware acceleration)
- Existing: `rand`, `rand_chacha`, `rayon`, `ark-ff`

## Verification

1. Unit tests verify RDCF correctness and PCF VOLE correlation
2. Unit tests verify daBit/edaBit algebraic consistency
3. Unit tests verify B2A: `combine(result) == F::from(x_plaintext)`
4. Integration test `dag_correct` verifies full proving pipeline
5. Key invariant: 1-round B2A produces identical results to direct field sharing
