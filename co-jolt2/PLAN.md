# Witness Generation Plan (Instructions Scope)

## Context

Phase 1 (foundation types) is complete: `Rep3Operand`, `Rep3RegisterState`, `Rep3InstructionFormat`, `Rep3RISCVCycle`, and trace conversion utilities all compile clean.

This plan adds witness generation infrastructure, scoped to instruction-related polynomials. We mirror vanilla's `witness.rs` and `Jolt` trait with Rep3 type analogs.

**Vanilla references:**
- `../../examples/jolt/jolt-core/src/zkvm/mod.rs` — `Jolt` trait
- `../../examples/jolt/jolt-core/src/zkvm/witness.rs` — `WitnessData`, `CommittedPolynomial`, `generate_witness_batch`
- `../../examples/jolt/jolt-core/src/zkvm/instruction/mod.rs` — `LookupQuery`, `CircuitFlags`, `InstructionFlags`

---

## 1. Rep3Jolt Trait

**File:** `src/zkvm/mod.rs`
**Mirrors:** vanilla `Jolt<F, PCS, FS>` trait in `jolt-core/src/zkvm/mod.rs:221`

Add `Rep3Jolt` trait with `todo!()` method bodies. The trait mirrors vanilla `Jolt` but adds MPC network context:

```rust
use mpc_core::protocols::rep3::network::Rep3Network;

pub trait Rep3Jolt<F: JoltField, PCS, FS: Transcript, N: Rep3Network>
where
    PCS: CommitmentScheme<Field = F>,
{
    fn prove(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[Cycle],
        io_ctx: &mut IoContext<N>,
    ) -> ... {
        todo!()
    }
}
```

**Rationale:** In MPC, tracing is done by vanilla (public). Rep3Jolt receives the already-traced `Vec<Cycle>` plus MPC network context. Preprocessing (`shared_preprocess`, `prover_preprocess`) is the same as vanilla (public data), so we reuse those directly without re-declaring. `verify` is also unchanged (verifier runs single-party).

**Methods (all `todo!()`):**
- `prove(preprocessing, trace, io_ctx)` — the only method that differs from vanilla

---

## 2. Rep3WitnessData

**File:** `src/zkvm/witness.rs`
**Mirrors:** vanilla `WitnessData` in `jolt-core/src/zkvm/witness.rs:84`

### Type Mapping

| Vanilla field | Vanilla type | Rep3 type | Notes |
|---|---|---|---|
| `left_instruction_input` | `Vec<u64>` | `Vec<Rep3RingShare<u64>>` | Always unsigned register value |
| `right_instruction_input` | `Vec<i128>` | `Vec<Rep3RingShare<u64>>` | See i128 memo below |
| `write_lookup_output_to_rd` | `Vec<u8>` | `Vec<Rep3RingShare<u8>>` | Flag × flag product |
| `write_pc_to_rd` | `Vec<u8>` | `Vec<Rep3RingShare<u8>>` | Flag × flag product |
| `should_branch` | `Vec<u8>` | `Vec<Rep3RingShare<u8>>` | output × flag product |
| `should_jump` | `Vec<u8>` | `Vec<Rep3RingShare<u8>>` | Flag × flag product |
| `rd_inc` | `Vec<i128>` | see i128 memo | post_rd − pre_rd |
| `ram_inc` | `Vec<i128>` | see i128 memo | Out of instruction scope |
| `instruction_ra` | `[Vec<Option<u8>>; D]` | `[Vec<Option<Rep3RingShare<u8>>>; D]` | One-hot indices |
| `bytecode_ra` | `Vec<Vec<Option<u8>>>` | — | Out of instruction scope |
| `ram_ra` | `Vec<Vec<Option<u8>>>` | — | Out of instruction scope |

### Struct Definition

```rust
pub struct Rep3WitnessData {
    // Instruction-scope polynomials
    pub left_instruction_input: Vec<Rep3RingShare<u64>>,
    pub right_instruction_input: Vec<Rep3RingShare<u64>>,  // mixed: shared or public
    pub write_lookup_output_to_rd: Vec<Rep3RingShare<u8>>,
    pub write_pc_to_rd: Vec<Rep3RingShare<u8>>,
    pub should_branch: Vec<Rep3RingShare<u8>>,
    pub should_jump: Vec<Rep3RingShare<u8>>,

    // Stored as separate pre/post to avoid signed arithmetic (see i128 memo)
    pub rd_pre: Vec<Rep3RingShare<u64>>,
    pub rd_post: Vec<Rep3RingShare<u64>>,

    pub instruction_ra: [Vec<Option<Rep3RingShare<u8>>>; D],

    // Out of scope for this phase (placeholders)
    // pub ram_inc: ...,
    // pub bytecode_ra: ...,
    // pub ram_ra: ...,
}
```

### Constructor and Stub Methods

```rust
impl Rep3WitnessData {
    pub fn new(trace_len: usize) -> Self { ... }  // allocate vecs
}
```

No `generate_witness_batch` implementation yet — just the data structure. Generation logic is a future task that will populate these fields from `Rep3RISCVCycle` trace.

### Module Declarations

Update `src/zkvm/mod.rs` to add:
```rust
pub mod witness;
```

---

## 3. Research Memo: i128 in Vanilla Witness and MPC Handling

### Why vanilla uses i128

Three fields in `WitnessData` use `i128`:

**a) `right_instruction_input: Vec<i128>`**
- Source: `LookupQuery::to_instruction_inputs(cycle).1`
- For R-format (ADD, SUB, etc.): returns `register_state.rs2 as i128` — always non-negative (u64 cast to i128)
- For I-format (ADDI, etc.): returns `instruction.operands.imm as i128` — the immediate field is `i64`, can be negative (e.g., `addi x1, x1, -4` → `-4i128`)
- **Key**: The signed-ness comes from immediate values in I-format instructions. The `imm` field is `i64` because RISC-V immediates are sign-extended.
- **Solution**: we make it `Vec<Rep3RingShare<u64>>` for now.

**b) `rd_inc: Vec<i128>`**
- Source: `post_rd as i128 - pre_rd as i128`
- Both `pre_rd` and `post_rd` are `u64` register values
- Difference can be negative (e.g., register decreases from 100 to 50 → `rd_inc = -50`)
- Range: `[-(2^64-1), 2^64-1]` (fits in i65, stored as i128)

**c) `ram_inc: Vec<i128>`**
- Source: `post_value as i128 - pre_value as i128`
- Same pattern as `rd_inc` but for RAM values
- Out of instruction scope for this phase
---

## 4. Rep3Cycle Enum

**File:** `src/zkvm/instruction/mod.rs`
**Mirrors:** vanilla `Cycle` enum in `tracer/src/instruction/mod.rs:394` (generated by `define_rv32im_enums!` macro)

### Context

Vanilla `Cycle` is an enum over all instruction types:
```rust
pub enum Cycle {
    NoOp,
    ADD(RISCVCycle<ADD>),
    ADDI(RISCVCycle<ADDI>),
    // ... ~60 variants
}
```

We need a `Rep3Cycle` enum with `Rep3RISCVCycle<T>` instead of `RISCVCycle<T>`, so that `generate_witness_batch_rep3` can access shared register values.

### Design

```rust
pub enum Rep3Cycle {
    NoOp,
    ADD(Rep3RISCVCycle<ADD>),
    ADDI(Rep3RISCVCycle<ADDI>),
    // ... all instruction variants
}
```

Generated by a macro (mirroring `define_rv32im_enums!`), including the `INLINE` variant.

### Methods on Rep3Cycle

Mirror each vanilla `Cycle` method with Rep3 return types where values are shared:

| Vanilla Method | Vanilla Return | Rep3 Method | Rep3 Return | Notes |
|---|---|---|---|---|
| `instruction(&self)` | `Instruction` | `instruction(&self)` | `Instruction` | Same — instruction encoding is public |
| `circuit_flags()` | `[bool; 18]` | via `instruction()` | `[bool; 18]` | Public — derived from opcode |
| `rs1_read(&self)` | `(u8, u64)` | `rs1_read(&self)` | `(u8, Rep3Operand)` | register index is public, value is shared |
| `rs2_read(&self)` | `(u8, u64)` | `rs2_read(&self)` | `(u8, Rep3Operand)` | register index is public, value is shared |
| `rd_write(&self)` | `(u8, u64, u64)` | `rd_write(&self)` | `(u8, Rep3Operand, Rep3Operand)` | index public, pre/post values shared |
| `ram_access(&self)` | `RAMAccess` | — | — | Out of scope |

### Rep3LookupQuery Trait

**File:** `src/zkvm/instruction/mod.rs`
**Mirrors:** vanilla `LookupQuery<XLEN>` trait in `jolt-core/src/zkvm/instruction/mod.rs:17`

Vanilla `LookupQuery` operates on `RISCVCycle<T>` / `Cycle` and returns plaintext values. We need a Rep3 version that returns shared values.

```rust
pub trait Rep3LookupQuery<const XLEN: usize> {
    fn to_instruction_inputs_rep3(&self) -> (Rep3Operand, Rep3Operand);
    fn to_lookup_index_rep3(&self) -> Rep3RingShare<u128>;  // deferred
    fn to_lookup_output_rep3(&self) -> Rep3Operand;         // deferred
}
```

**Impls needed:**
- `impl Rep3LookupQuery<XLEN> for Rep3Cycle` — dispatches to per-instruction impls (macro-generated)
- `impl Rep3LookupQuery<XLEN> for Rep3RISCVCycle<T>` — per instruction type

For `to_instruction_inputs_rep3`:
- R-type: returns `(rs1_operand, rs2_operand)` — both from `Rep3RegisterState`
- I-type: returns `(rs1_operand, Rep3Operand::Public(imm))` — imm from instruction encoding
- B-type: returns `(rs1_operand, rs2_operand)`
- J-type: returns `(Rep3Operand::Public(address), Rep3Operand::Public(imm))`
- U-type: returns `(Rep3Operand::Public(0), Rep3Operand::Public(imm))`
- VirtualAdvice: returns `(Rep3Operand::Public(0), Rep3Operand::Public(0))`

`to_lookup_index_rep3` and `to_lookup_output_rep3`: stub with `todo!()` for now.

### Conversion: `Cycle` → `Rep3Cycle`

```rust
impl Rep3Cycle {
    pub fn from_public_cycle(cycle: &Cycle) -> Self { ... }
    pub fn promote_to_shares(&mut self, party_id: PartyID) { ... }
}

pub fn convert_trace_to_rep3(trace: &[Cycle]) -> Vec<Rep3Cycle> { ... }
```

These dispatch to `Rep3RISCVCycle::from_public_cycle` per variant. Note: this requires `Rep3InstructionFormat` impls for all instruction format types (currently only `FormatR` is implemented — need to add `FormatI`, `FormatB`, `FormatJ`, `FormatU`, etc.).

### Missing Format Impls

Before `Rep3Cycle` can cover all instructions, we need `Rep3InstructionFormat` + `Rep3RegisterState` for:

| Format | Instructions | Status |
|---|---|---|
| `FormatR` | ADD, SUB, AND, OR, XOR, SLT, SLTU, MUL, MULHU, ANDN | Done |
| `FormatI` | ADDI, ANDI, ORI, XORI, SLTI, SLTIU, JALR, LD | TODO |
| `FormatB` | BEQ, BGE, BGEU, BLT, BLTU, BNE | TODO |
| `FormatJ` | JAL | TODO |
| `FormatU` | LUI, AUIPC | TODO |
| `FormatS` | SD | TODO |
| `FormatInline` | INLINE | TODO |
| Others (virtual) | VirtualAdvice, VirtualAssert*, VirtualMove, etc. | TODO |

Each format impl follows the same pattern as `format_r.rs`: define `Rep3RegisterStateFormatX`, impl `Rep3RegisterState`, impl `Rep3InstructionFormat for FormatX`.

---

## 5. Rep3WitnessData and `generate_witness_batch_rep3`

**File:** `src/zkvm/witness.rs`
**Mirrors:** vanilla `WitnessData` + `CommittedPolynomial::generate_witness_batch` in `jolt-core/src/zkvm/witness.rs`

### Vanilla witness.rs Inventory

| Vanilla Type/Item | Rep3 Counterpart | Action |
|---|---|---|
| `DTH_ROOT_OF_K`, `compute_d_parameter` | Reuse directly | Same (public constants) |
| `CommittedPolynomial` enum | Reuse directly | Same (polynomial labels, public) |
| `ALL_COMMITTED_POLYNOMIALS` static | Reuse directly | Same (public registry) |
| `AllCommittedPolynomials` struct + methods | Reuse directly | Same (public) |
| `CommittedPolynomial::len/from_index/to_index/ram_d` | Reuse directly | Same (public) |
| `WitnessData` struct | `Rep3WitnessData` | New — see Section 2 |
| `WitnessData::new` | `Rep3WitnessData::new` | New — see Section 2 |
| `SharedWitnessData` (UnsafeCell wrapper) | Not needed | Sequential iteration (MPC ops need `io_ctx`) |
| `CommittedPolynomial::generate_witness_batch` | `generate_witness_batch_rep3` | New — see below |
| `CommittedPolynomial::generate_witness` | Not needed | `generate_witness_batch` subsumes it |
| `VirtualPolynomial` enum | Out of scope | R1CS / virtual polynomials are a later phase |
| `ALL_VIRTUAL_POLYNOMIALS` | Out of scope | Same |

### `generate_witness_batch_rep3` Signature

```rust
pub fn generate_witness_batch_rep3<F, PCS, N>(
    polynomials: &[CommittedPolynomial],
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: &[Rep3Cycle],
    io_ctx: &mut IoContextPool<N>,
) -> eyre::Result<HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
```

Input is `&[Rep3Cycle]` (NOT vanilla `&[Cycle]`). The caller converts the vanilla trace to `Rep3Cycle` first via `convert_trace_to_rep3`.

### Per-Field Computation

Rep3 share types already implement `Add`, `Sub`, `Mul<scalar>` for linear operations, so most vanilla computations translate directly. We walk through each field:

**a) `left_instruction_input[i] = left`**
- Vanilla: `u64` from `to_instruction_inputs(cycle).0`
- Rep3: `Rep3Operand` from `Rep3LookupQuery::to_instruction_inputs_rep3(cycle).0`. Extract as `Rep3RingShare<u64>`.

**b) `right_instruction_input[i] = right`**
- Vanilla: `i128` from `to_instruction_inputs(cycle).1`
- Rep3: `Rep3Operand` from `.1`. Signed immediates handled as u64 in the ring.

**c) `write_lookup_output_to_rd[i] = rd_write_flag * circuit_flags[WriteLookupOutputToRD]`**
- Both PUBLIC. Compute as plain `u8`, store directly.

**d) `write_pc_to_rd[i] = rd_write_flag * circuit_flags[Jump]`**
- Both PUBLIC. Compute as plain `u8`, store directly.

**e) `should_branch[i] = lookup_output * circuit_flags[Branch]`**
- `lookup_output` is SHARED, `circuit_flags[Branch]` is PUBLIC.
- Linear op: `share * public_scalar`.

**f) `should_jump[i] = is_jump * (1 - is_next_noop)`**
- Both PUBLIC. Compute as plain `u8`, store directly.

**g) `rd_pre[i]`, `rd_post[i]`**
- Option A: store separately as `Rep3RingShare<u64>`. Source: `rep3_cycle.rd_write()` returns `(u8, Rep3Operand, Rep3Operand)`.

**h) `instruction_ra[j][i]`**
- Vanilla: extract byte chunks from `lookup_index: u128` via shift+mask.
- Rep3: `lookup_index` is `Rep3RingShare<u128>` (assumed). Byte extraction requires nonlinear MPC ops.
- **Approach**: `a2b` to binary share, extract 16 bytes locally as `Rep3RingShare<u8>`, `b2a` back.

### Ring-to-Field Conversion Pipeline

Vanilla creates `MultilinearPolynomial<F>::from(coeffs)` which implicitly converts to field. In Rep3:

1. Collect all `Rep3RingShare<T>` coefficient vectors from `Rep3WitnessData`
2. Batch-convert via `ring_to_field_a2b_many::<T, F, N>(shares, io_ctx)` → `Vec<Rep3PrimeFieldShare<F>>`
3. Create `Rep3MultilinearPolynomial::from(field_shares)` (the `From<Vec<Rep3PrimeFieldShare<F>>>` impl)

For PUBLIC-only fields (`write_lookup_output_to_rd`, `write_pc_to_rd`, `should_jump`): skip MPC conversion, create `Rep3MultilinearPolynomial::Public(MultilinearPolynomial::from(plain_coeffs))` directly.

### OneHot Polynomials (`instruction_ra`)

Vanilla: `OneHotPolynomial::from_indices(indices, K_CHUNK)` → `MultilinearPolynomial::OneHot(...)`.

In Rep3 the indices are secret (`Rep3RingShare<u8>`). One-hot expansion requires OHV protocol (as v1 does in `subtable_lookup_indices_rep3`). **Deferred** to Lasso/Shout phase.

For now, store `instruction_ra` as `[Vec<Option<Rep3RingShare<u8>>>; D]` and return them as-is (not yet converted to polynomials). The `CommittedPolynomial::InstructionRa(i)` entries will be skipped in the returned HashMap for this phase.

### Scope

**In scope:**
- `Rep3Cycle` enum + conversion from vanilla `Cycle`
- `Rep3LookupQuery` trait with `to_instruction_inputs_rep3`
- Missing format impls (`FormatI`, `FormatB`, `FormatJ`, `FormatU`, `FormatS`, virtual formats)
- `Rep3WitnessData` struct + `new()`
- `generate_witness_batch_rep3` — populates `Rep3WitnessData`, converts to `Rep3MultilinearPolynomial`
- `Rep3MultilinearPolynomial` type (copy from co-jolt v1)

**Out of scope (stubs/todo):**
- `to_lookup_index_rep3`, `to_lookup_output_rep3` — stub `todo!()`
- `ram_inc`, `bytecode_ra`, `ram_ra` — out of instruction scope
- OneHot polynomial construction — deferred to Lasso/Shout
- `VirtualPolynomial` — R1CS phase
- `CommittedPolynomial::generate_witness` (single-poly version) — not needed

---

## 6. Implementation Steps

1. **Format impls** — add `Rep3RegisterStateFormatI`, `FormatB`, `FormatJ`, `FormatU`, `FormatS`, and virtual instruction formats in `src/zkvm/instruction/format/`
2. **Rep3Cycle enum** — define in `src/zkvm/instruction/mod.rs` with macro, impl methods (`instruction()`, `rs1_read()`, `rs2_read()`, `rd_write()`), impl `Rep3LookupQuery`
3. **Rep3Cycle conversion** — `from_public_cycle`, `convert_trace_to_rep3`, `promote_to_shares`
4. **Rep3MultilinearPolynomial** — copy from `co-jolt/src/poly/multilinear_polynomial.rs` into `src/poly/`
5. **Rep3WitnessData** — struct + `new()` in `src/zkvm/witness.rs`
6. **`generate_witness_batch_rep3`** — implement in `src/zkvm/witness.rs`
7. **Wire up modules** — `src/lib.rs`, `src/zkvm/mod.rs`, `src/poly/mod.rs`

---

## Files to Create/Modify

| File | Action |
|---|---|
| `src/zkvm/instruction/format/format_i.rs` | New: `Rep3RegisterStateFormatI` + impls |
| `src/zkvm/instruction/format/format_b.rs` | New: `Rep3RegisterStateFormatB` + impls |
| `src/zkvm/instruction/format/format_j.rs` | New: `Rep3RegisterStateFormatJ` + impls |
| `src/zkvm/instruction/format/format_u.rs` | New: `Rep3RegisterStateFormatU` + impls |
| `src/zkvm/instruction/format/format_s.rs` | New: `Rep3RegisterStateFormatS` + impls |
| `src/zkvm/instruction/format/format_inline.rs` | New: `Rep3RegisterStateFormatInline` + impls |
| `src/zkvm/instruction/format/mod.rs` | Add new format modules |
| `src/zkvm/instruction/mod.rs` | Add `Rep3Cycle` enum, `Rep3LookupQuery` trait |
| `src/zkvm/witness.rs` | New: `Rep3WitnessData` + `generate_witness_batch_rep3` |
| `src/zkvm/mod.rs` | Add `pub mod witness;` and `Rep3Jolt` trait |
| `src/poly/multilinear_polynomial.rs` | Copy `Rep3MultilinearPolynomial` from co-jolt v1 |
| `src/poly/mod.rs` | New: `pub mod multilinear_polynomial;` |
| `src/lib.rs` | Add `pub mod poly;` |

## Verification

```bash
cargo check
```

All types are stubs/data structures — no logic to test beyond compilation.
