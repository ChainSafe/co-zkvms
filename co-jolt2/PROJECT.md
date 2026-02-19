# Co-Jolt2 MPC-ZK-VM

## Context

Building MPC version of Jolt zkVM by extending vanilla Jolt (`../../examples/jolt/tracer/` `../../examples/jolt/jolt-core/`) with MPC secret-share types. This plan uses a **hybrid approach** where vanilla Jolt generates public traces, and co-jolt2 converts them to MPC traces for witness generaion.

**Key Requirements:**
- Minimal modifications to vanilla Jolt codebase 
- No wrapper types (implement traits directly on vanilla types)

## References

This project is based on `../../examples/jolt` (refered to as "vanilla")
The previous version: `../co-jolt` (refered to as "v1"). It implements LEGACY Jolt VM (different from "vanilla" Jolt VM!)

- Summary of Jolt codebase wrt. Instructions `/Users/timofey/repos/examples/jolt/Instructions.md`
- Summary of Jolt codebase wrt. Witness `/Users/timofey/repos/examples/jolt/Instructions.md`
- Summary of `co-jolt` (v1) and `mpc-types` crates wrt. MPC types and core methods `../co-jolt/MpcTypes.md`
- Implementation docs:
  - Rep3 Dory commitment scheme `co-jolt2/docs/DORY.md`
  - Rep3 Polynomials `co-jolt2/docs/POLY.md`


## Rules

### Naming Conventions
- Use **suffix** pattern for Worker/Coordinator structs: `StateManagerWorker`, `SumcheckStagesCoordinator` (not `WorkerStateManager`)
- Methods on `*Worker`/`*Coordinator` structs **omit** the role from method names: `prove()`, not `prove_worker()`
- Worker and Coordinator are **separate structs in separate files**

### Imports
- When importing MPC ops from `(rep3|rep3_ring)::(arithmetic|binary)`, alias the module as `arithmetic`/`binary` (e.g. `use ...::rep3::arithmetic as arithmetic;`) unless the same file uses both `rep3` and `rep3_ring` `arithmetic`/`binary` (ambiguity).

### Usage of References

- Use vanilla Jolt (`../../examples/jolt/`) as the LOGIC reference w.r.t Jolt VM logic. 
- Co-jolt v1 (`../co-jolt/`) is LEGACY
  - use it as MPC arhitecture reference **but be extremely cautious not to use it as Jolt VM logic reference**
  - use it to copy proven MPC utility implementations (Rep3Operand, share operations) , never for Jolt related architectural patterns.

### Naming Conventions

1. Structs use **suffix** pattern: `StateManagerWorker`, `SumcheckStagesCoordinator` (not `WorkerStateManager`)
2. Methods on `*Worker`/`*Coordinator` structs **omit** "worker"/"coordinator" from method names (e.g. `prove()`, `stage1_prove()`, `stage2_instances()` — not `prove_worker()`, `stage1_prove_worker()`)
3. `JoltDAGWorker` and `JoltDAGCoordinator` are **separate structs in separate files** (not methods on a shared enum)

These conventions must also be added to PROJECT.md Rules section during implementation.


### Test Naming

Format: `<subject>_<what_is_checked>`. Rules:
1. No `test_` prefix — `#[test]` is enough
2. No `_and_` — split into separate tests or pick the primary check
3. No mentions of `vanilla` — correctness against vanilla is implied. Use `_correct` suffix
4. Do not use `_reconstructs` in test names — use `_correct` for correctness assertions
5. Use `open` not `reconstruct`
6. Keep names ≤ 5 snake_case tokens
7. Subject first — e.g. `witness_batch`, `one_hot_eval`, `dory_commit`
8. Abbreviations OK when unambiguous (`eval`, `commit`, `batch`). Use `one_hot` not `ohp`

### Modifying Vanilla Jolt
When the vanilla codebase (`../../examples/jolt/`) has private methods or attributes that co-jolt2 needs, you are allowed to make them `pub` directly in the vanilla code. This is a prototype — don't waste time on workarounds.

## File Structure Must Mirror Vanilla
co-jolt2's `src/` directory structure must mirror the vanilla Jolt codebase. We combine:
- `../../examples/jolt/tracer/src/instruction/` (tracer instruction types)
- `../../examples/jolt/jolt-core/src/zkvm/instruction/` (jolt-core instruction logic)

into a single `src/zkvm/instruction/` directory.

**Mapping:**
| Vanilla | co-jolt2 |
|---------|----------|
| `jolt-core/src/zkvm/` | `src/zkvm/` |
| `jolt-core/src/zkvm/instruction/` | `src/zkvm/instruction/` |
| `tracer/src/instruction/format/` | `src/zkvm/instruction/format/` |

When adding new modules (e.g., `bytecode/`, `r1cs/`, `witness.rs`), place them at the same relative path as vanilla's `jolt-core/src/zkvm/`.

---

## Trace Data: Shared vs Public Classification

All data extracted from a `Cycle` during witness generation:

### Source Values from Cycle

| Value | Source | Type | Shared/Public | Notes |
|-------|--------|------|---------------|-------|
| `rs1` | `register_state.rs1` | `u64` | **SHARED** | Secret register value |
| `rs2` | `register_state.rs2` | `u64` | **SHARED** | Secret register value |
| `rd.0` (pre) | `register_state.rd_values().0` | `u64` | **SHARED** | Secret register pre-value |
| `rd.1` (post) | `register_state.rd_values().1` | `u64` | **SHARED** | Secret register post-value |
| `rd_write_flag` | `NormalizedOperands::from(..).rd` | `u8` | **PUBLIC** | Destination register index (0 = no write). Part of instruction encoding. |
| `imm` | `instruction.operands.imm` | `i64/u64` | **PUBLIC** | Immediate value, part of program |
| `address` (PC) | `instruction.address` | `u64` | **PUBLIC** | Program counter |
| `circuit_flags[..]` | `instruction.circuit_flags()` | `[bool; 18]` | **PUBLIC** | Determined by opcode |
| `lookup_output` | `to_lookup_output(cycle)` | `u64` | **SHARED** | Result of instruction computation on secret inputs |
| `advice` | `instruction.advice` | `u64` | **SHARED** | VirtualAdvice: secret value |
| `ram_access.pre_value` | `cycle.ram_access()` | `u64` | **SHARED** | Secret RAM value |
| `ram_access.post_value` | `cycle.ram_access()` | `u64` | **SHARED** | Secret RAM value |
| `ram_access.address` | `cycle.ram_access()` | `u64` | **PUBLIC** | Memory address (public in Jolt model) |

### Instruction Inputs (`to_instruction_inputs`)

| Format | `left` (u64) | `right` (i128) |
|--------|-------------|----------------|
| R-type (ADD, SUB, AND, ...) | `rs1` **SHARED** | `rs2 as i128` **SHARED** |
| I-type (ADDI, ANDI, ...) | `rs1` **SHARED** | `imm as i128` **PUBLIC** |
| B-type (BEQ, BNE, ...) | `rs1` **SHARED** | `rs2 as i128` **SHARED** |
| J-type (JAL) | `address` (PC) **PUBLIC** | `imm as i128` **PUBLIC** |
| U-type (LUI) | `0` **PUBLIC** | `imm as i128` **PUBLIC** |
| AUIPC | `address` (PC) **PUBLIC** | `imm as i128` **PUBLIC** |
| VirtualAdvice | `0` **PUBLIC** | `0` **PUBLIC** |

### Lookup Index (`to_lookup_index`)

For most instructions: `interleave_bits(left, right as u64)` — bit-interleaving of two u64 values.
For ADD-type (AddOperands flag): `left + right` (single combined operand).
For VirtualAdvice: `advice` value directly.

This is a **nonlinear** operation on shared values when both operands are shared (R-type, B-type).


---

## MPC Architecture: Worker/Coordinator Pattern

Extracted from co-jolt v1 (`../co-jolt/src/jolt/vm/jolt/`). This pattern applies to the new vanilla DAG-based architecture.

### Roles

**Worker** (`io_ctx: IoContextPool<Network>`)
- Holds MPC-shared witness polynomials (`Rep3MultilinearPolynomial<F>`)
- Generates witness from shared trace data
- Commits polynomial shares independently, sends commitment shares to coordinator
- Performs distributed sumcheck: receives challenge points, computes evaluations, sends results; commitments, and opening proofs
- All MPC compute happens here

**Coordinator** (`network: &mut Network`)
- Owns the Fiat-Shamir transcript — sole generator of challenges
- Receives commitment shares from workers, combines into public commitments
- Broadcasts transcript challenges to workers at each protocol phase
- Receives evaluation shares from workers, combines into public values
- Assembles the final proof struct

### Communication Flow

```
COORDINATOR                              WORKER(s)
    │                                        │
    │  ←── commitment_shares ──────────────  │  (worker commits poly shares)
    │  (combine → public commitments)        │
    │  (append to transcript)                │
    │                                        │
    │  ── challenge (from transcript) ─────→ │  (broadcast_request)
    │                                        │  (worker computes on challenge)
    │  ←── evaluation_shares ──────────────  │  (send_response)
    │  (combine → public eval)               │
    │  (append proof to transcript)          │
    │                                        │
    │  ... repeat per protocol phase ...     │
    │                                        │
    │  ── opening_point ───────────────────→ │  (final batch opening)
    │  ←── opening_proof_shares ───────────  │
    │  (assemble JoltProof)                  │
```

### v1 Prove Flow (for reference)

**Coordinator** (`prove_rep3`):
1. `receive_commitments()` → combine shares → append to transcript
2. `BytecodeProof::coordinate_memory_checking()` — broadcast challenges, receive proof shares
3. `InstructionLookupsProof::prove_rep3()` — broadcast r_eq, coordinate sumcheck
4. `ReadWriteMemoryProof::prove_rep3()` — coordinate output sumcheck + timestamp
5. `UniformSpartanProof::prove_rep3()` — broadcast tau, coordinate outer sumcheck
6. `opening_accumulator.reduce_and_prove()` — batch all openings into single proof
7. Assemble `JoltProof` struct

**Worker** (`prove`):
1. `polynomials.commit()` → send commitment shares
2. `Rep3BytecodeProver::prove_memory_checking()` — receive challenges, compute grand products
3. `Rep3InstructionLookupsProver::prove()` — receive r_eq, compute sumcheck evals
4. `Rep3ReadWriteMemoryProver::prove()` — receive challenges, compute memory proofs
5. `Rep3UniformSpartanProver::prove()` — receive tau, compute Spartan evals
6. `opening_accumulator.reduce_and_prove()` — send opening shares

### Key Types from v1

| v1 Type | Role | Key Fields |
|---------|------|------------|
| `JoltRep3Prover` (worker) | Worker state machine | `io_ctx`, `polynomials`, `preprocessing`, `r1cs_builder`, `spartan_key` |
| `JoltRep3` trait (coordinator) | Coordinator interface | `init_rep3()`, `prove_rep3()` |
| `Rep3OpeningAccumulatorWorker` | Worker opening batching | Accumulates claims, sends to coordinator |
| `Rep3OpeningAccumulatorCoordinator` | Coordinator opening batching | Receives shares, combines, produces single proof |
| `JoltWitnessMeta` | Init metadata | `padded_trace_length`, `read_write_memory_size`, `memory_layout` |

---

## Vanilla DAG Architecture

The vanilla Jolt proof is organized as a 5-stage DAG pipeline in `jolt-core/src/zkvm/dag/`.

### DAG Components

| File | Purpose |
|------|---------|
| `jolt_dag.rs` | `JoltDAG` enum — orchestrates `prove()` and `verify()` |
| `state_manager.rs` | `StateManager` — central hub holding transcript, proofs, commitments, accumulator |
| `stage.rs` | `SumcheckStages` trait — interface for subsystem DAG nodes |
| `proof_serialization.rs` | `JoltProof` — serializable proof artifact with conversion methods |

### StateManager

Central state container threaded through all DAG stages:

```rust
struct StateManager<'a, F, ProofTranscript, PCS> {
    transcript: Rc<RefCell<ProofTranscript>>,      // Fiat-Shamir
    proofs: Rc<RefCell<BTreeMap<ProofKeys, ProofData>>>,  // Accumulated proofs
    commitments: Rc<RefCell<Vec<PCS::Commitment>>>,       // PCS commitments
    untrusted_advice_commitment: Option<PCS::Commitment>,
    trusted_advice_commitment: Option<PCS::Commitment>,
    ram_K: usize,
    twist_sumcheck_switch_index: usize,
    program_io: JoltDevice,
    prover_state: Option<ProverState<'a, F, PCS>>,   // Trace + accumulator
    verifier_state: Option<VerifierState<'a, F, PCS>>,
}
```

### SumcheckStages Trait

Each subsystem implements this to plug into the staged pipeline:

```rust
trait SumcheckStages<F, ProofTranscript, PCS> {
    fn stage1_prove(&mut self, state_manager: &mut StateManager) -> Result<()>;
    fn stage2_prover_instances(&mut self, state_manager: &mut StateManager)
        -> Vec<Box<dyn SumcheckInstance>>;
    fn stage3_prover_instances(...) -> Vec<Box<dyn SumcheckInstance>>;
    fn stage4_prover_instances(...) -> Vec<Box<dyn SumcheckInstance>>;
    // + verify counterparts
}
```

### Subsystem DAG Nodes

| Node | Stages Active | Purpose |
|------|---------------|---------|
| `SpartanDag` | 1, 2, 3 | R1CS constraint verification (outer sumcheck) |
| `RegistersDag` | 2, 3 | Register read/write consistency |
| `RamDag` | 2, 3, 4 | RAM read/write consistency |
| `LookupsDag` | 2, 3, 4 | Instruction lookup table proofs |
| `BytecodeDag` | 4 | Bytecode integrity proofs |

### Vanilla Prove Flow

```
JoltDAG::prove(state_manager):
  1. fiat_shamir_preamble()
  2. commit_untrusted_advice() (if any)
  3. generate_and_commit_polynomials() → all witness polys committed
  4. Append commitments + advice commitments to transcript
  
  Stage 1: SpartanDag::stage1_prove() — outer sumcheck
  
  Stage 2: BatchedSumcheck::prove([
      SpartanDag, RegistersDag, RamDag, LookupsDag
  ].stage2_prover_instances())
  
  Stage 3: BatchedSumcheck::prove([
      SpartanDag, RegistersDag, LookupsDag, RamDag
  ].stage3_prover_instances())
  
  Stage 4: BatchedSumcheck::prove([
      RamDag, BytecodeDag, LookupsDag
  ].stage4_prover_instances())
  
  Stage 5: Opening proofs
    - trusted_advice opening proof
    - untrusted_advice opening proof
    - accumulator.reduce_and_prove() → single batch PCS proof
  
  → JoltProof::from_prover_state_manager()
```

### Data Flow Between Stages

- **Fiat-Shamir transcript**: Each stage's proof appended → next stage derives challenges from it
- **Opening accumulator**: All stages append polynomial openings → Stage 5 batches them all
- **ProofKeys map**: Each stage inserts its proof under a key (Stage1Sumcheck..Stage4Sumcheck, ReducedOpeningProof, etc.)

---

## DAG Implementation Plan: MPC Version

### Design Principle

Map the vanilla DAG's `StateManager` pattern onto the worker/coordinator split:

| Vanilla | MPC Worker | MPC Coordinator |
|---------|------------|-----------------|
| `StateManager` | `StateManagerWorker` | `StateManagerCoordinator` |
| `ProverState` (trace + accumulator) | `ProverStateWorker` (shared polys + advice poly) | N/A (no trace) |
| `ProverOpeningAccumulator` | `Rep3OpeningAccumulatorWorker` | `Rep3OpeningAccumulatorCoordinator` |
| `transcript` (owns) | N/A (no transcript) | `transcript` (owns) |
| `commitments` (stores) | sends shares | receives + combines |
| `SumcheckStages` trait | `SumcheckStagesWorker` trait | `SumcheckStagesCoordinator` trait |

### Module Structure

```
src/zkvm/dag/
├── mod.rs                       // Module declarations
├── state_manager.rs             // StateManagerWorker, StateManagerCoordinator
├── stage.rs                     // SumcheckStagesWorker, SumcheckStagesCoordinator traits
├── jolt_dag_worker.rs           // JoltDAGWorker — worker prove flow
├── jolt_dag_coordinator.rs      // JoltDAGCoordinator — coordinator prove flow
└── (no proof_serialization.rs)  // Re-use vanilla's JoltProof (coordinator assembles it)
```

### Types

#### `StateManagerWorker`

```rust
pub struct StateManagerWorker<'a, F, PCS, N: Rep3Network> {
    pub io_ctx: IoContextPool<N>,
    pub commitments: Vec<PCS::Commitment>,  // worker's commitment shares
    pub ram_K: usize,
    pub twist_sumcheck_switch_index: usize,
    pub program_io: JoltDevice,  // public portion
    pub prover_state: ProverStateWorker<'a, F, PCS>,
}

pub struct ProverStateWorker<'a, F, PCS> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: Vec<Rep3Cycle>,  // MPC-shared trace
    pub final_memory_state: Memory,
    pub untrusted_advice_polynomial: Option<Rep3MultilinearPolynomial<F>>,
}
```

#### `StateManagerCoordinator`

```rust
pub struct StateManagerCoordinator<'a, F, ProofTranscript, PCS> {
    pub transcript: ProofTranscript,
    pub proofs: BTreeMap<ProofKeys, ProofData<F, PCS, ProofTranscript>>,
    pub commitments: Vec<PCS::Commitment>,  // combined public commitments
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub ram_K: usize,
    pub twist_sumcheck_switch_index: usize,
    pub program_io: JoltDevice,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
}
```

#### Split `SumcheckStages` Trait

```rust
// Worker: produces sumcheck instance contributions (shared polynomials)
pub trait SumcheckStagesWorker<F, PCS, N: Rep3Network> {
    fn stage1_prove(&mut self, state: &mut StateManagerWorker<F, PCS, N>) -> Result<()> { Ok(()) }
    fn stage2_instances(&mut self, state: &mut StateManagerWorker<F, PCS, N>)
        -> Vec<Box<dyn Rep3SumcheckInstance<F>>> { vec![] }
    fn stage3_instances(...) -> Vec<...> { vec![] }
    fn stage4_instances(...) -> Vec<...> { vec![] }
}

// Coordinator: drives sumcheck rounds via transcript
pub trait SumcheckStagesCoordinator<F, ProofTranscript, PCS> {
    fn stage1_prove(&mut self, state: &mut StateManagerCoordinator<...>) -> Result<()> { Ok(()) }
    fn stage2_instances(&mut self, state: &mut StateManagerCoordinator<...>) -> Vec<...> { vec![] }
    fn stage3_instances(...) -> Vec<...> { vec![] }
    fn stage4_instances(...) -> Vec<...> { vec![] }
}
```

#### `JoltDAGWorker` / `JoltDAGCoordinator`

```rust
// jolt_dag_worker.rs
pub struct JoltDAGWorker;

impl JoltDAGWorker {
    /// Worker side: generates shared witness, commits, participates in sumchecks
    pub fn prove<F, PCS, N>(
        state: StateManagerWorker<F, PCS, N>,
    ) -> Result<()>;
}

// jolt_dag_coordinator.rs
pub struct JoltDAGCoordinator;

impl JoltDAGCoordinator {
    /// Coordinator side: drives transcript, coordinates sumchecks, assembles proof
    pub fn prove<F, ProofTranscript, PCS, N>(
        state: StateManagerCoordinator<F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> Result<JoltProof<F, PCS, ProofTranscript>>;
}
```

### Prove Flow (MPC)

```
COORDINATOR                              WORKER(s)
    │                                        │
    │  fiat_shamir_preamble()                │
    │                                        │
    │  ←── commitment_shares ──────────────  │  generate_and_commit_polynomials()
    │  combine → set commitments             │
    │  append commitments to transcript      │
    │                                        │
    │  ── sync ────────────────────────────  │
    │                                        │
    │  Stage 1 (Spartan outer sumcheck):     │
    │  ── tau ─────────────────────────────→ │
    │  ←── sumcheck_evals ─────────────────  │  SpartanDagWorker::stage1_prove()
    │  SpartanDagCoordinator::stage1_prove() │
    │  insert Stage1Sumcheck proof           │
    │                                        │
    │  Stage 2 (batched sumcheck):           │
    │  coord gets instances                  │  worker gets instances
    │  ── batching_challenge ──────────────→ │
    │  for each round:                       │
    │    ←── round_evals ──────────────────  │  worker binds + evaluates
    │    combine, derive challenge           │
    │    ── round_challenge ───────────────→ │
    │  insert Stage2Sumcheck proof           │
    │                                        │
    │  Stage 3 (same pattern) ...            │
    │  Stage 4 (same pattern) ...            │
    │                                        │
    │  Stage 5 (batch opening):              │
    │  ── opening_point ───────────────────→ │
    │  ←── opening_proof_shares ───────────  │  accumulator.reduce_and_prove()
    │  combine → ReducedOpeningProof         │
    │                                        │
    │  Assemble JoltProof                    │
```



## Implementation Steps

### Worker (Done)
- **Trace sharing**: `host/program.rs` — `share_trace`, `share_cycle`, `generate_trace_shares`
- **Instructions**: 70+ `Rep3LookupQuery` impls in `zkvm/instruction/`
- **Operand promotion**: `populate_operands_casts` in `zkvm/instruction/mod.rs`
- **Witness generation**: `generate_witness_batch_rep3` in `zkvm/witness.rs`
- **Polynomial types**: `Rep3MultilinearPolynomial`, `Rep3DensePolynomial`, `Rep3OneHotPolynomial`
- **Commitment**: `Rep3CommitmentScheme` + Dory impl in `poly/commitment/`
- **DAG worker**: commit + hint exchange in `zkvm/dag/worker.rs`
- **Advice stubs**: `commit_untrusted_advice`, `compute_trusted_advice_poly` (empty-case only)

### Worker (Remaining)
- Advice polynomial construction (non-empty case)
- RAM initial/final memory state construction (use `Rep3Value<F>` for mixed public/shared)
- Virtual polynomial evaluation (`compute_claimed_witness_evals_rep3`)
- Subsystem DAG nodes: `SpartanDagWorker`, `RegistersDagWorker`, `RamDagWorker`, `LookupsDagWorker`, `BytecodeDagWorker`
- `Rep3BatchedSumcheck` (distributed round evaluation)
- `Rep3OpeningAccumulatorWorker`

### Coordinator (Done)
- **Commitment receive + combine** in `zkvm/dag/coordinator.rs`
- **Fiat-Shamir preamble** in `zkvm/dag/state_manager.rs`
- **Advice stubs**: empty-case advice checks, transcript append ordering
- **Stub JoltProof construction**: real commitments + default placeholders for unimplemented stages

### Coordinator (Remaining)
- Advice commitment handling (non-empty case)
- Subsystem DAG node coordination (stages 1-4)
- `Rep3BatchedSumcheck` transcript-driven round coordination
- `Rep3OpeningAccumulatorCoordinator`
- Proof assembly with real opening proofs

### Tests (Done)
- `witness_batch_rep3`: 3-party witness generation correctness (in `zkvm/witness.rs`)
- `commitment_correct`: 3 workers + 1 coordinator end-to-end commitment correctness (in `zkvm/mod.rs`)

### RAM Memory State Design Note
Use `Rep3Value<F>` (`src/utils/types/rep3_value.rs`) for mixed initial/final memory state:
- Bytecode words → `Rep3Value::Public(F::from(word))`
- Advice bytes → `Rep3Value::Shared(ring_to_field(packed_shared_u64))`
- Input bytes → `Rep3Value::Public(F::from(word))` (inputs are public)
- Cascading: `eq(r,k)` is public challenge → `Public * Shared = Shared` propagates correctly through Twist sumchecks
