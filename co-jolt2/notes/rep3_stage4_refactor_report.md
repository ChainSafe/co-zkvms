# Rep3 Jolt DAG Stage4 + `mles_product_sum` Refactor Report

Date: 2026-03-03

## Major structural changes

- **Stage4 DAG wiring** (`co-jolt2/src/zkvm/dag/stage.rs`)
  - Match vanilla: Lookups `InstructionRA` stage4 sumcheck is always instantiated (removed conditional/skip path).
  - Reduce coordinator→workers stage4 init broadcasts to **1** bundled message:
    - `(RamStage4Init<F>, BytecodeStage4Init<F>, ra_claim, r_address, r_cycle)`
  - Add coordinator-side `ensure!` that the InstructionRA opening exists (fail-fast instead of silently skipping).

- **Stage4 init serialization** (`co-jolt2/src/zkvm/ram/mod.rs`, `co-jolt2/src/zkvm/bytecode/mod.rs`)
  - `RamStage4Init<F>` / `BytecodeStage4Init<F>` now derive `CanonicalSerialize`/`CanonicalDeserialize`.

- **RAM stage4 worker hot path** (`co-jolt2/src/zkvm/ram/mod.rs`)
  - Remove `addresses.clone()` by constructing booleanity last and moving `addresses`.
  - Replace repeated full scans with a single-pass chunk histogram builder (`compute_address_chunk_hists`).

- **Bytecode stage4 worker hot path** (`co-jolt2/src/zkvm/bytecode/mod.rs`)
  - Fuse construction of `F_1` (d chunk-hists weighted by `eq_r_cycle`) and `F_polys` (3 K-hists weighted by `eq_evals`)
    into one parallel pass (`compute_pc_hists`).

- **`mles_product_sum`** (`co-jolt2/src/subprotocols/mles_product_sum.rs`)
  - Rewrite `compute_mles_product_16_rep3` to use flat, pre-sized buffers across levels, eliminating nested vectors and a
    separate flattening pass.
  - Keep the same 3 resharing points (`reshare_additive_many`), but parallelize local per-level computation.

## Perf deltas (bench + qualitative)

- **e2e bench (prebuilt, excludes compilation)**
  - Baseline (before changes): `46.41s` real, `2162982912` max RSS bytes, `1969781` msgs.
  - After changes: `44.36s` real, `2152169472` max RSS bytes, `1977318` msgs.
- **Comms**: stage4 init broadcasts reduced from 5+ down to 1 (coordinator→workers).
- **CPU/allocations**
  - Fewer passes over large arrays in RAM/Bytecode stage4 histogram construction.
  - `mles_product_sum`: fewer allocations and less temporary buffering; parallel local work on big independent loops.

## Semantic risks + validation

- Risk: stage4 lookups InstructionRA is now always present (intended to match vanilla ordering/semantics).
- Risk: refactors touch hot-path histogram construction and `mles_product_sum` buffer layout.
- Validation:
  - `cargo test -p co-jolt2 --test dag_correct --features test-utils -- --nocapture`
  - e2e bench rerun after prebuilding (numbers above).

## `promote_to_trivial_share` occurrences (notable ones)

- `co-jolt2/src/zkvm/dag/witness.rs` (`to_instruction_inputs`, `to_lookup_operands`):
  - Promotes public PC/imm/advice constants into Rep3 shares early.
  - Downstream impact: some consumers operate purely in “share” space and lose the ability to use cheaper “public×shared”
    fast paths (more below in the code-review note).
