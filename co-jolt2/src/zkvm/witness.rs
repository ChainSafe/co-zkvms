use mpc_core::protocols::rep3_ring::Rep3RingShare;

use super::instruction::{Rep3Cycle, Rep3Operand};

/// Number of chunks for instruction lookup decomposition.
/// Mirrors `instruction_lookups::D` in vanilla jolt-core.
pub const INSTRUCTION_D: usize = 16;

/// Rep3 version of vanilla WitnessData.
/// Holds shared witness polynomial coefficients for instruction-scope polynomials.
///
/// All register-derived values stored as `Rep3RingShare<T>` (extracted from Rep3Operand
/// via `as_arithmetic`). Flag fields that are PUBLIC in vanilla are stored as
/// `Rep3RingShare<u8>` (trivial shares) for uniform treatment in the polynomial pipeline.
pub struct Rep3WitnessData {
    pub left_instruction_input: Vec<Rep3RingShare<u64>>,
    pub right_instruction_input: Vec<Rep3RingShare<u64>>,
    pub write_lookup_output_to_rd: Vec<Rep3RingShare<u8>>,
    pub write_pc_to_rd: Vec<Rep3RingShare<u8>>,
    pub should_branch: Vec<Rep3RingShare<u8>>,
    pub should_jump: Vec<Rep3RingShare<u8>>,

    // Stored as separate pre/post to avoid signed arithmetic (see PLAN.md Section 3)
    pub rd_pre: Vec<Rep3RingShare<u64>>,
    pub rd_post: Vec<Rep3RingShare<u64>>,

    pub instruction_ra: [Vec<Option<Rep3RingShare<u8>>>; INSTRUCTION_D],
}

impl Rep3WitnessData {
    pub fn new(trace_len: usize) -> Self {
        Self {
            left_instruction_input: vec![Rep3RingShare::default(); trace_len],
            right_instruction_input: vec![Rep3RingShare::default(); trace_len],
            write_lookup_output_to_rd: vec![Rep3RingShare::default(); trace_len],
            write_pc_to_rd: vec![Rep3RingShare::default(); trace_len],
            should_branch: vec![Rep3RingShare::default(); trace_len],
            should_jump: vec![Rep3RingShare::default(); trace_len],
            rd_pre: vec![Rep3RingShare::default(); trace_len],
            rd_post: vec![Rep3RingShare::default(); trace_len],
            instruction_ra: std::array::from_fn(|_| vec![None; trace_len]),
        }
    }
}

/// Helper: extract `Rep3RingShare<u64>` from a `Rep3Operand`.
/// For shared operands, uses `as_arithmetic`. For public, creates a zero share
/// (the public value is embedded via the ring representation).
fn operand_to_ring_u64(op: &Rep3Operand) -> Rep3RingShare<u64> {
    match op {
        Rep3Operand::Shared { .. } => op.as_arithmetic_u64(),
        Rep3Operand::Public(_) => {
            // Public operands should have been promoted to trivial shares before
            // calling this function. Panic to catch misuse.
            panic!("operand_to_ring_u64: expected shared operand, got public. Promote to shares first.")
        }
    }
}

/// Populate `Rep3WitnessData` from a Rep3 trace.
///
/// Mirrors vanilla `CommittedPolynomial::generate_witness_batch` per-cycle logic.
/// The trace must have been promoted to shares (via `promote_to_shares`) and
/// arithmetic representations populated (via `populate_arithmetic`) before calling.
///
/// Fields populated:
/// - `left_instruction_input`, `right_instruction_input` — from rs1/rs2 register state
/// - `rd_pre`, `rd_post` — from rd_write()
///
/// Deferred fields (set to default zero shares):
/// - `write_lookup_output_to_rd`, `write_pc_to_rd`, `should_jump` — need CircuitFlags (jolt-core)
/// - `should_branch` — needs to_lookup_output_rep3 (Lasso/Shout phase)
/// - `instruction_ra` — needs to_lookup_index_rep3 (Lasso/Shout phase)
///
/// ## Ring-to-Field Conversion Pipeline (next step, not yet implemented)
///
/// After populating, the caller should:
/// 1. Batch-convert `Vec<Rep3RingShare<T>>` → `Vec<Rep3PrimeFieldShare<F>>`
///    via `ring_to_field_a2b_many::<T, F, N>(shares, io_ctx)`
/// 2. Create `Rep3MultilinearPolynomial::from(field_shares)`
///
/// This requires jolt-core types (`JoltField`, `CommittedPolynomial`, etc.)
/// which are out of scope for Phase 1.
pub fn populate_witness_data(trace: &[Rep3Cycle]) -> Rep3WitnessData {
    let mut data = Rep3WitnessData::new(trace.len());

    for (i, cycle) in trace.iter().enumerate() {
        // Left/right instruction inputs from register state
        let (_rs1_idx, rs1_val) = cycle.rs1_read();
        let (_rs2_idx, rs2_val) = cycle.rs2_read();
        data.left_instruction_input[i] = operand_to_ring_u64(rs1_val);
        data.right_instruction_input[i] = operand_to_ring_u64(rs2_val);

        // rd pre/post values
        let (_rd_idx, pre, post) = cycle.rd_write();
        data.rd_pre[i] = operand_to_ring_u64(pre);
        data.rd_post[i] = operand_to_ring_u64(post);

        // Flag fields (write_lookup_output_to_rd, write_pc_to_rd, should_jump)
        // require CircuitFlags from jolt-core which is not available in Phase 1.
        // These are PUBLIC values derived from the opcode, so they can be computed
        // once jolt-core is added as a dependency.

        // should_branch requires to_lookup_output_rep3 (deferred to Lasso/Shout)

        // instruction_ra requires to_lookup_index_rep3 (deferred to Lasso/Shout)
    }

    data
}

// NOTE: The full `generate_witness_batch_rep3` function with signature:
//
//   pub fn generate_witness_batch_rep3<F, PCS, N>(
//       polynomials: &[CommittedPolynomial],
//       preprocessing: &JoltProverPreprocessing<F, PCS>,
//       trace: &[Rep3Cycle],
//       io_ctx: &mut IoContextPool<N>,
//   ) -> eyre::Result<HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>>
//
// requires jolt-core types (JoltField, CommitmentScheme, CommittedPolynomial,
// JoltProverPreprocessing, MultilinearPolynomial) and Rep3MultilinearPolynomial.
// These are blocked on re-adding jolt-core as a dependency (ark version conflict).
//
// When jolt-core is available, this function should:
// 1. Call `populate_witness_data(trace)` to get `Rep3WitnessData`
// 2. For each CommittedPolynomial in `polynomials`:
//    - Extract the corresponding Vec<Rep3RingShare<T>> from Rep3WitnessData
//    - Convert via `ring_to_field_a2b_many::<T, F, N>(shares, io_ctx)`
//    - Create `Rep3MultilinearPolynomial::from(field_shares)`
//    - For public-only fields: `Rep3MultilinearPolynomial::Public(MultilinearPolynomial::from(plain))`
// 3. Return HashMap mapping CommittedPolynomial → Rep3MultilinearPolynomial
