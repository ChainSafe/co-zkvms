use std::array;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use itertools::Itertools;
use jolt2_common::constants::XLEN;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::zkvm::instruction::{CircuitFlags, InstructionFlags, InstructionLookup};
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::{CommittedPolynomial, DTH_ROOT_OF_K};
use jolt_core::zkvm::{instruction_lookups, JoltProverPreprocessing};
use mpc_core::protocols::rep3::network::{IoContext, IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3_ring::casts::ring_to_field_a2b_many;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use snarks_core::math::Math;
use tracing::info_span;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::Rep3MultilinearPolynomial;
use crate::utils::future_ring::{FutureRep3Ring, Rep3RingFutureExt};
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::instruction::{populate_operands_casts, Rep3LookupQuery, Rep3Operand};

use super::instruction::{Rep3Cycle, Rep3RAMAccess};

// ── Rep3WitnessData ─────────────────────────────────────────────────────────

/// Mirrors vanilla `WitnessData` but uses `Rep3RingShare` for shared fields.
struct Rep3WitnessData {
    rd_pre: Vec<Rep3RingShare<u64>>,
    rd_post: Vec<Rep3RingShare<u64>>,
    write_lookup_output_to_rd: Vec<u8>,
    write_pc_to_rd: Vec<u8>,
    should_jump: Vec<u8>,
    ram_pre: Vec<Rep3RingShare<u64>>,
    ram_post: Vec<Rep3RingShare<u64>>,
    // Deferred: instruction_ra (needs to_lookup_index_rep3)
    instruction_ra: [Vec<Option<Rep3RingShare<u8>>>; instruction_lookups::D],
    bytecode_ra: Vec<Vec<Option<u8>>>,
    ram_ra: Vec<Vec<Option<u8>>>,
}

impl Rep3WitnessData {
    fn new(trace_len: usize, ram_d: usize, bytecode_d: usize) -> Self {
        Self {
            rd_pre: vec![Rep3RingShare::default(); trace_len],
            rd_post: vec![Rep3RingShare::default(); trace_len],
            write_lookup_output_to_rd: vec![0; trace_len],
            write_pc_to_rd: vec![0; trace_len],
            should_jump: vec![0; trace_len],
            ram_pre: vec![Rep3RingShare::default(); trace_len],
            ram_post: vec![Rep3RingShare::default(); trace_len],

            instruction_ra: array::from_fn(|_| vec![None; trace_len]),
            bytecode_ra: (0..bytecode_d).map(|_| vec![None; trace_len]).collect(),
            ram_ra: (0..ram_d).map(|_| vec![None; trace_len]).collect(),
        }
    }
}

struct SharedRep3WitnessData(UnsafeCell<Rep3WitnessData>);
unsafe impl Sync for SharedRep3WitnessData {}

// ── compute_lookup_outputs ──────────────────────────────────────────────────

/// Compute the lookup output for each cycle in the trace.
///
/// Mirrors v1's `compute_lookup_outputs_rep3`:
/// 1. Initialize output futures as `Ready(zero_share)`
/// 2. Group `(cycle, &mut future)` pairs by instruction discriminant (skip NoOp/INLINE)
/// 3. Process each group via `par_chunks` → `to_lookup_output_batched`
/// 4. `fulfill_batched` all futures into `Rep3PrimeFieldShare<F>`
#[tracing::instrument(skip_all, name = "compute_lookup_outputs")]
pub fn compute_lookup_outputs<F, N>(
    trace: &[Rep3Cycle],
    io_ctx: &mut IoContextPool<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    F: JoltField,
    N: Rep3NetworkWorker,
    Standard: Distribution<u64>,
{
    let mut output_futures: Vec<FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>> =
        vec![FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share()); trace.len()];

    // Group by instruction type, skipping non-lookup cycles (NoOp, INLINE)
    let ops_by_instruction: Vec<(
        Vec<&Rep3Cycle>,
        Vec<&mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    )> = trace
        .iter()
        .zip(output_futures.iter_mut())
        .filter(|(cycle, _)| cycle.lookup_table().is_some())
        .group_by(|(cycle, _)| mem::discriminant(*cycle))
        .into_iter()
        .map(|(_, group)| group.unzip())
        .collect();

    // Process each instruction group via par_chunks
    io_ctx.par_chunks(
        ops_by_instruction,
        None,
        |groups, io_ctx: &mut IoContext<N>| -> eyre::Result<Vec<()>> {
            for (steps, out) in groups {
                Rep3LookupQuery::<XLEN>::to_lookup_output_batched(steps[0], &steps, io_ctx, out)?;
            }
            Ok(vec![()])
        },
    )?;

    // Fulfill all pending futures (batched casts via io_ctx)
    output_futures.fulfill_batched(io_ctx, |res, ()| res)
}

// ── populate_cycle_witness_rep3 ─────────────────────────────────────────────

/// Populate `state.prover_state.cycle_witness` from the (ring-shared) trace.
///
/// This cache is the field-domain, per-cycle witness representation used by
/// Stage 1 Spartan and later stages, allowing the ring-shared trace to be dropped.
#[tracing::instrument(skip_all, name = "populate_cycle_witness_rep3")]
pub fn populate_cycle_witness_rep3<F, PCS, N>(
    state: &mut StateManagerWorker<'_, F, PCS>,
    io_ctx: &mut IoContextPool<N>,
) -> eyre::Result<()>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
    Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
{
    let party_id = state.party_id;
    let preprocessing = state.prover_state.preprocessing;
    let trace = &mut state.prover_state.trace;
    eyre::ensure!(
        trace.len().is_power_of_two(),
        "trace length must be power-of-two"
    );

    // Ensure all shared operands have arithmetic representations (batched).
    populate_operands_casts(trace, io_ctx.main())?;

    // Compute lookup outputs (batched, MPC).
    let lookup_output = compute_lookup_outputs::<F, N>(trace, io_ctx)?;

    let n = trace.len();

    // Public columns
    let mut pc: Vec<u64> = Vec::with_capacity(n);
    let mut unexpanded_pc: Vec<u64> = Vec::with_capacity(n);
    let mut imm: Vec<i128> = Vec::with_capacity(n);
    let mut rd_addr: Vec<u8> = Vec::with_capacity(n);
    let mut rs1_addr: Vec<u8> = Vec::with_capacity(n);
    let mut rs2_addr: Vec<u8> = Vec::with_capacity(n);
    let mut ram_addr: Vec<u64> = Vec::with_capacity(n);
    let mut flags_bits: Vec<u32> = Vec::with_capacity(n);
    let mut advice: Vec<u64> = vec![0; n];

    // Ring-shared columns to cast to field
    let mut rs1_ring: Vec<Rep3RingShare<u64>> = Vec::with_capacity(n);
    let mut rs2_ring: Vec<Rep3RingShare<u64>> = Vec::with_capacity(n);
    let mut rd_write_ring: Vec<Rep3RingShare<u64>> = Vec::with_capacity(n);
    let mut ram_read_ring: Vec<Rep3RingShare<u64>> = Vec::with_capacity(n);
    let mut ram_write_ring: Vec<Rep3RingShare<u64>> = Vec::with_capacity(n);

    for (t, cycle) in trace.iter().enumerate() {
        let norm = cycle.instruction().normalize();
        let circuit_flags = cycle.instruction().circuit_flags();

        pc.push(cycle.get_pc(&preprocessing.shared.bytecode) as u64);
        unexpanded_pc.push(norm.address as u64);
        imm.push(norm.operands.imm);

        let (rs1_i, rs1_v) = cycle.rs1_read();
        let (rs2_i, rs2_v) = cycle.rs2_read();
        let (rd_i, _rd_pre, rd_post) = cycle.rd_write();

        rs1_addr.push(rs1_i);
        rs2_addr.push(rs2_i);
        rd_addr.push(rd_i);

        ram_addr.push(cycle.ram_access().address());

        // Pack circuit flags into a u32 bitmask
        let mut mask = 0u32;
        for (i, flag) in circuit_flags.iter().enumerate() {
            if *flag {
                mask |= 1u32 << i;
            }
        }
        flags_bits.push(mask);

        // Advice value (only meaningful for VirtualAdvice).
        if circuit_flags[CircuitFlags::Advice as usize] {
            if let Rep3Cycle::VirtualAdvice(c) = cycle {
                advice[t] = c.instruction.advice;
            }
        }

        rs1_ring.push(rs1_v.as_arithmetic_or_trivial(party_id));
        rs2_ring.push(rs2_v.as_arithmetic_or_trivial(party_id));
        rd_write_ring.push(rd_post.as_arithmetic_or_trivial(party_id));

        match cycle.ram_access() {
            Rep3RAMAccess::Read(r) => {
                // Match vanilla: for reads, RamReadValue == RamWriteValue == r.value
                ram_read_ring.push(r.value.as_arithmetic_or_trivial(party_id));
                ram_write_ring.push(r.value.as_arithmetic_or_trivial(party_id));
            }
            Rep3RAMAccess::Write(w) => {
                ram_read_ring.push(w.pre_value.as_arithmetic_or_trivial(party_id));
                ram_write_ring.push(w.post_value.as_arithmetic_or_trivial(party_id));
            }
            Rep3RAMAccess::NoOp => {
                let zero = Rep3Operand::Public(0);
                ram_read_ring.push(zero.as_arithmetic_or_trivial(party_id));
                ram_write_ring.push(zero.as_arithmetic_or_trivial(party_id));
            }
        }
    }

    let rs1_value = ring_to_field_a2b_many(&rs1_ring, io_ctx.main())?;
    let rs2_value = ring_to_field_a2b_many(&rs2_ring, io_ctx.main())?;
    let rd_write_value = ring_to_field_a2b_many(&rd_write_ring, io_ctx.main())?;
    let ram_read_value = ring_to_field_a2b_many(&ram_read_ring, io_ctx.main())?;
    let ram_write_value = ring_to_field_a2b_many(&ram_write_ring, io_ctx.main())?;

    state.prover_state.cycle_witness.pc = pc;
    state.prover_state.cycle_witness.unexpanded_pc = unexpanded_pc;
    state.prover_state.cycle_witness.imm = imm;
    state.prover_state.cycle_witness.rd_addr = rd_addr;
    state.prover_state.cycle_witness.rs1_addr = rs1_addr;
    state.prover_state.cycle_witness.rs2_addr = rs2_addr;
    state.prover_state.cycle_witness.ram_addr = ram_addr;
    state.prover_state.cycle_witness.flags_bits = flags_bits;
    state.prover_state.cycle_witness.advice = std::mem::take(&mut advice);

    state.prover_state.cycle_witness.lookup_output = lookup_output;
    state.prover_state.cycle_witness.rs1_value = rs1_value;
    state.prover_state.cycle_witness.rs2_value = rs2_value;
    state.prover_state.cycle_witness.rd_write_value = rd_write_value;
    state.prover_state.cycle_witness.ram_read_value = ram_read_value;
    state.prover_state.cycle_witness.ram_write_value = ram_write_value;

    Ok(())
}

// ── generate_witness_batch_rep3 ─────────────────────────────────────────────

/// Rep3 version of vanilla `CommittedPolynomial::generate_witness_batch`.
///
/// Populates witness data from a Rep3 trace and converts to `Rep3MultilinearPolynomial<F>`.
/// The trace must have been promoted to shares and arithmetic representations populated
/// before calling.
///
/// - Shared fields (register values) go through `ring_to_field_a2b_many` → `Shared(...)`
/// - Public fields (flags derived from opcode) → `Public(...)`
/// - Deferred fields (instruction_ra) are skipped or zeroed
#[tracing::instrument(skip_all, name = "Witness::generate_witness_rep3")]
pub fn generate_witness_batch_rep3<F, PCS, N>(
    polynomials: &[CommittedPolynomial],
    state: &mut StateManagerWorker<'_, F, PCS>,
    io_ctx: &mut IoContextPool<N>,
) -> eyre::Result<HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
    Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
{
    let party_id = state.party_id;
    let preprocessing: &JoltProverPreprocessing<F, PCS> = state.prover_state.preprocessing;
    let trace: &[Rep3Cycle] = &state.prover_state.trace;

    let lookup_outputs: Vec<Rep3PrimeFieldShare<F>> =
        state.prover_state.cycle_witness.lookup_output.clone();

    let mut ram_d = 0;
    let mut bytecode_d = 0;

    for poly in polynomials {
        match poly {
            CommittedPolynomial::BytecodeRa(i) => {
                bytecode_d = bytecode_d.max(*i + 1);
            }
            CommittedPolynomial::RamRa(i) => {
                ram_d = ram_d.max(*i + 1);
            }
            _ => {}
        }
    }
    let batch = Rep3WitnessData::new(trace.len(), ram_d, bytecode_d);

    // Precompute constants per cycle
    let bytecode_constants = if bytecode_d > 0 {
        let d = preprocessing.shared.bytecode.d;
        let log_K = preprocessing.shared.bytecode.code_size.log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let K_chunk = 1 << log_K_chunk;
        Some((d, log_K_chunk, K_chunk))
    } else {
        None
    };

    let dth_root_log = if ram_d > 0 {
        Some(DTH_ROOT_OF_K.log_2())
    } else {
        None
    };

    let instruction_ra_shifts: [usize; instruction_lookups::D] =
        array::from_fn(|i| instruction_lookups::LOG_K_CHUNK * (instruction_lookups::D - 1 - i));
    let batch_cell = Arc::new(SharedRep3WitnessData(UnsafeCell::new(batch)));

    // -- Parallel trace collection (mirrors vanilla par_iter) --
    // SAFETY: Each thread writes to a unique index of a pre-allocated vector
    //
    // Phase 1: Collect all data that doesn't require communication, plus
    //          FutureRep3Ring futures for instruction_ra indices.
    let index_futures: Vec<FutureRep3Ring<u128, Rep3RingShare<u128>>> = (0..trace.len())
        .into_par_iter()
        .map({
            let batch_cell = batch_cell.clone();
            move |i| {
                let cycle = &trace[i];
                let batch_ref = unsafe { &mut *batch_cell.0.get() };

                // Rd write: (rd_write_flag, pre, post)
                let (rd_write_flag, pre, post) = cycle.rd_write();
                batch_ref.rd_pre[i] = pre.as_arithmetic_or_trivial(party_id);
                batch_ref.rd_post[i] = post.as_arithmetic_or_trivial(party_id);

                let circuit_flags = cycle.instruction().circuit_flags();

                // WriteLookupOutputToRD (public)
                batch_ref.write_lookup_output_to_rd[i] = rd_write_flag
                    * (circuit_flags[CircuitFlags::WriteLookupOutputToRD as usize] as u8);

                // WritePCtoRD (public)
                batch_ref.write_pc_to_rd[i] =
                    rd_write_flag * (circuit_flags[CircuitFlags::Jump as usize] as u8);

                // ShouldJump (public)
                let is_jump = circuit_flags[CircuitFlags::Jump as usize] as u8;
                let is_next_noop = if i + 1 < trace.len() {
                    trace[i + 1].instruction().circuit_flags()[CircuitFlags::IsNoop as usize] as u8
                } else {
                    1 // Last cycle, treat as if next is NoOp
                };
                batch_ref.should_jump[i] = is_jump * (1 - is_next_noop);

                // RAM inc data
                if let Rep3RAMAccess::Write(w) = cycle.ram_access() {
                    batch_ref.ram_pre[i] = w.pre_value.as_arithmetic_u64();
                    batch_ref.ram_post[i] = w.post_value.as_arithmetic_u64();
                }

                // BytecodeRa indices
                if let Some((d, log_K_chunk, K_chunk)) = bytecode_constants {
                    let pc = cycle.get_pc(&preprocessing.shared.bytecode);

                    for j in 0..bytecode_d {
                        let index = (pc >> (log_K_chunk * (d - 1 - j))) % K_chunk;
                        batch_ref.bytecode_ra[j][i] = Some(index as u8);
                    }
                }

                if let Some(dth_log) = dth_root_log {
                    let address_opt = remap_address(
                        cycle.ram_access().address() as u64,
                        &preprocessing.shared.memory_layout,
                    );

                    for j in 0..ram_d {
                        let index = address_opt.map(|address| {
                            ((address as usize >> (dth_log * (ram_d - 1 - j))) % DTH_ROOT_OF_K)
                                as u8
                        });
                        batch_ref.ram_ra[j][i] = index;
                    }
                }

                // Return the lookup index future for batched fulfillment
                Rep3LookupQuery::<XLEN>::to_lookup_index(cycle, party_id)
            }
        })
        .collect();

    // Phase 2: Fulfill all pending index futures (batched A2B + MulA2B via io_ctx)
    let indices: Vec<Rep3RingShare<u128>> = {
        let _span = info_span!("fulfill_index_futures", count = index_futures.len()).entered();
        index_futures.fulfill_batched(io_ctx, |r, ()| r)?
    };

    // Phase 3: Chunk resolved indices into instruction_ra (parallel, no comms)
    // SAFETY: Each thread writes to a unique index i across the D arrays
    indices
        .par_iter()
        .enumerate()
        .for_each(|(i, lookup_index)| {
            let batch_ref = unsafe { &mut *batch_cell.0.get() };
            for j in 0..instruction_lookups::D {
                let k = (*lookup_index >> instruction_ra_shifts[j])
                    & RingElement(instruction_lookups::K_CHUNK as u128 - 1);
                batch_ref.instruction_ra[j][i] = Some(k.downcast());
            }
        });

    let mut batch = Arc::try_unwrap(batch_cell)
        .ok()
        .expect("Arc should have single owner")
        .0
        .into_inner();

    // -- Convert to polynomials --
    let mut results = HashMap::with_capacity(polynomials.len());
    let _span = info_span!("convert_to_polynomials", count = polynomials.len()).entered();

    // Instruction inputs are derived from the cached cycle witness (no extra casts).
    // Clone flags_bits to release borrow on cycle_witness, allowing assignment of rd_inc/ram_inc later.
    let flags_bits = state.prover_state.cycle_witness.flags_bits.clone();

    let mut left_input_field: Vec<Rep3PrimeFieldShare<F>> = Vec::new();
    let mut right_input_field: Vec<Rep3PrimeFieldShare<F>> = Vec::new();
    if polynomials.iter().any(|p| {
        matches!(
            p,
            CommittedPolynomial::LeftInstructionInput | CommittedPolynomial::RightInstructionInput
        )
    }) {
        let n = state.prover_state.cycle_witness.len();
        left_input_field = Vec::with_capacity(n);
        right_input_field = Vec::with_capacity(n);
        for t in 0..n {
            let (l, r) = state
                .prover_state
                .cycle_witness
                .row(t)
                .to_instruction_inputs(party_id);
            left_input_field.push(l);
            right_input_field.push(r);
        }
    }

    for poly in polynomials {
        match poly {
            CommittedPolynomial::LeftInstructionInput => {
                let field_shares = std::mem::take(&mut left_input_field);
                results.insert(*poly, Rep3MultilinearPolynomial::from(field_shares));
            }
            CommittedPolynomial::RightInstructionInput => {
                let field_shares = std::mem::take(&mut right_input_field);
                results.insert(*poly, Rep3MultilinearPolynomial::from(field_shares));
            }
            CommittedPolynomial::WriteLookupOutputToRD => {
                let coeffs = std::mem::take(&mut batch.write_lookup_output_to_rd);
                results.insert(
                    *poly,
                    Rep3MultilinearPolynomial::Public(MultilinearPolynomial::<F>::from(coeffs)),
                );
            }
            CommittedPolynomial::WritePCtoRD => {
                let coeffs = std::mem::take(&mut batch.write_pc_to_rd);
                results.insert(
                    *poly,
                    Rep3MultilinearPolynomial::Public(MultilinearPolynomial::<F>::from(coeffs)),
                );
            }
            CommittedPolynomial::ShouldBranch => {
                // should_branch[i] = lookup_output[i] * circuit_flags[Branch] (public scalar)
                let branch_mask = 1u32 << (CircuitFlags::Branch as usize);
                let should_branch: Vec<Rep3PrimeFieldShare<F>> = lookup_outputs
                    .iter()
                    .enumerate()
                    .map(|(t, output)| {
                        if (flags_bits[t] & branch_mask) != 0 {
                            *output
                        } else {
                            Rep3PrimeFieldShare::zero_share()
                        }
                    })
                    .collect();
                results.insert(*poly, Rep3MultilinearPolynomial::from(should_branch));
            }
            CommittedPolynomial::ShouldJump => {
                let coeffs = std::mem::take(&mut batch.should_jump);
                results.insert(
                    *poly,
                    Rep3MultilinearPolynomial::Public(MultilinearPolynomial::<F>::from(coeffs)),
                );
            }
            CommittedPolynomial::RdInc => {
                // rd_inc = rd_post - rd_pre in the field (MPC: subtract after conversion)
                let pre_field: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&batch.rd_pre, io_ctx.main())?;
                let post_field: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&batch.rd_post, io_ctx.main())?;
                let inc: Vec<Rep3PrimeFieldShare<F>> = post_field
                    .into_iter()
                    .zip(pre_field)
                    .map(|(post, pre)| post - pre)
                    .collect();
                // Store as Rep3DensePolynomial on cycle_witness for stage2 provers.
                // Arc internally — no data duplication.
                let dense = Rep3DensePolynomial::new(inc);
                state.prover_state.cycle_witness.rd_inc = Some(dense.clone());
                results.insert(*poly, Rep3MultilinearPolynomial::shared(dense));
            }
            CommittedPolynomial::RamInc => {
                // ram_inc = post_value - pre_value in the field (MPC: subtract after conversion)
                let pre_field: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&batch.ram_pre, io_ctx.main())?;
                let post_field: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&batch.ram_post, io_ctx.main())?;
                let inc: Vec<Rep3PrimeFieldShare<F>> = post_field
                    .into_iter()
                    .zip(pre_field)
                    .map(|(post, pre)| post - pre)
                    .collect();
                // Store as Rep3DensePolynomial on cycle_witness for stage2 provers.
                let dense = Rep3DensePolynomial::new(inc);
                state.prover_state.cycle_witness.ram_inc = Some(dense.clone());
                results.insert(*poly, Rep3MultilinearPolynomial::shared(dense));
            }
            CommittedPolynomial::InstructionRa(i) => {
                if *i < instruction_lookups::D {
                    let indices = std::mem::take(&mut batch.instruction_ra[*i]);
                    let one_hot = Rep3OneHotPolynomial::<F>::from_indices(
                        indices,
                        instruction_lookups::K_CHUNK,
                        io_ctx.main(),
                    )?;
                    results.insert(*poly, Rep3MultilinearPolynomial::shared_one_hot(one_hot));
                }
            }
            CommittedPolynomial::BytecodeRa(i) => {
                if *i < bytecode_d {
                    let indices = std::mem::take(&mut batch.bytecode_ra[*i]);
                    let d = preprocessing.shared.bytecode.d;
                    let log_K = preprocessing.shared.bytecode.code_size.log_2();
                    let log_K_chunk = log_K.div_ceil(d);
                    let K_chunk = 1 << log_K_chunk;
                    let one_hot = OneHotPolynomial::from_indices(indices, K_chunk);
                    results.insert(
                        *poly,
                        Rep3MultilinearPolynomial::Public(MultilinearPolynomial::OneHot(one_hot)),
                    );
                }
            }
            CommittedPolynomial::RamRa(i) => {
                if *i < ram_d {
                    let indices = std::mem::take(&mut batch.ram_ra[*i]);
                    let one_hot = OneHotPolynomial::from_indices(indices, DTH_ROOT_OF_K);
                    results.insert(
                        *poly,
                        Rep3MultilinearPolynomial::Public(MultilinearPolynomial::OneHot(one_hot)),
                    );
                }
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use ark_bn254::Fr;
    use ark_std::test_rng;
    use tracing::info;

    use crate::host::program::Rep3Program;
    use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
    use crate::poly::{Rep3MultilinearPolynomial, Rep3SharedPoly};
    use crate::utils::compute_ram_k;
    use crate::utils::test_utils::{check_poly, run_rep3_test};
    use crate::utils::tracing::init_tracing;
    use crate::zkvm::instruction::{populate_operands_casts, Rep3Cycle, Rep3Operand};
    use jolt_core::host::Program;
    use jolt_core::poly::commitment::dory::DoryGlobals;
    use jolt_core::poly::commitment::mock::MockCommitScheme;
    use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
    use jolt_core::zkvm::bytecode::BytecodePreprocessing;
    use jolt_core::zkvm::ram::RAMPreprocessing;
    use jolt_core::zkvm::witness::{
        compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
    };
    use jolt_core::zkvm::{JoltProverPreprocessing, JoltSharedPreprocessing};
    use tracer::instruction::Cycle;

    type F = Fr;
    type PCS = MockCommitScheme<F>;

    #[test]
    #[ignore = "requires QUIC network sockets (not available in sandboxed test env)"]
    fn witness_batch_rep3() {
        let _tracing_guard = init_tracing("witness_test.json", Path::new("/tmp/co-jolt2-traces"));

        // 1. Build and trace the fibonacci program
        let mut program = Program::new("fibonacci-guest");
        let elf_path = "/tmp/jolt-guest-targets/fibonacci-guest-/riscv64imac-unknown-none-elf/release/fibonacci-guest";
        program.elf = Some(PathBuf::from(elf_path));
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, memory_init, _) = program.decode();

        // 2. Generate trace and shares
        let mut rng = test_rng();
        let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);

        // Also get a vanilla trace for comparison
        let (mut vanilla_trace, _memory, io_device) = program.trace(&inputs, &[], &[]);

        // Pad traces to next power of 2 (mirrors StateManager / DAG init).
        // The +1 accounts for the implicit PC termination cycle.
        let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
        info!(raw_len = vanilla_trace.len(), padded_len, "padding traces");
        vanilla_trace.resize(padded_len, Cycle::NoOp);
        for (trace, _, _) in shares.iter_mut() {
            trace.resize(padded_len, Rep3Cycle::NoOp);
        }

        // 3. Build preprocessing (shared between all parties + vanilla)
        let shared = JoltSharedPreprocessing {
            memory_layout: io_device.memory_layout.clone(),
            bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
            ram: RAMPreprocessing::preprocess(memory_init.clone()),
        };
        let preprocessing: JoltProverPreprocessing<F, PCS> = JoltProverPreprocessing {
            generators: (),
            shared: shared.clone(),
        };

        // 4. Determine which polynomials to test.
        let ram_K = compute_ram_k(&vanilla_trace, &preprocessing.shared);
        let bytecode_d = preprocessing.shared.bytecode.d;
        let ram_d = compute_d_parameter(ram_K);
        let _guard = AllCommittedPolynomials::initialize(ram_d, bytecode_d);
        // DoryGlobals needed for vanilla OneHotPolynomial::from_indices (debug_assert)
        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);

        let all_polys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();

        info!(total = all_polys.len(), "polynomial counts");

        // 5. Run vanilla witness generation (including one-hot polys)
        info!("running vanilla witness generation");
        let vanilla_results =
            CommittedPolynomial::generate_witness_batch(&all_polys, &preprocessing, &vanilla_trace);
        info!(
            count = vanilla_results.len(),
            "vanilla witness generation complete"
        );

        // 6. Run MPC witness generation on 3 parties
        let preprocessing_arc = Arc::new(preprocessing);
        let io_device_arc = Arc::new(io_device);
        let base_port: u16 = 14200;

        info!("launching 3-party MPC witness generation");
        let mpc_results: [HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>; 3] =
            run_rep3_test(
                base_port,
                4,
                |party_idx| {
                    let (trace, memory, _io) = shares[party_idx].clone();
                    let preprocessing = Arc::clone(&preprocessing_arc);
                    let io_device = Arc::clone(&io_device_arc);
                    (
                        trace,
                        memory,
                        io_device,
                        preprocessing,
                        ram_K,
                        all_polys.clone(),
                    )
                },
                |input, mut io_ctx| {
                    let (mut trace, memory, io_device, preprocessing, ram_k, polys) = input;
                    let party = io_ctx.party_id();

                    info!(?party, "populate_operands_casts start");
                    populate_operands_casts(&mut trace, io_ctx.main())?;
                    info!(?party, "populate_operands_casts done");

                    // Verify arithmetic shares are populated
                    let mut unpopulated = 0usize;
                    let mut total_shared = 0usize;
                    for cycle in trace.iter_mut() {
                        for op in cycle.shared_operands_mut() {
                            if let Rep3Operand::Shared { arithmetic, .. } = op {
                                total_shared += 1;
                                if arithmetic.is_none() {
                                    unpopulated += 1;
                                }
                            }
                        }
                    }
                    info!(?party, total_shared, unpopulated, "operand check");
                    assert_eq!(unpopulated, 0, "unpopulated arithmetic shares remain");

                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        (*io_device).clone(),
                        memory,
                        io_ctx.party_id(),
                        ram_k,
                    );

                    info!(?party, "generate_witness_batch_rep3 start");
                    let results =
                        generate_witness_batch_rep3::<F, PCS, _>(&polys, &mut state, &mut io_ctx)?;
                    info!(
                        ?party,
                        count = results.len(),
                        "generate_witness_batch_rep3 done"
                    );
                    Ok(results)
                },
            );

        info!("MPC witness generation complete, reconstructing");

        // 7. Reconstruct and compare
        for poly_key in &all_polys {
            let vanilla_poly = match vanilla_results.get(poly_key) {
                Some(p) => p,
                None => continue,
            };

            let share_polys: Vec<Rep3MultilinearPolynomial<F>> = (0..3)
                .map(|i| {
                    mpc_results[i]
                        .get(poly_key)
                        .unwrap_or_else(|| panic!("party {i} missing poly {poly_key:?}"))
                        .clone()
                })
                .collect();

            match &share_polys[0] {
                Rep3MultilinearPolynomial::Public(MultilinearPolynomial::OneHot(mpc_ohp)) => {
                    let vanilla_indices = match vanilla_poly {
                        MultilinearPolynomial::OneHot(ohp) => &*ohp.nonzero_indices,
                        _ => panic!("expected vanilla OneHot for {poly_key:?}"),
                    };
                    assert_eq!(
                        &*mpc_ohp.nonzero_indices, vanilla_indices,
                        "{poly_key:?} public OneHot indices mismatch"
                    );
                    info!(?poly_key, "public one-hot indices match");
                }
                Rep3MultilinearPolynomial::Public(pub_poly) => {
                    check_poly(pub_poly, vanilla_poly, &format!("{poly_key:?} (public)"));
                }
                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(_)) => {
                    let reconstructed = Rep3MultilinearPolynomial::combine_shares(share_polys);
                    check_poly(
                        &reconstructed,
                        vanilla_poly,
                        &format!("{poly_key:?} (shared dense)"),
                    );
                }
                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(_)) => {
                    let ohps: Vec<&Rep3OneHotPolynomial<F>> = share_polys
                        .iter()
                        .map(|p| match p {
                            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(ohp)) => ohp,
                            _ => unreachable!(),
                        })
                        .collect();

                    let reconstructed_indices =
                        Rep3OneHotPolynomial::reconstruct_indices([ohps[0], ohps[1], ohps[2]]);

                    let vanilla_indices = match vanilla_poly {
                        MultilinearPolynomial::OneHot(ohp) => &*ohp.nonzero_indices,
                        _ => panic!(
                            "expected vanilla OneHot for {poly_key:?}, got {:?}",
                            std::mem::discriminant(vanilla_poly)
                        ),
                    };

                    assert_eq!(
                        reconstructed_indices.len(),
                        vanilla_indices.len(),
                        "length mismatch for {poly_key:?}"
                    );
                    for (j, (mpc, vanilla)) in reconstructed_indices
                        .iter()
                        .zip(vanilla_indices.iter())
                        .enumerate()
                    {
                        assert_eq!(
                            mpc, vanilla,
                            "{poly_key:?} index mismatch at cycle {j}: mpc={mpc:?} vanilla={vanilla:?}"
                        );
                    }
                    info!(?poly_key, "one-hot indices match");
                }
            }
        }

        info!("all polynomials match!");
    }
}
