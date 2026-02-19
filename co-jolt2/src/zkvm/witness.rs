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
use mpc_core::protocols::rep3::network::{
    IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3_ring::casts::ring_to_field_a2b_many;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use snarks_core::math::Math;
use tracing::{info, info_span};

use crate::field::JoltField;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::Rep3MultilinearPolynomial;
use crate::utils::future_ring::{FutureRep3Ring, Rep3RingFutureExt};
use crate::zkvm::instruction::Rep3LookupQuery;

use super::instruction::{Rep3Cycle, Rep3RAMAccess};

// ── Rep3WitnessData ─────────────────────────────────────────────────────────

/// Mirrors vanilla `WitnessData` but uses `Rep3RingShare` for shared fields.
struct Rep3WitnessData {
    left_instruction_input: Vec<Rep3RingShare<u64>>,
    right_instruction_input: Vec<Rep3RingShare<u128>>,
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
            left_instruction_input: vec![Rep3RingShare::default(); trace_len],
            right_instruction_input: vec![Rep3RingShare::default(); trace_len],
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
    Standard: Distribution<u32>,
{
    let mut output_futures: Vec<FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>> =
        vec![FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share()); trace.len()];

    // Group by instruction type, skipping non-lookup cycles (NoOp, INLINE)
    let ops_by_instruction: Vec<(
        Vec<&Rep3Cycle>,
        Vec<&mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
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
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: &[Rep3Cycle],
    io_ctx: &mut IoContextPool<N>,
) -> eyre::Result<HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
    Standard: Distribution<u64> + Distribution<u8> + Distribution<u128>,
{
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
    let party_id = io_ctx.party_id();
    let batch_cell = Arc::new(SharedRep3WitnessData(UnsafeCell::new(batch)));

    // -- Parallel trace collection (mirrors vanilla par_iter) --
    // SAFETY: Each thread writes to a unique index of a pre-allocated vector
    //
    // Phase 1: Collect all data that doesn't require communication, plus
    //          FutureRep3Ring futures for instruction_ra indices.
    eprintln!(
        "[witness] phase1: parallel trace collection start | len: {}",
        trace.len()
    );
    let index_futures: Vec<FutureRep3Ring<u128, Rep3RingShare<u128>>> = (0..trace.len())
        .into_par_iter()
        .map({
            let batch_cell = batch_cell.clone();
            move |i| {
                let cycle = &trace[i];
                let batch_ref = unsafe { &mut *batch_cell.0.get() };

                // Store raw instruction inputs (mirrors vanilla's to_instruction_inputs)
                let (left_op, right_op) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(cycle);
                batch_ref.left_instruction_input[i] =
                    left_op.as_arithmetic_or_trivial::<u64>(party_id);
                batch_ref.right_instruction_input[i] =
                    right_op.as_arithmetic_or_trivial::<u128>(party_id);

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

    eprintln!("[witness] phase1: done | futures: {}", index_futures.len());

    // Phase 2: Fulfill all pending index futures (batched A2B + MulA2B via io_ctx)
    let indices: Vec<Rep3RingShare<u128>> = {
        let _span = info_span!("fulfill_index_futures", count = index_futures.len()).entered();
        index_futures.fulfill_batched(io_ctx, |r, ()| r)?
    };

    eprintln!(
        "[witness] phase2: fulfill done | indices: {}",
        indices.len()
    );

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

    eprintln!("[witness] phase3: chunk indices done");

    // Phase 4: Compute lookup outputs (batched per instruction type)
    eprintln!("[witness] phase4: compute_lookup_outputs start");
    let lookup_outputs: Vec<Rep3PrimeFieldShare<F>> = compute_lookup_outputs(trace, io_ctx)?;
    eprintln!(
        "[witness] phase4: compute_lookup_outputs done | len: {}",
        lookup_outputs.len()
    );

    // -- Convert to polynomials --
    eprintln!(
        "[witness] phase5: convert_to_polynomials start | count: {}",
        polynomials.len()
    );
    let mut results = HashMap::with_capacity(polynomials.len());
    let _span = info_span!("convert_to_polynomials", count = polynomials.len()).entered();

    for poly in polynomials {
        match poly {
            CommittedPolynomial::LeftInstructionInput => {
                let coeffs = std::mem::take(&mut batch.left_instruction_input);
                let field_shares: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&coeffs, io_ctx.main())?;
                results.insert(*poly, Rep3MultilinearPolynomial::from(field_shares));
            }
            CommittedPolynomial::RightInstructionInput => {
                let coeffs = std::mem::take(&mut batch.right_instruction_input);
                let field_shares: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&coeffs, io_ctx.main())?;
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
                let should_branch: Vec<Rep3PrimeFieldShare<F>> = lookup_outputs
                    .iter()
                    .zip(trace.iter())
                    .map(|(output, cycle)| {
                        let is_branch =
                            cycle.instruction().circuit_flags()[CircuitFlags::Branch as usize];
                        if is_branch {
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
                results.insert(*poly, Rep3MultilinearPolynomial::from(inc));
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
                results.insert(*poly, Rep3MultilinearPolynomial::from(inc));
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
