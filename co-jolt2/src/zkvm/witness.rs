use std::array;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use jolt2_common::constants::XLEN;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::zkvm::instruction::{
    CircuitFlags, InstructionFlags, InstructionLookup, InterleavedBitsMarker,
};
use jolt_core::zkvm::lookup_table::LookupTables;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::{CommittedPolynomial, DTH_ROOT_OF_K};
use jolt_core::zkvm::{instruction_lookups, JoltProverPreprocessing};
use mpc_core::protocols::rep3::network::{IoContext, IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{arithmetic::promote_to_trivial_share, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use mpc_core::protocols::rep3_ring::preprocessing::edabits;
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
use crate::utils::types::Either;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::instruction::{populate_operands_casts, Rep3LookupQuery, Rep3Operand};
use crate::utils::memory::maybe_purge_jemalloc;

use super::instruction::{Rep3Cycle, Rep3RAMAccess};

// ── ring_to_field cast helpers ──────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
enum SparseCastCol {
    Rs1 = 0,
    Rs2 = 1,
    RdWrite = 2,
    RamRead = 3,
    RamWrite = 4,
}

#[derive(Clone, Debug)]
struct SparseCastJob {
    col: SparseCastCol,
    row: usize,
    share: Rep3RingShare<u64>,
}

struct SharedSparseFieldCols<F: JoltField> {
    rs1: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    rs2: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    rd_write: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    ram_read: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    ram_write: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
}

unsafe impl<F: JoltField> Sync for SharedSparseFieldCols<F> {}

#[tracing::instrument(
    skip_all,
    name = "fill_field_from_operands_sparse_u64",
    fields(jobs = jobs.len())
)]
fn fill_field_from_operands_sparse_u64<F, N>(
    io_ctx: &mut IoContextPool<N>,
    jobs: Vec<SparseCastJob>,
    out: Arc<SharedSparseFieldCols<F>>,
    preproc: &mut PreprocessingPool<F>,
) -> eyre::Result<()>
where
    F: JoltField,
    N: Rep3NetworkWorker,
    Standard: Distribution<u64>,
{
    if jobs.is_empty() {
        return Ok(());
    }

    let chunk_size: usize = std::env::var("B2A_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);

    for chunk in jobs.chunks(chunk_size) {
        let n = chunk.len();
        let mut shares: Vec<Rep3RingShare<u64>> = Vec::with_capacity(n);
        let mut targets: Vec<(SparseCastCol, usize)> = Vec::with_capacity(n);
        for job in chunk {
            shares.push(job.share);
            targets.push((job.col, job.row));
        }

        let batch = preproc.take_edabits::<u64>(n);
        let casted = io_ctx.par_chunks_preproc(shares, batch, None, |xs, batch, ctx| {
            edabits::ring_to_field_b2a_many::<u64, F, _>(&xs, &batch, ctx)
        })?;

        debug_assert_eq!(casted.len(), targets.len());
        for (value, (col, row)) in casted.into_iter().zip(targets.into_iter()) {
            unsafe {
                match col {
                    SparseCastCol::Rs1 => (&mut *out.rs1.get())[row] = value,
                    SparseCastCol::Rs2 => (&mut *out.rs2.get())[row] = value,
                    SparseCastCol::RdWrite => (&mut *out.rd_write.get())[row] = value,
                    SparseCastCol::RamRead => (&mut *out.ram_read.get())[row] = value,
                    SparseCastCol::RamWrite => (&mut *out.ram_write.get())[row] = value,
                }
            }
        }
    }

    Ok(())
}

/// Compute the plaintext lookup index for cycles with fully-public operands.
///
/// Returns `Some(plain_u128)` for instructions where both operands are public
/// (control-only: LUI, AUIPC, JAL, VirtualPow2*, VirtualShiftRightBitmask*).
/// These use the "add" path: index = left + right (u128 arithmetic).
///
/// Returns `None` for all other instructions (operands may be secret-shared).
fn compute_public_index(cycle: &Rep3Cycle) -> Option<u128> {
    let try_public_add = |inputs: (Rep3Operand, Rep3Operand)| -> Option<u128> {
        let (l, r) = inputs;
        let l_val = match l {
            Rep3Operand::Public(v)
            | Rep3Operand::Shared {
                public: Some(v), ..
            } => v,
            _ => return None,
        };
        let r_val = match r {
            Rep3Operand::Public(v)
            | Rep3Operand::Shared {
                public: Some(v), ..
            } => v,
            _ => return None,
        };
        Some((l_val as u128).wrapping_add(r_val as u128))
    };

    match cycle {
        Rep3Cycle::LUI(c) => try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::AUIPC(c) => try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::JAL(c) => try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::VirtualPow2(c) => {
            try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualPow2I(c) => {
            try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualPow2W(c) => {
            try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualPow2IW(c) => {
            try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualShiftRightBitmask(c) => {
            try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualShiftRightBitmaskI(c) => {
            try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        _ => None,
    }
}

/// Extract the public right-operand value for instructions with a public right operand.
///
/// Returns `Some(value_u64)` for:
/// - Shift/rotate: VirtualSRA/SRL/SRAI/SRLI (rs2 is public bitmask),
///   VirtualROTRI/ROTRIW (immediate bitmask)
/// - Immediate ALU: ADDI, ANDI, ORI, XORI, SLTI, SLTIU, VirtualMULI
///
/// Used by ReadRaf to exploit public right operands:
/// - Skip MPC for And/Xor/Or/shift suffixes when right operand is known
/// - Compute SignExtension suffix locally
fn compute_right_operand_public(cycle: &Rep3Cycle) -> Option<u64> {
    let extract_right = |inputs: (Rep3Operand, Rep3Operand)| -> Option<u64> {
        match inputs.1 {
            Rep3Operand::Public(v)
            | Rep3Operand::Shared {
                public: Some(v), ..
            } => Some(v as u64),
            _ => None,
        }
    };
    match cycle {
        // Shift/rotate — rs2 is public bitmask
        Rep3Cycle::VirtualSRA(c) => {
            extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualSRL(c) => {
            extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualSRAI(c) => {
            extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualSRLI(c) => {
            extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualROTRI(c) => {
            extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        Rep3Cycle::VirtualROTRIW(c) => {
            extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        // Immediate ALU — right operand is the immediate
        Rep3Cycle::ADDI(c) => extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::ANDI(c) => extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::ORI(c) => extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::XORI(c) => extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::SLTI(c) => extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::SLTIU(c) => extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c)),
        Rep3Cycle::VirtualMULI(c) => {
            extract_right(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        _ => None,
    }
}

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

    let _span = tracing::trace_span!("group_by_table").entered();

    // Parallel pre-pass: compute discriminant per cycle for lookup-enabled instructions.
    // We still build the final groups deterministically in trace order.
    let discriminants: Vec<Option<mem::Discriminant<Rep3Cycle>>> = trace
        .par_iter()
        .map(|cycle| {
            cycle
                .lookup_table()
                .is_some()
                .then_some(mem::discriminant(cycle))
        })
        .collect();

    // Deterministic, first-seen grouping by instruction discriminant (reduces tiny batches).
    let mut disc_to_group: HashMap<mem::Discriminant<Rep3Cycle>, usize> = HashMap::new();
    let mut group_ids: Vec<Option<usize>> = vec![None; trace.len()];
    let mut num_groups = 0usize;
    for (i, disc) in discriminants.into_iter().enumerate() {
        let Some(disc) = disc else { continue };
        let gid = *disc_to_group.entry(disc).or_insert_with(|| {
            let gid = num_groups;
            num_groups += 1;
            gid
        });
        group_ids[i] = Some(gid);
    }

    let mut ops_by_instruction: Vec<(
        Vec<&Rep3Cycle>,
        Vec<&mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    )> = (0..num_groups).map(|_| (Vec::new(), Vec::new())).collect();

    // TODO: parallelize
    for (i, (cycle, out)) in trace.iter().zip(output_futures.iter_mut()).enumerate() {
        let Some(gid) = group_ids[i] else { continue };
        ops_by_instruction[gid].0.push(cycle);
        ops_by_instruction[gid].1.push(out);
    }
    drop(_span);

    // Process each instruction group via par_chunks
    let _span = tracing::info_span!("to_lookup_output_batched", num_groups = num_groups).entered();
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
    drop(_span);

    // Fulfill all pending futures (batched casts via io_ctx)
    let _span = tracing::info_span!("fulfill_batched").entered();
    output_futures.fulfill_batched(io_ctx, |res, ()| res)
}

// ── populate_cycle_witness_rep3 ─────────────────────────────────────────────

/// Populate `state.prover_state.cycle_witness` from the (ring-shared) trace.
///
/// This cache is the field-domain, per-cycle witness representation used by
/// Stage 1 Spartan and later stages, allowing the ring-shared trace to be dropped.
#[tracing::instrument(skip_all, name = "populate_cycle_witness")]
pub fn populate_cycle_witness_rep3<F, PCS, N>(
    state: &mut StateManagerWorker<'_, F, PCS>,
    io_ctx: &mut IoContextPool<N>,
    preproc: &mut PreprocessingPool<F>,
) -> eyre::Result<()>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
    Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
{
    let party_id = state.party_id;
    let preprocessing = state.prover_state.preprocessing;
    let trace = state.trace_mut();
    eyre::ensure!(
        trace.len().is_power_of_two(),
        "trace length must be power-of-two"
    );

    // Ensure all shared operands have arithmetic representations (batched).
    populate_operands_casts(trace, io_ctx.main())?;

    // Compute lookup outputs (batched, MPC).
    let lookup_output = compute_lookup_outputs::<F, N>(trace, io_ctx)?;

    let n = trace.len();

    // Meta (AoS) + public stage-specific columns
    let mut meta: Vec<crate::zkvm::dag::witness::CycleMeta> = Vec::with_capacity(n);
    let mut unexpanded_pc: Vec<u64> = Vec::with_capacity(n);
    let mut flags_bits: Vec<u32> = Vec::with_capacity(n);

    let mut imm: Vec<i128> = Vec::with_capacity(n);
    let mut advice: Vec<u64> = vec![0; n];
    let mut lookup_tables: Vec<Option<LookupTables<XLEN>>> = Vec::with_capacity(n);
    let mut is_interleaved_operands: Vec<bool> = Vec::with_capacity(n);
    let mut right_operand_public_mask: Vec<Option<u64>> = Vec::with_capacity(n);

    // Shared columns (ring→field), populated sparsely:
    // - public/trivial values are injected directly into field trivial shares
    // - secret values are batched into EdaBits B2A (`ring_to_field_b2a_many`) via IoContextPool
    let shared_cols = Arc::new(SharedSparseFieldCols::<F> {
        rs1: UnsafeCell::new(vec![Rep3PrimeFieldShare::zero_share(); n]),
        rs2: UnsafeCell::new(vec![Rep3PrimeFieldShare::zero_share(); n]),
        rd_write: UnsafeCell::new(vec![Rep3PrimeFieldShare::zero_share(); n]),
        ram_read: UnsafeCell::new(vec![Rep3PrimeFieldShare::zero_share(); n]),
        ram_write: UnsafeCell::new(vec![Rep3PrimeFieldShare::zero_share(); n]),
    });
    let mut cast_jobs: Vec<SparseCastJob> = Vec::new();
    cast_jobs.reserve(n * 5);

    let mut maybe_push = |col: SparseCastCol, row: usize, op: &Rep3Operand| match op {
        Rep3Operand::Public(v)
        | Rep3Operand::Shared {
            public: Some(v), ..
        } => {
            let share = promote_to_trivial_share(party_id, F::from_u64(*v));
            unsafe {
                match col {
                    SparseCastCol::Rs1 => (&mut *shared_cols.rs1.get())[row] = share,
                    SparseCastCol::Rs2 => (&mut *shared_cols.rs2.get())[row] = share,
                    SparseCastCol::RdWrite => (&mut *shared_cols.rd_write.get())[row] = share,
                    SparseCastCol::RamRead => (&mut *shared_cols.ram_read.get())[row] = share,
                    SparseCastCol::RamWrite => (&mut *shared_cols.ram_write.get())[row] = share,
                }
            }
        }
        Rep3Operand::Shared { .. } => {
            cast_jobs.push(SparseCastJob {
                col,
                row,
                share: op.as_binary(),
            });
        }
    };

    for (t, cycle) in trace.iter().enumerate() {
        let norm = cycle.instruction().normalize();
        let circuit_flags = cycle.instruction().circuit_flags();

        let pc_index = cycle.get_pc(&preprocessing.shared.bytecode) as u64;
        unexpanded_pc.push(norm.address as u64);
        imm.push(norm.operands.imm);

        let (rs1_i, rs1_v) = cycle.rs1_read();
        let (rs2_i, rs2_v) = cycle.rs2_read();
        let (rd_i, _rd_pre, rd_post) = cycle.rd_write();

        let ram_addr = cycle.ram_access().address();

        meta.push(crate::zkvm::dag::witness::CycleMeta {
            pc_index,
            ram_addr,
            rd_addr: rd_i,
            rs1_addr: rs1_i,
            rs2_addr: rs2_i,
        });

        // Pack circuit flags into a u32 bitmask
        let mut mask = 0u32;
        for (i, flag) in circuit_flags.iter().enumerate() {
            if *flag {
                mask |= 1u32 << i;
            }
        }
        flags_bits.push(mask);

        // Lookup table and interleaved operand flag (public, derived from opcode).
        lookup_tables.push(InstructionLookup::<XLEN>::lookup_table(cycle));
        is_interleaved_operands.push(circuit_flags.is_interleaved_operands());
        right_operand_public_mask.push(compute_right_operand_public(cycle));

        // Advice value (only meaningful for VirtualAdvice).
        if circuit_flags[CircuitFlags::Advice as usize] {
            if let Rep3Cycle::VirtualAdvice(c) = cycle {
                advice[t] = c.instruction.advice;
            }
        }

        maybe_push(SparseCastCol::Rs1, t, &rs1_v);
        maybe_push(SparseCastCol::Rs2, t, &rs2_v);
        maybe_push(SparseCastCol::RdWrite, t, &rd_post);

        match cycle.ram_access() {
            Rep3RAMAccess::Read(r) => {
                // Match vanilla: for reads, RamReadValue == RamWriteValue == r.value
                maybe_push(SparseCastCol::RamRead, t, &r.value);
                maybe_push(SparseCastCol::RamWrite, t, &r.value);
            }
            Rep3RAMAccess::Write(w) => {
                maybe_push(SparseCastCol::RamRead, t, &w.pre_value);
                maybe_push(SparseCastCol::RamWrite, t, &w.post_value);
            }
            Rep3RAMAccess::NoOp => {
                let z = promote_to_trivial_share(party_id, F::zero());
                unsafe {
                    (&mut *shared_cols.ram_read.get())[t] = z;
                    (&mut *shared_cols.ram_write.get())[t] = z;
                }
            }
        }
    }

    fill_field_from_operands_sparse_u64::<F, N>(
        io_ctx,
        cast_jobs,
        Arc::clone(&shared_cols),
        preproc,
    )?;

    let _span = tracing::trace_span!("init_rep3_witnesses").entered();
    let shared_cols = Arc::try_unwrap(shared_cols)
        .ok()
        .expect("shared cols Arc should have single owner");
    let rs1_value = shared_cols.rs1.into_inner();
    let rs2_value = shared_cols.rs2.into_inner();
    let rd_write_value = shared_cols.rd_write.into_inner();
    let ram_read_value = shared_cols.ram_read.into_inner();
    let ram_write_value = shared_cols.ram_write.into_inner();

    let cw = &mut state.prover_state.cycle_witness;
    cw.set_len(n);
    cw.set_meta(meta);
    cw.set_stage1(
        imm,
        std::mem::take(&mut advice),
        lookup_output,
        rs1_value,
        rs2_value,
        rd_write_value,
        ram_read_value,
        ram_write_value,
    );
    cw.update_stage3(crate::zkvm::dag::witness::Stage3Update {
        pc_sumcheck: Some((unexpanded_pc, flags_bits)),
        read_raf_tables_and_masks: Some((
            lookup_tables,
            is_interleaved_operands,
            right_operand_public_mask,
        )),
        read_raf_lookup_indices: None,
        product_inputs: None,
    });

    // Precompute instruction inputs for Spartan Product virtualization (stage 3),
    // so we can drop the large stage1 witness vectors after Spartan stage 1.
    let mut left: Vec<Rep3PrimeFieldShare<F>> = Vec::with_capacity(n);
    let mut right: Vec<Rep3PrimeFieldShare<F>> = Vec::with_capacity(n);
    for t in 0..n {
        let (l, r) = cw.row_stage1(t).to_instruction_inputs(party_id);
        left.push(l);
        right.push(r);
    }
    cw.update_stage3(crate::zkvm::dag::witness::Stage3Update {
        pc_sumcheck: None,
        read_raf_tables_and_masks: None,
        read_raf_lookup_indices: None,
        product_inputs: Some(crate::zkvm::dag::witness::ProductInputs { left, right }),
    });

    #[cfg(debug_assertions)]
    cw.sanity_check_lengths();

    Ok(())
}

// ── generate_witness_batch_rep3 ─────────────────────────────────────────────

/// Rep3 version of vanilla `CommittedPolynomial::generate_witness_batch`.
///
/// Populates witness data from a Rep3 trace and converts to `Rep3MultilinearPolynomial<F>`.
/// The trace must have been promoted to shares and arithmetic representations populated
/// before calling.
///
/// - Shared fields (register values) go through EdaBits B2A (`ring_to_field_b2a_many`) → `Shared(...)`
/// - Public fields (flags derived from opcode) → `Public(...)`
/// - Deferred fields (instruction_ra) are skipped or zeroed
#[tracing::instrument(skip_all, name = "witness_batch_generate")]
pub fn generate_witness_batch_rep3<F, PCS, N>(
    polynomials: &[CommittedPolynomial],
    state: &mut StateManagerWorker<'_, F, PCS>,
    io_ctx: &mut IoContextPool<N>,
    preproc: &mut PreprocessingPool<F>,
) -> eyre::Result<HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
    Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
{
    let fulfill_index_chunk: usize = std::env::var("FULFILL_INDEX_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16 * 1024);
    let inc_b2a_chunk: usize = std::env::var("INC_B2A_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8 * 1024);
    let inc_b2a_max_forks: usize = std::env::var("INC_B2A_MAX_FORKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let party_id = state.party_id;
    let preprocessing: &JoltProverPreprocessing<F, PCS> = state.prover_state.preprocessing;
    let trace: &[Rep3Cycle] = state.trace_ref();

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
                batch_ref.rd_pre[i] = pre.as_binary_or_trivial(party_id);
                batch_ref.rd_post[i] = post.as_binary_or_trivial(party_id);

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
                    batch_ref.ram_pre[i] = w.pre_value.as_binary_or_trivial(party_id);
                    batch_ref.ram_post[i] = w.post_value.as_binary_or_trivial(party_id);
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
        let total = index_futures.len();
        let _span = info_span!("fulfill_index_futures", count = total).entered();
        let mut out: Vec<Rep3RingShare<u128>> = Vec::with_capacity(total);
        let mut iter = index_futures.into_iter();
        let mut chunk_id: usize = 0;
        loop {
            let mut chunk: Vec<FutureRep3Ring<u128, Rep3RingShare<u128>>> =
                Vec::with_capacity(fulfill_index_chunk);
            for _ in 0..fulfill_index_chunk {
                let Some(f) = iter.next() else { break };
                chunk.push(f);
            }
            if chunk.is_empty() {
                break;
            }
            let _chunk_span = info_span!(
                "fulfill_index_futures_chunk",
                chunk_id,
                chunk_len = chunk.len()
            )
            .entered();
            let resolved: Vec<Rep3RingShare<u128>> = chunk.fulfill_batched(io_ctx, |r, ()| r)?;
            drop(_chunk_span);
            out.extend(resolved);
            chunk_id += 1;
        }
        drop(_span);
        out
    };

    // Phase 3: Chunk resolved indices into instruction_ra (parallel, no comms)
    // SAFETY: Each thread writes to a unique index i across the D arrays.
    // NoOp padding cycles are left as None (the default) so that
    // masked_indices_c[j] = None downstream, excluding them from non_noop_cycles
    // and avoiding redundant B2A / edaBit consumption on padding.
    indices
        .par_iter()
        .enumerate()
        .for_each(|(i, lookup_index)| {
            if matches!(trace[i], Rep3Cycle::NoOp) {
                return;
            }
            let batch_ref = unsafe { &mut *batch_cell.0.get() };
            for j in 0..instruction_lookups::D {
                let k = (*lookup_index >> instruction_ra_shifts[j])
                    & RingElement(instruction_lookups::K_CHUNK as u128 - 1);
                batch_ref.instruction_ra[j][i] = Some(k.downcast());
            }
        });

    // Classify lookup indices as Either::Public or Either::Shared.
    // Public indices are for control-only instructions (LUI, AUIPC, JAL, VirtualPow2*, etc.)
    // where the entire lookup index is deterministic from public instruction fields.
    let either_indices: Vec<Either<u128, Rep3RingShare<u128>>> = indices
        .into_par_iter()
        .zip(trace.par_iter())
        .map(|(share, cycle)| match compute_public_index(cycle) {
            Some(plain) => Either::Public(plain),
            None => Either::Shared(share),
        })
        .collect();

    // Persist lookup indices for ReadRaf suffix evaluation
    state
        .prover_state
        .cycle_witness
        .update_stage3(crate::zkvm::dag::witness::Stage3Update {
            pc_sumcheck: None,
            read_raf_tables_and_masks: None,
            read_raf_lookup_indices: Some(either_indices),
            product_inputs: None,
        });

    let mut batch = Arc::try_unwrap(batch_cell)
        .ok()
        .expect("Arc should have single owner")
        .0
        .into_inner();

    // -- Convert to polynomials --
    let mut results = HashMap::with_capacity(polynomials.len());
    let _span = info_span!("convert_to_polynomials", count = polynomials.len()).entered();

    // should_branch[i] = lookup_output[i] * circuit_flags[Branch] (public scalar).
    // Compute this before RdInc/RamInc so we can borrow cycle_witness immutably (no cloning).
    if polynomials
        .iter()
        .any(|p| matches!(p, CommittedPolynomial::ShouldBranch))
    {
        let lookup_outputs = state.prover_state.cycle_witness.stage1_lookup_output();
        let flags_bits = state.prover_state.cycle_witness.pc_sumcheck_flags_bits();
        debug_assert_eq!(lookup_outputs.len(), flags_bits.len());

        let branch_mask = 1u32 << (CircuitFlags::Branch as usize);
        let mut should_branch: Vec<Rep3PrimeFieldShare<F>> =
            Vec::with_capacity(lookup_outputs.len());
        for (t, output) in lookup_outputs.iter().enumerate() {
            if (flags_bits[t] & branch_mask) != 0 {
                should_branch.push(*output);
            } else {
                should_branch.push(Rep3PrimeFieldShare::zero_share());
            }
        }
        results.insert(
            CommittedPolynomial::ShouldBranch,
            Rep3MultilinearPolynomial::from(should_branch),
        );
    }

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
                .row_stage1(t)
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
                // Already computed above to avoid borrowing + cloning cycle_witness flags.
            }
            CommittedPolynomial::ShouldJump => {
                let coeffs = std::mem::take(&mut batch.should_jump);
                results.insert(
                    *poly,
                    Rep3MultilinearPolynomial::Public(MultilinearPolynomial::<F>::from(coeffs)),
                );
            }
            CommittedPolynomial::RdInc => {
                // rd_inc = rd_post - rd_pre in the field (binary shares → EdaBits B2A)
                let n = batch.rd_pre.len();
                let rd_pre = std::mem::take(&mut batch.rd_pre);
                let rd_post = std::mem::take(&mut batch.rd_post);
                debug_assert_eq!(rd_pre.len(), n);
                debug_assert_eq!(rd_post.len(), n);

                let _span = info_span!(
                    "rd_inc_b2a",
                    n,
                    chunk = inc_b2a_chunk,
                    max_forks = inc_b2a_max_forks
                )
                .entered();
                let mut inc: Vec<Rep3PrimeFieldShare<F>> = Vec::with_capacity(n);
                for off in (0..n).step_by(inc_b2a_chunk) {
                    let end = (off + inc_b2a_chunk).min(n);
                    let chunk_len = end - off;
                    let mut combined: Vec<Rep3RingShare<u64>> = Vec::with_capacity(2 * chunk_len);
                    combined.extend_from_slice(&rd_pre[off..end]);
                    combined.extend_from_slice(&rd_post[off..end]);

                    let batch_eda = preproc.take_edabits::<u64>(2 * chunk_len);
                    let field_all: Vec<Rep3PrimeFieldShare<F>> = if inc_b2a_max_forks <= 1 {
                        edabits::ring_to_field_b2a_many::<u64, F, _>(
                            &combined,
                            &batch_eda,
                            io_ctx.main(),
                        )?
                    } else {
                        let chunk_size = (combined.len()).div_ceil(inc_b2a_max_forks);
                        io_ctx.par_chunks_preproc(combined, batch_eda, Some(chunk_size), |xs, b, c| {
                            edabits::ring_to_field_b2a_many::<u64, F, _>(&xs, &b, c)
                        })?
                    };
                    debug_assert_eq!(field_all.len(), 2 * chunk_len);
                    for i in 0..chunk_len {
                        inc.push(field_all[chunk_len + i] - field_all[i]);
                    }
                }
                drop(_span);
                let dense = Rep3DensePolynomial::new(inc);
                state
                    .prover_state
                    .cycle_witness
                    .set_stage2_incs(Some(dense.clone()), None);
                results.insert(*poly, Rep3MultilinearPolynomial::shared(dense));
            }
            CommittedPolynomial::RamInc => {
                // ram_inc = post_value - pre_value in the field (binary shares → EdaBits B2A)
                let n = batch.ram_pre.len();
                let ram_pre = std::mem::take(&mut batch.ram_pre);
                let ram_post = std::mem::take(&mut batch.ram_post);
                debug_assert_eq!(ram_pre.len(), n);
                debug_assert_eq!(ram_post.len(), n);

                let _span = info_span!(
                    "ram_inc_b2a",
                    n,
                    chunk = inc_b2a_chunk,
                    max_forks = inc_b2a_max_forks
                )
                .entered();
                let mut inc: Vec<Rep3PrimeFieldShare<F>> = Vec::with_capacity(n);
                for off in (0..n).step_by(inc_b2a_chunk) {
                    let end = (off + inc_b2a_chunk).min(n);
                    let chunk_len = end - off;
                    let mut combined: Vec<Rep3RingShare<u64>> = Vec::with_capacity(2 * chunk_len);
                    combined.extend_from_slice(&ram_pre[off..end]);
                    combined.extend_from_slice(&ram_post[off..end]);

                    let batch_eda = preproc.take_edabits::<u64>(2 * chunk_len);
                    let field_all: Vec<Rep3PrimeFieldShare<F>> = if inc_b2a_max_forks <= 1 {
                        edabits::ring_to_field_b2a_many::<u64, F, _>(
                            &combined,
                            &batch_eda,
                            io_ctx.main(),
                        )?
                    } else {
                        let chunk_size = (combined.len()).div_ceil(inc_b2a_max_forks);
                        io_ctx.par_chunks_preproc(combined, batch_eda, Some(chunk_size), |xs, b, c| {
                            edabits::ring_to_field_b2a_many::<u64, F, _>(&xs, &b, c)
                        })?
                    };
                    debug_assert_eq!(field_all.len(), 2 * chunk_len);
                    for i in 0..chunk_len {
                        inc.push(field_all[chunk_len + i] - field_all[i]);
                    }
                }
                drop(_span);
                let dense = Rep3DensePolynomial::new(inc);
                state
                    .prover_state
                    .cycle_witness
                    .set_stage2_incs(None, Some(dense.clone()));
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

    // Diagnostic: force arena purging at a known lifetime boundary (witness → commit)
    maybe_purge_jemalloc();

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

                    // Create EdaBits pool for B2A conversions in witness gen
                    let budget =
                        crate::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                    let counts = [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128];
                    let mut preproc =
                        mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool_batched::<F, _>(
                            counts,
                            budget.dabits,
                            &mut io_ctx,
                        )?;

                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        (*io_device).clone(),
                        memory,
                        io_ctx.party_id(),
                        ram_k,
                        None,
                    );

                    info!(?party, "generate_witness_batch_rep3 start");
                    let results = generate_witness_batch_rep3::<F, PCS, _>(
                        &polys,
                        &mut state,
                        &mut io_ctx,
                        &mut preproc,
                    )?;
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
                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RLC(_)) => {
                    unreachable!("RLC variant should not appear in witness polynomials");
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
