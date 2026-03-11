use std::array;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use jolt_common::constants::{LookupIndexInt, XlenInt, XLEN};
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::utils::math::Math;
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
use tracing::info_span;

#[cfg(feature = "ring-msm")]
use crate::poly::compact_polynomial::Rep3CompactPolynomial;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use crate::utils::future_ring::{FutureRep3Ring, Rep3RingFutureExt};
use crate::utils::memory::maybe_purge_jemalloc;
use crate::utils::types::Either;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::instruction::{populate_operands_casts, Rep3LookupQuery, Rep3Operand};

use super::instruction::{Rep3Cycle, Rep3RAMAccess};

// ── ring_to_field cast helpers ──────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
enum SparseCastCol {
    Rs1 = 0,
    Rs2 = 1,
    RdWrite = 2,
    RamRead = 3,
    RamWrite = 4,
    Advice = 5,
}

#[derive(Clone, Debug)]
struct SparseCastJob {
    col: SparseCastCol,
    row: usize,
    share: Rep3RingShare<XlenInt>,
}

struct SharedSparseFieldCols<F: JoltField> {
    rs1: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    rs2: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    rd_write: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    ram_read: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    ram_write: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
    advice: UnsafeCell<Vec<Rep3PrimeFieldShare<F>>>,
}

unsafe impl<F: JoltField> Sync for SharedSparseFieldCols<F> {}

#[tracing::instrument(
    skip_all,
    name = "fill_field_from_operands_sparse",
    fields(jobs = jobs.len())
)]
fn fill_field_from_operands_sparse<F, N>(
    io_ctx: &mut IoContextPool<N>,
    jobs: Vec<SparseCastJob>,
    out: Arc<SharedSparseFieldCols<F>>,
    preproc: &mut PreprocessingPool<F>,
) -> eyre::Result<()>
where
    F: JoltField,
    N: Rep3NetworkWorker,
    Standard: Distribution<XlenInt>,
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
        let mut shares: Vec<Rep3RingShare<XlenInt>> = Vec::with_capacity(n);
        let mut targets: Vec<(SparseCastCol, usize)> = Vec::with_capacity(n);
        for job in chunk {
            shares.push(job.share);
            targets.push((job.col, job.row));
        }

        let batch = preproc.take_edabits::<XlenInt>(n)?;
        let casted = io_ctx.par_chunks_preproc(shares, batch, None, |xs, batch, ctx| {
            edabits::ring_to_field_b2a_many::<XlenInt, F, _>(&xs, &batch, ctx)
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
                    SparseCastCol::Advice => (&mut *out.advice.get())[row] = value,
                }
            }
        }
    }

    Ok(())
}

/// Compute the plaintext lookup index for cycles with fully-public operands.
///
/// Returns `Some(plain)` for instructions where both operands are public
/// (control-only: LUI, AUIPC, JAL, VirtualAdvice, VirtualPow2*, VirtualShiftRightBitmask*).
/// These use the "add" path: index = left + right (LookupIndexInt arithmetic).
///
/// Returns `None` for all other instructions (operands may be secret-shared).
fn compute_public_index(cycle: &Rep3Cycle) -> Option<LookupIndexInt> {
    let try_public_add = |inputs: (Rep3Operand, Rep3Operand)| -> Option<LookupIndexInt> {
        let (l, r) = inputs;
        let l_val = match l {
            Rep3Operand::Public(v) => v,
            Rep3Operand::Shared {
                public: Some(v), ..
            } => v as i128,
            _ => return None,
        };
        let r_val = match r {
            Rep3Operand::Public(v) => v,
            Rep3Operand::Shared {
                public: Some(v), ..
            } => v as i128,
            _ => return None,
        };
        Some((l_val as XlenInt as LookupIndexInt) + (r_val as XlenInt as LookupIndexInt))
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
        #[cfg(feature = "rv64")]
        Rep3Cycle::VirtualPow2W(c) => {
            try_public_add(Rep3LookupQuery::<XLEN>::to_instruction_inputs(c))
        }
        #[cfg(feature = "rv64")]
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
            Rep3Operand::Public(v) => Some(v as u64),
            Rep3Operand::Shared {
                public: Some(v), ..
            } => Some(v as XlenInt as u64),
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
        #[cfg(feature = "rv64")]
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
    rd_pre: Vec<Rep3RingShare<XlenInt>>,
    rd_post: Vec<Rep3RingShare<XlenInt>>,
    write_lookup_output_to_rd: Vec<u8>,
    write_pc_to_rd: Vec<u8>,
    should_jump: Vec<u8>,
    ram_pre: Vec<Rep3RingShare<XlenInt>>,
    ram_post: Vec<Rep3RingShare<XlenInt>>,
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
    let mut advice: Vec<Rep3PrimeFieldShare<F>> = vec![Rep3PrimeFieldShare::zero_share(); n];
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
        advice: UnsafeCell::new(vec![Rep3PrimeFieldShare::zero_share(); n]),
    });
    let mut cast_jobs: Vec<SparseCastJob> = Vec::new();
    cast_jobs.reserve(n * 5);

    let mut maybe_push = |col: SparseCastCol, row: usize, op: &Rep3Operand| match op {
        Rep3Operand::Public(v) => {
            let share = promote_to_trivial_share(party_id, F::from_u64(*v as u64));
            unsafe {
                match col {
                    SparseCastCol::Rs1 => (&mut *shared_cols.rs1.get())[row] = share,
                    SparseCastCol::Rs2 => (&mut *shared_cols.rs2.get())[row] = share,
                    SparseCastCol::RdWrite => (&mut *shared_cols.rd_write.get())[row] = share,
                    SparseCastCol::RamRead => (&mut *shared_cols.ram_read.get())[row] = share,
                    SparseCastCol::RamWrite => (&mut *shared_cols.ram_write.get())[row] = share,
                    SparseCastCol::Advice => (&mut *shared_cols.advice.get())[row] = share,
                }
            }
        }
        Rep3Operand::Shared {
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
                    SparseCastCol::Advice => (&mut *shared_cols.advice.get())[row] = share,
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
        #[cfg(not(feature = "rv64"))]
        imm.push(if circuit_flags[CircuitFlags::Branch as usize] {
            norm.operands.imm as i32 as i128
        } else {
            norm.operands.imm as XlenInt as i128
        });
        #[cfg(feature = "rv64")]
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
                maybe_push(
                    SparseCastCol::Advice,
                    t,
                    c.advice
                        .as_ref()
                        .expect("VirtualAdvice shared advice payload missing"),
                );
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

    fill_field_from_operands_sparse::<F, N>(io_ctx, cast_jobs, Arc::clone(&shared_cols), preproc)?;

    let _span = tracing::trace_span!("init_rep3_witnesses").entered();
    let shared_cols = Arc::try_unwrap(shared_cols)
        .ok()
        .expect("shared cols Arc should have single owner");
    let rs1_value = shared_cols.rs1.into_inner();
    let rs2_value = shared_cols.rs2.into_inner();
    let rd_write_value = shared_cols.rd_write.into_inner();
    let ram_read_value = shared_cols.ram_read.into_inner();
    let ram_write_value = shared_cols.ram_write.into_inner();
    advice = shared_cols.advice.into_inner();

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
    let index_futures: Vec<FutureRep3Ring<LookupIndexInt, Rep3RingShare<LookupIndexInt>>> = (0
        ..trace.len())
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
    let indices: Vec<Rep3RingShare<LookupIndexInt>> = {
        let total = index_futures.len();
        let _span = info_span!("fulfill_index_futures", count = total).entered();
        let mut out: Vec<Rep3RingShare<LookupIndexInt>> = Vec::with_capacity(total);
        let mut iter = index_futures.into_iter();
        let mut chunk_id: usize = 0;
        loop {
            let mut chunk: Vec<FutureRep3Ring<LookupIndexInt, Rep3RingShare<LookupIndexInt>>> =
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
            let resolved: Vec<Rep3RingShare<LookupIndexInt>> =
                chunk.fulfill_batched(io_ctx, |r, ()| r)?;
            drop(_chunk_span);
            out.extend(resolved);
            chunk_id += 1;
        }
        drop(_span);
        out
    };

    // Phase 3: Chunk resolved indices into instruction_ra (parallel, no comms)
    // SAFETY: Each thread writes to a unique index i across the D arrays.
    //
    // Match vanilla witness generation exactly: padded NoOp cycles contribute
    // lookup index 0, so all InstructionRa chunks are `Some(0)`.
    indices
        .par_iter()
        .enumerate()
        .for_each(|(i, lookup_index)| {
            let batch_ref = unsafe { &mut *batch_cell.0.get() };
            for j in 0..instruction_lookups::D {
                let k = (*lookup_index >> instruction_ra_shifts[j])
                    & RingElement(instruction_lookups::K_CHUNK as LookupIndexInt - 1);
                batch_ref.instruction_ra[j][i] = Some(k.downcast());
            }
        });

    // Classify lookup indices as Either::Public or Either::Shared.
    // Public indices are for control-only instructions (LUI, AUIPC, JAL, VirtualPow2*, etc.)
    // where the entire lookup index is deterministic from public instruction fields.
    let either_indices: Vec<Either<LookupIndexInt, Rep3RingShare<LookupIndexInt>>> = indices
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

    // Build instruction input polynomials.
    #[cfg(feature = "ring-msm")]
    let mut left_ops: Vec<Rep3Operand> = Vec::new();
    #[cfg(feature = "ring-msm")]
    let mut right_ops: Vec<Rep3Operand> = Vec::new();
    #[cfg(feature = "ring-msm")]
    if polynomials.iter().any(|p| {
        matches!(
            p,
            CommittedPolynomial::LeftInstructionInput | CommittedPolynomial::RightInstructionInput
        )
    }) {
        let trace: &[Rep3Cycle] = state.trace_ref();
        let n = trace.len();
        left_ops = Vec::with_capacity(n);
        right_ops = Vec::with_capacity(n);
        for cycle in trace {
            let (left_op, right_op) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(cycle);
            left_ops.push(left_op);
            right_ops.push(right_op);
        }
    }

    for poly in polynomials {
        match poly {
            CommittedPolynomial::LeftInstructionInput => {
                #[cfg(feature = "ring-msm")]
                {
                    let compact = Rep3CompactPolynomial::from_operands(mem::take(&mut left_ops));
                    results.insert(
                        *poly,
                        Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::CompactRing(compact)),
                    );
                }
                #[cfg(not(feature = "ring-msm"))]
                {
                    let party_id = io_ctx.party_id();
                    let n = state.prover_state.cycle_witness.len();
                    let field_shares: Vec<Rep3PrimeFieldShare<F>> = (0..n)
                        .map(|t| {
                            state
                                .prover_state
                                .cycle_witness
                                .row_stage1(t)
                                .to_instruction_inputs(party_id)
                                .0
                        })
                        .collect();
                    results.insert(*poly, Rep3MultilinearPolynomial::from(field_shares));
                }
            }
            CommittedPolynomial::RightInstructionInput => {
                #[cfg(feature = "ring-msm")]
                {
                    let compact = Rep3CompactPolynomial::from_operands(mem::take(&mut right_ops));
                    results.insert(
                        *poly,
                        Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::CompactRing(compact)),
                    );
                }
                #[cfg(not(feature = "ring-msm"))]
                {
                    let party_id = io_ctx.party_id();
                    let n = state.prover_state.cycle_witness.len();
                    let field_shares: Vec<Rep3PrimeFieldShare<F>> = (0..n)
                        .map(|t| {
                            state
                                .prover_state
                                .cycle_witness
                                .row_stage1(t)
                                .to_instruction_inputs(party_id)
                                .1
                        })
                        .collect();
                    results.insert(*poly, Rep3MultilinearPolynomial::from(field_shares));
                }
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
                    let mut combined: Vec<Rep3RingShare<XlenInt>> =
                        Vec::with_capacity(2 * chunk_len);
                    combined.extend_from_slice(&rd_pre[off..end]);
                    combined.extend_from_slice(&rd_post[off..end]);

                    let batch_eda = preproc.take_edabits::<XlenInt>(2 * chunk_len)?;
                    let field_all: Vec<Rep3PrimeFieldShare<F>> = if inc_b2a_max_forks <= 1 {
                        edabits::ring_to_field_b2a_many::<XlenInt, F, _>(
                            &combined,
                            &batch_eda,
                            io_ctx.main(),
                        )?
                    } else {
                        let chunk_size = (combined.len()).div_ceil(inc_b2a_max_forks);
                        io_ctx.par_chunks_preproc(
                            combined,
                            batch_eda,
                            Some(chunk_size),
                            |xs, b, c| edabits::ring_to_field_b2a_many::<XlenInt, F, _>(&xs, &b, c),
                        )?
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
                    let mut combined: Vec<Rep3RingShare<XlenInt>> =
                        Vec::with_capacity(2 * chunk_len);
                    combined.extend_from_slice(&ram_pre[off..end]);
                    combined.extend_from_slice(&ram_post[off..end]);

                    let batch_eda = preproc.take_edabits::<XlenInt>(2 * chunk_len)?;
                    let field_all: Vec<Rep3PrimeFieldShare<F>> = if inc_b2a_max_forks <= 1 {
                        edabits::ring_to_field_b2a_many::<XlenInt, F, _>(
                            &combined,
                            &batch_eda,
                            io_ctx.main(),
                        )?
                    } else {
                        let chunk_size = (combined.len()).div_ceil(inc_b2a_max_forks);
                        io_ctx.par_chunks_preproc(
                            combined,
                            batch_eda,
                            Some(chunk_size),
                            |xs, b, c| edabits::ring_to_field_b2a_many::<XlenInt, F, _>(&xs, &b, c),
                        )?
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
                        preproc,
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
