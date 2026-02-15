use std::array;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;

use jolt2_common::constants::XLEN;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::zkvm::instruction::{CircuitFlags, InstructionFlags};
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::{CommittedPolynomial, DTH_ROOT_OF_K};
use jolt_core::zkvm::{instruction_lookups, JoltProverPreprocessing};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3_ring::casts::ring_to_field_a2b_many;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use snarks_core::math::Math;

use crate::field::JoltField;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::Rep3MultilinearPolynomial;
use crate::utils::future_ring::FutureRep3Ring;
use crate::zkvm::instruction::Rep3LookupQuery;

use super::instruction::{Rep3Cycle, Rep3RAMAccess};

// ── Rep3WitnessData ─────────────────────────────────────────────────────────

/// Mirrors vanilla `WitnessData` but uses `Rep3RingShare` for shared fields.
struct Rep3WitnessData {
    left_instruction_input: Vec<Rep3RingShare<u64>>,
    right_instruction_input: Vec<Rep3RingShare<u64>>,
    rd_pre: Vec<Rep3RingShare<u64>>,
    rd_post: Vec<Rep3RingShare<u64>>,
    write_lookup_output_to_rd: Vec<u8>,
    write_pc_to_rd: Vec<u8>,
    should_branch: Vec<Rep3RingShare<u8>>,
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
            should_branch: vec![Rep3RingShare::default(); trace_len],
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

// ── generate_witness_batch_rep3 ─────────────────────────────────────────────

/// Rep3 version of vanilla `CommittedPolynomial::generate_witness_batch`.
///
/// Populates witness data from a Rep3 trace and converts to `Rep3MultilinearPolynomial<F>`.
/// The trace must have been promoted to shares and arithmetic representations populated
/// before calling.
///
/// - Shared fields (register values) go through `ring_to_field_a2b_many` → `Shared(...)`
/// - Public fields (flags derived from opcode) → `Public(...)`
/// - Deferred fields (instruction_ra, should_branch) are skipped or zeroed
pub fn generate_witness_batch_rep3<F, PCS, N>(
    polynomials: &[CommittedPolynomial],
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: &[Rep3Cycle],
    io_ctx: &mut IoContext<N>,
) -> std::io::Result<HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3Network,
    Standard: Distribution<u64> + Distribution<u8>,
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
    let batch_cell = Arc::new(SharedRep3WitnessData(UnsafeCell::new(batch)));

    // -- Parallel trace collection (mirrors vanilla par_iter) --
    // SAFETY: Each thread writes to a unique index of a pre-allocated vector
    (0..trace.len()).into_par_iter().for_each({
        let batch_cell = batch_cell.clone();
        move |i| {
            let cycle = &trace[i];
            let batch_ref = unsafe { &mut *batch_cell.0.get() };

            // Instruction inputs: rs1 → left, rs2 → right
            let (_rs1_idx, rs1_val) = cycle.rs1_read();
            let (_rs2_idx, rs2_val) = cycle.rs2_read();
            batch_ref.left_instruction_input[i] = rs1_val.as_arithmetic_u64();
            batch_ref.right_instruction_input[i] = rs2_val.as_arithmetic_u64();

            // Rd write: (rd_write_flag, pre, post)
            let (rd_write_flag, pre, post) = cycle.rd_write();
            batch_ref.rd_pre[i] = pre.as_arithmetic_u64();
            batch_ref.rd_post[i] = post.as_arithmetic_u64();

            let circuit_flags = cycle.instruction().circuit_flags();

            // WriteLookupOutputToRD (public)
            batch_ref.write_lookup_output_to_rd[i] =
                rd_write_flag * (circuit_flags[CircuitFlags::WriteLookupOutputToRD as usize] as u8);

            // WritePCtoRD (public)
            batch_ref.write_pc_to_rd[i] =
                rd_write_flag * (circuit_flags[CircuitFlags::Jump as usize] as u8);

            // ShouldBranch: deferred (needs to_lookup_output_rep3)
            // Vanilla: should_branch[i] = (lookup_output as u8) * (circuit_flags[Branch] as u8)
            // batch_ref.should_branch[i] remains default zero share

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

            // InstructionRa indices
            let lookup_index = Rep3LookupQuery::<XLEN>::to_lookup_index(cycle);
            for j in 0..instruction_lookups::D {
                // $x \mod 2^t$ is simply the low $t$ bits.
                let k = (lookup_index >> instruction_ra_shifts[j])
                    & RingElement(instruction_lookups::K_CHUNK as u128 - 1);
                batch_ref.instruction_ra[j][i] = Some(k.downcast());
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
                        ((address as usize >> (dth_log * (ram_d - 1 - j))) % DTH_ROOT_OF_K) as u8
                    });
                    batch_ref.ram_ra[j][i] = index;
                }
            }
        }
    });

    let mut batch = Arc::try_unwrap(batch_cell)
        .ok()
        .expect("Arc should have single owner")
        .0
        .into_inner();

    // -- Convert to polynomials --
    let mut results = HashMap::with_capacity(polynomials.len());

    for poly in polynomials {
        match poly {
            CommittedPolynomial::LeftInstructionInput => {
                let coeffs = std::mem::take(&mut batch.left_instruction_input);
                let field_shares: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&coeffs, io_ctx)?;
                results.insert(*poly, Rep3MultilinearPolynomial::from(field_shares));
            }
            CommittedPolynomial::RightInstructionInput => {
                let coeffs = std::mem::take(&mut batch.right_instruction_input);
                let field_shares: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&coeffs, io_ctx)?;
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
                // Deferred: needs to_lookup_output_rep3 (Lasso/Shout phase)
                let coeffs = std::mem::take(&mut batch.should_branch);
                let field_shares: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&coeffs, io_ctx)?;
                results.insert(*poly, Rep3MultilinearPolynomial::from(field_shares));
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
                    ring_to_field_a2b_many(&batch.rd_pre, io_ctx)?;
                let post_field: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&batch.rd_post, io_ctx)?;
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
                    ring_to_field_a2b_many(&batch.ram_pre, io_ctx)?;
                let post_field: Vec<Rep3PrimeFieldShare<F>> =
                    ring_to_field_a2b_many(&batch.ram_post, io_ctx)?;
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
                    );
                    // results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                }
            }
            CommittedPolynomial::BytecodeRa(i) => {
                if *i < bytecode_d {
                    let indices = std::mem::take(&mut batch.bytecode_ra[*i]);
                    let d = preprocessing.shared.bytecode.d;
                    let log_K = preprocessing.shared.bytecode.code_size.log_2();
                    let log_K_chunk = log_K.div_ceil(d);
                    let K_chunk = 1 << log_K_chunk;
                    // let one_hot = OneHotPolynomial::from_indices(indices, K_chunk);
                    // results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                }
            }
            CommittedPolynomial::RamRa(i) => {
                if *i < ram_d {
                    let indices = std::mem::take(&mut batch.ram_ra[*i]);
                    // let one_hot = OneHotPolynomial::from_indices(indices, DTH_ROOT_OF_K);
                    // results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                }
            }
        }
    }

    Ok(results)
}
