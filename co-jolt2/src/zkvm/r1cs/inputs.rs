#![allow(clippy::too_many_arguments)]

use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::instruction::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use mpc_core::protocols::rep3::arithmetic;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rayon::prelude::*;
use strum::IntoEnumIterator;

use crate::field::JoltField;
use crate::utils::types::Rep3Value;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::dag::witness::Stage1RowRef;
use rayon::prelude::*;

pub use jolt_core::zkvm::r1cs::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS};

const NUM_R1CS_INPUTS: usize = ALL_R1CS_INPUTS.len();

/// Rep3 view of all R1CS inputs for a single cycle.
///
/// This is a cheap row view: it does not perform any MPC communication.
#[derive(Clone, Debug)]
pub struct Rep3R1CSCycleInputs<F: JoltField> {
    pub left_input: Rep3PrimeFieldShare<F>,
    pub right_input: Rep3PrimeFieldShare<F>,
    pub product: Rep3PrimeFieldShare<F>,
    pub left_lookup: Rep3PrimeFieldShare<F>,
    pub right_lookup: Rep3PrimeFieldShare<F>,
    pub lookup_output: Rep3PrimeFieldShare<F>,

    pub rd_addr: u8,
    pub rs1_read_value: Rep3PrimeFieldShare<F>,
    pub rs2_read_value: Rep3PrimeFieldShare<F>,
    pub rd_write_value: Rep3PrimeFieldShare<F>,

    pub ram_addr: u64,
    pub ram_read_value: Rep3PrimeFieldShare<F>,
    pub ram_write_value: Rep3PrimeFieldShare<F>,

    pub pc: u64,
    pub next_pc: u64,
    pub unexpanded_pc: u64,
    pub next_unexpanded_pc: u64,

    pub imm: i128,
    pub flags: [bool; NUM_CIRCUIT_FLAGS],
    pub next_is_noop: bool,
    pub should_jump: bool,
    pub should_branch: Rep3PrimeFieldShare<F>,
    pub write_lookup_output_to_rd_addr: u8,
    pub write_pc_to_rd_addr: u8,
}

impl<F: JoltField> Rep3R1CSCycleInputs<F> {
    pub fn from_trace(
        party_id: mpc_core::protocols::rep3::PartyID,
        row: Stage1RowRef<'_, F>,
        product: Rep3PrimeFieldShare<F>,
    ) -> Self {
        let (left_input, right_input) = row.to_instruction_inputs(party_id);
        let (left_lookup, right_lookup) = row.to_lookup_operands(party_id, product);
        let lookup_output = row.to_lookup_output();

        let rd_addr = row.rd_addr();
        let rs1_read_value = row.rs1_value();
        let rs2_read_value = row.rs2_value();
        let rd_write_value = row.rd_write_value();

        let ram_addr = row.ram_addr();
        let ram_read_value = row.ram_read_value();
        let ram_write_value = row.ram_write_value();

        let pc = row.pc_index();
        let next_pc = row.next_pc_index();
        let unexpanded_pc = row.unexpanded_pc();
        let next_unexpanded_pc = row.next_unexpanded_pc();
        let imm = row.imm();

        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        for flag in CircuitFlags::iter() {
            flags[flag as usize] = row.flag(flag);
        }
        let next_is_noop = row.next_is_noop();
        let should_jump = row.should_jump();

        let should_branch = if row.flag(CircuitFlags::Branch) {
            lookup_output
        } else {
            Rep3PrimeFieldShare::zero_share()
        };

        let write_lookup_output_to_rd_addr = if row.flag(CircuitFlags::WriteLookupOutputToRD) {
            rd_addr
        } else {
            0
        };
        let write_pc_to_rd_addr = if row.flag(CircuitFlags::Jump) {
            rd_addr
        } else {
            0
        };

        Self {
            left_input,
            right_input,
            product,
            left_lookup,
            right_lookup,
            lookup_output,
            rd_addr,
            rs1_read_value,
            rs2_read_value,
            rd_write_value,
            ram_addr,
            ram_read_value,
            ram_write_value,
            pc,
            next_pc,
            unexpanded_pc,
            next_unexpanded_pc,
            imm,
            flags,
            next_is_noop,
            should_jump,
            should_branch,
            write_lookup_output_to_rd_addr,
            write_pc_to_rd_addr,
        }
    }
}

/// Rep3 version of vanilla `compute_claimed_witness_evals`:
/// returns 41 evaluations in `ALL_R1CS_INPUTS` order.
#[tracing::instrument(skip_all)]
pub fn compute_claimed_witness_evals_rep3<F, PCS, N>(
    state: &mut StateManagerWorker<'_, F, PCS>,
    io_ctx: &mut IoContextPool<N>,
    r_cycle: &[F::Challenge],
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    F: JoltField,
    PCS: jolt_core::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
{
    let cycle_witness = &state.prover_state.cycle_witness;
    let trace_len = cycle_witness.len();
    eyre::ensure!(
        trace_len.is_power_of_two(),
        "trace length must be power-of-two"
    );
    eyre::ensure!(
        r_cycle.len() == trace_len.log_2(),
        "r_cycle length mismatch: got {}, expected {}",
        r_cycle.len(),
        trace_len.log_2()
    );

    let party_id = state.party_id;

    let flags_bits = cycle_witness.pc_sumcheck_flags_bits();

    // Batched shared×shared products: needed for ALL rows where both instruction
    // inputs are shared (vanilla computes Product = left * right unconditionally).
    let mask_left_rs1 = 1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
    let mask_right_rs2 = 1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);

    let mask_both_shared = mask_left_rs1 | mask_right_rs2;
    // Important: keep ordering deterministic across parties.
    // Do not build this list with a parallel filter+collect (ordering is not guaranteed).
    let shared_mul_rows: Vec<usize> = (0..trace_len)
        .into_par_iter()
        .filter(|&t| (flags_bits[t] & mask_both_shared) == mask_both_shared)
        .collect();
    let mut mul_map: Vec<u32> = vec![u32::MAX; trace_len];
    let (shared_mul_rows, mul_map) = build_shared_mul_rows_and_map(&flags_bits, mask_both_shared);

    let mul_products: Vec<Rep3PrimeFieldShare<F>> = if !shared_mul_rows.is_empty() {
        let (lhs, rhs): (Vec<_>, Vec<_>) = shared_mul_rows
            .par_iter()
            .map(|&t| {
                (
                    cycle_witness.row_stage1(t).rs1_value(),
                    cycle_witness.row_stage1(t).rs2_value(),
                )
            })
            .unzip();

        arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
    } else {
        vec![]
    };

    // Eq tables for r_cycle (same nested structure as vanilla)
    let m = r_cycle.len() / 2;
    let (r2, r1) = r_cycle.split_at(m);
    let (eq_one, eq_two) = rayon::join(
        || EqPolynomial::<F>::evals(r2),
        || EqPolynomial::<F>::evals(r1),
    );

    let n_inputs = NUM_R1CS_INPUTS;

    let idx_left = JoltR1CSInputs::LeftInstructionInput.to_index();
    let idx_right = JoltR1CSInputs::RightInstructionInput.to_index();
    let idx_product = JoltR1CSInputs::Product.to_index();
    let idx_write_lookup = JoltR1CSInputs::WriteLookupOutputToRD.to_index();
    let idx_write_pc = JoltR1CSInputs::WritePCtoRD.to_index();
    let idx_should_branch = JoltR1CSInputs::ShouldBranch.to_index();
    let idx_pc = JoltR1CSInputs::PC.to_index();
    let idx_unexp_pc = JoltR1CSInputs::UnexpandedPC.to_index();
    let idx_rd = JoltR1CSInputs::Rd.to_index();
    let idx_imm = JoltR1CSInputs::Imm.to_index();
    let idx_ram_addr = JoltR1CSInputs::RamAddress.to_index();
    let idx_rs1 = JoltR1CSInputs::Rs1Value.to_index();
    let idx_rs2 = JoltR1CSInputs::Rs2Value.to_index();
    let idx_rd_write = JoltR1CSInputs::RdWriteValue.to_index();
    let idx_ram_read = JoltR1CSInputs::RamReadValue.to_index();
    let idx_ram_write = JoltR1CSInputs::RamWriteValue.to_index();
    let idx_left_lookup = JoltR1CSInputs::LeftLookupOperand.to_index();
    let idx_right_lookup = JoltR1CSInputs::RightLookupOperand.to_index();
    let idx_next_unexp = JoltR1CSInputs::NextUnexpandedPC.to_index();
    let idx_next_pc = JoltR1CSInputs::NextPC.to_index();
    let idx_lookup_output = JoltR1CSInputs::LookupOutput.to_index();
    let idx_next_is_noop = JoltR1CSInputs::NextIsNoop.to_index();
    let idx_should_jump = JoltR1CSInputs::ShouldJump.to_index();

    // Parallel outer loop over eq_one, serial inner loop over eq_two (mirrors vanilla).
    // We accumulate into Rep3Value so public terms stay public until the final conversion.
    let acc: [Rep3Value<F>; NUM_R1CS_INPUTS] = (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let eq1_val = eq_one[x1];
            let mut inner: [Rep3Value<F>; NUM_R1CS_INPUTS] =
                core::array::from_fn(|_| Rep3Value::<F>::zero_public());

            for x2 in 0..eq_two.len() {
                let eq2_val = eq_two[x2];
                let t = x1 * eq_two.len() + x2;

                let row = cycle_witness.row_stage1(t);

                // Public per-cycle values
                inner[idx_pc].add_public_assign(F::from_u64(row.pc_index()) * eq2_val, party_id);
                inner[idx_unexp_pc]
                    .add_public_assign(F::from_u64(row.unexpanded_pc()) * eq2_val, party_id);
                inner[idx_rd]
                    .add_public_assign(F::from_u64(row.rd_addr() as u64) * eq2_val, party_id);
                inner[idx_imm].add_public_assign(F::from_i128(row.imm()) * eq2_val, party_id);
                inner[idx_ram_addr]
                    .add_public_assign(F::from_u64(row.ram_addr()) * eq2_val, party_id);

                inner[idx_next_unexp]
                    .add_public_assign(F::from_u64(row.next_unexpanded_pc()) * eq2_val, party_id);
                inner[idx_next_pc]
                    .add_public_assign(F::from_u64(row.next_pc_index()) * eq2_val, party_id);
                let next_is_noop = row.next_is_noop();
                inner[idx_next_is_noop]
                    .add_public_assign(F::from_bool(next_is_noop) * eq2_val, party_id);

                // Shared per-cycle values
                let lookup_output = row.to_lookup_output();
                inner[idx_lookup_output].add_assign(
                    &Rep3Value::Shared(arithmetic::mul_public(lookup_output, eq2_val)),
                    party_id,
                );
                inner[idx_rs1].add_assign(
                    &Rep3Value::Shared(arithmetic::mul_public(row.rs1_value(), eq2_val)),
                    party_id,
                );
                inner[idx_rs2].add_assign(
                    &Rep3Value::Shared(arithmetic::mul_public(row.rs2_value(), eq2_val)),
                    party_id,
                );
                inner[idx_rd_write].add_assign(
                    &Rep3Value::Shared(arithmetic::mul_public(row.rd_write_value(), eq2_val)),
                    party_id,
                );
                inner[idx_ram_read].add_assign(
                    &Rep3Value::Shared(arithmetic::mul_public(row.ram_read_value(), eq2_val)),
                    party_id,
                );
                inner[idx_ram_write].add_assign(
                    &Rep3Value::Shared(arithmetic::mul_public(row.ram_write_value(), eq2_val)),
                    party_id,
                );

                // Instruction inputs, product, and lookup operands.
                // Vanilla computes Product = left * right for ALL rows unconditionally.
                let (left_input, right_input) = row.to_instruction_inputs_value(party_id);
                let product = if mul_map[t] != u32::MAX {
                    Rep3Value::Shared(mul_products[mul_map[t] as usize])
                } else {
                    let fb = flags_bits[t];
                    let left_shared = (fb & mask_left_rs1) != 0;
                    let right_shared = (fb & mask_right_rs2) != 0;
                    match (left_shared, right_shared) {
                        (true, false) => left_input.mul(&right_input),
                        (false, true) => left_input.mul(&right_input),
                        (false, false) => left_input.mul(&right_input),
                        (true, true) => unreachable!("shared×shared row must be in mul_map"),
                    }
                };
                let (left_lookup, right_lookup) = row.to_lookup_operands_value(party_id, product);

                inner[idx_left].add_assign(&left_input.mul_public(eq2_val), party_id);
                inner[idx_right].add_assign(&right_input.mul_public(eq2_val), party_id);
                inner[idx_product].add_assign(&product.mul_public(eq2_val), party_id);
                inner[idx_left_lookup].add_assign(&left_lookup.mul_public(eq2_val), party_id);
                inner[idx_right_lookup].add_assign(&right_lookup.mul_public(eq2_val), party_id);

                let fb = flags_bits[t];
                if row.flag(CircuitFlags::WriteLookupOutputToRD) {
                    inner[idx_write_lookup]
                        .add_public_assign(F::from_u64(row.rd_addr() as u64) * eq2_val, party_id);
                }
                let is_jump = row.flag(CircuitFlags::Jump);
                if is_jump {
                    inner[idx_write_pc]
                        .add_public_assign(F::from_u64(row.rd_addr() as u64) * eq2_val, party_id);
                }
                if row.flag(CircuitFlags::Branch) {
                    inner[idx_should_branch].add_assign(
                        &Rep3Value::Shared(arithmetic::mul_public(lookup_output, eq2_val)),
                        party_id,
                    );
                }
                inner[idx_should_jump]
                    .add_public_assign(F::from_bool(is_jump && !next_is_noop) * eq2_val, party_id);

                for flag in CircuitFlags::iter() {
                    let idx = JoltR1CSInputs::OpFlags(flag).to_index();
                    let bit = 1u32 << (flag as usize);
                    inner[idx].add_public_assign(F::from_bool((fb & bit) != 0) * eq2_val, party_id);
                }
            }

            // Multiply inner sums by eq1[x1]
            for i in 0..n_inputs {
                inner[i] = inner[i].mul_public(eq1_val);
            }
            inner
        })
        .reduce(
            || core::array::from_fn(|_| Rep3Value::<F>::zero_public()),
            |mut acc, item| {
                for i in 0..n_inputs {
                    acc[i].add_assign(&item[i], party_id);
                }
                acc
            },
        );

    Ok(acc
        .into_iter()
        .map(|v| v.into_shared_rep3(party_id))
        .collect())
}

use crate::utils::send_ptr::SendPtr;

/// Build `(shared_mul_rows, mul_map)` for the predicate `(flags_bits[t] & mask) == mask`.
///
/// - `shared_mul_rows` is in ascending `t` order (deterministic across parties).
/// - `mul_map[t]` is the index of `t` in `shared_mul_rows`, or `u32::MAX` if not present.
pub(crate) fn build_shared_mul_rows_and_map(
    flags_bits: &[u32],
    mask: u32,
) -> (Vec<usize>, Vec<u32>) {
    let n = flags_bits.len();

    // For small traces, a single sequential pass is usually faster than Rayon overhead.
    const SEQ_THRESHOLD: usize = 1 << 14; // 16k
    if n < SEQ_THRESHOLD {
        let mut shared_mul_rows: Vec<usize> = Vec::new();
        let mut mul_map: Vec<u32> = vec![u32::MAX; n];
        for t in 0..n {
            if (flags_bits[t] & mask) == mask {
                mul_map[t] = shared_mul_rows.len() as u32;
                shared_mul_rows.push(t);
            }
        }
        return (shared_mul_rows, mul_map);
    }

    // Chunked parallel scan:
    // 1) count matches per chunk in parallel
    // 2) prefix-sum offsets (deterministic, small)
    // 3) fill `shared_mul_rows` and `mul_map` in parallel (disjoint writes)
    const CHUNK_SIZE: usize = 4096;
    let num_chunks = n.div_ceil(CHUNK_SIZE);

    let counts: Vec<usize> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * CHUNK_SIZE;
            let end = core::cmp::min(start + CHUNK_SIZE, n);
            flags_bits[start..end]
                .iter()
                .filter(|&&fb| (fb & mask) == mask)
                .count()
        })
        .collect();

    let mut offsets: Vec<usize> = Vec::with_capacity(num_chunks + 1);
    offsets.push(0);
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c);
    }
    let total = *offsets.last().unwrap();

    let mut shared_mul_rows: Vec<usize> = vec![0; total];
    let mut mul_map: Vec<u32> = vec![u32::MAX; n];

    let shared_mul_rows_ptr = SendPtr(shared_mul_rows.as_mut_ptr());
    let mul_map_ptr = SendPtr(mul_map.as_mut_ptr());

    (0..num_chunks).into_par_iter().for_each(move |i| {
        let start = i * CHUNK_SIZE;
        let end = core::cmp::min(start + CHUNK_SIZE, n);
        let mut out = offsets[i];

        for (j, &fb) in flags_bits[start..end].iter().enumerate() {
            if (fb & mask) == mask {
                let t = start + j;
                unsafe {
                    shared_mul_rows_ptr.write(out, t);
                    mul_map_ptr.write(t, out as u32);
                }
                out += 1;
            }
        }

        debug_assert_eq!(out, offsets[i + 1]);
    });

    (shared_mul_rows, mul_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use ark_bn254::Fr;
    use ark_std::{test_rng, UniformRand};
    use tracing::info;

    use crate::host::program::Rep3Program;
    use crate::utils::compute_ram_k;
    use crate::utils::test_utils::run_rep3_test;
    use crate::utils::tracing::init_tracing;
    use crate::zkvm::instruction::populate_operands_casts;
    use crate::zkvm::instruction::Rep3Cycle;
    use crate::zkvm::witness::{generate_witness_batch_rep3, populate_cycle_witness_rep3};
    use jolt_core::host::Program;
    use jolt_core::poly::commitment::mock::MockCommitScheme;
    use jolt_core::zkvm::bytecode::BytecodePreprocessing;
    use jolt_core::zkvm::ram::RAMPreprocessing;
    use jolt_core::zkvm::witness::{
        compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial,
    };
    use jolt_core::zkvm::{JoltProverPreprocessing, JoltSharedPreprocessing};
    use tracer::instruction::Cycle;

    type F = Fr;
    type PCS = MockCommitScheme<F>;
    type Challenge = <F as jolt_core::field::JoltField>::Challenge;

    #[test]
    #[ignore = "requires QUIC network sockets (not available in sandboxed test env)"]
    fn r1cs_claimed_evals_correct() {
        let _tracing_guard =
            init_tracing("r1cs_inputs_test.json", Path::new("/tmp/co-jolt2-traces"));

        let mut program = Program::new("fibonacci-guest");
        let elf_path = "/tmp/jolt-guest-targets/fibonacci-guest-/riscv64imac-unknown-none-elf/release/fibonacci-guest";
        program.elf = Some(PathBuf::from(elf_path));
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, memory_init, _) = program.decode();

        let mut rng = test_rng();
        let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);
        let (mut vanilla_trace, _memory, io_device) = program.trace(&inputs, &[], &[]);

        let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
        info!(raw_len = vanilla_trace.len(), padded_len, "padding traces");
        vanilla_trace.resize(padded_len, Cycle::NoOp);
        for (trace, _, _) in shares.iter_mut() {
            trace.resize(padded_len, Rep3Cycle::NoOp);
        }

        let shared = JoltSharedPreprocessing {
            memory_layout: io_device.memory_layout.clone(),
            bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
            ram: RAMPreprocessing::preprocess(memory_init.clone()),
        };
        let preprocessing: JoltProverPreprocessing<F, PCS> = JoltProverPreprocessing {
            generators: (),
            shared: shared.clone(),
        };

        let ram_k = compute_ram_k(&vanilla_trace, &preprocessing.shared);
        let bytecode_d = preprocessing.shared.bytecode.d;
        let ram_d = compute_d_parameter(ram_k);
        let _guard = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

        let log_t = padded_len.log_2();
        let r_cycle: Vec<Challenge> = (0..log_t).map(|_| Challenge::rand(&mut rng)).collect();

        let vanilla_evals = jolt_core::zkvm::r1cs::inputs::compute_claimed_witness_evals(
            &shared,
            &vanilla_trace,
            &r_cycle,
        );

        let preprocessing_arc = Arc::new(preprocessing);
        let io_device_arc = Arc::new(io_device);
        let base_port: u16 = 14300;

        let share_evals: [Vec<Rep3PrimeFieldShare<F>>; 3] = run_rep3_test(
            base_port,
            4,
            |party_idx| {
                let (trace, mem, _io_share) = shares[party_idx].clone();
                let preprocessing = Arc::clone(&preprocessing_arc);
                let io_device = Arc::clone(&io_device_arc);
                (trace, mem, io_device, preprocessing, ram_k, r_cycle.clone())
            },
            |input, mut io_ctx| {
                let (mut trace, mem, io_device, preprocessing, ram_k, r_cycle) = input;
                populate_operands_casts(&mut trace, io_ctx.main())?;
                let party_id = io_ctx.party_id();

                // Create EdaBits pool for B2A conversions in witness gen
                let budget =
                    crate::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                let counts = [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128];
                let mut edabits_pool =
                    mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool_batched::<F, _>(
                        counts,
                        budget.dabits,
                        &mut io_ctx,
                    )?;

                let mut state = StateManagerWorker::new(
                    &preprocessing,
                    trace,
                    (*io_device).clone(),
                    mem,
                    io_ctx.party_id(),
                    ram_k,
                    None,
                );

                populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut edabits_pool)?;

                let poly_keys: Vec<CommittedPolynomial> =
                    AllCommittedPolynomials::iter().copied().collect();
                let _witness_polys =
                    generate_witness_batch_rep3::<F, PCS, _>(&poly_keys, &mut state, &mut io_ctx, &mut edabits_pool)?;
                state.prover_state.trace = None;

                compute_claimed_witness_evals_rep3::<F, PCS, _>(&mut state, &mut io_ctx, &r_cycle)
            },
        );

        let opened = arithmetic::combine_field_elements_vec(vec![
            share_evals[0].clone(),
            share_evals[1].clone(),
            share_evals[2].clone(),
        ]);

        assert_eq!(opened.len(), ALL_R1CS_INPUTS.len());
        assert_eq!(opened.len(), vanilla_evals.len());
        for (i, (mpc, van)) in opened.iter().zip(vanilla_evals.iter()).enumerate() {
            assert_eq!(
                mpc, van,
                "mismatch at input index {i} ({:?})",
                ALL_R1CS_INPUTS[i]
            );
        }
    }
}
