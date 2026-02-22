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
use crate::zkvm::dag::state_manager::Rep3CycleWitnessRef;
use crate::zkvm::dag::state_manager::StateManagerWorker;

pub use jolt_core::zkvm::r1cs::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS};

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
        row: Rep3CycleWitnessRef<'_, F>,
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

        let pc = row.pc();
        let next_pc = row.next_pc();
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

    let pc = &cycle_witness.pc;
    let unexpanded_pc = &cycle_witness.unexpanded_pc;
    let imm = &cycle_witness.imm;
    let rd_addr = &cycle_witness.rd_addr;
    let ram_addr = &cycle_witness.ram_addr;
    let flags_bits = &cycle_witness.flags_bits;

    let rs1_field = &cycle_witness.rs1_value;
    let rs2_field = &cycle_witness.rs2_value;

    // Batched shared×shared products: needed for ALL rows where both instruction
    // inputs are shared (vanilla computes Product = left * right unconditionally).
    let mask_left_rs1 = 1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
    let mask_right_rs2 = 1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);

    let mask_both_shared = mask_left_rs1 | mask_right_rs2;
    let shared_mul_rows: Vec<usize> = (0..trace_len)
        .into_par_iter()
        .filter(|&t| (flags_bits[t] & mask_both_shared) == mask_both_shared)
        .collect();
    let mut mul_map = vec![None; trace_len];
    for (k, &t) in shared_mul_rows.iter().enumerate() {
        mul_map[t] = Some(k);
    }

    let mul_products: Vec<Rep3PrimeFieldShare<F>> = if !shared_mul_rows.is_empty() {
        let lhs: Vec<_> = shared_mul_rows.iter().map(|&t| rs1_field[t]).collect();
        let rhs: Vec<_> = shared_mul_rows.iter().map(|&t| rs2_field[t]).collect();
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

    let n_inputs = ALL_R1CS_INPUTS.len();

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
    let (acc_public, acc_shared) = (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let eq1_val = eq_one[x1];
            let mut inner_public = vec![F::zero(); n_inputs];
            let mut inner_shared = vec![Rep3PrimeFieldShare::zero_share(); n_inputs];

            for x2 in 0..eq_two.len() {
                let eq2_val = eq_two[x2];
                let t = x1 * eq_two.len() + x2;

                let row = cycle_witness.row(t);

                // Public per-cycle values
                inner_public[idx_pc] += F::from_u64(pc[t]) * eq2_val;
                inner_public[idx_unexp_pc] += F::from_u64(unexpanded_pc[t]) * eq2_val;
                inner_public[idx_rd] += F::from_u64(rd_addr[t] as u64) * eq2_val;
                inner_public[idx_imm] += F::from_i128(imm[t]) * eq2_val;
                inner_public[idx_ram_addr] += F::from_u64(ram_addr[t]) * eq2_val;
                let next_unexp = if t + 1 < trace_len {
                    unexpanded_pc[t + 1]
                } else {
                    0
                };
                let next_pc_val = if t + 1 < trace_len { pc[t + 1] } else { 0 };
                let next_is_noop = row.next_is_noop();
                inner_public[idx_next_unexp] += F::from_u64(next_unexp) * eq2_val;
                inner_public[idx_next_pc] += F::from_u64(next_pc_val) * eq2_val;
                inner_public[idx_next_is_noop] += F::from_bool(next_is_noop) * eq2_val;

                // Shared per-cycle values
                let lookup_output = row.to_lookup_output();
                inner_shared[idx_lookup_output] += arithmetic::mul_public(lookup_output, eq2_val);
                inner_shared[idx_rs1] += arithmetic::mul_public(row.rs1_value(), eq2_val);
                inner_shared[idx_rs2] += arithmetic::mul_public(row.rs2_value(), eq2_val);
                inner_shared[idx_rd_write] += arithmetic::mul_public(row.rd_write_value(), eq2_val);
                inner_shared[idx_ram_read] += arithmetic::mul_public(row.ram_read_value(), eq2_val);
                inner_shared[idx_ram_write] +=
                    arithmetic::mul_public(row.ram_write_value(), eq2_val);

                // Instruction inputs, product, and lookup operands.
                // Vanilla computes Product = left * right for ALL rows unconditionally.
                let (left_input, right_input) = row.to_instruction_inputs(party_id);
                let product = match mul_map[t] {
                    Some(k) => mul_products[k],
                    None => {
                        let fb = flags_bits[t];
                        let left_shared = (fb & mask_left_rs1) != 0;
                        let right_shared = (fb & mask_right_rs2) != 0;
                        match (left_shared, right_shared) {
                            (true, false) => {
                                arithmetic::mul_public(rs1_field[t], row.to_right_public_input())
                            }
                            (false, true) => {
                                arithmetic::mul_public(rs2_field[t], row.to_left_public_input())
                            }
                            (false, false) => {
                                let l = row.to_left_public_input();
                                let r = row.to_right_public_input();
                                arithmetic::promote_to_trivial_share(party_id, l * r)
                            }
                            (true, true) => unreachable!("should be in mul_map"),
                        }
                    }
                };
                let (left_lookup, right_lookup) = row.to_lookup_operands(party_id, product);

                inner_shared[idx_left] += arithmetic::mul_public(left_input, eq2_val);
                inner_shared[idx_right] += arithmetic::mul_public(right_input, eq2_val);
                inner_shared[idx_product] += arithmetic::mul_public(product, eq2_val);
                inner_shared[idx_left_lookup] += arithmetic::mul_public(left_lookup, eq2_val);
                inner_shared[idx_right_lookup] += arithmetic::mul_public(right_lookup, eq2_val);

                let fb = flags_bits[t];
                if row.flag(CircuitFlags::WriteLookupOutputToRD) {
                    inner_public[idx_write_lookup] += F::from_u64(rd_addr[t] as u64) * eq2_val;
                }
                let is_jump = row.flag(CircuitFlags::Jump);
                if is_jump {
                    inner_public[idx_write_pc] += F::from_u64(rd_addr[t] as u64) * eq2_val;
                }
                if row.flag(CircuitFlags::Branch) {
                    inner_shared[idx_should_branch] +=
                        arithmetic::mul_public(lookup_output, eq2_val);
                }
                inner_public[idx_should_jump] += F::from_bool(is_jump && !next_is_noop) * eq2_val;

                for flag in CircuitFlags::iter() {
                    let idx = JoltR1CSInputs::OpFlags(flag).to_index();
                    let bit = 1u32 << (flag as usize);
                    inner_public[idx] += F::from_bool((fb & bit) != 0) * eq2_val;
                }
            }

            // Multiply inner sums by eq1[x1]
            for i in 0..n_inputs {
                inner_public[i] *= eq1_val;
                inner_shared[i] = arithmetic::mul_public(inner_shared[i], eq1_val);
            }
            (inner_public, inner_shared)
        })
        .reduce(
            || {
                (
                    vec![F::zero(); n_inputs],
                    vec![Rep3PrimeFieldShare::zero_share(); n_inputs],
                )
            },
            |(mut acc_pub, mut acc_sh), (item_pub, item_sh)| {
                for i in 0..n_inputs {
                    acc_pub[i] += item_pub[i];
                    acc_sh[i] += item_sh[i];
                }
                (acc_pub, acc_sh)
            },
        );

    Ok((0..n_inputs)
        .map(|i| acc_shared[i] + arithmetic::promote_to_trivial_share(party_id, acc_public[i]))
        .collect())
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
    use crate::zkvm::witness::generate_witness_batch_rep3;
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

                let mut state = StateManagerWorker::new(
                    &preprocessing,
                    trace,
                    (*io_device).clone(),
                    mem,
                    io_ctx.party_id(),
                    ram_k,
                    todo!(),
                );

                // Populate lookup cache via witness generation
                let poly_keys: Vec<CommittedPolynomial> =
                    AllCommittedPolynomials::iter().copied().collect();
                let _witness_polys =
                    generate_witness_batch_rep3::<F, PCS, _>(&poly_keys, &mut state, &mut io_ctx)?;
                state.prover_state.trace.clear();
                state.prover_state.trace.shrink_to_fit();

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
