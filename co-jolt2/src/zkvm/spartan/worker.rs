use jolt_core::utils::math::Math;
use jolt_core::zkvm::r1cs::constraints::UNIFORM_R1CS;
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::spartan_interleaved_poly::Rep3SpartanInterleavedPolynomial;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::r1cs::inputs::{compute_claimed_witness_evals_rep3, Rep3R1CSCycleInputs};

pub struct Rep3SpartanDagWorker;

impl Rep3SpartanDagWorker {
    #[tracing::instrument(skip_all, name = "Rep3SpartanDagWorker::stage1_prove")]
    pub fn stage1_prove<F, PCS, N>(
        state: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        PCS: jolt_core::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>,
        N: Rep3NetworkWorker,
        Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
    {
        let party_id = io_ctx.party_id();

        let tau: Vec<F::Challenge> = io_ctx.network().receive_request()?;

        let cycle_witness = &state.prover_state.cycle_witness;
        let num_steps = cycle_witness.len();
        eyre::ensure!(num_steps.is_power_of_two(), "num_steps must be pow2");
        eyre::ensure!(
            !cycle_witness.lookup_output.is_empty(),
            "cycle_witness.lookup_output not populated"
        );

        let key = UniformSpartanKey::<F>::new(num_steps);
        let rows_per_step_padded = key.padded_row_constraint_per_step();
        eyre::ensure!(
            rows_per_step_padded >= UNIFORM_R1CS.len(),
            "padded constraint rows too small"
        );

        // Precompute Product shares for MUL rows that need shared×shared multiplication.
        let flags_bits = &cycle_witness.flags_bits;
        let rs1_field = &cycle_witness.rs1_value;
        let rs2_field = &cycle_witness.rs2_value;

        let mask_left_rs1 = 1u32 << (jolt_core::zkvm::instruction::CircuitFlags::LeftOperandIsRs1Value as usize);
        let mask_right_rs2 = 1u32 << (jolt_core::zkvm::instruction::CircuitFlags::RightOperandIsRs2Value as usize);
        let mask_mul = 1u32 << (jolt_core::zkvm::instruction::CircuitFlags::MultiplyOperands as usize);
        let mask_shared_mul = mask_mul | mask_left_rs1 | mask_right_rs2;

        let mul_rows: Vec<usize> = (0..num_steps)
            .into_par_iter()
            .filter(|&t| (flags_bits[t] & mask_shared_mul) == mask_shared_mul)
            .collect();
        let mut mul_map = vec![None; num_steps];
        for (k, &t) in mul_rows.iter().enumerate() {
            mul_map[t] = Some(k);
        }

        let mul_products: Vec<Rep3PrimeFieldShare<F>> = if mul_rows.is_empty() {
            vec![]
        } else {
            let lhs: Vec<_> = mul_rows.iter().map(|&t| rs1_field[t]).collect();
            let rhs: Vec<_> = mul_rows.iter().map(|&t| rs2_field[t]).collect();
            rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
        };

        let product_per_cycle: Vec<Rep3PrimeFieldShare<F>> = (0..num_steps)
            .map(|t| {
                mul_map[t]
                    .map(|k| mul_products[k])
                    .unwrap_or_else(Rep3PrimeFieldShare::zero_share)
            })
            .collect();

        // Materialize per-cycle R1CS inputs (cheap; uses cached cycle witness).
        let mut cycle_inputs: Vec<Rep3R1CSCycleInputs<F>> = Vec::with_capacity(num_steps);
        for t in 0..num_steps {
            let row = cycle_witness.row(t);
            cycle_inputs.push(Rep3R1CSCycleInputs::from_trace(
                party_id,
                row,
                product_per_cycle[t],
            ));
        }

        let mut az_bz_cz_poly = Rep3SpartanInterleavedPolynomial::<F>::new(
            party_id,
            &key,
            &tau,
            &cycle_inputs,
        )?;

        let num_rounds_x = key.num_rows_bits();
        let mut r: Vec<F::Challenge> = Vec::with_capacity(num_rounds_x);

        for _round in 0..num_rounds_x {
            let (t0, t_inf) = az_bz_cz_poly.quadratic_evals(io_ctx.main())?;
            io_ctx.network().send_response((t0, t_inf))?;
            let r_i: F::Challenge = io_ctx.network().receive_request()?;
            r.push(r_i);
            az_bz_cz_poly.bind(party_id, r_i);
        }

        // Send final Az,Bz,Cz eval shares at the full sumcheck point.
        let final_evals: [AdditiveShare<F>; 3] = az_bz_cz_poly.final_evals_additive();
        io_ctx.network().send_response(final_evals.to_vec())?;

        // Send claimed witness eval shares at r_cycle (outer sumcheck point restricted to step vars).
        let mut r_reversed = r;
        r_reversed.reverse();
        let num_steps_bits = num_steps.log_2();
        let r_cycle = &r_reversed[..num_steps_bits];

        let claimed_witness_evals = compute_claimed_witness_evals_rep3(state, io_ctx, r_cycle)?;
        let claimed_additive: Vec<AdditiveShare<F>> = claimed_witness_evals
            .into_iter()
            .map(|x| x.into_additive())
            .collect();
        io_ctx.network().send_response(claimed_additive)?;

        Ok(())
    }
}
