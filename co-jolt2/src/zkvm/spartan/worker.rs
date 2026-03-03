use jolt_core::utils::math::Math;
use jolt_core::zkvm::instruction::CircuitFlags;
use jolt_core::zkvm::r1cs::constraints::UNIFORM_R1CS;
use jolt_core::zkvm::r1cs::inputs::{ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS};
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::spartan_interleaved_poly::Rep3SpartanInterleavedPolynomial;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::r1cs::inputs::{
    build_shared_mul_rows_and_map, compute_claimed_witness_evals_rep3, Rep3R1CSCycleInputs,
};
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

pub struct Rep3SpartanDagWorker;

impl Rep3SpartanDagWorker {
    #[tracing::instrument(skip_all, name = "Rep3SpartanDagWorker::stage1_prove")]
    pub fn stage1_prove<F, PCS, N>(
        state: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<(Vec<F::Challenge>, Vec<Rep3PrimeFieldShare<F>>)>
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
            !cycle_witness.stage1_lookup_output().is_empty(),
            "cycle_witness.lookup_output not populated"
        );

        let key = UniformSpartanKey::<F>::new(num_steps);

        // Precompute Product shares = left_input * right_input for ALL rows.
        // We only batch MPC shared×shared multiplication for the rows where BOTH inputs are shared.
        // All other cases are computed locally using mul_public or a trivial share.
        //
        // Vanilla always computes Product = left_input * right_input regardless of instruction type.
        // The R1CS constraint RightLookupEqProductIfMul (index 13) pairs with RightLookupSub (index 12)
        // in the sparse interleaved polynomial, so Product must match vanilla for correct t_inf.
        let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
        let mask_left_rs1 = 1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
        let mask_right_rs2 = 1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
        let mask_both_shared = mask_left_rs1 | mask_right_rs2;

        // Important: keep ordering deterministic across parties.
        // Do not build this list with a parallel filter+collect (ordering is not guaranteed).
        let (shared_mul_rows, mul_map) =
            build_shared_mul_rows_and_map(&flags_bits, mask_both_shared);

        let mul_products: Vec<Rep3PrimeFieldShare<F>> = if !shared_mul_rows.is_empty() {
            let (lhs, rhs): (Vec<_>, Vec<_>) = shared_mul_rows
                .par_iter()
                .map(|&t| {
                    let row = cycle_witness.row_stage1(t);
                    (row.rs1_value(), row.rs2_value())
                })
                .unzip();
            rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
        } else {
            vec![]
        };

        let product_per_cycle: Vec<Rep3PrimeFieldShare<F>> = (0..num_steps)
            .into_par_iter()
            .map(|t| {
                if mul_map[t] != u32::MAX {
                    return mul_products[mul_map[t] as usize];
                }

                let row = cycle_witness.row_stage1(t);
                let fb = flags_bits[t];
                let left_shared = (fb & mask_left_rs1) != 0;
                let right_shared = (fb & mask_right_rs2) != 0;

                match (left_shared, right_shared) {
                    (true, false) => {
                        rep3_arithmetic::mul_public(row.rs1_value(), row.to_right_public_input())
                    }
                    (false, true) => {
                        rep3_arithmetic::mul_public(row.rs2_value(), row.to_left_public_input())
                    }
                    (false, false) => {
                        let l = row.to_left_public_input();
                        let r = row.to_right_public_input();
                        rep3_arithmetic::promote_to_trivial_share(party_id, l * r)
                    }
                    (true, true) => unreachable!("shared×shared row must be in mul_map"),
                }
            })
            .collect();

        // Materialize per-cycle R1CS inputs (cheap; uses cached cycle witness).
        let mut cycle_inputs: Vec<Rep3R1CSCycleInputs<F>> = Vec::with_capacity(num_steps);
        for t in 0..num_steps {
            let row = cycle_witness.row_stage1(t);
            cycle_inputs.push(Rep3R1CSCycleInputs::from_trace(
                party_id,
                row,
                product_per_cycle[t],
            ));
        }

        let mut az_bz_cz_poly = Rep3SpartanInterleavedPolynomial::<F>::new(
            &key,
            &cycle_inputs,
            &UNIFORM_R1CS,
            party_id,
        )?;

        let mut eq_poly = GruenSplitEqPolynomial::<F>::new(&tau, BindingOrder::LowToHigh);

        let num_rounds_x = key.num_rows_bits();
        let mut r: Vec<F::Challenge> = Vec::with_capacity(num_rounds_x);

        for round in 0..num_rounds_x {
            if round == 0 {
                az_bz_cz_poly.streaming_sumcheck_round(&mut eq_poly, &mut r, io_ctx)?;
            } else {
                az_bz_cz_poly.remaining_sumcheck_round(&mut eq_poly, &mut r, io_ctx)?;
            }
        }

        // Send final Az,Bz,Cz eval shares at the full sumcheck point.
        let final_evals: [AdditiveShare<F>; 3] = az_bz_cz_poly.final_evals_additive(party_id);
        io_ctx.network().send_response(final_evals.to_vec())?;

        // Send claimed witness eval shares at r_cycle (outer sumcheck point restricted to step vars).
        let mut r_reversed = r;
        r_reversed.reverse();
        let num_steps_bits = num_steps.log_2();
        let r_cycle = &r_reversed[..num_steps_bits];

        let claimed_witness_evals = compute_claimed_witness_evals_rep3(state, io_ctx, r_cycle)?;

        // Cache SpartanOuter openings in the worker accumulator for later stages.
        let opening_point_cycle = OpeningPoint::<BIG_ENDIAN, F>::new(r_cycle.to_vec());

        let committed_polys: Vec<CommittedPolynomial> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        let committed_claims: Vec<Rep3PrimeFieldShare<F>> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();
        state.accumulator.append_dense(
            committed_polys,
            SumcheckId::SpartanOuter,
            opening_point_cycle.r.clone(),
            &committed_claims,
        );

        for input in ALL_R1CS_INPUTS.iter() {
            if COMMITTED_R1CS_INPUTS.contains(input) {
                continue;
            }
            let poly = VirtualPolynomial::try_from(input).ok().unwrap();
            let eval = claimed_witness_evals[input.to_index()];
            state.accumulator.append_virtual(
                poly,
                SumcheckId::SpartanOuter,
                opening_point_cycle.clone(),
                eval,
            );
        }

        let claimed_additive: Vec<AdditiveShare<F>> = claimed_witness_evals
            .iter()
            .map(|x| x.into_additive())
            .collect();
        io_ctx.network().send_response(claimed_additive)?;

        Ok((r_reversed, claimed_witness_evals))
    }
}
