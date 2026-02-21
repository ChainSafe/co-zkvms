use jolt_core::zkvm::r1cs::constraints::UNIFORM_R1CS;
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use jolt_core::utils::math::Math;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rand::distributions::{Distribution, Standard};

use crate::field::JoltField;
use crate::poly::spartan_interleaved_poly::Rep3SpartanInterleavedPolynomial;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::r1cs::inputs::{compute_claimed_witness_evals_rep3, Rep3R1CSCycleInputs};
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;

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

        // Precompute Product shares = left_input * right_input for ALL rows.
        // Vanilla always computes Product = left_input * right_input regardless of instruction type.
        // The R1CS constraint RightLookupEqProductIfMul (index 13) pairs with RightLookupSub (index 12)
        // in the sparse interleaved polynomial, so Product must match vanilla for correct t_inf.
        let lhs_all: Vec<Rep3PrimeFieldShare<F>> = (0..num_steps)
            .map(|t| {
                let row = cycle_witness.row(t);
                row.to_instruction_inputs(party_id).0
            })
            .collect();
        let rhs_all: Vec<Rep3PrimeFieldShare<F>> = (0..num_steps)
            .map(|t| {
                let row = cycle_witness.row(t);
                row.to_instruction_inputs(party_id).1
            })
            .collect();
        let product_per_cycle: Vec<Rep3PrimeFieldShare<F>> =
            rep3_arithmetic::mul_vec_par(&lhs_all, &rhs_all, io_ctx.main())?;

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
        let claimed_additive: Vec<AdditiveShare<F>> = claimed_witness_evals
            .into_iter()
            .map(|x| x.into_additive())
            .collect();
        io_ctx.network().send_response(claimed_additive)?;

        Ok(())
    }
}
