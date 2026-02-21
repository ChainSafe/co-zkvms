use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::subprotocols::sumcheck::{process_eq_sumcheck_round, SumcheckInstanceProof};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::r1cs::inputs::{ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS};
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

use crate::field::JoltField;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManagerCoordinator};

pub struct Rep3SpartanDag;

impl Rep3SpartanDag {
    #[tracing::instrument(skip_all, name = "Rep3SpartanDag::stage1_prove")]
    pub fn stage1_prove<F, ProofTranscript, PCS, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: jolt_core::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>,
        N: Rep3NetworkCoordinator,
    {
        let trace_length = state.trace_length;
        let padded_trace_length = trace_length.next_power_of_two();
        let key = UniformSpartanKey::<F>::new(padded_trace_length);

        let num_rounds_x = key.num_rows_bits();
        let tau: Vec<F::Challenge> = state
            .transcript
            .challenge_vector_optimized::<F>(num_rounds_x);
        network.broadcast_request(tau.clone())?;

        let mut eq_poly = GruenSplitEqPolynomial::new(&tau, BindingOrder::LowToHigh);

        // DEBUG: Receive and print first 8 rows' Az/Bz cleartext values from workers.
        {
            let debug_shares: Vec<Vec<AdditiveShare<F>>> = network.receive_responses()?;
            let debug_vals = additive::combine_additive_vec(debug_shares);
            let padded_constraints = key.padded_row_constraint_per_step();
            for row in 0..std::cmp::min(8, debug_vals.len() / 2) {
                let step = row / padded_constraints;
                let ci = row % padded_constraints;
                let az = debug_vals[2 * row];
                let bz = debug_vals[2 * row + 1];
                eprintln!("[MPC-AZ] row={row} step={step} ci={ci} az={az:?} bz={bz:?}");
            }
        }

        let mut r: Vec<F::Challenge> = Vec::with_capacity(num_rounds_x);
        let mut polys = Vec::with_capacity(num_rounds_x);
        let mut claim = F::zero();

        for _round in 0..num_rounds_x {
            let round_shares: Vec<(AdditiveShare<F>, AdditiveShare<F>)> =
                network.receive_responses()?;
            eyre::ensure!(round_shares.len() == 3, "expected 3 parties");

            let t0 = additive::combine_additive_share(round_shares.iter().map(|x| x.0).collect());
            let t_inf =
                additive::combine_additive_share(round_shares.into_iter().map(|x| x.1).collect());

            if _round < 3 {
                eprintln!("[COORD] round {_round}: t0={t0:?} t_inf={t_inf:?}");
            }

            let r_i = process_eq_sumcheck_round(
                (t0, t_inf),
                &mut eq_poly,
                &mut polys,
                &mut r,
                &mut claim,
                &mut state.transcript,
            );
            network.broadcast_request(r_i)?;
        }

        // Open final Az/Bz/Cz evals at the full outer sumcheck point.
        let final_shares: Vec<Vec<AdditiveShare<F>>> = network.receive_responses()?;
        let final_evals = additive::combine_additive_vec(final_shares);
        let [claim_az, claim_bz, claim_cz]: [F; 3] = final_evals.try_into().unwrap();

        // Insert Stage1 sumcheck proof.
        let proof = SumcheckInstanceProof::new(polys);
        state.proofs.insert(
            ProofKeys::Stage1Sumcheck,
            ProofData::SumcheckProof(proof),
        );

        // Outer sumcheck is bound from the "top"; reverse challenges to match vanilla.
        let mut outer_sumcheck_r: Vec<F::Challenge> = r.into_iter().rev().collect();

        // Append Az/Bz/Cz claims to transcript (matching vanilla ordering).
        for v in [claim_az, claim_bz, claim_cz] {
            state.transcript.append_scalar(&v);
        }

        // Store Az/Bz/Cz virtual openings (append again via accumulator, matching vanilla).
        let opening_point = jolt_core::poly::opening_proof::OpeningPoint::new(outer_sumcheck_r.clone());
        state.accumulator.append_virtual(
            &mut state.transcript,
            VirtualPolynomial::SpartanAz,
            jolt_core::poly::opening_proof::SumcheckId::SpartanOuter,
            opening_point.clone(),
            claim_az,
        );
        state.accumulator.append_virtual(
            &mut state.transcript,
            VirtualPolynomial::SpartanBz,
            jolt_core::poly::opening_proof::SumcheckId::SpartanOuter,
            opening_point.clone(),
            claim_bz,
        );
        state.accumulator.append_virtual(
            &mut state.transcript,
            VirtualPolynomial::SpartanCz,
            jolt_core::poly::opening_proof::SumcheckId::SpartanOuter,
            opening_point,
            claim_cz,
        );

        // Compute r_cycle (step variables) and receive claimed witness evaluations in ALL_R1CS_INPUTS order.
        let num_steps_bits = key.num_steps.log_2();
        let (r_cycle, _) = outer_sumcheck_r.split_at(num_steps_bits);

        let claimed_shares: Vec<Vec<AdditiveShare<F>>> = network.receive_responses()?;
        let claimed_witness_evals: Vec<F> = additive::combine_additive_vec(claimed_shares);
        eyre::ensure!(
            claimed_witness_evals.len() == ALL_R1CS_INPUTS.len(),
            "claimed witness eval len mismatch"
        );

        // Append committed openings (PCS).
        let committed_polys: Vec<CommittedPolynomial> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        let committed_claims: Vec<F> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();
        state.accumulator.append_dense(
            &mut state.transcript,
            committed_polys,
            jolt_core::poly::opening_proof::SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
            committed_claims,
        );

        // Append virtual openings for the remaining R1CS inputs.
        for input in ALL_R1CS_INPUTS.iter() {
            if COMMITTED_R1CS_INPUTS.contains(input) {
                continue;
            }
            let poly = VirtualPolynomial::try_from(input).ok().unwrap();
            let eval = claimed_witness_evals[input.to_index()];
            state.accumulator.append_virtual(
                &mut state.transcript,
                poly,
                jolt_core::poly::opening_proof::SumcheckId::SpartanOuter,
                jolt_core::poly::opening_proof::OpeningPoint::new(r_cycle.to_vec()),
                eval,
            );
        }

        Ok(())
    }
}
