use jolt_core::curve::Bn254Curve;
#[cfg(feature = "zk")]
use jolt_core::poly::commitment::commitment_scheme::ZkEvalCommitment;
#[cfg(feature = "zk")]
use jolt_core::poly::commitment::pedersen::PedersenGenerators;
#[cfg(feature = "zk")]
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
#[cfg(feature = "zk")]
use jolt_core::poly::opening_proof::{OpeningId, SumcheckId};
use jolt_core::poly::unipoly::CompressedUniPoly;
#[cfg(feature = "zk")]
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::subprotocols::sumcheck::{process_eq_sumcheck_round, SumcheckInstanceProof};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::instruction_lookups::D as LOOKUP_D;
use jolt_core::zkvm::r1cs::inputs::{ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS};
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use jolt_core::zkvm::witness::{compute_d_parameter, CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
#[cfg(feature = "zk")]
use rand::thread_rng;

use crate::subprotocols::sumcheck::Rep3SumcheckInstance;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::spartan::inner::Rep3InnerSumcheck;
use jolt_core::field::JoltField;
#[cfg(feature = "zk")]
use jolt_core::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource, ZkStageData,
};

pub struct Rep3SpartanDag;

impl Rep3SpartanDag {
    #[tracing::instrument(skip_all, name = "SpartanDag::stage1_prove")]
    pub fn stage1_prove<F, ProofTranscript, PCS, N>(
        state: &mut StateManager<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: jolt_core::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>
            + ZkEvalCommitment<Bn254Curve>,
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

        let mut eq_poly = GruenSplitEqPolynomial::<F>::new(&tau, BindingOrder::LowToHigh);

        let mut r: Vec<F::Challenge> = Vec::with_capacity(num_rounds_x);
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds_x);
        let mut claim = F::zero();
        #[cfg(feature = "zk")]
        let experimental_zk = std::env::var_os("CO_JOLT2_EXPERIMENTAL_DAG_ZK_SUMCHECKS").is_some();
        #[cfg(feature = "zk")]
        let pedersen_gens = experimental_zk.then(|| {
            let pcs_setup = state
                .pcs_setup
                .expect("StateManager::pcs_setup must be set for experimental DAG ZK sumchecks");
            let blindfold_pedersen_count = [
                4usize,
                LOOKUP_D + 1,
                compute_d_parameter(state.ram_K) + 1,
                state.preprocessing.shared.bytecode.d + 1,
            ]
            .into_iter()
            .max()
            .unwrap_or(1)
            .next_power_of_two();
            let (message_generators, blinding_generator) =
                PCS::zk_generators(pcs_setup, blindfold_pedersen_count)
                .expect("PCS does not support BlindFold generators");
            PedersenGenerators::<Bn254Curve>::new(message_generators, blinding_generator)
        });
        #[cfg(feature = "zk")]
        let mut zk_rng = experimental_zk.then(thread_rng);
        #[cfg(feature = "zk")]
        let mut round_commitments = Vec::with_capacity(num_rounds_x);
        #[cfg(feature = "zk")]
        let mut poly_coeffs = Vec::with_capacity(num_rounds_x);
        #[cfg(feature = "zk")]
        let mut blinding_factors = Vec::with_capacity(num_rounds_x);

        for _round in 0..num_rounds_x {
            let round_shares: Vec<(AdditiveShare<F>, AdditiveShare<F>)> =
                network.receive_responses()?;
            eyre::ensure!(round_shares.len() == 3, "expected 3 parties");

            let t0: F = round_shares.iter().map(|x| x.0.into_fe()).sum();
            let t_inf: F = round_shares.iter().map(|x| x.1.into_fe()).sum();
            #[cfg(feature = "zk")]
            let r_i = if let (Some(pedersen_gens), Some(zk_rng)) =
                (pedersen_gens.as_ref(), zk_rng.as_mut())
            {
                let scalar_times_w_i =
                    eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];
                let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
                    [
                        eq_poly.current_scalar - scalar_times_w_i,
                        scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
                    ],
                    t0,
                    t_inf,
                    claim,
                );

                let blinding = F::random(zk_rng);
                let commitment = pedersen_gens.commit(&cubic_poly.coeffs, &blinding);
                state.transcript.append_message(b"sumcheck_commitment");
                state.transcript.append_serializable(&commitment);

                let r_i = state.transcript.challenge_scalar_optimized::<F>();
                r.push(r_i);
                round_commitments.push(commitment);
                poly_coeffs.push(cubic_poly.coeffs.clone());
                blinding_factors.push(blinding);
                claim = cubic_poly.evaluate(&r_i);
                eq_poly.bind(r_i);
                r_i
            } else {
                process_eq_sumcheck_round(
                    (t0, t_inf),
                    &mut eq_poly,
                    &mut polys,
                    &mut r,
                    &mut claim,
                    &mut state.transcript,
                )
            };
            #[cfg(not(feature = "zk"))]
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

        // Outer sumcheck is bound from the "top"; reverse challenges to match vanilla.
        let outer_sumcheck_r: Vec<F::Challenge> = r.iter().rev().cloned().collect();

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
        #[cfg(feature = "zk")]
        let proof = if let (Some(pedersen_gens), Some(zk_rng)) =
            (pedersen_gens.as_ref(), zk_rng.as_mut())
        {
            state.accumulator.set_zk_mode(true);
            let opening_point =
                jolt_core::poly::opening_proof::OpeningPoint::new(outer_sumcheck_r.clone());
            state.accumulator.append_virtual(
                &mut state.transcript,
                VirtualPolynomial::SpartanAz,
                SumcheckId::SpartanOuter,
                opening_point.clone(),
                claim_az,
            );
            state.accumulator.append_virtual(
                &mut state.transcript,
                VirtualPolynomial::SpartanBz,
                SumcheckId::SpartanOuter,
                opening_point.clone(),
                claim_bz,
            );
            state.accumulator.append_virtual(
                &mut state.transcript,
                VirtualPolynomial::SpartanCz,
                SumcheckId::SpartanOuter,
                opening_point,
                claim_cz,
            );

            state.accumulator.append_dense(
                &mut state.transcript,
                committed_polys.clone(),
                SumcheckId::SpartanOuter,
                r_cycle.to_vec(),
                committed_claims.clone(),
            );

            for input in ALL_R1CS_INPUTS.iter() {
                if COMMITTED_R1CS_INPUTS.contains(input) {
                    continue;
                }
                let poly = VirtualPolynomial::try_from(input).ok().unwrap();
                let eval = claimed_witness_evals[input.to_index()];
                state.accumulator.append_virtual(
                    &mut state.transcript,
                    poly,
                    SumcheckId::SpartanOuter,
                    jolt_core::poly::opening_proof::OpeningPoint::new(r_cycle.to_vec()),
                    eval,
                );
            }

            let output_claim_values = state.accumulator.take_pending_claims();
            let output_claim_ids = state.accumulator.take_pending_claim_ids();
            state.accumulator.set_zk_mode(false);

            let committed_output_claims =
                pedersen_gens.commit_chunked(&output_claim_values, zk_rng);
            let (output_claims_commitments, output_claims_blindings): (Vec<_>, Vec<_>) =
                committed_output_claims.into_iter().unzip();
            state.transcript.append_message(b"output_claims_coms");
            output_claims_commitments
                .iter()
                .for_each(|commitment| state.transcript.append_serializable(commitment));

            let eq_eval = EqPolynomial::<F>::mle(&tau, &outer_sumcheck_r);
            state.blindfold_accumulator.push_stage_data(ZkStageData {
                initial_claim: F::zero(),
                round_commitments: round_commitments.clone(),
                poly_coeffs: poly_coeffs.clone(),
                blinding_factors: blinding_factors.clone(),
                challenges: r.clone(),
                batching_coefficients: vec![F::one()],
                output_constraints: vec![Some(Self::stage1_output_constraint())],
                constraint_challenge_values: vec![vec![eq_eval]],
                input_constraints: vec![InputClaimConstraint::default()],
                input_constraint_challenge_values: vec![Vec::new()],
                input_claim_scaling_exponents: vec![0],
                output_claims: output_claim_ids
                    .into_iter()
                    .zip(output_claim_values)
                    .collect(),
                output_claims_blindings,
                output_claims_commitments: output_claims_commitments.clone(),
            });

            SumcheckInstanceProof::new_zk(
                round_commitments,
                poly_coeffs
                    .iter()
                    .map(|coeffs| coeffs.len().saturating_sub(1))
                    .collect(),
                output_claims_commitments,
            )
        } else {
            SumcheckInstanceProof::new(polys)
        };
        #[cfg(not(feature = "zk"))]
        let proof = SumcheckInstanceProof::new(polys);
        state
            .proofs
            .insert(ProofKeys::Stage1Sumcheck, ProofData::SumcheckProof(proof));
        #[cfg(feature = "zk")]
        let proof_is_zk = matches!(
            state.proofs.get(&ProofKeys::Stage1Sumcheck),
            Some(ProofData::SumcheckProof(SumcheckInstanceProof::Zk(_)))
        );
        #[cfg(not(feature = "zk"))]
        let proof_is_zk = false;

        if !proof_is_zk {
            state
                .transcript
                .append_scalars(&[claim_az, claim_bz, claim_cz]);

            let opening_point =
                jolt_core::poly::opening_proof::OpeningPoint::new(outer_sumcheck_r.clone());
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

            state.accumulator.append_dense(
                &mut state.transcript,
                committed_polys,
                jolt_core::poly::opening_proof::SumcheckId::SpartanOuter,
                r_cycle.to_vec(),
                committed_claims,
            );

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
        }

        Ok(())
    }

    #[cfg(feature = "zk")]
    fn stage1_output_constraint() -> OutputClaimConstraint {
        OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(
                ValueSource::challenge(0),
                vec![
                    ValueSource::opening(OpeningId::Virtual(
                        VirtualPolynomial::SpartanAz,
                        SumcheckId::SpartanOuter,
                    )),
                    ValueSource::opening(OpeningId::Virtual(
                        VirtualPolynomial::SpartanBz,
                        SumcheckId::SpartanOuter,
                    )),
                ],
            ),
            ProductTerm::scaled(
                ValueSource::challenge(0),
                vec![
                    ValueSource::constant(-1),
                    ValueSource::opening(OpeningId::Virtual(
                        VirtualPolynomial::SpartanCz,
                        SumcheckId::SpartanOuter,
                    )),
                ],
            ),
        ])
    }

    #[tracing::instrument(skip_all, name = "SpartanDag::stage2_instances")]
    pub fn stage2_instances<F, ProofTranscript, PCS, N>(
        state: &mut StateManager<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: jolt_core::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>,
        N: Rep3NetworkCoordinator,
    {
        let inner = Rep3InnerSumcheck::new::<ProofTranscript, PCS>(state);

        // Broadcast init bundle (gamma, input_claim) to workers
        network.broadcast_request((inner.gamma(), inner.input_claim()))?;

        Ok(vec![Box::new(inner)])
    }
}
