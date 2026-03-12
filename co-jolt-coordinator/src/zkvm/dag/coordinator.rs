use std::collections::HashMap;

use crate::poly::commitment::Rep3CommitmentScheme;
use crate::subprotocols::sumcheck::{BatchedSumcheckInstance, HybridBatchedSumcheck};
use crate::zkvm::dag::stage::{Rep3JoltDagStages, SumcheckStagesCoordinator};
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::spartan::Rep3SpartanDag;
use jolt_core::curve::Bn254Curve;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
use jolt_core::poly::commitment::dory::DoryGlobals;
#[cfg(feature = "zk")]
use jolt_core::poly::commitment::pedersen::PedersenGenerators;
use jolt_core::poly::opening_proof::{OpeningId, ReducedOpeningProof};
use jolt_core::subprotocols::blindfold::OpeningProofData;
#[cfg(feature = "zk")]
use jolt_core::subprotocols::blindfold::{
    pedersen_generator_count_for_r1cs, BakedPublicInputs, BlindFoldProof, BlindFoldProver, BlindFoldVerifier,
    BlindFoldVerifierInput, BlindFoldWitness, ExtraConstraintWitness, FinalOutputWitness, InputClaimConstraint,
    OutputClaimConstraint, RelaxedR1CSInstance, RoundWitness, StageConfig, StageWitness, ValueSource,
    VerifierR1CSBuilder, ZkStageData,
};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::dag::proof_serialization::{Claims, JoltProof};
use jolt_core::zkvm::instruction_lookups::D as LOOKUP_D;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::MaybeShared;
#[cfg(feature = "zk")]
use rand::thread_rng;
use tracing::info_span;

/// Coordinator side of the MPC DAG prover.
///
/// Owns the Fiat-Shamir transcript, drives sumcheck rounds by broadcasting
/// challenges, receives evaluation shares from workers, and assembles the
/// final proof.
pub struct Rep3JoltDag;

impl Rep3JoltDag {
    #[tracing::instrument(skip_all, name = "JoltDag::prove")]
    pub fn prove<'a, F, ProofTranscript, PCS, N>(
        mut state: StateManager<'a, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<JoltProof<F, Bn254Curve, PCS, ProofTranscript>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript> + ZkEvalCommitment<Bn254Curve>,
        N: Rep3NetworkCoordinator,
    {
        // --- Receive trace_length from workers ---
        let trace_lengths: Vec<usize> = network.receive_responses()?;
        let trace_length = trace_lengths[0];
        eyre::ensure!(trace_lengths.iter().all(|&t| t == trace_length), "trace_length mismatch across parties");
        state.trace_length = trace_length;
        let padded_trace_length = trace_length.next_power_of_two();

        // --- Fiat-Shamir preamble ---
        state.fiat_shamir_preamble(trace_length);

        // --- Initialize DoryGlobals and AllCommittedPolynomials ---
        let ram_K = state.ram_K;
        let bytecode_d = state.preprocessing.shared.bytecode.d;
        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length);
        let _poly_guard = AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d);
        #[cfg(feature = "zk")]
        let mut experimental_zk_rng = Self::experimental_zk_sumchecks_enabled().then(thread_rng);

        // --- Receive, combine, and store commitments ---
        let _recv_commits = info_span!("receive_commitments").entered();
        Self::receive_commitments::<F, PCS, ProofTranscript, N>(&mut state, network)?;

        // --- Receive untrusted advice commitment from workers ---
        Self::receive_untrusted_advice_commitment::<F, PCS, ProofTranscript, N>(&mut state, network)?;

        // --- Append advice commitments to transcript (matching vanilla ordering) ---
        if let Some(ref untrusted_advice_commitment) = state.untrusted_advice_commitment {
            state.transcript.append_serializable(untrusted_advice_commitment);
        }

        if let Some(ref trusted_advice_commitment) = state.trusted_advice_commitment {
            state.transcript.append_serializable(trusted_advice_commitment);
        }
        drop(_recv_commits);

        Rep3SpartanDag::stage1_prove(&mut state, network)?;

        // -------------------------------------------------------------------
        // Stage 2: batched sumcheck
        // -------------------------------------------------------------------

        let _stage2 = info_span!("stage2_prove").entered();
        let mut stages = Rep3JoltDagStages;
        let stage2_hybrid: Vec<BatchedSumcheckInstance<F, ProofTranscript>> =
            stages.stage2_instances(&mut state, network)?;

        #[cfg(feature = "zk")]
        let (proof, _r_stage2) = if let Some(rng) = experimental_zk_rng.as_mut() {
            let pedersen_gens = Self::pedersen_generators::<F, PCS>(
                state.pcs_setup.expect("StateManager::pcs_setup must be set for experimental DAG ZK sumchecks"),
                Self::blindfold_pedersen_generator_count(&state),
            );
            let (proof, _r_stage2, zk_material) = HybridBatchedSumcheck::prove_zk(
                &stage2_hybrid,
                &mut state.accumulator,
                &mut state.transcript,
                network,
                &pedersen_gens,
                rng,
            )?;
            state.blindfold_accumulator.push_stage_data(Self::stage_data_from_instances(
                "stage2",
                &stage2_hybrid,
                &state.accumulator,
                zk_material,
            ));
            (proof, _r_stage2)
        } else {
            HybridBatchedSumcheck::prove(&stage2_hybrid, &mut state.accumulator, &mut state.transcript, network)?
        };
        #[cfg(not(feature = "zk"))]
        let (proof, _r_stage2) =
            HybridBatchedSumcheck::prove(&stage2_hybrid, &mut state.accumulator, &mut state.transcript, network)?;
        state.proofs.insert(ProofKeys::Stage2Sumcheck, ProofData::SumcheckProof(proof));
        drop(_stage2);

        // -------------------------------------------------------------------
        // Stage 3: batched sumcheck (secret + public instances)
        // -------------------------------------------------------------------

        let _stage3 = info_span!("stage3_prove").entered();
        let stage3_instances = stages.stage3_instances(&mut state, network)?;

        #[cfg(feature = "zk")]
        let (stage3_proof, _r_stage3) = if let Some(rng) = experimental_zk_rng.as_mut() {
            let pedersen_gens = Self::pedersen_generators::<F, PCS>(
                state.pcs_setup.expect("StateManager::pcs_setup must be set for experimental DAG ZK sumchecks"),
                Self::blindfold_pedersen_generator_count(&state),
            );
            let (proof, _r_stage3, zk_material) = HybridBatchedSumcheck::prove_zk(
                &stage3_instances,
                &mut state.accumulator,
                &mut state.transcript,
                network,
                &pedersen_gens,
                rng,
            )?;
            state.blindfold_accumulator.push_stage_data(Self::stage_data_from_instances(
                "stage3",
                &stage3_instances,
                &state.accumulator,
                zk_material,
            ));
            (proof, _r_stage3)
        } else {
            HybridBatchedSumcheck::prove(&stage3_instances, &mut state.accumulator, &mut state.transcript, network)?
        };
        #[cfg(not(feature = "zk"))]
        let (stage3_proof, _r_stage3) =
            HybridBatchedSumcheck::prove(&stage3_instances, &mut state.accumulator, &mut state.transcript, network)?;
        state.proofs.insert(ProofKeys::Stage3Sumcheck, ProofData::SumcheckProof(stage3_proof));
        drop(_stage3);

        // -------------------------------------------------------------------
        // Stage 4: batched sumcheck (RAM + Bytecode public, Lookups RA secret)
        // -------------------------------------------------------------------

        let _stage4 = info_span!("stage4_prove").entered();
        let stage4_instances = stages.stage4_instances(&mut state, network)?;

        if !stage4_instances.is_empty() {
            #[cfg(feature = "zk")]
            let (stage4_proof, _r_stage4) = if let Some(rng) = experimental_zk_rng.as_mut() {
                let pedersen_gens = Self::pedersen_generators::<F, PCS>(
                    state.pcs_setup.expect("StateManager::pcs_setup must be set for experimental DAG ZK sumchecks"),
                    Self::blindfold_pedersen_generator_count(&state),
                );
                let (proof, _r_stage4, zk_material) = HybridBatchedSumcheck::prove_zk(
                    &stage4_instances,
                    &mut state.accumulator,
                    &mut state.transcript,
                    network,
                    &pedersen_gens,
                    rng,
                )?;
                state.blindfold_accumulator.push_stage_data(Self::stage_data_from_instances(
                    "stage4",
                    &stage4_instances,
                    &state.accumulator,
                    zk_material,
                ));
                (proof, _r_stage4)
            } else {
                HybridBatchedSumcheck::prove(&stage4_instances, &mut state.accumulator, &mut state.transcript, network)?
            };
            #[cfg(not(feature = "zk"))]
            let (stage4_proof, _r_stage4) = HybridBatchedSumcheck::prove(
                &stage4_instances,
                &mut state.accumulator,
                &mut state.transcript,
                network,
            )?;
            state.proofs.insert(ProofKeys::Stage4Sumcheck, ProofData::SumcheckProof(stage4_proof));
        }
        drop(_stage4);

        // -------------------------------------------------------------------
        // Stage 5: opening proof reduction
        // -------------------------------------------------------------------

        let _stage5 = info_span!("stage5_reduce_and_prove").entered();
        let poly_keys: Vec<CommittedPolynomial> = AllCommittedPolynomials::iter().copied().collect();
        let mut commitment_map: HashMap<CommittedPolynomial, PCS::Commitment> =
            poly_keys.into_iter().zip(state.commitments.iter().cloned()).collect();

        let pcs_setup = state.pcs_setup.expect("StateManager::pcs_setup must be set for reduce_and_prove (stage5)");
        let reduced = state.accumulator.reduce_and_prove::<PCS, ProofTranscript, N>(
            &mut commitment_map,
            pcs_setup,
            &mut state.transcript,
            network,
        )?;
        state.stage5_y_blinding = reduced.y_blinding;
        #[cfg(feature = "zk")]
        if let Some(y_blinding) = reduced.y_blinding {
            state.blindfold_accumulator.set_opening_proof_data(OpeningProofData {
                opening_ids: reduced.opening_ids.clone(),
                constraint_coeffs: reduced.constraint_coeffs.clone(),
                joint_claim: reduced.joint_claim,
                y_blinding,
            });
        }
        #[cfg(feature = "zk")]
        let blindfold_proof = if Self::experimental_zk_sumchecks_enabled() {
            Some(Self::prove_blindfold::<F, ProofTranscript, PCS>(&mut state, &reduced.joint_opening_proof))
        } else {
            None
        };
        state.proofs.insert(
            ProofKeys::ReducedOpeningProof,
            ProofData::ReducedOpeningProof(ReducedOpeningProof::<F, Bn254Curve, PCS, ProofTranscript> {
                sumcheck_proof: reduced.sumcheck_proof,
                sumcheck_claims: reduced.sumcheck_claims,
                joint_opening_proof: reduced.joint_opening_proof,
            }),
        );
        drop(_stage5);

        // --- Construct JoltProof ---
        let proof = JoltProof {
            opening_claims: Claims(std::mem::take(&mut state.accumulator.openings)),
            commitments: std::mem::take(&mut state.commitments),
            proofs: std::mem::take(&mut state.proofs),
            #[cfg(feature = "zk")]
            blindfold_proof,
            untrusted_advice_commitment: state.untrusted_advice_commitment.take(),
            trace_length,
            ram_K: state.ram_K,
            bytecode_d: state.preprocessing.shared.bytecode.d,
            twist_sumcheck_switch_index: state.twist_sumcheck_switch_index,
        };
        Ok(proof)
    }

    #[cfg(feature = "zk")]
    fn experimental_zk_sumchecks_enabled() -> bool {
        std::env::var_os("CO_JOLT2_EXPERIMENTAL_DAG_ZK_SUMCHECKS").is_some()
    }

    #[cfg(feature = "zk")]
    fn pedersen_generators<F, PCS>(pcs_setup: &PCS::ProverSetup, count: usize) -> PedersenGenerators<Bn254Curve>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<Bn254Curve>,
    {
        let (message_generators, blinding_generator) =
            PCS::zk_generators(pcs_setup, count).expect("PCS does not support BlindFold generators");
        PedersenGenerators::new(message_generators, blinding_generator)
    }

    #[cfg(feature = "zk")]
    fn blindfold_pedersen_generator_count<F, ProofTranscript, PCS>(
        state: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> usize
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    {
        let max_coeffs =
            [4usize, LOOKUP_D + 1, compute_d_parameter(state.ram_K) + 1, state.preprocessing.shared.bytecode.d + 1]
                .into_iter()
                .max()
                .unwrap_or(1);
        max_coeffs.next_power_of_two()
    }

    #[cfg(feature = "zk")]
    fn stage_data_from_instances<F, ProofTranscript>(
        stage_label: &str,
        instances: &[BatchedSumcheckInstance<F, ProofTranscript>],
        accumulator: &crate::poly::opening_proof::Rep3OpeningAccumulator<F>,
        zk_material: crate::subprotocols::sumcheck::HybridZkProofMaterial<F>,
    ) -> ZkStageData<F, Bn254Curve>
    where
        F: JoltField,
        ProofTranscript: Transcript,
    {
        let max_num_rounds = instances.iter().map(BatchedSumcheckInstance::num_rounds).max().unwrap_or(0);
        let num_instances = instances.len();

        for (instance_idx, instance) in instances.iter().enumerate() {
            let constraint = instance.input_claim_constraint();
            if constraint.terms.is_empty() {
                assert_eq!(
                    instance.input_claim_public(),
                    F::zero(),
                    "BlindFold missing input-claim constraint in {stage_label} instance {}",
                    instance_idx + 1,
                );
                continue;
            }
            let challenge_values = instance.input_constraint_challenge_values(accumulator);
            let opening_values: Vec<F> =
                constraint.required_openings.iter().map(|id| accumulator.get_opening(*id)).collect();
            let expected_input_claim = constraint.evaluate(&opening_values, &challenge_values);
            assert_eq!(
                expected_input_claim,
                instance.input_claim_public(),
                "BlindFold input-claim constraint mismatch in {stage_label} instance {}",
                instance_idx + 1,
            );
        }

        ZkStageData {
            initial_claim: zk_material.initial_claim,
            round_commitments: zk_material.round_commitments,
            poly_coeffs: zk_material.poly_coeffs,
            blinding_factors: zk_material.blinding_factors,
            challenges: zk_material.challenges.clone(),
            batching_coefficients: zk_material.batching_coefficients,
            output_constraints: instances.iter().map(BatchedSumcheckInstance::output_claim_constraint).collect(),
            constraint_challenge_values: instances
                .iter()
                .map(|instance| {
                    let offset = max_num_rounds - instance.num_rounds();
                    let r_slice = &zk_material.challenges[offset..offset + instance.num_rounds()];
                    instance.output_constraint_challenge_values(r_slice)
                })
                .collect(),
            input_constraints: instances.iter().map(BatchedSumcheckInstance::input_claim_constraint).collect(),
            input_constraint_challenge_values: instances
                .iter()
                .map(|instance| instance.input_constraint_challenge_values(accumulator))
                .collect(),
            input_claim_scaling_exponents: instances
                .iter()
                .map(|instance| max_num_rounds - instance.num_rounds())
                .collect(),
            output_claims: zk_material.output_claims,
            output_claims_blindings: zk_material.output_claims_blindings,
            output_claims_commitments: zk_material.output_claims_commitments,
        }
    }

    #[cfg(feature = "zk")]
    fn prove_blindfold<F, ProofTranscript, PCS>(
        state: &mut StateManager<'_, F, ProofTranscript, PCS>,
        joint_opening_proof: &PCS::Proof,
    ) -> BlindFoldProof<F, Bn254Curve>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<Bn254Curve>,
    {
        let stage5_data = state.blindfold_accumulator.take_opening_proof_data();
        let zk_stages = state.blindfold_accumulator.take_stage_data();
        assert_eq!(zk_stages.len(), 4, "expected 4 DAG BlindFold stages");

        let mut stage_configs = Vec::new();
        let mut stage_witnesses = Vec::new();
        let mut initial_claims = Vec::with_capacity(zk_stages.len());
        let mut baked_challenges = Vec::new();
        let mut baked_output_challenges = Vec::new();
        let mut baked_input_challenges = Vec::new();
        let mut oc_blocks: Vec<Vec<OpeningId>> = Vec::with_capacity(zk_stages.len());

        for (stage_idx, zk_data) in zk_stages.iter().enumerate() {
            initial_claims.push(zk_data.initial_claim);
            oc_blocks.push(zk_data.output_claims.iter().map(|(id, _)| *id).collect());

            let mut current_claim = zk_data.initial_claim;
            let num_rounds = zk_data.poly_coeffs.len();
            for (round_idx, coeffs) in zk_data.poly_coeffs.iter().enumerate() {
                let challenge: F = zk_data.challenges[round_idx].into();
                let poly_degree = coeffs.len() - 1;
                let mut next_claim = coeffs[coeffs.len() - 1];
                for coeff in coeffs[..coeffs.len() - 1].iter().rev() {
                    next_claim = *coeff + challenge * next_claim;
                }
                let expected_claimed_sum = F::from_u64(2) * coeffs[0] + coeffs[1..].iter().copied().sum::<F>();
                assert_eq!(
                    expected_claimed_sum,
                    current_claim,
                    "BlindFold round claimed-sum mismatch at stage {} round {}",
                    stage_idx + 1,
                    round_idx + 1,
                );
                let round_witness = RoundWitness::with_claimed_sum(coeffs.clone(), challenge, current_claim);
                let mut config = if round_idx == 0 {
                    StageConfig::new_chain(1, poly_degree)
                } else {
                    StageConfig::new(1, poly_degree)
                };
                let initial_input = if round_idx == 0 {
                    let batched_constraint = InputClaimConstraint::batch_required(
                        &zk_data.input_constraints,
                        zk_data.batching_coefficients.len(),
                    );
                    let mut challenge_values: Vec<F> = zk_data
                        .batching_coefficients
                        .iter()
                        .zip(&zk_data.input_claim_scaling_exponents)
                        .map(|(alpha, &scale)| alpha.mul_pow_2(scale))
                        .collect();
                    for values in &zk_data.input_constraint_challenge_values {
                        challenge_values.extend_from_slice(values);
                    }
                    let opening_values: Vec<F> = batched_constraint
                        .required_openings
                        .iter()
                        .map(|id| state.accumulator.get_opening(*id))
                        .collect();
                    if !batched_constraint.terms.is_empty() {
                        let expected_initial_claim = batched_constraint.evaluate(&opening_values, &challenge_values);
                        assert_eq!(
                            expected_initial_claim,
                            current_claim,
                            "BlindFold initial-claim constraint mismatch at stage {} start",
                            stage_idx + 1,
                        );
                        config = config.with_input_constraint(batched_constraint);
                        baked_input_challenges.extend_from_slice(&challenge_values);
                        Some(FinalOutputWitness::general(challenge_values, opening_values))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let final_output = if round_idx == num_rounds - 1 {
                    if let Some(constraint) = OutputClaimConstraint::batch(&zk_data.output_constraints) {
                        let mut challenge_values = zk_data.batching_coefficients.clone();
                        for values in &zk_data.constraint_challenge_values {
                            challenge_values.extend_from_slice(values);
                        }
                        let opening_values: Vec<F> =
                            constraint.required_openings.iter().map(|id| state.accumulator.get_opening(*id)).collect();
                        let expected_output_claim = constraint.evaluate(&opening_values, &challenge_values);
                        assert_eq!(
                            expected_output_claim,
                            next_claim,
                            "BlindFold output-claim constraint mismatch at stage {} end",
                            stage_idx + 1,
                        );
                        baked_output_challenges.extend_from_slice(&challenge_values);
                        config = config.with_constraint(constraint);
                        Some(FinalOutputWitness::general(challenge_values, opening_values))
                    } else {
                        None
                    }
                } else {
                    None
                };

                stage_witnesses.push(match (initial_input, final_output) {
                    (Some(initial_input), Some(final_output)) => {
                        StageWitness::with_both(vec![round_witness], initial_input, final_output)
                    }
                    (Some(initial_input), None) => StageWitness::with_initial_input(vec![round_witness], initial_input),
                    (None, Some(final_output)) => StageWitness::with_final_output(vec![round_witness], final_output),
                    (None, None) => StageWitness::new(vec![round_witness]),
                });

                stage_configs.push(config);
                baked_challenges.push(challenge);
                current_claim = next_claim;
            }
        }

        let extra_constraints = vec![OutputClaimConstraint::linear(
            stage5_data
                .opening_ids
                .iter()
                .enumerate()
                .map(|(idx, opening_id)| (ValueSource::challenge(idx), ValueSource::opening(*opening_id)))
                .collect(),
        )];
        let baked = BakedPublicInputs {
            challenges: baked_challenges,
            initial_claims: initial_claims.clone(),
            batching_coefficients: Vec::new(),
            output_constraint_challenges: baked_output_challenges,
            input_constraint_challenges: baked_input_challenges,
            extra_constraint_challenges: stage5_data.constraint_coeffs.clone(),
        };
        let r1cs =
            VerifierR1CSBuilder::<F>::new_with_extra(&stage_configs, &extra_constraints, &baked, oc_blocks.clone())
                .build();

        let hyrax_c = r1cs.hyrax.C;
        let mut output_claims_values = Vec::new();
        let mut output_claims_commitments = Vec::new();
        let mut output_claims_blindings = Vec::new();
        for zk_data in &zk_stages {
            output_claims_commitments.extend_from_slice(&zk_data.output_claims_commitments);
            output_claims_blindings.extend_from_slice(&zk_data.output_claims_blindings);
            let vals: Vec<F> = zk_data.output_claims.iter().map(|(_, value)| *value).collect();
            output_claims_values.extend_from_slice(&vals);
            let block_rows = vals.len().div_ceil(hyrax_c.max(1));
            output_claims_values.resize(output_claims_values.len() + block_rows * hyrax_c - vals.len(), F::zero());
        }

        let extra_opening_values: Vec<F> =
            stage5_data.opening_ids.iter().map(|id| state.accumulator.get_opening(*id)).collect();
        let expected_stage5_joint_claim: F = stage5_data
            .constraint_coeffs
            .iter()
            .zip(extra_opening_values.iter())
            .map(|(coeff, opening)| *coeff * *opening)
            .sum();
        assert_eq!(
            expected_stage5_joint_claim,
            stage5_data.joint_claim,
            "DAG stage5 BlindFold linear relation must match the stored joint claim",
        );
        let extra_witness = ExtraConstraintWitness {
            output_value: stage5_data.joint_claim,
            blinding: stage5_data.y_blinding,
            challenge_values: stage5_data.constraint_coeffs.clone(),
            opening_values: extra_opening_values,
        };
        let blindfold_witness = BlindFoldWitness::with_output_claims(
            initial_claims,
            stage_witnesses,
            vec![extra_witness],
            output_claims_values,
        );
        let z = blindfold_witness.assign(&r1cs);
        if let Err(row) = r1cs.check_satisfaction(&z) {
            panic!("DAG BlindFold witness must satisfy the verifier R1CS at row {row}");
        }
        let witness: Vec<F> = z[1..].to_vec();

        let mut round_commitments = Vec::new();
        let mut round_blindings = Vec::new();
        for zk_data in &zk_stages {
            round_commitments.extend_from_slice(&zk_data.round_commitments);
            round_blindings.extend_from_slice(&zk_data.blinding_factors);
        }

        let pcs_setup = state.pcs_setup.expect("PCS setup must be present for DAG BlindFold proving");
        let pedersen_generators =
            Self::pedersen_generators::<F, PCS>(pcs_setup, pedersen_generator_count_for_r1cs(&r1cs));
        let eval_commitments = vec![PCS::eval_commitment(joint_opening_proof).expect("missing eval commitment")];

        let hyrax = &r1cs.hyrax;
        let regular_noncoeff_start = (hyrax.R_coeff + hyrax.output_claims_rows) * hyrax.C;
        let mut noncoeff_row_commitments = Vec::with_capacity(hyrax.regular_noncoeff_rows());
        let mut noncoeff_row_blindings = Vec::with_capacity(hyrax.regular_noncoeff_rows());
        let mut rng = thread_rng();
        for row_idx in 0..hyrax.regular_noncoeff_rows() {
            let row_start = regular_noncoeff_start + row_idx * hyrax.C;
            let mut row_data = vec![F::zero(); hyrax.C];
            for k in 0..hyrax.C {
                if row_start + k < witness.len() {
                    row_data[k] = witness[row_start + k];
                }
            }
            let blinding = F::random(&mut rng);
            noncoeff_row_commitments.push(pedersen_generators.commit(&row_data, &blinding));
            noncoeff_row_blindings.push(blinding);
        }

        let mut w_row_blindings = Vec::with_capacity(hyrax.R_prime);
        w_row_blindings.extend_from_slice(&round_blindings);
        w_row_blindings.resize(hyrax.R_coeff, F::zero());
        w_row_blindings.extend_from_slice(&output_claims_blindings);
        w_row_blindings.resize(hyrax.R_coeff + hyrax.output_claims_rows, F::zero());
        w_row_blindings.extend_from_slice(&noncoeff_row_blindings);
        w_row_blindings.resize(hyrax.R_prime, F::zero());

        assert_eq!(
            round_commitments.len(),
            zk_stages.iter().map(|stage| stage.round_commitments.len()).sum::<usize>(),
            "BlindFold round commitment count mismatch",
        );
        assert_eq!(
            round_commitments.len(),
            hyrax.total_rounds,
            "BlindFold round commitments must match total Hyrax coeff rows",
        );
        assert_eq!(
            output_claims_commitments.len(),
            hyrax.output_claims_rows,
            "BlindFold output-claim row commitment count mismatch",
        );
        assert_eq!(
            noncoeff_row_commitments.len(),
            hyrax.regular_noncoeff_rows(),
            "BlindFold non-coeff row commitment count mismatch",
        );
        assert_eq!(eval_commitments.len(), extra_constraints.len(), "BlindFold eval commitment count mismatch",);
        assert_eq!(w_row_blindings.len(), hyrax.R_prime, "BlindFold W-row blinding count mismatch",);

        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &witness,
            r1cs.num_constraints,
            hyrax.C,
            round_commitments,
            output_claims_commitments,
            noncoeff_row_commitments,
            eval_commitments,
            w_row_blindings,
        );
        let eval_commitment_gens = PCS::eval_commitment_gens(pcs_setup);

        let prover = BlindFoldProver::<_, _>::new(&pedersen_generators, &r1cs, eval_commitment_gens);
        let mut blindfold_transcript = ProofTranscript::new(b"BlindFold");
        let blindfold_proof = prover.prove(&real_instance, &real_witness, &z, &mut blindfold_transcript);

        let verifier = BlindFoldVerifier::<_, _>::new(&pedersen_generators, &r1cs, eval_commitment_gens);
        let mut blindfold_verify_transcript = ProofTranscript::new(b"BlindFold");
        verifier
            .verify(
                &blindfold_proof,
                &BlindFoldVerifierInput {
                    round_commitments: real_instance.round_commitments.clone(),
                    output_claims_row_commitments: real_instance.output_claims_row_commitments.clone(),
                    eval_commitments: real_instance.eval_commitments.clone(),
                },
                &mut blindfold_verify_transcript,
            )
            .expect("Coordinator-constructed DAG BlindFold proof must self-verify");

        blindfold_proof
    }

    fn receive_commitments<F, PCS, ProofTranscript, N>(
        state: &mut StateManager<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        // Receive commitment shares from all 3 parties
        let all_commitment_shares: Vec<Vec<MaybeShared<PCS::Commitment>>> = network.receive_responses()?;

        eyre::ensure!(
            all_commitment_shares.len() == 3,
            "expected commitment shares from 3 parties, got {}",
            all_commitment_shares.len()
        );
        let num_polys = all_commitment_shares[0].len();

        // Combine commitment shares
        let combined_commitments: Vec<PCS::Commitment> = (0..num_polys)
            .map(|i| {
                let shares: Vec<&MaybeShared<PCS::Commitment>> =
                    all_commitment_shares.iter().map(|party_shares| &party_shares[i]).collect();
                <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::combine_commitment_shares(&shares)
            })
            .collect();

        // Store commitments and append to transcript
        state.commitments = combined_commitments;
        for commitment in &state.commitments {
            state.transcript.append_serializable(commitment);
        }

        Ok(())
    }

    fn receive_untrusted_advice_commitment<F, PCS, ProofTranscript, N>(
        state: &mut StateManager<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        let commitments: Vec<Option<MaybeShared<PCS::Commitment>>> = network.receive_responses()?;
        eyre::ensure!(
            commitments.len() == 3,
            "expected untrusted advice commitment from 3 parties, got {}",
            commitments.len()
        );

        let present: Vec<MaybeShared<PCS::Commitment>> = commitments.into_iter().flatten().collect();
        state.untrusted_advice_commitment = if present.is_empty() {
            None
        } else {
            eyre::ensure!(present.len() == 3, "expected untrusted advice commitment shares from all 3 parties");
            let shares: Vec<&MaybeShared<PCS::Commitment>> = present.iter().collect();
            Some(<PCS as Rep3CommitmentScheme<F, ProofTranscript>>::combine_commitment_shares(&shares))
        };
        Ok(())
    }
}
