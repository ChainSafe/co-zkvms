use std::collections::HashMap;

use crate::curve::{Bn254Curve, JoltCurve};
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{
    pedersen_generator_count_for_r1cs, BakedPublicInputs, BlindFoldVerifier, BlindFoldVerifierInput,
    InputClaimConstraint, OutputClaimConstraint, StageConfig, ValueSource, VerifierR1CSBuilder,
};
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance, SumcheckInstanceProof};
use crate::transcripts::Transcript;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial};
use anyhow::Context;

#[cfg(feature = "zk")]
use super::verifier_dags::Stage1BlindfoldData;
use super::verifier_dags::{BytecodeDag, LookupsDag, RamDag, RegistersDag, SpartanDag};

pub enum JoltDAG {}

#[cfg(feature = "zk")]
struct StageBlindfoldVerifyData<F: JoltField> {
    batching_coefficients: Vec<F>,
    challenges: Vec<F::Challenge>,
    output_claim_ids: Vec<OpeningId>,
}

impl JoltDAG {
    #[tracing::instrument(skip_all, name = "JoltDAG::verify")]
    pub fn verify<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<Bn254Curve>,
    >(
        mut state_manager: StateManager<'a, F, Bn254Curve, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        state_manager.fiat_shamir_preamble();

        let ram_K = state_manager.ram_K;
        let bytecode_d = state_manager.get_verifier_data().0.shared.bytecode.d;
        let _guard = AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d);

        // Append commitments to transcript
        let commitments = state_manager.get_commitments();
        let transcript = state_manager.get_transcript();
        for commitment in commitments.borrow().iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = state_manager.untrusted_advice_commitment {
            transcript.borrow_mut().append_serializable(untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = state_manager.trusted_advice_commitment {
            transcript.borrow_mut().append_serializable(trusted_advice_commitment);
        }

        // Stage 1:
        let trace_length = state_manager.get_verifier_data().2;
        let padded_trace_length = trace_length.next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::default();
        let mut registers_dag = RegistersDag::default();
        let mut ram_dag = RamDag::new_verifier(&state_manager);
        let mut bytecode_dag = BytecodeDag::default();
        #[cfg(feature = "zk")]
        let stage1_blindfold = spartan_dag.stage1_verify(&mut state_manager).context("Stage 1")?;
        #[cfg(not(feature = "zk"))]
        spartan_dag.stage1_verify(&mut state_manager).context("Stage 1")?;

        // Stage 2:
        let stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_verifier_instances(&mut state_manager))
            .chain(registers_dag.stage2_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage2_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage2_verifier_instances(&mut state_manager))
            .collect();
        let stage2_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> =
            stage2_instances.iter().map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>).collect();

        let proofs = state_manager.proofs.borrow();
        let stage2_proof_data = proofs.get(&ProofKeys::Stage2Sumcheck).expect("Stage 2 sumcheck proof not found");
        let stage2_proof = match stage2_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 2"),
        };

        let transcript = state_manager.get_transcript();
        let opening_accumulator = state_manager.get_verifier_accumulator();
        let (stage2_batching_coeffs, r_stage2, stage2_output_claim_ids) = BatchedSumcheck::verify(
            stage2_proof,
            stage2_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 2")?;

        drop(proofs);

        // Stage 3:
        let stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_verifier_instances(&mut state_manager))
            .chain(registers_dag.stage3_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage3_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage3_verifier_instances(&mut state_manager))
            .collect();
        let stage3_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> =
            stage3_instances.iter().map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>).collect();

        let proofs = state_manager.proofs.borrow();
        let stage3_proof_data = proofs.get(&ProofKeys::Stage3Sumcheck).expect("Stage 3 sumcheck proof not found");
        let stage3_proof = match stage3_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 3"),
        };

        let (stage3_batching_coeffs, r_stage3, stage3_output_claim_ids) = BatchedSumcheck::verify(
            stage3_proof,
            stage3_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 3")?;

        drop(proofs);

        // Stage 4:
        let stage4_instances: Vec<_> = std::iter::empty()
            .chain(ram_dag.stage4_verifier_instances(&mut state_manager))
            .chain(bytecode_dag.stage4_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage4_verifier_instances(&mut state_manager))
            .collect();
        let stage4_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> =
            stage4_instances.iter().map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>).collect();

        let proofs = state_manager.proofs.borrow();
        let stage4_proof_data = proofs.get(&ProofKeys::Stage4Sumcheck).expect("Stage 4 sumcheck proof not found");
        let stage4_proof = match stage4_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let (stage4_batching_coeffs, r_stage4, stage4_output_claim_ids) = BatchedSumcheck::verify(
            stage4_proof,
            stage4_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 4")?;

        // Verify trusted_advice opening proofs
        if state_manager.trusted_advice_commitment.is_some() {
            Self::verify_trusted_advice_proofs(
                &state_manager,
                &state_manager.preprocessing.generators,
                &mut *transcript.borrow_mut(),
            )
            .context("Stage 5")?;
        }

        // Verify untrusted_advice opening proofs
        if state_manager.untrusted_advice_commitment.is_some() {
            Self::verify_untrusted_advice_proofs(
                &state_manager,
                &state_manager.preprocessing.generators,
                &mut *transcript.borrow_mut(),
            )
            .context("Stage 5")?;
        }

        {
            // Batch-prove all openings
            let batched_opening_proof =
                proofs.get(&ProofKeys::ReducedOpeningProof).expect("Reduced opening proof not found");
            let batched_opening_proof = match batched_opening_proof {
                ProofData::ReducedOpeningProof(proof) => proof,
                _ => panic!("Invalid proof type for stage 4"),
            };
            let stage5_eval_commitment = PCS::eval_commitment(&batched_opening_proof.joint_opening_proof)
                .ok_or_else(|| anyhow::anyhow!("missing eval commitment"))?;

            let mut commitments_map = HashMap::new();
            for polynomial in AllCommittedPolynomials::iter() {
                commitments_map.insert(*polynomial, commitments.borrow()[polynomial.to_index()].clone());
            }
            let accumulator = state_manager.get_verifier_accumulator();
            accumulator
                .borrow_mut()
                .reduce_and_verify(
                    &state_manager.preprocessing.generators,
                    &mut commitments_map,
                    batched_opening_proof,
                    &mut *transcript.borrow_mut(),
                )
                .context("Stage 5")?;
            drop(proofs);

            #[cfg(feature = "zk")]
            if state_manager.blindfold_proof.is_some() {
                Self::verify_blindfold::<F, ProofTranscript, PCS>(
                    &mut state_manager,
                    stage1_blindfold,
                    &stage2_instances,
                    StageBlindfoldVerifyData {
                        batching_coefficients: stage2_batching_coeffs,
                        challenges: r_stage2,
                        output_claim_ids: stage2_output_claim_ids,
                    },
                    &stage3_instances,
                    StageBlindfoldVerifyData {
                        batching_coefficients: stage3_batching_coeffs,
                        challenges: r_stage3,
                        output_claim_ids: stage3_output_claim_ids,
                    },
                    &stage4_instances,
                    StageBlindfoldVerifyData {
                        batching_coefficients: stage4_batching_coeffs,
                        challenges: r_stage4,
                        output_claim_ids: stage4_output_claim_ids,
                    },
                    stage5_eval_commitment,
                )
                .context("BlindFold")?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "zk")]
    #[allow(clippy::too_many_arguments)]
    fn verify_blindfold<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<Bn254Curve>,
    >(
        state_manager: &mut StateManager<'a, F, Bn254Curve, ProofTranscript, PCS>,
        stage1_data: Stage1BlindfoldData<F>,
        stage2_instances: &[Box<dyn SumcheckInstance<F, ProofTranscript>>],
        stage2_data: StageBlindfoldVerifyData<F>,
        stage3_instances: &[Box<dyn SumcheckInstance<F, ProofTranscript>>],
        stage3_data: StageBlindfoldVerifyData<F>,
        stage4_instances: &[Box<dyn SumcheckInstance<F, ProofTranscript>>],
        stage4_data: StageBlindfoldVerifyData<F>,
        stage5_eval_commitment: <Bn254Curve as JoltCurve>::G1,
    ) -> Result<(), anyhow::Error> {
        let opening_data = state_manager
            .get_verifier_accumulator()
            .borrow_mut()
            .take_blindfold_opening_data()
            .ok_or_else(|| anyhow::anyhow!("missing BlindFold opening reduction data"))?;

        let stage2_input_constraint = InputClaimConstraint::batch_required(
            &stage2_instances.iter().map(|instance| instance.input_claim_constraint()).collect::<Vec<_>>(),
            stage2_data.batching_coefficients.len(),
        );
        let stage3_input_constraint = InputClaimConstraint::batch_required(
            &stage3_instances.iter().map(|instance| instance.input_claim_constraint()).collect::<Vec<_>>(),
            stage3_data.batching_coefficients.len(),
        );
        let stage4_input_constraint = InputClaimConstraint::batch_required(
            &stage4_instances.iter().map(|instance| instance.input_claim_constraint()).collect::<Vec<_>>(),
            stage4_data.batching_coefficients.len(),
        );
        let stage2_output_constraint = OutputClaimConstraint::batch(
            &stage2_instances.iter().map(|instance| instance.output_claim_constraint()).collect::<Vec<_>>(),
        );
        let stage3_output_constraint = OutputClaimConstraint::batch(
            &stage3_instances.iter().map(|instance| instance.output_claim_constraint()).collect::<Vec<_>>(),
        );
        let stage4_output_constraint = OutputClaimConstraint::batch(
            &stage4_instances.iter().map(|instance| instance.output_claim_constraint()).collect::<Vec<_>>(),
        );

        let stage2_input_challenges = Self::batched_input_challenge_values(
            stage2_instances,
            &stage2_data,
            state_manager.get_verifier_accumulator(),
        );
        let stage3_input_challenges = Self::batched_input_challenge_values(
            stage3_instances,
            &stage3_data,
            state_manager.get_verifier_accumulator(),
        );
        let stage4_input_challenges = Self::batched_input_challenge_values(
            stage4_instances,
            &stage4_data,
            state_manager.get_verifier_accumulator(),
        );
        let stage2_output_challenges = Self::batched_output_challenge_values(stage2_instances, &stage2_data);
        let stage3_output_challenges = Self::batched_output_challenge_values(stage3_instances, &stage3_data);
        let stage4_output_challenges = Self::batched_output_challenge_values(stage4_instances, &stage4_data);

        let proofs = state_manager.proofs.borrow();
        let stage1_proof = Self::sumcheck_proof(&proofs, ProofKeys::Stage1Sumcheck)?;
        let stage2_proof = Self::sumcheck_proof(&proofs, ProofKeys::Stage2Sumcheck)?;
        let stage3_proof = Self::sumcheck_proof(&proofs, ProofKeys::Stage3Sumcheck)?;
        let stage4_proof = Self::sumcheck_proof(&proofs, ProofKeys::Stage4Sumcheck)?;

        let stage1_eq_eval = crate::poly::eq_poly::EqPolynomial::<F>::mle(
            &stage1_data.tau,
            &stage1_data.challenges.iter().rev().copied().collect::<Vec<_>>(),
        );

        let stage_proofs = [stage1_proof, stage2_proof, stage3_proof, stage4_proof];
        let blindfold_hyrax_c = stage_proofs
            .iter()
            .map(|proof| match proof {
                SumcheckInstanceProof::Zk(zk_proof) => {
                    zk_proof.poly_degrees.iter().map(|degree| degree + 1).max().unwrap_or(1)
                }
                SumcheckInstanceProof::Clear(_) => 1,
            })
            .max()
            .unwrap_or(1)
            .next_power_of_two();
        let stage_batching_coeffs = [
            vec![F::one()],
            stage2_data.batching_coefficients.clone(),
            stage3_data.batching_coefficients.clone(),
            stage4_data.batching_coefficients.clone(),
        ];
        let stage_input_constraints = [
            InputClaimConstraint::default(),
            stage2_input_constraint,
            stage3_input_constraint,
            stage4_input_constraint,
        ];
        let stage_input_challenges =
            [vec![], stage2_input_challenges, stage3_input_challenges, stage4_input_challenges];
        let stage_output_constraints = [
            Some(Self::stage1_output_constraint()),
            stage2_output_constraint,
            stage3_output_constraint,
            stage4_output_constraint,
        ];
        let stage_output_challenges = [
            Some(vec![F::one(), stage1_eq_eval]),
            stage2_output_challenges,
            stage3_output_challenges,
            stage4_output_challenges,
        ];
        let stage_sumcheck_challenges = [
            stage1_data.challenges.clone(),
            stage2_data.challenges.clone(),
            stage3_data.challenges.clone(),
            stage4_data.challenges.clone(),
        ];
        let oc_blocks = vec![
            stage1_data.output_claim_ids,
            stage2_data.output_claim_ids,
            stage3_data.output_claim_ids,
            stage4_data.output_claim_ids,
        ];
        let initial_claims = vec![
            F::zero(),
            Self::batched_initial_claim(stage2_instances, &stage2_data.batching_coefficients),
            Self::batched_initial_claim(stage3_instances, &stage3_data.batching_coefficients),
            Self::batched_initial_claim(stage4_instances, &stage4_data.batching_coefficients),
        ];

        let mut stage_configs = Vec::new();
        let mut baked_challenges = Vec::new();
        let mut baked_input_challenges = Vec::new();
        let mut baked_output_challenges = Vec::new();
        let mut round_commitments = Vec::new();
        let mut output_claims_row_commitments = Vec::new();

        for stage_idx in 0..4 {
            let zk_proof = match stage_proofs[stage_idx] {
                SumcheckInstanceProof::Zk(zk_proof) => zk_proof,
                SumcheckInstanceProof::Clear(_) => {
                    return Err(anyhow::anyhow!("BlindFold requires ZK DAG sumcheck proofs"));
                }
            };
            let oc_block_rows = oc_blocks[stage_idx].len().div_ceil(blindfold_hyrax_c);
            anyhow::ensure!(
                zk_proof.output_claims_commitments.len() == oc_block_rows,
                "BlindFold stage {} OC rows mismatch: proof has {}, verifier reconstructed {} rows from {} claims",
                stage_idx + 1,
                zk_proof.output_claims_commitments.len(),
                oc_block_rows,
                oc_blocks[stage_idx].len(),
            );

            round_commitments.extend_from_slice(&zk_proof.round_commitments);
            output_claims_row_commitments.extend_from_slice(&zk_proof.output_claims_commitments);
            baked_input_challenges.extend_from_slice(&stage_input_challenges[stage_idx]);
            if let Some(values) = &stage_output_challenges[stage_idx] {
                baked_output_challenges.extend_from_slice(values);
            }

            for (round_idx, poly_degree) in zk_proof.poly_degrees.iter().copied().enumerate() {
                let mut config = if round_idx == 0 {
                    StageConfig::new_chain(1, poly_degree)
                } else {
                    StageConfig::new(1, poly_degree)
                };
                if round_idx == 0 && !stage_input_constraints[stage_idx].terms.is_empty() {
                    config = config.with_input_constraint(stage_input_constraints[stage_idx].clone());
                }
                if round_idx + 1 == zk_proof.poly_degrees.len() {
                    if let Some(constraint) = &stage_output_constraints[stage_idx] {
                        config = config.with_constraint(constraint.clone());
                    }
                }
                stage_configs.push(config);
                baked_challenges.push(stage_sumcheck_challenges[stage_idx][round_idx].into());
            }
        }

        let extra_constraint = OutputClaimConstraint::linear(
            opening_data
                .opening_ids
                .iter()
                .enumerate()
                .map(|(idx, opening_id)| (ValueSource::challenge(idx), ValueSource::opening(*opening_id)))
                .collect(),
        );
        let baked = BakedPublicInputs {
            challenges: baked_challenges,
            initial_claims,
            batching_coefficients: Vec::new(),
            output_constraint_challenges: baked_output_challenges,
            input_constraint_challenges: baked_input_challenges,
            extra_constraint_challenges: opening_data.constraint_coeffs.clone(),
        };
        let oc_block_lens: Vec<usize> = oc_blocks.iter().map(|b| b.len()).collect();
        let r1cs =
            VerifierR1CSBuilder::<F>::new_with_extra(&stage_configs, &[extra_constraint], &baked, oc_blocks).build();
        drop(proofs);

        let verifier_input = BlindFoldVerifierInput {
            round_commitments,
            output_claims_row_commitments,
            eval_commitments: vec![stage5_eval_commitment],
        };
        let blindfold_proof =
            state_manager.blindfold_proof.as_ref().ok_or_else(|| anyhow::anyhow!("missing blindfold proof"))?;
        let (expected_e_rows, _) = r1cs.hyrax.e_grid(r1cs.num_constraints);
        anyhow::ensure!(
            verifier_input.round_commitments.len() == r1cs.hyrax.total_rounds,
            "BlindFold verifier round commitments mismatch: got {}, expected {}",
            verifier_input.round_commitments.len(),
            r1cs.hyrax.total_rounds,
        );
        anyhow::ensure!(
            verifier_input.output_claims_row_commitments.len() == r1cs.hyrax.output_claims_rows,
            "BlindFold verifier OC row commitments mismatch: got {}, expected {}",
            verifier_input.output_claims_row_commitments.len(),
            r1cs.hyrax.output_claims_rows,
        );
        anyhow::ensure!(
            blindfold_proof.noncoeff_row_commitments.len() == r1cs.hyrax.regular_noncoeff_rows(),
            "BlindFold proof noncoeff rows mismatch: got {}, expected {}",
            blindfold_proof.noncoeff_row_commitments.len(),
            r1cs.hyrax.regular_noncoeff_rows(),
        );
        anyhow::ensure!(
            blindfold_proof.random_instance.output_claims_row_commitments.len() == r1cs.hyrax.output_claims_rows,
            "BlindFold random instance OC rows mismatch: got {}, expected {}",
            blindfold_proof.random_instance.output_claims_row_commitments.len(),
            r1cs.hyrax.output_claims_rows,
        );
        anyhow::ensure!(
            blindfold_proof.random_instance.noncoeff_row_commitments.len() == r1cs.hyrax.regular_noncoeff_rows(),
            "BlindFold random instance noncoeff rows mismatch: got {}, expected {}",
            blindfold_proof.random_instance.noncoeff_row_commitments.len(),
            r1cs.hyrax.regular_noncoeff_rows(),
        );
        anyhow::ensure!(
            blindfold_proof.random_instance.e_row_commitments.len() == expected_e_rows,
            "BlindFold random instance E rows mismatch: got {}, expected {}",
            blindfold_proof.random_instance.e_row_commitments.len(),
            expected_e_rows,
        );
        let pedersen_generators =
            state_manager.preprocessing.pedersen_generators(pedersen_generator_count_for_r1cs(&r1cs));
        let eval_commitment_gens = PCS::eval_commitment_gens_verifier(&state_manager.preprocessing.generators);
        let verifier = BlindFoldVerifier::<_, _>::new(&pedersen_generators, &r1cs, eval_commitment_gens);
        let mut blindfold_transcript = ProofTranscript::new(b"BlindFold");
        verifier
            .verify(blindfold_proof, &verifier_input, &mut blindfold_transcript)
            .map_err(|err| anyhow::anyhow!("BlindFold verification failed: {err:?}"))
    }

    #[cfg(feature = "zk")]
    fn sumcheck_proof<'a, F, C, PCS, ProofTranscript>(
        proofs: &'a crate::zkvm::dag::state_manager::Proofs<F, C, PCS, ProofTranscript>,
        key: ProofKeys,
    ) -> Result<&'a SumcheckInstanceProof<F, C, ProofTranscript>, anyhow::Error>
    where
        F: JoltField,
        C: JoltCurve,
        PCS: CommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        match proofs.get(&key).ok_or_else(|| anyhow::anyhow!("missing sumcheck proof for {key:?}"))? {
            ProofData::SumcheckProof(proof) => Ok(proof),
            _ => Err(anyhow::anyhow!("invalid proof type for {key:?}")),
        }
    }

    #[cfg(feature = "zk")]
    fn batched_input_challenge_values<F, ProofTranscript>(
        instances: &[Box<dyn SumcheckInstance<F, ProofTranscript>>],
        stage_data: &StageBlindfoldVerifyData<F>,
        opening_accumulator: std::rc::Rc<std::cell::RefCell<crate::poly::opening_proof::VerifierOpeningAccumulator<F>>>,
    ) -> Vec<F>
    where
        F: JoltField,
        ProofTranscript: Transcript,
    {
        let max_num_rounds = instances.iter().map(|instance| instance.num_rounds()).max().unwrap();
        let mut values: Vec<F> = stage_data
            .batching_coefficients
            .iter()
            .zip(instances.iter())
            .map(|(alpha, instance)| alpha.mul_pow_2(max_num_rounds - instance.num_rounds()))
            .collect();
        for instance in instances {
            values.extend(instance.input_constraint_challenge_values(Some(opening_accumulator.clone())));
        }
        values
    }

    #[cfg(feature = "zk")]
    fn batched_output_challenge_values<F, ProofTranscript>(
        instances: &[Box<dyn SumcheckInstance<F, ProofTranscript>>],
        stage_data: &StageBlindfoldVerifyData<F>,
    ) -> Option<Vec<F>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
    {
        let constraints: Vec<_> = instances.iter().map(|instance| instance.output_claim_constraint()).collect();
        OutputClaimConstraint::batch(&constraints)?;

        let max_num_rounds = instances.iter().map(|instance| instance.num_rounds()).max().unwrap();
        let mut values = stage_data.batching_coefficients.clone();
        for instance in instances {
            let offset = max_num_rounds - instance.num_rounds();
            values
                .extend(instance.output_constraint_challenge_values(
                    &stage_data.challenges[offset..offset + instance.num_rounds()],
                ));
        }
        Some(values)
    }

    #[cfg(feature = "zk")]
    fn batched_initial_claim<F, ProofTranscript>(
        instances: &[Box<dyn SumcheckInstance<F, ProofTranscript>>],
        batching_coefficients: &[F],
    ) -> F
    where
        F: JoltField,
        ProofTranscript: Transcript,
    {
        let max_num_rounds = instances.iter().map(|instance| instance.num_rounds()).max().unwrap_or(0);
        instances
            .iter()
            .zip(batching_coefficients.iter())
            .map(|(instance, coeff)| instance.input_claim().mul_pow_2(max_num_rounds - instance.num_rounds()) * coeff)
            .sum()
    }

    #[cfg(feature = "zk")]
    fn stage1_output_constraint() -> OutputClaimConstraint {
        OutputClaimConstraint::batch(&[Some(OutputClaimConstraint::sum_of_products(vec![
            crate::subprotocols::blindfold::ProductTerm::scaled(
                ValueSource::challenge(0),
                vec![
                    ValueSource::opening(OpeningId::Virtual(
                        crate::zkvm::witness::VirtualPolynomial::SpartanAz,
                        crate::poly::opening_proof::SumcheckId::SpartanOuter,
                    )),
                    ValueSource::opening(OpeningId::Virtual(
                        crate::zkvm::witness::VirtualPolynomial::SpartanBz,
                        crate::poly::opening_proof::SumcheckId::SpartanOuter,
                    )),
                ],
            ),
            crate::subprotocols::blindfold::ProductTerm::scaled(
                ValueSource::challenge(0),
                vec![
                    ValueSource::constant(-1),
                    ValueSource::opening(OpeningId::Virtual(
                        crate::zkvm::witness::VirtualPolynomial::SpartanCz,
                        crate::poly::opening_proof::SumcheckId::SpartanOuter,
                    )),
                ],
            ),
        ]))])
        .expect("single stage1 constraint should batch")
    }

    fn verify_trusted_advice_proofs<
        F: JoltField,
        C: JoltCurve,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, C, ProofTranscript, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut ProofTranscript,
    ) -> Result<(), anyhow::Error> {
        let trusted_advice_commitment = state_manager.trusted_advice_commitment.as_ref().unwrap();
        let accumulator = state_manager.get_verifier_accumulator();

        let (point, eval) = accumulator.borrow().get_trusted_advice_opening().unwrap();
        let proof = match state_manager.proofs.borrow().get(&ProofKeys::TrustedAdviceProof) {
            Some(ProofData::OpeningProof(proof)) => proof.clone(),
            _ => return Err(anyhow::anyhow!("Trusted advice proof not found")),
        };

        PCS::verify(&proof, verifier_setup, transcript, &point.r, &eval, trusted_advice_commitment)
            .map_err(|e| anyhow::anyhow!("Trusted advice opening proof verification failed: {e:?}"))?;

        Ok(())
    }

    fn verify_untrusted_advice_proofs<
        F: JoltField,
        C: JoltCurve,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, C, ProofTranscript, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut ProofTranscript,
    ) -> Result<(), anyhow::Error> {
        use crate::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
        use crate::utils::math::Math;
        use crate::zkvm::witness::VirtualPolynomial;

        let untrusted_advice_commitment = state_manager.untrusted_advice_commitment.as_ref().unwrap();
        let accumulator = state_manager.get_verifier_accumulator();

        // Reconstruct the advice opening point from the RamVal sumcheck point.
        // The serialized proof only stores the claim (scalar), not the opening
        // point, so the verifier must recompute it.
        let ws = common::constants::RAM_WORD_SIZE as usize;
        let max_size = state_manager.program_io.memory_layout.max_untrusted_advice_size as usize / ws;
        let log_advice_size = max_size.next_power_of_two().log_2();
        let total_memory_vars = state_manager.ram_K.log_2();
        let (r_val_point, _) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamVal, SumcheckId::RamReadWriteChecking);
        let r_address = &r_val_point.r[..total_memory_vars];
        let high_bits = total_memory_vars - log_advice_size;
        let advice_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(r_address[high_bits..].to_vec());

        // Directly populate the opening point in the accumulator without touching
        // the Fiat-Shamir transcript.  The prover (coordinator) inserts into
        // `openings` directly and never calls `append_untrusted_advice`, so the
        // verifier must match.
        {
            use crate::poly::opening_proof::OpeningId;
            let mut acc = accumulator.borrow_mut();
            if let Some((point, _)) = acc.openings.get_mut(&OpeningId::UntrustedAdvice) {
                *point = advice_opening_point;
            } else {
                acc.openings.insert(
                    OpeningId::UntrustedAdvice,
                    (advice_opening_point, F::zero()),
                );
            }
        }

        let (point, eval) = accumulator.borrow().get_untrusted_advice_opening().unwrap();
        let proof = match state_manager.proofs.borrow().get(&ProofKeys::UntrustedAdviceProof) {
            Some(ProofData::OpeningProof(proof)) => proof.clone(),
            _ => return Err(anyhow::anyhow!("Untrusted advice proof not found")),
        };

        PCS::verify(&proof, verifier_setup, transcript, &point.r, &eval, untrusted_advice_commitment)
            .map_err(|e| anyhow::anyhow!("Untrusted advice opening proof verification failed: {e:?}"))?;

        Ok(())
    }
}

#[cfg(all(test, feature = "zk"))]
mod tests {
    use super::JoltDAG;
    use crate::curve::Bn254Curve;
    use crate::field::JoltField;
    use crate::poly::commitment::pedersen::PedersenGenerators;
    use crate::poly::opening_proof::{OpeningId, SumcheckId};
    use crate::subprotocols::blindfold::{
        pedersen_generator_count_for_r1cs, BakedPublicInputs, BlindFoldProver, BlindFoldVerifier,
        BlindFoldVerifierInput, BlindFoldWitness, ExtraConstraintWitness, OutputClaimConstraint, RelaxedR1CSInstance,
        RelaxedR1CSWitness, ValueSource, VerifierR1CSBuilder,
    };
    use crate::transcripts::KeccakTranscript;
    use crate::zkvm::witness::CommittedPolynomial;
    use ark_bn254::Fr;
    use rand::thread_rng;

    type F = Fr;

    fn stage5_test_r1cs(
        opening_ids: &[OpeningId],
        constraint_coeffs: &[F],
    ) -> crate::subprotocols::blindfold::VerifierR1CS<F> {
        let extra_constraint = OutputClaimConstraint::linear(
            opening_ids
                .iter()
                .enumerate()
                .map(|(idx, opening_id)| (ValueSource::challenge(idx), ValueSource::opening(*opening_id)))
                .collect(),
        );
        let baked = BakedPublicInputs {
            challenges: Vec::new(),
            initial_claims: vec![F::zero()],
            batching_coefficients: Vec::new(),
            output_constraint_challenges: Vec::new(),
            input_constraint_challenges: Vec::new(),
            extra_constraint_challenges: constraint_coeffs.to_vec(),
        };

        VerifierR1CSBuilder::<F>::new_with_extra(&[], &[extra_constraint], &baked, vec![]).build()
    }

    fn make_stage5_test_instance(
        opening_values: &[F],
        constraint_coeffs: &[F],
        joint_claim: F,
        y_blinding: F,
    ) -> (
        crate::subprotocols::blindfold::VerifierR1CS<F>,
        PedersenGenerators<Bn254Curve>,
        RelaxedR1CSInstance<F, Bn254Curve>,
        RelaxedR1CSWitness<F>,
        Vec<F>,
        BlindFoldVerifierInput<Bn254Curve>,
        (<Bn254Curve as crate::curve::JoltCurve>::G1, <Bn254Curve as crate::curve::JoltCurve>::G1),
    ) {
        let opening_ids = vec![
            OpeningId::Committed(CommittedPolynomial::Bytecode, SumcheckId::OpeningReduction),
            OpeningId::Committed(CommittedPolynomial::ReadWriteMemory, SumcheckId::OpeningReduction),
        ];
        let r1cs = stage5_test_r1cs(&opening_ids, constraint_coeffs);
        let pedersen_generators =
            PedersenGenerators::<Bn254Curve>::deterministic(pedersen_generator_count_for_r1cs(&r1cs));
        let eval_commitment_gens = (pedersen_generators.message_generators[0], pedersen_generators.blinding_generator);
        let blindfold_witness = BlindFoldWitness::with_output_claims(
            vec![],
            vec![],
            vec![ExtraConstraintWitness {
                output_value: joint_claim,
                blinding: y_blinding,
                challenge_values: constraint_coeffs.to_vec(),
                opening_values: opening_values.to_vec(),
            }],
            Vec::new(),
        );

        let z = blindfold_witness.assign(&r1cs);
        r1cs.check_satisfaction(&z).unwrap();
        let witness: Vec<F> = z[1..].to_vec();

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
        w_row_blindings.extend_from_slice(&noncoeff_row_blindings);
        w_row_blindings.resize(hyrax.R_prime, F::zero());

        let eval_commitment =
            eval_commitment_gens.0.scalar_mul(&joint_claim) + eval_commitment_gens.1.scalar_mul(&y_blinding);
        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &witness,
            r1cs.num_constraints,
            hyrax.C,
            Vec::new(),
            Vec::new(),
            noncoeff_row_commitments,
            vec![eval_commitment],
            w_row_blindings,
        );

        let verifier_input = BlindFoldVerifierInput {
            round_commitments: real_instance.round_commitments.clone(),
            output_claims_row_commitments: real_instance.output_claims_row_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        (r1cs, pedersen_generators, real_instance, real_witness, z, verifier_input, eval_commitment_gens)
    }

    #[test]
    fn stage5_blindfold_tampered_constraint_coeff_fails() {
        let opening_values = [F::from_u64(7), F::from_u64(11)];
        let constraint_coeffs = [F::from_u64(3), F::from_u64(5)];
        let joint_claim = constraint_coeffs[0] * opening_values[0] + constraint_coeffs[1] * opening_values[1];
        let y_blinding = F::from_u64(13);

        let (r1cs, pedersen_generators, real_instance, real_witness, z, verifier_input, eval_commitment_gens) =
            make_stage5_test_instance(&opening_values, &constraint_coeffs, joint_claim, y_blinding);

        let prover = BlindFoldProver::<_, _>::new(&pedersen_generators, &r1cs, Some(eval_commitment_gens));
        let mut prover_transcript = KeccakTranscript::new(b"DAG_stage5_constraint");
        let proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        let tampered_constraint_coeffs = [constraint_coeffs[0] + F::one(), constraint_coeffs[1]];
        let tampered_r1cs = stage5_test_r1cs(
            &[
                OpeningId::Committed(CommittedPolynomial::Bytecode, SumcheckId::OpeningReduction),
                OpeningId::Committed(CommittedPolynomial::ReadWriteMemory, SumcheckId::OpeningReduction),
            ],
            &tampered_constraint_coeffs,
        );

        let verifier = BlindFoldVerifier::<_, _>::new(&pedersen_generators, &tampered_r1cs, Some(eval_commitment_gens));
        let mut verifier_transcript = KeccakTranscript::new(b"DAG_stage5_constraint");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);

        assert!(
            result.is_err(),
            "BlindFold verification should fail when DAG stage5 constraint coefficients are tampered"
        );
    }
}
