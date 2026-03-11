use std::collections::HashMap;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance};
use crate::transcripts::Transcript;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial};
use anyhow::Context;

use super::verifier_dags::{BytecodeDag, LookupsDag, RamDag, RegistersDag, SpartanDag};

pub enum JoltDAG {}

impl JoltDAG {
    #[tracing::instrument(skip_all, name = "JoltDAG::verify")]
    pub fn verify<
        'a,
        F: JoltField,
        C: JoltCurve,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        mut state_manager: StateManager<'a, F, C, ProofTranscript, PCS>,
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
            transcript
                .borrow_mut()
                .append_serializable(untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = state_manager.trusted_advice_commitment {
            transcript
                .borrow_mut()
                .append_serializable(trusted_advice_commitment);
        }

        // Stage 1:
        let trace_length = state_manager.get_verifier_data().2;
        let padded_trace_length = trace_length.next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::default();
        let mut registers_dag = RegistersDag::default();
        let mut ram_dag = RamDag::new_verifier(&state_manager);
        let mut bytecode_dag = BytecodeDag::default();
        spartan_dag
            .stage1_verify(&mut state_manager)
            .context("Stage 1")?;

        // Stage 2:
        let stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_verifier_instances(&mut state_manager))
            .chain(registers_dag.stage2_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage2_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage2_verifier_instances(&mut state_manager))
            .collect();
        let stage2_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = stage2_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage2_proof_data = proofs
            .get(&ProofKeys::Stage2Sumcheck)
            .expect("Stage 2 sumcheck proof not found");
        let stage2_proof = match stage2_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 2"),
        };

        let transcript = state_manager.get_transcript();
        let opening_accumulator = state_manager.get_verifier_accumulator();
        let _r_stage2 = BatchedSumcheck::verify(
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
        let stage3_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = stage3_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage3_proof_data = proofs
            .get(&ProofKeys::Stage3Sumcheck)
            .expect("Stage 3 sumcheck proof not found");
        let stage3_proof = match stage3_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 3"),
        };

        let _r_stage3 = BatchedSumcheck::verify(
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
        let stage4_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = stage4_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage4_proof_data = proofs
            .get(&ProofKeys::Stage4Sumcheck)
            .expect("Stage 4 sumcheck proof not found");
        let stage4_proof = match stage4_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let _r_stage4 = BatchedSumcheck::verify(
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

        // Batch-prove all openings
        let batched_opening_proof = proofs
            .get(&ProofKeys::ReducedOpeningProof)
            .expect("Reduced opening proof not found");
        let batched_opening_proof = match batched_opening_proof {
            ProofData::ReducedOpeningProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let mut commitments_map = HashMap::new();
        for polynomial in AllCommittedPolynomials::iter() {
            commitments_map.insert(
                *polynomial,
                commitments.borrow()[polynomial.to_index()].clone(),
            );
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

        Ok(())
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
        let proof = match state_manager
            .proofs
            .borrow()
            .get(&ProofKeys::TrustedAdviceProof)
        {
            Some(ProofData::OpeningProof(proof)) => proof.clone(),
            _ => return Err(anyhow::anyhow!("Trusted advice proof not found")),
        };

        PCS::verify(
            &proof,
            verifier_setup,
            transcript,
            &point.r,
            &eval,
            trusted_advice_commitment,
        )
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
        let untrusted_advice_commitment =
            state_manager.untrusted_advice_commitment.as_ref().unwrap();
        let accumulator = state_manager.get_verifier_accumulator();

        let (point, eval) = accumulator.borrow().get_untrusted_advice_opening().unwrap();
        let proof = match state_manager
            .proofs
            .borrow()
            .get(&ProofKeys::UntrustedAdviceProof)
        {
            Some(ProofData::OpeningProof(proof)) => proof.clone(),
            _ => return Err(anyhow::anyhow!("Untrusted advice proof not found")),
        };

        PCS::verify(
            &proof,
            verifier_setup,
            transcript,
            &point.r,
            &eval,
            untrusted_advice_commitment,
        )
        .map_err(|e| {
            anyhow::anyhow!("Untrusted advice opening proof verification failed: {e:?}")
        })?;

        Ok(())
    }
}
