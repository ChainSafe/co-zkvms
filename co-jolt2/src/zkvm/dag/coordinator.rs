use std::collections::HashMap;

use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::subprotocols::sumcheck::{BatchedSumcheckInstance, HybridBatchedSumcheck};
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::stage::{Rep3JoltDagStagesCoordinator, SumcheckStagesCoordinator};
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManagerCoordinator};
use crate::zkvm::spartan::Rep3SpartanDag;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::opening_proof::ReducedOpeningProof;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::dag::proof_serialization::{Claims, JoltProof};
use jolt_core::zkvm::witness::{
    compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

/// Coordinator side of the MPC DAG prover.
///
/// Owns the Fiat-Shamir transcript, drives sumcheck rounds by broadcasting
/// challenges, receives evaluation shares from workers, and assembles the
/// final proof.
pub struct Rep3JoltDAGCoordinator;

impl Rep3JoltDAGCoordinator {
    #[tracing::instrument(skip_all, name = "Rep3JoltDAGCoordinator::prove")]
    pub fn prove<'a, F, ProofTranscript, PCS, N>(
        mut state: StateManagerCoordinator<'a, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<JoltProof<F, PCS, ProofTranscript>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        // --- Receive trace_length from workers ---
        let trace_lengths: Vec<usize> = network.receive_responses()?;
        let trace_length = trace_lengths[0];
        eyre::ensure!(
            trace_lengths.iter().all(|&t| t == trace_length),
            "trace_length mismatch across parties"
        );
        state.trace_length = trace_length;
        let padded_trace_length = trace_length.next_power_of_two();

        // --- Fiat-Shamir preamble ---
        state.fiat_shamir_preamble(trace_length);

        // --- Initialize DoryGlobals and AllCommittedPolynomials ---
        let ram_K = state.ram_K;
        let bytecode_d = state.preprocessing.shared.bytecode.d;
        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length);
        let _poly_guard =
            AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d);

        // --- Receive, combine, and store commitments ---
        Self::receive_commitments::<F, PCS, ProofTranscript, N>(&mut state, network)?;

        // --- Receive untrusted advice commitment from workers ---
        Self::receive_untrusted_advice_commitment::<F, PCS, ProofTranscript, N>(
            &mut state, network,
        )?;

        // --- Append advice commitments to transcript (matching vanilla ordering) ---
        if let Some(ref untrusted_advice_commitment) = state.untrusted_advice_commitment {
            state
                .transcript
                .append_serializable(untrusted_advice_commitment);
        }

        if let Some(ref trusted_advice_commitment) = state.trusted_advice_commitment {
            state
                .transcript
                .append_serializable(trusted_advice_commitment);
        }

        Rep3SpartanDag::stage1_prove(&mut state, network)?;

        // -------------------------------------------------------------------
        // Stage 2: batched sumcheck
        // -------------------------------------------------------------------

        let mut stages = Rep3JoltDagStagesCoordinator;
        let stage2_hybrid: Vec<BatchedSumcheckInstance<F, ProofTranscript>> =
            stages.stage2_instances(&mut state, network)?;

        let (proof, _r_stage2) = HybridBatchedSumcheck::prove(
            &stage2_hybrid,
            &mut state.accumulator,
            &mut state.transcript,
            network,
        )?;
        state
            .proofs
            .insert(ProofKeys::Stage2Sumcheck, ProofData::SumcheckProof(proof));

        // -------------------------------------------------------------------
        // Stage 3: batched sumcheck (secret + public instances)
        // -------------------------------------------------------------------

        let stage3_instances = stages.stage3_instances(&mut state, network)?;

        let (stage3_proof, _r_stage3) = HybridBatchedSumcheck::prove(
            &stage3_instances,
            &mut state.accumulator,
            &mut state.transcript,
            network,
        )?;
        state.proofs.insert(
            ProofKeys::Stage3Sumcheck,
            ProofData::SumcheckProof(stage3_proof),
        );

        // -------------------------------------------------------------------
        // Stage 4: batched sumcheck (RAM + Bytecode public, Lookups RA secret)
        // -------------------------------------------------------------------

        let stage4_instances = stages.stage4_instances(&mut state, network)?;

        if !stage4_instances.is_empty() {
            let (stage4_proof, _r_stage4) = HybridBatchedSumcheck::prove(
                &stage4_instances,
                &mut state.accumulator,
                &mut state.transcript,
                network,
            )?;
            state.proofs.insert(
                ProofKeys::Stage4Sumcheck,
                ProofData::SumcheckProof(stage4_proof),
            );
        }

        // --- Construct stub JoltProof with real commitments, deferred stages ---
        // -------------------------------------------------------------------
        // Stage 5: opening proof reduction
        // -------------------------------------------------------------------

        let poly_keys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();
        let mut commitment_map: HashMap<CommittedPolynomial, PCS::Commitment> = poly_keys
            .into_iter()
            .zip(state.commitments.iter().cloned())
            .collect();

        let pcs_setup = state
            .pcs_setup
            .expect("StateManagerCoordinator::pcs_setup must be set for reduce_and_prove (stage5)");
        let reduced = state
            .accumulator
            .reduce_and_prove::<PCS, ProofTranscript, N>(
                &mut commitment_map,
                pcs_setup,
                &mut state.transcript,
                network,
            )?;
        state.proofs.insert(
            ProofKeys::ReducedOpeningProof,
            ProofData::ReducedOpeningProof(ReducedOpeningProof {
                sumcheck_proof: reduced.sumcheck_proof,
                sumcheck_claims: reduced.sumcheck_claims,
                joint_opening_proof: reduced.joint_opening_proof,
            }),
        );

        // --- Construct JoltProof ---
        let proof = JoltProof {
            opening_claims: Claims(std::mem::take(&mut state.accumulator.openings)),
            commitments: std::mem::take(&mut state.commitments),
            proofs: std::mem::take(&mut state.proofs),
            untrusted_advice_commitment: state.untrusted_advice_commitment.take(),
            trace_length,
            ram_K: state.ram_K,
            bytecode_d: state.preprocessing.shared.bytecode.d,
            twist_sumcheck_switch_index: state.twist_sumcheck_switch_index,
        };
        Ok(proof)
    }

    fn receive_commitments<F, PCS, ProofTranscript, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        // Receive commitment shares from all 3 parties
        // Each party sends Vec<MaybeShared<Commitment>> aligned with AllCommittedPolynomials
        let all_commitment_shares: Vec<Vec<MaybeShared<PCS::Commitment>>> =
            network.receive_responses()?;

        eyre::ensure!(
            all_commitment_shares.len() == 3,
            "expected commitment shares from 3 parties, got {}",
            all_commitment_shares.len()
        );
        let num_polys = all_commitment_shares[0].len();

        // Combine commitment shares
        let combined_commitments: Vec<PCS::Commitment> = (0..num_polys)
            .map(|i| {
                let shares: Vec<&MaybeShared<PCS::Commitment>> = all_commitment_shares
                    .iter()
                    .map(|party_shares| &party_shares[i])
                    .collect();
                <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::combine_commitment_shares(
                    &shares,
                )
            })
            .collect();

        // Store commitments and append to transcript
        state.commitments = combined_commitments;
        for commitment in &state.commitments {
            state.transcript.append_serializable(commitment);
        }

        Ok(())
    }

    /// Receive the untrusted advice commitment from all 3 workers.
    ///
    /// All workers compute the same public commitment, so we verify consistency
    /// and store the first non-None value.
    fn receive_untrusted_advice_commitment<F, PCS, ProofTranscript, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        let commitments: Vec<Option<PCS::Commitment>> = network.receive_responses()?;
        eyre::ensure!(
            commitments.len() == 3,
            "expected untrusted advice commitment from 3 parties, got {}",
            commitments.len()
        );

        // All workers should produce identical results (public computation)
        eyre::ensure!(
            commitments[0] == commitments[1] && commitments[1] == commitments[2],
            "untrusted advice commitment mismatch across parties"
        );

        state.untrusted_advice_commitment = commitments.into_iter().next().unwrap();
        Ok(())
    }
}
