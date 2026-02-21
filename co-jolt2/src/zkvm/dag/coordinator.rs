use std::collections::BTreeMap;

use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::state_manager::StateManagerCoordinator;
use crate::zkvm::dag::Rep3DagStop;
use crate::zkvm::spartan::Rep3SpartanDag;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::dag::proof_serialization::{Claims, JoltProof};
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
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
        Self::prove_with_stop(state, network, Rep3DagStop::AfterStage1)
    }

    #[tracing::instrument(skip_all, name = "Rep3JoltDAGCoordinator::prove_with_stop")]
    pub fn prove_with_stop<'a, F, ProofTranscript, PCS, N>(
        mut state: StateManagerCoordinator<'a, F, ProofTranscript, PCS>,
        network: &mut N,
        stop: Rep3DagStop,
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

        if stop != Rep3DagStop::AfterCommitments {
            Rep3SpartanDag::stage1_prove(&mut state, network)?;
        }
        // --- Construct stub JoltProof with real commitments, deferred stages ---
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
