use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::state_manager::StateManagerCoordinator;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::dag::proof_serialization::JoltProof;
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
        // --- Receive trace_length from workers ---
        let trace_lengths: Vec<usize> = network.receive_responses()?;
        let trace_length = trace_lengths[0];
        eyre::ensure!(
            trace_lengths.iter().all(|&t| t == trace_length),
            "trace_length mismatch across parties"
        );
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

        // Stage 1: coordinate Spartan outer sumcheck
        // Stage 2: coordinate batched sumcheck
        // Stage 3: coordinate batched sumcheck
        // Stage 4: coordinate batched sumcheck
        // Stage 5: coordinate opening proof, assemble JoltProof
        todo!("remaining stages")
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
}
