use crate::field::JoltField;
use crate::zkvm::dag::state_manager::StateManagerCoordinator;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::dag::proof_serialization::JoltProof;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

/// Coordinator side of the MPC DAG prover.
///
/// Owns the Fiat-Shamir transcript, drives sumcheck rounds by broadcasting
/// challenges, receives evaluation shares from workers, and assembles the
/// final proof.
pub struct JoltDAGCoordinator;

impl JoltDAGCoordinator {
    #[allow(unused_variables)]
    pub fn prove<'a, F, ProofTranscript, PCS, N>(
        mut state: StateManagerCoordinator<'a, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<JoltProof<F, PCS, ProofTranscript>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
        N: Rep3NetworkCoordinator,
    {
        // Step 2+: fiat_shamir_preamble
        // Step 2: receive commitment shares, combine, append to transcript
        // Stage 1: coordinate Spartan outer sumcheck
        // Stage 2: coordinate batched sumcheck
        // Stage 3: coordinate batched sumcheck
        // Stage 4: coordinate batched sumcheck
        // Stage 5: coordinate opening proof, assemble JoltProof
        todo!("implement coordinator prove flow")
    }
}
