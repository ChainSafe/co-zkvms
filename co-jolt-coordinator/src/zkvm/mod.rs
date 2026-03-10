pub mod bytecode;
pub mod dag;
pub use co_jolt2::zkvm::instruction;
pub mod instruction_lookups;
pub use co_jolt2::zkvm::r1cs;
pub mod ram;
pub mod registers;
pub mod spartan;
pub use co_jolt2::zkvm::suffixes;
pub use co_jolt2::zkvm::witness;

use co_jolt2::field::JoltField;
use co_jolt2::poly::commitment::Rep3CommitmentScheme;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::dag::proof_serialization::JoltProof;
use jolt_core::zkvm::JoltVerifierPreprocessing;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use tracer::JoltDevice;

use crate::poly::commitment::Rep3CoordinatorCommitmentScheme;
use crate::zkvm::dag::coordinator::Rep3JoltDag;
use crate::zkvm::dag::state_manager::StateManager;

// ---------------------------------------------------------------------------
// Coordinator trait
// ---------------------------------------------------------------------------

pub trait Rep3Jolt<F: JoltField, PCS, ProofTranscript: Transcript>
where
    PCS: CommitmentScheme<Field = F>
        + Rep3CommitmentScheme<F, ProofTranscript>
        + Rep3CoordinatorCommitmentScheme<F, ProofTranscript>,
{
    fn prove<N: Rep3NetworkCoordinator>(
        preprocessing: &JoltVerifierPreprocessing<F, PCS>,
        pcs_setup: &PCS::ProverSetup,
        program_io: JoltDevice,
        network: &mut N,
        ram_K: usize,
        trace_length: usize,
    ) -> eyre::Result<JoltProof<F, PCS, ProofTranscript>>;
}

// ---------------------------------------------------------------------------
// Implementation for JoltArch
// ---------------------------------------------------------------------------

use co_jolt2::zkvm::JoltArch;
use jolt_core::ark_bn254::Fr;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;

impl Rep3Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltArch {
    fn prove<N: Rep3NetworkCoordinator>(
        preprocessing: &JoltVerifierPreprocessing<Fr, DoryCommitmentScheme>,
        pcs_setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
        program_io: JoltDevice,
        network: &mut N,
        ram_K: usize,
        trace_length: usize,
    ) -> eyre::Result<JoltProof<Fr, DoryCommitmentScheme, Blake2bTranscript>> {
        // Compute twist_sumcheck_switch_index the same way as the worker
        let T = trace_length;
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = if num_chunks > 0 { T / num_chunks } else { T };
        let twist_sumcheck_switch_index = if chunk_size > 0 {
            chunk_size.trailing_zeros() as usize
        } else {
            0
        };

        let state = StateManager::new(
            preprocessing,
            program_io,
            ram_K,
            twist_sumcheck_switch_index,
        )
        .with_pcs_setup(pcs_setup);
        Rep3JoltDag::prove(state, network)
    }
}
