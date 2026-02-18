pub mod dag;
pub mod instruction;
pub mod witness;

use std::collections::HashMap;

use crate::field::JoltField;
use crate::host::memory::Rep3Memory;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::zkvm::dag::coordinator::Rep3JoltDAGCoordinator;
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};
use crate::zkvm::dag::worker::Rep3JoltDAGWorker;
use crate::zkvm::instruction::Rep3Cycle;
use jolt_core::ark_bn254::Fr;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};
use jolt_core::zkvm::dag::proof_serialization::JoltProof;
use jolt_core::zkvm::witness::CommittedPolynomial;
use jolt_core::zkvm::{JoltProverPreprocessing, JoltRV64IMAC, JoltVerifierPreprocessing};
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use tracer::JoltDevice;

// ---------------------------------------------------------------------------
// Worker trait
// ---------------------------------------------------------------------------

pub trait Rep3JoltWorker<F: JoltField, PCS, ProofTranscript: Transcript>
where
    PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
{
    fn preprocess(
        bytecode: Vec<tracer::instruction::Instruction>,
        memory_layout: jolt2_common::jolt_device::MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS>;

    fn prove<N: Rep3NetworkWorker>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: Vec<Rep3Cycle>,
        program_io: JoltDevice,
        final_memory_state: Rep3Memory,
        io_ctx: IoContextPool<N>,
        ram_K: usize,
    ) -> eyre::Result<HashMap<CommittedPolynomial, PCS::OpeningProofHint>>;
}

// ---------------------------------------------------------------------------
// Coordinator trait
// ---------------------------------------------------------------------------

pub trait Rep3Jolt<F: JoltField, PCS, ProofTranscript: Transcript>
where
    PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
{
    fn prove<N: Rep3NetworkCoordinator>(
        preprocessing: &JoltVerifierPreprocessing<F, PCS>,
        program_io: JoltDevice,
        network: &mut N,
        ram_K: usize,
        trace_length: usize,
    ) -> eyre::Result<JoltProof<F, PCS, ProofTranscript>>;
}

// ---------------------------------------------------------------------------
// Implementations for JoltRV64IMAC
// ---------------------------------------------------------------------------

impl Rep3JoltWorker<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltRV64IMAC {
    fn preprocess(
        bytecode: Vec<tracer::instruction::Instruction>,
        memory_layout: jolt2_common::jolt_device::MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<Fr, DoryCommitmentScheme> {
        use jolt_core::utils::math::Math;
        use jolt_core::zkvm::witness::DTH_ROOT_OF_K;
        use jolt_core::zkvm::Jolt;

        // Delegate to vanilla Jolt::prover_preprocess — preprocessing is public
        <JoltRV64IMAC as Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript>>::prover_preprocess(
            bytecode,
            memory_layout,
            memory_init,
            max_trace_length,
        )
    }

    fn prove<N: Rep3NetworkWorker>(
        preprocessing: &JoltProverPreprocessing<Fr, DoryCommitmentScheme>,
        trace: Vec<Rep3Cycle>,
        program_io: JoltDevice,
        final_memory_state: Rep3Memory,
        io_ctx: IoContextPool<N>,
        ram_K: usize,
    ) -> eyre::Result<
        HashMap<CommittedPolynomial, <DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint>,
    > {
        let state = StateManagerWorker::new(
            preprocessing,
            trace,
            program_io,
            final_memory_state,
            io_ctx,
            ram_K,
        );
        Rep3JoltDAGWorker::prove::<Fr, DoryCommitmentScheme, Blake2bTranscript, N>(state)
    }
}

impl Rep3Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltRV64IMAC {
    fn prove<N: Rep3NetworkCoordinator>(
        preprocessing: &JoltVerifierPreprocessing<Fr, DoryCommitmentScheme>,
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

        let state = StateManagerCoordinator::new(
            preprocessing,
            program_io,
            ram_K,
            twist_sumcheck_switch_index,
        );
        Rep3JoltDAGCoordinator::prove(state, network)
    }
}
