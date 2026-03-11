pub mod bytecode;
pub mod dag;
pub mod instruction;
pub mod instruction_lookups;
pub mod r1cs;
pub mod ram;
pub mod registers;
pub mod spartan;
pub mod suffixes;
pub mod witness;

use crate::host::memory::Rep3Memory;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::dag::worker::Rep3JoltDagWorker;
use crate::zkvm::instruction::Rep3Cycle;
use jolt_core::ark_bn254::Fr;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};
use jolt_core::zkvm::{Jolt, JoltProverPreprocessing, JoltRV64IMAC};

pub use jolt_core::zkvm::JoltRV32IM;

#[cfg(not(feature = "rv64"))]
pub type JoltArch = JoltRV32IM;
#[cfg(feature = "rv64")]
pub type JoltArch = JoltRV64IMAC;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
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
        memory_layout: jolt_common::jolt_device::MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS>;

    fn prove<N: Rep3NetworkWorker>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: Vec<Rep3Cycle>,
        program_io: crate::host::jolt_device::Rep3ProgramIOInput,
        final_memory_state: Rep3Memory,
        io_ctx: &mut IoContextPool<N>,
        ram_K: usize,
        preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<()>;
}

// ---------------------------------------------------------------------------
// Implementation for JoltArch
// ---------------------------------------------------------------------------

impl Rep3JoltWorker<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltArch {
    #[tracing::instrument(skip_all, name = "jolt_preprocess")]
    fn preprocess(
        bytecode: Vec<tracer::instruction::Instruction>,
        memory_layout: jolt_common::jolt_device::MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<Fr, DoryCommitmentScheme> {
        // Delegate to vanilla Jolt::prover_preprocess — preprocessing is public
        <JoltArch as Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript>>::prover_preprocess(
            bytecode,
            memory_layout,
            memory_init,
            max_trace_length,
        )
    }

    fn prove<N: Rep3NetworkWorker>(
        preprocessing: &JoltProverPreprocessing<Fr, DoryCommitmentScheme>,
        trace: Vec<Rep3Cycle>,
        program_io: crate::host::jolt_device::Rep3ProgramIOInput,
        final_memory_state: Rep3Memory,
        io_ctx: &mut IoContextPool<N>,
        ram_K: usize,
        preproc: &mut PreprocessingPool<Fr>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.party_id();
        let state = StateManagerWorker::new(preprocessing, trace, program_io, final_memory_state, party_id, ram_K);
        Rep3JoltDagWorker::prove::<Fr, DoryCommitmentScheme, Blake2bTranscript, N>(state, io_ctx, preproc)
    }
}
