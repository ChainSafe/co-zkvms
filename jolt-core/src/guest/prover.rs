use super::program::Program;
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::zkvm::{Jolt, JoltProverPreprocessing, JoltRV64IMAC};
use common::jolt_device::MemoryLayout;

#[allow(clippy::type_complexity)]
#[cfg(feature = "prover")]
pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
) -> JoltProverPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> {
    let (bytecode, memory_init, program_size) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);

    JoltRV64IMAC::prover_preprocess(bytecode, memory_layout, memory_init, max_trace_length)
}
