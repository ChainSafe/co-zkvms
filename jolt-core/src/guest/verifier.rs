use crate::guest::program::Program;
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::zkvm::{Jolt, JoltRV64IMAC, JoltVerifierPreprocessing};
use common::jolt_device::MemoryLayout;

pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
) -> JoltVerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> {
    let (bytecode, memory_init, program_size) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);

    let prover_preprocessing = JoltRV64IMAC::prover_preprocess(
        bytecode.to_vec(),
        memory_layout,
        memory_init.to_vec(),
        max_trace_length,
    );

    JoltVerifierPreprocessing::from(&prover_preprocessing)
}
