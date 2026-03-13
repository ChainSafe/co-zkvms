#[cfg(feature = "host")]
pub use co_jolt2::host::program::generate_trace_shares;
#[cfg(feature = "host")]
pub use jolt_core::host;
#[cfg(feature = "host")]
pub use jolt_core::zkvm::dag::proof_serialization::serialize_and_print_size;

pub use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
pub use jolt_core::ark_bn254::Fr as F;
pub use jolt_core::curve::Bn254Curve;
pub use jolt_core::field::JoltField;
pub use jolt_core::guest;
pub use jolt_core::poly::commitment::dory::DoryCommitmentScheme as PCS;
pub use jolt_core::transcripts::Blake2bTranscript;
pub use jolt_core::zkvm::{
    dag::proof_serialization::JoltProof, Jolt, JoltProverPreprocessing, JoltRV32IM, JoltRV64IMAC, JoltRVArch,
    JoltVerifierPreprocessing, RV64IMACJoltProof, Serializable,
};

pub use crate::client::Client;
pub use eyre;

// Re-exports needed by the provable macro
pub use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
pub use jolt_core::poly::commitment::dory::DoryGlobals;
pub use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
