//! Dory polynomial commitment scheme
//!
//! This module provides a Dory commitment scheme implementation that bridges
//! between Jolt's types and final-dory's arkworks backend.

mod commitment_scheme;
mod dory_globals;
mod jolt_dory_routines;
mod wrappers;

#[cfg(test)]
mod tests;

#[cfg(feature = "zk")]
pub use commitment_scheme::bind_opening_inputs_zk;
pub use commitment_scheme::{
    bind_opening_inputs, DoryCommitmentScheme, DoryOpeningProofHint,
};
pub use commitment_scheme::{DoryBatchedProof, DoryCommitment, DoryProofData};
pub use dory_globals::{DoryContext, DoryGlobals, DoryLayout};
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use wrappers::{
    ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT,
    ArkworksProverSetup, ArkworksVerifierSetup,
    JoltFieldWrapper, JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254, JoltGTWrapper,
    JoltGroupWrapper, JoltToDoryTranscript, JoltToDoryTranscriptRef, BN254,
};
