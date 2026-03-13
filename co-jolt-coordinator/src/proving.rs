//! Coordinator proving logic — extracted for reuse across transport modes.

use crate::types::ProofRequest;
use crate::zkvm::{JoltArch, Rep3Jolt};
use ark_serialize::CanonicalSerialize;
use eyre::Context;
use jolt_core::ark_bn254::Fr;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use jolt_core::zkvm::{Jolt, JoltVerifierPreprocessing};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::protocols::rep3::PartyID;
use mpc_net::topology::MpcStarNetCoordinator;
use tracer::JoltDevice;
use tracing::info;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;

/// Drive one proof iteration.
///
/// 1. Sync with workers (barrier)
/// 2. Receive `ProofRequest` (public data) from workers
/// 3. Compute preprocessing
/// 4. Drive MPC proof
/// 5. Send serialized proof to worker 0
pub fn coordinate_once<N: Rep3NetworkCoordinator>(network: &mut N) -> eyre::Result<()> {
    info!("waiting for workers...");
    network.sync_with_parties()?;

    // Receive bincode-serialized ProofRequest from each worker (all identical)
    let requests: Vec<Vec<u8>> = network.receive_responses()?;
    let request: ProofRequest = bincode::deserialize(&requests[0]).context("deserializing ProofRequest")?;

    info!(
        padded_len = request.padded_len,
        ram_k = request.ram_k,
        bytecode_len = request.bytecode.len(),
        "received proof request"
    );

    // Reconstruct JoltDevice from public fields (no advice — workers commit those)
    // Truncate trailing zeros from outputs, matching what vanilla Jolt::prove does.
    let mut outputs = request.outputs;
    outputs.truncate(outputs.iter().rposition(|&b| b != 0).map_or(0, |pos| pos + 1));
    let program_io = JoltDevice {
        inputs: request.inputs,
        outputs,
        panic: request.panic,
        memory_layout: request.memory_layout.clone(),
        trusted_advice: vec![],
        untrusted_advice: vec![],
    };

    // Compute preprocessing from public data (same as vanilla Jolt)
    let preprocessing = <JoltArch as Jolt<F, PCS, FS>>::prover_preprocess(
        request.bytecode,
        request.memory_layout,
        request.memory_init,
        request.padded_len,
    );
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, request.padded_len),
        AllCommittedPolynomials::initialize(compute_d_parameter(request.ram_k), preprocessing.shared.bytecode.d),
    );

    // Drive MPC proof
    let proof = <JoltArch as Rep3Jolt<F, PCS, _>>::prove(
        &verifier_preprocessing,
        &preprocessing.generators,
        program_io,
        network,
        request.ram_k,
        request.padded_len,
    )?;

    info!("proof complete, sending to worker 0");

    // Send proof to worker 0 only (who relays to user).
    let mut proof_bytes = Vec::new();
    proof.serialize_compressed(&mut proof_bytes).context("serializing proof")?;
    network.send_request(PartyID::ID0, 0, proof_bytes)?;

    info!("proof sent, returning to stand-by");
    Ok(())
}
