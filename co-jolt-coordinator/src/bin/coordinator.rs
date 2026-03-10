use co_jolt_coordinator::transport::ephemeral_identity::EphemeralIdentity;
use co_jolt_coordinator::types::ProofRequest;
use co_jolt_coordinator::zkvm::{JoltArch, Rep3Jolt};
use eyre::Context;
use jolt_core::ark_bn254::Fr;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use jolt_core::zkvm::{Jolt, JoltVerifierPreprocessing};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_net::topology::MpcStarNetCoordinator;
use mpc_types::protocols::rep3::id::PartyID;
use tracer::JoltDevice;
use tracing::info;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;

fn main() -> eyre::Result<()> {
    // 1. Generate ephemeral ECDSA P-256 identity
    let identity = EphemeralIdentity::generate().context("generating ephemeral identity")?;
    info!(
        pubkey_len = identity.public_key_bytes.len(),
        "generated ephemeral ECDSA P-256 identity"
    );

    // 2. [if aws_nitro] Request NSM attestation binding the ephemeral pubkey
    #[cfg(feature = "aws_nitro")]
    let attestation_doc: Option<Vec<u8>> = {
        // TODO: integrate aws-nitro-enclaves-nsm-api
        None
    };

    #[cfg(not(feature = "aws_nitro"))]
    let attestation_doc: Option<Vec<u8>> = None;

    // 3. Accept 3 worker connections over vsock+TLS
    #[cfg(feature = "aws_nitro")]
    {
        use co_jolt_coordinator::transport::vsock_tls::VsockTlsCoordinator;

        let vsock_port: u32 = std::env::var("VSOCK_PORT")
            .unwrap_or_else(|_| "9000".to_string())
            .parse()
            .context("parsing VSOCK_PORT")?;

        let mut network = VsockTlsCoordinator::accept(
            vsock_port,
            &identity,
            attestation_doc.as_deref(),
        )
        .context("accepting vsock+TLS connections")?;

        info!("accepted 3 worker connections, entering stand-by loop");
        prove_loop(&mut network)?;
    }

    #[cfg(not(feature = "aws_nitro"))]
    {
        let _ = (attestation_doc, identity);
        info!("coordinator stub (no aws_nitro feature) — nothing to do");
    }

    Ok(())
}

/// Main proving service loop.
///
/// Waits for workers to signal readiness (they received shares from a user),
/// receives public proof metadata, computes preprocessing, and drives the proof.
fn prove_loop<N: Rep3NetworkCoordinator>(network: &mut N) -> eyre::Result<()> {
    loop {
        info!("waiting for workers...");
        network.sync_with_parties()?;

        // Receive bincode-serialized ProofRequest from each worker (all identical)
        let requests: Vec<Vec<u8>> = network.receive_responses()?;
        let request: ProofRequest =
            bincode::deserialize(&requests[0]).context("deserializing ProofRequest")?;

        info!(
            padded_len = request.padded_len,
            ram_k = request.ram_k,
            bytecode_len = request.bytecode.len(),
            "received proof request"
        );

        // Reconstruct JoltDevice from public fields (no advice — workers commit those)
        let program_io = JoltDevice {
            inputs: request.inputs,
            outputs: request.outputs,
            panic: request.panic,
            memory_layout: request.memory_layout.clone(),
            trusted_advice: vec![],
            untrusted_advice: vec![],
        };

        // Compute preprocessing from public data (same as vanilla Jolt)
        let preprocessing =
            <JoltArch as Jolt<F, PCS, FS>>::prover_preprocess(
                request.bytecode,
                request.memory_layout,
                request.memory_init,
                request.padded_len,
            );
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

        let _guard = (
            DoryGlobals::initialize(DTH_ROOT_OF_K, request.padded_len),
            AllCommittedPolynomials::initialize(
                compute_d_parameter(request.ram_k),
                preprocessing.shared.bytecode.d,
            ),
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
        // Use ark CanonicalSerialize since JoltProof doesn't impl serde.
        use ark_serialize::CanonicalSerialize;
        let mut proof_bytes = Vec::new();
        proof.serialize_compressed(&mut proof_bytes)
            .context("serializing proof")?;
        network.send_request(PartyID::ID0, 0, proof_bytes)?;

        info!("proof sent, returning to stand-by");
    }
}
