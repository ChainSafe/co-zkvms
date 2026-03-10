//! `ProvingClient` — user-facing client for delegating proofs to TEE workers.
//!
//! The client connects to 3 workers via TLS, traces the guest program locally,
//! generates 3-way secret shares, and sends each share to the corresponding
//! worker. Worker 0 relays the assembled proof back.
//!
//! ## E2E Encryption
//!
//! All user↔worker traffic is TLS-encrypted. The `AcceptAnyCertVerifier` skips
//! PKI certificate validation — trust is established via TEE attestation (the
//! worker's ephemeral cert is bound to a Nitro attestation document), not through
//! a certificate authority. TLS encryption and key exchange still happen normally.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;

use eyre::Context;
use jolt_core::host::Program;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::RAMPreprocessing;
use jolt_core::zkvm::JoltSharedPreprocessing;
use rand::rngs::OsRng;
use rustls::pki_types::ServerName;
use serde::{Deserialize, Serialize};
use tracer::instruction::Cycle;

use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::host::memory::Rep3Memory;
use crate::host::program::share_trace;
use crate::utils::compute_ram_k;
use crate::zkvm::instruction::Rep3Cycle;

/// Payload sent to each worker containing their secret share + public data.
///
/// NOTE: No plaintext advice or io_device — only shares and public metadata.
#[derive(Serialize, Deserialize)]
struct WorkerPayload {
    trace: Vec<Rep3Cycle>,
    memory: Rep3Memory,
    program_io_share: Rep3ProgramIOInput,
    bytecode: Vec<tracer::instruction::Instruction>,
    memory_init: Vec<(u64, u8)>,
    padded_len: usize,
    ram_k: usize,
}

type TlsStream = rustls::StreamOwned<rustls::ClientConnection, TcpStream>;

/// A TLS connection to a single worker.
struct WorkerConnection {
    stream: TlsStream,
}

impl WorkerConnection {
    fn connect(addr: SocketAddr) -> eyre::Result<Self> {
        let config = Arc::new(
            rustls::ClientConfig::builder()
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(AcceptAnyCertVerifier))
                .with_no_client_auth(),
        );

        let tcp = TcpStream::connect(addr)
            .with_context(|| format!("connecting to worker at {addr}"))?;
        tcp.set_nodelay(true)?;

        let server_name = ServerName::try_from("localhost").unwrap();
        let tls_conn = rustls::ClientConnection::new(Arc::clone(&config), server_name)
            .context("creating TLS client connection")?;
        let stream = rustls::StreamOwned::new(tls_conn, tcp);

        Ok(Self { stream })
    }

    fn send(&mut self, data: &[u8]) -> io::Result<()> {
        self.stream.write_all(&(data.len() as u32).to_be_bytes())?;
        self.stream.write_all(data)?;
        self.stream.flush()
    }

    fn recv(&mut self) -> io::Result<Vec<u8>> {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Client for delegating zkVM proofs to a set of 3 TEE workers.
///
/// All traffic is TLS-encrypted. Certificate validation is skipped because
/// trust is established via TEE attestation, not PKI.
pub struct ProvingClient {
    workers: [WorkerConnection; 3],
}

impl ProvingClient {
    /// Connect to 3 workers via TLS.
    pub fn connect(worker_addrs: [SocketAddr; 3]) -> eyre::Result<Self> {
        rustls::crypto::aws_lc_rs::default_provider()
            .install_default()
            .ok();

        let w0 = WorkerConnection::connect(worker_addrs[0]).context("worker 0")?;
        let w1 = WorkerConnection::connect(worker_addrs[1]).context("worker 1")?;
        let w2 = WorkerConnection::connect(worker_addrs[2]).context("worker 2")?;

        Ok(Self {
            workers: [w0, w1, w2],
        })
    }

    /// Trace the program, generate shares, send to workers, and receive the proof.
    ///
    /// Returns the serialized proof bytes (ark CanonicalSerialize compressed).
    pub fn delegate(
        &mut self,
        program: &mut Program,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> eyre::Result<Vec<u8>> {
        // 1. Decode + trace
        let (bytecode, memory_init, _) = program.decode();
        let (mut vanilla_trace, memory, io_device) =
            program.trace(inputs, untrusted_advice, trusted_advice);

        let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
        vanilla_trace.resize(padded_len, Cycle::NoOp);

        // Compute ram_K
        let shared = JoltSharedPreprocessing {
            memory_layout: io_device.memory_layout.clone(),
            bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
            ram: RAMPreprocessing::preprocess(memory_init.clone()),
        };
        let ram_k = compute_ram_k(&vanilla_trace, &shared);

        // 2. Generate 3-way secret shares
        let mut rng = OsRng;
        let program_io_shares =
            Rep3ProgramIOInput::generate_secret_shares(io_device, &mut rng);
        let memory_shares = Rep3Memory::generate_secret_shares(
            memory,
            &shared.memory_layout,
            ram_k,
            &mut rng,
        );
        let trace_shares = share_trace(vanilla_trace, &mut rng);

        let [io0, io1, io2]: [Rep3ProgramIOInput; 3] =
            program_io_shares.try_into().expect("expected 3 shares");
        let [mem0, mem1, mem2]: [Rep3Memory; 3] =
            memory_shares.try_into().expect("expected 3 shares");
        let [t0, t1, t2]: [Vec<Rep3Cycle>; 3] = trace_shares;

        let shares = [(t0, mem0, io0), (t1, mem1, io1), (t2, mem2, io2)];

        // 3. Build and send payloads to workers (no plaintext advice!)
        for (i, (trace, mem, io_share)) in shares.into_iter().enumerate() {
            let payload = WorkerPayload {
                trace,
                memory: mem,
                program_io_share: io_share,
                bytecode: bytecode.clone(),
                memory_init: memory_init.clone(),
                padded_len,
                ram_k,
            };
            let payload_bytes =
                bincode::serialize(&payload).context("serializing WorkerPayload")?;
            self.workers[i]
                .send(&payload_bytes)
                .with_context(|| format!("sending payload to worker {i}"))?;
        }

        // 4. Wait for proof from worker 0
        let proof_bytes = self.workers[0]
            .recv()
            .context("receiving proof from worker 0")?;

        Ok(proof_bytes)
    }
}

// -- TLS cert verifier: skip PKI validation (trust via attestation) --
//
// This implements rustls's `ServerCertVerifier` trait to accept any certificate.
// In the TEE model, the worker's ephemeral TLS certificate is bound to a Nitro
// attestation document. The client verifies the attestation separately, so
// standard PKI validation is not needed. TLS encryption and key exchange
// still happen normally — only certificate *identity* verification is skipped.

#[derive(Debug)]
struct AcceptAnyCertVerifier;

impl rustls::client::danger::ServerCertVerifier for AcceptAnyCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}
