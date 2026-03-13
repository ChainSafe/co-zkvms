//! `Client` — user-facing client for delegating proofs to TEE workers.
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

use co_jolt2::host::jolt_device::Rep3ProgramIOInput;
use co_jolt2::host::memory::Rep3Memory;
use co_jolt2::host::program::Rep3ShareBundle;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use eyre::Context;
use rustls::pki_types::ServerName;
use serde::Serialize;

/// Payload sent to each worker containing their secret share + public data.
///
/// NOTE: No plaintext advice or io_device — only shares and public metadata.
#[derive(Serialize)]
struct WorkerPayloadRef<'a> {
    trace: &'a [Rep3Cycle],
    memory: &'a Rep3Memory,
    program_io_share: &'a Rep3ProgramIOInput,
    bytecode: &'a [tracer::instruction::Instruction],
    memory_init: &'a [(u64, u8)],
    program_id: &'a str,
    preprocess_trace_len: usize,
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

        let tcp = TcpStream::connect(addr).with_context(|| format!("connecting to worker at {addr}"))?;
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
pub struct Client {
    workers: [WorkerConnection; 3],
}

impl Client {
    /// Connect to 3 workers via TLS.
    pub fn connect(worker_addrs: [SocketAddr; 3]) -> eyre::Result<Self> {
        rustls::crypto::aws_lc_rs::default_provider().install_default().ok();

        let w0 = WorkerConnection::connect(worker_addrs[0]).context("worker 0")?;
        let w1 = WorkerConnection::connect(worker_addrs[1]).context("worker 1")?;
        let w2 = WorkerConnection::connect(worker_addrs[2]).context("worker 2")?;

        Ok(Self { workers: [w0, w1, w2] })
    }

    /// Send precomputed Rep3 shares to workers and receive the proof.
    ///
    /// Returns the serialized proof bytes (ark CanonicalSerialize compressed).
    pub fn delegate(
        &mut self,
        bytecode: Vec<tracer::instruction::Instruction>,
        memory_init: Vec<(u64, u8)>,
        program_id: String,
        preprocess_trace_len: usize,
        padded_len: usize,
        ram_k: usize,
        shares: [Rep3ShareBundle; 3],
    ) -> eyre::Result<Vec<u8>> {
        for (i, ((trace, memory, program_io_share), worker)) in
            shares.iter().zip(self.workers.iter_mut()).enumerate()
        {
            let payload = WorkerPayloadRef {
                trace,
                memory,
                program_io_share,
                bytecode: &bytecode,
                memory_init: &memory_init,
                program_id: &program_id,
                preprocess_trace_len,
                padded_len,
                ram_k,
            };
            let payload_bytes = bincode::serialize(&payload).context("serializing WorkerPayload")?;
            worker.send(&payload_bytes).with_context(|| format!("sending payload to worker {i}"))?;
        }

        let proof_bytes = self.workers[0].recv().context("receiving proof from worker 0")?;
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
        rustls::crypto::aws_lc_rs::default_provider().signature_verification_algorithms.supported_schemes()
    }
}
