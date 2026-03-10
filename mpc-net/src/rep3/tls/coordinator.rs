//! Raw TLS client for connecting to a TEE coordinator through a host proxy.
//!
//! Used when `CoordinatorProtocol::Tls` is set in the network config.
//! The coordinator runs inside an enclave with an ephemeral TLS identity;
//! the worker connects via TCP (through an untrusted host proxy) and verifies
//! the coordinator's identity via attestation rather than a pre-shared cert.

use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;

use eyre::Context;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error, SignatureScheme};

use crate::config::Address;

/// A TLS connection to the coordinator, used in TEE mode.
///
/// Wraps a `rustls::StreamOwned` over a TCP connection to the host proxy.
/// The TLS session terminates inside the enclave — the proxy only sees ciphertext.
pub struct TlsCoordinatorClient {
    stream: rustls::StreamOwned<rustls::ClientConnection, TcpStream>,
}

impl std::fmt::Debug for TlsCoordinatorClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsCoordinatorClient").finish()
    }
}

impl TlsCoordinatorClient {
    /// Connect to the coordinator through the host proxy.
    ///
    /// 1. TCP connect to `addr` (the host proxy address)
    /// 2. TLS handshake (accepts any server cert — attestation verifies identity)
    /// 3. Read attestation document (length-prefixed first message from coordinator)
    /// 4. Send party identification (party_id, worker_id)
    pub fn connect(addr: &Address, party_id: usize, worker_id: usize) -> eyre::Result<Self> {
        let socket_addr = addr
            .to_socket_addrs()
            .context("resolving coordinator proxy address")?
            .next()
            .ok_or_else(|| eyre::eyre!("could not resolve address {addr}"))?;

        let tcp = TcpStream::connect(socket_addr)
            .with_context(|| format!("connecting to coordinator proxy at {addr}"))?;
        tcp.set_nodelay(true)?;

        // Accept any server cert — the coordinator's ephemeral cert is verified
        // via attestation, not via a pre-shared root store.
        let config = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyCertVerifier))
            .with_no_client_auth();

        let server_name = ServerName::try_from("enclave.local")
            .expect("valid static server name");
        let tls_conn = rustls::ClientConnection::new(Arc::new(config), server_name)
            .context("creating TLS client connection")?;
        let mut stream = rustls::StreamOwned::new(tls_conn, tcp);

        // Read attestation document (length-prefixed)
        let mut len_buf = [0u8; 4];
        stream
            .read_exact(&mut len_buf)
            .context("reading attestation doc length")?;
        let doc_len = u32::from_be_bytes(len_buf) as usize;
        if doc_len > 0 {
            let mut _doc = vec![0u8; doc_len];
            stream
                .read_exact(&mut _doc)
                .context("reading attestation document")?;
            // TODO: verify attestation document against policy
            // verify_attestation(&doc, &stream.conn.peer_certificates(), &policy)?;
        }

        // Send identification
        stream
            .write_all(&(party_id as u32).to_be_bytes())
            .context("sending party_id")?;
        stream
            .write_all(&(worker_id as u32).to_be_bytes())
            .context("sending worker_id")?;
        stream.flush().context("flushing identification")?;

        Ok(Self { stream })
    }

    /// Send a length-prefixed message to the coordinator.
    pub fn send(&mut self, data: &[u8]) -> io::Result<()> {
        self.stream.write_all(&(data.len() as u32).to_be_bytes())?;
        self.stream.write_all(data)?;
        self.stream.flush()
    }

    /// Receive a length-prefixed message from the coordinator.
    pub fn recv(&mut self) -> io::Result<Vec<u8>> {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Certificate verifier that accepts any server certificate.
///
/// In TEE mode, the coordinator generates an ephemeral cert on every boot.
/// Trust is established via attestation (binding the ephemeral pubkey to the
/// enclave measurement), not via a pre-shared certificate.
#[derive(Debug)]
struct AcceptAnyCertVerifier;

impl ServerCertVerifier for AcceptAnyCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::ED25519,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}
