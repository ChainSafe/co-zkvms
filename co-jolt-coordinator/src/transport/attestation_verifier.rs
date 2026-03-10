use std::collections::HashMap;

use rustls::pki_types::CertificateDer;

/// Policy for verifying NSM attestation documents from the enclave.
pub enum AttestationPolicy {
    /// Accept any (or no) attestation — for local dev and fystack simulation.
    AcceptAll,
    /// Verify against AWS Nitro root of trust + expected PCR values.
    #[allow(dead_code)]
    AwsNitro {
        expected_pcrs: HashMap<usize, Vec<u8>>,
    },
}

/// Verify an attestation document against the given policy.
///
/// - `doc_bytes`: raw attestation document bytes received from the enclave
///   (empty slice if no attestation was provided)
/// - `tls_cert`: the TLS certificate presented by the enclave during handshake
/// - `policy`: determines what level of verification to perform
pub fn verify_attestation(
    _doc_bytes: &[u8],
    _tls_cert: &CertificateDer,
    policy: &AttestationPolicy,
) -> eyre::Result<()> {
    match policy {
        AttestationPolicy::AcceptAll => Ok(()),
        AttestationPolicy::AwsNitro { expected_pcrs: _ } => {
            // Future implementation:
            // 1. Parse COSE_Sign1 attestation document
            // 2. Verify signature against AWS Nitro root cert chain
            // 3. Check PCR0 (enclave image hash) matches expected
            // 4. Extract public_key from attestation doc
            // 5. Verify public_key matches TLS cert's public key
            //    (proves the ephemeral key was generated inside THIS enclave)
            Err(eyre::eyre!(
                "AwsNitro attestation verification not yet implemented"
            ))
        }
    }
}
