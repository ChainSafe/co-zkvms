use eyre::Context;
use rcgen::CertifiedKey;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// Ephemeral ECDSA P-256 identity generated fresh on every enclave boot.
/// The private key never touches disk (enclaves have no persistent storage).
pub struct EphemeralIdentity {
    pub cert_der: CertificateDer<'static>,
    pub key_der: PrivateKeyDer<'static>,
    /// Raw ECDSA P-256 public key bytes — embedded in NSM attestation request
    /// so the attestation document cryptographically binds to this identity.
    pub public_key_bytes: Vec<u8>,
}

impl EphemeralIdentity {
    pub fn generate() -> eyre::Result<Self> {
        let CertifiedKey { cert, key_pair } = rcgen::generate_simple_self_signed(vec!["enclave.local".into()])
            .context("generating ephemeral ECDSA P-256 certificate")?;

        let public_key_bytes = key_pair.public_key_raw().to_vec();
        let cert_der = CertificateDer::from(cert.der().to_vec());
        let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pair.serialize_der())).clone_key();

        Ok(Self { cert_der, key_der, public_key_bytes })
    }
}
