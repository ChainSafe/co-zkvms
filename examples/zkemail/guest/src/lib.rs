#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

use rsa::pkcs1::DecodeRsaPublicKey;
use rsa::pkcs1v15::Pkcs1v15Sign;
use rsa::RsaPublicKey;
use sha2::{Digest, Sha256};
use zkemail_core::{DKIMInput, DKIMOutput};

#[jolt::provable(stack_size = 131072, memory_size = 1048576, max_input_size = 65536)]
fn verify_dkim(input: DKIMInput) -> DKIMOutput {
    // Decode RSA public key from PKCS#1 DER
    let public_key =
        RsaPublicKey::from_pkcs1_der(&input.public_key_der).expect("invalid public key DER");

    // Hash the canonicalized signed headers
    let mut hasher = Sha256::new();
    hasher.update(&input.signed_headers);
    let header_hash: [u8; 32] = hasher.finalize().into();

    // Verify RSA PKCS#1 v1.5 signature (SHA-256)
    let scheme = Pkcs1v15Sign::new::<Sha256>();
    let verified = public_key
        .verify(scheme, &header_hash, &input.signature)
        .is_ok();

    // Hash from_domain and public_key for output commitment
    let from_domain_hash: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(&input.from_domain);
        h.finalize().into()
    };

    let public_key_hash: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(&input.public_key_der);
        h.finalize().into()
    };

    DKIMOutput {
        from_domain_hash,
        public_key_hash,
        verified,
    }
}
