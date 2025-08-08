#![allow(incomplete_features)]

pub mod indexer;
pub mod logup;
pub mod math;
pub mod r1cs;
pub mod utils;
pub mod verifier;
pub mod zk;

use ark_ec::pairing::Pairing;
use ark_poly_commit::multilinear_pc::data_structures::Commitment;
use ark_serialize::CanonicalSerialize;
use bytesize::ByteSize;
pub use indexer::{IndexProverKey, IndexVerifierKey, Indexer};
pub use logup::LogLookupProof;
pub use r1cs::R1CS;
pub use zk::SRS;
use zk::{ZKMLProof, ZKSumcheckProof};

pub use snarks_core::transcript;

/// The SNARK proof, composed of all prover's messages sent throughout the protocol.
#[derive(CanonicalSerialize)]
pub struct R1CSProof<E: Pairing> {
    pub witness_commitment: Commitment<E>,

    pub first_sumcheck_msgs: ZKSumcheckProof<E>,
    pub va: E::ScalarField,
    pub vb: E::ScalarField,
    pub vc: E::ScalarField,

    pub second_sumcheck_msgs: ZKSumcheckProof<E>,
    pub witness_eval: E::ScalarField,
    pub val_m: E::ScalarField,
    pub witness_proof: ZKMLProof<E>,
    pub eq_tilde_rx_commitment: Commitment<E>,
    pub eq_tilde_ry_commitment: Commitment<E>,

    pub lookup_proof: LogLookupProof<E>,
}

impl<E: Pairing> R1CSProof<E> {
    pub fn log_size_report(&self) {
        tracing::info!(
            "witness_commitment: {}",
            ByteSize(self.witness_commitment.compressed_size() as u64)
        );
        tracing::info!(
            "first_sumcheck_msgs: {}",
            ByteSize(self.first_sumcheck_msgs.compressed_size() as u64)
        );
        tracing::info!("va: {}", ByteSize(self.va.compressed_size() as u64));
        tracing::info!("vb: {}", ByteSize(self.vb.compressed_size() as u64));
        tracing::info!("vc: {}", ByteSize(self.vc.compressed_size() as u64));
        tracing::info!(
            "second_sumcheck_msgs: {}",
            ByteSize(self.second_sumcheck_msgs.compressed_size() as u64)
        );
        tracing::info!(
            "witness_eval: {}",
            ByteSize(self.witness_eval.compressed_size() as u64)
        );
        tracing::info!("val_m: {}", ByteSize(self.val_m.compressed_size() as u64));
        tracing::info!(
            "witness_proof: {}",
            ByteSize(self.witness_proof.compressed_size() as u64)
        );
        tracing::info!(
            "eq_tilde_rx_commitment: {}",
            ByteSize(self.eq_tilde_rx_commitment.compressed_size() as u64)
        );
        tracing::info!(
            "eq_tilde_ry_commitment: {}",
            ByteSize(self.eq_tilde_ry_commitment.compressed_size() as u64)
        );
        tracing::info!(
            "lookup_proof: {}",
            ByteSize(self.lookup_proof.compressed_size() as u64)
        );
    }
}
