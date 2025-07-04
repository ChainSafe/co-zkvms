#![allow(incomplete_features)]

pub mod indexer;
pub mod logup;
pub mod math;
pub mod r1cs;
pub mod transcript;
pub mod utils;
pub mod verifier;
pub mod zk;

use ark_ec::pairing::Pairing;
use ark_poly_commit::multilinear_pc::data_structures::Commitment;
use ark_serialize::CanonicalSerialize;
pub use indexer::{IndexProverKey, IndexVerifierKey, Indexer};
pub use logup::LogLookupProof;
pub use r1cs::R1CS;
pub use zk::SRS;
use zk::{ZKMLProof, ZKSumcheckProof};

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
