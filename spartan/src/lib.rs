pub mod indexer;
pub mod logup;
pub mod verifier;
pub mod zk;
pub mod r1cs;
pub mod math;
pub mod transcript;
pub mod utils;

use ark_ec::pairing::Pairing;
use ark_poly_commit::multilinear_pc::data_structures::Commitment;
use ark_serialize::CanonicalSerialize;
pub use indexer::{IndexProverKey, IndexVerifierKey, Indexer};
pub use logup::LogLookupProof;
pub use zk::SRS;
use zk::{ZKMLProof, ZKSumcheckProof};
pub use r1cs::R1CS;

/// The SNARK proof, composed of all prover's messages sent throughout the protocol.
#[derive(CanonicalSerialize)]
pub struct R1CSProof<E: Pairing> {
    pub witness_commitment: Commitment<E>,

    //first_sumcheck_polynomial_info: PolynomialInfo,
    pub first_sumcheck_msgs: ZKSumcheckProof<E>,
    pub va: E::ScalarField,
    pub vb: E::ScalarField,
    pub vc: E::ScalarField,

    //second_sumcheck_polynomial_info: PolynomialInfo,
    pub second_sumcheck_msgs: ZKSumcheckProof<E>,
    pub witness_eval: E::ScalarField,
    pub val_M: E::ScalarField,
    pub witness_proof: ZKMLProof<E>,
    pub eq_tilde_rx_commitment: Commitment<E>,
    pub eq_tilde_ry_commitment: Commitment<E>,

    //third_sumcheck_polynomial_info: PolynomialInfo,
    // third_sumcheck_msgs: SumcheckProof<E::ScalarField>,
    // third_round_message: Vec<E::ScalarField>,
    // third_round_proof: Vec<PCProof<E>>, // in eq_tilde_rx, eq_tilde_ry, val_a, val_b, val_c sequence
    pub lookup_proof: LogLookupProof<E>,
}
