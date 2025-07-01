use ark_ec::pairing::Pairing;
use ark_linear_sumcheck::ml_sumcheck::{protocol::PolynomialInfo, Proof as SumcheckProof};
use ark_poly_commit::multilinear_pc::{
    data_structures::Commitment, data_structures::Proof as PCProof,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use blake2::digest::generic_array::typenum::Log2;

use crate::utils::{
    aggregate_comm, aggregate_eval, aggregate_poly, boost_degree, generate_eq, map_poly, two_pow_n,
};
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::{DenseMultilinearExtension, Polynomial};
use ark_poly_commit::multilinear_pc::{
    data_structures::CommitterKey, data_structures::VerifierKey, MultilinearPC,
};
use rand::RngCore;

use crate::logup::LogLookupProof;

use self::zk::{ZKMLProof, ZKSumcheckProof};
pub mod prover;

pub mod verifier;

pub mod indexer;

mod tests;
pub mod zk;
mod fixed_base;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct OracleEval<E: Pairing> {
    pub val: E::ScalarField,
    pub commitment: Commitment<E>,
    pub proof: PCProof<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct BatchOracleEval<E: Pairing> {
    pub val: Vec<E::ScalarField>,
    pub debug_val: Vec<E::ScalarField>,
    pub commitment: Vec<Commitment<E>>,
    pub proof: PCProof<E>,
}

impl<E: Pairing> OracleEval<E> {
    pub fn verify(&self, vk: &VerifierKey<E>, point: &[E::ScalarField]) -> bool {
        MultilinearPC::check(vk, &self.commitment, point, self.val, &self.proof)
    }
}

impl<E: Pairing> BatchOracleEval<E> {
    pub fn verify(
        &self,
        eta: E::ScalarField,
        vk: &VerifierKey<E>,
        point: &[E::ScalarField],
    ) -> bool {
        let batch_comm = aggregate_comm(eta, &self.commitment);
        let batch_eval = aggregate_eval(eta, &self.val[0..self.commitment.len()]);
        MultilinearPC::check(vk, &batch_comm, point, batch_eval, &self.proof)
    }
}

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

/// Batch opening polynomial
pub fn batch_open_poly<E: Pairing>(
    polys: &[&DenseMultilinearExtension<E::ScalarField>],
    comms: &[Commitment<E>],
    ck: &CommitterKey<E>,
    final_point: &[E::ScalarField],
    eta: E::ScalarField,
) -> BatchOracleEval<E> {
    let batch_poly = aggregate_poly(eta, &polys[0..comms.len()]);
    let batch_proof = MultilinearPC::open(&ck, &batch_poly, &final_point);
    let mut evals = Vec::new();
    let mut debug_evals = Vec::new();
    for p in &polys[0..comms.len()] {
        evals.push(p.evaluate(&final_point.to_vec()))
    }
    for p in &polys[comms.len()..] {
        debug_evals.push(p.evaluate(&final_point.to_vec()))
    }
    let batch_oracle = BatchOracleEval {
        val: evals,
        debug_val: debug_evals,
        commitment: comms.to_vec(),
        proof: batch_proof,
    };
    batch_oracle
}

/// Batch verify polynomial
pub fn batch_verify_poly<E: Pairing>(
    comms: &[Commitment<E>],
    evals: &[E::ScalarField],
    vk: &VerifierKey<E>,
    proof: &PCProof<E>,
    final_point: &[E::ScalarField],
    eta: E::ScalarField,
    // rng: &mut R,
) -> bool {
    // let eta: E::ScalarField = get_scalar_challenge(rng);

    let batch_comm = aggregate_comm(eta, comms);
    let batch_eval = aggregate_eval(eta, &evals);
    let res = MultilinearPC::check(vk, &batch_comm, final_point, batch_eval, proof);
    res
}
