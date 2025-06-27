use std::marker::PhantomData;
use std::ops::Index;

use ark_ec::pairing::{self, Pairing};
use ark_ff::UniformRand;
use ark_ff::Zero;
use ark_linear_sumcheck::ml_sumcheck::protocol::PolynomialInfo;
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::{MultilinearExtension, SparseMultilinearExtension};
use ark_poly_commit::challenge::ChallengeGenerator;
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_std::rand::RngCore;

use crate::snark::indexer::*;
use crate::snark::zk::zk_sumcheck_verifier_wrapper;
use crate::snark::zk::ZKMLCommit;
use crate::snark::OracleEval;
use crate::subprotocols::loglookup::LogLookupProof;
use crate::utils::aggregate_comm;
use crate::utils::aggregate_eval;
use crate::utils::generate_dumb_sponge;
use crate::utils::{eq_eval, generate_eq};
use crate::{transcript::LookupTranscript, R1CSInstance};
use crate::{VerificationResult, PROTOCOL_NAME};
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use super::{
    prover::{self, MaskPolynomial, ProverMessage},
    R1CSProof,
};
impl<E: Pairing> R1CSProof<E> {
    /// Verification function for SNARK proof.
    /// The input contains the R1CS instance and the verification key
    /// of polynomial commitment.
    pub fn verify(
        &self,
        vk: &IndexVerifierKey<E>,
        assignment: &Vec<E::ScalarField>,
    ) -> VerificationResult {
        let mut rng = Blake2s512Rng::setup();
        assert!(rng.feed(&"initialize".as_bytes()).is_ok());
        let mut challenge_gen =
            ChallengeGenerator::new_univariate(&mut generate_dumb_sponge::<E::ScalarField>());
        let mut v_state: VerifierState<E> = DFSVerifier::verifier_init(vk.padded_num_var);
        let mle_io_1_evals = assignment.iter().copied().enumerate().collect::<Vec<_>>();
        let mle_io_1 =
            SparseMultilinearExtension::from_evaluations(vk.padded_num_var, mle_io_1_evals.iter());
        let w_commitment = &self.witness_commitment;

        rng.feed(w_commitment);
        let verifier_first_message = DFSVerifier::verifier_first_round(&mut v_state, &mut rng);

        let r1_aux = PolynomialInfo {
            max_multiplicands: 3,
            num_variables: v_state.num_variables,
        };

        let (flag_wrap_1, sub_claim_1) = zk_sumcheck_verifier_wrapper(
            &vk.vk_mask,
            &self.first_sumcheck_msgs,
            &mut rng,
            &mut challenge_gen,
            E::ScalarField::zero(),
        );
        let r_x = sub_claim_1.point;
        let flag_1: bool = sub_claim_1.expected_evaluation
            == (self.va * self.vb - self.vc) * eq_eval(&v_state.self_randomness[0][..], &r_x[..]);
        assert!(flag_1 && flag_wrap_1);

        let val_r1 = vec![self.va, self.vb, self.vc];
        rng.feed(&val_r1);
        rng.feed(&self.first_sumcheck_msgs);
        let verifier_second_message = DFSVerifier::verifier_second_round(&mut v_state, &mut rng);

        let sumcheck_second_round = &self.second_sumcheck_msgs;
        println!("val_r1: {:?}", val_r1);
        println!("v_state.self_randomness[1]: {:?}", v_state.self_randomness[1]);
        let checksum_2: E::ScalarField = val_r1
            .iter()
            .zip(v_state.self_randomness[1].iter())
            .map(|(x, y)| *x * y)
            .sum();

        println!("checksum_2: {:?}", checksum_2);
        let r2_aux = PolynomialInfo {
            max_multiplicands: 2,
            num_variables: v_state.num_variables,
        };

        let (flag_wrap_2, sub_claim_2) = zk_sumcheck_verifier_wrapper(
            &vk.vk_mask,
            &sumcheck_second_round,
            &mut rng,
            &mut challenge_gen,
            checksum_2,
        );
        let r_y = sub_claim_2.point;

        let w_proof = &self.witness_proof;
        let w_value = self.witness_eval;
        let flag_2 = ZKMLCommit::<E, MaskPolynomial<E>>::check(
            &vk.vk_w,
            &w_commitment,
            &r_y,
            w_value,
            &w_proof,
        );

        assert!(flag_2 && flag_wrap_2);

        // let z = mle_io_1.evaluate(&r_y[..]).unwrap() + w_value;
        let z = crate::utils::eval_sparse_mle(&mle_io_1, &r_y[..]) + w_value;

        let flag_3 = sub_claim_2.expected_evaluation == self.val_M * z;
        assert!(flag_3);

        let _ = rng.feed(&vec![self.witness_eval, self.val_M]);
        let _ = rng.feed(&w_proof.clone());
        let _ = rng.feed(&self.eq_tilde_rx_commitment);
        let _ = rng.feed(&self.eq_tilde_ry_commitment);
        let _ = rng.feed(&self.second_sumcheck_msgs);

        let verifier_fifth_message = DFSVerifier::verifier_fifth_round(&mut v_state, &mut rng);

        let (lookup_x, z, lambda) = LogLookupProof::<E>::verify_before_sumcheck(
            &self.lookup_proof.info,
            &self.lookup_proof.batch_oracle,
            2,
            &mut rng,
        );

        let aux_eval = self.lookup_proof.batch_oracle.val[4]
            * self.lookup_proof.batch_oracle.val[5]
            * (self.lookup_proof.batch_oracle.val[6] * v_state.self_randomness[1][0]
                + self.lookup_proof.batch_oracle.val[7] * v_state.self_randomness[1][1]
                + self.lookup_proof.batch_oracle.val[8] * v_state.self_randomness[1][2]);

        println!("verifier aux_sum (val_M): {:?}", self.val_M);
        assert!(LogLookupProof::<E>::verify(
            &self.lookup_proof.info,
            &self.lookup_proof.sumcheck_pfs,
            &self.lookup_proof.batch_oracle,
            self.lookup_proof.degree_diff,
            &lookup_x,
            &z,
            &lambda,
            &vk.vk_index,
            &mut rng,
            aux_eval,
            self.val_M,
        )
        .is_ok());

        Ok(())
    }
}

pub struct DFSVerifier<E: Pairing> {
    _marker: PhantomData<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct VerifierState<E: Pairing> {
    pub self_randomness: Vec<Vec<E::ScalarField>>,
    pub num_variables: usize,
}
#[derive(Debug)]
pub struct VerifierMessage<E: Pairing> {
    pub verifier_message: Vec<E::ScalarField>,
}

impl<E: Pairing> DFSVerifier<E> {
    // io = (io, 1), |w|=|io,1|
    /// initialize verifier. Verifier only need to store its randomness when interactive with prover.
    pub fn verifier_init(num_variables: usize) -> VerifierState<E> {
        VerifierState {
            self_randomness: Vec::new(),
            num_variables: num_variables,
        }
    }
    ///On receiving message from prover_first_round, use the rng indicated by prover message to output a challenge vector.
    pub fn verifier_first_round<R: RngCore>(
        state: &mut VerifierState<E>,
        rng: &mut R,
    ) -> VerifierMessage<E> {
        let mut message: Vec<E::ScalarField> = Vec::new();
        for _ in 0..state.num_variables {
            message.push(E::ScalarField::rand(rng));
        }
        state.self_randomness.push(message.clone());
        VerifierMessage {
            verifier_message: message,
        }
    }
    ///On receiving message from prover_second_round, use the rng indicated by prover message to output 3 challenges .
    pub fn verifier_second_round<R: RngCore>(
        state: &mut VerifierState<E>,
        rng: &mut R,
    ) -> VerifierMessage<E> {
        let message = vec![
            E::ScalarField::rand(rng),
            E::ScalarField::rand(rng),
            E::ScalarField::rand(rng),
        ];
        state.self_randomness.push(message.clone());
        VerifierMessage {
            verifier_message: message,
        }
    }
    // verifier do nothing in round 3

    pub fn verifier_third_round_unused<R: RngCore>(
        state: &mut VerifierState<E>,
        rng: &mut R,
    ) -> VerifierMessage<E> {
        let message = vec![E::ScalarField::rand(rng)];
        state.self_randomness.push(message.clone());
        VerifierMessage {
            verifier_message: message,
        }
    }

    pub fn verifier_fifth_round<R: RngCore>(
        state: &mut VerifierState<E>,
        rng: &mut R,
    ) -> VerifierMessage<E> {
        let message = vec![E::ScalarField::rand(rng)];
        state.self_randomness.push(message.clone());
        VerifierMessage {
            verifier_message: message,
        }
    }
    // The function of verifier_check should be moved to method 'verify', and it will not be called in method 'prove'.
    /* pub fn verifier_check<R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>>(
        prover_message: &R1CSProof<E>,
        state: &VerifierState<E>,
        index: &IndexVerifierKey<E>,
        mle_io_1: &DenseMultilinearExtension<E::ScalarField>,
        rng: &mut R
    ) -> bool {
        let w_commitment = &prover_message.witness_commitment;

        let sumcheck_first_round = &prover_message.first_sumcheck_msgs;
        let val_r1 = vec![prover_message.va, prover_message.vb, prover_message.vc];
        let r1_aux = PolynomialInfo {
            max_multiplicands: 3,
            num_variables: state.num_variables,
        };

        let fsrng = rng;

        let sub_claim_1 = MLSumcheck::verify_as_subprotocol(
            fsrng,
            &r1_aux,
            E::ScalarField::zero(),
            &sumcheck_first_round,
        )
        .unwrap();
        let r_x = sub_claim_1.point;
        let flag_1: bool = sub_claim_1.expected_evaluation
            == (val_r1[0] * val_r1[1] - val_r1[2])
                * eq_eval(&state.self_randomness[0][..], &r_x[..]);

        let sumcheck_second_round = &prover_message.second_sumcheck_msgs;
        let checksum_2: E::ScalarField = val_r1
            .iter()
            .zip(state.self_randomness[1].iter())
            .map(|(x, y)| *x * y)
            .sum();
        let r2_aux = PolynomialInfo {
            max_multiplicands: 2,
            num_variables: state.num_variables,
        };

        let sub_claim_2 =
            MLSumcheck::verify_as_subprotocol(fsrng, &r2_aux, checksum_2, &sumcheck_second_round)
                .unwrap();
        let r_y = sub_claim_2.point;

        let w_proof = &prover_message.witness_proof;
        let w_value = prover_message.witness_eval;
        let flag_2 = MultilinearPC::check(&index.vk, &w_commitment, &r_y, w_value, &w_proof);

        let z = (E::ScalarField::one() - r_y[state.num_variables - 1])
            *    mle_io_1
                .evaluate(&r_y[..(state.num_variables - 1)])
                .unwrap()
            + r_y[state.num_variables - 1] * w_value;

        let eq_rx = generate_eq(&r_x[..]);
        let eq_ry = generate_eq(&r_y[..]);

        let mut val_a = E::ScalarField::zero();
        let mut val_b = E::ScalarField::zero();
        let mut val_c = E::ScalarField::zero();
        index
            .val_a
            .evaluations
            .iter()
            .zip(index.val_b.evaluations.iter())
            .zip(index.val_c.evaluations.iter())
            .zip(index.row.iter())
            .zip(index.col.iter())
            .for_each(|((((v_a, v_b), v_c), row), col)| {
                val_a += *v_a * eq_rx.index(*row) * eq_ry.index(*col);
                val_b += *v_b * eq_rx.index(*row) * eq_ry.index(*col);
                val_c += *v_c * eq_rx.index(*row) * eq_ry.index(*col)
            });
        let flag_3 = sub_claim_2.expected_evaluation
            == (val_a * state.self_randomness[1][0]
                + val_b * state.self_randomness[1][1]
                + val_c * state.self_randomness[1][2])
                * z;

        flag_1 & flag_2 & flag_3
    } */
}
