use std::rc::Rc;

use ark_ec::pairing::Pairing;
use ark_ff::{AdditiveGroup, Field, One, Zero};
use ark_linear_sumcheck::ml_sumcheck::{
    protocol::{ListOfProductsOfPolynomials, PolynomialInfo},
    MLSumcheck,
};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, CommitterKey, VerifierKey},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_iter;
use rayon::prelude::*;

use crate::{
    transcript::Transcript,
    utils::{boost_degree, eq_eval, map_poly, two_pow_n},
    verifier::{batch_verify_poly, BatchOracleEval, VerificationResult},
};

type SumcheckProof<E> = ark_linear_sumcheck::ml_sumcheck::Proof<<E as Pairing>::ScalarField>;

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct LogLookupProof<E: Pairing> {
    pub sumcheck_pfs: SumcheckProof<E>,
    pub info: PolynomialInfo,
    pub point: Vec<E::ScalarField>,
    pub batch_oracle: BatchOracleEval<E>,
    pub degree_diff: usize,
}

impl<E: Pairing> LogLookupProof<E> {
    #[tracing::instrument(skip_all, name = "LogLookupProof::prove")]
    pub fn prove(
        query: &DenseMultilinearExtension<E::ScalarField>,
        table: &DenseMultilinearExtension<E::ScalarField>,
        m: &DenseMultilinearExtension<E::ScalarField>,
        ck: &CommitterKey<E>,
        // q_polys: &mut ListOfProductsOfPolynomials<E::ScalarField>,
        x: &E::ScalarField,
    ) -> (
        Vec<DenseMultilinearExtension<E::ScalarField>>,
        Vec<Commitment<E>>,
        Vec<DenseMultilinearExtension<E::ScalarField>>,
    ) {
        let phi_0 = map_poly(&table, |t| *x + t);

        let phi_1 = map_poly(&query, |q| *x + q);

        let mut minusone = E::ScalarField::one();
        minusone.neg_in_place();

        let mut phi_0_inv = phi_0.to_evaluations().clone();
        ark_ff::batch_inversion(&mut phi_0_inv);

        let h_0 = DenseMultilinearExtension::from_evaluations_vec(
            table.num_vars(),
            cfg_iter!(m.evaluations)
                .zip(&phi_0_inv)
                .map(|(m_val, inv)| *m_val * inv)
                .collect(),
        );

        let h_0 = boost_degree(&h_0, query.num_vars);
        let phi_0 = boost_degree(&phi_0, query.num_vars);
        // let m = boost_degree(&m, query.num_vars);

        let h_1 = map_poly(&phi_1, |p| E::ScalarField::one() / p);

        let h_0_commitment = MultilinearPC::commit(&ck, &h_0);
        let h_1_commitment = MultilinearPC::commit(&ck, &h_1);

        (
            [h_0, h_1].to_vec(),
            [h_0_commitment, h_1_commitment].to_vec(),
            [phi_0, phi_1].to_vec(),
        )
    }

    pub fn get_sumcheck_verifier_challenges<T: Transcript>(
        info: &PolynomialInfo,
        batch_oracle: &BatchOracleEval<E>,
        num_instance: usize,
        transcript: &mut T,
    ) -> (
        Vec<E::ScalarField>,
        Vec<Vec<E::ScalarField>>,
        Vec<E::ScalarField>,
    ) {
        let dimension = info.num_variables;

        let mut lookup_x = Vec::new();
        let labels = [b"x_r", b"x_c"];
        for i in 0..num_instance {
            lookup_x.push(transcript.get_scalar_challenge(labels[i]));
        }

        for i in 0..num_instance {
            transcript.append_serializable(b"batch_comm1", &batch_oracle.commitment[2 * i]);
            transcript.append_serializable(b"batch_comm2", &batch_oracle.commitment[2 * i + 1]);
        }

        let mut z = Vec::new();
        let mut lambda = Vec::new();
        for _ in 0..num_instance {
            z.push(transcript.get_vector_challenge(b"z", dimension));
            lambda.push(transcript.get_scalar_challenge(b"lambda"));
        }

        (lookup_x, z, lambda)
    }

    #[tracing::instrument(skip_all, name = "LogLookupProof::verify")]
    pub fn verify<T: Transcript>(
        info: &PolynomialInfo,
        sumcheck_pfs: &SumcheckProof<E>,
        batch_oracle: &BatchOracleEval<E>,
        degree_diff: usize,
        lookup_x: &[E::ScalarField],
        z: &Vec<Vec<E::ScalarField>>,
        lambda: &[E::ScalarField],
        vk: &VerifierKey<E>,
        transcript: &mut T,
        aux_eval: E::ScalarField,
        aux_sum: E::ScalarField,
    ) -> VerificationResult {
        let subclaim = MLSumcheck::verify_as_subprotocol(
            transcript,
            &info,
            E::ScalarField::zero() + aux_sum,
            &sumcheck_pfs,
        );
        if let Err(err) = subclaim {
            return Err(anyhow::anyhow!(err).context("sumcheck claim error"));
        };

        let subclaim = subclaim.unwrap();
        let scaling_factor = two_pow_n::<E::ScalarField>(degree_diff).inverse().unwrap();
        let point = subclaim.point;

        let mut res = aux_eval;

        for i in 0..z.len() {
            let batch_oracle_val = &batch_oracle.val[2 * i..2 * i + 2];
            let batch_oracle_debug_val = &batch_oracle.debug_val[3 * i..3 * i + 3];

            let mut eta = lambda[i];
            eta = eta * lambda[i];
            let q0 = batch_oracle_val[0] * lambda[i]
                + eq_eval(&point, &z[i])
                    * eta
                    * (batch_oracle_val[0]
                        * (batch_oracle_debug_val[2] + (lookup_x[i] * scaling_factor))
                        - scaling_factor * batch_oracle_debug_val[0]);

            eta = eta * lambda[i];
            let mut neg_h_1_oracle = batch_oracle_val[1];
            neg_h_1_oracle.neg_in_place();
            let q1 = neg_h_1_oracle * lambda[i]
                + eq_eval(&point, &z[i])
                    * eta
                    * (batch_oracle_val[1] * (batch_oracle_debug_val[1] + lookup_x[i])
                        - E::ScalarField::one());
            res = res + q0 + q1;
        }

        let eta = transcript.get_scalar_challenge(b"eta");
        let poly_oracle_verifications = batch_verify_poly(
            &batch_oracle.commitment,
            &batch_oracle.val,
            vk,
            &batch_oracle.proof,
            &point,
            eta,
        );

        if !poly_oracle_verifications {
            return Err(anyhow::anyhow!("poly oracle verification error")
                .context("poly oracle verification error"));
        }

        if res == subclaim.expected_evaluation {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "unexpected evaluation. expected: {:?}, actual: {:?}",
                subclaim.expected_evaluation,
                res
            )
            .context("unexpected evaluation"))
        }
    }
}
