use std::marker::PhantomData;

use anyhow::{ensure, Context};
use ark_ec::pairing::Pairing;
use ark_ff::{UniformRand, Zero};
use ark_poly::SparseMultilinearExtension;
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, Proof as PCProof, VerifierKey},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use super::{
    indexer::IndexVerifierKey,
    logup::LogLookupProof,
    zk::{zk_sumcheck_verifier_wrapper, ZKMLCommit},
    R1CSProof,
};
use crate::{
    math::MaskPolynomial,
    transcript::{Transcript, TranscriptMerlin},
    utils::{aggregate_comm, aggregate_eval, eq_eval, generate_dumb_sponge},
};

/// Verification result.
pub type VerificationResult = anyhow::Result<()>;

impl<E: Pairing> R1CSProof<E> {
    /// Verification function for SNARK proof.
    /// The input contains the R1CS instance and the verification key
    /// of polynomial commitment.
    #[tracing::instrument(skip_all, name = "R1CSProof::verify")]
    pub fn verify(
        &self,
        vk: &IndexVerifierKey<E>,
        assignment: &Vec<E::ScalarField>,
    ) -> VerificationResult {
        let mut transcript = TranscriptMerlin::new(b"dfs");
        let mut challenge_gen = generate_dumb_sponge::<E::ScalarField>();
        let mut v_state: VerifierState<E> = DFSVerifier::verifier_init(vk.padded_num_var);
        let mle_io_1_evals = assignment.iter().copied().enumerate().collect::<Vec<_>>();
        let mle_io_1 =
            SparseMultilinearExtension::from_evaluations(vk.padded_num_var, mle_io_1_evals.iter());
        let w_commitment = &self.witness_commitment;

        transcript.append_serializable(b"w_commitment", w_commitment);
        let _ = DFSVerifier::verifier_first_round(&mut v_state, &mut transcript);

        let sub_claim_1 = zk_sumcheck_verifier_wrapper(
            &vk.vk_mask,
            &self.first_sumcheck_msgs,
            &mut transcript,
            &mut challenge_gen,
            E::ScalarField::zero(),
        )
        .context("while verifying first sumcheck")?;
        let r_x = sub_claim_1.point;
        let actual_eval =
            (self.va * self.vb - self.vc) * eq_eval(&v_state.self_randomness[0][..], &r_x[..]);
        ensure!(
            sub_claim_1.expected_evaluation == actual_eval,
            anyhow::anyhow!(
                "unexpected evaluation. expected: {:?}, actual: {:?}",
                sub_claim_1.expected_evaluation,
                actual_eval
            )
            .context("while verifying first sumcheck")
        );

        let val_r1 = vec![self.va, self.vb, self.vc];
        transcript.append_serializable(b"val_r1", &val_r1);
        transcript.append_serializable(b"first_sumcheck_msgs", &self.first_sumcheck_msgs);

        let _ = DFSVerifier::verifier_second_round(&mut v_state, &mut transcript);

        let sumcheck_second_round = &self.second_sumcheck_msgs;
        let checksum_2: E::ScalarField = val_r1
            .iter()
            .zip(v_state.self_randomness[1].iter())
            .map(|(x, y)| *x * y)
            .sum();

        let sub_claim_2 = zk_sumcheck_verifier_wrapper(
            &vk.vk_mask,
            &sumcheck_second_round,
            &mut transcript,
            &mut challenge_gen,
            checksum_2,
        )
        .context("while verifying second sumcheck")?;
        transcript.append_serializable(b"second_sumcheck_msgs", &self.second_sumcheck_msgs);
        let r_y = sub_claim_2.point;

        let w_proof = &self.witness_proof;
        let w_value = self.witness_eval;
        let flag_zkml = ZKMLCommit::<E, MaskPolynomial<E>>::check(
            &vk.vk_w,
            &w_commitment,
            &r_y,
            w_value,
            &w_proof,
        );

        ensure!(
            flag_zkml,
            anyhow::anyhow!("zkml openning check failed")
                .context("while verifying second sumcheck")
        );

        let z = crate::utils::eval_sparse_mle(&mle_io_1, &r_y[..]) + w_value;
        ensure!(
            sub_claim_2.expected_evaluation == self.val_M * z,
            anyhow::anyhow!(
                "unexpected evaluation. expected: {:?}, actual: {:?}",
                sub_claim_2.expected_evaluation,
                self.val_M * z
            )
            .context("while verifying second sumcheck")
        );

        transcript.append_serializable(b"witness_eval", &[self.witness_eval, self.val_M]);
        transcript.append_serializable(b"eq_tilde_rx_comm", &self.eq_tilde_rx_commitment);
        transcript.append_serializable(b"eq_tilde_ry_comm", &self.eq_tilde_ry_commitment);
        transcript.append_serializable(b"w_proof", &w_proof.clone());

        let _ = DFSVerifier::verifier_fifth_round(&mut v_state, &mut transcript);

        let (lookup_x, z, lambda) = LogLookupProof::<E>::get_sumcheck_verifier_challenges(
            &self.lookup_proof.info,
            &self.lookup_proof.batch_oracle,
            2,
            &mut transcript,
        );

        let aux_eval = self.lookup_proof.batch_oracle.val[4]
            * self.lookup_proof.batch_oracle.val[5]
            * (self.lookup_proof.batch_oracle.val[6] * v_state.self_randomness[1][0]
                + self.lookup_proof.batch_oracle.val[7] * v_state.self_randomness[1][1]
                + self.lookup_proof.batch_oracle.val[8] * v_state.self_randomness[1][2]);

        LogLookupProof::<E>::verify(
            &self.lookup_proof.info,
            &self.lookup_proof.sumcheck_pfs,
            &self.lookup_proof.batch_oracle,
            self.lookup_proof.degree_diff,
            &lookup_x,
            &z,
            &lambda,
            &vk.vk_index,
            &mut transcript,
            aux_eval,
            self.val_M,
        )
        .context("while verifying lookup proof")?;

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
}

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
