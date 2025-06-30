use ark_std::time::Instant;
use std::collections::hash_map::RandomState;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::ops::Index;
use std::ops::Neg;
use std::rc::Rc;

use crate::snark::indexer::IndexVerifierKey;
use crate::utils::split_poly;
use crate::utils::split_vec;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::pairing::Pairing;
use ark_ff::{One, Zero};
use ark_linear_sumcheck::ml_sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::multivariate::{SparsePolynomial, SparseTerm};
use ark_poly::Polynomial;
use ark_poly::DenseMVPolynomial;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{cfg_into_iter, cfg_iter, cfg_iter_mut, cmp::max};

use crate::snark::{batch_open_poly, batch_verify_poly, BatchOracleEval, OracleEval};
use ark_poly_commit::multilinear_pc::{
    data_structures::Commitment, data_structures::Proof, MultilinearPC,
};
use ark_poly_commit::LabeledPolynomial;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use rand::Rng;
use rand::RngCore;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::verifier::VerifierMessage;
use super::zk::ZKMLCommit;
use super::{verifier, R1CSProof};
use crate::snark::indexer::IndexProverKey;
use crate::snark::verifier::{DFSVerifier, VerifierState};
use crate::snark::zk::zk_sumcheck_prover_wrapper;
use crate::snark::zk::ZKSumcheckProof;

use super::zk::{ZKMLCommitment, ZKMLCommitterKey, ZKMLProof};
use crate::subprotocols::loglookup::sumcheck_polynomial_list;
use crate::subprotocols::loglookup::LogLookupProof;
use crate::transcript::{get_scalar_challenge, get_vector_challenge};
use crate::utils::generate_dumb_sponge;
use crate::utils::{
    aggregate_comm, aggregate_eval, aggregate_poly, boost_degree, generate_eq, map_poly,
    normalized_multiplicities, two_pow_n,
};
use crate::utils::{dense_scalar_prod, feed_message};
use crate::Math;
use crate::{R1CSInstance, PROTOCOL_NAME};

type SumcheckProof<E> = ark_linear_sumcheck::ml_sumcheck::Proof<<E as Pairing>::ScalarField>;
pub type MaskPolynomial<E: Pairing> = SparsePolynomial<E::ScalarField, SparseTerm>;
impl<E: Pairing> R1CSProof<E> {
    /// Given as input the R1CS instance `r1cs` and the committer key `ck` for the polynomial commitment scheme,
    /// produce a new SNARK proof.
    pub fn new(
        r1cs: &R1CSInstance<E::ScalarField>,
        pk: &IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
        witness: &Vec<E::ScalarField>,
        assignment: &Vec<E::ScalarField>,
    ) -> R1CSProof<E>
    where
        E: Pairing,
    {
        let mut fs_rng = Blake2s512Rng::setup();
        let mut mask_rng = Blake2s512Rng::setup();
        assert!(fs_rng.feed(&"initialize".as_bytes()).is_ok());
        assert!(mask_rng.feed(&"initialize".as_bytes()).is_ok());
        let mut challenge_gen = generate_dumb_sponge::<E::ScalarField>();
        let (mut prover_message, _, _num_variables) = R1CSProof::prove(
            pk,
            vk,
            assignment.clone(),
            witness.clone(),
            r1cs.clone(),
            &mut fs_rng,
            &mut mask_rng,
            &mut challenge_gen,
        );
        let msg4 = prover_message.pop().unwrap();
        // let msg3 = prover_message.pop().unwrap();
        let msg2 = prover_message.pop().unwrap();
        let msg1 = prover_message.pop().unwrap();
        let msg0 = prover_message.pop().unwrap();
        let msg1_gm = msg1.group_message.unwrap();
        let msg2_gm = msg2.group_message.unwrap();
        let mut msg2_pm = msg2.zk_proof_message.unwrap();
        // let msg3_gm = msg3.group_message.unwrap();
        // let mut msg3_pm = msg3.proof_message.unwrap();

        R1CSProof {
            witness_commitment: (msg0.commitment_message.unwrap()),
            // first_sumcheck_polynomial_info: (PolynomialInfo { max_multiplicands: (3), num_variables: (num_variables) }),
            // highest level
            first_sumcheck_msgs: (msg1.zksumcheck_message.unwrap()),
            va: (msg1_gm[0]),
            vb: (msg1_gm[1]),
            vc: (msg1_gm[2]),
            // second_sumcheck_polynomial_info: (PolynomialInfo { max_multiplicands: (2), num_variables: (num_variables) }),
            // verify well formedness of eq(x)
            second_sumcheck_msgs: (msg2.zksumcheck_message.unwrap()),
            witness_eval: (msg2_gm[0]),
            witness_proof: msg2_pm,
            val_M: (msg2_gm[1]),
            eq_tilde_rx_commitment: (msg2.commitment_message.unwrap()),
            eq_tilde_ry_commitment: (msg2.commitment_message_2.unwrap()),
            // third_sumcheck_polynomial_info: (PolynomialInfo { max_multiplicands: (3), num_variables: (pk.num_variables_val) }),
            // verify well formedness of M~(rx, ry)
            // third_sumcheck_msgs: msg3.sumcheck_message.unwrap(),
            // third_round_message: msg3_gm,
            // third_round_proof: msg3_pm,
            // Lookup stuff for each component of M~(rx, ry)
            lookup_proof: msg4.lookup_message.unwrap(),
        }
    }
    /// Applying non-interactive prove by calling method of verifier.
    /// The whole process will be prover_init -> verifier_init -> prover_first_round -> verifier_first_round ->...-> prover_third_round.
    /// The prover will generate whole proof after the process. The transcript will be used to keep consinstency between method 'prove' and 'verify'
    fn prove<
        R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
        S: CryptographicSponge,
    >(
        index_pk: &IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
        io: Vec<E::ScalarField>,
        witness: Vec<E::ScalarField>,
        r1cs: R1CSInstance<E::ScalarField>,
        fs_rng: &mut R,
        mask_rng: &mut R,
        mask_challenge_generator_for_open: &mut S,
    ) -> (Vec<ProverMessage<E>>, Vec<VerifierMessage<E>>, usize) {
        //todo: padding the R1CS instance, witness and io
        let init_time = start_timer!(|| "Prover init");
        let mut prover_state =
            DFSProver::prover_init(&r1cs, index_pk, vk, witness.clone(), io.clone());
        end_timer!(init_time);
        let mut prover_message: Vec<ProverMessage<E>> = Vec::new();
        let mut verifier_state: VerifierState<E> =
            DFSVerifier::verifier_init(prover_state.num_variables);
        let mut verifier_message: Vec<VerifierMessage<E>> = Vec::new();

        // First round: the prover message is just the witness commitment
        let first_time = start_timer!(|| "Prover first round");
        let mut prover_first_message = DFSProver::prover_first_round(&mut prover_state, mask_rng);
        println!(
            "prover_first_message: {:?}",
            prover_first_message.commitment_message
        );
        prover_first_message = feed_message(fs_rng, prover_first_message);
        prover_message.push(prover_first_message);
        end_timer!(first_time);

        // This first challenge is used for checking the hadamard product of AB - C ?= 0.
        // The following sumcheck, in prover_second_round, doesn't verify the well formedness of its components, which future rounds will do.
        let verifier_first_message = DFSVerifier::verifier_first_round(&mut verifier_state, fs_rng);

        println!(
            "verifier_first_message.verifier_message: {:?}",
            verifier_first_message.verifier_message
        );

        let second_time = start_timer!(|| "Prover second round");
        // println!("verifier_first_message.verifier_message: {:?}", verifier_first_message.verifier_message);
        let mut prover_second_message = DFSProver::prover_second_round(
            &mut prover_state,
            &verifier_first_message.verifier_message,
            fs_rng,
            mask_rng,
            mask_challenge_generator_for_open,
        );
        prover_second_message = feed_message(fs_rng, prover_second_message);
        prover_message.push(prover_second_message);
        end_timer!(second_time);

        // This challenge is used for verifying the well formedness of the subclaim evaluation from the previous round.
        // In particular, this challenge batches a, b, and c polynomials using 3 scalars
        let verifier_second_message =
            DFSVerifier::verifier_second_round(&mut verifier_state, fs_rng);

        println!(
            "verifier_second_message: {:?}",
            verifier_second_message.verifier_message
        );

        // println!("verifier_state self_randomness: {:?}", verifier_state.self_randomness[1]);

        let third_time = start_timer!(|| "Prover third round");
        let mut prover_third_message = DFSProver::prover_third_round(
            &mut prover_state,
            &verifier_second_message.verifier_message,
            fs_rng,
            mask_rng,
            mask_challenge_generator_for_open,
        );
        prover_third_message = feed_message(fs_rng, prover_third_message);
        prover_message.push(prover_third_message);
        end_timer!(third_time);

        let verifier_fifth_message = DFSVerifier::verifier_fifth_round(&mut verifier_state, fs_rng);
        println!("verifier_fifth_message: {:?}", verifier_fifth_message.verifier_message);
        let holo_time = Instant::now();

        let fifth_time = start_timer!(|| "Prover fifth round");
        let mut prover_fifth_message = DFSProver::prover_fifth_round(
            &mut prover_state,
            fs_rng,
            verifier_fifth_message.verifier_message[0],
        );
        prover_fifth_message = feed_message(fs_rng, prover_fifth_message);
        prover_message.push(prover_fifth_message);
        end_timer!(fifth_time);

        println!("holography time: {:?}", holo_time.elapsed());

        verifier_message.push(verifier_first_message);
        verifier_message.push(verifier_second_message);
        verifier_message.push(verifier_fifth_message);
        (prover_message, verifier_message, prover_state.num_variables)
    }
}
pub struct ProverState<'a, E: Pairing> {
    witness_poly: DenseMultilinearExtension<E::ScalarField>,
    witness_mask: LabeledPolynomial<E::ScalarField, SparsePolynomial<E::ScalarField, SparseTerm>>,
    z: DenseMultilinearExtension<E::ScalarField>,
    z_a: DenseMultilinearExtension<E::ScalarField>,
    z_b: DenseMultilinearExtension<E::ScalarField>,
    z_c: DenseMultilinearExtension<E::ScalarField>,
    index: &'a IndexProverKey<E>,
    vk: Option<IndexVerifierKey<E>>,
    num_variables: usize,
    r_x: Vec<E::ScalarField>,
    r_y: Vec<E::ScalarField>,
    eq_rx: DenseMultilinearExtension<E::ScalarField>,
    eq_ry: DenseMultilinearExtension<E::ScalarField>,
    eq_tilde_rx: DenseMultilinearExtension<E::ScalarField>,
    eq_tilde_ry: DenseMultilinearExtension<E::ScalarField>,
    eq_tilde_rx_comm: Option<Commitment<E>>,
    eq_tilde_ry_comm: Option<Commitment<E>>,
    val_M: DenseMultilinearExtension<E::ScalarField>,
    prover_round: usize,
}
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct ProverMessage<E: Pairing> {
    pub sumcheck_message: Option<SumcheckProof<E>>,
    pub group_message: Option<Vec<E::ScalarField>>,
    pub zk_proof_message: Option<ZKMLProof<E>>,
    pub proof_message: Option<Vec<Proof<E>>>,
    pub commitment_message: Option<Commitment<E>>,
    pub commitment_message_2: Option<Commitment<E>>,
    pub lookup_message: Option<LogLookupProof<E>>,
    pub zksumcheck_message: Option<ZKSumcheckProof<E>>,
}

pub struct DFSProver<E: Pairing> {
    _pairing: PhantomData<E>,
}

impl<E: Pairing> DFSProver<E> {
    //using little endian as ml_sumcheck, i.e. io[0,0,0,0,1] = F_{io}(1,0,0,0,0).
    pub fn prover_init<'a>(
        r1cs: &R1CSInstance<E::ScalarField>,
        index: &'a IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
        mut witness: Vec<E::ScalarField>,
        io: Vec<E::ScalarField>,
    ) -> ProverState<'a, E> {
        let mut z_eval = cfg_iter!(io)
            .chain(&witness)
            .copied()
            .collect::<Vec<E::ScalarField>>();
        z_eval.resize(index.padded_num_var.pow2(), E::ScalarField::zero());
        let z = DenseMultilinearExtension::from_evaluations_vec(index.padded_num_var, z_eval);
        let mut w_eval = vec![E::ScalarField::zero(); io.len()];
        w_eval.append(&mut witness);
        w_eval.resize(index.padded_num_var.pow2(), E::ScalarField::zero());
        let num_variables = z.num_vars;
        let mut z_a = vec![E::ScalarField::zero(); num_variables.pow2()];
        let mut z_b = vec![E::ScalarField::zero(); num_variables.pow2()];
        let mut z_c = vec![E::ScalarField::zero(); num_variables.pow2()];

        //assert_eq!(r1cs.num_vars.log_2(), num_variables);
        for entry in &r1cs.A.M {
            z_a[entry.row] += entry.val * z.evaluations[entry.col];
        }
        for entry in &r1cs.B.M {
            z_b[entry.row] += entry.val * z.evaluations[entry.col];
        }
        for entry in &r1cs.C.M {
            z_c[entry.row] += entry.val * z.evaluations[entry.col];
        }
        ProverState {
            witness_poly: (DenseMultilinearExtension {
                evaluations: (w_eval),
                num_vars: (index.padded_num_var),
            }),
            witness_mask: LabeledPolynomial::<E::ScalarField, MaskPolynomial<E>>::new(
                "init_mask".into(),
                MaskPolynomial::<E>::zero(),
                None,
                None,
            ),
            z: (z),
            z_a: (DenseMultilinearExtension {
                evaluations: (z_a),
                num_vars: (num_variables),
            }),
            z_b: (DenseMultilinearExtension {
                evaluations: (z_b),
                num_vars: (num_variables),
            }),
            z_c: (DenseMultilinearExtension {
                evaluations: (z_c),
                num_vars: (num_variables),
            }),
            index: (index),
            vk: Some(vk.clone()),
            num_variables: (num_variables),
            r_x: (Vec::new()),
            r_y: (Vec::new()),
            eq_rx: DenseMultilinearExtension {
                evaluations: vec![E::ScalarField::zero()],
                num_vars: 0,
            },
            eq_ry: DenseMultilinearExtension {
                evaluations: vec![E::ScalarField::zero()],
                num_vars: 0,
            },
            eq_tilde_rx: DenseMultilinearExtension {
                evaluations: vec![E::ScalarField::zero()],
                num_vars: 0,
            },
            eq_tilde_ry: DenseMultilinearExtension {
                evaluations: vec![E::ScalarField::zero()],
                num_vars: 0,
            },
            eq_tilde_rx_comm: None,
            eq_tilde_ry_comm: None,
            val_M: DenseMultilinearExtension {
                evaluations: vec![E::ScalarField::zero()],
                num_vars: 0,
            },
            prover_round: 0,
        }
    }
    /// In round 1, prover only need to commit to witness w
    pub fn prover_first_round<R: RngCore>(
        state: &mut ProverState<E>,
        mask_rng: &mut R,
    ) -> ProverMessage<E> {
        assert_eq!(state.prover_round, 0);
        state.prover_round = 1;
        let zk_commitment_w = ZKMLCommit::<E, MaskPolynomial<E>>::commit(
            &state.index.ck_w,
            &state.witness_poly,
            2,
            None,
            mask_rng,
        );
        state.witness_mask = zk_commitment_w.1;
        ProverMessage {
            sumcheck_message: None,
            group_message: None,
            zk_proof_message: None,
            proof_message: None,
            commitment_message: Some(zk_commitment_w.0),
            commitment_message_2: None,
            lookup_message: None,
            zksumcheck_message: None,
        }
    }

    /// On input from verifier_first_round, compute and output first-round sumcheck message and v_A, v_B, v_C.
    /// Compute all things BEFORE Sumcheck#2 in Spartan.
    /// By this point, we should have computed values for v_A, v_B, and v_C.
    /// This sumcheck is for ensuring the correctness of e_x,
    ///
    pub fn prover_second_round<
        R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
        S: CryptographicSponge,
    >(
        state: &mut ProverState<E>,
        v_msg: &Vec<E::ScalarField>,
        fs_rng: &mut R,
        mask_rng: &mut R,
        mask_challenge_generator_for_open: &mut S,
    ) -> ProverMessage<E> {
        assert_eq!(state.prover_round, 1);
        let num_variables = state.num_variables;
        // println!("v_msg: {:?}", v_msg);
        let eq_func = generate_eq(&v_msg);
        let mut product_list = ListOfProductsOfPolynomials::new(num_variables);
        let A_B_hat = vec![
            Rc::new(state.z_a.clone()),
            Rc::new(state.z_b.clone()),
            Rc::new(eq_func.clone()),
        ];
        let C_hat = vec![Rc::new(state.z_c.clone()), Rc::new(eq_func.clone())];

        product_list.add_product(A_B_hat, E::ScalarField::one());
        product_list.add_product(C_hat, E::ScalarField::one().neg());

        let (pf, final_state) = zk_sumcheck_prover_wrapper(
            &product_list,
            fs_rng,
            mask_rng,
            &state.index.ck_mask,
            mask_challenge_generator_for_open,
        );

        let randomness = &final_state.prover_state.randomness;
        let (val_a, val_b, val_c) = (
            state.z_a.evaluate(randomness),
            state.z_b.evaluate(randomness),
            state.z_c.evaluate(randomness),
        );

        println!("val_a: {:?}", val_a);
        println!("val_b: {:?}", val_b);
        println!("val_c: {:?}", val_c);

        state.prover_round = 2;
        state.r_x = final_state.prover_state.randomness;
        state.eq_rx = generate_eq(&state.r_x[..]);

        ProverMessage {
            sumcheck_message: None,
            group_message: Some(vec![val_a, val_b, val_c]),
            zk_proof_message: None,
            proof_message: None,
            commitment_message: None,
            commitment_message_2: None,
            lookup_message: None,
            zksumcheck_message: Some(pf),
        }
    }

    /// Compute all things from Sumcheck#2.
    /// For the time prover does not need to compute M~(r_x, r_y) and it is left to linear verifier.

    pub fn prover_third_round<
        R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
        S: CryptographicSponge,
    >(
        state: &mut ProverState<E>,
        v_msg: &Vec<E::ScalarField>,
        fs_rng: &mut R,
        mask_rng: &mut R,
        mask_challenge_generator_for_open: &mut S,
    ) -> ProverMessage<E> {
        assert_eq!(state.prover_round, 2);
        println!("v_msg: {:?}", v_msg);

        let mut product_list = ListOfProductsOfPolynomials::new(state.num_variables);
        let mut A_rx = vec![E::ScalarField::zero(); state.num_variables.pow2()];
        let mut B_rx = vec![E::ScalarField::zero(); state.num_variables.pow2()];
        let mut C_rx = vec![E::ScalarField::zero(); state.num_variables.pow2()];

        for i in 0..state.num_variables.pow2() {
            if state.index.col[i] == usize::MAX {
                continue;
            }
            A_rx[state.index.col[i]] +=
                state.index.val_a[i] * state.eq_rx.index(state.index.row[i]);
            B_rx[state.index.col[i]] +=
                state.index.val_b[i] * state.eq_rx.index(state.index.row[i]);
            C_rx[state.index.col[i]] +=
                state.index.val_c[i] * state.eq_rx.index(state.index.row[i]);
        }

        println!("--------------------------------");
        println!("full");
        println!("A_rx: {:?}", A_rx[..5].to_vec());
        println!("B_rx: {:?}", B_rx[..5].to_vec());
        println!("C_rx: {:?}", C_rx[..5].to_vec());
        println!("--------");
        println!(
            "A_rx: {:?}",
            A_rx[state.num_variables.pow2() / 2..state.num_variables.pow2() / 2 + 5].to_vec()
        );
        println!(
            "B_rx: {:?}",
            B_rx[state.num_variables.pow2() / 2..state.num_variables.pow2() / 2 + 5].to_vec()
        );
        println!(
            "C_rx: {:?}",
            C_rx[state.num_variables.pow2() / 2..state.num_variables.pow2() / 2 + 5].to_vec()
        );
        println!("--------------------------------");

        println!("state.index.real_len_val: {:?}", state.index.real_len_val);
        println!(
            "state.num_variables.pow2()]: {:?}",
            state.num_variables.pow2()
        );

        let mut A_rx1 = vec![E::ScalarField::zero(); state.num_variables.pow2() / 2];
        let mut B_rx1 = vec![E::ScalarField::zero(); state.num_variables.pow2() / 2];
        let mut C_rx1 = vec![E::ScalarField::zero(); state.num_variables.pow2() / 2];
        // todo make 0..state.index.real_len_val
        for i in 0..state.num_variables.pow2() {
            if state.index.col[i] == usize::MAX {
                continue;
            }
            // if i >= state.num_variables.pow2() / 2 {
            //     continue;
            // }
            let mut col = state.index.col[i];
            // if state.index.col[i] >= state.num_variables.pow2() / 2 {
            //     col = state.index.col[i] - state.num_variables.pow2() / 2;
            // }
            // let i_chunk = i % (state.num_variables.pow2() / 2);
            A_rx1[col] += state.index.val_a[i] * state.eq_rx.index(state.index.row[i]);
            B_rx1[col] += state.index.val_b[i] * state.eq_rx.index(state.index.row[i]);
            C_rx1[col] += state.index.val_c[i] * state.eq_rx.index(state.index.row[i]);
        }

        println!("--------------------------------");
        println!("half 1");
        println!("eq_rx: {:?}", state.eq_rx.evaluations[..5].to_vec());
        println!("val_a: {:?}", state.index.val_a.evaluations[..5].to_vec());
        println!("val_b: {:?}", state.index.val_b.evaluations[..5].to_vec());
        println!("val_c: {:?}", state.index.val_c.evaluations[..5].to_vec());

        println!("A_rx1: {:?}", A_rx1[..5].to_vec());
        println!("B_rx1: {:?}", B_rx1[..5].to_vec());
        println!("C_rx1: {:?}", C_rx1[..5].to_vec());
        println!("--------------------------------");

        let mut A_rx2 = vec![E::ScalarField::zero(); state.num_variables.pow2() / 2];
        let mut B_rx2 = vec![E::ScalarField::zero(); state.num_variables.pow2() / 2];
        let mut C_rx2 = vec![E::ScalarField::zero(); state.num_variables.pow2() / 2];
        for i in 0..state.num_variables.pow2() {
            if state.index.col[i] == usize::MAX {
                continue;
            }
            let mut col = state.index.col[i];
            // if state.index.col[i] >= state.num_variables.pow2() / 2 {
            //     col = state.index.col[i] - state.num_variables.pow2() / 2;
            // }

            // A_rx2[col] +=
            //     state.index.val_a[i] * state.eq_rx.index(state.index.row[i]);
            // B_rx2[col] +=
            //     state.index.val_b[i] * state.eq_rx.index(state.index.row[i]);
            // C_rx2[col] +=
            //     state.index.val_c[i] * state.eq_rx.index(state.index.row[i]);
        }
        println!("--------------------------------");
        println!("half 2");
        println!(
            "eq_rx: {:?}",
            state.eq_rx.evaluations
                [state.num_variables.pow2() / 2..state.num_variables.pow2() / 2 + 5]
                .to_vec()
        );
        println!(
            "state.index.row: {:?}",
            state.index.row[state.num_variables.pow2() / 2..state.num_variables.pow2() / 2 + 5]
                .to_vec()
        );
        println!(
            "state.index.row: {:?}",
            state.index.row[state.num_variables.pow2() - 5..].to_vec()
        );
        // println!(
        //     "eq_rx2: {:?}",
        //     state.eq_rx.evaluations
        //         [state.eq_rx.evaluations.len() - 5..]
        //         .to_vec()
        // );

        println!(
            "val_a: {:?}",
            state.index.val_a.evaluations
                [state.num_variables.pow2() / 2..state.num_variables.pow2() / 2 + 5]
                .to_vec()
        );
        println!(
            "val_b: {:?}",
            state.index.val_b.evaluations
                [state.num_variables.pow2() / 2..state.num_variables.pow2() / 2 + 5]
                .to_vec()
        );
        println!(
            "val_c: {:?}",
            state.index.val_c.evaluations
                [state.num_variables.pow2() / 2..state.num_variables.pow2() / 2 + 5]
                .to_vec()
        );
        // println!("val_a: {:?}", state.index.val_a.evaluations[state.num_variables.pow2() - 5..].to_vec());
        // println!("val_b: {:?}", state.index.val_b.evaluations[state.num_variables.pow2() - 5..].to_vec());
        // println!("val_c: {:?}", state.index.val_c.evaluations[state.num_variables.pow2() - 5..].to_vec());
        println!("A_rx2: {:?}", A_rx2[..5].to_vec());
        println!("B_rx2: {:?}", B_rx2[..5].to_vec());
        println!("C_rx2: {:?}", C_rx2[..5].to_vec());
        println!("--------------------------------");

        // let [mut A_rx1, mut A_rx2] = split_vec(&A_rx, 1).try_into().unwrap();
        // let [mut B_rx1, mut B_rx2] = split_vec(&B_rx, 1).try_into().unwrap();
        // let [mut C_rx1, mut C_rx2] = split_vec(&C_rx, 1).try_into().unwrap();

        A_rx1.append(&mut A_rx2);
        B_rx1.append(&mut B_rx2);
        C_rx1.append(&mut C_rx2);

        A_rx = A_rx1;
        B_rx = B_rx1;
        C_rx = C_rx1;

        let A_hat = vec![
            Rc::new(DenseMultilinearExtension {
                evaluations: (A_rx),
                num_vars: (state.num_variables),
            }),
            Rc::new(state.z.clone()),
        ];
        let B_hat = vec![
            Rc::new(DenseMultilinearExtension {
                evaluations: (B_rx),
                num_vars: (state.num_variables),
            }),
            Rc::new(state.z.clone()),
        ];
        let C_hat = vec![
            Rc::new(DenseMultilinearExtension {
                evaluations: (C_rx),
                num_vars: (state.num_variables),
            }),
            Rc::new(state.z.clone()),
        ];

        product_list.add_product(A_hat, v_msg[0]);
        product_list.add_product(B_hat, v_msg[1]);
        product_list.add_product(C_hat, v_msg[2]);

        let (pf, final_state) = zk_sumcheck_prover_wrapper(
            &product_list,
            fs_rng,
            mask_rng,
            &state.index.ck_mask,
            mask_challenge_generator_for_open,
        );
        state.prover_round = 3;
        println!("final_point: {:?}", final_state.prover_state.randomness);
        state.r_y = final_state.prover_state.randomness;

        let eq_ry = generate_eq(&state.r_y[..]);

        let val_w = state.witness_poly.evaluate(&state.r_y);

        state.eq_ry = eq_ry;

        let mut val_a = E::ScalarField::zero();
        let mut val_b = E::ScalarField::zero();
        let mut val_c = E::ScalarField::zero();
        let mut val_a_temp = E::ScalarField::zero();
        let mut val_b_temp = E::ScalarField::zero();
        let mut val_c_temp = E::ScalarField::zero();
        let chunk_size = state.index.num_variables_val.pow2() / 4;
        println!("chunk_size: {:?}", chunk_size);
        let mut j = 0;
        for (i, ((((v_a, v_b), v_c), row), col)) in state
            .index
            .val_a
            .evaluations
            .iter()
            .zip(state.index.val_b.evaluations.iter())
            .zip(state.index.val_c.evaluations.iter())
            .zip(state.index.row.iter())
            .zip(state.index.col.iter())
            .enumerate()
        {
            if i < state.index.real_len_val {
                val_a += *v_a * state.eq_rx.index(*row) * state.eq_ry.index(*col);
                val_b += *v_b * state.eq_rx.index(*row) * state.eq_ry.index(*col);
                val_c += *v_c * state.eq_rx.index(*row) * state.eq_ry.index(*col);
                val_a_temp += *v_a * state.eq_rx.index(*row) * state.eq_ry.index(*col);
                val_b_temp += *v_b * state.eq_rx.index(*row) * state.eq_ry.index(*col);
                val_c_temp += *v_c * state.eq_rx.index(*row) * state.eq_ry.index(*col);

                if i == chunk_size * (j + 1) - 1 || i == state.index.real_len_val - 1 {
                    println!("--------------------------------");
                    println!("chunk_size: {:?}", i + 1);
                    println!("{j} val_a: {:?}", val_a_temp);
                    println!("{j} val_b: {:?}", val_b_temp);
                    println!("{j} val_c: {:?}", val_c_temp);
                    println!("--------------------------------");
                    j += 1;
                    val_a_temp = E::ScalarField::zero();
                    val_b_temp = E::ScalarField::zero();
                    val_c_temp = E::ScalarField::zero();
                }
            }
        }

        println!("--------------------------------");
        println!("agg val_a: {:?}", val_a);
        println!("agg val_b: {:?}", val_b);
        println!("agg val_c: {:?}", val_c);
        println!("--------------------------------");

        let val_m = val_a * v_msg[0] + val_b * v_msg[1] + val_c * v_msg[2];

        state.val_M = dense_scalar_prod(&v_msg[0], &state.index.val_a)
            + dense_scalar_prod(&v_msg[1], &state.index.val_b)
            + dense_scalar_prod(&v_msg[2], &state.index.val_c);

        let [val_m0, val_m1, val_m2, val_m3] = split_poly(&state.val_M, 2).try_into().unwrap();

        println!("--------------------------------");
        println!("val_m0 poly: {:?}", val_m0.evaluations[..5].to_vec());
        println!("val_m1 poly: {:?}", val_m1.evaluations[..5].to_vec());
        println!("val_m2 poly: {:?}", val_m2.evaluations[..5].to_vec());
        println!("val_m3 poly: {:?}", val_m3.evaluations[..5].to_vec());
        println!("val_m: {:?}", val_m);
        println!("--------------------------------");

        let mut eq_tilde_x: Vec<E::ScalarField> =
            vec![E::ScalarField::zero(); state.index.num_variables_val.pow2()];
        let mut eq_tilde_y: Vec<E::ScalarField> =
            vec![E::ScalarField::zero(); state.index.num_variables_val.pow2()];
        for i in 0..state.index.real_len_val {
            eq_tilde_x[i] = *state.eq_rx.index(state.index.row[i]);
            eq_tilde_y[i] = *state.eq_ry.index(state.index.col[i]);
        }
        // assert!(eq_tilde_y == state.eq_ry.evaluations);
        state.eq_tilde_rx = DenseMultilinearExtension::from_evaluations_vec(
            state.index.num_variables_val,
            eq_tilde_x,
        );
        state.eq_tilde_ry = DenseMultilinearExtension::from_evaluations_vec(
            state.index.num_variables_val,
            eq_tilde_y,
        );

        let [eq_tilde_x0, eq_tilde_x1, eq_tilde_x2, eq_tilde_x3] = split_poly(&state.eq_tilde_rx, 2).try_into().unwrap();
        let [eq_tilde_y0, eq_tilde_y1, eq_tilde_y2, eq_tilde_y3] = split_poly(&state.eq_tilde_ry, 2).try_into().unwrap();

        println!("--------------------------------");
        println!("eq_tilde_x0: {:?}", eq_tilde_x0.evaluations[..5].to_vec());
        println!("eq_tilde_y0: {:?}", eq_tilde_y0.evaluations[..5].to_vec());
        println!("--------------------------------");
        println!("eq_tilde_x1: {:?}", eq_tilde_x1.evaluations[..5].to_vec());
        println!("eq_tilde_y1: {:?}", eq_tilde_y1.evaluations[..5].to_vec());
        println!("--------------------------------");
        println!("eq_tilde_x2: {:?}", eq_tilde_x2.evaluations[..5].to_vec());
        println!("eq_tilde_y2: {:?}", eq_tilde_y2.evaluations[..5].to_vec());
        println!("--------------------------------");
        println!("eq_tilde_x3: {:?}", eq_tilde_x3.evaluations[..5].to_vec());
        println!("eq_tilde_y3: {:?}", eq_tilde_y3.evaluations[..5].to_vec());
        println!("--------------------------------");

        let rx_comm = MultilinearPC::commit(&state.index.ck_index, &state.eq_tilde_rx);
        let ry_comm = MultilinearPC::commit(&state.index.ck_index, &state.eq_tilde_ry);
        state.eq_tilde_rx_comm = Some(rx_comm.clone());
        state.eq_tilde_ry_comm = Some(ry_comm.clone());

        ProverMessage {
            sumcheck_message: None,
            group_message: Some(vec![val_w, val_m]),
            zk_proof_message: Some(ZKMLCommit::<E, MaskPolynomial<E>>::open(
                &state.index.ck_w,
                &state.witness_poly,
                &state.witness_mask,
                &state.r_y[..],
            )),
            proof_message: None,
            commitment_message: Some(rx_comm.clone()),
            commitment_message_2: Some(ry_comm.clone()),
            lookup_message: None,
            zksumcheck_message: Some(pf),
        }
    }

    // /// Sumcheck proof for integrity of M~(rx, ry)
    // pub fn prover_fourth_round_unused<
    //     R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
    // >(
    //     state: &mut ProverState<E>,
    //     v_msg: &Vec<E::ScalarField>,
    //     rng: &mut R,
    // ) -> ProverMessage<E> {
    //     let fourth_round_sc = start_timer!(|| "fourth round sumcheck");
    //     let prod = vec![
    //         Rc::new(state.eq_tilde_rx.clone()),
    //         Rc::new(state.eq_tilde_ry.clone()),
    //         Rc::new(state.val_M.clone()),
    //     ];
    //     let mut product_list = ListOfProductsOfPolynomials::new(state.index.num_variables_val);
    //     product_list.add_product(prod, E::ScalarField::one());
    //     let (pf, final_state) = MLSumcheck::prove_as_subprotocol(rng, &product_list).unwrap();
    //     let point = final_state.randomness;
    //     end_timer!(fourth_round_sc);
    //     let eta = v_msg[0];
    //     let aggrerate_polynomial = aggregate_poly(
    //         eta,
    //         &vec![
    //             &state.eq_tilde_rx,
    //             &state.eq_tilde_ry,
    //             &state.index.val_a,
    //             &state.index.val_b,
    //             &state.index.val_c,
    //         ][..],
    //     );
    //     let fourth_round_open = start_timer!(|| "fourth round open");

    //     let proof_aggrerate =
    //         MultilinearPC::open(&state.index.ck_index, &aggrerate_polynomial, &point);

    //     let val_x = state.eq_tilde_rx.evaluate(&point).unwrap();
    //     let val_y = state.eq_tilde_ry.evaluate(&point).unwrap();
    //     let val_a = state.index.val_a.evaluate(&point).unwrap();
    //     let val_b = state.index.val_b.evaluate(&point).unwrap();
    //     let val_c = state.index.val_c.evaluate(&point).unwrap();
    //     end_timer!(fourth_round_open);
    //     //send proof of val_a, val_b, val_c at the point as well
    //     ProverMessage {
    //         sumcheck_message: Some(pf),
    //         zk_proof_message: None,
    //         group_message: Some(vec![val_x, val_y, val_a, val_b, val_c]),
    //         proof_message: Some(vec![proof_aggrerate]),
    //         commitment_message: None,
    //         commitment_message_2: None,
    //         lookup_message: None,
    //         zksumcheck_message: None,
    //     }
    // }

    /// Prover and verifier will call loglookup to validate M~(r_x,r_y)
    /// Specifically, they'll verify that all of the eq~ used to calculate M~ are valid eq~ evaluations.
    /// Here, verifier message is batching scalar gamma. All other verifier challenges are implied in the proof generation.
    pub fn prover_fifth_round<R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>>(
        state: &mut ProverState<E>,
        rng: &mut R,
        v_msg: E::ScalarField,
    ) -> ProverMessage<E> {
        let q_num_vars = state.index.real_len_val.log_2();
        let [_, _, _, eq_tilde_x3] = split_poly(&state.eq_tilde_ry, 2).try_into().unwrap();
        let [_, _, _, col3] = split_vec(&state.index.col, 2).try_into().unwrap();
        println!("--------------------------------");
        println!("eq_tilde_ry3: {:?}", eq_tilde_x3.evaluations[eq_tilde_x3.evaluations.len() - 5..].to_vec());
        println!("state.index.col3: {:?}", col3[col3.len() - 5..].to_vec());
        println!("v_msg: {:?}", v_msg);
        println!("--------------------------------");
        let q_row: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                q_num_vars,
                crate::utils::hash_tuple::<E::ScalarField>(
                    &state.index.row[..state.index.real_len_val],
                    &state.eq_tilde_rx,
                    &v_msg,
                ),
            );
            println!("---------------q_col-----------------");

        let q_col = DenseMultilinearExtension::from_evaluations_vec(
            q_num_vars,
            crate::utils::hash_tuple::<E::ScalarField>(
                &state.index.col[..state.index.real_len_val],
                &state.eq_tilde_ry,
                &v_msg,
            ),
        );
        println!("---------------q_col-----------------");

        let domain = (0usize..1 << state.eq_tilde_rx.num_vars).collect::<Vec<_>>();
        let t_row = DenseMultilinearExtension::from_evaluations_vec(
            state.eq_tilde_rx.num_vars,
            crate::utils::hash_tuple::<E::ScalarField>(&domain, &state.eq_tilde_rx, &v_msg),
        );
        let t_col = DenseMultilinearExtension::from_evaluations_vec(
            state.index.num_variables_val,
            crate::utils::hash_tuple::<E::ScalarField>(&domain, &state.eq_tilde_ry, &v_msg),
        );

        let mut q_polys =
            ListOfProductsOfPolynomials::new(max(q_num_vars, state.index.num_variables_val));

        let prod = vec![
            Rc::new(state.eq_tilde_rx.clone()),
            Rc::new(state.eq_tilde_ry.clone()),
            Rc::new(state.val_M.clone()),
        ];
        q_polys.add_product(prod, E::ScalarField::one());

        let x_r: E::ScalarField = get_scalar_challenge(rng);
        let x_c: E::ScalarField = get_scalar_challenge(rng);
        println!("--------------------------------");
        println!("x_c: {:?}", x_c);
        println!("q_row: {:?}", q_row.evaluations[..5].to_vec());
        println!("t_row: {:?}", t_row.evaluations[..5].to_vec());
        println!("--------------------------------");

        let lookup_pf_row = LogLookupProof::prove(
            &q_row,
            &t_row,
            &state.index.freq_r,
            &state.index.ck_index,
            &x_r,
        );

        let lookup_pf_col = LogLookupProof::prove(
            &q_col,
            &t_col,
            &state.index.freq_c,
            &state.index.ck_index,
            &x_c,
        );

        _ = rng.feed(&lookup_pf_row.1[0]);
        _ = rng.feed(&lookup_pf_row.1[1]);
        _ = rng.feed(&lookup_pf_col.1[0]);
        _ = rng.feed(&lookup_pf_col.1[1]);

        let z = get_vector_challenge(rng, q_row.num_vars);
        let lambda = get_scalar_challenge(rng);

        let [q_row0, q_row1, q_row2, q_row3] = split_poly(&q_row, 2).try_into().unwrap();
        let [t_row0, t_row1, t_row2, t_row3] = split_poly(&t_row, 2).try_into().unwrap();
        let [q_col0, q_col1, q_col2, q_col3] = split_poly(&q_col, 2).try_into().unwrap();
        let [t_col0, t_col1, t_col2, t_col3] = split_poly(&t_col, 2).try_into().unwrap();
        let [freq_r0, freq_r1, freq_r2, freq_r3] = split_poly(&state.index.freq_r, 2).try_into().unwrap();
        let [pf_row00_0, pf_row00_1, pf_row00_2, pf_row00_3] = split_poly(&lookup_pf_row.0[0], 2).try_into().unwrap();
        let [pf_row01_0, pf_row01_1, pf_row01_2, pf_row01_3] = split_poly(&lookup_pf_row.0[1], 2).try_into().unwrap();
        let [pf_row20_0, pf_row20_1, pf_row20_2, pf_row20_3] = split_poly(&lookup_pf_row.2[0], 2).try_into().unwrap();
        let [pf_row21_0, pf_row21_1, pf_row21_2, pf_row21_3] = split_poly(&lookup_pf_row.2[1], 2).try_into().unwrap();

        let [pf_col00_0, pf_col00_1, pf_col00_2, pf_col00_3] = split_poly(&lookup_pf_col.0[0], 2).try_into().unwrap();
        let [pf_col01_0, pf_col01_1, pf_col01_2, pf_col01_3] = split_poly(&lookup_pf_col.0[1], 2).try_into().unwrap();
        let [pf_col20_0, pf_col20_1, pf_col20_2, pf_col20_3] = split_poly(&lookup_pf_col.2[0], 2).try_into().unwrap();
        let [pf_col21_0, pf_col21_1, pf_col21_2, pf_col21_3] = split_poly(&lookup_pf_col.2[1], 2).try_into().unwrap();
        
        // println!("--------------------------------");
        // println!("freq_r0: {:?}", freq_r0.evaluations[..5].to_vec());
        // println!("q_row0: {:?}", q_row0.evaluations[..5].to_vec());
        // println!("t_row0: {:?}", t_row0.evaluations[..5].to_vec());
        // println!("pf_row00_0: {:?}", pf_row00_0.evaluations[..5].to_vec());
        // println!("pf_row01_0: {:?}", pf_row01_0.evaluations[..5].to_vec());
        // println!("pf_row20_0: {:?}", pf_row20_0.evaluations[..5].to_vec());
        // println!("pf_row21_0: {:?}", pf_row21_0.evaluations[..5].to_vec());
        // println!("--------------------------------");
        // println!("freq_r1: {:?}", freq_r1.evaluations[..5].to_vec());
        // println!("q_row1: {:?}", q_row1.evaluations[..5].to_vec());
        // println!("t_row1: {:?}", t_row1.evaluations[..5].to_vec());
        // println!("pf_row00_1: {:?}", pf_row00_1.evaluations[..5].to_vec());
        // println!("pf_row01_1: {:?}", pf_row01_1.evaluations[..5].to_vec());
        // println!("pf_row20_1: {:?}", pf_row20_1.evaluations[..5].to_vec());
        // println!("pf_row21_1: {:?}", pf_row21_1.evaluations[..5].to_vec());
        // println!("--------------------------------");
        // println!("freq_r2: {:?}", freq_r2.evaluations[..5].to_vec());
        // println!("q_row2: {:?}", q_row2.evaluations[..5].to_vec());
        // println!("t_row2: {:?}", t_row2.evaluations[..5].to_vec());
        // println!("pf_row00_2: {:?}", pf_row00_2.evaluations[..5].to_vec());
        // println!("pf_row01_2: {:?}", pf_row01_2.evaluations[..5].to_vec());
        // println!("pf_row20_2: {:?}", pf_row20_2.evaluations[..5].to_vec());
        // println!("pf_row21_2: {:?}", pf_row21_2.evaluations[..5].to_vec());
        // println!("--------------------------------");
        // println!("freq_r3: {:?}", freq_r3.evaluations[..5].to_vec());
        // println!("q_row3: {:?}", q_row3.evaluations[..5].to_vec());
        // println!("t_row3: {:?}", t_row3.evaluations[..5].to_vec());
        // println!("pf_row00_3: {:?}", pf_row00_3.evaluations[..5].to_vec());
        // println!("pf_row01_3: {:?}", pf_row01_3.evaluations[..5].to_vec());
        // println!("pf_row20_3: {:?}", pf_row20_3.evaluations[..5].to_vec());
        // println!("pf_row21_3: {:?}", pf_row21_3.evaluations[..5].to_vec());
        // println!("--------------------------------");

        println!("--------------------------------");
        println!("q_col0: {:?}", q_col0.evaluations[q_col0.evaluations.len() - 5..].to_vec());
        println!("t_col0: {:?}", t_col0.evaluations[t_col0.evaluations.len() - 5..].to_vec());
        println!("pf_col00_0: {:?}", pf_col00_0.evaluations[pf_col00_0.evaluations.len() - 5..].to_vec());
        println!("pf_col01_0: {:?}", pf_col01_0.evaluations[pf_col01_0.evaluations.len() - 5..].to_vec());
        println!("pf_col20_0: {:?}", pf_col20_0.evaluations[pf_col20_0.evaluations.len() - 5..].to_vec());
        println!("pf_col21_0: {:?}", pf_col21_0.evaluations[pf_col21_0.evaluations.len() - 5..].to_vec());
        println!("--------------------------------");
        println!("q_col1: {:?}", q_col1.evaluations[q_col1.evaluations.len() - 5..].to_vec());
        println!("t_col1: {:?}", t_col1.evaluations[t_col1.evaluations.len() - 5..].to_vec());
        println!("pf_col00_1: {:?}", pf_col00_1.evaluations[pf_col00_1.evaluations.len() - 5..].to_vec());
        println!("pf_col01_1: {:?}", pf_col01_1.evaluations[pf_col01_1.evaluations.len() - 5..].to_vec());
        println!("pf_col20_1: {:?}", pf_col20_1.evaluations[pf_col20_1.evaluations.len() - 5..].to_vec());
        println!("pf_col21_1: {:?}", pf_col21_1.evaluations[pf_col21_1.evaluations.len() - 5..].to_vec());
        println!("--------------------------------");
        println!("q_col2: {:?}", q_col2.evaluations[q_col2.evaluations.len() - 5..].to_vec());
        println!("t_col2: {:?}", t_col2.evaluations[t_col2.evaluations.len() - 5..].to_vec());
        println!("pf_col00_2: {:?}", pf_col00_2.evaluations[pf_col00_2.evaluations.len() - 5..].to_vec());
        println!("pf_col01_2: {:?}", pf_col01_2.evaluations[pf_col01_2.evaluations.len() - 5..].to_vec());
        println!("pf_col20_2: {:?}", pf_col20_2.evaluations[pf_col20_2.evaluations.len() - 5..].to_vec());
        println!("pf_col21_2: {:?}", pf_col21_2.evaluations[pf_col21_2.evaluations.len() - 5..].to_vec());
        println!("--------------------------------");
        println!("q_col3: {:?}", q_col3.evaluations[q_col3.evaluations.len() - 5..].to_vec());
        println!("t_col3: {:?}", t_col3.evaluations[t_col3.evaluations.len() - 5..].to_vec());
        println!("pf_col00_3: {:?}", pf_col00_3.evaluations[pf_col00_3.evaluations.len() - 5..].to_vec());
        println!("pf_col01_3: {:?}", pf_col01_3.evaluations[pf_col01_3.evaluations.len() - 5..].to_vec());
        println!("pf_col20_3: {:?}", pf_col20_3.evaluations[pf_col20_3.evaluations.len() - 5..].to_vec());
        println!("pf_col21_3: {:?}", pf_col21_3.evaluations[pf_col21_3.evaluations.len() - 5..].to_vec());
        println!("--------------------------------");

        sumcheck_polynomial_list(
            (lookup_pf_row.0[0].clone(), lookup_pf_row.0[1].clone()),
            (lookup_pf_row.2[0].clone(), lookup_pf_row.2[1].clone()),
            boost_degree(&state.index.freq_r.clone(), q_row.num_vars),
            q_row.num_vars - t_row.num_vars,
            &mut q_polys,
            &z,
            &lambda,
            0,
            z.len(),
        );

        let z = get_vector_challenge(rng, q_col.num_vars);
        let lambda = get_scalar_challenge(rng);
        sumcheck_polynomial_list(
            (lookup_pf_col.0[0].clone(), lookup_pf_col.0[1].clone()),
            (lookup_pf_col.2[0].clone(), lookup_pf_col.2[1].clone()),
            boost_degree(&state.index.freq_c.clone(), q_col.num_vars),
            q_col.num_vars - t_col.num_vars,
            &mut q_polys,
            &z,
            &lambda,
            0,
            z.len(),
        );

        let (pf, final_state) = MLSumcheck::prove_as_subprotocol(rng, &q_polys).unwrap();
        let final_point = final_state.randomness;
        println!("final_point: {:?}", final_point[..5].to_vec());

        let eta: E::ScalarField = get_scalar_challenge(rng);
        let vk = state.vk.clone().unwrap();
        let batch_oracle = batch_open_poly(
            &[
                &lookup_pf_row.0[0],
                &lookup_pf_row.0[1],
                &lookup_pf_col.0[0],
                &lookup_pf_col.0[1],
                &state.eq_tilde_rx,
                &state.eq_tilde_ry,
                &state.index.val_a,
                &state.index.val_b,
                &state.index.val_c,
                &state.index.freq_r,
                &q_row,
                &t_row,
                &state.index.freq_c,
                &q_col,
                &t_col,
            ],
            &[
                lookup_pf_row.1[0].clone(),
                lookup_pf_row.1[1].clone(),
                lookup_pf_col.1[0].clone(),
                lookup_pf_col.1[1].clone(),
                state.eq_tilde_rx_comm.clone().unwrap(),
                state.eq_tilde_ry_comm.clone().unwrap(),
                vk.val_a_oracle.clone(),
                vk.val_b_oracle.clone(),
                vk.val_c_oracle.clone(),
            ],
            &state.index.ck_index,
            &final_point,
            eta,
        );

        assert!(q_row.num_vars - t_row.num_vars == q_col.num_vars - t_col.num_vars);
        ProverMessage {
            sumcheck_message: None,
            group_message: None,
            zk_proof_message: None,
            proof_message: None,
            commitment_message: None,
            commitment_message_2: None,
            lookup_message: Some(LogLookupProof {
                sumcheck_pfs: pf,
                info: q_polys.info(),
                point: final_point.clone(),
                degree_diff: q_row.num_vars - t_row.num_vars,
                batch_oracle,
            }),
            zksumcheck_message: None,
        }
    }
}
