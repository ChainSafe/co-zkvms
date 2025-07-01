use crate::logup::default_sumcheck_poly_list;
use crate::logup::poly_list_to_prover_state;
use crate::logup::LogLookupProof;
use crate::math::Math;
use crate::mpc::ass::AssShare;
use crate::mpc::rss::ProverFirstMsg;
use crate::mpc::rss::ProverSecondMsg;
use crate::mpc::rss::Rep3SumcheckProverMsg;
use crate::mpc::rss::RssShare;
use crate::mpc::utils::ShareType;
use crate::mpi_snark::coordinator::PartialProof;
use crate::mpi_utils::combine_partial_proof;
use crate::mpi_utils::gather_responses;
use crate::mpi_utils::merge_list_of_distributed_poly;
use crate::mpi_utils::scatter_requests;
use crate::mpi_utils::DistrbutedSumcheckProverState;
use crate::network::NetworkCoordinator;
use crate::snark::indexer::IndexProverKey;
use crate::snark::indexer::IndexVerifierKey;
use crate::snark::prover::MaskPolynomial;
use crate::snark::prover::ProverMessage;
use crate::snark::verifier::DFSVerifier;
use crate::snark::verifier::VerifierMessage;
use crate::snark::verifier::VerifierState;
use crate::snark::zk::generate_mask_polynomial;
use crate::snark::zk::zk_sumcheck_verifier_wrapper;
use crate::snark::zk::ZKMLCommit;
use crate::snark::zk::ZKMLCommitment;
use crate::snark::zk::ZKMLCommitterKey;
use crate::snark::zk::ZKMLProof;
use crate::snark::zk::ZKSumcheckProof;
use crate::snark::BatchOracleEval;
use crate::snark::R1CSProof;
use crate::transcript::Transcript;
use crate::utils::aggregate_proof;
use crate::utils::combine_comm;
use crate::utils::feed_message;
use crate::utils::generate_dumb_sponge;
use crate::utils::merge_proof;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_ff::One;
use ark_ff::UniformRand;
use ark_ff::Zero;
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::data_structures::PolynomialInfo;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::MaskProverState;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::IPForMLSumcheck;
use ark_linear_sumcheck::rng::Blake2s512Rng;
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_poly::multivariate::SparsePolynomial;
use ark_poly::multivariate::SparseTerm;
use ark_poly::DenseMultilinearExtension;
use ark_poly::MultilinearExtension;
use ark_poly::Polynomial;
use ark_poly_commit::marlin_pst13_pc::CommitterKey as MaskCommitterKey;
use ark_poly_commit::marlin_pst13_pc::MarlinPST13;
use ark_poly_commit::multilinear_pc::data_structures::Commitment;
use ark_poly_commit::multilinear_pc::data_structures::CommitterKey;
use ark_poly_commit::multilinear_pc::data_structures::Proof;
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_poly_commit::LabeledPolynomial;
use ark_poly_commit::PolynomialCommitment;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use ark_std::marker::PhantomData;
use ark_std::ops::Neg;
use ark_std::rc::Rc;
use ark_std::time::Duration;
use ark_std::time::Instant;
use ark_std::{end_timer, start_timer};
use rand::Rng;
use rand::RngCore;

pub struct SpartanProverCoordinator<E: Pairing, N: NetworkCoordinator> {
    _network: PhantomData<N>,
    _pairing: PhantomData<E>,
}

#[derive(Clone)]
struct ProverState<E: Pairing> {
    pub val_a: E::ScalarField,
    pub val_b: E::ScalarField,
    pub val_c: E::ScalarField,
    pub val_w: E::ScalarField,
    pub val_m: E::ScalarField,
    pub first_sumcheck_msgs: Option<ZKSumcheckProof<E>>,
    pub second_sumcheck_msgs: Option<ZKSumcheckProof<E>>,
    pub zk_open_pf: Option<ZKMLProof<E>>,
    pub witness_mask:
        LabeledPolynomial<E::ScalarField, SparsePolynomial<E::ScalarField, SparseTerm>>,
    pub witness_comm: Option<Commitment<E>>,
    pub r_x: Vec<E::ScalarField>,
    pub r_y: Vec<E::ScalarField>,
    pub eq_tilde_rx_comm: Option<Commitment<E>>,
    pub eq_tilde_ry_comm: Option<Commitment<E>>,
    pub time_elapsed: Duration,
}

impl<E: Pairing> Default for ProverState<E> {
    fn default() -> Self {
        Self {
            witness_mask: LabeledPolynomial::<E::ScalarField, MaskPolynomial<E>>::new(
                "init_mask".into(),
                MaskPolynomial::<E>::zero(),
                None,
                None,
            ),
            witness_comm: None,
            r_x: Vec::new(),
            r_y: Vec::new(),
            eq_tilde_rx_comm: None,
            eq_tilde_ry_comm: None,
            val_a: E::ScalarField::zero(),
            val_b: E::ScalarField::zero(),
            val_c: E::ScalarField::zero(),
            first_sumcheck_msgs: None,
            second_sumcheck_msgs: None,
            val_m: E::ScalarField::zero(),
            val_w: E::ScalarField::zero(),
            zk_open_pf: None,
            time_elapsed: Duration::from_secs(0),
        }
    }
}

impl<E: Pairing, N: NetworkCoordinator> SpartanProverCoordinator<E, N> {
    pub fn prove(
        index: &IndexProverKey<E>,
        pub_index: &IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
        transcript: &mut impl Transcript,
        network: &mut N,
    ) -> (R1CSProof<E>, Duration)
    where
        E: Pairing,
    {
        let mut state = ProverState::default();
        let mask_rng = &mut Blake2s512Rng::setup();
        transcript.append_serializable(b"initialize", &());
        assert!(mask_rng.feed(&"initialize".as_bytes()).is_ok());
        let mut challenge_gen = generate_dumb_sponge::<E::ScalarField>();

        //todo: padding the R1CS instance, witness and io

        let time = Instant::now();
        let mut verifier_state: VerifierState<E> = DFSVerifier::verifier_init(index.padded_num_var);
        state.time_elapsed += time.elapsed();

        // let mut prover_first_message: ProverMessage<E>;
        Self::first_round(&mut state, &index, 2, None, mask_rng, network, transcript);

        // This first challenge is used for checking the hadamard product of AB - C ?= 0.
        // The following sumcheck, in prover_second_round, doesn't verify the well formedness of its components, which future rounds will do.

        let verifier_first_message =
            DFSVerifier::verifier_first_round(&mut verifier_state, transcript);

        Self::second_round(
            &index,
            &mut state,
            &verifier_first_message.verifier_message,
            mask_rng,
            &mut challenge_gen,
            network,
            transcript,
        );

        // This challenge is used for verifying the well formedness of the subclaim evaluation from the previous round.
        // In particular, this challenge batches a, b, and c polynomials using 3 scalars
        let verifier_second_message =
            DFSVerifier::verifier_second_round(&mut verifier_state, transcript);

        Self::third_round(
            &index,
            &pub_index,
            &mut state,
            &verifier_second_message.verifier_message,
            mask_rng,
            &mut challenge_gen,
            network,
            transcript,
        );

        let verifier_fifth_message =
            DFSVerifier::verifier_fifth_round(&mut verifier_state, transcript);

        let holo_time = Instant::now();

        let lookup_proof = Self::fifth_round(
            pub_index,
            vk,
            &mut state,
            verifier_fifth_message.verifier_message[0],
            network,
            transcript,
        );
        transcript.append_serializable(b"lookup_proof", &lookup_proof);

        println!("holography time: {:?}", holo_time.elapsed());

        (
            R1CSProof {
                witness_commitment: (state.witness_comm.unwrap()),
                first_sumcheck_msgs: (state.first_sumcheck_msgs.unwrap()),
                va: state.val_a,
                vb: state.val_b,
                vc: state.val_c,
                second_sumcheck_msgs: state.second_sumcheck_msgs.unwrap(),
                witness_eval: state.val_w,
                witness_proof: state.zk_open_pf.unwrap(),
                val_M: state.val_m,
                eq_tilde_rx_commitment: state.eq_tilde_rx_comm.unwrap(),
                eq_tilde_ry_commitment: state.eq_tilde_ry_comm.unwrap(),
                lookup_proof: lookup_proof,
            },
            state.time_elapsed,
        )
    }

    // hiding_poly_commit
    fn first_round<'a>(
        state: &mut ProverState<E>,
        index: &IndexProverKey<E>,
        hiding_bound: usize,
        mask_num_var: Option<usize>,
        rng: &mut impl Rng,
        network: &mut N,
        transcript: &mut impl Transcript,
    ) {
        let (base_commitment_vec, time): (Vec<Commitment<E>>, Duration) =
            rep3_poly_commit_coordinator(1, index.ck_w.0.g, network);

        state.time_elapsed += time;

        let time = Instant::now();

        let mut p_hat = SparsePolynomial::<E::ScalarField, SparseTerm>::zero();
        if let Some(mask_num_vars) = mask_num_var {
            p_hat = generate_mask_polynomial(rng, mask_num_vars, hiding_bound, false);
        } else {
            p_hat = generate_mask_polynomial(rng, index.padded_num_var, hiding_bound, false);
        }
        let labeled_p_hat =
            LabeledPolynomial::new("p_hat".to_owned(), p_hat, Some(hiding_bound), None);
        let hiding_commitment: E::G1Affine = ZKMLCommit::<
            E,
            SparsePolynomial<E::ScalarField, SparseTerm>,
        >::commit_mask(
            &index.ck_w.1, &labeled_p_hat, rng
        );

        let hidden_commitment: E::G1Affine =
            (base_commitment_vec[0].g_product + hiding_commitment).into();
        let commitment = Commitment {
            g_product: hidden_commitment,
            nv: index.padded_num_var,
        };

        transcript.append_serializable(b"w_commitment", &commitment);
        state.witness_comm = Some(commitment.clone());
        state.witness_mask = labeled_p_hat.clone();
        state.time_elapsed += time.elapsed();
    }

    fn second_round<'a, R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>>(
        index: &IndexProverKey<E>,
        state: &mut ProverState<E>,
        v_msg: &Vec<E::ScalarField>,
        mask_rng: &mut R,
        mask_challenge_generator_for_open: &mut impl CryptographicSponge,
        network: &mut N,
        transcript: &mut impl Transcript,
    ) {
        network.broadcast_requests(v_msg.clone());

        let num_variables = index.padded_num_var;
        let poly_info = PolynomialInfo {
            max_multiplicands: 3,
            num_variables: num_variables,
        };

        let sumcheck_polys_builder = |responses_chunked: &[(
            E::ScalarField,
            E::ScalarField,
            E::ScalarField,
            E::ScalarField,
        )],
                                      log_num_workers_per_party: usize|
         -> ListOfProductsOfPolynomials<E::ScalarField> {
            let mut merge_poly = ListOfProductsOfPolynomials::new(log_num_workers_per_party);
            let mut za = Vec::new();
            let mut zb = Vec::new();
            let mut zc = Vec::new();
            let mut eq = Vec::new();

            for i in 0..1 << log_num_workers_per_party {
                za.push(
                    responses_chunked[3 * i].0
                        + responses_chunked[3 * i + 1].0
                        + responses_chunked[3 * i + 2].0,
                );
                zb.push(
                    responses_chunked[3 * i].1
                        + responses_chunked[3 * i + 1].1
                        + responses_chunked[3 * i + 2].1,
                );
                zc.push(
                    responses_chunked[3 * i].2
                        + responses_chunked[3 * i + 1].2
                        + responses_chunked[3 * i + 2].2,
                );
                eq.push(responses_chunked[3 * i].3);
                assert!(responses_chunked[3 * i].3 == responses_chunked[3 * i + 1].3);
                assert!(responses_chunked[3 * i].3 == responses_chunked[3 * i + 2].3);
            }

            let eq_func = DenseMultilinearExtension::from_evaluations_vec(
                log_num_workers_per_party,
                eq.clone(),
            );
            let A_B_hat = vec![
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    log_num_workers_per_party,
                    za.clone(),
                )),
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    log_num_workers_per_party,
                    zb.clone(),
                )),
                Rc::new(eq_func.clone()),
            ];
            let C_hat = vec![
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    log_num_workers_per_party,
                    zc.clone(),
                )),
                Rc::new(eq_func.clone()),
            ];

            merge_poly.add_product(A_B_hat, E::ScalarField::one());
            merge_poly.add_product(C_hat, E::ScalarField::one().neg());

            merge_poly
        };

        let (pf, final_point, time1) = rep3_zk_sumcheck_coordinator::<ProverFirstMsg<E>, E, N, R, _>(
            poly_info,
            mask_rng,
            &index.ck_mask,
            mask_challenge_generator_for_open,
            network,
            transcript,
            sumcheck_polys_builder,
        );

        let (evals, time2) =
            rep3_eval_poly_coordinator::<E, _>(index.padded_num_var, 3, &final_point, network);

        let (val_a, val_b, val_c) = (evals[0].clone(), evals[1].clone(), evals[2].clone());

        state.r_x = final_point.to_vec();

        transcript.append_serializable(b"group_message", &[val_a, val_b, val_c]);
        transcript.append_serializable(b"first_sumcheck_msgs", &pf);

        state.val_a = val_a;
        state.val_b = val_b;
        state.val_c = val_c;
        state.first_sumcheck_msgs = Some(pf);
        state.time_elapsed += time1 + time2;
    }

    fn third_round<'a, R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>>(
        index: &IndexProverKey<E>,
        pub_index: &IndexProverKey<E>,
        state: &mut ProverState<E>,
        v_msg: &Vec<E::ScalarField>,
        mask_rng: &mut R,
        mask_challenge_generator_for_open: &mut impl CryptographicSponge,
        network: &mut N,
        transcript: &mut impl Transcript,
    ) {
        network.broadcast_requests(v_msg.clone());

        let num_variables = index.padded_num_var;
        let poly_info = PolynomialInfo {
            max_multiplicands: 2,
            num_variables: num_variables,
        };

        let sumcheck_polys_builder = |responses_chunked: &[(
            E::ScalarField,
            E::ScalarField,
            E::ScalarField,
            E::ScalarField,
        )],
                                      log_num_workers_per_party: usize|
         -> ListOfProductsOfPolynomials<E::ScalarField> {
            let mut merge_poly = ListOfProductsOfPolynomials::new(log_num_workers_per_party);
            let mut A_rx = Vec::new();
            let mut B_rx = Vec::new();
            let mut C_rx = Vec::new();
            let mut z = Vec::new();

            for i in 0..1 << log_num_workers_per_party {
                A_rx.push(responses_chunked[3 * i].0);
                B_rx.push(responses_chunked[3 * i].1);
                C_rx.push(responses_chunked[3 * i].2);
                z.push(
                    responses_chunked[3 * i].3
                        + responses_chunked[3 * i + 1].3
                        + responses_chunked[3 * i + 2].3,
                );
            }

            let z = DenseMultilinearExtension::from_evaluations_vec(log_num_workers_per_party, z);
            let A_hat = vec![
                Rc::new(DenseMultilinearExtension {
                    evaluations: (A_rx),
                    num_vars: (log_num_workers_per_party),
                }),
                Rc::new(z.clone()),
            ];
            let B_hat = vec![
                Rc::new(DenseMultilinearExtension {
                    evaluations: (B_rx),
                    num_vars: (log_num_workers_per_party),
                }),
                Rc::new(z.clone()),
            ];
            let C_hat = vec![
                Rc::new(DenseMultilinearExtension {
                    evaluations: (C_rx),
                    num_vars: (log_num_workers_per_party),
                }),
                Rc::new(z.clone()),
            ];

            merge_poly.add_product(A_hat, v_msg[0]);
            merge_poly.add_product(B_hat, v_msg[1]);
            merge_poly.add_product(C_hat, v_msg[2]);

            merge_poly
        };

        let (pf, final_point, time) = rep3_zk_sumcheck_coordinator::<ProverSecondMsg<E>, E, N, R, _>(
            poly_info,
            mask_rng,
            &index.ck_mask,
            mask_challenge_generator_for_open,
            network,
            transcript,
            sumcheck_polys_builder,
        );
        state.second_sumcheck_msgs = Some(pf);

        state.r_y = final_point.to_vec();
        state.time_elapsed += time;

        let (val_ws, time) =
            rep3_eval_poly_coordinator::<E, N>(index.padded_num_var, 1, &final_point, network);
        state.val_w = val_ws[0];
        state.time_elapsed += time;
        transcript.append_serializable(b"val_w", &state.val_w);

        let responses_chunked: Vec<(E::ScalarField, E::ScalarField, E::ScalarField)> =
            network.receive_responses(Default::default());

        let mut val_a = E::ScalarField::zero();
        let mut val_b = E::ScalarField::zero();
        let mut val_c = E::ScalarField::zero();
        for i in 0..1 << network.log_num_pub_workers() {
            val_a = val_a + responses_chunked[i].0;
            val_b = val_b + responses_chunked[i].1;
            val_c = val_c + responses_chunked[i].2;
        }

        state.val_m = val_a * v_msg[0] + val_b * v_msg[1] + val_c * v_msg[2];
        transcript.append_serializable(b"val_m", &state.val_m);

        let (comms, time) = rep3_poly_commit_coordinator(2, pub_index.ck_index.g, network);
        let [eq_tilde_rx_comm, eq_tilde_ry_comm] = comms.try_into().unwrap();
        transcript.append_serializable(b"eq_tilde_rx_comm", &eq_tilde_rx_comm);
        transcript.append_serializable(b"eq_tilde_ry_comm", &eq_tilde_ry_comm);
        state.eq_tilde_rx_comm = Some(eq_tilde_rx_comm);
        state.eq_tilde_ry_comm = Some(eq_tilde_ry_comm);
        state.time_elapsed += time;

        let (zk_open_pf, time) = rep3_zk_open_poly_coordinator(
            index.padded_num_var,
            &state.witness_comm.clone().unwrap(),
            &state.r_y[..],
            &index.ck_w,
            &state.witness_mask,
            network,
        );
        transcript.append_serializable(b"zk_proof_message", &zk_open_pf);
        state.zk_open_pf = Some(zk_open_pf);
        state.time_elapsed += time;
    }

    fn fifth_round(
        pub_index: &IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
        state: &mut ProverState<E>,
        v_msg: E::ScalarField,
        network: &mut N,
        transcript: &mut impl Transcript,
    ) -> LogLookupProof<E> {
        network.broadcast_requests(v_msg.clone());

        let time = Instant::now();

        let q_num_vars = pub_index.real_len_val.log_2();

        let mut q_polys = ListOfProductsOfPolynomials::new(1);
        let default_poly = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![E::ScalarField::zero(), E::ScalarField::zero()],
        );

        let prod = vec![
            Rc::new(default_poly.clone()),
            Rc::new(default_poly.clone()),
            Rc::new(default_poly.clone()),
        ];
        q_polys.add_product(prod, E::ScalarField::one());

        let x_r: E::ScalarField = transcript.get_scalar_challenge(b"x_r");
        let x_c: E::ScalarField = transcript.get_scalar_challenge(b"x_c");

        state.time_elapsed += time.elapsed();

        network.broadcast_requests((x_r, x_c));

        let (comms, time) = rep3_poly_commit_coordinator(4, pub_index.ck_index.g, network);

        state.time_elapsed += time;

        transcript.append_serializable(b"comm0", &comms[0]);
        transcript.append_serializable(b"comm1", &comms[1]);
        transcript.append_serializable(b"comm2", &comms[2]);
        transcript.append_serializable(b"comm3", &comms[3]);

        let z: Vec<E::ScalarField> = transcript.get_vector_challenge(b"z", q_num_vars);
        let lambda: E::ScalarField = transcript.get_scalar_challenge(b"lambda");

        network.broadcast_requests((z.clone(), lambda.clone()));

        default_sumcheck_poly_list(
            &lambda,
            q_num_vars - pub_index.num_variables_val,
            &mut q_polys,
        );

        let z: Vec<E::ScalarField> = transcript.get_vector_challenge(b"z", q_num_vars);
        let lambda: E::ScalarField = transcript.get_scalar_challenge(b"lambda");

        network.broadcast_requests((z.clone(), lambda.clone()));

        default_sumcheck_poly_list(
            &lambda,
            q_num_vars - pub_index.num_variables_val,
            &mut q_polys,
        );

        let poly_info = PolynomialInfo {
            max_multiplicands: 3,
            num_variables: q_num_vars,
        };

        let (prover_msgs, final_point, time) =
            distributed_sumcheck_coordinator(&poly_info, &q_polys, network, transcript);

        state.time_elapsed += time;

        let eta: E::ScalarField = transcript.get_scalar_challenge(b"eta");
        network.broadcast_requests(eta.clone());

        let (batch_oracle, time) = batch_open_poly_coordinator(
            poly_info.num_variables,
            15,
            &pub_index.ck_index,
            &[
                comms[0].clone(),
                comms[1].clone(),
                comms[2].clone(),
                comms[3].clone(),
                state.eq_tilde_rx_comm.clone().unwrap(),
                state.eq_tilde_ry_comm.clone().unwrap(),
                vk.val_a_oracle.clone(),
                vk.val_b_oracle.clone(),
                vk.val_c_oracle.clone(),
            ],
            &final_point,
            pub_index.ck_index.g,
            network,
            false,
        );

        state.time_elapsed += time;

        LogLookupProof {
            sumcheck_pfs: prover_msgs,
            info: poly_info,
            point: final_point.clone(),
            degree_diff: q_num_vars - pub_index.num_variables_val,
            batch_oracle,
        }
    }
}

pub fn rep3_zk_sumcheck_coordinator<
    M: Rep3SumcheckProverMsg<E>,
    E: Pairing,
    N: NetworkCoordinator,
    R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
    T: CanonicalSerialize + CanonicalDeserialize + Clone + Default,
>(
    poly_info: PolynomialInfo,
    mask_rng: &mut R,
    mask_key: &MaskCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    opening_challenge: &mut impl CryptographicSponge,
    network: &mut N,
    transcript: &mut impl Transcript,
    sumcheck_polys_builder: impl Fn(&[T], usize) -> ListOfProductsOfPolynomials<E::ScalarField>,
) -> (ZKSumcheckProof<E>, Vec<E::ScalarField>, Duration) {
    let time = Instant::now();

    let mask_poly = generate_mask_polynomial(
        mask_rng,
        poly_info.num_variables,
        poly_info.max_multiplicands,
        true,
    );
    let vec_mask_poly = vec![LabeledPolynomial::new(
        String::from("mask_poly_for_sumcheck"),
        mask_poly.clone(),
        Some(poly_info.max_multiplicands),
        None,
    )];
    let (mask_commit, mask_randomness) =
        MarlinPST13::<_, _>::commit(mask_key, &vec_mask_poly, Some(mask_rng)).unwrap();
    let g_commit = mask_commit[0].commitment();
    let _ = transcript.append_serializable(b"g_commit", g_commit);
    let challenge = transcript.get_scalar_challenge(b"r1");

    _ = transcript.append_serializable(b"poly_info", &poly_info.clone());
    let mut prover_zk_state = IPForMLSumcheck::mask_init(
        &mask_poly,
        poly_info.num_variables,
        poly_info.max_multiplicands,
        challenge,
    );

    let mut prover_msgs = Vec::new();
    let mut final_point = Vec::new();
    let mut v_msg = None;

    let mut tot_time = time.elapsed();
    // assert!(1 << log_num_workers == size);

    println!("coordinator sumcheck start");
    for _round in 0..poly_info.num_variables - network.log_num_workers_per_party() {
        let responses_chunked: Vec<_> = network.receive_responses(M::default());
        println!("coordinator sumcheck received responses for round {}", _round);
        let time = Instant::now();

        let mut prover_message = M::open_to_msg(&vec![
            responses_chunked[0].clone(),
            responses_chunked[1].clone(),
            responses_chunked[2].clone(),
        ]);
        for i in 1..1 << network.log_num_workers_per_party() {
            let tmp = M::open_to_msg(&vec![
                responses_chunked[3 * i + 0].clone(),
                responses_chunked[3 * i + 1].clone(),
                responses_chunked[3 * i + 2].clone(),
            ]);
            // Aggregate results from different parties
            for j in 0..prover_message.evaluations.len() {
                prover_message.evaluations[j] = prover_message.evaluations[j] + tmp.evaluations[j]
            }
        }
        let mask = IPForMLSumcheck::mask_round(&mut prover_zk_state, &v_msg);

        let final_msg = ProverMsg {
            evaluations: prover_message
                .evaluations
                .iter()
                .zip(mask.evaluations.iter())
                .map(|(msg, sum)| *msg + sum)
                .collect(),
        };

        let _ = transcript.append_serializable(b"final_msg", &final_msg);
        prover_msgs.push(final_msg.clone());
        let verifier_msg = Some(IPForMLSumcheck::sample_round(transcript));
        // Using the aggregate results to generate the verifier's message.
        let r: E::ScalarField = verifier_msg.clone().unwrap().randomness;
        final_point.push(r);

        tot_time += time.elapsed();

        println!(
            "coordinator sumcheck broadcast requests for round {}",
            _round
        );
        network.broadcast_requests(r);

        v_msg = verifier_msg.clone();
    }

    let responses = network.receive_responses(Default::default());

    let sumcheck_polys = sumcheck_polys_builder(&responses, network.log_num_workers_per_party());

    let time = Instant::now();

    let mut prover_state = IPForMLSumcheck::prover_init(&sumcheck_polys);
    let mut verifier_msg = None;
    for _ in poly_info.num_variables - network.log_num_workers_per_party()..poly_info.num_variables
    {
        let prover_message = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
        let mask = IPForMLSumcheck::mask_round(&mut prover_zk_state, &v_msg);

        let final_msg = ProverMsg {
            evaluations: prover_message
                .evaluations
                .iter()
                .zip(mask.evaluations.iter())
                .map(|(msg, sum)| *msg + sum)
                .collect(),
        };

        _ = transcript.append_serializable(b"final_msg", &final_msg);
        prover_msgs.push(final_msg.clone());
        verifier_msg = Some(IPForMLSumcheck::sample_round(transcript));
        final_point.push(verifier_msg.clone().unwrap().randomness);
        v_msg = verifier_msg.clone();
    }

    tot_time += time.elapsed();

    network.broadcast_requests(final_point.clone());

    let time = Instant::now();

    let opening = MarlinPST13::<_, SparsePolynomial<E::ScalarField, SparseTerm>>::open(
        &mask_key,
        &vec_mask_poly,
        &mask_commit,
        &final_point,
        opening_challenge,
        &mask_randomness,
        None,
    );

    tot_time += time.elapsed();

    (
        ZKSumcheckProof {
            g_commit: *g_commit,
            sumcheck_proof: prover_msgs,
            poly_info: poly_info.clone(),
            g_proof: opening.unwrap(),
            g_value: mask_poly.evaluate(&final_point),
        },
        final_point,
        tot_time,
    )
}

pub fn distributed_sumcheck_coordinator<F: Field, N: NetworkCoordinator>(
    poly_info: &PolynomialInfo,
    q_polys: &ListOfProductsOfPolynomials<F>,
    network: &mut N,
    transcript: &mut impl Transcript,
) -> (Vec<ProverMsg<F>>, Vec<F>, Duration) {
    let time = Instant::now();
    let log_num_workers = network.log_num_pub_workers();

    let mut prover_msgs = Vec::new();
    let mut final_point = Vec::new();

    _ = transcript.append_serializable(b"poly_info", &poly_info.clone());
    // assert!(1 << log_num_workers == size);

    let mut tot_time = time.elapsed();

    for _round in 0..poly_info.num_variables - log_num_workers {
        let default_response = ProverMsg {
            evaluations: vec![F::zero(); poly_info.max_multiplicands + 1],
        };

        let responses_chunked: Vec<_> = network.receive_responses(default_response);

        let time = Instant::now();

        let mut prover_message = responses_chunked[0].clone();
        for i in 1..1 << log_num_workers {
            let tmp = responses_chunked[i].clone();
            // Aggregate results from different parties
            for j in 0..prover_message.evaluations.len() {
                prover_message.evaluations[j] = prover_message.evaluations[j] + tmp.evaluations[j]
            }
        }
        let _ = transcript.append_serializable(b"prover_message", &prover_message);
        prover_msgs.push(prover_message.clone());
        let verifier_msg = Some(IPForMLSumcheck::sample_round(transcript));
        // Using the aggregate results to generate the verifier's message.
        let r = verifier_msg.clone().unwrap().randomness;
        final_point.push(r);

        tot_time += time.elapsed();

        network.broadcast_requests(r);
    }

    if log_num_workers != 0 {
        let default_response = poly_list_to_prover_state(q_polys);

        let responses_chunked: Vec<_> = network.receive_responses(default_response);

        let time = Instant::now();

        let merge_poly =
            merge_list_of_distributed_poly(&responses_chunked, poly_info, log_num_workers);

        let mut prover_state = IPForMLSumcheck::prover_init(&merge_poly);
        let mut verifier_msg = None;
        for _ in poly_info.num_variables - log_num_workers..poly_info.num_variables {
            let prover_message = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
            _ = transcript.append_serializable(b"prover_message", &prover_message);
            prover_msgs.push(prover_message.clone());
            verifier_msg = Some(IPForMLSumcheck::sample_round(transcript));
            final_point.push(verifier_msg.clone().unwrap().randomness);
        }

        tot_time += time.elapsed();
    }

    network.broadcast_requests(final_point.clone());

    (prover_msgs, final_point, tot_time)
}

fn rep3_poly_commit_coordinator<E: Pairing, N: NetworkCoordinator>(
    num_comms: usize,
    g: E::G1Affine,
    network: &mut N,
) -> (Vec<Commitment<E>>, Duration) {
    let default_response = vec![
        Commitment::<E> {
            nv: 0,
            g_product: g,
        };
        num_comms
    ];

    let responses_chunked = network.receive_responses(default_response);
    let time = Instant::now();
    // Round 1 process
    let mut res = Vec::new();
    for j in 0..num_comms {
        let mut comm = Vec::new();
        for i in 0..responses_chunked.len() {
            comm.push(responses_chunked[i][j].clone())
        }
        // TODO: do we need to do this??
        // let mut tmp = ;
        // tmp.nv = tmp.nv / 3;
        res.push(combine_comm(&comm))
    }

    (res, time.elapsed())
}

pub fn rep3_eval_poly_coordinator<E: Pairing, N: NetworkCoordinator>(
    num_var: usize,
    num_poly: usize,
    final_point: &[E::ScalarField],
    network: &mut N,
) -> (Vec<E::ScalarField>, Duration) {
    let default_response = vec![E::ScalarField::one(); num_poly];
    let responses_chunked: Vec<_> = network.receive_responses(default_response);

    let time = Instant::now();

    let mut evals = Vec::new();
    for i in 0..num_poly {
        let mut e = Vec::new();
        for j in 0..1 << network.log_num_workers_per_party() {
            e.push(
                responses_chunked[3 * j + 0][i]
                    + responses_chunked[3 * j + 1][i]
                    + responses_chunked[3 * j + 2][i],
            );
        }
        let ep =
            DenseMultilinearExtension::from_evaluations_vec(network.log_num_workers_per_party(), e);

        evals.push(ep.evaluate(
            &final_point[num_var - network.log_num_workers_per_party()..num_var].to_vec(),
        ));
    }

    (evals, time.elapsed())
}

pub fn batch_open_poly_coordinator<'a, E: Pairing, N: NetworkCoordinator>(
    num_var: usize,
    num_poly: usize,
    merge_ck: &CommitterKey<E>,
    comms: &[Commitment<E>],
    final_point: &[E::ScalarField],
    g: E::G1Affine,
    network: &mut N,
    rep3: bool,
) -> (BatchOracleEval<E>, Duration) {
    let log_num_workers = if rep3 {
        network.log_num_workers_per_party()
    } else {
        network.log_num_pub_workers()
    };
    let default_response = PartialProof {
        proofs: Proof {
            proofs: vec![g; num_var],
        },
        val: E::ScalarField::zero(),
        evals: vec![E::ScalarField::one(); num_poly],
    };
    let responses_chunked: Vec<_> = network.receive_responses(default_response);

    let time = Instant::now();

    let mut pfs = Vec::new();
    let mut rs = Vec::new();
    let mut es = Vec::new();
    for i in 0..1 << log_num_workers {
        let partial = if rep3 {
            combine_partial_proof(&[
                responses_chunked[3 * i].clone(),
                responses_chunked[3 * i + 1].clone(),
                responses_chunked[3 * i + 2].clone(),
            ])
        } else {
            responses_chunked[i].clone()
        };
        pfs.push(partial.proofs.clone());
        rs.push(partial.val);
        es.push(partial.evals);
    }

    let pf1 = aggregate_proof(E::ScalarField::one(), &pfs);
    let rp = DenseMultilinearExtension::from_evaluations_vec(log_num_workers, rs);

    let pf2 = MultilinearPC::<E>::open(
        merge_ck,
        &rp,
        &final_point[num_var - log_num_workers..num_var],
    );

    let batch_proof = merge_proof(&pf1, &pf2);

    let mut evals = Vec::new();
    let mut debug_evals = Vec::new();
    for i in 0..num_poly {
        let mut e = Vec::new();
        for j in 0..1 << log_num_workers {
            e.push(es[j][i]);
        }
        let ep = DenseMultilinearExtension::from_evaluations_vec(log_num_workers, e);
        if i < comms.len() {
            evals.push(ep.evaluate(&final_point[num_var - log_num_workers..num_var].to_vec()));
        } else {
            debug_evals
                .push(ep.evaluate(&final_point[num_var - log_num_workers..num_var].to_vec()));
        }
    }

    let batch_oracle = BatchOracleEval {
        val: evals,
        debug_val: debug_evals,
        commitment: comms.to_vec(),
        proof: batch_proof,
    };
    (batch_oracle, time.elapsed())
}

pub fn rep3_zk_open_poly_coordinator<'a, E: Pairing, N: NetworkCoordinator>(
    num_var: usize,
    comm: &Commitment<E>,
    final_point: &[E::ScalarField],
    ck: &ZKMLCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    p_hat: &LabeledPolynomial<E::ScalarField, SparsePolynomial<E::ScalarField, SparseTerm>>,
    network: &mut N,
) -> (ZKMLProof<E>, Duration) {
    let (batch_oracle, tot_time) = batch_open_poly_coordinator(
        num_var,
        1,
        &ck.0,
        &[comm.clone()],
        final_point,
        ck.0.g,
        network,
        true,
    );

    let time = Instant::now();

    let base_proof = batch_oracle.proof;
    let point = final_point.to_vec(); // todo add lifetime restriction
    let (hiding_proof, evaluation) =
        ZKMLCommit::<E, SparsePolynomial<E::ScalarField, SparseTerm>>::open_mask(
            &ck.1, p_hat, &point,
        );
    let hidden_proof_evals = base_proof
        .proofs
        .iter()
        .zip(hiding_proof.w.iter())
        .map(|(base_eval, hiding_eval)| (*base_eval + hiding_eval).into())
        .collect::<Vec<E::G1Affine>>();

    (
        (
            Proof {
                proofs: hidden_proof_evals,
            },
            evaluation,
        ),
        tot_time + time.elapsed(),
    )
}
