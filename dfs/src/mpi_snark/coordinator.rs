use crate::end_timer_buf;
use crate::math::Math;
use crate::mpi_utils::merge_list_of_distributed_poly;
use crate::mpi_utils::obtain_distrbuted_sumcheck_prover_state;
use crate::mpi_utils::{gather_responses, scatter_requests, DistrbutedSumcheckProverState};
use crate::snark::indexer::IndexProverKey;
use crate::snark::indexer::IndexVerifierKey;
use crate::snark::prover::MaskPolynomial;
use crate::snark::prover::ProverMessage;
use crate::snark::verifier::DFSVerifier;
use crate::snark::verifier::VerifierMessage;
use crate::snark::verifier::VerifierState;
use crate::snark::zk::generate_mask_polynomial;
use crate::snark::zk::ZKMLCommit;
use crate::snark::zk::ZKMLCommitment;
use crate::snark::zk::ZKMLCommitterKey;
use crate::snark::zk::ZKMLProof;
use crate::snark::zk::ZKSumcheckProof;
use crate::snark::BatchOracleEval;
use crate::snark::R1CSProof;
use crate::start_timer_buf;
use crate::subprotocols::loglookup::default_sumcheck_poly_list;
use crate::subprotocols::loglookup::poly_list_to_prover_state;
use crate::subprotocols::loglookup::LogLookupProof;
use crate::transcript::get_scalar_challenge;
use crate::transcript::get_vector_challenge;
use crate::utils::aggregate_poly;
use crate::utils::aggregate_proof;
use crate::utils::combine_comm;
use crate::utils::feed_message;
use crate::utils::generate_dumb_sponge;
use crate::utils::merge_proof;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_ff::UniformRand;
use ark_ff::{One, Zero};
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::IPForMLSumcheck;
use ark_linear_sumcheck::ml_sumcheck::protocol::PolynomialInfo;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::DenseMVPolynomial;
use ark_poly::DenseMultilinearExtension;
use ark_poly::MultilinearExtension;
use ark_poly::Polynomial;
use ark_poly_commit::challenge::ChallengeGenerator;
use ark_poly_commit::marlin_pst13_pc::CommitterKey as MaskCommitterKey;
use ark_poly_commit::marlin_pst13_pc::MarlinPST13;
use ark_poly_commit::multilinear_pc::data_structures::{
    Commitment, CommitterKey, Proof, UniversalParams, VerifierKey,
};
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_poly_commit::LabeledPolynomial;
use ark_poly_commit::PolynomialCommitment;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::end_timer;
use ark_std::rc::Rc;
use ark_std::start_timer;
use ark_std::time::Duration;
use ark_std::time::Instant;
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::Process,
    Count,
};
use mpi::{request, traits::*};
use rand::Rng;
use rand::RngCore;
use std::marker::PhantomData;
use std::ops::Neg;

pub struct ProverState<'a, E: Pairing> {
    pub index: &'a IndexProverKey<E>,
    pub num_variables: usize,
    pub prover_round: usize,
    pub log_parties: usize,
    pub witness_mask:
        LabeledPolynomial<E::ScalarField, SparsePolynomial<E::ScalarField, SparseTerm>>,
    pub witness_comm: Option<Commitment<E>>,
    pub r_x: Vec<E::ScalarField>,
    pub r_y: Vec<E::ScalarField>,
    pub eq_tilde_rx_comm: Option<Commitment<E>>,
    pub eq_tilde_ry_comm: Option<Commitment<E>>,
    pub vk: Option<IndexVerifierKey<E>>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
/// proof of opening
pub struct PartialProof<E: Pairing> {
    /// Evaluation of quotients
    pub proofs: Proof<E>,
    pub val: E::ScalarField,
    pub evals: Vec<E::ScalarField>,
}

pub fn mpi_sumcheck_coordinator<
    'a,
    F: Field,
    C: 'a + Communicator,
    R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    root_process: &Process<'a, C>,
    poly_info: &PolynomialInfo,
    prover_transcript: &mut R,
    default_last_sumcheck_state: DistrbutedSumcheckProverState<F>,
    log_num_workers: usize,
    num_nodes: usize,
) -> (Vec<ProverMsg<F>>, Vec<F>, Duration) {
    let time = Instant::now();

    let mut prover_msgs = Vec::new();
    let mut final_point = Vec::new();

    _ = prover_transcript.feed(&poly_info.clone());
    // assert!(1 << log_num_workers == size);

    let mut tot_time = time.elapsed();

    for round in 0..poly_info.num_variables - log_num_workers {
        let default_response = ProverMsg {
            evaluations: vec![F::zero(); poly_info.max_multiplicands + 1],
        };

        let responses_chunked: Vec<_> = gather_responses(
            log,
            &(stage.to_owned() + "_" + &round.to_string()),
            size,
            root_process,
            default_response,
        );

        let time = Instant::now();

        let mut prover_message = responses_chunked[0].clone();
        for i in 1..1 << log_num_workers {
            let tmp = responses_chunked[i].clone();
            // Aggregate results from different parties
            for j in 0..prover_message.evaluations.len() {
                prover_message.evaluations[j] = prover_message.evaluations[j] + tmp.evaluations[j]
            }
        }
        let _ = prover_transcript.feed(&prover_message);
        prover_msgs.push(prover_message.clone());
        let verifier_msg = Some(IPForMLSumcheck::sample_round(prover_transcript));
        // Using the aggregate results to generate the verifier's message.
        let r = verifier_msg.clone().unwrap().randomness;
        final_point.push(r);

        tot_time += time.elapsed();

        let requests_chunked = vec![r; num_nodes];
        scatter_requests(
            log,
            &(stage.to_owned() + "_" + &round.to_string()),
            root_process,
            &requests_chunked,
        );
    }

    if log_num_workers != 0 {
        let default_response = default_last_sumcheck_state;

        let responses_chunked: Vec<_> = gather_responses(
            log,
            &(stage.to_owned() + "_rest_rounds"),
            size,
            root_process,
            default_response,
        );

        let time = Instant::now();

        let merge_poly =
            merge_list_of_distributed_poly(&responses_chunked, poly_info, log_num_workers);

        let mut prover_state = IPForMLSumcheck::prover_init(&merge_poly);
        let mut verifier_msg = None;
        for _ in poly_info.num_variables - log_num_workers..poly_info.num_variables {
            let prover_message = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
            _ = prover_transcript.feed(&prover_message);
            prover_msgs.push(prover_message.clone());
            verifier_msg = Some(IPForMLSumcheck::sample_round(prover_transcript));
            final_point.push(verifier_msg.clone().unwrap().randomness);
        }

        tot_time += time.elapsed();
    }

    let requests_chunked = vec![final_point.clone(); num_nodes];
    scatter_requests(
        log,
        &(stage.to_owned() + "_final_point"),
        root_process,
        &requests_chunked,
    );

    (prover_msgs, final_point, tot_time)
}

pub fn mpi_zk_sumcheck_coordinators<
    'a,
    E: Pairing,
    C: 'a + Communicator,
    R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
    S: CryptographicSponge,
>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    root_process: &Process<'a, C>,
    poly_info: &PolynomialInfo,
    prover_transcript: &mut R,
    default_last_sumcheck_state: DistrbutedSumcheckProverState<E::ScalarField>,
    log_num_workers: usize,
    mask_rng: &mut R,
    mask_key: &MaskCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    opening_challenge: &mut ChallengeGenerator<E::ScalarField, S>,
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
    let (mask_commit, mut mask_randomness) =
        MarlinPST13::<_, _, PoseidonSponge<E::ScalarField>>::commit(
            mask_key,
            &vec_mask_poly,
            Some(mask_rng),
        )
        .unwrap();
    let g_commit = mask_commit[0].commitment();
    let _ = prover_transcript.feed(g_commit);
    let challenge = E::ScalarField::rand(prover_transcript);

    _ = prover_transcript.feed(&poly_info.clone());
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

    for round in 0..poly_info.num_variables - log_num_workers {
        let default_response = ProverMsg {
            evaluations: vec![E::ScalarField::zero(); poly_info.max_multiplicands + 1],
        };

        let responses_chunked: Vec<_> = gather_responses(
            log,
            &(stage.to_owned() + "_" + &round.to_string()),
            size,
            root_process,
            default_response,
        );

        let time = Instant::now();

        let mut prover_message = responses_chunked[0].clone();
        for i in 1..1 << log_num_workers {
            let tmp = responses_chunked[i].clone();
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

        let _ = prover_transcript.feed(&final_msg);
        prover_msgs.push(final_msg.clone());
        let verifier_msg = Some(IPForMLSumcheck::sample_round(prover_transcript));
        // Using the aggregate results to generate the verifier's message.
        let r = verifier_msg.clone().unwrap().randomness;
        final_point.push(r);

        tot_time += time.elapsed();

        let requests_chunked = vec![r; 1 << log_num_workers];
        scatter_requests(
            log,
            &(stage.to_owned() + "_" + &round.to_string()),
            root_process,
            &requests_chunked,
        );

        v_msg = verifier_msg.clone();
    }

    if log_num_workers != 0 {
        let default_response = default_last_sumcheck_state;

        let responses_chunked: Vec<_> = gather_responses(
            log,
            &(stage.to_owned() + "_rest_rounds"),
            size,
            root_process,
            default_response,
        );

        let time = Instant::now();

        let merge_poly =
            merge_list_of_distributed_poly(&responses_chunked, poly_info, log_num_workers);

        let mut prover_state = IPForMLSumcheck::prover_init(&merge_poly);
        let mut verifier_msg = None;
        for _ in poly_info.num_variables - log_num_workers..poly_info.num_variables {
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

            _ = prover_transcript.feed(&final_msg);
            prover_msgs.push(final_msg.clone());
            verifier_msg = Some(IPForMLSumcheck::sample_round(prover_transcript));
            final_point.push(verifier_msg.clone().unwrap().randomness);
            v_msg = verifier_msg.clone();
        }

        tot_time += time.elapsed();
    }

    let requests_chunked = vec![final_point.clone(); 1 << log_num_workers];
    scatter_requests(
        log,
        &(stage.to_owned() + "_final_point"),
        root_process,
        &requests_chunked,
    );

    let time = Instant::now();

    let opening = MarlinPST13::<_, SparsePolynomial<E::ScalarField, SparseTerm>, _>::open(
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

pub fn mpi_poly_commit_coordinator<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: Count,
    root_process: &Process<'a, C>,
    num_comms: usize,
    g: E::G1Affine,
    log_num_pub_workers: usize,
) -> (Vec<Commitment<E>>, Duration) {
    let default_response = vec![
        Commitment::<E> {
            nv: 0,
            g_product: g,
        };
        num_comms
    ];

    let responses_chunked: Vec<_> =
        gather_responses(log, stage, size, &root_process, default_response);

    // Round 1 process

    let time = Instant::now();

    let mut res = Vec::new();
    for j in 0..num_comms {
        let mut comm = Vec::new();
        for i in 0..1 << log_num_pub_workers {
            comm.push(responses_chunked[i][j].clone())
        }
        res.push(combine_comm(&comm))
    }

    (res, time.elapsed())
}

pub fn mpi_hiding_poly_commit_coordinator<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: Count,
    root_process: &Process<'a, C>,
    g: E::G1Affine,
    hiding_bound: usize,
    mask_num_var: Option<usize>,
    rng: &mut impl Rng,
    num_vars: usize,
    mask_ck: &MaskCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    log_num_pub_workers: usize,
) -> (
    ZKMLCommitment<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    Duration,
) {
    let (base_commitment_vec, mut tot_time): (Vec<Commitment<E>>, Duration) =
        mpi_poly_commit_coordinator(log, stage, size, root_process, 1, g, log_num_pub_workers);

    let time = Instant::now();
    let mut p_hat = SparsePolynomial::<E::ScalarField, SparseTerm>::zero();
    if let Some(mask_num_vars) = mask_num_var {
        p_hat = generate_mask_polynomial(rng, mask_num_vars, hiding_bound, false);
    } else {
        p_hat = generate_mask_polynomial(rng, num_vars, hiding_bound, false);
    }
    let labeled_p_hat = LabeledPolynomial::new("p_hat".to_owned(), p_hat, Some(hiding_bound), None);
    let hiding_commitment: E::G1Affine = ZKMLCommit::<
        E,
        SparsePolynomial<E::ScalarField, SparseTerm>,
    >::commit_mask(&mask_ck, &labeled_p_hat, rng);

    let hidden_commitment: E::G1Affine =
        (base_commitment_vec[0].g_product + hiding_commitment).into();
    let commitment = Commitment {
        g_product: hidden_commitment,
        nv: num_vars,
    };

    ((commitment, labeled_p_hat), tot_time + time.elapsed())
}

// /// Batch opening polynomial
pub fn mpi_batch_open_poly_coordinator<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    root_process: &Process<'a, C>,
    log_num_workers: usize,
    num_var: usize,
    num_poly: usize,
    merge_ck: &CommitterKey<E>,
    comms: &[Commitment<E>],
    final_point: &[E::ScalarField],
    eta: E::ScalarField,
    g: E::G1Affine,
    log_num_pub_workers: usize,
) -> (BatchOracleEval<E>, Duration) {
    let default_response = PartialProof {
        proofs: Proof {
            proofs: vec![g; num_var],
        },
        val: E::ScalarField::zero(),
        evals: vec![E::ScalarField::one(); num_poly],
    };
    let responses_chunked: Vec<_> = gather_responses(
        log,
        &(stage.to_owned() + "_obtain_proof"),
        size,
        &root_process,
        default_response,
    );

    let time = Instant::now();

    let mut pfs = Vec::new();
    let mut rs = Vec::new();
    for i in 0..(1 << log_num_pub_workers) {
        pfs.push(responses_chunked[i].proofs.clone());
        rs.push(responses_chunked[i].val);
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
        for j in 0..(1 << log_num_pub_workers) {
            e.push(responses_chunked[j].evals[i]);
        }
        let ep = DenseMultilinearExtension::from_evaluations_vec(log_num_workers, e);
        if i < comms.len() {
            evals.push(
                ep.evaluate(&final_point[num_var - log_num_workers..num_var])
                    .unwrap(),
            );
        } else {
            debug_evals.push(
                ep.evaluate(&final_point[num_var - log_num_workers..num_var])
                    .unwrap(),
            );
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

// /// Batch opening polynomial
pub fn mpi_zk_open_poly_coordinator<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    root_process: &Process<'a, C>,
    log_num_workers: usize,
    num_var: usize,
    // merge_ck: &CommitterKey<E>,
    comm: &Commitment<E>,
    final_point: &[E::ScalarField],
    // mask_ck: &MaskCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    ck: &ZKMLCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    p_hat: &LabeledPolynomial<E::ScalarField, SparsePolynomial<E::ScalarField, SparseTerm>>, // TODO: Represent this as a prg seed? how many bits?
    log_num_pub_workers: usize,
) -> (ZKMLProof<E>, Duration) {
    let (batch_oracle, mut tot_time) = mpi_batch_open_poly_coordinator(
        log,
        stage,
        size,
        root_process,
        log_num_workers,
        num_var,
        1,
        &ck.0,
        &[comm.clone()],
        final_point,
        E::ScalarField::one(),
        ck.0.g,
        log_num_pub_workers,
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
    //modify MLProof
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

// /// Batch opening polynomial
pub fn mpi_eval_poly_coordinator<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    root_process: &Process<'a, C>,
    log_num_workers: usize,
    num_var: usize,
    num_poly: usize,
    final_point: &[E::ScalarField],
    log_num_pub_workers: usize,
) -> (Vec<E::ScalarField>, Duration) {
    let default_response = vec![E::ScalarField::one(); num_poly];
    let responses_chunked: Vec<_> = gather_responses(
        log,
        &(stage.to_owned() + "_obtain_eval"),
        size,
        &root_process,
        default_response,
    );

    let time = Instant::now();

    let mut evals = Vec::new();
    for i in 0..num_poly {
        let mut e = Vec::new();
        for j in 0..(1 << log_num_pub_workers) {
            e.push(responses_chunked[j][i]);
        }
        let ep = DenseMultilinearExtension::from_evaluations_vec(log_num_workers, e);

        evals.push(
            ep.evaluate(&final_point[num_var - log_num_workers..num_var])
                .unwrap(),
        );
    }

    (evals, time.elapsed())
}

pub struct PublicProver<E: Pairing> {
    _pairing: PhantomData<E>,
}

impl<E: Pairing> PublicProver<E> {
    pub fn new<'a, C: 'a + Communicator>(
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
        pk: &IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
    ) -> (R1CSProof<E>, Duration)
    where
        E: Pairing,
    {
        let mut fs_rng = Blake2s512Rng::setup();
        let mut mask_rng = Blake2s512Rng::setup();
        assert!(fs_rng.feed(&"initialize".as_bytes()).is_ok());
        assert!(mask_rng.feed(&"initialize".as_bytes()).is_ok());
        let mut challenge_gen =
            ChallengeGenerator::new_univariate(&mut generate_dumb_sponge::<E::ScalarField>());
        let (mut prover_message, _, _num_variables, time) = Self::prove(
            log,
            stage,
            size,
            root_process,
            log_num_workers,
            pk.padded_num_var,
            pk,
            vk,
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

        (
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
            },
            time,
        )
    }

    pub fn prove<
        'a,
        R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
        S: CryptographicSponge,
        C: 'a + Communicator,
    >(
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
        num_variables: usize,
        index_pk: &IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
        fs_rng: &mut R,
        mask_rng: &mut R,
        mask_challenge_generator_for_open: &mut ChallengeGenerator<E::ScalarField, S>,
    ) -> (
        Vec<ProverMessage<E>>,
        Vec<VerifierMessage<E>>,
        usize,
        Duration,
    ) {
        //todo: padding the R1CS instance, witness and io

        let time = Instant::now();

        let init_time = start_timer!(|| "Prover init");
        let mut prover_state = Self::prover_init(index_pk, vk, num_variables, log_num_workers);
        end_timer!(init_time);

        let mut prover_message: Vec<ProverMessage<E>> = Vec::new();
        let mut verifier_state: VerifierState<E> =
            DFSVerifier::verifier_init(prover_state.num_variables);
        let mut verifier_message: Vec<VerifierMessage<E>> = Vec::new();

        let mut tot_time = time.elapsed();

        // First round: the prover message is just the witness commitment
        let first_time = start_timer!(|| "Prover first round");

        let (mut prover_first_message, time) = Self::prover_first_round(
            &mut prover_state,
            mask_rng,
            log,
            stage,
            size,
            root_process,
            log_num_workers,
        );

        prover_first_message = feed_message(fs_rng, prover_first_message);
        prover_message.push(prover_first_message);
        end_timer!(first_time);

        tot_time += time;

        // This first challenge is used for checking the hadamard product of AB - C ?= 0.
        // The following sumcheck, in prover_second_round, doesn't verify the well formedness of its components, which future rounds will do.
        let verifier_first_message = DFSVerifier::verifier_first_round(&mut verifier_state, fs_rng);

        let second_time = start_timer!(|| "Prover second round");
        let (mut prover_second_message, time) = Self::prover_second_round(
            &mut prover_state,
            &verifier_first_message.verifier_message,
            fs_rng,
            mask_rng,
            mask_challenge_generator_for_open,
            log,
            stage,
            size,
            root_process,
            log_num_workers,
        );
        prover_second_message = feed_message(fs_rng, prover_second_message);
        prover_message.push(prover_second_message);
        end_timer!(second_time);

        tot_time += time;

        // This challenge is used for verifying the well formedness of the subclaim evaluation from the previous round.
        // In particular, this challenge batches a, b, and c polynomials using 3 scalars
        let verifier_second_message =
            DFSVerifier::verifier_second_round(&mut verifier_state, fs_rng);

        let third_time = start_timer!(|| "Prover third round");
        let (mut prover_third_message, time) = Self::prover_third_round(
            &mut prover_state,
            &verifier_second_message.verifier_message,
            fs_rng,
            mask_rng,
            mask_challenge_generator_for_open,
            log,
            stage,
            size,
            root_process,
            log_num_workers,
        );
        prover_third_message = feed_message(fs_rng, prover_third_message);
        prover_message.push(prover_third_message);
        end_timer!(third_time);

        tot_time += time;

        let verifier_fifth_message = DFSVerifier::verifier_fifth_round(&mut verifier_state, fs_rng);
        println!("verifier_fifth_message: {:?}", verifier_fifth_message.verifier_message);

        let holo_time = Instant::now();

        let fifth_time = start_timer!(|| "Prover fifth round");
        let (mut prover_fifth_message, time) = Self::prover_fifth_round(
            &mut prover_state,
            fs_rng,
            verifier_fifth_message.verifier_message[0],
            log_num_workers,
            log,
            stage,
            size,
            root_process,
            1 << log_num_workers,
        );
        prover_fifth_message = feed_message(fs_rng, prover_fifth_message);
        prover_message.push(prover_fifth_message);
        end_timer!(fifth_time);

        tot_time += time;

        println!("holography time: {:?}", holo_time.elapsed());

        verifier_message.push(verifier_first_message);
        verifier_message.push(verifier_second_message);
        verifier_message.push(verifier_fifth_message);
        (
            prover_message,
            verifier_message,
            prover_state.num_variables,
            tot_time,
        )
    }

    //using little endian as ml_sumcheck, i.e. io[0,0,0,0,1] = F_{io}(1,0,0,0,0).
    pub fn prover_init<'a>(
        index: &'a IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
        num_variables: usize,
        log_parties: usize,
    ) -> ProverState<'a, E> {
        ProverState {
            index: (index),
            num_variables,
            prover_round: (0),
            log_parties,
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
            vk: Some(vk.clone()),
        }
    }

    pub fn prover_init_from<'a>(
        index: &'a IndexProverKey<E>,
        vk: &IndexVerifierKey<E>,
        num_variables: usize,
        log_parties: usize,
        state: &ProverState<E>,
    ) -> ProverState<'a, E> {
        ProverState {
            index: (index),
            num_variables,
            prover_round: state.prover_round,
            log_parties,
            witness_mask: LabeledPolynomial::<E::ScalarField, MaskPolynomial<E>>::new(
                "init_mask".into(),
                MaskPolynomial::<E>::zero(),
                None,
                None,
            ),
            witness_comm: None,
            r_x: state.r_x.clone(),
            r_y: state.r_y.clone(),
            eq_tilde_rx_comm: state.eq_tilde_rx_comm.clone(),
            eq_tilde_ry_comm: state.eq_tilde_ry_comm.clone(),
            vk: Some(vk.clone()),
        }
    }
    /// In round 1, prover only need to commit to witness w
    pub fn prover_first_round<'a, R: RngCore, C: 'a + Communicator>(
        state: &mut ProverState<E>,
        mask_rng: &mut R,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
    ) -> (ProverMessage<E>, Duration) {
        assert_eq!(state.prover_round, 0);
        state.prover_round = 1;
        let (zk_commitment_w, time) = mpi_hiding_poly_commit_coordinator(
            log,
            stage,
            size,
            root_process,
            state.index.ck_w.0.g,
            2,
            None,
            mask_rng,
            state.num_variables,
            &state.index.ck_w.1,
            log_num_workers,
        );

        state.witness_comm = Some(zk_commitment_w.0.clone());
        state.witness_mask = zk_commitment_w.1.clone();

        (
            ProverMessage {
                sumcheck_message: None,
                group_message: None,
                zk_proof_message: None,
                proof_message: None,
                commitment_message: Some(zk_commitment_w.0),
                commitment_message_2: None,
                lookup_message: None,
                zksumcheck_message: None,
            },
            time,
        )
    }

    pub fn prover_second_round<
        'a,
        R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
        S: CryptographicSponge,
        C: 'a + Communicator,
    >(
        state: &mut ProverState<E>,
        v_msg: &Vec<E::ScalarField>,
        fs_rng: &mut R,
        mask_rng: &mut R,
        mask_challenge_generator_for_open: &mut ChallengeGenerator<E::ScalarField, S>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
    ) -> (ProverMessage<E>, Duration) {
        let start = start_timer_buf!(log, || format!("Coord: Generating stage0 requests"));
        let requests_chunked = vec![v_msg.clone(); 1 << state.log_parties];
        end_timer_buf!(log, start);

        scatter_requests(log, "stage0", &root_process, &requests_chunked);

        let time = Instant::now();

        assert_eq!(state.prover_round, 1);
        let num_variables = state.num_variables;
        let poly_info = PolynomialInfo {
            max_multiplicands: 3,
            num_variables: num_variables,
        };
        let mut q_polys = ListOfProductsOfPolynomials::new(1);
        let default_poly = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![E::ScalarField::zero(), E::ScalarField::zero()],
        );

        let A_B_hat = vec![
            Rc::new(default_poly.clone()),
            Rc::new(default_poly.clone()),
            Rc::new(default_poly.clone()),
        ];
        let C_hat = vec![Rc::new(default_poly.clone()), Rc::new(default_poly.clone())];

        q_polys.add_product(A_B_hat, E::ScalarField::one());
        q_polys.add_product(C_hat, E::ScalarField::one().neg());
        let sumcheck_state = IPForMLSumcheck::prover_init(&q_polys);
        let default_last_sumcheck_state = obtain_distrbuted_sumcheck_prover_state(&sumcheck_state);

        let mut tot_time = time.elapsed();

        let (pf, final_point, time) = mpi_zk_sumcheck_coordinators(
            log,
            stage,
            size,
            root_process,
            &poly_info,
            fs_rng,
            default_last_sumcheck_state,
            log_num_workers,
            mask_rng,
            &state.index.ck_mask,
            mask_challenge_generator_for_open,
        );

        tot_time = tot_time + time;

        let (evals, time): (Vec<E::ScalarField>, Duration) = mpi_eval_poly_coordinator::<E, C>(
            log,
            &(stage.to_owned() + "_za_zb_zc_eval"),
            size,
            root_process,
            state.log_parties,
            state.num_variables,
            3,
            &final_point,
            log_num_workers,
        );

        tot_time = tot_time + time;

        let (val_a, val_b, val_c) = (evals[0].clone(), evals[1].clone(), evals[2].clone());

        state.prover_round = 2;
        state.r_x = final_point.to_vec();

        (
            ProverMessage {
                sumcheck_message: None,
                group_message: (Some(vec![val_a, val_b, val_c])),
                zk_proof_message: None,
                proof_message: None,
                commitment_message: None,
                commitment_message_2: None,
                lookup_message: None,
                zksumcheck_message: Some(pf),
            },
            tot_time,
        )
    }

    pub fn prover_third_round<
        'a,
        R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
        S: CryptographicSponge,
        C: 'a + Communicator,
    >(
        state: &mut ProverState<E>,
        v_msg: &Vec<E::ScalarField>,
        fs_rng: &mut R,
        mask_rng: &mut R,
        mask_challenge_generator_for_open: &mut ChallengeGenerator<E::ScalarField, S>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
    ) -> (ProverMessage<E>, Duration) {
        assert_eq!(state.prover_round, 2);

        let start = start_timer_buf!(log, || format!("Coord: Generating stage1 requests"));
        let requests_chunked = vec![v_msg.clone(); 1 << state.log_parties];
        end_timer_buf!(log, start);

        scatter_requests(log, "stage1", &root_process, &requests_chunked);

        let time = Instant::now();

        let mut q_polys = ListOfProductsOfPolynomials::new(1);
        let default_poly = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![E::ScalarField::zero(), E::ScalarField::zero()],
        );

        let A_hat = vec![Rc::new(default_poly.clone()), Rc::new(default_poly.clone())];
        let B_hat = vec![Rc::new(default_poly.clone()), Rc::new(default_poly.clone())];
        let C_hat = vec![Rc::new(default_poly.clone()), Rc::new(default_poly.clone())];

        q_polys.add_product(A_hat, v_msg[0]);
        q_polys.add_product(B_hat, v_msg[1]);
        q_polys.add_product(C_hat, v_msg[2]);

        let num_variables = state.num_variables;
        let poly_info = PolynomialInfo {
            max_multiplicands: 2,
            num_variables: num_variables,
        };

        let sumcheck_state = IPForMLSumcheck::prover_init(&q_polys);
        let default_last_sumcheck_state = obtain_distrbuted_sumcheck_prover_state(&sumcheck_state);

        let mut tot_time = time.elapsed();

        let (pf, final_point, time) = mpi_zk_sumcheck_coordinators(
            log,
            stage,
            size,
            root_process,
            &poly_info,
            fs_rng,
            default_last_sumcheck_state,
            log_num_workers,
            mask_rng,
            &state.index.ck_mask,
            mask_challenge_generator_for_open,
        );
        state.prover_round = 3;
        state.r_y = final_point.to_vec();

        tot_time += time;

        let (val_ws, time) = mpi_eval_poly_coordinator::<E, C>(
            log,
            &(stage.to_owned() + "w_eval"),
            size,
            root_process,
            state.log_parties,
            state.num_variables,
            1,
            &final_point,
            log_num_workers,
        );

        let val_w = val_ws[0];

        tot_time += time;

        let default_response = (
            E::ScalarField::one(),
            E::ScalarField::one(),
            E::ScalarField::one(),
        );
        let responses_chunked: Vec<_> = gather_responses(
            log,
            &(stage.to_owned() + "_obtain_eval"),
            size,
            &root_process,
            default_response,
        );

        let time = Instant::now();

        let mut val_a = E::ScalarField::zero();
        let mut val_b = E::ScalarField::zero();
        let mut val_c = E::ScalarField::zero();
        for i in 0..responses_chunked.len() {
            val_a = val_a + responses_chunked[i].0;
            val_b = val_b + responses_chunked[i].1;
            val_c = val_c + responses_chunked[i].2;
        }

        let val_m = val_a * v_msg[0] + val_b * v_msg[1] + val_c * v_msg[2];

        tot_time += time.elapsed();

        let (comms, time) = mpi_poly_commit_coordinator(
            log,
            stage,
            size,
            root_process,
            2,
            state.index.ck_index.g,
            log_num_workers,
        );
        state.eq_tilde_rx_comm = Some(comms[0].clone());
        state.eq_tilde_ry_comm = Some(comms[1].clone());

        tot_time += time;

        let (zk_open_pf, time) = mpi_zk_open_poly_coordinator(
            log,
            stage,
            size,
            root_process,
            log_num_workers,
            state.num_variables,
            &state.witness_comm.clone().unwrap(),
            &state.r_y[..],
            &state.index.ck_w,
            &state.witness_mask,
            log_num_workers,
        );
        tot_time += time;

        (
            ProverMessage {
                sumcheck_message: None,
                group_message: Some(vec![val_w, val_m]),
                zk_proof_message: Some(zk_open_pf),
                proof_message: None,
                commitment_message: Some(comms[0].clone()),
                commitment_message_2: Some(comms[1].clone()),
                lookup_message: None,
                zksumcheck_message: Some(pf),
            },
            tot_time,
        )
    }

    pub fn prover_fifth_round<
        'a,
        R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
        C: 'a + Communicator,
    >(
        state: &mut ProverState<E>,
        rng: &mut R,
        v_msg: E::ScalarField,
        log_num_workers: usize,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        root_process: &Process<'a, C>,
        num_nodes: usize,
    ) -> (ProverMessage<E>, Duration) {
        let start = start_timer_buf!(log, || format!("Coord: Generating stage5 requests"));
        let requests_chunked = vec![v_msg.clone(); num_nodes];
        end_timer_buf!(log, start);

        scatter_requests(
            log,
            &(stage.to_owned() + "_5"),
            &root_process,
            &requests_chunked,
        );

        let time = Instant::now();

        let q_num_vars = state.index.real_len_val.log_2();

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

        let x_r: E::ScalarField = get_scalar_challenge(rng);
        let x_c: E::ScalarField = get_scalar_challenge(rng);

        let mut tot_time = time.elapsed();

        let requests_chunked = vec![(x_r, x_c); num_nodes];
        scatter_requests(log, "stage5", &root_process, &requests_chunked);

        let (comms, time) = mpi_poly_commit_coordinator(
            log,
            "stage0_poly_commit",
            size,
            &root_process,
            4,
            state.index.ck_index.g,
            log_num_workers,
        );

        tot_time += time;

        _ = rng.feed(&comms[0]);
        _ = rng.feed(&comms[1]);
        _ = rng.feed(&comms[2]);
        _ = rng.feed(&comms[3]);

        let z: Vec<E::ScalarField> = get_vector_challenge(rng, q_num_vars);
        let lambda: E::ScalarField = get_scalar_challenge(rng);

        let requests_chunked = vec![(z.clone(), lambda.clone()); num_nodes];
        scatter_requests(log, "stage5", &root_process, &requests_chunked);

        println!("state.index.num_variables_val: {:?}", state.index.num_variables_val);
        default_sumcheck_poly_list(
            &lambda,
            q_num_vars - state.index.num_variables_val,
            &mut q_polys,
        );

        let z: Vec<E::ScalarField> = get_vector_challenge(rng, q_num_vars);
        let lambda: E::ScalarField = get_scalar_challenge(rng);

        let requests_chunked = vec![(z.clone(), lambda.clone()); num_nodes];
        scatter_requests(log, "stage5", &root_process, &requests_chunked);

        default_sumcheck_poly_list(
            &lambda,
            q_num_vars - state.index.num_variables_val,
            &mut q_polys,
        );

        let poly_info = PolynomialInfo {
            max_multiplicands: 3,
            num_variables: q_num_vars,
        };
        let default_last_sumcheck_state = poly_list_to_prover_state(&q_polys);

        let (prover_msgs, final_point, time) = mpi_sumcheck_coordinator(
            log,
            "lookup_sumcheck",
            size,
            &root_process,
            &poly_info,
            rng,
            default_last_sumcheck_state,
            log_num_workers,
            num_nodes,
        );

        tot_time += time;

        let eta: E::ScalarField = get_scalar_challenge(rng);
        let requests_chunked = vec![eta.clone(); num_nodes];
        scatter_requests(log, "stage5", &root_process, &requests_chunked);

        let vk = state.vk.clone().unwrap();

        let (batch_oracle, time) = mpi_batch_open_poly_coordinator(
            log,
            "stage_lookup_poly_open",
            size,
            &root_process,
            log_num_workers,
            poly_info.num_variables,
            15,
            &state.index.ck_index,
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
            eta,
            state.index.ck_index.g,
            log_num_workers,
        );

        tot_time += time;

        (
            ProverMessage {
                sumcheck_message: None,
                group_message: None,
                zk_proof_message: None,
                proof_message: None,
                commitment_message: None,
                commitment_message_2: None,
                lookup_message: Some(LogLookupProof {
                    sumcheck_pfs: prover_msgs,
                    info: poly_info,
                    point: final_point.clone(),
                    degree_diff: q_num_vars - state.index.num_variables_val,
                    batch_oracle,
                }),
                zksumcheck_message: None,
            },
            tot_time,
        )
    }
}
