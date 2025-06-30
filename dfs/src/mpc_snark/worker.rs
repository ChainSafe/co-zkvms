use crate::math::Math;
use crate::mpc::rss::RssPoly;
use crate::mpc::rss::RssSumcheck;
use crate::mpc::rss::SSRandom;
use crate::mpc::SSOpen;
use crate::mpi_snark::coordinator::PartialProof;
use crate::mpi_snark::worker::DistributedProverKey;
use crate::mpi_utils::receive_requests;
use crate::mpi_utils::send_and_receive;
use crate::mpi_utils::send_responses;
use crate::mpi_utils::DistrbutedSumcheckProverState;
use crate::snark::indexer::IndexProverKey;
use crate::subprotocols::loglookup::default_sumcheck_poly_list;
use crate::subprotocols::loglookup::poly_list_to_prover_state;
use crate::utils::aggregate_poly;
use crate::utils::dense_scalar_prod;
use crate::utils::distributed_open;
use crate::utils::generate_eq;
use crate::utils::partial_generate_eq;
use crate::utils::split_vec;
use crate::R1CSInstance;
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_ff::One;
use ark_ff::Zero;
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::data_structures::PolynomialInfo;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::verifier::VerifierMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::IPForMLSumcheck;
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_poly::DenseMultilinearExtension;
use ark_poly::Polynomial;
use ark_poly_commit::multilinear_pc::data_structures::Commitment;
use ark_poly_commit::multilinear_pc::data_structures::CommitterKey;
use ark_poly_commit::multilinear_pc::data_structures::Proof;
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use ark_std::marker::PhantomData;
use ark_std::rc::Rc;
use mpi::topology::Process;
use mpi::traits::Communicator;
use mpi::Count;
use rand::RngCore;
use std::cmp::min;
use std::ops::Index;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistributedRSSProverKey<E: Pairing> {
    pub party_id: usize,
    pub num_parties: usize,
    pub ipk: IndexProverKey<E>,
    pub pub_ipk: IndexProverKey<E>,

    pub row: Vec<usize>,
    pub col: Vec<usize>,
    pub val_a: DenseMultilinearExtension<E::ScalarField>,
    pub val_b: DenseMultilinearExtension<E::ScalarField>,
    pub val_c: DenseMultilinearExtension<E::ScalarField>,
    // pub r1cs: R1CSInstance<E::ScalarField>,
    pub z: RssPoly<E>,
    pub za: RssPoly<E>,
    pub zb: RssPoly<E>,
    pub zc: RssPoly<E>,
    pub num_variables: usize,
    pub seed_0: String,
    pub seed_1: String,

    // todo: remove
    pub za_poly: Option<DenseMultilinearExtension<E::ScalarField>>,
    pub zb_poly: Option<DenseMultilinearExtension<E::ScalarField>>,
    pub zc_poly: Option<DenseMultilinearExtension<E::ScalarField>>,
}

pub struct RssPrivateProver<E: Pairing> {
    _pairing: PhantomData<E>,
}

impl<E: Pairing> RssPrivateProver<E> {
    pub fn prove<'a, C: 'a + Communicator, R: RngCore + FeedableRNG>(
        pk: &DistributedRSSProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
        log_num_workers_per_party: usize,
        pub_log_num_workers: usize,
        start_eq: usize,
        log_chunk_size: usize,
        random_rng: &mut SSRandom<R>,
        active: bool,
        pub_start_eq: usize,
        pub_log_chunk_size: usize,
    ) -> (usize, usize) {
        let mut send_size = 0;
        let mut recv_size = 0;

        let (size1, size2) = Self::prover_first_round(
            pk,
            log,
            &(stage.to_owned() + "round 1"),
            size,
            rank,
            &root_process,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let ((full_eq_rx, rx), size1, size2) = Self::prover_second_round(
            pk,
            log,
            &(stage.to_owned() + "round 2"),
            size,
            rank,
            root_process,
            log_num_workers_per_party,
            start_eq,
            log_chunk_size,
            random_rng,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        // let mut eq_tilde_rx: Vec<E::ScalarField> =
        //     vec![E::ScalarField::zero(); pk.ipk.num_variables_val.pow2()];

        // for i in 0..min(pk.ipk.rows_indexed.len(), pk.ipk.num_variables_val.pow2()) {
        //     if pk.row[i] != usize::MAX {
        //         eq_tilde_rx[i] = *full_eq_rx.index(pk.ipk.rows[i]);
        //     }
        // }
        // let eq_tilde_rx = DenseMultilinearExtension::from_evaluations_vec(
        //     pk.ipk.num_variables_val,
        //     eq_tilde_rx,
        // );

        let ((pub_eq_tilde_rx, pub_eq_tilde_ry, pub_val_M), full_eq_ry, size1, size2) =
            Self::prover_third_round(
                pk,
                &pk.pub_ipk,
                log,
                stage,
                size,
                rank,
                root_process,
                log_num_workers_per_party,
                start_eq,
                log_chunk_size,
                &full_eq_rx,
                random_rng,
                active,
                rx,
                pub_start_eq,
                pub_log_chunk_size,
            );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let mut full_eq_tilde_rx: Vec<E::ScalarField> =
            vec![E::ScalarField::zero(); pk.num_variables.pow2()];
        let mut full_eq_tilde_ry: Vec<E::ScalarField> =
            vec![E::ScalarField::zero(); pk.num_variables.pow2()];
        for i in 0..pk.num_variables.pow2() {
            if pk.row[i] != usize::MAX {
                full_eq_tilde_rx[i] = *full_eq_rx.index(pk.row[i]);
            }
            if pk.col[i] != usize::MAX {
                full_eq_tilde_ry[i] = *full_eq_ry.index(pk.col[i]);
            }
        }
        println!("pk.num_variables.pow2(): {:?}", pk.num_variables.pow2());

        let full_eq_tilde_rx =
            DenseMultilinearExtension::from_evaluations_vec(pk.num_variables, full_eq_tilde_rx);
        let full_eq_tilde_ry =
            DenseMultilinearExtension::from_evaluations_vec(pk.num_variables, full_eq_tilde_ry);

        let (size1, size2) = Self::prover_fifth_round(
            pk,
            &pk.pub_ipk,
            log,
            stage,
            size,
            rank,
            root_process,
            pub_log_num_workers,
            pub_start_eq,
            pub_log_chunk_size,
            &full_eq_tilde_rx,
            &full_eq_tilde_ry,
            &pub_eq_tilde_rx,
            &pub_eq_tilde_ry,
            &pub_val_M,
            active,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        (send_size, recv_size)
    }

    /// In round 1, prover only need to commit to witness w
    pub fn prover_first_round<'a, C: 'a + Communicator>(
        pk: &DistributedRSSProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
    ) -> (usize, usize) {
        rss_poly_commit_worker(
            log,
            stage,
            size,
            rank,
            root_process,
            &vec![&pk.z],
            &pk.ipk.ck_w.0,
        )
    }

    pub fn prover_second_round<'a, C: 'a + Communicator, R: RngCore + FeedableRNG>(
        pk: &DistributedRSSProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
        log_num_workers_per_party: usize,
        start_eq: usize,
        log_chunk_size: usize,
        random_rng: &mut SSRandom<R>,
    ) -> (
        (
            DenseMultilinearExtension<E::ScalarField>,
            Vec<E::ScalarField>,
        ),
        usize,
        usize,
    ) {
        let mut send_size = 0;
        let mut recv_size = 0;

        let start = start_timer_buf!(log, || format!("Coord: Generating stage0 requests"));
        let (v_msg, size1): (Vec<E::ScalarField>, usize) =
            receive_requests(log, rank, stage, &root_process, 0);
        recv_size = recv_size + size1;
        end_timer_buf!(log, start);

        let num_variables = pk.ipk.padded_num_var;

        let eq_func = partial_generate_eq(&v_msg, start_eq, log_chunk_size);

        let (final_point, size1, size2) = rss_first_sumcheck_worker(
            log,
            stage,
            size,
            rank,
            root_process,
            &pk.za,
            &pk.zb,
            &pk.zc,
            &eq_func,
            random_rng,
            log_num_workers_per_party,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let randomness = final_point[0..num_variables].to_vec();
        // println!("final_point: {:?}", final_point.len());
        // println!("num_variables: {:?}", num_variables);

        let (val_a, val_b, val_c) = (
            pk.za.share_0.evaluate(&randomness),
            pk.zb.share_0.evaluate(&randomness),
            pk.zc.share_0.evaluate(&randomness),
        );

        // let val_a_check = pk.za_poly.evaluate(&final_point);
        // let val_b_check = pk.zb_poly.evaluate(&final_point);
        // let val_c_check = pk.zc_poly.evaluate(&final_point);

        // println!("val_a_check: {:?}", val_a_check);
        // println!("val_b_check: {:?}", val_b_check);
        // println!("val_c_check: {:?}", val_c_check);

        // [BigInt([15766573602034106474, 14910596996200590167, 13955810961206256749, 1225194845514926437])]

        let response = vec![val_a, val_b, val_c];
        let size1 = send_responses(
            log,
            rank,
            &(stage.to_owned() + "round 2_eval"),
            &root_process,
            &response,
            0,
        );
        send_size = send_size + size1;

        let eq_rx = generate_eq(&final_point);
        ((eq_rx, final_point), send_size, recv_size)
    }

    pub fn prover_third_round<'a, C: 'a + Communicator, R: RngCore + FeedableRNG>(
        pk: &DistributedRSSProverKey<E>,
        pub_ipk: &IndexProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
        log_num_workers_per_party: usize,
        start_eq: usize,
        log_chunk_size: usize,
        full_eq_rx: &DenseMultilinearExtension<E::ScalarField>,
        random_rng: &mut SSRandom<R>,
        active: bool,
        r_x: Vec<E::ScalarField>, // todo this *full* final point from first sumcheck ie. full_eq_rx = generate_eq(&r_x)
        pub_start_eq: usize,
        pub_log_chunk_size: usize,
    ) -> (
        (
            DenseMultilinearExtension<E::ScalarField>,
            DenseMultilinearExtension<E::ScalarField>,
            DenseMultilinearExtension<E::ScalarField>,
        ),
        DenseMultilinearExtension<E::ScalarField>,
        usize,
        usize,
    ) {
        let mut send_size = 0;
        let mut recv_size = 0;

        let start = start_timer_buf!(log, || format!("Coord: receiving stage1 requests"));
        let (v_msg, size1): (Vec<E::ScalarField>, usize) =
            receive_requests(log, rank, stage, &root_process, 0);
        recv_size = recv_size + size1;
        end_timer_buf!(log, start);

        let num_variables = pk.ipk.padded_num_var;
        let instance_size = pk.num_variables.pow2();
        let chunk_size = pk.ipk.num_variables_val.pow2();
        let c_start = pk.party_id * instance_size / pk.num_parties;

        let mut A_rx = vec![E::ScalarField::zero(); chunk_size];
        let mut B_rx = vec![E::ScalarField::zero(); chunk_size];
        let mut C_rx = vec![E::ScalarField::zero(); chunk_size];

        for i in 0..pk.ipk.cols_indexed.len() {
            let col = pk.ipk.cols_indexed[i] - c_start; // local offset 0..range_len-1
            let row = pk.ipk.rows_indexed[i];
            let v = full_eq_rx.index(row);

            A_rx[col] += pk.ipk.val_a_indexed[i] * v;
            B_rx[col] += pk.ipk.val_b_indexed[i] * v;
            C_rx[col] += pk.ipk.val_c_indexed[i] * v;
        }

        let (final_point, size1, size2) = rss_second_sumcheck_worker(
            log,
            stage,
            size,
            rank,
            root_process,
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, A_rx),
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, B_rx),
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, C_rx),
            &pk.z,
            random_rng,
            log_num_workers_per_party,
            &v_msg,
        );
        println!("final_point: {:?}", final_point);
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let (size1, size2) = rss_eval_poly_worker::<E, C>(
            log,
            &(stage.to_owned() + "round 3_w_eval"),
            size,
            rank,
            &root_process,
            vec![&pk.z],
            1,
            &final_point,
            pk.num_variables,
            log_num_workers_per_party,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;
        let r_y = final_point.to_vec();

        let mut worker_eq_tilde_rx: DenseMultilinearExtension<E::ScalarField> =
            DenseMultilinearExtension {
                evaluations: vec![E::ScalarField::zero()],
                num_vars: 0,
            };
        let mut worker_eq_tilde_ry: DenseMultilinearExtension<E::ScalarField> =
            DenseMultilinearExtension {
                evaluations: vec![E::ScalarField::zero()],
                num_vars: 0,
            };
        let mut pub_val_M: DenseMultilinearExtension<E::ScalarField> = DenseMultilinearExtension {
            evaluations: vec![E::ScalarField::zero()],
            num_vars: 0,
        };

        let full_eq_ry: DenseMultilinearExtension<<E as Pairing>::ScalarField> = generate_eq(&r_y);
        if active {
            let mut worker_eq_tilde_rx_evals =
                vec![E::ScalarField::zero(); pub_log_chunk_size.pow2()];
            let mut worker_eq_tilde_ry_evals =
                vec![E::ScalarField::zero(); pub_log_chunk_size.pow2()];

            let mut val_a = E::ScalarField::zero();
            let mut val_b = E::ScalarField::zero();
            let mut val_c = E::ScalarField::zero();

            for (i, ((((v_a, v_b), v_c), row), col)) in pub_ipk
                .val_a
                .evaluations
                .iter()
                .zip(pub_ipk.val_b.evaluations.iter())
                .zip(pub_ipk.val_c.evaluations.iter())
                .zip(pub_ipk.row.iter())
                .zip(pub_ipk.col.iter())
                .enumerate()
            {
                if i < pub_ipk.real_len_val {
                    val_a += *v_a * full_eq_rx.index(*row) * full_eq_ry.index(*col);
                    val_b += *v_b * full_eq_rx.index(*row) * full_eq_ry.index(*col);
                    val_c += *v_c * full_eq_rx.index(*row) * full_eq_ry.index(*col);

                    worker_eq_tilde_rx_evals[i] = *full_eq_rx.index(*row);
                    worker_eq_tilde_ry_evals[i] = *full_eq_ry.index(*col);
                }
            }

            worker_eq_tilde_rx = DenseMultilinearExtension::from_evaluations_vec(
                pub_log_chunk_size,
                worker_eq_tilde_rx_evals,
            );
            worker_eq_tilde_ry = DenseMultilinearExtension::from_evaluations_vec(
                pub_log_chunk_size,
                worker_eq_tilde_ry_evals,
            );

            let response = (val_a, val_b, val_c);
            let size1 = send_responses(
                log,
                rank,
                &(stage.to_owned() + "round 3_a_b_c_eval"),
                &root_process,
                &response,
                0,
            );
            send_size = send_size + size1;

            pub_val_M = dense_scalar_prod(&v_msg[0], &pub_ipk.val_a)
                + dense_scalar_prod(&v_msg[1], &pub_ipk.val_b)
                + dense_scalar_prod(&v_msg[2], &pub_ipk.val_c);
            println!(
                "{rank} pub_val_M poly: {:?}",
                pub_val_M.evaluations[..5].to_vec()
            );

            let (size1, size2) = crate::mpi_snark::worker::mpi_poly_commit_worker(
                log,
                stage,
                size,
                rank,
                root_process,
                &vec![&worker_eq_tilde_rx, &worker_eq_tilde_ry],
                &pub_ipk.ck_index,
            );
            send_size = send_size + size1;
            recv_size = recv_size + size2;
        } else {
            let response = (
                E::ScalarField::zero(),
                E::ScalarField::zero(),
                E::ScalarField::zero(),
            );
            send_responses(
                log,
                rank,
                &(stage.to_owned() + "round 3_a_b_c_eval"),
                &root_process,
                &response,
                1,
            );

            let default_response = vec![
                Commitment::<E> {
                    nv: 0,
                    g_product: pub_ipk.ck_index.g,
                };
                2
            ];

            send_responses(log, rank, stage, &root_process, &default_response, 1);
        }

        let (size1, size2) = rss_batch_open_poly_worker(
            log,
            "third_round_open",
            size,
            rank,
            root_process,
            log_num_workers_per_party,
            pk.num_variables,
            &pk.ipk.ck_w.0,
            1,
            &[&pk.z],
            &r_y,
            E::ScalarField::one(),
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        (
            (worker_eq_tilde_rx, worker_eq_tilde_ry, pub_val_M),
            full_eq_ry,
            send_size,
            recv_size,
        )
    }

    pub fn prover_fifth_round<'a, C: 'a + Communicator>(
        pk: &DistributedRSSProverKey<E>,
        pub_ipk: &IndexProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
        pub_log_num_workers: usize,
        pub_start_eq: usize,
        pub_log_chunk_size: usize,
        full_eq_tilde_x: &DenseMultilinearExtension<E::ScalarField>,
        full_eq_tilde_y: &DenseMultilinearExtension<E::ScalarField>,
        pub_eq_tilde_rx: &DenseMultilinearExtension<E::ScalarField>,
        pub_eq_tilde_ry: &DenseMultilinearExtension<E::ScalarField>,
        pub_val_M: &DenseMultilinearExtension<E::ScalarField>,
        active: bool,
    ) -> (usize, usize) {
        if active {
            return crate::mpi_snark::worker::PublicProver::prover_fifth_round(
                &DistributedProverKey::<E> {
                    party_id: pk.party_id,
                    num_parties: pk.num_parties,
                    ipk: pk.ipk.clone(),
                    row: pk.row.clone(),
                    col: pk.col.clone(),
                    z: Vec::default(),
                    za: Vec::default(),
                    zb: Vec::default(),
                    zc: Vec::default(),
                    num_variables: pk.num_variables,
                },
                &pub_ipk,
                pk.num_variables,
                log,
                stage,
                size,
                rank,
                root_process,
                pub_log_num_workers,
                pub_start_eq,
                pub_log_chunk_size,
                full_eq_tilde_x,
                full_eq_tilde_y,
                pub_eq_tilde_rx,
                pub_eq_tilde_ry,
                pub_val_M,
            );
        } else {
            dummy_fifth_round(
                &pub_ipk,
                log,
                stage,
                size,
                rank,
                root_process,
                pub_log_num_workers,
            )
        }
        return (0, 0);
    }
}

pub fn rss_first_sumcheck_worker<'a, E: Pairing, C: 'a + Communicator, R: RngCore + FeedableRNG>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    za: &RssPoly<E>,
    zb: &RssPoly<E>,
    zc: &RssPoly<E>,
    eq: &DenseMultilinearExtension<E::ScalarField>,
    random_rng: &mut SSRandom<R>,
    log_num_workers_per_party: usize,
) -> (Vec<E::ScalarField>, usize, usize) {
    let mut send_msg_size = 0;
    let mut recv_msg_size = 0;

    let mut prover_state = RssSumcheck::<E>::first_sumcheck_init(za, zb, zc, eq);
    let num_vars = prover_state.num_vars;
    let mut verifier_msg = None;
    let mut final_point = Vec::new();

    for round in 0..num_vars {
        let prover_message = RssSumcheck::<E>::first_sumcheck_prove_round(
            &mut prover_state,
            &verifier_msg,
            random_rng,
        );
        let responses = prover_message.clone();
        let size1 = send_responses(
            log,
            rank,
            &(stage.to_owned() + "round " + &round.to_string()),
            &root_process,
            &responses,
            0,
        );
        send_msg_size = send_msg_size + size1;

        let (r, size1) = receive_requests(
            log,
            rank,
            &(stage.to_owned() + "round " + &round.to_string()),
            &root_process,
            0,
        );
        recv_msg_size = recv_msg_size + size1;

        verifier_msg = Some(VerifierMsg { randomness: r });
        final_point.push(r);
    }

    if log_num_workers_per_party != 0 {
        let _ = RssSumcheck::<E>::first_sumcheck_prove_round(
            &mut prover_state,
            &verifier_msg,
            random_rng,
        );

        let responses = (
            prover_state.secret_polys[0].share_0[0],
            prover_state.secret_polys[1].share_0[0],
            prover_state.secret_polys[2].share_0[0],
            prover_state.pub_polys[0][0],
        );
        let size1 = send_responses(
            log,
            rank,
            &(stage.to_owned() + "final_round "),
            &root_process,
            &responses,
            0,
        );
        send_msg_size = send_msg_size + size1;
    }

    let (final_point, size1) = receive_requests(
        log,
        rank,
        &(stage.to_owned() + "_final_point"),
        &root_process,
        0,
    );

    recv_msg_size = recv_msg_size + size1;

    (final_point, send_msg_size, recv_msg_size)
}

pub fn rss_second_sumcheck_worker<
    'a,
    E: Pairing,
    C: 'a + Communicator,
    R: RngCore + FeedableRNG,
>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    a_r: &DenseMultilinearExtension<E::ScalarField>,
    b_r: &DenseMultilinearExtension<E::ScalarField>,
    c_r: &DenseMultilinearExtension<E::ScalarField>,
    z: &RssPoly<E>,
    random_rng: &mut SSRandom<R>,
    log_num_workers_per_party: usize,
    coeffs: &Vec<E::ScalarField>,
) -> (Vec<E::ScalarField>, usize, usize) {
    let mut send_msg_size = 0;
    let mut recv_msg_size = 0;

    let mut prover_state = RssSumcheck::<E>::second_sumcheck_init(a_r, b_r, c_r, z, coeffs);
    let num_vars = prover_state.num_vars;
    let mut verifier_msg = None;
    let mut final_point = Vec::new();

    for round in 0..num_vars {
        let prover_message = RssSumcheck::<E>::second_sumcheck_prove_round(
            &mut prover_state,
            &verifier_msg,
            random_rng,
        );
        let responses = prover_message.clone();
        let size1 = send_responses(
            log,
            rank,
            &(stage.to_owned() + "round " + &round.to_string()),
            &root_process,
            &responses,
            0,
        );
        send_msg_size = send_msg_size + size1;

        let (r, size1) = receive_requests(
            log,
            rank,
            &(stage.to_owned() + "round " + &round.to_string()),
            &root_process,
            0,
        );
        recv_msg_size = recv_msg_size + size1;
        verifier_msg = Some(VerifierMsg { randomness: r });
        final_point.push(r);
    }

    if log_num_workers_per_party != 0 {
        let _ = RssSumcheck::<E>::second_sumcheck_prove_round(
            &mut prover_state,
            &verifier_msg,
            random_rng,
        );
        let responses = (
            prover_state.pub_polys[0][0],
            prover_state.pub_polys[1][0],
            prover_state.pub_polys[2][0],
            prover_state.secret_polys[0].share_0[0],
        );
        let size1 = send_responses(
            log,
            rank,
            &(stage.to_owned() + "final_round "),
            &root_process,
            &responses,
            0,
        );
        send_msg_size = send_msg_size + size1;
    }

    let (final_point, size1) = receive_requests(
        log,
        rank,
        &(stage.to_owned() + "_final_point"),
        &root_process,
        0,
    );
    recv_msg_size = recv_msg_size + size1;

    (final_point, send_msg_size, recv_msg_size)
}

pub fn rss_poly_commit_worker<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    polys: &Vec<&RssPoly<E>>,
    ck: &CommitterKey<E>,
) -> (usize, usize) {
    let mut res = Vec::new();

    for p in polys {
        let comm = MultilinearPC::commit(ck, &p.share_0);
        res.push(comm);
    }

    (send_responses(log, rank, stage, &root_process, &res, 0), 0)
}

pub fn rss_eval_poly_worker<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    polys: Vec<&RssPoly<E>>,
    num_poly: usize,
    final_point: &[E::ScalarField],
    num_vars: usize,
    log_num_workers_per_party: usize,
) -> (usize, usize) {
    let mut res = Vec::new();
    for p in polys {
        res.push(
            p.share_0
                .evaluate(&final_point[0..num_vars - log_num_workers_per_party].to_vec()),
        )
    }

    (
        send_responses(
            log,
            rank,
            &(stage.to_owned() + "poly_eval"),
            &root_process,
            &res,
            0,
        ),
        0,
    )
}

pub fn rss_batch_open_poly_worker<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    log_num_workers: usize,
    num_var: usize,
    ck: &CommitterKey<E>,
    num_comms: usize,
    rss_polys: &[&RssPoly<E>],
    point: &[E::ScalarField],
    eta: E::ScalarField,
) -> (usize, usize) {
    let mut polys = Vec::new();
    for i in 0..rss_polys.len() {
        polys.push(&rss_polys[i].share_0)
    }
    let agg_poly = aggregate_poly(eta, &polys[0..num_comms]);

    let (pf, r) = distributed_open(&ck, &agg_poly, &point[0..num_var - log_num_workers]);
    let mut evals = Vec::new();
    for p in polys.iter() {
        evals.push(p.evaluate(&point[0..num_var - log_num_workers].to_vec()));
    }

    let response = PartialProof {
        proofs: pf,
        val: r,
        evals,
    };

    (
        send_responses(
            log,
            rank,
            &(stage.to_owned() + "round 1"),
            &root_process,
            &response,
            0,
        ),
        0,
    )
}

pub fn dummy_sumcheck_worker<'a, F: Field, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    default_last_sumcheck_state: DistrbutedSumcheckProverState<F>,
    num_variables: usize,
    max_multiplicands: usize,
    log_num_workers: usize,
) {
    for round in 0..num_variables {
        let default_response = ProverMsg {
            evaluations: vec![F::zero(); max_multiplicands + 1],
        };

        send_responses(
            log,
            rank,
            &(stage.to_owned() + "round " + &round.to_string()),
            &root_process,
            &default_response,
            1,
        );

        let r: (F, usize) = receive_requests(
            log,
            rank,
            &(stage.to_owned() + "round " + &round.to_string()),
            &root_process,
            1,
        );
    }

    if log_num_workers != 0 {
        let default_response = default_last_sumcheck_state;
        send_responses(
            log,
            rank,
            &(stage.to_owned() + "final_round "),
            &root_process,
            &default_response,
            1,
        );
    }

    let _: (Vec<F>, usize) = receive_requests(
        log,
        rank,
        &(stage.to_owned() + "_final_point"),
        &root_process,
        1,
    );
}

pub fn dummy_batch_open_poly_worker<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    num_var: usize,
    num_poly: usize,
    g: E::G1Affine,
) {
    let default_response: PartialProof<E> = PartialProof {
        proofs: Proof {
            proofs: vec![g; num_var],
        },
        val: E::ScalarField::zero(),
        evals: vec![E::ScalarField::one(); num_poly],
    };
    send_responses(
        log,
        rank,
        &(stage.to_owned() + "round 1"),
        &root_process,
        &default_response,
        1,
    );
}

pub fn dummy_fifth_round<'a, E: Pairing, C: 'a + Communicator>(
    ipk: &IndexProverKey<E>,
    log: &mut Vec<String>,
    stage: &str,
    size: Count,
    rank: i32,
    root_process: &Process<'a, C>,
    log_num_workers: usize,
) {
    let start = start_timer_buf!(log, || format!("Coord: receiving stage1 requests"));
    let (v_msg, _): (E::ScalarField, usize) = receive_requests(log, rank, stage, &root_process, 1);
    end_timer_buf!(log, start);

    let ((x_r, x_c), _): ((E::ScalarField, E::ScalarField), usize) =
        receive_requests(log, rank, stage, &root_process, 1);

    let default_response = vec![
        Commitment::<E> {
            nv: 0,
            g_product: ipk.ck_index.g,
        };
        4
    ];

    send_responses(log, rank, "round 5", &root_process, &default_response, 1);

    let ((z, lambda), _): ((Vec<E::ScalarField>, E::ScalarField), usize) =
        receive_requests(log, rank, stage, &root_process, 1);

    let ((z, lambda), _): ((Vec<E::ScalarField>, E::ScalarField), usize) =
        receive_requests(log, rank, stage, &root_process, 1);

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

    default_sumcheck_poly_list(&lambda, 0, &mut q_polys);
    default_sumcheck_poly_list(&lambda, 0, &mut q_polys);

    let default_last_sumcheck_state = poly_list_to_prover_state(&q_polys);

    dummy_sumcheck_worker(
        log,
        "stage1_sumcheck",
        size,
        rank,
        &root_process,
        default_last_sumcheck_state,
        ipk.real_len_val.log_2(),
        3,
        log_num_workers,
    );

    let (eta, _): (E::ScalarField, usize) = receive_requests(log, rank, stage, &root_process, 1);

    dummy_batch_open_poly_worker::<E, C>(
        log,
        "stage5_poly_open",
        size,
        rank,
        &root_process,
        ipk.real_len_val.log_2(),
        15,
        ipk.ck_index.g,
    );
}
