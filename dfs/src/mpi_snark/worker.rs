use crate::end_timer_buf;
use crate::math::Math;
use crate::mpi_snark::coordinator::PartialProof;
use crate::mpi_utils::hash_tuple;
use crate::mpi_utils::merge_list_of_distributed_poly;
use crate::mpi_utils::obtain_distrbuted_sumcheck_prover_state;
use crate::mpi_utils::{
    gather_responses, receive_requests, scatter_requests, send_responses,
    DistrbutedSumcheckProverState,
};
use crate::snark::indexer::IndexProverKey;
use crate::snark::indexer::IndexVerifierKey;
use crate::snark::BatchOracleEval;
use crate::start_timer_buf;
use crate::subprotocols::loglookup::sumcheck_polynomial_list;
use crate::subprotocols::loglookup::LogLookupProof;
use crate::utils::aggregate_proof;
use crate::utils::boost_degree;
use crate::utils::combine_comm;
use crate::utils::dense_scalar_prod;
use crate::utils::generate_eq;
use crate::utils::merge_proof;
use crate::utils::partial_generate_eq;
use crate::utils::split_poly;
use crate::utils::{aggregate_poly, distributed_open};
use crate::R1CSInstance;
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_ff::{One, Zero};
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::verifier::VerifierMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::PolynomialInfo;
use ark_linear_sumcheck::ml_sumcheck::protocol::{IPForMLSumcheck, ListOfProductsOfPolynomials};
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::DenseMultilinearExtension;
use ark_poly::MultilinearExtension;
use ark_poly_commit::multilinear_pc::data_structures::{
    Commitment, CommitterKey, Proof, UniversalParams, VerifierKey,
};
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cmp::max;
use ark_std::marker::PhantomData;
use ark_std::rc::Rc;
use crossbeam::channel::Sender;
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::Process,
    Count,
};
use mpi::{request, traits::*};
use rand::RngCore;
use std::ops::Neg;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistributedProverKey<E: Pairing> {
    pub party_id: usize,
    pub num_parties: usize,
    pub col: Vec<usize>,
    pub row: Vec<usize>,
    pub ipk: IndexProverKey<E>,
    // pub r1cs: R1CSInstance<E::ScalarField>,
    pub z: Vec<E::ScalarField>,
    pub za: Vec<E::ScalarField>,
    pub zb: Vec<E::ScalarField>,
    pub zc: Vec<E::ScalarField>,
    pub num_variables: usize,
}

pub fn mpi_sumcheck_worker<'a, F: Field, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    distributed_q_polys: &ListOfProductsOfPolynomials<F>,
    log_num_workers: usize,
) -> (Vec<F>, usize, usize) {
    let mut send_size = 0;
    let mut recv_size = 0;

    let mut prover_state = IPForMLSumcheck::prover_init(&distributed_q_polys);
    let mut verifier_msg = None;
    let mut final_point = Vec::new();

    for round in 0..distributed_q_polys.num_variables {
        let prover_message = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
        let responses = prover_message.clone();
        let size1 = send_responses(
            log,
            rank,
            &(stage.to_owned() + "round " + &round.to_string()),
            &root_process,
            &responses,
            0,
        );
        send_size = send_size + size1;

        let (r, size1): (F, usize) = receive_requests(
            log,
            rank,
            &(stage.to_owned() + "round " + &round.to_string()),
            &root_process,
            0,
        );
        recv_size = recv_size + size1;
        verifier_msg = Some(VerifierMsg { randomness: r });
        final_point.push(r);
    }

    if log_num_workers != 0 {
        let _ = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
        let responses = obtain_distrbuted_sumcheck_prover_state(&prover_state);
        let size1 = send_responses(
            log,
            rank,
            &(stage.to_owned() + "final_round "),
            &root_process,
            &responses,
            0,
        );
        send_size = send_size + size1;
    }

    let (final_point, size1) = receive_requests(
        log,
        rank,
        &(stage.to_owned() + "_final_point"),
        &root_process,
        0,
    );
    recv_size = recv_size + size1;

    (final_point, send_size, recv_size)
}

pub fn mpi_poly_commit_worker<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    polys: &Vec<&DenseMultilinearExtension<E::ScalarField>>,
    ck: &CommitterKey<E>,
) -> (usize, usize) {
    let mut res = Vec::new();

    for p in polys {
        let comm = MultilinearPC::commit(ck, *p);
        res.push(comm);
    }

    (send_responses(log, rank, stage, &root_process, &res, 0), 0)
}

// /// Batch evluating polynomial
pub fn mpi_eval_poly_worker<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    polys: Vec<&DenseMultilinearExtension<E::ScalarField>>,
    num_poly: usize,
    final_point: &[E::ScalarField],
    num_vars: usize,
    log_num_workers: usize,
) -> (usize, usize) {
    let mut res = Vec::new();
    for p in polys {
        res.push(
            p.evaluate(&final_point[0..num_vars - log_num_workers])
                .unwrap(),
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

// /// Batch opening polynomial
pub fn mpi_batch_open_poly_worker<'a, E: Pairing, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    size: i32,
    rank: i32,
    root_process: &Process<'a, C>,
    log_num_workers: usize,
    num_var: usize,
    ck: &CommitterKey<E>,
    num_comms: usize,
    polys: &[&DenseMultilinearExtension<E::ScalarField>],
    point: &[E::ScalarField],
    eta: E::ScalarField,
) -> (usize, usize) {
    let agg_poly = aggregate_poly(eta, &polys[0..num_comms]);

    let (pf, r) = distributed_open(&ck, &agg_poly, &point[0..num_var - log_num_workers]);
    let mut evals = Vec::new();
    for p in polys.iter() {
        evals.push(p.evaluate(&point[0..num_var - log_num_workers]).unwrap());
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

pub struct PublicProver<E: Pairing> {
    _pairing: PhantomData<E>,
}

impl<E: Pairing> PublicProver<E> {
    pub fn prove<'a, C: 'a + Communicator>(
        pk: &DistributedProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
        start_eq: usize,
        log_chunk_size: usize,
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

        let (eq_rx, size1, size2) = Self::prover_second_round(
            pk,
            log,
            &(stage.to_owned() + "round 2"),
            size,
            rank,
            root_process,
            log_num_workers,
            start_eq,
            log_chunk_size,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let ((eq_ry, val_M), size1, size2) = Self::prover_third_round(
            pk,
            log,
            stage,
            size,
            rank,
            root_process,
            log_num_workers,
            start_eq,
            log_chunk_size,
            &eq_rx,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let (size1, size2) = Self::prover_fifth_round(
            pk,
            &pk.ipk,
            pk.num_variables,
            log,
            stage,
            size,
            rank,
            root_process,
            log_num_workers,
            start_eq,
            log_chunk_size,
            &DenseMultilinearExtension::default(), // TODO
            &DenseMultilinearExtension::default(),
            &eq_rx,
            &eq_ry,
            &val_M,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        (send_size, recv_size)
    }

    /// In round 1, prover only need to commit to witness w
    pub fn prover_first_round<'a, C: 'a + Communicator>(
        pk: &DistributedProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
    ) -> (usize, usize) {
        mpi_poly_commit_worker(
            log,
            stage,
            size,
            rank,
            root_process,
            &vec![&DenseMultilinearExtension {
                evaluations: (pk.z.clone()),
                num_vars: pk.ipk.padded_num_var,
            }],
            &pk.ipk.ck_w.0,
        )
    }

    pub fn prover_second_round<'a, C: 'a + Communicator>(
        pk: &DistributedProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
        start_eq: usize,
        log_chunk_size: usize,
    ) -> (DenseMultilinearExtension<E::ScalarField>, usize, usize) {
        let mut send_size = 0;
        let mut recv_size = 0;

        let start = start_timer_buf!(log, || format!("Coord: Generating stage0 requests"));
        let (v_msg, size1): (Vec<E::ScalarField>, usize) =
            receive_requests(log, rank, stage, &root_process, 0);
        recv_size = recv_size + size1;
        end_timer_buf!(log, start);

        let num_variables = pk.ipk.padded_num_var;

        let za = DenseMultilinearExtension::from_evaluations_vec(num_variables, pk.za.clone());
        let zb = DenseMultilinearExtension::from_evaluations_vec(num_variables, pk.zb.clone());
        let zc = DenseMultilinearExtension::from_evaluations_vec(num_variables, pk.zc.clone());

        let mut q_polys = ListOfProductsOfPolynomials::new(num_variables);

        let eq_func = partial_generate_eq(&v_msg, start_eq, log_chunk_size);
        let A_B_hat = vec![
            Rc::new(za.clone()),
            Rc::new(zb.clone()),
            Rc::new(eq_func.clone()),
        ];
        let C_hat = vec![Rc::new(zc.clone()), Rc::new(eq_func.clone())];

        q_polys.add_product(A_B_hat, E::ScalarField::one());
        q_polys.add_product(C_hat, E::ScalarField::one().neg());
        let sumcheck_state = IPForMLSumcheck::prover_init(&q_polys);

        let (final_point, size1, size2) = mpi_sumcheck_worker(
            log,
            stage,
            size,
            rank,
            root_process,
            &q_polys,
            log_num_workers,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let randomness = &final_point[0..num_variables];

        let (val_a, val_b, val_c) = (
            za.evaluate(randomness).unwrap(),
            zb.evaluate(randomness).unwrap(),
            zc.evaluate(randomness).unwrap(),
        );

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

        let eq_rx = partial_generate_eq(&final_point, start_eq, log_chunk_size);
        (eq_rx, send_size, recv_size)
    }

    pub fn prover_third_round<'a, C: 'a + Communicator>(
        pk: &DistributedProverKey<E>,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
        start_eq: usize,
        log_chunk_size: usize,
        eq_rx: &DenseMultilinearExtension<E::ScalarField>,
    ) -> (
        (
            DenseMultilinearExtension<E::ScalarField>,
            DenseMultilinearExtension<E::ScalarField>,
        ),
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
        let mut q_polys = ListOfProductsOfPolynomials::new(num_variables);

        let mut A_rx = vec![E::ScalarField::zero(); num_variables.pow2()];
        let mut B_rx = vec![E::ScalarField::zero(); num_variables.pow2()];
        let mut C_rx = vec![E::ScalarField::zero(); num_variables.pow2()];
        for i in 0..pk.ipk.real_len_val {
            A_rx[i] += pk.ipk.val_a[i] * eq_rx[i];
            B_rx[i] += pk.ipk.val_b[i] * eq_rx[i];
            C_rx[i] += pk.ipk.val_c[i] * eq_rx[i];
        }

        let z_poly = DenseMultilinearExtension::from_evaluations_vec(num_variables, pk.z.clone());
        let A_hat = vec![
            Rc::new(DenseMultilinearExtension {
                evaluations: (A_rx),
                num_vars: (num_variables),
            }),
            Rc::new(z_poly.clone()),
        ];
        let B_hat = vec![
            Rc::new(DenseMultilinearExtension {
                evaluations: (B_rx),
                num_vars: (num_variables),
            }),
            Rc::new(z_poly.clone()),
        ];
        let C_hat = vec![
            Rc::new(DenseMultilinearExtension {
                evaluations: (C_rx),
                num_vars: (num_variables),
            }),
            Rc::new(z_poly.clone()),
        ];

        q_polys.add_product(A_hat, v_msg[0]);
        q_polys.add_product(B_hat, v_msg[1]);
        q_polys.add_product(C_hat, v_msg[2]);

        let (final_point, size1, size2) = mpi_sumcheck_worker(
            log,
            stage,
            size,
            rank,
            root_process,
            &q_polys,
            log_num_workers,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;
        let r_y = final_point.to_vec();
        let eq_ry = partial_generate_eq(&r_y, start_eq, log_chunk_size);

        let (size1, size2) = mpi_eval_poly_worker::<E, C>(
            log,
            &(stage.to_owned() + "round 3_w_eval"),
            size,
            rank,
            &root_process,
            vec![&z_poly],
            1,
            &final_point,
            pk.num_variables,
            log_num_workers,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let mut val_a = E::ScalarField::zero();
        let mut val_b = E::ScalarField::zero();
        let mut val_c = E::ScalarField::zero();
        for (i, ((v_a, v_b), v_c)) in pk
            .ipk
            .val_a
            .evaluations
            .iter()
            .zip(pk.ipk.val_b.evaluations.iter())
            .zip(pk.ipk.val_c.evaluations.iter())
            .enumerate()
        {
            if i < pk.ipk.real_len_val {
                val_a += *v_a * eq_rx[i] * eq_ry[i];
                val_b += *v_b * eq_rx[i] * eq_ry[i];
                val_c += *v_c * eq_rx[i] * eq_ry[i]
            }
        }

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

        let val_M = dense_scalar_prod(&v_msg[0], &pk.ipk.val_a)
            + dense_scalar_prod(&v_msg[1], &pk.ipk.val_b)
            + dense_scalar_prod(&v_msg[2], &pk.ipk.val_c);

        let (size1, size2) = mpi_poly_commit_worker(
            log,
            stage,
            size,
            rank,
            root_process,
            &vec![eq_rx, &eq_ry],
            &pk.ipk.ck_index,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        let (size1, size2) = mpi_batch_open_poly_worker(
            log,
            "third_round_open",
            size,
            rank,
            root_process,
            log_num_workers,
            pk.num_variables,
            &pk.ipk.ck_w.0,
            1,
            &[&z_poly],
            &r_y,
            E::ScalarField::one(),
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;

        ((eq_ry, val_M), send_size, recv_size)
    }

    pub fn prover_fifth_round<'a, C: 'a + Communicator>(
        pk: &DistributedProverKey<E>,
        ipk: &IndexProverKey<E>,
        pk_num_variables: usize,
        log: &mut Vec<String>,
        stage: &str,
        size: Count,
        rank: i32,
        root_process: &Process<'a, C>,
        log_num_workers: usize,
        start_eq: usize,
        log_chunk_size: usize,
        full_eq_tilde_x: &DenseMultilinearExtension<E::ScalarField>,
        full_eq_tilde_y: &DenseMultilinearExtension<E::ScalarField>,
        eq_tilde_rx: &DenseMultilinearExtension<E::ScalarField>,
        eq_tilde_ry: &DenseMultilinearExtension<E::ScalarField>,
        val_M: &DenseMultilinearExtension<E::ScalarField>,
    ) -> (usize, usize) {
        let mut send_size = 0;
        let mut recv_size = 0;

        let start = start_timer_buf!(log, || format!("Coord: receiving stage1 requests"));
        let (v_msg, size1): (E::ScalarField, usize) =
            receive_requests(log, rank, stage, &root_process, 0);
        recv_size = recv_size + size1;
        end_timer_buf!(log, start);

        let q_num_vars = eq_tilde_rx.num_vars; //ipk.real_len_val.log_2();

        let mut q_row: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                q_num_vars,
                // `crate::utils::hash_tuple` method reindexes eq_tilde_rx based ipk.row
                // which we did in third round is eq_tilde_rx =? reindex(eq_rx)
                crate::utils::hash_tuple::<E::ScalarField>(
                    &ipk.row[..ipk.real_len_val],
                    full_eq_tilde_x,
                    &v_msg,
                ),
            );
        let first_row = *pk.row.iter().filter(|r| **r != usize::MAX).next().unwrap();
        let full_q_row_first = E::ScalarField::from(first_row as u64) + v_msg * full_eq_tilde_x[first_row];

        for i in ipk.real_len_val..q_num_vars.pow2() {
            q_row.evaluations[i] = full_q_row_first;
        }
    
        let mut q_col: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                q_num_vars,
                crate::utils::hash_tuple::<E::ScalarField>(
                    &ipk.col[..ipk.num_variables_val.pow2()],
                    full_eq_tilde_y,
                    &v_msg,
                ),
            );
        let first_col = *pk.col.iter().filter(|r| **r != usize::MAX).next().unwrap();
        let full_q_col_first = E::ScalarField::from(first_col as u64) + v_msg * full_eq_tilde_y[first_col];

        for i in ipk.real_len_val..q_num_vars.pow2() {
            q_col.evaluations[i] = full_q_col_first;
        }

        let domain = (start_eq..start_eq + (1 << log_chunk_size)).collect::<Vec<_>>();
        let t_row = DenseMultilinearExtension::from_evaluations_vec(
            eq_tilde_rx.num_vars,
            hash_tuple::<E::ScalarField>(&domain, eq_tilde_rx, &v_msg),
        );
      
        assert!(eq_tilde_rx.num_vars == ipk.num_variables_val);
        let t_col: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                ipk.num_variables_val,
                hash_tuple::<E::ScalarField>(&domain, eq_tilde_ry, &v_msg),
            );

        let mut q_polys = ListOfProductsOfPolynomials::new(max(q_num_vars, ipk.num_variables_val));

        // println!("--------------------------------");
        // println!("rank: {:?}", rank);
        // println!("eq_rx: {:?}", eq_tilde_rx.evaluations[..5].to_vec());
        // println!("eq_ry: {:?}", eq_tilde_ry.evaluations[..5].to_vec());
        // println!("val_M: {:?}", val_M.evaluations[..5].to_vec());
        // println!("--------------------------------");

        let prod = vec![
            Rc::new(eq_tilde_rx.clone()),
            Rc::new(eq_tilde_ry.clone()),
            Rc::new(val_M.clone()),
        ];
        q_polys.add_product(prod, E::ScalarField::one());

        let ((x_r, x_c), size1): ((E::ScalarField, E::ScalarField), usize) =
            receive_requests(log, rank, stage, &root_process, 0);
        recv_size = recv_size + size1;
        // println!("--------------------------------");
        // println!("rank: {:?}", rank);
        // println!("x_c: {:?}", x_c);
        // // println!("q_row: {:?}", q_row.evaluations[..5].to_vec());
        // // println!("t_row: {:?}", t_row.evaluations[..5].to_vec());
        // println!(
        //     "q_col: {:?}",
        //     q_col.evaluations[q_col.evaluations.len() - 5..].to_vec()
        // );
        // println!(
        //     "t_col: {:?}",
        //     t_col.evaluations[t_col.evaluations.len() - 5..].to_vec()
        // );
        // println!("--------------------------------");

        let lookup_pf_row = LogLookupProof::prove(&q_row, &t_row, &ipk.freq_r, &ipk.ck_index, &x_r);

        let lookup_pf_col = LogLookupProof::prove(&q_col, &t_col, &ipk.freq_c, &ipk.ck_index, &x_c);

        let responses = vec![
            lookup_pf_row.1[0].clone(),
            lookup_pf_row.1[1].clone(),
            lookup_pf_col.1[0].clone(),
            lookup_pf_col.1[1].clone(),
        ];
        let size1 = send_responses(log, rank, "round 5", &root_process, &responses, 0);
        send_size = send_size + size1;

        let ((z, lambda), size1): ((Vec<E::ScalarField>, E::ScalarField), usize) =
            receive_requests(log, rank, stage, &root_process, 0);
        recv_size = recv_size + size1;

        // println!("--------------------------------");
        // println!("rank: {:?}", rank);
        // println!("ipk.freq_r: {:?}", ipk.freq_r.evaluations[..5].to_vec());
        // println!("lookup_pf_row.0[0]: {:?}", lookup_pf_row.0[0].evaluations[..5].to_vec());
        // println!("lookup_pf_row.0[1]: {:?}", lookup_pf_row.0[1].evaluations[..5].to_vec());
        // println!("lookup_pf_row.2[0]: {:?}", lookup_pf_row.2[0].evaluations[..5].to_vec());
        // println!("lookup_pf_row.2[1]: {:?}", lookup_pf_row.2[1].evaluations[..5].to_vec());
        // println!("--------------------------------");

        sumcheck_polynomial_list(
            (lookup_pf_row.0[0].clone(), lookup_pf_row.0[1].clone()),
            (lookup_pf_row.2[0].clone(), lookup_pf_row.2[1].clone()),
            boost_degree(&ipk.freq_r.clone(), q_row.num_vars),
            // ipk.freq_r.clone(),
            q_row.num_vars - t_row.num_vars,
            &mut q_polys,
            &z,
            &lambda,
            start_eq,
            log_chunk_size,
        );

        let ((z, lambda), size1): ((Vec<E::ScalarField>, E::ScalarField), usize) =
            receive_requests(log, rank, stage, &root_process, 0);
        recv_size = recv_size + size1;

        sumcheck_polynomial_list(
            (lookup_pf_col.0[0].clone(), lookup_pf_col.0[1].clone()),
            (lookup_pf_col.2[0].clone(), lookup_pf_col.2[1].clone()),
            ipk.freq_c.clone(),
            q_col.num_vars - t_col.num_vars,
            &mut q_polys,
            &z,
            &lambda,
            start_eq,
            log_chunk_size,
        );

        // println!("--------------------------------");
        // println!("rank: {:?}", rank);
        // println!("ipk.freq_c: {:?}", ipk.freq_c.evaluations[..5].to_vec());
        // println!(
        //     "lookup_pf_col.0[0]: {:?}",
        //     lookup_pf_col.0[0].evaluations[lookup_pf_col.0[0].evaluations.len() - 5..].to_vec()
        // );
        // println!(
        //     "lookup_pf_col.0[1]: {:?}",
        //     lookup_pf_col.0[1].evaluations[lookup_pf_col.0[1].evaluations.len() - 5..].to_vec()
        // );
        // println!(
        //     "lookup_pf_col.2[0]: {:?}",
        //     lookup_pf_col.2[0].evaluations[lookup_pf_col.2[0].evaluations.len() - 5..].to_vec()
        // );
        // println!(
        //     "lookup_pf_col.2[1]: {:?}",
        //     lookup_pf_col.2[1].evaluations[lookup_pf_col.2[1].evaluations.len() - 5..].to_vec()
        // );
        // println!("--------------------------------");

        let (final_point, size1, size2) = mpi_sumcheck_worker(
            log,
            "stage1_sumcheck",
            size,
            rank,
            &root_process,
            &q_polys,
            log_num_workers,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;
        // println!("--------------------------------");
        // println!("rank: {:?}", rank);
        // println!("final_point: {:?}", final_point[..5].to_vec());
        // println!("--------------------------------");

        let (eta, size1): (E::ScalarField, usize) =
            receive_requests(log, rank, stage, &root_process, 0);
        recv_size = recv_size + size1;

        let (size1, size2) = mpi_batch_open_poly_worker(
            log,
            "stage5_poly_open",
            size,
            rank,
            &root_process,
            log_num_workers,
            pk_num_variables,
            &ipk.ck_index,
            9,
            &[
                &lookup_pf_row.0[0],
                &lookup_pf_row.0[1],
                &lookup_pf_col.0[0],
                &lookup_pf_col.0[1],
                &eq_tilde_rx,
                &eq_tilde_ry,
                &ipk.val_a,
                &ipk.val_b,
                &ipk.val_c,
                &ipk.freq_r,
                &q_row,
                &t_row,
                &ipk.freq_c,
                &q_col,
                &t_col,
            ],
            &final_point,
            eta,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;
        (send_size, recv_size)
    }
}
