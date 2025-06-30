use crate::math::Math;
use crate::mpc::rss::RssPoly;
use crate::mpc::rss::RssSumcheck;
use crate::mpc::rss::SSRandom;
use crate::mpi_snark::coordinator::PartialProof;
use crate::mpi_snark::worker::DistributedProverKey;
use crate::mpi_utils::receive_requests;
use crate::mpi_utils::send_responses;
use crate::mpi_utils::DistrbutedSumcheckProverState;
use crate::network::NetworkWorker;
use crate::snark::indexer::IndexProverKey;
use crate::subprotocols::loglookup::default_sumcheck_poly_list;
use crate::subprotocols::loglookup::poly_list_to_prover_state;
use crate::subprotocols::loglookup::LogLookupProof;
use crate::utils::aggregate_poly;
use crate::utils::dense_scalar_prod;
use crate::utils::distributed_open;
use crate::utils::generate_eq;
use crate::utils::partial_generate_eq;
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_ff::One;
use ark_ff::Zero;
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::verifier::VerifierMsg;
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_poly::{DenseMultilinearExtension, Polynomial};
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
use std::cmp::max;
use std::ops::Index;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3ProverKey<E: Pairing> {
    pub party_id: usize,
    pub num_parties: usize,
    pub ipk: IndexProverKey<E>,
    pub pub_ipk: IndexProverKey<E>,

    pub row: Vec<usize>,
    pub col: Vec<usize>,
    pub val_a: DenseMultilinearExtension<E::ScalarField>,
    pub val_b: DenseMultilinearExtension<E::ScalarField>,
    pub val_c: DenseMultilinearExtension<E::ScalarField>,
    pub z: RssPoly<E>,
    pub za: RssPoly<E>,
    pub zb: RssPoly<E>,
    pub zc: RssPoly<E>,
    pub num_variables: usize,
    pub seed_0: String,
    pub seed_1: String,
}

pub struct SpartanProverWorker<E: Pairing, N: NetworkWorker> {
    pub num_variables: usize,
    pub size: Count,
    pub log_chunk_size: usize,
    pub start_eq: usize,
    pub pub_log_chunk_size: usize,
    pub pub_start_eq: usize,

    _network: PhantomData<N>,
    _pairing: PhantomData<E>,
}

#[derive(Clone, Default)]
struct ProverState<E: Pairing> {
    pub r_x: Vec<E::ScalarField>,
    pub r_y: Vec<E::ScalarField>,
    pub eq_rx: Option<DenseMultilinearExtension<E::ScalarField>>,
    pub eq_ry: Option<DenseMultilinearExtension<E::ScalarField>>,
    pub eq_tilde_rx: Option<DenseMultilinearExtension<E::ScalarField>>,
    pub eq_tilde_ry: Option<DenseMultilinearExtension<E::ScalarField>>,
    pub eq_tilde_rx_chunk: Option<DenseMultilinearExtension<E::ScalarField>>,
    pub eq_tilde_ry_chunk: Option<DenseMultilinearExtension<E::ScalarField>>,
    pub val_m_poly_chunk: Option<DenseMultilinearExtension<E::ScalarField>>,
}

impl<E: Pairing, N: NetworkWorker> SpartanProverWorker<E, N> {
    pub fn prove<'a, C: 'a + Communicator, R: RngCore + FeedableRNG>(
        &mut self,
        pk: &Rep3ProverKey<E>,
        random_rng: &mut SSRandom<R>,
        active: bool,
        network: &mut N,
    ) -> (usize, usize) {
        let mut state = ProverState::default();

        self.first_round(&vec![&pk.z], &pk.ipk.ck_w.0, network);

        self.second_round(pk, &mut state, random_rng, network);

        self.third_round(pk, &mut state, random_rng, active, network);

        let (size1, size2) = Self::prover_fifth_round(
            pk,
            &mut state,
            &pub_eq_tilde_rx,
            &pub_eq_tilde_ry,
            &pub_val_M,
            active,
        );
        // send_size = send_size + size1;
        // recv_size = recv_size + size2;

        (0, 0)
    }

    pub fn first_round(&self, polys: &Vec<&RssPoly<E>>, ck: &CommitterKey<E>, network: &mut N) {
        poly_commit_worker(polys.iter().map(|p| &p.share_0), ck, network);
    }

    pub fn second_round<R: RngCore + FeedableRNG>(
        &self,
        pk: &Rep3ProverKey<E>,
        state: &mut ProverState<E>,
        random_rng: &mut SSRandom<R>,
        network: &mut N,
    ) {
        let v_msg: Vec<_> = network.receive_request();

        let num_variables = pk.ipk.padded_num_var;

        let eq_func = partial_generate_eq(&v_msg, self.start_eq, self.log_chunk_size);

        let final_point =
            rep3_first_sumcheck_worker(&pk.za, &pk.zb, &pk.zc, &eq_func, random_rng, network);

        let randomness = &final_point[0..num_variables].to_vec();

        let (val_a, val_b, val_c) = (
            pk.za.share_0.evaluate(&randomness),
            pk.zb.share_0.evaluate(&randomness),
            pk.zc.share_0.evaluate(&randomness),
        );

        let response = vec![val_a, val_b, val_c];
        network.send_response(response);

        state.eq_rx = Some(generate_eq(&final_point));
        state.r_x = final_point;
    }

    pub fn third_round<R: RngCore + FeedableRNG>(
        &self,
        pk: &Rep3ProverKey<E>,
        state: &mut ProverState<E>,
        random_rng: &mut SSRandom<R>,
        active: bool,
        network: &mut N,
    ) {
        let v_msg: Vec<_> = network.receive_request();
        let eq_rx = state.eq_rx.as_ref().unwrap();

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
            let v = eq_rx.index(row);

            A_rx[col] += pk.ipk.val_a_indexed[i] * v;
            B_rx[col] += pk.ipk.val_b_indexed[i] * v;
            C_rx[col] += pk.ipk.val_c_indexed[i] * v;
        }

        let final_point = rep3_second_sumcheck_worker(
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, A_rx),
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, B_rx),
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, C_rx),
            &pk.z,
            random_rng,
            &v_msg,
            network,
        );

        rep3_eval_poly_worker(vec![&pk.z], &final_point, 1, pk.num_variables, network);
        state.r_y = final_point.to_vec();
        state.eq_ry = Some(generate_eq(&final_point));
        let eq_ry = state.eq_ry.as_ref().unwrap();

        let chunk_size = self.pub_log_chunk_size.pow2();
        if active {
            let mut eq_tilde_rx_chunk_evals = vec![E::ScalarField::zero(); chunk_size];
            let mut eq_tilde_ry_chunk_evals = vec![E::ScalarField::zero(); chunk_size];

            let mut val_a = E::ScalarField::zero();
            let mut val_b = E::ScalarField::zero();
            let mut val_c = E::ScalarField::zero();

            for (i, ((((v_a, v_b), v_c), row), col)) in pk
                .pub_ipk
                .val_a
                .evaluations
                .iter()
                .zip(pk.pub_ipk.val_b.evaluations.iter())
                .zip(pk.pub_ipk.val_c.evaluations.iter())
                .zip(pk.pub_ipk.row.iter())
                .zip(pk.pub_ipk.col.iter())
                .enumerate()
            {
                if i < pk.pub_ipk.real_len_val {
                    val_a += *v_a * eq_rx.index(*row) * eq_ry.index(*col);
                    val_b += *v_b * eq_rx.index(*row) * eq_ry.index(*col);
                    val_c += *v_c * eq_rx.index(*row) * eq_ry.index(*col);

                    eq_tilde_rx_chunk_evals[i] = *eq_rx.index(*row);
                    eq_tilde_ry_chunk_evals[i] = *eq_ry.index(*col);
                }
            }

            state.eq_tilde_rx_chunk = Some(DenseMultilinearExtension::from_evaluations_vec(
                self.pub_log_chunk_size,
                eq_tilde_rx_chunk_evals,
            ));
            state.eq_tilde_ry_chunk = Some(DenseMultilinearExtension::from_evaluations_vec(
                self.pub_log_chunk_size,
                eq_tilde_ry_chunk_evals,
            ));
            let eq_tilde_rx_chunk = state.eq_tilde_rx_chunk.as_ref().unwrap();
            let eq_tilde_ry_chunk = state.eq_tilde_ry_chunk.as_ref().unwrap();

            let response = (val_a, val_b, val_c);
            network.send_response(response);

            let val_m_poly_chunk = dense_scalar_prod(&v_msg[0], &pk.pub_ipk.val_a)
                + dense_scalar_prod(&v_msg[1], &pk.pub_ipk.val_b)
                + dense_scalar_prod(&v_msg[2], &pk.pub_ipk.val_c);
            state.val_m_poly_chunk = Some(val_m_poly_chunk);

            poly_commit_worker(
                [eq_tilde_rx_chunk, eq_tilde_ry_chunk],
                &pk.pub_ipk.ck_index,
                network,
            );
        } else {
            let response = (
                E::ScalarField::zero(),
                E::ScalarField::zero(),
                E::ScalarField::zero(),
            );
            network.send_response(response);

            let default_response = vec![
                Commitment::<E> {
                    nv: 0,
                    g_product: pk.pub_ipk.ck_index.g
                };
                2
            ];
            network.send_response(default_response);
        }

        batch_open_poly_worker(
            self.num_variables,
            &pk.ipk.ck_w.0,
            1,
            &[&pk.z],
            &state.r_y,
            E::ScalarField::one(),
            network,
            true,
        );

        let mut eq_tilde_rx_evals = vec![E::ScalarField::zero(); pk.num_variables.pow2()];
        let mut eq_tilde_ry_evals = vec![E::ScalarField::zero(); pk.num_variables.pow2()];
        for i in 0..pk.num_variables.pow2() {
            if pk.row[i] != usize::MAX {
                eq_tilde_rx_evals[i] = *eq_rx.index(pk.row[i]);
            }
            if pk.col[i] != usize::MAX {
                eq_tilde_ry_evals[i] = *eq_ry.index(pk.col[i]);
            }
        }

        state.eq_tilde_rx = Some(DenseMultilinearExtension::from_evaluations_vec(
            pk.num_variables,
            eq_tilde_rx_evals,
        ));
        state.eq_tilde_ry = Some(DenseMultilinearExtension::from_evaluations_vec(
            pk.num_variables,
            eq_tilde_ry_evals,
        ));
    }

    pub fn prover_fifth_round(
        &self,
        pk: &Rep3ProverKey<E>,
        state: &mut ProverState<E>,
        active: bool,
        network: &mut N,
    ) {
        if !active {
            // return dummy_fifth_round(
            //     &pub_ipk,
            //     log,
            //     stage,
            //     size,
            //     rank,
            //     root_process,
            //     pub_log_num_workers,
            // );

            return todo!();
        }

        let start_eq = self.pub_start_eq;
        let log_chunk_size = self.pub_log_chunk_size;
        let eq_tilde_rx = state.eq_tilde_rx.as_ref().unwrap();
        let eq_tilde_ry = state.eq_tilde_ry.as_ref().unwrap();
        let val_m_poly_chunk = state.val_m_poly_chunk.as_ref().unwrap();

        let v_msg = network.receive_request();

        let q_num_vars = eq_tilde_rx.num_vars; //ipk.real_len_val.log_2();

        let mut q_row: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                q_num_vars,
                // `crate::utils::hash_tuple` method reindexes eq_tilde_rx based ipk.row
                // which we did in third round is eq_tilde_rx =? reindex(eq_rx)
                crate::utils::hash_tuple::<E::ScalarField>(
                    &pk.row[..pk.ipk.real_len_val],
                    eq_tilde_rx,
                    &v_msg,
                ),
            );
        let first_row = *pk.row.iter().filter(|r| **r != usize::MAX).next().unwrap();
        let full_q_row_first =
            E::ScalarField::from(first_row as u64) + v_msg * eq_tilde_rx[first_row];

        for i in pk.ipk.real_len_val..q_num_vars.pow2() {
            q_row.evaluations[i] = full_q_row_first;
        }

        let mut q_col: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                q_num_vars,
                crate::utils::hash_tuple::<E::ScalarField>(
                    &pk.col[..pk.ipk.num_variables_val.pow2()],
                    eq_tilde_ry,
                    &v_msg,
                ),
            );
        let first_col = *pk.col.iter().filter(|r| **r != usize::MAX).next().unwrap();
        let full_q_col_first =
            E::ScalarField::from(first_col as u64) + v_msg * eq_tilde_ry[first_col];

        for i in pk.ipk.real_len_val..q_num_vars.pow2() {
            q_col.evaluations[i] = full_q_col_first;
        }

        let domain = (start_eq..start_eq + (1 << log_chunk_size)).collect::<Vec<_>>();
        let t_row = DenseMultilinearExtension::from_evaluations_vec(
            eq_tilde_rx.num_vars,
            crate::mpi_utils::hash_tuple::<E::ScalarField>(&domain, eq_tilde_rx, &v_msg),
        );

        assert!(eq_tilde_rx.num_vars == pk.ipk.num_variables_val);
        let t_col: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                pk.ipk.num_variables_val,
                crate::mpi_utils::hash_tuple::<E::ScalarField>(&domain, eq_tilde_ry, &v_msg),
            );

        let mut q_polys = ListOfProductsOfPolynomials::new(max(q_num_vars, pk.ipk.num_variables_val));

        let prod = vec![
            Rc::new(eq_tilde_rx.clone()),
            Rc::new(eq_tilde_ry.clone()),
            Rc::new(val_m_poly_chunk.clone()),
        ];
        q_polys.add_product(prod, E::ScalarField::one());

        let (x_r, x_c) = network.receive_request();

        let lookup_pf_row = LogLookupProof::prove(&q_row, &t_row, &pk.ipk.freq_r, &pk.ipk.ck_index, &x_r);

        let lookup_pf_col = LogLookupProof::prove(&q_col, &t_col, &pk.ipk.freq_c, &pk.ipk.ck_index, &x_c);

        let responses = vec![
            lookup_pf_row.1[0].clone(),
            lookup_pf_row.1[1].clone(),
            lookup_pf_col.1[0].clone(),
            lookup_pf_col.1[1].clone(),
        ];
        network.send_response(responses);

        let (z, lambda) = network.receive_request();

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
    }
}

pub fn poly_commit_worker<'a, E: Pairing, N: NetworkWorker>(
    polys: impl IntoIterator<Item = &'a DenseMultilinearExtension<E::ScalarField>>,
    ck: &CommitterKey<E>,
    network: &mut N,
) {
    let mut res = Vec::new();

    for p in polys {
        let comm = MultilinearPC::commit(ck, p);
        res.push(comm);
    }

    network.send_response(res);
}

pub fn rep3_first_sumcheck_worker<E: Pairing, R: RngCore + FeedableRNG, N: NetworkWorker>(
    za: &RssPoly<E>,
    zb: &RssPoly<E>,
    zc: &RssPoly<E>,
    eq: &DenseMultilinearExtension<E::ScalarField>,
    random_rng: &mut SSRandom<R>,
    network: &mut N,
) -> Vec<E::ScalarField> {
    let mut prover_state = RssSumcheck::<E>::first_sumcheck_init(za, zb, zc, eq);
    let num_vars = prover_state.num_vars;
    let mut verifier_msg = None;
    let mut final_point = Vec::new();

    for _round in 0..num_vars {
        let prover_message = RssSumcheck::<E>::first_sumcheck_prove_round(
            &mut prover_state,
            &verifier_msg,
            random_rng,
        );
        network.send_response(prover_message.clone());

        let r = network.receive_request();

        verifier_msg = Some(VerifierMsg { randomness: r });
        final_point.push(r);
    }

    let _ =
        RssSumcheck::<E>::first_sumcheck_prove_round(&mut prover_state, &verifier_msg, random_rng);

    let response = (
        prover_state.secret_polys[0].share_0[0],
        prover_state.secret_polys[1].share_0[0],
        prover_state.secret_polys[2].share_0[0],
        prover_state.pub_polys[0][0],
    );
    network.send_response(response);

    let final_point = network.receive_request();

    final_point
}

pub fn rep3_second_sumcheck_worker<E: Pairing, R: RngCore + FeedableRNG, N: NetworkWorker>(
    a_r: &DenseMultilinearExtension<E::ScalarField>,
    b_r: &DenseMultilinearExtension<E::ScalarField>,
    c_r: &DenseMultilinearExtension<E::ScalarField>,
    z: &RssPoly<E>,
    random_rng: &mut SSRandom<R>,
    v_msg: &Vec<E::ScalarField>,
    network: &mut N,
) -> Vec<E::ScalarField> {
    let mut prover_state = RssSumcheck::<E>::second_sumcheck_init(a_r, b_r, c_r, z, v_msg);
    let num_vars = prover_state.num_vars;
    let mut verifier_msg = None;
    let mut final_point = Vec::new();

    for _round in 0..num_vars {
        let prover_message = RssSumcheck::<E>::second_sumcheck_prove_round(
            &mut prover_state,
            &verifier_msg,
            random_rng,
        );
        network.send_response(prover_message.clone());

        let r = network.receive_request();
        verifier_msg = Some(VerifierMsg { randomness: r });
        final_point.push(r);
    }

    let _ =
        RssSumcheck::<E>::second_sumcheck_prove_round(&mut prover_state, &verifier_msg, random_rng);
    let responses = (
        prover_state.pub_polys[0][0],
        prover_state.pub_polys[1][0],
        prover_state.pub_polys[2][0],
        prover_state.secret_polys[0].share_0[0],
    );
    network.send_response(responses);

    let final_point = network.receive_request();

    final_point
}

pub fn rep3_eval_poly_worker<E: Pairing, N: NetworkWorker>(
    polys: Vec<&RssPoly<E>>,
    final_point: &[E::ScalarField],
    num_poly: usize,
    num_vars: usize,
    network: &mut N,
) {
    let mut res = Vec::new();
    for p in polys {
        res.push(
            p.share_0
                .evaluate(&final_point[0..num_vars - network.log_num_workers_per_party()].to_vec()),
        )
    }

    network.send_response(res);
}

pub fn batch_open_poly_worker<E: Pairing, N: NetworkWorker>(
    num_var: usize,
    ck: &CommitterKey<E>,
    num_comms: usize,
    rss_polys: &[&RssPoly<E>],
    point: &[E::ScalarField],
    eta: E::ScalarField,
    network: &mut N,
    rep3: bool,
) {
    let log_num_workers = if rep3 {
        network.log_num_workers_per_party()
    } else {
        network.log_num_pub_workers()
    };
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

    network.send_response(response);
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
