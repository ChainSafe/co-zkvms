use std::{cmp::max, iter, ops::Index};

use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_linear_sumcheck::{
    ml_sumcheck::{
        data_structures::ListOfProductsOfPolynomials,
        protocol::{prover::ProverMsg, verifier::VerifierMsg, IPForMLSumcheck},
    },
    rng::FeedableRNG,
};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension, Polynomial};
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, CommitterKey, Proof},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter, marker::PhantomData, rc::Rc};
use color_eyre::eyre::Context;
use color_eyre::eyre::Result;
use mpc_core::protocols::rep3::{
    arithmetic::Rep3PrimeFieldShare, poly::Rep3DensePolynomial, rngs::SSRandom,
};
use mpc_net::mpc_star::MpcStarNetWorker;
use rand::RngCore;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use snarks_core::math::Math;
use spartan::{
    logup::LogLookupProof,
    utils::{boost_degree, dense_scalar_prod, generate_eq, partial_generate_eq},
    IndexProverKey,
};

use crate::{
    sumcheck::{
        append_sumcheck_polys, default_sumcheck_poly_list, obtain_distrbuted_sumcheck_prover_state,
        poly_list_to_prover_state, DistrbutedSumcheckProverState, Rep3Sumcheck,
    },
    utils::aggregate_poly,
    witness::{Rep3R1CSWitnessShare, Rep3WitnessShare},
};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
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
    pub num_variables: usize,
    pub seed_0: String,
    pub seed_1: String,
}

pub struct SpartanProverWorker<E: Pairing, N: MpcStarNetWorker> {
    pub log_chunk_size: usize,
    pub start_eq: usize,
    pub pub_log_chunk_size: usize,
    pub pub_start_eq: usize,
    _network: PhantomData<N>,
    _pairing: PhantomData<E>,
}

#[derive(Clone)]
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

impl<E: Pairing> Default for ProverState<E> {
    fn default() -> Self {
        Self {
            r_x: vec![],
            r_y: vec![],
            eq_rx: None,
            eq_ry: None,
            eq_tilde_rx: None,
            eq_tilde_ry: None,
            eq_tilde_rx_chunk: None,
            eq_tilde_ry_chunk: None,
            val_m_poly_chunk: None,
        }
    }
}

impl<E: Pairing, N: MpcStarNetWorker> SpartanProverWorker<E, N> {
    pub fn new(
        log_chunk_size: usize,
        start_eq: usize,
        pub_log_chunk_size: usize,
        pub_start_eq: usize,
    ) -> Self {
        Self {
            log_chunk_size,
            start_eq,
            pub_log_chunk_size,
            pub_start_eq,
            _network: PhantomData,
            _pairing: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "SpartanProverWorker::prove")]
    pub fn prove<R: RngCore + FeedableRNG>(
        &mut self,
        pk: &Rep3ProverKey<E>,
        z: Rep3WitnessShare<E::ScalarField>,
        random_rng: &mut SSRandom<R>,
        active: bool,
        network: &mut N,
    ) -> Result<()> {
        let mut state = ProverState::default();

        let witness_share = self.zero_round(pk, &z);

        self.first_round(&vec![&witness_share.z], &pk.ipk.ck_w.0, network)
            .context("while running first round")?;

        self.second_round(pk, &witness_share, &mut state, random_rng, network)
            .context("while running second round")?;

        self.third_round(pk, &witness_share, &mut state, random_rng, active, network)
            .context("while running third round")?;

        if active {
            self.fourth_round(pk, &mut state, network)
                .context("while running fourth round")?;
        } else {
            // todo fork network to avoid dummy fourth round
            dummy_fourth_round(&pk.pub_ipk, network);
        }

        Ok(())
    }

    // Compute Az, Bz, Cz
    #[tracing::instrument(skip_all, name = "SpartanProverWorker::zero_round")]
    fn zero_round(
        &self,
        pk: &Rep3ProverKey<E>,
        z: &Rep3WitnessShare<E::ScalarField>,
    ) -> Rep3R1CSWitnessShare<E::ScalarField> {
        let chunk_size = pk.ipk.num_variables_val.exp2();
        let mut za = vec![Rep3PrimeFieldShare::zero(); chunk_size];
        let mut zb = vec![Rep3PrimeFieldShare::zero(); chunk_size];
        let mut zc = vec![Rep3PrimeFieldShare::zero(); chunk_size];

        let c_start = pk.party_id * pk.num_variables.exp2() / pk.num_parties;

        assert_eq!(pk.ipk.cols_indexed.len(), pk.ipk.rows_indexed.len());

        for i in 0..pk.ipk.cols_indexed.len() {
            let row = pk.ipk.rows_indexed[i] - c_start;
            let col = pk.ipk.cols_indexed[i] - c_start;
            let z_share = z.get_share_by_idx(col);
            za[row] += z_share * pk.ipk.val_a_indexed[i];
            zb[row] += z_share * pk.ipk.val_b_indexed[i];
            zc[row] += z_share * pk.ipk.val_c_indexed[i];
        }

        Rep3R1CSWitnessShare {
            z: z.clone(),
            za: Rep3DensePolynomial::new_with_vars(za, pk.ipk.num_variables_val),
            zb: Rep3DensePolynomial::new_with_vars(zb, pk.ipk.num_variables_val),
            zc: Rep3DensePolynomial::new_with_vars(zc, pk.ipk.num_variables_val),
        }
    }

    #[tracing::instrument(skip_all, name = "SpartanProverWorker::first_round")]
    fn first_round(
        &self,
        polys: &Vec<&Rep3DensePolynomial<E::ScalarField>>,
        ck: &CommitterKey<E>,
        network: &mut N,
    ) -> Result<()> {
        poly_commit_worker(polys.iter().map(|p| &p.share_0), ck, network)
            .context("while committing polynomials")
    }

    #[tracing::instrument(skip_all, name = "SpartanProverWorker::second_round")]
    fn second_round<R: RngCore + FeedableRNG>(
        &self,
        pk: &Rep3ProverKey<E>,
        witness_share: &Rep3R1CSWitnessShare<E::ScalarField>,
        state: &mut ProverState<E>,
        random_rng: &mut SSRandom<R>,
        network: &mut N,
    ) -> Result<()> {
        let v_msg: Vec<_> = network.receive_request()?;

        let num_variables = pk.ipk.padded_num_var;

        let eq_func = partial_generate_eq(&v_msg, self.start_eq, self.log_chunk_size);

        let final_point = rep3_first_sumcheck_worker(
            &witness_share.za,
            &witness_share.zb,
            &witness_share.zc,
            &eq_func,
            random_rng,
            network,
        )
        .context("while running first sumcheck")?;

        let randomness = &final_point[0..num_variables].to_vec();

        let (val_a, val_b, val_c) = (
            witness_share.za.share_0.evaluate(&randomness),
            witness_share.zb.share_0.evaluate(&randomness),
            witness_share.zc.share_0.evaluate(&randomness),
        );

        let response = vec![val_a, val_b, val_c];
        network.send_response(response)?;

        state.eq_rx = Some(generate_eq(&final_point));
        state.r_x = final_point;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "SpartanProverWorker::third_round")]
    fn third_round<R: RngCore + FeedableRNG>(
        &self,
        pk: &Rep3ProverKey<E>,
        witness_share: &Rep3R1CSWitnessShare<E::ScalarField>,
        state: &mut ProverState<E>,
        random_rng: &mut SSRandom<R>,
        active: bool,
        network: &mut N,
    ) -> Result<()> {
        let v_msg: Vec<_> = network.receive_request().context("while receiving v_msg")?;
        let eq_rx = state.eq_rx.as_ref().unwrap();

        let num_variables = pk.ipk.padded_num_var;
        let instance_size = pk.num_variables.exp2();
        let chunk_size = pk.ipk.num_variables_val.exp2();
        let c_start = pk.party_id * instance_size / pk.num_parties;

        let mut a_rx = vec![E::ScalarField::zero(); chunk_size];
        let mut b_rx = vec![E::ScalarField::zero(); chunk_size];
        let mut c_rx = vec![E::ScalarField::zero(); chunk_size];

        for i in 0..pk.ipk.cols_indexed.len() {
            let col = pk.ipk.cols_indexed[i] - c_start; // local offset 0..range_len-1
            let row = pk.ipk.rows_indexed[i];
            let eq = eq_rx.index(row);

            a_rx[col] += pk.ipk.val_a_indexed[i] * eq;
            b_rx[col] += pk.ipk.val_b_indexed[i] * eq;
            c_rx[col] += pk.ipk.val_c_indexed[i] * eq;
        }

        let final_point = rep3_second_sumcheck_worker(
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, a_rx),
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, b_rx),
            &DenseMultilinearExtension::from_evaluations_vec(num_variables, c_rx),
            &witness_share.z,
            random_rng,
            &v_msg,
            network,
        )
        .context("while running second sumcheck")?;

        rep3_eval_poly_worker(
            vec![&witness_share.z],
            &final_point,
            pk.num_variables,
            network,
        )
        .context("while running eval poly")?;
        state.r_y = final_point.to_vec();
        state.eq_ry = Some(generate_eq(&final_point));
        let eq_ry = state.eq_ry.as_ref().unwrap();

        let chunk_size = self.pub_log_chunk_size.exp2();
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
                .zip(pk.pub_ipk.rows.iter())
                .zip(pk.pub_ipk.cols.iter())
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
            network.send_response(response)?;

            let val_m_poly_chunk = dense_scalar_prod(&v_msg[0], &pk.pub_ipk.val_a)
                + dense_scalar_prod(&v_msg[1], &pk.pub_ipk.val_b)
                + dense_scalar_prod(&v_msg[2], &pk.pub_ipk.val_c);
            state.val_m_poly_chunk = Some(val_m_poly_chunk);

            poly_commit_worker(
                [eq_tilde_rx_chunk, eq_tilde_ry_chunk],
                &pk.pub_ipk.ck_index,
                network,
            )
            .context("while committing polynomials")?;
        } else {
            let response = (
                E::ScalarField::zero(),
                E::ScalarField::zero(),
                E::ScalarField::zero(),
            );
            network.send_response(response)?;

            let default_response = vec![
                Commitment::<E> {
                    nv: 0,
                    g_product: pk.pub_ipk.ck_index.g
                };
                2
            ];
            network.send_response(default_response)?;
        }

        distributed_batch_open_poly_worker(
            iter::once(&witness_share.z).map(|p| &p.share_0),
            &pk.ipk.ck_w.0,
            &state.r_y,
            E::ScalarField::one(),
            1,
            pk.num_variables,
            network.log_num_workers_per_party(),
            network,
        )
        .context("while running batch open poly")?;

        let mut eq_tilde_rx_evals = vec![E::ScalarField::zero(); pk.num_variables.exp2()];
        let mut eq_tilde_ry_evals = vec![E::ScalarField::zero(); pk.num_variables.exp2()];
        for i in 0..pk.num_variables.exp2() {
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

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "SpartanProverWorker::fourth_round")]
    fn fourth_round(
        &self,
        pk: &Rep3ProverKey<E>,
        state: &mut ProverState<E>,
        network: &mut N,
    ) -> Result<()> {
        let start_eq = self.pub_start_eq;
        let log_chunk_size = self.pub_log_chunk_size;
        let eq_tilde_rx = state.eq_tilde_rx.as_ref().unwrap();
        let eq_tilde_ry = state.eq_tilde_ry.as_ref().unwrap();
        let eq_tilde_rx_chunk = state.eq_tilde_rx_chunk.as_ref().unwrap();
        let eq_tilde_ry_chunk = state.eq_tilde_ry_chunk.as_ref().unwrap();
        let val_m_poly_chunk = state.val_m_poly_chunk.as_ref().unwrap();

        let v_msg = network.receive_request().context("while receiving v_msg")?;

        let q_num_vars = pk.pub_ipk.num_variables_val;

        let mut q_row: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                q_num_vars,
                // `hash_tuple` method reindexes eq_tilde_rx based ipk.row
                // which we did in third round is eq_tilde_rx =? reindex(eq_rx)
                hash_tuple::<E::ScalarField>(
                    &pk.pub_ipk.rows[..pk.pub_ipk.real_len_val],
                    eq_tilde_rx,
                    &v_msg,
                ),
            );
        let first_row = *pk.row.iter().filter(|r| **r != usize::MAX).next().unwrap();
        let full_q_row_first =
            E::ScalarField::from(first_row as u64) + v_msg * eq_tilde_rx[first_row];

        for i in pk.pub_ipk.real_len_val..q_num_vars.exp2() {
            q_row.evaluations[i] = full_q_row_first;
        }

        let mut q_col: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                q_num_vars,
                hash_tuple::<E::ScalarField>(
                    &pk.pub_ipk.cols[..pk.pub_ipk.num_variables_val.exp2()],
                    eq_tilde_ry,
                    &v_msg,
                ),
            );

        // TODO: we just need first_col value in pk -- can avoid storing entire col, eq_tilde_ry vector?
        let first_col = *pk.col.iter().filter(|r| **r != usize::MAX).next().unwrap();
        let full_q_col_first =
            E::ScalarField::from(first_col as u64) + v_msg * eq_tilde_ry[first_col];

        for i in pk.pub_ipk.real_len_val..q_num_vars.exp2() {
            q_col.evaluations[i] = full_q_col_first;
        }

        let domain = (start_eq..start_eq + (1 << log_chunk_size)).collect::<Vec<_>>();
        let t_row = DenseMultilinearExtension::from_evaluations_vec(
            pk.pub_ipk.num_variables_val,
            hash_tuple::<E::ScalarField>(&domain, eq_tilde_rx, &v_msg),
        );

        assert!(eq_tilde_rx_chunk.num_vars == pk.pub_ipk.num_variables_val);
        let t_col: DenseMultilinearExtension<<E as Pairing>::ScalarField> =
            DenseMultilinearExtension::from_evaluations_vec(
                pk.pub_ipk.num_variables_val,
                hash_tuple::<E::ScalarField>(&domain, eq_tilde_ry, &v_msg),
            );

        let mut q_polys =
            ListOfProductsOfPolynomials::new(max(q_num_vars, pk.pub_ipk.num_variables_val));

        let prod = vec![
            Rc::new(eq_tilde_rx_chunk.clone()),
            Rc::new(eq_tilde_ry_chunk.clone()),
            Rc::new(val_m_poly_chunk.clone()),
        ];
        q_polys.add_product(prod, E::ScalarField::one());

        let (x_r, x_c) = network
            .receive_request()
            .context("while receiving x_r and x_c")?;

        let lookup_pf_row = LogLookupProof::prove(
            &q_row,
            &t_row,
            &pk.pub_ipk.freq_r,
            &pk.pub_ipk.ck_index,
            &x_r,
        );

        let lookup_pf_col = LogLookupProof::prove(
            &q_col,
            &t_col,
            &pk.pub_ipk.freq_c,
            &pk.pub_ipk.ck_index,
            &x_c,
        );

        let responses = vec![
            lookup_pf_row.1[0].clone(),
            lookup_pf_row.1[1].clone(),
            lookup_pf_col.1[0].clone(),
            lookup_pf_col.1[1].clone(),
        ];

        network.send_response(responses);

        let (z, lambda) = network
            .receive_request()
            .context("while receiving first z and lambda")?;

        append_sumcheck_polys(
            (lookup_pf_row.0[0].clone(), lookup_pf_row.0[1].clone()),
            (lookup_pf_row.2[0].clone(), lookup_pf_row.2[1].clone()),
            boost_degree(&pk.pub_ipk.freq_r.clone(), q_row.num_vars),
            q_row.num_vars - t_row.num_vars,
            &mut q_polys,
            &z,
            &lambda,
            start_eq,
            log_chunk_size,
        );

        let (z, lambda) = network
            .receive_request()
            .context("while receiving second z and lambda")?;

        append_sumcheck_polys(
            (lookup_pf_col.0[0].clone(), lookup_pf_col.0[1].clone()),
            (lookup_pf_col.2[0].clone(), lookup_pf_col.2[1].clone()),
            pk.pub_ipk.freq_c.clone(),
            q_col.num_vars - t_col.num_vars,
            &mut q_polys,
            &z,
            &lambda,
            start_eq,
            log_chunk_size,
        );

        let final_point = distributed_sumcheck_worker(&q_polys, network)
            .context("while running distributed sumcheck")?;

        let eta = network.receive_request().context("while receiving eta")?;

        distributed_batch_open_poly_worker(
            [
                &lookup_pf_row.0[0],
                &lookup_pf_row.0[1],
                &lookup_pf_col.0[0],
                &lookup_pf_col.0[1],
                &eq_tilde_rx_chunk,
                &eq_tilde_ry_chunk,
                &pk.pub_ipk.val_a,
                &pk.pub_ipk.val_b,
                &pk.pub_ipk.val_c,
                &pk.pub_ipk.freq_r,
                &q_row,
                &t_row,
                &pk.pub_ipk.freq_c,
                &q_col,
                &t_col,
            ],
            &pk.pub_ipk.ck_index,
            &final_point,
            eta,
            9,
            pk.num_variables,
            network.log_num_pub_workers(),
            network,
        )
        .context("while batch opening polynomials")?;

        Ok(())
    }
}

pub fn poly_commit_worker<'a, E: Pairing, N: MpcStarNetWorker>(
    polys: impl IntoIterator<Item = &'a DenseMultilinearExtension<E::ScalarField>>,
    ck: &CommitterKey<E>,
    network: &mut N,
) -> Result<()> {
    let mut res = Vec::new();

    for p in polys {
        let comm = MultilinearPC::commit(ck, p);
        res.push(comm);
    }

    network.send_response(res)
}

#[tracing::instrument(skip_all, name = "rep3_first_sumcheck_worker")]
pub fn rep3_first_sumcheck_worker<F: PrimeField, R: RngCore + FeedableRNG, N: MpcStarNetWorker>(
    za: &Rep3DensePolynomial<F>,
    zb: &Rep3DensePolynomial<F>,
    zc: &Rep3DensePolynomial<F>,
    eq: &DenseMultilinearExtension<F>,
    random_rng: &mut SSRandom<R>,
    network: &mut N,
) -> Result<Vec<F>> {
    let mut prover_state = Rep3Sumcheck::<F>::first_sumcheck_init(za, zb, zc, eq);
    let num_vars = prover_state.num_vars;
    let mut verifier_msg = None;
    let mut final_point = Vec::new();

    for _round in 0..num_vars {
        let prover_message = Rep3Sumcheck::<F>::first_sumcheck_prove_round(
            &mut prover_state,
            &verifier_msg,
            random_rng,
        );
        network
            .send_response(prover_message.clone())
            .context("while sending prover message")?;
        let r = network
            .receive_request()
            .context("while receiving randomness")?;

        verifier_msg = Some(VerifierMsg { randomness: r });
        final_point.push(r);
    }

    let _ =
        Rep3Sumcheck::<F>::first_sumcheck_prove_round(&mut prover_state, &verifier_msg, random_rng);

    let response = (
        prover_state.secret_polys[0].share_0[0],
        prover_state.secret_polys[1].share_0[0],
        prover_state.secret_polys[2].share_0[0],
        prover_state.pub_polys[0][0],
    );
    network.send_response(response)?;

    let final_point = network
        .receive_request()
        .context("while receiving final point")?;

    Ok(final_point)
}

#[tracing::instrument(skip_all, name = "rep3_second_sumcheck_worker")]
pub fn rep3_second_sumcheck_worker<F: PrimeField, R: RngCore + FeedableRNG, N: MpcStarNetWorker>(
    a_r: &DenseMultilinearExtension<F>,
    b_r: &DenseMultilinearExtension<F>,
    c_r: &DenseMultilinearExtension<F>,
    z: &Rep3DensePolynomial<F>,
    random_rng: &mut SSRandom<R>,
    v_msg: &Vec<F>,
    network: &mut N,
) -> Result<Vec<F>> {
    let mut prover_state = Rep3Sumcheck::<F>::second_sumcheck_init(a_r, b_r, c_r, z, v_msg);
    let num_vars = prover_state.num_vars;
    let mut verifier_msg = None;
    let mut final_point = Vec::new();

    for _round in 0..num_vars {
        let prover_message = Rep3Sumcheck::<F>::second_sumcheck_prove_round(
            &mut prover_state,
            &verifier_msg,
            random_rng,
        );
        network
            .send_response(prover_message.clone())
            .context("while sending prover message")?;

        let r = network
            .receive_request()
            .context("while receiving randomness")?;
        verifier_msg = Some(VerifierMsg { randomness: r });
        final_point.push(r);
    }

    let _ = Rep3Sumcheck::<F>::second_sumcheck_prove_round(
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
    network.send_response(responses)?;

    let final_point = network
        .receive_request()
        .context("while receiving final point")?;

    Ok(final_point)
}

#[tracing::instrument(skip_all, name = "distributed_sumcheck_worker")]
pub fn distributed_sumcheck_worker<F: Field, N: MpcStarNetWorker>(
    distributed_q_polys: &ListOfProductsOfPolynomials<F>,
    network: &mut N,
) -> Result<Vec<F>> {
    let mut prover_state = IPForMLSumcheck::prover_init(&distributed_q_polys);
    let mut verifier_msg = None;
    let mut final_point = Vec::new();

    for _round in 0..distributed_q_polys.num_variables {
        let prover_message = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
        network
            .send_response(prover_message.clone())
            .context("while sending prover message")?;
        let r = network
            .receive_request()
            .context("while receiving randomness")?;
        verifier_msg = Some(VerifierMsg { randomness: r });
        final_point.push(r);
    }

    let _ = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
    let responses = obtain_distrbuted_sumcheck_prover_state(&prover_state);
    network.send_response(responses)?;

    let final_point = network
        .receive_request()
        .context("while receiving final point")?;

    Ok(final_point)
}

#[tracing::instrument(skip_all, name = "rep3_eval_poly_worker")]
pub fn rep3_eval_poly_worker<F: PrimeField, N: MpcStarNetWorker>(
    polys: Vec<&Rep3DensePolynomial<F>>,
    final_point: &[F],
    num_vars: usize,
    network: &mut N,
) -> Result<()> {
    let mut res = Vec::new();
    for p in polys {
        res.push(
            p.share_0
                .evaluate(&final_point[0..num_vars - network.log_num_workers_per_party()].to_vec()),
        )
    }

    network.send_response(res)
}

#[tracing::instrument(skip_all, name = "distributed_batch_open_poly_worker")]
pub fn distributed_batch_open_poly_worker<'a, E: Pairing, N: MpcStarNetWorker>(
    polys: impl IntoIterator<Item = &'a DenseMultilinearExtension<E::ScalarField>>,
    ck: &CommitterKey<E>,
    point: &[E::ScalarField],
    eta: E::ScalarField,
    num_comms: usize,
    num_var: usize,
    log_num_workers: usize,
    network: &mut N,
) -> Result<()> {
    let polys = polys.into_iter().collect::<Vec<_>>();

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

    network.send_response(response)
}

fn distributed_open<E: Pairing>(
    ck: &CommitterKey<E>,
    polynomial: &impl MultilinearExtension<E::ScalarField>,
    point: &[E::ScalarField],
) -> (Proof<E>, E::ScalarField) {
    assert_eq!(polynomial.num_vars(), ck.nv, "Invalid size of polynomial");
    let nv = polynomial.num_vars();
    let mut r: Vec<Vec<E::ScalarField>> = (0..nv + 1).map(|_| Vec::new()).collect();
    let mut q: Vec<Vec<E::ScalarField>> = (0..nv + 1).map(|_| Vec::new()).collect();

    r[nv] = polynomial.to_evaluations();

    let mut proofs = Vec::new();
    for i in 0..nv {
        let k = nv - i;
        let point_at_k = point[i];
        q[k] = (0..(1 << (k - 1)))
            .map(|_| E::ScalarField::zero())
            .collect();
        r[k - 1] = (0..(1 << (k - 1)))
            .map(|_| E::ScalarField::zero())
            .collect();
        for b in 0..(1 << (k - 1)) {
            q[k][b] = r[k][(b << 1) + 1] - &r[k][b << 1];
            r[k - 1][b] = r[k][b << 1] * &(E::ScalarField::one() - &point_at_k)
                + &(r[k][(b << 1) + 1] * &point_at_k);
        }
        let scalars: Vec<_> = (0..(1 << k)).map(|x| q[k][x >> 1].into_bigint()).collect();

        let pi_g =
            <E::G1 as VariableBaseMSM>::msm_bigint(&ck.powers_of_g[i], &scalars).into_affine(); // no need to move outside and partition
        proofs.push(pi_g);
    }

    (Proof { proofs }, r[0][0])
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
/// proof of opening
pub struct PartialProof<E: Pairing> {
    /// Evaluation of quotients
    pub proofs: Proof<E>,
    pub val: E::ScalarField,
    pub evals: Vec<E::ScalarField>,
}

impl<E: Pairing> PartialProof<E> {
    pub fn combine_partial_proof(pfs: &[Self]) -> Self {
        let mut res = pfs[0].clone();
        for i in 1..pfs.len() {
            for j in 0..res.proofs.proofs.len() {
                res.proofs.proofs[j] = (res.proofs.proofs[j] + pfs[i].proofs.proofs[j]).into();
            }
            res.val = res.val + pfs[i].val;
            for j in 0..res.evals.len() {
                res.evals[j] = res.evals[j] + pfs[i].evals[j];
            }
        }
        res
    }
}

fn hash_tuple<F: Field>(v: &[usize], eq: &DenseMultilinearExtension<F>, v_msg: &F) -> Vec<F> {
    let mut result = cfg_iter!(v)
        .filter(|v_i| **v_i != usize::MAX)
        .map(|v_i| F::from(*v_i as u64) + *v_msg * eq[*v_i])
        .collect::<Vec<_>>();

    for _ in 0..result.len().next_power_of_two() - result.len() {
        result.push(result[0]);
    }
    result
}

fn dummy_sumcheck_worker<F: Field, N: MpcStarNetWorker>(
    default_last_sumcheck_state: DistrbutedSumcheckProverState<F>,
    num_variables: usize,
    max_multiplicands: usize,
    network: &mut N,
) {
    for _ in 0..num_variables {
        let default_response = ProverMsg {
            evaluations: vec![F::zero(); max_multiplicands + 1],
        };

        network.send_response(default_response);

        let _: F = network.receive_request().unwrap();
    }

    let default_response = default_last_sumcheck_state;
    network.send_response(default_response);

    let _: Vec<F> = network.receive_request().unwrap();
}

fn dummy_batch_open_poly_worker<'a, E: Pairing, N: MpcStarNetWorker>(
    num_var: usize,
    num_poly: usize,
    g: E::G1Affine,
    network: &mut N,
) {
    let default_response: PartialProof<E> = PartialProof {
        proofs: Proof {
            proofs: vec![g; num_var],
        },
        val: E::ScalarField::zero(),
        evals: vec![E::ScalarField::one(); num_poly],
    };
    network.send_response(default_response);
}

fn dummy_fourth_round<'a, E: Pairing, N: MpcStarNetWorker>(
    ipk: &IndexProverKey<E>,
    network: &mut N,
) {
    let _v_msg: E::ScalarField = network.receive_request().unwrap();

    let (_x_r, _x_c): (E::ScalarField, E::ScalarField) = network.receive_request().unwrap();

    let default_response = vec![
        Commitment::<E> {
            nv: 0,
            g_product: ipk.ck_index.g,
        };
        4
    ];

    network.send_response(default_response);

    let (_z, _lambda): (Vec<E::ScalarField>, E::ScalarField) = network.receive_request().unwrap();
    let (_z, lambda): (Vec<E::ScalarField>, E::ScalarField) = network.receive_request().unwrap();

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
        default_last_sumcheck_state,
        ipk.real_len_val.log_2(),
        3,
        network,
    );

    let _eta: E::ScalarField = network.receive_request().unwrap();

    dummy_batch_open_poly_worker::<E, N>(ipk.real_len_val.log_2(), 15, ipk.ck_index.g, network);
}
