use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::RngCore;
use spartan::{math::Math, IndexProverKey, IndexVerifierKey, Indexer, R1CS, SRS};

use crate::{
    utils::{split_ck, split_poly, split_vec},
    Rep3ProverKey,
};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct CoordinatorKey<E: Pairing> {
    pub log_instance_size: usize,
    pub ipk: IndexProverKey<E>,
    pub pub_ipk: IndexProverKey<E>,
    pub ivk: IndexVerifierKey<E>,
}

pub fn setup_rep3<E: Pairing>(
    r1cs: &R1CS<E::ScalarField>,
    log_num_workers_per_party: usize,
    log_num_public_workers: usize,
    rng: &mut impl RngCore,
) -> (CoordinatorKey<E>, Vec<[Rep3ProverKey<E>; 3]>) {
    let log_instance_size = r1cs.log2_instance_size();
    let srs = SRS::<E, _>::generate_srs(log_instance_size + 2, 4, rng);

    let (pk, vk) = Indexer::index_for_prover_and_verifier(&r1cs, &srs);

    let mut prover_keys = Vec::new();

    let (ipk_vec, root_ipk) = split_ipk(&pk, log_num_workers_per_party);
    let (pub_ipk_vec, pub_root_ipk) = split_ipk(&pk, log_num_public_workers);

    let mut cnt = 0;
    for i in 0..1 << log_num_workers_per_party {
        let mut pk_vec = Vec::new();
        for j in 0..3 {
            let next = (j + 1) % 3;
            let seed_0 = format!("seed_{j}");
            let seed_1 = format!("seed_{next}");

            let pk = Rep3ProverKey {
                party_id: i,
                num_parties: log_num_public_workers,
                ipk: ipk_vec[i].clone(),
                pub_ipk: pub_ipk_vec[cnt].clone(),
                row: pk.row.clone(),
                col: pk.col.clone(),
                val_a: pk.val_a.clone(),
                val_b: pk.val_b.clone(),
                val_c: pk.val_c.clone(),
                num_variables: pk.num_variables_val,
                seed_0,
                seed_1,
            };

            if cnt < (1 << log_num_public_workers) - 1 {
                cnt = cnt + 1;
            }

            pk_vec.push(pk);
        }

        prover_keys.push(pk_vec.try_into().unwrap());
    }

    let pk = CoordinatorKey {
        log_instance_size,
        ipk: root_ipk,
        pub_ipk: pub_root_ipk,
        ivk: vk.clone(),
    };

    (pk, prover_keys)
}

pub fn split_ipk<E: Pairing>(
    pk: &IndexProverKey<E>,
    log_parties: usize,
) -> (Vec<IndexProverKey<E>>, IndexProverKey<E>) {
    let num_parties = 1 << log_parties;
    let chunk_size = 1 << (pk.num_variables_val - log_parties);

    let mut res = Vec::new();
    let row_vec = split_vec(&pk.row, log_parties);
    let col_vec = split_vec(&pk.col, log_parties);
    let val_a_vec = split_poly(&pk.val_a, log_parties);
    let val_b_vec = split_poly(&pk.val_b, log_parties);
    let val_c_vec = split_poly(&pk.val_c, log_parties);
    let freq_r_vec = split_poly(&pk.freq_r, log_parties);
    let freq_c_vec = split_poly(&pk.freq_c, log_parties);

    let n_cols = pk.num_variables_val.pow2();
    let mut bucket_cols_index: Vec<Vec<usize>> = vec![Vec::new(); num_parties];
    let mut bucket_rows_index: Vec<Vec<usize>> = vec![Vec::new(); num_parties];
    let mut bucket_val_a_index: Vec<Vec<E::ScalarField>> = vec![Vec::new(); num_parties];
    let mut bucket_val_b_index: Vec<Vec<E::ScalarField>> = vec![Vec::new(); num_parties];
    let mut bucket_val_c_index: Vec<Vec<E::ScalarField>> = vec![Vec::new(); num_parties];

    for i in 0..pk.real_len_val {
        // scan once
        let dest = pk.col[i] * num_parties / n_cols; // owner-computes-column rule
        bucket_cols_index[dest].push(pk.col[i]);
        bucket_rows_index[dest].push(pk.row[i]);
        bucket_val_a_index[dest].push(pk.val_a[i]);
        bucket_val_b_index[dest].push(pk.val_b[i]);
        bucket_val_c_index[dest].push(pk.val_c[i]);
    }

    let (ck_w_vec, merge_w_ck) = split_ck(&pk.ck_w.0, log_parties);
    let (ck_index_vec, merge_index_ck) = split_ck(&pk.ck_index, log_parties);

    let default_poly =
        DenseMultilinearExtension::from_evaluations_vec(0, vec![E::ScalarField::zero()]);
    let root_ipk = IndexProverKey {
        row: Vec::new(),
        col: Vec::new(),
        real_len_val: pk.real_len_val,
        padded_num_var: pk.padded_num_var,
        val_a: default_poly.clone(),
        val_b: default_poly.clone(),
        val_c: default_poly.clone(),
        freq_r: default_poly.clone(),
        freq_c: default_poly.clone(),
        num_variables_val: pk.num_variables_val,
        ck_w: (merge_w_ck, pk.ck_w.1.clone()),
        ck_index: merge_index_ck.clone(),
        ck_mask: pk.ck_mask.clone(),

        rows_indexed: Vec::new(),
        cols_indexed: Vec::new(),
        val_a_indexed: default_poly.clone(),
        val_b_indexed: default_poly.clone(),
        val_c_indexed: default_poly.clone(),
    };

    for i in 0..1 << log_parties {
        let ipk = IndexProverKey {
            row: row_vec[i].clone(),
            col: col_vec[i].clone(),
            real_len_val: real_chunk_size(i, chunk_size, pk.real_len_val),
            padded_num_var: pk.padded_num_var - log_parties,
            val_a: val_a_vec[i].clone(),
            val_b: val_b_vec[i].clone(),
            val_c: val_c_vec[i].clone(),
            freq_r: freq_r_vec[i].clone(),
            freq_c: freq_c_vec[i].clone(),
            num_variables_val: pk.num_variables_val - log_parties,
            ck_w: (ck_w_vec[i].clone(), pk.ck_w.1.clone()),
            ck_index: ck_index_vec[i].clone(),
            ck_mask: pk.ck_mask.clone(),

            rows_indexed: bucket_rows_index[i].clone(),
            cols_indexed: bucket_cols_index[i].clone(),
            val_a_indexed: DenseMultilinearExtension {
                evaluations: bucket_val_a_index[i].clone(),
                num_vars: bucket_val_a_index[i].len().log_2(),
            },
            val_b_indexed: DenseMultilinearExtension {
                evaluations: bucket_val_b_index[i].clone(),
                num_vars: bucket_val_b_index[i].len().log_2(),
            },
            val_c_indexed: DenseMultilinearExtension {
                evaluations: bucket_val_c_index[i].clone(),
                num_vars: bucket_val_c_index[i].len().log_2(),
            },
        };
        res.push(ipk);
    }

    (res, root_ipk)
}

fn real_chunk_size(i: usize, chunk_size: usize, n: usize) -> usize {
    debug_assert!(chunk_size.is_power_of_two());
    let start_index = i * chunk_size;

    if start_index >= n {
        0
    } else {
        (n - start_index).min(chunk_size)
    }
}
