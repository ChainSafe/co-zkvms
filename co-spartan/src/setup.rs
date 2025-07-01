use ark_ec::pairing::Pairing;
use ark_ff::{Field, Zero};
use ark_poly::DenseMultilinearExtension;

use crate::utils::{split_ck, split_poly, split_vec};
use spartan::{
    math::Math,
    IndexProverKey,
};

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
