use std::ops::Index;
use std::{collections::HashMap, rc::Rc};

use ark_crypto_primitives::sponge::{
    poseidon::{PoseidonConfig, PoseidonSponge},
    CryptographicSponge,
};
use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_poly::{Polynomial, DenseMultilinearExtension, MultilinearExtension};
use ark_poly_commit::multilinear_pc::data_structures::{
    Commitment, CommitterKey, Proof, UniversalParams, VerifierKey,
};

use crate::math::Math;
use crate::mpi_utils::DistrbutedSumcheckProverState;
use crate::snark::indexer::IndexProverKey;
use crate::R1CSInstance;
use crate::SparseMatPolynomial;
use ark_poly::SparseMultilinearExtension;
use ark_poly_commit::multilinear_pc::data_structures::EvaluationHyperCubeOnG1;
use ark_serialize::CanonicalSerialize;
use ark_std::{cfg_into_iter, cfg_iter_mut, test_rng};
use rand::RngCore;

use crate::snark::prover::ProverState;

use ark_ec::CurveGroup;
use ark_ec::VariableBaseMSM;
use ark_ff::{One, Zero};
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::data_structures::PolynomialInfo;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverState as SumcheckProverState;
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_std::cfg_iter;
use rand::Rng;
use rayon::prelude::*;

use crate::snark::prover::ProverMessage;

pub fn map_poly<F: Field, MapFn>(
    poly: &DenseMultilinearExtension<F>,
    f: MapFn,
) -> DenseMultilinearExtension<F>
where
    MapFn: Fn(&F) -> F + Sync + Send,
{
    DenseMultilinearExtension::from_evaluations_vec(
        poly.num_vars,
        cfg_iter!(poly.evaluations).map(f).collect::<Vec<_>>(),
    )
}

pub fn boost_degree<F: Field>(
    g: &DenseMultilinearExtension<F>,
    new_dim: usize,
) -> DenseMultilinearExtension<F> {
    assert!(new_dim >= g.num_vars);
    //The factor will be count many times when used in product
    let factor = two_pow_n::<F>(new_dim - g.num_vars).inverse().unwrap();
    // let factor = F::one();
    let scaled_g = map_poly(g, |x| factor * x);
    let evals = cfg_into_iter!(vec![
        scaled_g.evaluations.clone();
        1 << (new_dim - g.num_vars)
    ])
    .flatten()
    .collect();
    DenseMultilinearExtension::from_evaluations_vec(new_dim, evals)
}

#[inline]
pub fn two_pow_n<F: Field>(n: usize) -> F {
    let mut result = F::one();
    for _ in 0..n {
        result.double_in_place();
    }
    result
}

// TODO: generate via hyperplonk code
pub fn generate_eq<F: Field>(z: &[F]) -> DenseMultilinearExtension<F> {
    let mut evals = vec![F::one(); 1 << z.len()];
    evals[0] = z.iter().map(|z_i| F::one() - z_i).product();
    let mut z_inv: Vec<F> = z.iter().map(|z_i| F::one() - z_i).collect();
    ark_ff::batch_inversion(&mut z_inv);
    for i in 1usize..1 << z.len() {
        let prev: F = evals[i - (1 << i.trailing_zeros())];
        // let factor = z[i.trailing_zeros() as usize] / (F::one() - z[i.trailing_zeros() as usize]);
        let factor = z[i.trailing_zeros() as usize] * z_inv[i.trailing_zeros() as usize];
        evals[i] = prev * factor;
    }
    DenseMultilinearExtension::from_evaluations_vec(z.len(), evals)
}

pub fn generate_eq_point<F: Field>(z: &[F], point: usize) -> F {
    let mut x = point;
    let mut res = F::one();
    for i in 0..z.len() {
        if (x & 1) == 1 {
            res = res * z[i]
        } else {
            res = res * (F::one() - z[i])
        }
        x >>= 1;
    }
    res
}

pub fn partial_generate_eq<F: Field>(
    z: &[F],
    start: usize,
    log_chunk_size: usize,
) -> DenseMultilinearExtension<F> {
    let length = 1 << log_chunk_size;
    let mut evals = vec![F::one(); length];
    let mut z_inv: Vec<F> = z.iter().map(|z_i| F::one() - z_i).collect();
    ark_ff::batch_inversion(&mut z_inv);

    evals[0] = generate_eq_point(z, start);

    println!("log_chunk_size: {:?}", log_chunk_size);

    for i in 1usize..length {
        // if i.trailing_zeros() >= z.len() as u32 {
        //     continue;
        // }
        let prev: F = evals[i - (1 << i.trailing_zeros())];
        // let factor = z[i.trailing_zeros() as usize] / (F::one() - z[i.trailing_zeros() as usize]);
        let factor = z[i.trailing_zeros() as usize] * z_inv[i.trailing_zeros() as usize];
        evals[i] = prev * factor;
    }
    DenseMultilinearExtension::from_evaluations_vec(log_chunk_size, evals)
}

pub fn dense_scalar_prod<F: Field>(
    scalar: &F,
    dense: &DenseMultilinearExtension<F>,
) -> DenseMultilinearExtension<F> {
    let scalar = *scalar;
    let evaluations = cfg_iter!(dense.evaluations).map(|x| scalar * x).collect();
    DenseMultilinearExtension {
        evaluations,
        num_vars: dense.num_vars,
    }
}

pub fn eq_eval<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y)
        .map(|(x, y)| *x * y + (F::one() - x) * (F::one() - y))
        .product()
}

pub fn usize_to_vec<F: Field>(x: usize, num_variable: usize) -> Vec<F> {
    let mut v: Vec<F> = Vec::new();
    let mut t = x;
    for _ in 0..num_variable {
        if t & 1 == 1 {
            v.push(F::one());
        } else {
            v.push(F::zero());
        }
        t >>= 1;
    }
    v
}

pub fn feed_some<T, R>(rng: &mut R, element: Option<T>) -> Option<T>
where
    T: CanonicalSerialize,
    R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
{
    let temp = element.unwrap();
    assert!(rng.feed(&temp).is_ok());
    Some(temp)
}

pub fn feed_message<R, E>(rng: &mut R, mut message: ProverMessage<E>) -> ProverMessage<E>
where
    R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
    E: Pairing,
{
    if message.sumcheck_message.is_some() {
        message.sumcheck_message = feed_some(rng, message.sumcheck_message);
    }

    if message.group_message.is_some() {
        message.group_message = feed_some(rng, message.group_message);
    }

    if message.zk_proof_message.is_some() {
        message.zk_proof_message = feed_some(rng, message.zk_proof_message);
    }

    if message.proof_message.is_some() {
        message.proof_message = feed_some(rng, message.proof_message);
    }

    if message.commitment_message.is_some() {
        message.commitment_message = feed_some(rng, message.commitment_message);
    }

    if message.commitment_message_2.is_some() {
        message.commitment_message_2 = feed_some(rng, message.commitment_message_2);
    }
    if message.zksumcheck_message.is_some() {
        message.zksumcheck_message = feed_some(rng, message.zksumcheck_message);
    }
    if message.lookup_message.is_some() {
        message.lookup_message = feed_some(rng, message.lookup_message);
    }

    message
}

pub fn hash_tuple<F: Field>(v: &[usize], eq: &DenseMultilinearExtension<F>, v_msg: &F) -> Vec<F> {
    let mut result = cfg_iter!(v)
        .filter(|v_i| **v_i != usize::MAX)
        .map(|v_i| F::from(*v_i as u64) + *v_msg * eq[*v_i])
        .collect::<Vec<_>>();
    println!(
        "result: {:?} -> {} <- {:?}",
        result.len(),
        result.len().next_power_of_two() - result.len(),
        result[0]
    );

    for _ in 0..result.len().next_power_of_two() - result.len() {
        result.push(result[0]);
    }
    result
}

pub fn hash_usize<F: Field>(v: &[usize]) -> Vec<F> {
    let mut result = cfg_iter!(v)
        .map(|v_i| F::from(*v_i as u64))
        .collect::<Vec<_>>();

    for _ in 0..result.len().next_power_of_two() - result.len() {
        result.push(result[0]);
    }
    result
}

pub fn hash_usize_custom_pad<F: Field>(v: &[usize], pad_to_log_size: usize) -> Vec<F> {
    let mut result = cfg_iter!(v)
        .map(|v_i| F::from(*v_i as u64))
        .collect::<Vec<_>>();

    for _ in 0..(1 << pad_to_log_size) - result.len() {
        result.push(result[0]);
    }
    result
}

pub fn split_poly<F: Field>(
    polys: &DenseMultilinearExtension<F>,
    log_parties: usize,
) -> Vec<DenseMultilinearExtension<F>> {
    let nv = polys.num_vars - log_parties;
    let chunk_size = 1 << nv;
    let mut res = Vec::new();

    for i in 0..1 << log_parties {
        res.push(DenseMultilinearExtension {
            evaluations: polys.evaluations[chunk_size * i..chunk_size * (i + 1)].to_vec(),
            num_vars: nv,
        })
    }

    res
}

pub fn split_vec<T: Clone>(polys: &Vec<T>, log_parties: usize) -> Vec<Vec<T>> {
    let chunk_size = polys.len() / (1 << log_parties);
    let mut res = Vec::new();

    for i in 0..1 << log_parties {
        res.push(polys[chunk_size * i..chunk_size * (i + 1)].to_vec())
    }

    res
}

pub fn split_ck<E: Pairing>(
    ck: &CommitterKey<E>,
    log_parties: usize,
) -> (Vec<CommitterKey<E>>, CommitterKey<E>) {
    let nv = ck.nv - log_parties;
    let h = ck.h;
    let g = ck.g;
    let mut chunk_size = 1 << nv;
    let mut res = vec![
        CommitterKey {
            nv: nv,
            powers_of_g: Vec::new(),
            powers_of_h: Vec::new(),
            g: g.clone(),
            h: h.clone(),
        };
        1 << log_parties
    ];

    for i in 0..nv {
        for j in 0..1 << log_parties {
            res[j]
                .powers_of_g
                .push(ck.powers_of_g[i][chunk_size * j..chunk_size * (j + 1)].to_vec());
            res[j]
                .powers_of_h
                .push(ck.powers_of_h[i][chunk_size * j..chunk_size * (j + 1)].to_vec());
        }
        chunk_size /= 2;
    }

    let mut res_2 = CommitterKey {
        nv: log_parties,
        powers_of_g: Vec::new(),
        powers_of_h: Vec::new(),
        g: g.clone(),
        h: h.clone(),
    };

    for i in 0..log_parties {
        res_2.powers_of_g.push(ck.powers_of_g[nv + i].clone());
        res_2.powers_of_h.push(ck.powers_of_h[nv + i].clone());
    }

    (res, res_2)
}

pub fn aggregate_poly<F: Field>(
    eta: F,
    polys: &[&DenseMultilinearExtension<F>],
) -> DenseMultilinearExtension<F> {
    let mut vars = 0;
    for p in polys {
        if p.num_vars > vars {
            vars = p.num_vars;
        }
    }
    let mut evals = vec![F::zero(); 1 << vars];
    let mut x = F::one();
    for p in polys {
        cfg_iter_mut!(evals)
            .zip(&p.evaluations)
            .for_each(|(a, b)| *a += x * b);
        x *= eta
    }
    DenseMultilinearExtension {
        evaluations: evals,
        num_vars: vars,
    }
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
    println!("pk.row -> row_vec: {:?}", row_vec[0].len());
    println!("pk.col -> col_vec: {:?}", col_vec[0].len());
    // let row_vec_reindex = split_index_vec(&pk.row, log_parties);
    // let col_vec_reindex = split_index_vec(&pk.col, log_parties);
    let val_a_vec = split_poly(&pk.val_a, log_parties);
    let val_b_vec = split_poly(&pk.val_b, log_parties);
    let val_c_vec = split_poly(&pk.val_c, log_parties);
    let freq_r_vec = split_poly(&pk.freq_r, log_parties);
    let freq_c_vec = split_poly(&pk.freq_c, log_parties);

    let n_cols = pk.num_variables_val.pow2();
    println!("n_cols: {:?}", n_cols);
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

    println!("pk.ck_w.0.nv: {:?}", pk.ck_w.0.nv);
    println!("log_parties: {:?}", log_parties);
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

    println!("| num_parties: {:?}", num_parties);

    for i in 0..1 << log_parties {
        println!("--------------------------------");
        println!("party {:?}", i);
        println!(
            "bucket_rows_index[i].len(): {:?}",
            bucket_rows_index[i].len()
        );
        println!(
            "bucket_cols_index[i].len(): {:?}",
            bucket_cols_index[i].len()
        );
        println!(
            "bucket_val_a_index[i].len(): {:?}",
            bucket_val_a_index[i].len()
        );
        println!(
            "bucket_val_b_index[i].len(): {:?}",
            bucket_val_b_index[i].len()
        );
        println!(
            "bucket_val_c_index[i].len(): {:?}",
            bucket_val_c_index[i].len()
        );
        println!("--------------------------------");
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
        println!("ipk.real_len_val: {:?}", ipk.real_len_val);
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

pub fn split_sparse_poly<F: PrimeField>(
    poly: &SparseMatPolynomial<F>,
    log_parties: usize,
) -> Vec<SparseMatPolynomial<F>> {
    let mut res = Vec::new();
    let len = poly.M.len();
    let chunk_size = len / (1 << log_parties);
    for i in 0..(1 << log_parties) {
        let p = SparseMatPolynomial {
            num_vars_x: poly.num_vars_x,
            num_vars_y: poly.num_vars_y,
            M: poly.M[i * chunk_size..(i + 1) * chunk_size].to_vec(),
        };
        res.push(p);
    }

    res
}

pub fn split_r1cs<F: PrimeField>(
    r1cs: &R1CSInstance<F>,
    log_parties: usize,
) -> Vec<R1CSInstance<F>> {
    let mut res = Vec::new();
    let a_vec = split_sparse_poly(&r1cs.A, log_parties);
    let b_vec = split_sparse_poly(&r1cs.B, log_parties);
    let c_vec = split_sparse_poly(&r1cs.C, log_parties);
    for i in 0..1 << log_parties {
        let r = R1CSInstance {
            num_cons: r1cs.num_cons / (1 << log_parties),
            num_vars: r1cs.num_vars / (1 << log_parties),
            num_inputs: r1cs.num_inputs / (1 << log_parties),
            A: a_vec[i].clone(),
            B: b_vec[i].clone(),
            C: c_vec[i].clone(),
        };
        res.push(r);
    }

    res
}

pub fn aggregate_comm<E: Pairing>(eta: E::ScalarField, comms: &[Commitment<E>]) -> Commitment<E> {
    let mut res = comms[0].clone();
    let mut x = eta;
    for i in 1..comms.len() {
        res.g_product = (res.g_product + (comms[i].g_product * x)).into();
        x *= eta
    }
    res
}

pub fn eval_sparse_mle<F: Field>(mle: &SparseMultilinearExtension<F>, point: &[F]) -> F {
    let mut res = F::zero();
    for (&i, &v) in mle.evaluations.iter() {
        res += v * generate_eq_point(point, i)
    }
    res
}

pub fn combine_comm<E: Pairing>(comms: &[Commitment<E>]) -> Commitment<E> {
    let mut res = comms[0].clone();
    for i in 1..comms.len() {
        res.g_product = (res.g_product + (comms[i].g_product)).into();
    }
    res.nv = res.nv + comms.len().log_2();
    res
}

pub fn aggregate_proof<E: Pairing>(eta: E::ScalarField, pfs: &[Proof<E>]) -> Proof<E> {
    let mut res = pfs[0].clone();
    let mut x = eta;
    for i in 1..pfs.len() {
        for j in 0..res.proofs.len() {
            res.proofs[j] = (res.proofs[j] + pfs[i].proofs[j] * x).into();
        }
        x *= eta
    }
    res
}

pub fn merge_proof<E: Pairing>(pf1: &Proof<E>, pf2: &Proof<E>) -> Proof<E> {
    let mut res = pf1.proofs.clone();
    res.append(&mut pf2.proofs.clone());
    Proof { proofs: res }
}

pub fn aggregate_eval<F: Field>(eta: F, vals: &[F]) -> F {
    let mut res = vals[0];
    let mut x = eta;
    for i in 1..vals.len() {
        res = res + x * vals[i];
        x *= eta
    }
    res
}

#[test]
fn test_generate_eq() {
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use rand::thread_rng;

    const NUM_VARS: usize = 1;

    let z = <[Fr; NUM_VARS]>::rand(&mut thread_rng()).to_vec();
    let lh = generate_eq(&z);

    let f = DenseMultilinearExtension::from_evaluations_vec(
        NUM_VARS,
        <[Fr; 1 << NUM_VARS]>::rand(&mut thread_rng()).to_vec(),
    );

    let s: Fr = f
        .evaluations
        .iter()
        .zip(lh.evaluations.iter())
        .map(|(a, b)| a * b)
        .sum();

    assert_eq!(s, f.evaluate(&z));
}

pub fn generate_dumb_sponge<F: PrimeField>() -> PoseidonSponge<F> {
    PoseidonSponge::new(&poseidon_parameters_for_test())
}

pub fn poseidon_parameters_for_test<F: PrimeField>() -> PoseidonConfig<F> {
    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;

    let mds = vec![
        vec![F::one(), F::zero(), F::one()],
        vec![F::one(), F::one(), F::zero()],
        vec![F::zero(), F::one(), F::one()],
    ];

    let mut ark = Vec::new();
    let mut ark_rng = test_rng();

    for _ in 0..(full_rounds + partial_rounds) {
        let mut res = Vec::new();

        for _ in 0..3 {
            res.push(F::rand(&mut ark_rng));
        }
        ark.push(res);
    }
    PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, ark, 2, 1)
}

pub fn normalized_multiplicities<F: Field>(
    query: &DenseMultilinearExtension<F>,
    table: &DenseMultilinearExtension<F>,
) -> DenseMultilinearExtension<F> {
    let mut table_freqs: HashMap<F, F> = HashMap::new();
    for t in table.evaluations.iter() {
        if table_freqs.contains_key(t) {
            *table_freqs.get_mut(t).unwrap() += F::one();
        } else {
            table_freqs.insert(*t, F::one());
        }
    }

    let mut table_index = Vec::new();
    let mut table_freqs_inv = Vec::new();
    for t in table_freqs.iter() {
        table_index.push(*t.0);
        table_freqs_inv.push(*t.1);
    }
    ark_ff::batch_inversion(&mut table_freqs_inv);
    let mut table_invs: HashMap<F, F> = HashMap::new();
    for (idx, inv) in table_index.iter().zip(table_freqs_inv.iter()) {
        table_invs.insert(*idx, *inv);
    }

    let mut query_freqs: HashMap<F, F> = HashMap::new();
    for t in query.evaluations.iter() {
        if query_freqs.contains_key(t) {
            *query_freqs.get_mut(t).unwrap() += F::one();
        } else {
            query_freqs.insert(*t, F::one());
        }
    }
    let mut numerator = vec![F::zero(); table.evaluations.len()];
    for (i, e) in table.evaluations.iter().enumerate() {
        numerator[i] += query_freqs.get(e).unwrap_or(&F::zero());
    }

    for (i, mut n) in numerator.iter_mut().enumerate() {
        // *n /= table_freqs.get(&table[i]).unwrap();
        *n *= table_invs.get(&table[i]).unwrap();
    }
    DenseMultilinearExtension::from_evaluations_vec(table.num_vars, numerator)
}

pub fn distributed_open<E: Pairing>(
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
        let scalars: Vec<_> = (0..(1 << k))
            .map(|x| q[k][x >> 1].into_bigint()) // fine
            .collect();

        let pi_g =
            <E::G1 as VariableBaseMSM>::msm_bigint(&ck.powers_of_g[i], &scalars).into_affine(); // no need to move outside and partition
        proofs.push(pi_g);
    }

    (Proof { proofs }, r[0][0])
}

pub fn merge_list_of_distributed_poly<F: Field>(
    prover_states: Vec<SumcheckProverState<F>>,
    poly_info: PolynomialInfo,
    nv: usize,
    log_num_parties: usize,
) -> ListOfProductsOfPolynomials<F> {
    let mut merge_poly = ListOfProductsOfPolynomials::new(log_num_parties);
    merge_poly.max_multiplicands = poly_info.max_multiplicands;
    for j in 0..prover_states[0].list_of_products.len() {
        let mut evals: Vec<Vec<F>> = vec![Vec::new(); prover_states[0].list_of_products[j].1.len()];
        for i in 0..prover_states.len() {
            let (coeff, prods) = &prover_states[i].list_of_products[j];
            for k in 0..prods.len() {
                assert!(
                    prover_states[i].flattened_ml_extensions[prods[k]]
                        .evaluations
                        .len()
                        == 1
                );
                evals[k].push(prover_states[i].flattened_ml_extensions[prods[k]].evaluations[0]);
            }
        }
        let mut prod: Vec<Rc<DenseMultilinearExtension<F>>> = Vec::new();
        for e in &evals {
            prod.push(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                log_num_parties,
                e.clone(),
            )))
        }

        merge_poly.add_product(prod.into_iter(), prover_states[0].list_of_products[j].0);
    }

    merge_poly
}

pub fn produce_test_r1cs<F: PrimeField>(
    log_dim: usize,
    mut prng: &mut impl Rng,
) -> (R1CSInstance<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
    let mut z_vec = Vec::new();
    let mut a_vec = Vec::new();
    let mut b_vec = Vec::new();
    let mut c_vec = Vec::new();

    let mut za_vec = Vec::new();
    let mut zb_vec = Vec::new();
    let mut zc_vec = Vec::new();

    // use ark_std::time::{Duration, Instant};
    // let mut tot_time = Duration::ZERO;
    for i in 0..1 << log_dim {
        let a = F::rand(&mut prng);
        let b = F::rand(&mut prng);
        let z = F::rand(&mut prng);
        let c = a * b * z;
        z_vec.push(z);
        a_vec.push((i, i, a));
        b_vec.push((i, i, b));
        c_vec.push((i, i, c));

        // let time = Instant::now();
        za_vec.push(a * z);
        zb_vec.push(b * z);
        zc_vec.push(c * z);
        // let final_time = time.elapsed();
        // tot_time += final_time;
    }
    // println!(
    //     "Time to compute za_zb_zc_vec: {:?} for {:?} elements",
    //     tot_time,
    //     1 << log_dim
    // );
    // std::process::exit(0);

    (
        R1CSInstance::new(1 << log_dim, log_dim, 0, &a_vec, &b_vec, &c_vec),
        z_vec,
        za_vec,
        zb_vec,
        zc_vec,
    )
}

#[test]
fn test_serialize() {
    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
    use std::mem;

    let mut rng = StdRng::seed_from_u64(12);
    let file1 = Fr::rand(&mut rng);
    let file2 = Fr::rand(&mut rng);

    let mut buf = Vec::new();
    file1.serialize_uncompressed(&mut buf).unwrap();

    file2.serialize_uncompressed(&mut buf).unwrap();

    let buf_slice = buf.as_slice();

    let out1 = Fr::deserialize_uncompressed_unchecked(buf_slice).unwrap();
    let out2 = Fr::deserialize_uncompressed_unchecked(&buf_slice[mem::size_of::<Fr>()..]).unwrap();
    assert!(file1 == out1);
    assert!(file2 == out2);
}

#[macro_export]
macro_rules! start_timer_buf {
    ($buf:ident, $msg:expr) => {{
        // use std::time::Instant;

        // let msg = $msg();
        // let start_info = "Start:";

        // $buf.push(format!("{:8} {}", start_info, msg));
        // (msg.to_string(), Instant::now())
    }};
}

#[macro_export]
macro_rules! end_timer_buf {
    ($buf:ident, $time:expr) => {{
        // let time = $time.1;
        // let final_time = time.elapsed();

        // let end_info = "End:";
        // let message = format!("{}", $time.0);

        // $buf.push(format!(
        //     "{:8} {} {}μs",
        //     end_info,
        //     message,
        //     final_time.as_micros()
        // ));
    }};
}
