use std::{collections::HashMap, ops::Index, rc::Rc};

use ark_crypto_primitives::sponge::{
    poseidon::{PoseidonConfig, PoseidonSponge},
    CryptographicSponge,
};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_linear_sumcheck::{
    ml_sumcheck::{
        data_structures::{ListOfProductsOfPolynomials, PolynomialInfo},
        protocol::prover::ProverState as SumcheckProverState,
    },
    rng::FeedableRNG,
};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension, SparseMultilinearExtension};
use ark_poly_commit::multilinear_pc::data_structures::{Commitment, CommitterKey, Proof};
use ark_serialize::CanonicalSerialize;
use ark_std::{cfg_into_iter, cfg_iter, cfg_iter_mut, test_rng};
use rand::{Rng, RngCore};
use rayon::prelude::*;

use crate::{math::Math};

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

pub fn hash_usize<F: Field>(v: &[usize]) -> Vec<F> {
    let mut result = cfg_iter!(v)
        .map(|v_i| F::from(*v_i as u64))
        .collect::<Vec<_>>();

    for _ in 0..result.len().next_power_of_two() - result.len() {
        result.push(result[0]);
    }
    result
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

#[test]
fn test_serialize() {
    use std::mem;

    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

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
