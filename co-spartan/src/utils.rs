use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::multilinear_pc::data_structures::CommitterKey;
use ark_std::cfg_iter_mut;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

/// Pads the vector with 0 so that the number of elements in the vector is a
/// power of 2
pub fn pad_to_power_of_two<T: Default>(witness: &mut Vec<T>, log2n: usize) {
    let target_len = 1 << log2n;
    witness.reserve_exact(target_len - witness.len());
    while witness.len() < target_len {
        witness.push(T::default());
    }
}

/// Pads the vector with 0 so that the number of elements in the vector is a
/// power of 2
pub fn pad_to_next_power_of_two<T: Default>(witness: &mut Vec<T>) {
    let target_len = 1 << next_power_of_two(witness.len());
    witness.reserve_exact(target_len - witness.len());
    while witness.len() < target_len {
        witness.push(T::default());
    }
}

/// Calculates the degree of the next smallest power of two
pub fn next_power_of_two(n: usize) -> usize {
    let mut power = 1;
    let mut ans = 0;
    while power < n {
        power <<= 1;
        ans += 1;
    }
    ans
}
