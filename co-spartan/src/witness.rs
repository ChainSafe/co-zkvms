use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::RngCore;

use crate::{
    mpc::rep3::{generate_poly_shares_rss, Rep3DensePolynomial},
    utils::{pad_to_power_of_two, split_vec},
};

pub type Rep3WitnessShare<F: PrimeField> = Rep3DensePolynomial<F>;

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3R1CSWitnessShare<F: PrimeField> {
    pub z: Rep3DensePolynomial<F>,
    pub za: Rep3DensePolynomial<F>,
    pub zb: Rep3DensePolynomial<F>,
    pub zc: Rep3DensePolynomial<F>,
}

#[tracing::instrument(skip_all, name = "split_witness")]
pub fn split_witness<F: PrimeField>(
    mut z: Vec<F>,
    log_instance_size: usize,
    log_num_workers_per_party: usize,
    rng: &mut impl RngCore,
) -> Vec<[(usize, Rep3WitnessShare<F>); 3]> {
    pad_to_power_of_two(&mut z, log_instance_size);

    let mut z_vec = split_vec(&z, log_num_workers_per_party);

    let num_vars = log_instance_size - log_num_workers_per_party;

    let mut witness_shares = Vec::new();

    for i in 0..1 << log_num_workers_per_party {
        let z = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            std::mem::take(&mut z_vec[i]),
        );

        let z_shares = generate_poly_shares_rss(&z, rng);

        let mut wit_vec = Vec::new();
        for j in 0..3 {
            let worker_id = i * 3 + j;
            let next = (j + 1) % 3;
            let z = Rep3DensePolynomial::<F>::from_poly_shares(z_shares[j].clone(), z_shares[next].clone());
            wit_vec.push((worker_id, z));
        }

        witness_shares.push(wit_vec.try_into().unwrap());
    }

    witness_shares
}
