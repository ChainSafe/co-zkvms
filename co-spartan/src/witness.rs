use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::RngCore;

use crate::{
    mpc::rep3::{generate_poly_shares_rss, Rep3Poly},
    utils::{pad_to_power_of_two, split_vec},
};

pub type Rep3WitnessShare<E> = Rep3Poly<E>;

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3R1CSWitnessShare<E: Pairing> {
    pub z: Rep3Poly<E>,
    pub za: Rep3Poly<E>,
    pub zb: Rep3Poly<E>,
    pub zc: Rep3Poly<E>,
}

#[tracing::instrument(skip_all, name = "split_witness")]
pub fn split_witness<E: Pairing>(
    mut z: Vec<E::ScalarField>,
    log_instance_size: usize,
    log_num_workers_per_party: usize,
    rng: &mut impl RngCore,
) -> Vec<[(usize, Rep3WitnessShare<E>); 3]> {
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
            let z = Rep3Poly::<E>::new(j, z_shares[j].clone(), z_shares[next].clone());
            wit_vec.push((worker_id, z));
        }

        witness_shares.push(wit_vec.try_into().unwrap());
    }

    witness_shares
}
