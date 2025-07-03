use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::RngCore;
use spartan::R1CS;

use crate::{
    mpc::rep3::{generate_poly_shares_rss, RssPoly},
    utils::{pad_to_power_of_two, split_vec},
};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct WitnessShare<E: Pairing> {
    pub worker_id: usize,
    pub z: RssPoly<E>,
    pub za: RssPoly<E>,
    pub zb: RssPoly<E>,
    pub zc: RssPoly<E>,
}

#[tracing::instrument(skip_all, name = "split_witness")]
pub fn split_witness<E: Pairing>(
    r1cs: &R1CS<E::ScalarField>,
    mut z: Vec<E::ScalarField>,
    log_num_workers_per_party: usize,
    rng: &mut impl RngCore,
) -> Vec<[WitnessShare<E>; 3]> {
    let log_instance_size = r1cs.log2_instance_size();

    let mut za = r1cs.a() * &z;
    let mut zb = r1cs.b() * &z;
    let mut zc = r1cs.c() * &z;

    pad_to_power_of_two(&mut z, log_instance_size);
    pad_to_power_of_two(&mut za, log_instance_size);
    pad_to_power_of_two(&mut zb, log_instance_size);
    pad_to_power_of_two(&mut zc, log_instance_size);

    let mut z_vec = split_vec(&z, log_num_workers_per_party);
    let mut za_vec = split_vec(&za, log_num_workers_per_party);
    let mut zb_vec = split_vec(&zb, log_num_workers_per_party);
    let mut zc_vec = split_vec(&zc, log_num_workers_per_party);

    let num_vars = log_instance_size - log_num_workers_per_party;

    let mut witness_shares = Vec::new();

    for i in 0..1 << log_num_workers_per_party {
        let z = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            std::mem::take(&mut z_vec[i]),
        );
        let za = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            std::mem::take(&mut za_vec[i]),
        );
        let zb = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            std::mem::take(&mut zb_vec[i]),
        );
        let zc = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            std::mem::take(&mut zc_vec[i]),
        );

        let z_shares = generate_poly_shares_rss(&z, rng);
        let za_shares = generate_poly_shares_rss(&za, rng);
        let zb_shares = generate_poly_shares_rss(&zb, rng);
        let zc_shares = generate_poly_shares_rss(&zc, rng);

        let mut wit_vec = Vec::new();
        for j in 0..3 {
            let worker_id = i * 3 + j;
            let next = (j + 1) % 3;
            let z = RssPoly::<E>::new(j, z_shares[j].clone(), z_shares[next].clone());
            let za = RssPoly::<E>::new(j, za_shares[j].clone(), za_shares[next].clone());
            let zb = RssPoly::<E>::new(j, zb_shares[j].clone(), zb_shares[next].clone());
            let zc = RssPoly::<E>::new(j, zc_shares[j].clone(), zc_shares[next].clone());

            wit_vec.push(WitnessShare {
                worker_id,
                z,
                za,
                zb,
                zc,
            });
        }

        witness_shares.push(wit_vec.try_into().unwrap());
    }

    witness_shares
}
