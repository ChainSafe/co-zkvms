use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly_commit::multilinear_pc::data_structures::{Commitment, Proof};

use crate::math::Math;

pub fn aggregate_eval<F: Field>(eta: F, vals: &[F]) -> F {
    let mut res = vals[0];
    let mut x = eta;
    for i in 1..vals.len() {
        res = res + x * vals[i];
        x *= eta
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

pub fn aggregate_comm_with_powers<E: Pairing>(
    powers: &[E::ScalarField],
    comms: &[Commitment<E>],
) -> Commitment<E> {
    let mut res = comms[0].clone();
    for i in 1..comms.len() {
        res.g_product = (res.g_product + (comms[i].g_product * powers[i])).into();
    }
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

pub fn combine_comm<E: Pairing>(comms: &[Commitment<E>]) -> Commitment<E> {
    let mut res = comms[0].clone();
    for i in 1..comms.len() {
        res.g_product = (res.g_product + (comms[i].g_product)).into();
    }
    res.nv = res.nv + comms.len().log_2();
    res
}
