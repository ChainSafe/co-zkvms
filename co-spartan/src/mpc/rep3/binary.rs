use ark_ff::{One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use mpc_core::protocols::rep3::PartyID;
use num_bigint::BigUint;
use rand::{CryptoRng, Rng};
use std::marker::PhantomData;

use crate::mpc::SSOpen;


