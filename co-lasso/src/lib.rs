pub mod grand_product;
pub mod lasso;
pub mod memory_checking;
pub mod subtables;
pub mod sumcheck;
mod utils;
mod poly;
mod witness_solver;

pub use witness_solver::{Rep3LassoWitnessSolver, Rep3LassoPolynomials};

use ark_ff::{BigInteger, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_chunks, cfg_into_iter, cfg_iter};
use color_eyre::eyre::Context;
use itertools::{multizip, Itertools};
use jolt_core::poly::{dense_mlpoly::DensePolynomial, field::JoltField};
use mpc_core::protocols::{
    rep3::{
        self, arithmetic,
        network::{IoContext, Rep3Network},
        Rep3BigUintShare, Rep3PrimeFieldShare,
    },
    rep3_ring::lut::{PublicPrivateLut, Rep3LookupTable},
};
use std::{iter, marker::PhantomData};

