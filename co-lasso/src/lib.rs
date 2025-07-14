#![feature(bool_to_result)]

pub mod coordinator;
pub mod instructions;
pub mod lasso;
mod poly;
pub mod subprotocols;
pub mod subtables;
mod utils;
mod witness_solver;
pub mod worker;

pub use witness_solver::{Rep3LassoPolynomials, Rep3LassoWitnessSolver};

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

use enum_dispatch;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount, EnumIter};

#[macro_export]
macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
        #[repr(usize)]
        #[enum_dispatch(LassoSubtable<F>)]
        #[derive(EnumCount, EnumIter)]
        pub enum $enum_name<F: JoltField> { $($alias($struct)),+ }
        impl<F: JoltField> From<SubtableId> for $enum_name<F> {
          fn from(subtable_id: SubtableId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id {:?}", subtable_id) }
          }
        }

        impl<F: JoltField> From<$enum_name<F>> for usize {
            fn from(subtable: $enum_name<F>) -> usize {
                unsafe { *<*const _>::from(&subtable).cast::<usize>() }
            }
        }
        impl<F: JoltField> JoltSubtableSet<F> for $enum_name<F> {}
    };
}
