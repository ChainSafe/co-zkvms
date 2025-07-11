pub mod range_check;
pub mod xor;

mod utils;

use co_spartan::mpc::rep3::Rep3PrimeFieldShare;
use enum_dispatch::enum_dispatch;
use jolt_core::poly::field::JoltField;
use mpc_core::protocols::rep3::{
    network::{IoContext, Rep3Network},
    Rep3BigUintShare,
};
use std::fmt::Debug;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount, EnumIter};

pub use jolt_core::jolt::instruction::SubtableIndices;

use crate::subtables::LassoSubtable;

#[enum_dispatch]
pub trait LookupType<F: JoltField>: 'static + Send + Sync + Debug + Clone {
    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F;

    /// The degree of the `g` polynomial described by `combine_lookups`
    fn g_poly_degree(&self, C: usize) -> usize;

    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)>;

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize>;

    fn lookup_entry(&self) -> F;
}

// pub enum Rep3Share<F: JoltField> {
//     Arithmetic(Rep3PrimeFieldShare<F>),
//     Binary(Rep3BigUintShare<F>),
// }

#[enum_dispatch]
pub trait Rep3LookupType<F: JoltField>: 'static + Send + Sync + Debug + Clone {
    // fn operands(&self) -> Vec<Rep3PrimeFieldShare<F>>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, vals: &[Rep3PrimeFieldShare<F>], C: usize, M: usize) -> Rep3PrimeFieldShare<F>;

    /// The degree of the `g` polynomial described by `combine_lookups`
    fn g_poly_degree(&self, C: usize) -> usize;

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>>;

    fn output<N: Rep3Network>(&self, io_ctx: &mut IoContext<N>) -> Rep3PrimeFieldShare<F>;
}

pub trait LookupSet<F: JoltField>:
    LookupType<F> + IntoEnumIterator + EnumCount + Send + Sync
{
    fn enum_index(lookup: &Self) -> usize {
        let byte = unsafe { *(lookup as *const Self as *const u8) };
        byte as usize
    }
}

pub trait Rep3LookupSet<F: JoltField>:
    Rep3LookupType<F> + IntoEnumIterator + EnumCount + Send + Sync
{
    fn enum_index(lookup: &Self) -> usize {
        let byte = unsafe { *(lookup as *const Self as *const u8) };
        byte as usize
    }
}

use paste::paste;
macro_rules! lookup_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        paste! {
            #[allow(non_camel_case_types)]
            #[repr(u8)]
            #[derive(Clone, Debug, PartialEq, EnumIter, EnumCount)]
            #[enum_dispatch(LookupType<F>, Rep3LookupType<F>)]
            pub enum $enum_name<F: JoltField> {
                $([<$alias>]($struct)),+
            }
        }
        impl<F: JoltField> LookupSet<F> for $enum_name<F> {}
        impl<F: JoltField> Rep3LookupSet<F> for $enum_name<F> {}

        // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
        // impl<F: JoltField> Default for $enum_name<F> {
        //     fn default() -> Self {
        //         $enum_name::iter().collect::<Vec<_>>()[0]
        //     }
        // }
    };
}

lookup_set!(
  TestLookups,
  Range256: range_check::RangeLookup<256, F>,
  Range320: range_check::RangeLookup<320, F>
);

lookup_set!(
  TestInstructions,
  XOR: xor::XORInstruction<F>
);
