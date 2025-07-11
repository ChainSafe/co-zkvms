pub mod range_check;

use co_spartan::mpc::rep3::Rep3PrimeFieldShare;
use enum_dispatch::enum_dispatch;
use jolt_core::poly::field::JoltField;
use mpc_core::protocols::rep3::Rep3BigUintShare;
use std::any::TypeId;
use std::fmt::Debug;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount, EnumIter};

pub use jolt_core::jolt::instruction::SubtableIndices;
// pub use jolt_core::jolt::subtable::{LassoSubtable, JoltSubtableSet as SubtableSet};

// pub type SubtableId = TypeId;
pub type LookupId = String;

#[enum_dispatch]
pub trait LassoSubtable<F: JoltField>: 'static + Sync {
    /// Returns the TypeId of this subtable.
    /// The `Jolt` trait has associated enum types `InstructionSet` and `Subtables`.
    /// This function is used to resolve the many-to-many mapping between `InstructionSet` variants
    /// and `Subtables` variants,
    fn subtable_id(&self) -> SubtableId {
        TypeId::of::<Self>()
    }
    /// Fully materializes a subtable of size `M`, reprensented as a Vec of length `M`.
    fn materialize(&self, M: usize) -> Vec<F>;
    /// Evaluates the multilinear extension polynomial for this subtable at the given `point`,
    /// interpreted to be of size log_2(M), where M is the size of the subtable.
    fn evaluate_mle(&self, point: &[F]) -> F;
}

pub type SubtableId = TypeId;
pub trait SubtableSet<F: JoltField>:
    LassoSubtable<F>
    + IntoEnumIterator
    + EnumCount
    + From<SubtableId>
    + Into<usize>
    + std::fmt::Debug
    + Send
    + Sync
{
    fn enum_index(subtable: Box<dyn LassoSubtable<F>>) -> usize {
        Self::from(subtable.subtable_id()).into()
    }
}

#[enum_dispatch]
pub trait LookupType<F: JoltField>: 'static + Send + Sync + Debug + Clone {
    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F;

    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)>;

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize>;

    fn output(&self) -> F;

    /// Returns the indices of each subtable lookups
    /// The length of `index_bits` is same as actual bit length of table index
    fn subtable_indices(&self, index_bits: Vec<bool>, log_M: usize) -> Vec<Vec<bool>>;

    // fn num_memories(&self) -> usize;
}

#[enum_dispatch]
pub trait Rep3LookupType<F: JoltField>: 'static + Send + Sync + Debug + Clone {
    // fn operands(&self) -> Vec<Rep3PrimeFieldShare<F>>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
    ) -> Rep3PrimeFieldShare<F>;

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>>;

    fn output(&self) -> Rep3PrimeFieldShare<F>;
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

// pub trait LookupClone<F: JoltField> {
//     fn clone_box(&self) -> Box<dyn LookupType<F>>;
// }

// impl<T, F: JoltField> LookupClone<F> for T
// where
//     T: LookupType<F> + Clone + 'static,
// {
//     fn clone_box(&self) -> Box<dyn LookupType<F>> {
//         Box::new(self.clone())
//     }
// }

// impl<F> Clone for Box<dyn LookupType<F>> {
//     fn clone(&self) -> Self {
//         unimplemented!()
//         // self.clone_box()
//     }
// }

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

macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        paste! {
            #[allow(non_camel_case_types)]
            #[repr(u8)]
            #[enum_dispatch(LassoSubtable<F>)]
            #[derive(Debug, EnumCount, EnumIter)]
            pub enum $enum_name<F: JoltField> { $([<$alias>]($struct)),+ }
        }
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
                // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
                let byte = unsafe { *(&subtable as *const $enum_name<F> as *const u8) };
                byte as usize
            }
        }
        impl<F: JoltField> SubtableSet<F> for $enum_name<F> {}
    };
}

lookup_set!(
  TestLookups,
  Range256: range_check::RangeLookup<256, F>,
  Range320: range_check::RangeLookup<320, F>
);

subtable_enum!(
  TestSubtables,
  Full: range_check::FullLimbSubtable<F>,
  Bound: range_check::BoundSubtable<320,F>
);

// impl<F: JoltField> TestLookups<F> {
//     fn a2b(lookups: &[Self]) -> Vec<Rep3BigUintShare<F>> {
//         lookups
//             .iter()
//             .enumerate()
//             .filter(|(i, lookup)| matches!(lookup, Lookups::Range256(_)))
//             .collect()
//     }
// }
