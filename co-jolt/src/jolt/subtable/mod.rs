use enum_dispatch::enum_dispatch;
use jolt_core::poly::field::JoltField;
use mpc_core::protocols::rep3::{Rep3BigUintShare,Rep3PrimeFieldShare};
use std::any::TypeId;
use std::fmt::Debug;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount, EnumIter};

pub use jolt_core::jolt::instruction::SubtableIndices;

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
pub trait JoltSubtableSet<F: JoltField>:
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
        impl<F: JoltField> JoltSubtableSet<F> for $enum_name<F> {}
    };
}

pub mod range_check;
pub mod xor;

pub use range_check::{BoundSubtable, FullLimbSubtable};
pub use xor::XorSubtable;

subtable_enum!(
  TestSubtables,
  Full: range_check::FullLimbSubtable<F>,
  Bound: range_check::BoundSubtable<320,F>
);

subtable_enum!(
  TestInstructionSubtables,
  XOR: XorSubtable<F>
);
