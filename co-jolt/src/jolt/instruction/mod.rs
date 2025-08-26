use crate::field::JoltField;
use crate::utils::future::FutureVal;
use crate::utils::instruction_utils::chunk_operand;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use enum_dispatch::enum_dispatch;
use jolt_tracer::ELFInstruction;
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare,
};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use strum::{EnumCount, IntoEnumIterator};

pub use jolt_core::jolt::instruction::SubtableIndices;
use jolt_core::jolt::subtable::LassoSubtable;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[enum_dispatch]
pub trait JoltInstruction<F: JoltField>: 'static + Send + Sync + Debug + Clone {
    fn operands(&self) -> (u64, u64);

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F;

    /// The degree of the `g` polynomial described by `combine_lookups`
    fn g_poly_degree(&self, C: usize) -> usize;

    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)>;

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize>;

    fn lookup_entry(&self) -> F;

    fn operand_chunks(&self, C: usize, log_M: usize) -> (Vec<u8>, Vec<u8>) {
        assert!(
            log_M % 2 == 0,
            "log_M must be even for operand_chunks to work"
        );
        let (left_operand, right_operand) = self.operands();
        (
            chunk_operand(left_operand, C, log_M / 2),
            chunk_operand(right_operand, C, log_M / 2),
        )
    }
    fn random(&self, rng: &mut StdRng) -> Self;

    fn slice_values_ref<'a, T>(&self, vals: &'a [T], C: usize, M: usize) -> Vec<&'a [T]> {
        let mut offset = 0;
        let mut slices = vec![];
        for (_, indices) in self.subtables(C, M) {
            slices.push(&vals[offset..offset + indices.len()]);
            offset += indices.len();
        }
        assert_eq!(offset, vals.len());
        slices
    }

    fn slice_values<T: Default>(&self, mut vals: Vec<T>, C: usize, M: usize) -> Vec<Vec<T>> {
        let mut slices = vec![];
        for (_, indices) in self.subtables(C, M) {
            slices.push(vals.drain(..indices.len()).collect());
        }
        slices
    }
}

#[enum_dispatch]
pub trait Rep3JoltInstruction<F: JoltField>: JoltInstruction<F> {
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>);

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>);

    fn lhs(&self) -> &Rep3Operand<F>;
    fn rhs(&self) -> Option<&Rep3Operand<F>>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>;

    fn to_indices_intermediate(
        &self,
        z: &Rep3PrimeFieldShare<F>,
    ) -> FutureVal<F, Option<Rep3BigUintShare<F>>> {
        FutureVal::Ready(None)
    }

    fn to_indices_rep3(
        &self,
        z: Option<Rep3BigUintShare<F>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3BigUintShare<F>>;

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>>;

    fn output_batched<'a, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction<F>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureVal<F, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        Err(eyre::eyre!(
            "output_batched not implemented for instruction"
        ))
    }
}

pub trait JoltInstructionSet<F: JoltField>:
    JoltInstruction<F>
    + IntoEnumIterator
    + EnumCount
    + for<'a> TryFrom<&'a ELFInstruction>
    + AsRef<str>
    + Send
    + Sync
    + CanonicalSerialize
    + CanonicalDeserialize
{
    fn enum_index(lookup: &Self) -> usize {
        let byte = unsafe { *(lookup as *const Self as *const u8) };
        byte as usize
    }

    fn name(&self) -> &str {
        self.as_ref()
    }
}

pub trait Rep3JoltInstructionSet<F: JoltField>:
    Rep3JoltInstruction<F> + IntoEnumIterator + EnumCount + AsRef<str> + Send + Sync
{
    fn enum_index(lookup: &Self) -> usize {
        let byte = unsafe { *(lookup as *const Self as *const u8) };
        byte as usize
    }

    fn promote_public_operands_to_shared<'a>(
        ops: impl ParallelIterator<Item = &'a mut Option<Self>>,
        id: PartyID,
    ) {
        ops.filter_map(|op| op.as_mut()).for_each(|op| {
            let (op1, op2) = op.operands_mut();

            if let Rep3Operand::Public(x) = op1 {
                *op1 = Rep3Operand::Shared {
                    binary: rep3::binary::promote_to_trivial_share(id, &(*x).into()),
                    arithmetic: Some(rep3::arithmetic::promote_to_trivial_share(id, (*x).into())),
                    public: Some(*x),
                };
            }

            if let Some(Rep3Operand::Public(x)) = op2 {
                *op2.unwrap() = Rep3Operand::Shared {
                    binary: rep3::binary::promote_to_trivial_share(id, &(*x).into()),
                    arithmetic: Some(rep3::arithmetic::promote_to_trivial_share(id, (*x).into())),
                    public: Some(*x),
                };
            }
        });
    }

    #[tracing::instrument(skip_all, name = "Rep3JoltInstructionSet::operands_to_binary")]
    fn operands_b2a_many<'a, N: Rep3Network>(
        ops: impl ParallelIterator<Item = &'a mut Option<Self>>,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<()> {
        let (inputs, field_operands): (
            Vec<Vec<Rep3BigUintShare<F>>>,
            Vec<Vec<&mut Rep3Operand<F>>>,
        ) = ops
            .filter_map(|op| op.as_mut())
            .map(|op| {
                let (op1, op2) = op.operands_mut();
                match (&op1, &op2) {
                    (
                        Rep3Operand::Shared {
                            arithmetic: None,
                            binary: x,
                            ..
                        },
                        Some(Rep3Operand::Shared {
                            arithmetic: None,
                            binary: y,
                            ..
                        }),
                    ) => {
                        let res = vec![x.clone(), y.clone()];
                        (res, vec![op1, op2.unwrap()])
                    }
                    (
                        Rep3Operand::Shared {
                            arithmetic: None,
                            binary: x,
                            ..
                        },
                        _,
                    ) => {
                        let res = vec![x.clone()];
                        (res, vec![op1])
                    }
                    (
                        _,
                        Some(Rep3Operand::Shared {
                            arithmetic: None,
                            binary: y,
                            ..
                        }),
                    ) => {
                        let res = vec![y.clone()];
                        (res, vec![op2.unwrap()])
                    }
                    _ => (vec![], vec![]),
                }
            })
            .unzip();

        if inputs.iter().flatten().next().is_none() {
            return Ok(());
        }
        let mut outputs =
            rep3::conversion::b2a_many(&inputs.into_iter().flatten().collect::<Vec<_>>(), io_ctx)?;
        for operands in field_operands.into_iter() {
            for (output, operand) in outputs.drain(..operands.len()).zip(operands) {
                match operand {
                    Rep3Operand::Shared {
                        arithmetic: None,
                        binary,
                        public,
                    } => {
                        *operand = Rep3Operand::Shared {
                            binary: std::mem::take(binary),
                            arithmetic: Some(output),
                            public: std::mem::take(public),
                        };
                    }
                    _ => panic!("Expected shared operand"),
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        self.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "u64", into = "u64")]
pub enum Rep3Operand<F: JoltField> {
    Shared {
        binary: Rep3BigUintShare<F>,
        arithmetic: Option<Rep3PrimeFieldShare<F>>,
        public: Option<u64>, // Some for trivial shares
    },
    Public(u64),
}

impl<F: JoltField> Rep3Operand<F> {
    pub fn as_public(&self) -> u64 {
        match self {
            Rep3Operand::Public(x)
            | Rep3Operand::Shared {
                public: Some(x), ..
            } => *x,
            _ => panic!("Not a public operand"),
        }
    }

    pub fn as_arithmetic_share(&self) -> Rep3PrimeFieldShare<F> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => arithmetic.unwrap(),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_binary_share(&self) -> Rep3BigUintShare<F> {
        match self {
            Rep3Operand::Shared { binary, .. } => binary.clone(),
            _ => panic!("Not a binary operand"),
        }
    }
}

impl<F: JoltField> Default for Rep3Operand<F> {
    fn default() -> Self {
        Rep3Operand::Public(0)
    }
}

impl<F: JoltField> From<Rep3BigUintShare<F>> for Rep3Operand<F> {
    fn from(value: Rep3BigUintShare<F>) -> Self {
        Rep3Operand::Shared {
            binary: value,
            arithmetic: None,
            public: None,
        }
    }
}

impl<F: JoltField> From<u64> for Rep3Operand<F> {
    fn from(value: u64) -> Self {
        Rep3Operand::Public(value)
    }
}

impl<F: JoltField> Into<u64> for Rep3Operand<F> {
    fn into(self) -> u64 {
        match self {
            Rep3Operand::Public(x) => x,
            _ => panic!("Cannot convert Rep3Operand to u64"),
        }
    }
}

impl<F: JoltField> CanonicalSerialize for Rep3Operand<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Rep3Operand::Public(x) => {
                (0_u8).serialize_with_mode(&mut writer, compress)?;
                x.serialize_with_mode(&mut writer, compress)?;
            }
            Rep3Operand::Shared {
                binary,
                arithmetic,
                public,
            } => {
                (1_u8).serialize_with_mode(&mut writer, compress)?;
                binary.serialize_with_mode(&mut writer, compress)?;
                arithmetic.serialize_with_mode(&mut writer, compress)?;
                public.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Rep3Operand::Public(x) => {
                (0_u8).serialized_size(compress) + x.serialized_size(compress)
            }
            Rep3Operand::Shared {
                binary,
                arithmetic,
                public,
            } => {
                (1_u8).serialized_size(compress)
                    + binary.serialized_size(compress)
                    + arithmetic.serialized_size(compress)
                    + public.serialized_size(compress)
            }
        }
    }
}

impl<F: JoltField> CanonicalDeserialize for Rep3Operand<F> {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let res = match discriminant {
            0 => Rep3Operand::Public(u64::deserialize_with_mode(&mut reader, compress, validate)?),
            1 => Rep3Operand::Shared {
                binary: Rep3BigUintShare::<F>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?,
                arithmetic: Option::<Rep3PrimeFieldShare<F>>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?,
                public: Option::<u64>::deserialize_with_mode(&mut reader, compress, validate)?,
            },
            _ => Err(SerializationError::InvalidData)?,
        };
        Ok(res)
    }
}

impl<F: JoltField> Valid for Rep3Operand<F> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Rep3Operand::Public(value) => value.check(),
            Rep3Operand::Shared {
                binary,
                arithmetic,
                public,
            } => {
                binary.check()?;
                arithmetic.check()?;
                public.check()?;
                Ok(())
            }
        }
    }
}

#[macro_export]
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        paste! {
            #[allow(non_camel_case_types)]
            #[repr(u8)]
            #[derive(Clone, Debug, PartialEq, EnumIter, EnumCount, AsRefStr, Serialize, Deserialize)]
            #[enum_dispatch(JoltInstruction<F>, Rep3JoltInstruction<F>)]
            pub enum $enum_name<F: JoltField> {
                $([<$alias>]($struct)),+
            }
        }
        impl<F: JoltField> JoltInstructionSet<F> for $enum_name<F> {}
        impl<F: JoltField> Rep3JoltInstructionSet<F> for $enum_name<F> {}

        // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
        impl<F: JoltField> Default for $enum_name<F> {
            fn default() -> Self {
                $enum_name::iter().collect::<Vec<_>>()[0].clone()
            }
        }
    };
}

// pub mod range_check;

pub mod add;
pub mod and;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod bne;
pub mod mul;
pub mod or;
pub mod sll;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod srl;
pub mod sub;
// pub mod sw;
pub mod mulhu;
pub mod mulu;
pub mod virtual_advice;
pub mod virtual_assert_halfword_alignment;
pub mod virtual_assert_lte;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_signed_remainder;
pub mod virtual_assert_valid_unsigned_remainder;
pub mod virtual_move;
pub mod virtual_movsign;
pub mod virtual_pow2;
pub mod virtual_right_shift_padding;
pub mod xor;

// instruction_set!(
//   TestLookups,
//   Range256: range_check::RangeLookup<256, F>,
//   Range320: range_check::RangeLookup<320, F>
// );

// impl<F: JoltField> TryFrom<&ELFInstruction> for TestLookups<F> {
//     type Error = &'static str;

//     fn try_from(instruction: &ELFInstruction) -> Result<Self, Self::Error> {
//         unimplemented!()
//     }
// }

// instruction_set!(
//   TestInstructions,
//   XOR: xor::XORInstruction<F>
// );

// impl<F: JoltField> TryFrom<&ELFInstruction> for TestInstructions<F> {
//     type Error = &'static str;

//     fn try_from(instruction: &ELFInstruction) -> Result<Self, Self::Error> {
//         unimplemented!()
//     }
// }
