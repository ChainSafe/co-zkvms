use crate::field::JoltField;
use crate::utils::future::FutureVal;
use crate::utils::instruction_utils::chunk_operand;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use enum_dispatch::enum_dispatch;
use jolt_tracer::ELFInstruction;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        Rep3BigUintShare,
    },
    rep3_ring::Rep3RingShare,
};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;
use strum::{EnumCount, IntoEnumIterator};

pub use jolt_core::jolt::instruction::SubtableIndices;
use jolt_core::jolt::subtable::LassoSubtable;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[enum_dispatch]
pub trait JoltInstruction: 'static + Send + Sync + Debug + Clone {
    fn operands(&self) -> (u64, u64);

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F;

    /// The degree of the `g` polynomial described by `combine_lookups`
    fn g_poly_degree(&self, C: usize) -> usize;

    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)>;

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize>;

    fn lookup_entry<F: JoltField>(&self) -> F;

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

    fn slice_values_ref<'a, F: JoltField, T>(
        &self,
        vals: &'a [T],
        C: usize,
        M: usize,
    ) -> Vec<&'a [T]> {
        let mut offset = 0;
        let mut slices = vec![];
        for (_, indices) in self.subtables::<F>(C, M) {
            slices.push(&vals[offset..offset + indices.len()]);
            offset += indices.len();
        }
        assert_eq!(offset, vals.len());
        slices
    }

    fn slice_values<F: JoltField, T: Default>(
        &self,
        mut vals: Vec<T>,
        C: usize,
        M: usize,
    ) -> Vec<Vec<T>> {
        let mut slices = vec![];
        for (_, indices) in self.subtables::<F>(C, M) {
            slices.push(vals.drain(..indices.len()).collect());
        }
        slices
    }
}

#[enum_dispatch]
pub trait Rep3JoltInstruction: JoltInstruction {
    fn operands_rep3(&self) -> (Rep3Operand, Rep3Operand);

    fn operands_mut(&mut self) -> (&mut Rep3Operand, Option<&mut Rep3Operand>);

    fn lhs(&self) -> &Rep3Operand;
    fn rhs(&self) -> Option<&Rep3Operand>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>;

    fn to_indices_intermediate<F: JoltField>(
        &self,
        z: &Rep3PrimeFieldShare<F>,
    ) -> FutureVal<F, Option<Rep3RingShare<u32>>> {
        FutureVal::Ready(None)
    }

    fn to_indices_rep3(
        &self,
        z: Option<Rep3RingShare<u32>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>>;

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureVal<F, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        Err(eyre::eyre!(
            "output_batched not implemented for instruction"
        ))
    }
}

pub trait JoltInstructionSet:
    JoltInstruction
    + IntoEnumIterator
    + EnumCount
    + for<'a> TryFrom<&'a ELFInstruction>
    + AsRef<str>
    + Send
    + Sync
{
    fn enum_index(lookup: &Self) -> usize {
        let byte = unsafe { *(lookup as *const Self as *const u8) };
        byte as usize
    }

    fn name(&self) -> &str {
        self.as_ref()
    }
}

pub trait Rep3JoltInstructionSet:
    Rep3JoltInstruction + IntoEnumIterator + EnumCount + AsRef<str> + Send + Sync
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
                    binary: rep3_ring::binary::promote_to_trivial_share(
                        id,
                        &RingElement(*x as u32),
                    ),
                    arithmetic: Some(rep3_ring::arithmetic::promote_to_trivial_share(
                        id,
                        RingElement(*x as u32),
                    )),
                    public: Some(*x),
                };
            }

            if let Some(Rep3Operand::Public(x)) = op2 {
                *op2.unwrap() = Rep3Operand::Shared {
                    binary: rep3_ring::binary::promote_to_trivial_share(
                        id,
                        &RingElement(*x as u32),
                    ),
                    arithmetic: Some(rep3_ring::arithmetic::promote_to_trivial_share(
                        id,
                        RingElement(*x as u32),
                    )),
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
        let (inputs, field_operands): (Vec<Vec<Rep3RingShare<u32>>>, Vec<Vec<&mut Rep3Operand>>) =
            ops.filter_map(|op| op.as_mut())
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
        let mut outputs = rep3_ring::conversion::a2b_many(
            &inputs.into_iter().flatten().collect::<Vec<_>>(),
            io_ctx,
        )?;
        for operands in field_operands.into_iter() {
            for (output, operand) in outputs.drain(..operands.len()).zip(operands) {
                match operand {
                    Rep3Operand::Shared {
                        arithmetic: None,
                        binary,
                        public,
                        ..
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
pub enum Rep3Operand {
    Shared {
        binary: Rep3RingShare<u32>,
        arithmetic: Option<Rep3RingShare<u32>>,
        public: Option<u64>, // Some for trivial shares
    },
    Public(u64),
}

impl Rep3Operand {
    pub fn from_binary(share: Rep3RingShare<u32>) -> Self {
        Rep3Operand::Shared {
            binary: share,
            arithmetic: None,
            public: None,
        }
    }

    pub fn from_arithmetic(binary: Rep3RingShare<u32>, arithmetic: Rep3RingShare<u32>) -> Self {
        Rep3Operand::Shared {
            binary,
            arithmetic: Some(arithmetic),
            public: None,
        }
    }

    pub fn as_public(&self) -> u64 {
        match self {
            Rep3Operand::Public(x)
            | Rep3Operand::Shared {
                public: Some(x), ..
            } => *x,
            _ => panic!("Not a public operand"),
        }
    }

    pub fn as_arithmetic_share(&self) -> Rep3RingShare<u32> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => arithmetic.unwrap(),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_binary_share(&self) -> Rep3RingShare<u32> {
        match self {
            Rep3Operand::Shared { binary, .. } => binary.clone(),
            _ => panic!("Not a binary operand"),
        }
    }
}

impl Default for Rep3Operand {
    fn default() -> Self {
        Rep3Operand::Public(0)
    }
}

// impl<F: JoltField> From<Rep3BigUintShare<F>> for Rep3Operand {
//     fn from(value: Rep3BigUintShare<F>) -> Self {
//         Rep3Operand::Shared {
//             binary: value,
//             arithmetic: None,
//             public: None,
//         }
//     }
// }

impl From<u64> for Rep3Operand {
    fn from(value: u64) -> Self {
        Rep3Operand::Public(value)
    }
}

impl From<u32> for Rep3Operand {
    fn from(value: u32) -> Self {
        Rep3Operand::Public(value as u64)
    }
}

impl Into<u64> for Rep3Operand {
    fn into(self) -> u64 {
        match self {
            Rep3Operand::Public(x) => x,
            _ => panic!("Cannot convert Rep3Operand to u64"),
        }
    }
}

impl Into<u32> for Rep3Operand {
    fn into(self) -> u32 {
        match self {
            Rep3Operand::Public(x) => x as u32,
            _ => panic!("Cannot convert Rep3Operand to u32"),
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
            #[enum_dispatch(JoltInstruction, Rep3JoltInstruction)]
            pub enum $enum_name {
                $([<$alias>]($struct)),+
            }
        }
        impl JoltInstructionSet for $enum_name {}
        impl Rep3JoltInstructionSet for $enum_name {}

        // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
        impl Default for $enum_name {
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
