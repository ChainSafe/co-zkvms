use crate::field::JoltField;
use crate::utils::future::FutureRep3;
use crate::utils::future_ring::FutureRep3Ring;
use crate::utils::instruction_utils::{chunk_operand, rep3_chunk_operand};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use enum_dispatch::enum_dispatch;
use jolt_tracer::ELFInstruction;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::downcast;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self, arithmetic};
use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        Rep3BigUintShare,
    },
    rep3_ring::Rep3RingShare,
};
use num_traits::AsPrimitive;
use rand::rngs::StdRng;
use rand_chacha::rand_core::le;
use serde::{Deserialize, Serialize};
use std::any::type_name_of_val;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::panic;
use strum::{EnumCount, IntoEnumIterator};

pub use jolt_core::jolt::instruction::SubtableIndices;
use jolt_core::jolt::subtable::LassoSubtable;

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
        id: PartyID,
    ) -> FutureRep3Ring<u128, Option<Rep3RingShare<u128>>> {
        FutureRep3Ring::Ready(None)
    }

    fn to_indices_rep3(
        &self,
        z: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>>;

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        Err(eyre::eyre!(
            "output_batched not implemented for instruction"
        ))
    }

    fn operand_chunks_rep3(
        &self,
        C: usize,
        log_M: usize,
        party_id: PartyID,
    ) -> (Vec<Rep3RingShare<u8>>, Vec<Rep3RingShare<u8>>) {
        assert!(
            log_M % 2 == 0,
            "log_M must be even for operand_chunks to work"
        );
        let (x, y) = self.operands_rep3();
        (
            rep3_chunk_operand(x.as_binary_or_trivial(party_id), C, log_M / 2),
            rep3_chunk_operand(y.as_binary_or_trivial(party_id), C, log_M / 2),
        )
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
    JoltInstructionSet + Rep3JoltInstruction + IntoEnumIterator + EnumCount + AsRef<str> + Send + Sync
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
                        RingElement(*x as u128),
                    )),
                    public: Some(*x),
                };
            }

            if let Some(Rep3Operand::Public(y)) = op2 {
                *op2.unwrap() = Rep3Operand::Shared {
                    binary: rep3_ring::binary::promote_to_trivial_share(
                        id,
                        &RingElement(*y as u32),
                    ),
                    arithmetic: Some(rep3_ring::arithmetic::promote_to_trivial_share(
                        id,
                        RingElement(*y as u128),
                    )),
                    public: Some(*y),
                };
            }
        });
    }

    #[tracing::instrument(skip_all, name = "Rep3JoltInstructionSet::populate_operands_casts")]
    fn populate_operands_casts<'a, N: Rep3Network>(
        ops: impl ParallelIterator<Item = &'a mut Option<Self>>,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<()> {
        let (binary, field_operands): (Vec<Rep3RingShare<u32>>, Vec<&mut Rep3Operand>) = ops
            .filter_map(|op| op.as_mut())
            .flat_map(|op| {
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
                    ) => (vec![x.clone(), y.clone()], vec![op1, op2.unwrap()]),
                    (
                        Rep3Operand::Shared {
                            arithmetic: None,
                            binary: x,
                            ..
                        },
                        _,
                    ) => (vec![x.clone()], vec![op1]),
                    (
                        _,
                        Some(Rep3Operand::Shared {
                            arithmetic: None,
                            binary: y,
                            ..
                        }),
                    ) => (vec![y.clone()], vec![op2.unwrap()]),
                    _ => (vec![], vec![]),
                }
            })
            .unzip();

        if binary.is_empty() {
            return Ok(());
        }

        let arithmetic = rep3_ring::casts::upcast_many_from_binary(&binary, io_ctx)?;

        field_operands
            .into_par_iter()
            .zip_eq(arithmetic)
            .for_each(|(operand, arithmetic)| match operand {
                Rep3Operand::Shared {
                    arithmetic: None,
                    binary,
                    public,
                } => {
                    *operand = Rep3Operand::Shared {
                        binary: std::mem::take(binary),
                        arithmetic: Some(arithmetic),
                        public: std::mem::take(public),
                    };
                }
                _ => panic!("Expected shared operand"),
            });
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
        arithmetic: Option<Rep3RingShare<u128>>,
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

    pub fn from_arithmetic(binary: Rep3RingShare<u32>, arithmetic: Rep3RingShare<u128>) -> Self {
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

    pub fn as_arithmetic<T: IntRing2k>(&self) -> Rep3RingShare<T>
    where
        u128: AsPrimitive<T>,
    {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u32(&self) -> Rep3RingShare<u32> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u64(&self) -> Rep3RingShare<u64> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u128(&self) -> Rep3RingShare<u128> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_binary(&self) -> Rep3RingShare<u32> {
        match self {
            Rep3Operand::Shared { binary, .. } => binary.clone(),
            _ => panic!("Not a binary operand"),
        }
    }

    pub fn as_binary_or_trivial(&self, id: PartyID) -> Rep3RingShare<u32> {
        match *self {
            Rep3Operand::Shared { binary, .. } => binary,
            Rep3Operand::Public(value) => {
                rep3_ring::binary::promote_to_trivial_share(id, &(value as u32).into())
            }
        }
    }
}

impl Default for Rep3Operand {
    fn default() -> Self {
        Rep3Operand::Public(0)
    }
}

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
pub mod mulhu;
pub mod mulu;
pub mod or;
pub mod sll;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod srl;
pub mod sub;
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
