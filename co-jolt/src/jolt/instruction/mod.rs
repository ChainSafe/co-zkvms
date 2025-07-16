use crate::utils::instruction_utils::chunk_operand;
use enum_dispatch::enum_dispatch;
use jolt_core::poly::field::JoltField;
use jolt_tracer::ELFInstruction;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare,
};
use paste::paste;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount, AsRefStr, EnumIter};
use std::fmt::Debug;

pub use jolt_core::jolt::instruction::SubtableIndices;

use super::subtable::LassoSubtable;

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

    fn operand_chunks(&self, C: usize, log_M: usize) -> (Vec<u64>, Vec<u64>) {
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

    fn slice_values<'a>(&self, vals: &'a [F], C: usize, M: usize) -> Vec<&'a [F]> {
        let mut offset = 0;
        let mut slices = vec![];
        for (_, indices) in self.subtables(C, M) {
            slices.push(&vals[offset..offset + indices.len()]);
            offset += indices.len();
        }
        assert_eq!(offset, vals.len());
        slices
    }
}

#[enum_dispatch]
pub trait Rep3JoltInstruction<F: JoltField>: JoltInstruction<F> {
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>);

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>);

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>>;

    fn to_indices_rep3(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>>;

    fn output<N: Rep3Network>(&self, io_ctx: &mut IoContext<N>) -> eyre::Result<Rep3PrimeFieldShare<F>>;
}

pub trait JoltInstructionSet<F: JoltField>:
    JoltInstruction<F> + IntoEnumIterator + EnumCount + for<'a> TryFrom<&'a ELFInstruction> + AsRef<str> + Send + Sync 
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

    #[tracing::instrument(skip_all, name = "Rep3JoltInstructionSet::operands_to_binary")]
    fn operands_to_binary<N: Rep3Network>(
        ops: &mut [Option<Self>],
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<()> {
        let id = io_ctx.network.get_id();
        ark_std::cfg_iter_mut!(ops)
            .filter_map(|op| op.as_mut())
            .for_each(|op| {
                let (op1, op2) = op.operands_mut();
                match (&op1, &op2) {
                    (Rep3Operand::Public(x), Some(Rep3Operand::Public(y))) => {
                        *op1 = Rep3Operand::Binary(rep3::binary::promote_to_trivial_share(
                            id,
                            &(*x).into(),
                        ));
                        *op2.unwrap() = Rep3Operand::Binary(
                            rep3::binary::promote_to_trivial_share(id, &(*y).into()),
                        );
                    }
                    (Rep3Operand::Public(x), _) => {
                        *op1 = Rep3Operand::Binary(rep3::binary::promote_to_trivial_share(
                            id,
                            &(*x).into(),
                        ));
                    }
                    (_, Some(Rep3Operand::Public(y))) => {
                        *op2.unwrap() = Rep3Operand::Binary(
                            rep3::binary::promote_to_trivial_share(id, &(*y).into()),
                        );
                    }
                    _ => {}
                }
            });

        let (inputs, field_operands): (
            Vec<Vec<Rep3PrimeFieldShare<F>>>,
            Vec<Vec<&mut Rep3Operand<F>>>,
        ) = ark_std::cfg_iter_mut!(ops)
            .filter_map(|op| op.as_mut())
            .map(|op| {
                let (op1, op2) = op.operands_mut();
                match (&op1, &op2) {
                    (Rep3Operand::Arithmetic(x), Some(Rep3Operand::Arithmetic(y))) => {
                        let res = vec![*x, *y];
                        (res, vec![op1, op2.unwrap()])
                    }
                    (Rep3Operand::Arithmetic(x), _) => {
                        let res = vec![*x];
                        (res, vec![op1])
                    }
                    (_, Some(Rep3Operand::Arithmetic(y))) => {
                        let res = vec![*y];
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
            rep3::conversion::a2b_many(&inputs.into_iter().flatten().collect::<Vec<_>>(), io_ctx)?;
        for operands in field_operands.into_iter() {
            for (output, operand) in outputs.drain(..operands.len()).zip(operands) {
                *operand = Rep3Operand::Binary(output);
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
    Arithmetic(Rep3PrimeFieldShare<F>),
    Binary(Rep3BigUintShare<F>),
    Public(u64),
}

impl<F: JoltField> Default for Rep3Operand<F> {
    fn default() -> Self {
        Rep3Operand::Public(0)
    }
}

impl<F: JoltField> From<Rep3PrimeFieldShare<F>> for Rep3Operand<F> {
    fn from(value: Rep3PrimeFieldShare<F>) -> Self {
        Rep3Operand::Arithmetic(value)
    }
}

impl<F: JoltField> From<Rep3BigUintShare<F>> for Rep3Operand<F> {
    fn from(value: Rep3BigUintShare<F>) -> Self {
        Rep3Operand::Binary(value)
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
        // impl<F: JoltField> Default for $enum_name<F> {
        //     fn default() -> Self {
        //         $enum_name::iter().collect::<Vec<_>>()[0]
        //     }
        // }
    };
}

pub mod range_check;

pub mod add;
pub mod and;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod bne;
pub mod lb;
pub mod lh;
pub mod or;
pub mod sb;
pub mod sh;
pub mod sll;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod srl;
pub mod sub;
pub mod sw;
pub mod xor;

instruction_set!(
  TestLookups,
  Range256: range_check::RangeLookup<256, F>,
  Range320: range_check::RangeLookup<320, F>
);

impl<F: JoltField> TryFrom<&ELFInstruction> for TestLookups<F> {
    type Error = &'static str;

    fn try_from(instruction: &ELFInstruction) -> Result<Self, Self::Error> {
        unimplemented!()
    }
}

instruction_set!(
  TestInstructions,
  XOR: xor::XORInstruction<F>
);

impl<F: JoltField> TryFrom<&ELFInstruction> for TestInstructions<F> {
    type Error = &'static str;

    fn try_from(instruction: &ELFInstruction) -> Result<Self, Self::Error> {
        unimplemented!()
    }
}
