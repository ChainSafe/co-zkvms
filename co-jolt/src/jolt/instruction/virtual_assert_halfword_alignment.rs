use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3PrimeFieldShare,
};

use jolt_core::jolt::subtable::{low_bit::LowBitSubtable, LassoSubtable};
use jolt_core::{
    field::JoltField,
    utils::instruction_utils::{add_and_chunk_operands, assert_valid_parameters},
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

/// (address, offset)
#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssertHalfwordAlignmentInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F>
    for AssertHalfwordAlignmentInstruction<WORD_SIZE, F>
{
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups(&self, vals: &[F], _: usize, _: usize) -> F {
        assert_eq!(vals.len(), 1);
        let lowest_bit = vals[0];
        F::one() - lowest_bit
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(
            Box::new(LowBitSubtable::<F>::new()),
            SubtableIndices::from(C - 1),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        add_and_chunk_operands(
            self.0.as_public() as u128,
            self.1.as_public() as u128,
            C,
            log_M,
        )
    }

    fn lookup_entry(&self) -> F {
        match WORD_SIZE {
            32 => (((self.0.as_public() as u32 as i32 + self.1.as_public() as u32 as i32) % 2 == 0)
                as u64)
                .into(),
            64 => {
                (((self.0.as_public() as i64 + self.1.as_public() as i64) % 2 == 0) as u64).into()
            }
            _ => panic!("Only 32-bit and 64-bit word sizes are supported"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self(
                (rng.next_u32() as u64).into(),
                ((rng.next_u32() % (1 << 12)) as u64).into(),
            ),
            64 => Self(rng.next_u64().into(), (rng.next_u64() % (1 << 12)).into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F>
    for AssertHalfwordAlignmentInstruction<WORD_SIZE, F>
{
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        assert_eq!(vals.len(), 1);
        let lowest_bit = vals[0];
        Ok(rep3::arithmetic::sub_public_by_shared(
            F::one(),
            lowest_bit,
            io_ctx.id,
        ))
    }

    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        assert_eq!(vals.len(), 1);
        Ok(vals[0]
            .iter()
            .map(|lowest_bit| {
                rep3::arithmetic::sub_public_by_shared(F::one(), *lowest_bit, io_ctx.id)
            })
            .collect::<Vec<_>>())
    }

    fn to_indices_rep3(&self, C: usize, log_M: usize) -> Vec<rep3::Rep3BigUintShare<F>> {
        // add_and_chunk_operands_rep3(self.0, self.1, C, log_M)
        todo!()
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        todo!()
    }
}
