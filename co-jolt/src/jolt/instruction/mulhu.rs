use ark_std::log2;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{Rep3BigUintShare, Rep3PrimeFieldShare};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::field::JoltField;
use crate::utils::instruction_utils::{
    assert_valid_parameters, concatenate_lookups, concatenate_lookups_rep3, multiply_and_chunk_operands
};
use jolt_core::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MULHUInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for MULHUInstruction<WORD_SIZE, F> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups(&self, vals: &[F], _: usize, M: usize) -> F {
        concatenate_lookups(vals, vals.len(), log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        assert_eq!(C * log2(M) as usize, 2 * WORD_SIZE);
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(0..C / 2),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        multiply_and_chunk_operands(
            self.0.as_public() as u128,
            self.1.as_public() as u128,
            C,
            log_M,
        )
    }

    // fn materialize_entry(&self, index: u64) -> u64 {
    //     index >> WORD_SIZE
    // }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => match WORD_SIZE {
                32 => ((*x).wrapping_mul(*y) >> 32).into(),
                64 => ((*x as u128).wrapping_mul(*y as u128) >> 64).into(),
                _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
            },
            _ => panic!("MULHUInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self(
                Rep3Operand::Public(rng.next_u32() as u64),
                Rep3Operand::Public(rng.next_u32() as u64),
            ),
            64 => Self(
                Rep3Operand::Public(rng.next_u64()),
                Rep3Operand::Public(rng.next_u64()),
            ),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F>
    for MULHUInstruction<WORD_SIZE, F>
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
        Ok(concatenate_lookups_rep3(vals, vals.len(), log2(M) as usize))
    }

    fn to_indices_rep3(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>> {
        todo!()
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        todo!()
    }
}
