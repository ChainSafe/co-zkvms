use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::{
    field::JoltField,
    jolt::subtable::{identity::IdentitySubtable, LassoSubtable},
    utils::instruction_utils::{chunk_operand_usize, concatenate_lookups},
};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{Rep3BigUintShare, Rep3PrimeFieldShare};

use crate::utils::instruction_utils::concatenate_lookups_rep3;

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MOVEInstruction<const WORD_SIZE: usize, F: JoltField>(pub Rep3Operand<F>);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for MOVEInstruction<WORD_SIZE, F> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), 0)
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        assert!(M == 1 << 16);
        vec![(
            // Implicitly range-checks all query chunks
            Box::new(IdentitySubtable::<F>::new()),
            SubtableIndices::from(0..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0.as_public(), C, log_M)
    }

    fn lookup_entry(&self) -> F {
        // Same for both 32-bit and 64-bit word sizes
        self.0.as_public().into()
    }
    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self((rng.next_u32() as u64).into()),
            64 => Self(rng.next_u64().into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F>
    for MOVEInstruction<WORD_SIZE, F>
{
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), Rep3Operand::default())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, None)
    }

    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        Ok(concatenate_lookups_rep3(vals, C, log2(M) as usize))
    }

    fn to_indices_rep3(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>> {
        // chunk_operand_usize_rep3(
        //     self.0.as_binary_share(),
        //     Rep3Operand::zero().as_binary_share(),
        //     C,
        //     log_M,
        // )
        todo!()
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        Ok(self.0.as_arithmetic_share())
    }
}
