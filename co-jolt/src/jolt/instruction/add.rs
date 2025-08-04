use ark_std::log2;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, Rep3Operand, SubtableIndices};
use crate::jolt::instruction::Rep3JoltInstruction;
use crate::utils::instruction_utils::{
    add_and_chunk_operands, assert_valid_parameters, concatenate_lookups, concatenate_lookups_rep3, concatenate_lookups_rep3_batched,
};
use jolt_core::field::JoltField;
use jolt_core::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ADDInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for ADDInstruction<WORD_SIZE, F> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(vals.len() == C / 2);
        // The output is the TruncateOverflow(most significant chunk) || identity of other chunks
        concatenate_lookups(vals, C / 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(msb_chunk_index + 1..C),
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
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                if WORD_SIZE == 32 {
                    (*x as u32).overflowing_add(*y as u32).0.into()
                } else if WORD_SIZE == 64 {
                    (*x as u64).overflowing_add(*y as u64).0.into()
                } else {
                    panic!("only implemented for u32 / u64")
                }
            }
            _ => panic!("ADDInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F> for ADDInstruction<WORD_SIZE, F> {
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
        _: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        assert!(vals.len() == C / 2);
        Ok(concatenate_lookups_rep3(vals, C / 2, log2(M) as usize))
    }

    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        Ok(concatenate_lookups_rep3_batched(vals, C / 2, log2(M) as usize))
    }

    fn to_indices_rep3(
        &self,
        C: usize,
        log_M: usize,
    ) -> Vec<mpc_core::protocols::rep3::Rep3BigUintShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Binary(x), Rep3Operand::Binary(y)) => {
                unimplemented!()
            }
            _ => panic!("ADDInstruction::to_indices called with non-binary operands"),
        }
    }

    fn output<N: Rep3Network>(&self, _: &mut IoContext<N>) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Binary(x), Rep3Operand::Binary(y)) => {
                unimplemented!()
            }
            _ => panic!("ADDInstruction::output called with non-binary operands"),
        }
    }
}
