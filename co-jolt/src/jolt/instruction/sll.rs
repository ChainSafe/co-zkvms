use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::rep3::{
    network::{IoContext, Rep3Network},
    Rep3PrimeFieldShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::instruction_utils::{
    assert_valid_parameters, chunk_and_concatenate_for_shift, concatenate_lookups,
    concatenate_lookups_rep3,
};
use jolt_core::field::JoltField;
use jolt_core::jolt::subtable::{sll::SllSubtable, LassoSubtable};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SLLInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for SLLInstruction<WORD_SIZE, F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("SLLInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(C <= 10);
        concatenate_lookups(vals, C, (log2(M) / 2) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // We have to pre-define subtables in this way because `CHUNK_INDEX` needs to be a constant,
        // i.e. known at compile time (so we cannot do a `map` over the range of `C`,
        // which only happens at runtime).
        let mut subtables: Vec<Box<dyn LassoSubtable<F>>> = vec![
            Box::new(SllSubtable::<F, 0, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 1, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 2, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 3, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 4, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 5, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 6, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 7, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 8, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 9, WORD_SIZE>::new()),
        ];
        subtables.truncate(C);
        subtables.reverse();

        let indices = (0..C).map(SubtableIndices::from);
        subtables.into_iter().zip(indices).collect()
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_for_shift(*x, *y, C, log_M)
            }
            _ => panic!("SLLInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                // SLL is specified to ignore all but the last 5 bits of y: https://jemu.oscc.cc/SLL
                (*x as u32)
                    .checked_shl(*y as u32 % WORD_SIZE as u32)
                    .unwrap_or(0)
                    .into()
            }
            _ => panic!("SLLInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F> for SLLInstruction<WORD_SIZE, F> {
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
        assert!(C <= 10);
        Ok(concatenate_lookups_rep3(vals, C, (log2(M) / 2) as usize))
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
            _ => panic!("SLLInstruction::to_indices called with non-binary operands"),
        }
    }

    fn output<N: Rep3Network>(&self, _: &mut IoContext<N>) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Binary(x), Rep3Operand::Binary(y)) => {
                unimplemented!()
            }
            _ => panic!("SLLInstruction::output called with non-binary operands"),
        }
    }
}
