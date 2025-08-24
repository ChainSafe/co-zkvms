use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::ops::Shl;

use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare, Rep3PrimeFieldShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::field::JoltField;
use crate::utils::future::FutureVal;
use crate::utils::instruction_utils::{
    assert_valid_parameters, chunk_and_concatenate_for_shift, concatenate_lookups,
    concatenate_lookups_rep3, concatenate_lookups_rep3_batched,
    rep3_chunk_and_concatenate_for_shift,
};
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

    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        assert!(C <= 10);
        Ok(concatenate_lookups_rep3_batched(
            vals,
            C,
            (log2(M) / 2) as usize,
        ))
    }

    fn to_indices_rep3(
        &self,
        _: &Rep3BigUintShare<F>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3BigUintShare<F>> {
        rep3_chunk_and_concatenate_for_shift(
            self.0.as_binary_share(),
            self.1.as_binary_share(),
            C,
            log_M,
        )
    }

    fn output<N: Rep3Network>(&self, _: &mut IoContext<N>) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        unimplemented!()
    }

    fn output_batched<N: Rep3Network>(
        &self,
        steps: &[Self],
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<FutureVal<F, Rep3PrimeFieldShare<F>>>> {
        let z = steps
            .into_iter()
            .map(|Self(x, y)| {
                FutureVal::b2a(
                    x.as_binary_share()
                        .shl((y.as_public() as u32 % WORD_SIZE as u32) as usize),
                )
            })
            .collect::<Vec<_>>();
        Ok(z)
    }
}
