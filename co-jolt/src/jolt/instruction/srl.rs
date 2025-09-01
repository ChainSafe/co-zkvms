use itertools::izip;
use std::iter::Sum;
use std::ops::Shr;

use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        Rep3BigUintShare, Rep3PrimeFieldShare,
    },
    rep3_ring::Rep3RingShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::{
    future::FutureRep3,
    instruction_utils::{
        assert_valid_parameters, chunk_and_concatenate_for_shift,
        rep3_chunk_and_concatenate_for_shift,
    },
};
use jolt_core::jolt::subtable::{srl::SrlSubtable, LassoSubtable};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SRLInstruction<const WORD_SIZE: usize>(pub Rep3Operand, pub Rep3Operand);

impl<const WORD_SIZE: usize> JoltInstruction for SRLInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("SRLInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, _: usize) -> F {
        assert!(C <= 10);
        assert!(vals.len() == C);
        vals.iter().sum()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // We have to pre-define subtables in this way because `CHUNK_INDEX` needs to be a constant,
        // i.e. known at compile time (so we cannot do a `map` over the range of `C`,
        // which only happens at runtime).
        let mut subtables: Vec<Box<dyn LassoSubtable<F>>> = vec![
            Box::new(SrlSubtable::<F, 0, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 1, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 2, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 3, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 4, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 5, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 6, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 7, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 8, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 9, WORD_SIZE>::new()),
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
            _ => panic!("SRLInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                let x = *x as u32;
                let y = (*y % WORD_SIZE as u64) as u32;
                // x.checked_shr(y).unwrap_or(0).into()
                (x.wrapping_shr(y)).into()
            }
            _ => panic!("SRLInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl<const WORD_SIZE: usize> Rep3JoltInstruction for SRLInstruction<WORD_SIZE> {
    fn operands_rep3(&self) -> (Rep3Operand, Rep3Operand) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand, Option<&mut Rep3Operand>) {
        (&mut self.0, Some(&mut self.1))
    }

    fn lhs(&self) -> &Rep3Operand {
        &self.0
    }

    fn rhs(&self) -> Option<&Rep3Operand> {
        Some(&self.1)
    }

    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        assert!(C <= 10);
        assert_eq!(vals.len(), C);
        Ok(rep3::arithmetic::sum_batched(&vals))
    }

    fn to_indices_rep3(
        &self,
        _: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        rep3_chunk_and_concatenate_for_shift(self.0.as_binary(), self.1.as_binary(), C, log_M)
    }

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        izip!(steps, out).for_each(|(st, out)| {
            *out = FutureRep3Ring::cast_to_field_b2a(
                st.lhs()
                    .as_binary()
                    .shr((st.rhs().unwrap().as_public() % WORD_SIZE as u64) as usize),
            )
        });
        Ok(())
    }
}
