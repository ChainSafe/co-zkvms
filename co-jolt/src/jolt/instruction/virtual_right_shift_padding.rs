use itertools::izip;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::field::JoltField;
use crate::utils::future_ring::FutureRep3Ring;
use jolt_core::jolt::subtable::LassoSubtable;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, Rep3PrimeFieldShare};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct RightShiftPaddingInstruction<const WORD_SIZE: usize>(pub Rep3Operand);

impl<const WORD_SIZE: usize> JoltInstruction for RightShiftPaddingInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), 0)
    }

    fn combine_lookups<F: JoltField>(&self, _: &[F], _: usize, _: usize) -> F {
        F::zero()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        _: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![]
    }

    fn to_indices(&self, C: usize, _: usize) -> Vec<usize> {
        vec![0; C]
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        let shift = self.0.as_public() % WORD_SIZE as u64;
        let ones = (1 << shift) - 1;
        (ones << (WORD_SIZE as u64 - shift)).into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self((rng.next_u32() as u64).into()),
            64 => Self(rng.next_u64().into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize> Rep3JoltInstruction for RightShiftPaddingInstruction<WORD_SIZE> {
    fn operands_rep3(&self) -> (Rep3Operand, Rep3Operand) {
        (self.0.clone(), Rep3Operand::default())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand, Option<&mut Rep3Operand>) {
        (&mut self.0, None)
    }

    fn lhs(&self) -> &Rep3Operand {
        &self.0
    }

    fn rhs(&self) -> Option<&Rep3Operand> {
        None
    }

    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        _: usize,
        _: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        Ok(vec![Rep3PrimeFieldShare::zero_share(); vals[0].len()])
    }

    fn to_indices_rep3(
        &self,
        _: Option<Rep3RingShare<u128>>,
        C: usize,
        _: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        vec![Rep3RingShare::zero_share(); C]
    }

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        izip!(steps, out).for_each(|(step, out)| {
            let shift = step.lhs().as_public() % WORD_SIZE as u64;
            let ones = (1 << shift) - 1;
            *out = FutureRep3Ring::Ready(
                rep3::arithmetic::promote_to_trivial_share(
                    io_ctx.id,
                    F::from(ones << (WORD_SIZE as u64 - shift)),
                )
                .into(),
            );
        });
        Ok(())
    }
}
