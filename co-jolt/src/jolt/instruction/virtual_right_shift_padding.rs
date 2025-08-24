use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::field::JoltField;
use crate::utils::future::FutureVal;
use jolt_core::jolt::subtable::LassoSubtable;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct RightShiftPaddingInstruction<const WORD_SIZE: usize, F: JoltField>(pub Rep3Operand<F>);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F>
    for RightShiftPaddingInstruction<WORD_SIZE, F>
{
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), 0)
    }

    fn combine_lookups(&self, _: &[F], _: usize, _: usize) -> F {
        F::zero()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, _: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![]
    }

    fn to_indices(&self, C: usize, _: usize) -> Vec<usize> {
        vec![0; C]
    }

    fn lookup_entry(&self) -> F {
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

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F>
    for RightShiftPaddingInstruction<WORD_SIZE, F>
{
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), Rep3Operand::default())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, None)
    }

    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        _: &[Rep3PrimeFieldShare<F>],
        _: usize,
        _: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        Ok(Rep3PrimeFieldShare::zero_share())
    }

    fn combine_lookups_rep3_batched<N: Rep3Network>(
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
        _: &Rep3BigUintShare<F>,
        C: usize,
        _: usize,
    ) -> Vec<Rep3BigUintShare<F>> {
        vec![Rep3BigUintShare::zero_share(); C]
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        let shift = self.0.as_public() % WORD_SIZE as u64;
        let ones = (1 << shift) - 1;
        Ok(rep3::arithmetic::promote_to_trivial_share(
            io_ctx.id,
            F::from(ones << (WORD_SIZE as u64 - shift)),
        )
        .into())
    }

    fn output_batched<N: Rep3Network>(
        &self,
        steps: &[Self],
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<FutureVal<F, Rep3PrimeFieldShare<F>>>> {
        steps
            .into_iter()
            .map(|step| {
                let shift = step.0.as_public() % WORD_SIZE as u64;
                let ones = (1 << shift) - 1;
                Ok(FutureVal::Ready(
                    rep3::arithmetic::promote_to_trivial_share(
                        io_ctx.id,
                        F::from(ones << (WORD_SIZE as u64 - shift)),
                    )
                    .into(),
                ))
            })
            .collect()
    }
}
