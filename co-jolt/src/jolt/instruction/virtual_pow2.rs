use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::izip;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::field::JoltField;
use crate::utils::future::FutureVal;
use jolt_core::jolt::subtable::LassoSubtable;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(
    Clone,
    Default,
    Debug,
    Serialize,
    Deserialize,
    PartialEq,
)]
pub struct POW2Instruction<const WORD_SIZE: usize>(pub Rep3Operand);

impl<F: JoltField, const WORD_SIZE: usize> JoltInstruction<F> for POW2Instruction<WORD_SIZE> {
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
        (1 << (self.0.as_public() % WORD_SIZE as u64)).into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self((rng.next_u32() as u64).into()),
            64 => Self(rng.next_u64().into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<F: JoltField, const WORD_SIZE: usize> Rep3JoltInstruction<F>
    for POW2Instruction<WORD_SIZE>
{
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
        _: Option<Rep3RingShare<u32>>,
        C: usize,
        _: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        vec![Rep3RingShare::zero_share(); C]
    }

    fn output_batched<'a, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction<F>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureVal<F, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        izip!(steps, out).for_each(|(step, out)| {
            *out = FutureVal::Ready(
                rep3::arithmetic::promote_to_trivial_share(
                    io_ctx.id,
                    F::from(1 << (step.lhs().as_public() % WORD_SIZE as u64)),
                )
                .into(),
            )
        });
        Ok(())
    }
}
