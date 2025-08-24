use ark_std::log2;
use eyre::Context;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::rep3::{self,
    network::{IoContext, Rep3Network}, Rep3BigUintShare, Rep3PrimeFieldShare
};

use super::{JoltInstruction, SubtableIndices};
use crate::utils::instruction_utils::{
    chunk_and_concatenate_operands, concatenate_lookups, concatenate_lookups_rep3,
    rep3_chunk_and_concatenate_operands,
};
use crate::utils::future::FutureVal;
use crate::{
    jolt::instruction::{Rep3JoltInstruction, Rep3Operand},
    utils::instruction_utils::concatenate_lookups_rep3_batched,
};
use crate::field::JoltField;
use jolt_core::jolt::subtable::{or::OrSubtable, LassoSubtable};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ORInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for ORInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("ORInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(OrSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_operands(*x, *y, C, log_M)
            }
            _ => panic!("ORInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => F::from(*x | *y),
            _ => panic!("ORInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl<F: JoltField> Rep3JoltInstruction<F> for ORInstruction<F> {
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
        Ok(concatenate_lookups_rep3(vals, C, log2(M) as usize / 2))
    }

    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        Ok(concatenate_lookups_rep3_batched(
            vals,
            C,
            log2(M) as usize / 2,
        ))
    }

    fn to_indices_rep3(
        &self,
        _: &Rep3BigUintShare<F>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3BigUintShare<F>> {
        rep3_chunk_and_concatenate_operands(self.0.as_binary_share(), self.1.as_binary_share(), C, log_M)
    }

    fn output<N: Rep3Network>(&self, _: &mut IoContext<N>) -> eyre::Result<Rep3PrimeFieldShare<F>> {
       unimplemented!()
    }

    fn output_batched<N: Rep3Network>(
        &self,
        steps: &[Self],
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<FutureVal<F, Rep3PrimeFieldShare<F>>>> {
        let (a, b): (Vec<_>, Vec<_>) = steps
            .into_iter()
            .map(|Self(x, y)| (x.as_binary_share(), y.as_binary_share()))
            .unzip();

        let z = rep3::binary::or_vec(&a, &b, io_ctx).context("ORInstruction::output_batched")?;
        Ok(z.into_iter().map(FutureVal::b2a).collect())
    }
}
