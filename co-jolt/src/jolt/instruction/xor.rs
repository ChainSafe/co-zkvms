use crate::field::JoltField;
use ark_std::log2;
use eyre::Context;
use jolt_core::jolt::instruction::SubtableIndices;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand};
use crate::utils::future::FutureVal;
use crate::utils::instruction_utils::{
    chunk_and_concatenate_operands, concatenate_lookups, concatenate_lookups_rep3,
    concatenate_lookups_rep3_batched, rep3_chunk_and_concatenate_operands,
};
use jolt_core::jolt::subtable::{xor::XorSubtable, LassoSubtable};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct XORInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> XORInstruction<F> {
    pub fn public(x: u64, y: u64) -> Self {
        Self(Rep3Operand::Public(x), Rep3Operand::Public(y))
    }
}

impl<F: JoltField> JoltInstruction<F> for XORInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => unreachable!(),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            Rep3Operand::Public(rng.next_u32() as u64),
            Rep3Operand::Public(rng.next_u32() as u64),
        )
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(XorSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_operands(*x, *y, C, log_M)
            }
            _ => unreachable!(),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => F::from(*x ^ *y),
            _ => unreachable!(),
        }
    }
}

impl<F: JoltField> Rep3JoltInstruction<F> for XORInstruction<F> {
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
        rep3_chunk_and_concatenate_operands(
            self.0.as_binary_share(),
            self.1.as_binary_share(),
            C,
            log_M,
        )
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        unimplemented!()
    }

    fn output_batched<N: Rep3Network>(
        &self,
        steps: &[Self],
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<FutureVal<F, Rep3PrimeFieldShare<F>>>> {
        let z = steps
            .into_iter()
            .map(|step| FutureVal::b2a(step.0.as_binary_share() ^ step.1.as_binary_share()))
            .collect::<Vec<_>>();
        Ok(z)
    }
}
