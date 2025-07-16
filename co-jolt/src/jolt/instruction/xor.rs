use ark_std::log2;
use eyre::Context;
use jolt_core::jolt::instruction::SubtableIndices;
use jolt_core::poly::field::JoltField;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand};
use crate::jolt::subtable::{xor::XorSubtable, LassoSubtable};
use crate::utils::instruction_utils::{
    chunk_and_concatenate_operands, concatenate_lookups, concatenate_lookups_rep3,
    rep3_chunk_and_concatenate_operands,
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct XORInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> XORInstruction<F> {
    pub fn public(x: u64, y: u64) -> Self {
        Self(Rep3Operand::Public(x), Rep3Operand::Public(y))
    }

    pub fn shared_binary(x: Rep3BigUintShare<F>, y: Rep3BigUintShare<F>) -> Self {
        Self(Rep3Operand::Binary(x), Rep3Operand::Binary(y))
    }

    pub fn shared(x: Rep3PrimeFieldShare<F>, y: Rep3PrimeFieldShare<F>) -> Self {
        Self(Rep3Operand::Arithmetic(x), Rep3Operand::Arithmetic(y))
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
    fn operands(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    fn combine_lookups<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        Ok(concatenate_lookups_rep3(vals, C, log2(M) as usize / 2))
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn to_indices(
        &self,
        C: usize,
        log_M: usize,
    ) -> Vec<mpc_core::protocols::rep3::Rep3BigUintShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Binary(x), Rep3Operand::Binary(y)) => {
                rep3_chunk_and_concatenate_operands(x.clone(), y.clone(), C, log_M)
            }
            _ => panic!("XORInstruction::to_indices called with non-binary operands"),
        }
    }

    fn output<N: Rep3Network>(&self, io_ctx: &mut IoContext<N>) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Binary(x), Rep3Operand::Binary(y)) => {
                rep3::conversion::b2a_selector(&(x.clone() ^ y.clone()), io_ctx)
                    .context("while evaluating XORInstruction")
            }
            _ => panic!("XORInstruction::output called with non-binary operands"),
        }
    }
}
