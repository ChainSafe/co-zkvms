use crate::field::JoltField;
use crate::utils::future_ring::FutureRep3Ring;
use ark_std::log2;
use itertools::izip;
use jolt_core::jolt::instruction::SubtableIndices;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand};
use crate::utils::instruction_utils::{
    chunk_and_concatenate_operands, concatenate_lookups,
    concatenate_lookups_rep3_batched, rep3_chunk_and_concatenate_operands,
};
use jolt_core::jolt::subtable::{xor::XorSubtable, LassoSubtable};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct XORInstruction(pub Rep3Operand, pub Rep3Operand);

impl JoltInstruction for XORInstruction {
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

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
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

    fn lookup_entry<F: JoltField>(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => F::from(*x ^ *y),
            _ => unreachable!(),
        }
    }
}

impl Rep3JoltInstruction for XORInstruction {
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
        Ok(concatenate_lookups_rep3_batched(
            vals,
            C,
            log2(M) as usize / 2,
        ))
    }

    fn to_indices_rep3(
        &self,
        _: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        rep3_chunk_and_concatenate_operands(self.0.as_binary(), self.1.as_binary(), C, log_M)
    }

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        izip!(steps, out).for_each(|(step, out)| {
            *out = FutureRep3Ring::cast_to_field_b2a(
                step.lhs().as_binary() ^ step.rhs().unwrap().as_binary(),
            )
        });
        Ok(())
    }
}
