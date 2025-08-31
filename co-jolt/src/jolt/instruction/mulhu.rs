use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use eyre::Context;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::field::JoltField;
use crate::utils::future::FutureVal;
use crate::utils::instruction_utils::{
    assert_valid_parameters, concatenate_lookups, concatenate_lookups_rep3,
    concatenate_lookups_rep3_batched, multiply_and_chunk_operands,
    rep3_multiply_and_chunk_operands,
};
use jolt_core::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MULHUInstruction<const WORD_SIZE: usize>(pub Rep3Operand, pub Rep3Operand);

impl<F: JoltField, const WORD_SIZE: usize> JoltInstruction<F> for MULHUInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups(&self, vals: &[F], _: usize, M: usize) -> F {
        concatenate_lookups(vals, vals.len(), log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        assert_eq!(C * log2(M) as usize, 2 * WORD_SIZE);
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(0..C / 2),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        multiply_and_chunk_operands(
            self.0.as_public() as u128,
            self.1.as_public() as u128,
            C,
            log_M,
        )
    }

    // fn materialize_entry(&self, index: u64) -> u64 {
    //     index >> WORD_SIZE
    // }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => match WORD_SIZE {
                32 => ((*x).wrapping_mul(*y) >> 32).into(),
                64 => ((*x as u128).wrapping_mul(*y as u128) >> 64).into(),
                _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
            },
            _ => panic!("MULHUInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self(
                Rep3Operand::Public(rng.next_u32() as u64),
                Rep3Operand::Public(rng.next_u32() as u64),
            ),
            64 => Self(
                Rep3Operand::Public(rng.next_u64()),
                Rep3Operand::Public(rng.next_u64()),
            ),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<F: JoltField, const WORD_SIZE: usize> Rep3JoltInstruction<F> for MULHUInstruction<WORD_SIZE> {
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

    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        _C: usize,
        M: usize,
        _io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        Ok(concatenate_lookups_rep3(vals, vals.len(), log2(M) as usize))
    }

    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        Ok(concatenate_lookups_rep3_batched(vals, C, log2(M) as usize))
    }

    fn to_indices_intermediate(
        &self,
        z: &Rep3PrimeFieldShare<F>,
    ) -> FutureVal<F, Option<Rep3RingShare<u32>>> {
        FutureVal::a2b(*z)
    }

    fn to_indices_rep3(
        &self,
        z: Option<Rep3RingShare<u32>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        rep3_multiply_and_chunk_operands(&z.unwrap(), C, log_M)
    }

    fn output_batched<'a, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction<F>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureVal<F, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        let (a, b): (Vec<_>, Vec<_>) = steps
            .into_iter()
            .map(|st| {
                (
                    st.lhs().as_arithmetic_share(),
                    st.rhs().unwrap().as_arithmetic_share(),
                )
            })
            .unzip();

        rep3_ring::arithmetic::mul_vec(&a, &b, io_ctx)
            .context("MULInstruction::output_batched")?
            .into_iter()
            .zip(out)
            .for_each(|(ready, out)| {
                *out = FutureVal::cast_to_field(ready);
            });

        Ok(())
    }
}
