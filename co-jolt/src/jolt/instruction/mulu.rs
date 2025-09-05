use ark_std::log2;
use eyre::Context;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::field::JoltField;
use crate::utils::future_ring::FutureRep3Ring;
use crate::utils::instruction_utils::{
    assert_valid_parameters, concatenate_lookups, concatenate_lookups_rep3_batched,
    multiply_and_chunk_operands, rep3_multiply_and_chunk_operands,
};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MULUInstruction<const WORD_SIZE: usize>(pub Rep3Operand, pub Rep3Operand);

impl<const WORD_SIZE: usize> JoltInstruction for MULUInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("MULU instruction requires public operands"),
        }
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(vals.len() == C / 2);
        concatenate_lookups(vals, C / 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(msb_chunk_index + 1..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                multiply_and_chunk_operands(*x as u128, *y as u128, C, log_M)
            }
            _ => panic!("MULU instruction requires public operands"),
        }
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        if WORD_SIZE == 32 {
            (self.0.as_public().wrapping_mul(self.1.as_public()) as u32 as u64).into()
        } else if WORD_SIZE == 64 {
            (self.0.as_public().wrapping_mul(self.1.as_public())).into()
        } else {
            panic!("MULU is only implemented for 32-bit or 64-bit word sizes")
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self(
                (rng.next_u32() as u64).into(),
                (rng.next_u32() as u64).into(),
            ),
            64 => Self(rng.next_u64().into(), rng.next_u64().into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize> Rep3JoltInstruction for MULUInstruction<WORD_SIZE> {
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
            C / 2,
            log2(M) as usize,
        ))
    }

    fn to_indices_intermediate<F: JoltField>(
        &self,
        _: PartyID,
    ) -> FutureRep3Ring<u128, Option<Rep3RingShare<u128>>> {
        FutureRep3Ring::mul_a2b(self.0.as_arithmetic(), self.1.as_arithmetic())
    }

    fn to_indices_rep3(
        &self,
        z: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        rep3_multiply_and_chunk_operands(&z.unwrap(), C, log_M)
    }

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        let (a, b): (Vec<_>, Vec<_>) = steps
            .into_iter()
            .map(|st| {
                (
                    st.lhs().as_arithmetic_u32(),
                    st.rhs().unwrap().as_arithmetic_u32(),
                )
            })
            .unzip();

        rep3_ring::arithmetic::mul_vec(&a, &b, io_ctx)
            .context("MULInstruction::output_batched")?
            .into_iter()
            .zip(out)
            .for_each(|(ready, out)| {
                *out = FutureRep3Ring::cast_to_field(ready);
            });

        Ok(())
    }
}
