use ark_std::log2;
use eyre::Context;
use jolt_core::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare, Rep3PrimeFieldShare,
};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::field::JoltField;
use crate::utils::future::FutureVal;
use crate::{
    jolt::instruction::Rep3JoltInstruction,
    utils::instruction_utils::{
        concatenate_lookups_rep3, concatenate_lookups_rep3_batched,
        rep3_multiply_and_chunk_operands,
    },
};

use super::{JoltInstruction, Rep3Operand, SubtableIndices};
use jolt_core::utils::instruction_utils::{
    assert_valid_parameters, concatenate_lookups, multiply_and_chunk_operands,
};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MULInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for MULInstruction<WORD_SIZE, F> {
    // fn to_lookup_index(&self) -> u64 {
    //     self.0 * self.1
    // }

    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("MULInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(vals.len() == C / 2);
        concatenate_lookups(vals, C / 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
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
            _ => panic!("MULInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                if WORD_SIZE == 32 {
                    let x = *x as i32;
                    let y = *y as i32;
                    (x.wrapping_mul(y) as u32 as u64).into()
                } else if WORD_SIZE == 64 {
                    let x = *x as i64;
                    let y = *y as i64;
                    (x.wrapping_mul(y) as u64).into()
                } else {
                    panic!("MUL is only implemented for 32-bit or 64-bit word sizes")
                }
            }
            _ => panic!("MULInstruction::lookup_entry called with non-public operands"),
        }
    }

    // fn materialize_entry(&self, index: u64) -> u64 {
    //     index % (1 << WORD_SIZE)
    // }

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

    // fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
    //     debug_assert_eq!(r.len(), 2 * WORD_SIZE);
    //     let mut result = F::zero();
    //     for i in 0..WORD_SIZE {
    //         result += F::from_u64_unchecked(1 << (WORD_SIZE - 1 - i)) * r[WORD_SIZE + i];
    //     }
    //     result
    // }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F> for MULInstruction<WORD_SIZE, F> {
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
        assert!(vals.len() == C / 2);
        Ok(concatenate_lookups_rep3(vals, C / 2, log2(M) as usize))
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
            C / 2,
            log2(M) as usize,
        ))
    }

    fn to_indices_rep3(
        &self,
        z: &Rep3BigUintShare<F>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3BigUintShare<F>> {
        rep3_multiply_and_chunk_operands(z, C, log_M)
    }

    fn output<N: mpc_core::protocols::rep3::network::Rep3Network>(
        &self,
        _: &mut mpc_core::protocols::rep3::network::IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        unimplemented!()
    }

    fn output_batched<N: Rep3Network>(
        &self,
        steps: &[Self],
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<FutureVal<F, Rep3PrimeFieldShare<F>>>> {
        let (a, b): (Vec<_>, Vec<_>) = steps
            .into_iter()
            .map(|Self(x, y)| (x.as_arithmetic_share(), y.as_arithmetic_share()))
            .unzip();

        Ok(rep3::arithmetic::mul_vec(&a, &b, io_ctx)
            .context("MULInstruction::output_batched")?
            .into_iter()
            .map(FutureVal::Ready)
            .collect())
    }
}
