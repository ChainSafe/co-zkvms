use std::mem;

use itertools::{chain, Itertools};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::{
    field::JoltField,
    jolt::subtable::LassoSubtable,
    jolt::subtable::{div_by_zero::DivByZeroSubtable, left_is_zero::LeftIsZeroSubtable},
    utils::instruction_utils::chunk_and_concatenate_operands,
    utils::uninterleave_bits,
};
use mpc_core::protocols::additive;
use mpc_core::protocols::rep3::{Rep3BigUintShare, Rep3PrimeFieldShare};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        self,
        network::{IoContext, Rep3Network},
    },
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::instruction_utils::{
    assert_valid_parameters, concatenate_lookups, multiply_and_chunk_operands,
    rep3_chunk_and_concatenate_operands,
};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (divisor, quotient)
pub struct AssertValidDiv0Instruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F>
    for AssertValidDiv0Instruction<WORD_SIZE, F>
{
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref(vals, C, M);
        let divisor_is_zero: F = vals_by_subtable[0].iter().product();
        let is_valid_div_by_zero: F = vals_by_subtable[1].iter().product();

        F::one() - divisor_is_zero + is_valid_div_by_zero
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (
                Box::new(LeftIsZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
            (
                Box::new(DivByZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0.as_public(), self.1.as_public(), C, log_M)
    }

    fn lookup_entry(&self) -> F {
        let divisor = self.0.as_public();
        let quotient = self.1.as_public();
        if divisor == 0 {
            match WORD_SIZE {
                32 => (quotient == u32::MAX as u64).into(),
                64 => (quotient == u64::MAX).into(),
                _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
            }
        } else {
            F::one()
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

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F>
    for AssertValidDiv0Instruction<WORD_SIZE, F>
{
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    #[tracing::instrument(
        skip_all,
        name = "AssertValidDiv0Instruction::combine_lookups_rep3",
        level = "trace"
    )]
    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        let vals_by_subtable = self.slice_values_ref(vals, C, M);

        #[cfg(not(feature = "public-eq"))]
        {
            use crate::utils::instruction_utils::transpose;

            let [divisor_is_zero, is_valid_div_by_zero] = rep3::arithmetic::product_many(
                &transpose(vec![
                    vals_by_subtable[0].to_vec(),
                    vals_by_subtable[1].to_vec(),
                ]),
                io_ctx,
            )?
            .try_into()
            .unwrap();

            return Ok(rep3::arithmetic::sub_public_by_shared(
                F::one(),
                divisor_is_zero + is_valid_div_by_zero,
                io_ctx.id,
            ));
        }

        #[cfg(feature = "public-eq")]
        {
            let opened = rep3::arithmetic::open_vec(&vals_by_subtable[..2].concat(), io_ctx)?;

            let (divisor_is_zero_vals, is_valid_div_by_zero_vals) =
                opened.split_at(vals_by_subtable[0].len());

            let divisor_is_zero: F = divisor_is_zero_vals.iter().product();
            let is_valid_div_by_zero: F = is_valid_div_by_zero_vals.iter().product();

            return Ok(rep3::arithmetic::promote_to_trivial_share(
                io_ctx.id,
                F::one() - divisor_is_zero + is_valid_div_by_zero,
            ));
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "AssertValidDiv0Instruction::combine_lookups_rep3_batched",
        level = "trace"
    )]
    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals_many: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        let batch_size = vals_many[0].len();
        let mut batched_vals_by_subtable = self.slice_values(vals_many, C, M);

        #[cfg(not(feature = "public-eq"))]
        {
            let products = rep3::arithmetic::product_many(
                (0..C).map(|i| {
                    (0..batch_size)
                        .flat_map(|j| {
                            [
                                mem::take(&mut batched_vals_by_subtable[0][i][j]),
                                mem::take(&mut batched_vals_by_subtable[1][i][j]),
                            ]
                        })
                        .collect::<Vec<_>>()
                }),
                io_ctx,
            )?;
            let res = products
                .chunks(2)
                .map(|chunk| {
                    let [divisor_is_zero, is_valid_div_by_zero] = chunk.try_into().unwrap();
                    rep3::arithmetic::sub_public_by_shared(
                        F::one(),
                        divisor_is_zero + is_valid_div_by_zero,
                        io_ctx.id,
                    )
                })
                .collect::<Vec<_>>();

            return Ok(res);
        }

        #[cfg(feature = "public-eq")]
        {
            let res = rep3::arithmetic::open_vec(
                (0..batch_size).flat_map(|j| {
                    chain!(
                        (0..C).map(|i| &batched_vals_by_subtable[0][i][j]),
                        (0..C).map(|i| &batched_vals_by_subtable[1][i][j]),
                    )
                    .collect::<Vec<_>>()
                }),
                io_ctx,
            )?
            .chunks(C * 2)
            .map(|chunk| {
                let (divisor_is_zero_vals, is_valid_div_by_zero_vals) = chunk.split_at(C);
                let divisor_is_zero: F = divisor_is_zero_vals.iter().product();
                let is_valid_div_by_zero: F = is_valid_div_by_zero_vals.iter().product();
                rep3::arithmetic::promote_to_trivial_share(
                    io_ctx.id,
                    F::one() - divisor_is_zero + is_valid_div_by_zero,
                )
            })
            .collect::<Vec<_>>();

            return Ok(res);
        }
    }

    fn to_indices_rep3(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>> {
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
        todo!()
    }
}
