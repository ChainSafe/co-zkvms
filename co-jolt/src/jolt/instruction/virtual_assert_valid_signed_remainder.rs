use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::{
    field::JoltField,
    jolt::subtable::right_is_zero::RightIsZeroSubtable,
    jolt::subtable::{
        eq::EqSubtable, eq_abs::EqAbsSubtable, left_is_zero::LeftIsZeroSubtable,
        left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable, ltu::LtuSubtable,
        right_msb::RightMSBSubtable, LassoSubtable,
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
};
use mpc_core::protocols::rep3::{Rep3BigUintShare, Rep3PrimeFieldShare};

use crate::utils::instruction_utils::rep3_chunk_and_concatenate_operands;

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (remainder, divisor)
pub struct AssertValidSignedRemainderInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F>
    for AssertValidSignedRemainderInstruction<WORD_SIZE, F>
{
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref(vals, C, M);

        let left_msb = vals_by_subtable[0];
        let right_msb = vals_by_subtable[1];
        let eq = vals_by_subtable[2];
        let ltu = vals_by_subtable[3];
        let eq_abs = vals_by_subtable[4];
        let lt_abs = vals_by_subtable[5];
        let remainder_is_zero: F = vals_by_subtable[6].iter().product();
        let divisor_is_zero: F = vals_by_subtable[7].iter().product();

        // Accumulator for LTU(x_{<s}, y_{<s})
        let mut ltu_sum = lt_abs[0];
        // Accumulator for EQ(x_{<s}, y_{<s})
        let mut eq_prod = eq_abs[0];

        for (ltu_i, eq_i) in ltu.iter().zip(eq) {
            ltu_sum += *ltu_i * eq_prod;
            eq_prod *= *eq_i;
        }

        // (1 - x_s - y_s) * LTU(x_{<s}, y_{<s}) + x_s * y_s * (1 - EQ(x_{<s}, y_{<s})) + (1 - x_s) * y_s * EQ(x, 0) + EQ(y, 0)
        (F::one() - left_msb[0] - right_msb[0]) * ltu_sum
            + left_msb[0] * right_msb[0] * (F::one() - eq_prod)
            + (F::one() - left_msb[0]) * right_msb[0] * remainder_is_zero
            + divisor_is_zero
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C + 2
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LeftMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(RightMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(LtuSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(EqAbsSubtable::new()), SubtableIndices::from(0)),
            (Box::new(LtAbsSubtable::new()), SubtableIndices::from(0)),
            (
                Box::new(LeftIsZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
            (
                Box::new(RightIsZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0.as_public(), self.1.as_public(), C, log_M)
    }

    fn lookup_entry(&self) -> F {
        match WORD_SIZE {
            32 => {
                let remainder = self.0.as_public() as u32 as i32;
                let divisor = self.1.as_public() as u32 as i32;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    F::one()
                } else {
                    let remainder_sign = remainder >> 31;
                    let divisor_sign = divisor >> 31;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            64 => {
                let remainder = self.0.as_public() as i64;
                let divisor = self.1.as_public() as i64;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    F::one()
                } else {
                    let remainder_sign = remainder >> 63;
                    let divisor_sign = divisor >> 63;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self(
                (rng.next_u32() as u64).into(),
                (rng.next_u32() as u64).into(),
            ),
            64 => Self((rng.next_u64()).into(), (rng.next_u64()).into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F>
    for AssertValidSignedRemainderInstruction<WORD_SIZE, F>
{
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    #[tracing::instrument(
        skip_all,
        name = "AssertValidSignedRemainderInstruction::combine_lookups_rep3",
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

        let left_msb = vals_by_subtable[0];
        let right_msb = vals_by_subtable[1];
        let ltu = vals_by_subtable[3];
        let lt_abs = vals_by_subtable[5];

        #[cfg(not(feature = "public-eq"))]
        let (eq, eq_abs) = (vals_by_subtable[3], vals_by_subtable[5]);
        #[cfg(feature = "public-eq")]
        let (eq_abs, eq) = {
            let mut eq = rep3::arithmetic::open_vec(
                &[vals_by_subtable[3], &[vals_by_subtable[5][0]]].concat(),
                io_ctx,
            )?;
            (vec![eq.pop().unwrap()], eq)
        };

        let [remainder_is_zero, divisor_is_zero] =
            rep3::arithmetic::product_many(&vals_by_subtable[6..8], io_ctx)?
                .try_into()
                .unwrap();

        // Accumulator for LTU(x_{<s}, y_{<s}) * eq_eval * flag_eval
        let mut ltu_sum = lt_abs[0].into_additive();
        // Accumulator for EQ(x_{<s}, y_{<s}) * eq_eval * flag_eval
        let mut eq_prod = eq_abs[0];

        for (ltu, eq) in ltu.iter().zip(eq) {
            #[cfg(not(feature = "public-eq"))]
            {
                ltu_sum += *ltu * eq_prod;
                eq_prod = rep3::arithmetic::mul(eq_prod, *eq, io_ctx)?;
            }
            #[cfg(feature = "public-eq")]
            {
                ltu_sum += ltu.into_additive() * eq_prod;
                eq_prod *= eq;
            }
        }

        let not_left_msb = rep3::arithmetic::sub_public_by_shared(F::one(), left_msb[0], io_ctx.id);
        let not_left_msb_minus_right_msb =
            rep3::arithmetic::sub_public_by_shared(F::one(), left_msb[0] - right_msb[0], io_ctx.id);

        #[cfg(not(feature = "public-eq"))]
        let not_eq_prod = rep3::arithmetic::sub_public_by_shared(F::one(), eq_prod, io_ctx.id);
        #[cfg(feature = "public-eq")]
        let not_eq_prod = F::one() - eq_prod;

        let res = rep3::arithmetic::reshare_additive_many(
            &[
                ltu_sum,
                left_msb[0] * right_msb[0],
                not_left_msb * right_msb[0],
            ],
            io_ctx,
        )?;

        #[cfg(not(feature = "public-eq"))]
        let res = rep3::arithmetic::reshare_additive_many(
            &[
                res[0] * not_left_msb_minus_right_msb,
                res[1] * not_eq_prod,
                res[2] * remainder_is_zero,
            ],
            io_ctx,
        )?;

        #[cfg(feature = "public-eq")]
        let res = {
            let mut t = rep3::arithmetic::reshare_additive_many(
                &[
                    res[0] * not_left_msb_minus_right_msb,
                    res[2] * remainder_is_zero,
                ],
                io_ctx,
            )?;
            t.insert(1, res[1] * not_eq_prod);
            t
        };

        // (x_s * (1 - y_s) + EQ(x_s, y_s) * LTU(x_{<s}, y_{<s}))
        Ok(
            res[0] // (1 - x_s - y_s) * LTU(x_{<s}, y_{<s})
                + res[1] // x_s * y_s * (1 - EQ(x_{<s}, y_{<s}))
            + res[2] // (1 - x_s) * y_s * EQ(x, 0)
            + divisor_is_zero, // EQ(y, 0)
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "AssertValidSignedRemainderInstruction::combine_lookups_rep3_batched",
        level = "trace"
    )]
    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals_many: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        todo!()
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
