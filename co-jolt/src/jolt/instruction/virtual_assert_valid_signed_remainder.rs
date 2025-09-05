use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::field::JoltField;

use jolt_core::{
    jolt::subtable::right_is_zero::RightIsZeroSubtable,
    jolt::subtable::{
        eq::EqSubtable, eq_abs::EqAbsSubtable, left_is_zero::LeftIsZeroSubtable,
        left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable, ltu::LtuSubtable,
        right_msb::RightMSBSubtable, LassoSubtable,
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::{
    rep3::network::{IoContext, Rep3Network},
    rep3_ring::Rep3RingShare,
};

use crate::utils::instruction_utils::rep3_chunk_and_concatenate_operands;

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (remainder, divisor)
pub struct AssertValidSignedRemainderInstruction<const WORD_SIZE: usize>(
    pub Rep3Operand,
    pub Rep3Operand,
);

impl<const WORD_SIZE: usize> JoltInstruction for AssertValidSignedRemainderInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref::<F, _>(vals, C, M);

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

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
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

    fn lookup_entry<F: JoltField>(&self) -> F {
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

impl<const WORD_SIZE: usize> Rep3JoltInstruction
    for AssertValidSignedRemainderInstruction<WORD_SIZE>
{
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

    #[tracing::instrument(
        skip_all,
        name = "AssertValidSignedRemainderInstruction::combine_lookups_rep3_batched",
        level = "trace"
    )]
    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals_many: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        todo!()
    }

    fn to_indices_rep3(
        &self,
        _: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        rep3_chunk_and_concatenate_operands(self.0.as_binary(), self.1.as_binary(), C, log_M)
    }

    // fn output_batched<'a, F: JoltField, N: Rep3Network>(
    //     &self,
    //     steps: &[&impl Rep3JoltInstruction],
    //     io_ctx: &mut IoContext<N>,
    //     out: impl IntoIterator<
    //         Item = &'a mut crate::utils::future_ring::FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>,
    //     >,
    // ) -> eyre::Result<()> {

    //     Ok(())
    // }
}
