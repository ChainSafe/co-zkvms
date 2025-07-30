use jolt_core::field::JoltField;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::jolt::subtable::{
    eq::EqSubtable, eq_abs::EqAbsSubtable, left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable,
    ltu::LtuSubtable, right_msb::RightMSBSubtable, LassoSubtable,
};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        Rep3PrimeFieldShare,
    },
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::instruction_utils::{
    chunk_and_concatenate_operands, rep3_chunk_and_concatenate_operands,
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SLTInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for SLTInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("SLTInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);

        let left_msb = vals_by_subtable[0];
        let right_msb = vals_by_subtable[1];
        let ltu = vals_by_subtable[2];
        let eq = vals_by_subtable[3];
        let lt_abs = vals_by_subtable[4];
        let eq_abs = vals_by_subtable[5];

        // Accumulator for LTU(x_{<s}, y_{<s})
        let mut ltu_sum = lt_abs[0];
        // Accumulator for EQ(x_{<s}, y_{<s})
        let mut eq_prod = eq_abs[0];

        for i in 0..C - 2 {
            ltu_sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }
        // Do not need to update `eq_prod` for the last iteration
        ltu_sum += ltu[C - 2] * eq_prod;

        // x_s * (1 - y_s) + EQ(x_s, y_s) * LTU(x_{<s}, y_{<s})
        left_msb[0] * (F::one() - right_msb[0])
            + (left_msb[0] * right_msb[0] + (F::one() - left_msb[0]) * (F::one() - right_msb[0]))
                * ltu_sum
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C + 1
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LeftMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(RightMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(LtuSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(1..C - 1)),
            (Box::new(LtAbsSubtable::new()), SubtableIndices::from(0)),
            (Box::new(EqAbsSubtable::new()), SubtableIndices::from(0)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_operands(*x, *y, C, log_M)
            }
            _ => panic!("SLTInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => ((*x as i32) < (*y as i32)).into(),
            _ => panic!("SLTInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl<F: JoltField> Rep3JoltInstruction<F> for SLTInstruction<F> {
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    #[tracing::instrument(skip_all, name = "SLTInstruction::combine_lookups", level = "trace")]
    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        eq_flag_eval: F,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<AdditiveShare<F>> {
        let vals_by_subtable = self.slice_values(vals, C, M);

        let left_msb = vals_by_subtable[0];
        let right_msb = vals_by_subtable[1];
        let ltu = vals_by_subtable[2];
        let lt_abs = vals_by_subtable[4];
        #[cfg(not(feature = "public-eq"))]
        let (eq, eq_abs) = (vals_by_subtable[3], vals_by_subtable[5]);
        #[cfg(feature = "public-eq")]
        let (eq_abs, eq) = {
            let eq = rep3::arithmetic::open_vec(
                &[vals_by_subtable[3], &[vals_by_subtable[5][0]]].concat(),
                io_ctx,
            )?;
            (vec![eq.pop().unwrap()], eq)
        };

        // Accumulator for LTU(x_{<s}, y_{<s})
        let mut ltu_sum = lt_abs[0].into_additive();
        // Accumulator for EQ(x_{<s}, y_{<s})
        let mut eq_prod = eq_abs[0];

        for i in 0..C - 2 {
            #[cfg(not(feature = "public-eq"))]
            {
                ltu_sum += ltu[i] * eq_prod;
                eq_prod = rep3::arithmetic::mul(eq_prod, eq[i], io_ctx)?;
            }
            #[cfg(feature = "public-eq")]
            {
                ltu_sum += rep3::arithmetic::mul_public(ltu[i], eq_prod).into_additive();
                eq_prod *= eq[i];
            }
        }

        #[cfg(not(feature = "public-eq"))]
        {
            ltu_sum += ltu[C - 2] * eq_prod;
        }
        #[cfg(feature = "public-eq")]
        {
            ltu_sum += rep3::arithmetic::mul_public(ltu[C - 2], eq_prod).into_additive();
        }


        // x_s * (1 - y_s) + EQ(x_s, y_s) * LTU(x_{<s}, y_{<s})

        let not_left_msb = rep3::arithmetic::sub_public_by_shared(F::one(), left_msb[0], io_ctx.id);
        let not_right_msb =
            rep3::arithmetic::sub_public_by_shared(F::one(), right_msb[0], io_ctx.id);

        let left_msb_toggled = rep3::arithmetic::mul_public(left_msb[0], eq_flag_eval);

        let res = rep3::arithmetic::reshare_additive_many(
            &[
                left_msb[0] * right_msb[0],
                not_left_msb * not_right_msb,
                ltu_sum,
            ],
            io_ctx,
        )?;

        let ltu_sum_toggled = rep3::arithmetic::mul_public(res[2], eq_flag_eval);

        Ok(
            left_msb_toggled * not_right_msb // x_s * (1 - y_s) * eq_eval * flag_eval
            + res[0] * ltu_sum_toggled // x_s * y_s * LTU(x_{<s}, y_{<s}) * eq_eval * flag_eval
            + res[1] * ltu_sum_toggled, // (1 - x_s) * (1 - y_s) * LTU(x_{<s}, y_{<s}) * eq_eval * flag_eval
        )
    }

    fn to_indices_rep3(
        &self,
        C: usize,
        log_M: usize,
    ) -> Vec<mpc_core::protocols::rep3::Rep3BigUintShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Binary(x), Rep3Operand::Binary(y)) => {
                rep3_chunk_and_concatenate_operands(x.clone(), y.clone(), C, log_M)
            }
            _ => panic!("SLTInstruction::to_indices called with non-binary operands"),
        }
    }

    fn output<N: Rep3Network>(&self, _: &mut IoContext<N>) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Binary(x), Rep3Operand::Binary(y)) => {
                unimplemented!()
            }
            _ => panic!("SLTInstruction::output called with non-binary operands"),
        }
    }
}
