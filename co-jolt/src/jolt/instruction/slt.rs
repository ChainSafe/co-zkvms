use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};
use itertools::{chain, izip, multizip};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::jolt::subtable::{
    eq::EqSubtable, eq_abs::EqAbsSubtable, left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable,
    ltu::LtuSubtable, right_msb::RightMSBSubtable, LassoSubtable,
};
use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        Rep3PrimeFieldShare,
    },
    rep3_ring::{self, Rep3RingShare},
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::instruction_utils::{
    chunk_and_concatenate_operands, rep3_chunk_and_concatenate_operands,
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SLTInstruction(pub Rep3Operand, pub Rep3Operand);

impl JoltInstruction for SLTInstruction {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("SLTInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref::<F, _>(vals, C, M);

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

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
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

    fn lookup_entry<F: JoltField>(&self) -> F {
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

impl Rep3JoltInstruction for SLTInstruction {
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
        name = "SLTInstruction::combine_lookups_rep3_batched",
        level = "trace"
    )]
    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals_many: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        let batch_size = vals_many[0].len();
        let mut vals_by_subtable_by_term = self.slice_values::<F, _>(vals_many, C, M);

        let (eq, eq_abs) = (
            vals_by_subtable_by_term.remove(3),
            vals_by_subtable_by_term.remove(4).pop().unwrap(), // fifth subtable
        );

        let [left_msb, right_msb, ltu, lt_abs] = vals_by_subtable_by_term.try_into().unwrap();

        // Accumulator for LTU(x_{<s}, y_{<s})
        let mut ltu_sums = lt_abs[0]
            .iter()
            .map(|x| x.into_additive())
            .collect::<Vec<_>>();
        // Accumulator for EQ(x_{<s}, y_{<s})
        let mut eq_prods = eq_abs;

        for i in 0..C - 2 {
            multizip((ltu_sums.iter_mut(), ltu[i].iter(), eq_prods.iter())).for_each(
                |(sum, ltu_i, eq_prod)| {
                    *sum += *ltu_i * *eq_prod;
                },
            );
            eq_prods = rep3::arithmetic::mul_vec(&eq_prods, &eq[i], io_ctx)?;
        }

        let ltu_sum_eq_prod = izip!(ltu_sums, &ltu[C - 2], eq_prods)
            .map(|(sum, ltu, eq_prod)| sum + *ltu * eq_prod)
            .collect::<Vec<_>>();

        let not_left_msb = left_msb[0]
            .iter()
            .map(|x| rep3::arithmetic::sub_public_by_shared(F::one(), *x, io_ctx.id))
            .collect::<Vec<_>>();
        let not_right_msb = right_msb[0]
            .iter()
            .map(|y| rep3::arithmetic::sub_public_by_shared(F::one(), *y, io_ctx.id))
            .collect::<Vec<_>>();

        let res = rep3::arithmetic::reshare_additive_many(
            &chain![
                izip!(left_msb[0].iter(), not_right_msb.iter()).map(|(x, y)| x * y),
                izip!(left_msb[0].iter(), right_msb[0].iter()).map(|(x, y)| x * y),
                izip!(not_left_msb.iter(), not_right_msb.iter()).map(|(x, y)| x * y),
                ltu_sum_eq_prod,
            ]
            .collect::<Vec<_>>(),
            io_ctx,
        )?
        .chunks(batch_size)
        .map(|x| x.to_vec())
        .collect::<Vec<_>>();

        let [left_not_right, left_right, not_left_not_right, ltu_sum_eq_prod] =
            res.try_into().unwrap();

        // x_s * (1 - y_s) + EQ(x_s, y_s) * LTU(x_{<s}, y_{<s})
        let res = izip!(
            left_not_right,
            rep3::arithmetic::mul_vec(
                &izip!(left_right, not_left_not_right)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<_>>(),
                &ltu_sum_eq_prod,
                io_ctx,
            )?
        )
        .map(|(x, y)| x + y)
        .collect::<Vec<_>>();

        Ok(res)
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
        let (a, b): (Vec<_>, Vec<_>) = steps
            .into_iter()
            .map(|st| (st.lhs().as_binary(), st.rhs().unwrap().as_binary())) // TODO: as i32
            .unzip();

        // a < b is equivalent to !(a >= b)
        let tmp = rep3_ring::arithmetic::ge_many(&a, &b, io_ctx)?;
        tmp.into_iter()
            .zip(out)
            .for_each(|(x, out)| *out = FutureRep3Ring::bit_inject_to_field(!x));
        Ok(())
    }
}
