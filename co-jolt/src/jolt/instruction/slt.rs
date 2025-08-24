use crate::field::JoltField;
use itertools::{chain, izip, multizip, Itertools};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::jolt::subtable::{
    eq::EqSubtable, eq_abs::EqAbsSubtable, left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable,
    ltu::LtuSubtable, right_msb::RightMSBSubtable, LassoSubtable,
};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare, Rep3PrimeFieldShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::future::FutureVal;
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
        let vals_by_subtable = self.slice_values_ref(vals, C, M);

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
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        let vals_by_subtable = self.slice_values_ref(vals, C, M);

        let left_msb = vals_by_subtable[0];
        let right_msb = vals_by_subtable[1];
        let ltu = vals_by_subtable[2];
        let lt_abs = vals_by_subtable[4];
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

        let not_left_msb = rep3::arithmetic::sub_public_by_shared(F::one(), left_msb[0], io_ctx.id);
        let not_right_msb =
            rep3::arithmetic::sub_public_by_shared(F::one(), right_msb[0], io_ctx.id);

        let res = rep3::arithmetic::reshare_additive_many(
            &[
                left_msb[0] * not_right_msb,
                left_msb[0] * right_msb[0],
                not_left_msb * not_right_msb,
                ltu_sum,
            ],
            io_ctx,
        )?;

        // x_s * (1 - y_s) + EQ(x_s, y_s) * LTU(x_{<s}, y_{<s})
        Ok(res[0] + rep3::arithmetic::mul(res[1] + res[2], res[3], io_ctx)?)
    }

    #[tracing::instrument(
        skip_all,
        name = "SLTInstruction::combine_lookups_rep3_batched",
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
        let mut vals_by_subtable_by_term = self.slice_values(vals_many, C, M);

        #[cfg(not(feature = "public-eq"))]
        let (eq, eq_abs) = (
            vals_by_subtable_by_term.remove(3),
            vals_by_subtable_by_term.remove(4).pop().unwrap(), // fifth subtable
        );
        #[cfg(feature = "public-eq")]
        let (eq, eq_abs) = {
            let eq_abs = vals_by_subtable_by_term.remove(5);
            let eq = vals_by_subtable_by_term.remove(3);
            let mut openned =
                rep3::arithmetic::open_vec::<F, _>(chain![&eq, &eq_abs].flatten(), io_ctx)?;

            let eq = vec![
                openned.drain(..batch_size).collect::<Vec<_>>(),
                openned.drain(..batch_size).collect::<Vec<_>>(),
            ];
            let eq_abs = openned.drain(..).collect::<Vec<_>>();
            assert_eq!(eq_abs.len(), batch_size);

            (eq, eq_abs)
        };

        let [left_msb, right_msb, ltu, lt_abs] = vals_by_subtable_by_term.try_into().unwrap();

        // Accumulator for LTU(x_{<s}, y_{<s})
        let mut ltu_sums = lt_abs[0]
            .iter()
            .map(|x| x.into_additive())
            .collect::<Vec<_>>();
        // Accumulator for EQ(x_{<s}, y_{<s})
        let mut eq_prods = eq_abs;

        for i in 0..C - 2 {
            #[cfg(not(feature = "public-eq"))]
            {
                multizip((ltu_sums.iter_mut(), ltu[i].iter(), eq_prods.iter())).for_each(
                    |(sum, ltu_i, eq_prod)| {
                        *sum += *ltu_i * *eq_prod;
                    },
                );
                eq_prods = rep3::arithmetic::mul_vec(&eq_prods, &eq[i], io_ctx)?;
            }
            #[cfg(feature = "public-eq")]
            {
                multizip((ltu_sums.iter_mut(), ltu[i].iter(), eq_prods.iter())).for_each(
                    |(sum, ltu_i, eq_prod)| {
                        *sum += rep3::arithmetic::mul_public(*ltu_i, *eq_prod).into_additive();
                    },
                );

                izip!(eq_prods.iter_mut(), eq[i].iter()).for_each(|(eq_prod, eq_i)| {
                    *eq_prod *= *eq_i;
                });
            }
        }

        #[cfg(not(feature = "public-eq"))]
        let ltu_sum_eq_prod = izip!(ltu_sums, &ltu[C - 2], eq_prods)
            .map(|(sum, ltu, eq_prod)| sum + *ltu * eq_prod)
            .collect::<Vec<_>>();
        #[cfg(feature = "public-eq")]
        let ltu_sum_eq_prod = izip!(ltu_sums, &ltu[C - 2], eq_prods)
            .map(|(sum, ltu, eq_prod)| {
                sum + rep3::arithmetic::mul_public(*ltu, eq_prod).into_additive()
            })
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
        _: &Rep3BigUintShare<F>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3BigUintShare<F>> {
        rep3_chunk_and_concatenate_operands(
            self.0.as_binary_share(),
            self.1.as_binary_share(),
            C,
            log_M,
        )
    }

    fn output<N: Rep3Network>(&self, _: &mut IoContext<N>) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        unimplemented!()
    }

    fn output_batched<N: Rep3Network>(
        &self,
        steps: &[Self],
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<FutureVal<F, Rep3PrimeFieldShare<F>>>> {
        let (a, b): (Vec<_>, Vec<_>) = steps
            .into_iter()
            .map(|Self(x, y)| (x.as_binary_share(), y.as_binary_share())) // TODO: as i32
            .unzip();

        // a < b is equivalent to !(a >= b)
        let tmp = rep3::arithmetic::ge_many(&a, &b, io_ctx)?;
        Ok(tmp
            .into_iter()
            .map(|x| {
                FutureVal::Ready(rep3::arithmetic::sub_public_by_shared(
                    F::one(),
                    x,
                    io_ctx.id,
                ))
            })
            .collect())
    }
}
