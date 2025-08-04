use itertools::{Itertools, multizip};
use mpc_core::protocols::additive::AdditiveShare;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::jolt::subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable};
use jolt_core::utils::instruction_utils::chunk_and_concatenate_operands;
use jolt_core::{
    field::JoltField, jolt::subtable::right_is_zero::RightIsZeroSubtable, utils::uninterleave_bits,
};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};

use crate::utils::instruction_utils::{double_transpose, rep3_chunk_and_concatenate_operands, transpose};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssertValidUnsignedRemainderInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F>
    for AssertValidUnsignedRemainderInstruction<WORD_SIZE, F>
{
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref(vals, C, M);
        let ltu = vals_by_subtable[0];
        let eq = vals_by_subtable[1];
        let divisor_is_zero: F = vals_by_subtable[2].iter().product();

        let mut sum = F::zero();
        let mut eq_prod = F::one();

        for i in 0..C - 1 {
            sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }
        // LTU(x, y) + EQ(y, 0)
        sum + ltu[C - 1] * eq_prod + divisor_is_zero
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C - 1)),
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
        // Same for both 32-bit and 64-bit word sizes
        let remainder = self.0.as_public();
        let divisor = self.1.as_public();
        (divisor == 0 || remainder < divisor).into()
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
    for AssertValidUnsignedRemainderInstruction<WORD_SIZE, F>
{
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    #[tracing::instrument(
        skip_all,
        name = "AssertValidUnsignedRemainderInstruction::combine_lookups_rep3",
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
        let ltu = vals_by_subtable[0];
        #[cfg(not(feature = "public-eq"))]
        let eq = vals_by_subtable[1];
        #[cfg(feature = "public-eq")]
        let eq = rep3::arithmetic::open_vec(&vals_by_subtable[1], io_ctx)?;

        #[cfg(not(feature = "public-eq"))]
        let divisor_is_zero = rep3::arithmetic::product(vals_by_subtable[2], io_ctx)?;
        #[cfg(feature = "public-eq")]
        let divisor_is_zero = {
            let divisor_is_zero_vals = rep3::arithmetic::open_vec(&vals_by_subtable[2], io_ctx)?;
            rep3::arithmetic::promote_to_trivial_share(
                io_ctx.id,
                divisor_is_zero_vals.iter().product::<F>(),
            )
        };

        let mut sum = ltu[0].into_additive();
        let mut eq_prod = eq[0];

        for i in 1..C - 1 {
            #[cfg(not(feature = "public-eq"))]
            {
                sum += ltu[i] * eq_prod;
                eq_prod = rep3::arithmetic::mul(eq_prod, eq[i], io_ctx)?;
            }
            #[cfg(feature = "public-eq")]
            {
                sum += rep3::arithmetic::mul_public(ltu[i], eq_prod).into_additive();
                eq_prod *= eq[i];
            }
        }
        #[cfg(not(feature = "public-eq"))]
        let ltu_sum_eq_prod = ltu[C - 1] * eq_prod;
        #[cfg(feature = "public-eq")]
        let ltu_sum_eq_prod = rep3::arithmetic::mul_public(ltu[C - 1], eq_prod).into_additive();

        Ok(rep3::arithmetic::reshare_additive(sum + ltu_sum_eq_prod, io_ctx)? + divisor_is_zero)
    }

    // fn combine_lookups_rep3_batched<N: Rep3Network>(
    //     &self,
    //     vals_many: Vec<Vec<Rep3PrimeFieldShare<F>>>,
    //     C: usize,
    //     M: usize,
    //     io_ctx: &mut IoContext<N>,
    // ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    //     #[cfg(feature = "public-eq")]
    //     let terms_len = vals_many.len();
    //     let mut val_bathes_by_subtable = transpose(
    //         self.slice_values(vals_many, C, M),
    //     );

    //     let ltu = std::mem::take(&mut val_bathes_by_subtable[0]);
    //     #[cfg(not(feature = "public-eq"))]
    //     let mut eq = std::mem::take(&mut val_bathes_by_subtable[1]);
    //     #[cfg(feature = "public-eq")]
    //     let mut eq: Vec<_> =
    //         rep3::arithmetic::open_vec(&std::mem::take(&mut val_bathes_by_subtable[1]).concat(), io_ctx)?
    //             .chunks(terms_len)
    //             .map(|vals| vals.to_vec())
    //             .collect();

    //     #[cfg(not(feature = "public-eq"))]
    //     let divisor_is_zero = rep3::arithmetic::product_many(
    //         &std::mem::take(&mut val_bathes_by_subtable[2]),
    //         io_ctx,
    //     )?;
    //     #[cfg(feature = "public-eq")]
    //     let divisor_is_zero = {
    //         rep3::arithmetic::open_vec(&vals_by_subtable_by_term[2].concat(), io_ctx)?
    //             .chunks(C)
    //             .map(|vals| {
    //                 rep3::arithmetic::promote_to_trivial_share(
    //                     io_ctx.id,
    //                     vals.iter().product::<F>(),
    //                 )
    //             })
    //             .collect_vec()
    //     };

    //     let mut sums = ltu[0].iter().map(|x| x.into_additive()).collect::<Vec<_>>();
    //     let mut eq_prods = std::mem::take(&mut eq[0]);

    //     for i in 1..C - 1 {
    //         #[cfg(not(feature = "public-eq"))]
    //         {
    //             multizip((sums.iter_mut(), ltu[i].iter(), eq_prods.iter())).for_each(|(sum, ltu_i, eq_prod)| {
    //                 *sum += *ltu_i * *eq_prod;
    //             });
    //             eq_prods = rep3::arithmetic::mul_vec(&eq_prods, &eq[i], io_ctx)?;
    //         }
    //         #[cfg(feature = "public-eq")]
    //         {
    //             multizip((sums.iter_mut(), ltu[i].iter(), eq_prods.iter())).for_each(|(sum, ltu_i, eq_prod)| {
    //                 *sum += rep3::arithmetic::mul_public(*ltu_i, *eq_prod).into_additive();
    //             });
    //             eq_prods
    //                 .iter_mut()
    //                 .zip(eq[i].iter())
    //                 .for_each(|(eq_prod, eq_i)| {
    //                     *eq_prod *= *eq_i;
    //                 });
    //         }
    //     }

    //     #[cfg(not(feature = "public-eq"))]
    //     let ltu_sum_eq_prod = ltu[C - 1]
    //         .iter()
    //         .zip_eq(eq_prods.into_iter())
    //         .map(|(ltu, eq_prod)| *ltu * eq_prod)
    //         .collect::<Vec<_>>();
    //     #[cfg(feature = "public-eq")]
    //     let ltu_sum_eq_prod = ltu
    //         .iter()
    //         .zip(eq_prods.into_iter())
    //         .map(|(ltu, eq_prod)| rep3::arithmetic::mul_public(ltu[C - 1], eq_prod).into_additive())
    //         .collect::<Vec<_>>();

    //     let res = rep3::arithmetic::reshare_additive_many(
    //         &sums
    //             .iter()
    //             .zip_eq(ltu_sum_eq_prod.iter())
    //             .map(|(sum, ltu_sum_eq_prod)| *sum + ltu_sum_eq_prod)
    //             .collect::<Vec<_>>(),
    //         io_ctx,
    //     )?
    //     .into_iter()
    //     .zip_eq(divisor_is_zero.into_iter())
    //     .map(|(sum, divisor_is_zero)| sum + divisor_is_zero)
    //     .collect::<Vec<_>>();

    //     Ok(res)
    // }

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
