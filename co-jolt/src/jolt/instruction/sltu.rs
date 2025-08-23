use itertools::multizip;
use crate::field::JoltField;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::jolt::subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3PrimeFieldShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand};
use crate::{
    jolt::instruction::SubtableIndices,
    utils::instruction_utils::{
        chunk_and_concatenate_operands, rep3_chunk_and_concatenate_operands,
    },
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SLTUInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for SLTUInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("SLTUInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref(vals, C, M);
        let ltu = vals_by_subtable[0];
        let eq = vals_by_subtable[1];

        let mut sum = F::zero();
        let mut eq_prod = F::one();

        for i in 0..C - 1 {
            sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }
        // Do not need to update `eq_prod` for the last iteration
        sum + ltu[C - 1] * eq_prod
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C - 1)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_operands(*x, *y, C, log_M)
            }
            _ => panic!("SLTUInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x < *y).into(),
            _ => panic!("SLTUInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl<F: JoltField> Rep3JoltInstruction<F> for SLTUInstruction<F> {
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    #[tracing::instrument(skip_all, name = "SLTUInstruction::combine_lookups", level = "trace")]
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
        let eq = rep3::arithmetic::open_vec(vals_by_subtable[1], io_ctx)?;

        #[cfg(not(feature = "public-eq"))]
        let mut sum = ltu[0].into_additive();
        #[cfg(feature = "public-eq")]
        let mut sum = ltu[0];
        let mut eq_prod = eq[0];

        for i in 1..C - 1 {
            #[cfg(not(feature = "public-eq"))]
            {
                sum += ltu[i] * eq_prod;
                eq_prod = rep3::arithmetic::mul(eq_prod, eq[i], io_ctx)?;
            }
            #[cfg(feature = "public-eq")]
            {
                sum += rep3::arithmetic::mul_public(ltu[i], eq_prod);
                eq_prod *= eq[i];
            }
        }

        #[cfg(not(feature = "public-eq"))]
        return rep3::arithmetic::reshare_additive(sum + ltu[C - 1] * eq_prod, io_ctx);

        #[cfg(feature = "public-eq")]
        Ok(sum + (ltu[C - 1] * eq_prod))
    }

    #[tracing::instrument(
        skip_all,
        name = "SLTUInstruction::combine_lookups_rep3_batched",
        level = "trace"
    )]
    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals_many: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        let mut batched_vals_by_subtable = self.slice_values(vals_many, C, M);

        let ltu = std::mem::take(&mut batched_vals_by_subtable[0]);
        let mut eq = std::mem::take(&mut batched_vals_by_subtable[1]);

        let mut sums = ltu[0].iter().map(|x| x.into_additive()).collect::<Vec<_>>();
        let mut eq_prods = std::mem::take(&mut eq[0]);

        for i in 1..C - 1 {
            multizip((sums.iter_mut(), ltu[i].iter(), eq_prods.iter())).for_each(
                |(sum, ltu_i, eq_prod)| {
                    *sum += *ltu_i * *eq_prod;
                },
            );
            eq_prods = rep3::arithmetic::mul_vec(&eq_prods, &eq[i], io_ctx)?;
        }

        rep3::arithmetic::reshare_additive_many(
            &itertools::multizip((sums, &ltu[C - 1], eq_prods))
                .map(|(sum, ltu, eq_prod)| sum + *ltu * eq_prod)
                .collect::<Vec<_>>(),
            io_ctx,
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
            _ => panic!("SLTUInstruction::to_indices called with non-binary operands"),
        }
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Binary(x), Rep3Operand::Binary(y)) => {
                unimplemented!()
            }
            _ => panic!("SLTUInstruction::output called with non-binary operands"),
        }
    }
}
