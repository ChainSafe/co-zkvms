use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};
use itertools::multizip;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::jolt::subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable};
use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        Rep3PrimeFieldShare,
    },
    rep3_ring::{self, Rep3RingShare},
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand};
use crate::{
    jolt::instruction::SubtableIndices,
    utils::instruction_utils::{
        chunk_and_concatenate_operands, rep3_chunk_and_concatenate_operands,
    },
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SLTUInstruction(pub Rep3Operand, pub Rep3Operand);

impl JoltInstruction for SLTUInstruction {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("SLTUInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref::<F, _>(vals, C, M);
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

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
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

    fn lookup_entry<F: JoltField>(&self) -> F {
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

impl Rep3JoltInstruction for SLTUInstruction {
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
        name = "SLTUInstruction::combine_lookups_rep3_batched",
        level = "trace"
    )]
    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals_many: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        let mut batched_vals_by_subtable = self.slice_values::<F, _>(vals_many, C, M);

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
            .map(|st| (st.lhs().as_binary(), st.rhs().unwrap().as_binary()))
            .unzip();

        // a < b is equivalent to !(a >= b)
        let tmp = rep3_ring::arithmetic::ge_many(&a, &b, io_ctx)?;
        tmp.into_iter().zip(out).for_each(|(x, out)| {
            *out = FutureRep3Ring::bit_inject_to_field(!x);
        });
        Ok(())
    }
}
