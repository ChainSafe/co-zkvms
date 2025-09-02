use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Context;
#[cfg(feature = "public-eq")]
use mpc_core::protocols::additive;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::utils::future::FutureRep3;
use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};
use itertools::multizip;
use jolt_core::{
    jolt::subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    utils::instruction_utils::chunk_and_concatenate_operands,
};
use mpc_core::protocols::{
    rep3::network::{IoContext, Rep3Network},
    rep3_ring,
};
use mpc_core::protocols::{
    rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare},
    rep3_ring::Rep3RingShare,
};

use crate::utils::instruction_utils::rep3_chunk_and_concatenate_operands;

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ASSERTLTEInstruction<const WORD_SIZE: usize>(pub Rep3Operand, pub Rep3Operand);

impl<const WORD_SIZE: usize> JoltInstruction for ASSERTLTEInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref::<F, _>(vals, C, M);
        let ltu = vals_by_subtable[0];
        let eq = vals_by_subtable[1];

        // Accumulator for LTU(x, y)
        let mut ltu_sum = F::zero();
        // Accumulator for EQ(x, y)
        let mut eq_prod = F::one();

        for i in 0..C {
            ltu_sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }

        // LTU(x,y) || EQ(x,y)
        ltu_sum + eq_prod
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
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0.as_public(), self.1.as_public(), C, log_M)
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        (self.0.as_public() <= self.1.as_public()).into()
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

impl<const WORD_SIZE: usize> Rep3JoltInstruction for ASSERTLTEInstruction<WORD_SIZE> {
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
        name = "ASSERTLTEInstruction::combine_lookups_rep3_batched",
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
        let mut batched_vals_by_subtable = self.slice_values::<F, _>(vals_many, C, M);
        let ltu = std::mem::take(&mut batched_vals_by_subtable[0]);
        #[cfg(not(feature = "public-eq"))]
        let mut eq = std::mem::take(&mut batched_vals_by_subtable[1]);
        #[cfg(feature = "public-eq")]
        let mut eq = rep3::arithmetic::open_vec(&batched_vals_by_subtable[1].concat(), io_ctx)?
            .chunks(batch_size)
            .map(|vals| vals.to_vec())
            .collect::<Vec<_>>();

        // Accumulator for LTU(x, y)
        #[cfg(not(feature = "public-eq"))]
        let mut ltu_sums = ltu[0].iter().map(|x| x.into_additive()).collect::<Vec<_>>();
        #[cfg(feature = "public-eq")]
        let mut ltu_sums = std::mem::take(&mut ltu[0]);
        // Accumulator for EQ(x, y)
        let mut eq_prods = std::mem::take(&mut eq[0]);

        for i in 1..C {
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
                        *sum += rep3::arithmetic::mul_public(*ltu_i, *eq_prod);
                    },
                );
                eq_prods
                    .iter_mut()
                    .zip(eq[i].iter())
                    .for_each(|(eq_prod, eq_i)| {
                        *eq_prod *= *eq_i;
                    });
            }
        }
        #[cfg(not(feature = "public-eq"))]
        return rep3::arithmetic::reshare_additive_many(
            &itertools::multizip((ltu_sums, eq_prods))
                .map(|(sum, eq_prod)| sum + eq_prod.into_additive())
                .collect::<Vec<_>>(),
            io_ctx,
        );

        // #[cfg(not(feature = "public-eq"))]
        // return rep3::arithmetic::reshare_additive(ltu_sum + eq_prod.into_additive(), io_ctx);
        #[cfg(feature = "public-eq")]
        Ok(itertools::multizip((ltu_sums, eq_prods))
            .map(|(sum, eq_prod)| rep3::arithmetic::add_public(sum, eq_prod, io_ctx.id))
            .collect::<Vec<_>>())
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

        // a <= b is equivalent to b >= a
        rep3_ring::arithmetic::ge_many(&b, &a, io_ctx)
            .context("ASSERTLTEInstruction::output_batched")?
            .into_iter()
            .zip(out)
            .for_each(|(r, out)| {
                *out = FutureRep3Ring::bit_inject_to_field(r);
            });
        Ok(())
    }
}
