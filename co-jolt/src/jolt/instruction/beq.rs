use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use eyre::Context;

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand};
use crate::field::JoltField;
use crate::utils::future::FutureVal;
use jolt_core::jolt::subtable::{eq::EqSubtable, LassoSubtable};

use mpc_core::protocols::rep3::{
    self, Rep3BigUintShare,
    network::{IoContext, Rep3Network},
    Rep3PrimeFieldShare,
};

#[cfg(feature = "public-eq")]
use crate::utils::instruction_utils::transpose;
use crate::{
    jolt::instruction::SubtableIndices,
    utils::instruction_utils::{
        chunk_and_concatenate_operands, rep3_chunk_and_concatenate_operands,
    },
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BEQInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for BEQInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("BEQInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], _: usize, _: usize) -> F {
        vals.iter().product::<F>()
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables(&self, C: usize, _: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(EqSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_operands(*x, *y, C, log_M)
            }
            _ => panic!("BEQInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x == *y).into(),
            _ => panic!("BEQInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl<F: JoltField> Rep3JoltInstruction<F> for BEQInstruction<F> {
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    #[tracing::instrument(skip_all, name = "BEQInstruction::combine_lookups", level = "trace")]
    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        _C: usize,
        _M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        #[cfg(feature = "public-eq")]
        {
            let opened = rep3::arithmetic::open_vec(vals, io_ctx)?;
            return Ok(rep3::arithmetic::promote_to_trivial_share(
                io_ctx.id,
                opened.iter().product::<F>(),
            ));
        }

        #[cfg(not(feature = "public-eq"))]
        rep3::arithmetic::product(vals, io_ctx)
    }

    #[tracing::instrument(
        skip_all,
        name = "BEQInstruction::combine_lookups_rep3_batched",
        level = "trace"
    )]
    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        vals_many: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        _C: usize,
        _M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        #[cfg(feature = "public-eq")]
        {
            use crate::utils::instruction_utils::chunks_take_nth;

            return Ok(chunks_take_nth(
                &rep3::arithmetic::open_vec(&vals_many.concat(), io_ctx)?,
                vals_many.len(),
                vals_many[0].len(),
            )
            .map(|chunk| {
                rep3::arithmetic::promote_to_trivial_share(io_ctx.id, chunk.product::<F>())
            })
            .collect::<Vec<_>>());
        }

        #[cfg(not(feature = "public-eq"))]
        rep3::arithmetic::product_many(vals_many, io_ctx)
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
            .map(|Self(x, y)| (x.as_arithmetic_share(), y.as_arithmetic_share()))
            .unzip();

        Ok(rep3::arithmetic::eq_many(&a, &b, io_ctx)
            .context("BEQInstruction::output_batched")?
            .into_iter()
            .map(FutureVal::Ready)
            .collect())
    }
}
