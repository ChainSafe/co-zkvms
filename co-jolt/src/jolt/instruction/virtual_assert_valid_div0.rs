use std::mem;

use itertools::izip;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};
use jolt_core::{
    jolt::subtable::LassoSubtable,
    jolt::subtable::{div_by_zero::DivByZeroSubtable, left_is_zero::LeftIsZeroSubtable},
    utils::instruction_utils::chunk_and_concatenate_operands,
};
use mpc_core::protocols::{rep3::Rep3PrimeFieldShare, rep3_ring};
use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network},
    },
    rep3_ring::{ring::bit::Bit, Rep3RingShare},
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::instruction_utils::rep3_chunk_and_concatenate_operands;

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (divisor, quotient)
pub struct AssertValidDiv0Instruction<const WORD_SIZE: usize>(pub Rep3Operand, pub Rep3Operand);

impl<const WORD_SIZE: usize> JoltInstruction for AssertValidDiv0Instruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values_ref::<F, _>(vals, C, M);
        let divisor_is_zero: F = vals_by_subtable[0].iter().product();
        let is_valid_div_by_zero: F = vals_by_subtable[1].iter().product();

        F::one() - divisor_is_zero + is_valid_div_by_zero
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
            (
                Box::new(LeftIsZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
            (
                Box::new(DivByZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0.as_public(), self.1.as_public(), C, log_M)
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        let divisor = self.0.as_public();
        let quotient = self.1.as_public();
        if divisor == 0 {
            match WORD_SIZE {
                32 => (quotient == u32::MAX as u64).into(),
                64 => (quotient == u64::MAX).into(),
                _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
            }
        } else {
            F::one()
        }
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

impl<const WORD_SIZE: usize> Rep3JoltInstruction for AssertValidDiv0Instruction<WORD_SIZE> {
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
        name = "AssertValidDiv0Instruction::combine_lookups_rep3_batched",
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

        let products = rep3::arithmetic::product_many(
            (0..C).map(|i| {
                (0..batch_size)
                    .flat_map(|j| {
                        [
                            mem::take(&mut batched_vals_by_subtable[0][i][j]),
                            mem::take(&mut batched_vals_by_subtable[1][i][j]),
                        ]
                    })
                    .collect::<Vec<_>>()
            }),
            io_ctx,
        )?;
        let res = products
            .chunks(2)
            .map(|chunk| {
                let [divisor_is_zero, is_valid_div_by_zero] = chunk.try_into().unwrap();
                rep3::arithmetic::sub_public_by_shared(
                    F::one(),
                    divisor_is_zero + is_valid_div_by_zero,
                    io_ctx.id,
                )
            })
            .collect::<Vec<_>>();

        return Ok(res);
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
        out: impl IntoIterator<
            Item = &'a mut crate::utils::future_ring::FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>,
        >,
    ) -> eyre::Result<()> {
        let (devisors, quotients): (Vec<_>, Vec<_>) = steps
            .into_iter()
            .map(|st| (st.lhs().as_binary(), st.rhs().unwrap().as_arithmetic_u32()))
            .unzip();
        // let one = rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, 1u32.into());
        let one = rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, Bit::one().into());
        let max = rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, u32::MAX.into());
        // let devisor_is_zero = rep3_ring::conversion::bit_inject_many(
        //     &rep3_ring::binary::is_zero_many(&devisors, io_ctx)?,
        //     io_ctx,
        // )?;
        let devisor_is_zero = rep3_ring::binary::is_zero_many(&devisors, io_ctx)?;
        let quotient_eq_max =
            rep3_ring::arithmetic::eq_many(&quotients, &vec![max; quotients.len()], io_ctx)?;

        let res = rep3_ring::binary::cmux_many(
            &devisor_is_zero,
            &quotient_eq_max,
            &vec![one; quotients.len()],
            io_ctx,
        )?;
        izip!(out, res).for_each(|(out, res)| {
            *out = FutureRep3Ring::bit_inject_to_field(res);
        });

        Ok(())
    }
}
