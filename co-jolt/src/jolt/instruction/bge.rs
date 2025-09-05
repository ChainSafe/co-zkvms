use eyre::Context;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};
use jolt_core::jolt::subtable::{
    eq::EqSubtable, eq_abs::EqAbsSubtable, left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable,
    ltu::LtuSubtable, right_msb::RightMSBSubtable, LassoSubtable,
};
use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network}, Rep3PrimeFieldShare,
    },
    rep3_ring::{self, Rep3RingShare},
};

use super::{
    slt::SLTInstruction, JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices,
};
use crate::utils::instruction_utils::{
    chunk_and_concatenate_operands, rep3_chunk_and_concatenate_operands,
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BGEInstruction(pub Rep3Operand, pub Rep3Operand);

impl JoltInstruction for BGEInstruction {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("BGEInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // 1 - LTS(x, y) =
        F::one()
            - <SLTInstruction as JoltInstruction>::combine_lookups(
                &SLTInstruction(self.0.clone(), self.1.clone()),
                vals,
                C,
                M,
            )
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
            _ => panic!("BGEInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => ((*x as i32) >= (*y as i32)).into(),
            _ => panic!("BGEInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl Rep3JoltInstruction for BGEInstruction {
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
        name = "BGEInstruction::combine_lookups_rep3_batched",
        level = "trace"
    )]
    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        let res = <SLTInstruction as Rep3JoltInstruction>::combine_lookups_rep3_batched(
            &SLTInstruction(self.0.clone(), self.1.clone()),
            vals,
            C,
            M,
            io_ctx,
        )?
        .into_iter()
        .map(|slt| rep3::arithmetic::sub_public_by_shared(F::one(), slt, io_ctx.network.get_id()))
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
            .map(|st| (st.lhs().as_binary(), st.rhs().unwrap().as_binary()))
            .unzip();

        rep3_ring::arithmetic::ge_many(&a, &b, io_ctx)
            .context("BGEInstruction::output_batched")?
            .into_iter()
            .zip(out)
            .for_each(|(ge, out)| {
                *out = FutureRep3Ring::bit_inject_to_field(ge);
            });
        Ok(())
    }
}
