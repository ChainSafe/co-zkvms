use crate::field::JoltField;
use crate::utils::future_ring::FutureRep3Ring;
use ark_std::log2;
use itertools::izip;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::instruction_utils::{
    add_and_chunk_operands, assert_valid_parameters, concatenate_lookups,
    concatenate_lookups_rep3_batched, rep3_add_and_chunk_operands,
};
use jolt_core::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SUBInstruction<const WORD_SIZE: usize>(pub Rep3Operand, pub Rep3Operand);

impl<const WORD_SIZE: usize> JoltInstruction for SUBInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("SUBInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(vals.len() == C / 2);
        // The output is the TruncateOverflow(most significant chunk) || Identity of other chunks
        concatenate_lookups(vals, C / 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(msb_chunk_index + 1..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        add_and_chunk_operands(
            self.0.as_public() as u128,
            (1u128 << WORD_SIZE) - self.1.as_public() as u128,
            C,
            log_M,
        )
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                (*x as u32).overflowing_sub(*y as u32).0.into()
            }
            _ => panic!("SUBInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

impl<const WORD_SIZE: usize> Rep3JoltInstruction for SUBInstruction<WORD_SIZE> {
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

    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        assert!(vals.len() == C / 2);
        Ok(concatenate_lookups_rep3_batched(
            vals,
            C / 2,
            log2(M) as usize,
        ))
    }

    fn to_indices_intermediate<F: JoltField>(
        &self,
        id: PartyID,
    ) -> FutureRep3Ring<u128, Option<Rep3RingShare<u128>>> {
        FutureRep3Ring::a2b(
            self.lhs().as_arithmetic()
                + rep3_ring::arithmetic::sub_public_by_shared(
                    (1u128 << WORD_SIZE).into(),
                    self.rhs().unwrap().as_arithmetic(),
                    id,
                ),
        )
    }

    fn to_indices_rep3(
        &self,
        z: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        // add_and_chunk_operands(*x as u128, (1u128 << WORD_SIZE) - *y as u128, C, log_M)
        rep3_add_and_chunk_operands(&z.unwrap(), C, log_M)
    }

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        _: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        izip!(steps, out).for_each(|(st, out)| {
            *out = FutureRep3Ring::cast_to_field(
                st.lhs().as_arithmetic_u32() - st.rhs().unwrap().as_arithmetic_u32(),
            );
        });
        Ok(())
    }
}
