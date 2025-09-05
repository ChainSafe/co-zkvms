use ark_std::log2;
use itertools::izip;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};

use jolt_core::{
    jolt::subtable::{identity::IdentitySubtable, LassoSubtable},
    utils::instruction_utils::{chunk_operand_usize, concatenate_lookups},
};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::{
    rep3::network::{IoContext, Rep3Network},
    rep3_ring::Rep3RingShare,
};

use crate::utils::instruction_utils::{concatenate_lookups_rep3_batched, rep3_chunk_operand};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MOVEInstruction<const WORD_SIZE: usize>(pub Rep3Operand);

impl<const WORD_SIZE: usize> JoltInstruction for MOVEInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        assert!(M == 1 << 16);
        vec![(
            // Implicitly range-checks all query chunks
            Box::new(IdentitySubtable::<F>::new()),
            SubtableIndices::from(0..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0.as_public(), C, log_M)
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        // Same for both 32-bit and 64-bit word sizes
        self.0.as_public().into()
    }
    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self((rng.next_u32() as u64).into()),
            64 => Self(rng.next_u64().into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize> Rep3JoltInstruction for MOVEInstruction<WORD_SIZE> {
    fn operands_rep3(&self) -> (Rep3Operand, Rep3Operand) {
        (self.0.clone(), Rep3Operand::default())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand, Option<&mut Rep3Operand>) {
        (&mut self.0, None)
    }

    fn lhs(&self) -> &Rep3Operand {
        &self.0
    }

    fn rhs(&self) -> Option<&Rep3Operand> {
        None
    }

    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        Ok(concatenate_lookups_rep3_batched(vals, C, log2(M) as usize))
    }

    fn to_indices_rep3(
        &self,
        _: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        rep3_chunk_operand(self.0.as_binary(), C, log_M)
    }

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        _: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        izip!(steps, out).for_each(|(step, out)| {
            *out = FutureRep3Ring::cast_to_field(step.lhs().as_arithmetic_u32());
        });
        Ok(())
    }
}
