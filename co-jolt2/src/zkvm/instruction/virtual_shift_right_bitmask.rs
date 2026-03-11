use mpc_core::protocols::rep3;

use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualShiftRightBitmask> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (self.register_state.rs1_operand(), Rep3Operand::Public(0))
    }

    fn to_lookup_index(&self, party_id: PartyID) -> FutureRep3Ring<LookupIndexInt, Rep3RingShare<LookupIndexInt>> {
        let (left, right) = <Self as Rep3LookupQuery<XLEN>>::to_instruction_inputs(self);
        let l = left.as_arithmetic_or_trivial_wide(party_id);
        let r = right.as_arithmetic_or_trivial_wide(party_id);
        FutureRep3Ring::a2b(l + r)
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // Shift amount is public, compute bitmask directly.
        // Vanilla formula (XLEN=64): ones = (1u128 << (64-shift)) - 1; mask = (ones << shift) as u64
        // Equivalent: mask has the lowest `shift` bits cleared and all higher bits set.
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, _) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let shift = l.as_public() % XLEN as u64;
            let mask = if shift == 0 {
                u64::MAX >> (64 - XLEN as u64)
            } else {
                let ones = (1u128 << (XLEN as u128 - shift as u128)) - 1;
                (ones << shift as u128) as u64
            };
            *out = FutureRep3Ring::Ready(rep3::arithmetic::promote_to_trivial_share(io_ctx.id, F::from(mask)).into());
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualShiftRightBitmaskI> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(self.instruction.operands.imm.into()), Rep3Operand::Public(0))
    }

    fn to_lookup_index(&self, party_id: PartyID) -> FutureRep3Ring<LookupIndexInt, Rep3RingShare<LookupIndexInt>> {
        let (left, right) = <Self as Rep3LookupQuery<XLEN>>::to_instruction_inputs(self);
        let l = left.as_arithmetic_or_trivial_wide(party_id);
        let r = right.as_arithmetic_or_trivial_wide(party_id);
        FutureRep3Ring::a2b(l + r)
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // Same bitmask formula as VirtualShiftRightBitmask.
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, _) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let shift = l.as_public() % XLEN as u64;
            let mask = if shift == 0 {
                u64::MAX >> (64 - XLEN as u64)
            } else {
                let ones = (1u128 << (XLEN as u128 - shift as u128)) - 1;
                (ones << shift as u128) as u64
            };
            *out = FutureRep3Ring::Ready(rep3::arithmetic::promote_to_trivial_share(io_ctx.id, F::from(mask)).into());
        });
        Ok(())
    }
}
