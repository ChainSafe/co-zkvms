use mpc_core::protocols::rep3;

use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualPow2W> {
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
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, _) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let val = l.as_public();
            *out = FutureRep3Ring::Ready(
                rep3::arithmetic::promote_to_trivial_share(io_ctx.id, F::from(1u64 << (val % 32))).into(),
            );
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualPow2IW> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(self.instruction.operands.imm.into()))
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
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (_, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let val = r.as_public();
            *out = FutureRep3Ring::Ready(
                rep3::arithmetic::promote_to_trivial_share(io_ctx.id, F::from(1u64 << (val % 32))).into(),
            );
        });
        Ok(())
    }
}
