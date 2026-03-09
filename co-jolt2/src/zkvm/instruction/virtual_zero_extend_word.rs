use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualZeroExtendWord> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (self.register_state.rs1_operand(), Rep3Operand::Public(0))
    }

    fn to_lookup_index(
        &self,
        party_id: PartyID,
    ) -> FutureRep3Ring<LookupIndexInt, Rep3RingShare<LookupIndexInt>> {
        let (left, right) = <Self as Rep3LookupQuery<XLEN>>::to_instruction_inputs(self);
        let l = left.as_arithmetic_or_trivial_wide(party_id);
        let r = right.as_arithmetic_or_trivial_wide(party_id);
        FutureRep3Ring::a2b(l + r)
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, _) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            // Mask to lower XLEN/2 bits
            let mask = RingElement((1u64 << (XLEN / 2)) - 1);
            *out = FutureRep3Ring::cast_to_field_b2a(l.as_binary_or_trivial(io_ctx.id) & mask);
        });
        Ok(())
    }
}
