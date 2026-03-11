use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<LUI> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        let sums: Vec<_> = steps
            .iter()
            .map(|step| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
                l.as_arithmetic_or_trivial::<u64>(io_ctx.id) + r.as_arithmetic_or_trivial::<u64>(io_ctx.id)
            })
            .collect();
        cast_wrapped_lookup_output_many(&sums, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(share, out)| *out = FutureRep3Ring::Ready(share));
        Ok(())
    }
}
