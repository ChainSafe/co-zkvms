use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualAssertWordAlignment> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (self.register_state.rs1_operand(), Rep3Operand::Public(self.instruction.operands.imm as u64 as i128))
    }

    fn to_lookup_index(&self, party_id: PartyID) -> FutureRep3Ring<LookupIndexInt, Rep3RingShare<LookupIndexInt>> {
        let (left, right) = <Self as Rep3LookupQuery<XLEN>>::to_instruction_inputs(self);
        let l = left.as_arithmetic_or_trivial_wide(party_id);
        let r = right.as_arithmetic_or_trivial_wide(party_id);
        FutureRep3Ring::a2b(l + r)
    }

    #[tracing::instrument(skip_all, name = "VirtualAssertWordAlignment::output", level = "trace")]
    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // Check (lhs + rhs) % 4 == 0: add in binary, check 2 LSBs are zero
        let (a, b): (Vec<_>, Vec<_>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (l.as_binary_or_trivial(io_ctx.id), r.as_binary_or_trivial(io_ctx.id))
            })
            .unzip();
        let sums = rep3_ring::binary::add_many(&a, &b, io_ctx)?;
        // Mask to get 2 LSBs (as u32 share), then check if zero
        let low_bits: Vec<_> = sums.iter().map(|s| *s & RingElement(3 as XlenInt)).collect();
        let is_aligned = rep3_ring::binary::is_zero_many(&low_bits, io_ctx)?;
        is_aligned.into_iter().zip(out).for_each(|(x, out)| {
            *out = FutureRep3Ring::bit_inject_to_field(x);
        });
        Ok(())
    }
}
