use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<XOR> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (
            self.register_state.rs1_operand(),
            self.register_state.rs2_operand(),
        )
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let z = l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id);
            *out = FutureRep3Ring::cast_to_field_b2a(binary_to_output(z));
        });
        Ok(())
    }
}
