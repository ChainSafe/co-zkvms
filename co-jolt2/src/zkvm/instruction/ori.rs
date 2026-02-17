use super::*;

// TODO: figure out how to deal with signed operands.
impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<ORI> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (
            self.register_state.rs1_operand(),
            Rep3Operand::Public(self.instruction.operands.imm),
        )
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        let (x, y): (Vec<_>, Vec<_>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (l.as_binary(), r.as_binary())
            })
            .unzip();
        rep3_ring::binary::or_many(&x, &y, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(z, out)| {
                *out = FutureRep3Ring::cast_to_field_b2a(z);
            });
        Ok(())
    }
}
