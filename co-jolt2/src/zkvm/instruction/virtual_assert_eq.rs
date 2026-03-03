use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualAssertEQ> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (
            self.register_state.rs1_operand(),
            self.register_state.rs2_operand(),
        )
    }

    #[tracing::instrument(skip_all, name = "VirtualAssertEQ::output", level = "trace")]
    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        let (a, b): (Vec<Rep3RingShare<u64>>, Vec<Rep3RingShare<u64>>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (
                    downcast(l.as_arithmetic_or_trivial_u128(io_ctx.id)),
                    downcast(r.as_arithmetic_or_trivial_u128(io_ctx.id)),
                )
            })
            .unzip();
        rep3_ring::arithmetic::eq_many(&a, &b, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(x, out)| {
                *out = FutureRep3Ring::bit_inject_to_field(x);
            });
        Ok(())
    }
}
