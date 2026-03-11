use super::*;

// TODO: figure out how to deal with signed operands.
impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<SLTI> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (self.register_state.rs1_operand(), Rep3Operand::Public(self.instruction.operands.imm.into()))
    }

    #[tracing::instrument(skip_all, name = "SLTI::output", level = "trace")]
    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // ge_many's internal unsigned_ge_many expects binary (XOR-domain) shares.
        // Flip the sign bit (binary XOR) to convert signed comparison to unsigned.
        let sign_bit = RingElement((1 as XlenInt) << (XLEN - 1));
        let (a, b): (Vec<Rep3RingShare<XlenInt>>, Vec<Rep3RingShare<XlenInt>>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (
                    rep3_ring::binary::xor_public(&l.as_binary_or_trivial(io_ctx.id), &sign_bit, io_ctx.id),
                    rep3_ring::binary::xor_public(&r.as_binary_or_trivial(io_ctx.id), &sign_bit, io_ctx.id),
                )
            })
            .unzip();
        rep3_ring::arithmetic::ge_many(&a, &b, io_ctx)?.into_iter().zip(out).for_each(|(x, out)| {
            *out = FutureRep3Ring::bit_inject_to_field(!x);
        });
        Ok(())
    }
}
