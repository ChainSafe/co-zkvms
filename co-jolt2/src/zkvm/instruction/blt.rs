use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<BLT> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        let sign_flip = RingElement(1u32 << 31);
        let (a, b): (Vec<_>, Vec<_>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (l.as_binary() ^ sign_flip, r.as_binary() ^ sign_flip)
            })
            .unzip();
        rep3_ring::arithmetic::ge_many(&a, &b, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(x, out)| {
                *out = FutureRep3Ring::bit_inject_to_field(!x);
            });
        Ok(())
    }
}
