use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualMovsign> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (self.register_state.rs1_operand(), Rep3Operand::Public(self.instruction.operands.imm.into()))
    }

    #[tracing::instrument(skip_all, name = "VirtualMovsign::output", level = "trace")]
    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // Extract sign bit (MSB), then cmux: if sign_bit then 0xFFFFFFFF else 0
        let all_ones = RingElement(u64::from(u32::MAX));
        let vals: Vec<_> = steps
            .iter()
            .map(|st| {
                let (l, _r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                l.as_binary_or_trivial(io_ctx.id)
            })
            .collect();
        let sign_bits: Vec<_> = vals.iter().map(|v| (*v >> (XLEN - 1))).collect();
        let is_negative = rep3_ring::binary::is_zero_many(&sign_bits, io_ctx)?;
        // is_negative[i] == 1 means sign bit was 0 (positive), == 0 means sign bit was 1 (negative)
        // We want: if sign_bit then all_ones else 0
        // sign_bit = !is_zero(val >> (XLEN-1))
        let results: Vec<_> = is_negative
            .into_iter()
            .map(|z| {
                let neg = !z; // neg == 1 when sign bit is 1
                let neg_u64 = bit_to_ring64(neg);
                // neg_u32 * 0xFFFFFFFF
                neg_u64 * all_ones
            })
            .collect();
        results.into_iter().zip(out).for_each(|(r, out)| {
            *out = FutureRep3Ring::cast_to_field_b2a(downcast(r));
        });
        Ok(())
    }
}
