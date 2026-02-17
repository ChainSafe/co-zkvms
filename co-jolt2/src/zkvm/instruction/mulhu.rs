use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<MULHU> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (
            self.register_state.rs1_operand(),
            self.register_state.rs2_operand(),
        )
    }

    fn to_lookup_operands(&self) -> (Rep3Operand, Rep3Operand) {
        todo!("to_lookup_operands: mul-index")
    }

    fn to_lookup_index(&self, party_id: PartyID) -> FutureRep3Ring<u128, Rep3RingShare<u128>> {
        let (left, right) = <Self as Rep3LookupQuery<XLEN>>::to_instruction_inputs(self);
        let l = left.as_arithmetic_or_trivial_u128(party_id);
        let r = right.as_arithmetic_or_trivial_u128(party_id);
        FutureRep3Ring::mul_a2b(l, r)
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        let (a, b): (Vec<_>, Vec<_>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (l.as_arithmetic_u64(), r.as_arithmetic_u64())
            })
            .unzip();
        rep3_ring::arithmetic::mul_vec(&a, &b, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(product, out)| {
                let upper: Rep3RingShare<u32> = downcast(product >> 32);
                *out = FutureRep3Ring::cast_to_field(upper);
            });
        Ok(())
    }
}
