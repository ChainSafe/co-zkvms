use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualROTRI> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let n = (r.as_public() % XLEN as u64) as u32;
            let x = l.as_binary_or_trivial(io_ctx.id);
            // Right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(x.a.0.rotate_right(n)),
                RingElement(x.b.0.rotate_right(n)),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(rotated);
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualROTRIW> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let n = (r.as_public() % 32) as u32;
            let x: Rep3RingShare<u32> = downcast(l.as_binary_or_trivial(io_ctx.id));
            // W-variant: 32-bit right rotation
            let rotated = Rep3RingShare::new_ring(
                RingElement(x.a.0.rotate_right(n)),
                RingElement(x.b.0.rotate_right(n)),
            );
            let rotated_u64 = Rep3RingShare::new_ring(
                RingElement(rotated.a.0 as u64),
                RingElement(rotated.b.0 as u64),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(rotated_u64);
        });
        Ok(())
    }
}
