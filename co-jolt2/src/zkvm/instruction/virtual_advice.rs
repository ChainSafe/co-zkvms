use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualAdvice> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        // Match vanilla: VirtualAdvice does not contribute to instruction inputs.
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_public_lookup_output(&self) -> Option<u64> {
        let advice = self.instruction.advice.unwrap_or(0);
        Some(match XLEN {
            #[cfg(test)]
            8 => advice as u8 as u64,
            32 => advice as u32 as u64,
            64 => advice as u64,
            _ => panic!("{XLEN}-bit word size is unsupported"),
        })
    }

    fn to_lookup_index(
        &self,
        party_id: PartyID,
    ) -> FutureRep3Ring<LookupIndexInt, Rep3RingShare<LookupIndexInt>> {
        let advice = self.instruction.advice.unwrap_or(0);
        let advice = match XLEN {
            #[cfg(test)]
            8 => advice as u8 as LookupIndexInt,
            32 => advice as u32 as LookupIndexInt,
            64 => advice as LookupIndexInt,
            _ => panic!("{XLEN}-bit word size is unsupported"),
        };
        FutureRep3Ring::Ready(rep3_ring::arithmetic::promote_to_trivial_share(
            party_id,
            RingElement(advice),
        ))
    }

    // TODO: can advice be public?
    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // RangeCheckTable is the identity function: output = input = advice value.
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let advice_val = step
                .to_public_lookup_output()
                .expect("VirtualAdvice lookup output must be public");
            // Advice is public instruction metadata, so each party can promote it locally.
            *out = FutureRep3Ring::Ready(
                mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                    io_ctx.id,
                    F::from_u64(advice_val),
                ),
            );
        });
        Ok(())
    }
}
