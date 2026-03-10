use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualAdvice> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        // Match vanilla: VirtualAdvice does not contribute to instruction inputs.
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_index(
        &self,
        party_id: PartyID,
    ) -> FutureRep3Ring<LookupIndexInt, Rep3RingShare<LookupIndexInt>> {
        debug_assert!(
            self.instruction.advice.is_none(),
            "VirtualAdvice plaintext advice must be scrubbed from Rep3 traces"
        );
        let advice = self
            .advice
            .as_ref()
            .expect("VirtualAdvice shared advice payload missing")
            .as_binary_or_trivial(party_id);
        let advice = match XLEN {
            #[cfg(test)]
            8 => Rep3RingShare::new_ring(
                RingElement(advice.a.0 as u8 as LookupIndexInt),
                RingElement(advice.b.0 as u8 as LookupIndexInt),
            ),
            32 => Rep3RingShare::new_ring(
                RingElement(advice.a.0 as u32 as LookupIndexInt),
                RingElement(advice.b.0 as u32 as LookupIndexInt),
            ),
            64 => Rep3RingShare::new_ring(
                RingElement(advice.a.0 as LookupIndexInt),
                RingElement(advice.b.0 as LookupIndexInt),
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        };
        FutureRep3Ring::Ready(advice)
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // RangeCheckTable is the identity function: output = input = advice value.
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let advice = step.to_lookup_index(io_ctx.id);
            *out = match advice {
                FutureRep3Ring::Ready(advice) => {
                    let advice_u64 = match XLEN {
                        #[cfg(test)]
                        8 => Rep3RingShare::new_ring(
                            RingElement(advice.a.0 as u8 as u64),
                            RingElement(advice.b.0 as u8 as u64),
                        ),
                        32 => Rep3RingShare::new_ring(
                            RingElement(advice.a.0 as u32 as u64),
                            RingElement(advice.b.0 as u32 as u64),
                        ),
                        64 => Rep3RingShare::new_ring(
                            RingElement(advice.a.0 as u64),
                            RingElement(advice.b.0 as u64),
                        ),
                        _ => panic!("{XLEN}-bit word size is unsupported"),
                    };
                    FutureRep3Ring::cast_to_field_b2a(advice_u64)
                }
                FutureRep3Ring::Pending(_, _) => {
                    panic!("VirtualAdvice lookup index must be immediately available")
                }
            };
        });
        Ok(())
    }
}
