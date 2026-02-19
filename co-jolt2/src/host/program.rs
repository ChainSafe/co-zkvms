use itertools::Itertools;
use jolt_core::host::Program;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rand::RngCore;
use rayon::prelude::*;

use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::host::memory::Rep3Memory;
use crate::zkvm::instruction::{Rep3Cycle, Rep3Operand};

pub trait Rep3Program {
    fn generate_trace_shares<R: RngCore>(
        &mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        rng: &mut R,
    ) -> [(Vec<Rep3Cycle>, Rep3Memory, Rep3ProgramIOInput); 3];
}

impl Rep3Program for Program {
    fn generate_trace_shares<R: RngCore>(
        &mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        rng: &mut R,
    ) -> [(Vec<Rep3Cycle>, Rep3Memory, Rep3ProgramIOInput); 3] {
        let (trace, memory, program_io) = self.trace(inputs, untrusted_advice, trusted_advice);

        let program_io_shares = Rep3ProgramIOInput::generate_secret_shares(program_io, rng);
        let memory_shares = Rep3Memory::generate_secret_shares(memory, rng);

        let trace_shares = share_trace(trace, rng);

        let [io0, io1, io2]: [Rep3ProgramIOInput; 3] =
            program_io_shares.try_into().expect("expected 3 shares");
        let [mem0, mem1, mem2]: [Rep3Memory; 3] =
            memory_shares.try_into().expect("expected 3 shares");
        let [t0, t1, t2]: [Vec<Rep3Cycle>; 3] = trace_shares;

        [(t0, mem0, io0), (t1, mem1, io1), (t2, mem2, io2)]
    }
}

/// Share a vanilla trace into 3 Rep3 traces with binary-shared operands.
///
/// For each cycle:
/// 1. Extract operand values from the vanilla Cycle
/// 2. Generate binary shares for each value
/// 3. Build 3 Rep3Cycles via `from_cycle_shared`
fn share_trace<R: RngCore>(
    trace: Vec<tracer::instruction::Cycle>,
    rng: &mut R,
) -> [Vec<Rep3Cycle>; 3] {
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    let root = rng.next_u64();

    // Process each cycle in parallel, producing 3 shared copies per cycle
    let (t0, t1, t2): (Vec<Rep3Cycle>, Vec<Rep3Cycle>, Vec<Rep3Cycle>) = trace
        .into_par_iter()
        .map_init(
            move || {
                let tid = rayon::current_thread_index().unwrap_or(0) as u64;
                ChaCha12Rng::seed_from_u64(root ^ tid)
            },
            |rng, cycle| share_cycle(&cycle, rng),
        )
        .collect::<Vec<_>>()
        .into_iter()
        .multiunzip();

    [t0, t1, t2]
}

/// Returns the indices of register operands that must remain public (not
/// secret-shared) for a given cycle.  Certain virtual instructions compute
/// their lookup output using the plaintext value of a register operand
/// (e.g. `2^x` in VirtualPow2).  Mirroring v1, we keep that operand public
/// so the MPC lookup-output code can call `as_public()` on it.
///
/// The indices refer to positions in the vector returned by
/// `Rep3Cycle::extract_operand_values` / consumed by `from_cycle_shared`.
/// For FormatI this is `[rs1=0, rd_old=1, rd_new=2, ...ram]`.
fn public_operand_indices(cycle: &tracer::instruction::Cycle) -> &'static [usize] {
    use tracer::instruction::Cycle;
    match cycle {
        // rs1 holds the shift/exponent amount — keep it public
        Cycle::VirtualPow2(_) | Cycle::VirtualPow2W(_) | Cycle::VirtualShiftRightBitmask(_) => &[0], // rs1
        // rs2 holds the bitmask (from VirtualShiftRightBitmask) — keep it public
        Cycle::VirtualSRL(_) | Cycle::VirtualSRA(_) => &[1], // rs2
        _ => &[],
    }
}

/// Share a single vanilla Cycle into 3 Rep3Cycles with binary-shared operands.
///
/// Extracts operand values directly from the vanilla Cycle, generates binary
/// shares, and builds 3 Rep3Cycles via `from_cycle_shared`.
/// Operands at indices returned by `public_operand_indices` are kept public.
fn share_cycle(
    cycle: &tracer::instruction::Cycle,
    rng: &mut impl rand::Rng,
) -> (Rep3Cycle, Rep3Cycle, Rep3Cycle) {
    let values = Rep3Cycle::extract_operand_values(cycle);
    let public_indices = public_operand_indices(cycle);

    // Generate shares for each operand — public indices get replicated as
    // Rep3Operand::Public(v) for all 3 parties instead of binary shares.
    let operands_per_party: Vec<[Rep3Operand; 3]> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if public_indices.contains(&i) {
                let op = Rep3Operand::Public(v);
                [op, op, op]
            } else {
                let s = rep3_ring::binary::generate_shares_rep3(v, rng);
                [
                    Rep3Operand::from_binary(s[0]),
                    Rep3Operand::from_binary(s[1]),
                    Rep3Operand::from_binary(s[2]),
                ]
            }
        })
        .collect();

    // Build 3 Rep3Cycles, one per party
    let mut s0 = operands_per_party.iter().map(|s| s[0]);
    let mut s1 = operands_per_party.iter().map(|s| s[1]);
    let mut s2 = operands_per_party.iter().map(|s| s[2]);
    (
        Rep3Cycle::from_cycle_shared(cycle, &mut s0),
        Rep3Cycle::from_cycle_shared(cycle, &mut s1),
        Rep3Cycle::from_cycle_shared(cycle, &mut s2),
    )
}
