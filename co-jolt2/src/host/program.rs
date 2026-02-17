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

/// Share a single vanilla Cycle into 3 Rep3Cycles with binary-shared operands.
///
/// Extracts operand values directly from the vanilla Cycle, generates binary
/// shares, and builds 3 Rep3Cycles via `from_cycle_shared`.
fn share_cycle(
    cycle: &tracer::instruction::Cycle,
    rng: &mut impl rand::Rng,
) -> (Rep3Cycle, Rep3Cycle, Rep3Cycle) {
    let values = Rep3Cycle::extract_operand_values(cycle);

    // Generate binary shares for each operand value
    let shares_per_op: Vec<[Rep3RingShare<u32>; 3]> = values
        .iter()
        .map(|&v| {
            let s = rep3_ring::binary::generate_shares_rep3(v as u32, rng);
            [s[0], s[1], s[2]]
        })
        .collect();

    // Build 3 Rep3Cycles, one per party
    let mut s0 = shares_per_op.iter().map(|s| Rep3Operand::from_binary(s[0]));
    let mut s1 = shares_per_op.iter().map(|s| Rep3Operand::from_binary(s[1]));
    let mut s2 = shares_per_op.iter().map(|s| Rep3Operand::from_binary(s[2]));
    (
        Rep3Cycle::from_cycle_shared(cycle, &mut s0),
        Rep3Cycle::from_cycle_shared(cycle, &mut s1),
        Rep3Cycle::from_cycle_shared(cycle, &mut s2),
    )
}
