use itertools::Itertools;
use jolt_core::host::Program;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::RAMPreprocessing;
use jolt_core::zkvm::JoltSharedPreprocessing;
use mpc_core::protocols::rep3_ring::{self};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;

use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::host::memory::Rep3Memory;
use crate::utils::compute_ram_k;
use crate::zkvm::instruction::{Rep3Cycle, Rep3Operand};

pub type Rep3ShareBundle = (Vec<Rep3Cycle>, Rep3Memory, Rep3ProgramIOInput);

/// Execute `program`, generate 3-way Rep3 secret shares of the trace, and return
/// all public metadata needed by the client to build `WorkerPayload`.
///
/// Returns `(bytecode, memory_init, program_io, shares)`:
/// - `bytecode` and `memory_init`: public data sent to workers for preprocessing
/// - `program_io`: vanilla JoltDevice (needed for verification and output decoding)
/// - `shares`: 3 Rep3ShareBundles (one per party), traces already padded to next power of 2
#[tracing::instrument(skip_all, name = "Program::generate_trace_shares")]
pub fn generate_trace_shares<R: RngCore + CryptoRng>(
    program: &mut Program,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    rng: &mut R,
) -> (
    Vec<tracer::instruction::Instruction>,
    Vec<(u64, u8)>,
    tracer::JoltDevice,
    [Rep3ShareBundle; 3],
) {
    let (bytecode, memory_init, _) = program.decode();
    let (mut trace, memory, program_io) = program.trace(inputs, untrusted_advice, trusted_advice);

    let padded_len = (trace.len() + 1).next_power_of_two();
    trace.resize(padded_len, tracer::instruction::Cycle::NoOp);

    let shared_preprocessing = JoltSharedPreprocessing {
        memory_layout: program_io.memory_layout.clone(),
        bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
        ram: RAMPreprocessing::preprocess(memory_init.clone()),
    };

    // Share the trace, then compute ram_k from the (public) addresses in the shared trace.
    let trace_shares = share_trace(trace, rng);
    let ram_k = compute_ram_k(&trace_shares[0], &shared_preprocessing);

    let program_io_shares = Rep3ProgramIOInput::generate_secret_shares(program_io.clone(), rng);
    let memory_shares = Rep3Memory::generate_secret_shares(memory, &program_io.memory_layout, ram_k, rng);

    let [io0, io1, io2]: [Rep3ProgramIOInput; 3] = program_io_shares.try_into().expect("expected 3 shares");
    let [mem0, mem1, mem2]: [Rep3Memory; 3] = memory_shares.try_into().expect("expected 3 shares");
    let [t0, t1, t2]: [Vec<Rep3Cycle>; 3] = trace_shares;

    (bytecode, memory_init, program_io, [(t0, mem0, io0), (t1, mem1, io1), (t2, mem2, io2)])
}

/// Share a vanilla trace into 3 Rep3 traces with binary-shared operands.
///
/// For each cycle:
/// 1. Extract operand values from the vanilla Cycle
/// 2. Generate binary shares for each value
/// 3. Build 3 Rep3Cycles via `from_cycle_shared`
pub fn share_trace<R: RngCore>(trace: Vec<tracer::instruction::Cycle>, rng: &mut R) -> [Vec<Rep3Cycle>; 3] {
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
        Cycle::VirtualPow2(_) | Cycle::VirtualShiftRightBitmask(_) => &[0], // rs1
        #[cfg(feature = "rv64")]
        Cycle::VirtualPow2W(_) => &[0],     // rs1
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
    rng: &mut (impl rand::Rng + rand::CryptoRng),
) -> (Rep3Cycle, Rep3Cycle, Rep3Cycle) {
    let mut copied_cycle = cycle.clone();
    let values = Rep3Cycle::extract_operand_values(&copied_cycle);
    let public_indices = public_operand_indices(cycle);
    if let tracer::instruction::Cycle::VirtualAdvice(c) = &mut copied_cycle {
        c.instruction.advice = None;
    }

    // Generate shares for each operand — public indices get replicated as
    // Rep3Operand::Public(v) for all 3 parties instead of binary shares.
    let operands_per_party: Vec<[Rep3Operand; 3]> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if public_indices.contains(&i) {
                let op = Rep3Operand::Public(v as i128);
                [op, op, op]
            } else {
                let s = rep3_ring::share_ring_element_binary(
                    rep3_ring::ring::ring_impl::RingElement(v as jolt_common::constants::XlenInt),
                    rng,
                );
                [Rep3Operand::from_binary(s[0]), Rep3Operand::from_binary(s[1]), Rep3Operand::from_binary(s[2])]
            }
        })
        .collect();

    // Build 3 Rep3Cycles, one per party
    let mut s0 = operands_per_party.iter().map(|s| s[0]);
    let mut s1 = operands_per_party.iter().map(|s| s[1]);
    let mut s2 = operands_per_party.iter().map(|s| s[2]);
    (
        Rep3Cycle::from_cycle_shared(&copied_cycle, &mut s0),
        Rep3Cycle::from_cycle_shared(&copied_cycle, &mut s1),
        Rep3Cycle::from_cycle_shared(&copied_cycle, &mut s2),
    )
}
