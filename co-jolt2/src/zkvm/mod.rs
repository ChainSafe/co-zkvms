pub mod bytecode;
pub mod dag;
pub mod instruction;
pub mod instruction_lookups;
pub mod r1cs;
pub mod ram;
pub mod registers;
pub mod spartan;
pub mod suffixes;
pub mod witness;

use std::collections::HashMap;

use crate::field::JoltField;
use crate::host::memory::Rep3Memory;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::zkvm::dag::coordinator::Rep3JoltDAGCoordinator;
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};
use crate::zkvm::dag::worker::Rep3JoltDAGWorker;
use crate::zkvm::instruction::Rep3Cycle;
use jolt_core::ark_bn254::Fr;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};
use jolt_core::zkvm::dag::proof_serialization::JoltProof;
use jolt_core::zkvm::witness::CommittedPolynomial;
use jolt_core::zkvm::{JoltProverPreprocessing, JoltRV64IMAC, JoltVerifierPreprocessing};
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3_ring::edabits::EdaBitsPool;
use tracer::JoltDevice;

// ---------------------------------------------------------------------------
// Worker trait
// ---------------------------------------------------------------------------

pub trait Rep3JoltWorker<F: JoltField, PCS, ProofTranscript: Transcript>
where
    PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
{
    fn preprocess(
        bytecode: Vec<tracer::instruction::Instruction>,
        memory_layout: jolt2_common::jolt_device::MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS>;

    fn prove<N: Rep3NetworkWorker>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: Vec<Rep3Cycle>,
        program_io: JoltDevice,
        final_memory_state: Rep3Memory,
        io_ctx: &mut IoContextPool<N>,
        ram_K: usize,
        advice_shares: Option<crate::host::jolt_device::Rep3ProgramIOInput>,
        edabits_pool: EdaBitsPool<F>,
    ) -> eyre::Result<()>;
}

// ---------------------------------------------------------------------------
// Coordinator trait
// ---------------------------------------------------------------------------

pub trait Rep3Jolt<F: JoltField, PCS, ProofTranscript: Transcript>
where
    PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
{
    fn prove<N: Rep3NetworkCoordinator>(
        preprocessing: &JoltVerifierPreprocessing<F, PCS>,
        program_io: JoltDevice,
        network: &mut N,
        ram_K: usize,
        trace_length: usize,
    ) -> eyre::Result<JoltProof<F, PCS, ProofTranscript>>;
}

// ---------------------------------------------------------------------------
// Implementations for JoltRV64IMAC
// ---------------------------------------------------------------------------

impl Rep3JoltWorker<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltRV64IMAC {
    fn preprocess(
        bytecode: Vec<tracer::instruction::Instruction>,
        memory_layout: jolt2_common::jolt_device::MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<Fr, DoryCommitmentScheme> {
        use jolt_core::utils::math::Math;
        use jolt_core::zkvm::witness::DTH_ROOT_OF_K;
        use jolt_core::zkvm::Jolt;

        // Delegate to vanilla Jolt::prover_preprocess — preprocessing is public
        <JoltRV64IMAC as Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript>>::prover_preprocess(
            bytecode,
            memory_layout,
            memory_init,
            max_trace_length,
        )
    }

    fn prove<N: Rep3NetworkWorker>(
        preprocessing: &JoltProverPreprocessing<Fr, DoryCommitmentScheme>,
        trace: Vec<Rep3Cycle>,
        program_io: JoltDevice,
        final_memory_state: Rep3Memory,
        io_ctx: &mut IoContextPool<N>,
        ram_K: usize,
        advice_shares: Option<crate::host::jolt_device::Rep3ProgramIOInput>,
        edabits_pool: EdaBitsPool<Fr>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.party_id();
        let state = StateManagerWorker::new(
            preprocessing,
            trace,
            program_io,
            final_memory_state,
            party_id,
            ram_K,
            advice_shares,
        );
        Rep3JoltDAGWorker::prove::<Fr, DoryCommitmentScheme, Blake2bTranscript, N>(
            state,
            io_ctx,
            edabits_pool,
        )
    }
}

impl Rep3Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltRV64IMAC {
    fn prove<N: Rep3NetworkCoordinator>(
        preprocessing: &JoltVerifierPreprocessing<Fr, DoryCommitmentScheme>,
        program_io: JoltDevice,
        network: &mut N,
        ram_K: usize,
        trace_length: usize,
    ) -> eyre::Result<JoltProof<Fr, DoryCommitmentScheme, Blake2bTranscript>> {
        // Compute twist_sumcheck_switch_index the same way as the worker
        let T = trace_length;
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = if num_chunks > 0 { T / num_chunks } else { T };
        let twist_sumcheck_switch_index = if chunk_size > 0 {
            chunk_size.trailing_zeros() as usize
        } else {
            0
        };

        let state = StateManagerCoordinator::new(
            preprocessing,
            program_io,
            ram_K,
            twist_sumcheck_switch_index,
        );
        Rep3JoltDAGCoordinator::prove(state, network)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use ark_bn254::Fr;
    use ark_std::test_rng;
    use tracing::{info, info_span};

    use crate::host::program::Rep3Program;
    use crate::utils::compute_ram_k;
    use crate::utils::test_utils::run_rep3_test_with_coordinator;
    use crate::utils::tracing::init_tracing;
    use crate::zkvm::instruction::{populate_operands_casts, Rep3Cycle};
    use crate::zkvm::{Rep3Jolt, Rep3JoltWorker};
    use jolt_core::host::Program;
    use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
    use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
    use jolt_core::zkvm::bytecode::BytecodePreprocessing;
    use jolt_core::zkvm::ram::RAMPreprocessing;
    use jolt_core::zkvm::witness::{
        compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
    };
    use jolt_core::zkvm::{
        JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing, JoltVerifierPreprocessing,
    };
    use tracer::instruction::Cycle;

    type F = Fr;
    type PCS = DoryCommitmentScheme;

    #[test]
    #[ignore = "requires QUIC network sockets (not available in sandboxed test env)"]
    fn commitment_correct() {
        let _tracing_guard =
            init_tracing("commitment_test.json", Path::new("/tmp/co-jolt2-traces"));

        // 1. Build and trace the fibonacci program
        let mut program = Program::new("fibonacci-guest");
        let elf_path = "/tmp/jolt-guest-targets/fibonacci-guest-/riscv64imac-unknown-none-elf/release/fibonacci-guest";
        program.elf = Some(PathBuf::from(elf_path));
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, memory_init, _) = program.decode();

        // 2. Generate trace and shares
        let mut rng = test_rng();
        let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);

        let (mut vanilla_trace, _memory, io_device) = program.trace(&inputs, &[], &[]);

        // Pad traces
        let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
        info!(raw_len = vanilla_trace.len(), padded_len, "padding traces");
        vanilla_trace.resize(padded_len, Cycle::NoOp);
        for (trace, _, _) in shares.iter_mut() {
            trace.resize(padded_len, Rep3Cycle::NoOp);
        }

        // 3. Build preprocessing with Dory generators
        let shared = JoltSharedPreprocessing {
            memory_layout: io_device.memory_layout.clone(),
            bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
            ram: RAMPreprocessing::preprocess(memory_init.clone()),
        };
        let preprocessing: JoltProverPreprocessing<F, PCS> =
            <JoltRV64IMAC as Rep3JoltWorker<F, PCS, _>>::preprocess(
                bytecode,
                io_device.memory_layout.clone(),
                memory_init,
                padded_len,
            );
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

        // 4. Compute ram_K
        let ram_K = compute_ram_k(&vanilla_trace, &shared);
        let bytecode_d = shared.bytecode.d;
        let ram_d = compute_d_parameter(ram_K);

        // 5. Vanilla: generate witness polys and commit
        // DoryGlobals guard kept alive for entire test — workers get non-owning guards
        let _vanilla_span = info_span!("vanilla_commitments").entered();
        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);
        let _poly_guard = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

        let all_polys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();
        let mut vanilla_witness =
            CommittedPolynomial::generate_witness_batch(&all_polys, &preprocessing, &vanilla_trace);

        let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
            .filter_map(|poly| vanilla_witness.remove(poly))
            .collect();

        let vanilla_commits: Vec<<PCS as CommitmentScheme>::Commitment> =
            PCS::batch_commit(&committed_polys, &preprocessing.generators)
                .into_iter()
                .map(|(c, _)| c)
                .collect();
        drop(committed_polys);
        drop(_vanilla_span);

        // 6. MPC: run 3 workers + 1 coordinator
        let preprocessing_arc = Arc::new(preprocessing);
        let io_device_arc = Arc::new(io_device);
        let base_port: u16 = 14300;

        let _mpc_span = info_span!("mpc_commitment").entered();
        let (_worker_results, coordinator_result) = run_rep3_test_with_coordinator(
            base_port,
            4,
            |party_idx| {
                let (trace, memory, advice) = shares[party_idx].clone();
                let preprocessing = Arc::clone(&preprocessing_arc);
                let io_device = (*io_device_arc).clone();
                (trace, memory, preprocessing, io_device, ram_K, advice)
            },
            || {
                let verifier_preprocessing = verifier_preprocessing.clone();
                let io_device = (*io_device_arc).clone();
                (verifier_preprocessing, io_device, ram_K, padded_len)
            },
            |input, mut io_ctx| {
                let (mut trace, memory, preprocessing, io_device, ram_K, advice) = input;

                let party = io_ctx.party_id();
                let _span = info_span!("populate_operands_casts", ?party).entered();
                populate_operands_casts(&mut trace, io_ctx.main())?;
                drop(_span);

                // Preprocessing: create EdaBits pool for B2A conversions.
                let edabits_pool = {
                    use crate::zkvm::instruction_lookups::read_raf_checking::compute_edabit_budget;
                    use mpc_core::protocols::rep3_ring::edabits;
                    use mpc_core::protocols::rep3_ring::pcg::edabits_pcg;
                    let budget = compute_edabit_budget(trace.len());
                    let lazy_u8 = edabits::random_edabits_lazy::<u8, F, _>(budget.u8, &mut io_ctx)?;
                    let lazy_u16 =
                        edabits::random_edabits_lazy::<u16, F, _>(budget.u16, &mut io_ctx)?;
                    let lazy_u32 =
                        edabits::random_edabits_lazy::<u32, F, _>(budget.u32, &mut io_ctx)?;
                    let lazy_u64 =
                        edabits::random_edabits_lazy::<u64, F, _>(budget.u64, &mut io_ctx)?;
                    let lazy_u128 =
                        edabits::random_edabits_lazy::<u128, F, _>(budget.u128, &mut io_ctx)?;
                    let dabit_setup = edabits_pcg::random_pcg_dabit_setup::<F, _>(&mut io_ctx)?;
                    edabits::EdaBitsPool::new(
                        lazy_u8,
                        lazy_u16,
                        lazy_u32,
                        lazy_u64,
                        lazy_u128,
                        dabit_setup,
                        512 * trace.len(),
                    )
                };

                <JoltRV64IMAC as Rep3JoltWorker<F, PCS, _>>::prove(
                    &preprocessing,
                    trace,
                    io_device,
                    memory,
                    &mut io_ctx,
                    ram_K,
                    Some(advice),
                    edabits_pool,
                )?;
                Ok(())
            },
            |input, network| {
                let (verifier_preprocessing, io_device, ram_K, trace_length) = input;
                let _span = info_span!("coordinator_prove").entered();
                let proof = <JoltRV64IMAC as Rep3Jolt<F, PCS, _>>::prove(
                    &verifier_preprocessing,
                    io_device,
                    network,
                    ram_K,
                    trace_length,
                )?;
                info!(commitments = proof.commitments.len(), "coordinator done");
                Ok(proof)
            },
        );

        drop(_mpc_span);

        // 7. Compare commitments
        let proof = coordinator_result;
        assert_eq!(
            proof.commitments.len(),
            vanilla_commits.len(),
            "commitment count mismatch: mpc={} vanilla={}",
            proof.commitments.len(),
            vanilla_commits.len()
        );
        for (i, (mpc, vanilla)) in proof
            .commitments
            .iter()
            .zip(vanilla_commits.iter())
            .enumerate()
        {
            assert_eq!(mpc, vanilla, "commitment mismatch at index {i}");
        }
        info!("all {} commitments match!", proof.commitments.len());
    }
}
