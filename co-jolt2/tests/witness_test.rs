//! Integration test for `generate_witness_batch_rep3`.
//!
//! Spawns 3 MPC worker threads connected via local QUIC, runs the MPC witness
//! generation on shared fibonacci traces, reconstructs the polynomials, and
//! compares them against the vanilla (cleartext) witness generation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ark_bn254::Fr;
use ark_std::test_rng;
use tracing::info;

use co_jolt2::host::program::Rep3Program;
use co_jolt2::poly::Rep3MultilinearPolynomial;
use co_jolt2::utils::test_utils::{check_poly, run_rep3_test};
use co_jolt2::utils::tracing::init_tracing;
use co_jolt2::zkvm::instruction::{populate_operands_casts, Rep3Cycle, Rep3Operand};
use co_jolt2::zkvm::witness::generate_witness_batch_rep3;
use jolt_core::host::Program;
use jolt_core::poly::commitment::mock::MockCommitScheme;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::{remap_address, RAMPreprocessing};
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial};
use jolt_core::zkvm::{JoltProverPreprocessing, JoltSharedPreprocessing};
use rayon::prelude::*;
use tracer::instruction::Cycle;

type F = Fr;
type PCS = MockCommitScheme<F>;

// ── Compute ram_K from trace (mirrors vanilla StateManager::new_prover) ─────

fn compute_ram_k(
    trace: &[tracer::instruction::Cycle],
    preprocessing: &JoltSharedPreprocessing,
) -> usize {
    let max_from_trace = trace
        .par_iter()
        .filter_map(|cycle| {
            remap_address(
                cycle.ram_access().address() as u64,
                &preprocessing.memory_layout,
            )
        })
        .max()
        .unwrap_or(0);

    let max_from_bytecode = remap_address(
        preprocessing.ram.min_bytecode_address,
        &preprocessing.memory_layout,
    )
    .unwrap_or(0)
        + preprocessing.ram.bytecode_words.len() as u64
        + 1;

    max_from_trace.max(max_from_bytecode).next_power_of_two() as usize
}

// ── The Test ────────────────────────────────────────────────────────────────

#[test]
fn test_generate_witness_batch_rep3() {
    let _tracing_guard = init_tracing("witness_test.json", Path::new("/tmp/co-jolt2-traces"));

    // 1. Build and trace the fibonacci program
    let mut program = Program::new("fibonacci-guest");
    // Use pre-built ELF to avoid needing the guest package in this workspace.
    // Build with: cd $JOLT_FORK && cargo build --release --features guest -p fibonacci-guest \
    //   --target riscv64imac-unknown-none-elf --target-dir /tmp/jolt-guest-targets/fibonacci-guest-
    let elf_path = "/tmp/jolt-guest-targets/fibonacci-guest-/riscv64imac-unknown-none-elf/release/fibonacci-guest";
    program.elf = Some(PathBuf::from(elf_path));
    let inputs = postcard::to_stdvec(&9u32).unwrap();
    let (bytecode, memory_init, _) = program.decode();

    // 2. Generate trace and shares
    let mut rng = test_rng();
    let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);

    // Also get a vanilla trace for comparison
    let (mut vanilla_trace, _memory, io_device) = program.trace(&inputs, &[], &[]);

    // Pad traces to next power of 2 (mirrors StateManager / DAG init).
    // The +1 accounts for the implicit PC termination cycle.
    let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
    info!(raw_len = vanilla_trace.len(), padded_len, "padding traces");
    vanilla_trace.resize(padded_len, Cycle::NoOp);
    for (trace, _, _) in shares.iter_mut() {
        trace.resize(padded_len, Rep3Cycle::NoOp);
    }

    // 3. Build preprocessing (shared between all parties + vanilla)
    let shared = JoltSharedPreprocessing {
        memory_layout: io_device.memory_layout.clone(),
        bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
        ram: RAMPreprocessing::preprocess(memory_init.clone()),
    };
    let preprocessing: JoltProverPreprocessing<F, PCS> = JoltProverPreprocessing {
        generators: (),
        shared: shared.clone(),
    };

    // 4. Determine which polynomials to test.
    //    Initialize AllCommittedPolynomials (global static) before calling vanilla.
    let ram_K = compute_ram_k(&vanilla_trace, &preprocessing.shared);
    let bytecode_d = preprocessing.shared.bytecode.d;
    let ram_d = compute_d_parameter(ram_K);
    let _guard = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

    let all_polys: Vec<CommittedPolynomial> = AllCommittedPolynomials::iter().copied().collect();

    // Filter to non-one-hot polynomials (the ones our MPC code populates)
    let testable_polys: Vec<CommittedPolynomial> = all_polys
        .iter()
        .copied()
        .filter(|p| {
            !matches!(
                p,
                CommittedPolynomial::InstructionRa(_)
                    | CommittedPolynomial::BytecodeRa(_)
                    | CommittedPolynomial::RamRa(_)
            )
        })
        .collect();

    info!(
        total = all_polys.len(),
        testable = testable_polys.len(),
        "polynomial counts"
    );

    // 5. Run vanilla witness generation (only for testable polys — one-hot polys
    //    require DoryGlobals which we don't initialize for MockCommitScheme)
    info!("running vanilla witness generation");
    let vanilla_results = CommittedPolynomial::generate_witness_batch(
        &testable_polys,
        &preprocessing,
        &vanilla_trace,
    );
    info!(
        count = vanilla_results.len(),
        "vanilla witness generation complete"
    );

    // 6. Run MPC witness generation on 3 parties
    let preprocessing_arc = Arc::new(preprocessing);

    // Pick a base port unlikely to collide with other tests
    let base_port: u16 = 14200;

    info!("launching 3-party MPC witness generation");
    let mpc_results: [HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>; 3] =
        run_rep3_test(
            base_port,
            4, // num_io_forks
            |party_idx| {
                let (trace, _memory, _io) = shares[party_idx].clone();
                let preprocessing = Arc::clone(&preprocessing_arc);
                (trace, preprocessing, testable_polys.clone())
            },
            |input, io_ctx| {
                let (mut trace, preprocessing, polys) = input;
                let party = io_ctx.party_id();

                // Populate arithmetic shares from binary shares (requires network)
                info!(?party, "populate_operands_casts start");
                populate_operands_casts(&mut trace, io_ctx.main())?;
                info!(?party, "populate_operands_casts done");

                // Verify arithmetic shares are populated
                let mut unpopulated = 0usize;
                let mut total_shared = 0usize;
                for cycle in trace.iter_mut() {
                    for op in cycle.shared_operands_mut() {
                        if let Rep3Operand::Shared { arithmetic, .. } = op {
                            total_shared += 1;
                            if arithmetic.is_none() {
                                unpopulated += 1;
                            }
                        }
                    }
                }
                info!(?party, total_shared, unpopulated, "operand check");
                assert_eq!(unpopulated, 0, "unpopulated arithmetic shares remain");

                // Generate witness polynomials
                info!(?party, "generate_witness_batch_rep3 start");
                let results = generate_witness_batch_rep3::<F, PCS, _>(
                    &polys,
                    &preprocessing,
                    &trace,
                    io_ctx,
                )?;
                info!(
                    ?party,
                    count = results.len(),
                    "generate_witness_batch_rep3 done"
                );
                Ok(results)
            },
        );

    info!("MPC witness generation complete, reconstructing");

    // 7. Reconstruct and compare
    for poly_key in &testable_polys {
        let vanilla_poly = match vanilla_results.get(poly_key) {
            Some(p) => p,
            None => continue,
        };

        // Collect the 3 shares for this polynomial
        let share_polys: Vec<Rep3MultilinearPolynomial<F>> = (0..3)
            .map(|i| {
                mpc_results[i]
                    .get(poly_key)
                    .unwrap_or_else(|| panic!("party {i} missing poly {poly_key:?}"))
                    .clone()
            })
            .collect();

        match &share_polys[0] {
            Rep3MultilinearPolynomial::Public(pub_poly) => {
                check_poly(pub_poly, vanilla_poly, &format!("{poly_key:?} (public)"));
            }
            Rep3MultilinearPolynomial::Shared(_) => {
                let reconstructed = Rep3MultilinearPolynomial::combine_shares(share_polys);
                check_poly(
                    &reconstructed,
                    vanilla_poly,
                    &format!("{poly_key:?} (shared)"),
                );
            }
        }
    }

    info!("all polynomials match!");
}
