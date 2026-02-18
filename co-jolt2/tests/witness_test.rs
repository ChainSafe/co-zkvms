//! Integration test for `generate_witness_batch_rep3`.
//!
//! Spawns 3 MPC worker threads connected via local QUIC, runs the MPC witness
//! generation on shared fibonacci traces, reconstructs the polynomials, and
//! compares them against the vanilla (cleartext) witness generation.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::thread;

use ark_bn254::Fr;
use ark_std::test_rng;
use eyre::Context;
use tracing::info;
use tracing_chrome::ChromeLayerBuilder;
use tracing_forest::ForestLayer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;

use co_jolt2::host::program::Rep3Program;
use co_jolt2::poly::Rep3MultilinearPolynomial;
use co_jolt2::zkvm::instruction::{populate_operands_casts, Rep3Cycle, Rep3Operand};
use co_jolt2::zkvm::witness::generate_witness_batch_rep3;
use jolt_core::host::Program;
use jolt_core::poly::commitment::mock::MockCommitScheme;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::{remap_address, RAMPreprocessing};
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial};
use jolt_core::zkvm::{JoltProverPreprocessing, JoltSharedPreprocessing};
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3MpcNet};
use mpc_net::config::{Address, NetworkConfig, NetworkParty};
use rayon::prelude::*;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tracer::instruction::Cycle;

type F = Fr;
type PCS = MockCommitScheme<F>;

// ── Tracing ─────────────────────────────────────────────────────────────────

struct TracingGuard {
    _guard: Option<tracing_chrome::FlushGuard>,
    file: String,
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Some(ref file) = Some(&self.file) {
            info!("tracing_chrome flushing to {file}");
        }
    }
}

fn init_tracing(file: &str, trace_dir: &Path) -> Option<TracingGuard> {
    std::fs::create_dir_all(trace_dir).unwrap();
    let trace_path = trace_dir.join(file);
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("jolt_core=info".parse().unwrap())
        .add_directive("co_jolt2=info".parse().unwrap())
        .add_directive("mpc_net=info".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap());

    let current_level = env_filter.max_level_hint().unwrap_or(LevelFilter::INFO);
    let subscriber = Registry::default().with(env_filter);

    if current_level == LevelFilter::TRACE {
        let (chrome_layer, _guard) = ChromeLayerBuilder::new().file(trace_path).build();
        let _ = tracing::subscriber::set_global_default(
            subscriber
                .with(chrome_layer)
                .with(ForestLayer::default().with_filter(LevelFilter::TRACE)),
        );
        info!("tracing_chrome writes to file: {}", file);
        Some(TracingGuard {
            _guard: Some(_guard),
            file: file.to_string(),
        })
    } else {
        let _ = tracing::subscriber::set_global_default(subscriber.with(ForestLayer::default()));
        None
    }
}

// ── Test Network Helpers ────────────────────────────────────────────────────

/// Generate a self-signed certificate + private key for localhost.
fn generate_cert() -> (CertificateDer<'static>, PrivateKeyDer<'static>) {
    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string(), "127.0.0.1".to_string()])
            .expect("cert generation");
    let cert_der = CertificateDer::from(cert.der().to_vec());
    let key_der =
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pair.serialize_der())).clone_key();
    (cert_der, key_der)
}

/// Build `NetworkConfig`s for 3 localhost parties using the given base port.
fn build_test_configs(base_port: u16) -> [NetworkConfig; 3] {
    let certs_keys: Vec<_> = (0..3).map(|_| generate_cert()).collect();

    let parties: Vec<NetworkParty> = (0..3)
        .map(|i| NetworkParty {
            id: i,
            worker: 0,
            dns_name: Address::new("localhost".into(), base_port + i as u16),
            cert: certs_keys[i].0.clone(),
        })
        .collect();

    std::array::from_fn(|i| NetworkConfig {
        parties: parties.clone(),
        coordinator: None,
        is_coordinator: false,
        my_id: i,
        worker: 0,
        bind_addr: SocketAddr::new(IpAddr::from_str("127.0.0.1").unwrap(), base_port + i as u16),
        key: certs_keys[i].1.clone_key(),
        timeout: Some(std::time::Duration::from_secs(30)),
    })
}

/// Spawn 3 MPC worker threads. Each worker receives its input via the closure
/// `make_input(party_index)`, runs `work_fn`, and returns the result.
///
/// Returns an array of 3 results.
fn run_3party_test<I, O, W>(
    base_port: u16,
    num_io_forks: u32,
    make_input: impl Fn(usize) -> I,
    work_fn: W,
) -> [O; 3]
where
    I: Send + 'static,
    O: Send + 'static,
    W: Fn(I, &mut IoContextPool<Rep3MpcNet>) -> eyre::Result<O> + Send + Sync + 'static,
{
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .ok(); // ignore if already installed

    let configs = build_test_configs(base_port);
    let work_fn = Arc::new(work_fn);

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let config = configs[i].clone();
            let input = make_input(i);
            let work_fn = Arc::clone(&work_fn);
            thread::spawn(move || {
                // Each party gets its own rayon thread pool to avoid deadlocks.
                // In production each party is a separate process; in tests they
                // share a process and would contend on the global rayon pool.
                let pool = rayon::ThreadPoolBuilder::new()
                    .thread_name(move |idx| format!("party-{i}-rayon-{idx}"))
                    .build()
                    .unwrap();
                pool.install(|| {
                    let network = Rep3MpcNet::new(config, 0)
                        .with_context(|| format!("party {i} network init"))
                        .unwrap();
                    let mut io_ctx = IoContextPool::init(network, num_io_forks)
                        .with_context(|| format!("party {i} io_ctx init"))
                        .unwrap();
                    work_fn(input, &mut io_ctx)
                        .with_context(|| format!("party {i} work"))
                        .unwrap()
                })
            })
        })
        .collect();

    let results: Vec<O> = handles
        .into_iter()
        .map(|h| h.join().expect("worker thread panicked"))
        .collect();

    results.try_into().unwrap_or_else(|_| unreachable!())
}

// ── Polynomial Comparison ───────────────────────────────────────────────────

fn check_poly(poly: &MultilinearPolynomial<F>, check: &MultilinearPolynomial<F>, label: &str) {
    assert_eq!(poly.len(), check.len(), "len mismatch {label}");
    let len = poly.len();
    let mut mismatches = Vec::new();
    for i in 0..len {
        let a = poly.get_coeff(i);
        let b = check.get_coeff(i);
        if a != b {
            mismatches.push((i, a, b));
        }
    }
    if !mismatches.is_empty() {
        eprintln!(
            "[check_poly] {label}: {}/{len} mismatches",
            mismatches.len()
        );
        for &(i, a, b) in mismatches.iter().take(20) {
            eprintln!("  pos {i}: mpc={a} vanilla={b}");
        }
        panic!(
            "{label}: {} mismatches (first at pos {})",
            mismatches.len(),
            mismatches[0].0,
        );
    }
}

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
        run_3party_test(
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
