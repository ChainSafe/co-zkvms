use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;
use std::sync::Arc;
use std::thread;

use eyre::Context;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

use jolt_core::field::JoltField;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3MpcNet};
use mpc_net::config::{Address, NetworkConfig, NetworkParty};
use mpc_net::rep3::quic::Rep3QuicNetCoordinator;

// ── Test Network Helpers ────────────────────────────────────────────────────

/// Generate a self-signed certificate + private key for localhost.
fn generate_cert() -> (CertificateDer<'static>, PrivateKeyDer<'static>) {
    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string(), "127.0.0.1".to_string()])
            .expect("cert generation");
    let cert_der = CertificateDer::from(cert.der().to_vec());
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pair.serialize_der())).clone_key();
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
            protocol: Default::default(),
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
        user_listen_addr: None,
    })
}

/// Spawn 3 MPC worker threads. Each worker receives its input via the closure
/// `make_input(party_index)`, runs `work_fn`, and returns the result.
///
/// Returns an array of 3 results.
pub fn run_rep3_test<I, O, W>(base_port: u16, num_io_forks: u32, make_input: impl Fn(usize) -> I, work_fn: W) -> [O; 3]
where
    I: Send + 'static,
    O: Send + 'static,
    W: Fn(I, IoContextPool<Rep3MpcNet>) -> eyre::Result<O> + Send + Sync + 'static,
{
    rustls::crypto::aws_lc_rs::default_provider().install_default().ok(); // ignore if already installed

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
                    let network =
                        Rep3MpcNet::new(config, 0).with_context(|| format!("party {i} network init")).unwrap();
                    let io_ctx = IoContextPool::init(network, num_io_forks)
                        .with_context(|| format!("party {i} io_ctx init"))
                        .unwrap();
                    work_fn(input, io_ctx).with_context(|| format!("party {i} work")).unwrap()
                })
            })
        })
        .collect();

    let results: Vec<O> = handles.into_iter().map(|h| h.join().expect("worker thread panicked")).collect();

    results.try_into().unwrap_or_else(|_| unreachable!())
}

// ── Test with Coordinator ───────────────────────────────────────────────────

/// Build `NetworkConfig`s for 3 worker parties + 1 coordinator on localhost.
/// Workers get ports `base_port..base_port+2`, coordinator gets `base_port+3`.
fn build_test_configs_with_coordinator(base_port: u16) -> ([NetworkConfig; 3], NetworkConfig) {
    let certs_keys: Vec<_> = (0..4).map(|_| generate_cert()).collect();

    // Worker parties (indices 0..3)
    let worker_parties: Vec<NetworkParty> = (0..3)
        .map(|i| NetworkParty {
            id: i,
            worker: 0,
            dns_name: Address::new("localhost".into(), base_port + i as u16),
            cert: certs_keys[i].0.clone(),
            protocol: Default::default(),
        })
        .collect();

    // Coordinator party
    let coordinator_party = NetworkParty {
        id: 0,
        worker: 0,
        dns_name: Address::new("localhost".into(), base_port + 3),
        cert: certs_keys[3].0.clone(),
        protocol: Default::default(),
    };

    let worker_configs: [NetworkConfig; 3] = std::array::from_fn(|i| NetworkConfig {
        parties: worker_parties.clone(),
        coordinator: Some(coordinator_party.clone()),
        is_coordinator: false,
        my_id: i,
        worker: 0,
        bind_addr: SocketAddr::new(IpAddr::from_str("127.0.0.1").unwrap(), base_port + i as u16),
        key: certs_keys[i].1.clone_key(),
        timeout: Some(std::time::Duration::from_secs(30)),
        user_listen_addr: None,
    });

    let coordinator_config = NetworkConfig {
        parties: worker_parties,
        coordinator: Some(coordinator_party),
        is_coordinator: true,
        my_id: 0,
        worker: 0,
        bind_addr: SocketAddr::new(IpAddr::from_str("127.0.0.1").unwrap(), base_port + 3),
        key: certs_keys[3].1.clone_key(),
        timeout: Some(std::time::Duration::from_secs(30)),
        user_listen_addr: None,
    };

    (worker_configs, coordinator_config)
}

/// Spawn 3 MPC worker threads + 1 coordinator thread.
///
/// Each worker receives its input via `make_worker_input(party_index)` and runs
/// `worker_fn` with an `IoContextPool` (passed by value for ownership transfer).
/// The coordinator receives its input via `make_coordinator_input()` and runs
/// `coordinator_fn` with a `Rep3QuicNetCoordinator`.
///
/// Returns `([worker_results; 3], coordinator_result)`.
pub fn run_rep3_test_with_coordinator<WI, WO, CI, CO, WF, CF>(
    base_port: u16,
    num_io_forks: u32,
    make_worker_input: impl Fn(usize) -> WI,
    make_coordinator_input: impl FnOnce() -> CI,
    worker_fn: WF,
    coordinator_fn: CF,
) -> ([WO; 3], CO)
where
    WI: Send + 'static,
    WO: Send + 'static,
    CI: Send + 'static,
    CO: Send + 'static,
    WF: Fn(WI, IoContextPool<Rep3MpcNet>) -> eyre::Result<WO> + Send + Sync + 'static,
    CF: FnOnce(CI, &mut Rep3QuicNetCoordinator) -> eyre::Result<CO> + Send + 'static,
{
    rustls::crypto::aws_lc_rs::default_provider().install_default().ok();

    let (worker_configs, coordinator_config) = build_test_configs_with_coordinator(base_port);
    let worker_fn = Arc::new(worker_fn);

    // Spawn worker threads
    let worker_handles: Vec<_> = (0..3)
        .map(|i| {
            let config = worker_configs[i].clone();
            let input = make_worker_input(i);
            let worker_fn = Arc::clone(&worker_fn);
            thread::spawn(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .thread_name(move |idx| format!("party-{i}-rayon-{idx}"))
                    .build()
                    .unwrap();
                pool.install(|| {
                    let network =
                        Rep3MpcNet::new(config, 0).with_context(|| format!("party {i} network init")).unwrap();
                    let io_ctx = IoContextPool::init(network, num_io_forks)
                        .with_context(|| format!("party {i} io_ctx init"))
                        .unwrap();
                    worker_fn(input, io_ctx).with_context(|| format!("party {i} work")).unwrap()
                })
            })
        })
        .collect();

    // Spawn coordinator thread
    let coordinator_input = make_coordinator_input();
    let coordinator_handle = thread::spawn(move || {
        let pool =
            rayon::ThreadPoolBuilder::new().thread_name(|idx| format!("coordinator-rayon-{idx}")).build().unwrap();
        pool.install(|| {
            let mut network =
                Rep3QuicNetCoordinator::new(coordinator_config, 0).context("coordinator network init").unwrap();
            coordinator_fn(coordinator_input, &mut network).context("coordinator work").unwrap()
        })
    });

    // Collect results
    let worker_results: Vec<WO> =
        worker_handles.into_iter().map(|h| h.join().expect("worker thread panicked")).collect();
    let coordinator_result = coordinator_handle.join().expect("coordinator thread panicked");

    let worker_array = worker_results.try_into().unwrap_or_else(|_| unreachable!());
    (worker_array, coordinator_result)
}

#[cfg(feature = "test-utils")]
pub use mpc_core::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;

// ── Polynomial Comparison ───────────────────────────────────────────────────

/// Compare two multilinear polynomials coefficient-by-coefficient.
/// Panics with a detailed mismatch report if they differ.
pub fn check_poly<F: JoltField>(poly: &MultilinearPolynomial<F>, check: &MultilinearPolynomial<F>, label: &str) {
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
        panic!("{label}: {} mismatches (first at pos {}, len {len})", mismatches.len(), mismatches[0].0,);
    }
}
