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
pub fn run_rep3_test<I, O, W>(
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

/// Compare two multilinear polynomials coefficient-by-coefficient.
/// Panics with a detailed mismatch report if they differ.
pub fn check_poly<F: JoltField>(
    poly: &MultilinearPolynomial<F>,
    check: &MultilinearPolynomial<F>,
    label: &str,
) {
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
