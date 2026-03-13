use color_eyre::{eyre::Context, Result};
use mpc_net::config::{Address, CoordinatorProtocol, NetworkConfig};
use rcgen::CertifiedKey;
use serde::Serialize;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;

/// Certificate Generator for MPC-NET
#[derive(Debug, PartialEq, Parser)]
struct CliArgs {
    /// The path to config files
    #[clap(short, long, default_value = "./examples/")]
    out_dir: PathBuf,

    /// The path to the .der certificate file
    #[clap(short, long, default_value = "./data/")]
    cert_dir: PathBuf,
    /// The path to the .der key file
    #[clap(short, long, default_value = "./data/")]
    key_dir: PathBuf,

    // /// The subject alternative names for the certificate
    // #[clap(short, long)]
    // sans: Vec<String>,
    /// The number of workers to generate configs for
    #[clap(short, long)]
    num_workers: usize,

    /// Base port for user-facing TLS listener on workers (port = base + party_id).
    /// When set, each worker config gets a `user_listen_addr`.
    #[clap(long)]
    user_listen_base_port: Option<u16>,

    /// Base port for inter-party QUIC/TLS ring (port = base + party_id).
    #[clap(long, default_value = "10000")]
    inter_party_base_port: u16,

    /// Coordinator bind port.
    #[clap(long, default_value = "20000")]
    coordinator_port: u16,

    /// Coordinator protocol: quic or tls (default: quic).
    #[clap(long, default_value = "quic")]
    coordinator_protocol: String,

    /// Override coordinator address in worker configs (default: localhost:<coordinator-port>).
    #[clap(long)]
    coordinator_addr: Option<String>,
}

#[derive(Serialize)]
struct DelegatorConfig {
    workers: Vec<String>,
}

fn main() -> Result<()> {
    let args = CliArgs::parse();

    let coordinator_protocol = match args.coordinator_protocol.as_str() {
        "quic" => CoordinatorProtocol::Quic,
        "tls" => CoordinatorProtocol::Tls,
        other => {
            return Err(color_eyre::eyre::eyre!("unknown coordinator protocol: {other} (expected 'quic' or 'tls')"))
        }
    };

    let data_dir = args.cert_dir.to_str().expect("cert_dir must be valid UTF-8");
    let (mut workers, mut coordinator) = NetworkConfig::generate_worker_configs_full(
        args.num_workers,
        data_dir,
        args.inter_party_base_port,
        args.coordinator_port,
    );

    // Apply user_listen_addr to worker configs
    if let Some(base_port) = args.user_listen_base_port {
        for (id, config) in workers.iter_mut() {
            let party_id = usize::from(id.party_id()) as u16;
            config.user_listen_addr = Some(SocketAddr::new(IpAddr::from_str("0.0.0.0").unwrap(), base_port + party_id));
        }

        let delegator = DelegatorConfig {
            workers: workers
                .iter()
                .filter_map(|(_, config)| config.user_listen_addr.map(|addr| format!("127.0.0.1:{}", addr.port())))
                .collect(),
        };
        let toml = toml::to_string_pretty(&delegator).context("serializing delegator config")?;
        std::fs::write(args.out_dir.join("config_delegator.toml"), toml).context("writing delegator config")?;
    }

    // Apply coordinator protocol
    if let Some(ref mut coord) = coordinator.coordinator {
        coord.protocol = coordinator_protocol;
    }
    for (_, config) in workers.iter_mut() {
        if let Some(ref mut coord) = config.coordinator {
            coord.protocol = coordinator_protocol;
        }
    }

    // Override coordinator address if specified
    if let Some(ref addr) = args.coordinator_addr {
        let parsed: Address =
            addr.parse().map_err(|e| color_eyre::eyre::eyre!("parsing --coordinator-addr '{addr}': {e}"))?;
        for (_, config) in workers.iter_mut() {
            if let Some(ref mut coord) = config.coordinator {
                coord.dns_name = parsed.clone();
            }
        }
    }

    for (id, config) in &workers {
        let toml = toml::to_string_pretty(config).context("serializing config")?;
        std::fs::write(
            args.out_dir.join(format!("config_worker{}_{}.toml", id.worker_idx(), id.party_id() as usize)),
            toml,
        )
        .context("writing config file")?;

        // Worker key & certificate
        let CertifiedKey { cert, key_pair } = rcgen::generate_simple_self_signed(vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
            format!("worker{}_{}", id.worker_idx(), id.party_id()),
        ])
        .context("generating self-signed cert")?;
        let key = key_pair.serialize_der();
        std::fs::write(args.key_dir.join(format!("key{}_{}.der", id.worker_idx(), id.party_id())), key)
            .context("writing key file")?;
        let cert = cert.der();
        std::fs::write(args.cert_dir.join(format!("cert{}_{}.der", id.worker_idx(), id.party_id())), cert)
            .context("writing certificate file")?;
    }

    let toml = toml::to_string_pretty(&coordinator).context("serializing config")?;
    std::fs::write(args.out_dir.join(format!("config_coordinator.toml")), toml).context("writing config file")?;

    // Coordinator key & certificate
    {
        let CertifiedKey { cert, key_pair } = rcgen::generate_simple_self_signed(vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
            "coordinator".to_string(),
        ])
        .context("generating self-signed cert")?;
        let key = key_pair.serialize_der();
        std::fs::write(args.key_dir.join("key_coordinator.der"), key).context("writing key file")?;
        let cert = cert.der();
        std::fs::write(args.cert_dir.join("cert_coordinator.der"), cert).context("writing certificate file")?;
    }

    Ok(())
}
