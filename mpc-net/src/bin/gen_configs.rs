use color_eyre::{eyre::Context, Result};
use mpc_net::config::NetworkConfig;
use rcgen::CertifiedKey;
use std::path::PathBuf;

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
}

fn main() -> Result<()> {
    let args = CliArgs::parse();

    let data_dir = args.cert_dir.to_str().expect("cert_dir must be valid UTF-8");
    let (workers, coordinator) =
        NetworkConfig::generate_worker_configs_with_dir(args.num_workers, data_dir);

    for (id, config) in workers {
        let toml = toml::to_string_pretty(&config).context("serializing config")?;
        std::fs::write(
            args.out_dir.join(format!(
                "config_worker{}_{}.toml",
                id.worker_idx(),
                id.party_id() as usize
            )),
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
        std::fs::write(
            args.key_dir
                .join(format!("key{}_{}.der", id.worker_idx(), id.party_id())),
            key,
        )
        .context("writing key file")?;
        let cert = cert.der();
        std::fs::write(
            args.cert_dir
                .join(format!("cert{}_{}.der", id.worker_idx(), id.party_id())),
            cert,
        )
        .context("writing certificate file")?;
    }

    let toml = toml::to_string_pretty(&coordinator).context("serializing config")?;
    std::fs::write(args.out_dir.join(format!("config_coordinator.toml")), toml)
        .context("writing config file")?;

    // Coordinator key & certificate
    {
        let CertifiedKey { cert, key_pair } = rcgen::generate_simple_self_signed(vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
            "coordinator".to_string(),
        ])
        .context("generating self-signed cert")?;
        let key = key_pair.serialize_der();
        std::fs::write(args.key_dir.join("key_coordinator.der"), key)
            .context("writing key file")?;
        let cert = cert.der();
        std::fs::write(args.cert_dir.join("cert_coordinator.der"), cert)
            .context("writing certificate file")?;
    }

    Ok(())
}
