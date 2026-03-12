use std::path::PathBuf;

use clap::Parser;
use co_jolt_coordinator::proving::coordinate_once;
use co_jolt_coordinator::transport::ephemeral_identity::EphemeralIdentity;
use eyre::Context;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use tracing::info;

#[derive(Parser)]
struct Args {
    /// Path to network config TOML (required for non-enclave modes)
    #[clap(short = 'c', long)]
    config_file: Option<PathBuf>,

    /// Transport mode: quic or tls (ignored in aws_nitro builds)
    #[clap(long, default_value = "quic")]
    transport: String,

    /// Number of Rayon threads (must match workers for twist_sumcheck_switch_index)
    #[clap(long)]
    rayon_threads: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();

    if let Some(threads) = args.rayon_threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    }

    // 1. Generate ephemeral ECDSA P-256 identity
    let identity = EphemeralIdentity::generate().context("generating ephemeral identity")?;
    info!(pubkey_len = identity.public_key_bytes.len(), "generated ephemeral ECDSA P-256 identity");

    // 2. [if aws_nitro] Request NSM attestation binding the ephemeral pubkey
    #[cfg(feature = "aws_nitro")]
    let attestation_doc: Option<Vec<u8>> = {
        // TODO: integrate aws-nitro-enclaves-nsm-api
        None
    };

    // 3. Select transport and enter proving loop
    #[cfg(feature = "aws_nitro")]
    {
        use co_jolt_coordinator::transport::vsock_tls::VsockTlsCoordinator;

        let vsock_port: u32 =
            std::env::var("VSOCK_PORT").unwrap_or_else(|_| "9000".to_string()).parse().context("parsing VSOCK_PORT")?;

        let mut network = VsockTlsCoordinator::accept(vsock_port, &identity, attestation_doc.as_deref())
            .context("accepting vsock+TLS connections")?;

        info!("accepted 3 worker connections, entering stand-by loop");
        prove_loop(&mut network)?;
    }

    #[cfg(not(feature = "aws_nitro"))]
    {
        let config_file =
            args.config_file.ok_or_else(|| eyre::eyre!("--config-file is required in non-enclave mode"))?;

        match args.transport.as_str() {
            "quic" => {
                use mpc_net::config::{NetworkConfig, NetworkConfigFile};
                use mpc_net::rep3::quic::Rep3QuicNetCoordinator;

                let config: NetworkConfigFile =
                    toml::from_str(&std::fs::read_to_string(&config_file).context("reading config file")?)
                        .context("parsing config file")?;
                let config = NetworkConfig::try_from(config).context("converting network config")?;

                info!("creating QUIC coordinator network");
                let mut network = Rep3QuicNetCoordinator::new(config, 0)?;
                info!("accepted 3 worker connections, entering stand-by loop");
                prove_loop(&mut network)?;
            }
            "tls" => {
                use co_jolt_coordinator::transport::tcp_tls::TcpTlsCoordinator;
                use mpc_net::config::{NetworkConfig, NetworkConfigFile};

                let config: NetworkConfigFile =
                    toml::from_str(&std::fs::read_to_string(&config_file).context("reading config file")?)
                        .context("parsing config file")?;
                let config = NetworkConfig::try_from(config).context("converting network config")?;

                info!("creating TLS coordinator network");
                let mut network = TcpTlsCoordinator::accept(
                    config.bind_addr,
                    &identity,
                    None, // no attestation in emulated mode
                )
                .context("accepting TLS connections")?;
                info!("accepted 3 worker connections, entering stand-by loop");
                prove_loop(&mut network)?;
            }
            other => eyre::bail!("unknown transport: {other} (expected 'quic' or 'tls')"),
        }
    }

    Ok(())
}

fn prove_loop<N: Rep3NetworkCoordinator>(network: &mut N) -> eyre::Result<()> {
    loop {
        coordinate_once(network)?;
    }
}
