use co_jolt_coordinator::transport::ephemeral_identity::EphemeralIdentity;
use eyre::Context;
use tracing::info;

fn main() -> eyre::Result<()> {
    // 1. Generate ephemeral ECDSA P-256 identity
    let identity = EphemeralIdentity::generate().context("generating ephemeral identity")?;
    info!(
        pubkey_len = identity.public_key_bytes.len(),
        "generated ephemeral ECDSA P-256 identity"
    );

    // 2. [if aws_nitro] Request NSM attestation binding the ephemeral pubkey
    #[cfg(feature = "aws_nitro")]
    let attestation_doc: Option<Vec<u8>> = {
        // TODO: integrate aws-nitro-enclaves-nsm-api
        // let nsm_fd = nsm_init();
        // let doc = nsm_process_request(nsm_fd, Request::Attestation {
        //     public_key: Some(identity.public_key_bytes.clone()),
        //     user_data: None,
        //     nonce: None,
        // })?;
        // Some(doc)
        None
    };

    #[cfg(not(feature = "aws_nitro"))]
    let attestation_doc: Option<Vec<u8>> = None;

    // 3. Accept 3 worker connections over vsock+TLS
    #[cfg(feature = "aws_nitro")]
    {
        use co_jolt_coordinator::transport::vsock_tls::VsockTlsCoordinator;

        let vsock_port: u32 = std::env::var("VSOCK_PORT")
            .unwrap_or_else(|_| "9000".to_string())
            .parse()
            .context("parsing VSOCK_PORT")?;

        let mut _network = VsockTlsCoordinator::accept(
            vsock_port,
            &identity,
            attestation_doc.as_deref(),
        )
        .context("accepting vsock+TLS connections")?;

        info!("accepted 3 worker connections");

        // TODO: receive trace metadata, run preprocessing, drive proof:
        // <JoltRV32IM as Rep3Jolt<F, PCS, _>>::prove(
        //     &verifier_preprocessing,
        //     &pcs_setup,
        //     io_device,
        //     &mut network,
        //     ram_k,
        //     trace_length,
        // )?;
    }

    #[cfg(not(feature = "aws_nitro"))]
    {
        let _ = attestation_doc;
        info!("coordinator stub (no aws_nitro feature) — nothing to do");
    }

    Ok(())
}
