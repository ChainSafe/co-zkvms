use std::net::SocketAddr;
use std::path::PathBuf;

use ::guest::{
    build_delegate_verify_dkim, build_verifier_verify_dkim, compile_verify_dkim, memory_config_verify_dkim, verify_dkim,
};
use ark_bn254::Fr;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use cfdkim::{dns::from_tokio_resolver, public_key::retrieve_public_key};
use clap::Parser;
use eyre::Context;
use mailparse::MailHeaderMap;
use serde::Deserialize;
use slog::{o, Discard, Logger};
use tracing::info;
use tracing_forest::ForestLayer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::{EnvFilter, Layer};
use trust_dns_resolver::TokioAsyncResolver;
use zkemail_core::{DKIMInput, DKIMOutput};

use jolt_sdk::*;

type F = Fr;
type PCS = jolt_sdk::PCS;

#[derive(Deserialize)]
struct DelegatorConfig {
    workers: Vec<String>,
}

#[derive(Parser)]
struct Args {
    /// Path to delegator config TOML.
    #[clap(long, default_value = ".artifacts/config_delegator.toml")]
    config_path: PathBuf,

    /// Path to the .eml file
    #[clap(long)]
    email_path: PathBuf,

    /// Expected sender domain (e.g., "google.com")
    #[clap(long)]
    from_domain: String,
}

fn init_tracing() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("rustls=off".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap())
        .add_directive("dory=off".parse().unwrap());

    let _ = tracing::subscriber::set_global_default(
        Registry::default().with(env_filter).with(ForestLayer::default().with_filter(LevelFilter::INFO)),
    );
}

/// Extract the value of a DKIM tag (e.g., "s", "d", "b", "h", "c", "a") from a DKIM-Signature header value.
fn get_dkim_tag(header_value: &str, tag: &str) -> Option<String> {
    let prefix = format!("{}=", tag);
    for part in header_value.split(';') {
        let trimmed = part.trim();
        if trimmed.starts_with(&prefix) {
            return Some(trimmed[prefix.len()..].trim().to_string());
        }
    }
    None
}

/// Perform relaxed header canonicalization per RFC 6376 Section 3.4.2:
/// - Convert header name to lowercase
/// - Unfold header continuation lines
/// - Reduce sequences of WSP to a single SP
/// - Strip trailing WSP before CRLF
fn canonicalize_header_relaxed(name: &str, value: &str) -> String {
    let lowered_name = name.to_lowercase();
    // Unfold: remove CRLF followed by WSP
    let unfolded = value.replace("\r\n", "").replace('\n', "");
    // Reduce runs of whitespace to a single space, trim leading/trailing
    let reduced: String = unfolded.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("{}:{}", lowered_name, reduced)
}

/// Build the canonicalized header block that was signed by DKIM.
/// Returns the bytes that the RSA signature covers.
fn build_signed_headers(raw_email: &[u8], dkim_header_value: &str, signed_header_names: &str) -> eyre::Result<Vec<u8>> {
    let parsed = mailparse::parse_mail(raw_email)?;

    let header_names: Vec<&str> = signed_header_names.split(':').map(|s| s.trim()).collect();

    let mut lines = Vec::new();
    for name in &header_names {
        let name_lower = name.to_lowercase();
        if let Some(header) = parsed.headers.iter().find(|h| h.get_key().to_lowercase() == name_lower) {
            lines.push(canonicalize_header_relaxed(&header.get_key(), &header.get_value()));
        }
    }

    // Append the DKIM-Signature header itself with the b= value emptied
    let dkim_no_sig = remove_b_value(dkim_header_value);
    lines.push(canonicalize_header_relaxed("dkim-signature", &dkim_no_sig));

    // Join with CRLF; the final header does NOT get a trailing CRLF
    let result = lines.join("\r\n");
    Ok(result.into_bytes())
}

/// Remove the b= tag value from a DKIM-Signature header value,
/// leaving "b=" with an empty value.
fn remove_b_value(header_value: &str) -> String {
    let mut result = String::new();
    let mut in_b_tag = false;
    for part in header_value.split(';') {
        if !result.is_empty() {
            result.push(';');
        }
        let trimmed = part.trim();
        if trimmed.starts_with("b=") && !trimmed.starts_with("bh=") {
            // Keep "b=" but empty the value
            result.push_str(" b=");
            in_b_tag = true;
        } else {
            in_b_tag = false;
            result.push_str(part);
        }
    }
    let _ = in_b_tag;
    result
}

/// Prepare DKIM input by parsing the email, looking up DNS, and extracting
/// the canonicalized signed headers + RSA public key + signature.
async fn prepare_dkim_input(email_path: &PathBuf, from_domain: &str) -> eyre::Result<DKIMInput> {
    let logger = Logger::root(Discard, o!());
    let raw_email = std::fs::read(email_path).context("reading email file")?;
    let parsed = mailparse::parse_mail(&raw_email).map_err(|e| eyre::eyre!("parse email: {}", e))?;

    // Find DKIM-Signature header for the target domain
    let dkim_headers = parsed.headers.get_all_headers("DKIM-Signature");
    if dkim_headers.is_empty() {
        return Err(eyre::eyre!("no DKIM-Signature headers found"));
    }

    let resolver = TokioAsyncResolver::tokio_from_system_conf().map_err(|e| eyre::eyre!("DNS resolver init: {}", e))?;
    let cfdkim_resolver = from_tokio_resolver(resolver);

    let mut found_header_value = None;
    let mut found_public_key_der = None;

    for header in &dkim_headers {
        let header_value = String::from_utf8_lossy(header.get_value_raw()).to_string();

        let d = match get_dkim_tag(&header_value, "d") {
            Some(d) => d,
            None => continue,
        };
        if d.to_lowercase() != from_domain.to_lowercase() {
            continue;
        }

        let algo = match get_dkim_tag(&header_value, "a") {
            Some(a) => a,
            None => continue,
        };
        if !algo.starts_with("rsa-") {
            continue;
        }

        let selector = match get_dkim_tag(&header_value, "s") {
            Some(s) => s,
            None => continue,
        };
        match retrieve_public_key(&logger, cfdkim_resolver.clone(), from_domain.to_string(), selector).await {
            Ok(pk) => {
                found_public_key_der = Some(pk.to_vec() as Vec<u8>);
                found_header_value = Some(header_value);
                break;
            }
            Err(e) => {
                info!(error = %e, "retrieve_public_key failed");
                continue;
            }
        }
    }

    let header_value = found_header_value.ok_or_else(|| eyre::eyre!("no matching DKIM header found"))?;
    let public_key_der = found_public_key_der.ok_or_else(|| eyre::eyre!("no public key retrieved"))?;

    // Extract signature bytes from b= tag
    let b_value = get_dkim_tag(&header_value, "b").ok_or_else(|| eyre::eyre!("missing b= tag"))?;
    let b_clean: String = b_value.chars().filter(|c| !c.is_whitespace()).collect();
    let signature = BASE64.decode(&b_clean).context("base64 decode DKIM signature")?;

    // Extract signed header names from h= tag
    let h_value = get_dkim_tag(&header_value, "h").ok_or_else(|| eyre::eyre!("missing h= tag"))?;

    // Build canonicalized signed headers
    let signed_headers = build_signed_headers(&raw_email, &header_value, &h_value)?;

    info!(
        domain = from_domain,
        signed_headers_len = signed_headers.len(),
        signature_len = signature.len(),
        public_key_len = public_key_der.len(),
        "prepared DKIM input"
    );

    Ok(DKIMInput { signed_headers, signature, public_key_der, from_domain: from_domain.as_bytes().to_vec() })
}

fn main() -> eyre::Result<()> {
    init_tracing();

    let args = Args::parse();
    let config: DelegatorConfig =
        toml::from_str(&std::fs::read_to_string(&args.config_path).context("reading config")?)
            .context("parsing config")?;

    // Parse worker addresses
    let addrs: Vec<SocketAddr> = config
        .workers
        .iter()
        .map(|s| s.trim().parse::<SocketAddr>())
        .collect::<Result<_, _>>()
        .context("parsing worker addresses")?;
    let worker_addrs: [SocketAddr; 3] =
        addrs.try_into().map_err(|v: Vec<_>| eyre::eyre!("expected 3 worker addresses, got {}", v.len()))?;

    // Prepare DKIM input (async DNS lookup)
    let rt = tokio::runtime::Runtime::new()?;
    let dkim_input = rt.block_on(prepare_dkim_input(&args.email_path, &args.from_domain))?;

    // Compile guest program (before connecting so failures don't drop worker connections)
    let target_dir = "/tmp/jolt-guest-targets";
    let mut preprocessing_program = compile_verify_dkim(target_dir);
    let delegate = build_delegate_verify_dkim(compile_verify_dkim(target_dir));

    // Native execution
    let native_output: DKIMOutput = verify_dkim(dkim_input.clone());
    info!(?native_output, "native DKIM verification result");

    // Connect to workers
    info!(?worker_addrs, "connecting to workers");
    let mut client = Client::connect(worker_addrs)?;
    info!("connected to all 3 workers");

    // Delegate proof to workers
    info!("delegating proof...");
    let program_id = "zkemail-verify";
    let (output, proof, program_io) = delegate(&mut client, dkim_input, program_id)?;
    info!(trace_length = proof.trace_length, "proof received");

    // Verify the proof
    let (bytecode, memory_init, program_size) = preprocessing_program.decode();
    let mut memory_config = memory_config_verify_dkim();
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    let prover_preprocessing: JoltProverPreprocessing<F, PCS> =
        JoltRVArch::prover_preprocess(bytecode, memory_layout, memory_init, proof.trace_length);
    let verifier = build_verifier_verify_dkim(JoltVerifierPreprocessing::from(&prover_preprocessing));
    info!("verifying proof...");
    let is_valid = verifier(output.clone(), program_io.panic, proof);

    if !is_valid {
        return Err(eyre::eyre!("proof verification failed"));
    }
    if output != native_output {
        return Err(eyre::eyre!("output mismatch with native execution"));
    }

    info!(verified = output.verified, "proof verified successfully!");
    Ok(())
}
