use std::io::{Read, Write};
use std::sync::Arc;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::{eyre, Context};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_net::topology::MpcStarNetCoordinator;
use mpc_core::protocols::rep3::PartyID;
use vsock::VsockListener;

use super::ephemeral_identity::EphemeralIdentity;

const NUM_PARTIES: usize = 3;
const VMADDR_CID_ANY: u32 = u32::MAX; // VMADDR_CID_ANY = -1 as u32

type TlsStream = rustls::StreamOwned<rustls::ServerConnection, vsock::VsockStream>;

/// Coordinator network transport over vsock + TLS.
///
/// Listens on a vsock port inside the enclave, accepts TLS connections
/// from 3 worker parties. The host EC2 runs an untrusted TCP→vsock proxy
/// so workers connect via standard TCP.
pub struct VsockTlsCoordinator {
    /// TLS streams indexed by party_id (0, 1, 2)
    streams: [TlsStream; NUM_PARTIES],
}

impl VsockTlsCoordinator {
    /// Accept 3 worker connections over vsock+TLS.
    ///
    /// After TLS handshake, sends the attestation document (if any) as a
    /// length-prefixed first message, then reads the worker's party_id.
    pub fn accept(
        vsock_port: u32,
        identity: &EphemeralIdentity,
        attestation_doc: Option<&[u8]>,
    ) -> eyre::Result<Self> {
        rustls::crypto::ring::default_provider()
            .install_default()
            .ok();

        let server_config = Arc::new(
            rustls::ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(
                    vec![identity.cert_der.clone()],
                    identity.key_der.clone_key(),
                )
                .context("building rustls ServerConfig")?,
        );

        let listener = VsockListener::bind_with_cid_port(VMADDR_CID_ANY, vsock_port)
            .context("binding vsock listener")?;

        let mut indexed_streams: Vec<(usize, TlsStream)> = Vec::with_capacity(NUM_PARTIES);

        for i in 0..NUM_PARTIES {
            let (vsock_stream, _addr) = listener
                .accept()
                .with_context(|| format!("accepting vsock connection {i}"))?;

            let tls_conn = rustls::ServerConnection::new(Arc::clone(&server_config))
                .context("creating TLS server connection")?;
            let mut tls_stream = rustls::StreamOwned::new(tls_conn, vsock_stream);

            // Send attestation doc as length-prefixed first message
            match attestation_doc {
                Some(doc) => {
                    tls_stream.write_all(&(doc.len() as u32).to_be_bytes())?;
                    tls_stream.write_all(doc)?;
                }
                None => {
                    tls_stream.write_all(&0u32.to_be_bytes())?;
                }
            }

            // Read party_id and worker_id from worker
            let mut id_buf = [0u8; 4];
            tls_stream.read_exact(&mut id_buf)?;
            let party_id = u32::from_be_bytes(id_buf) as usize;
            let mut _worker_id_buf = [0u8; 4];
            tls_stream.read_exact(&mut _worker_id_buf)?;

            if party_id >= NUM_PARTIES {
                return Err(eyre!("invalid party_id {party_id} from connection {i}"));
            }

            indexed_streams.push((party_id, tls_stream));
        }

        // Sort by party_id and convert to fixed array
        indexed_streams.sort_by_key(|(id, _)| *id);

        // Verify we got exactly parties 0, 1, 2
        for (i, (id, _)) in indexed_streams.iter().enumerate() {
            if *id != i {
                return Err(eyre!(
                    "expected party_id {i} at index {i}, got {id} (duplicate or missing party)"
                ));
            }
        }

        let [s0, s1, s2] = <[(usize, TlsStream); 3]>::try_from(indexed_streams)
            .map_err(|v| eyre!("expected 3 streams, got {}", v.len()))?;

        Ok(Self {
            streams: [s0.1, s1.1, s2.1],
        })
    }
}

// -- Serialization helpers (matching mpc-core test_utils pattern) --

fn serialize_uncompressed<T: CanonicalSerialize>(data: &T) -> eyre::Result<Vec<u8>> {
    let size = data.uncompressed_size();
    let mut buf = Vec::with_capacity(size);
    data.serialize_uncompressed(&mut buf)
        .context("serialize_uncompressed")?;
    Ok(buf)
}

fn deserialize_uncompressed<T: CanonicalDeserialize>(bytes: &[u8]) -> eyre::Result<T> {
    T::deserialize_uncompressed_unchecked(bytes).context("deserialize_uncompressed")
}

fn send_to_stream(stream: &mut TlsStream, bytes: &[u8]) -> eyre::Result<()> {
    let len = (bytes.len() as u32).to_be_bytes();
    stream.write_all(&len)?;
    stream.write_all(bytes)?;
    stream.flush()?;
    Ok(())
}

fn recv_from_stream<T: CanonicalDeserialize>(stream: &mut TlsStream) -> eyre::Result<T> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    deserialize_uncompressed(&buf)
}

impl MpcStarNetCoordinator for VsockTlsCoordinator {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> eyre::Result<Vec<T>> {
        let mut responses = Vec::with_capacity(NUM_PARTIES);
        for stream in self.streams.iter_mut() {
            responses.push(recv_from_stream(stream)?);
        }
        Ok(responses)
    }

    fn receive_responses_from_subnets<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> eyre::Result<Vec<Vec<T>>> {
        Ok(vec![self.receive_responses()?])
    }

    fn receive_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
    ) -> eyre::Result<T> {
        if worker_id != 0 {
            return Err(eyre!("vsock transport only supports worker_id=0"));
        }
        let idx = usize::from(party_id);
        recv_from_stream(&mut self.streams[idx])
    }

    fn receive_response_from_workers<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
    ) -> eyre::Result<Vec<T>> {
        Ok(vec![self.receive_response(party_id, 0)?])
    }

    fn broadcast_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: T,
    ) -> eyre::Result<()> {
        let bytes = serialize_uncompressed(&data)?;
        for stream in self.streams.iter_mut() {
            send_to_stream(stream, &bytes)?;
        }
        Ok(())
    }

    fn send_requests<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> eyre::Result<()> {
        if data.len() != NUM_PARTIES {
            return Err(eyre!(
                "send_requests expects {} items, got {}",
                NUM_PARTIES,
                data.len()
            ));
        }
        for (item, stream) in data.iter().zip(self.streams.iter_mut()) {
            let bytes = serialize_uncompressed(item)?;
            send_to_stream(stream, &bytes)?;
        }
        Ok(())
    }

    fn send_request_to_workers<T: CanonicalSerialize + CanonicalDeserialize + Clone>(
        &mut self,
        party_id: PartyID,
        data: T,
    ) -> eyre::Result<()> {
        self.send_request(party_id, 0, data)
    }

    fn send_requests_to_workers<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> eyre::Result<()> {
        self.send_requests(data)
    }

    fn send_requests_blocking<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> eyre::Result<()> {
        self.send_requests(data)
    }

    fn send_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
        data: T,
    ) -> eyre::Result<()> {
        if worker_id != 0 {
            return Err(eyre!("vsock transport only supports worker_id=0"));
        }
        let idx = usize::from(party_id);
        let bytes = serialize_uncompressed(&data)?;
        send_to_stream(&mut self.streams[idx], &bytes)
    }

    fn log_num_workers(&self) -> usize {
        0
    }

    fn active_num_workers(&self) -> usize {
        1
    }

    fn total_bandwidth_used(&self) -> (u64, u64) {
        (0, 0)
    }

    fn log_connection_stats(&self, _label: Option<&str>) {}

    fn reset_stats(&mut self) {}

    fn fork(&mut self) -> eyre::Result<Self> {
        Err(eyre!("VsockTlsCoordinator does not support fork"))
    }

    fn set_num_workers(&mut self, _num_workers: usize) {}

    fn reset_num_workers(&mut self) {}
}

impl Rep3NetworkCoordinator for VsockTlsCoordinator {
    fn sync_with_parties(&mut self) -> eyre::Result<()> {
        self.broadcast_request(true)?;
        let _ = self.receive_responses::<bool>()?;
        Ok(())
    }
}
