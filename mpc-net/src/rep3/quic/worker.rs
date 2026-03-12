use crate::{
    channel::{BulkBytesChannelHandle, BytesChannel, Channel},
    codecs::BincodeCodec,
    rep3::{PartyID, PartyWorkerID},
    MpcNetworkHandlerShutdown, DEFAULT_CONNECT_TIMEOUT,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::{Bytes, BytesMut};
use bytesize::ByteSize;
use color_eyre::eyre::{self, Report};
use color_eyre::eyre::{bail, Context};
use eyre::ensure;
use once_cell::sync::Lazy;
use quinn::{
    crypto::rustls::QuicClientConfig,
    rustls::{pki_types::CertificateDer, RootCertStore},
};
use quinn::{ClientConfig, Connection, Endpoint, IdleTimeout, RecvStream, SendStream, TransportConfig, VarInt};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};
use std::{
    collections::HashMap,
    io,
    net::{SocketAddr, ToSocketAddrs},
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    runtime::Runtime,
};
use tokio_util::codec::{Decoder, Encoder, LengthDelimitedCodec};

use crate::{
    channel::ChannelHandle, config::NetworkConfig, topology::MpcStarNetWorker, MpcNetworkHandlerWrapper, Result,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuicForkTopology {
    ConnectionPool,
    StreamPool,
}

impl QuicForkTopology {
    fn from_env() -> Self {
        match std::env::var("MPC_QUIC_TOPOLOGY").ok().as_deref().map(str::trim) {
            Some("stream-pool") => Self::StreamPool,
            Some("conn-pool") | None => Self::ConnectionPool,
            Some(other) => {
                tracing::warn!(topology = other, "unknown MPC_QUIC_TOPOLOGY, using conn-pool");
                Self::ConnectionPool
            }
        }
    }

    fn physical_connection_count(self, configured_lanes: usize) -> usize {
        match self {
            Self::ConnectionPool => configured_lanes.max(1),
            Self::StreamPool => 1,
        }
    }
}

fn configured_transport_lanes() -> usize {
    std::env::var("MPC_QUIC_CONN_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| std::env::var("NETWORK_FORKS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0))
        .unwrap_or(8)
}

fn parse_quic_limit_mb(var: &str, default_mb: usize) -> u32 {
    let bytes = std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_mb)
        .saturating_mul(1024 * 1024)
        .min(u32::MAX as usize);
    u32::try_from(bytes).expect("bounded to u32::MAX")
}

pub(crate) fn quic_conn_rx_window_bytes() -> u32 {
    parse_quic_limit_mb("MPC_QUIC_CONN_RX_WINDOW_MB", 256)
}

pub(crate) fn quic_stream_rx_window_bytes() -> u32 {
    parse_quic_limit_mb("MPC_QUIC_STREAM_RX_WINDOW_MB", 64)
}

pub(crate) fn quic_max_bidi_streams() -> u32 {
    std::env::var("MPC_QUIC_MAX_BIDI_STREAMS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(256)
}

pub(crate) fn quic_transport_config() -> Arc<TransportConfig> {
    let mut transport_config = TransportConfig::default();
    transport_config.receive_window(VarInt::from(quic_conn_rx_window_bytes()));
    transport_config.stream_receive_window(VarInt::from(quic_stream_rx_window_bytes()));
    transport_config.max_concurrent_bidi_streams(VarInt::from(quic_max_bidi_streams()));
    transport_config.max_idle_timeout(Some(IdleTimeout::try_from(Duration::from_secs(180)).unwrap()));
    transport_config.keep_alive_interval(Some(Duration::from_secs(1)));
    Arc::new(transport_config)
}

pub static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        // .worker_threads(8)
        .enable_all()
        .build()
        .unwrap()
});

#[derive(Clone)]
pub struct Rep3QuicMpcNetWorker {
    pub id: PartyWorkerID,
    pub chan_next: ChannelHandle<Bytes, BytesMut>,
    pub chan_prev: ChannelHandle<Bytes, BytesMut>,
    pub chan_next_bulk: Option<BulkBytesChannelHandle>,
    pub chan_prev_bulk: Option<BulkBytesChannelHandle>,
    pub chan_coordinator: Option<ChannelHandle<Bytes, BytesMut>>,
    pub log_num_workers_per_party: usize,
    pub current_log_num_workers: usize,
    pub net_handler: Arc<MpcNetworkHandlerWrapper>,
    pub config: NetworkConfig,

    pub fork_id: u32,
    pub seq: Arc<AtomicU64>,
    pub alloc: Arc<ForkAlloc>,
    pub transport_lanes: usize,
}

fn fork_bulk_channels() -> bool {
    std::env::var("MPC_FORK_BULK_CHANNELS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1) != 0
}

impl Rep3QuicMpcNetWorker {
    pub fn new(config: NetworkConfig, log_num_workers_per_party: usize) -> Result<Self> {
        ensure!(config.parties.len() == 3, "REP3 protocol requires exactly 3 parties");

        let alloc = Arc::new(ForkAlloc::new());
        let fork_id = alloc.alloc();
        let seq = Arc::new(AtomicU64::new(0));
        let id = PartyWorkerID::new(config.my_id, config.worker);

        let (net_handler, chan_next, chan_prev, chan_next_bulk, chan_prev_bulk, mut chan_coordinator, transport_lanes) =
            RUNTIME.block_on(async {
                let net_handler = MpcNetworkHandlerWorker::establish(config.clone()).await?;
                let chan_coordinator =
                    net_handler.get_coordinator_byte_channel().await?.map(ChannelHandle::manage_bytes_quic);

                let mut channels = net_handler.get_byte_channels_for_lane(0).await?;
                let mut bulk_channels = net_handler.get_byte_channels_for_lane(0).await?;
                let chan_next =
                    channels.remove(&id.party_id().next_id().into()).ok_or(eyre::eyre!("no next channel found"))?;
                let chan_prev =
                    channels.remove(&id.party_id().prev_id().into()).ok_or(eyre::eyre!("no prev channel found"))?;
                let chan_next_bulk = bulk_channels
                    .remove(&id.party_id().next_id().into())
                    .ok_or(eyre::eyre!("no next bulk channel found"))?;
                let chan_prev_bulk = bulk_channels
                    .remove(&id.party_id().prev_id().into())
                    .ok_or(eyre::eyre!("no prev bulk channel found"))?;
                if !channels.is_empty() {
                    bail!("unexpected channels found")
                }
                if !bulk_channels.is_empty() {
                    bail!("unexpected bulk channels found")
                }
                let chan_next = ChannelHandle::manage_bytes_quic(chan_next);
                let chan_prev = ChannelHandle::manage_bytes_quic(chan_prev);
                let chan_next_bulk = BulkBytesChannelHandle::manage_quic(chan_next_bulk);
                let chan_prev_bulk = BulkBytesChannelHandle::manage_quic(chan_prev_bulk);

                let transport_lanes = net_handler.transport_lanes;
                eyre::Ok((
                    net_handler,
                    chan_next,
                    chan_prev,
                    chan_next_bulk,
                    chan_prev_bulk,
                    chan_coordinator,
                    transport_lanes,
                ))
            })?;

        // If coordinator uses TLS protocol, connect via TlsCoordinatorClient
        #[cfg(feature = "tls")]
        if chan_coordinator.is_none() {
            if let Some(ref coord) = config.coordinator {
                if coord.protocol == crate::config::CoordinatorProtocol::Tls {
                    tracing::info!("connecting to coordinator via TLS");
                    let tls_client = crate::rep3::tls::coordinator::TlsCoordinatorClient::connect(
                        &coord.dns_name,
                        config.my_id,
                        config.worker,
                    )?;
                    chan_coordinator = Some(ChannelHandle::manage_tls_coordinator(tls_client));
                }
            }
        }

        Ok(Self {
            id,
            net_handler: Arc::new(MpcNetworkHandlerWrapper::new(RUNTIME.handle().clone(), net_handler)),
            chan_next,
            chan_prev,
            chan_next_bulk: Some(chan_next_bulk),
            chan_prev_bulk: Some(chan_prev_bulk),
            chan_coordinator,
            log_num_workers_per_party,
            current_log_num_workers: log_num_workers_per_party,
            config,
            alloc,
            fork_id,
            seq,
            transport_lanes,
        })
    }

    /// Sends bytes over the network to the target party.
    pub fn send_bytes(&mut self, target: PartyID, data: Bytes) -> std::io::Result<()> {
        if target == self.id.party_id().next_id() {
            std::mem::drop(self.chan_next.blocking_send(data));
            Ok(())
        } else if target == self.id.party_id().prev_id() {
            std::mem::drop(self.chan_prev.blocking_send(data));
            Ok(())
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Cannot send to self"));
        }
    }

    pub fn send_bytes_bulk(&mut self, target: PartyID, data: Bytes) -> std::io::Result<()> {
        let chan = if target == self.id.party_id().next_id() {
            self.chan_next_bulk.as_ref()
        } else if target == self.id.party_id().prev_id() {
            self.chan_prev_bulk.as_ref()
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Cannot send to self"));
        };
        let chan = chan.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "bulk channel not allocated (set MPC_FORK_BULK_CHANNELS=1)",
            )
        })?;
        self.net_handler.runtime.block_on(chan.send(data))
    }

    pub async fn send_bytes_async(&mut self, target: PartyID, data: Bytes) -> std::io::Result<()> {
        if target == self.id.party_id().next_id() {
            std::mem::drop(self.chan_next.send(data).await);
            Ok(())
        } else if target == self.id.party_id().prev_id() {
            std::mem::drop(self.chan_prev.send(data).await);
            Ok(())
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Cannot send to self"));
        }
    }

    /// Receives bytes over the network from the party with the given id.
    pub fn recv_bytes(&mut self, from: PartyID) -> std::io::Result<BytesMut> {
        let data = if from == self.id.party_id().prev_id() {
            self.chan_prev.blocking_recv().blocking_recv()
        } else if from == self.id.party_id().next_id() {
            self.chan_next.blocking_recv().blocking_recv()
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Cannot recv from self"));
        };
        let data =
            data.map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "receive channel end died"))??;
        Ok(data)
    }

    fn bulk_chan(&self, party: PartyID) -> std::io::Result<&BulkBytesChannelHandle> {
        let chan = if party == self.id.party_id().prev_id() {
            self.chan_prev_bulk.as_ref()
        } else if party == self.id.party_id().next_id() {
            self.chan_next_bulk.as_ref()
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Cannot use bulk channel with self"));
        };
        chan.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "bulk channel not allocated (set MPC_FORK_BULK_CHANNELS=1)",
            )
        })
    }

    pub fn recv_bytes_bulk(&mut self, from: PartyID) -> std::io::Result<Vec<u8>> {
        let chan = self.bulk_chan(from)?;
        self.net_handler.runtime.block_on(chan.recv_bytes())
    }

    pub fn recv_bytes_bulk_into(&mut self, from: PartyID, dst: &mut [u8]) -> std::io::Result<()> {
        let chan = self.bulk_chan(from)?;
        self.net_handler.runtime.block_on(chan.recv_into(dst))
    }

    pub async fn recv_bytes_async(&mut self, from: PartyID) -> std::io::Result<BytesMut> {
        let data = if from == self.id.party_id().prev_id() {
            self.chan_prev.recv().await.await
        } else if from == self.id.party_id().next_id() {
            self.chan_next.recv().await.await
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Cannot recv from self"));
        };
        let data =
            data.map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "receive channel end died"))??;
        Ok(data)
    }

    /// Print the IO stats of **fork** subnetwork. Don't use if forks create new connections.
    pub fn log_connection_stats(&self) {
        // hack: wait arbitrary time for all send/recv tasks till now to complete
        std::thread::sleep(std::time::Duration::from_secs(1));
        self.net_handler.runtime.block_on(async { self.net_handler.inner.log_connection_stats() })
    }
}

#[derive(Debug)]
pub struct ForkAlloc {
    next: AtomicU32,
}
impl ForkAlloc {
    pub fn new() -> Self {
        Self { next: AtomicU32::new(0) }
    }
    #[inline]
    pub fn alloc(&self) -> u32 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        id
    }
}

impl MpcStarNetWorker for Rep3QuicMpcNetWorker {
    fn send_response<T: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: T) -> Result<()> {
        let size = data.uncompressed_size();
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
            .context("while serializing data")?;

        std::mem::drop(self.chan_coordinator.as_ref().unwrap().blocking_send(Bytes::from(ser_data)));
        Ok(())
    }

    fn receive_request<T: CanonicalSerialize + CanonicalDeserialize>(&mut self) -> Result<T> {
        let response = self
            .chan_coordinator
            .as_ref()
            .unwrap()
            .blocking_recv()
            .blocking_recv()
            .context("while receiving request")??;

        let ret = T::deserialize_uncompressed_unchecked(&response[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
            .context("while deserializing request")?;
        Ok(ret)
    }

    fn log_num_workers(&self) -> usize {
        self.current_log_num_workers
    }

    fn io_stats_total(&self) -> (u64, u64) {
        let sent_bytes = self
            .net_handler
            .inner
            .parties_connections
            .iter()
            .map(|(_, conns)| conns.iter().map(|conn| conn.stats().udp_tx.bytes as u64).sum::<u64>())
            .sum();
        let recv_bytes = self
            .net_handler
            .inner
            .parties_connections
            .iter()
            .map(|(_, conns)| conns.iter().map(|conn| conn.stats().udp_rx.bytes as u64).sum::<u64>())
            .sum();
        (sent_bytes, recv_bytes)
    }

    fn io_stats_per_party(&self) -> BTreeMap<usize, (u64, u64)> {
        self.net_handler
            .inner
            .parties_connections
            .iter()
            .map(|(id, conns)| {
                let (sent, recv) = conns.iter().fold((0u64, 0u64), |(sent, recv), conn| {
                    let stats = conn.stats();
                    (sent + stats.udp_tx.bytes, recv + stats.udp_rx.bytes)
                });
                (*id, (sent, recv))
            })
            .collect()
    }

    fn party_id(&self) -> PartyID {
        self.id.party_id()
    }

    fn fork(&self) -> Self {
        let fork_id = self.alloc.alloc();
        let id = self.id.clone();
        let lane_idx = (fork_id as usize) % self.transport_lanes.max(1);
        let want_bulk = fork_bulk_channels();

        let (chan_next, chan_prev, chan_next_bulk, chan_prev_bulk) = RUNTIME
            .block_on(async {
                let mut channels = self.net_handler.inner.get_byte_channels_for_lane(lane_idx).await?;
                let chan_next =
                    channels.remove(&id.party_id().next_id().into()).ok_or(eyre::eyre!("no next channel found"))?;
                let chan_prev =
                    channels.remove(&id.party_id().prev_id().into()).ok_or(eyre::eyre!("no prev channel found"))?;
                if !channels.is_empty() {
                    bail!("unexpected channels found")
                }
                let chan_next = ChannelHandle::manage_bytes_quic(chan_next);
                let chan_prev = ChannelHandle::manage_bytes_quic(chan_prev);

                let (chan_next_bulk, chan_prev_bulk) = if want_bulk {
                    let mut bulk_channels = self.net_handler.inner.get_byte_channels_for_lane(lane_idx).await?;
                    let chan_next_bulk = bulk_channels
                        .remove(&id.party_id().next_id().into())
                        .ok_or(eyre::eyre!("no next bulk channel found"))?;
                    let chan_prev_bulk = bulk_channels
                        .remove(&id.party_id().prev_id().into())
                        .ok_or(eyre::eyre!("no prev bulk channel found"))?;
                    if !bulk_channels.is_empty() {
                        bail!("unexpected bulk channels found")
                    }
                    (
                        Some(BulkBytesChannelHandle::manage_quic(chan_next_bulk)),
                        Some(BulkBytesChannelHandle::manage_quic(chan_prev_bulk)),
                    )
                } else {
                    (None, None)
                };

                eyre::Ok((chan_next, chan_prev, chan_next_bulk, chan_prev_bulk))
            })
            .unwrap();

        Self {
            id,
            net_handler: Arc::clone(&self.net_handler),
            chan_next,
            chan_prev,
            chan_next_bulk,
            chan_prev_bulk,
            chan_coordinator: None,
            log_num_workers_per_party: self.log_num_workers_per_party,
            current_log_num_workers: self.current_log_num_workers,
            config: self.config.clone(),
            alloc: self.alloc.clone(),
            fork_id,
            seq: Arc::new(AtomicU64::new(0)),
            transport_lanes: self.transport_lanes,
        }
    }

    fn fork_with_coordinator(&mut self) -> Result<Self> {
        let id = self.id.clone();
        let net_handler = Arc::clone(&self.net_handler);
        let chan_coordinator = net_handler.runtime.block_on(async {
            let chan_coordinator =
                net_handler.inner.get_coordinator_byte_channel().await?.map(ChannelHandle::manage_bytes_quic);

            Ok::<_, Report>(chan_coordinator)
        })?;

        Ok(Self {
            id,
            net_handler,
            chan_next: self.chan_next.clone(),
            chan_prev: self.chan_prev.clone(),
            chan_next_bulk: self.chan_next_bulk.clone(), // shares parent's bulk channels (if any)
            chan_prev_bulk: self.chan_prev_bulk.clone(),
            chan_coordinator,
            log_num_workers_per_party: self.log_num_workers_per_party,
            current_log_num_workers: self.current_log_num_workers,
            config: self.config.clone(),
            fork_id: self.alloc.alloc(),
            seq: Arc::new(AtomicU64::new(0)),
            alloc: self.alloc.clone(),
            transport_lanes: self.transport_lanes,
        })
    }

    // #[tracing::instrument(skip_all, name = "MpcStarNetWorker::get_worker_subnets")]
    fn get_worker_subnets(&self, num_workers: usize) -> Result<Vec<Self>> {
        let config = self.config.clone();
        let log_num_workers_per_party = self.log_num_workers_per_party;
        (1..num_workers)
            .map(|worker_id| Self::new(config.for_worker(worker_id), log_num_workers_per_party))
            .collect::<Result<Vec<_>>>()
    }

    fn worker_idx(&self) -> usize {
        self.id.1
    }

    fn set_log_num_workers(&mut self, log_num_workers: usize) {
        self.current_log_num_workers = log_num_workers;
    }

    fn reset_log_num_workers(&mut self) {
        self.current_log_num_workers = self.log_num_workers_per_party;
    }
}

pub fn codec_cfg() -> tokio_util::codec::LengthDelimitedCodec {
    pub const LEN_BYTES: usize = 5;
    pub const MAX_FRAME: usize = 1 << 30; // 1 GiB (pick a real bound)

    tokio_util::codec::LengthDelimitedCodec::builder()
        .length_field_type::<u64>()
        .length_field_length(LEN_BYTES)
        .max_frame_length(MAX_FRAME)
        .new_codec()
}

/// A network handler for MPC protocols.
#[derive(Debug)]
pub struct MpcNetworkHandlerWorker {
    // this is a btreemap because we rely on iteration order
    parties_connections: BTreeMap<usize, Vec<Connection>>,
    coordinator_connection: Option<Connection>,
    endpoints: Vec<Endpoint>,
    my_id: usize,
    worker: usize,
    topology: QuicForkTopology,
    transport_lanes: usize,
}

impl MpcNetworkHandlerWorker {
    /// Tries to establish a connection to other parties in the network based on the provided [NetworkConfig].
    pub async fn establish(config: NetworkConfig) -> Result<Self, Report> {
        config.check_config()?;
        let certs: HashMap<usize, CertificateDer> = config.parties.iter().map(|p| (p.id, p.cert.clone())).collect();

        let mut root_store = RootCertStore::empty();
        for (id, cert) in &certs {
            root_store.add(cert.clone()).with_context(|| format!("adding certificate for party {id} to root store"))?;
        }
        if let Some(coordinator) = &config.coordinator {
            root_store
                .add(coordinator.cert.clone())
                .with_context(|| format!("adding certificate for coordinator to root store"))?;
        }

        let crypto = quinn::rustls::ClientConfig::builder().with_root_certificates(root_store).with_no_client_auth();

        let transport_config = quic_transport_config();
        let client_config = {
            let mut client_config = ClientConfig::new(Arc::new(QuicClientConfig::try_from(crypto)?));
            client_config.transport_config(Arc::clone(&transport_config));
            client_config
        };

        let mut server_config = quinn::ServerConfig::with_single_cert(vec![certs[&config.my_id].clone()], config.key)
            .context("creating our server config")?;
        server_config.transport_config(transport_config);
        let our_socket_addr = config.bind_addr;

        let mut endpoints = Vec::new();
        let server_endpoint = {
            // Retry binding if the port is still lingering from a previous run.
            let mut last_err = None;
            let mut ep = None;
            for attempt in 0..10 {
                match quinn::Endpoint::server(server_config.clone(), our_socket_addr) {
                    Ok(e) => {
                        ep = Some(e);
                        break;
                    }
                    Err(e) => {
                        if attempt < 9 {
                            tracing::warn!(
                                attempt,
                                addr = %our_socket_addr,
                                "server bind failed, retrying: {e}"
                            );
                            std::thread::sleep(std::time::Duration::from_secs(1));
                        }
                        last_err = Some(e);
                    }
                }
            }
            ep.ok_or_else(|| last_err.unwrap())?
        };

        let coordinator_connection = if let Some(coordinator) = config.coordinator {
            match coordinator.protocol {
                crate::config::CoordinatorProtocol::Quic => {
                    tracing::trace!("my id: {:?}, connecting to coordinator via QUIC", config.my_id);

                    let addresses: Vec<SocketAddr> = coordinator
                        .dns_name
                        .to_socket_addrs()
                        .with_context(|| format!("while resolving DNS name for {}", coordinator.dns_name))?
                        .collect();
                    if addresses.is_empty() {
                        return Err(eyre::eyre!("could not resolve DNS name {}", coordinator.dns_name));
                    }
                    let party_addr = addresses[0];
                    let local_client_socket: SocketAddr = match party_addr {
                        SocketAddr::V4(_) => "0.0.0.0:0".parse().expect("hardcoded IP address is valid"),
                        SocketAddr::V6(_) => "[::]:0".parse().expect("hardcoded IP address is valid"),
                    };
                    let endpoint = quinn::Endpoint::client(local_client_socket)
                        .with_context(|| format!("creating client endpoint to coordinator"))?;
                    let conn = endpoint
                        .connect_with(client_config.clone(), party_addr, &coordinator.dns_name.hostname)
                        .with_context(|| format!("setting up client connection with coordinator"))?
                        .await
                        .with_context(|| format!("connecting as a client to coordinator"))?;
                    let mut uni = conn.open_uni().await?;
                    uni.write_u32(u32::try_from(config.my_id).expect("party id fits into u32")).await?;
                    uni.write_u32(config.worker as u32).await?;
                    uni.flush().await?;
                    uni.finish()?;

                    tracing::trace!(
                        "coordinator conn with id {} from {} to {}",
                        conn.stable_id(),
                        endpoint.local_addr().unwrap(),
                        conn.remote_address(),
                    );
                    endpoints.push(endpoint);
                    Some(conn)
                }
                crate::config::CoordinatorProtocol::Tls => {
                    // TLS coordinator connection is handled in Rep3QuicMpcNetWorker::new()
                    // via ChannelHandle::manage_tls_coordinator.
                    None
                }
            }
        } else {
            None
        };

        let topology = QuicForkTopology::from_env();
        let transport_lanes = configured_transport_lanes();
        let physical_connections = topology.physical_connection_count(transport_lanes);
        let mut parties_connections_slots: BTreeMap<usize, Vec<Option<Connection>>> = config
            .parties
            .iter()
            .filter(|party| party.id != config.my_id)
            .map(|party| (party.id, (0..physical_connections).map(|_| None).collect::<Vec<_>>()))
            .collect();

        for party in config.parties.iter().filter(|party| party.id < config.my_id) {
            tracing::trace!(
                "my id: {:?}, connecting to party: {:?} with {} lanes",
                config.my_id,
                party.id,
                physical_connections
            );

            let party_addresses: Vec<SocketAddr> = party
                .dns_name
                .to_socket_addrs()
                .with_context(|| format!("while resolving DNS name for {}", party.dns_name))?
                .collect();
            if party_addresses.is_empty() {
                return Err(eyre::eyre!("could not resolve DNS name {}", party.dns_name));
            }
            let party_addr = party_addresses[0];

            for lane_idx in 0..physical_connections {
                let local_client_socket: SocketAddr = match party_addr {
                    SocketAddr::V4(_) => "0.0.0.0:0".parse().expect("hardcoded IP address is valid"),
                    SocketAddr::V6(_) => "[::]:0".parse().expect("hardcoded IP address is valid"),
                };
                let endpoint = quinn::Endpoint::client(local_client_socket)
                    .with_context(|| format!("creating client endpoint to party {} lane {}", party.id, lane_idx))?;
                let conn = endpoint
                    .connect_with(client_config.clone(), party_addr, &party.dns_name.hostname)
                    .with_context(|| format!("setting up client connection with party {} lane {}", party.id, lane_idx))?
                    .await
                    .with_context(|| format!("connecting as a client to party {} lane {}", party.id, lane_idx))?;
                let mut uni = conn.open_uni().await?;
                uni.write_u32(u32::try_from(config.my_id).expect("party id fits into u32")).await?;
                uni.write_u32(lane_idx as u32).await?;
                uni.flush().await?;
                uni.finish()?;
                tracing::trace!(
                    lane = lane_idx,
                    "Conn with id {} from {} to {}",
                    conn.stable_id(),
                    endpoint.local_addr().unwrap(),
                    conn.remote_address(),
                );
                let slots = parties_connections_slots.get_mut(&party.id).expect("lane slots exist");
                assert!(slots[lane_idx].replace(conn).is_none());
                endpoints.push(endpoint);
            }
        }

        let expected_incoming =
            config.parties.iter().filter(|party| party.id > config.my_id).count() * physical_connections;
        for _ in 0..expected_incoming {
            match tokio::time::timeout(config.timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT), server_endpoint.accept())
                .await
            {
                Ok(Some(maybe_conn)) => {
                    let conn = maybe_conn.await?;
                    tracing::trace!(
                        "Conn with id {} from {} to {}",
                        conn.stable_id(),
                        server_endpoint.local_addr().unwrap(),
                        conn.remote_address(),
                    );
                    let mut uni = conn.accept_uni().await?;
                    let other_party_id = usize::try_from(uni.read_u32().await?).expect("u32 fits into usize");
                    let lane_idx = usize::try_from(uni.read_u32().await?).expect("u32 fits into usize");
                    ensure!(
                        lane_idx < physical_connections,
                        "peer lane {lane_idx} out of range for party {other_party_id}"
                    );
                    let slots = parties_connections_slots
                        .get_mut(&other_party_id)
                        .ok_or_else(|| eyre::eyre!("unexpected connection from party {other_party_id}"))?;
                    ensure!(
                        slots[lane_idx].is_none(),
                        "duplicate connection for party {other_party_id} lane {lane_idx}"
                    );
                    slots[lane_idx] = Some(conn);
                }
                Ok(None) => {
                    return Err(eyre::eyre!("server endpoint did not accept an expected peer connection",));
                }
                Err(_) => {
                    return Err(eyre::eyre!("a party did not connect within 60 seconds - timeout",));
                }
            }
        }

        let parties_connections = parties_connections_slots
            .into_iter()
            .map(|(party_id, slots)| {
                let conns = slots
                    .into_iter()
                    .enumerate()
                    .map(|(lane_idx, conn)| {
                        conn.ok_or_else(|| eyre::eyre!("missing connection for party {party_id} lane {lane_idx}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok::<_, Report>((party_id, conns))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        endpoints.push(server_endpoint);

        Ok(MpcNetworkHandlerWorker {
            parties_connections,
            coordinator_connection,
            endpoints,
            my_id: config.my_id,
            worker: config.worker,
            topology,
            transport_lanes,
        })
    }

    /// Returns the number of sent and received bytes.
    pub fn get_send_receive(&self, i: usize) -> std::io::Result<(u64, u64)> {
        let conns = self
            .parties_connections
            .get(&i)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such connection"))?;
        Ok(conns.iter().fold((0u64, 0u64), |(sent, recv), conn| {
            let stats = conn.stats();
            (sent + stats.udp_tx.bytes, recv + stats.udp_rx.bytes)
        }))
    }

    /// Prints the IO statistics for connections in fork. Don't use if forks create new connections.
    pub fn log_connection_stats(&self) {
        for (i, conns) in &self.parties_connections {
            let (sent, recv) = conns.iter().fold((0u64, 0u64), |(sent, recv), conn| {
                let stats = conn.stats();
                (sent + stats.udp_tx.bytes, recv + stats.udp_rx.bytes)
            });
            tracing::info!(
                "IO: P{}->P{} | SENT: {} bytes | RECV: {} bytes",
                self.my_id,
                i,
                ByteSize(sent),
                ByteSize(recv)
            );
        }

        if let Some(conn) = self.coordinator_connection.as_ref() {
            let stats = conn.stats();
            tracing::info!(
                "IO: P{}->C | SENT: {} bytes | RECV: {} bytes",
                self.my_id,
                ByteSize(stats.udp_tx.bytes),
                ByteSize(stats.udp_rx.bytes)
            );
        }
    }

    /// Sets up a new [BytesChannel] between each party. The resulting map maps the id of the party to its respective [BytesChannel].
    pub async fn get_byte_channels(&self) -> std::io::Result<HashMap<usize, BytesChannel<RecvStream, SendStream>>> {
        self.get_byte_channels_for_lane(0).await
    }

    pub async fn get_byte_channels_for_lane(
        &self,
        lane_idx: usize,
    ) -> std::io::Result<HashMap<usize, BytesChannel<RecvStream, SendStream>>> {
        // set max frame length to 1Tb and length_field_length to 5 bytes
        const NUM_BYTES: usize = 5;
        let codec = LengthDelimitedCodec::builder()
            .length_field_type::<u64>() // u64 because this is the type the length is decoded into, and u32 doesnt fit 5 bytes
            .length_field_length(NUM_BYTES)
            .max_frame_length(1usize << (NUM_BYTES * 8))
            .new_codec();
        self.get_custom_channels_for_lane(codec, lane_idx).await
    }

    /// Set up a new [Channel] using [BincodeCodec] between each party. The resulting map maps the id of the party to its respective [Channel].
    pub async fn get_serde_bincode_channels<M: Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> std::io::Result<HashMap<usize, Channel<RecvStream, SendStream, BincodeCodec<M>>>> {
        let bincodec = BincodeCodec::<M>::new();
        self.get_custom_channels_for_lane(bincodec, 0).await
    }

    fn select_connection<'a>(&self, conns: &'a [Connection], lane_idx: usize) -> &'a Connection {
        match self.topology {
            QuicForkTopology::ConnectionPool => &conns[lane_idx % conns.len().max(1)],
            QuicForkTopology::StreamPool => &conns[0],
        }
    }

    /// Set up a new [Channel] using the provided codec between each party. The resulting map maps the id of the party to its respective [Channel].
    pub async fn get_custom_channels<
        MSend,
        MRecv,
        C: Encoder<MSend, Error = io::Error> + Decoder<Item = MRecv, Error = io::Error> + 'static + Clone,
    >(
        &self,
        codec: C,
    ) -> std::io::Result<HashMap<usize, Channel<RecvStream, SendStream, C>>> {
        self.get_custom_channels_for_lane(codec, 0).await
    }

    pub async fn get_custom_channels_for_lane<
        MSend,
        MRecv,
        C: Encoder<MSend, Error = io::Error> + Decoder<Item = MRecv, Error = io::Error> + 'static + Clone,
    >(
        &self,
        codec: C,
        lane_idx: usize,
    ) -> std::io::Result<HashMap<usize, Channel<RecvStream, SendStream, C>>> {
        let mut channels = HashMap::with_capacity(self.parties_connections.len());
        for (&id, conns) in self.parties_connections.iter() {
            let conn = self.select_connection(conns, lane_idx);
            if id < self.my_id {
                // we are the client, so we are the receiver
                let (mut send_stream, mut recv_stream) = conn.open_bi().await?;
                send_stream.write_u32(self.my_id as u32).await?;
                let their_id = recv_stream.read_u32().await?;
                assert!(their_id == id as u32);
                let conn = Channel::new(recv_stream, send_stream, codec.clone());
                assert!(channels.insert(id, conn).is_none());
            } else {
                // we are the server, so we are the sender
                let (mut send_stream, mut recv_stream) = conn.accept_bi().await?;
                let their_id = recv_stream.read_u32().await?;
                assert!(their_id == id as u32);
                send_stream.write_u32(self.my_id as u32).await?;
                let conn = Channel::new(recv_stream, send_stream, codec.clone());
                assert!(channels.insert(id, conn).is_none());
            }
        }
        Ok(channels)
    }

    /// Sets up a new [BytesChannel] between each party. The resulting map maps the id of the party to its respective [BytesChannel].
    pub async fn get_coordinator_byte_channel(&self) -> std::io::Result<Option<BytesChannel<RecvStream, SendStream>>> {
        if let Some(conn) = self.coordinator_connection.as_ref() {
            // set max frame length to 1Tb and length_field_length to 5 bytes
            const NUM_BYTES: usize = 5;
            let codec = LengthDelimitedCodec::builder()
                .length_field_type::<u64>() // u64 because this is the type the length is decoded into, and u32 doesnt fit 5 bytes
                .length_field_length(NUM_BYTES)
                .max_frame_length(1usize << (NUM_BYTES * 8))
                .new_codec();

            // we are the client, so we are the receiver
            let (mut send_stream, recv_stream) = conn.open_bi().await?;
            send_stream.write_u32(self.my_id as u32).await?;
            send_stream.write_u32(self.worker as u32).await?;

            Ok(Some(Channel::new(recv_stream, send_stream, codec.clone())))
        } else {
            Ok(None)
        }
    }
}

impl MpcNetworkHandlerShutdown for MpcNetworkHandlerWorker {
    /// Shutdown all connections, and call [`quinn::Endpoint::wait_idle`] on all of them
    async fn shutdown(&self) -> std::io::Result<()> {
        for (id, conns) in self.parties_connections.iter() {
            for conn in conns {
                let res = async {
                    if self.my_id < *id {
                        let mut send = conn.open_uni().await?;
                        send.write_all(b"done").await?;
                    } else {
                        let mut recv = conn.accept_uni().await?;
                        let mut buffer = vec![0u8; b"done".len()];
                        recv.read_exact(&mut buffer).await.map_err(|_| {
                            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "failed to recv done msg")
                        })?;

                        conn.close(0u32.into(), format!("close from party {}", self.my_id).as_bytes());
                    }
                    Ok::<_, std::io::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::trace!(party = id, "shutdown handshake failed (peer may have exited): {e}");
                }
            }
        }

        if let Some(conn) = self.coordinator_connection.as_ref() {
            let res = async {
                let mut send = conn.open_uni().await?;
                send.write_all(b"done").await
            }
            .await;
            if let Err(e) = res {
                tracing::trace!("coordinator shutdown handshake failed (coordinator may have exited): {e}");
            }
        }

        // Close all known connections so wait_idle can complete.
        for (_, conns) in self.parties_connections.iter() {
            for conn in conns {
                conn.close(0u32.into(), b"shutdown");
            }
        }
        if let Some(conn) = self.coordinator_connection.as_ref() {
            conn.close(0u32.into(), b"shutdown");
        }

        for endpoint in self.endpoints.iter() {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), endpoint.wait_idle()).await;
            endpoint.close(VarInt::from_u32(0), &[]);
        }
        Ok(())
    }
}
