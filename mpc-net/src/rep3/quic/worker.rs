use crate::{
    channel::{BytesChannel, Channel},
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
use quinn::{
    crypto::rustls::QuicClientConfig,
    rustls::{pki_types::CertificateDer, RootCertStore},
};
use quinn::{
    ClientConfig, Connection, Endpoint, IdleTimeout, RecvStream, SendStream, TransportConfig,
    VarInt,
};
use serde::{de::DeserializeOwned, Serialize};
use std::{collections::BTreeMap, iter, sync::Arc};
use std::{
    collections::HashMap,
    io,
    net::{SocketAddr, ToSocketAddrs},
    time::Duration,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::codec::{Decoder, Encoder, LengthDelimitedCodec};

use crate::{
    channel::ChannelHandle, config::NetworkConfig, mpc_star::MpcStarNetWorker,
    MpcNetworkHandlerWrapper, Result,
};

pub struct Rep3QuicMpcNetWorker {
    pub id: PartyWorkerID,
    pub chan_next: ChannelHandle<Bytes, BytesMut>,
    pub chan_prev: ChannelHandle<Bytes, BytesMut>,
    pub chan_coordinator: Option<ChannelHandle<Bytes, BytesMut>>,
    pub log_num_pub_workers: usize,
    pub log_num_workers_per_party: usize,
    pub net_handler: Arc<MpcNetworkHandlerWrapper>,
    pub config: NetworkConfig,
}

impl Rep3QuicMpcNetWorker {
    pub fn new(config: NetworkConfig) -> Result<Self> {
        Self::new_with_coordinator(config, 0, 0)
    }

    pub fn new_with_coordinator(
        config: NetworkConfig,
        log_num_workers_per_party: usize,
        log_num_pub_workers: usize,
    ) -> Result<Self> {
        ensure!(
            config.parties.len() == 3,
            "REP3 protocol requires exactly 3 parties"
        );
        ensure!(
            config.coordinator.is_some(),
            "REP3 star protocol requires a coordinator"
        );
        let id = PartyWorkerID::new(config.my_id, config.worker);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let (net_handler, chan_next, chan_prev, chan_coordinator) = runtime.block_on(async {
            let net_handler = MpcNetworkHandlerWorker::establish(config.clone()).await?;
            let chan_coordinator = net_handler
                .get_coordinator_byte_channel()
                .await?
                .map(ChannelHandle::manage);
            let mut channels = net_handler.get_byte_channels().await?;
            let chan_next = channels
                .remove(&id.party_id().next_id().into())
                .ok_or(eyre::eyre!("no next channel found"))?;
            let chan_prev = channels
                .remove(&id.party_id().prev_id().into())
                .ok_or(eyre::eyre!("no prev channel found"))?;
            if !channels.is_empty() {
                bail!("unexpected channels found")
            }

            let chan_next = ChannelHandle::manage(chan_next);
            let chan_prev = ChannelHandle::manage(chan_prev);
            Ok((net_handler, chan_next, chan_prev, chan_coordinator))
        })?;
        Ok(Self {
            id,
            net_handler: Arc::new(MpcNetworkHandlerWrapper::new(runtime, net_handler)),
            chan_next,
            chan_prev,
            chan_coordinator,
            log_num_pub_workers,
            log_num_workers_per_party,
            config,
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot send to self",
            ));
        }
    }

    /// Receives bytes over the network from the party with the given id.
    pub fn recv_bytes(&mut self, from: PartyID) -> std::io::Result<BytesMut> {
        let data = if from == self.id.party_id().prev_id() {
            self.chan_prev.blocking_recv().blocking_recv()
        } else if from == self.id.party_id().next_id() {
            self.chan_next.blocking_recv().blocking_recv()
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot recv from self",
            ));
        };
        let data = data.map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "receive channel end died")
        })??;
        Ok(data)
    }

    /// Print the connection stats of the network
    pub fn print_connection_stats(&self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        self.net_handler.inner.print_connection_stats(out)
    }

    /// Print the connection stats of the network
    pub fn log_connection_stats(&self) {
        // hack: wait arbitrary time for all send/recv tasks till now to complete
        std::thread::sleep(std::time::Duration::from_secs(1));
        self.net_handler
            .runtime
            .block_on(async { self.net_handler.inner.log_connection_stats() })
    }
}

impl MpcStarNetWorker for Rep3QuicMpcNetWorker {
    fn send_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: T,
    ) -> Result<()> {
        let size = data.uncompressed_size();
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
            .context("while serializing data")?;

        std::mem::drop(
            self.chan_coordinator
                .as_ref()
                .unwrap()
                .blocking_send(Bytes::from(ser_data)),
        );
        Ok(())
    }

    fn receive_request<T: CanonicalSerialize + CanonicalDeserialize>(&mut self) -> Result<T> {
        let response = self
            .chan_coordinator
            .as_ref()
            .unwrap()
            .blocking_recv()
            .blocking_recv()
            .context("while receiving response")??;

        let ret = T::deserialize_uncompressed_unchecked(&response[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
            .context("while deserializing response")?;
        Ok(ret)
    }

    fn log_num_pub_workers(&self) -> usize {
        self.log_num_pub_workers
    }

    fn log_num_workers_per_party(&self) -> usize {
        self.log_num_workers_per_party
    }

    fn total_bandwidth_used(&self) -> (u64, u64) {
        let sent_bytes = self
            .net_handler
            .inner
            .parties_connections
            .iter()
            .map(|(_, conn)| conn.stats().udp_tx.bytes as u64)
            .sum();
        let recv_bytes = self
            .net_handler
            .inner
            .parties_connections
            .iter()
            .map(|(_, conn)| conn.stats().udp_rx.bytes as u64)
            .sum();
        (sent_bytes, recv_bytes)
    }

    fn party_id(&self) -> PartyID {
        self.id.party_id()
    }

    fn fork_with_coordinator(&mut self) -> Result<Self> {
        let id = self.id.clone();
        let net_handler = Arc::clone(&self.net_handler);
        let (chan_next, chan_prev, chan_coordinator) = net_handler.runtime.block_on(async {
            let chan_coordinator = net_handler
                .inner
                .get_coordinator_byte_channel()
                .await?
                .map(ChannelHandle::manage);

            let mut channels = net_handler.inner.get_byte_channels().await?;

            let chan_next = channels
                .remove(&id.party_id().next_id().into())
                .expect("to find next channel");
            let chan_prev = channels
                .remove(&id.party_id().prev_id().into())
                .expect("to find prev channel");
            if !channels.is_empty() {
                panic!("unexpected channels found")
            }

            let chan_next = ChannelHandle::manage(chan_next);
            let chan_prev = ChannelHandle::manage(chan_prev);

            Ok::<_, Report>((chan_next, chan_prev, chan_coordinator))
        })?;

        Ok(Self {
            id,
            net_handler: net_handler,
            chan_next,
            chan_prev,
            chan_coordinator,
            log_num_pub_workers: self.log_num_pub_workers,
            log_num_workers_per_party: self.log_num_workers_per_party,
            config: self.config.clone(),
        })
    }

    #[tracing::instrument(skip_all, name = "MpcStarNetWorker::fork_into_worker_subnets")]
    fn fork_into_worker_subnets(&mut self, num_workers: usize) -> Result<Vec<Self>> {
        let config = self.config.clone();
        let log_num_workers_per_party = self.log_num_workers_per_party;
        let log_num_pub_workers = self.log_num_pub_workers;
        iter::once(self.fork_with_coordinator())
            .chain((1..num_workers).map(|worker_id| {
                Self::new_with_coordinator(
                    config.for_worker(worker_id),
                    log_num_workers_per_party,
                    log_num_pub_workers,
                )
            }))
            .collect::<Result<Vec<_>>>()
    }
}

/// A network handler for MPC protocols.
#[derive(Debug)]
pub struct MpcNetworkHandlerWorker {
    // this is a btreemap because we rely on iteration order
    parties_connections: BTreeMap<usize, Connection>,
    coordinator_connection: Option<Connection>,
    endpoints: Vec<Endpoint>,
    my_id: usize,
    worker: usize,
}

impl MpcNetworkHandlerWorker {
    /// Tries to establish a connection to other parties in the network based on the provided [NetworkConfig].
    pub async fn establish(config: NetworkConfig) -> Result<Self, Report> {
        config.check_config()?;
        let certs: HashMap<usize, CertificateDer> = config
            .parties
            .iter()
            .map(|p| (p.id, p.cert.clone()))
            .collect();

        let mut root_store = RootCertStore::empty();
        for (id, cert) in &certs {
            root_store
                .add(cert.clone())
                .with_context(|| format!("adding certificate for party {id} to root store"))?;
        }
        if let Some(coordinator) = &config.coordinator {
            root_store
                .add(coordinator.cert.clone())
                .with_context(|| format!("adding certificate for coordinator to root store"))?;
        }
        let crypto = quinn::rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let client_config = {
            let mut transport_config = TransportConfig::default();
            // we dont set this to timeout, because it is the timeout for a idle connection
            // maybe we want to make this configurable too?
            transport_config.max_idle_timeout(Some(
                IdleTimeout::try_from(Duration::from_secs(60)).unwrap(),
            ));
            // atm clients send keepalive packets
            transport_config.keep_alive_interval(Some(Duration::from_secs(1)));
            let mut client_config =
                ClientConfig::new(Arc::new(QuicClientConfig::try_from(crypto)?));
            client_config.transport_config(Arc::new(transport_config));
            client_config
        };

        let server_config =
            quinn::ServerConfig::with_single_cert(vec![certs[&config.my_id].clone()], config.key)
                .context("creating our server config")?;
        let our_socket_addr = config.bind_addr;

        let mut endpoints = Vec::new();
        let server_endpoint = quinn::Endpoint::server(server_config.clone(), our_socket_addr)?;

        let coordinator_connection = if let Some(coordinator) = config.coordinator {
            tracing::trace!("my id: {:?}, connecting to coordinator", config.my_id);

            let addresses: Vec<SocketAddr> = coordinator
                .dns_name
                .to_socket_addrs()
                .with_context(|| format!("while resolving DNS name for {}", coordinator.dns_name))?
                .collect();
            if addresses.is_empty() {
                return Err(eyre::eyre!(
                    "could not resolve DNS name {}",
                    coordinator.dns_name
                ));
            }
            let party_addr = addresses[0];
            let local_client_socket: SocketAddr = match party_addr {
                SocketAddr::V4(_) => "0.0.0.0:0".parse().expect("hardcoded IP address is valid"),
                SocketAddr::V6(_) => "[::]:0".parse().expect("hardcoded IP address is valid"),
            };
            let endpoint = quinn::Endpoint::client(local_client_socket)
                .with_context(|| format!("creating client endpoint to coordinator"))?;
            let conn = endpoint
                .connect_with(
                    client_config.clone(),
                    party_addr,
                    &coordinator.dns_name.hostname,
                )
                .with_context(|| format!("setting up client connection with coordinator"))?
                .await
                .with_context(|| format!("connecting as a client to coordinator"))?;
            let mut uni = conn.open_uni().await?;
            uni.write_u32(u32::try_from(config.my_id).expect("party id fits into u32"))
                .await?;
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
        } else {
            None
        };

        let mut parties_connections = BTreeMap::new();

        for party in config.parties {
            if party.id == config.my_id {
                // skip self
                continue;
            }
            if party.id < config.my_id {
                tracing::trace!(
                    "my id: {:?}, connecting to party: {:?}",
                    config.my_id,
                    party.id
                );
                // connect to party, we are client

                let party_addresses: Vec<SocketAddr> = party
                    .dns_name
                    .to_socket_addrs()
                    .with_context(|| format!("while resolving DNS name for {}", party.dns_name))?
                    .collect();
                if party_addresses.is_empty() {
                    return Err(eyre::eyre!("could not resolve DNS name {}", party.dns_name));
                }
                let party_addr = party_addresses[0];
                let local_client_socket: SocketAddr = match party_addr {
                    SocketAddr::V4(_) => {
                        "0.0.0.0:0".parse().expect("hardcoded IP address is valid")
                    }
                    SocketAddr::V6(_) => "[::]:0".parse().expect("hardcoded IP address is valid"),
                };
                let endpoint = quinn::Endpoint::client(local_client_socket)
                    .with_context(|| format!("creating client endpoint to party {}", party.id))?;
                let conn = endpoint
                    .connect_with(client_config.clone(), party_addr, &party.dns_name.hostname)
                    .with_context(|| {
                        format!("setting up client connection with party {}", party.id)
                    })?
                    .await
                    .with_context(|| format!("connecting as a client to party {}", party.id))?;
                let mut uni = conn.open_uni().await?;
                uni.write_u32(u32::try_from(config.my_id).expect("party id fits into u32"))
                    .await?;
                uni.flush().await?;
                uni.finish()?;
                tracing::trace!(
                    "Conn with id {} from {} to {}",
                    conn.stable_id(),
                    endpoint.local_addr().unwrap(),
                    conn.remote_address(),
                );
                assert!(parties_connections.insert(party.id, conn).is_none());
                endpoints.push(endpoint);
            } else {
                tracing::trace!(
                    "my id: {:?}, accepting connection from party: {:?}",
                    config.my_id,
                    party.id
                );

                // we are the server, accept a connection
                match tokio::time::timeout(
                    config.timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT),
                    server_endpoint.accept(),
                )
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
                        let other_party_id = uni.read_u32().await?;
                        tracing::trace!(
                            "my id: {:?}, accepted connection from other_party_id: {:?}",
                            config.my_id,
                            other_party_id
                        );
                        assert!(parties_connections
                            .insert(
                                usize::try_from(other_party_id).expect("u32 fits into usize"),
                                conn
                            )
                            .is_none());
                    }
                    Ok(None) => {
                        return Err(eyre::eyre!(
                            "server endpoint did not accept a connection from party {}",
                            party.id
                        ));
                    }
                    Err(_) => {
                        return Err(eyre::eyre!(
                            "party {} did not connect within 60 seconds - timeout",
                            party.id
                        ));
                    }
                }
            }
        }
        endpoints.push(server_endpoint);

        Ok(MpcNetworkHandlerWorker {
            parties_connections,
            coordinator_connection,
            endpoints,
            my_id: config.my_id,
            worker: config.worker,
        })
    }

    /// Returns the number of sent and received bytes.
    pub fn get_send_receive(&self, i: usize) -> std::io::Result<(u64, u64)> {
        let conn = self
            .parties_connections
            .get(&i)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such connection"))?;
        let stats = conn.stats();
        Ok((stats.udp_tx.bytes, stats.udp_rx.bytes))
    }

    /// Prints the connection statistics.
    pub fn print_connection_stats(&self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        for (i, conn) in &self.parties_connections {
            let stats = conn.stats();
            writeln!(
                out,
                "Connection {} stats:\n\tSENT: {} bytes\n\tRECV: {} bytes",
                i, stats.udp_tx.bytes, stats.udp_rx.bytes
            )?;
        }
        Ok(())
    }

    /// Prints the connection statistics.
    pub fn log_connection_stats(&self) {
        for (i, conn) in &self.parties_connections {
            let stats = conn.stats();
            tracing::info!(
                "Connection {} stats: SENT: {} bytes RECV: {} bytes",
                i,
                ByteSize(stats.udp_tx.bytes),
                ByteSize(stats.udp_rx.bytes)
            );
        }

        if let Some(conn) = self.coordinator_connection.as_ref() {
            let stats = conn.stats();
            tracing::info!(
                "Coordinator connection stats: SENT: {} bytes RECV: {} bytes",
                ByteSize(stats.udp_tx.bytes),
                ByteSize(stats.udp_rx.bytes)
            );
        }
    }

    /// Sets up a new [BytesChannel] between each party. The resulting map maps the id of the party to its respective [BytesChannel].
    pub async fn get_byte_channels(
        &self,
    ) -> std::io::Result<HashMap<usize, BytesChannel<RecvStream, SendStream>>> {
        // set max frame length to 1Tb and length_field_length to 5 bytes
        const NUM_BYTES: usize = 5;
        let codec = LengthDelimitedCodec::builder()
            .length_field_type::<u64>() // u64 because this is the type the length is decoded into, and u32 doesnt fit 5 bytes
            .length_field_length(NUM_BYTES)
            .max_frame_length(1usize << (NUM_BYTES * 8))
            .new_codec();
        self.get_custom_channels(codec).await
    }

    /// Set up a new [Channel] using [BincodeCodec] between each party. The resulting map maps the id of the party to its respective [Channel].
    pub async fn get_serde_bincode_channels<M: Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> std::io::Result<HashMap<usize, Channel<RecvStream, SendStream, BincodeCodec<M>>>> {
        let bincodec = BincodeCodec::<M>::new();
        self.get_custom_channels(bincodec).await
    }

    /// Set up a new [Channel] using the provided codec between each party. The resulting map maps the id of the party to its respective [Channel].
    pub async fn get_custom_channels<
        MSend,
        MRecv,
        C: Encoder<MSend, Error = io::Error>
            + Decoder<Item = MRecv, Error = io::Error>
            + 'static
            + Clone,
    >(
        &self,
        codec: C,
    ) -> std::io::Result<HashMap<usize, Channel<RecvStream, SendStream, C>>> {
        let mut channels = HashMap::with_capacity(self.parties_connections.len() - 1);
        for (&id, conn) in self.parties_connections.iter() {
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
    pub async fn get_coordinator_byte_channel(
        &self,
    ) -> std::io::Result<Option<BytesChannel<RecvStream, SendStream>>> {
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
        tracing::trace!(
            "party {} worker {} shutting down, conns = {:?}",
            self.my_id,
            self.worker,
            self.parties_connections.keys()
        );

        for (id, conn) in self.parties_connections.iter() {
            if self.my_id < *id {
                let mut send = conn.open_uni().await?;
                send.write_all(b"done").await?;
            } else {
                let mut recv = conn.accept_uni().await?;
                let mut buffer = vec![0u8; b"done".len()];
                recv.read_exact(&mut buffer).await.map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "failed to recv done msg")
                })?;

                conn.close(
                    0u32.into(),
                    format!("close from party {}", self.my_id).as_bytes(),
                );
            }
        }


        if let Some(conn) = self.coordinator_connection.as_ref() {
            let mut send = conn.open_uni().await?;
            send.write_all(b"done").await?;
        }

        for endpoint in self.endpoints.iter() {
            endpoint.wait_idle().await;
            endpoint.close(VarInt::from_u32(0), &[]);
        }
        tracing::trace!("party {} worker {} shut down", self.my_id, self.worker);
        Ok(())
    }
}
