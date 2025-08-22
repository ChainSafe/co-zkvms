use crate::{
    channel::{BytesChannel, Channel},
    codecs::BincodeCodec,
    rep3::PartyWorkerID,
    MpcNetworkHandlerShutdown, DEFAULT_CONNECT_TIMEOUT,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::{Bytes, BytesMut};
use bytesize::ByteSize;
use color_eyre::eyre::{self, Report};
use color_eyre::eyre::{bail, Context};
use mpc_types::protocols::rep3::id::PartyID;
use quinn::{Connection, Endpoint, IdleTimeout, RecvStream, SendStream, TransportConfig, VarInt};
use serde::{de::DeserializeOwned, Serialize};
use std::io;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio::io::AsyncReadExt;
use tokio_util::codec::{Decoder, Encoder, LengthDelimitedCodec};

use rayon::prelude::*;

use crate::{
    channel::ChannelHandle, config::NetworkConfig, mpc_star::MpcStarNetCoordinator, Result,
    MpcNetworkHandlerWrapperMut,
};

#[derive(Clone)]
pub struct Rep3QuicNetCoordinator {
    pub(crate) channels: BTreeMap<usize, ChannelHandle<Bytes, BytesMut>>,
    pub(crate) net_handler: Arc<MpcNetworkHandlerWrapperMut<MpcNetworkCoordinatorHandler>>,
    pub(crate) log_num_workers_per_party: usize,
    pub(crate) stats_checkpoints: Vec<(u64, u64)>,
    pub(crate) config: NetworkConfig,
    pub(crate) current_num_workers: usize,
}

impl Rep3QuicNetCoordinator {
    pub fn new(config: NetworkConfig, log_num_workers_per_party: usize) -> Result<Self> {
        if config.parties.len() % 3 != 0 {
            bail!("REP3 protocol requires exactly 3 workers per party")
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let (net_handler, channels) = runtime.block_on(async {
            let net_handler = MpcNetworkCoordinatorHandler::establish(config.clone())
                .await
                .context("establishing network handler")?;
            let channels: BTreeMap<usize, ChannelHandle<Bytes, BytesMut>> = net_handler
                .get_byte_channels()
                .await
                .context("getting byte channels")?
                .into_iter()
                .map(|(id, channel)| (id, ChannelHandle::manage(channel)))
                .collect();

            Ok::<_, Report>((net_handler, channels))
        })?;

        let num_parties = channels.len();

        Ok(Self {
            channels,
            net_handler: Arc::new(MpcNetworkHandlerWrapperMut::new(runtime, net_handler)),
            log_num_workers_per_party,
            stats_checkpoints: vec![(0, 0); num_parties * (1 << log_num_workers_per_party)],
            config,
            current_num_workers: num_parties / 3,
        })
    }

    fn channels(&mut self) -> impl Iterator<Item = (&usize, &mut ChannelHandle<Bytes, BytesMut>)> {
        self.channels
            .iter_mut()
            .filter(|(&id, _)| id < self.current_num_workers * 3)
    }

    fn channels_par(
        &mut self,
    ) -> impl ParallelIterator<Item = (&usize, &mut ChannelHandle<Bytes, BytesMut>)> {
        self.channels
            .par_iter_mut()
            .filter(|(&id, _)| id < self.current_num_workers * 3)
    }
}

impl MpcStarNetCoordinator for Rep3QuicNetCoordinator {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> Result<Vec<T>> {
        let mut responses_bytes = Vec::new();
        // todo try recieve in parallel
        for (_, channel) in self.channels() {
            let response = channel
                .blocking_recv()
                .blocking_recv()
                .context("while receiving response")??;
            responses_bytes.push(response);
        }

        responses_bytes
            .iter()
            .map(|data| {
                T::deserialize_uncompressed_unchecked(&data[..])
                    .map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                    })
                    .context("while deserializing response")
            })
            .collect::<Result<Vec<_>>>()
    }

    fn receive_responses_from_subnets<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> Result<Vec<Vec<T>>> {
        let mut responses_bytes: Vec<Vec<_>> = Vec::with_capacity(2);
        for (global_worker_id, channel) in self.channels() {
            let worker_idx = PartyWorkerID::from_global_worker_id(*global_worker_id).worker_idx();
            let response = channel
                .blocking_recv()
                .blocking_recv()
                .context("while receiving response")??;
            match responses_bytes.get_mut(worker_idx) {
                None => {
                    responses_bytes.insert(worker_idx, vec![response]);
                }
                Some(responses) => responses.push(response),
            }
        }

        responses_bytes
            .iter()
            .map(|data| {
                data.iter()
                    .map(|data| {
                        T::deserialize_uncompressed_unchecked(&data[..])
                            .map_err(|e| {
                                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                            })
                            .context("while deserializing response")
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()
    }

    fn broadcast_request<
        T: ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
    >(
        &mut self,
        data: T,
    ) -> Result<()> {
        let size = data.uncompressed_size();
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
            .context("while serializing data")?;

        self.channels_par().for_each(|(_, channel)| {
            std::mem::drop(channel.blocking_send(Bytes::from(ser_data.clone())));
        });

        Ok(())
    }

    fn send_requests<T: ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> Result<()> {
        self.channels_par()
            .map(|(i, channel)| {
                let size = data[*i].uncompressed_size();
                let mut ser_data = Vec::with_capacity(size);
                data[*i]
                    .serialize_uncompressed(&mut ser_data)
                    .map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                    })
                    .context("while serializing data")?;
                std::mem::drop(channel.blocking_send(Bytes::from(ser_data)));
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    fn send_requests_blocking<
        T: ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
    >(
        &mut self,
        data: Vec<T>,
    ) -> Result<()> {
        let resvs = self
            .channels_par()
            .map(|(i, channel)| {
                let size = data[*i].uncompressed_size();
                let mut ser_data = Vec::with_capacity(size);
                data[*i]
                    .serialize_uncompressed(&mut ser_data)
                    .map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                    })
                    .context("while serializing data")?;
                Ok(channel.blocking_send(Bytes::from(ser_data)))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(data);
        resvs
            .into_iter()
            .map(|resv| resv.blocking_recv().context("while sending request"))
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    fn send_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
        data: T,
    ) -> Result<()> {
        let channel = self
            .channels
            .get_mut(&PartyWorkerID::new(party_id.into(), worker_id).global_worker_id())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no such party"))?;
        let size = data.uncompressed_size();
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
            .context("while serializing data")?;
        std::mem::drop(channel.blocking_send(Bytes::from(ser_data.clone())));
        Ok(())
    }

    fn receive_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
    ) -> Result<T> {
        let channel = self
            .channels
            .get_mut(&PartyWorkerID::new(party_id.into(), worker_id).global_worker_id())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no such party"))?;
        let response = channel
            .blocking_recv()
            .blocking_recv()
            .context("while receiving response")??;
        T::deserialize_uncompressed_unchecked(&response[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
            .context("while deserializing response")
    }

    fn log_num_workers_per_party(&self) -> usize {
        self.log_num_workers_per_party
    }

    fn reset_stats(&mut self) {
        let net_handler = self.net_handler.inner.lock().unwrap();
        for (i, conn) in &net_handler.connections {
            let stats = conn.stats();
            self.stats_checkpoints[*i].0 += stats.udp_tx.bytes;
            self.stats_checkpoints[*i].1 += stats.udp_rx.bytes;
        }
    }

    fn total_bandwidth_used(&self) -> (u64, u64) {
        let net_handler = self.net_handler.inner.lock().unwrap();
        let sent_bytes = net_handler
            .connections
            .iter()
            .zip(self.stats_checkpoints.iter())
            .map(|((_, conn), (sent, _))| conn.stats().udp_tx.bytes as u64 - sent)
            .sum();
        let recv_bytes = net_handler
            .connections
            .iter()
            .zip(self.stats_checkpoints.iter())
            .map(|((_, conn), (_, recv))| conn.stats().udp_rx.bytes as u64 - recv)
            .sum();
        (sent_bytes, recv_bytes)
    }

    fn log_connection_stats(&self, label: Option<&str>) {
        // hack: wait arbitrary time for all send/recv tasks till now to complete
        std::thread::sleep(std::time::Duration::from_secs(1));
        let net_handler = self.net_handler.inner.lock().unwrap();
        for (i, conn) in &net_handler.connections {
            let id = PartyWorkerID::from_global_worker_id(*i);
            let stats = conn.stats();
            tracing::info!(
                "{}: connection with {} | stats: SENT: {} bytes | RECV: {} bytes",
                label.unwrap_or("Coordinator stats"),
                id,
                ByteSize(stats.udp_tx.bytes - self.stats_checkpoints[*i].0),
                ByteSize(stats.udp_rx.bytes - self.stats_checkpoints[*i].1)
            );
        }
        drop(net_handler);
        let (sent_bytes, recv_bytes) = self.total_bandwidth_used();
        tracing::info!(
            "{} total: SENT: {} bytes RECV: {} bytes",
            label.unwrap_or("Coordinator stats"),
            ByteSize(sent_bytes),
            ByteSize(recv_bytes)
        );
    }

    fn fork(&mut self) -> Result<Self> {
        // let id = self.id.clone();
        let net_handler = Arc::clone(&self.net_handler);
        let channels = net_handler.runtime.block_on(async {
            let channels: BTreeMap<usize, ChannelHandle<Bytes, BytesMut>> = net_handler
                .inner
                .lock()
                .unwrap()
                .get_byte_channels()
                .await
                .context("getting byte channels")?
                .into_iter()
                .map(|(id, channel)| (id, ChannelHandle::manage(channel)))
                .collect();

            Ok::<_, Report>(channels)
        })?;

        Ok(Self {
            channels,
            net_handler: net_handler,
            log_num_workers_per_party: self.log_num_workers_per_party,
            stats_checkpoints: self.stats_checkpoints.clone(),
            config: self.config.clone(),
            current_num_workers: self.current_num_workers,
        })
    }

    fn extend_with_worker_subnets(&mut self, new_num_workers: usize) -> eyre::Result<()> {
        self.current_num_workers = new_num_workers;
        Ok(())
    }

    fn trim_subnets(&mut self, num_workers: usize) -> Result<()> {
        self.current_num_workers = num_workers;
        Ok(())
    }
}

/// A network handler for MPC protocols.
#[derive(Debug)]
pub struct MpcNetworkCoordinatorHandler {
    // this is a btreemap because we rely on iteration order
    connections: BTreeMap<usize, Connection>,
    server_endpoint: Endpoint,
}

impl MpcNetworkCoordinatorHandler {
    /// Tries to establish a connection to other parties in the network based on the provided [NetworkConfig].
    pub async fn establish(config: NetworkConfig) -> Result<Self, Report> {
        // config.check_config()?;

        let our_cert = config.coordinator.as_ref().unwrap().cert.clone();

        let mut transport_config = TransportConfig::default();
        transport_config.max_idle_timeout(Some(
            IdleTimeout::try_from(Duration::from_secs(5 * 60)).unwrap(),
        ));
        transport_config.keep_alive_interval(Some(Duration::from_secs(1)));

        let mut server_config = quinn::ServerConfig::with_single_cert(vec![our_cert], config.key)
            .context("creating our server config")?;
        server_config.transport_config(Arc::new(transport_config));
        let our_socket_addr = config.bind_addr;

        let server_endpoint = quinn::Endpoint::server(server_config.clone(), our_socket_addr)?;

        let mut connections = BTreeMap::new();

        tracing::trace!(
            "Coordinator expecting {} total worker connections",
            config.parties.len()
        );

        // Accept all connections first, then identify each one
        for _ in 0..config.parties.len() {
            match tokio::time::timeout(
                config.timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT),
                server_endpoint.accept(),
            )
            .await
            {
                Ok(Some(maybe_conn)) => {
                    let conn = maybe_conn.await?;
                    tracing::trace!(
                        "Coordinator accepted connection with id {} from {}",
                        conn.stable_id(),
                        conn.remote_address(),
                    );

                    // Now identify which worker this is
                    let mut uni: RecvStream = conn.accept_uni().await?;
                    let party_id = uni.read_u32().await?;
                    let worker_id = uni.read_u32().await?;

                    let id = PartyWorkerID::new(party_id as usize, worker_id as usize);
                    let global_worker_id = id.global_worker_id();

                    tracing::trace!(
                        "Coordinator identified connection: party {}, worker {}, global_id {}",
                        party_id,
                        worker_id,
                        global_worker_id
                    );

                    assert!(
                        connections.insert(global_worker_id, conn).is_none(),
                        "Duplicate global worker ID: {}",
                        global_worker_id
                    );
                }
                Ok(None) => {
                    return Err(eyre::eyre!(
                        "server endpoint did not accept a connection from party",
                    ));
                }
                Err(_) => {
                    return Err(eyre::eyre!("timeout waiting for worker connection"));
                }
            }
        }

        Ok(MpcNetworkCoordinatorHandler {
            connections,
            server_endpoint,
        })
    }

    /// Tries to establish a connection to other parties in the network based on the provided [NetworkConfig].
    pub async fn extend(&mut self, config: NetworkConfig) -> eyre::Result<()> {
        // config.check_config()?;

        // Accept all connections first, then identify each one
        let mut new_connections = vec![];
        for party in &config.parties {
            let id = PartyWorkerID::new(party.id.into(), party.worker);
            let global_worker_id = id.global_worker_id();
            if self.connections.contains_key(&global_worker_id) {
                continue;
            }

            match tokio::time::timeout(
                config.timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT),
                self.server_endpoint.accept(),
            )
            .await
            {
                Ok(Some(maybe_conn)) => {
                    let conn = maybe_conn.await?;
                    tracing::trace!(
                        "Coordinator accepted connection with id {} from {}",
                        conn.stable_id(),
                        conn.remote_address(),
                    );

                    // Now identify which worker this is
                    let mut uni: RecvStream = conn.accept_uni().await?;
                    let party_id = uni.read_u32().await?;
                    let worker_id = uni.read_u32().await?;

                    let id = PartyWorkerID::new(party_id as usize, worker_id as usize);
                    let global_worker_id = id.global_worker_id();

                    tracing::trace!(
                        "Coordinator identified connection: party {}, worker {}, global_id {}",
                        party_id,
                        worker_id,
                        global_worker_id
                    );

                    new_connections.push((global_worker_id, conn));
                }
                Ok(None) => {
                    return Err(eyre::eyre!(
                        "server endpoint did not accept a connection from party",
                    ));
                }
                Err(_) => {
                    return Err(eyre::eyre!("timeout waiting for worker connection"));
                }
            }
        }

        for (global_worker_id, conn) in new_connections {
            assert!(
                self.connections.insert(global_worker_id, conn).is_none(),
                "Duplicate global worker ID: {}",
                global_worker_id
            );
        }

        Ok(())
    }

    pub async fn trim(&mut self, new_num_workers: usize) -> eyre::Result<()> {
        for (id, conn) in self.connections.iter() {
            if *id > 3 * new_num_workers - 1 {
                let mut recv = conn.accept_uni().await?;
                let mut buffer = vec![0u8; b"done".len()];
                recv.read_exact(&mut buffer).await.map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "failed to recv done msg")
                })?;

                conn.close(0u32.into(), format!("close from coordinator").as_bytes());
            }
        }

        // keep connections for communication stats
        Ok(())
    }

    /// Returns the number of sent and received bytes.
    pub fn get_send_receive(&self, i: usize) -> std::io::Result<(u64, u64)> {
        let conn = self
            .connections
            .get(&i)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such connection"))?;
        let stats = conn.stats();
        Ok((stats.udp_tx.bytes, stats.udp_rx.bytes))
    }

    /// Prints the connection statistics.
    pub fn print_connection_stats(&self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        for (i, conn) in &self.connections {
            let stats = conn.stats();
            writeln!(
                out,
                "Connection {} stats:\n\tSENT: {} bytes\n\tRECV: {} bytes",
                i, stats.udp_tx.bytes, stats.udp_rx.bytes
            )?;
        }
        Ok(())
    }

    /// Sets up a new [BytesChannel] between each party. The resulting map maps the id of the party to its respective [BytesChannel].
    pub async fn get_byte_channels(
        &self,
    ) -> std::io::Result<BTreeMap<usize, BytesChannel<RecvStream, SendStream>>> {
        // set max frame length to 1Tb and length_field_length to 5 bytes
        const NUM_BYTES: usize = 5;
        let codec = LengthDelimitedCodec::builder()
            .length_field_type::<u64>() // u64 because this is the type the length is decoded into, and u32 doesnt fit 5 bytes
            .length_field_length(NUM_BYTES)
            .max_frame_length(1usize << (NUM_BYTES * 8))
            .new_codec();
        self.get_custom_channels(codec, |_| true).await
    }

    pub async fn get_byte_channels_filter(
        &self,
        filter: impl Fn(usize) -> bool,
    ) -> std::io::Result<BTreeMap<usize, BytesChannel<RecvStream, SendStream>>> {
        // set max frame length to 1Tb and length_field_length to 5 bytes
        const NUM_BYTES: usize = 5;
        let codec = LengthDelimitedCodec::builder()
            .length_field_type::<u64>() // u64 because this is the type the length is decoded into, and u32 doesnt fit 5 bytes
            .length_field_length(NUM_BYTES)
            .max_frame_length(1usize << (NUM_BYTES * 8))
            .new_codec();
        self.get_custom_channels(codec, filter).await
    }

    /// Set up a new [Channel] using [BincodeCodec] between each party. The resulting map maps the id of the party to its respective [Channel].
    pub async fn get_serde_bincode_channels<M: Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> std::io::Result<BTreeMap<usize, Channel<RecvStream, SendStream, BincodeCodec<M>>>> {
        let bincodec = BincodeCodec::<M>::new();
        self.get_custom_channels(bincodec, |_| true).await
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
        filter: impl Fn(usize) -> bool,
    ) -> std::io::Result<BTreeMap<usize, Channel<RecvStream, SendStream, C>>> {
        let mut channels = BTreeMap::new();

        // For coordinator, all connections are already established and identified
        // We just need to create channels from the existing bidirectional streams
        for (&global_worker_id, conn) in self.connections.iter().filter(|(&id, _)| filter(id)) {
            // The streams were already established during connection setup
            // We need to create new streams for data communication
            let (send_stream, mut recv_stream) = conn.accept_bi().await?;
            let party_id = recv_stream.read_u32().await?;
            let worker_id = recv_stream.read_u32().await?;
            let id = PartyWorkerID::new(party_id as usize, worker_id as usize);
            assert!(global_worker_id == id.global_worker_id());
            let channel = Channel::new(recv_stream, send_stream, codec.clone());
            assert!(channels.insert(global_worker_id, channel).is_none());
        }

        Ok(channels)
    }
}

impl MpcNetworkHandlerShutdown for MpcNetworkCoordinatorHandler {
    /// Shutdown all connections, and call [`quinn::Endpoint::wait_idle`] on all of them
    async fn shutdown(&self) -> std::io::Result<()> {
        tracing::debug!(
            "coordinator shutting down, conns = {:?}",
            self.connections.keys()
        );

        for (id, conn) in self.connections.iter() {
            let mut recv = conn.accept_uni().await?;
            let mut buffer = vec![0u8; b"done".len()];
            recv.read_exact(&mut buffer).await.map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "failed to recv done msg")
            })?;

            tracing::debug!("coordinator closing conn = {id}");

            conn.close(0u32.into(), format!("close from coordinator").as_bytes());
        }
        self.server_endpoint.wait_idle().await;
        self.server_endpoint.close(VarInt::from_u32(0), &[]);

        Ok(())
    }
}
