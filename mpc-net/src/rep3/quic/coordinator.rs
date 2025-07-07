use crate::{
    channel::{BytesChannel, Channel},
    codecs::BincodeCodec,
    rep3::PartyWorkerID,
    MpcNetworkHandlerShutdown, MpcNetworkHandlerWrapper, DEFAULT_CONNECT_TIMEOUT,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::{Bytes, BytesMut};
use color_eyre::eyre::{self, Report};
use color_eyre::eyre::{bail, Context};
use quinn::{
    rustls::pki_types::CertificateDer, Connection, Endpoint, RecvStream, SendStream, VarInt,
};
use serde::{de::DeserializeOwned, Serialize};
use std::{collections::BTreeMap, sync::Arc};
use std::{collections::HashMap, io};
use tokio::io::AsyncReadExt;
use tokio_util::codec::{Decoder, Encoder, LengthDelimitedCodec};

use crate::{
    channel::ChannelHandle, config::NetworkConfig, mpc_star::MpcStarNetCoordinator, Result,
};

pub struct Rep3QuicNetCoordinator {
    pub(crate) channels: BTreeMap<usize, ChannelHandle<Bytes, BytesMut>>,
    pub(crate) net_handler: Arc<MpcNetworkHandlerWrapper<MpcNetworkCoordinatorHandler>>,
    pub(crate) log_num_pub_workers: usize,
    pub(crate) log_num_workers_per_party: usize,
}

impl Rep3QuicNetCoordinator {
    pub fn new(
        config: NetworkConfig,
        log_num_pub_workers: usize,
        log_num_workers_per_party: usize,
    ) -> Result<Self> {
        if (config.parties.len() - 1) % 3 != 0 {
            bail!("REP3 protocol requires exactly 3 workers per party")
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let (net_handler, channels) = runtime.block_on(async {
            let net_handler = MpcNetworkCoordinatorHandler::establish(config)
                .await
                .context("establishing network handler")?;
            let channels: BTreeMap<usize, ChannelHandle<Bytes, BytesMut>> = net_handler
                .get_byte_channels()
                .await
                .context("getting byte channels")?
                .into_iter()
                .map(|(id, channel)| (id, ChannelHandle::manage(channel)))
                .collect();
            if !channels.is_empty() {
                bail!("unexpected channels found")
            }

            Ok::<_, Report>((net_handler, channels))
        })?;

        Ok(Self {
            channels,
            net_handler: Arc::new(MpcNetworkHandlerWrapper::new(runtime, net_handler)),
            log_num_pub_workers,
            log_num_workers_per_party,
        })
    }
}

impl MpcStarNetCoordinator for Rep3QuicNetCoordinator {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        _default_response: T,
    ) -> Result<Vec<T>> {
        let mut responses_bytes = Vec::new();
        for (_, channel) in self.channels.iter_mut() {
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
                    .context("while deserializing response")
            })
            .collect::<Result<Vec<_>>>()
    }

    fn broadcast_request<
        T: ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize + Clone,
    >(
        &mut self,
        data: T,
    ) -> Result<()> {
        let size = data.uncompressed_size();
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        for (_, channel) in self.channels.iter_mut() {
            std::mem::drop(channel.blocking_send(Bytes::from(ser_data.clone())));
        }

        Ok(())
    }

    fn send_requests<
        T: ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize + Clone,
    >(
        &mut self,
        data: Vec<T>,
    ) -> Result<()> {
        for (&i, channel) in self.channels.iter_mut() {
            let size = data[i].uncompressed_size();
            let mut ser_data = Vec::with_capacity(size);
            data[i]
                .serialize_uncompressed(&mut ser_data)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
            std::mem::drop(channel.blocking_send(Bytes::from(ser_data.clone())));
        }
        Ok(())
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
            .connections
            .iter()
            .map(|(_, conn)| conn.stats().udp_tx.bytes as u64)
            .sum();
        let recv_bytes = self
            .net_handler
            .inner
            .connections
            .iter()
            .map(|(_, conn)| conn.stats().udp_rx.bytes as u64)
            .sum();
        (sent_bytes, recv_bytes)
    }
}

/// A network handler for MPC protocols.
#[derive(Debug)]
pub struct MpcNetworkCoordinatorHandler {
    // this is a btreemap because we rely on iteration order
    connections: BTreeMap<usize, Connection>,
    endpoints: Vec<Endpoint>,
    my_id: usize,
}

impl MpcNetworkCoordinatorHandler {
    /// Tries to establish a connection to other parties in the network based on the provided [NetworkConfig].
    pub async fn establish(config: NetworkConfig) -> Result<Self, Report> {
        config.check_config()?;
        let certs: HashMap<usize, CertificateDer> = config
            .parties
            .iter()
            .map(|p| (p.id, p.cert.clone()))
            .collect();

        let server_config =
            quinn::ServerConfig::with_single_cert(vec![certs[&config.my_id].clone()], config.key)
                .context("creating our server config")?;
        let our_socket_addr = config.bind_addr;

        let mut endpoints = Vec::new();
        let server_endpoint = quinn::Endpoint::server(server_config.clone(), our_socket_addr)?;

        let mut connections = BTreeMap::new();

        tracing::info!(
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

                    tracing::info!(
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

        endpoints.push(server_endpoint);

        Ok(MpcNetworkCoordinatorHandler {
            connections,
            endpoints,
            my_id: config.my_id,
        })
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
        self.get_custom_channels(codec).await
    }

    /// Set up a new [Channel] using [BincodeCodec] between each party. The resulting map maps the id of the party to its respective [Channel].
    pub async fn get_serde_bincode_channels<M: Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> std::io::Result<BTreeMap<usize, Channel<RecvStream, SendStream, BincodeCodec<M>>>> {
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
    ) -> std::io::Result<BTreeMap<usize, Channel<RecvStream, SendStream, C>>> {
        let mut channels = BTreeMap::new();

        // For coordinator, all connections are already established and identified
        // We just need to create channels from the existing bidirectional streams
        for (&global_worker_id, conn) in self.connections.iter() {
            // The streams were already established during connection setup
            // We need to create new streams for data communication
            let (send_stream, mut recv_stream) = conn.open_bi().await?;
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
            "party {} shutting down, conns = {:?}",
            self.my_id,
            self.connections.keys()
        );

        for (id, conn) in self.connections.iter() {
            if self.my_id < *id {
                let mut send = conn.open_uni().await?;
                send.write_all(b"done").await?;
            } else {
                let mut recv = conn.accept_uni().await?;
                let mut buffer = vec![0u8; b"done".len()];
                recv.read_exact(&mut buffer).await.map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "failed to recv done msg")
                })?;

                tracing::debug!("party {} closing conn = {id}", self.my_id);

                conn.close(
                    0u32.into(),
                    format!("close from party {}", self.my_id).as_bytes(),
                );
            }
        }
        for endpoint in self.endpoints.iter() {
            endpoint.wait_idle().await;
            endpoint.close(VarInt::from_u32(0), &[]);
        }
        Ok(())
    }
}
