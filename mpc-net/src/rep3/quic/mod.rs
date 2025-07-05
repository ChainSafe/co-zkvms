pub mod channel;
pub mod codecs;
pub mod handler;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::{Bytes, BytesMut};
use color_eyre::eyre::{bail, Context};
use std::{collections::BTreeMap, sync::Arc};

use crate::{
    config::NetworkConfig,
    mpc_star::MpcStarNetCoordinator,
    rep3::quic::{channel::ChannelHandle, handler::MpcNetworkHandlerWrapper},
    Result,
};

use handler::MpcNetworkHandler;

pub struct Rep3QuicNetCoordinator {
    pub(crate) channels: BTreeMap<usize, ChannelHandle<Bytes, BytesMut>>,
    pub(crate) net_handler: Arc<MpcNetworkHandlerWrapper>,
}

impl Rep3QuicNetCoordinator {
    pub fn new(config: NetworkConfig) -> Result<Self> {
        if (config.parties.len() - 1) % 3 != 0 {
            bail!("REP3 protocol requires exactly 3 workers per party")
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let (net_handler, channels) = runtime.block_on(async {
            let net_handler = MpcNetworkHandler::establish(config).await.context("establishing network handler")?;
            let mut channels = BTreeMap::new();
            for (id, channel) in net_handler.get_byte_channels().await.context("getting byte channels")? {
                channels.insert(id, ChannelHandle::manage(channel));
            }
            if !channels.is_empty() {
                bail!("unexpected channels found")
            }

            Ok((net_handler, channels))
        })?;

        Ok(Self {
            channels,
            net_handler: Arc::new(MpcNetworkHandlerWrapper::new(runtime, net_handler)),
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
                T::deserialize_compressed_unchecked(&data[..])
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
        todo!()
    }

    fn log_num_workers_per_party(&self) -> usize {
        todo!()
    }

    fn total_bandwidth_used(&self) -> (usize, usize) {
        todo!()
    }
}
