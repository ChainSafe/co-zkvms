//! Rep3 Network
//!
//! This module contains implementation of the rep3 mpc network

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use async_trait::async_trait;
use bytes::Bytes;
use bytesize::ByteSize;
use eyre::Context;
use mpc_types::field::PrimeField;
use std::iter;
use std::sync::{Arc, OnceLock};

use itertools::Itertools;
use mpc_net::topology::{MpcStarNetCoordinator, MpcStarNetWorker};

use mpc_net::rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator};
use rand::{CryptoRng, Rng, SeedableRng, distributions::Standard, prelude::Distribution};

use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::rngs::RngForker;
use crate::{IoResult, RngType};

use rayon::iter::Either;
use rayon::prelude::*;

use super::{
    conversion::A2BType,
    rngs::{Rep3CorrelatedRng, Rep3Rand, Rep3RandBitComp},
};

// this will be moved later
/// This struct handles networking and rng
pub struct IoContext<N: Rep3Network> {
    /// The party id
    pub id: PartyID,
    /// The correlated rng
    pub rngs: Rep3CorrelatedRng,
    /// The underlying unique rng used for, e.g., Yao
    pub rng: RngType,
    /// The underlying network
    pub network: N,
    /// The used arithmetic/binary conversion protocol
    pub a2b_type: A2BType,

    rng_src: Arc<RngForker<RngType>>,
    rngs_src: Arc<RngForker<Rep3CorrelatedRng>>,
}

impl<N: Rep3Network> Clone for IoContext<N> {
    fn clone(&self) -> Self {
        let child_rngs = self.rngs_src.fork(); // lock once, derive new RNG
        let child_rng = self.rng_src.fork(); // lock once, derive new RNG

        IoContext {
            id: self.id,
            rngs: child_rngs,
            rng: child_rng,
            network: self.network.clone(),
            a2b_type: self.a2b_type,
            rng_src: Arc::clone(&self.rng_src),
            rngs_src: Arc::clone(&self.rngs_src),
        }
    }
}

impl<N: Rep3Network> IoContext<N> {
    fn setup_prf<R: Rng + CryptoRng>(network: &mut N, rng: &mut R) -> IoResult<Rep3Rand> {
        let seed1: [u8; crate::SEED_SIZE] = rng.r#gen();
        network.send_next(seed1)?;
        let seed2: [u8; crate::SEED_SIZE] = network.recv_prev()?;

        Ok(Rep3Rand::new(seed1, seed2))
    }

    fn setup_bitcomp(
        network: &mut N,
        rands: &mut Rep3Rand,
    ) -> IoResult<(Rep3RandBitComp, Rep3RandBitComp)> {
        let (k1a, k1c) = rands.random_seeds();
        let (k2a, k2c) = rands.random_seeds();
        match network.get_id() {
            PartyID::ID0 => {
                network.send_next(k1c)?;
                let (k1b, k2b): ([u8; crate::SEED_SIZE], [u8; crate::SEED_SIZE]) =
                    network.recv_prev()?;
                let bitcomp1 = Rep3RandBitComp::new_3keys(k1a, k1b, k1c);
                let bitcomp2 = Rep3RandBitComp::new_3keys(k2a, k2b, k2c);
                Ok((bitcomp1, bitcomp2))
            }
            PartyID::ID1 => {
                network.send_next((k1c, k2c))?;
                let k1b: [u8; crate::SEED_SIZE] = network.recv_prev()?;
                let bitcomp1 = Rep3RandBitComp::new_3keys(k1a, k1b, k1c);
                let bitcomp2 = Rep3RandBitComp::new_2keys(k2a, k2c);
                Ok((bitcomp1, bitcomp2))
            }
            PartyID::ID2 => {
                network.send_next((k1c, k2c))?;
                let (k1b, k2b): ([u8; crate::SEED_SIZE], [u8; crate::SEED_SIZE]) =
                    network.recv_prev()?;
                let bitcomp1 = Rep3RandBitComp::new_3keys(k1a, k1b, k1c);
                let bitcomp2 = Rep3RandBitComp::new_3keys(k2a, k2b, k2c);
                Ok((bitcomp1, bitcomp2))
            }
        }
    }

    /// Construct  a new [`IoContext`] with the given network
    pub fn init(mut network: N) -> IoResult<Self> {
        let mut rng = RngType::from_entropy();
        let mut rand = Self::setup_prf(&mut network, &mut rng)?;
        let bitcomps = Self::setup_bitcomp(&mut network, &mut rand)?;
        let mut master_rngs = Rep3CorrelatedRng::new(rand, bitcomps.0, bitcomps.1);

        Ok(Self {
            id: network.get_id(), //shorthand access
            network,
            rngs: master_rngs.fork(),
            rng: rng.clone(),
            a2b_type: A2BType::default(),
            rng_src: Arc::new(RngForker::new(rng)),
            rngs_src: Arc::new(RngForker::new(master_rngs)),
        })
    }

    /// Allows to change the used arithmetic/binary conversion protocol
    pub fn set_a2b_type(&mut self, a2b_type: A2BType) {
        self.a2b_type = a2b_type;
    }

    /// Cronstruct a fork of the [`IoContext`]. This fork can be used concurrently with its parent.
    pub fn fork(&self) -> IoResult<Self> {
        let child_rngs = self.rngs_src.fork(); // lock once, derive new RNG
        let child_rng = self.rng_src.fork(); // lock once, derive new RNG

        Ok(IoContext {
            id: self.id,
            rngs: child_rngs,
            rng: child_rng,
            network: self.network.fork(),
            a2b_type: self.a2b_type,
            rng_src: Arc::clone(&self.rng_src),
            rngs_src: Arc::clone(&self.rngs_src),
        })
    }

    /// Generate two random elements
    pub fn random_elements<T>(&mut self) -> (T, T)
    where
        Standard: Distribution<T>,
    {
        self.rngs.rand.random_elements()
    }

    /// Generate two random field elements
    pub fn random_fes<F: PrimeField>(&mut self) -> (F, F) {
        self.rngs.rand.random_fes()
    }

    /// Generate a masking field element
    pub fn masking_field_element<F: PrimeField>(&mut self) -> F {
        let (a, b) = self.random_fes::<F>();
        a - b
    }
}

/// This trait defines the network interface for the REP3 protocol.
#[async_trait]
pub trait Rep3Network: Send + Clone {
    /// Returns the id of the party. The id is in the range 0 <= id < 3
    fn get_id(&self) -> PartyID;

    /// Sends `data` to the next party and receives from the previous party. Use this whenever
    /// possible in contrast to calling [`Self::send_next()`] and [`Self::recv_prev()`] sequential. This method
    /// executes send/receive concurrently.
    fn reshare<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: F,
    ) -> std::io::Result<F> {
        let mut res = self.reshare_many(&[data])?;
        if res.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            //we checked that there is really one element
            Ok(res.pop().unwrap())
        }
    }

    async fn reshare_async<F: CanonicalSerialize + CanonicalDeserialize + Send>(
        &mut self,
        data: F,
    ) -> std::io::Result<F> {
        let mut res = self.reshare_many_async(vec![data]).await?;
        if res.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            Ok(res.pop().unwrap())
        }
    }

    /// Perform multiple reshares with one networking round
    fn reshare_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<Vec<F>>;

    async fn reshare_many_async<F: CanonicalSerialize + CanonicalDeserialize + Send>(
        &mut self,
        data: Vec<F>,
    ) -> std::io::Result<Vec<F>>;

    /// Broadcast data to the other two parties and receive data from them
    fn broadcast<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: F,
    ) -> std::io::Result<(F, F)> {
        let (mut prev, mut next) = self.broadcast_many(&[data])?;
        if prev.len() != 1 || next.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            //we checked that there is really one element
            let prev = prev.pop().unwrap();
            let next = next.pop().unwrap();
            Ok((prev, next))
        }
    }

    /// Broadcast data to the other two parties and receive data from them
    fn broadcast_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<(Vec<F>, Vec<F>)>;

    /// Sends data to the target party. This function has a default implementation for calling [Rep3Network::send_many].
    fn send<F: CanonicalSerialize>(&mut self, target: PartyID, data: F) -> std::io::Result<()> {
        self.send_many(target, &[data])
    }

    /// Sends data to the target party. This function has a default implementation for calling [Rep3Network::send_many].
    async fn send_async<F: CanonicalSerialize + Send>(
        &mut self,
        target: PartyID,
        data: F,
    ) -> std::io::Result<()> {
        self.send_many_async(target, vec![data]).await
    }

    /// Sends a vector of data to the target party.
    fn send_many<F: CanonicalSerialize>(
        &mut self,
        target: PartyID,
        data: &[F],
    ) -> std::io::Result<()>;

    /// Sends a vector of data to the target party.
    async fn send_many_async<F: CanonicalSerialize + Send>(
        &mut self,
        target: PartyID,
        data: Vec<F>,
    ) -> std::io::Result<()>;

    /// Sends data to the party with id = next_id (i.e., my_id + 1 mod 3). This function has a default implementation for calling [Rep3Network::send] with the next_id.
    fn send_next<F: CanonicalSerialize>(&mut self, data: F) -> std::io::Result<()> {
        self.send(self.get_id().next_id(), data)
    }

    async fn send_next_async<F: CanonicalSerialize + Send>(
        &mut self,
        data: F,
    ) -> std::io::Result<()> {
        self.send_async(self.get_id().next_id(), data).await
    }

    /// Sends a vector data to the party with id = next_id (i.e., my_id + 1 mod 3). This function has a default implementation for calling [Rep3Network::send_many] with the next_id.
    fn send_next_many<F: CanonicalSerialize>(&mut self, data: &[F]) -> std::io::Result<()> {
        self.send_many(self.get_id().next_id(), data)
    }

    async fn send_next_many_async<F: CanonicalSerialize + Send>(
        &mut self,
        data: Vec<F>,
    ) -> std::io::Result<()> {
        self.send_many_async(self.get_id().next_id(), data).await
    }

    /// Receives data from the party with the given id. This function has a default implementation for calling [Rep3Network::recv_many] and checking for the correct length of 1.
    fn recv<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<F> {
        let mut res = self.recv_many(from)?;
        if res.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            Ok(res.pop().unwrap())
        }
    }

    /// Receives data from the party with the given id. This function has a default implementation for calling [Rep3Network::recv_many] and checking for the correct length of 1.
    async fn recv_async<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<F> {
        let mut res = self.recv_many_async(from).await?;
        if res.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            Ok(res.pop().unwrap())
        }
    }

    /// Receives a vector of data from the party with the given id.
    fn recv_many<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<Vec<F>>;

    /// Receives a vector of data from the party with the given id.
    async fn recv_many_async<F: CanonicalDeserialize>(
        &mut self,
        from: PartyID,
    ) -> std::io::Result<Vec<F>>;

    /// Receives data from the party with the id = prev_id (i.e., my_id + 2 mod 3). This function has a default implementation for calling [Rep3Network::recv] with the prev_id.
    fn recv_prev<F: CanonicalDeserialize>(&mut self) -> std::io::Result<F> {
        self.recv(self.get_id().prev_id())
    }

    async fn recv_prev_async<F: CanonicalDeserialize>(&mut self) -> std::io::Result<F> {
        self.recv_async(self.get_id().prev_id()).await
    }

    /// Receives a vector of data from the party with the id = prev_id (i.e., my_id + 2 mod 3). This function has a default implementation for calling [Rep3Network::recv_many] with the prev_id.
    fn recv_prev_many<F: CanonicalDeserialize>(&mut self) -> std::io::Result<Vec<F>> {
        self.recv_many(self.get_id().prev_id())
    }

    async fn recv_prev_many_async<F: CanonicalDeserialize>(&mut self) -> std::io::Result<Vec<F>> {
        self.recv_many_async(self.get_id().prev_id()).await
    }

    /// Fork the network into two separate instances with their own connections
    fn fork(&self) -> Self
    where
        Self: Sized;
}

pub type Rep3MpcNet = mpc_net::rep3::quic::Rep3QuicMpcNetWorker;

#[async_trait]
impl Rep3Network for Rep3MpcNet {
    fn get_id(&self) -> PartyID {
        self.id.party_id()
    }

    fn reshare_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<Vec<F>> {
        self.send_many(self.get_id().next_id(), data)?;
        self.recv_many(self.get_id().prev_id())
    }

    async fn reshare_many_async<F: CanonicalSerialize + CanonicalDeserialize + Send>(
        &mut self,
        data: Vec<F>,
    ) -> std::io::Result<Vec<F>> {
        self.send_many_async(self.get_id().next_id(), data).await?;
        self.recv_many_async(self.get_id().prev_id()).await
    }

    fn broadcast_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<(Vec<F>, Vec<F>)> {
        self.send_many(self.get_id().next_id(), data)?;
        self.send_many(self.get_id().prev_id(), data)?;
        let recv_next = self.recv_many(self.get_id().next_id())?;
        let recv_prev = self.recv_many(self.get_id().prev_id())?;
        Ok((recv_prev, recv_next))
    }

    fn send_many<F: CanonicalSerialize>(
        &mut self,
        target: PartyID,
        data: &[F],
    ) -> std::io::Result<()> {
        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        self.send_bytes(target, Bytes::from(ser_data))
    }

    async fn send_many_async<F: CanonicalSerialize + Send>(
        &mut self,
        target: PartyID,
        data: Vec<F>,
    ) -> std::io::Result<()> {
        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        self.send_bytes_async(target, Bytes::from(ser_data)).await
    }

    fn recv_many<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<Vec<F>> {
        let data = self.recv_bytes(from)?;
        let len = data.len();

        let res = Vec::<F>::deserialize_uncompressed_unchecked(&data[..]).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "to {} from {} error: {e}: got {} bytes type {}",
                    self.id.party_id(),
                    from,
                    len,
                    std::any::type_name::<F>()
                ),
            )
        })?;

        Ok(res)
    }

    async fn recv_many_async<F: CanonicalDeserialize>(
        &mut self,
        from: PartyID,
    ) -> std::io::Result<Vec<F>> {
        let data = self.recv_bytes_async(from).await?;

        let res = Vec::<F>::deserialize_uncompressed_unchecked(&data[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(res)
    }

    fn fork(&self) -> Self {
        MpcStarNetWorker::fork(&self)
    }
}

pub trait Rep3NetworkWorker: Rep3Network + MpcStarNetWorker + 'static {
    fn exchange<Req, Resp>(&mut self, request: Req) -> eyre::Result<Resp>
    where
        Req: CanonicalSerialize + CanonicalDeserialize,
        Resp: CanonicalSerialize + CanonicalDeserialize,
    {
        self.send_response(request).context("send_response")?;
        self.receive_request().context("receive_request")
    }
}
pub trait Rep3NetworkCoordinator: MpcStarNetCoordinator + 'static {
    fn sync_with_parties(&mut self) -> eyre::Result<()>;
}

impl Rep3NetworkWorker for Rep3QuicMpcNetWorker {}

impl Rep3NetworkCoordinator for Rep3QuicNetCoordinator {
    #[tracing::instrument(skip_all, name = "sync_with_parties", level = "trace")]
    fn sync_with_parties(&mut self) -> eyre::Result<()> {
        self.broadcast_request(true)?;
        self.receive_responses::<bool>()?;
        Ok(())
    }
}

pub struct IoContextPool<Network: Rep3NetworkWorker> {
    pub worker_id: usize,
    main: IoContext<Network>,
    forks: Vec<IoContext<Network>>,
    num_workers: usize,
}

impl<Network: Rep3NetworkWorker> IoContextPool<Network> {
    pub fn init(network: Network, num_forks: u32) -> eyre::Result<Self> {
        let worker_id = network.worker_idx();
        let num_workers = 1 << network.log_num_workers();
        let main = IoContext::init(network)?;

        let forks = iter::repeat_with(|| main.fork())
            .take(num_forks as usize)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            worker_id,
            main,
            forks,
            num_workers,
        })
    }

    pub fn main(&mut self) -> &mut IoContext<Network> {
        &mut self.main
    }

    pub fn network(&mut self) -> &mut Network {
        &mut self.main.network
    }

    pub fn fork(&mut self) -> eyre::Result<IoContext<Network>> {
        self.main.fork().context("while forking io context")
    }

    /// Create a sub-pool by forking a new main context and `num_forks` children.
    pub fn fork_pool(&mut self, num_forks: usize) -> eyre::Result<Self> {
        let main = self.main.fork().context("while forking pool main")?;
        let forks = iter::repeat_with(|| main.fork())
            .take(num_forks)
            .collect::<Result<Vec<_>, _>>()
            .context("while forking pool children")?;
        Ok(Self {
            worker_id: self.worker_id,
            main,
            forks,
            num_workers: self.num_workers,
        })
    }

    pub fn log_num_workers(&self) -> usize {
        self.main.network.log_num_workers()
    }

    pub fn num_workers(&self) -> usize {
        1 << self.log_num_workers()
    }

    pub fn party_id(&self) -> PartyID {
        self.main.id
    }

    pub fn party_idx(&self) -> usize {
        self.party_id().into()
    }

    pub fn forks(&mut self, num_forks: usize) -> &mut [IoContext<Network>] {
        &mut self.forks[..num_forks]
    }

    pub fn forks_owned(&mut self, num_forks: usize) -> Vec<IoContext<Network>> {
        self.forks[..num_forks].to_vec()
    }

    pub fn worker_idx(&self) -> usize {
        self.worker_id
    }

    pub fn total_num_workers(&self) -> usize {
        self.num_workers
    }

    /// Parallelize the computation of `map` over the `inputs` using the forked `IoContext`s.
    ///
    /// The `inputs` are split into chunks, and each chunk is processed in parallel using the `forks`.
    ///
    /// The `map` is a function that takes an input and an `IoContext` and returns a flattened result.
    ///
    /// The `max_forks` is the maximum number of forks to use.
    /// If `None`, all forks are used. (Default: rayon::current_num_threads() / num_workers)
    pub fn par_iter<T, R, MapFn>(
        &mut self,
        inputs: impl IntoParallelIterator<Item = T, Iter: IndexedParallelIterator>,
        max_forks: Option<usize>,
        map: MapFn,
    ) -> eyre::Result<Vec<R>>
    where
        MapFn: Fn(T, &mut IoContext<Network>) -> eyre::Result<R> + Sync + Send,
        T: Sized + Send + Clone,
        R: Sync + Send,
    {
        let inputs_iter = inputs.into_par_iter();
        let max_forks = max_forks.unwrap_or(self.forks.len());
        let len = inputs_iter.len();

        if max_forks == 0 {
            return inputs_iter
                .collect::<Vec<_>>()
                .into_iter()
                .map(|val| map(val, self.main()))
                .collect::<eyre::Result<Vec<_>>>();
        }

        if len == 1 {
            return Ok(vec![map(
                inputs_iter.collect::<Vec<_>>().pop().unwrap(),
                self.main(),
            )?]);
        }

        let chunk_size = len.div_ceil(max_forks);
        assert!(chunk_size != 0);
        let forks = len.div_ceil(chunk_size);

        inputs_iter
            .into_par_iter()
            .chunks(chunk_size)
            .zip_eq(self.forks(forks).par_iter_mut())
            .map(|(chunk, mut ctx)| {
                chunk
                    .into_iter()
                    .map(|val| map(val, &mut ctx))
                    .collect_vec()
            })
            .flatten()
            .collect::<eyre::Result<Vec<_>>>()
    }

    /// Network parallel **cyclic** (aka round-robin) iterator map with deterministic fork assignment.
    /// - Fork `f` handles indices `f, f+N, f+2N,…` where `N = min(self.forks.len(), inputs.len())`.
    /// - Each fork owns an exclusive `IoContext`; output order matches input order.
    /// - If `N==0` or `len==1`, runs on `self.main()`.
    ///
    /// Bounds: `T: Send+Sync+Clone`, `R: Send+Sync`,
    /// `map: Fn(T, &mut IoContext<Network>) -> eyre::Result<R> + Send+Sync`.
    pub fn par_iter_cyclic<T, R, MapFn>(
        &mut self,
        inputs: impl IntoIterator<Item = T>,
        map: MapFn,
    ) -> eyre::Result<Vec<R>>
    where
        MapFn: Fn(T, &mut IoContext<Network>) -> eyre::Result<R> + Sync + Send,
        T: Send + Sync + Clone,
        R: Send + Sync,
    {
        let items: Vec<T> = inputs.into_iter().collect();
        let m = items.len();
        if m == 0 {
            return Ok(Vec::new());
        }

        // use up to m forks
        let forks = self.forks.len().min(m);
        if forks == 0 {
            return items.into_iter().map(|v| map(v, self.main())).collect();
        }
        if m == 1 {
            return Ok(vec![map(items[0].clone(), self.main())?]);
        }

        let results: Vec<OnceLock<eyre::Result<R>>> = (0..m).map(|_| OnceLock::new()).collect();

        // N parallel tasks; fork f processes indices f, f+N, f+2N, ...
        (0..forks)
            .into_par_iter()
            .zip(self.forks(forks).par_iter_mut())
            .for_each(|(f, mut ctx)| {
                for i in (f..m).step_by(forks) {
                    let r = map(items[i].clone(), &mut ctx);
                    let _ = results[i].set(r);
                }
            });

        results
            .into_par_iter()
            .map(|cell| cell.into_inner().expect("missing result"))
            .collect::<Result<Vec<_>, _>>()
    }

    /// Parallelize the computation of `map` over the `inputs` using the forked `IoContext`s.
    ///
    /// The `inputs` are split into chunks, and each chunk is processed in parallel using the `forks`.
    ///
    /// The `map` is a function that takes a chunk of inputs and an `IoContext` and returns a flattened result.
    ///
    /// The `chunk_size` is the size of each chunk.
    /// If `None`, the chunk size is the number of inputs divided by the number of available forks.
    pub fn par_chunks<T, R, MapFn, Err>(
        &mut self,
        inputs: impl IntoParallelIterator<Item = T, Iter: IndexedParallelIterator>,
        chunk_size: Option<usize>,
        map: MapFn,
    ) -> eyre::Result<Vec<R>>
    where
        MapFn: Fn(Vec<T>, &mut IoContext<Network>) -> Result<Vec<R>, Err> + Sync + Send,
        T: Sized + Send,
        R: Sync + Send + Clone,
        eyre::Report: From<Err>,
        Err: Send + Sync,
    {
        let inputs_iter = inputs.into_par_iter();
        let len = inputs_iter.len();

        if self.forks.len() == 0 {
            return Ok(map(inputs_iter.collect(), self.main())?);
        }

        let chunk_size = chunk_size.unwrap_or(len.div_ceil(self.forks.len()));
        assert!(chunk_size != 0);
        if len <= chunk_size {
            return Ok(map(inputs_iter.collect(), self.main())?);
        }
        let forks = len.div_ceil(chunk_size);

        inputs_iter
            .into_par_iter()
            .chunks(chunk_size)
            .zip_eq(self.forks(forks).par_iter_mut())
            .flat_map(|(chunk, mut ctx)| match map(chunk, &mut ctx) {
                Ok(result) => Either::Left(result.into_par_iter().map(|r| eyre::Ok(r))),
                Err(err) => Either::Right(rayon::iter::once(Err(eyre::Error::from(err)))),
            })
            .collect::<Result<Vec<_>, _>>()
    }

    // /// Deterministic assignment based on input values:
    // /// stable sort by `inputs[i]` (tie-break by index), then round-robin to forks.
    // pub fn par_iter_range<R, E, MapFn>(
    //     &mut self,
    //     range: Vec<usize>,
    //     map_fn: MapFn,
    // ) -> Result<Vec<R>, E>
    // where
    //     R: Send + Sync,
    //     E: Send + Sync,
    //     MapFn: Fn(usize, &mut IoContext<Network>) -> Result<R, E> + Sync,
    // {
    //     let n = self.forks.len();
    //     assert!(n > 0);
    //     let m = range.len();

    //     // stable order (by value, tie by index)
    //     let order: Vec<usize> = (0..m).collect();

    //     // deterministic RR split: fork f gets i=f, f+n, f+2n, …
    //     let per_fork: Vec<Vec<usize>> = (0..n)
    //         .map(|f| order.iter().skip(f).step_by(n).copied().collect())
    //         .collect();

    //     // take contexts
    //     let ctxs = self.forks_owned(n); // Vec<IoContext<Network>>

    //     // results without locks on hot path
    //     let slots: Vec<OnceLock<Result<R, E>>> = (0..m).map(|_| OnceLock::new()).collect();
    //     let map_ref = &map_fn;

    //     per_fork
    //         .into_par_iter() // N parallel tasks
    //         .zip(ctxs.into_par_iter()) // pair each with its ctx
    //         .for_each(|(idxs, mut ctx)| {
    //             for i in idxs {
    //                 let r = map_ref(range[i], &mut ctx);
    //                 let _ = slots[i].set(r);
    //             }
    //         });

    //     // gather in original order
    //     let mut out = Vec::with_capacity(m);
    //     for cell in slots {
    //         match cell.into_inner().expect("missing result") {
    //             Ok(v) => out.push(v),
    //             Err(e) => return Err(e),
    //         }
    //     }
    //     Ok(out)
    // }

    #[tracing::instrument(skip_all, name = "sync_with_parties", level = "trace")]
    pub fn sync_with_parties(&mut self) -> eyre::Result<()> {
        self.main().network.send_next(true)?;
        self.main().network.recv_prev::<bool>()?;
        Ok(())
    }

    #[tracing::instrument(skip_all, name = "sync_with_coordinator", level = "trace")]
    pub fn sync_with_coordinator(&mut self) -> eyre::Result<()> {
        self.main().network.receive_request::<bool>()?;
        self.main().network.send_response(true)?;
        Ok(())
    }
    /// Prints the connection statistics.
    pub fn log_connection_stats(&self) {
        let acc_io = self
            .forks
            .iter()
            .map(|io_ctx| io_ctx.network.io_stats_per_party())
            .fold(self.main.network.io_stats_per_party(), |mut acc, stats| {
                acc.iter_mut().for_each(|(id, (tx, rx))| {
                    let (tx_, rx_) = stats.get(id).unwrap();
                    *tx += tx_;
                    *rx += rx_;
                });
                acc
            });
        for (i, (tx, rx)) in acc_io {
            tracing::info!(
                "IO: P{}->P{} | SENT: {} bytes | RECV: {} bytes",
                self.party_idx(),
                i,
                ByteSize(tx),
                ByteSize(rx)
            );
        }
    }
}
