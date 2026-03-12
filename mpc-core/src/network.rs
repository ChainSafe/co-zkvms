//! Rep3 Network
//!
//! This module contains implementation of the rep3 mpc network

use crate::field::PrimeField;
use crate::protocols::rep3_ring::Rep3RingShare;
use crate::protocols::rep3_ring::ring::bit::Bit;
use crate::protocols::rep3_ring::ring::int_ring::IntRing2k;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::Bytes;
use bytesize::ByteSize;
use eyre::Context;
use std::io;
use std::iter;
use std::mem::MaybeUninit;
use std::slice;
use std::sync::{Arc, OnceLock};

use crate::preprocessing::backing_store::assert_field_layout;
use crate::protocols::rep3_ring::dabits::DaBitBatch;
use crate::protocols::rep3_ring::edabits::EdaBitsBatch;

use itertools::Itertools;
use mpc_net::topology::{MpcStarNetCoordinator, MpcStarNetWorker};

use mpc_net::rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator};
use rand::{CryptoRng, Rng, SeedableRng, distributions::Standard, prelude::Distribution};

use crate::protocols::rep3::rngs::RngForker;
use crate::{IoResult, RngType};
pub use mpc_net::id::PartyID;

use rayon::iter::Either;
use rayon::prelude::*;

use crate::protocols::rep3::{
    conversion::MPCType,
    rngs::{Rep3CorrelatedRng, Rep3Rand, Rep3RandBitComp},
};

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
    /// Online or preprocessed MPC execution mode
    pub mpc_type: MPCType,

    rng_src: Arc<RngForker<RngType>>,
    rngs_src: Arc<RngForker<Rep3CorrelatedRng>>,
}

// NOTE: IoContext intentionally does NOT implement Clone.
// Clone on Rep3Network shares streams (mpsc::Sender clone), which would cause
// message interleaving if two clones are used concurrently.
// Use IoContext::fork() to create an independent context with its own streams.

impl<N: Rep3Network> IoContext<N> {
    fn setup_prf<R: Rng + CryptoRng>(network: &mut N, rng: &mut R) -> IoResult<Rep3Rand> {
        let seed1: [u8; crate::SEED_SIZE] = rng.r#gen();
        network.send_next(seed1)?;
        let seed2: [u8; crate::SEED_SIZE] = network.recv_prev()?;

        Ok(Rep3Rand::new(seed1, seed2))
    }

    fn setup_bitcomp(network: &mut N, rands: &mut Rep3Rand) -> IoResult<(Rep3RandBitComp, Rep3RandBitComp)> {
        let (k1a, k1c) = rands.random_seeds();
        let (k2a, k2c) = rands.random_seeds();
        match network.get_id() {
            PartyID::ID0 => {
                network.send_next(k1c)?;
                let (k1b, k2b): ([u8; crate::SEED_SIZE], [u8; crate::SEED_SIZE]) = network.recv_prev()?;
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
                let (k1b, k2b): ([u8; crate::SEED_SIZE], [u8; crate::SEED_SIZE]) = network.recv_prev()?;
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
            mpc_type: MPCType::default(),
            rng_src: Arc::new(RngForker::new(rng)),
            rngs_src: Arc::new(RngForker::new(master_rngs)),
        })
    }

    /// Set the MPC execution mode (online or preprocessed).
    pub fn set_mpc_type(&mut self, mpc_type: MPCType) {
        self.mpc_type = mpc_type;
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
            mpc_type: self.mpc_type,
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
pub trait Rep3Network: Send + Clone {
    /// Returns the id of the party. The id is in the range 0 <= id < 3
    fn get_id(&self) -> PartyID;

    /// Sends `data` to the next party and receives from the previous party. Use this whenever
    /// possible in contrast to calling [`Self::send_next()`] and [`Self::recv_prev()`] sequential. This method
    /// executes send/receive concurrently.
    fn reshare<F: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: F) -> std::io::Result<F> {
        let mut res = self.reshare_many(&[data])?;
        if res.len() != 1 {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Expected 1 element, got more"))
        } else {
            //we checked that there is really one element
            Ok(res.pop().unwrap())
        }
    }

    /// Perform multiple reshares with one networking round
    fn reshare_many<F: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: &[F]) -> std::io::Result<Vec<F>>;

    /// Broadcast data to the other two parties and receive data from them
    fn broadcast<F: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: F) -> std::io::Result<(F, F)> {
        let (mut prev, mut next) = self.broadcast_many(&[data])?;
        if prev.len() != 1 || next.len() != 1 {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Expected 1 element, got more"))
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

    /// Sends a vector of data to the target party.
    fn send_many<F: CanonicalSerialize>(&mut self, target: PartyID, data: &[F]) -> std::io::Result<()>;

    /// Sends data to the party with id = next_id (i.e., my_id + 1 mod 3). This function has a default implementation for calling [Rep3Network::send] with the next_id.
    fn send_next<F: CanonicalSerialize>(&mut self, data: F) -> std::io::Result<()> {
        self.send(self.get_id().next_id(), data)
    }

    /// Sends a vector data to the party with id = next_id (i.e., my_id + 1 mod 3). This function has a default implementation for calling [Rep3Network::send_many] with the next_id.
    fn send_next_many<F: CanonicalSerialize>(&mut self, data: &[F]) -> std::io::Result<()> {
        self.send_many(self.get_id().next_id(), data)
    }

    /// Receives data from the party with the given id. This function has a default implementation for calling [Rep3Network::recv_many] and checking for the correct length of 1.
    fn recv<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<F> {
        let mut res = self.recv_many(from)?;
        if res.len() != 1 {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Expected 1 element, got more"))
        } else {
            Ok(res.pop().unwrap())
        }
    }

    /// Receives a vector of data from the party with the given id.
    fn recv_many<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<Vec<F>>;

    /// Receives data from the party with the id = prev_id (i.e., my_id + 2 mod 3). This function has a default implementation for calling [Rep3Network::recv] with the prev_id.
    fn recv_prev<F: CanonicalDeserialize>(&mut self) -> std::io::Result<F> {
        self.recv(self.get_id().prev_id())
    }

    /// Receives a vector of data from the party with the id = prev_id (i.e., my_id + 2 mod 3). This function has a default implementation for calling [Rep3Network::recv_many] with the prev_id.
    fn recv_prev_many<F: CanonicalDeserialize>(&mut self) -> std::io::Result<Vec<F>> {
        self.recv_many(self.get_id().prev_id())
    }

    /// Fork the network into a new independent instance.
    /// Note: for QUIC transport, this opens new bidi streams on existing connections
    /// and spawns tokio tasks — it is not free.
    fn fork(&self) -> Self
    where
        Self: Sized;
}

pub type Rep3MpcNet = mpc_net::rep3::quic::Rep3QuicMpcNetWorker;

impl Rep3Network for Rep3MpcNet {
    fn get_id(&self) -> PartyID {
        self.id.party_id()
    }

    fn reshare_many<F: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: &[F]) -> std::io::Result<Vec<F>> {
        self.send_many(self.get_id().next_id(), data)?;
        self.recv_many(self.get_id().prev_id())
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

    fn send_many<F: CanonicalSerialize>(&mut self, target: PartyID, data: &[F]) -> std::io::Result<()> {
        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        self.send_bytes(target, Bytes::from(ser_data))
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

pub trait Rep3RawFieldTransport {
    fn send_field_slice_raw<F: PrimeField>(&mut self, target: PartyID, data: &[F]) -> io::Result<()>;

    fn send_field_vec_raw<F: PrimeField>(&mut self, target: PartyID, data: Vec<F>) -> io::Result<()> {
        self.send_field_slice_raw(target, &data)
    }

    fn recv_field_bytes_raw<F: PrimeField>(&mut self, from: PartyID, elems: usize) -> io::Result<Vec<u8>>;

    fn recv_field_vec_raw<F: PrimeField>(&mut self, from: PartyID, elems: usize) -> io::Result<Vec<F>> {
        let bytes = self.recv_field_bytes_raw::<F>(from, elems)?;
        field_vec_from_bytes::<F>(&bytes)
    }

    fn recv_field_bytes_bulk_into<F: PrimeField>(&mut self, from: PartyID, dst: &mut [u8]) -> io::Result<()> {
        let elem_size = std::mem::size_of::<F>();
        if dst.len() % elem_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "recv_field_bytes_bulk_into: buffer length {} not divisible by element size {}",
                    dst.len(),
                    elem_size,
                ),
            ));
        }
        let elems = dst.len() / elem_size;
        let bytes = self.recv_field_bytes_raw::<F>(from, elems)?;
        dst.copy_from_slice(&bytes);
        Ok(())
    }

    fn recv_field_vec_raw_owned<F: PrimeField>(&mut self, from: PartyID, elems: usize) -> io::Result<Vec<F>> {
        self.recv_field_vec_raw(from, elems)
    }
}

fn field_vec_from_bytes<F: PrimeField>(bytes: &[u8]) -> io::Result<Vec<F>> {
    const { assert_field_layout::<F>() };
    let elem_size = std::mem::size_of::<F>();
    if bytes.len() % elem_size != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "raw field payload length {} is not divisible by element size {} for {}",
                bytes.len(),
                elem_size,
                std::any::type_name::<F>()
            ),
        ));
    }
    let elems = bytes.len() / elem_size;
    if elems == 0 {
        return Ok(Vec::new());
    }

    let mut out: Vec<MaybeUninit<F>> = Vec::with_capacity(elems);
    // SAFETY: every element is fully initialized by the byte copy below.
    unsafe { out.set_len(elems) };
    let out_bytes = unsafe { slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, bytes.len()) };
    out_bytes.copy_from_slice(bytes);
    let out: Vec<F> = unsafe { std::mem::transmute(out) };
    Ok(out)
}

fn field_slice_to_bytes<F: PrimeField>(data: &[F]) -> &[u8] {
    const { assert_field_layout::<F>() };
    unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

fn field_vec_into_bytes<F: PrimeField>(data: Vec<F>) -> Vec<u8> {
    const { assert_field_layout::<F>() };
    let mut data = std::mem::ManuallyDrop::new(data);
    let len = data.len().saturating_mul(std::mem::size_of::<F>());
    let cap = data.capacity().saturating_mul(std::mem::size_of::<F>());
    unsafe { Vec::from_raw_parts(data.as_mut_ptr() as *mut u8, len, cap) }
}

impl Rep3NetworkWorker for Rep3QuicMpcNetWorker {}

impl Rep3RawFieldTransport for Rep3QuicMpcNetWorker {
    fn send_field_slice_raw<F: PrimeField>(&mut self, target: PartyID, data: &[F]) -> io::Result<()> {
        self.send_bytes_bulk(target, Bytes::copy_from_slice(field_slice_to_bytes(data)))
    }

    fn send_field_vec_raw<F: PrimeField>(&mut self, target: PartyID, data: Vec<F>) -> io::Result<()> {
        self.send_bytes_bulk(target, Bytes::from(field_vec_into_bytes(data)))
    }

    fn recv_field_bytes_raw<F: PrimeField>(&mut self, from: PartyID, elems: usize) -> io::Result<Vec<u8>> {
        let bytes = self.recv_bytes_bulk(from)?;
        let expected = elems.saturating_mul(std::mem::size_of::<F>());
        if bytes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "raw field payload size mismatch from {:?}: expected {} bytes ({} elems of {}), got {}",
                    from,
                    expected,
                    elems,
                    std::any::type_name::<F>(),
                    bytes.len()
                ),
            ));
        }
        Ok(bytes)
    }

    fn recv_field_bytes_bulk_into<F: PrimeField>(&mut self, from: PartyID, dst: &mut [u8]) -> io::Result<()> {
        self.recv_bytes_bulk_into(from, dst)
    }

    fn recv_field_vec_raw_owned<F: PrimeField>(&mut self, from: PartyID, elems: usize) -> io::Result<Vec<F>> {
        const { assert_field_layout::<F>() };
        let mut out: Vec<MaybeUninit<F>> = Vec::with_capacity(elems);
        unsafe { out.set_len(elems) };
        let out_bytes = unsafe {
            slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, elems.saturating_mul(std::mem::size_of::<F>()))
        };
        self.recv_bytes_bulk_into(from, out_bytes)?;
        Ok(unsafe { std::mem::transmute(out) })
    }
}

impl Rep3NetworkCoordinator for Rep3QuicNetCoordinator {
    #[tracing::instrument(skip_all, name = "sync_with_parties", level = "trace")]
    fn sync_with_parties(&mut self) -> eyre::Result<()> {
        self.broadcast_request(true)?;
        self.receive_responses::<bool>()?;
        Ok(())
    }
}

pub struct IoContextPool<Network: Rep3NetworkWorker> {
    worker_id: usize,
    main: IoContext<Network>,
    forks: Vec<IoContext<Network>>,
    num_workers: usize,
}

impl<Network: Rep3NetworkWorker> IoContextPool<Network> {
    pub fn init(network: Network, num_forks: u32) -> eyre::Result<Self> {
        let worker_id = network.worker_idx();
        let num_workers = 1 << network.log_num_workers();
        let main = IoContext::init(network)?;

        let forks = iter::repeat_with(|| main.fork()).take(num_forks as usize).collect::<Result<Vec<_>, _>>()?;

        Ok(Self { worker_id, main, forks, num_workers })
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
        Ok(Self { worker_id: self.worker_id, main, forks, num_workers: self.num_workers })
    }

    /// Drop all forks and re-create with `num_forks` children.
    /// New forks inherit current env settings (e.g. `MPC_FORK_BULK_CHANNELS`).
    pub fn reconfigure(&mut self, num_forks: u32) -> eyre::Result<()> {
        self.forks.clear();
        self.forks = iter::repeat_with(|| self.main.fork())
            .take(num_forks as usize)
            .collect::<Result<Vec<_>, _>>()
            .context("while reconfiguring fork pool")?;
        Ok(())
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

    pub fn max_forks(&self) -> usize {
        self.forks.len()
    }

    pub fn forks(&mut self, num_forks: usize) -> &mut [IoContext<Network>] {
        &mut self.forks[..num_forks]
    }

    /// Drain `n` forks from the pool, transferring ownership to the caller.
    pub fn take_forks(&mut self, n: usize) -> Vec<IoContext<Network>> {
        self.forks.drain(..n).collect()
    }

    /// Return previously taken forks back to the pool.
    pub fn return_forks(&mut self, forks: Vec<IoContext<Network>>) {
        self.forks.extend(forks);
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
            return Ok(vec![map(inputs_iter.collect::<Vec<_>>().pop().unwrap(), self.main())?]);
        }

        let chunk_size = len.div_ceil(max_forks);
        assert!(chunk_size != 0);
        let forks = len.div_ceil(chunk_size);

        inputs_iter
            .into_par_iter()
            .chunks(chunk_size)
            .zip_eq(self.forks(forks).par_iter_mut())
            .map(|(chunk, mut ctx)| chunk.into_iter().map(|val| map(val, &mut ctx)).collect_vec())
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
        (0..forks).into_par_iter().zip(self.forks(forks).par_iter_mut()).for_each(|(f, mut ctx)| {
            for i in (f..m).step_by(forks) {
                let r = map(items[i].clone(), &mut ctx);
                let _ = results[i].set(r);
            }
        });

        results.into_par_iter().map(|cell| cell.into_inner().expect("missing result")).collect::<Result<Vec<_>, _>>()
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

    /// Like `par_chunks` but also splits an `EdaBitsBatch` in lockstep with inputs.
    /// Each fork receives a sub-batch with matching gammas (1:1 with inputs) and
    /// alphas_flat (K:1 with inputs, where K = T::K bits per ring element).
    // NOTE: Chunking boilerplate is intentionally duplicated from `par_chunks` because
    // abstracting the batch co-splitting across EdaBitsBatch/DaBitBatch would add
    // trait complexity for minimal gain.
    pub fn par_chunks_preproc<T, R, F, MapFn, Err>(
        &mut self,
        inputs: Vec<Rep3RingShare<T>>,
        batch: EdaBitsBatch<T, F>,
        chunk_size: Option<usize>,
        map: MapFn,
    ) -> eyre::Result<Vec<R>>
    where
        T: IntRing2k + Send + Sync,
        F: PrimeField + Send + Sync,
        MapFn:
            Fn(Vec<Rep3RingShare<T>>, EdaBitsBatch<T, F>, &mut IoContext<Network>) -> Result<Vec<R>, Err> + Sync + Send,
        R: Sync + Send + Clone,
        eyre::Report: From<Err>,
        Err: Send + Sync,
    {
        let len = inputs.len();

        if self.forks.is_empty() || len == 0 {
            return Ok(map(inputs, batch, self.main())?);
        }

        let chunk_size = chunk_size.unwrap_or(len.div_ceil(self.forks.len()));
        assert!(chunk_size != 0);
        if len <= chunk_size {
            return Ok(map(inputs, batch, self.main())?);
        }
        let num_forks = len.div_ceil(chunk_size);
        let k = T::K;

        inputs
            .into_par_iter()
            .chunks(chunk_size)
            .zip_eq(batch.gammas.into_par_iter().chunks(chunk_size))
            .zip_eq(batch.alphas_flat.into_par_iter().chunks(chunk_size * k))
            .zip_eq(self.forks(num_forks).par_iter_mut())
            .flat_map(|(((inputs, gammas), alphas), ctx)| {
                let sub_batch = EdaBitsBatch { gammas, alphas_flat: alphas };
                match map(inputs, sub_batch, ctx) {
                    Ok(r) => Either::Left(r.into_par_iter().map(eyre::Ok)),
                    Err(e) => Either::Right(rayon::iter::once(Err(eyre::Error::from(e)))),
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }

    /// Like `par_chunks` but also splits a `DaBitBatch` in lockstep with inputs.
    /// Each fork receives a sub-batch with matching gammas, thetas, and v_shares
    /// (all 1:1 with inputs).
    pub fn par_chunks_dabits<R, F, MapFn, Err>(
        &mut self,
        inputs: Vec<Rep3RingShare<Bit>>,
        batch: DaBitBatch<F>,
        chunk_size: Option<usize>,
        map: MapFn,
    ) -> eyre::Result<Vec<R>>
    where
        F: PrimeField + Send + Sync,
        MapFn: Fn(Vec<Rep3RingShare<Bit>>, DaBitBatch<F>, &mut IoContext<Network>) -> Result<Vec<R>, Err> + Sync + Send,
        R: Sync + Send + Clone,
        eyre::Report: From<Err>,
        Err: Send + Sync,
    {
        let len = inputs.len();

        if self.forks.is_empty() || len == 0 {
            return Ok(map(inputs, batch, self.main())?);
        }

        let chunk_size = chunk_size.unwrap_or(len.div_ceil(self.forks.len()));
        assert!(chunk_size != 0);
        if len <= chunk_size {
            return Ok(map(inputs, batch, self.main())?);
        }
        let num_forks = len.div_ceil(chunk_size);

        inputs
            .into_par_iter()
            .chunks(chunk_size)
            .zip_eq(batch.gammas.into_par_iter().chunks(chunk_size))
            .zip_eq(batch.thetas.into_par_iter().chunks(chunk_size))
            .zip_eq(batch.v_shares.into_par_iter().chunks(chunk_size))
            .zip_eq(self.forks(num_forks).par_iter_mut())
            .flat_map(|((((inputs, gammas), thetas), v_shares), ctx)| {
                let sub_batch = DaBitBatch { gammas, thetas, v_shares };
                match map(inputs, sub_batch, ctx) {
                    Ok(r) => Either::Left(r.into_par_iter().map(eyre::Ok)),
                    Err(e) => Either::Right(rayon::iter::once(Err(eyre::Error::from(e)))),
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }

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
        let acc_io = self.forks.iter().map(|io_ctx| io_ctx.network.io_stats_per_party()).fold(
            self.main.network.io_stats_per_party(),
            |mut acc, stats| {
                acc.iter_mut().for_each(|(id, (tx, rx))| {
                    let (tx_, rx_) = stats.get(id).unwrap();
                    *tx += tx_;
                    *rx += rx_;
                });
                acc
            },
        );
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
