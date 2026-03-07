//! In-memory Rep3 test network (no OS networking).
//!
//! This is intended for integration tests in downstream crates that want to run
//! coordinator/worker protocols without QUIC / localhost sockets.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use async_trait::async_trait;
use color_eyre::eyre::{Context, Result, eyre};
use mpc_types::field::PrimeField;

use mpc_net::topology::{MpcStarNetCoordinator, MpcStarNetWorker};

use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::network::{
    IoContextPool, Rep3Network, Rep3NetworkCoordinator, Rep3NetworkWorker,
    Rep3RawFieldTransport,
};

fn to_io_err(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, msg.into())
}

fn serialize_uncompressed<T: CanonicalSerialize>(data: &T) -> Result<Vec<u8>> {
    let size = data.uncompressed_size();
    let mut ser = Vec::with_capacity(size);
    data.serialize_uncompressed(&mut ser)
        .map_err(|e| to_io_err(e.to_string()))
        .context("serialize_uncompressed")?;
    Ok(ser)
}

fn deserialize_uncompressed<T: CanonicalDeserialize>(bytes: &[u8]) -> Result<T> {
    Ok(T::deserialize_uncompressed_unchecked(bytes)
        .map_err(|e| to_io_err(e.to_string()))
        .context("deserialize_uncompressed")?)
}

fn serialize_slice_uncompressed<T: CanonicalSerialize>(data: &[T]) -> std::io::Result<Vec<u8>> {
    let size = data.serialized_size(ark_serialize::Compress::No);
    let mut ser = Vec::with_capacity(size);
    data.serialize_uncompressed(&mut ser)
        .map_err(|e| to_io_err(e.to_string()))?;
    Ok(ser)
}

fn deserialize_vec_uncompressed<T: CanonicalDeserialize>(bytes: &[u8]) -> std::io::Result<Vec<T>> {
    Vec::<T>::deserialize_uncompressed_unchecked(bytes).map_err(|e| to_io_err(e.to_string()))
}

#[derive(Clone)]
pub struct LocalRep3TestWorkerNet {
    id: PartyID,
    worker_idx: usize,
    log_num_workers: usize,

    // star (coordinator <-> worker)
    star_req_rx: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
    star_resp_tx: mpsc::Sender<Vec<u8>>,

    // ring (party <-> party) for IoContext::init RNG setup
    ring_txs: [mpsc::Sender<Vec<u8>>; 3],
    ring_rxs: [Arc<Mutex<mpsc::Receiver<Vec<u8>>>>; 3],
}

impl LocalRep3TestWorkerNet {
    fn party_index(id: PartyID) -> usize {
        usize::from(id)
    }

    fn ring_send_bytes(&mut self, target: PartyID, bytes: Vec<u8>) -> std::io::Result<()> {
        let idx = Self::party_index(target);
        self.ring_txs[idx]
            .send(bytes)
            .map_err(|_| to_io_err("ring send failed"))
    }

    fn ring_recv_bytes(&mut self, from: PartyID) -> std::io::Result<Vec<u8>> {
        let idx = Self::party_index(from);
        self.ring_rxs[idx]
            .lock()
            .unwrap()
            .recv()
            .map_err(|_| to_io_err("ring recv failed"))
    }
}

#[async_trait]
impl Rep3Network for LocalRep3TestWorkerNet {
    fn get_id(&self) -> PartyID {
        self.id
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
        let bytes = serialize_slice_uncompressed(data)?;
        self.ring_send_bytes(target, bytes)
    }

    async fn send_many_async<F: CanonicalSerialize + Send>(
        &mut self,
        target: PartyID,
        data: Vec<F>,
    ) -> std::io::Result<()> {
        self.send_many(target, &data)
    }

    fn recv_many<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<Vec<F>> {
        let bytes = self.ring_recv_bytes(from)?;
        deserialize_vec_uncompressed(&bytes)
    }

    async fn recv_many_async<F: CanonicalDeserialize>(
        &mut self,
        from: PartyID,
    ) -> std::io::Result<Vec<F>> {
        self.recv_many(from)
    }

    fn fork(&self) -> Self {
        self.clone()
    }
}

impl MpcStarNetWorker for LocalRep3TestWorkerNet {
    fn send_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: T,
    ) -> Result<()> {
        let bytes = serialize_uncompressed(&data)?;
        self.star_resp_tx
            .send(bytes)
            .map_err(|_| eyre!("star send_response failed"))?;
        Ok(())
    }

    fn receive_request<T: CanonicalSerialize + CanonicalDeserialize>(&mut self) -> Result<T> {
        let bytes = self
            .star_req_rx
            .lock()
            .unwrap()
            .recv()
            .map_err(|_| eyre!("star receive_request failed"))?;
        deserialize_uncompressed(&bytes)
    }

    fn log_num_workers(&self) -> usize {
        self.log_num_workers
    }

    fn set_log_num_workers(&mut self, log_num_workers: usize) {
        self.log_num_workers = log_num_workers;
    }

    fn reset_log_num_workers(&mut self) {
        self.log_num_workers = 0;
    }

    fn io_stats_total(&self) -> (u64, u64) {
        (0, 0)
    }

    fn io_stats_per_party(&self) -> BTreeMap<usize, (u64, u64)> {
        BTreeMap::new()
    }

    fn party_id(&self) -> PartyID {
        self.id
    }

    fn worker_idx(&self) -> usize {
        self.worker_idx
    }

    fn fork(&self) -> Self {
        self.clone()
    }

    fn fork_with_coordinator(&mut self) -> Result<Self> {
        Ok(self.clone())
    }

    fn get_worker_subnets(&self, num_workers: usize) -> Result<Vec<Self>> {
        if num_workers != 1 {
            return Err(eyre!("in-memory net only supports num_workers=1"));
        }
        Ok(vec![self.clone()])
    }
}

impl Rep3NetworkWorker for LocalRep3TestWorkerNet {}

impl Rep3RawFieldTransport for LocalRep3TestWorkerNet {
    fn send_field_slice_raw<F: PrimeField>(
        &mut self,
        target: PartyID,
        data: &[F],
    ) -> std::io::Result<()> {
        use crate::protocols::rep3_ring::preprocessing::backing_store::assert_field_layout;
        const { assert_field_layout::<F>() };
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.ring_send_bytes(target, bytes.to_vec())
    }

    fn recv_field_bytes_raw<F: PrimeField>(
        &mut self,
        from: PartyID,
        elems: usize,
    ) -> std::io::Result<Vec<u8>> {
        use crate::protocols::rep3_ring::preprocessing::backing_store::assert_field_layout;
        const { assert_field_layout::<F>() };
        let bytes = self.ring_recv_bytes(from)?;
        let expected = elems.saturating_mul(std::mem::size_of::<F>());
        if bytes.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
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
}

#[derive(Clone)]
pub struct LocalRep3TestCoordinatorNet {
    log_num_workers: usize,
    active_num_workers: usize,
    star_req_txs: [mpsc::Sender<Vec<u8>>; 3],
    star_resp_rxs: [Arc<Mutex<mpsc::Receiver<Vec<u8>>>>; 3],
}

impl LocalRep3TestCoordinatorNet {
    fn recv_from_party<T: CanonicalDeserialize>(&mut self, party: PartyID) -> Result<T> {
        let idx = usize::from(party);
        let bytes = self.star_resp_rxs[idx]
            .lock()
            .unwrap()
            .recv()
            .map_err(|_| eyre!("coordinator recv failed"))?;
        deserialize_uncompressed(&bytes)
    }
}

impl MpcStarNetCoordinator for LocalRep3TestCoordinatorNet {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> Result<Vec<T>> {
        Ok(vec![
            self.recv_from_party(PartyID::ID0)?,
            self.recv_from_party(PartyID::ID1)?,
            self.recv_from_party(PartyID::ID2)?,
        ])
    }

    fn receive_responses_from_subnets<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> Result<Vec<Vec<T>>> {
        Ok(vec![self.receive_responses()?])
    }

    fn receive_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
    ) -> Result<T> {
        if worker_id != 0 {
            return Err(eyre!("in-memory net only supports worker_id=0"));
        }
        self.recv_from_party(party_id)
    }

    fn receive_response_from_workers<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
    ) -> Result<Vec<T>> {
        Ok(vec![self.receive_response(party_id, 0)?])
    }

    fn broadcast_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: T,
    ) -> Result<()> {
        let bytes = serialize_uncompressed(&data)?;
        for tx in self.star_req_txs.iter() {
            tx.send(bytes.clone())
                .map_err(|_| eyre!("broadcast_request failed"))?;
        }
        Ok(())
    }

    fn send_requests<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> Result<()> {
        if data.len() != 3 {
            return Err(eyre!("send_requests expects 3 items, got {}", data.len()));
        }
        for (i, item) in data.into_iter().enumerate() {
            let bytes = serialize_uncompressed(&item)?;
            self.star_req_txs[i]
                .send(bytes)
                .map_err(|_| eyre!("send_requests failed"))?;
        }
        Ok(())
    }

    fn send_request_to_workers<T: CanonicalSerialize + CanonicalDeserialize + Clone>(
        &mut self,
        party_id: PartyID,
        data: T,
    ) -> Result<()> {
        self.send_request(party_id, 0, data)
    }

    fn send_requests_to_workers<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> Result<()> {
        self.send_requests(data)
    }

    fn send_requests_blocking<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> Result<()> {
        self.send_requests(data)
    }

    fn send_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
        data: T,
    ) -> Result<()> {
        if worker_id != 0 {
            return Err(eyre!("in-memory net only supports worker_id=0"));
        }
        let idx = usize::from(party_id);
        let bytes = serialize_uncompressed(&data)?;
        self.star_req_txs[idx]
            .send(bytes)
            .map_err(|_| eyre!("send_request failed"))?;
        Ok(())
    }

    fn log_num_workers(&self) -> usize {
        self.log_num_workers
    }

    fn active_num_workers(&self) -> usize {
        self.active_num_workers
    }

    fn total_bandwidth_used(&self) -> (u64, u64) {
        (0, 0)
    }

    fn log_connection_stats(&self, _label: Option<&str>) {}

    fn reset_stats(&mut self) {}

    fn fork(&mut self) -> Result<Self> {
        Ok(self.clone())
    }

    fn set_num_workers(&mut self, num_workers: usize) {
        self.active_num_workers = num_workers;
        self.log_num_workers = num_workers.next_power_of_two().trailing_zeros() as usize;
    }

    fn reset_num_workers(&mut self) {
        self.active_num_workers = 1;
        self.log_num_workers = 0;
    }
}

impl Rep3NetworkCoordinator for LocalRep3TestCoordinatorNet {
    fn sync_with_parties(&mut self) -> eyre::Result<()> {
        self.broadcast_request(true)?;
        let _ = self.receive_responses::<bool>()?;
        Ok(())
    }
}

pub fn run_rep3_local_test_with_coordinator<WI, WO, CI, CO, WF, CF>(
    num_io_forks: u32,
    make_worker_input: impl Fn(usize) -> WI,
    make_coordinator_input: impl FnOnce() -> CI,
    worker_fn: WF,
    coordinator_fn: CF,
) -> ([WO; 3], CO)
where
    WI: Send + 'static,
    WO: Send + 'static,
    CI: Send + 'static,
    CO: Send + 'static,
    WF: Fn(WI, IoContextPool<LocalRep3TestWorkerNet>) -> eyre::Result<WO> + Send + Sync + 'static,
    CF: FnOnce(CI, &mut LocalRep3TestCoordinatorNet) -> eyre::Result<CO> + Send + 'static,
{
    // Star channels (coordinator <-> each party).
    let mut star_req_txs = Vec::with_capacity(3);
    let mut star_req_rxs = Vec::with_capacity(3);
    let mut star_resp_txs = Vec::with_capacity(3);
    let mut star_resp_rxs = Vec::with_capacity(3);
    for _ in 0..3 {
        let (req_tx, req_rx) = mpsc::channel::<Vec<u8>>();
        let (resp_tx, resp_rx) = mpsc::channel::<Vec<u8>>();
        star_req_txs.push(req_tx);
        star_req_rxs.push(Arc::new(Mutex::new(req_rx)));
        star_resp_txs.push(resp_tx);
        star_resp_rxs.push(Arc::new(Mutex::new(resp_rx)));
    }

    let star_req_txs_arr: [mpsc::Sender<Vec<u8>>; 3] =
        star_req_txs.try_into().unwrap_or_else(|_| unreachable!());
    let star_req_rxs_arr: [Arc<Mutex<mpsc::Receiver<Vec<u8>>>>; 3] =
        star_req_rxs.try_into().unwrap_or_else(|_| unreachable!());
    let star_resp_txs_arr: [mpsc::Sender<Vec<u8>>; 3] =
        star_resp_txs.try_into().unwrap_or_else(|_| unreachable!());
    let star_resp_rxs_arr: [Arc<Mutex<mpsc::Receiver<Vec<u8>>>>; 3] =
        star_resp_rxs.try_into().unwrap_or_else(|_| unreachable!());

    // Ring channels (party <-> party).
    let mut ring_txs: [[Option<mpsc::Sender<Vec<u8>>>; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| None));
    let mut ring_rxs: [[Option<Arc<Mutex<mpsc::Receiver<Vec<u8>>>>>; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| None));
    for from in 0..3 {
        for to in 0..3 {
            if from == to {
                continue;
            }
            let (tx, rx) = mpsc::channel::<Vec<u8>>();
            ring_txs[from][to] = Some(tx);
            ring_rxs[to][from] = Some(Arc::new(Mutex::new(rx)));
        }
    }

    let worker_fn = std::sync::Arc::new(worker_fn);

    // Spawn worker threads.
    let worker_handles: Vec<_> = (0..3)
        .map(|i| {
            let input = make_worker_input(i);
            let worker_fn = std::sync::Arc::clone(&worker_fn);

            let id = PartyID::try_from(i).unwrap();
            let mut senders: Vec<mpsc::Sender<Vec<u8>>> = Vec::with_capacity(3);
            let mut receivers: Vec<Arc<Mutex<mpsc::Receiver<Vec<u8>>>>> = Vec::with_capacity(3);
            for j in 0..3 {
                if i == j {
                    // Dummy channel for unused self-send/recv.
                    let (tx, rx) = mpsc::channel::<Vec<u8>>();
                    senders.push(tx);
                    receivers.push(Arc::new(Mutex::new(rx)));
                } else {
                    senders.push(ring_txs[i][j].as_ref().unwrap().clone());
                    receivers.push(ring_rxs[i][j].as_ref().unwrap().clone());
                }
            }
            let ring_txs_arr: [mpsc::Sender<Vec<u8>>; 3] =
                senders.try_into().unwrap_or_else(|_| unreachable!());
            let ring_rxs_arr: [Arc<Mutex<mpsc::Receiver<Vec<u8>>>>; 3] =
                receivers.try_into().unwrap_or_else(|_| unreachable!());

            let net = LocalRep3TestWorkerNet {
                id,
                worker_idx: 0,
                log_num_workers: 0,
                star_req_rx: star_req_rxs_arr[i].clone(),
                star_resp_tx: star_resp_txs_arr[i].clone(),
                ring_txs: ring_txs_arr,
                ring_rxs: ring_rxs_arr,
            };

            thread::spawn(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .thread_name(move |idx| format!("party-{i}-rayon-{idx}"))
                    .build()
                    .unwrap();
                pool.install(|| {
                    let io_ctx = IoContextPool::init(net, num_io_forks)
                        .with_context(|| format!("party {i} io_ctx init"))
                        .unwrap();
                    worker_fn(input, io_ctx)
                        .with_context(|| format!("party {i} work"))
                        .unwrap()
                })
            })
        })
        .collect();

    // Spawn coordinator thread.
    let coordinator_input = make_coordinator_input();
    let coordinator_handle = thread::spawn(move || {
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|idx| format!("coordinator-rayon-{idx}"))
            .build()
            .unwrap();
        pool.install(|| {
            let mut net = LocalRep3TestCoordinatorNet {
                log_num_workers: 0,
                active_num_workers: 1,
                star_req_txs: star_req_txs_arr,
                star_resp_rxs: star_resp_rxs_arr,
            };
            coordinator_fn(coordinator_input, &mut net)
                .context("coordinator work")
                .unwrap()
        })
    });

    let worker_results: Vec<WO> = worker_handles
        .into_iter()
        .map(|h| h.join().expect("worker thread panicked"))
        .collect();
    let coordinator_result = coordinator_handle
        .join()
        .expect("coordinator thread panicked");

    let worker_array = worker_results.try_into().unwrap_or_else(|_| unreachable!());
    (worker_array, coordinator_result)
}
