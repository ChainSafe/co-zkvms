use std::iter;

use eyre::Context;
use itertools::Itertools;
use mpc_core::protocols::rep3::PartyID;
use mpc_net::mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker};

pub use mpc_core::protocols::rep3::network::*;
use mpc_net::rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator};
use rand::{Rng, SeedableRng};

use rayon::prelude::*;

pub trait Rep3NetworkWorker: Rep3Network + MpcStarNetWorker + 'static {}
pub trait Rep3NetworkCoordinator: MpcStarNetCoordinator + 'static {}

impl Rep3NetworkWorker for Rep3QuicMpcNetWorker {}

impl Rep3NetworkCoordinator for Rep3QuicNetCoordinator {}

pub struct WorkerIoContext<Network: Rep3NetworkWorker> {
    pub worker_id: usize,
    main: IoContext<Network>,
    forks: Vec<IoContext<Network>>,
}

impl<Network: Rep3NetworkWorker> WorkerIoContext<Network> {
    pub fn main(&mut self) -> &mut IoContext<Network> {
        &mut self.main
    }

    pub fn forks(&mut self, num_forks: usize) -> &mut [IoContext<Network>] {
        &mut self.forks[..num_forks]
    }

    pub fn network(&mut self) -> &mut Network {
        &mut self.main.network
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

    /// Parallelize the computation of `map` over the `inputs` using the forked `IoContext`s.
    ///
    /// The `inputs` are split into chunks, and each chunk is processed in parallel using the `forks`.
    ///
    /// The `map` is a function that takes a chunk of inputs and an `IoContext` and returns a flattened result.
    ///
    /// The `chunk_size` is the size of each chunk.
    /// If `None`, the chunk size is the number of inputs divided by the number of available forks.
    pub fn par_chunks<T, R, MapFn>(
        &mut self,
        inputs: impl IntoParallelIterator<Item = T, Iter: IndexedParallelIterator>,
        chunk_size: Option<usize>,
        map: MapFn,
    ) -> eyre::Result<Vec<R>>
    where
        MapFn: Fn(Vec<T>, &mut IoContext<Network>) -> eyre::Result<Vec<R>> + Sync + Send,
        T: Sized + Send + Clone,
        R: Sync + Send,
    {
        let inputs_iter = inputs.into_par_iter();
        let len = inputs_iter.len();

        let chunk_size = chunk_size.unwrap_or(len.div_ceil(self.forks.len()));
        assert!(chunk_size != 0);
        if len <= chunk_size {
            return map(inputs_iter.collect(), self.main());
        }
        let forks = len.div_ceil(chunk_size);

        Ok(inputs_iter
            .into_par_iter()
            .chunks(chunk_size)
            .zip_eq(self.forks(forks).par_iter_mut())
            .flat_map_iter(|(chunk, mut ctx)| {
                map(chunk, &mut ctx)
                    // .map(|r| r.into_iter().map(|r| Ok(r)))
                    // .map_err(|e| iter::once(e))
                    .unwrap()
            })
            .collect::<Vec<_>>())
    }
}

// impl<Network: Rep3NetworkWorker> WorkerIoContext<Network> {
//     pub fn init(worker_id: usize, network: Network, num_forks: usize) -> eyre::Result<Self> {
//         let mut main = IoContext::init(network)?;
//         let forks = iter::repeat_with(|| main.fork())
//             .take(num_forks)
//             .collect::<Result<Vec<_>, _>>()?;
//         Ok(Self {
//             worker_id,
//             main,
//             forks,
//         })
//     }
// }

pub struct IoContextPool<Network: Rep3NetworkWorker> {
    /// The party id
    pub id: PartyID,

    pub workers: Vec<WorkerIoContext<Network>>,
}

impl<Network: Rep3NetworkWorker> IoContextPool<Network> {
    #[tracing::instrument(skip_all, name = "IoContextPool::init")]
    pub fn init(network: Network, num_forks: usize) -> eyre::Result<Self> {
        let num_workers = 1 << network.log_num_workers_per_party();
        let mut main_worker = IoContext::init(network)?;
        let rngs = &mut main_worker.rngs;
        let rng = &mut main_worker.rng;
        let a2b_type = main_worker.a2b_type;
        let id = main_worker.id;

        let workers: Vec<_> = main_worker
            .network
            .get_worker_subnets(num_workers)
            .context("while setting up worker subnets")?
            .into_iter()
            .map(|network| {
                let rngs = rngs.fork();
                let rng = rand_chacha::ChaCha12Rng::from_seed(rng.r#gen());
                IoContext {
                    network,
                    rngs,
                    rng,
                    id,
                    a2b_type,
                }
            })
            .collect();
        let workers = rayon::iter::once(main_worker)
            .chain(workers)
            .enumerate()
            .map(|(worker_id, mut worker)| {
                let forks = iter::repeat_with(|| worker.fork())
                    .take(num_forks)
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(WorkerIoContext {
                    worker_id,
                    main: worker,
                    forks,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        Ok(Self { id, workers })
    }

    pub fn main(&mut self) -> &mut IoContext<Network> {
        &mut self.workers[0].main
    }

    pub fn worker(&mut self, worker_id: usize) -> &mut WorkerIoContext<Network> {
        &mut self.workers[worker_id]
    }

    pub fn network(&mut self) -> &mut Network {
        &mut self.workers[0].main.network
    }

    pub fn fork(&mut self) -> eyre::Result<IoContext<Network>> {
        self.workers[0]
            .main
            .fork()
            .context("while forking io context")
    }

    pub fn log_num_workers_per_party(&self) -> usize {
        self.workers[0].main.network.log_num_workers_per_party()
    }

    pub fn num_workers(&self) -> usize {
        1 << self.log_num_workers_per_party()
    }

    pub fn workers_mut(&mut self) -> &mut [WorkerIoContext<Network>] {
        &mut self.workers
    }

    pub fn party_id(&self) -> PartyID {
        self.id
    }
}
