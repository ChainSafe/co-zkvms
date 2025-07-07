use itertools::Itertools;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn fork_map<F, T, R, N: Rep3Network>(
    i: impl IntoIterator<Item = T>,
    io_context0: &mut IoContext<N>,
    io_context1: &mut IoContext<N>,
    map_fn: F,
) -> Vec<R>
where
    F: Fn(T, IoContext<N>, IoContext<N>) -> R + Sync + Send,
    T: Sized + Send,
    R: Sync + Send,
{
    let iter = tracing::info_span!("setup forked networks").in_scope(|| {
        let iter = i.into_iter().map(|val| {
            (
                val,
                io_context0.fork().unwrap(),
                io_context1.fork().unwrap(),
            )
        });
        #[cfg(feature = "parallel")]
        let iter = iter.collect_vec().into_par_iter();
        iter
    });

    iter.map(|(val, io_context0, io_context1)| map_fn(val, io_context0, io_context1))
        .collect()
}

pub fn fork_chunks_flat_map<F, T, R, N: Rep3Network>(
    i: &[T],
    io_context0: &mut IoContext<N>,
    io_context1: &mut IoContext<N>,
    chunk_size: usize,
    map_fn: F,
) -> Vec<R>
where
    F: Fn(T, &mut IoContext<N>, &mut IoContext<N>) -> R + Sync + Send,
    T: Sized + Send + Clone,
    R: Send,
{
    let iter = i.chunks(chunk_size).into_iter().map(|chunk| {
        (
            chunk.to_vec(),
            io_context0.fork().unwrap(),
            io_context1.fork().unwrap(),
        )
    });

    #[cfg(feature = "parallel")]
    let iter = iter.collect_vec().into_par_iter();

    iter.map(|(chunk, mut io_context0, mut io_context1)| {
        chunk
            .into_iter()
            .map(|val| map_fn(val, &mut io_context0, &mut io_context1))
            .collect_vec()
    })
    .flatten()
    .collect()
}
