pub mod shared_or_public;
pub mod future;
pub mod instruction_utils;
pub mod transcript;

pub use jolt_core::{
    field::JoltField,
    poly::dense_mlpoly::DensePolynomial,
    utils::{
        compute_dotproduct, errors, gaussian_elimination, gen_random_point,
        index_to_field_bitvector, is_power_of_two, math, mul_0_1_optimized, mul_0_optimized,
        split_bits, thread,
    },
};

use eyre::{Context, Result};
use itertools::Itertools;

use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub trait Forkable: Sized + Send {
    fn fork(&mut self) -> Result<Self>;
}

impl<N: Rep3Network> Forkable for IoContext<N> {
    fn fork(&mut self) -> Result<Self> {
        self.fork().context("while trying to fork IoContext")
    }
}

pub fn fork_map<F: Forkable, T, R, M>(
    i: impl IntoIterator<Item = T>,
    ctx: &mut F,
    map_fn: M,
) -> Result<Vec<R>>
where
    M: Fn(T, &mut F) -> R + Sync + Send,
    T: Sized + Send,
    R: Sync + Send,
{
    let iter = tracing::info_span!("setup forked networks").in_scope(|| {
        let iter_forked = i
            .into_iter()
            .map(|val| ctx.fork().map(|ctx| (val, ctx)))
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "parallel")]
        let iter = iter_forked.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = iter_forked.into_iter();
        Ok::<_, eyre::Report>(iter)
    })?;

    Ok(iter.map(|(val, mut ctx)| map_fn(val, &mut ctx)).collect())
}

pub fn try_fork_map<F: Forkable, T, R, M>(
    i: impl IntoIterator<Item = T>,
    ctx: &mut F,
    map_fn: M,
) -> Result<Vec<R>>
where
    M: Fn(T, &mut F) -> eyre::Result<R> + Sync + Send,
    T: Sized + Send,
    R: Sync + Send,
{
    let iter = tracing::info_span!("setup forked networks").in_scope(|| {
        let iter_forked = i
            .into_iter()
            .map(|val| ctx.fork().map(|ctx| (val, ctx)))
            .collect::<Result<Vec<_>>>()?;
        Ok::<_, eyre::Report>(iter_forked.into_par_iter())
    })?;

    iter.map(|(val, mut ctx)| map_fn(val, &mut ctx))
        .collect::<Result<Vec<_>>>()
}

pub fn try_fork_chunks<F, T, R, M>(
    i: impl IntoIterator<Item = T> + ExactSizeIterator,
    ctx: &mut F,
    max_forks: usize,
    map_fn: M,
) -> eyre::Result<Vec<R>>
where
    F: Forkable,
    M: Fn(T, &mut F) -> eyre::Result<R> + Sync + Send,
    T: Sized + Send + Clone,
    R: Sync + Send,
{
    let len = i.len();

    if max_forks == 0 {
        return i
            .into_iter()
            .map(|val| map_fn(val, ctx))
            .collect::<eyre::Result<Vec<_>>>();
    }

    if len == 1 {
        return Ok(vec![map_fn(i.into_iter().next().unwrap(), ctx)?]);
    }

    let chunk_size = len.div_ceil(max_forks);
    assert!(chunk_size != 0);
    let forks = len.div_ceil(chunk_size);

    let iter = tracing::info_span!("setup forked networks", len, forks).in_scope(|| {
        let iter_forked = i
            .into_iter()
            .chunks(chunk_size)
            .into_iter()
            .map(|chunk| ctx.fork().map(|ctx| (chunk.collect_vec(), ctx)))
            .collect::<Result<Vec<_>>>()?;
        Ok::<_, eyre::Report>(iter_forked.into_par_iter())
    })?;

    iter.map(|(chunk, mut ctx)| {
        chunk
            .into_iter()
            .map(|val| map_fn(val, &mut ctx))
            .collect_vec()
    })
    .flatten()
    .collect::<eyre::Result<Vec<_>>>()
}
