pub mod instruction_utils;
pub mod element;
pub mod transcript;

pub use jolt_core::{
    field::JoltField,
    utils::{
        compute_dotproduct, gaussian_elimination, gen_random_point, index_to_field_bitvector,
        is_power_of_two, math, mul_0_1_optimized, mul_0_optimized, split_bits, thread,
        errors,
    },
};
use jolt_core::utils::transcript::Transcript;

use eyre::{Context, Result};
use itertools::Itertools;

use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    PartyID,
};

use crate::poly::{dense_mlpoly::DensePolynomial, Rep3DensePolynomial};

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
        #[cfg(feature = "parallel")]
        let iter = iter_forked.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = iter_forked.into_iter();
        Ok::<_, eyre::Report>(iter)
    })?;

    iter.map(|(val, mut ctx)| map_fn(val, &mut ctx))
        .collect::<Result<Vec<_>>>()
}

pub fn fork_chunks<F, T, R, N: Rep3Network>(
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
    let chunk_size = len.div_ceil(max_forks);
    let forks = len.div_ceil(chunk_size);
    assert!(chunk_size != 0);

    let iter = tracing::trace_span!("setup forked networks", len, forks).in_scope(|| {
        let iter_forked = i
            .into_iter()
            .chunks(chunk_size)
            .into_iter()
            .map(|chunk| ctx.fork().map(|ctx| (chunk.collect_vec(), ctx)))
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "parallel")]
        let iter = iter_forked.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = iter_forked.into_iter();
        Ok::<_, eyre::Report>(iter)
    })?;

    iter.map(|(chunk, mut ctx)| {
        chunk
            .into_iter()
            .map(|val| map_fn(val, &mut ctx))
            .collect_vec()
    })
    .flatten()
    .collect::<eyre::Result<Vec<_>>>()
    .context("while trying to fork_chunks_flat_map")
}

#[inline]
pub fn split_rep3_poly_flagged<F: JoltField>(
    poly: &Rep3DensePolynomial<F>,
    flags: &DensePolynomial<F>,
    id: PartyID,
) -> (Rep3DensePolynomial<F>, Rep3DensePolynomial<F>) {
    // let (a, b) = poly.copy_poly_shares();

    // let (left_a, right_a) = split_poly_flagged(&a, flags);
    // let (left_b, right_b) = split_poly_flagged(&b, flags);

    // let left = Rep3DensePolynomial::from_vec_shares(left_a, left_b);
    // let right = Rep3DensePolynomial::from_vec_shares(right_a, right_b);

    // (left, right)

    let poly_evals = poly.evals_ref();
    let len = poly_evals.len();
    let half = len / 2;
    let mut left = Vec::with_capacity(half);
    let mut right = Vec::with_capacity(half);

    let one = rep3::arithmetic::promote_to_trivial_share(id, F::one());

    for i in 0..len {
        if flags[i].is_zero() {
            if i < half {
                left.push(one);
            } else {
                right.push(one);
            }
        } else if i < half {
            left.push(poly_evals[i]);
        } else {
            right.push(poly_evals[i]);
        }
    }
    (
        Rep3DensePolynomial::new(left),
        Rep3DensePolynomial::new(right),
    )
}
