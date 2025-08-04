use ark_ff::PrimeField;
use ark_linear_sumcheck::rng::FeedableRNG;
use eyre::Context;
use itertools::Itertools;
use itertools::izip;
use rand::Rng;
use rand::RngCore;

pub use mpc_core::protocols::rep3::arithmetic::*;
pub use mpc_core::protocols::rep3::{
    Rep3PrimeFieldShare,
    network::{IoContext, Rep3Network},
};

use crate::protocols::additive::AdditiveShare;
use crate::protocols::rep3::rngs::SSRandom;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn generate_shares_rep3<F: PrimeField, R: Rng>(
    val: F,
    rng: &mut R,
) -> Vec<Rep3PrimeFieldShare<F>> {
    let t0 = F::rand(rng);
    let t1 = F::rand(rng);
    let t2 = val - t0 - t1;

    let p_share_0 = Rep3PrimeFieldShare::new(t0, t2); // Party 0 gets (t_0, t_2)
    let p_share_1 = Rep3PrimeFieldShare::new(t1, t0); // Party 1 gets (t_1, t_0)
    let p_share_2 = Rep3PrimeFieldShare::new(t2, t1); // Party 2 gets (t_2, t_1)
    vec![p_share_0, p_share_1, p_share_2]
}

pub fn open<F: PrimeField>(shares: [Rep3PrimeFieldShare<F>; 3], id: usize) -> F {
    shares[id].a + shares[id].b + shares[(id + 2) % 3].b
}

pub fn get_mask_scalar_rep3<F: PrimeField, R: RngCore + FeedableRNG>(
    rng: &mut SSRandom<R>,
) -> (F, F) {
    let mask_share = (
        F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0),
        F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0),
    );
    rng.update();
    mask_share
}

#[tracing::instrument(skip_all, name = "product", level = "trace")]
pub fn product<F: PrimeField, N: Rep3Network>(
    shares: &[Rep3PrimeFieldShare<F>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Rep3PrimeFieldShare<F>> {
    shares
        .iter()
        .skip(1)
        .try_fold(shares[0], |acc, x| mul(acc, *x, io_ctx))
        .context("while computing product")
}

pub fn product_into_additive<F: PrimeField, N: Rep3Network>(
    shares: &[Rep3PrimeFieldShare<F>],
    io_ctx: &mut IoContext<N>,
    public_extra: Option<F>,
) -> eyre::Result<AdditiveShare<F>> {
    let num_multiplications = shares.len();

    if num_multiplications == 1 {
        return Ok((shares[0] * public_extra.unwrap_or(F::one())).into_additive());
    } else if num_multiplications == 2 {
        return Ok(shares[0] * (shares[1] * public_extra.unwrap_or(F::one())));
    }

    let product_except_last = shares
        .iter()
        .skip(1)
        .take(num_multiplications - 2)
        .try_fold(shares[0] * public_extra.unwrap_or(F::one()), |acc, x| {
            mul(acc, *x, io_ctx)
        })
        .context("while computing product")?;

    Ok(product_except_last * *shares.last().unwrap())
}

#[tracing::instrument(skip_all, name = "product_many", level = "trace")]
pub fn product_many<F: PrimeField, N: Rep3Network, S>(
    shares: impl IntoIterator<Item = S>,
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    S: AsRef<[Rep3PrimeFieldShare<F>]>,
{
    let mut shares = shares.into_iter();
    let first_share = shares.next().unwrap();
    let first_share_ref = first_share.as_ref();
    shares
        .try_fold(
            (0..first_share_ref.len()).map(|i| first_share_ref[i]).collect_vec(),
            |acc, x| mul_vec(&acc, x.as_ref(), io_ctx),
        )
        .context("while computing product")
}

pub fn product_many_into_additive<F: PrimeField, N: Rep3Network>(
    shares: &[&[Rep3PrimeFieldShare<F>]],
    io_ctx: &mut IoContext<N>,
    public_extra: Option<F>,
) -> eyre::Result<Vec<AdditiveShare<F>>> {
    let num_multiplications = shares[0].len();
    if num_multiplications == 1 {
        return Ok(shares
            .iter()
            .map(|x| (x[0] * public_extra.unwrap_or(F::one())).into_additive())
            .collect());
    } else if num_multiplications == 2 {
        return Ok(shares
            .iter()
            .map(|x| x[0] * (x[1] * public_extra.unwrap_or(F::one())))
            .collect());
    }

    let products_except_last = shares
        .iter()
        .skip(1)
        .take(num_multiplications - 2)
        .try_fold(
            shares
                .iter()
                .map(|x| x[0] * public_extra.unwrap_or(F::one()))
                .collect_vec(),
            |acc, x| mul_vec(&acc, *x, io_ctx),
        )
        .context("while computing product")?;
    Ok(products_except_last
        .into_iter()
        .zip(shares.into_iter().map(|x| x.last().unwrap()))
        .map(|(a, &b)| a * b)
        .collect())
}

#[tracing::instrument(skip_all, name = "reshare_additive", level = "trace")]
pub fn reshare_additive<F: PrimeField, N: Rep3Network>(
    additive: AdditiveShare<F>,
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Rep3PrimeFieldShare<F>> {
    let prev_share: F = io_ctx.network.reshare(additive)?;
    Ok(Rep3PrimeFieldShare::new(additive, prev_share))
}

#[tracing::instrument(skip_all, name = "reshare_additive_many", level = "trace")]
pub fn reshare_additive_many<F: PrimeField, N: Rep3Network>(
    additive_shares: &[AdditiveShare<F>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let b_shares: Vec<F> = io_ctx.network.reshare_many(additive_shares)?;
    Ok(additive_shares
        .into_par_iter()
        .zip(b_shares.into_par_iter())
        .map(|(a, b)| Rep3PrimeFieldShare::new(*a, b))
        .collect())
}

/// Convenience method for \[a\] * (\[b\] * c)
#[tracing::instrument(skip_all, name = "mul_mul_public", level = "trace")]
pub fn mul_mul_public<F: PrimeField>(a: FieldShare<F>, b: FieldShare<F>, c: F) -> F {
    a * mul_public(b, c)
}

#[tracing::instrument(skip_all, name = "sum_batched", level = "trace")]
pub fn sum_batched<F: PrimeField>(
    vals: &[Vec<Rep3PrimeFieldShare<F>>],
) -> Vec<Rep3PrimeFieldShare<F>> {
    let bathes_len = vals[0].len();
    (0..bathes_len)
        .map(|i| {
            vals.iter()
                .map(|val| val[i])
                .sum::<Rep3PrimeFieldShare<F>>()
        })
        .collect()
}

/// Reconstructs a vector of field elements from its arithmetic replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
#[tracing::instrument(skip_all, name = "combine_field_elements_vec", level = "trace")]
pub fn combine_field_elements_vec<F: PrimeField>(
    shares: Vec<Vec<Rep3PrimeFieldShare<F>>>,
) -> Vec<F> {
    let [s0, s1, s2]: [Vec<_>; 3] = shares.try_into().unwrap();
    mpc_core::protocols::rep3::combine_field_elements(&s0, &s1, &s2)
}
