//! Bitwise suffix evaluation helpers: AND, NotAnd, XOR, OR, XOR-rotate.

use super::future::{B2ABucketExtend, SuffixFutureBatch};
use super::{to_u32_share, MixedBatch, Uninterleavable};
use crate::field::JoltField;
use crate::utils::types::Either;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self as rep3_ring, Rep3RingShare};
use rand::distributions::Standard;
use rand::prelude::Distribution;

/// Shared 3-way MixedBatch dispatch for And, NotAnd, Or suffix variants.
///
/// Handles Public (local), Shared (MPC), and Mixed (split + recombine) cases.
/// `local_fn` computes the per-element result for public y values.
/// `mpc_fn` is the MPC operation for fully-shared batches (and_many or or_many).
pub(crate) fn eval_bitwise_mixed<T, F, N>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    base: usize,
    orig: impl Fn(usize) -> usize,
    io_ctx: &mut IoContext<N>,
    out: &mut SuffixFutureBatch<F>,
    local_fn: impl Fn(&Rep3RingShare<T::Half>, u64) -> Rep3RingShare<T::Half>,
    mpc_fn: impl FnOnce(
        &[Rep3RingShare<T::Half>],
        &[Rep3RingShare<T::Half>],
        &mut IoContext<N>,
    ) -> std::io::Result<Vec<Rep3RingShare<T::Half>>>,
) -> eyre::Result<()>
where
    T: Uninterleavable,
    T::Half: B2ABucketExtend,
    F: JoltField,
    N: Rep3Network,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    match right {
        MixedBatch::Public(y_pubs) => {
            out.extend_b2a_ring::<T::Half>(
                indices_iter,
                xs.iter()
                    .enumerate()
                    .map(|(j, x)| local_fn(x, y_pubs[orig(j)])),
            );
        }
        MixedBatch::Shared(ys) => {
            let result = mpc_fn(xs, ys, io_ctx)?;
            out.extend_b2a_ring::<T::Half>(indices_iter, result.into_iter());
        }
        MixedBatch::Mixed(mixed) => {
            let mut local_idx = Vec::new();
            let mut local_vals = Vec::new();
            let mut mpc_idx = Vec::new();
            let mut mpc_xs = Vec::new();
            let mut mpc_ys = Vec::new();
            for (j, x) in xs.iter().enumerate() {
                let i = orig(j);
                match &mixed[i] {
                    Either::Public(yp) => {
                        local_idx.push(base + i);
                        local_vals.push(local_fn(x, *yp));
                    }
                    Either::Shared(y) => {
                        mpc_idx.push(base + i);
                        mpc_xs.push(*x);
                        mpc_ys.push(*y);
                    }
                }
            }
            out.extend_b2a_ring::<T::Half>(local_idx.into_iter(), local_vals.into_iter());
            if !mpc_xs.is_empty() {
                let result = mpc_fn(&mpc_xs, &mpc_ys, io_ctx)?;
                out.extend_b2a_ring::<T::Half>(mpc_idx.into_iter(), result.into_iter());
            }
        }
    }
    Ok(())
}

/// XOR suffix: always local — no MPC for any MixedBatch variant.
pub(crate) fn eval_xor<T, F, N>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    base: usize,
    orig: impl Fn(usize) -> usize,
    party_id: PartyID,
    out: &mut SuffixFutureBatch<F>,
) where
    T: Uninterleavable,
    T::Half: B2ABucketExtend,
    F: JoltField,
    N: Rep3Network,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    out.extend_b2a_ring::<T::Half>(
        indices_iter,
        xs.iter().enumerate().map(|(j, x)| {
            let i = orig(j);
            match right {
                MixedBatch::Public(y_pubs) => {
                    let mask = RingElement(
                        T::Half::try_from(y_pubs[i] as u128)
                            .unwrap_or_else(|_| unreachable!()),
                    );
                    rep3_ring::binary::xor_public(x, &mask, party_id)
                }
                MixedBatch::Shared(ys) => *x ^ ys[i],
                MixedBatch::Mixed(mixed) => match &mixed[i] {
                    Either::Public(yp) => {
                        let mask = RingElement(
                            T::Half::try_from(*yp as u128)
                                .unwrap_or_else(|_| unreachable!()),
                        );
                        rep3_ring::binary::xor_public(x, &mask, party_id)
                    }
                    Either::Shared(y) => *x ^ *y,
                },
            }
        }),
    );
}

/// XorRot with pre-uninterleaved operands: (x ^ y) rotate_right by ROTATION.
pub(crate) fn eval_xor_rot_uninterleaved<
    const ROTATION: u32,
    T: Uninterleavable,
    F: JoltField,
>(
    xs: &[Rep3RingShare<T::Half>],
    ys: &[Rep3RingShare<T::Half>],
    indices: impl IntoIterator<Item = usize>,
    out: &mut SuffixFutureBatch<F>,
) where
    Standard: Distribution<T::Half>,
{
    let k = T::Half::K;
    let rot = (ROTATION as usize) % k;
    let result: Vec<Rep3RingShare<T::Half>> = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let xored = *x ^ *y;
            (xored >> rot) ^ (xored << (k - rot))
        })
        .collect();
    out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
}

/// XorRotW with pre-uninterleaved operands: u32 rotate then push as u32 or T::Half.
pub(crate) fn eval_xor_rot_w_uninterleaved<
    const ROTATION: u32,
    T: Uninterleavable,
    F: JoltField,
>(
    xs: &[Rep3RingShare<T::Half>],
    ys: &[Rep3RingShare<T::Half>],
    indices: impl IntoIterator<Item = usize>,
    out: &mut SuffixFutureBatch<F>,
) where
    Standard: Distribution<T::Half>,
{
    let rotated_u32: Vec<Rep3RingShare<u32>> = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let x32 = to_u32_share(*x);
            let y32 = to_u32_share(*y);
            let xored = x32 ^ y32;
            let rot = (ROTATION as usize) % 32;
            (xored >> rot) ^ (xored << (32 - rot))
        })
        .collect();

    let idx: Vec<usize> = indices.into_iter().collect();
    if T::Half::K >= 32 {
        let result: Vec<Rep3RingShare<T::Half>> = rotated_u32
            .iter()
            .map(|r| Rep3RingShare {
                a: RingElement(T::Half::try_from(r.a.0 as u128).unwrap_or_else(|_| unreachable!())),
                b: RingElement(T::Half::try_from(r.b.0 as u128).unwrap_or_else(|_| unreachable!())),
            })
            .collect();
        out.extend_b2a_ring::<T::Half>(idx.into_iter(), result.into_iter());
    } else {
        out.extend_b2a_ring::<u32>(idx.into_iter(), rotated_u32.into_iter());
    }
}
