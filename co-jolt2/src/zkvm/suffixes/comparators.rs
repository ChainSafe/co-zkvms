//! Comparison and division-check suffix evaluation helpers.
//!
//! Contains `ge_many_mixed` (batched >= with mixed public/shared operands)
//! and named eval functions for Eq, RightOperandIsZero, DivByZero,
//! ChangeDivisor, and ChangeDivisorW suffix arms.

use super::future::{B2ABucketExtend, SuffixFutureBatch};
use super::{to_u32_share, MixedBatch, Uninterleavable};
use crate::utils::types::rep3_value::Rep3Value;
use crate::utils::types::Either;
use jolt_common::constants::XLEN;
use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self as rep3_ring, Rep3RingShare};
use num_traits::AsPrimitive;
use rand::distributions::Standard;
use rand::prelude::Distribution;

// ---------------------------------------------------------------------------
// ge_many_mixed
// ---------------------------------------------------------------------------

/// Batched comparison with mixed public/shared right operand.
/// Computes `x >= y` (swap=false) or `y >= x` (swap=true) without promoting
/// public y values to trivial shares. Instead:
/// - Public y: compute propagate/generate bits locally (no communication)
/// - Shared y: compute propagate locally, generate via `and_many` (1 round)
/// - Single Kogge-Stone carry tree for all elements (log2(K) rounds)
///
/// Total: 1 + log2(K) rounds (same as `ge_many`), but the AND round only
/// processes shared-y elements, and public-y elements contribute zero AND cost.
pub(crate) fn ge_many_mixed<T: Uninterleavable, N: Rep3Network>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    n: usize,
    orig: &dyn Fn(usize) -> usize,
    party_id: PartyID,
    io_ctx: &mut IoContext<N>,
    swap: bool,
) -> eyre::Result<Vec<Rep3RingShare<Bit>>>
where
    Standard: Distribution<T::Half>,
{
    let mut p = vec![Rep3RingShare::<T::Half>::zero_share(); n];
    let mut g = vec![Rep3RingShare::<T::Half>::zero_share(); n];

    // Shared-y elements that need and_many for generate bits
    let mut shared_a: Vec<Rep3RingShare<T::Half>> = Vec::new();
    let mut shared_b: Vec<Rep3RingShare<T::Half>> = Vec::new();
    let mut shared_pos: Vec<usize> = Vec::new();

    for j in 0..n {
        let i = orig(j);

        // Extract y as public constant or shared value
        let y_val: Either<RingElement<T::Half>, Rep3RingShare<T::Half>> = match right {
            MixedBatch::Public(y_pubs) => Either::Public(RingElement(
                T::Half::try_from(y_pubs[i] as u128).unwrap_or_else(|_| unreachable!()),
            )),
            MixedBatch::Shared(ys) => Either::Shared(ys[i]),
            MixedBatch::Mixed(mixed) => match &mixed[i] {
                Either::Public(yp) => Either::Public(RingElement(
                    T::Half::try_from(*yp as u128).unwrap_or_else(|_| unreachable!()),
                )),
                Either::Shared(y) => Either::Shared(*y),
            },
        };

        match y_val {
            Either::Public(y_const) => {
                if swap {
                    // y_const - x: a=y_const, b'=~x
                    let x_neg = !xs[j];
                    p[j] = rep3_ring::binary::xor_public(&x_neg, &y_const, party_id);
                    g[j] = &x_neg & &y_const;
                } else {
                    // x - y_const: a=x, b'=~y_const
                    let y_neg = !y_const;
                    p[j] = rep3_ring::binary::xor_public(&xs[j], &y_neg, party_id);
                    g[j] = &xs[j] & &y_neg;
                }
            }
            Either::Shared(y_shared) => {
                if swap {
                    // y - x: a=y, b'=~x
                    let x_neg = !xs[j];
                    p[j] = y_shared ^ x_neg;
                    shared_a.push(y_shared);
                    shared_b.push(x_neg);
                } else {
                    // x - y: a=x, b'=~y
                    let y_neg = !y_shared;
                    p[j] = xs[j] ^ y_neg;
                    shared_a.push(xs[j]);
                    shared_b.push(y_neg);
                }
                shared_pos.push(j);
            }
        }
    }

    // AND for shared-y elements only (1 communication round, smaller batch)
    if !shared_a.is_empty() {
        let and_results = rep3_ring::binary::and_many::<T::Half, _>(&shared_a, &shared_b, io_ctx)?;
        for (k, &pos) in shared_pos.iter().enumerate() {
            g[pos] = and_results[k];
        }
    }

    // carry_in = 1: g ^= p & 1
    let one = RingElement(T::Half::try_from(1u128).unwrap_or_else(|_| unreachable!()));
    for (gi, pi) in g.iter_mut().zip(p.iter()) {
        *gi ^= *pi & one;
    }

    // Kogge-Stone carry tree (log2(K) rounds)
    let carries = rep3_ring::arithmetic::kogge_stone_carries_many::<T::Half, _>(p, g, io_ctx)?;
    Ok(carries)
}

// ---------------------------------------------------------------------------
// Named eval functions
// ---------------------------------------------------------------------------

/// Eq suffix: XOR (local for both public and shared y) then is_zero_many (MPC).
pub(crate) fn eval_eq<T, F, N>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    orig: impl Fn(usize) -> usize,
    party_id: PartyID,
    io_ctx: &mut IoContext<N>,
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) -> eyre::Result<()>
where
    T: Uninterleavable + AsPrimitive<Bit>,
    Standard: Distribution<T> + Distribution<T::Half>,
    T::Half: AsPrimitive<T> + AsPrimitive<Bit>,
    F: JoltField,
    N: Rep3Network,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    let diff: Vec<Rep3RingShare<T::Half>> = xs
        .iter()
        .enumerate()
        .map(|(j, x)| {
            let i = orig(j);
            match right {
                MixedBatch::Public(y_pubs) => {
                    let mask = RingElement(
                        T::Half::try_from(y_pubs[i] as u128).unwrap_or_else(|_| unreachable!()),
                    );
                    rep3_ring::binary::xor_public(x, &mask, party_id)
                }
                MixedBatch::Shared(ys) => *x ^ ys[i],
                MixedBatch::Mixed(mixed) => match &mixed[i] {
                    Either::Public(yp) => {
                        let mask = RingElement(
                            T::Half::try_from(*yp as u128).unwrap_or_else(|_| unreachable!()),
                        );
                        rep3_ring::binary::xor_public(x, &mask, party_id)
                    }
                    Either::Shared(y) => *x ^ *y,
                },
            }
        })
        .collect();
    let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&diff, io_ctx)?;
    out.extend_bitinject(indices_iter, eq_bits.into_iter());
    Ok(())
}

/// RightOperandIsZero suffix: 3-way MixedBatch dispatch.
pub(crate) fn eval_right_is_zero<T, F, N>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    orig: impl Fn(usize) -> usize,
    io_ctx: &mut IoContext<N>,
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) -> eyre::Result<()>
where
    T: Uninterleavable,
    T::Half: B2ABucketExtend,
    Standard: Distribution<T::Half>,
    F: JoltField,
    N: Rep3Network,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    match right {
        MixedBatch::Public(y_pubs) => {
            out.extend_ready(
                indices_iter,
                (0..n).map(|j| {
                    let val = if y_pubs[orig(j)] == 0 { 1u64 } else { 0u64 };
                    Rep3Value::Public(F::from_u64(val))
                }),
            );
        }
        MixedBatch::Shared(ys) => {
            let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(ys, io_ctx)?;
            out.extend_bitinject(indices_iter, eq_bits.into_iter());
        }
        MixedBatch::Mixed(mixed) => {
            let mut mpc_idx = Vec::new();
            let mut mpc_ys = Vec::new();
            for (j, _) in xs.iter().enumerate() {
                let i = orig(j);
                match &mixed[i] {
                    Either::Public(yp) => {
                        let val = if *yp == 0 { 1u64 } else { 0u64 };
                        out.extend_ready(
                            std::iter::once(base + i),
                            std::iter::once(Rep3Value::Public(F::from_u64(val))),
                        );
                    }
                    Either::Shared(y) => {
                        mpc_idx.push(base + i);
                        mpc_ys.push(*y);
                    }
                }
            }
            if !mpc_ys.is_empty() {
                let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&mpc_ys, io_ctx)?;
                out.extend_bitinject(mpc_idx.into_iter(), eq_bits.into_iter());
            }
        }
    }
    Ok(())
}

/// DivByZero suffix: is_zero(divisor) AND is_all_ones(quotient).
pub(crate) fn eval_div_by_zero<T, F, N>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    suffix_len: usize,
    party_id: PartyID,
    io_ctx: &mut IoContext<N>,
    base: usize,
    orig: impl Fn(usize) -> usize,
    out: &mut SuffixFutureBatch<F>,
) -> eyre::Result<()>
where
    T: Uninterleavable,
    T::Half: B2ABucketExtend,
    Standard: Distribution<T::Half>,
    F: JoltField,
    N: Rep3Network,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    let ys = right.as_shared();
    let quotient_bits = suffix_len / 2;
    let all_ones_val: u128 = if quotient_bits >= T::Half::K {
        (1u128 << T::Half::K) - 1
    } else {
        (1u128 << quotient_bits) - 1
    };
    let all_ones_mask =
        RingElement(T::Half::try_from(all_ones_val).unwrap_or_else(|_| unreachable!()));
    let q_xor: Vec<Rep3RingShare<T::Half>> = ys
        .iter()
        .map(|q| rep3_ring::binary::xor_public(q, &all_ones_mask, party_id))
        .collect();
    // Batch both is_zero_many calls into one (halves rounds)
    let split = xs.len();
    let mut combined = Vec::with_capacity(split + q_xor.len());
    combined.extend_from_slice(xs);
    combined.extend_from_slice(&q_xor);
    let combined_result = rep3_ring::binary::is_zero_many::<T::Half, _>(&combined, io_ctx)?;
    let (divisor_zero, quotient_all_ones) = combined_result.split_at(split);
    let result = rep3_ring::binary::and_many::<Bit, _>(divisor_zero, quotient_all_ones, io_ctx)?;
    out.extend_bitinject(indices_iter, result.into_iter());
    Ok(())
}

/// ChangeDivisor suffix: is_all_ones(y) AND is_zero(x).
pub(crate) fn eval_change_divisor<T, F, N>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    suffix_len: usize,
    party_id: PartyID,
    io_ctx: &mut IoContext<N>,
    base: usize,
    orig: impl Fn(usize) -> usize,
    out: &mut SuffixFutureBatch<F>,
) -> eyre::Result<()>
where
    T: Uninterleavable,
    T::Half: B2ABucketExtend,
    Standard: Distribution<T::Half>,
    F: JoltField,
    N: Rep3Network,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    let ys = right.as_shared();
    let y_len = suffix_len / 2;
    let all_ones_val: u128 = if y_len >= T::Half::K {
        (1u128 << T::Half::K) - 1
    } else {
        (1u128 << y_len) - 1
    };
    let all_ones_mask =
        RingElement(T::Half::try_from(all_ones_val).unwrap_or_else(|_| unreachable!()));
    let y_xor: Vec<Rep3RingShare<T::Half>> = ys
        .iter()
        .map(|y| rep3_ring::binary::xor_public(y, &all_ones_mask, party_id))
        .collect();
    // Batch both is_zero_many calls into one (halves rounds)
    let split = y_xor.len();
    let mut combined = Vec::with_capacity(split + xs.len());
    combined.extend_from_slice(&y_xor);
    combined.extend_from_slice(xs);
    let combined_result = rep3_ring::binary::is_zero_many::<T::Half, _>(&combined, io_ctx)?;
    let (y_eq_all_ones, x_eq_zero) = combined_result.split_at(split);
    let result = rep3_ring::binary::and_many::<Bit, _>(y_eq_all_ones, x_eq_zero, io_ctx)?;
    out.extend_bitinject(indices_iter, result.into_iter());
    Ok(())
}

/// ChangeDivisorW suffix: W-variant, operates on u32.
pub(crate) fn eval_change_divisor_w<T, F, N>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    suffix_len: usize,
    party_id: PartyID,
    io_ctx: &mut IoContext<N>,
    base: usize,
    orig: impl Fn(usize) -> usize,
    out: &mut SuffixFutureBatch<F>,
) -> eyre::Result<()>
where
    T: Uninterleavable,
    T::Half: B2ABucketExtend,
    F: JoltField,
    N: Rep3Network,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    let ys = right.as_shared();
    let xs32: Vec<Rep3RingShare<u32>> = xs.iter().map(|x| to_u32_share(*x)).collect();
    let ys32: Vec<Rep3RingShare<u32>> = ys.iter().map(|y| to_u32_share(*y)).collect();
    let y_len = (suffix_len / 2).min(XLEN / 2);
    let all_ones_mask = RingElement(if y_len >= 32 {
        u32::MAX
    } else {
        (1u32 << y_len) - 1
    });
    let y_xor: Vec<Rep3RingShare<u32>> = ys32
        .iter()
        .map(|y| rep3_ring::binary::xor_public(y, &all_ones_mask, party_id))
        .collect();
    // Batch both is_zero_many calls into one (halves rounds)
    let split = y_xor.len();
    let mut combined = Vec::with_capacity(split + xs32.len());
    combined.extend_from_slice(&y_xor);
    combined.extend_from_slice(&xs32);
    let combined_result = rep3_ring::binary::is_zero_many::<u32, _>(&combined, io_ctx)?;
    let (y_eq_all_ones, x_eq_zero) = combined_result.split_at(split);
    let result = rep3_ring::binary::and_many::<Bit, _>(y_eq_all_ones, x_eq_zero, io_ctx)?;
    out.extend_bitinject(indices_iter, result.into_iter());
    Ok(())
}
