//! MPC suffix evaluation for the ReadRaf sumcheck.
//!
//! Each vanilla `Suffixes` variant evaluates `suffix_mle(LookupBits) -> u64`.
//! This module provides the MPC equivalent: given a batch of secret
//! `Rep3RingShare<T>` suffix bits (one per cycle), produce a batch of
//! `FutureRep3Ring<T::Half, Rep3PrimeFieldShare<F>>` suffix futures.
//!
//! The ring type `T: Uninterleavable` is chosen per-phase to be the smallest
//! ring that fits the suffix_len bits, minimising EdaBit alpha count and
//! communication during Protocol Π₂ B2A conversion.
//!
//! Lookup indices from witness gen are already in the **binary (XOR) domain**
//! (fulfilled via `RingA2B`). All local operations (mask, shift, XOR, uninterleave)
//! preserve the binary domain. Interactive operations (AND, OR, is_zero, ge)
//! also operate in binary domain. Field conversion is deferred via `FutureRep3Ring`
//! and batched by the caller using `fulfill_batched`.

use crate::field::JoltField;
use crate::utils::types::rep3_value::Rep3Value;
use crate::utils::types::Either;
use jolt2_common::constants::{LookupIndexInt, XLEN};
use jolt_core::utils::interleave_bits;
use jolt_core::utils::lookup_bits::LookupBits;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::lookup_table::suffixes::Suffixes;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::downcast;
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self as rep3_ring, Rep3RingShare};
use num_traits::AsPrimitive;
use rand::distributions::Standard;
use rand::prelude::Distribution;

mod bitwise;
mod comparators;
pub mod future;
pub use future::{B2ABucketExtend, SuffixFutureBatch};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Types whose binary-domain shares can be uninterleaved into half-sized shares.
///
/// A suffix of `suffix_len` interleaved bits packs two operands (x, y) in
/// alternating bit positions. Uninterleaving extracts x and y into separate
/// `T::Half`-sized shares, each holding `suffix_len/2` bits.
pub trait Uninterleavable: IntRing2k + B2ABucketExtend + AsPrimitive<Self::Half> {
    type Half: IntRing2k + B2ABucketExtend + AsPrimitive<Self>;
    fn uninterleave(
        s: Rep3RingShare<Self>,
    ) -> (Rep3RingShare<Self::Half>, Rep3RingShare<Self::Half>)
    where
        Standard: Distribution<Self::Half>;
}

/// Morton-decode uninterleave: ~6 mask+shift+OR steps per component instead of
/// O(half_k) iterations. Applies decode to each share component (.a.0, .b.0)
/// independently — purely local, zero communication.
///
/// Reference: `jolt_core::utils::uninterleave_bits` in vanilla Jolt.
macro_rules! impl_uninterleavable {
    ($full:ty, $half:ty, $even_mask:expr, [$(($shift:expr, $mask:expr)),+ $(,)?]) => {
        impl Uninterleavable for $full {
            type Half = $half;
            #[inline(always)]
            fn uninterleave(s: Rep3RingShare<Self>) -> (Rep3RingShare<$half>, Rep3RingShare<$half>)
            where
                Standard: Distribution<$half>,
            {
                #[inline(always)]
                fn decode(val: $full) -> ($half, $half) {
                    let mut x = (val >> 1) & $even_mask;
                    let mut y = val & $even_mask;
                    $( x = (x | (x >> $shift)) & $mask;
                       y = (y | (y >> $shift)) & $mask; )+
                    (x as $half, y as $half)
                }
                let (xa, ya) = decode(s.a.0);
                let (xb, yb) = decode(s.b.0);
                (Rep3RingShare { a: RingElement(xa), b: RingElement(xb) },
                 Rep3RingShare { a: RingElement(ya), b: RingElement(yb) })
            }
        }
    };
}

impl_uninterleavable!(
    u16,
    u8,
    0x5555u16,
    [(1, 0x3333u16), (2, 0x0F0Fu16), (4, 0x00FFu16)]
);
impl_uninterleavable!(
    u32,
    u16,
    0x5555_5555u32,
    [
        (1, 0x3333_3333u32),
        (2, 0x0F0F_0F0Fu32),
        (4, 0x00FF_00FFu32),
        (8, 0x0000_FFFFu32)
    ]
);
impl_uninterleavable!(
    u64,
    u32,
    0x5555_5555_5555_5555u64,
    [
        (1, 0x3333_3333_3333_3333u64),
        (2, 0x0F0F_0F0F_0F0F_0F0Fu64),
        (4, 0x00FF_00FF_00FF_00FFu64),
        (8, 0x0000_FFFF_0000_FFFFu64),
        (16, 0x0000_0000_FFFF_FFFFu64)
    ]
);
impl_uninterleavable!(
    u128,
    u64,
    0x5555_5555_5555_5555_5555_5555_5555_5555u128,
    [
        (1, 0x3333_3333_3333_3333_3333_3333_3333_3333u128),
        (2, 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128),
        (4, 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128),
        (8, 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128),
        (16, 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128),
        (32, 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFFu128)
    ]
);

/// Per-cycle data that may be public or shared.
pub enum MixedBatch<P, S: IntRing2k> {
    /// All cycles have this value public.
    Public(Vec<P>),
    /// All cycles have this value shared.
    Shared(Vec<Rep3RingShare<S>>),
    /// Mix of public and shared cycles.
    Mixed(Vec<Either<P, Rep3RingShare<S>>>),
}

impl<P: Copy, S: IntRing2k> MixedBatch<P, S> {
    pub fn len(&self) -> usize {
        match self {
            Self::Public(v) => v.len(),
            Self::Shared(v) => v.len(),
            Self::Mixed(v) => v.len(),
        }
    }

    pub fn as_public(&self) -> &[P] {
        match self {
            Self::Public(v) => v,
            _ => panic!("expected Public"),
        }
    }

    pub fn as_shared(&self) -> &[Rep3RingShare<S>] {
        match self {
            Self::Shared(v) => v,
            _ => panic!("expected Shared"),
        }
    }

    pub fn as_mixed(&self) -> &[Either<P, Rep3RingShare<S>>] {
        match self {
            Self::Mixed(v) => v,
            _ => panic!("expected Mixed"),
        }
    }

    pub fn into_shared(self) -> Vec<Rep3RingShare<S>> {
        match self {
            Self::Shared(v) => v,
            _ => panic!("expected Shared"),
        }
    }

    pub fn is_public(&self) -> bool {
        matches!(self, Self::Public(_))
    }

    /// Classify a vec of `Either<P, Rep3RingShare<S>>` into Public/Shared/Mixed.
    pub fn classify(entries: Vec<Either<P, Rep3RingShare<S>>>) -> Self {
        let mut all_public = true;
        let mut all_shared = true;
        for e in &entries {
            match e {
                Either::Public(_) => all_shared = false,
                Either::Shared(_) => all_public = false,
            }
        }
        if all_public {
            MixedBatch::Public(
                entries
                    .into_iter()
                    .map(|e| match e {
                        Either::Public(p) => p,
                        _ => unreachable!(),
                    })
                    .collect(),
            )
        } else if all_shared {
            MixedBatch::Shared(
                entries
                    .into_iter()
                    .map(|e| match e {
                        Either::Shared(s) => s,
                        _ => unreachable!(),
                    })
                    .collect(),
            )
        } else {
            MixedBatch::Mixed(entries)
        }
    }
}

/// Per-table suffix bits, either interleaved or split into left/right operands.
///
/// Parameterized by the full ring T (chosen per-phase by suffix_len).
/// Interleaved variant stores `Rep3RingShare<T>`, Uninterleaved stores `Rep3RingShare<T::Half>`.
pub enum SuffixBitsBatch<T: Uninterleavable> {
    /// Separate left (x) and right (y) operands in T::Half ring.
    /// Used by tables whose suffixes all operate on uninterleaved operands
    /// (And, Or, Xor, Eq, LessThan, shift tables, etc.)
    Uninterleaved(MixedBatch<u64, T::Half>, MixedBatch<u64, T::Half>),
    /// Raw interleaved bits in full ring T.
    /// Used by tables whose suffixes operate on raw interleaved bits
    /// (UpperWord, LowerWord, LowerHalfWord, Rev8W, Pow2, etc.)
    Interleaved(MixedBatch<LookupIndexInt, T>),
}

impl<T: Uninterleavable> SuffixBitsBatch<T> {
    pub fn len(&self) -> usize {
        match self {
            Self::Uninterleaved(left, _) => left.len(),
            Self::Interleaved(bits) => bits.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Evaluate a single suffix for one table's cycles, pushing results into the batch.
///
/// This replaces the old `evaluate_suffix_mle_batched` + `split_and_eval` approach.
/// Instead of evaluating each suffix over ALL active cycles, this evaluates a suffix
/// only for the cycles belonging to a specific table.
///
/// `data`: per-table operand data (Interleaved or Uninterleaved)
/// `base`: base index in the output batch (from `out.reserve(n)`)
/// `out`: accumulator for all suffix results
pub fn evaluate_suffix_for_table<T, F, N>(
    suffix: &Suffixes,
    data: &SuffixBitsBatch<T>,
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
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
    let n = data.len();
    debug_assert!(suffix_len > 0);

    // One suffix: constant 1 for all cycles
    if matches!(suffix, Suffixes::One) {
        out.extend_ready(
            base..base + n,
            std::iter::repeat(Rep3Value::Public(F::one())).take(n),
        );
        return Ok(());
    }

    match data {
        SuffixBitsBatch::Uninterleaved(left, right) => eval_suffix_uninterleaved::<T, F, N>(
            suffix, left, right, suffix_len, io_ctx, party_id, base, out,
        ),
        SuffixBitsBatch::Interleaved(bits) => eval_suffix_interleaved::<T, F, N>(
            suffix, bits, suffix_len, io_ctx, party_id, base, out,
        ),
    }
}

/// Check if a set of suffixes requires interleaved data (full ring T).
///
/// Tables whose non-One suffixes include any of the interleaved-only suffixes
/// (UpperWord, LowerWord, LowerHalfWord, Rev8W, Pow2, Pow2W,
/// SignExtensionUpperHalf, OverflowBitsZero) need `SuffixBitsBatch::Interleaved`.
/// All other tables use `SuffixBitsBatch::Uninterleaved`.
pub fn table_uses_interleaved_data(suffixes: &[Suffixes]) -> bool {
    suffixes.iter().any(|s| {
        if matches!(
            s,
            Suffixes::UpperWord
                | Suffixes::LowerWord
                | Suffixes::LowerHalfWord
                | Suffixes::Pow2
                | Suffixes::Pow2W
                | Suffixes::SignExtensionUpperHalf
                | Suffixes::OverflowBitsZero
        ) {
            return true;
        }
        #[cfg(feature = "rv64")]
        if matches!(s, Suffixes::Rev8W) {
            return true;
        }
        false
    })
}

/// Returns the EdaBit ring bit-width consumed by this suffix, or `None` if
/// the suffix produces Ready or BitInject results (no EdaBits needed).
///
/// `t_k`: bit-width of full ring T (e.g. 128, 64, 32, 16)
/// `t_half_k`: bit-width of half ring T::Half (e.g. 64, 32, 16, 8)
///
/// The returned value is the bit-width of the ring to use for B2A conversion.
pub fn suffix_edabit_ring_bits(suffix: &Suffixes, t_k: usize, t_half_k: usize) -> Option<usize> {
    match suffix {
        // --- B2A(T::Half): standard bitwise ops and value extraction ---
        Suffixes::And
        | Suffixes::NotAnd
        | Suffixes::Xor
        | Suffixes::Or
        | Suffixes::RightOperand
        | Suffixes::XorRot16
        | Suffixes::XorRot24
        | Suffixes::XorRot32
        | Suffixes::XorRot63 => Some(t_half_k),

        // --- B2A(T::Half): shift suffixes with public right operand ---
        Suffixes::RightShift | Suffixes::LeftShift => Some(t_half_k),

        // --- B2A(T::Half): W-variant shifts with public right operand ---
        Suffixes::RightShiftW | Suffixes::LeftShiftW => Some(t_half_k),

        // --- B2A(T::Half or u32): W-variant ops ---
        Suffixes::RightOperandW
        | Suffixes::XorRotW7
        | Suffixes::XorRotW8
        | Suffixes::XorRotW12
        | Suffixes::XorRotW16 => {
            if t_half_k >= 32 {
                Some(t_half_k)
            } else {
                Some(32)
            }
        }

        #[cfg(feature = "rv64")]
        Suffixes::Rev8W => {
            if t_half_k >= 32 {
                Some(t_half_k)
            } else {
                Some(32)
            }
        }

        // --- B2A(T::Half or T): interleaved extraction ---
        Suffixes::UpperWord => {
            if t_k <= XLEN {
                None // All zero → Ready
            } else {
                let result_bits = t_k - XLEN;
                if result_bits <= t_half_k {
                    Some(t_half_k)
                } else {
                    Some(t_k)
                }
            }
        }
        Suffixes::LowerWord => {
            let result_bits = XLEN.min(t_k);
            if result_bits <= t_half_k {
                Some(t_half_k)
            } else {
                Some(t_k)
            }
        }
        Suffixes::LowerHalfWord => {
            let result_bits = (XLEN / 2).min(t_k);
            if result_bits <= t_half_k {
                Some(t_half_k)
            } else {
                Some(t_k)
            }
        }

        // --- Ready: no EdaBits ---
        Suffixes::One
        | Suffixes::Pow2
        | Suffixes::Pow2W
        | Suffixes::SignExtension
        | Suffixes::SignExtensionUpperHalf
        | Suffixes::SignExtensionRightOperand
        | Suffixes::RightShiftHelper
        | Suffixes::RightShiftPadding
        | Suffixes::RightShiftWHelper
        | Suffixes::LeftShiftWHelper => None,

        // --- BitInject (daBits, not edaBits) ---
        Suffixes::Lsb
        | Suffixes::TwoLsb
        | Suffixes::LessThan
        | Suffixes::GreaterThan
        | Suffixes::Eq
        | Suffixes::LeftOperandIsZero
        | Suffixes::RightOperandIsZero
        | Suffixes::DivByZero
        | Suffixes::OverflowBitsZero
        | Suffixes::ChangeDivisor
        | Suffixes::ChangeDivisorW => None,
    }
}

/// Returns true if this suffix variant produces B2A futures (consuming edaBits).
/// Convenience wrapper around `suffix_edabit_ring_bits`.
pub fn suffix_uses_b2a_edabits(suffix: &Suffixes) -> bool {
    suffix_edabit_ring_bits(suffix, 128, 64).is_some()
}

// ---------------------------------------------------------------------------
// Private evaluation functions
// ---------------------------------------------------------------------------

/// Handle public entries for an interleaved suffix: compute via suffix_mle.
/// Returns the positions and shares for entries that need MPC evaluation.
fn split_interleaved_public<T: Uninterleavable, F: JoltField>(
    suffix: &Suffixes,
    bits: &MixedBatch<LookupIndexInt, T>,
    suffix_len: usize,
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) -> (Vec<usize>, Vec<Rep3RingShare<T>>)
where
    Standard: Distribution<T>,
{
    let mask = if suffix_len >= LookupIndexInt::BITS as usize {
        LookupIndexInt::MAX
    } else {
        ((1 as LookupIndexInt) << suffix_len) - 1
    };

    match bits {
        MixedBatch::Shared(shares) => {
            let positions: Vec<usize> = (0..shares.len()).collect();
            (positions, shares.clone())
        }
        MixedBatch::Public(pubs) => {
            for (i, &p) in pubs.iter().enumerate() {
                let val = suffix.suffix_mle::<XLEN>(LookupBits::new((p & mask) as u128, suffix_len));
                out.extend_ready(
                    std::iter::once(base + i),
                    std::iter::once(Rep3Value::Public(F::from_u64(val))),
                );
            }
            (Vec::new(), Vec::new())
        }
        MixedBatch::Mixed(mixed) => {
            let mut positions = Vec::new();
            let mut shares = Vec::new();
            for (i, entry) in mixed.iter().enumerate() {
                match entry {
                    Either::Public(p) => {
                        let val = suffix.suffix_mle::<XLEN>(LookupBits::new((*p & mask) as u128, suffix_len));
                        out.extend_ready(
                            std::iter::once(base + i),
                            std::iter::once(Rep3Value::Public(F::from_u64(val))),
                        );
                    }
                    Either::Shared(s) => {
                        positions.push(i);
                        shares.push(*s);
                    }
                }
            }
            (positions, shares)
        }
    }
}

/// Evaluate an uninterleaved suffix, pushing results into SuffixFutureBatch.
///
/// Handles public left operands inline (promoting to trivial or resolving via suffix_mle).
/// For the common case (left=Shared), borrows slices directly with zero allocation.
fn eval_suffix_uninterleaved<T, F, N>(
    suffix: &Suffixes,
    left: &MixedBatch<u64, T::Half>,
    right: &MixedBatch<u64, T::Half>,
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
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
    // Handle all-public left: both operands must be public → suffix_mle → Ready
    if left.is_public() {
        let x_pubs = left.as_public();
        let y_pubs = right.as_public(); // left=public implies right=public
        for (i, (&xp, &yp)) in x_pubs.iter().zip(y_pubs.iter()).enumerate() {
            let interleaved = interleave_bits(xp, yp);
            let val = suffix.suffix_mle::<XLEN>(LookupBits::new(interleaved, suffix_len));
            out.extend_ready(
                std::iter::once(base + i),
                std::iter::once(Rep3Value::Public(F::from_u64(val))),
            );
        }
        return Ok(());
    }

    // Extract shared xs. For Shared left, borrow directly (Cow::Borrowed).
    // For Mixed left, filter (pub, pub) pairs as Ready and collect remaining shared xs.
    use std::borrow::Cow;

    let (xs_cow, positions): (Cow<'_, [Rep3RingShare<T::Half>]>, Option<Vec<usize>>) = match left {
        MixedBatch::Shared(xs) => (Cow::Borrowed(xs.as_slice()), None),
        MixedBatch::Mixed(xs_mixed) => {
            let mut pos = Vec::new();
            let mut shared_xs = Vec::new();
            for (i, x_entry) in xs_mixed.iter().enumerate() {
                match x_entry {
                    Either::Public(xp) => {
                        let yp = match right {
                            MixedBatch::Public(y_pubs) => y_pubs[i],
                            MixedBatch::Mixed(ys_mixed) => match ys_mixed[i] {
                                Either::Public(yp) => yp,
                                Either::Shared(_) => unreachable!("left=public but right=shared"),
                            },
                            MixedBatch::Shared(_) => {
                                unreachable!("left=mixed but right=all-shared")
                            }
                        };
                        let interleaved = interleave_bits(*xp, yp);
                        let val =
                            suffix.suffix_mle::<XLEN>(LookupBits::new(interleaved, suffix_len));
                        out.extend_ready(
                            std::iter::once(base + i),
                            std::iter::once(Rep3Value::Public(F::from_u64(val))),
                        );
                    }
                    Either::Shared(s) => {
                        pos.push(i);
                        shared_xs.push(*s);
                    }
                }
            }
            (Cow::Owned(shared_xs), Some(pos))
        }
        MixedBatch::Public(_) => unreachable!("handled above"),
    };

    let xs: &[Rep3RingShare<T::Half>] = &xs_cow;
    let pos_slice = positions.as_deref();
    if xs.is_empty() {
        return Ok(());
    }
    let n = xs.len();

    // Map local index j → original index in the batch.
    // None positions = all entries (identity), Some = filtered subset.
    let orig = |j: usize| -> usize { pos_slice.map_or(j, |p| p[j]) };
    let indices_iter = (0..n).map(|j| base + orig(j));

    match suffix {
        Suffixes::One => unreachable!("handled above"),

        // --- B2A(H): bitwise ops (right can be public or shared) ---
        Suffixes::And => {
            bitwise::eval_bitwise_mixed::<T, F, N>(
                xs,
                right,
                base,
                orig,
                io_ctx,
                out,
                |x, yp| {
                    let mask = RingElement(
                        T::Half::try_from(yp as u128).unwrap_or_else(|_| unreachable!()),
                    );
                    *x & mask
                },
                |xs, ys, ctx| rep3_ring::binary::and_many::<T::Half, _>(xs, ys, ctx),
            )?;
        }
        Suffixes::NotAnd => {
            bitwise::eval_bitwise_mixed::<T, F, N>(
                xs,
                right,
                base,
                &orig,
                io_ctx,
                out,
                |x, yp| {
                    let mask = RingElement(
                        !T::Half::try_from(yp as u128).unwrap_or_else(|_| unreachable!()),
                    );
                    *x & mask
                },
                |xs, ys, ctx| {
                    let not_ys: Vec<_> = ys.iter().map(|y| !y).collect();
                    rep3_ring::binary::and_many::<T::Half, _>(xs, &not_ys, ctx)
                },
            )?;
        }
        Suffixes::Xor => {
            bitwise::eval_xor::<T, F, N>(xs, right, base, &orig, party_id, out);
        }
        Suffixes::Or => {
            bitwise::eval_bitwise_mixed::<T, F, N>(
                xs,
                right,
                base,
                &orig,
                io_ctx,
                out,
                |x, yp| {
                    let mask = RingElement(
                        T::Half::try_from(yp as u128).unwrap_or_else(|_| unreachable!()),
                    );
                    rep3_ring::binary::xor_public(&(*x & RingElement(!mask.0)), &mask, party_id)
                },
                |xs, ys, ctx| rep3_ring::binary::or_many::<T::Half, _>(xs, ys, ctx),
            )?;
        }

        // --- B2A(H): value extraction (right can be public or shared) ---
        Suffixes::RightOperand => match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_ready(
                    indices_iter,
                    (0..n).map(|j| Rep3Value::Public(F::from_u64(y_pubs[orig(j)]))),
                );
            }
            MixedBatch::Shared(ys) => {
                out.extend_b2a_ring::<T::Half>(indices_iter, ys.iter().copied());
            }
            MixedBatch::Mixed(mixed) => {
                for (j, _) in xs.iter().enumerate() {
                    let i = orig(j);
                    match &mixed[i] {
                        Either::Public(yp) => {
                            out.extend_ready(
                                std::iter::once(base + i),
                                std::iter::once(Rep3Value::Public(F::from_u64(*yp))),
                            );
                        }
                        Either::Shared(y) => {
                            out.extend_b2a_ring::<T::Half>(
                                std::iter::once(base + i),
                                std::iter::once(*y),
                            );
                        }
                    }
                }
            }
        },
        Suffixes::RightOperandW => {
            eval_right_operand_w::<T, F>(xs, right, base, &orig, out);
        }
        Suffixes::Lsb => match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_ready(
                    indices_iter,
                    (0..n).map(|j| Rep3Value::Public(F::from_u64(y_pubs[orig(j)] & 1))),
                );
            }
            MixedBatch::Shared(ys) => {
                out.extend_bitinject(
                    indices_iter,
                    ys.iter().map(|y| downcast::<T::Half, Bit>(*y)),
                );
            }
            MixedBatch::Mixed(mixed) => {
                for (j, _) in xs.iter().enumerate() {
                    let i = orig(j);
                    match &mixed[i] {
                        Either::Public(yp) => {
                            out.extend_ready(
                                std::iter::once(base + i),
                                std::iter::once(Rep3Value::Public(F::from_u64(*yp & 1))),
                            );
                        }
                        Either::Shared(y) => {
                            out.extend_bitinject(
                                std::iter::once(base + i),
                                std::iter::once(downcast::<T::Half, Bit>(*y)),
                            );
                        }
                    }
                }
            }
        },

        // --- BitInject: comparisons ---
        // Uses ge_many_mixed: computes (p, g) locally for public y elements,
        // and_many only for shared y elements, then a single Kogge-Stone tree.
        Suffixes::LessThan => {
            // lt(x, y) = !(x >= y)
            let ge_bits =
                comparators::ge_many_mixed::<T, _>(xs, right, n, &orig, party_id, io_ctx, false)?;
            let lt_bits: Vec<_> = ge_bits.iter().map(|b| !b).collect();
            out.extend_bitinject(indices_iter, lt_bits.into_iter());
        }
        Suffixes::GreaterThan => {
            // gt(x, y) = !(y >= x)
            let ge_bits =
                comparators::ge_many_mixed::<T, _>(xs, right, n, &orig, party_id, io_ctx, true)?;
            let gt_bits: Vec<_> = ge_bits.iter().map(|b| !b).collect();
            out.extend_bitinject(indices_iter, gt_bits.into_iter());
        }
        Suffixes::Eq => {
            comparators::eval_eq::<T, F, N>(xs, right, &orig, party_id, io_ctx, base, out)?;
        }
        Suffixes::LeftOperandIsZero => {
            // Only uses xs — right operand irrelevant
            let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(xs, io_ctx)?;
            out.extend_bitinject(indices_iter, eq_bits.into_iter());
        }
        Suffixes::RightOperandIsZero => {
            comparators::eval_right_is_zero::<T, F, N>(xs, right, &orig, io_ctx, base, out)?;
        }

        // --- BitInject: division checks (right always shared — division tables) ---
        Suffixes::DivByZero => {
            comparators::eval_div_by_zero::<T, F, N>(
                xs, right, suffix_len, party_id, io_ctx, base, &orig, out,
            )?;
        }
        Suffixes::TwoLsb => {
            let ys = right.as_shared();
            let x_lsb: Vec<Rep3RingShare<Bit>> =
                xs.iter().map(|x| downcast::<T::Half, Bit>(*x)).collect();
            let y_lsb: Vec<Rep3RingShare<Bit>> =
                ys.iter().map(|y| downcast::<T::Half, Bit>(*y)).collect();
            let x_or_y = rep3_ring::binary::or_many::<Bit, _>(&x_lsb, &y_lsb, io_ctx)?;
            let result: Vec<_> = x_or_y.iter().map(|b| !b).collect();
            out.extend_bitinject(indices_iter, result.into_iter());
        }
        Suffixes::ChangeDivisor => {
            comparators::eval_change_divisor::<T, F, N>(
                xs, right, suffix_len, party_id, io_ctx, base, &orig, out,
            )?;
        }
        Suffixes::ChangeDivisorW => {
            comparators::eval_change_divisor_w::<T, F, N>(
                xs, right, suffix_len, party_id, io_ctx, base, &orig, out,
            )?;
        }

        // --- Ready: sign extension (right always public for shift tables) ---
        Suffixes::SignExtension => {
            let y_pubs = right.as_public();
            out.extend_ready(
                indices_iter,
                (0..n).map(|j| {
                    let val = compute_sign_extension_from_mask(y_pubs[orig(j)], suffix_len);
                    Rep3Value::Public(F::from_u64(val))
                }),
            );
        }
        Suffixes::SignExtensionRightOperand => {
            eval_sign_extension_right_operand::<T, F>(xs, right, suffix_len, base, &orig, out);
        }

        // --- B2A(H): XOR-rotate (right always shared — hash tables) ---
        Suffixes::XorRot16 => {
            let ys = right.as_shared();
            bitwise::eval_xor_rot_uninterleaved::<16, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRot24 => {
            let ys = right.as_shared();
            bitwise::eval_xor_rot_uninterleaved::<24, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRot32 => {
            let ys = right.as_shared();
            bitwise::eval_xor_rot_uninterleaved::<32, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRot63 => {
            let ys = right.as_shared();
            bitwise::eval_xor_rot_uninterleaved::<63, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRotW7 => {
            let ys = right.as_shared();
            bitwise::eval_xor_rot_w_uninterleaved::<7, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRotW8 => {
            let ys = right.as_shared();
            bitwise::eval_xor_rot_w_uninterleaved::<8, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRotW12 => {
            let ys = right.as_shared();
            bitwise::eval_xor_rot_w_uninterleaved::<12, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRotW16 => {
            let ys = right.as_shared();
            bitwise::eval_xor_rot_w_uninterleaved::<16, T, F>(xs, ys, indices_iter, out);
        }

        // --- Shift suffixes (right operand always public for shift tables) ---
        Suffixes::RightShift => {
            let y_pubs = right.as_public();
            out.extend_b2a_ring::<T::Half>(
                indices_iter,
                xs.iter().enumerate().map(|(j, x)| {
                    let shift = (y_pubs[orig(j)] as u128).trailing_zeros() as usize;
                    *x >> shift
                }),
            );
        }
        Suffixes::RightShiftHelper => {
            let y_pubs = right.as_public();
            let y_len = suffix_len / 2;
            out.extend_ready(
                indices_iter,
                (0..n).map(|j| {
                    let yp = y_pubs[orig(j)];
                    let lo = LookupBits::new(yp as u128, y_len).leading_ones() as u64;
                    let val = 1u64 << lo;
                    Rep3Value::Public(F::from_u64(val))
                }),
            );
        }
        Suffixes::RightShiftPadding => {
            let y_pubs = right.as_public();
            let log_xlen = XLEN.log_2();
            out.extend_ready(
                indices_iter,
                (0..n).map(|j| {
                    let yp = y_pubs[orig(j)];
                    let shift_mask = (1u64 << log_xlen.min(suffix_len / 2)) - 1;
                    let shift = (yp & shift_mask) as usize;
                    let val = 1u128 << (XLEN - 1 - shift);
                    Rep3Value::Public(F::from_u128(val))
                }),
            );
        }
        Suffixes::LeftShift => {
            let y_pubs = right.as_public();
            out.extend_b2a_ring::<T::Half>(
                indices_iter,
                xs.iter().enumerate().map(|(j, x)| {
                    let yp = y_pubs[orig(j)];
                    let y_mask = T::Half::try_from(yp as u128).unwrap_or_else(|_| {
                        T::Half::try_from(yp as u128 & ((1u128 << T::Half::K) - 1))
                            .unwrap_or_else(|_| unreachable!())
                    });
                    let masked = *x & RingElement(!y_mask);
                    let shift = LookupBits::new(yp as u128, suffix_len / 2).leading_ones() as usize;
                    masked << shift
                }),
            );
        }
        Suffixes::RightShiftW => {
            let y_pubs = right.as_public();
            out.extend_b2a_ring::<T::Half>(
                indices_iter,
                xs.iter().enumerate().map(|(j, x)| {
                    let yp = y_pubs[orig(j)];
                    let x32 = to_u32_share(*x);
                    let shift = (yp as u128).trailing_zeros().min(XLEN as u32 / 2) as usize;
                    let shifted = x32 >> shift;
                    Rep3RingShare {
                        a: RingElement(
                            T::Half::try_from(shifted.a.0 as u128)
                                .unwrap_or_else(|_| unreachable!()),
                        ),
                        b: RingElement(
                            T::Half::try_from(shifted.b.0 as u128)
                                .unwrap_or_else(|_| unreachable!()),
                        ),
                    }
                }),
            );
        }
        Suffixes::RightShiftWHelper => {
            let y_pubs = right.as_public();
            let half_xlen = XLEN / 2;
            let y_bits = (suffix_len / 2).min(half_xlen);
            out.extend_ready(
                indices_iter,
                (0..n).map(|j| {
                    let yp = y_pubs[orig(j)];
                    let y_truncated = LookupBits::new(yp as u128, y_bits);
                    let lo = y_truncated.leading_ones() as u64;
                    let val = 1u64 << lo;
                    Rep3Value::Public(F::from_u64(val))
                }),
            );
        }
        Suffixes::LeftShiftW => {
            let y_pubs = right.as_public();
            let half_xlen = XLEN / 2;
            out.extend_b2a_ring::<T::Half>(
                indices_iter,
                xs.iter().enumerate().map(|(j, x)| {
                    let yp = y_pubs[orig(j)];
                    let y_truncated_bits = yp & ((1u64 << half_xlen) - 1);
                    let x32 = to_u32_share(*x);
                    let y32_mask = y_truncated_bits as u32;
                    let masked = x32 & RingElement(!y32_mask);
                    let lo = LookupBits::new(y_truncated_bits as u128, half_xlen).leading_ones();
                    let shifted = masked << lo as usize;
                    Rep3RingShare {
                        a: RingElement(
                            T::Half::try_from(shifted.a.0 as u128)
                                .unwrap_or_else(|_| unreachable!()),
                        ),
                        b: RingElement(
                            T::Half::try_from(shifted.b.0 as u128)
                                .unwrap_or_else(|_| unreachable!()),
                        ),
                    }
                }),
            );
        }
        Suffixes::LeftShiftWHelper => {
            let y_pubs = right.as_public();
            out.extend_ready(
                indices_iter,
                (0..n).map(|j| {
                    let yp = y_pubs[orig(j)];
                    let lo = LookupBits::new(yp as u128, suffix_len / 2).leading_ones() as u64;
                    let val = (1u64 << lo) as u32 as u64;
                    Rep3Value::Public(F::from_u64(val))
                }),
            );
        }

        // These suffixes are interleaved-only — should not appear here
        Suffixes::UpperWord
        | Suffixes::LowerWord
        | Suffixes::LowerHalfWord
        | Suffixes::Pow2
        | Suffixes::Pow2W
        | Suffixes::SignExtensionUpperHalf
        | Suffixes::OverflowBitsZero => {
            unreachable!(
                "Interleaved suffix {:?} received Uninterleaved data",
                suffix
            );
        }
        #[cfg(feature = "rv64")]
        Suffixes::Rev8W => {
            unreachable!(
                "Interleaved suffix {:?} received Uninterleaved data",
                suffix
            );
        }
    }

    Ok(())
}

/// Evaluate an interleaved suffix on shared bits, pushing into SuffixFutureBatch.
fn eval_suffix_interleaved<T, F, N>(
    suffix: &Suffixes,
    bits: &MixedBatch<LookupIndexInt, T>,
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
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
    let (positions, shared_bits) =
        split_interleaved_public::<T, F>(suffix, bits, suffix_len, base, out);

    if shared_bits.is_empty() {
        return Ok(());
    }

    let indices = positions.iter().map(|&i| base + i);

    match suffix {
        Suffixes::One => unreachable!("handled above"),

        Suffixes::UpperWord => {
            if T::K <= XLEN {
                // All bits below XLEN → zero
                out.extend_ready(
                    indices,
                    std::iter::repeat(Rep3Value::Public(F::zero())).take(shared_bits.len()),
                );
            } else {
                let result_bits = T::K - XLEN;
                let shifted: Vec<Rep3RingShare<T>> =
                    shared_bits.iter().map(|b| *b >> XLEN).collect();
                if result_bits <= T::Half::K {
                    let dc: Vec<Rep3RingShare<T::Half>> =
                        shifted.iter().map(|s| downcast(*s)).collect();
                    out.extend_b2a_ring::<T::Half>(indices, dc.into_iter());
                } else {
                    // Push as full T ring B2A
                    out.extend_b2a_ring::<T>(indices, shifted.into_iter());
                }
            }
        }
        Suffixes::LowerWord => {
            let result_bits = XLEN.min(T::K);
            if result_bits <= T::Half::K {
                let result: Vec<Rep3RingShare<T::Half>> = if XLEN >= T::K {
                    shared_bits.iter().map(|b| downcast(*b)).collect()
                } else {
                    let mask_val =
                        T::try_from((1u128 << XLEN) - 1).unwrap_or_else(|_| unreachable!());
                    shared_bits
                        .iter()
                        .map(|b| downcast(*b & RingElement(mask_val)))
                        .collect()
                };
                out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
            } else {
                let masked: Vec<Rep3RingShare<T>> = if XLEN >= T::K {
                    shared_bits.clone()
                } else {
                    let mask_val =
                        T::try_from((1u128 << XLEN) - 1).unwrap_or_else(|_| unreachable!());
                    shared_bits
                        .iter()
                        .map(|b| *b & RingElement(mask_val))
                        .collect()
                };
                out.extend_b2a_ring::<T>(indices, masked.into_iter());
            }
        }
        Suffixes::LowerHalfWord => {
            let half = XLEN / 2;
            let result_bits = half.min(T::K);
            if result_bits <= T::Half::K {
                let result: Vec<Rep3RingShare<T::Half>> = if half >= T::K {
                    shared_bits.iter().map(|b| downcast(*b)).collect()
                } else {
                    let mask_val =
                        T::try_from((1u128 << half) - 1).unwrap_or_else(|_| unreachable!());
                    shared_bits
                        .iter()
                        .map(|b| downcast(*b & RingElement(mask_val)))
                        .collect()
                };
                out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
            } else {
                let masked: Vec<Rep3RingShare<T>> = if half >= T::K {
                    shared_bits.clone()
                } else {
                    let mask_val =
                        T::try_from((1u128 << half) - 1).unwrap_or_else(|_| unreachable!());
                    shared_bits
                        .iter()
                        .map(|b| *b & RingElement(mask_val))
                        .collect()
                };
                out.extend_b2a_ring::<T>(indices, masked.into_iter());
            }
        }
        #[cfg(feature = "rv64")]
        Suffixes::Rev8W => {
            let mask_byte = RingElement(0xFFu32);
            let reversed_u32: Vec<Rep3RingShare<u32>> = shared_bits
                .iter()
                .map(|b| {
                    let a_u128: u128 = b.a.0.into();
                    let b_u128: u128 = b.b.0.into();
                    let v = Rep3RingShare {
                        a: RingElement(a_u128 as u32),
                        b: RingElement(b_u128 as u32),
                    };
                    let byte0 = v & mask_byte;
                    let byte1 = (v >> 8) & mask_byte;
                    let byte2 = (v >> 16) & mask_byte;
                    let byte3 = (v >> 24) & mask_byte;
                    (byte0 << 24) ^ (byte1 << 16) ^ (byte2 << 8) ^ byte3
                })
                .collect();
            if T::Half::K >= 32 {
                let result: Vec<Rep3RingShare<T::Half>> = reversed_u32
                    .iter()
                    .map(|r| Rep3RingShare {
                        a: RingElement(
                            T::Half::try_from(r.a.0 as u128).unwrap_or_else(|_| unreachable!()),
                        ),
                        b: RingElement(
                            T::Half::try_from(r.b.0 as u128).unwrap_or_else(|_| unreachable!()),
                        ),
                    })
                    .collect();
                out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
            } else {
                out.extend_b2a_ring::<u32>(indices, reversed_u32.into_iter());
            }
        }
        Suffixes::Pow2 => {
            let log_xlen = XLEN.log_2();
            let num_bits = log_xlen.min(suffix_len);
            let shift_mask_val =
                T::try_from((1u128 << num_bits) - 1).unwrap_or_else(|_| unreachable!());
            let shifts: Vec<Rep3RingShare<T>> = shared_bits
                .iter()
                .map(|b| *b & RingElement(shift_mask_val))
                .collect();
            let result =
                eval_pow2_from_shift_bits_ready::<T, F, N>(&shifts, num_bits, io_ctx, party_id)?;
            out.extend_ready(indices, result.into_iter().map(Rep3Value::Shared));
        }
        Suffixes::Pow2W => {
            let num_bits = 5usize.min(suffix_len);
            let shift_mask_val =
                T::try_from((1u128 << num_bits) - 1).unwrap_or_else(|_| unreachable!());
            let shifts: Vec<Rep3RingShare<T>> = shared_bits
                .iter()
                .map(|b| *b & RingElement(shift_mask_val))
                .collect();
            let result =
                eval_pow2_from_shift_bits_ready::<T, F, N>(&shifts, num_bits, io_ctx, party_id)?;
            out.extend_ready(indices, result.into_iter().map(Rep3Value::Shared));
        }
        Suffixes::SignExtensionUpperHalf => {
            let half = XLEN / 2;
            if suffix_len < half {
                out.extend_ready(
                    indices,
                    std::iter::repeat(Rep3Value::Public(F::one())).take(shared_bits.len()),
                );
            } else {
                let sign_bit_pos = half - 1;
                let weight = F::from_u128(((1u64 << half) - 1) as u128 * (1u128 << half));
                out.extend_bitinject_scaled(
                    indices,
                    shared_bits
                        .iter()
                        .map(|b| downcast::<T, Bit>(*b >> sign_bit_pos)),
                    weight,
                );
            }
        }
        Suffixes::OverflowBitsZero => {
            if T::K <= XLEN {
                out.extend_ready(
                    indices,
                    std::iter::repeat(Rep3Value::Public(F::one())).take(shared_bits.len()),
                );
            } else {
                let upper: Vec<Rep3RingShare<T>> = shared_bits.iter().map(|b| *b >> XLEN).collect();
                let eq_bits = rep3_ring::binary::is_zero_many::<T, _>(&upper, io_ctx)?;
                out.extend_bitinject(indices, eq_bits.into_iter());
            }
        }

        // These suffixes are uninterleaved-only — should not appear here
        _ => {
            unreachable!(
                "Uninterleaved suffix {:?} received Interleaved data",
                suffix
            );
        }
    }

    Ok(())
}

/// RightOperandW suffix: truncate right operand to 32 bits.
fn eval_right_operand_w<T, F>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    base: usize,
    orig: impl Fn(usize) -> usize,
    out: &mut SuffixFutureBatch<F>,
) where
    T: Uninterleavable,
    T::Half: B2ABucketExtend,
    F: JoltField,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    match right {
        MixedBatch::Public(y_pubs) => {
            out.extend_ready(
                indices_iter,
                (0..n).map(|j| Rep3Value::Public(F::from_u64(y_pubs[orig(j)] & 0xFFFF_FFFF))),
            );
        }
        MixedBatch::Shared(ys) => {
            if T::Half::K >= 32 {
                let m: u128 = (1u128 << 32) - 1;
                let mask_val = T::Half::try_from(m).unwrap_or_else(|_| unreachable!());
                out.extend_b2a_ring::<T::Half>(
                    indices_iter,
                    ys.iter().map(|y| *y & RingElement(mask_val)),
                );
            } else {
                out.extend_b2a_ring::<T::Half>(indices_iter, ys.iter().copied());
            }
        }
        MixedBatch::Mixed(mixed) => {
            for (j, _) in xs.iter().enumerate() {
                let i = orig(j);
                match &mixed[i] {
                    Either::Public(yp) => {
                        out.extend_ready(
                            std::iter::once(base + i),
                            std::iter::once(Rep3Value::Public(F::from_u64(*yp & 0xFFFF_FFFF))),
                        );
                    }
                    Either::Shared(y) => {
                        if T::Half::K >= 32 {
                            let m: u128 = (1u128 << 32) - 1;
                            let mask_val = T::Half::try_from(m).unwrap_or_else(|_| unreachable!());
                            out.extend_b2a_ring::<T::Half>(
                                std::iter::once(base + i),
                                std::iter::once(*y & RingElement(mask_val)),
                            );
                        } else {
                            out.extend_b2a_ring::<T::Half>(
                                std::iter::once(base + i),
                                std::iter::once(*y),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// SignExtensionRightOperand suffix: extract sign bit, apply weight.
fn eval_sign_extension_right_operand<T, F>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    suffix_len: usize,
    base: usize,
    orig: impl Fn(usize) -> usize,
    out: &mut SuffixFutureBatch<F>,
) where
    T: Uninterleavable,
    T::Half: AsPrimitive<Bit>,
    F: JoltField,
{
    let n = xs.len();
    let indices_iter = (0..n).map(|j| base + orig(j));
    if suffix_len < XLEN {
        out.extend_ready(
            indices_iter,
            std::iter::repeat(Rep3Value::Public(F::one())).take(n),
        );
    } else {
        let sign_bit_pos = XLEN / 2 - 1;
        let weight = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)));
        match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_ready(
                    indices_iter,
                    (0..n).map(|j| {
                        let sign = (y_pubs[orig(j)] >> sign_bit_pos) & 1;
                        Rep3Value::Public(if sign == 1 { weight } else { F::zero() })
                    }),
                );
            }
            MixedBatch::Shared(ys) => {
                out.extend_bitinject_scaled(
                    indices_iter,
                    ys.iter()
                        .map(|y| downcast::<T::Half, Bit>(*y >> sign_bit_pos)),
                    weight,
                );
            }
            MixedBatch::Mixed(mixed) => {
                for j in 0..n {
                    let i = orig(j);
                    let idx = base + i;
                    match &mixed[i] {
                        Either::Public(yp) => {
                            let sign = (*yp >> sign_bit_pos) & 1;
                            out.extend_ready(
                                std::iter::once(idx),
                                std::iter::once(Rep3Value::Public(if sign == 1 {
                                    weight
                                } else {
                                    F::zero()
                                })),
                            );
                        }
                        Either::Shared(y) => {
                            out.extend_bitinject_scaled(
                                std::iter::once(idx),
                                std::iter::once(downcast::<T::Half, Bit>(*y >> sign_bit_pos)),
                                weight,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Pow2 from shift bits → produces Ready field shares directly.
/// Same as eval_pow2_from_shift_bits but returns Vec<Rep3PrimeFieldShare<F>>.
fn eval_pow2_from_shift_bits_ready<T: IntRing2k, F: JoltField, N: Rep3Network>(
    shift_vals: &[Rep3RingShare<T>],
    num_bits: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    let table_size = 1usize << num_bits;
    let n = shift_vals.len();

    let mut xored = Vec::with_capacity(n * table_size);
    for shift in shift_vals {
        for s in 0..table_size {
            let target = RingElement(T::try_from(s as u128).unwrap_or_else(|_| unreachable!()));
            xored.push(rep3_ring::binary::xor_public(shift, &target, party_id));
        }
    }

    let eq_bits: Vec<Rep3RingShare<Bit>> = rep3_ring::binary::is_zero_many::<T, _>(&xored, io_ctx)?;
    let eq_field: Vec<Rep3PrimeFieldShare<F>> =
        rep3_ring::conversion::bit_inject_from_bits_to_field_many(&eq_bits, io_ctx)?;

    let mut result = Vec::with_capacity(n);
    for j in 0..n {
        let mut acc = Rep3PrimeFieldShare::<F>::zero_share();
        for s in 0..table_size {
            let weight = F::from_u128(1u128 << s);
            acc = acc + eq_field[j * table_size + s] * weight;
        }
        result.push(acc);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the sign-extension suffix value from a public right-operand bitmask.
///
/// Mirrors vanilla `SignExtensionSuffix::suffix_mle` on the uninterleaved `y` bits.
fn compute_sign_extension_from_mask(mask_u64: u64, suffix_len: usize) -> u64 {
    let y_len = suffix_len / 2;
    let y_low = if y_len >= 64 {
        mask_u64
    } else {
        mask_u64 & ((1u64 << y_len) - 1)
    };
    let padding_len = std::cmp::min(y_low.trailing_zeros() as usize, y_len);
    if padding_len == 0 {
        0
    } else {
        ((1u128 << XLEN) - (1u128 << (XLEN - padding_len))) as u64
    }
}

/// Truncate a share to u32 (used by W-variant instructions).
fn to_u32_share<H: IntRing2k>(s: Rep3RingShare<H>) -> Rep3RingShare<u32> {
    // Via Into<u128> → truncate, since H may not impl AsPrimitive<u32> directly.
    let a: u128 = s.a.0.into();
    let b: u128 = s.b.0.into();
    Rep3RingShare {
        a: RingElement(a as u32),
        b: RingElement(b as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_core::utils::uninterleave_bits;

    #[test]
    fn test_uninterleave_bin_correctness() {
        let test_vals: Vec<u128> = vec![
            0b1010,
            0b1111,
            0b01_10_01_10,
            0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0u128,
            0,
            u128::MAX,
            1,
            0x5555_5555_5555_5555_5555_5555_5555_5555u128,
            0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAAu128,
        ];

        for &val in &test_vals {
            let (vx, vy) = uninterleave_bits(val);

            // 3-party XOR sharing: a ^ b ^ c = val
            let a: u128 = 0x1234_5678_ABCD_EF01_2345_6789_ABCD_EF01u128;
            let b: u128 = 0xFEDC_BA98_7654_3210_FEDC_BA98_7654_3210u128;
            let c: u128 = val ^ a ^ b;

            // Party 0: (a, b), Party 1: (b, c), Party 2: (c, a)
            let share0 = Rep3RingShare {
                a: RingElement(a),
                b: RingElement(b),
            };
            let share1 = Rep3RingShare {
                a: RingElement(b),
                b: RingElement(c),
            };
            let share2 = Rep3RingShare {
                a: RingElement(c),
                b: RingElement(a),
            };

            let (x0, y0) = u128::uninterleave(share0);
            let (x1, y1) = u128::uninterleave(share1);
            let (x2, y2) = u128::uninterleave(share2);

            // Reconstruct: piece_0 ^ piece_1 ^ piece_2
            let rx = x0.a.0 ^ x1.a.0 ^ x2.a.0;
            let ry = y0.a.0 ^ y1.a.0 ^ y2.a.0;

            assert_eq!(rx, vx, "x mismatch for val=0x{:032X}", val);
            assert_eq!(ry, vy, "y mismatch for val=0x{:032X}", val);

            // Share consistency: party_i.b == party_{i-1 mod 3}.a
            // In rep3 binary: party0=(a,b), party1=(b,c), party2=(c,a)
            // So: party0.b = b = party1.a ✓, party1.b = c = party2.a ✓, party2.b = a = party0.a ✓
            assert_eq!(x0.b.0, x1.a.0, "x share consistency: p0.b == p1.a");
            assert_eq!(x1.b.0, x2.a.0, "x share consistency: p1.b == p2.a");
            assert_eq!(x2.b.0, x0.a.0, "x share consistency: p2.b == p0.a");
            assert_eq!(y0.b.0, y1.a.0, "y share consistency: p0.b == p1.a");
            assert_eq!(y1.b.0, y2.a.0, "y share consistency: p1.b == p2.a");
            assert_eq!(y2.b.0, y0.a.0, "y share consistency: p2.b == p0.a");
        }
    }
}
