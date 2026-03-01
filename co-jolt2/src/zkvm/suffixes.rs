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
use crate::utils::types::Either;
use jolt2_common::constants::XLEN;
use jolt_core::utils::interleave_bits;
use jolt_core::utils::lookup_bits::LookupBits;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::lookup_table::suffixes::Suffixes;
use mpc_core::protocols::rep3::network::{
    IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::downcast;
use mpc_core::protocols::rep3_ring::edabits::EdaBitsPool;
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self as rep3_ring, Rep3RingShare};
use num_traits::AsPrimitive;
use rand::distributions::Standard;
use rand::prelude::Distribution;

// ---------------------------------------------------------------------------
// Uninterleavable trait — maps interleaved ring T → half-sized ring T::Half
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

/// Compute the sign-extension suffix value from a public right-operand bitmask.
///
/// Mirrors vanilla `SignExtensionSuffix::suffix_mle` logic:
/// sign extension padding length = trailing_zeros of the right operand's low bits.
fn compute_sign_extension_from_mask(mask_u64: u64, suffix_len: usize) -> u64 {
    let y_len = suffix_len / 2;
    let y_low = if y_len >= 64 {
        mask_u64
    } else {
        mask_u64 & ((1u64 << y_len) - 1)
    };
    let padding_len = y_low.trailing_zeros() as u64;
    // The sign-extension MLE value: 2^padding_len
    1u64 << padding_len.min(63)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MixedBatch / SuffixBitsBatch — per-table operand data
// ---------------------------------------------------------------------------

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
    Interleaved(MixedBatch<u128, T>),
}

impl<T: Uninterleavable> SuffixBitsBatch<T> {
    pub fn len(&self) -> usize {
        match self {
            Self::Uninterleaved(left, _) => left.len(),
            Self::Interleaved(bits) => bits.len(),
        }
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
        matches!(
            s,
            Suffixes::UpperWord
                | Suffixes::LowerWord
                | Suffixes::LowerHalfWord
                | Suffixes::Rev8W
                | Suffixes::Pow2
                | Suffixes::Pow2W
                | Suffixes::SignExtensionUpperHalf
                | Suffixes::OverflowBitsZero
        )
    })
}

// ---------------------------------------------------------------------------
// SuffixFutureBatch — pre-bucketed suffix evaluation results
// ---------------------------------------------------------------------------

/// Trait for compile-time dispatch of B2A bucket extension.
/// Each ring type maps to the correct typed bucket in `SuffixFutureBatch`.
pub trait B2ABucketExtend: IntRing2k {
    fn extend_bucket<F: JoltField>(
        batch: &mut SuffixFutureBatch<F>,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<Self>>,
    );
}

macro_rules! impl_b2a_bucket_extend {
    ($ring:ty, $idx_field:ident, $val_field:ident) => {
        impl B2ABucketExtend for $ring {
            fn extend_bucket<F: JoltField>(
                batch: &mut SuffixFutureBatch<F>,
                indices: impl IntoIterator<Item = usize>,
                vals: impl IntoIterator<Item = Rep3RingShare<Self>>,
            ) {
                batch.$idx_field.extend(indices);
                batch.$val_field.extend(vals);
            }
        }
    };
}

impl_b2a_bucket_extend!(u8, b2a_u8_idx, b2a_u8);
impl_b2a_bucket_extend!(u16, b2a_u16_idx, b2a_u16);
impl_b2a_bucket_extend!(u32, b2a_u32_idx, b2a_u32);
impl_b2a_bucket_extend!(u64, b2a_u64_idx, b2a_u64);
impl_b2a_bucket_extend!(u128, b2a_u128_idx, b2a_u128);

/// Pre-bucketed collection of suffix evaluation results, replacing the
/// `Vec<SuffixFuture>` + rayon fold/reduce classification scan.
///
/// Values are pushed into typed buckets during suffix evaluation, then
/// fulfilled in a single batched pass per ring type.
pub struct SuffixFutureBatch<F: JoltField> {
    len: usize,

    // Scatter indices (position in output vec)
    ready_idx: Vec<usize>,
    bitinject_idx: Vec<usize>,
    b2a_u8_idx: Vec<usize>,
    b2a_u16_idx: Vec<usize>,
    b2a_u32_idx: Vec<usize>,
    b2a_u64_idx: Vec<usize>,
    b2a_u128_idx: Vec<usize>,

    // Values
    ready: Vec<Rep3PrimeFieldShare<F>>,
    bitinject: Vec<Rep3RingShare<Bit>>,
    b2a_u8: Vec<Rep3RingShare<u8>>,
    b2a_u16: Vec<Rep3RingShare<u16>>,
    b2a_u32: Vec<Rep3RingShare<u32>>,
    b2a_u64: Vec<Rep3RingShare<u64>>,
    b2a_u128: Vec<Rep3RingShare<u128>>,
}

impl<F: JoltField> SuffixFutureBatch<F> {
    pub fn new() -> Self {
        Self {
            len: 0,
            ready_idx: Vec::new(),
            bitinject_idx: Vec::new(),
            b2a_u8_idx: Vec::new(),
            b2a_u16_idx: Vec::new(),
            b2a_u32_idx: Vec::new(),
            b2a_u64_idx: Vec::new(),
            b2a_u128_idx: Vec::new(),
            ready: Vec::new(),
            bitinject: Vec::new(),
            b2a_u8: Vec::new(),
            b2a_u16: Vec::new(),
            b2a_u32: Vec::new(),
            b2a_u64: Vec::new(),
            b2a_u128: Vec::new(),
        }
    }

    /// Reserve `n` output slots, returning the base index for this segment.
    pub fn reserve(&mut self, n: usize) -> usize {
        let base = self.len;
        self.len += n;
        base
    }

    /// Push Ready (field) values with their output indices.
    pub fn extend_ready(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3PrimeFieldShare<F>>,
    ) {
        self.ready_idx.extend(indices);
        self.ready.extend(vals);
    }

    /// Push BitInject (single-bit) values with their output indices.
    pub fn extend_bitinject(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<Bit>>,
    ) {
        self.bitinject_idx.extend(indices);
        self.bitinject.extend(vals);
    }

    /// Push B2A values into the correct typed bucket via compile-time dispatch.
    pub fn extend_b2a_ring<R: B2ABucketExtend>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<R>>,
    ) {
        R::extend_bucket(self, indices, vals);
    }

    /// Fulfill all pending conversions and scatter into output vec.
    #[tracing::instrument(skip_all, name = "suffixes_fulfill")]
    pub fn fulfill_with_pool<N: Rep3NetworkWorker>(
        self,
        io_ctx: &mut IoContextPool<N>,
        pool: &mut EdaBitsPool<F>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        use mpc_core::protocols::rep3_ring::edabits;
        use rayon::prelude::*;

        let mut out = vec![Rep3PrimeFieldShare::default(); self.len];

        // Phase 1: Sequential conversions, collect all (idx, val) pairs.
        let mut scatter: Vec<(usize, Rep3PrimeFieldShare<F>)> = Vec::with_capacity(self.len);

        // Ready — direct
        scatter.extend(self.ready_idx.into_iter().zip(self.ready.into_iter()));

        // BitInject — single-bit → field via daBits
        if !self.bitinject.is_empty() {
            let dabits = pool.take_dabits(self.bitinject.len());
            let fields = edabits::bit_inject_field_many(&self.bitinject, &dabits, io_ctx.main())?;
            scatter.extend(self.bitinject_idx.into_iter().zip(fields.into_iter()));
        }

        // B2A per ring type
        macro_rules! fulfill_b2a {
            ($ring:ty, $idx:ident, $val:ident) => {
                if !self.$val.is_empty() {
                    let batch = pool.take_edabits::<$ring>(self.$val.len());
                    let fields = edabits::ring_to_field_b2a_many::<$ring, F, _>(
                        &self.$val,
                        &batch,
                        io_ctx.main(),
                    )?;
                    scatter.extend(self.$idx.into_iter().zip(fields.into_iter()));
                }
            };
        }
        fulfill_b2a!(u8, b2a_u8_idx, b2a_u8);
        fulfill_b2a!(u16, b2a_u16_idx, b2a_u16);
        fulfill_b2a!(u32, b2a_u32_idx, b2a_u32);
        fulfill_b2a!(u64, b2a_u64_idx, b2a_u64);
        fulfill_b2a!(u128, b2a_u128_idx, b2a_u128);

        // Phase 2: Parallel scatter — all indices are disjoint by construction.
        #[derive(Copy, Clone)]
        struct SendPtr<T>(*mut T);
        // SAFETY: indices are unique (each output position reserved by exactly one segment),
        // so distinct threads write to distinct memory locations.
        unsafe impl<T: Send> Send for SendPtr<T> {}
        unsafe impl<T: Send> Sync for SendPtr<T> {}

        let ptr = SendPtr(out.as_mut_ptr());
        scatter.into_par_iter().for_each(|(idx, val)| {
            let p = ptr; // force capture of SendPtr wrapper, not the raw *mut field
                         // SAFETY: idx is unique across all entries; Rep3PrimeFieldShare<F> is Copy.
            unsafe {
                p.0.add(idx).write(val);
            }
        });

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Per-table suffix evaluation (new path)
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
        let one = Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id);
        out.extend_ready(base..base + n, std::iter::repeat(one).take(n));
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

/// Handle public entries for an interleaved suffix: compute via suffix_mle.
/// Returns the positions and shares for entries that need MPC evaluation.
fn split_interleaved_public<T: Uninterleavable, F: JoltField>(
    suffix: &Suffixes,
    bits: &MixedBatch<u128, T>,
    suffix_len: usize,
    party_id: PartyID,
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) -> (Vec<usize>, Vec<Rep3RingShare<T>>)
where
    Standard: Distribution<T>,
{
    let mask = if suffix_len >= 128 {
        u128::MAX
    } else {
        (1u128 << suffix_len) - 1
    };

    match bits {
        MixedBatch::Shared(shares) => {
            let positions: Vec<usize> = (0..shares.len()).collect();
            (positions, shares.clone())
        }
        MixedBatch::Public(pubs) => {
            for (i, &p) in pubs.iter().enumerate() {
                let val = suffix.suffix_mle::<XLEN>(LookupBits::new(p & mask, suffix_len));
                out.extend_ready(
                    std::iter::once(base + i),
                    std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                        &F::from_u64(val),
                        party_id,
                    )),
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
                        let val = suffix.suffix_mle::<XLEN>(LookupBits::new(*p & mask, suffix_len));
                        out.extend_ready(
                            std::iter::once(base + i),
                            std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                                &F::from_u64(val),
                                party_id,
                            )),
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
                std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                    &F::from_u64(val),
                    party_id,
                )),
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
                            std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                                &F::from_u64(val),
                                party_id,
                            )),
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
        Suffixes::And => match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_b2a_ring::<T::Half>(
                    indices_iter,
                    xs.iter().enumerate().map(|(j, x)| {
                        let yp = y_pubs[orig(j)];
                        *x & RingElement(
                            T::Half::try_from(yp as u128).unwrap_or_else(|_| unreachable!()),
                        )
                    }),
                );
            }
            MixedBatch::Shared(ys) => {
                let result = rep3_ring::binary::and_many::<T::Half, _>(xs, ys, io_ctx)?;
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
                            let mask = RingElement(
                                T::Half::try_from(*yp as u128).unwrap_or_else(|_| unreachable!()),
                            );
                            local_idx.push(base + i);
                            local_vals.push(*x & mask);
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
                    let result =
                        rep3_ring::binary::and_many::<T::Half, _>(&mpc_xs, &mpc_ys, io_ctx)?;
                    out.extend_b2a_ring::<T::Half>(mpc_idx.into_iter(), result.into_iter());
                }
            }
        },
        Suffixes::NotAnd => match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_b2a_ring::<T::Half>(
                    indices_iter,
                    xs.iter().enumerate().map(|(j, x)| {
                        let yp = y_pubs[orig(j)];
                        *x & RingElement(
                            !T::Half::try_from(yp as u128).unwrap_or_else(|_| unreachable!()),
                        )
                    }),
                );
            }
            MixedBatch::Shared(ys) => {
                let not_ys: Vec<_> = ys.iter().map(|y| !y).collect();
                let result = rep3_ring::binary::and_many::<T::Half, _>(xs, &not_ys, io_ctx)?;
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
                            let mask = RingElement(
                                !T::Half::try_from(*yp as u128).unwrap_or_else(|_| unreachable!()),
                            );
                            local_idx.push(base + i);
                            local_vals.push(*x & mask);
                        }
                        Either::Shared(y) => {
                            mpc_idx.push(base + i);
                            mpc_xs.push(*x);
                            mpc_ys.push(!y);
                        }
                    }
                }
                out.extend_b2a_ring::<T::Half>(local_idx.into_iter(), local_vals.into_iter());
                if !mpc_xs.is_empty() {
                    let result =
                        rep3_ring::binary::and_many::<T::Half, _>(&mpc_xs, &mpc_ys, io_ctx)?;
                    out.extend_b2a_ring::<T::Half>(mpc_idx.into_iter(), result.into_iter());
                }
            }
        },
        Suffixes::Xor => {
            // XOR is always local — no MPC for any variant
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
        Suffixes::Or => {
            match right {
                MixedBatch::Public(y_pubs) => {
                    // x | pub = (x & !pub) ^ pub  (local)
                    out.extend_b2a_ring::<T::Half>(
                        indices_iter,
                        xs.iter().enumerate().map(|(j, x)| {
                            let mask = RingElement(
                                T::Half::try_from(y_pubs[orig(j)] as u128)
                                    .unwrap_or_else(|_| unreachable!()),
                            );
                            let x_and_not_mask = *x & RingElement(!mask.0);
                            rep3_ring::binary::xor_public(&x_and_not_mask, &mask, party_id)
                        }),
                    );
                }
                MixedBatch::Shared(ys) => {
                    let result = rep3_ring::binary::or_many::<T::Half, _>(xs, ys, io_ctx)?;
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
                                let mask = RingElement(
                                    T::Half::try_from(*yp as u128)
                                        .unwrap_or_else(|_| unreachable!()),
                                );
                                local_idx.push(base + i);
                                local_vals.push(rep3_ring::binary::xor_public(
                                    &(*x & RingElement(!mask.0)),
                                    &mask,
                                    party_id,
                                ));
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
                        let result =
                            rep3_ring::binary::or_many::<T::Half, _>(&mpc_xs, &mpc_ys, io_ctx)?;
                        out.extend_b2a_ring::<T::Half>(mpc_idx.into_iter(), result.into_iter());
                    }
                }
            }
        }

        // --- B2A(H): value extraction (right can be public or shared) ---
        Suffixes::RightOperand => match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_ready(
                    indices_iter,
                    (0..n).map(|j| {
                        Rep3PrimeFieldShare::promote_from_trivial(
                            &F::from_u64(y_pubs[orig(j)]),
                            party_id,
                        )
                    }),
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
                                std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                                    &F::from_u64(*yp),
                                    party_id,
                                )),
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
        Suffixes::RightOperandW => match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_ready(
                    indices_iter,
                    (0..n).map(|j| {
                        Rep3PrimeFieldShare::promote_from_trivial(
                            &F::from_u64(y_pubs[orig(j)] & 0xFFFF_FFFF),
                            party_id,
                        )
                    }),
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
                                std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                                    &F::from_u64(*yp & 0xFFFF_FFFF),
                                    party_id,
                                )),
                            );
                        }
                        Either::Shared(y) => {
                            if T::Half::K >= 32 {
                                let m: u128 = (1u128 << 32) - 1;
                                let mask_val =
                                    T::Half::try_from(m).unwrap_or_else(|_| unreachable!());
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
        },
        Suffixes::Lsb => match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_ready(
                    indices_iter,
                    (0..n).map(|j| {
                        Rep3PrimeFieldShare::promote_from_trivial(
                            &F::from_u64(y_pubs[orig(j)] & 1),
                            party_id,
                        )
                    }),
                );
            }
            MixedBatch::Shared(ys) => {
                let one = T::Half::try_from(1u128).unwrap_or_else(|_| unreachable!());
                out.extend_b2a_ring::<T::Half>(
                    indices_iter,
                    ys.iter().map(|y| *y & RingElement(one)),
                );
            }
            MixedBatch::Mixed(mixed) => {
                for (j, _) in xs.iter().enumerate() {
                    let i = orig(j);
                    match &mixed[i] {
                        Either::Public(yp) => {
                            out.extend_ready(
                                std::iter::once(base + i),
                                std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                                    &F::from_u64(*yp & 1),
                                    party_id,
                                )),
                            );
                        }
                        Either::Shared(y) => {
                            let one = T::Half::try_from(1u128).unwrap_or_else(|_| unreachable!());
                            out.extend_b2a_ring::<T::Half>(
                                std::iter::once(base + i),
                                std::iter::once(*y & RingElement(one)),
                            );
                        }
                    }
                }
            }
        },

        // --- BitInject: comparisons (always need shared ys for MPC) ---
        Suffixes::LessThan => {
            let ys: Vec<Rep3RingShare<T::Half>> = (0..n)
                .map(|j| {
                    let i = orig(j);
                    match right {
                        MixedBatch::Shared(ys) => ys[i],
                        MixedBatch::Public(y_pubs) => {
                            let h = T::Half::try_from(y_pubs[i] as u128)
                                .unwrap_or_else(|_| unreachable!());
                            rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(h))
                        }
                        MixedBatch::Mixed(mixed) => match &mixed[i] {
                            Either::Shared(y) => *y,
                            Either::Public(yp) => {
                                let h = T::Half::try_from(*yp as u128)
                                    .unwrap_or_else(|_| unreachable!());
                                rep3_ring::binary::promote_to_trivial_share(
                                    party_id,
                                    &RingElement(h),
                                )
                            }
                        },
                    }
                })
                .collect();
            let ge_bits: Vec<Rep3RingShare<Bit>> =
                rep3_ring::arithmetic::ge_many::<T::Half, _>(xs, &ys, io_ctx)?;
            let lt_bits: Vec<_> = ge_bits.iter().map(|b| !b).collect();
            out.extend_bitinject(indices_iter, lt_bits.into_iter());
        }
        Suffixes::GreaterThan => {
            let ys: Vec<Rep3RingShare<T::Half>> = (0..n)
                .map(|j| {
                    let i = orig(j);
                    match right {
                        MixedBatch::Shared(ys) => ys[i],
                        MixedBatch::Public(y_pubs) => {
                            let h = T::Half::try_from(y_pubs[i] as u128)
                                .unwrap_or_else(|_| unreachable!());
                            rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(h))
                        }
                        MixedBatch::Mixed(mixed) => match &mixed[i] {
                            Either::Shared(y) => *y,
                            Either::Public(yp) => {
                                let h = T::Half::try_from(*yp as u128)
                                    .unwrap_or_else(|_| unreachable!());
                                rep3_ring::binary::promote_to_trivial_share(
                                    party_id,
                                    &RingElement(h),
                                )
                            }
                        },
                    }
                })
                .collect();
            let ge_bits: Vec<Rep3RingShare<Bit>> =
                rep3_ring::arithmetic::ge_many::<T::Half, _>(&ys, xs, io_ctx)?;
            let gt_bits: Vec<_> = ge_bits.iter().map(|b| !b).collect();
            out.extend_bitinject(indices_iter, gt_bits.into_iter());
        }
        Suffixes::Eq => {
            // XOR (local for both public and shared y) then is_zero_many (MPC)
            let diff: Vec<Rep3RingShare<T::Half>> = xs
                .iter()
                .enumerate()
                .map(|(j, x)| {
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
                })
                .collect();
            let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&diff, io_ctx)?;
            out.extend_bitinject(indices_iter, eq_bits.into_iter());
        }
        Suffixes::LeftOperandIsZero => {
            // Only uses xs — right operand irrelevant
            let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(xs, io_ctx)?;
            out.extend_bitinject(indices_iter, eq_bits.into_iter());
        }
        Suffixes::RightOperandIsZero => match right {
            MixedBatch::Public(y_pubs) => {
                out.extend_ready(
                    indices_iter,
                    (0..n).map(|j| {
                        let val = if y_pubs[orig(j)] == 0 { 1u64 } else { 0u64 };
                        Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(val), party_id)
                    }),
                );
            }
            MixedBatch::Shared(ys) => {
                let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(ys, io_ctx)?;
                out.extend_bitinject(indices_iter, eq_bits.into_iter());
            }
            MixedBatch::Mixed(mixed) => {
                // Split: public y → Ready, shared y → collect for is_zero_many
                let mut mpc_idx = Vec::new();
                let mut mpc_ys = Vec::new();
                for (j, _) in xs.iter().enumerate() {
                    let i = orig(j);
                    match &mixed[i] {
                        Either::Public(yp) => {
                            let val = if *yp == 0 { 1u64 } else { 0u64 };
                            out.extend_ready(
                                std::iter::once(base + i),
                                std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                                    &F::from_u64(val),
                                    party_id,
                                )),
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
        },

        // --- BitInject: division checks (right always shared — division tables) ---
        Suffixes::DivByZero => {
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
            let divisor_zero = rep3_ring::binary::is_zero_many::<T::Half, _>(xs, io_ctx)?;
            let quotient_all_ones = rep3_ring::binary::is_zero_many::<T::Half, _>(&q_xor, io_ctx)?;
            let result =
                rep3_ring::binary::and_many::<Bit, _>(&divisor_zero, &quotient_all_ones, io_ctx)?;
            out.extend_bitinject(indices_iter, result.into_iter());
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
            let y_eq_all_ones = rep3_ring::binary::is_zero_many::<T::Half, _>(&y_xor, io_ctx)?;
            let x_eq_zero = rep3_ring::binary::is_zero_many::<T::Half, _>(xs, io_ctx)?;
            let result = rep3_ring::binary::and_many::<Bit, _>(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
            out.extend_bitinject(indices_iter, result.into_iter());
        }
        Suffixes::ChangeDivisorW => {
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
            let y_eq_all_ones = rep3_ring::binary::is_zero_many::<u32, _>(&y_xor, io_ctx)?;
            let x_eq_zero = rep3_ring::binary::is_zero_many::<u32, _>(&xs32, io_ctx)?;
            let result = rep3_ring::binary::and_many::<Bit, _>(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
            out.extend_bitinject(indices_iter, result.into_iter());
        }

        // --- Ready: sign extension (right always public for shift tables) ---
        Suffixes::SignExtension => {
            let y_pubs = right.as_public();
            out.extend_ready(
                indices_iter,
                (0..n).map(|j| {
                    let val = compute_sign_extension_from_mask(y_pubs[orig(j)], suffix_len);
                    Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(val), party_id)
                }),
            );
        }
        Suffixes::SignExtensionRightOperand => {
            if suffix_len < XLEN {
                let one = Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id);
                out.extend_ready(indices_iter, std::iter::repeat(one).take(n));
            } else {
                // Need shared ys for sign bit extraction — promote inline
                let ys: Vec<Rep3RingShare<T::Half>> = (0..n)
                    .map(|j| {
                        let i = orig(j);
                        match right {
                            MixedBatch::Shared(ys) => ys[i],
                            MixedBatch::Public(y_pubs) => {
                                let h = T::Half::try_from(y_pubs[i] as u128)
                                    .unwrap_or_else(|_| unreachable!());
                                rep3_ring::binary::promote_to_trivial_share(
                                    party_id,
                                    &RingElement(h),
                                )
                            }
                            MixedBatch::Mixed(mixed) => match &mixed[i] {
                                Either::Shared(y) => *y,
                                Either::Public(yp) => {
                                    let h = T::Half::try_from(*yp as u128)
                                        .unwrap_or_else(|_| unreachable!());
                                    rep3_ring::binary::promote_to_trivial_share(
                                        party_id,
                                        &RingElement(h),
                                    )
                                }
                            },
                        }
                    })
                    .collect();
                let sign_bit_pos = XLEN / 2 - 1;
                let sign_bits: Vec<Rep3RingShare<Bit>> = ys
                    .iter()
                    .map(|y| downcast::<T::Half, Bit>(*y >> sign_bit_pos))
                    .collect();
                let weight = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)));
                let sign_field: Vec<Rep3PrimeFieldShare<F>> =
                    rep3_ring::conversion::bit_inject_from_bits_to_field_many(&sign_bits, io_ctx)?;
                out.extend_ready(
                    indices_iter,
                    sign_field.into_iter().map(move |s| s * weight),
                );
            }
        }

        // --- B2A(H): XOR-rotate (right always shared — hash tables) ---
        Suffixes::XorRot16 => {
            let ys = right.as_shared();
            eval_xor_rot_uninterleaved::<16, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRot24 => {
            let ys = right.as_shared();
            eval_xor_rot_uninterleaved::<24, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRot32 => {
            let ys = right.as_shared();
            eval_xor_rot_uninterleaved::<32, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRot63 => {
            let ys = right.as_shared();
            eval_xor_rot_uninterleaved::<63, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRotW7 => {
            let ys = right.as_shared();
            eval_xor_rot_w_uninterleaved::<7, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRotW8 => {
            let ys = right.as_shared();
            eval_xor_rot_w_uninterleaved::<8, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRotW12 => {
            let ys = right.as_shared();
            eval_xor_rot_w_uninterleaved::<12, T, F>(xs, ys, indices_iter, out);
        }
        Suffixes::XorRotW16 => {
            let ys = right.as_shared();
            eval_xor_rot_w_uninterleaved::<16, T, F>(xs, ys, indices_iter, out);
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
                    Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(val), party_id)
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
                    Rep3PrimeFieldShare::promote_from_trivial(&F::from_u128(val), party_id)
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
                    Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(val), party_id)
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
                    Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(val), party_id)
                }),
            );
        }

        // These suffixes are interleaved-only — should not appear here
        Suffixes::UpperWord
        | Suffixes::LowerWord
        | Suffixes::LowerHalfWord
        | Suffixes::Rev8W
        | Suffixes::Pow2
        | Suffixes::Pow2W
        | Suffixes::SignExtensionUpperHalf
        | Suffixes::OverflowBitsZero => {
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
    bits: &MixedBatch<u128, T>,
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
        split_interleaved_public::<T, F>(suffix, bits, suffix_len, party_id, base, out);

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
                    std::iter::repeat(Rep3PrimeFieldShare::zero_share()).take(shared_bits.len()),
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
            out.extend_ready(indices, result.into_iter());
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
            out.extend_ready(indices, result.into_iter());
        }
        Suffixes::SignExtensionUpperHalf => {
            let half = XLEN / 2;
            if suffix_len < half {
                let one = Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id);
                out.extend_ready(indices, std::iter::repeat(one).take(shared_bits.len()));
            } else {
                let sign_bit_pos = half - 1;
                let sign_bits: Vec<Rep3RingShare<Bit>> = shared_bits
                    .iter()
                    .map(|b| downcast::<T, Bit>(*b >> sign_bit_pos))
                    .collect();
                let weight = F::from_u128(((1u64 << half) - 1) as u128 * (1u128 << half));
                let sign_field: Vec<Rep3PrimeFieldShare<F>> =
                    rep3_ring::conversion::bit_inject_from_bits_to_field_many(&sign_bits, io_ctx)?;
                let result: Vec<_> = sign_field.into_iter().map(|s| s * weight).collect();
                out.extend_ready(indices, result.into_iter());
            }
        }
        Suffixes::OverflowBitsZero => {
            if T::K <= XLEN {
                let one_bit = rep3_ring::binary::promote_to_trivial_share(
                    PartyID::ID0,
                    &RingElement(Bit::one()),
                );
                out.extend_bitinject(indices, std::iter::repeat(one_bit).take(shared_bits.len()));
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

// ---------------------------------------------------------------------------
// Helpers for sign extension and XOR-rotate
// ---------------------------------------------------------------------------

/// XorRot with pre-uninterleaved operands: (x ^ y) rotate_right by ROTATION
fn eval_xor_rot_uninterleaved<const ROTATION: u32, T: Uninterleavable, F: JoltField>(
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
fn eval_xor_rot_w_uninterleaved<const ROTATION: u32, T: Uninterleavable, F: JoltField>(
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
// Suffix classification for EdaBit budget estimation
// ---------------------------------------------------------------------------

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
        | Suffixes::Lsb
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
        | Suffixes::XorRotW16
        | Suffixes::Rev8W => {
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
        Suffixes::TwoLsb
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
