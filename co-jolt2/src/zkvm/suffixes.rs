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
use jolt_core::utils::lookup_bits::LookupBits;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::lookup_table::suffixes::Suffixes;
use mpc_core::protocols::rep3::network::{IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::EdaBitsPool;
use mpc_core::protocols::rep3_ring::casts::downcast;
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
pub trait Uninterleavable: IntRing2k + AsPrimitive<Self::Half> {
    type Half: IntRing2k + AsPrimitive<Self>;
    fn uninterleave(
        s: Rep3RingShare<Self>,
    ) -> (Rep3RingShare<Self::Half>, Rep3RingShare<Self::Half>)
    where
        Standard: Distribution<Self::Half>;
}

/// Generic single-element uninterleave: extract alternating bits from T into two T::Half values.
fn uninterleave_generic<T: Uninterleavable>(
    s: Rep3RingShare<T>,
) -> (Rep3RingShare<T::Half>, Rep3RingShare<T::Half>)
where
    Standard: Distribution<T::Half>,
{
    T::uninterleave(s)
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

impl_uninterleavable!(u16, u8, 0x5555u16, [
    (1, 0x3333u16), (2, 0x0F0Fu16), (4, 0x00FFu16)]);
impl_uninterleavable!(u32, u16, 0x5555_5555u32, [
    (1, 0x3333_3333u32), (2, 0x0F0F_0F0Fu32), (4, 0x00FF_00FFu32), (8, 0x0000_FFFFu32)]);
impl_uninterleavable!(u64, u32, 0x5555_5555_5555_5555u64, [
    (1, 0x3333_3333_3333_3333u64), (2, 0x0F0F_0F0F_0F0F_0F0Fu64),
    (4, 0x00FF_00FF_00FF_00FFu64), (8, 0x0000_FFFF_0000_FFFFu64),
    (16, 0x0000_0000_FFFF_FFFFu64)]);
impl_uninterleavable!(u128, u64, 0x5555_5555_5555_5555_5555_5555_5555_5555u128, [
    (1, 0x3333_3333_3333_3333_3333_3333_3333_3333u128),
    (2, 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128),
    (4, 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128),
    (8, 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128),
    (16, 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128),
    (32, 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFFu128)]);

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

/// Batch uninterleave: local, no communication.
fn uninterleave_batch<T: Uninterleavable>(
    bits: &[Rep3RingShare<T>],
) -> (Vec<Rep3RingShare<T::Half>>, Vec<Rep3RingShare<T::Half>>)
where
    Standard: Distribution<T::Half>,
{
    bits.iter().map(|b| uninterleave_generic(*b)).unzip()
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

/// Re-interleave two public operand values for `suffix_mle` computation.
///
/// x occupies odd bit positions (1,3,5,...) and y occupies even positions (0,2,4,...),
/// matching the Jolt interleaving convention.
fn interleave_public_pair(x: u64, y: u64, half_bits: usize) -> u128 {
    let mut result = 0u128;
    for i in 0..half_bits {
        result |= (((x >> i) & 1) as u128) << (2 * i + 1);
        result |= (((y >> i) & 1) as u128) << (2 * i);
    }
    result
}

// ---------------------------------------------------------------------------
// SuffixFutureBatch — pre-bucketed suffix evaluation results
// ---------------------------------------------------------------------------

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
        for (idx, val) in indices.into_iter().zip(vals) {
            self.ready_idx.push(idx);
            self.ready.push(val);
        }
    }

    /// Push BitInject (single-bit) values with their output indices.
    pub fn extend_bitinject(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<Bit>>,
    ) {
        for (idx, val) in indices.into_iter().zip(vals) {
            self.bitinject_idx.push(idx);
            self.bitinject.push(val);
        }
    }

    /// Push B2A values for a specific ring type, dispatched by `TypeId`.
    pub fn extend_b2a_ring<R: IntRing2k>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<R>>,
    ) {
        use std::any::TypeId;
        let tid = TypeId::of::<R>();
        if tid == TypeId::of::<u8>() {
            for (idx, val) in indices.into_iter().zip(vals) {
                self.b2a_u8_idx.push(idx);
                // Safety: we just checked R == u8
                let raw: u128 = val.a.0.into();
                let raw_b: u128 = val.b.0.into();
                self.b2a_u8.push(Rep3RingShare {
                    a: RingElement(raw as u8),
                    b: RingElement(raw_b as u8),
                });
            }
        } else if tid == TypeId::of::<u16>() {
            for (idx, val) in indices.into_iter().zip(vals) {
                self.b2a_u16_idx.push(idx);
                let raw: u128 = val.a.0.into();
                let raw_b: u128 = val.b.0.into();
                self.b2a_u16.push(Rep3RingShare {
                    a: RingElement(raw as u16),
                    b: RingElement(raw_b as u16),
                });
            }
        } else if tid == TypeId::of::<u32>() {
            for (idx, val) in indices.into_iter().zip(vals) {
                self.b2a_u32_idx.push(idx);
                let raw: u128 = val.a.0.into();
                let raw_b: u128 = val.b.0.into();
                self.b2a_u32.push(Rep3RingShare {
                    a: RingElement(raw as u32),
                    b: RingElement(raw_b as u32),
                });
            }
        } else if tid == TypeId::of::<u64>() {
            for (idx, val) in indices.into_iter().zip(vals) {
                self.b2a_u64_idx.push(idx);
                let raw: u128 = val.a.0.into();
                let raw_b: u128 = val.b.0.into();
                self.b2a_u64.push(Rep3RingShare {
                    a: RingElement(raw as u64),
                    b: RingElement(raw_b as u64),
                });
            }
        } else if tid == TypeId::of::<u128>() {
            for (idx, val) in indices.into_iter().zip(vals) {
                self.b2a_u128_idx.push(idx);
                let raw: u128 = val.a.0.into();
                let raw_b: u128 = val.b.0.into();
                self.b2a_u128.push(Rep3RingShare {
                    a: RingElement(raw),
                    b: RingElement(raw_b),
                });
            }
        } else {
            panic!("Unsupported ring type for B2A");
        }
    }

    /// Typed B2A push for u8.
    pub fn extend_b2a_u8(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<u8>>,
    ) {
        for (idx, val) in indices.into_iter().zip(vals) {
            self.b2a_u8_idx.push(idx);
            self.b2a_u8.push(val);
        }
    }

    /// Typed B2A push for u16.
    pub fn extend_b2a_u16(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<u16>>,
    ) {
        for (idx, val) in indices.into_iter().zip(vals) {
            self.b2a_u16_idx.push(idx);
            self.b2a_u16.push(val);
        }
    }

    /// Typed B2A push for u32.
    pub fn extend_b2a_u32(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<u32>>,
    ) {
        for (idx, val) in indices.into_iter().zip(vals) {
            self.b2a_u32_idx.push(idx);
            self.b2a_u32.push(val);
        }
    }

    /// Typed B2A push for u64.
    pub fn extend_b2a_u64(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<u64>>,
    ) {
        for (idx, val) in indices.into_iter().zip(vals) {
            self.b2a_u64_idx.push(idx);
            self.b2a_u64.push(val);
        }
    }

    /// Typed B2A push for u128.
    pub fn extend_b2a_u128(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<u128>>,
    ) {
        for (idx, val) in indices.into_iter().zip(vals) {
            self.b2a_u128_idx.push(idx);
            self.b2a_u128.push(val);
        }
    }

    /// Fulfill all pending conversions and scatter into output vec.
    pub fn fulfill_with_pool<N: Rep3NetworkWorker>(
        self,
        io_ctx: &mut IoContextPool<N>,
        pool: &mut EdaBitsPool<F>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        use mpc_core::protocols::rep3_ring::edabits;

        let mut out = vec![Rep3PrimeFieldShare::default(); self.len];

        // Ready — direct scatter
        for k in 0..self.ready.len() {
            out[self.ready_idx[k]] = self.ready[k];
        }

        // BitInject — single-bit → field via daBits
        if !self.bitinject.is_empty() {
            let dabits = pool.take_dabits(self.bitinject.len());
            let fields =
                edabits::bit_inject_field_many(&self.bitinject, &dabits, io_ctx.main())?;
            for k in 0..fields.len() {
                out[self.bitinject_idx[k]] = fields[k];
            }
        }

        // B2A per ring type
        macro_rules! fulfill_b2a {
            ($ring:ty, $idx:ident, $val:ident) => {
                if !self.$val.is_empty() {
                    let batch = pool.take_edabits::<$ring>(self.$val.len());
                    let fields = edabits::ring_to_field_b2a_many::<$ring, F, _>(
                        &self.$val, &batch, io_ctx.main(),
                    )?;
                    for k in 0..fields.len() {
                        out[self.$idx[k]] = fields[k];
                    }
                }
            };
        }
        fulfill_b2a!(u8, b2a_u8_idx, b2a_u8);
        fulfill_b2a!(u16, b2a_u16_idx, b2a_u16);
        fulfill_b2a!(u32, b2a_u32_idx, b2a_u32);
        fulfill_b2a!(u64, b2a_u64_idx, b2a_u64);
        fulfill_b2a!(u128, b2a_u128_idx, b2a_u128);

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
        SuffixBitsBatch::Uninterleaved(left, right) => {
            eval_suffix_uninterleaved::<T, F, N>(
                suffix, left, right, suffix_len, io_ctx, party_id, base, out,
            )
        }
        SuffixBitsBatch::Interleaved(bits) => {
            eval_suffix_interleaved::<T, F, N>(
                suffix, bits, suffix_len, io_ctx, party_id, base, out,
            )
        }
    }
}

/// Handle public entries for an uninterleaved suffix: compute via suffix_mle.
/// Returns the positions and shares for entries that need MPC evaluation.
fn split_uninterleaved_public<T: Uninterleavable, F: JoltField>(
    suffix: &Suffixes,
    left: &MixedBatch<u64, T::Half>,
    right: &MixedBatch<u64, T::Half>,
    suffix_len: usize,
    party_id: PartyID,
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) -> (Vec<usize>, Vec<Rep3RingShare<T::Half>>, Vec<Rep3RingShare<T::Half>>)
where
    Standard: Distribution<T::Half>,
{
    let half_bits = suffix_len / 2;

    match (left, right) {
        (MixedBatch::Shared(xs), MixedBatch::Shared(ys)) => {
            let positions: Vec<usize> = (0..xs.len()).collect();
            (positions, xs.clone(), ys.clone())
        }
        (MixedBatch::Shared(xs), MixedBatch::Public(y_pubs)) => {
            // All right operands are public → promote to trivial shares
            let ys: Vec<Rep3RingShare<T::Half>> = y_pubs
                .iter()
                .map(|&yp| {
                    let val = T::Half::try_from(yp as u128).unwrap_or_else(|_| {
                        // Truncate to H bits
                        T::Half::try_from(yp as u128 & ((1u128 << T::Half::K) - 1))
                            .unwrap_or_else(|_| unreachable!())
                    });
                    rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(val))
                })
                .collect();
            let positions: Vec<usize> = (0..xs.len()).collect();
            (positions, xs.clone(), ys)
        }
        (MixedBatch::Public(x_pubs), MixedBatch::Public(y_pubs)) => {
            // All public → compute locally via suffix_mle
            for (i, (&xp, &yp)) in x_pubs.iter().zip(y_pubs.iter()).enumerate() {
                let interleaved = interleave_public_pair(xp, yp, half_bits);
                let val = suffix.suffix_mle::<XLEN>(LookupBits::new(interleaved, suffix_len));
                out.extend_ready(
                    std::iter::once(base + i),
                    std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                        &F::from_u64(val),
                        party_id,
                    )),
                );
            }
            (Vec::new(), Vec::new(), Vec::new())
        }
        (MixedBatch::Shared(xs), MixedBatch::Mixed(ys_mixed)) => {
            // Some right operands are public, some shared.
            // Promote public right to trivial share for uniform MPC treatment.
            let mut positions = Vec::with_capacity(xs.len());
            let mut out_xs = Vec::with_capacity(xs.len());
            let mut out_ys = Vec::with_capacity(xs.len());
            for (i, y_entry) in ys_mixed.iter().enumerate() {
                positions.push(i);
                out_xs.push(xs[i]);
                match y_entry {
                    Either::Public(yp) => {
                        let val = T::Half::try_from(*yp as u128).unwrap_or_else(|_| {
                            T::Half::try_from(*yp as u128 & ((1u128 << T::Half::K) - 1))
                                .unwrap_or_else(|_| unreachable!())
                        });
                        out_ys.push(rep3_ring::binary::promote_to_trivial_share(
                            party_id,
                            &RingElement(val),
                        ));
                    }
                    Either::Shared(ys) => {
                        out_ys.push(*ys);
                    }
                }
            }
            (positions, out_xs, out_ys)
        }
        (MixedBatch::Mixed(xs_mixed), MixedBatch::Mixed(ys_mixed)) => {
            // Both can be public or shared. Separate fully-public from rest.
            let mut positions = Vec::new();
            let mut out_xs = Vec::new();
            let mut out_ys = Vec::new();
            for (i, (x_entry, y_entry)) in xs_mixed.iter().zip(ys_mixed.iter()).enumerate() {
                match (x_entry, y_entry) {
                    (Either::Public(xp), Either::Public(yp)) => {
                        let interleaved = interleave_public_pair(*xp, *yp, half_bits);
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
                    _ => {
                        positions.push(i);
                        let x_share = match x_entry {
                            Either::Shared(s) => *s,
                            Either::Public(xp) => {
                                let val =
                                    T::Half::try_from(*xp as u128).unwrap_or_else(|_| {
                                        T::Half::try_from(
                                            *xp as u128 & ((1u128 << T::Half::K) - 1),
                                        )
                                        .unwrap_or_else(|_| unreachable!())
                                    });
                                rep3_ring::binary::promote_to_trivial_share(
                                    party_id,
                                    &RingElement(val),
                                )
                            }
                        };
                        let y_share = match y_entry {
                            Either::Shared(s) => *s,
                            Either::Public(yp) => {
                                let val =
                                    T::Half::try_from(*yp as u128).unwrap_or_else(|_| {
                                        T::Half::try_from(
                                            *yp as u128 & ((1u128 << T::Half::K) - 1),
                                        )
                                        .unwrap_or_else(|_| unreachable!())
                                    });
                                rep3_ring::binary::promote_to_trivial_share(
                                    party_id,
                                    &RingElement(val),
                                )
                            }
                        };
                        out_xs.push(x_share);
                        out_ys.push(y_share);
                    }
                }
            }
            (positions, out_xs, out_ys)
        }
        // Other combinations (Mixed left, Shared/Public right) — shouldn't occur
        // in practice but handle by promoting to shares.
        (MixedBatch::Mixed(xs_mixed), MixedBatch::Shared(ys)) => {
            let mut positions = Vec::new();
            let mut out_xs = Vec::new();
            let mut out_ys = Vec::new();
            for (i, x_entry) in xs_mixed.iter().enumerate() {
                match x_entry {
                    Either::Public(xp) => {
                        // We don't have the public y value, so promote x and treat as shared
                        let val = T::Half::try_from(*xp as u128).unwrap_or_else(|_| {
                            T::Half::try_from(*xp as u128 & ((1u128 << T::Half::K) - 1))
                                .unwrap_or_else(|_| unreachable!())
                        });
                        positions.push(i);
                        out_xs.push(rep3_ring::binary::promote_to_trivial_share(
                            party_id,
                            &RingElement(val),
                        ));
                        out_ys.push(ys[i]);
                    }
                    Either::Shared(s) => {
                        positions.push(i);
                        out_xs.push(*s);
                        out_ys.push(ys[i]);
                    }
                }
            }
            (positions, out_xs, out_ys)
        }
        (MixedBatch::Public(x_pubs), MixedBatch::Shared(ys)) => {
            let xs: Vec<Rep3RingShare<T::Half>> = x_pubs
                .iter()
                .map(|&xp| {
                    let val = T::Half::try_from(xp as u128).unwrap_or_else(|_| {
                        T::Half::try_from(xp as u128 & ((1u128 << T::Half::K) - 1))
                            .unwrap_or_else(|_| unreachable!())
                    });
                    rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(val))
                })
                .collect();
            let positions: Vec<usize> = (0..xs.len()).collect();
            (positions, xs, ys.clone())
        }
        (MixedBatch::Public(x_pubs), MixedBatch::Mixed(ys_mixed)) => {
            let mut positions = Vec::new();
            let mut out_xs = Vec::new();
            let mut out_ys = Vec::new();
            for (i, (xp, y_entry)) in x_pubs.iter().zip(ys_mixed.iter()).enumerate() {
                match y_entry {
                    Either::Public(yp) => {
                        let interleaved = interleave_public_pair(*xp, *yp, half_bits);
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
                    Either::Shared(ys) => {
                        let val = T::Half::try_from(*xp as u128).unwrap_or_else(|_| {
                            T::Half::try_from(*xp as u128 & ((1u128 << T::Half::K) - 1))
                                .unwrap_or_else(|_| unreachable!())
                        });
                        positions.push(i);
                        out_xs.push(rep3_ring::binary::promote_to_trivial_share(
                            party_id,
                            &RingElement(val),
                        ));
                        out_ys.push(*ys);
                    }
                }
            }
            (positions, out_xs, out_ys)
        }
        (MixedBatch::Mixed(xs_mixed), MixedBatch::Public(y_pubs)) => {
            let mut positions = Vec::new();
            let mut out_xs = Vec::new();
            let mut out_ys = Vec::new();
            for (i, (x_entry, yp)) in xs_mixed.iter().zip(y_pubs.iter()).enumerate() {
                match x_entry {
                    Either::Public(xp) => {
                        let interleaved = interleave_public_pair(*xp, *yp, half_bits);
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
                        let val = T::Half::try_from(*yp as u128).unwrap_or_else(|_| {
                            T::Half::try_from(*yp as u128 & ((1u128 << T::Half::K) - 1))
                                .unwrap_or_else(|_| unreachable!())
                        });
                        positions.push(i);
                        out_xs.push(*s);
                        out_ys.push(rep3_ring::binary::promote_to_trivial_share(
                            party_id,
                            &RingElement(val),
                        ));
                    }
                }
            }
            (positions, out_xs, out_ys)
        }
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
                        let val =
                            suffix.suffix_mle::<XLEN>(LookupBits::new(*p & mask, suffix_len));
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

/// Evaluate an uninterleaved suffix on shared operands, pushing into SuffixFutureBatch.
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
    // Split public/shared entries. Public entries are pushed as Ready directly.
    let (positions, xs, ys) =
        split_uninterleaved_public::<T, F>(suffix, left, right, suffix_len, party_id, base, out);

    if xs.is_empty() {
        return Ok(());
    }

    let indices = positions.iter().map(|&i| base + i);

    match suffix {
        Suffixes::One => unreachable!("handled above"),

        // --- B2A(H): bitwise ops ---
        Suffixes::And => {
            let result = rep3_ring::binary::and_many::<T::Half, _>(&xs, &ys, io_ctx)?;
            out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
        }
        Suffixes::NotAnd => {
            let not_ys: Vec<_> = ys.iter().map(|y| !y).collect();
            let result = rep3_ring::binary::and_many::<T::Half, _>(&xs, &not_ys, io_ctx)?;
            out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
        }
        Suffixes::Xor => {
            let result: Vec<_> = xs.iter().zip(ys.iter()).map(|(x, y)| *x ^ *y).collect();
            out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
        }
        Suffixes::Or => {
            let result = rep3_ring::binary::or_many::<T::Half, _>(&xs, &ys, io_ctx)?;
            out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
        }

        // --- B2A(H): value extraction ---
        Suffixes::RightOperand => {
            out.extend_b2a_ring::<T::Half>(indices, ys.into_iter());
        }
        Suffixes::RightOperandW => {
            if T::Half::K >= 32 {
                let m: u128 = (1u128 << 32) - 1;
                let mask_val = T::Half::try_from(m).unwrap_or_else(|_| unreachable!());
                let masked: Vec<_> = ys.iter().map(|y| *y & RingElement(mask_val)).collect();
                out.extend_b2a_ring::<T::Half>(indices, masked.into_iter());
            } else {
                out.extend_b2a_ring::<T::Half>(indices, ys.into_iter());
            }
        }
        Suffixes::Lsb => {
            let one = T::Half::try_from(1u128).unwrap_or_else(|_| unreachable!());
            let result: Vec<_> = ys.iter().map(|y| *y & RingElement(one)).collect();
            out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
        }

        // --- BitInject: comparisons ---
        Suffixes::LessThan => {
            let ge_bits: Vec<Rep3RingShare<Bit>> =
                rep3_ring::arithmetic::ge_many::<T::Half, _>(&xs, &ys, io_ctx)?;
            let lt_bits: Vec<_> = ge_bits.iter().map(|b| !b).collect();
            out.extend_bitinject(indices, lt_bits.into_iter());
        }
        Suffixes::GreaterThan => {
            let ge_bits: Vec<Rep3RingShare<Bit>> =
                rep3_ring::arithmetic::ge_many::<T::Half, _>(&ys, &xs, io_ctx)?;
            let gt_bits: Vec<_> = ge_bits.iter().map(|b| !b).collect();
            out.extend_bitinject(indices, gt_bits.into_iter());
        }
        Suffixes::Eq => {
            let diff: Vec<Rep3RingShare<T::Half>> =
                xs.iter().zip(ys.iter()).map(|(x, y)| *x ^ *y).collect();
            let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&diff, io_ctx)?;
            out.extend_bitinject(indices, eq_bits.into_iter());
        }
        Suffixes::LeftOperandIsZero => {
            let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&xs, io_ctx)?;
            out.extend_bitinject(indices, eq_bits.into_iter());
        }
        Suffixes::RightOperandIsZero => {
            let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&ys, io_ctx)?;
            out.extend_bitinject(indices, eq_bits.into_iter());
        }
        Suffixes::DivByZero => {
            // divisor==0 AND quotient==all_ones
            // Here x=divisor (left), y=quotient (right) based on interleaving convention
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
            let divisor_zero =
                rep3_ring::binary::is_zero_many::<T::Half, _>(&xs, io_ctx)?;
            let quotient_all_ones =
                rep3_ring::binary::is_zero_many::<T::Half, _>(&q_xor, io_ctx)?;
            let result =
                rep3_ring::binary::and_many::<Bit, _>(&divisor_zero, &quotient_all_ones, io_ctx)?;
            out.extend_bitinject(indices, result.into_iter());
        }
        Suffixes::TwoLsb => {
            // is_zero_bit(x[0]) AND is_zero_bit(y[0])
            let x_lsb: Vec<Rep3RingShare<Bit>> =
                xs.iter().map(|x| downcast::<T::Half, Bit>(*x)).collect();
            let y_lsb: Vec<Rep3RingShare<Bit>> =
                ys.iter().map(|y| downcast::<T::Half, Bit>(*y)).collect();
            // NOT x_lsb AND NOT y_lsb = NOT(x_lsb OR y_lsb)
            let x_or_y = rep3_ring::binary::or_many::<Bit, _>(&x_lsb, &y_lsb, io_ctx)?;
            let result: Vec<_> = x_or_y.iter().map(|b| !b).collect();
            out.extend_bitinject(indices, result.into_iter());
        }

        // --- BitInject: change divisor ---
        Suffixes::ChangeDivisor => {
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
            let y_eq_all_ones =
                rep3_ring::binary::is_zero_many::<T::Half, _>(&y_xor, io_ctx)?;
            let x_eq_zero =
                rep3_ring::binary::is_zero_many::<T::Half, _>(&xs, io_ctx)?;
            let result =
                rep3_ring::binary::and_many::<Bit, _>(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
            out.extend_bitinject(indices, result.into_iter());
        }
        Suffixes::ChangeDivisorW => {
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
            let y_eq_all_ones =
                rep3_ring::binary::is_zero_many::<u32, _>(&y_xor, io_ctx)?;
            let x_eq_zero =
                rep3_ring::binary::is_zero_many::<u32, _>(&xs32, io_ctx)?;
            let result =
                rep3_ring::binary::and_many::<Bit, _>(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
            out.extend_bitinject(indices, result.into_iter());
        }

        // --- Ready: sign extension ---
        Suffixes::SignExtension => {
            // Check if right operand is public — use local computation
            let has_public_right = matches!(right, MixedBatch::Public(_));
            if has_public_right {
                // All right operands public → compute sign extension locally
                let y_pubs = match right {
                    MixedBatch::Public(pubs) => pubs,
                    _ => unreachable!(),
                };
                for (i, &yp) in y_pubs.iter().enumerate() {
                    let val = compute_sign_extension_from_mask(yp, suffix_len);
                    out.extend_ready(
                        std::iter::once(base + i),
                        std::iter::once(Rep3PrimeFieldShare::promote_from_trivial(
                            &F::from_u64(val),
                            party_id,
                        )),
                    );
                }
            } else {
                // Fall back to MPC eval (already have xs, ys from split)
                let result = eval_sign_extension_from_ys::<T, F, N>(
                    &ys, suffix_len, io_ctx, party_id,
                )?;
                out.extend_ready(indices, result.into_iter());
            }
        }
        Suffixes::SignExtensionRightOperand => {
            if suffix_len < XLEN {
                let one = Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id);
                out.extend_ready(indices, std::iter::repeat(one).take(xs.len()));
            } else {
                let sign_bit_pos = XLEN / 2 - 1;
                let sign_bits: Vec<Rep3RingShare<Bit>> = ys
                    .iter()
                    .map(|y| downcast::<T::Half, Bit>(*y >> sign_bit_pos))
                    .collect();
                let weight = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)));
                let sign_field: Vec<Rep3PrimeFieldShare<F>> =
                    rep3_ring::conversion::bit_inject_from_bits_to_field_many(
                        &sign_bits, io_ctx,
                    )?;
                let result: Vec<_> = sign_field.into_iter().map(|s| s * weight).collect();
                out.extend_ready(indices, result.into_iter());
            }
        }

        // --- B2A(H): XOR-rotate ---
        Suffixes::XorRot16 => {
            eval_xor_rot_uninterleaved::<16, T, F>(
                &xs, &ys, indices, out,
            );
        }
        Suffixes::XorRot24 => {
            eval_xor_rot_uninterleaved::<24, T, F>(
                &xs, &ys, indices, out,
            );
        }
        Suffixes::XorRot32 => {
            eval_xor_rot_uninterleaved::<32, T, F>(
                &xs, &ys, indices, out,
            );
        }
        Suffixes::XorRot63 => {
            eval_xor_rot_uninterleaved::<63, T, F>(
                &xs, &ys, indices, out,
            );
        }

        // --- B2A(u32 or H): XOR-rotate W-variants ---
        Suffixes::XorRotW7 => {
            eval_xor_rot_w_uninterleaved::<7, T, F>(&xs, &ys, &positions, base, out);
        }
        Suffixes::XorRotW8 => {
            eval_xor_rot_w_uninterleaved::<8, T, F>(&xs, &ys, &positions, base, out);
        }
        Suffixes::XorRotW12 => {
            eval_xor_rot_w_uninterleaved::<12, T, F>(&xs, &ys, &positions, base, out);
        }
        Suffixes::XorRotW16 => {
            eval_xor_rot_w_uninterleaved::<16, T, F>(&xs, &ys, &positions, base, out);
        }

        // --- Shift suffixes (right operand is always public for shift tables) ---
        Suffixes::RightShift => {
            eval_right_shift_public_y::<T, F>(&xs, right, &positions, base, out);
        }
        Suffixes::RightShiftHelper => {
            eval_right_shift_helper_public_y::<T, F>(right, &positions, base, party_id, out);
        }
        Suffixes::RightShiftPadding => {
            // RightShiftPadding uses a different formula: 1 << (XLEN - 1 - shift)
            // where shift = low log2(XLEN) bits. But this is only used with
            // a prefix that provides the leading bits. For the suffix part:
            eval_right_shift_padding_public_y::<T, F>(
                right, suffix_len, &positions, base, party_id, out,
            );
        }
        Suffixes::LeftShift => {
            eval_left_shift_public_y::<T, F>(&xs, right, &positions, base, out);
        }
        Suffixes::RightShiftW => {
            eval_right_shift_w_public_y::<T, F>(&xs, right, &positions, base, out);
        }
        Suffixes::RightShiftWHelper => {
            eval_right_shift_w_helper_public_y::<T, F>(
                right, suffix_len, &positions, base, party_id, out,
            );
        }
        Suffixes::LeftShiftW => {
            eval_left_shift_w_public_y::<T, F>(&xs, right, &positions, base, out);
        }
        Suffixes::LeftShiftWHelper => {
            // LeftShiftWHelper: (1 << y.leading_ones()) as u32
            eval_left_shift_w_helper_public_y::<T, F>(
                right, &positions, base, party_id, out,
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
                    .map(|r| {
                        Rep3RingShare {
                            a: RingElement(
                                T::Half::try_from(r.a.0 as u128)
                                    .unwrap_or_else(|_| unreachable!()),
                            ),
                            b: RingElement(
                                T::Half::try_from(r.b.0 as u128)
                                    .unwrap_or_else(|_| unreachable!()),
                            ),
                        }
                    })
                    .collect();
                out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
            } else {
                out.extend_b2a_u32(indices, reversed_u32.into_iter());
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
                    rep3_ring::conversion::bit_inject_from_bits_to_field_many(
                        &sign_bits, io_ctx,
                    )?;
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
                let upper: Vec<Rep3RingShare<T>> =
                    shared_bits.iter().map(|b| *b >> XLEN).collect();
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
// Shift suffix helpers (public right operand)
// ---------------------------------------------------------------------------

/// RightShift with public right operand: x >> trailing_zeros(public_y)
fn eval_right_shift_public_y<T: Uninterleavable, F: JoltField>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    positions: &[usize],
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) where
    Standard: Distribution<T::Half>,
{
    let y_pubs = extract_public_right_values(right, positions);
    let result: Vec<Rep3RingShare<T::Half>> = xs
        .iter()
        .zip(y_pubs.iter())
        .map(|(x, &yp)| {
            let shift = (yp as u128).trailing_zeros() as usize;
            *x >> shift
        })
        .collect();
    let indices = positions.iter().map(|&i| base + i);
    out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
}

/// RightShiftHelper with public right operand: 1 << leading_ones(public_y)
fn eval_right_shift_helper_public_y<T: Uninterleavable, F: JoltField>(
    right: &MixedBatch<u64, T::Half>,
    positions: &[usize],
    base: usize,
    party_id: PartyID,
    out: &mut SuffixFutureBatch<F>,
) {
    let y_pubs = extract_public_right_values(right, positions);
    let indices = positions.iter().map(|&i| base + i);
    let result: Vec<Rep3PrimeFieldShare<F>> = y_pubs
        .iter()
        .map(|&yp| {
            let lo = yp.leading_ones() as u64;
            let val = 1u64 << lo;
            Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(val), party_id)
        })
        .collect();
    out.extend_ready(indices, result.into_iter());
}

/// RightShiftPadding with public right operand: 1 << (XLEN - 1 - shift)
fn eval_right_shift_padding_public_y<T: Uninterleavable, F: JoltField>(
    right: &MixedBatch<u64, T::Half>,
    suffix_len: usize,
    positions: &[usize],
    base: usize,
    party_id: PartyID,
    out: &mut SuffixFutureBatch<F>,
) {
    let y_pubs = extract_public_right_values(right, positions);
    let indices = positions.iter().map(|&i| base + i);
    let log_xlen = XLEN.log_2();
    let result: Vec<Rep3PrimeFieldShare<F>> = y_pubs
        .iter()
        .map(|&yp| {
            // Extract shift from low log2(XLEN) bits of y
            let shift_mask = (1u64 << log_xlen.min(suffix_len / 2)) - 1;
            let shift = (yp & shift_mask) as usize;
            let val = 1u128 << (XLEN - 1 - shift);
            Rep3PrimeFieldShare::promote_from_trivial(&F::from_u128(val), party_id)
        })
        .collect();
    out.extend_ready(indices, result.into_iter());
}

/// LeftShift with public right operand: (x & !y_mask) << leading_ones(y)
fn eval_left_shift_public_y<T: Uninterleavable, F: JoltField>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    positions: &[usize],
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) where
    Standard: Distribution<T::Half>,
{
    let y_pubs = extract_public_right_values(right, positions);
    let result: Vec<Rep3RingShare<T::Half>> = xs
        .iter()
        .zip(y_pubs.iter())
        .map(|(x, &yp)| {
            let y_mask = T::Half::try_from(yp as u128)
                .unwrap_or_else(|_| {
                    T::Half::try_from(yp as u128 & ((1u128 << T::Half::K) - 1))
                        .unwrap_or_else(|_| unreachable!())
                });
            // x & !y_mask
            let masked = *x & RingElement(!y_mask);
            let shift = (yp as u128).leading_ones() as usize;
            masked << shift
        })
        .collect();
    let indices = positions.iter().map(|&i| base + i);
    out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
}

/// RightShiftW with public right operand: (x as u32) >> trailing_zeros(y).min(XLEN/2)
fn eval_right_shift_w_public_y<T: Uninterleavable, F: JoltField>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    positions: &[usize],
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) where
    Standard: Distribution<T::Half>,
{
    let y_pubs = extract_public_right_values(right, positions);
    let result: Vec<Rep3RingShare<T::Half>> = xs
        .iter()
        .zip(y_pubs.iter())
        .map(|(x, &yp)| {
            let x32 = to_u32_share(*x);
            let shift = (yp as u128).trailing_zeros().min(XLEN as u32 / 2) as usize;
            let shifted = x32 >> shift;
            // Convert back to T::Half
            Rep3RingShare {
                a: RingElement(
                    T::Half::try_from(shifted.a.0 as u128).unwrap_or_else(|_| unreachable!()),
                ),
                b: RingElement(
                    T::Half::try_from(shifted.b.0 as u128).unwrap_or_else(|_| unreachable!()),
                ),
            }
        })
        .collect();
    let indices = positions.iter().map(|&i| base + i);
    out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
}

/// RightShiftWHelper with public right operand: 1 << leading_ones(y truncated to XLEN/2 bits)
fn eval_right_shift_w_helper_public_y<T: Uninterleavable, F: JoltField>(
    right: &MixedBatch<u64, T::Half>,
    suffix_len: usize,
    positions: &[usize],
    base: usize,
    party_id: PartyID,
    out: &mut SuffixFutureBatch<F>,
) {
    let y_pubs = extract_public_right_values(right, positions);
    let half_xlen = XLEN / 2;
    let y_bits = (suffix_len / 2).min(half_xlen);
    let indices = positions.iter().map(|&i| base + i);
    let result: Vec<Rep3PrimeFieldShare<F>> = y_pubs
        .iter()
        .map(|&yp| {
            let y_truncated = LookupBits::new(yp as u128, y_bits);
            let lo = y_truncated.leading_ones() as u64;
            let val = 1u64 << lo;
            Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(val), party_id)
        })
        .collect();
    out.extend_ready(indices, result.into_iter());
}

/// LeftShiftW with public right operand
fn eval_left_shift_w_public_y<T: Uninterleavable, F: JoltField>(
    xs: &[Rep3RingShare<T::Half>],
    right: &MixedBatch<u64, T::Half>,
    positions: &[usize],
    base: usize,
    out: &mut SuffixFutureBatch<F>,
) where
    Standard: Distribution<T::Half>,
{
    let y_pubs = extract_public_right_values(right, positions);
    let half_xlen = XLEN / 2;
    let result: Vec<Rep3RingShare<T::Half>> = xs
        .iter()
        .zip(y_pubs.iter())
        .map(|(x, &yp)| {
            let y_truncated_bits = yp & ((1u64 << half_xlen) - 1);
            let x32 = to_u32_share(*x);
            let y32_mask = y_truncated_bits as u32;
            let masked = x32 & RingElement(!y32_mask);
            let lo = LookupBits::new(y_truncated_bits as u128, half_xlen).leading_ones();
            let shifted = masked << lo as usize;
            Rep3RingShare {
                a: RingElement(
                    T::Half::try_from(shifted.a.0 as u128).unwrap_or_else(|_| unreachable!()),
                ),
                b: RingElement(
                    T::Half::try_from(shifted.b.0 as u128).unwrap_or_else(|_| unreachable!()),
                ),
            }
        })
        .collect();
    let indices = positions.iter().map(|&i| base + i);
    out.extend_b2a_ring::<T::Half>(indices, result.into_iter());
}

/// LeftShiftWHelper with public right operand: (1 << y.leading_ones()) as u32
fn eval_left_shift_w_helper_public_y<T: Uninterleavable, F: JoltField>(
    right: &MixedBatch<u64, T::Half>,
    positions: &[usize],
    base: usize,
    party_id: PartyID,
    out: &mut SuffixFutureBatch<F>,
) {
    let y_pubs = extract_public_right_values(right, positions);
    let indices = positions.iter().map(|&i| base + i);
    let result: Vec<Rep3PrimeFieldShare<F>> = y_pubs
        .iter()
        .map(|&yp| {
            let lo = yp.leading_ones() as u64;
            let val = (1u64 << lo) as u32 as u64;
            Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(val), party_id)
        })
        .collect();
    out.extend_ready(indices, result.into_iter());
}

/// Extract public right operand values from the MixedBatch.
/// For Shared right operand, panics (shift tables should always have public right).
fn extract_public_right_values<H: IntRing2k>(
    right: &MixedBatch<u64, H>,
    positions: &[usize],
) -> Vec<u64> {
    match right {
        MixedBatch::Public(pubs) => {
            positions.iter().map(|&i| pubs[i]).collect()
        }
        MixedBatch::Mixed(mixed) => {
            positions
                .iter()
                .map(|&i| match &mixed[i] {
                    Either::Public(p) => *p,
                    Either::Shared(_) => panic!("Shift suffix expects public right operand"),
                })
                .collect()
        }
        MixedBatch::Shared(_) => {
            panic!("Shift suffix expects public right operand, got all Shared");
        }
    }
}

// ---------------------------------------------------------------------------
// New-path helpers for sign extension and XOR-rotate
// ---------------------------------------------------------------------------

/// Sign extension from uninterleaved ys (right operands).
/// Same logic as eval_sign_extension but takes ys directly (no uninterleave).
fn eval_sign_extension_from_ys<T, F, N>(
    ys: &[Rep3RingShare<T::Half>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    T: Uninterleavable,
    Standard: Distribution<T::Half>,
    T::Half: AsPrimitive<Bit>,
    F: JoltField,
    N: Rep3Network,
{
    let y_len = suffix_len / 2;
    let n = ys.len();
    let mut result = vec![Rep3PrimeFieldShare::<F>::zero_share(); n];

    let mut bit_shares: Vec<Vec<Rep3RingShare<Bit>>> = Vec::with_capacity(y_len);
    for i in 0..y_len {
        let bits_i: Vec<Rep3RingShare<Bit>> = ys
            .iter()
            .map(|y| downcast::<T::Half, Bit>(*y >> i))
            .collect();
        bit_shares.push(bits_i);
    }

    let not_bits: Vec<Vec<Rep3RingShare<Bit>>> = bit_shares
        .iter()
        .map(|bs| bs.iter().map(|b| !b).collect())
        .collect();

    let mut running: Vec<Rep3RingShare<Bit>> = vec![
        rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(Bit::one()));
        n
    ];

    for p in 0..y_len {
        let indicator: Vec<Rep3RingShare<Bit>> =
            rep3_ring::binary::and_many::<Bit, _>(&running, &bit_shares[p], io_ctx)?;

        let padding_len = p;
        if XLEN >= padding_len {
            let weight = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN - padding_len)));
            if weight != F::zero() {
                let indicator_field: Vec<Rep3PrimeFieldShare<F>> =
                    rep3_ring::conversion::bit_inject_from_bits_to_field_many(&indicator, io_ctx)?;
                for j in 0..n {
                    result[j] = result[j] + indicator_field[j] * weight;
                }
            }
        }

        running = rep3_ring::binary::and_many::<Bit, _>(&running, &not_bits[p], io_ctx)?;
    }

    if XLEN >= y_len {
        let weight_y_len = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN - y_len)));
        if weight_y_len != F::zero() {
            let indicator_field: Vec<Rep3PrimeFieldShare<F>> =
                rep3_ring::conversion::bit_inject_from_bits_to_field_many(&running, io_ctx)?;
            for j in 0..n {
                result[j] = result[j] + indicator_field[j] * weight_y_len;
            }
        }
    }

    Ok(result)
}

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
    positions: &[usize],
    base: usize,
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

    let indices = positions.iter().map(|&i| base + i);
    if T::Half::K >= 32 {
        let result: Vec<Rep3RingShare<T::Half>> = rotated_u32
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
        out.extend_b2a_u32(indices, rotated_u32.into_iter());
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

    let eq_bits: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::is_zero_many::<T, _>(&xored, io_ctx)?;
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

// ---------------------------------------------------------------------------
// Operand Q suffix evaluations
// ---------------------------------------------------------------------------

/// Per-cycle suffix values for the operand Q polynomials.
pub struct OperandQSuffixEvals<F: JoltField> {
    pub left_operand: Vec<Rep3PrimeFieldShare<F>>,
    pub right_operand: Vec<Rep3PrimeFieldShare<F>>,
    pub identity: Vec<Rep3PrimeFieldShare<F>>,
}

/// Compute per-cycle suffix values for the operand Q polynomials.
///
/// Generic over `T: Uninterleavable` — the caller downcasts lookup_indices to the
/// smallest ring fitting `suffix_len` bits, so the EdaBits and B2A use fewer bits.
///
/// For suffix_len > 0, computes:
///   - left_operand[j] = uninterleave(suffix_bits_j).0 as field (T::Half ring B2A)
///   - right_operand[j] = uninterleave(suffix_bits_j).1 as field (T::Half ring B2A)
///   - identity[j] = suffix_bits_j as field (T ring B2A)
///
/// Input `suffix_bits` are pre-masked and downcast to `T` in **binary (XOR) domain**.
#[tracing::instrument(skip_all, name = "compute_operand_q_suffix_evals", fields(phase))]
pub fn compute_operand_q_suffix_evals<T, F, N>(
    suffix_bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
    pool: &mut mpc_core::protocols::rep3_ring::edabits::EdaBitsPool<F>,
) -> eyre::Result<OperandQSuffixEvals<F>>
where
    T: Uninterleavable,
    Standard: Distribution<T> + Distribution<T::Half>,
    T::Half: AsPrimitive<T>,
    F: JoltField,
    N: Rep3Network,
{
    use mpc_core::protocols::rep3_ring::edabits;

    // Identity: suffix_bits as field (T ring B2A via edaBits)
    let identity = {
        let batch = pool.take_edabits::<T>(suffix_bits.len());
        let xs: Vec<_> = suffix_bits.iter().copied().collect();
        edabits::ring_to_field_b2a_many::<T, F, _>(&xs, &batch, io_ctx)?
    };

    // Uninterleave (local) for left/right operands, then B2A via edaBits (T::Half ring)
    let (xs, ys) = uninterleave_batch(suffix_bits);
    let left_operand = {
        let batch = pool.take_edabits::<T::Half>(xs.len());
        edabits::ring_to_field_b2a_many::<T::Half, F, _>(&xs, &batch, io_ctx)?
    };
    let right_operand = {
        let batch = pool.take_edabits::<T::Half>(ys.len());
        edabits::ring_to_field_b2a_many::<T::Half, F, _>(&ys, &batch, io_ctx)?
    };

    Ok(OperandQSuffixEvals {
        left_operand,
        right_operand,
        identity,
    })
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

            let (x0, y0) = uninterleave_generic::<u128>(share0);
            let (x1, y1) = uninterleave_generic::<u128>(share1);
            let (x2, y2) = uninterleave_generic::<u128>(share2);

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
