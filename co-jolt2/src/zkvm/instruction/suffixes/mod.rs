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
use crate::utils::future_ring::{FutureOp, FutureRep3Ring};
use jolt2_common::constants::XLEN;
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

macro_rules! impl_uninterleavable {
    ($full:ty, $half:ty) => {
        impl Uninterleavable for $full {
            type Half = $half;
            fn uninterleave(s: Rep3RingShare<Self>) -> (Rep3RingShare<$half>, Rep3RingShare<$half>)
            where
                Standard: Distribution<$half>,
            {
                let one: $full = 1;
                let half_k = <$half>::K;
                let mut x = Rep3RingShare {
                    a: RingElement(0 as $half),
                    b: RingElement(0 as $half),
                };
                let mut y = Rep3RingShare {
                    a: RingElement(0 as $half),
                    b: RingElement(0 as $half),
                };
                for i in 0..half_k {
                    // x[i] = (s >> (2*i + 1)) & 1, placed at bit position i
                    let x_bit = (s >> (2 * i + 1)) & RingElement(one);
                    x.a.0 |= (x_bit.a.0 as $half) << i;
                    x.b.0 |= (x_bit.b.0 as $half) << i;
                    // y[i] = (s >> (2*i)) & 1, placed at bit position i
                    let y_bit = (s >> (2 * i)) & RingElement(one);
                    y.a.0 |= (y_bit.a.0 as $half) << i;
                    y.b.0 |= (y_bit.b.0 as $half) << i;
                }
                (x, y)
            }
        }
    };
}

impl_uninterleavable!(u16, u8);
impl_uninterleavable!(u32, u16);
impl_uninterleavable!(u64, u32);
impl_uninterleavable!(u128, u64);

/// Suffix evaluation future parameterized by half-ring type H.
pub type SuffixFuture<H, F> = FutureRep3Ring<H, Rep3PrimeFieldShare<F>>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Zero-extend a smaller binary share to a larger ring.
/// Local operation (no communication).
fn zext<S: IntRing2k, D: IntRing2k>(s: Rep3RingShare<S>) -> Rep3RingShare<D>
where
    S: AsPrimitive<D>,
{
    Rep3RingShare {
        a: RingElement(s.a.0.as_()),
        b: RingElement(s.b.0.as_()),
    }
}

/// Truncate or zero-extend a share to u32 (used by W-variant instructions).
fn to_u32_share<H: IntRing2k>(s: Rep3RingShare<H>) -> Rep3RingShare<u32> {
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
// Top-level dispatch
// ---------------------------------------------------------------------------

/// Evaluate a suffix MLE on a batch of secret suffix bits, producing
/// per-cycle suffix futures (deferred field conversion).
///
/// `bits[j]` is a **binary-domain** `Rep3RingShare<T>` representing the low
/// `suffix_len` bits of cycle j's lookup index, where `T` is the smallest
/// ring fitting `suffix_len` bits.
///
/// Returns `Vec<SuffixFuture<T::Half, F>>` — callers must call
/// `fulfill_batched_with_pool` to convert to `Vec<Rep3PrimeFieldShare<F>>`.
pub fn evaluate_suffix_mle_batched<T, F, N>(
    suffix: &Suffixes,
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    T: Uninterleavable + AsPrimitive<Bit>,
    Standard: Distribution<T> + Distribution<T::Half>,
    T::Half: AsPrimitive<T> + AsPrimitive<Bit>,
    F: JoltField,
    N: Rep3Network,
{
    let n = bits.len();

    // suffix_len == 0 is handled by caller (constant value).
    debug_assert!(suffix_len > 0);

    match suffix {
        Suffixes::One => Ok(vec![
            SuffixFuture::Ready(
                Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id,)
            );
            n
        ]),

        // --- Simple bitwise (uninterleave + local bitwise op) ---
        Suffixes::And => eval_and(bits, io_ctx),
        Suffixes::NotAnd => eval_notand(bits, io_ctx),
        Suffixes::Xor => eval_xor(bits),
        Suffixes::Or => eval_or(bits, io_ctx),

        // --- Value extraction ---
        Suffixes::RightOperand => Ok(eval_right_operand(bits)),
        Suffixes::RightOperandW => Ok(eval_right_operand_w(bits)),
        Suffixes::UpperWord => eval_upper_word(bits, io_ctx),
        Suffixes::LowerWord => eval_lower_word(bits, io_ctx),
        Suffixes::LowerHalfWord => eval_lower_half_word(bits, io_ctx),
        Suffixes::Lsb => Ok(eval_lsb(bits)),
        Suffixes::TwoLsb => eval_two_lsb(bits, io_ctx),

        // --- Comparisons ---
        Suffixes::LessThan => eval_less_than(bits, io_ctx),
        Suffixes::GreaterThan => eval_greater_than(bits, io_ctx),
        Suffixes::Eq => eval_eq(bits, io_ctx),
        Suffixes::LeftOperandIsZero => eval_left_is_zero(bits, io_ctx),
        Suffixes::RightOperandIsZero => eval_right_is_zero(bits, io_ctx),
        Suffixes::DivByZero => eval_div_by_zero(bits, suffix_len, io_ctx, party_id),
        Suffixes::OverflowBitsZero => eval_overflow_bits_zero(bits, io_ctx),

        // --- Change divisor ---
        Suffixes::ChangeDivisor => eval_change_divisor(bits, suffix_len, io_ctx, party_id),
        Suffixes::ChangeDivisorW => eval_change_divisor_w(bits, suffix_len, io_ctx, party_id),

        // --- Pow2 ---
        Suffixes::Pow2 => eval_pow2(bits, suffix_len, io_ctx, party_id),
        Suffixes::Pow2W => eval_pow2_w(bits, suffix_len, io_ctx, party_id),

        // --- Sign extension ---
        Suffixes::SignExtension => eval_sign_extension(bits, suffix_len, io_ctx, party_id),
        Suffixes::SignExtensionUpperHalf => {
            eval_sign_extension_upper_half(bits, suffix_len, io_ctx, party_id)
        }
        Suffixes::SignExtensionRightOperand => {
            eval_sign_extension_right_operand(bits, suffix_len, io_ctx, party_id)
        }

        // --- Right shift / left shift (bitmask-based) ---
        Suffixes::RightShift
        | Suffixes::RightShiftHelper
        | Suffixes::RightShiftPadding
        | Suffixes::LeftShift
        | Suffixes::RightShiftW
        | Suffixes::RightShiftWHelper
        | Suffixes::LeftShiftWHelper
        | Suffixes::LeftShiftW => {
            eval_with_vanilla_open(bits, suffix_len, suffix, io_ctx, party_id)
        }

        // --- XOR-rotate ---
        Suffixes::XorRot16 => Ok(eval_xor_rot::<16, T, F>(bits)),
        Suffixes::XorRot24 => Ok(eval_xor_rot::<24, T, F>(bits)),
        Suffixes::XorRot32 => Ok(eval_xor_rot::<32, T, F>(bits)),
        Suffixes::XorRot63 => Ok(eval_xor_rot::<63, T, F>(bits)),
        Suffixes::XorRotW7 => eval_xor_rot_w::<7, T, F, N>(bits, io_ctx),
        Suffixes::XorRotW8 => eval_xor_rot_w::<8, T, F, N>(bits, io_ctx),
        Suffixes::XorRotW12 => eval_xor_rot_w::<12, T, F, N>(bits, io_ctx),
        Suffixes::XorRotW16 => eval_xor_rot_w::<16, T, F, N>(bits, io_ctx),

        // --- Rev8W ---
        Suffixes::Rev8W => eval_rev8w(bits, io_ctx),
    }
}

// ---------------------------------------------------------------------------
// Bitwise suffixes
// ---------------------------------------------------------------------------

/// AND: uninterleave → x & y (binary domain, interactive)
fn eval_and<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
    let result = rep3_ring::binary::and_many::<T::Half, _>(&xs, &ys, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|r| SuffixFuture::cast_to_field_b2a(r))
        .collect())
}

/// NotAnd: x & !y (interactive AND)
fn eval_notand<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
    let not_ys: Vec<_> = ys.iter().map(|y| !y).collect();
    let result = rep3_ring::binary::and_many::<T::Half, _>(&xs, &not_ys, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|r| SuffixFuture::cast_to_field_b2a(r))
        .collect())
}

/// XOR: x ^ y (local in binary domain, no communication)
fn eval_xor<T: Uninterleavable, F: JoltField>(
    bits: &[Rep3RingShare<T>],
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
    Ok(xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| SuffixFuture::cast_to_field_b2a(*x ^ *y))
        .collect())
}

/// OR: x | y (interactive)
fn eval_or<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
    let result = rep3_ring::binary::or_many::<T::Half, _>(&xs, &ys, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|r| SuffixFuture::cast_to_field_b2a(r))
        .collect())
}

// ---------------------------------------------------------------------------
// Value extraction suffixes
// ---------------------------------------------------------------------------

/// RightOperand: extract y from uninterleave → B2A deferred (T::Half ring)
fn eval_right_operand<T: Uninterleavable, F: JoltField>(
    bits: &[Rep3RingShare<T>],
) -> Vec<SuffixFuture<T::Half, F>>
where
    Standard: Distribution<T::Half>,
{
    let (_, ys) = uninterleave_batch(bits);
    ys.into_iter()
        .map(|y| SuffixFuture::cast_to_field_b2a(y))
        .collect()
}

/// RightOperandW: extract y, truncate to u32 (W-variant).
/// Result always fits in T::Half because T::Half::K >= suffix_len/2 and the
/// truncation only removes bits above bit 31 (which are zero when T::Half::K ≤ 32).
fn eval_right_operand_w<T: Uninterleavable, F: JoltField>(
    bits: &[Rep3RingShare<T>],
) -> Vec<SuffixFuture<T::Half, F>>
where
    Standard: Distribution<T::Half>,
{
    let (_, ys) = uninterleave_batch(bits);
    // Truncate to 32 bits (local mask). If T::Half::K ≤ 32, this is a no-op.
    let mask_val = if T::Half::K >= 32 {
        // Mask to lower 32 bits within T::Half
        let m: u128 = (1u128 << 32) - 1;
        T::Half::try_from(m).unwrap_or_else(|_| unreachable!())
    } else {
        // T::Half is smaller than 32 bits — all bits are already "lower 32"
        // Just keep as-is (the value fits since the operand has T::Half::K bits)
        return ys
            .into_iter()
            .map(|y| SuffixFuture::cast_to_field_b2a(y))
            .collect();
    };
    ys.into_iter()
        .map(|y| SuffixFuture::cast_to_field_b2a(y & RingElement(mask_val)))
        .collect()
}

/// UpperWord: bits >> XLEN, interpreted as field.
/// Result is the interleaved bits above position XLEN.
/// Eagerly converts to field when result exceeds T::Half capacity.
fn eval_upper_word<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T> + Distribution<T::Half>,
{
    if T::K <= XLEN {
        // All bits are below XLEN → result is zero
        return Ok(bits
            .iter()
            .map(|_| SuffixFuture::Ready(Rep3PrimeFieldShare::zero_share()))
            .collect());
    }
    // Result has up to (T::K - XLEN) bits. Check if it fits in T::Half.
    let result_bits = T::K - XLEN;
    if result_bits <= T::Half::K {
        let shifted: Vec<Rep3RingShare<T::Half>> =
            bits.iter().map(|b| downcast(*b >> XLEN)).collect();
        Ok(shifted
            .into_iter()
            .map(|s| SuffixFuture::cast_to_field_b2a(s))
            .collect())
    } else {
        // Rare: eagerly convert full-width shifted value to field
        let shifted: Vec<Rep3RingShare<T>> = bits.iter().map(|b| *b >> XLEN).collect();
        let fields: Vec<Rep3PrimeFieldShare<F>> =
            rep3_ring::casts::binary_ring_to_field_many(&shifted, io_ctx)?;
        Ok(fields.into_iter().map(SuffixFuture::Ready).collect())
    }
}

/// LowerWord: bits & ((1 << XLEN) - 1), interpreted as field.
/// Eagerly converts when result exceeds T::Half capacity.
fn eval_lower_word<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T> + Distribution<T::Half>,
{
    let result_bits = XLEN.min(T::K);
    if result_bits <= T::Half::K {
        let mask_val = if XLEN >= T::K {
            // All bits are "lower" — no mask needed, just downcast
            return Ok(bits
                .iter()
                .map(|b| SuffixFuture::cast_to_field_b2a(downcast(*b)))
                .collect());
        } else {
            T::try_from((1u128 << XLEN) - 1).unwrap_or_else(|_| unreachable!())
        };
        Ok(bits
            .iter()
            .map(|b| SuffixFuture::cast_to_field_b2a(downcast(*b & RingElement(mask_val))))
            .collect())
    } else {
        // Result exceeds T::Half — eagerly convert via full ring
        let masked: Vec<Rep3RingShare<T>> = if XLEN >= T::K {
            // All bits of T are "lower" — no masking needed
            bits.to_vec()
        } else {
            let mask_val = T::try_from((1u128 << XLEN) - 1).unwrap_or_else(|_| unreachable!());
            bits.iter().map(|b| *b & RingElement(mask_val)).collect()
        };
        let fields: Vec<Rep3PrimeFieldShare<F>> =
            rep3_ring::casts::binary_ring_to_field_many(&masked, io_ctx)?;
        Ok(fields.into_iter().map(SuffixFuture::Ready).collect())
    }
}

/// LowerHalfWord: bits & ((1 << (XLEN/2)) - 1), interpreted as field.
/// Eagerly converts when result exceeds T::Half capacity.
fn eval_lower_half_word<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T> + Distribution<T::Half>,
{
    let half = XLEN / 2;
    let result_bits = half.min(T::K);
    if result_bits <= T::Half::K {
        let mask_val = if half >= T::K {
            return Ok(bits
                .iter()
                .map(|b| SuffixFuture::cast_to_field_b2a(downcast(*b)))
                .collect());
        } else {
            T::try_from((1u128 << half) - 1).unwrap_or_else(|_| unreachable!())
        };
        Ok(bits
            .iter()
            .map(|b| SuffixFuture::cast_to_field_b2a(downcast(*b & RingElement(mask_val))))
            .collect())
    } else {
        // Result exceeds T::Half — eagerly convert via full ring
        let masked: Vec<Rep3RingShare<T>> = if half >= T::K {
            // All bits of T fit in the lower half — no masking needed
            bits.to_vec()
        } else {
            let mask_val = T::try_from((1u128 << half) - 1).unwrap_or_else(|_| unreachable!());
            bits.iter().map(|b| *b & RingElement(mask_val)).collect()
        };
        let fields: Vec<Rep3PrimeFieldShare<F>> =
            rep3_ring::casts::binary_ring_to_field_many(&masked, io_ctx)?;
        Ok(fields.into_iter().map(SuffixFuture::Ready).collect())
    }
}

/// Lsb: least significant bit → B2A deferred (always fits in T::Half)
fn eval_lsb<T: Uninterleavable, F: JoltField>(
    bits: &[Rep3RingShare<T>],
) -> Vec<SuffixFuture<T::Half, F>>
where
    Standard: Distribution<T::Half>,
{
    let one = T::one();
    bits.iter()
        .map(|b| {
            let masked = *b & RingElement(one);
            SuffixFuture::cast_to_field_b2a(downcast(masked))
        })
        .collect()
}

/// TwoLsb: 1 if the two LSBs are both 0, else 0.
/// Binary domain: extract 2 LSBs, check if zero via is_zero_many.
fn eval_two_lsb<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T> + Distribution<T::Half>,
{
    let mask = T::try_from(0b11u128).unwrap_or_else(|_| unreachable!());
    let lsbs: Vec<Rep3RingShare<T>> = bits.iter().map(|b| *b & RingElement(mask)).collect();
    // is_zero_many on T-sized shares (benefits from smaller T)
    let result: Vec<Rep3RingShare<Bit>> = rep3_ring::binary::is_zero_many::<T, _>(&lsbs, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

// ---------------------------------------------------------------------------
// Comparison suffixes
// ---------------------------------------------------------------------------

/// LessThan: x < y (unsigned comparison on uninterleaved binary operands)
fn eval_less_than<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
    // x < y ≡ !(x >= y); ge_many expects binary shares
    let ge_bits: Vec<Rep3RingShare<Bit>> =
        rep3_ring::arithmetic::ge_many::<T::Half, _>(&xs, &ys, io_ctx)?;
    let lt_bits: Vec<Rep3RingShare<Bit>> = ge_bits.iter().map(|b| !b).collect();
    Ok(lt_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// GreaterThan: x > y ≡ y < x ≡ !(y >= x)
fn eval_greater_than<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
    let ge_bits: Vec<Rep3RingShare<Bit>> =
        rep3_ring::arithmetic::ge_many::<T::Half, _>(&ys, &xs, io_ctx)?;
    let gt_bits: Vec<Rep3RingShare<Bit>> = ge_bits.iter().map(|b| !b).collect();
    Ok(gt_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// Eq: x == y iff (x ^ y) == 0 in binary domain
fn eval_eq<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
    let diff: Vec<Rep3RingShare<T::Half>> =
        xs.iter().zip(ys.iter()).map(|(x, y)| *x ^ *y).collect();
    let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&diff, io_ctx)?;
    Ok(eq_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// LeftIsZero: x == 0
fn eval_left_is_zero<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, _) = uninterleave_batch(bits);
    let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&xs, io_ctx)?;
    Ok(eq_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// RightIsZero: y == 0
fn eval_right_is_zero<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (_, ys) = uninterleave_batch(bits);
    let eq_bits = rep3_ring::binary::is_zero_many::<T::Half, _>(&ys, io_ctx)?;
    Ok(eq_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// DivByZero: divisor==0 AND quotient==all_ones.
/// All in binary domain.
fn eval_div_by_zero<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (divisors, quotients) = uninterleave_batch(bits);
    let quotient_bits = suffix_len / 2;
    let all_ones_val: u128 = if quotient_bits >= T::Half::K {
        // All bits of T::Half are ones
        (1u128 << T::Half::K) - 1
    } else {
        (1u128 << quotient_bits) - 1
    };
    let all_ones_mask =
        RingElement(T::Half::try_from(all_ones_val).unwrap_or_else(|_| unreachable!()));
    let q_xor: Vec<Rep3RingShare<T::Half>> = quotients
        .iter()
        .map(|q| rep3_ring::binary::xor_public(q, &all_ones_mask, party_id))
        .collect();

    let divisor_zero: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::is_zero_many::<T::Half, _>(&divisors, io_ctx)?;
    let quotient_all_ones: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::is_zero_many::<T::Half, _>(&q_xor, io_ctx)?;
    let result: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::and_many::<Bit, _>(&divisor_zero, &quotient_all_ones, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// OverflowBitsZero: upper (T::K - XLEN) bits are all zero.
/// Binary domain: shift right by XLEN, check is_zero.
fn eval_overflow_bits_zero<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T> + Distribution<T::Half>,
{
    if T::K <= XLEN {
        // No overflow bits exist — result is always 1 (all zero above XLEN)
        return Ok(bits
            .iter()
            .map(|_| {
                SuffixFuture::Pending(
                    FutureOp::BitInject(rep3_ring::binary::promote_to_trivial_share(
                        // Use party 0 since trivial shares are the same for all
                        PartyID::ID0,
                        &RingElement(Bit::one()),
                    )),
                    (),
                )
            })
            .collect());
    }
    // Shift right by XLEN, then is_zero on the upper bits
    let upper: Vec<Rep3RingShare<T>> = bits.iter().map(|b| *b >> XLEN).collect();
    let eq_bits = rep3_ring::binary::is_zero_many::<T, _>(&upper, io_ctx)?;
    Ok(eq_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

// ---------------------------------------------------------------------------
// Change divisor
// ---------------------------------------------------------------------------

/// ChangeDivisor: (y == all_ones) AND (x == 0). Binary domain.
fn eval_change_divisor<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
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

    let y_eq_all_ones: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::is_zero_many::<T::Half, _>(&y_xor, io_ctx)?;
    let x_eq_zero: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::is_zero_many::<T::Half, _>(&xs, io_ctx)?;
    let result: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::and_many::<Bit, _>(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// ChangeDivisorW: same but with W-variant (truncate to 32 bits)
fn eval_change_divisor_w<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let (xs, ys) = uninterleave_batch(bits);
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

    let y_eq_all_ones: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::is_zero_many::<u32, _>(&y_xor, io_ctx)?;
    let x_eq_zero: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::is_zero_many::<u32, _>(&xs32, io_ctx)?;
    let result: Vec<Rep3RingShare<Bit>> =
        rep3_ring::binary::and_many::<Bit, _>(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

// ---------------------------------------------------------------------------
// Pow2 suffixes
// ---------------------------------------------------------------------------

/// Pow2: 1 << shift where shift = low log2(XLEN) bits of the suffix.
/// Input is binary domain. Shift bits extracted via binary mask.
/// Uses table lookup with eq_many on binary shares (via is_zero on XOR).
fn eval_pow2<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T> + Distribution<T::Half>,
{
    let log_xlen = XLEN.log_2();
    let num_bits = log_xlen.min(suffix_len);
    let shift_mask_val = T::try_from((1u128 << num_bits) - 1).unwrap_or_else(|_| unreachable!());
    // Extract shift bits, use T for is_zero (benefits from smaller ring)
    let shifts: Vec<Rep3RingShare<T>> = bits
        .iter()
        .map(|b| *b & RingElement(shift_mask_val))
        .collect();
    eval_pow2_from_shift_bits(&shifts, num_bits, io_ctx, party_id)
}

/// Pow2W: same but shift = low 5 bits (modulo 32)
fn eval_pow2_w<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T> + Distribution<T::Half>,
{
    let num_bits = 5usize.min(suffix_len);
    let shift_mask_val = T::try_from((1u128 << num_bits) - 1).unwrap_or_else(|_| unreachable!());
    let shifts: Vec<Rep3RingShare<T>> = bits
        .iter()
        .map(|b| *b & RingElement(shift_mask_val))
        .collect();
    eval_pow2_from_shift_bits(&shifts, num_bits, io_ctx, party_id)
}

/// Compute `1 << shift` where shift is a secret binary value with at most `num_bits` bits.
/// Uses binary-domain equality: for each possible s, check (shift ^ s) == 0,
/// then accumulate: result[j] = Σ_s indicator(shift[j] == s) * F::from(1 << s).
///
/// Returns Ready futures since the accumulation produces field shares directly.
fn eval_pow2_from_shift_bits<T: IntRing2k, H: IntRing2k, F: JoltField, N: Rep3Network>(
    shift_vals: &[Rep3RingShare<T>],
    num_bits: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<H, F>>>
where
    Standard: Distribution<T>,
{
    let table_size = 1usize << num_bits;
    let n = shift_vals.len();

    // For each (shift, s) pair, compute shift ^ s (local), then batch is_zero
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

    // Accumulate: result[j] = Σ_s eq(shift[j], s) * F::from(1 << s)
    let mut result = Vec::with_capacity(n);
    for j in 0..n {
        let mut acc = Rep3PrimeFieldShare::<F>::zero_share();
        for s in 0..table_size {
            let weight = F::from_u128(1u128 << s);
            acc = acc + eq_field[j * table_size + s] * weight;
        }
        result.push(SuffixFuture::Ready(acc));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Sign extension suffixes
// ---------------------------------------------------------------------------

/// SignExtension: computes ((1 << XLEN) - (1 << (XLEN - padding_len)))
/// where padding_len = min(y.trailing_zeros(), y.len()).
///
/// Uses binary domain bit extraction + sequential AND for trailing zeros.
/// Returns Ready futures since the accumulation produces field shares directly.
fn eval_sign_extension<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
    T::Half: AsPrimitive<Bit>,
{
    let (_, ys) = uninterleave_batch(bits);
    let y_len = suffix_len / 2;
    let n = bits.len();
    let mut result = vec![Rep3PrimeFieldShare::<F>::zero_share(); n];

    // Extract individual bit shares (binary domain)
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

    // running[j] = product of (bit_i == 0) for i < p
    let mut running: Vec<Rep3RingShare<Bit>> =
        vec![rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(Bit::one())); n];

    for p in 0..y_len {
        // indicator[p] = running AND bit_p
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

        // Update running: running = running AND not_bit_p
        running = rep3_ring::binary::and_many::<Bit, _>(&running, &not_bits[p], io_ctx)?;
    }

    // Handle p == y_len case: all bits were 0
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

    Ok(result.into_iter().map(SuffixFuture::Ready).collect())
}

/// SignExtensionUpperHalf: if suffix_len >= XLEN/2, extract sign bit at position (XLEN/2 - 1)
/// from the raw interleaved suffix bits, then multiply by weight.
///
/// Vanilla does `(bits >> (half_word_size - 1)) & 1` directly on interleaved LookupBits.
/// In the interleaved format, bit 31 corresponds to x[15] (left operand bit 15).
/// We must NOT uninterleave — just extract the bit directly from the interleaved T.
fn eval_sign_extension_upper_half<
    T: Uninterleavable + AsPrimitive<Bit>,
    F: JoltField,
    N: Rep3Network,
>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    let half = XLEN / 2;
    if suffix_len < half {
        return Ok(vec![
            SuffixFuture::Ready(
                Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id,)
            );
            bits.len()
        ]);
    }

    // Extract sign bit at position (half - 1) from the raw interleaved suffix bits,
    // matching vanilla: `(bits >> sign_bit_position) & 1` where sign_bit_position = half - 1.
    let sign_bit_pos = half - 1; // = 31 for XLEN=64
    let sign_bits: Vec<Rep3RingShare<Bit>> = bits
        .iter()
        .map(|b| downcast::<T, Bit>(*b >> sign_bit_pos))
        .collect();
    let weight = F::from_u128(((1u64 << half) - 1) as u128 * (1u128 << half));
    // Bit inject then multiply by weight → we do this eagerly since weight is public
    let sign_field: Vec<Rep3PrimeFieldShare<F>> =
        rep3_ring::conversion::bit_inject_from_bits_to_field_many(&sign_bits, io_ctx)?;
    Ok(sign_field
        .into_iter()
        .map(|s| SuffixFuture::Ready(s * weight))
        .collect())
}

/// SignExtensionRightOperand: if suffix_len >= XLEN, extract sign bit at position (XLEN - 2)
/// in the y operand (uninterleaved), multiply by weight.
fn eval_sign_extension_right_operand<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
    T::Half: AsPrimitive<Bit>,
{
    if suffix_len < XLEN {
        return Ok(vec![
            SuffixFuture::Ready(
                Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id,)
            );
            bits.len()
        ]);
    }

    // In uninterleaved y operand, the sign bit is at position (XLEN/2 - 1)
    // because the interleaved position XLEN-2 maps to y-bit at (XLEN-2)/2 = XLEN/2-1
    let sign_bit_pos = XLEN / 2 - 1;
    let (_, ys) = uninterleave_batch(bits);
    let sign_bits: Vec<Rep3RingShare<Bit>> = ys
        .iter()
        .map(|y| downcast::<T::Half, Bit>(*y >> sign_bit_pos))
        .collect();
    let weight = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)));
    let sign_field: Vec<Rep3PrimeFieldShare<F>> =
        rep3_ring::conversion::bit_inject_from_bits_to_field_many(&sign_bits, io_ctx)?;
    Ok(sign_field
        .into_iter()
        .map(|s| SuffixFuture::Ready(s * weight))
        .collect())
}

// ---------------------------------------------------------------------------
// XOR-rotate suffixes (all local — no communication)
// ---------------------------------------------------------------------------

/// XorRot: uninterleave → x^y → rotate_right(result, ROTATION).
/// All ops in binary domain. Rotate by constant is local.
/// Rotation is modulo T::Half::K bits (64-bit for u128 input, 32-bit for u64, etc.)
fn eval_xor_rot<const ROTATION: u32, T: Uninterleavable, F: JoltField>(
    bits: &[Rep3RingShare<T>],
) -> Vec<SuffixFuture<T::Half, F>>
where
    Standard: Distribution<T::Half>,
{
    let k = T::Half::K;
    let rot = (ROTATION as usize) % k;
    let (xs, ys) = uninterleave_batch(bits);
    xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let xored = *x ^ *y;
            let rotated = (xored >> rot) ^ (xored << (k - rot));
            SuffixFuture::cast_to_field_b2a(rotated)
        })
        .collect()
}

/// XorRotW: same but truncate to u32 first, then rotate in 32-bit ring.
/// Uses u32 shares for the rotation to get correct 32-bit wrap-around.
/// When T::Half < u32, eagerly converts the u32 result to field.
fn eval_xor_rot_w<const ROTATION: u32, T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
    T::Half: AsPrimitive<T>,
{
    let (xs, ys) = uninterleave_batch(bits);
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

    u32_results_to_suffix_futures::<T, F, N>(&rotated_u32, io_ctx)
}

// ---------------------------------------------------------------------------
// Rev8W (local — no communication)
// ---------------------------------------------------------------------------

/// Rev8W: byte reversal of lower 32 bits.
/// Binary domain: extract bytes via mask+shift, recombine with XOR.
/// Uses u32 shares internally for correct 32-bit byte extraction.
/// When T::Half < u32, eagerly converts the u32 result to field.
fn eval_rev8w<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
    T::Half: AsPrimitive<T>,
{
    let mask_byte = RingElement(0xFFu32);
    let reversed_u32: Vec<Rep3RingShare<u32>> = bits
        .iter()
        .map(|b| {
            // Convert interleaved value to u32 for byte operations
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
            // Non-overlapping positions → XOR == OR
            (byte0 << 24) ^ (byte1 << 16) ^ (byte2 << 8) ^ byte3
        })
        .collect();

    u32_results_to_suffix_futures::<T, F, N>(&reversed_u32, io_ctx)
}

/// Helper: convert u32 results to SuffixFuture<T::Half, F>.
/// If T::Half >= 32 bits, pack into T::Half and defer B2A.
/// Otherwise, eagerly convert the u32 values to field shares.
fn u32_results_to_suffix_futures<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    u32_results: &[Rep3RingShare<u32>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    if T::Half::K >= 32 {
        // Result fits in T::Half — pack manually and defer B2A
        Ok(u32_results
            .iter()
            .map(|r| {
                let half = Rep3RingShare {
                    a: RingElement(
                        T::Half::try_from(r.a.0 as u128).unwrap_or_else(|_| unreachable!()),
                    ),
                    b: RingElement(
                        T::Half::try_from(r.b.0 as u128).unwrap_or_else(|_| unreachable!()),
                    ),
                };
                SuffixFuture::cast_to_field_b2a(half)
            })
            .collect())
    } else {
        // T::Half < u32 — eagerly convert u32 result to field
        let fields: Vec<Rep3PrimeFieldShare<F>> =
            rep3_ring::casts::binary_ring_to_field_many::<u32, _, _>(u32_results, io_ctx)?;
        Ok(fields.into_iter().map(SuffixFuture::Ready).collect())
    }
}

// ---------------------------------------------------------------------------
// Suffix classification for EdaBit budget estimation
// ---------------------------------------------------------------------------

/// Returns true if this suffix variant produces `CastToFieldB2A` futures
/// (consuming edaBits of type T::Half) rather than `BitInject` or `Ready`.
pub fn suffix_uses_b2a_edabits(suffix: &Suffixes) -> bool {
    match suffix {
        // CastToFieldB2A — consumes T::Half edaBits
        Suffixes::And
        | Suffixes::NotAnd
        | Suffixes::Xor
        | Suffixes::Or
        | Suffixes::RightOperand
        | Suffixes::RightOperandW
        | Suffixes::UpperWord
        | Suffixes::LowerWord
        | Suffixes::LowerHalfWord
        | Suffixes::Lsb
        | Suffixes::XorRot16
        | Suffixes::XorRot24
        | Suffixes::XorRot32
        | Suffixes::XorRot63
        | Suffixes::XorRotW7
        | Suffixes::XorRotW8
        | Suffixes::XorRotW12
        | Suffixes::XorRotW16
        | Suffixes::Rev8W => true,

        // Ready (constant) — no edaBits
        Suffixes::One => false,

        // BitInject (daBits, not edaBits) — boolean results
        Suffixes::TwoLsb
        | Suffixes::LessThan
        | Suffixes::GreaterThan
        | Suffixes::Eq
        | Suffixes::LeftOperandIsZero
        | Suffixes::RightOperandIsZero
        | Suffixes::DivByZero
        | Suffixes::OverflowBitsZero
        | Suffixes::ChangeDivisor
        | Suffixes::ChangeDivisorW
        | Suffixes::Pow2
        | Suffixes::Pow2W
        | Suffixes::SignExtension
        | Suffixes::SignExtensionUpperHalf
        | Suffixes::SignExtensionRightOperand => false,

        // Shift suffixes — evaluated via vanilla open, produce Ready
        Suffixes::RightShift
        | Suffixes::RightShiftHelper
        | Suffixes::RightShiftPadding
        | Suffixes::LeftShift
        | Suffixes::RightShiftW
        | Suffixes::RightShiftWHelper
        | Suffixes::LeftShiftWHelper
        | Suffixes::LeftShiftW => false,
    }
}

// ---------------------------------------------------------------------------
// Shift-by-bitmask suffixes (complex — using open for prototype)
// ---------------------------------------------------------------------------

/// Evaluate suffix by opening the secret bits (not secure, for testing).
/// TODO: Replace with proper oblivious shift implementations.
fn eval_with_vanilla_open<T: Uninterleavable, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<T>],
    suffix_len: usize,
    suffix: &Suffixes,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<T::Half, F>>>
where
    Standard: Distribution<T::Half>,
{
    use jolt_core::utils::lookup_bits::LookupBits;

    // Open binary shares: upcast to u128 for reshare, then reconstruct via XOR.
    let bs: Vec<RingElement<u128>> = bits.iter().map(|s| RingElement(s.b.0.into())).collect();
    let cs: Vec<RingElement<u128>> = io_ctx.network.reshare_many(&bs)?;
    let result: Vec<SuffixFuture<T::Half, F>> = bits
        .iter()
        .zip(cs.iter())
        .map(|(s, c)| {
            let a_u128: u128 = s.a.0.into();
            let b_u128: u128 = s.b.0.into();
            let plain = a_u128 ^ b_u128 ^ c.0;
            let eval = suffix.suffix_mle::<XLEN>(LookupBits::new(plain, suffix_len));
            SuffixFuture::Ready(Rep3PrimeFieldShare::promote_from_trivial(
                &F::from_u64(eval),
                party_id,
            ))
        })
        .collect();
    Ok(result)
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
    pool: &mut mpc_core::protocols::rep3_ring::pcg::edabits_pcg::PcgEdaBitsPool<F>,
) -> eyre::Result<OperandQSuffixEvals<F>>
where
    T: Uninterleavable,
    Standard: Distribution<T> + Distribution<T::Half>,
    T::Half: AsPrimitive<T>,
    F: JoltField,
    N: Rep3Network,
{
    use mpc_core::protocols::rep3_ring::pcg::edabits_pcg;

    // Identity: suffix_bits as field (T ring B2A via PCG edaBits)
    let identity = {
        let edas = pool.take_edabits::<T>(suffix_bits.len());
        let xs: Vec<_> = suffix_bits.iter().copied().collect();
        edabits_pcg::ring_to_field_b2a_many::<T, F, _>(&xs, edas, io_ctx)?
    };

    // Uninterleave (local) for left/right operands, then B2A via PCG edaBits (T::Half ring)
    let (xs, ys) = uninterleave_batch(suffix_bits);
    let left_operand = {
        let edas = pool.take_edabits::<T::Half>(xs.len());
        edabits_pcg::ring_to_field_b2a_many::<T::Half, F, _>(&xs, edas, io_ctx)?
    };
    let right_operand = {
        let edas = pool.take_edabits::<T::Half>(ys.len());
        edabits_pcg::ring_to_field_b2a_many::<T::Half, F, _>(&ys, edas, io_ctx)?
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
