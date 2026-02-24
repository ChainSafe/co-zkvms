//! MPC suffix evaluation for the ReadRaf sumcheck.
//!
//! Each vanilla `Suffixes` variant evaluates `suffix_mle(LookupBits) -> u64`.
//! This module provides the MPC equivalent: given a batch of secret
//! `Rep3RingShare<u128>` suffix bits (one per cycle), produce a batch of
//! `FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>` suffix futures.
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
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self as rep3_ring, Rep3RingShare};

/// Suffix evaluation future: deferred ring→field conversion.
pub type SuffixFuture<F> = FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Zero-extend a binary `Rep3RingShare<u32>` to `Rep3RingShare<u64>`.
/// Local operation (no communication).
fn zext32(s: Rep3RingShare<u32>) -> Rep3RingShare<u64> {
    Rep3RingShare {
        a: RingElement(s.a.0 as u64),
        b: RingElement(s.b.0 as u64),
    }
}

/// MPC version of `uninterleave_bits(val: u128) -> (u64, u64)`.
///
/// Input must be in **binary** (XOR) domain.
///
/// Extracts bits one-by-one: x[i] = bit at position (2i+1), y[i] = bit at position (2i).
/// Each extraction uses `>> constant` then `& 1` (both local on binary shares),
/// then `<< i` to place the bit at the correct output position. Since different `i`
/// target different bit positions, the XOR-sum (`^`) to combine them is correct
/// even on share components.
///
/// Zero MPC communication.
fn uninterleave_bin(s: Rep3RingShare<u128>) -> (Rep3RingShare<u64>, Rep3RingShare<u64>) {
    let one = RingElement(1u128);
    let mut x = Rep3RingShare {
        a: RingElement(0u64),
        b: RingElement(0u64),
    };
    let mut y = Rep3RingShare {
        a: RingElement(0u64),
        b: RingElement(0u64),
    };
    for i in 0..64 {
        // x[i] = (s >> (2*i + 1)) & 1, placed at bit position i
        let x_bit = (s >> (2 * i + 1)) & one; // u128 share with only bit 0
        x.a.0 |= (x_bit.a.0 as u64) << i;
        x.b.0 |= (x_bit.b.0 as u64) << i;

        // y[i] = (s >> (2*i)) & 1, placed at bit position i
        let y_bit = (s >> (2 * i)) & one; // u128 share with only bit 0
        y.a.0 |= (y_bit.a.0 as u64) << i;
        y.b.0 |= (y_bit.b.0 as u64) << i;
    }
    (x, y)
}

/// Batch uninterleave: local, no communication.
fn uninterleave_batch(
    bits: &[Rep3RingShare<u128>],
) -> (Vec<Rep3RingShare<u64>>, Vec<Rep3RingShare<u64>>) {
    bits.iter().map(|b| uninterleave_bin(*b)).unzip()
}

// ---------------------------------------------------------------------------
// Top-level dispatch
// ---------------------------------------------------------------------------

/// Evaluate a suffix MLE on a batch of secret suffix bits, producing
/// per-cycle suffix futures (deferred field conversion).
///
/// `bits[j]` is a **binary-domain** `Rep3RingShare<u128>` representing the low
/// `suffix_len` bits of cycle j's lookup index.
///
/// Returns `Vec<SuffixFuture<F>>` — callers must call `fulfill_batched` to
/// convert to `Vec<Rep3PrimeFieldShare<F>>`.
// #[tracing::instrument(skip_all, name = "evaluate_suffix_mle_batched")]
pub fn evaluate_suffix_mle_batched<F: JoltField, N: Rep3Network>(
    suffix: &Suffixes,
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
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
        Suffixes::UpperWord => Ok(eval_upper_word(bits)),
        Suffixes::LowerWord => Ok(eval_lower_word(bits)),
        Suffixes::LowerHalfWord => Ok(eval_lower_half_word(bits)),
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
        Suffixes::XorRot16 => Ok(eval_xor_rot::<16, F>(bits)),
        Suffixes::XorRot24 => Ok(eval_xor_rot::<24, F>(bits)),
        Suffixes::XorRot32 => Ok(eval_xor_rot::<32, F>(bits)),
        Suffixes::XorRot63 => Ok(eval_xor_rot::<63, F>(bits)),
        Suffixes::XorRotW7 => Ok(eval_xor_rot_w::<7, F>(bits)),
        Suffixes::XorRotW8 => Ok(eval_xor_rot_w::<8, F>(bits)),
        Suffixes::XorRotW12 => Ok(eval_xor_rot_w::<12, F>(bits)),
        Suffixes::XorRotW16 => Ok(eval_xor_rot_w::<16, F>(bits)),

        // --- Rev8W ---
        Suffixes::Rev8W => Ok(eval_rev8w(bits)),
    }
}

// ---------------------------------------------------------------------------
// Bitwise suffixes
// ---------------------------------------------------------------------------

/// AND: uninterleave → x & y (binary domain, interactive)
fn eval_and<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    let result = rep3_ring::binary::and_many(&xs, &ys, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|r| SuffixFuture::cast_to_field_b2a(r))
        .collect())
}

/// NotAnd: x & !y (interactive AND)
fn eval_notand<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    let not_ys: Vec<_> = ys.iter().map(|y| !y).collect();
    let result = rep3_ring::binary::and_many(&xs, &not_ys, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|r| SuffixFuture::cast_to_field_b2a(r))
        .collect())
}

/// XOR: x ^ y (local in binary domain, no communication)
fn eval_xor<F: JoltField>(bits: &[Rep3RingShare<u128>]) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    Ok(xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| SuffixFuture::cast_to_field_b2a(*x ^ *y))
        .collect())
}

/// OR: x | y (interactive)
fn eval_or<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    let result = rep3_ring::binary::or_many(&xs, &ys, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|r| SuffixFuture::cast_to_field_b2a(r))
        .collect())
}

// ---------------------------------------------------------------------------
// Value extraction suffixes (all local — no communication)
// ---------------------------------------------------------------------------

/// RightOperand: extract y from uninterleave → B2A deferred
fn eval_right_operand<F: JoltField>(bits: &[Rep3RingShare<u128>]) -> Vec<SuffixFuture<F>> {
    let (_, ys) = uninterleave_batch(bits);
    ys.into_iter()
        .map(|y| SuffixFuture::cast_to_field_b2a(y))
        .collect()
}

/// RightOperandW: extract y, truncate to u32 → zero-extend to u64 → B2A deferred
fn eval_right_operand_w<F: JoltField>(bits: &[Rep3RingShare<u128>]) -> Vec<SuffixFuture<F>> {
    let (_, ys) = uninterleave_batch(bits);
    ys.into_iter()
        .map(|y| SuffixFuture::cast_to_field_b2a(zext32(downcast::<u64, u32>(y))))
        .collect()
}

/// UpperWord: bits >> XLEN (binary shift), downcast to u64 → B2A deferred
fn eval_upper_word<F: JoltField>(bits: &[Rep3RingShare<u128>]) -> Vec<SuffixFuture<F>> {
    bits.iter()
        .map(|b| SuffixFuture::cast_to_field_b2a(downcast(*b >> XLEN)))
        .collect()
}

/// LowerWord: bits & ((1 << XLEN) - 1) → B2A deferred
fn eval_lower_word<F: JoltField>(bits: &[Rep3RingShare<u128>]) -> Vec<SuffixFuture<F>> {
    let mask = RingElement((1u128 << XLEN) - 1);
    bits.iter()
        .map(|b| SuffixFuture::cast_to_field_b2a(downcast(*b & mask)))
        .collect()
}

/// LowerHalfWord: bits & ((1 << (XLEN/2)) - 1) → B2A deferred
fn eval_lower_half_word<F: JoltField>(bits: &[Rep3RingShare<u128>]) -> Vec<SuffixFuture<F>> {
    let half = XLEN / 2;
    let mask = if half >= 128 {
        RingElement(u128::MAX)
    } else {
        RingElement((1u128 << half) - 1)
    };
    bits.iter()
        .map(|b| SuffixFuture::cast_to_field_b2a(downcast(*b & mask)))
        .collect()
}

/// Lsb: least significant bit → B2A deferred
fn eval_lsb<F: JoltField>(bits: &[Rep3RingShare<u128>]) -> Vec<SuffixFuture<F>> {
    bits.iter()
        .map(|b| {
            let bit: Rep3RingShare<u64> = Rep3RingShare {
                a: RingElement((b.a.0 & 1) as u64),
                b: RingElement((b.b.0 & 1) as u64),
            };
            SuffixFuture::cast_to_field_b2a(bit)
        })
        .collect()
}

/// TwoLsb: 1 if the two LSBs are both 0, else 0.
/// Binary domain: extract 2 LSBs, check if zero via is_zero_many.
fn eval_two_lsb<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let mask = RingElement(0b11u128);
    let lsbs: Vec<Rep3RingShare<u128>> = bits.iter().map(|b| *b & mask).collect();
    // In binary domain, use is_zero_many (works on binary shares)
    let lsbs_downcast: Vec<Rep3RingShare<u64>> = lsbs.iter().map(|v| downcast(*v)).collect();
    let result = rep3_ring::binary::is_zero_many(&lsbs_downcast, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

// ---------------------------------------------------------------------------
// Comparison suffixes
// ---------------------------------------------------------------------------

/// LessThan: x < y (unsigned comparison on uninterleaved binary operands)
fn eval_less_than<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    // x < y ≡ !(x >= y); ge_many expects binary shares
    let ge_bits: Vec<Rep3RingShare<Bit>> = rep3_ring::arithmetic::ge_many(&xs, &ys, io_ctx)?;
    let lt_bits: Vec<Rep3RingShare<Bit>> = ge_bits.iter().map(|b| !b).collect();
    Ok(lt_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// GreaterThan: x > y ≡ y < x ≡ !(y >= x)
fn eval_greater_than<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    let ge_bits: Vec<Rep3RingShare<Bit>> = rep3_ring::arithmetic::ge_many(&ys, &xs, io_ctx)?;
    let gt_bits: Vec<Rep3RingShare<Bit>> = ge_bits.iter().map(|b| !b).collect();
    Ok(gt_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// Eq: x == y iff (x ^ y) == 0 in binary domain
fn eval_eq<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    let diff: Vec<Rep3RingShare<u64>> = xs.iter().zip(ys.iter()).map(|(x, y)| *x ^ *y).collect();
    let eq_bits = rep3_ring::binary::is_zero_many(&diff, io_ctx)?;
    Ok(eq_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// LeftIsZero: x == 0
fn eval_left_is_zero<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, _) = uninterleave_batch(bits);
    let eq_bits = rep3_ring::binary::is_zero_many(&xs, io_ctx)?;
    Ok(eq_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// RightIsZero: y == 0
fn eval_right_is_zero<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (_, ys) = uninterleave_batch(bits);
    let eq_bits = rep3_ring::binary::is_zero_many(&ys, io_ctx)?;
    Ok(eq_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// DivByZero: divisor==0 AND quotient==all_ones.
/// All in binary domain.
fn eval_div_by_zero<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (divisors, quotients) = uninterleave_batch(bits);
    let quotient_bits = suffix_len / 2;
    let all_ones_mask = RingElement(if quotient_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << quotient_bits) - 1
    });
    let q_xor: Vec<Rep3RingShare<u64>> = quotients
        .iter()
        .map(|q| rep3_ring::binary::xor_public(q, &all_ones_mask, party_id))
        .collect();

    let divisor_zero = rep3_ring::binary::is_zero_many(&divisors, io_ctx)?;
    let quotient_all_ones = rep3_ring::binary::is_zero_many(&q_xor, io_ctx)?;
    let result = rep3_ring::binary::and_many(&divisor_zero, &quotient_all_ones, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// OverflowBitsZero: upper (128 - XLEN) bits are all zero.
/// Binary domain: shift right by XLEN, check is_zero.
fn eval_overflow_bits_zero<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let upper: Vec<Rep3RingShare<u64>> = bits.iter().map(|b| downcast(*b >> XLEN)).collect();
    let eq_bits = rep3_ring::binary::is_zero_many(&upper, io_ctx)?;
    Ok(eq_bits
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

// ---------------------------------------------------------------------------
// Change divisor
// ---------------------------------------------------------------------------

/// ChangeDivisor: (y == all_ones) AND (x == 0). Binary domain.
fn eval_change_divisor<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    let y_len = suffix_len / 2;
    let all_ones_mask = RingElement(if y_len >= 64 {
        u64::MAX
    } else {
        (1u64 << y_len) - 1
    });
    let y_xor: Vec<Rep3RingShare<u64>> = ys
        .iter()
        .map(|y| rep3_ring::binary::xor_public(y, &all_ones_mask, party_id))
        .collect();

    let y_eq_all_ones = rep3_ring::binary::is_zero_many(&y_xor, io_ctx)?;
    let x_eq_zero = rep3_ring::binary::is_zero_many(&xs, io_ctx)?;
    let result = rep3_ring::binary::and_many(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
    Ok(result
        .into_iter()
        .map(|b| SuffixFuture::Pending(FutureOp::BitInject(b), ()))
        .collect())
}

/// ChangeDivisorW: same but with W-variant (truncate to 32 bits)
fn eval_change_divisor_w<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (xs, ys) = uninterleave_batch(bits);
    let xs32: Vec<Rep3RingShare<u32>> = xs.iter().map(|x| downcast::<u64, u32>(*x)).collect();
    let ys32: Vec<Rep3RingShare<u32>> = ys.iter().map(|y| downcast::<u64, u32>(*y)).collect();

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

    let y_eq_all_ones = rep3_ring::binary::is_zero_many(&y_xor, io_ctx)?;
    let x_eq_zero = rep3_ring::binary::is_zero_many(&xs32, io_ctx)?;
    let result = rep3_ring::binary::and_many(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
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
fn eval_pow2<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let log_xlen = XLEN.log_2();
    let num_bits = log_xlen.min(suffix_len);
    let shift_mask = RingElement((1u128 << num_bits) - 1);
    let shifts: Vec<Rep3RingShare<u64>> = bits.iter().map(|b| downcast(*b & shift_mask)).collect();
    eval_pow2_from_shift_bits(&shifts, num_bits, io_ctx, party_id)
}

/// Pow2W: same but shift = low 5 bits (modulo 32)
fn eval_pow2_w<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let num_bits = 5usize.min(suffix_len);
    let shift_mask = RingElement((1u128 << num_bits) - 1);
    let shifts: Vec<Rep3RingShare<u64>> = bits.iter().map(|b| downcast(*b & shift_mask)).collect();
    eval_pow2_from_shift_bits(&shifts, num_bits, io_ctx, party_id)
}

/// Compute `1 << shift` where shift is a secret binary value with at most `num_bits` bits.
/// Uses binary-domain equality: for each possible s, check (shift ^ s) == 0,
/// then accumulate: result[j] = Σ_s indicator(shift[j] == s) * F::from(1 << s).
///
/// Returns Ready futures since the accumulation produces field shares directly.
fn eval_pow2_from_shift_bits<F: JoltField, N: Rep3Network>(
    shift_vals: &[Rep3RingShare<u64>],
    num_bits: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let table_size = 1usize << num_bits;
    let n = shift_vals.len();

    // For each (shift, s) pair, compute shift ^ s (local), then batch is_zero
    let mut xored = Vec::with_capacity(n * table_size);
    for shift in shift_vals {
        for s in 0..table_size {
            let target = RingElement(s as u64);
            xored.push(rep3_ring::binary::xor_public(shift, &target, party_id));
        }
    }

    let eq_bits: Vec<Rep3RingShare<Bit>> = rep3_ring::binary::is_zero_many(&xored, io_ctx)?;
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
fn eval_sign_extension<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    let (_, ys) = uninterleave_batch(bits);
    let y_len = suffix_len / 2;
    let n = bits.len();
    let mut result = vec![Rep3PrimeFieldShare::<F>::zero_share(); n];

    // Extract individual bit shares (binary domain)
    let mut bit_shares: Vec<Vec<Rep3RingShare<Bit>>> = Vec::with_capacity(y_len);
    for i in 0..y_len {
        let bits_i: Vec<Rep3RingShare<Bit>> =
            ys.iter().map(|y| downcast::<u64, Bit>(*y >> i)).collect();
        bit_shares.push(bits_i);
    }

    let not_bits: Vec<Vec<Rep3RingShare<Bit>>> = bit_shares
        .iter()
        .map(|bs| bs.iter().map(|b| !b).collect())
        .collect();

    // running[j] = product of (bit_i == 0) for i < p
    let mut running =
        vec![rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(Bit::one())); n];

    for p in 0..y_len {
        // indicator[p] = running AND bit_p
        let indicator = rep3_ring::binary::and_many(&running, &bit_shares[p], io_ctx)?;

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
        running = rep3_ring::binary::and_many(&running, &not_bits[p], io_ctx)?;
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
/// We must NOT uninterleave — just extract the bit directly from the interleaved u128.
fn eval_sign_extension_upper_half<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
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
        .map(|b| downcast::<u128, Bit>(*b >> sign_bit_pos))
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
fn eval_sign_extension_right_operand<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
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
        .map(|y| downcast::<u64, Bit>(*y >> sign_bit_pos))
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
fn eval_xor_rot<const ROTATION: u32, F: JoltField>(
    bits: &[Rep3RingShare<u128>],
) -> Vec<SuffixFuture<F>> {
    let (xs, ys) = uninterleave_batch(bits);
    xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let xored = *x ^ *y;
            let rotated = (xored >> (ROTATION as usize)) ^ (xored << (64 - ROTATION as usize));
            SuffixFuture::cast_to_field_b2a(rotated)
        })
        .collect()
}

/// XorRotW: same but truncate to u32 first, then zero-extend back to u64
fn eval_xor_rot_w<const ROTATION: u32, F: JoltField>(
    bits: &[Rep3RingShare<u128>],
) -> Vec<SuffixFuture<F>> {
    let (xs, ys) = uninterleave_batch(bits);
    xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let x32 = downcast::<u64, u32>(*x);
            let y32 = downcast::<u64, u32>(*y);
            let xored = x32 ^ y32;
            let rotated = (xored >> (ROTATION as usize)) ^ (xored << (32 - ROTATION as usize));
            SuffixFuture::cast_to_field_b2a(zext32(rotated))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Rev8W (local — no communication)
// ---------------------------------------------------------------------------

/// Rev8W: byte reversal of lower 32 bits.
/// Binary domain: extract bytes via mask+shift, recombine with XOR.
/// Input is already binary — no A2B needed.
fn eval_rev8w<F: JoltField>(bits: &[Rep3RingShare<u128>]) -> Vec<SuffixFuture<F>> {
    let mask_byte = RingElement(0xFFu64);
    bits.iter()
        .map(|b| {
            let v: Rep3RingShare<u64> = downcast::<u128, u64>(*b);
            let byte0 = v & mask_byte;
            let byte1 = (v >> 8) & mask_byte;
            let byte2 = (v >> 16) & mask_byte;
            let byte3 = (v >> 24) & mask_byte;
            // Non-overlapping positions → XOR == OR
            let reversed = (byte0 << 24) ^ (byte1 << 16) ^ (byte2 << 8) ^ byte3;
            SuffixFuture::cast_to_field_b2a(reversed)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Shift-by-bitmask suffixes (complex — using open for prototype)
// ---------------------------------------------------------------------------

/// Evaluate suffix by opening the secret bits (not secure, for testing).
/// TODO: Replace with proper oblivious shift implementations.
fn eval_with_vanilla_open<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    suffix: &Suffixes,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<SuffixFuture<F>>> {
    use jolt_core::utils::lookup_bits::LookupBits;

    // Open binary shares: reshare b-component, then reconstruct via XOR.
    let bs: Vec<RingElement<u128>> = bits.iter().map(|s| s.b).collect();
    let cs: Vec<RingElement<u128>> = io_ctx.network.reshare_many(&bs)?;
    let result: Vec<SuffixFuture<F>> = bits
        .iter()
        .zip(cs.iter())
        .map(|(s, c)| {
            let plain = (s.a ^ s.b ^ *c).0;
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
/// For suffix_len == 0 (final phase), all values are zero.
/// For suffix_len > 0, computes:
///   - left_operand[j] = u64::from(uninterleave(suffix_bits_j).0) as field
///   - right_operand[j] = u64::from(uninterleave(suffix_bits_j).1) as field
///   - identity[j] = suffix_bits_j as field
///
/// Input `lookup_indices` are in **binary (XOR) domain**.
#[tracing::instrument(skip_all, name = "compute_operand_q_suffix_evals", fields(phase))]
pub fn compute_operand_q_suffix_evals<F: JoltField, N: Rep3Network>(
    phase: usize,
    lookup_indices: &[Rep3RingShare<u128>],
    num_cycles: usize,
    io_ctx: &mut IoContext<N>,
    _party_id: PartyID,
) -> eyre::Result<OperandQSuffixEvals<F>> {
    const PHASES: usize = 8;
    const LOG_M: usize = 16;

    let suffix_len = (PHASES - 1 - phase) * LOG_M;

    if suffix_len == 0 {
        return Ok(OperandQSuffixEvals {
            left_operand: vec![Rep3PrimeFieldShare::zero_share(); num_cycles],
            right_operand: vec![Rep3PrimeFieldShare::zero_share(); num_cycles],
            identity: vec![Rep3PrimeFieldShare::zero_share(); num_cycles],
        });
    }

    // Extract suffix bits (binary domain)
    let suffix_mask = RingElement((1u128 << suffix_len) - 1);
    let suffix_bits: Vec<Rep3RingShare<u128>> = lookup_indices
        .iter()
        .map(|idx| *idx & suffix_mask)
        .collect();

    // Identity: suffix_bits as field (binary domain → field via B2A)
    // NOTE: suffix_len can be up to 112 bits (phase 0), so we must NOT downcast to u64.
    // binary_ring_to_field_many supports u128 directly.
    let identity = rep3_ring::casts::binary_ring_to_field_many(&suffix_bits, io_ctx)?;

    // Uninterleave (local) for left/right operands, then B2A
    let (xs, ys) = uninterleave_batch(&suffix_bits);
    let left_operand = rep3_ring::casts::binary_ring_to_field_many(&xs, io_ctx)?;
    let right_operand = rep3_ring::casts::binary_ring_to_field_many(&ys, io_ctx)?;

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
            let share0 = Rep3RingShare { a: RingElement(a), b: RingElement(b) };
            let share1 = Rep3RingShare { a: RingElement(b), b: RingElement(c) };
            let share2 = Rep3RingShare { a: RingElement(c), b: RingElement(a) };

            let (x0, y0) = uninterleave_bin(share0);
            let (x1, y1) = uninterleave_bin(share1);
            let (x2, y2) = uninterleave_bin(share2);

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
