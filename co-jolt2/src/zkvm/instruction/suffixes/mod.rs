//! MPC suffix evaluation for the ReadRaf sumcheck.
//!
//! Each vanilla `Suffixes` variant evaluates `suffix_mle(LookupBits) -> u64`.
//! This module provides the MPC equivalent: given a batch of secret
//! `Rep3RingShare<u128>` suffix bits (one per cycle), produce a batch of
//! `Rep3PrimeFieldShare<F>` suffix values.
//!
//! Suffix bits are extracted from the full 128-bit lookup index per phase:
//!   `suffix_bits = lookup_index & ((1 << suffix_len) - 1)`
//! This is a local operation (public AND on arithmetic shares).

use crate::field::JoltField;
use jolt2_common::constants::XLEN;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::lookup_table::suffixes::Suffixes;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self as rep3_ring, Rep3RingShare};
use mpc_core::protocols::rep3_ring::casts::downcast;

// ---------------------------------------------------------------------------
// uninterleave on binary Rep3RingShare<u128>
// ---------------------------------------------------------------------------

/// MPC version of `uninterleave_bits(val: u128) -> (u64, u64)`.
///
/// Input must be in **binary** (XOR) domain.
/// Uses `&` (AND with public mask), `>>` (right shift by constant), and `^` (XOR)
/// which are all local on binary rep3 shares. The `|` from vanilla is replaced
/// with `^` since bit positions are non-overlapping at each compaction step.
///
/// Zero MPC communication.
fn uninterleave_bin(s: Rep3RingShare<u128>) -> (Rep3RingShare<u64>, Rep3RingShare<u64>) {
    let mask_1 = RingElement(0x5555_5555_5555_5555_5555_5555_5555_5555u128);
    let mask_2 = RingElement(0x3333_3333_3333_3333_3333_3333_3333_3333u128);
    let mask_4 = RingElement(0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128);
    let mask_8 = RingElement(0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128);
    let mask_16 = RingElement(0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128);
    let mask_32 = RingElement(0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128);
    let mask_64 = RingElement(0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFFu128);

    // Extract odd bits (x) and even bits (y)
    let mut x_bits = (s >> 1) & mask_1;
    let mut y_bits = s & mask_1;

    // Compact using XOR instead of OR (bits are non-overlapping)
    x_bits = (x_bits ^ (x_bits >> 1)) & mask_2;
    x_bits = (x_bits ^ (x_bits >> 2)) & mask_4;
    x_bits = (x_bits ^ (x_bits >> 4)) & mask_8;
    x_bits = (x_bits ^ (x_bits >> 8)) & mask_16;
    x_bits = (x_bits ^ (x_bits >> 16)) & mask_32;
    x_bits = (x_bits ^ (x_bits >> 32)) & mask_64;

    y_bits = (y_bits ^ (y_bits >> 1)) & mask_2;
    y_bits = (y_bits ^ (y_bits >> 2)) & mask_4;
    y_bits = (y_bits ^ (y_bits >> 4)) & mask_8;
    y_bits = (y_bits ^ (y_bits >> 8)) & mask_16;
    y_bits = (y_bits ^ (y_bits >> 16)) & mask_32;
    y_bits = (y_bits ^ (y_bits >> 32)) & mask_64;

    (downcast(x_bits), downcast(y_bits))
}

/// Convert arithmetic suffix bits to binary and uninterleave.
/// Returns binary-domain (x, y) pairs.
fn a2b_and_uninterleave<N: Rep3Network>(
    bits_arith: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<(Vec<Rep3RingShare<u64>>, Vec<Rep3RingShare<u64>>)> {
    let bits_bin = rep3_ring::conversion::a2b_many(bits_arith, io_ctx)?;
    let (xs, ys): (Vec<_>, Vec<_>) = bits_bin.iter().map(|b| uninterleave_bin(*b)).unzip();
    Ok((xs, ys))
}

// ---------------------------------------------------------------------------
// Top-level dispatch
// ---------------------------------------------------------------------------

/// Evaluate a suffix MLE on a batch of secret suffix bits, producing
/// per-cycle field shares.
///
/// `suffix_bits_arith[j]` is the arithmetic-domain `Rep3RingShare<u128>`
/// representing the low `suffix_len` bits of cycle j's lookup index.
///
/// Returns `Vec<Rep3PrimeFieldShare<F>>` of length `suffix_bits_arith.len()`.
pub fn evaluate_suffix_mle_batched<F: JoltField, N: Rep3Network>(
    suffix: &Suffixes,
    suffix_bits_arith: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let n = suffix_bits_arith.len();

    // suffix_len == 0 is handled by caller (constant value).
    debug_assert!(suffix_len > 0);

    match suffix {
        Suffixes::One => Ok(vec![
            Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id);
            n
        ]),

        // --- Simple bitwise (uninterleave + local bitwise op) ---
        Suffixes::And => eval_and(suffix_bits_arith, io_ctx),
        Suffixes::NotAnd => eval_notand(suffix_bits_arith, io_ctx),
        Suffixes::Xor => eval_xor(suffix_bits_arith, io_ctx),
        Suffixes::Or => eval_or(suffix_bits_arith, io_ctx),

        // --- Value extraction ---
        Suffixes::RightOperand => eval_right_operand(suffix_bits_arith, io_ctx),
        Suffixes::RightOperandW => eval_right_operand_w(suffix_bits_arith, io_ctx),
        Suffixes::UpperWord => eval_upper_word(suffix_bits_arith, io_ctx),
        Suffixes::LowerWord => eval_lower_word(suffix_bits_arith, io_ctx),
        Suffixes::LowerHalfWord => eval_lower_half_word(suffix_bits_arith, io_ctx),
        Suffixes::Lsb => eval_lsb(suffix_bits_arith, io_ctx),
        Suffixes::TwoLsb => eval_two_lsb(suffix_bits_arith, io_ctx),

        // --- Comparisons ---
        Suffixes::LessThan => eval_less_than(suffix_bits_arith, io_ctx),
        Suffixes::GreaterThan => eval_greater_than(suffix_bits_arith, io_ctx),
        Suffixes::Eq => eval_eq(suffix_bits_arith, io_ctx),
        Suffixes::LeftOperandIsZero => eval_left_is_zero(suffix_bits_arith, io_ctx),
        Suffixes::RightOperandIsZero => eval_right_is_zero(suffix_bits_arith, io_ctx),
        Suffixes::DivByZero => eval_div_by_zero(suffix_bits_arith, suffix_len, io_ctx, party_id),
        Suffixes::OverflowBitsZero => eval_overflow_bits_zero(suffix_bits_arith, io_ctx),

        // --- Change divisor ---
        Suffixes::ChangeDivisor => eval_change_divisor(suffix_bits_arith, suffix_len, io_ctx, party_id),
        Suffixes::ChangeDivisorW => eval_change_divisor_w(suffix_bits_arith, suffix_len, io_ctx, party_id),

        // --- Pow2 ---
        Suffixes::Pow2 => eval_pow2(suffix_bits_arith, suffix_len, io_ctx, party_id),
        Suffixes::Pow2W => eval_pow2_w(suffix_bits_arith, suffix_len, io_ctx, party_id),

        // --- Sign extension ---
        Suffixes::SignExtension => eval_sign_extension(suffix_bits_arith, suffix_len, io_ctx, party_id),
        Suffixes::SignExtensionUpperHalf => eval_sign_extension_upper_half(suffix_bits_arith, suffix_len, io_ctx, party_id),
        Suffixes::SignExtensionRightOperand => eval_sign_extension_right_operand(suffix_bits_arith, suffix_len, io_ctx, party_id),

        // --- Right shift / left shift (bitmask-based) ---
        Suffixes::RightShift => eval_with_vanilla_open(suffix_bits_arith, suffix_len, suffix, io_ctx, party_id),
        Suffixes::RightShiftHelper => eval_with_vanilla_open(suffix_bits_arith, suffix_len, suffix, io_ctx, party_id),
        Suffixes::RightShiftPadding => eval_with_vanilla_open(suffix_bits_arith, suffix_len, suffix, io_ctx, party_id),
        Suffixes::LeftShift => eval_with_vanilla_open(suffix_bits_arith, suffix_len, suffix, io_ctx, party_id),
        Suffixes::RightShiftW => eval_with_vanilla_open(suffix_bits_arith, suffix_len, suffix, io_ctx, party_id),
        Suffixes::RightShiftWHelper => eval_with_vanilla_open(suffix_bits_arith, suffix_len, suffix, io_ctx, party_id),
        Suffixes::LeftShiftWHelper => eval_with_vanilla_open(suffix_bits_arith, suffix_len, suffix, io_ctx, party_id),
        Suffixes::LeftShiftW => eval_with_vanilla_open(suffix_bits_arith, suffix_len, suffix, io_ctx, party_id),

        // --- XOR-rotate ---
        Suffixes::XorRot16 => eval_xor_rot::<16, F, N>(suffix_bits_arith, io_ctx),
        Suffixes::XorRot24 => eval_xor_rot::<24, F, N>(suffix_bits_arith, io_ctx),
        Suffixes::XorRot32 => eval_xor_rot::<32, F, N>(suffix_bits_arith, io_ctx),
        Suffixes::XorRot63 => eval_xor_rot::<63, F, N>(suffix_bits_arith, io_ctx),
        Suffixes::XorRotW7 => eval_xor_rot_w::<7, F, N>(suffix_bits_arith, io_ctx),
        Suffixes::XorRotW8 => eval_xor_rot_w::<8, F, N>(suffix_bits_arith, io_ctx),
        Suffixes::XorRotW12 => eval_xor_rot_w::<12, F, N>(suffix_bits_arith, io_ctx),
        Suffixes::XorRotW16 => eval_xor_rot_w::<16, F, N>(suffix_bits_arith, io_ctx),

        // --- Rev8W ---
        Suffixes::Rev8W => eval_rev8w(suffix_bits_arith, io_ctx),
    }
}

// ---------------------------------------------------------------------------
// Bitwise suffixes
// ---------------------------------------------------------------------------

/// AND: uninterleave → x & y (binary domain)
fn eval_and<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let result = rep3_ring::binary::and_many(&xs_bin, &ys_bin, io_ctx)?;
    let field = rep3_ring::casts::binary_ring_to_field_many(&result, io_ctx)?;
    Ok(field)
}

/// NotAnd: x & !y
fn eval_notand<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let not_ys: Vec<_> = ys_bin.iter().map(|y| !y).collect();
    let result = rep3_ring::binary::and_many(&xs_bin, &not_ys, io_ctx)?;
    let field = rep3_ring::casts::binary_ring_to_field_many(&result, io_ctx)?;
    Ok(field)
}

/// XOR: x ^ y (local in binary domain)
fn eval_xor<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let result: Vec<_> = xs_bin.iter().zip(ys_bin.iter()).map(|(x, y)| *x ^ *y).collect();
    let field = rep3_ring::casts::binary_ring_to_field_many(&result, io_ctx)?;
    Ok(field)
}

/// OR: x | y
fn eval_or<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let result = rep3_ring::binary::or_many(&xs_bin, &ys_bin, io_ctx)?;
    let field = rep3_ring::casts::binary_ring_to_field_many(&result, io_ctx)?;
    Ok(field)
}

// ---------------------------------------------------------------------------
// Value extraction suffixes
// ---------------------------------------------------------------------------

/// RightOperand: extract y from uninterleave, cast to field.
/// Needs A2B for uninterleave, then B2A for field conversion.
fn eval_right_operand<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (_, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let field = rep3_ring::casts::binary_ring_to_field_many(&ys_bin, io_ctx)?;
    Ok(field)
}

/// RightOperandW: extract y, truncate to u32, cast to field
fn eval_right_operand_w<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (_, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let ys32: Vec<Rep3RingShare<u32>> = ys_bin.iter().map(|y| downcast::<u64, u32>(*y)).collect();
    let field = rep3_ring::casts::binary_ring_to_field_many(&ys32, io_ctx)?;
    Ok(field)
}

/// UpperWord: u128 >> XLEN, cast to field as u64.
/// This is a simple mask+shift on arithmetic shares — no uninterleave needed.
fn eval_upper_word<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    // shift by constant on arithmetic shares: both components get shifted
    // This gives us the upper bits as an arithmetic share
    let vals: Vec<Rep3RingShare<u64>> = bits.iter().map(|b| downcast(*b >> XLEN)).collect();
    let field = rep3_ring::casts::ring_to_field_many_selector(&vals, io_ctx)?;
    Ok(field)
}

/// LowerWord: u128 % (1 << XLEN), cast to field as u64.
/// Public AND on arithmetic shares is correct.
fn eval_lower_word<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let mask = RingElement((1u128 << XLEN) - 1);
    let vals: Vec<Rep3RingShare<u64>> = bits.iter().map(|b| downcast(*b & mask)).collect();
    let field = rep3_ring::casts::ring_to_field_many_selector(&vals, io_ctx)?;
    Ok(field)
}

/// LowerHalfWord: u128 % (1 << (XLEN/2))
fn eval_lower_half_word<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let half = XLEN / 2;
    let mask = if half >= 128 {
        RingElement(u128::MAX)
    } else {
        RingElement((1u128 << half) - 1)
    };
    let vals: Vec<Rep3RingShare<u64>> = bits.iter().map(|b| downcast(*b & mask)).collect();
    let field = rep3_ring::casts::ring_to_field_many_selector(&vals, io_ctx)?;
    Ok(field)
}

/// Lsb: least significant bit.
/// `& 1` on arithmetic shares is correct (public AND).
fn eval_lsb<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let mask = RingElement(1u128);
    let vals: Vec<Rep3RingShare<u128>> = bits.iter().map(|b| *b & mask).collect();
    let field = rep3_ring::casts::ring_to_field_many_selector(&vals, io_ctx)?;
    Ok(field)
}

/// TwoLsb: 1 if the two LSBs are 0, else 0.
fn eval_two_lsb<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let mask = RingElement(0b11u128);
    let lsbs: Vec<Rep3RingShare<u128>> = bits.iter().map(|b| *b & mask).collect();
    let zeros = vec![Rep3RingShare::default(); lsbs.len()];
    let result: Vec<Rep3RingShare<Bit>> = rep3_ring::arithmetic::eq_many(&lsbs, &zeros, io_ctx)?;
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&result, io_ctx)?;
    Ok(field)
}

// ---------------------------------------------------------------------------
// Comparison suffixes
// ---------------------------------------------------------------------------

/// LessThan: x < y (unsigned comparison on uninterleaved operands)
fn eval_less_than<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    // A2B + uninterleave gives binary shares; ge_many expects binary shares.
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    // x < y ≡ !(x >= y)
    let ge_bits: Vec<Rep3RingShare<Bit>> = rep3_ring::arithmetic::ge_many(&xs_bin, &ys_bin, io_ctx)?;
    let lt_bits: Vec<Rep3RingShare<Bit>> = ge_bits.iter().map(|b| !b).collect();
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&lt_bits, io_ctx)?;
    Ok(field)
}

/// GreaterThan: x > y ≡ y < x ≡ !(y >= x)
fn eval_greater_than<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let ge_bits: Vec<Rep3RingShare<Bit>> = rep3_ring::arithmetic::ge_many(&ys_bin, &xs_bin, io_ctx)?;
    let gt_bits: Vec<Rep3RingShare<Bit>> = ge_bits.iter().map(|b| !b).collect();
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&gt_bits, io_ctx)?;
    Ok(field)
}

/// Eq: x == y. eq_many does A2B internally — but we need uninterleaved operands.
/// So we A2B + uninterleave to get binary x,y, then use binary is_zero on x^y.
fn eval_eq<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    // x == y iff (x ^ y) == 0 in binary domain
    let diff: Vec<Rep3RingShare<u64>> = xs_bin.iter().zip(ys_bin.iter()).map(|(x, y)| *x ^ *y).collect();
    let eq_bits = rep3_ring::binary::is_zero_many(&diff, io_ctx)?;
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&eq_bits, io_ctx)?;
    Ok(field)
}

/// LeftIsZero: x == 0
fn eval_left_is_zero<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, _) = a2b_and_uninterleave(bits, io_ctx)?;
    let eq_bits = rep3_ring::binary::is_zero_many(&xs_bin, io_ctx)?;
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&eq_bits, io_ctx)?;
    Ok(field)
}

/// RightIsZero: y == 0
fn eval_right_is_zero<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (_, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let eq_bits = rep3_ring::binary::is_zero_many(&ys_bin, io_ctx)?;
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&eq_bits, io_ctx)?;
    Ok(field)
}

/// DivByZero: divisor==0 AND quotient==all_ones.
/// Uses binary domain for both checks.
fn eval_div_by_zero<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (divisors_bin, quotients_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let quotient_bits = suffix_len / 2;
    // quotient == all_ones iff quotient ^ all_ones == 0
    let all_ones_mask = RingElement(if quotient_bits >= 64 { u64::MAX } else { (1u64 << quotient_bits) - 1 });
    let q_xor: Vec<Rep3RingShare<u64>> = quotients_bin
        .iter()
        .map(|q| rep3_ring::binary::xor_public(q, &all_ones_mask, party_id))
        .collect();

    let divisor_zero = rep3_ring::binary::is_zero_many(&divisors_bin, io_ctx)?;
    let quotient_all_ones = rep3_ring::binary::is_zero_many(&q_xor, io_ctx)?;
    let result = rep3_ring::binary::and_many(&divisor_zero, &quotient_all_ones, io_ctx)?;
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&result, io_ctx)?;
    Ok(field)
}

/// OverflowBitsZero: upper (128 - XLEN) bits are all zero.
/// Arithmetic domain: (val >> XLEN) == 0.
fn eval_overflow_bits_zero<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let upper: Vec<Rep3RingShare<u128>> = bits.iter().map(|b| *b >> XLEN).collect();
    let zeros = vec![Rep3RingShare::default(); bits.len()];
    let eq_bits: Vec<Rep3RingShare<Bit>> = rep3_ring::arithmetic::eq_many(&upper, &zeros, io_ctx)?;
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&eq_bits, io_ctx)?;
    Ok(field)
}

// ---------------------------------------------------------------------------
// Change divisor
// ---------------------------------------------------------------------------

/// ChangeDivisor: (y == all_ones) AND (x == 0)
fn eval_change_divisor<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let y_len = suffix_len / 2;
    let all_ones_mask = RingElement(if y_len >= 64 { u64::MAX } else { (1u64 << y_len) - 1 });
    let y_xor: Vec<Rep3RingShare<u64>> = ys_bin
        .iter()
        .map(|y| rep3_ring::binary::xor_public(y, &all_ones_mask, party_id))
        .collect();

    let y_eq_all_ones = rep3_ring::binary::is_zero_many(&y_xor, io_ctx)?;
    let x_eq_zero = rep3_ring::binary::is_zero_many(&xs_bin, io_ctx)?;
    let result = rep3_ring::binary::and_many(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&result, io_ctx)?;
    Ok(field)
}

/// ChangeDivisorW: same but with W-variant (truncate to XLEN/2 bits)
fn eval_change_divisor_w<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let xs32: Vec<Rep3RingShare<u32>> = xs_bin.iter().map(|x| downcast::<u64, u32>(*x)).collect();
    let ys32: Vec<Rep3RingShare<u32>> = ys_bin.iter().map(|y| downcast::<u64, u32>(*y)).collect();

    let y_len = (suffix_len / 2).min(XLEN / 2);
    let all_ones_mask = RingElement(if y_len >= 32 { u32::MAX } else { (1u32 << y_len) - 1 });
    let y_xor: Vec<Rep3RingShare<u32>> = ys32
        .iter()
        .map(|y| rep3_ring::binary::xor_public(y, &all_ones_mask, party_id))
        .collect();

    let y_eq_all_ones = rep3_ring::binary::is_zero_many(&y_xor, io_ctx)?;
    let x_eq_zero = rep3_ring::binary::is_zero_many(&xs32, io_ctx)?;
    let result = rep3_ring::binary::and_many(&y_eq_all_ones, &x_eq_zero, io_ctx)?;
    let field = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&result, io_ctx)?;
    Ok(field)
}

// ---------------------------------------------------------------------------
// Pow2 suffixes
// ---------------------------------------------------------------------------

/// Pow2: 1 << shift where shift = low log2(XLEN) bits of the suffix.
fn eval_pow2<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let log_xlen = XLEN.log_2();
    let shift_mask = RingElement((1u128 << log_xlen.min(suffix_len)) - 1);
    let shifts: Vec<_> = bits.iter().map(|b| *b & shift_mask).collect();
    eval_pow2_from_shift_bits(&shifts, log_xlen.min(suffix_len), io_ctx, party_id)
}

/// Pow2W: same but shift = low 5 bits (modulo 32)
fn eval_pow2_w<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let shift_bits = 5usize.min(suffix_len);
    let shift_mask = RingElement((1u128 << shift_bits) - 1);
    let shifts: Vec<_> = bits.iter().map(|b| *b & shift_mask).collect();
    eval_pow2_from_shift_bits(&shifts, shift_bits, io_ctx, party_id)
}

/// Compute `1 << shift` where shift is a secret arithmetic value with at most `num_bits` bits.
/// Uses table lookup: for each possible shift value s in [0, 2^num_bits),
/// compute eq(shift, s) * (1 << s) and sum.
fn eval_pow2_from_shift_bits<F: JoltField, N: Rep3Network>(
    shift_vals: &[Rep3RingShare<u128>],
    num_bits: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let table_size = 1usize << num_bits;
    let n = shift_vals.len();

    // Batch all equality checks
    let mut all_shifts = Vec::with_capacity(n * table_size);
    let mut all_targets = Vec::with_capacity(n * table_size);
    for shift in shift_vals {
        for s in 0..table_size {
            all_shifts.push(*shift);
            all_targets.push(rep3_ring::arithmetic::promote_to_trivial_share(
                party_id,
                RingElement(s as u128),
            ));
        }
    }

    let eq_bits: Vec<Rep3RingShare<Bit>> =
        rep3_ring::arithmetic::eq_many(&all_shifts, &all_targets, io_ctx)?;
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
        result.push(acc);
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
fn eval_sign_extension<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (_, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let y_len = suffix_len / 2;
    let n = bits.len();
    let mut result = vec![Rep3PrimeFieldShare::<F>::zero_share(); n];

    // Extract individual bit shares (binary domain)
    let mut bit_shares: Vec<Vec<Rep3RingShare<Bit>>> = Vec::with_capacity(y_len);
    for i in 0..y_len {
        let bits_i: Vec<Rep3RingShare<Bit>> = ys_bin
            .iter()
            .map(|y| downcast::<u64, Bit>(*y >> i))
            .collect();
        bit_shares.push(bits_i);
    }

    let not_bits: Vec<Vec<Rep3RingShare<Bit>>> = bit_shares
        .iter()
        .map(|bs| bs.iter().map(|b| !b).collect())
        .collect();

    // running[j] = product of (bit_i == 0) for i < p
    let mut running = vec![
        rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(Bit::one()));
        n
    ];

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

    Ok(result)
}

/// SignExtensionUpperHalf: if suffix_len >= XLEN/2, check sign bit at position XLEN/2-1.
/// Uses arithmetic domain: extract single bit via mask+shift, convert to field, multiply by weight.
fn eval_sign_extension_upper_half<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let half = XLEN / 2;
    if suffix_len < half {
        return Ok(vec![
            Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id);
            bits.len()
        ]);
    }

    // Extract sign bit at position half-1 (arithmetic: mask + shift)
    let sign_bit_mask = RingElement(1u128 << (half - 1));
    let sign_bits: Vec<Rep3RingShare<u128>> = bits.iter().map(|b| (*b & sign_bit_mask) >> (half - 1)).collect();
    let weight = F::from_u128(((1u64 << half) - 1) as u128 * (1u128 << half));
    let sign_field: Vec<Rep3PrimeFieldShare<F>> =
        rep3_ring::casts::ring_to_field_many_selector(&sign_bits, io_ctx)?;
    let result: Vec<_> = sign_field.iter().map(|s| *s * weight).collect();
    Ok(result)
}

/// SignExtensionRightOperand: if suffix_len >= XLEN, check sign bit at XLEN-2.
fn eval_sign_extension_right_operand<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    suffix_len: usize,
    io_ctx: &mut IoContext<N>,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    if suffix_len < XLEN {
        return Ok(vec![
            Rep3PrimeFieldShare::promote_from_trivial(&F::one(), party_id);
            bits.len()
        ]);
    }

    let sign_bit_pos = XLEN - 2;
    let sign_bit_mask = RingElement(1u128 << sign_bit_pos);
    let sign_bits: Vec<Rep3RingShare<u128>> = bits.iter().map(|b| (*b & sign_bit_mask) >> sign_bit_pos).collect();
    let weight = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)));
    let sign_field: Vec<Rep3PrimeFieldShare<F>> =
        rep3_ring::casts::ring_to_field_many_selector(&sign_bits, io_ctx)?;
    let result: Vec<_> = sign_field.iter().map(|s| *s * weight).collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// XOR-rotate suffixes
// ---------------------------------------------------------------------------

/// XorRot: uninterleave → x^y → rotate_right(result, ROTATION).
/// All ops in binary domain. Rotate by constant is local.
fn eval_xor_rot<const ROTATION: u32, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let xor_results: Vec<Rep3RingShare<u64>> = xs_bin
        .iter()
        .zip(ys_bin.iter())
        .map(|(x, y)| *x ^ *y)
        .collect();
    // rotate_right = (val >> ROT) ^ (val << (64 - ROT)) — XOR since bit positions are disjoint
    let rotated: Vec<Rep3RingShare<u64>> = xor_results
        .iter()
        .map(|v| (*v >> (ROTATION as usize)) ^ (*v << (64 - ROTATION as usize)))
        .collect();
    let field = rep3_ring::casts::binary_ring_to_field_many(&rotated, io_ctx)?;
    Ok(field)
}

/// XorRotW: same but truncate to u32 first
fn eval_xor_rot_w<const ROTATION: u32, F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let (xs_bin, ys_bin) = a2b_and_uninterleave(bits, io_ctx)?;
    let xs32: Vec<Rep3RingShare<u32>> = xs_bin.iter().map(|x| downcast::<u64, u32>(*x)).collect();
    let ys32: Vec<Rep3RingShare<u32>> = ys_bin.iter().map(|y| downcast::<u64, u32>(*y)).collect();
    let xor_results: Vec<Rep3RingShare<u32>> = xs32
        .iter()
        .zip(ys32.iter())
        .map(|(x, y)| *x ^ *y)
        .collect();
    let rotated: Vec<Rep3RingShare<u32>> = xor_results
        .iter()
        .map(|v| (*v >> (ROTATION as usize)) ^ (*v << (32 - ROTATION as usize)))
        .collect();
    let field = rep3_ring::casts::binary_ring_to_field_many(&rotated, io_ctx)?;
    Ok(field)
}

// ---------------------------------------------------------------------------
// Rev8W
// ---------------------------------------------------------------------------

/// Rev8W: byte reversal of lower 32 bits.
/// Binary domain: extract bytes via mask+shift, recombine with XOR.
fn eval_rev8w<F: JoltField, N: Rep3Network>(
    bits: &[Rep3RingShare<u128>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let vals: Vec<Rep3RingShare<u64>> = bits.iter().map(|b| downcast::<u128, u64>(*b)).collect();
    let vals_bin = rep3_ring::conversion::a2b_many(&vals, io_ctx)?;
    let mask_byte = RingElement(0xFFu64);
    let reversed: Vec<Rep3RingShare<u64>> = vals_bin
        .iter()
        .map(|v| {
            let byte0 = *v & mask_byte;
            let byte1 = (*v >> 8) & mask_byte;
            let byte2 = (*v >> 16) & mask_byte;
            let byte3 = (*v >> 24) & mask_byte;
            // Non-overlapping positions → XOR == OR
            (byte0 << 24) ^ (byte1 << 16) ^ (byte2 << 8) ^ byte3
        })
        .collect();
    let field = rep3_ring::casts::binary_ring_to_field_many(&reversed, io_ctx)?;
    Ok(field)
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
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    use jolt_core::utils::lookup_bits::LookupBits;

    let opened = rep3_ring::arithmetic::open_vec(bits, io_ctx)?;
    let result: Vec<Rep3PrimeFieldShare<F>> = opened
        .iter()
        .map(|val| {
            let plain = val.0;
            let eval = suffix.suffix_mle::<XLEN>(LookupBits::new(plain, suffix_len));
            Rep3PrimeFieldShare::promote_from_trivial(&F::from_u64(eval), party_id)
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

    // Extract suffix bits (arithmetic domain)
    let suffix_mask = RingElement((1u128 << suffix_len) - 1);
    let suffix_bits: Vec<Rep3RingShare<u128>> = lookup_indices
        .iter()
        .map(|idx| *idx & suffix_mask)
        .collect();

    // Identity: suffix_bits as field (arithmetic domain → field)
    let identity = rep3_ring::casts::ring_to_field_many_selector(&suffix_bits, io_ctx)?;

    // A2B + uninterleave for left/right operands
    let (xs_bin, ys_bin) = a2b_and_uninterleave(&suffix_bits, io_ctx)?;
    let left_operand = rep3_ring::casts::binary_ring_to_field_many(&xs_bin, io_ctx)?;
    let right_operand = rep3_ring::casts::binary_ring_to_field_many(&ys_bin, io_ctx)?;

    Ok(OperandQSuffixEvals {
        left_operand,
        right_operand,
        identity,
    })
}
