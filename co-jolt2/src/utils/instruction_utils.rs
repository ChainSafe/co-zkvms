use jolt2_common::constants::{LookupIndexInt, XlenInt};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};

use crate::zkvm::instruction::Rep3Operand;

/// Convert a Rep3Operand to a binary Rep3RingShare<LookupIndexInt> for use in
/// interleave_bits_shared and bit extraction.
/// For Shared: zero-extend the binary XlenInt share components to LookupIndexInt.
/// For Public: promote to trivial binary share.
pub fn operand_to_binary_wide(op: &Rep3Operand, id: PartyID) -> Rep3RingShare<LookupIndexInt> {
    match op {
        Rep3Operand::Shared { binary, .. } => {
            // Binary share zero-extension: upper bits are 0 in both components.
            // This is valid because in XOR sharing (a XOR b = value),
            // casting each component preserves correctness: 0 XOR 0 = 0 for upper bits.
            Rep3RingShare::new_ring(
                RingElement(binary.a.0 as LookupIndexInt),
                RingElement(binary.b.0 as LookupIndexInt),
            )
        }
        Rep3Operand::Public(v) => {
            rep3_ring::binary::promote_to_trivial_share(id, &RingElement(*v as LookupIndexInt))
        }
    }
}

/// Mirrors vanilla `interleave_bits` on Rep3RingShare<LookupIndexInt>.
///
/// **Precondition:** Each input share component must have only the low XLEN bits set
/// (i.e., operands are XlenInt values zero-extended to LookupIndexInt). This is
/// guaranteed by `operand_to_binary_wide`.
///
/// The OR-based interleave algorithm is correct on components with at most XLEN bits
/// because each `x | (x << k)` step's overlapping region is immediately masked away.
///
/// Output: bit 2i+1 = even_bits[i], bit 2i = odd_bits[i] (matching vanilla convention).
/// Zero MPC communication.
pub fn interleave_bits_shared(
    even_bits: Rep3RingShare<LookupIndexInt>,
    odd_bits: Rep3RingShare<LookupIndexInt>,
) -> Rep3RingShare<LookupIndexInt> {
    #[cfg(not(feature = "rv32"))]
    fn interleave_cleartext(even: u128, odd: u128) -> u128 {
        let mut x = even;
        x = (x | (x << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128;
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333u128;
        x = (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555u128;

        let mut y = odd;
        y = (y | (y << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128;
        y = (y | (y << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128;
        y = (y | (y << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128;
        y = (y | (y << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128;
        y = (y | (y << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333u128;
        y = (y | (y << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555u128;

        (x << 1) | y
    }

    #[cfg(feature = "rv32")]
    fn interleave_cleartext(even: u64, odd: u64) -> u64 {
        let mut x = even;
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFFu64;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FFu64;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0Fu64;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333u64;
        x = (x | (x << 1)) & 0x5555_5555_5555_5555u64;

        let mut y = odd;
        y = (y | (y << 16)) & 0x0000_FFFF_0000_FFFFu64;
        y = (y | (y << 8)) & 0x00FF_00FF_00FF_00FFu64;
        y = (y | (y << 4)) & 0x0F0F_0F0F_0F0F_0F0Fu64;
        y = (y | (y << 2)) & 0x3333_3333_3333_3333u64;
        y = (y | (y << 1)) & 0x5555_5555_5555_5555u64;

        (x << 1) | y
    }

    Rep3RingShare::new_ring(
        RingElement(interleave_cleartext(even_bits.a.0, odd_bits.a.0)),
        RingElement(interleave_cleartext(even_bits.b.0, odd_bits.b.0)),
    )
}

/// Upcast a `Rep3RingShare<Bit>` to `Rep3RingShare<u32>` (zero-extend in XOR domain).
pub fn bit_to_ring32(b: Rep3RingShare<Bit>) -> Rep3RingShare<u32> {
    Rep3RingShare::new(u8::from(b.a.0) as u32, u8::from(b.b.0) as u32)
}

/// Upcast a `Rep3RingShare<Bit>` to `Rep3RingShare<u64>` (zero-extend in XOR domain).
pub fn bit_to_ring64(b: Rep3RingShare<Bit>) -> Rep3RingShare<u64> {
    Rep3RingShare::new(u8::from(b.a.0) as u64, u8::from(b.b.0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_core::utils::interleave_bits;

    #[test]
    fn test_interleave_bits_shared_correctness() {
        // Test values: (x_operand u64, y_operand u64)
        let test_cases: Vec<(u64, u64)> = vec![
            (0, 0),
            (1, 0),
            (0, 1),
            (0xDEAD, 0xBEEF),
            (u64::MAX, u64::MAX),
            (0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210),
            (0x8000_0000_0000_0000, 0x0000_0000_0000_0001),
        ];

        for (x_val, y_val) in &test_cases {
            let vanilla = interleave_bits(*x_val, *y_val);

            // 3-party binary (XOR) sharing of x and y as u128
            // IMPORTANT: share components must only have low 64 bits set
            // (matching operand_to_binary_u128 which zero-extends u64 to u128)
            let ax: u128 = 0x0000_0000_0000_0000_2345_6789_ABCD_EF01;
            let bx: u128 = 0x0000_0000_0000_0000_FEDC_BA98_7654_3210;
            let cx: u128 = (*x_val as u128) ^ ax ^ bx;

            let ay: u128 = 0x0000_0000_0000_0000_1111_2222_3333_4444;
            let by: u128 = 0x0000_0000_0000_0000_9999_AAAA_BBBB_CCCC;
            let cy: u128 = (*y_val as u128) ^ ay ^ by;

            // Party 0: (a, b), Party 1: (b, c), Party 2: (c, a)
            let x_share0 = Rep3RingShare {
                a: RingElement(ax),
                b: RingElement(bx),
            };
            let y_share0 = Rep3RingShare {
                a: RingElement(ay),
                b: RingElement(by),
            };

            let x_share1 = Rep3RingShare {
                a: RingElement(bx),
                b: RingElement(cx),
            };
            let y_share1 = Rep3RingShare {
                a: RingElement(by),
                b: RingElement(cy),
            };

            let x_share2 = Rep3RingShare {
                a: RingElement(cx),
                b: RingElement(ax),
            };
            let y_share2 = Rep3RingShare {
                a: RingElement(cy),
                b: RingElement(ay),
            };

            let r0 = interleave_bits_shared(x_share0, y_share0);
            let r1 = interleave_bits_shared(x_share1, y_share1);
            let r2 = interleave_bits_shared(x_share2, y_share2);

            // Reconstruct: a ^ b ^ c
            let reconstructed = (r0.a ^ r0.b ^ r1.b).0;
            assert_eq!(
                reconstructed, vanilla,
                "interleave mismatch for x=0x{:016X}, y=0x{:016X}: got 0x{:032X}, expected 0x{:032X}",
                x_val, y_val, reconstructed, vanilla
            );

            // Check share consistency
            assert_eq!(r0.b.0, r1.a.0, "share consistency p0.b == p1.a failed");
            assert_eq!(r1.b.0, r2.a.0, "share consistency p1.b == p2.a failed");
            assert_eq!(r2.b.0, r0.a.0, "share consistency p2.b == p0.a failed");
        }
    }
}
