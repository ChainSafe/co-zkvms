use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};

use crate::zkvm::instruction::Rep3Operand;

/// Convert a Rep3Operand to a binary Rep3RingShare<u128> for use in interleave_bits_shared.
/// For Shared: zero-extend the binary u32 share components to u128.
/// For Public: promote to trivial binary share.
pub fn operand_to_binary_u128(op: &Rep3Operand, id: PartyID) -> Rep3RingShare<u128> {
    match op {
        Rep3Operand::Shared { binary, .. } => {
            // Binary share zero-extension: upper bits are 0 in both components.
            // This is valid because in XOR sharing (a XOR b = value),
            // casting each component preserves correctness: 0 XOR 0 = 0 for upper bits.
            Rep3RingShare::new_ring(
                RingElement(binary.a.0 as u128),
                RingElement(binary.b.0 as u128),
            )
        }
        Rep3Operand::Public(v) => {
            rep3_ring::binary::promote_to_trivial_share(id, &RingElement(*v as u128))
        }
    }
}

/// Mirrors vanilla `interleave_bits` on Rep3RingShare<u128>.
/// Interleave is a bit-permutation, so it can be applied to each XOR-share
/// component independently (preserving the XOR sharing). No communication.
pub fn interleave_bits_shared(
    even_bits: Rep3RingShare<u128>,
    odd_bits: Rep3RingShare<u128>,
) -> Rep3RingShare<u128> {
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

    Rep3RingShare::new_ring(
        RingElement(interleave_cleartext(even_bits.a.0, odd_bits.a.0)),
        RingElement(interleave_cleartext(even_bits.b.0, odd_bits.b.0)),
    )
}

/// Upcast a `Rep3RingShare<Bit>` to `Rep3RingShare<u32>` (zero-extend in XOR domain).
pub fn bit_to_ring32(b: Rep3RingShare<Bit>) -> Rep3RingShare<u32> {
    Rep3RingShare::new(u8::from(b.a.0) as u32, u8::from(b.b.0) as u32)
}
