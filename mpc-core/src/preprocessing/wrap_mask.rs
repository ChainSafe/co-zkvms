//! Wrap-mask preprocessing for DaBit-based extraction of 2-bit wrap counts.
//!
//! Used by Dory U64Scalars commit to extract the wrap count m ∈ {0,1,2} from
//! diff = m·2^64 in Z_{2^66}, replacing the expensive A2B Kogge-Stone adder
//! (7 rounds for u128) with a single batched open (1 round).

use crate::IoResult;
use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::network::{IoContext, Rep3Network};
use crate::protocols::rep3_ring::Rep3RingShare;
use crate::protocols::rep3_ring::ring::bit::Bit;
use crate::protocols::rep3_ring::ring::int_ring::IntRing2k;
use crate::protocols::rep3_ring::ring::ring_impl::RingElement;
use crate::protocols::rep3_ring::ring::u66::U66;
use crate::protocols::rep3_ring::{arithmetic, binary, conversion};
use rand::{Rng, SeedableRng};

use super::backing_store;

/// Preprocessed wrap masks: 2 correlated random bits per coefficient,
/// shared in both binary and arithmetic (U66) domains.
pub struct WrapMaskBatch {
    /// Binary shares of random bit r0 (one per coefficient).
    pub r0_bin: Vec<Rep3RingShare<Bit>>,
    /// Binary shares of random bit r1 (one per coefficient).
    pub r1_bin: Vec<Rep3RingShare<Bit>>,
    /// Arithmetic U66 shares of R = (r0 + 2·r1) · 2^64.
    pub mask_arith: Vec<Rep3RingShare<U66>>,
}

// ---------------------------------------------------------------------------
// LazyWrapMasks — lazy wrap masks with BackingStore persistence
// ---------------------------------------------------------------------------

/// Lazy wrap masks source. Binary shares are regenerated from seeds; arithmetic
/// shares (communication-dependent) are stored in a `BackingStore`.
///
/// ALL parties store `mask_arith_flat` (the bit_inject reshare results are not
/// seed-regenerable). `r0_bin` / `r1_bin` are regenerated from forked RNG seeds.
pub struct LazyWrapMasks {
    seed1: [u8; crate::SEED_SIZE],
    pos1: u128,
    seed2: [u8; crate::SEED_SIZE],
    pos2: u128,
    total: usize,
    cursor: usize,
    /// Interleaved [a₀, b₀, a₁, b₁, ...] for mask_arith shares.
    /// ALL parties store this (bit_inject results not seed-regenerable).
    mask_arith_flat: backing_store::BackingStore<RingElement<U66>>,
    party_id: PartyID,
    meta_path: Option<std::path::PathBuf>,
}

impl LazyWrapMasks {
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            seed1: [0u8; crate::SEED_SIZE],
            pos1: 0,
            seed2: [0u8; crate::SEED_SIZE],
            pos2: 0,
            total: 0,
            cursor: 0,
            mask_arith_flat: backing_store::BackingStore::Empty,
            party_id,
            meta_path: None,
        }
    }

    pub fn new(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        total: usize,
        mask_arith_flat: Vec<RingElement<U66>>,
        party_id: PartyID,
    ) -> Self {
        Self {
            seed1,
            pos1,
            seed2,
            pos2,
            total,
            cursor: 0,
            mask_arith_flat: backing_store::BackingStore::from_vec(mask_arith_flat),
            party_id,
            meta_path: None,
        }
    }

    pub fn remaining(&self) -> usize {
        self.total - self.cursor
    }

    /// Drain `n` wrap masks. Binary shares regenerated from seeds;
    /// arithmetic shares sliced from backing store.
    pub fn take_batch(&mut self, n: usize) -> eyre::Result<WrapMaskBatch> {
        eyre::ensure!(
            self.cursor + n <= self.total,
            "LazyWrapMasks: need {n}, have {} (cursor={}, total={})",
            self.remaining(),
            self.cursor,
            self.total
        );

        if n == 0 {
            return Ok(WrapMaskBatch { r0_bin: Vec::new(), r1_bin: Vec::new(), mask_arith: Vec::new() });
        }

        let cursor_base = self.cursor;

        // Regenerate binary shares by replaying the forked RNG from the snapshot.
        // Generation order: 2*total calls, items [0..total) → r0, [total..2*total) → r1.
        // We skip the first `cursor_base` items, take n for r0, skip to total, take n for r1.
        let mut rng1 = crate::RngType::from_seed(self.seed1);
        rng1.set_word_pos(self.pos1);
        let mut rng2 = crate::RngType::from_seed(self.seed2);
        rng2.set_word_pos(self.pos2);

        // Skip past items [0..cursor_base)
        for _ in 0..cursor_base {
            let _: RingElement<Bit> = rng1.r#gen();
            let _: RingElement<Bit> = rng2.r#gen();
        }
        // Take n items for r0_bin
        let r0_bin: Vec<Rep3RingShare<Bit>> = (0..n)
            .map(|_| {
                let a: RingElement<Bit> = rng1.r#gen();
                let b: RingElement<Bit> = rng2.r#gen();
                Rep3RingShare { a, b }
            })
            .collect();
        // Skip remaining r0 items [cursor_base+n..total)
        for _ in 0..(self.total - cursor_base - n) {
            let _: RingElement<Bit> = rng1.r#gen();
            let _: RingElement<Bit> = rng2.r#gen();
        }
        // Skip past r1 items [0..cursor_base)
        for _ in 0..cursor_base {
            let _: RingElement<Bit> = rng1.r#gen();
            let _: RingElement<Bit> = rng2.r#gen();
        }
        // Take n items for r1_bin
        let r1_bin: Vec<Rep3RingShare<Bit>> = (0..n)
            .map(|_| {
                let a: RingElement<Bit> = rng1.r#gen();
                let b: RingElement<Bit> = rng2.r#gen();
                Rep3RingShare { a, b }
            })
            .collect();

        // Slice mask_arith from backing store (interleaved [a, b, a, b, ...]).
        let flat_start = cursor_base * 2;
        let flat_end = flat_start + n * 2;
        let flat = &self.mask_arith_flat.as_slice()[flat_start..flat_end];
        let mask_arith: Vec<Rep3RingShare<U66>> =
            (0..n).map(|i| Rep3RingShare { a: flat[2 * i], b: flat[2 * i + 1] }).collect();

        self.cursor += n;
        self.persist_cursor();
        self.mask_arith_flat.consume(flat_start, flat_end);

        Ok(WrapMaskBatch { r0_bin, r1_bin, mask_arith })
    }

    pub fn save(&mut self, dir: &std::path::Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        if !self.mask_arith_flat.is_empty() {
            let data_path = dir.join("wrap_masks.data");
            self.mask_arith_flat.save_to_file(&data_path)?;
        }
        let meta_path = dir.join("wrap_masks.meta");
        backing_store::write_meta(
            &meta_path,
            &backing_store::MetaData {
                seed1: self.seed1,
                pos1: self.pos1,
                seed2: self.seed2,
                pos2: self.pos2,
                total: self.total,
                party_id_byte: backing_store::party_id_to_byte(self.party_id),
                cursor: self.cursor,
                field_bytes: std::mem::size_of::<RingElement<U66>>(),
            },
        )?;
        self.meta_path = Some(meta_path);
        std::result::Result::Ok(())
    }

    pub fn load(dir: &std::path::Path, party_id: PartyID) -> std::io::Result<Self> {
        let meta_path = dir.join("wrap_masks.meta");
        if !meta_path.exists() {
            return std::result::Result::Ok(Self::empty(party_id));
        }
        let meta = backing_store::read_meta(&meta_path)?;
        assert_eq!(meta.party_id_byte, backing_store::party_id_to_byte(party_id));
        let mask_arith_flat = if meta.total > 0 {
            let data_path = dir.join("wrap_masks.data");
            backing_store::BackingStore::load_from_file(&data_path)?
        } else {
            backing_store::BackingStore::Empty
        };
        std::result::Result::Ok(Self {
            seed1: meta.seed1,
            pos1: meta.pos1,
            seed2: meta.seed2,
            pos2: meta.pos2,
            total: meta.total,
            cursor: meta.cursor,
            mask_arith_flat,
            party_id,
            meta_path: Some(meta_path),
        })
    }

    fn persist_cursor(&self) {
        if let Some(ref path) = self.meta_path {
            let _ = backing_store::update_cursor(path, self.cursor);
        }
    }
}

/// Generate lazy wrap masks for `n` coefficients (offline, 2 rounds for bit_inject).
///
/// Binary shares are generated from a forked RNG (seed-regenerable).
/// Arithmetic shares are stored in a BackingStore (all parties).
pub fn generate_wrap_masks_lazy<N: Rep3Network>(n: usize, io: &mut IoContext<N>) -> IoResult<LazyWrapMasks> {
    let party_id = io.id;
    if n == 0 {
        return Ok(LazyWrapMasks::empty(party_id));
    }

    // Fork RNG for binary share generation and snapshot seeds.
    let mut bit_rand = io.rngs.rand.fork();
    let (seed1, pos1, seed2, pos2) = bit_rand.snapshot();

    // Generate 2n random binary-domain Bit shares from FORKED RNG (no communication).
    let mut all_bits: Vec<Rep3RingShare<Bit>> = Vec::with_capacity(2 * n);
    for _ in 0..2 * n {
        let (r1, r2) = bit_rand.random_elements::<RingElement<Bit>>();
        all_bits.push(Rep3RingShare { a: r1, b: r2 });
    }

    // Convert binary bit shares → arithmetic U66 shares (2 rounds, uses MAIN io RNG).
    let all_arith: Vec<Rep3RingShare<U66>> = conversion::bit_inject_from_bits_many::<U66, N>(&all_bits, io)?;

    let r0_arith = &all_arith[..n];
    let r1_arith = &all_arith[n..];

    // R_A[i] = (r0_A[i] + 2·r1_A[i]) * 2^64 in Z_{2^66}
    let two = RingElement(U66::new(2));
    let shift = RingElement(U66::new(1u128 << 64));
    let mask_arith: Vec<Rep3RingShare<U66>> = r0_arith
        .iter()
        .zip(r1_arith.iter())
        .map(|(r0, r1)| {
            let r1_scaled = *r1 * two;
            let sum = *r0 + r1_scaled;
            sum * shift
        })
        .collect();

    // Flatten to interleaved [a₀, b₀, a₁, b₁, ...] for BackingStore.
    let flat: Vec<RingElement<U66>> = mask_arith.iter().flat_map(|s| [s.a, s.b]).collect();

    Ok(LazyWrapMasks::new(seed1, pos1, seed2, pos2, n, flat, party_id))
}

/// Extract binary shares of wrap bits m0, m1 from diff_u66 = m·2^64 in Z_{2^66}.
///
/// Online cost: 1 round (batched open). All other work is local.
pub fn extract_wrap_m2_from_diff_u66_many<N: Rep3Network>(
    diff_u66: &[Rep3RingShare<U66>],
    masks: &WrapMaskBatch,
    io: &mut IoContext<N>,
) -> IoResult<(Vec<Rep3RingShare<Bit>>, Vec<Rep3RingShare<Bit>>)> {
    let n = diff_u66.len();
    assert_eq!(n, masks.r0_bin.len());
    assert_eq!(n, masks.r1_bin.len());
    assert_eq!(n, masks.mask_arith.len());

    // c = diff - R (local, arithmetic in U66)
    let c: Vec<Rep3RingShare<U66>> = diff_u66.iter().zip(masks.mask_arith.iter()).map(|(d, r)| *d - *r).collect();

    // Open c (1 round)
    let c_open: Vec<RingElement<U66>> = arithmetic::open_vec(&c, io)?;

    let party_id = io.id;
    let mut m0_bin = Vec::with_capacity(n);
    let mut m1_bin = Vec::with_capacity(n);

    for i in 0..n {
        let ctop = ((c_open[i].0.inner() >> 64) & 3) as u8;
        let (m0, m1) = two_bit_add_public_const_into_binary_share(ctop, masks.r0_bin[i], masks.r1_bin[i], party_id);
        m0_bin.push(m0);
        m1_bin.push(m1);
    }

    Ok((m0_bin, m1_bin))
}

/// Recover binary shares of m = (ctop + r) mod 4 from public ctop and binary
/// shares of r0, r1 (where r = r0 + 2·r1).
///
/// Returns (m0_B, m1_B) where m = m0 + 2·m1.
/// Entirely local — no communication.
fn two_bit_add_public_const_into_binary_share(
    ctop: u8,
    r0_b: Rep3RingShare<Bit>,
    r1_b: Rep3RingShare<Bit>,
    party_id: PartyID,
) -> (Rep3RingShare<Bit>, Rep3RingShare<Bit>) {
    let c0 = (ctop & 1) != 0;
    let c1 = (ctop >> 1 & 1) != 0;

    // m0 = r0 XOR c0
    let m0 = binary::xor_public(&r0_b, &RingElement(Bit::new(c0)), party_id);

    // carry = c0 AND r0 (public AND: if c0=0 → 0, if c0=1 → r0)
    let carry = if c0 { r0_b } else { Rep3RingShare::default() };

    // m1 = (r1 XOR carry) XOR c1
    let r1_xor_carry = Rep3RingShare { a: r1_b.a ^ carry.a, b: r1_b.b ^ carry.b };
    let m1 = binary::xor_public(&r1_xor_carry, &RingElement(Bit::new(c1)), party_id);

    (m0, m1)
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::protocols::rep3::network::IoContextPool;
    use crate::protocols::rep3::test_utils::LocalRep3TestWorkerNet;
    use crate::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;

    #[test]
    fn wrap_mask_extraction_correct() {
        use crate::protocols::rep3_ring::{share_ring_element, share_ring_element_binary};
        use rand::SeedableRng;

        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(42);

        // Test wrap values m ∈ {0, 1, 2} for several coefficients.
        let test_ms: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 1, 0];
        let n = test_ms.len();

        // diff = m * 2^64 in U66
        let diffs: Vec<RingElement<U66>> = test_ms.iter().map(|&m| RingElement(U66::new((m as u128) << 64))).collect();

        // Share each diff arithmetically
        let diff_shares: Vec<[Rep3RingShare<U66>; 3]> =
            diffs.iter().map(|d| share_ring_element(*d, &mut rng)).collect();

        let (results, _) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| {
                let party_diffs: Vec<Rep3RingShare<U66>> = diff_shares.iter().map(|s| s[party_idx]).collect();
                (party_diffs, n)
            },
            || (),
            |(party_diffs, n): (Vec<Rep3RingShare<U66>>, usize), mut io_ctx: IoContextPool<LocalRep3TestWorkerNet>| {
                let io = io_ctx.main();
                let mut lazy_masks = generate_wrap_masks_lazy(n, io)?;
                let masks = lazy_masks.take_batch(n)?;
                let (m0_bin, m1_bin) = extract_wrap_m2_from_diff_u66_many(&party_diffs, &masks, io)?;
                Ok((m0_bin, m1_bin))
            },
            |(), _net| Ok(()),
        );

        // Reconstruct m0, m1 from all 3 parties and verify.
        for i in 0..n {
            let m0_a = results[0].0[i].a.0.convert() as u8;
            let m0_b = results[1].0[i].a.0.convert() as u8;
            let m0_c = results[2].0[i].a.0.convert() as u8;
            let m0 = m0_a ^ m0_b ^ m0_c;

            let m1_a = results[0].1[i].a.0.convert() as u8;
            let m1_b = results[1].1[i].a.0.convert() as u8;
            let m1_c = results[2].1[i].a.0.convert() as u8;
            let m1 = m1_a ^ m1_b ^ m1_c;

            let reconstructed_m = m0 + 2 * m1;
            assert_eq!(
                reconstructed_m, test_ms[i],
                "mismatch at index {i}: expected {}, got {}",
                test_ms[i], reconstructed_m
            );
        }
    }
}
