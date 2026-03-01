//! Standard edaBits and B2A conversion built on PCG daBits.
//!
//! An **edaBit** packs K individual daBits into a single unit:
//! - `r_bits`: K boolean Rep3 shares `[r_i]_2`
//! - `r_packed`: the packed ring share `[r]_{2^K}` where `r = Σ 2^i * r_i`
//! - `bit_values`: K additive (3-of-3) field shares of the same bits
//!
//! The **B2A** conversion (2 online rounds):
//! 1. Open `c = x XOR r_packed` (1 round)
//! 2. Local: each party computes additive share of `[x]_p`
//! 3. Reshare additive → Rep3 (1 round, O(N) elements)

use super::dabit_gen::{self, PcgDaBitSetup};
use crate::protocols::rep3::{
    PartyID,
    network::{IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker},
};
use crate::protocols::rep3_ring::binary;
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3::Rep3PrimeFieldShare;
use mpc_types::protocols::rep3_ring::{
    Rep3RingShare,
    ring::{bit::Bit, int_ring::IntRing2k},
};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use std::marker::PhantomData;

// ── PcgEdaBit ───────────────────────────────────────────────────────────────

/// A standard edaBit: K daBits packed into a ring element.
#[derive(Debug, Clone)]
pub struct PcgEdaBit<T: IntRing2k, F: PrimeField> {
    /// K individual boolean shares of random bits.
    pub r_bits: Vec<Rep3RingShare<Bit>>,
    /// Packed ring share: `r = Σ 2^i * r_i`.
    pub r_packed: Rep3RingShare<T>,
    /// K additive (3-of-3) field shares of the same bits.
    /// Each party holds one `F` per bit; the sum over 3 parties equals the bit value.
    pub bit_values: Vec<F>,
}

// ── Lazy edaBit source ──────────────────────────────────────────────────────

/// Lazy PCG-based edaBit source that expands daBits from seeds on demand.
///
/// **O(1) storage** — only 3 pairwise seeds (96 bytes for P0/P2, 64 bytes for P1).
/// No O(N) corrections. Online `take(n)` expands n edaBits locally with no communication.
pub struct LazyPcgEdaBits<T: IntRing2k, F: PrimeField> {
    setup: PcgDaBitSetup,
    total: usize,
    cursor: usize,
    _phantom: PhantomData<(T, F)>,
}

impl<T: IntRing2k, F: PrimeField> LazyPcgEdaBits<T, F>
where
    Standard: Distribution<T>,
{
    /// Create a lazy edaBit source from a daBit setup.
    ///
    /// `total` is the number of edaBits available (each edaBit uses T::K daBits).
    pub fn new(setup: PcgDaBitSetup, total: usize) -> Self {
        Self {
            setup,
            total,
            cursor: 0,
            _phantom: PhantomData,
        }
    }

    /// Create an empty source.
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            setup: PcgDaBitSetup {
                party_id,
                seed_next: [0u8; 32],
                seed_prev: [0u8; 32],
                seed_third: None,
            },
            total: 0,
            cursor: 0,
            _phantom: PhantomData,
        }
    }

    /// Number of remaining edaBits.
    pub fn remaining(&self) -> usize {
        self.total - self.cursor
    }

    /// Expand `n` edaBits starting from the current cursor.
    pub fn take(&mut self, n: usize) -> Vec<PcgEdaBit<T, F>> {
        assert!(
            self.cursor + n <= self.total,
            "LazyPcgEdaBits: need {n}, have {}",
            self.remaining()
        );

        let result = dabit_gen::expand_edabits::<T, F>(&self.setup, self.cursor, n);
        self.cursor += n;
        result
    }
}

// ── B2A Conversion ──────────────────────────────────────────────────────────

/// Convert binary (XOR-shared) ring shares to arithmetic field shares.
///
/// B2A with 2 online rounds (additive edaBit shares):
/// 1. Open `c = x XOR r_packed` (1 round)
/// 2. Local: each party computes additive share of `[x]_p`
/// 3. Reshare additive → Rep3 via `masking_field_element` + `reshare_many` (1 round)
pub fn ring_to_field_b2a_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    eda: Vec<PcgEdaBit<T, F>>,
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    if x_binary.len() != eda.len() {
        return Err(eyre::anyhow!("ring_to_field_b2a_pcg: length mismatch"));
    }

    let n = x_binary.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let k = T::K;

    // Precompute powers of 2 in Fp.
    let pow2 = {
        let mut v = Vec::with_capacity(k);
        let mut cur = F::one();
        for _ in 0..k {
            v.push(cur);
            cur = cur + cur;
        }
        v
    };

    // Round 1: open c = x XOR r_packed (binary/XOR domain)
    let c_shares: Vec<Rep3RingShare<T>> = x_binary
        .par_iter()
        .zip(&eda)
        .map(|(x, e)| *x ^ e.r_packed)
        .collect();

    let c_values = binary::open_vec(&c_shares, io)?;
    let party_id = io.id;

    // Local: compute additive share of [x]_p = Σ 2^i * xor_additive(c_i, s_i)
    // where s_i is the additive field share of bit r_i.
    // xor_additive(0, s) = s, xor_additive(1, s) = -s (+ 1 for P0)
    let additive_results: Vec<F> = c_values
        .into_par_iter()
        .zip(&eda)
        .map(|(c, e)| {
            let mut result = F::zero();
            for i in 0..k {
                let c_bit = ((c >> i) & T::one()) == T::one();
                let s = e.bit_values[i];
                if c_bit {
                    result -= s * pow2[i];
                    if party_id == PartyID::ID0 {
                        result += pow2[i];
                    }
                } else {
                    result += s * pow2[i];
                }
            }
            result
        })
        .collect();

    // Round 2: reshare additive → Rep3 (same pattern as edabits.rs:798-807)
    let s_selfs: Vec<F> = additive_results
        .iter()
        .map(|v| *v + io.masking_field_element::<F>())
        .collect();
    let s_prevs = io.network.reshare_many(&s_selfs)?;

    Ok(s_selfs
        .into_iter()
        .zip(s_prevs)
        .map(|(s_self, s_prev)| Rep3PrimeFieldShare::new(s_self, s_prev))
        .collect())
}

// ── Distributed Setup ──────────────────────────────────────────────────────

/// Encode a 32-byte seed as 4 field elements (8 bytes each).
fn seed_to_fields<F: PrimeField>(seed: &[u8; 32]) -> [F; 4] {
    [
        F::from(u64::from_le_bytes(seed[0..8].try_into().unwrap())),
        F::from(u64::from_le_bytes(seed[8..16].try_into().unwrap())),
        F::from(u64::from_le_bytes(seed[16..24].try_into().unwrap())),
        F::from(u64::from_le_bytes(seed[24..32].try_into().unwrap())),
    ]
}

/// Decode a 32-byte seed from 4 field elements.
fn fields_to_seed<F: PrimeField>(fields: &[F]) -> [u8; 32] {
    assert!(fields.len() >= 4);
    let mut seed = [0u8; 32];
    for (i, f) in fields[..4].iter().enumerate() {
        let bigint = f.into_bigint();
        let limbs: &[u64] = bigint.as_ref();
        seed[i * 8..(i + 1) * 8].copy_from_slice(&limbs[0].to_le_bytes());
    }
    seed
}

/// Generate a PCG daBit setup distributed across 3 parties.
///
/// P0 (trusted dealer) generates pairwise seeds and distributes them.
///
/// **Communication:**
///   P0 → P1: 8 field elements (2 seeds = 64 bytes).
///   P0 → P2: 12 field elements (3 seeds = 96 bytes).
///
/// **Storage per party:** O(1) — just seeds.
pub fn random_pcg_dabit_setup<F: PrimeField, N: Rep3NetworkWorker>(
    io: &mut IoContextPool<N>,
) -> eyre::Result<PcgDaBitSetup> {
    let party_id = io.party_id();

    if party_id == PartyID::ID0 {
        let mut rng = rand::thread_rng();
        let dealer = dabit_gen::dealer_setup(&mut rng);

        // Send P1's seeds (8 field elements = 2 × 32-byte seeds)
        let p1_data: Vec<F> = {
            let mut v = Vec::with_capacity(8);
            v.extend(seed_to_fields::<F>(&dealer.party1.seed_next));
            v.extend(seed_to_fields::<F>(&dealer.party1.seed_prev));
            v
        };
        io.network().send_many(PartyID::ID1, &p1_data)?;

        // Send P2's seeds (12 field elements = 3 × 32-byte seeds)
        let p2_data: Vec<F> = {
            let mut v = Vec::with_capacity(12);
            v.extend(seed_to_fields::<F>(&dealer.party2.seed_next));
            v.extend(seed_to_fields::<F>(&dealer.party2.seed_prev));
            // P2 also gets the third seed (seed_01)
            let seed_third = dealer.party2.seed_third.unwrap();
            v.extend(seed_to_fields::<F>(&seed_third));
            v
        };
        io.network().send_many(PartyID::ID2, &p2_data)?;

        Ok(dealer.party0)
    } else if party_id == PartyID::ID1 {
        let data: Vec<F> = io.network().recv_many(PartyID::ID0)?;
        assert!(data.len() >= 8, "P1: expected 8 field elements for seeds");
        let seed_next = fields_to_seed::<F>(&data[0..4]);
        let seed_prev = fields_to_seed::<F>(&data[4..8]);
        Ok(PcgDaBitSetup {
            party_id,
            seed_next,
            seed_prev,
            seed_third: None,
        })
    } else {
        let data: Vec<F> = io.network().recv_many(PartyID::ID0)?;
        assert!(
            data.len() >= 12,
            "P2: expected 12 field elements, got {}",
            data.len()
        );
        let seed_next = fields_to_seed::<F>(&data[0..4]);
        let seed_prev = fields_to_seed::<F>(&data[4..8]);
        let seed_third = fields_to_seed::<F>(&data[8..12]);
        Ok(PcgDaBitSetup {
            party_id,
            seed_next,
            seed_prev,
            seed_third: Some(seed_third),
        })
    }
}

/// Generate a lazy PCG edaBit source for ring type `T`.
///
/// P0 runs dealer setup and distributes seeds (O(1) communication).
/// Each party constructs a `LazyPcgEdaBits` that can expand edaBits locally.
#[tracing::instrument(skip_all, name = "pcg_edabits_lazy", level = "trace", fields(num))]
pub fn random_pcg_edabits_lazy<T: IntRing2k, F: PrimeField, N: Rep3NetworkWorker>(
    num: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<LazyPcgEdaBits<T, F>>
where
    Standard: Distribution<T>,
{
    if num == 0 {
        return Ok(LazyPcgEdaBits::empty(io.party_id()));
    }

    let setup = random_pcg_dabit_setup::<F, N>(io)?;
    Ok(LazyPcgEdaBits::new(setup, num))
}

// ── PcgEdaBitsPool ─────────────────────────────────────────────────────────

/// A pool of PCG-generated edaBits and daBits for batched binary→field conversions.
///
/// Drop-in replacement for `EdaBitsPool<F>`, backed by `LazyPcgEdaBits` sources
/// that expand from compact seeds with zero online communication.
///
/// **Storage:** O(1) per ring width — just seeds, no O(N) corrections.
pub struct PcgEdaBitsPool<F: PrimeField> {
    edabits_u8: LazyPcgEdaBits<u8, F>,
    edabits_u16: LazyPcgEdaBits<u16, F>,
    edabits_u32: LazyPcgEdaBits<u32, F>,
    edabits_u64: LazyPcgEdaBits<u64, F>,
    edabits_u128: LazyPcgEdaBits<u128, F>,
    /// Lazy daBit source (same setup, expanded on demand).
    dabit_setup: PcgDaBitSetup,
    dabit_total: usize,
    dabit_cursor: usize,
}

impl<F: PrimeField> PcgEdaBitsPool<F> {
    /// Create an empty pool.
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            edabits_u8: LazyPcgEdaBits::empty(party_id),
            edabits_u16: LazyPcgEdaBits::empty(party_id),
            edabits_u32: LazyPcgEdaBits::empty(party_id),
            edabits_u64: LazyPcgEdaBits::empty(party_id),
            edabits_u128: LazyPcgEdaBits::empty(party_id),
            dabit_setup: PcgDaBitSetup {
                party_id,
                seed_next: [0u8; 32],
                seed_prev: [0u8; 32],
                seed_third: None,
            },
            dabit_total: 0,
            dabit_cursor: 0,
        }
    }

    /// Create a pool from lazy PCG edaBit sources and a daBit setup.
    pub fn new(
        edabits_u8: LazyPcgEdaBits<u8, F>,
        edabits_u16: LazyPcgEdaBits<u16, F>,
        edabits_u32: LazyPcgEdaBits<u32, F>,
        edabits_u64: LazyPcgEdaBits<u64, F>,
        edabits_u128: LazyPcgEdaBits<u128, F>,
        dabit_setup: PcgDaBitSetup,
        dabit_total: usize,
    ) -> Self {
        Self {
            edabits_u8,
            edabits_u16,
            edabits_u32,
            edabits_u64,
            edabits_u128,
            dabit_setup,
            dabit_total,
            dabit_cursor: 0,
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn take_edabits_u8(&mut self, n: usize) -> Vec<PcgEdaBit<u8, F>> {
        self.edabits_u8.take(n)
    }

    #[tracing::instrument(skip(self))]
    pub fn take_edabits_u16(&mut self, n: usize) -> Vec<PcgEdaBit<u16, F>> {
        self.edabits_u16.take(n)
    }

    #[tracing::instrument(skip(self))]
    pub fn take_edabits_u32(&mut self, n: usize) -> Vec<PcgEdaBit<u32, F>> {
        self.edabits_u32.take(n)
    }

    #[tracing::instrument(skip(self))]
    pub fn take_edabits_u64(&mut self, n: usize) -> Vec<PcgEdaBit<u64, F>> {
        self.edabits_u64.take(n)
    }

    #[tracing::instrument(skip(self))]
    pub fn take_edabits_u128(&mut self, n: usize) -> Vec<PcgEdaBit<u128, F>> {
        self.edabits_u128.take(n)
    }

    /// Expand `n` daBits on demand from the lazy daBit source.
    #[tracing::instrument(skip(self))]
    pub fn take_dabits(&mut self, n: usize) -> Vec<crate::protocols::rep3_ring::edabits::DaBit<F>> {
        assert!(
            self.dabit_cursor + n <= self.dabit_total,
            "PcgEdaBitsPool: need {n} dabits, have {}",
            self.dabit_total - self.dabit_cursor
        );
        let dabits = dabit_gen::expand_dabits(&self.dabit_setup, self.dabit_cursor, n);
        self.dabit_cursor += n;
        dabits
    }

    pub fn remaining_u64(&self) -> usize {
        self.edabits_u64.remaining()
    }

    pub fn remaining_u128(&self) -> usize {
        self.edabits_u128.remaining()
    }

    pub fn remaining_dabits(&self) -> usize {
        self.dabit_total - self.dabit_cursor
    }

    pub fn is_empty(&self) -> bool {
        self.edabits_u8.remaining() == 0
            && self.edabits_u16.remaining() == 0
            && self.edabits_u32.remaining() == 0
            && self.edabits_u64.remaining() == 0
            && self.edabits_u128.remaining() == 0
            && self.remaining_dabits() == 0
    }

    /// Generic edaBits drain, dispatched by `TypeId`.
    ///
    /// Panics if `T` is not one of u8, u16, u32, u64, u128.
    #[tracing::instrument(skip(self))]
    pub fn take_edabits<T: IntRing2k>(&mut self, n: usize) -> Vec<PcgEdaBit<T, F>>
    where
        Standard: Distribution<T>,
    {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<u8>() {
            let v = self.edabits_u8.take(n);
            // SAFETY: T == u8 confirmed by TypeId check.
            unsafe { std::mem::transmute::<Vec<PcgEdaBit<u8, F>>, Vec<PcgEdaBit<T, F>>>(v) }
        } else if tid == TypeId::of::<u16>() {
            let v = self.edabits_u16.take(n);
            unsafe { std::mem::transmute::<Vec<PcgEdaBit<u16, F>>, Vec<PcgEdaBit<T, F>>>(v) }
        } else if tid == TypeId::of::<u32>() {
            let v = self.edabits_u32.take(n);
            unsafe { std::mem::transmute::<Vec<PcgEdaBit<u32, F>>, Vec<PcgEdaBit<T, F>>>(v) }
        } else if tid == TypeId::of::<u64>() {
            let v = self.edabits_u64.take(n);
            unsafe { std::mem::transmute::<Vec<PcgEdaBit<u64, F>>, Vec<PcgEdaBit<T, F>>>(v) }
        } else if tid == TypeId::of::<u128>() {
            let v = self.edabits_u128.take(n);
            unsafe { std::mem::transmute::<Vec<PcgEdaBit<u128, F>>, Vec<PcgEdaBit<T, F>>>(v) }
        } else {
            panic!("PcgEdaBitsPool::take_edabits: unsupported ring type");
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use mpc_types::protocols::rep3::combine_field_elements;
    use mpc_types::protocols::rep3_ring::{
        combine_ring_element, combine_ring_element_binary, ring::ring_impl::RingElement,
        share_ring_element_binary,
    };
    use rand::{RngCore, SeedableRng};

    #[test]
    fn edabit_packing_consistent() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        // Need K=64 daBits per edaBit for u64
        let k = u64::K;
        let num_edabits = 10;

        let dealer = dabit_gen::dealer_setup(&mut rng);

        let mut lazy0 = LazyPcgEdaBits::<u64, Fr>::new(dealer.party0, num_edabits);
        let mut lazy1 = LazyPcgEdaBits::<u64, Fr>::new(dealer.party1, num_edabits);
        let mut lazy2 = LazyPcgEdaBits::<u64, Fr>::new(dealer.party2, num_edabits);

        let eda0 = lazy0.take(num_edabits);
        let eda1 = lazy1.take(num_edabits);
        let eda2 = lazy2.take(num_edabits);

        for i in 0..num_edabits {
            // Check r_packed = pack(r_bits)
            let packed_0 = binary::pack_bits::<u64>(&eda0[i].r_bits);
            assert_eq!(packed_0, eda0[i].r_packed, "eda {i}: pack mismatch P0");

            // Check bits and values match (additive shares: sum = bit value)
            for j in 0..k {
                let bit: RingElement<Bit> =
                    combine_ring_element(eda0[i].r_bits[j], eda1[i].r_bits[j], eda2[i].r_bits[j]);
                let b: bool = bit.0.convert();

                let val = eda0[i].bit_values[j] + eda1[i].bit_values[j] + eda2[i].bit_values[j];
                let expected = if b { Fr::from(1u64) } else { Fr::zero() };
                assert_eq!(val, expected, "eda {i}, bit {j}: bool/arith mismatch");
            }

            // Check packed value matches individual bits (binary world = XOR reconstruction)
            let packed_val: RingElement<u64> =
                combine_ring_element_binary(eda0[i].r_packed, eda1[i].r_packed, eda2[i].r_packed);
            let mut expected_packed = 0u64;
            for j in 0..k {
                let bit: RingElement<Bit> =
                    combine_ring_element(eda0[i].r_bits[j], eda1[i].r_bits[j], eda2[i].r_bits[j]);
                if bit.0.convert() {
                    expected_packed |= 1u64 << j;
                }
            }
            assert_eq!(
                packed_val.0, expected_packed,
                "eda {i}: packed bits mismatch"
            );
        }
    }

    #[test]
    fn b2a_pcg_recovers_xs_u64() {
        const NUM: usize = 16;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xB2A001);

        // Generate random u64 values
        let xs: Vec<u64> = (0..NUM).map(|_| rng.next_u64()).collect();

        // Create binary (XOR) shares for each party
        let per_val_shares: Vec<[Rep3RingShare<u64>; 3]> = xs
            .iter()
            .map(|&x| share_ring_element_binary::<u64, _>(RingElement(x), &mut rng))
            .collect();
        let x_bin_shares: [Vec<Rep3RingShare<u64>>; 3] =
            std::array::from_fn(|pid| per_val_shares.iter().map(|s| s[pid]).collect());

        // Generate PCG daBit setup (dealer generates for all 3 parties)
        let mut dealer_rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xB2A002);
        let dealer = dabit_gen::dealer_setup(&mut dealer_rng);
        let setups = [dealer.party0, dealer.party1, dealer.party2];

        // Run 3-party B2A protocol
        let (outputs, _) = run_rep3_local_test_with_coordinator(
            1,
            |i| (x_bin_shares[i].clone(), setups[i].clone()),
            || (),
            |(x_shares, setup), mut io_ctx| {
                let mut lazy = LazyPcgEdaBits::<u64, Fr>::new(setup, NUM);
                let edas = lazy.take(NUM);
                ring_to_field_b2a_many::<u64, Fr, _>(&x_shares, edas, io_ctx.main())
                    .map_err(Into::into)
            },
            |(), _net| Ok(()),
        );

        // Combine shares and verify
        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected: Vec<Fr> = xs.into_iter().map(Fr::from).collect();
        assert_eq!(
            combined, expected,
            "B2A PCG did not recover the original values"
        );
    }

    #[test]
    fn b2a_pcg_with_distributed_setup_u64() {
        const NUM: usize = 8;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xB2A003);

        // Generate random u64 values
        let xs: Vec<u64> = (0..NUM).map(|_| rng.next_u64()).collect();

        // Create binary (XOR) shares for each party
        let per_val_shares: Vec<[Rep3RingShare<u64>; 3]> = xs
            .iter()
            .map(|&x| share_ring_element_binary::<u64, _>(RingElement(x), &mut rng))
            .collect();
        let x_bin_shares: [Vec<Rep3RingShare<u64>>; 3] =
            std::array::from_fn(|pid| per_val_shares.iter().map(|s| s[pid]).collect());

        // Run 3-party B2A with distributed setup (via network)
        let (outputs, _) = run_rep3_local_test_with_coordinator(
            1,
            |i| x_bin_shares[i].clone(),
            || (),
            move |x_shares, mut io_ctx| {
                // Distributed setup via network
                let setup = random_pcg_dabit_setup::<Fr, _>(&mut io_ctx)?;
                let mut lazy = LazyPcgEdaBits::<u64, Fr>::new(setup, NUM);
                let edas = lazy.take(NUM);
                ring_to_field_b2a_many::<u64, Fr, _>(&x_shares, edas, io_ctx.main())
                    .map_err(Into::into)
            },
            |(), _net| Ok(()),
        );

        // Combine shares and verify
        let combined = combine_field_elements(&outputs[0], &outputs[1], &outputs[2]);
        let expected: Vec<Fr> = xs.into_iter().map(Fr::from).collect();
        assert_eq!(combined, expected, "B2A with distributed setup failed");
    }
}
