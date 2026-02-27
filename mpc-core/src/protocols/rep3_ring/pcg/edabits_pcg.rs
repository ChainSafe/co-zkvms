//! Standard edaBits and B2A conversion built on PCG daBits.
//!
//! An **edaBit** packs K individual daBits into a single unit:
//! - `r_bits`: K boolean Rep3 shares `[r_i]_2`
//! - `r_packed`: the packed ring share `[r]_{2^K}` where `r = Σ 2^i * r_i`
//! - `bit_values`: K arithmetic field shares `[r_i]_p`
//!
//! The **B2A** conversion (1 online round):
//! 1. Open `c = x XOR r_packed`
//! 2. `[x]_p = Σ 2^i * XOR_p(c_i, [r_i]_p)`
//!    where `XOR_p(0, [v]) = [v]` and `XOR_p(1, [v]) = 1 - [v]`.

use super::dabit_gen::{self, PcgDaBitSetup};
use crate::protocols::rep3::{
    PartyID, arithmetic as rep3_arith,
    network::{IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker},
};
use crate::protocols::rep3_ring::{arithmetic as rep3_ring_arith, binary};
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
    /// K arithmetic field shares of the same bits.
    pub bit_values: Vec<Rep3PrimeFieldShare<F>>,
}

// ── Lazy edaBit source ──────────────────────────────────────────────────────

/// Lazy PCG-based edaBit source that expands daBits from seeds on demand.
///
/// Stores pairwise PRF seeds + corrections (~O(N) for P0/P2 corrections).
/// Online `take(n)` expands n edaBits locally with no communication.
pub struct LazyPcgEdaBits<T: IntRing2k, F: PrimeField> {
    setup: PcgDaBitSetup<F>,
    total: usize,
    cursor: usize,
    _phantom: PhantomData<T>,
}

impl<T: IntRing2k, F: PrimeField> LazyPcgEdaBits<T, F>
where
    Standard: Distribution<T>,
{
    /// Create a lazy edaBit source from a daBit setup.
    ///
    /// `total` is the number of edaBits available (each edaBit uses T::K daBits).
    pub fn new(setup: PcgDaBitSetup<F>, total: usize) -> Self {
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
                corrections: Vec::new(),
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

        if n == 0 {
            return Vec::new();
        }

        let k = T::K;
        let dabit_start = self.cursor * k;
        let dabit_count = n * k;

        // Expand all needed daBits in parallel
        let dabits = dabit_gen::expand_dabits::<F>(&self.setup, dabit_start, dabit_count);

        // Pack K daBits into each edaBit
        let result: Vec<PcgEdaBit<T, F>> = (0..n)
            .into_par_iter()
            .with_min_len(64)
            .map(|i| {
                let chunk = &dabits[i * k..(i + 1) * k];
                let r_bits: Vec<Rep3RingShare<Bit>> = chunk.iter().map(|d| d.bit).collect(); // TODO: avoid iterating here; unzip or mem from parts?
                let r_packed = binary::pack_bits::<T>(&r_bits);
                let bit_values: Vec<Rep3PrimeFieldShare<F>> =
                    chunk.iter().map(|d| d.value).collect(); // TODO: avoid iterating here
                PcgEdaBit {
                    r_bits,
                    r_packed,
                    bit_values,
                }
            })
            .collect();

        self.cursor += n;
        result
    }
}

// ── B2A Conversion ──────────────────────────────────────────────────────────

/// Convert binary (XOR-shared) ring shares to arithmetic field shares.
///
/// Standard B2A with 1 online round:
/// 1. Open `c = x XOR r_packed`
/// 2. `[x]_p = Σ 2^i * XOR_p(c_i, [r_i]_p)`
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

    // Local: [x]_p = Σ 2^i * XOR_p(c_i, [r_i]_p)
    let results: Vec<Rep3PrimeFieldShare<F>> = c_values
        .into_par_iter()
        .zip(&eda)
        .map(|(c, e)| {
            let mut result = Rep3PrimeFieldShare::zero_share();
            for i in 0..k {
                let c_bit = ((c >> i) & T::one()) == T::one();
                let contrib = if !c_bit {
                    // XOR_p(0, [r_i]) = [r_i]
                    e.bit_values[i]
                } else {
                    // XOR_p(1, [r_i]) = 1 - [r_i]
                    rep3_arith::sub_public_by_shared(F::one(), e.bit_values[i], party_id)
                };
                result = result + contrib * pow2[i];
            }
            result
        })
        .collect();

    Ok(results)
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
///
/// Each field element encodes one u64 chunk. Since p >> 2^64 for bn254,
/// `F::from(val).into_bigint()` has `val` in the first u64 limb.
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
/// P0 (trusted dealer) generates pairwise seeds and corrections, then
/// distributes the per-party setups via the network.
///
/// **Communication:** P0 → P1: 8 field elements (2 seeds).
///                    P0 → P2: 8 + num field elements (2 seeds + corrections).
pub fn random_pcg_dabit_setup<F: PrimeField, N: Rep3NetworkWorker>(
    num: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PcgDaBitSetup<F>> {
    let party_id = io.party_id();

    if party_id == PartyID::ID0 {
        let mut rng = rand::thread_rng();
        let dealer = dabit_gen::dealer_setup::<F>(num, &mut rng);

        // Send P1's seeds (8 field elements = 2 × 32-byte seeds)
        let p1_data: Vec<F> = {
            let mut v = Vec::with_capacity(8);
            v.extend(seed_to_fields::<F>(&dealer.party1.seed_next));
            v.extend(seed_to_fields::<F>(&dealer.party1.seed_prev));
            v
        };
        io.network().send_many(PartyID::ID1, &p1_data)?;

        // Send P2's seeds + corrections
        let p2_data: Vec<F> = {
            let mut v = Vec::with_capacity(8 + num);
            v.extend(seed_to_fields::<F>(&dealer.party2.seed_next));
            v.extend(seed_to_fields::<F>(&dealer.party2.seed_prev));
            v.extend_from_slice(&dealer.party2.corrections);
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
            corrections: Vec::new(),
        })
    } else {
        let data: Vec<F> = io.network().recv_many(PartyID::ID0)?;
        assert!(
            data.len() >= 8 + num,
            "P2: expected {} field elements, got {}",
            8 + num,
            data.len()
        );
        let seed_next = fields_to_seed::<F>(&data[0..4]);
        let seed_prev = fields_to_seed::<F>(&data[4..8]);
        let corrections = data[8..].to_vec();
        Ok(PcgDaBitSetup {
            party_id,
            seed_next,
            seed_prev,
            seed_third: None,
            corrections,
        })
    }
}

/// Generate a lazy PCG edaBit source for ring type `T`.
///
/// P0 runs dealer setup for `num * T::K` daBits and distributes setups.
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

    let total_dabits = num * T::K;
    let setup = random_pcg_dabit_setup::<F, N>(total_dabits, io)?;
    Ok(LazyPcgEdaBits::new(setup, num))
}

// ── PcgEdaBitsPool ─────────────────────────────────────────────────────────

/// A pool of PCG-generated edaBits and daBits for batched binary→field conversions.
///
/// Drop-in replacement for `EdaBitsPool<F>`, backed by `LazyPcgEdaBits` sources
/// that expand from compact seeds with zero online communication.
pub struct PcgEdaBitsPool<F: PrimeField> {
    edabits_u8: LazyPcgEdaBits<u8, F>,
    edabits_u16: LazyPcgEdaBits<u16, F>,
    edabits_u32: LazyPcgEdaBits<u32, F>,
    edabits_u64: LazyPcgEdaBits<u64, F>,
    edabits_u128: LazyPcgEdaBits<u128, F>,
    dabits: Vec<crate::protocols::rep3_ring::edabits::DaBit<F>>, // TODO: lazy dabits
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
            dabits: Vec::new(),
        }
    }

    /// Create a pool from lazy PCG edaBits sources and eager daBits.
    pub fn new(
        edabits_u8: LazyPcgEdaBits<u8, F>,
        edabits_u16: LazyPcgEdaBits<u16, F>,
        edabits_u32: LazyPcgEdaBits<u32, F>,
        edabits_u64: LazyPcgEdaBits<u64, F>,
        edabits_u128: LazyPcgEdaBits<u128, F>,
        dabits: Vec<crate::protocols::rep3_ring::edabits::DaBit<F>>,
    ) -> Self {
        Self {
            edabits_u8,
            edabits_u16,
            edabits_u32,
            edabits_u64,
            edabits_u128,
            dabits,
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

    /// Drain `n` dabits from the pool. Panics if insufficient.
    #[tracing::instrument(skip(self))]
    pub fn take_dabits(&mut self, n: usize) -> Vec<crate::protocols::rep3_ring::edabits::DaBit<F>> {
        assert!(
            self.dabits.len() >= n,
            "PcgEdaBitsPool: need {n} dabits, have {}",
            self.dabits.len()
        );
        self.dabits.drain(..n).collect()
    }

    pub fn remaining_u64(&self) -> usize {
        self.edabits_u64.remaining()
    }

    pub fn remaining_u128(&self) -> usize {
        self.edabits_u128.remaining()
    }

    pub fn remaining_dabits(&self) -> usize {
        self.dabits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edabits_u8.remaining() == 0
            && self.edabits_u16.remaining() == 0
            && self.edabits_u32.remaining() == 0
            && self.edabits_u64.remaining() == 0
            && self.edabits_u128.remaining() == 0
            && self.dabits.is_empty()
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
    use mpc_types::protocols::rep3::{combine_field_element, combine_field_elements};
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
        let num_dabits = num_edabits * k;

        let dealer = dabit_gen::dealer_setup::<Fr>(num_dabits, &mut rng);

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

            // Check bits and values match
            for j in 0..k {
                let bit: RingElement<Bit> =
                    combine_ring_element(eda0[i].r_bits[j], eda1[i].r_bits[j], eda2[i].r_bits[j]);
                let b: bool = bit.0.convert();

                let val = combine_field_element(
                    eda0[i].bit_values[j],
                    eda1[i].bit_values[j],
                    eda2[i].bit_values[j],
                );
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
        let k = u64::K;
        let dealer = dabit_gen::dealer_setup::<Fr>(NUM * k, &mut dealer_rng);
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

        let k = u64::K;

        // Run 3-party B2A with distributed setup (via network)
        let (outputs, _) = run_rep3_local_test_with_coordinator(
            1,
            |i| x_bin_shares[i].clone(),
            || (),
            move |x_shares, mut io_ctx| {
                // Distributed setup via network
                let setup = random_pcg_dabit_setup::<Fr, _>(NUM * k, &mut io_ctx)?;
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
