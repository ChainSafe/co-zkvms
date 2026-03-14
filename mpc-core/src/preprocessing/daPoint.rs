//! daPoint: scalar-mul-free correlated randomness for `bit × public point` in Rep3.
//!
//! Provides secure multiplication of a secret-shared binary bit `b ∈ {0,1}` (Rep3 XOR shares)
//! with a public curve point `Q`, returning an **additive** per-party share of `b·Q`.
//!
//! Protocol overview:
//! - **Offline:** `random_dapoints` generates gamma-masked tuples using correlated RNGs.
//!   P0 holds secret bit gamma; P1/P2 hold additive shares of `gamma·Q` (no scalar muls).
//! - **Online:** `dot_product_dapoints` broadcasts `m = x ⊕ gamma` (1 round, P0 → P1,P2),
//!   then P1/P2 locally accumulate point additions/subtractions.

use std::path::PathBuf;

use crate::preprocessing::backing_store;
use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::network::{IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker};
use crate::protocols::rep3_ring::Rep3RingShare;
use crate::protocols::rep3_ring::ring::bit::Bit;
use crate::protocols::rep3_ring::ring::int_ring::IntRing2k;
use crate::protocols::rep3_ring::ring::ring_impl::RingElement;
use ark_ec::CurveGroup;
use itertools::izip;
use rand::Rng;

// ---------------------------------------------------------------------------
// EdaPoints: scalar-mul-free preprocessing for bit × public point
// ---------------------------------------------------------------------------

/// Flat batch of daPoint correlated tuples.
///
/// - `gammas[i]`: P0's secret mask bit (zeroed for P1/P2)
/// - `alphas[i]`: additive group share of `gamma_i * Q_i`
///   (P1 holds A1_i, P2 holds A2_i, P0 empty)
pub struct DaPointsBatch<C: CurveGroup> {
    pub gammas: Vec<Bit>,
    pub alphas: Vec<C>,
}

impl<C: CurveGroup> DaPointsBatch<C> {
    /// Extract a contiguous sub-batch `[start..start+len]`.
    pub fn slice(&self, start: usize, len: usize) -> Self {
        let gammas = self.gammas[start..start + len].to_vec();
        let alphas = if self.alphas.is_empty() {
            Vec::new()
        } else {
            self.alphas[start..start + len].to_vec()
        };
        Self { gammas, alphas }
    }

    /// Select a subset of entries by index, preserving the given order.
    pub fn select(&self, indices: &[usize]) -> Self {
        let gammas = indices.iter().map(|&i| self.gammas[i]).collect();
        let alphas = if self.alphas.is_empty() {
            // P0 has no alphas
            Vec::new()
        } else {
            indices.iter().map(|&i| self.alphas[i]).collect()
        };
        Self { gammas, alphas }
    }
}

/// Cursor-based daPoints source with persistence support.
///
/// P0 stores only gamma bits (tiny). P1/P2 store materialized curve points
/// (cannot regenerate lazily due to non-deterministic `C::rand()` consumption).
pub struct LazyDaPoints<C: CurveGroup> {
    total: usize,
    cursor: usize,
    gammas: Vec<Bit>,
    alphas: backing_store::BackingStore<C>,
    party_id: PartyID,
    meta_path: Option<PathBuf>,
}

impl<C: CurveGroup> LazyDaPoints<C> {
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            total: 0,
            cursor: 0,
            gammas: Vec::new(),
            alphas: backing_store::BackingStore::Empty,
            party_id,
            meta_path: None,
        }
    }

    pub fn new(total: usize, gammas: Vec<Bit>, alphas: Vec<C>, party_id: PartyID) -> Self {
        Self {
            total,
            cursor: 0,
            gammas,
            alphas: backing_store::BackingStore::from_vec(alphas),
            party_id,
            meta_path: None,
        }
    }

    pub fn remaining(&self) -> usize {
        self.total - self.cursor
    }

    /// Drain `n` daPoint tuples as a `DaPointsBatch`.
    pub fn take_batch(&mut self, n: usize) -> eyre::Result<DaPointsBatch<C>> {
        eyre::ensure!(
            self.cursor + n <= self.total,
            "LazyDaPoints: need {n}, have {} (cursor={}, total={})",
            self.remaining(),
            self.cursor,
            self.total,
        );

        if n == 0 {
            return Ok(DaPointsBatch { gammas: Vec::new(), alphas: Vec::new() });
        }

        let start = self.cursor;
        let end = start + n;

        let gammas =
            if self.party_id == PartyID::ID0 { self.gammas[start..end].to_vec() } else { vec![Bit::new(false); n] };

        let alphas = if self.party_id != PartyID::ID0 {
            let slice = self.alphas.as_slice();
            let points = slice[start..end].to_vec();
            self.alphas.consume(start, end);
            points
        } else {
            vec![C::zero(); n]
        };

        self.cursor += n;
        Ok(DaPointsBatch { gammas, alphas })
    }
}

/// Generate daPoint correlated tuples without scalar multiplications.
///
/// For each `Q_i`, produces a tuple `(gamma_i, A_i)` where:
/// - P0 holds `gamma_i` (secret bit), `A_i` is empty
/// - P1 holds `A1_i` (random curve point)
/// - P2 holds `A2_i = gamma_i * Q_i - A1_i`
///
/// **Communication:** 1 round, P0 → P2: `n` curve points.
/// **P2 does zero curve point generation.**
#[tracing::instrument(skip_all, name = "dapoints_preprocess", fields(n = qs.len()))]
pub fn random_dapoints<C: CurveGroup, N: Rep3NetworkWorker>(
    qs: &[C],
    io: &mut IoContextPool<N>,
) -> eyre::Result<LazyDaPoints<C>> {
    let n = qs.len();
    let party_id = io.party_id();
    if n == 0 {
        return Ok(LazyDaPoints::empty(party_id));
    }

    // Fork a dedicated Rep3Rand (all parties must call to keep main RNG aligned).
    let mut forked = io.main().rngs.rand.fork();

    let mut gammas = Vec::new();
    let mut alphas = Vec::new();

    match party_id {
        PartyID::ID0 => {
            // Generate gammas from XOR of both RNG streams (private to P0).
            gammas.reserve(n);
            for _ in 0..n {
                let g1: u8 = forked.rng1.r#gen();
                let g2: u8 = forked.rng2.r#gen();
                gammas.push(Bit::new((g1 ^ g2) & 1 == 1));
            }

            // Generate A1 from rng1 (shared with P1).
            let a1_points: Vec<C> = (0..n).map(|_| C::rand(&mut forked.rng1)).collect();

            // Compute A2 = (gamma ? Q : 0) - A1 and send to P2.
            let a2_all: Vec<C> = izip!(&gammas, qs, &a1_points)
                .map(|(gamma, q, a1)| {
                    let gamma_q = if gamma.convert() { *q } else { C::zero() };
                    gamma_q - a1
                })
                .collect();
            io.network().send_many(PartyID::ID2, &a2_all)?;
        }
        PartyID::ID1 => {
            // Advance rng2 past gamma bytes to align with P0's rng1.
            for _ in 0..n {
                let _: u8 = forked.rng2.r#gen();
            }
            // Generate A1 from rng2 (same stream as P0's rng1).
            alphas = (0..n).map(|_| C::rand(&mut forked.rng2)).collect();
        }
        PartyID::ID2 => {
            // Receive A2 from P0 (no curve point generation at all).
            alphas = io.network().recv_many(PartyID::ID0)?;
            debug_assert_eq!(alphas.len(), n);
        }
    }

    Ok(LazyDaPoints::new(n, gammas, alphas, party_id))
}

/// Like [`random_dapoints`] but takes column templates instead of a pre-expanded Q array.
///
/// Avoids materializing `2 * num_coeffs` Q points in memory. The Q-value ordering
/// matches [`precompute_dapoint_qs`]: per row `[q0_cols[0..seg], q1_cols[0..seg]]`.
///
/// `n_total = 2 * num_coeffs` daPoint tuples are generated.
///
/// Accepts a single `IoContext` (not pool) so it can run on a fork for parallelism.
#[tracing::instrument(skip_all, name = "dapoints_preprocess", fields(n = 2 * num_coeffs))]
pub fn random_dapoints_from_columns<C: CurveGroup, N: Rep3Network>(
    q0_cols: &[C],
    q1_cols: &[C],
    num_coeffs: usize,
    num_columns: usize,
    io: &mut IoContext<N>,
) -> eyre::Result<LazyDaPoints<C>> {
    let n = 2 * num_coeffs;
    let party_id = io.id;
    if n == 0 {
        return Ok(LazyDaPoints::empty(party_id));
    }

    let num_full_rows = num_coeffs / num_columns;
    let remainder = num_coeffs % num_columns;

    // Fork a dedicated Rep3Rand (all parties must call to keep RNG aligned).
    let mut forked = io.rngs.rand.fork();

    let mut gammas = Vec::new();
    let mut alphas = Vec::new();

    match party_id {
        PartyID::ID0 => {
            gammas.reserve(n);
            for _ in 0..n {
                let g1: u8 = forked.rng1.r#gen();
                let g2: u8 = forked.rng2.r#gen();
                gammas.push(Bit::new((g1 ^ g2) & 1 == 1));
            }

            let a1_points: Vec<C> = (0..n).map(|_| C::rand(&mut forked.rng1)).collect();

            // Iterate Q values from column templates (no full Q array allocation).
            let q_iter = (0..num_full_rows)
                .flat_map(|_| q0_cols[..num_columns].iter().chain(q1_cols[..num_columns].iter()))
                .chain(q0_cols[..remainder].iter().chain(q1_cols[..remainder].iter()));

            let a2_all: Vec<C> = izip!(&gammas, q_iter, &a1_points)
                .map(|(gamma, q, a1)| {
                    let gamma_q = if gamma.convert() { *q } else { C::zero() };
                    gamma_q - a1
                })
                .collect();
            io.network.send_many(PartyID::ID2, &a2_all)?;
        }
        PartyID::ID1 => {
            for _ in 0..n {
                let _: u8 = forked.rng2.r#gen();
            }
            alphas = (0..n).map(|_| C::rand(&mut forked.rng2)).collect();
        }
        PartyID::ID2 => {
            alphas = io.network.recv_many(PartyID::ID0)?;
            debug_assert_eq!(alphas.len(), n);
        }
    }

    Ok(LazyDaPoints::new(n, gammas, alphas, party_id))
}

// dot_product_dapoints moved to rep3/pointshare.rs

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::protocols::rep3::pointshare::dot_product_dapoints;
    use crate::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;
    use crate::protocols::rep3_ring::ring::ring_impl::RingElement;
    use ark_bn254::{Fr, G1Projective};
    use ark_std::UniformRand;
    use ark_std::Zero;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn dapoints_preproc_consistent() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let n = 16usize;
        let qs: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();

        // Return (gammas, alphas) from each party
        let outs: [(Vec<Bit>, Vec<G1Projective>); 3] = run_rep3_local_test_with_coordinator(
            1,
            |_party_idx| qs.clone(),
            || (),
            |qs, mut io_ctx| {
                let mut lazy = random_dapoints(&qs, &mut io_ctx)?;
                let batch = lazy.take_batch(qs.len())?;
                Ok((batch.gammas, batch.alphas))
            },
            |(), _net| Ok(()),
        )
        .0;

        let (gammas_p0, _alphas_p0) = &outs[0];
        let (_gammas_p1, alphas_p1) = &outs[1];
        let (_gammas_p2, alphas_p2) = &outs[2];

        // Verify: A1[i] + A2[i] == gamma[i] * Q[i]
        for i in 0..n {
            let gamma = gammas_p0[i];
            let a1 = alphas_p1[i];
            let a2 = alphas_p2[i];
            let expected = if gamma.convert() { qs[i] } else { G1Projective::zero() };
            assert_eq!(a1 + a2, expected, "A1[{i}] + A2[{i}] != gamma[{i}]*Q[{i}]");
        }
    }

    /// Test the online phase with hand-crafted preprocessing (bypasses RNG).
    #[test]
    fn dot_product_dapoints_manual_preproc() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let n = 4usize;

        let qs: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bits_plain: Vec<bool> = vec![true, false, true, true];

        // Hand-craft gamma values and corresponding A1/A2
        let gammas_plain: Vec<bool> = vec![false, true, true, false];
        let a1_points: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let a2_points: Vec<G1Projective> = izip!(&gammas_plain, &qs, &a1_points)
            .map(|(&g, q, a1)| {
                let gamma_q = if g { *q } else { G1Projective::zero() };
                gamma_q - a1
            })
            .collect();

        // Verify consistency: A1 + A2 == gamma * Q
        for i in 0..n {
            let expected = if gammas_plain[i] { qs[i] } else { G1Projective::zero() };
            assert_eq!(a1_points[i] + a2_points[i], expected);
        }

        let bit_shares_per_item: Vec<[Rep3RingShare<Bit>; 3]> = bits_plain
            .iter()
            .map(|&b| crate::protocols::rep3_ring::share_ring_element_binary(RingElement(Bit::new(b)), &mut rng))
            .collect();

        let bits_by_party: [Vec<Rep3RingShare<Bit>>; 3] =
            std::array::from_fn(|pid| bit_shares_per_item.iter().map(|s| s[pid]).collect::<Vec<_>>());

        // Build per-party batches
        let gammas_for_p0: Vec<Bit> = gammas_plain.iter().map(|&g| Bit::new(g)).collect();
        let batches: [DaPointsBatch<G1Projective>; 3] = [
            DaPointsBatch { gammas: gammas_for_p0, alphas: vec![G1Projective::zero(); n] },
            DaPointsBatch { gammas: vec![Bit::new(false); n], alphas: a1_points.clone() },
            DaPointsBatch { gammas: vec![Bit::new(false); n], alphas: a2_points.clone() },
        ];

        let outs: [G1Projective; 3] = run_rep3_local_test_with_coordinator(
            1,
            |party_idx| {
                (
                    bits_by_party[party_idx].clone(),
                    qs.clone(),
                    batches[party_idx].gammas.clone(),
                    batches[party_idx].alphas.clone(),
                )
            },
            || (),
            |(bits, qs, gammas, alphas), mut io_ctx| {
                let batch = DaPointsBatch { gammas, alphas };
                dot_product_dapoints(&bits, &qs, &batch, io_ctx.main())
            },
            |(), _net| Ok(()),
        )
        .0;

        let rec = outs[0] + outs[1] + outs[2];
        let exp = bits_plain
            .iter()
            .zip(qs.iter())
            .filter(|&(b, _)| *b)
            .map(|(_, q)| *q)
            .fold(G1Projective::zero(), |acc, p| acc + p);
        assert_eq!(rec, exp, "manual preproc dot product failed");
    }

    #[test]
    fn dot_product_dapoints_correct() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let n = 128usize;

        let qs: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bits_plain: Vec<bool> = (0..n).map(|_| (rng.next_u32() & 1) == 1).collect();

        let bit_shares_per_item: Vec<[Rep3RingShare<Bit>; 3]> = bits_plain
            .iter()
            .map(|&b| crate::protocols::rep3_ring::share_ring_element_binary(RingElement(Bit::new(b)), &mut rng))
            .collect();

        let bits_by_party: [Vec<Rep3RingShare<Bit>>; 3] =
            std::array::from_fn(|pid| bit_shares_per_item.iter().map(|s| s[pid]).collect::<Vec<_>>());

        let outs: [G1Projective; 3] = run_rep3_local_test_with_coordinator(
            1,
            |party_idx| (bits_by_party[party_idx].clone(), qs.clone()),
            || (),
            |(bits, qs), mut io_ctx| {
                let mut lazy = random_dapoints(&qs, &mut io_ctx)?;
                let batch = lazy.take_batch(qs.len())?;
                dot_product_dapoints(&bits, &qs, &batch, io_ctx.main())
            },
            |(), _net| Ok(()),
        )
        .0;

        let rec = outs[0] + outs[1] + outs[2];
        let exp = bits_plain
            .iter()
            .zip(qs.iter())
            .filter(|&(b, _)| *b)
            .map(|(_, q)| *q)
            .fold(G1Projective::zero(), |acc, p| acc + p);
        assert_eq!(rec, exp, "daPoints dot product must match cleartext");
    }

    #[test]
    fn dot_product_dapoints_all_zeros() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let n = 32usize;

        let qs: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bits_plain: Vec<bool> = vec![false; n];

        let bit_shares_per_item: Vec<[Rep3RingShare<Bit>; 3]> = bits_plain
            .iter()
            .map(|&b| crate::protocols::rep3_ring::share_ring_element_binary(RingElement(Bit::new(b)), &mut rng))
            .collect();

        let bits_by_party: [Vec<Rep3RingShare<Bit>>; 3] =
            std::array::from_fn(|pid| bit_shares_per_item.iter().map(|s| s[pid]).collect::<Vec<_>>());

        let outs: [G1Projective; 3] = run_rep3_local_test_with_coordinator(
            1,
            |party_idx| (bits_by_party[party_idx].clone(), qs.clone()),
            || (),
            |(bits, qs), mut io_ctx| {
                let mut lazy = random_dapoints(&qs, &mut io_ctx)?;
                let batch = lazy.take_batch(qs.len())?;
                dot_product_dapoints(&bits, &qs, &batch, io_ctx.main())
            },
            |(), _net| Ok(()),
        )
        .0;

        let rec = outs[0] + outs[1] + outs[2];
        assert_eq!(rec, G1Projective::zero(), "all-zero bits should give identity");
    }

    #[test]
    fn dot_product_dapoints_all_ones() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let n = 32usize;

        let qs: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bits_plain: Vec<bool> = vec![true; n];

        let bit_shares_per_item: Vec<[Rep3RingShare<Bit>; 3]> = bits_plain
            .iter()
            .map(|&b| crate::protocols::rep3_ring::share_ring_element_binary(RingElement(Bit::new(b)), &mut rng))
            .collect();

        let bits_by_party: [Vec<Rep3RingShare<Bit>>; 3] =
            std::array::from_fn(|pid| bit_shares_per_item.iter().map(|s| s[pid]).collect::<Vec<_>>());

        let outs: [G1Projective; 3] = run_rep3_local_test_with_coordinator(
            1,
            |party_idx| (bits_by_party[party_idx].clone(), qs.clone()),
            || (),
            |(bits, qs), mut io_ctx| {
                let mut lazy = random_dapoints(&qs, &mut io_ctx)?;
                let batch = lazy.take_batch(qs.len())?;
                dot_product_dapoints(&bits, &qs, &batch, io_ctx.main())
            },
            |(), _net| Ok(()),
        )
        .0;

        let rec = outs[0] + outs[1] + outs[2];
        let exp = qs.iter().fold(G1Projective::zero(), |acc, q| acc + q);
        assert_eq!(rec, exp, "all-one bits should give sum of all Q_i");
    }
}
