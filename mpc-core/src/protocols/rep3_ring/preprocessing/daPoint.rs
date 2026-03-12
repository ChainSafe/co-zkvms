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

use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::network::{IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker};
use crate::protocols::rep3_ring::preprocessing::backing_store;
use ark_ec::CurveGroup;
use itertools::izip;
use crate::protocols::rep3_ring::Rep3RingShare;
use crate::protocols::rep3_ring::ring::bit::Bit;
use crate::protocols::rep3_ring::ring::int_ring::IntRing2k;
use crate::protocols::rep3_ring::ring::ring_impl::RingElement;
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
            return Ok(DaPointsBatch {
                gammas: Vec::new(),
                alphas: Vec::new(),
            });
        }

        let start = self.cursor;
        let end = start + n;

        let gammas = if self.party_id == PartyID::ID0 {
            self.gammas[start..end].to_vec()
        } else {
            vec![Bit::new(false); n]
        };

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
#[tracing::instrument(skip_all, name = "dapoints_preprocess")]
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

/// Securely compute this party's additive share of `Σ_i bits[i] · qs[i]`
/// using gamma-based daPoint tuples (no scalar muls online).
///
/// **Communication:** 1 round — P0 broadcasts N bits to P1 and P2.
/// Returns `C` (additive share): P0 contributes 0, P1 contributes X1, P2 contributes X2.
pub fn dot_product_dapoints<C, N>(
    bits: &[Rep3RingShare<Bit>],
    qs: &[C],
    batch: &DaPointsBatch<C>,
    io: &mut IoContext<N>,
) -> eyre::Result<C>
where
    C: CurveGroup,
    N: Rep3Network,
{
    let n = bits.len();
    eyre::ensure!(qs.len() == n, "dot_product_dapoints: qs length mismatch");
    eyre::ensure!(
        batch.alphas.len() == n || (io.id == PartyID::ID0 && batch.alphas.is_empty()),
        "dot_product_dapoints: batch.alphas length mismatch"
    );
    eyre::ensure!(
        batch.gammas.len() == n || (io.id != PartyID::ID0 && batch.gammas.len() == n),
        "dot_product_dapoints: batch.gammas length mismatch"
    );
    if n == 0 {
        return Ok(C::zero());
    }

    // Round 1: P0 broadcasts masked bits m[i] = x.a ^ x.b ^ gamma[i].
    // Use RingElement<Bit> for serialization compatibility with Rep3Network.
    let ms: Vec<RingElement<Bit>> = if io.id == PartyID::ID0 {
        let ms: Vec<RingElement<Bit>> = izip!(bits, &batch.gammas)
            .map(|(x, gamma)| {
                RingElement(Bit::new(
                    x.a.0.convert() ^ x.b.0.convert() ^ gamma.convert(),
                ))
            })
            .collect();
        io.network.send_many(PartyID::ID1, &ms)?;
        io.network.send_many(PartyID::ID2, &ms)?;
        ms
    } else {
        io.network.recv_many(PartyID::ID0)?
    };

    // P0 contributes nothing.
    if io.id == PartyID::ID0 {
        return Ok(C::zero());
    }

    // P1/P2 compute beta and accumulate.
    let mut acc = C::zero();
    for (i, (m, x)) in ms.iter().zip(bits).enumerate() {
        // beta = m ^ x_2 where x_2 is the component known to both P1 and P2.
        // Rep3 share layout: P0=(t0,t2), P1=(t1,t0), P2=(t2,t1).
        // m = t0 ^ t2 ^ gamma. The missing share is t1.
        // P1 has t1 as x.a; P2 has t1 as x.b.
        let missing = match io.id {
            PartyID::ID1 => x.a.0,
            PartyID::ID2 => x.b.0,
            _ => unreachable!(),
        };
        let beta = m.0.convert() ^ missing.convert();

        let alpha = &batch.alphas[i];
        if beta {
            // beta=1: x_i * Q_i = Q_i - Gamma_i
            // P1 adds Q_i - A1_i, P2 adds -A2_i
            if io.id == PartyID::ID1 {
                acc += qs[i] - *alpha;
            } else {
                acc -= *alpha;
            }
        } else {
            // beta=0: x_i * Q_i = Gamma_i
            // P1 adds A1_i, P2 adds A2_i
            acc += *alpha;
        }
    }

    Ok(acc)
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;
    use ark_bn254::{Fr, G1Projective};
    use ark_std::UniformRand;
    use ark_std::Zero;
    use ark_std::test_rng;
    use rand::RngCore;

    #[test]
    fn dapoints_preproc_consistent() {
        let mut rng = test_rng();
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
            let expected = if gamma.convert() {
                qs[i]
            } else {
                G1Projective::zero()
            };
            assert_eq!(a1 + a2, expected, "A1[{i}] + A2[{i}] != gamma[{i}]*Q[{i}]");
        }
    }

    /// Test the online phase with hand-crafted preprocessing (bypasses RNG).
    #[test]
    fn dot_product_dapoints_manual_preproc() {
        let mut rng = test_rng();
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
            let expected = if gammas_plain[i] {
                qs[i]
            } else {
                G1Projective::zero()
            };
            assert_eq!(a1_points[i] + a2_points[i], expected);
        }

        let bit_shares_per_item: Vec<Vec<Rep3RingShare<Bit>>> = bits_plain
            .iter()
            .map(|&b| {
                crate::protocols::rep3_ring::binary::generate_shares_rep3::<Bit, _>(
                    Bit::new(b),
                    &mut rng,
                )
            })
            .collect();

        let bits_by_party: [Vec<Rep3RingShare<Bit>>; 3] = std::array::from_fn(|pid| {
            bit_shares_per_item
                .iter()
                .map(|s| s[pid])
                .collect::<Vec<_>>()
        });

        // Build per-party batches
        let gammas_for_p0: Vec<Bit> = gammas_plain.iter().map(|&g| Bit::new(g)).collect();
        let batches: [DaPointsBatch<G1Projective>; 3] = [
            DaPointsBatch {
                gammas: gammas_for_p0,
                alphas: vec![G1Projective::zero(); n],
            },
            DaPointsBatch {
                gammas: vec![Bit::new(false); n],
                alphas: a1_points.clone(),
            },
            DaPointsBatch {
                gammas: vec![Bit::new(false); n],
                alphas: a2_points.clone(),
            },
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
        let mut rng = test_rng();
        let n = 128usize;

        let qs: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bits_plain: Vec<bool> = (0..n).map(|_| (rng.next_u32() & 1) == 1).collect();

        let bit_shares_per_item: Vec<Vec<Rep3RingShare<Bit>>> = bits_plain
            .iter()
            .map(|&b| {
                crate::protocols::rep3_ring::binary::generate_shares_rep3::<Bit, _>(
                    Bit::new(b),
                    &mut rng,
                )
            })
            .collect();

        let bits_by_party: [Vec<Rep3RingShare<Bit>>; 3] = std::array::from_fn(|pid| {
            bit_shares_per_item
                .iter()
                .map(|s| s[pid])
                .collect::<Vec<_>>()
        });

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
        let mut rng = test_rng();
        let n = 32usize;

        let qs: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bits_plain: Vec<bool> = vec![false; n];

        let bit_shares_per_item: Vec<Vec<Rep3RingShare<Bit>>> = bits_plain
            .iter()
            .map(|&b| {
                crate::protocols::rep3_ring::binary::generate_shares_rep3::<Bit, _>(
                    Bit::new(b),
                    &mut rng,
                )
            })
            .collect();

        let bits_by_party: [Vec<Rep3RingShare<Bit>>; 3] = std::array::from_fn(|pid| {
            bit_shares_per_item
                .iter()
                .map(|s| s[pid])
                .collect::<Vec<_>>()
        });

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
        assert_eq!(
            rec,
            G1Projective::zero(),
            "all-zero bits should give identity"
        );
    }

    #[test]
    fn dot_product_dapoints_all_ones() {
        let mut rng = test_rng();
        let n = 32usize;

        let qs: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bits_plain: Vec<bool> = vec![true; n];

        let bit_shares_per_item: Vec<Vec<Rep3RingShare<Bit>>> = bits_plain
            .iter()
            .map(|&b| {
                crate::protocols::rep3_ring::binary::generate_shares_rep3::<Bit, _>(
                    Bit::new(b),
                    &mut rng,
                )
            })
            .collect();

        let bits_by_party: [Vec<Rep3RingShare<Bit>>; 3] = std::array::from_fn(|pid| {
            bit_shares_per_item
                .iter()
                .map(|s| s[pid])
                .collect::<Vec<_>>()
        });

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
