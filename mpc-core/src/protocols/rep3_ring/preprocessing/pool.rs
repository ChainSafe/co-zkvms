use std::path::Path;

use super::backing_store;
use super::dabits::{DaBitBatch, LazyDaBits};
use super::edabits::{EdaBitsBatch, EdaBitsRingBatch, LazyEdaBits, LazyEdaBitsRing};
use super::rand_ohv::{LazyRandOhvs, RandOhvBatch, generate_rand_ohvs_lazy};
use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::network::Rep3RawFieldTransport;
use crate::protocols::rep3::network::{IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker};
use eyre::Ok;
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3_ring::ring::int_ring::IntRing2k;
use rand::RngCore;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rayon::prelude::*;
use tracing::info_span;

#[cfg(feature = "ring-msm")]
use mpc_types::protocols::rep3_ring::ring::u66::U66;

#[cfg(not(feature = "ring-msm"))]
use std::marker::PhantomData;

/// A pool of pre-generated edaBits, daBits, and (optionally) ring-MSM
/// preprocessing for batched conversions.
///
/// EdaBits are stored lazily via [`LazyEdaBits`] (O(1) storage for P0/P1).
/// DaBits are stored via [`LazyDaBits`] (Cheng23 Π₁ partial-lazy).
///
/// When the `ring-msm` feature is enabled, also stores:
/// - DaPoints via [`super::daPoint::LazyDaPoints`]
/// - Wrap masks via [`super::wrap_mask::LazyWrapMasks`]
/// - Ring edaBits (U66) via [`LazyEdaBitsRing`]
pub struct PreprocessingPool<F: PrimeField, C: ark_ec::CurveGroup = ark_bn254::G1Projective> {
    pub(crate) edabits_u8: LazyEdaBits<u8, F>,
    pub(crate) edabits_u16: LazyEdaBits<u16, F>,
    pub(crate) edabits_u32: LazyEdaBits<u32, F>,
    pub(crate) edabits_u64: LazyEdaBits<u64, F>,
    pub(crate) edabits_u128: LazyEdaBits<u128, F>,
    pub(crate) dabits: LazyDaBits<F>,
    pub(crate) rand_ohvs_u8_k4: LazyRandOhvs<F>,
    #[cfg(feature = "ring-msm")]
    pub(crate) dapoints: super::daPoint::LazyDaPoints<C>,
    #[cfg(feature = "ring-msm")]
    pub(crate) wrap_masks: super::wrap_mask::LazyWrapMasks,
    pub(crate) ring_edabits_u64: LazyEdaBitsRing<u64>,
    pub(crate) ring_edabits_u128: LazyEdaBitsRing<u128>,
    #[cfg(feature = "ring-msm")]
    pub(crate) ring_edabits_u66: LazyEdaBitsRing<U66>,
    #[cfg(not(feature = "ring-msm"))]
    _phantom: PhantomData<C>,
}

impl<F: PrimeField, C: ark_ec::CurveGroup> PreprocessingPool<F, C> {
    /// Create an empty pool.
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            edabits_u8: LazyEdaBits::empty(party_id),
            edabits_u16: LazyEdaBits::empty(party_id),
            edabits_u32: LazyEdaBits::empty(party_id),
            edabits_u64: LazyEdaBits::empty(party_id),
            edabits_u128: LazyEdaBits::empty(party_id),
            dabits: LazyDaBits::empty(party_id),
            rand_ohvs_u8_k4: LazyRandOhvs::empty(party_id),
            ring_edabits_u64: LazyEdaBitsRing::empty(party_id),
            ring_edabits_u128: LazyEdaBitsRing::empty(party_id),
            #[cfg(feature = "ring-msm")]
            dapoints: super::daPoint::LazyDaPoints::empty(party_id),
            #[cfg(feature = "ring-msm")]
            wrap_masks: super::wrap_mask::LazyWrapMasks::empty(party_id),
            #[cfg(feature = "ring-msm")]
            ring_edabits_u66: LazyEdaBitsRing::empty(party_id),
            #[cfg(not(feature = "ring-msm"))]
            _phantom: PhantomData,
        }
    }

    /// Create a pool from lazy edaBits sources and lazy daBits.
    pub fn new(
        party_id: PartyID,
        edabits_u8: LazyEdaBits<u8, F>,
        edabits_u16: LazyEdaBits<u16, F>,
        edabits_u32: LazyEdaBits<u32, F>,
        edabits_u64: LazyEdaBits<u64, F>,
        edabits_u128: LazyEdaBits<u128, F>,
        dabits: LazyDaBits<F>,
        rand_ohvs_u8_k4: LazyRandOhvs<F>,
    ) -> Self {
        Self {
            edabits_u8,
            edabits_u16,
            edabits_u32,
            edabits_u64,
            edabits_u128,
            dabits,
            rand_ohvs_u8_k4,
            ring_edabits_u64: LazyEdaBitsRing::empty(party_id),
            ring_edabits_u128: LazyEdaBitsRing::empty(party_id),
            #[cfg(feature = "ring-msm")]
            dapoints: super::daPoint::LazyDaPoints::empty(party_id),
            #[cfg(feature = "ring-msm")]
            wrap_masks: super::wrap_mask::LazyWrapMasks::empty(party_id),
            #[cfg(feature = "ring-msm")]
            ring_edabits_u66: LazyEdaBitsRing::empty(party_id),
            #[cfg(not(feature = "ring-msm"))]
            _phantom: PhantomData,
        }
    }

    // --- ring-msm gated methods ---

    /// Inject pre-generated daPoints into this pool.
    #[cfg(feature = "ring-msm")]
    pub fn set_dapoints(&mut self, dp: super::daPoint::LazyDaPoints<C>) {
        self.dapoints = dp;
    }

    /// Drain `n` daPoint tuples from the lazy source.
    #[cfg(feature = "ring-msm")]
    pub fn take_dapoints(&mut self, n: usize) -> eyre::Result<super::daPoint::DaPointsBatch<C>> {
        self.dapoints.take_batch(n)
    }

    #[cfg(feature = "ring-msm")]
    pub fn remaining_dapoints(&self) -> usize {
        self.dapoints.remaining()
    }

    /// Inject pre-generated lazy wrap masks into this pool.
    #[cfg(feature = "ring-msm")]
    pub fn set_wrap_masks(&mut self, wm: super::wrap_mask::LazyWrapMasks) {
        self.wrap_masks = wm;
    }

    /// Drain `n` wrap masks from the lazy source.
    #[cfg(feature = "ring-msm")]
    pub fn take_wrap_masks(&mut self, n: usize) -> eyre::Result<super::wrap_mask::WrapMaskBatch> {
        self.wrap_masks.take_batch(n)
    }

    #[cfg(feature = "ring-msm")]
    pub fn remaining_wrap_masks(&self) -> usize {
        self.wrap_masks.remaining()
    }

    /// Inject pre-generated lazy ring edaBits (U66) into this pool.
    #[cfg(feature = "ring-msm")]
    pub fn set_ring_edabits_u66(&mut self, eb: LazyEdaBitsRing<U66>) {
        self.ring_edabits_u66 = eb;
    }

    /// Drain `n` ring edaBits (U66) from the lazy source.
    #[cfg(feature = "ring-msm")]
    pub fn take_ring_edabits_u66(&mut self, n: usize) -> eyre::Result<EdaBitsRingBatch<U66>> {
        self.ring_edabits_u66.take_batch(n)
    }

    #[cfg(feature = "ring-msm")]
    pub fn remaining_ring_edabits_u66(&self) -> usize {
        self.ring_edabits_u66.remaining()
    }

    // --- ring edaBits (upcast B2A) ---

    pub fn set_ring_edabits_u64(&mut self, eb: LazyEdaBitsRing<u64>) {
        self.ring_edabits_u64 = eb;
    }

    pub fn set_ring_edabits_u128(&mut self, eb: LazyEdaBitsRing<u128>) {
        self.ring_edabits_u128 = eb;
    }

    pub fn remaining_ring_edabits_u64(&self) -> usize {
        self.ring_edabits_u64.remaining()
    }

    pub fn remaining_ring_edabits_u128(&self) -> usize {
        self.ring_edabits_u128.remaining()
    }

    /// Generic ring-edaBits drain, dispatched by `TypeId`.
    #[tracing::instrument(skip(self), level = "trace")]
    pub fn take_ring_edabits<T: IntRing2k>(
        &mut self,
        n: usize,
    ) -> eyre::Result<EdaBitsRingBatch<T>>
    where
        Standard: Distribution<T>,
    {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<u64>() {
            let v = self.ring_edabits_u64.take_batch(n)?;
            Ok(unsafe { std::mem::transmute::<EdaBitsRingBatch<u64>, EdaBitsRingBatch<T>>(v) })
        } else if tid == TypeId::of::<u128>() {
            let v = self.ring_edabits_u128.take_batch(n)?;
            Ok(unsafe { std::mem::transmute::<EdaBitsRingBatch<u128>, EdaBitsRingBatch<T>>(v) })
        } else {
            eyre::bail!(
                "PreprocessingPool::take_ring_edabits: unsupported ring type u{}",
                T::K
            );
        }
    }

    // --- always-available methods ---

    /// Drain `n` daBit tuples (Cheng23 Π₁) from the lazy source.
    #[tracing::instrument(skip(self))]
    pub fn take_dabits(&mut self, n: usize) -> eyre::Result<DaBitBatch<F>> {
        self.dabits.take_batch(n)
    }

    pub fn remaining_dabits(&self) -> usize {
        self.dabits.remaining()
    }

    #[tracing::instrument(skip(self))]
    pub fn take_rand_ohvs_u8_k4(&mut self, n: usize) -> eyre::Result<RandOhvBatch<F>> {
        self.rand_ohvs_u8_k4.take_batch(n)
    }

    pub fn remaining_rand_ohvs_u8_k4(&self) -> usize {
        self.rand_ohvs_u8_k4.remaining()
    }

    pub fn is_empty(&self) -> bool {
        self.edabits_u8.remaining() == 0
            && self.edabits_u16.remaining() == 0
            && self.edabits_u32.remaining() == 0
            && self.edabits_u64.remaining() == 0
            && self.edabits_u128.remaining() == 0
            && self.dabits.remaining() == 0
            && self.rand_ohvs_u8_k4.remaining() == 0
    }

    /// Return remaining counts for each edaBit ring type and daBits.
    pub fn remaining_counts(&self) -> ([usize; 5], usize) {
        (
            [
                self.edabits_u8.remaining(),
                self.edabits_u16.remaining(),
                self.edabits_u32.remaining(),
                self.edabits_u64.remaining(),
                self.edabits_u128.remaining(),
            ],
            self.dabits.remaining(),
        )
    }

    /// Reset all internal cursors to 0 in `reuse-preproc` mode.
    ///
    /// This makes the pool re-usable across multiple proof iterations in a single process.
    /// Not safe for production use (re-using preprocessing randomness breaks security),
    /// hence gated behind `reuse-preproc`.
    #[cfg(feature = "reuse-preproc")]
    pub fn reset_cursors_for_reuse(&mut self) {
        self.edabits_u8.reset_cursor_for_reuse();
        self.edabits_u16.reset_cursor_for_reuse();
        self.edabits_u32.reset_cursor_for_reuse();
        self.edabits_u64.reset_cursor_for_reuse();
        self.edabits_u128.reset_cursor_for_reuse();
        self.dabits.reset_cursor_for_reuse();
        self.rand_ohvs_u8_k4.reset_cursor_for_reuse();
        self.ring_edabits_u64.reset_cursor_for_reuse();
        self.ring_edabits_u128.reset_cursor_for_reuse();
    }

    /// Generic edaBits drain as flat batch, dispatched by `TypeId`.
    ///
    /// Returns an error if `T` is not one of u8, u16, u32, u64, u128, or if
    /// there are not enough edaBits remaining.
    #[tracing::instrument(skip(self), level = "trace")]
    pub fn take_edabits<T: IntRing2k>(&mut self, n: usize) -> eyre::Result<EdaBitsBatch<T, F>>
    where
        Standard: Distribution<T>,
    {
        use std::any::TypeId;
        // Safety: We transmute between EdaBitsBatch<concrete, F> and EdaBitsBatch<T, F>
        // only when TypeId confirms T == concrete. The struct layout is
        // identical for the same concrete type, so the transmute is a no-op.
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<u8>() {
            let v = self.edabits_u8.take_batch(n)?;
            // SAFETY: T == u8 confirmed by TypeId check.
            Ok(unsafe { std::mem::transmute::<EdaBitsBatch<u8, F>, EdaBitsBatch<T, F>>(v) })
        } else if tid == TypeId::of::<u16>() {
            let v = self.edabits_u16.take_batch(n)?;
            Ok(unsafe { std::mem::transmute::<EdaBitsBatch<u16, F>, EdaBitsBatch<T, F>>(v) })
        } else if tid == TypeId::of::<u32>() {
            let v = self.edabits_u32.take_batch(n)?;
            Ok(unsafe { std::mem::transmute::<EdaBitsBatch<u32, F>, EdaBitsBatch<T, F>>(v) })
        } else if tid == TypeId::of::<u64>() {
            let v = self.edabits_u64.take_batch(n)?;
            Ok(unsafe { std::mem::transmute::<EdaBitsBatch<u64, F>, EdaBitsBatch<T, F>>(v) })
        } else if tid == TypeId::of::<u128>() {
            let v = self.edabits_u128.take_batch(n)?;
            Ok(unsafe { std::mem::transmute::<EdaBitsBatch<u128, F>, EdaBitsBatch<T, F>>(v) })
        } else {
            eyre::bail!("EdaBitsPool::take_edabits: unsupported ring type u{}", T::K);
        }
    }

    /// Write all lazy sources to `dir` concurrently.
    ///
    /// All data+meta pairs are written in parallel via `thread::scope`.
    /// Each save writes the large data file first (page-cache, no fsync), then
    /// fsyncs the tiny 117-byte meta file as the durability barrier.
    #[tracing::instrument(skip_all, name = "Preprocessing::save")]
    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        std::thread::scope(|s| -> std::io::Result<()> {
            let h0 = s.spawn(|| self.edabits_u8.save(dir));
            let h1 = s.spawn(|| self.edabits_u16.save(dir));
            let h2 = s.spawn(|| self.edabits_u32.save(dir));
            let h3 = s.spawn(|| self.edabits_u64.save(dir));
            let h4 = s.spawn(|| self.edabits_u128.save(dir));
            let h5 = s.spawn(|| self.dabits.save(dir));
            let h6 = s.spawn(|| self.rand_ohvs_u8_k4.save(dir));
            let h_r64 = s.spawn(|| self.ring_edabits_u64.save(dir));
            let h_r128 = s.spawn(|| self.ring_edabits_u128.save(dir));
            #[cfg(feature = "ring-msm")]
            let h7 = s.spawn(|| self.wrap_masks.save(dir));
            #[cfg(feature = "ring-msm")]
            let h8 = s.spawn(|| self.ring_edabits_u66.save(dir));
            h0.join().unwrap()?;
            h1.join().unwrap()?;
            h2.join().unwrap()?;
            h3.join().unwrap()?;
            h4.join().unwrap()?;
            h5.join().unwrap()?;
            h6.join().unwrap()?;
            h_r64.join().unwrap()?;
            h_r128.join().unwrap()?;
            #[cfg(feature = "ring-msm")]
            h7.join().unwrap()?;
            #[cfg(feature = "ring-msm")]
            h8.join().unwrap()?;
            std::result::Result::Ok(())
        })
    }

    /// Load all lazy sources from `dir`.
    ///
    /// Note: daPoints are NOT persisted — they are regenerated via
    /// `set_dapoints()` after loading, since they depend on the SRS.
    pub fn load(dir: &std::path::Path, party_id: PartyID) -> std::io::Result<Self> {
        std::result::Result::Ok(Self {
            edabits_u8: LazyEdaBits::<u8, F>::load(dir, party_id)?,
            edabits_u16: LazyEdaBits::<u16, F>::load(dir, party_id)?,
            edabits_u32: LazyEdaBits::<u32, F>::load(dir, party_id)?,
            edabits_u64: LazyEdaBits::<u64, F>::load(dir, party_id)?,
            edabits_u128: LazyEdaBits::<u128, F>::load(dir, party_id)?,
            dabits: LazyDaBits::<F>::load(dir, party_id)?,
            rand_ohvs_u8_k4: LazyRandOhvs::<F>::load(dir, party_id)?,
            ring_edabits_u64: LazyEdaBitsRing::<u64>::load(dir, party_id)?,
            ring_edabits_u128: LazyEdaBitsRing::<u128>::load(dir, party_id)?,
            #[cfg(feature = "ring-msm")]
            dapoints: super::daPoint::LazyDaPoints::empty(party_id),
            #[cfg(feature = "ring-msm")]
            wrap_masks: super::wrap_mask::LazyWrapMasks::load(dir, party_id)?,
            #[cfg(feature = "ring-msm")]
            ring_edabits_u66: LazyEdaBitsRing::<U66>::load(dir, party_id)?,
            #[cfg(not(feature = "ring-msm"))]
            _phantom: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Pool generation / extension functions (moved from edabits.rs)
// ---------------------------------------------------------------------------

/// Batched preprocessing (in-memory): generate all edaBits + daBits in **2 network rounds**
/// instead of 7 sequential rounds (5 edaBit + 2 daBit).
///
/// Round 1: P0→P2 sends all edaBit α₂ + daBit α₂; P1→P2 sends daBit s₁₂.
/// Round 2: P2→P0 sends daBit s₂₀.
#[tracing::instrument(skip_all, name = "preprocess_pool_batched")]
fn preprocess_pool_batched<F: PrimeField, N: Rep3NetworkWorker>(
    counts: [usize; 5], // [u8, u16, u32, u64, u128]
    num_dabits: usize,
    num_rand_ohvs_u8_k4: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PreprocessingPool<F>> {
    use super::dabits;
    use crate::protocols::rep3::rngs::Rep3Rand;

    let party_id = io.party_id();
    let fb = usize::try_from(F::MODULUS_BIT_SIZE)
        .expect("u32 fits into usize")
        .div_ceil(8);

    // Phase 1: Fork 6 Rep3Rands and snapshot seeds (local, no communication).
    let mut rands: [Rep3Rand; 6] = std::array::from_fn(|_| io.main().rngs.rand.fork());
    let snaps: [_; 6] = std::array::from_fn(|i| rands[i].snapshot());

    // Helper: compute edaBit α₂ for P0 given a forked Rep3Rand.
    #[tracing::instrument(skip(eda_rand))]
    fn edabit_alpha2_p0<T: IntRing2k, Fp: PrimeField>(
        num: usize,
        eda_rand: &mut Rep3Rand,
        fb: usize,
    ) -> Vec<Fp>
    where
        Standard: Distribution<T>,
    {
        if num == 0 {
            return Vec::new();
        }
        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let stride = t_bytes + k * fb;
        let gamma_total = num * t_bytes;
        let (all1, g2) = {
            let mut a = vec![0u8; num * stride];
            let mut b = vec![0u8; gamma_total];
            rayon::join(
                || eda_rand.rng1.fill_bytes(&mut a),
                || eda_rand.rng2.fill_bytes(&mut b),
            );
            (a, b)
        };
        let mut out = vec![Fp::zero(); num * k];
        out.par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each(|(i, chunk)| {
                let base = i * stride;
                let g1v = T::from_le_bytes(&all1[base..base + t_bytes]);
                let g2v = T::from_le_bytes(&g2[i * t_bytes..(i + 1) * t_bytes]);
                let gamma = g1v ^ g2v;
                for j in 0..k {
                    let s = base + t_bytes + j * fb;
                    let alpha1 = dabits::parse_field::<Fp>(&all1, s);
                    let gbit = ((gamma >> j) & T::one()) == T::one();
                    chunk[j] = Fp::from(gbit as u64) - alpha1;
                }
            });
        out
    }

    // Phase 2: Compute data to send (local, parallelizable).
    match party_id {
        PartyID::ID0 => {
            // Compute all edaBit α₂ + daBit α₂.
            let ea0 = edabit_alpha2_p0::<u8, F>(counts[0], &mut rands[0], fb);
            let ea1 = edabit_alpha2_p0::<u16, F>(counts[1], &mut rands[1], fb);
            let ea2 = edabit_alpha2_p0::<u32, F>(counts[2], &mut rands[2], fb);
            let ea3 = edabit_alpha2_p0::<u64, F>(counts[3], &mut rands[3], fb);
            let ea4 = edabit_alpha2_p0::<u128, F>(counts[4], &mut rands[4], fb);

            // daBit α₂ (same logic as random_dabits_lazy P0 branch)
            // daBit α₂ — per-item interleaved layout in rng1:
            //   [g₀(1) a1₀(fb) r1₀(fb) | g₁(1) a1₁(fb) r1₁(fb) | ...]
            let da_alpha2 = if num_dabits > 0 {
                let r = &mut rands[5];
                let da_stride = 1 + 2 * fb;
                let slen1 = num_dabits * da_stride;
                let slen2 = num_dabits;
                let (s1, s2) = {
                    let mut a = vec![0u8; slen1];
                    let mut b = vec![0u8; slen2];
                    rayon::join(|| r.rng1.fill_bytes(&mut a), || r.rng2.fill_bytes(&mut b));
                    (a, b)
                };
                (0..num_dabits)
                    .into_par_iter()
                    .with_min_len(256)
                    .map(|i| {
                        let gbit = ((s1[i * da_stride] ^ s2[i]) & 1) != 0;
                        let alpha1: F = dabits::parse_field(&s1, i * da_stride + 1);
                        F::from(gbit as u64) - alpha1
                    })
                    .collect::<Vec<F>>()
            } else {
                Vec::new()
            };

            // Round 1: P0 → P2 — distribute across parallel fork channels
            let mut combined: Vec<F> = Vec::with_capacity(
                ea0.len() + ea1.len() + ea2.len() + ea3.len() + ea4.len() + da_alpha2.len(),
            );
            combined.extend_from_slice(&ea0);
            combined.extend_from_slice(&ea1);
            combined.extend_from_slice(&ea2);
            combined.extend_from_slice(&ea3);
            combined.extend_from_slice(&ea4);
            combined.extend_from_slice(&da_alpha2);
            io.par_chunks(
                combined.into_par_iter(),
                None,
                |chunk: Vec<F>, ctx| -> eyre::Result<Vec<()>> {
                    ctx.network.send_many(PartyID::ID2, &chunk)?;
                    Ok(vec![])
                },
            )?;

            // Round 2: P0 ← P2 receives daBit s₂₀ across fork channels
            let s20: Vec<F> = if num_dabits > 0 {
                let _span = info_span!("resv_s20").entered();
                io.par_chunks(
                    0..num_dabits,
                    None,
                    |_: Vec<usize>, ctx| -> eyre::Result<Vec<F>> {
                        Ok(ctx.network.recv_many::<F>(PartyID::ID2)?)
                    },
                )?
            } else {
                Vec::new()
            };

            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new(s1, p1, s2, p2, counts[0], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new(s1, p1, s2, p2, counts[1], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new(s1, p1, s2, p2, counts[2], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new(s1, p1, s2, p2, counts[3], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 = LazyEdaBits::<u128, F>::new(s1, p1, s2, p2, counts[4], Vec::new(), party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            let rand_ohvs_u8_k4 = generate_rand_ohvs_lazy(num_rand_ohvs_u8_k4, io.main())?;
            Ok(PreprocessingPool::new(
                party_id,
                e0,
                e1,
                e2,
                e3,
                e4,
                dabits::LazyDaBits::new(ds1, dp1, ds2, dp2, num_dabits, s20, party_id),
                rand_ohvs_u8_k4,
            ))
        }
        PartyID::ID1 => {
            // P1 only sends daBit s₁₂ to P2 (edaBits are local-only for P1).
            // Per-item interleaved layout in rng2 (P1↔P0 stream):
            //   [g₀(1) a1₀(fb) r1₀(fb) | g₁(1) a1₁(fb) r1₁(fb) | ...]
            if num_dabits > 0 {
                let r = &mut rands[5];
                let da_stride = 1 + 2 * fb;
                let slen2 = num_dabits * da_stride;
                let slen1 = num_dabits;
                let (s2, s1) = {
                    let mut a = vec![0u8; slen2];
                    let mut b = vec![0u8; slen1];
                    rayon::join(|| r.rng2.fill_bytes(&mut a), || r.rng1.fill_bytes(&mut b));
                    (a, b)
                };
                let theta_bytes = &s1;

                let s12: Vec<F> = (0..num_dabits)
                    .into_par_iter()
                    .with_min_len(256)
                    .map(|i| {
                        let theta = (theta_bytes[i] & 1) != 0;
                        let neg1_theta = if theta { -F::one() } else { F::one() };
                        let alpha1: F = dabits::parse_field(&s2, i * da_stride + 1);
                        let r1_val: F = dabits::parse_field(&s2, i * da_stride + 1 + fb);
                        neg1_theta * alpha1 - r1_val
                    })
                    .collect();
                io.network().send_many(PartyID::ID2, &s12)?;
            }

            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new(s1, p1, s2, p2, counts[0], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new(s1, p1, s2, p2, counts[1], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new(s1, p1, s2, p2, counts[2], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new(s1, p1, s2, p2, counts[3], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 = LazyEdaBits::<u128, F>::new(s1, p1, s2, p2, counts[4], Vec::new(), party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            let rand_ohvs_u8_k4 = generate_rand_ohvs_lazy(num_rand_ohvs_u8_k4, io.main())?;
            Ok(PreprocessingPool::new(
                party_id,
                e0,
                e1,
                e2,
                e3,
                e4,
                dabits::LazyDaBits::new(ds1, dp1, ds2, dp2, num_dabits, Vec::new(), party_id),
                rand_ohvs_u8_k4,
            ))
        }
        PartyID::ID2 => {
            // P2 advances daBit RNGs (to stay in sync) but doesn't use them.
            if num_dabits > 0 {
                let r = &mut rands[5];
                let slen2 = num_dabits;
                let slen1 = num_dabits;
                let mut s2 = vec![0u8; slen2];
                let mut s1 = vec![0u8; slen1];
                rayon::join(|| r.rng2.fill_bytes(&mut s2), || r.rng1.fill_bytes(&mut s1));
            }

            // Round 1: receive combined α₂ from P0 + s₁₂ from P1
            let total_eda = counts
                .iter()
                .enumerate()
                .map(|(i, &c)| c * [u8::K, u16::K, u32::K, u64::K, u128::K][i])
                .sum::<usize>();
            let total_recv = total_eda + num_dabits;

            let combined: Vec<F> = if total_recv > 0 {
                let _span = info_span!("resv_combined").entered();
                io.par_chunks(
                    0..total_recv,
                    None,
                    |_: Vec<usize>, ctx| -> eyre::Result<Vec<F>> {
                        Ok(ctx.network.recv_many::<F>(PartyID::ID0)?)
                    },
                )?
            } else {
                Vec::new()
            };
            debug_assert_eq!(combined.len(), total_recv);

            let s12_recv: Vec<F> = if num_dabits > 0 {
                let _span = info_span!("s12_recv").entered();
                io.network().recv_many(PartyID::ID1)?
            } else {
                Vec::new()
            };

            // Split combined into 5 edaBit α₂ slices + 1 daBit α₂
            let mut offset = 0;
            let mut eda_alphas: [Vec<F>; 5] = Default::default();
            for (idx, &c) in counts.iter().enumerate() {
                let k = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                let len = c * k;
                eda_alphas[idx] = combined[offset..offset + len].to_vec();
                offset += len;
            }
            let dabit_alpha2 = &combined[offset..offset + num_dabits];

            // Compute daBit s₂₀ and send to P0
            let dabit_stored = if num_dabits > 0 {
                // Re-derive theta from P2↔P1 seed (rng2 was advanced above, use snapshot).
                let (_, _, ds2_seed, ds2_pos) = snaps[5];
                let theta_buf = dabits::seek_and_generate(ds2_seed, ds2_pos, 0, num_dabits);

                let s20: Vec<F> = (0..num_dabits)
                    .into_par_iter()
                    .with_min_len(256)
                    .map(|i| {
                        let theta = (theta_buf[i] & 1) != 0;
                        let neg1_theta = if theta { -F::one() } else { F::one() };
                        neg1_theta * dabit_alpha2[i]
                    })
                    .collect();

                // Interleave s20 + s12 for LazyDaBits stored format (before send consumes s20)
                let mut stored = Vec::with_capacity(2 * num_dabits);
                for i in 0..num_dabits {
                    stored.push(s20[i]);
                    stored.push(s12_recv[i]);
                }

                // Round 2: P2 → P0 sends s₂₀ across fork channels
                io.par_chunks(
                    s20.into_par_iter(),
                    None,
                    |chunk: Vec<F>, ctx| -> eyre::Result<Vec<()>> {
                        ctx.network.send_many(PartyID::ID0, &chunk)?;
                        Ok(vec![])
                    },
                )?;

                stored
            } else {
                Vec::new()
            };

            let [a0, a1, a2, a3, a4] = eda_alphas;
            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new(s1, p1, s2, p2, counts[0], a0, party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new(s1, p1, s2, p2, counts[1], a1, party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new(s1, p1, s2, p2, counts[2], a2, party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new(s1, p1, s2, p2, counts[3], a3, party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 = LazyEdaBits::<u128, F>::new(s1, p1, s2, p2, counts[4], a4, party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            let rand_ohvs_u8_k4 = generate_rand_ohvs_lazy(num_rand_ohvs_u8_k4, io.main())?;
            Ok(PreprocessingPool::new(
                party_id,
                e0,
                e1,
                e2,
                e3,
                e4,
                dabits::LazyDaBits::new(ds1, dp1, ds2, dp2, num_dabits, dabit_stored, party_id),
                rand_ohvs_u8_k4,
            ))
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn preproc_max_msg_mb() -> usize {
    env_usize("PREPROC_MAX_MSG_MB", 2)
}

fn preproc_store_batch_mb() -> usize {
    env_usize("PREPROC_STORE_BATCH_MB", 16)
}

fn preproc_lanes() -> usize {
    env_usize("PREPROC_LANES", 8)
}

fn preproc_segment_mb() -> usize {
    env_usize("PREPROC_SEGMENT_MB", 64)
}

fn configured_transport_lanes() -> usize {
    std::env::var("MPC_QUIC_CONN_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| {
            std::env::var("NETWORK_FORKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v > 0)
        })
        .unwrap_or(8)
}

fn preproc_max_elems_per_msg<F: PrimeField>() -> usize {
    let mb = preproc_max_msg_mb();
    let bytes = mb.saturating_mul(1024 * 1024);
    let elem = std::mem::size_of::<F>();
    if elem == 0 {
        1
    } else {
        bytes.div_ceil(elem).max(1)
    }
}

fn preproc_store_batch_elems<F: PrimeField>() -> usize {
    let mb = preproc_store_batch_mb();
    let bytes = mb.saturating_mul(1024 * 1024);
    let elem = std::mem::size_of::<F>();
    if elem == 0 {
        1
    } else {
        bytes.div_ceil(elem).max(1)
    }
}

fn preproc_segment_elems<F: PrimeField>() -> usize {
    let mb = preproc_segment_mb();
    let bytes = mb.saturating_mul(1024 * 1024);
    let elem = std::mem::size_of::<F>();
    if elem == 0 {
        1
    } else {
        bytes.div_ceil(elem).max(1)
    }
}

#[derive(Clone, Copy)]
struct SegmentPlan {
    start_item: usize,
    items: usize,
}

fn build_segment_plans(total_items: usize, segment_items: usize) -> Vec<SegmentPlan> {
    if total_items == 0 {
        return Vec::new();
    }
    let segment_items = segment_items.max(1);
    let mut start_item = 0usize;
    let mut out = Vec::with_capacity(total_items.div_ceil(segment_items));
    while start_item < total_items {
        let items = (total_items - start_item).min(segment_items);
        out.push(SegmentPlan { start_item, items });
        start_item += items;
    }
    out
}

fn par_segment_plans<N, MapFn>(
    forks: &mut [IoContext<N>],
    plans: Vec<SegmentPlan>,
    map: MapFn,
) -> eyre::Result<()>
where
    N: Rep3NetworkWorker,
    MapFn: Fn(SegmentPlan, &mut IoContext<N>) -> eyre::Result<()> + Sync + Send,
{
    let plans_len = plans.len();
    if plans_len == 0 {
        return Ok(());
    }

    let fork_count = forks.len().min(plans_len);
    if fork_count == 0 {
        return Ok(());
    }
    if fork_count == 1 {
        for plan in plans {
            map(plan, &mut forks[0])?;
        }
        return Ok(());
    }

    (0..fork_count)
        .into_par_iter()
        .zip(forks[..fork_count].par_iter_mut())
        .map(|(fork_idx, mut ctx)| {
            let start = (fork_idx * plans_len) / fork_count;
            let end = ((fork_idx + 1) * plans_len) / fork_count;
            for plan in &plans[start..end] {
                map(*plan, &mut ctx)?;
            }
            std::result::Result::<(), eyre::Report>::Ok(())
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(())
}

enum RawFieldChunk<F> {
    Borrowed { bytes: Vec<u8>, elems: usize },
    Owned(Vec<F>),
}

impl<F: PrimeField> RawFieldChunk<F> {
    fn as_slice(&self) -> &[F] {
        match self {
            Self::Borrowed { bytes, elems } => {
                const { backing_store::assert_field_layout::<F>() };
                let ptr = bytes.as_ptr();
                let align = std::mem::align_of::<F>();
                debug_assert_eq!(bytes.len(), elems * std::mem::size_of::<F>());
                debug_assert_eq!((ptr as usize) % align.max(1), 0);
                unsafe { std::slice::from_raw_parts(ptr.cast::<F>(), *elems) }
            }
            Self::Owned(vec) => vec.as_slice(),
        }
    }
}

fn recv_field_chunk_ctx<F: PrimeField, N: Rep3Network + Rep3RawFieldTransport>(
    from: PartyID,
    elems: usize,
    io: &mut IoContext<N>,
) -> eyre::Result<RawFieldChunk<F>> {
    let bytes = io.network.recv_field_bytes_raw::<F>(from, elems)?;
    let align = std::mem::align_of::<F>();
    if (bytes.as_ptr() as usize) % align.max(1) == 0 {
        return Ok(RawFieldChunk::Borrowed { bytes, elems });
    }
    Ok(RawFieldChunk::Owned(field_vec_from_raw_bytes::<F>(&bytes)?))
}

fn recv_field_chunk<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    from: PartyID,
    elems: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<RawFieldChunk<F>> {
    recv_field_chunk_ctx(from, elems, io.main())
}

fn send_field_superchunk_ctx<F: PrimeField, N: Rep3Network + Rep3RawFieldTransport>(
    mut data: Vec<F>,
    target: PartyID,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
) -> eyre::Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let _span = tracing::trace_span!(
        "send_field_superchunk",
        party_id = ?io.id,
        target = ?target,
        elems = data.len(),
        bytes = data.len() * std::mem::size_of::<F>(),
    )
    .entered();
    if data.len() <= max_msg_elems {
        io.network.send_field_vec_raw(target, data)?;
        return Ok(());
    }

    while data.len() > max_msg_elems {
        let tail = data.split_off(max_msg_elems);
        io.network.send_field_vec_raw(target, data)?;
        data = tail;
    }
    io.network.send_field_vec_raw(target, data)?;
    Ok(())
}

fn send_field_superchunk<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    data: Vec<F>,
    target: PartyID,
    io: &mut IoContextPool<N>,
    max_msg_elems: usize,
) -> eyre::Result<()> {
    send_field_superchunk_ctx(data, target, io.main(), max_msg_elems)
}

fn recv_field_bytes_collect_ctx<F: PrimeField, N: Rep3Network + Rep3RawFieldTransport>(
    from: PartyID,
    expected_len: usize,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
) -> eyre::Result<Vec<u8>> {
    if expected_len == 0 {
        return Ok(Vec::new());
    }

    let num_msgs = expected_len.div_ceil(max_msg_elems.max(1));
    let _span = tracing::trace_span!(
        "recv_field_messages_collect",
        party_id = ?io.id,
        from = ?from,
        elems = expected_len,
        bytes = expected_len * std::mem::size_of::<F>(),
        num_msgs
    )
    .entered();

    let mut out = Vec::with_capacity(expected_len * std::mem::size_of::<F>());
    let mut received = 0usize;
    while received < expected_len {
        let chunk_elems = (expected_len - received).min(max_msg_elems.max(1));
        let recv = io.network.recv_field_bytes_raw::<F>(from, chunk_elems)?;
        out.extend_from_slice(&recv);
        received += chunk_elems;
    }
    debug_assert_eq!(received, expected_len);
    Ok(out)
}

fn recv_field_messages_collect_ctx<F: PrimeField, N: Rep3Network + Rep3RawFieldTransport>(
    from: PartyID,
    expected_len: usize,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
) -> eyre::Result<Vec<F>> {
    if expected_len == 0 {
        return Ok(Vec::new());
    }

    if expected_len <= max_msg_elems.max(1) {
        return Ok(io
            .network
            .recv_field_vec_raw_owned::<F>(from, expected_len)?);
    }

    let bytes = recv_field_bytes_collect_ctx::<F, _>(from, expected_len, io, max_msg_elems)?;
    field_vec_from_raw_bytes::<F>(&bytes)
}

fn recv_field_messages_collect<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    from: PartyID,
    expected_len: usize,
    io: &mut IoContextPool<N>,
    max_msg_elems: usize,
) -> eyre::Result<Vec<F>> {
    recv_field_messages_collect_ctx(from, expected_len, io.main(), max_msg_elems)
}

fn recv_field_messages_into_store<
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
>(
    from: PartyID,
    start_elem: usize,
    expected_len: usize,
    io: &mut IoContextPool<N>,
    max_msg_elems: usize,
    store: &mut backing_store::BackingStore<F>,
) -> eyre::Result<()> {
    if expected_len == 0 {
        return Ok(());
    }

    let elem_size = std::mem::size_of::<F>();
    let num_msgs = expected_len.div_ceil(max_msg_elems.max(1));
    let _span = tracing::trace_span!(
        "recv_field_messages_into_store",
        party_id = ?io.party_id(),
        from = ?from,
        start_elem,
        elems = expected_len,
        bytes = expected_len * elem_size,
        num_msgs
    )
    .entered();

    let buf_elems = max_msg_elems.max(1).min(expected_len);
    let mut buf = vec![0u8; buf_elems * elem_size];
    let mut write_elem_offset = start_elem;
    let mut received = 0usize;
    while received < expected_len {
        let chunk_elems = (expected_len - received).min(buf_elems);
        let chunk_bytes = chunk_elems * elem_size;
        io.network()
            .recv_field_bytes_bulk_into::<F>(from, &mut buf[..chunk_bytes])?;
        write_bytes_to_store(store, write_elem_offset, &buf[..chunk_bytes])?;
        write_elem_offset += chunk_elems;
        received += chunk_elems;
    }
    debug_assert_eq!(received, expected_len);

    Ok(())
}

fn receive_field_store_batch<F: PrimeField + Copy, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    from: PartyID,
    start_elem: usize,
    expected_len: usize,
    io: &mut IoContextPool<N>,
    max_msg_elems: usize,
    store: &mut backing_store::BackingStore<F>,
) -> eyre::Result<()> {
    recv_field_messages_into_store(from, start_elem, expected_len, io, max_msg_elems, store)
}

fn receive_field_writer_batch_ctx<F: PrimeField + Copy, N: Rep3Network + Rep3RawFieldTransport>(
    from: PartyID,
    start_elem: usize,
    expected_len: usize,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
    writer: &backing_store::FileBackedWriter<F>,
) -> eyre::Result<()> {
    recv_field_messages_into_writer_ctx(from, start_elem, expected_len, io, max_msg_elems, writer)
}

fn recv_field_messages_into_writer_ctx<
    F: PrimeField + Copy,
    N: Rep3Network + Rep3RawFieldTransport,
>(
    from: PartyID,
    start_elem: usize,
    expected_len: usize,
    io: &mut IoContext<N>,
    max_msg_elems: usize,
    writer: &backing_store::FileBackedWriter<F>,
) -> eyre::Result<()> {
    if expected_len == 0 {
        return Ok(());
    }

    let elem_size = std::mem::size_of::<F>();
    let num_msgs = expected_len.div_ceil(max_msg_elems.max(1));
    let _span = tracing::trace_span!(
        "recv_field_messages_into_store",
        party_id = ?io.id,
        from = ?from,
        start_elem,
        elems = expected_len,
        bytes = expected_len * elem_size,
        num_msgs
    )
    .entered();

    let buf_elems = max_msg_elems.max(1).min(expected_len);
    let mut buf = vec![0u8; buf_elems * elem_size];
    let mut write_elem_offset = start_elem;
    let mut received = 0usize;
    while received < expected_len {
        let chunk_elems = (expected_len - received).min(buf_elems);
        let chunk_bytes = chunk_elems * elem_size;
        io.network
            .recv_field_bytes_bulk_into::<F>(from, &mut buf[..chunk_bytes])?;
        let _span = tracing::trace_span!(
            "append_to_store",
            start_elem = write_elem_offset,
            elems = chunk_elems,
            bytes = chunk_bytes
        )
        .entered();
        writer.write_bytes_at(write_elem_offset, &buf[..chunk_bytes])?;
        write_elem_offset += chunk_elems;
        received += chunk_elems;
    }
    debug_assert_eq!(received, expected_len);
    Ok(())
}

fn write_to_store<F: PrimeField + Copy>(
    store: &mut backing_store::BackingStore<F>,
    start_elem: usize,
    data: &[F],
) -> eyre::Result<()> {
    let _span = tracing::trace_span!(
        "append_to_store",
        start_elem,
        elems = data.len(),
        bytes = data.len() * std::mem::size_of::<F>()
    )
    .entered();
    store.write_at(start_elem, data)?;
    Ok(())
}

fn write_bytes_to_store<F: PrimeField + Copy>(
    store: &mut backing_store::BackingStore<F>,
    start_elem: usize,
    bytes: &[u8],
) -> eyre::Result<()> {
    let elem_size = std::mem::size_of::<F>();
    debug_assert_eq!(bytes.len() % elem_size, 0);
    let _span = tracing::trace_span!(
        "append_to_store",
        start_elem,
        elems = bytes.len() / elem_size,
        bytes = bytes.len()
    )
    .entered();
    store.write_bytes_at(start_elem, bytes)?;
    Ok(())
}

fn field_vec_from_raw_bytes<F: PrimeField>(bytes: &[u8]) -> eyre::Result<Vec<F>> {
    const { backing_store::assert_field_layout::<F>() };
    let elem_size = std::mem::size_of::<F>();
    if bytes.len() % elem_size != 0 {
        eyre::bail!(
            "raw field payload length {} is not divisible by element size {} for {}",
            bytes.len(),
            elem_size,
            std::any::type_name::<F>()
        );
    }
    let elems = bytes.len() / elem_size;
    if elems == 0 {
        return Ok(Vec::new());
    }

    let mut out: Vec<std::mem::MaybeUninit<F>> = Vec::with_capacity(elems);
    unsafe { out.set_len(elems) };
    let out_bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, bytes.len()) };
    out_bytes.copy_from_slice(bytes);
    let out: Vec<F> = unsafe { std::mem::transmute(out) };
    Ok(out)
}

fn read_file_backed_range<F: PrimeField + Copy>(
    file: &std::fs::File,
    start: usize,
    end: usize,
) -> std::io::Result<Vec<F>> {
    const { backing_store::assert_field_layout::<F>() };
    if start >= end {
        return std::result::Result::Ok(Vec::new());
    }
    let count = end - start;
    let elem_size = std::mem::size_of::<F>();
    let byte_offset = start * elem_size;
    let byte_len = count * elem_size;

    // SAFETY: same contract as BackingStore::read_file_backed_range.
    let mut out: Vec<std::mem::MaybeUninit<F>> = Vec::with_capacity(count);
    unsafe { out.set_len(count) };
    let out_bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };

    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(out_bytes, byte_offset as u64)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(out_bytes, byte_offset as u64)?;
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file.try_clone()?;
        f.seek(SeekFrom::Start(byte_offset as u64))?;
        f.read_exact(out_bytes)?;
    }

    let out: Vec<F> = unsafe { std::mem::transmute(out) };
    std::result::Result::Ok(out)
}

/// Batched preprocessing into a persistence directory, using bounded-memory chunking.
///
/// This is intended for large-trace benchmark harnesses. It avoids allocating
/// monolithic `Vec<F>` buffers for all α₂ values at once, and for P2 it writes
/// α₂/stored data directly to file-backed stores under `dir`.
#[tracing::instrument(skip_all, name = "preprocess_pool")]
fn preprocess_pool_base<F, N>(
    dir: &Path,
    counts: [usize; 5], // [u8, u16, u32, u64, u128]
    num_dabits: usize,
    num_rand_ohvs_u8_k4: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PreprocessingPool<F>>
where
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
{
    use super::dabits;
    use crate::protocols::rep3::rngs::Rep3Rand;

    std::fs::create_dir_all(dir)?;

    let party_id = io.party_id();
    let fb = usize::try_from(F::MODULUS_BIT_SIZE)
        .expect("u32 fits into usize")
        .div_ceil(8);
    let max_msg_elems = preproc_max_elems_per_msg::<F>();
    let store_batch_elems = preproc_store_batch_elems::<F>();
    let segment_elems = preproc_segment_elems::<F>();
    let active_edabit_lanes = preproc_lanes()
        .max(1)
        .min(io.max_forks().max(1))
        .min(configured_transport_lanes().max(1));
    let active_dabit_lanes = 1; // daBits benefit more from intra-chunk rayon parallelism than from lane splitting

    // Phase 1: Fork 6 Rep3Rands and snapshot seeds (local, no communication).
    let mut rands: [Rep3Rand; 6] = std::array::from_fn(|_| io.main().rngs.rand.fork());
    let snaps: [_; 6] = std::array::from_fn(|i| rands[i].snapshot());

    // Helper: compute edaBit α2 for a contiguous item range from seed snapshots.
    fn edabit_alpha2_seed_chunk<T: IntRing2k, Fp: PrimeField>(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        start_item: usize,
        num: usize,
        fb: usize,
        parallel: bool,
    ) -> Vec<Fp>
    where
        Standard: Distribution<T>,
    {
        if num == 0 {
            return Vec::new();
        }
        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let stride = t_bytes + k * fb;
        let _span = tracing::trace_span!(
            "edabit_alpha2_p0_chunk",
            ring_bits = T::K,
            items = num,
            start_item,
            elems = num * k,
            bytes = num * k * std::mem::size_of::<Fp>()
        )
        .entered();
        let all1 = dabits::seek_and_generate(seed1, pos1, start_item * stride, num * stride);
        let g2 = dabits::seek_and_generate(seed2, pos2, start_item * t_bytes, num * t_bytes);
        let mut out = vec![Fp::zero(); num * k];
        let fill_chunk = |(i, chunk): (usize, &mut [Fp])| {
            let base = i * stride;
            let g1v = T::from_le_bytes(&all1[base..base + t_bytes]);
            let g2v = T::from_le_bytes(&g2[i * t_bytes..(i + 1) * t_bytes]);
            let gamma = g1v ^ g2v;
            for j in 0..k {
                let s = base + t_bytes + j * fb;
                let alpha1 = dabits::parse_field::<Fp>(&all1, s);
                let gbit = ((gamma >> j) & T::one()) == T::one();
                chunk[j] = Fp::from(gbit as u64) - alpha1;
            }
        };
        if parallel {
            out.par_chunks_mut(k)
                .enumerate()
                .with_min_len(256)
                .for_each(fill_chunk);
        } else {
            out.chunks_mut(k).enumerate().for_each(fill_chunk);
        }
        out
    }

    fn dabit_alpha2_seed_chunk<Fp: PrimeField>(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        start_item: usize,
        num: usize,
        fb: usize,
        parallel: bool,
    ) -> Vec<Fp> {
        if num == 0 {
            return Vec::new();
        }
        let da_stride = 1 + 2 * fb;
        let s1 = dabits::seek_and_generate(seed1, pos1, start_item * da_stride, num * da_stride);
        let s2 = dabits::seek_and_generate(seed2, pos2, start_item, num);
        let compute = |i: usize| {
            let gbit = ((s1[i * da_stride] ^ s2[i]) & 1) != 0;
            let alpha1: Fp = dabits::parse_field(&s1, i * da_stride + 1);
            Fp::from(gbit as u64) - alpha1
        };
        if parallel {
            (0..num)
                .into_par_iter()
                .with_min_len(256)
                .map(compute)
                .collect()
        } else {
            (0..num).map(compute).collect()
        }
    }

    fn dabit_s12_seed_chunk<Fp: PrimeField>(
        theta_seed: [u8; crate::SEED_SIZE],
        theta_pos: u128,
        interleaved_seed: [u8; crate::SEED_SIZE],
        interleaved_pos: u128,
        start_item: usize,
        num: usize,
        fb: usize,
        parallel: bool,
    ) -> Vec<Fp> {
        if num == 0 {
            return Vec::new();
        }
        let da_stride = 1 + 2 * fb;
        let theta_bytes = dabits::seek_and_generate(theta_seed, theta_pos, start_item, num);
        let s2 = dabits::seek_and_generate(
            interleaved_seed,
            interleaved_pos,
            start_item * da_stride,
            num * da_stride,
        );
        let compute = |i: usize| {
            let theta = (theta_bytes[i] & 1) != 0;
            let neg1_theta = if theta { -Fp::one() } else { Fp::one() };
            let alpha1: Fp = dabits::parse_field(&s2, i * da_stride + 1);
            let r1_val: Fp = dabits::parse_field(&s2, i * da_stride + 1 + fb);
            neg1_theta * alpha1 - r1_val
        };
        if parallel {
            (0..num)
                .into_par_iter()
                .with_min_len(256)
                .map(compute)
                .collect()
        } else {
            (0..num).map(compute).collect()
        }
    }

    fn dabit_s20_from_theta_seed<Fp: PrimeField>(
        theta_seed: [u8; crate::SEED_SIZE],
        theta_pos: u128,
        start_item: usize,
        alpha2: &[Fp],
        parallel: bool,
    ) -> Vec<Fp> {
        if alpha2.is_empty() {
            return Vec::new();
        }
        let theta_buf = dabits::seek_and_generate(theta_seed, theta_pos, start_item, alpha2.len());
        let compute = |(alpha2_i, theta_b): (&Fp, &u8)| {
            let theta = (theta_b & 1) != 0;
            let neg1_theta = if theta { -Fp::one() } else { Fp::one() };
            neg1_theta * *alpha2_i
        };
        if parallel {
            alpha2
                .par_iter()
                .zip(theta_buf.par_iter())
                .map(compute)
                .collect()
        } else {
            alpha2.iter().zip(theta_buf.iter()).map(compute).collect()
        }
    }

    match party_id {
        PartyID::ID0 => {
            // Round 1: P0 → P2 sends edaBit α₂ streams per type, then daBit α₂.
            let mut send_edabit_type = |idx: usize, num: usize| {
                if num == 0 {
                    return Ok(());
                }
                let _span = info_span!("edabits_send_alphas", k = idx, n = num).entered();
                let k = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                let seeds = snaps[idx];
                let max_items_per_msg = (max_msg_elems / k).max(1);
                let segment_items = (segment_elems / k).max(max_items_per_msg);
                let plans = build_segment_plans(num, segment_items);
                let chunk_parallel = active_edabit_lanes == 1;
                if active_edabit_lanes > 1 && plans.len() > 1 {
                    par_segment_plans(io.forks(active_edabit_lanes), plans, |plan, ctx| {
                        let mut done = 0usize;
                        while done < plan.items {
                            let items = (plan.items - done).min(max_items_per_msg);
                            let start_item = plan.start_item + done;
                            let alpha2 = match idx {
                                0 => edabit_alpha2_seed_chunk::<u8, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                1 => edabit_alpha2_seed_chunk::<u16, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                2 => edabit_alpha2_seed_chunk::<u32, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                3 => edabit_alpha2_seed_chunk::<u64, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                4 => edabit_alpha2_seed_chunk::<u128, F>(
                                    seeds.0,
                                    seeds.1,
                                    seeds.2,
                                    seeds.3,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                ),
                                _ => unreachable!(),
                            };
                            debug_assert_eq!(alpha2.len(), items * k);
                            send_field_superchunk_ctx(alpha2, PartyID::ID2, ctx, max_msg_elems)?;
                            done += items;
                        }
                        Ok(())
                    })?;
                } else {
                    let mut done = 0usize;
                    while done < num {
                        let items = (num - done).min(max_items_per_msg);
                        let alpha2 = match idx {
                            0 => edabit_alpha2_seed_chunk::<u8, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            1 => edabit_alpha2_seed_chunk::<u16, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            2 => edabit_alpha2_seed_chunk::<u32, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            3 => edabit_alpha2_seed_chunk::<u64, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            4 => edabit_alpha2_seed_chunk::<u128, F>(
                                seeds.0,
                                seeds.1,
                                seeds.2,
                                seeds.3,
                                done,
                                items,
                                fb,
                                chunk_parallel,
                            ),
                            _ => unreachable!(),
                        };
                        debug_assert_eq!(alpha2.len(), items * k);
                        send_field_superchunk(alpha2, PartyID::ID2, io, max_msg_elems)?;
                        done += items;
                    }
                }
                Ok(())
            };

            // Collect active (non-zero) edaBit types.
            let active_types: Vec<(usize, usize)> = counts
                .iter()
                .copied()
                .enumerate()
                .filter(|(_i, c)| *c > 0)
                .collect();

            if active_types.len() > 1 && active_edabit_lanes >= active_types.len() {
                // Pipeline: each edaBit type runs on its own fork in parallel.
                let forks = io.forks(active_types.len());
                active_types
                    .into_par_iter()
                    .zip(forks.par_iter_mut())
                    .try_for_each(|((idx, num), ctx)| {
                        let _span = info_span!("edabits_send_alphas", k = idx, n = num).entered();
                        let k: usize = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                        let seeds = snaps[idx];
                        let max_items_per_msg: usize = (max_msg_elems / k).max(1);
                        let mut done = 0usize;
                        while done < num {
                            let items = (num - done).min(max_items_per_msg);
                            let alpha2 = match idx {
                                0 => edabit_alpha2_seed_chunk::<u8, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                1 => edabit_alpha2_seed_chunk::<u16, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                2 => edabit_alpha2_seed_chunk::<u32, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                3 => edabit_alpha2_seed_chunk::<u64, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                4 => edabit_alpha2_seed_chunk::<u128, F>(
                                    seeds.0, seeds.1, seeds.2, seeds.3, done, items, fb, true,
                                ),
                                _ => unreachable!(),
                            };
                            debug_assert_eq!(alpha2.len(), items * k);
                            send_field_superchunk_ctx(alpha2, PartyID::ID2, ctx, max_msg_elems)?;
                            done += items;
                        }
                        eyre::Result::<()>::Ok(())
                    })?;
            } else {
                send_edabit_type(0, counts[0])?;
                send_edabit_type(1, counts[1])?;
                send_edabit_type(2, counts[2])?;
                send_edabit_type(3, counts[3])?;
                send_edabit_type(4, counts[4])?;
            }
            let _span =
                tracing::trace_span!("edabits_to_dabits_sync", party_id = ?party_id).entered();
            io.sync_with_parties()?;

            let (ds1, dp1, ds2, dp2) = snaps[5];
            let dabits_store_path = dir.join("dabits.stored");
            let mut dabits_store = backing_store::BackingStore::create_file_backed_sized(
                &dabits_store_path,
                num_dabits,
            )?;
            if num_dabits > 0 {
                let max_items_per_msg = max_msg_elems.max(1);
                let segment_items = segment_elems.max(max_items_per_msg);
                let _span = info_span!("dabits_stream_chunks", n = num_dabits).entered();
                if let Some(writer) = dabits_store.writer()? {
                    let plans = build_segment_plans(num_dabits, segment_items);
                    let chunk_parallel = active_dabit_lanes == 1;
                    if active_dabit_lanes > 1 && plans.len() > 1 {
                        par_segment_plans(io.forks(active_dabit_lanes), plans, |plan, ctx| {
                            let mut offset = 0usize;
                            while offset < plan.items {
                                let items = (plan.items - offset).min(max_items_per_msg);
                                let start_item = plan.start_item + offset;
                                let da_alpha2 = dabit_alpha2_seed_chunk::<F>(
                                    ds1,
                                    dp1,
                                    ds2,
                                    dp2,
                                    start_item,
                                    items,
                                    fb,
                                    chunk_parallel,
                                );
                                send_field_superchunk_ctx(
                                    da_alpha2,
                                    PartyID::ID2,
                                    ctx,
                                    max_msg_elems,
                                )?;
                                recv_field_messages_into_writer_ctx::<F, _>(
                                    PartyID::ID2,
                                    start_item,
                                    items,
                                    ctx,
                                    max_msg_elems,
                                    &writer,
                                )?;
                                offset += items;
                            }
                            Ok(())
                        })?;
                    } else {
                        let mut offset = 0usize;
                        while offset < num_dabits {
                            let items = (num_dabits - offset).min(max_items_per_msg);
                            let da_alpha2 = dabit_alpha2_seed_chunk::<F>(
                                ds1,
                                dp1,
                                ds2,
                                dp2,
                                offset,
                                items,
                                fb,
                                chunk_parallel,
                            );
                            send_field_superchunk(da_alpha2, PartyID::ID2, io, max_msg_elems)?;

                            recv_field_messages_into_store::<F, _>(
                                PartyID::ID2,
                                offset,
                                items,
                                io,
                                max_msg_elems,
                                &mut dabits_store,
                            )?;
                            offset += items;
                        }
                    }
                }
            }

            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new(s1, p1, s2, p2, counts[0], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new(s1, p1, s2, p2, counts[1], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new(s1, p1, s2, p2, counts[2], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new(s1, p1, s2, p2, counts[3], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 = LazyEdaBits::<u128, F>::new(s1, p1, s2, p2, counts[4], Vec::new(), party_id);

            let d =
                LazyDaBits::new_with_store(ds1, dp1, ds2, dp2, num_dabits, dabits_store, party_id);
            let rand_ohvs_u8_k4 = generate_rand_ohvs_lazy(num_rand_ohvs_u8_k4, io.main())?;
            let pool = PreprocessingPool::new(party_id, e0, e1, e2, e3, e4, d, rand_ohvs_u8_k4);
            pool.save(dir)?;
            Ok(pool)
        }
        PartyID::ID1 => {
            // P1 only sends daBit s₁₂ to P2 (edaBits are local-only for P1).
            let _span =
                tracing::trace_span!("edabits_to_dabits_sync", party_id = ?party_id).entered();
            io.sync_with_parties()?;
            if num_dabits > 0 {
                let _span = info_span!("dabits_send_thetas", n = num_dabits).entered();
                let (theta_seed, theta_pos, interleaved_seed, interleaved_pos) = snaps[5];
                let max_items_per_msg = max_msg_elems.max(1);
                let segment_items = segment_elems.max(max_items_per_msg);
                let plans = build_segment_plans(num_dabits, segment_items);
                let chunk_parallel = active_dabit_lanes == 1;
                if active_dabit_lanes > 1 && plans.len() > 1 {
                    par_segment_plans(io.forks(active_dabit_lanes), plans, |plan, ctx| {
                        let mut offset = 0usize;
                        while offset < plan.items {
                            let items = (plan.items - offset).min(max_items_per_msg);
                            let start_item = plan.start_item + offset;
                            let s12 = dabit_s12_seed_chunk::<F>(
                                theta_seed,
                                theta_pos,
                                interleaved_seed,
                                interleaved_pos,
                                start_item,
                                items,
                                fb,
                                chunk_parallel,
                            );
                            send_field_superchunk_ctx(s12, PartyID::ID2, ctx, max_msg_elems)?;
                            offset += items;
                        }
                        Ok(())
                    })?;
                } else {
                    let mut offset = 0usize;
                    while offset < num_dabits {
                        let items = (num_dabits - offset).min(max_items_per_msg);
                        let s12 = dabit_s12_seed_chunk::<F>(
                            theta_seed,
                            theta_pos,
                            interleaved_seed,
                            interleaved_pos,
                            offset,
                            items,
                            fb,
                            chunk_parallel,
                        );
                        send_field_superchunk(s12, PartyID::ID2, io, max_msg_elems)?;
                        offset += items;
                    }
                }
            }

            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new(s1, p1, s2, p2, counts[0], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new(s1, p1, s2, p2, counts[1], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new(s1, p1, s2, p2, counts[2], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new(s1, p1, s2, p2, counts[3], Vec::new(), party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 = LazyEdaBits::<u128, F>::new(s1, p1, s2, p2, counts[4], Vec::new(), party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            let d = LazyDaBits::new(ds1, dp1, ds2, dp2, num_dabits, Vec::new(), party_id);
            let rand_ohvs_u8_k4 = generate_rand_ohvs_lazy(num_rand_ohvs_u8_k4, io.main())?;
            let pool = PreprocessingPool::new(party_id, e0, e1, e2, e3, e4, d, rand_ohvs_u8_k4);
            pool.save(dir)?;
            Ok(pool)
        }
        PartyID::ID2 => {
            // Round 1: receive edaBit α₂ streams from P0 and append to file-backed stores.
            let mut eda_stores_vec: Vec<backing_store::BackingStore<F>> = Vec::with_capacity(5);
            for idx in 0..5 {
                let suffix = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                let path = dir.join(format!("edabits_{suffix}.alpha2"));
                let total = counts[idx] * suffix;
                eda_stores_vec.push(backing_store::BackingStore::create_file_backed_sized(
                    &path, total,
                )?);
            }
            let mut eda_stores_iter = eda_stores_vec.into_iter();
            let mut eda_stores: [backing_store::BackingStore<F>; 5] = [
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
                eda_stores_iter
                    .next()
                    .expect("eda store vec must have len=5"),
            ];
            debug_assert!(eda_stores_iter.next().is_none());

            // Collect active (non-zero) edaBit types with their writers.
            let active_types: Vec<(usize, usize)> = counts
                .iter()
                .copied()
                .enumerate()
                .filter(|(_i, c)| *c > 0)
                .collect();

            if active_types.len() > 1 && active_edabit_lanes >= active_types.len() {
                // Pipeline: each edaBit type receives on its own fork in parallel.
                let writers: Vec<_> = active_types
                    .iter()
                    .map(|(idx, _)| eda_stores[*idx].writer())
                    .collect::<std::io::Result<Vec<_>>>()?;
                let forks = io.forks(active_types.len());
                active_types
                    .into_par_iter()
                    .zip(writers.into_par_iter())
                    .zip(forks.par_iter_mut())
                    .try_for_each(|(((idx, c), writer), ctx)| {
                        let _span = info_span!("edabits_resv_store", k = idx, n = c).entered();
                        let k: usize = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                        let max_items_per_msg: usize = (max_msg_elems / k).max(1);
                        let store_batch_items: usize =
                            (store_batch_elems / k).max(max_items_per_msg);
                        if let Some(writer) = writer {
                            let mut done = 0usize;
                            while done < c {
                                let items = (c - done).min(store_batch_items);
                                recv_field_messages_into_writer_ctx::<F, _>(
                                    PartyID::ID0,
                                    done * k,
                                    items * k,
                                    ctx,
                                    max_msg_elems,
                                    &writer,
                                )?;
                                done += items;
                            }
                        }
                        eyre::Result::<()>::Ok(())
                    })?;
            } else {
                for (idx, &c) in counts.iter().enumerate() {
                    let _span = info_span!("edabits_resv_store", k = idx, n = c).entered();
                    let k = [u8::K, u16::K, u32::K, u64::K, u128::K][idx];
                    let total_elems = c * k;
                    if total_elems == 0 {
                        continue;
                    }

                    let max_items_per_msg = (max_msg_elems / k).max(1);
                    let store_batch_items = (store_batch_elems / k).max(max_items_per_msg);
                    let segment_items = (segment_elems / k).max(max_items_per_msg);
                    let plans = build_segment_plans(c, segment_items);
                    if let Some(writer) = eda_stores[idx].writer()? {
                        if active_edabit_lanes > 1 && plans.len() > 1 {
                            par_segment_plans(
                                io.forks(active_edabit_lanes),
                                plans,
                                |plan, ctx| {
                                    let mut offset = 0usize;
                                    while offset < plan.items {
                                        let items = (plan.items - offset).min(store_batch_items);
                                        let start_item = plan.start_item + offset;
                                        recv_field_messages_into_writer_ctx::<F, _>(
                                            PartyID::ID0,
                                            start_item * k,
                                            items * k,
                                            ctx,
                                            max_msg_elems,
                                            &writer,
                                        )?;
                                        offset += items;
                                    }
                                    Ok(())
                                },
                            )?;
                        } else {
                            let mut write_elem_offset = 0usize;
                            let mut remaining = c;
                            while remaining > 0 {
                                let items = remaining.min(store_batch_items);
                                let expected = items * k;
                                recv_field_messages_into_store::<F, _>(
                                    PartyID::ID0,
                                    write_elem_offset,
                                    expected,
                                    io,
                                    max_msg_elems,
                                    &mut eda_stores[idx],
                                )?;
                                write_elem_offset += expected;
                                remaining -= items;
                            }
                        }
                    }
                }
            }
            let _span =
                tracing::trace_span!("edabits_to_dabits_sync", party_id = ?party_id).entered();
            io.sync_with_parties()?;

            // daBits: receive alpha2 from P0 and s12 from P1 in matching chunks, compute s20,
            // append local stored data, and forward s20 to P0 immediately.
            let dabits_store_path = dir.join("dabits.stored");
            let mut dabits_store = backing_store::BackingStore::create_file_backed_sized(
                &dabits_store_path,
                num_dabits.saturating_mul(2),
            )?;

            if num_dabits > 0 {
                let _span = info_span!("dabits_resv_store", n = num_dabits).entered();
                let (_, _, ds2_seed, ds2_pos) = snaps[5];
                let max_items_per_msg = max_msg_elems.max(1);
                let segment_items = segment_elems.max(max_items_per_msg);
                let plans = build_segment_plans(num_dabits, segment_items);
                if let Some(writer) = dabits_store.writer()? {
                    let chunk_parallel = active_dabit_lanes == 1;
                    if active_dabit_lanes > 1 && plans.len() > 1 {
                        par_segment_plans(io.forks(active_dabit_lanes), plans, |plan, ctx| {
                            let mut buffered_s20 =
                                Vec::with_capacity(plan.items.min(store_batch_elems));
                            let mut buffered_s12 =
                                Vec::with_capacity(plan.items.min(store_batch_elems));
                            let mut pending_pair_start = plan.start_item;
                            let mut offset = 0usize;
                            while offset < plan.items {
                                let items = (plan.items - offset).min(max_items_per_msg);
                                let start_item = plan.start_item + offset;
                                let alpha2 =
                                    recv_field_chunk_ctx::<F, _>(PartyID::ID0, items, ctx)?;
                                let s12 = recv_field_chunk_ctx::<F, _>(PartyID::ID1, items, ctx)?;
                                let _span_chunk = tracing::trace_span!(
                                    "dabit_s20_forward_chunk",
                                    items,
                                    bytes = items * std::mem::size_of::<F>()
                                )
                                .entered();
                                let s20 = dabit_s20_from_theta_seed::<F>(
                                    ds2_seed,
                                    ds2_pos,
                                    start_item,
                                    alpha2.as_slice(),
                                    chunk_parallel,
                                );
                                if buffered_s20.is_empty() {
                                    pending_pair_start = start_item;
                                }
                                buffered_s20.extend_from_slice(&s20);
                                buffered_s12.extend_from_slice(s12.as_slice());
                                if buffered_s20.len() >= store_batch_elems {
                                    let _span = tracing::trace_span!(
                                        "append_to_store",
                                        start_elem = pending_pair_start * 2,
                                        elems = buffered_s20.len() * 2,
                                        bytes = buffered_s20.len() * 2 * std::mem::size_of::<F>()
                                    )
                                    .entered();
                                    writer.write_interleaved_at(
                                        pending_pair_start,
                                        &buffered_s20,
                                        &buffered_s12,
                                    )?;
                                    buffered_s20.clear();
                                    buffered_s12.clear();
                                }
                                send_field_superchunk_ctx(s20, PartyID::ID0, ctx, max_msg_elems)?;
                                offset += items;
                            }
                            if !buffered_s20.is_empty() {
                                let _span = tracing::trace_span!(
                                    "append_to_store",
                                    start_elem = pending_pair_start * 2,
                                    elems = buffered_s20.len() * 2,
                                    bytes = buffered_s20.len() * 2 * std::mem::size_of::<F>()
                                )
                                .entered();
                                writer.write_interleaved_at(
                                    pending_pair_start,
                                    &buffered_s20,
                                    &buffered_s12,
                                )?;
                            }
                            Ok(())
                        })?;
                    } else {
                        let mut offset = 0usize;
                        while offset < num_dabits {
                            let items = (num_dabits - offset).min(max_items_per_msg);

                            let alpha2 = recv_field_chunk::<F, _>(PartyID::ID0, items, io)?;
                            let s12 = recv_field_chunk::<F, _>(PartyID::ID1, items, io)?;

                            let _span_chunk = tracing::trace_span!(
                                "dabit_s20_forward_chunk",
                                items,
                                bytes = items * std::mem::size_of::<F>()
                            )
                            .entered();
                            let s20 = dabit_s20_from_theta_seed::<F>(
                                ds2_seed,
                                ds2_pos,
                                offset,
                                alpha2.as_slice(),
                                chunk_parallel,
                            );

                            dabits_store.write_interleaved_at(offset, &s20, s12.as_slice())?;
                            send_field_superchunk(s20, PartyID::ID0, io, max_msg_elems)?;

                            offset += items;
                        }
                    }
                }
            }

            let [a0, a1, a2, a3, a4] = eda_stores;
            let mk = |i: usize| snaps[i];
            let (s1, p1, s2, p2) = mk(0);
            let e0 = LazyEdaBits::<u8, F>::new_with_store(s1, p1, s2, p2, counts[0], a0, party_id);
            let (s1, p1, s2, p2) = mk(1);
            let e1 = LazyEdaBits::<u16, F>::new_with_store(s1, p1, s2, p2, counts[1], a1, party_id);
            let (s1, p1, s2, p2) = mk(2);
            let e2 = LazyEdaBits::<u32, F>::new_with_store(s1, p1, s2, p2, counts[2], a2, party_id);
            let (s1, p1, s2, p2) = mk(3);
            let e3 = LazyEdaBits::<u64, F>::new_with_store(s1, p1, s2, p2, counts[3], a3, party_id);
            let (s1, p1, s2, p2) = mk(4);
            let e4 =
                LazyEdaBits::<u128, F>::new_with_store(s1, p1, s2, p2, counts[4], a4, party_id);
            let (ds1, dp1, ds2, dp2) = snaps[5];
            let d =
                LazyDaBits::new_with_store(ds1, dp1, ds2, dp2, num_dabits, dabits_store, party_id);

            let rand_ohvs_u8_k4 = generate_rand_ohvs_lazy(num_rand_ohvs_u8_k4, io.main())?;
            let pool = PreprocessingPool::new(party_id, e0, e1, e2, e3, e4, d, rand_ohvs_u8_k4);
            pool.save(dir)?;
            Ok(pool)
        }
    }
}

/// Extend an existing preprocessing pool with additional items.
///
/// For each edaBit type and daBits where `deficit > 0`, generates additional
/// items by seeking into the existing RNG streams past the already-generated
/// region. Same 2-round communication pattern as `preprocess_pool_batched`.
///
/// **Communication:** P0→P2 (combined alpha2 for deficit items), P1→P2 (daBit s₁₂),
/// P2→P0 (daBit s₂₀).
#[tracing::instrument(skip_all, name = "extend_pool_batched")]
fn extend_pool_batched_base<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    pool: &mut PreprocessingPool<F>,
    deficit_counts: [usize; 5],
    deficit_dabits: usize,
    deficit_rand_ohvs_u8_k4: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<()> {
    use super::dabits;

    let party_id = io.party_id();
    let fb = usize::try_from(F::MODULUS_BIT_SIZE)
        .expect("u32 fits into usize")
        .div_ceil(8);
    let max_msg_elems = preproc_max_elems_per_msg::<F>();
    let forks_cap = io.max_forks().max(1);

    fn edabit_alpha2_ext_chunk<T: IntRing2k, Fp: PrimeField>(
        seed1: [u8; crate::SEED_SIZE],
        pos1: u128,
        seed2: [u8; crate::SEED_SIZE],
        pos2: u128,
        start_item: usize,
        n: usize,
        fb: usize,
    ) -> Vec<Fp>
    where
        Standard: Distribution<T>,
    {
        if n == 0 {
            return Vec::new();
        }
        let t_bytes = std::mem::size_of::<T>();
        let k = T::K;
        let stride = t_bytes + k * fb;

        let all1 = dabits::seek_and_generate(seed1, pos1, start_item * stride, n * stride);
        let g2 = dabits::seek_and_generate(seed2, pos2, start_item * t_bytes, n * t_bytes);

        let mut out = vec![Fp::zero(); n * k];
        out.par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each(|(i, chunk)| {
                let base = i * stride;
                let g1v = T::from_le_bytes(&all1[base..base + t_bytes]);
                let g2v = T::from_le_bytes(&g2[i * t_bytes..(i + 1) * t_bytes]);
                let gamma = g1v ^ g2v;
                for j in 0..k {
                    let s = base + t_bytes + j * fb;
                    let alpha1 = dabits::parse_field::<Fp>(&all1, s);
                    let gbit = ((gamma >> j) & T::one()) == T::one();
                    chunk[j] = Fp::from(gbit as u64) - alpha1;
                }
            });
        out
    }

    // Collect active edaBit types with their metadata.
    let active_types: Vec<(
        usize,
        usize,
        usize,
        ([u8; crate::SEED_SIZE], u128, [u8; crate::SEED_SIZE], u128),
    )> = {
        let ks = [u8::K, u16::K, u32::K, u64::K, u128::K];
        let totals = [
            pool.edabits_u8.total(),
            pool.edabits_u16.total(),
            pool.edabits_u32.total(),
            pool.edabits_u64.total(),
            pool.edabits_u128.total(),
        ];
        let seeds = [
            pool.edabits_u8.extension_seeds(),
            pool.edabits_u16.extension_seeds(),
            pool.edabits_u32.extension_seeds(),
            pool.edabits_u64.extension_seeds(),
            pool.edabits_u128.extension_seeds(),
        ];
        (0..5)
            .filter(|&i| deficit_counts[i] > 0)
            .map(|i| (i, ks[i], totals[i], seeds[i]))
            .collect()
    };
    let active_edabit_lanes = preproc_lanes()
        .max(1)
        .min(forks_cap)
        .min(configured_transport_lanes().max(1));
    let segment_elems = preproc_segment_elems::<F>();

    match party_id {
        PartyID::ID0 => {
            // Round 1: send edaBit α₂ extensions per type, using intra-type
            // lane parallelism (par_segment_plans) just like initial preprocessing.
            for &(ty, k, old_total, seeds) in &active_types {
                let deficit = deficit_counts[ty];
                let (seed1, pos1, seed2, pos2) = seeds;
                let max_items_per_msg = (max_msg_elems / k).max(1);
                let segment_items = (segment_elems / k).max(max_items_per_msg);
                let plans = build_segment_plans(deficit, segment_items);

                if active_edabit_lanes > 1 && plans.len() > 1 {
                    par_segment_plans(io.forks(active_edabit_lanes), plans, |plan, ctx| {
                        let mut done = 0usize;
                        while done < plan.items {
                            let items = (plan.items - done).min(max_items_per_msg);
                            let start = old_total + plan.start_item + done;
                            let alpha2 = match ty {
                                0 => edabit_alpha2_ext_chunk::<u8, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                1 => edabit_alpha2_ext_chunk::<u16, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                2 => edabit_alpha2_ext_chunk::<u32, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                3 => edabit_alpha2_ext_chunk::<u64, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                4 => edabit_alpha2_ext_chunk::<u128, F>(
                                    seed1, pos1, seed2, pos2, start, items, fb,
                                ),
                                _ => unreachable!(),
                            };
                            send_field_superchunk_ctx(alpha2, PartyID::ID2, ctx, max_msg_elems)?;
                            done += items;
                        }
                        Ok(())
                    })?;
                } else {
                    let mut done = 0usize;
                    while done < deficit {
                        let items = (deficit - done).min(max_items_per_msg);
                        let start = old_total + done;
                        let alpha2 = match ty {
                            0 => edabit_alpha2_ext_chunk::<u8, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            1 => edabit_alpha2_ext_chunk::<u16, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            2 => edabit_alpha2_ext_chunk::<u32, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            3 => edabit_alpha2_ext_chunk::<u64, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            4 => edabit_alpha2_ext_chunk::<u128, F>(
                                seed1, pos1, seed2, pos2, start, items, fb,
                            ),
                            _ => unreachable!(),
                        };
                        send_field_superchunk(alpha2, PartyID::ID2, io, max_msg_elems)?;
                        done += items;
                    }
                }
            }

            // Barrier: all parties must finish edaBit phase before daBit phase.
            io.sync_with_parties()?;

            // daBit α₂ extension (chunked, interleaved send/recv to avoid deadlock).

            if deficit_dabits > 0 {
                let (ds1, dp1, ds2, dp2) = pool.dabits.extension_seeds();
                let old_total = pool.dabits.total();
                let da_stride = 1 + 2 * fb;
                let max_items_per_msg = max_msg_elems.max(1);
                let mut done = 0usize;
                while done < deficit_dabits {
                    let items = (deficit_dabits - done).min(max_items_per_msg);
                    let start = old_total + done;
                    let s1_buf =
                        dabits::seek_and_generate(ds1, dp1, start * da_stride, items * da_stride);
                    let g2_buf = dabits::seek_and_generate(ds2, dp2, start, items);
                    let da_alpha2 = (0..items)
                        .into_par_iter()
                        .with_min_len(256)
                        .map(|i| {
                            let base = i * da_stride;
                            let gbit = ((s1_buf[base] ^ g2_buf[i]) & 1) != 0;
                            let alpha1: F = dabits::parse_field(&s1_buf, base + 1);
                            F::from(gbit as u64) - alpha1
                        })
                        .collect::<Vec<F>>();
                    send_field_superchunk(da_alpha2, PartyID::ID2, io, max_msg_elems)?;
                    // Interleaved recv: must drain s20 each iteration to prevent
                    // circular buffer deadlock (P0 blocked sending, P2 blocked sending back).
                    let s20 = recv_field_messages_collect::<F, _>(
                        PartyID::ID2,
                        items,
                        io,
                        max_msg_elems,
                    )?;
                    pool.dabits.apply_extension(items, s20);
                    done += items;
                }
            }

            // Apply edaBits total extensions (P0 stores no α₂).
            pool.edabits_u8
                .apply_extension(deficit_counts[0], Vec::new());
            pool.edabits_u16
                .apply_extension(deficit_counts[1], Vec::new());
            pool.edabits_u32
                .apply_extension(deficit_counts[2], Vec::new());
            pool.edabits_u64
                .apply_extension(deficit_counts[3], Vec::new());
            pool.edabits_u128
                .apply_extension(deficit_counts[4], Vec::new());
        }
        PartyID::ID1 => {
            // Barrier: all parties must finish edaBit phase before daBit phase.
            io.sync_with_parties()?;

            // P1: send daBit s₁₂ extension in chunks.
            if deficit_dabits > 0 {
                let (ds1, dp1, ds2, dp2) = pool.dabits.extension_seeds();
                let old_total = pool.dabits.total();
                let da_stride = 1 + 2 * fb;
                let max_items_per_msg = max_msg_elems.max(1);
                let mut done = 0usize;
                while done < deficit_dabits {
                    let items = (deficit_dabits - done).min(max_items_per_msg);
                    let start = old_total + done;
                    let s2_buf =
                        dabits::seek_and_generate(ds2, dp2, start * da_stride, items * da_stride);
                    let theta_buf = dabits::seek_and_generate(ds1, dp1, start, items);
                    let s12: Vec<F> = (0..items)
                        .into_par_iter()
                        .with_min_len(256)
                        .map(|i| {
                            let base = i * da_stride;
                            let theta = (theta_buf[i] & 1) != 0;
                            let neg1_theta = if theta { -F::one() } else { F::one() };
                            let alpha1: F = dabits::parse_field(&s2_buf, base + 1);
                            let r1: F = dabits::parse_field(&s2_buf, base + 1 + fb);
                            neg1_theta * alpha1 - r1
                        })
                        .collect();
                    send_field_superchunk(s12, PartyID::ID2, io, max_msg_elems)?;
                    done += items;
                }
                pool.dabits.apply_extension(deficit_dabits, Vec::new());
            }
            // Apply edaBits total extensions (P1 stores no α₂).
            pool.edabits_u8
                .apply_extension(deficit_counts[0], Vec::new());
            pool.edabits_u16
                .apply_extension(deficit_counts[1], Vec::new());
            pool.edabits_u32
                .apply_extension(deficit_counts[2], Vec::new());
            pool.edabits_u64
                .apply_extension(deficit_counts[3], Vec::new());
            pool.edabits_u128
                .apply_extension(deficit_counts[4], Vec::new());
        }
        PartyID::ID2 => {
            // Receive edaBit α₂ extensions per type, using intra-type lane
            // parallelism (par_segment_plans) with FileBackedWriter for
            // parallel pwrite to different file offsets.
            for &(ty, k, _old_total, _seeds) in &active_types {
                let deficit = deficit_counts[ty];
                let bs = match ty {
                    0 => &mut pool.edabits_u8.alpha2_flat,
                    1 => &mut pool.edabits_u16.alpha2_flat,
                    2 => &mut pool.edabits_u32.alpha2_flat,
                    3 => &mut pool.edabits_u64.alpha2_flat,
                    4 => &mut pool.edabits_u128.alpha2_flat,
                    _ => unreachable!(),
                };
                let old_len = bs.len();
                let max_items_per_msg = (max_msg_elems / k).max(1);
                let segment_items = (segment_elems / k).max(max_items_per_msg);
                let plans = build_segment_plans(deficit, segment_items);
                if let Some(writer) = bs.pre_extended_writer(deficit * k)? {
                    if active_edabit_lanes > 1 && plans.len() > 1 {
                        par_segment_plans(io.forks(active_edabit_lanes), plans, |plan, ctx| {
                            let mut done = 0usize;
                            while done < plan.items {
                                let items = (plan.items - done).min(max_items_per_msg);
                                let write_start = old_len + (plan.start_item + done) * k;
                                recv_field_messages_into_writer_ctx::<F, _>(
                                    PartyID::ID0,
                                    write_start,
                                    items * k,
                                    ctx,
                                    max_msg_elems,
                                    &writer,
                                )?;
                                done += items;
                            }
                            Ok(())
                        })?;
                    } else {
                        let mut write_offset = old_len;
                        let mut remaining = deficit;
                        while remaining > 0 {
                            let items = remaining.min(max_items_per_msg);
                            recv_field_messages_into_writer_ctx::<F, _>(
                                PartyID::ID0,
                                write_offset,
                                items * k,
                                io.main(),
                                max_msg_elems,
                                &writer,
                            )?;
                            write_offset += items * k;
                            remaining -= items;
                        }
                    }
                }
                // Bump total (file was already extended by grow_and_writer).
                match ty {
                    0 => pool.edabits_u8.total += deficit,
                    1 => pool.edabits_u16.total += deficit,
                    2 => pool.edabits_u32.total += deficit,
                    3 => pool.edabits_u64.total += deficit,
                    4 => pool.edabits_u128.total += deficit,
                    _ => unreachable!(),
                }
            }

            // Barrier: all parties must finish edaBit phase before daBit phase.
            io.sync_with_parties()?;

            // daBits: receive alpha2 from P0 and s12 from P1 in matching chunks, compute s20,
            // append interleaved stored_ext, and forward s20 to P0 immediately.

            if deficit_dabits > 0 {
                let (_ds1, _dp1, ds2, dp2) = pool.dabits.extension_seeds();
                let old_total = pool.dabits.total();

                let max_items_per_msg = max_msg_elems.max(1);
                let mut done = 0usize;
                while done < deficit_dabits {
                    let items = (deficit_dabits - done).min(max_items_per_msg);
                    let alpha2 = recv_field_messages_collect::<F, _>(
                        PartyID::ID0,
                        items,
                        io,
                        max_msg_elems,
                    )?;
                    let s12 = recv_field_messages_collect::<F, _>(
                        PartyID::ID1,
                        items,
                        io,
                        max_msg_elems,
                    )?;
                    let theta_buf = dabits::seek_and_generate(ds2, dp2, old_total + done, items);

                    let _span = tracing::trace_span!(
                        "dabit_s20_forward_chunk",
                        party_id = ?party_id,
                        items,
                        bytes = items * std::mem::size_of::<F>()
                    )
                    .entered();
                    let s20: Vec<F> = (0..items)
                        .into_par_iter()
                        .with_min_len(256)
                        .map(|i| {
                            let theta = (theta_buf[i] & 1) != 0;
                            let neg1_theta = if theta { -F::one() } else { F::one() };
                            neg1_theta * alpha2[i]
                        })
                        .collect();

                    let mut stored_chunk = Vec::with_capacity(2 * items);
                    for i in 0..items {
                        stored_chunk.push(s20[i]);
                        stored_chunk.push(s12[i]);
                    }
                    pool.dabits.apply_extension(items, stored_chunk);
                    send_field_superchunk(s20, PartyID::ID0, io, max_msg_elems)?;
                    done += items;
                }
            }
        }
    }

    if deficit_rand_ohvs_u8_k4 > 0 {
        let mut ext = generate_rand_ohvs_lazy(deficit_rand_ohvs_u8_k4, io.main())?;
        let batch = ext.take_batch(deficit_rand_ohvs_u8_k4)?;
        let r_a_ext: Vec<_> = batch.r_shares.iter().map(|s| s.a.0).collect();
        let r_b_ext: Vec<_> = batch.r_shares.iter().map(|s| s.b.0).collect();
        let e_a_ext: Vec<_> = batch.e_fields_flat.iter().map(|s| s.a).collect();
        let e_b_ext: Vec<_> = batch.e_fields_flat.iter().map(|s| s.b).collect();
        pool.rand_ohvs_u8_k4.apply_extension(
            deficit_rand_ohvs_u8_k4,
            r_a_ext,
            r_b_ext,
            e_a_ext,
            e_b_ext,
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public wrapper functions
// ---------------------------------------------------------------------------

/// File-backed preprocessing: generate all edaBits + daBits + ring edaBits into `dir`.
#[cfg(not(feature = "ring-msm"))]
pub fn preprocess_pool<F, N>(
    dir: &Path,
    counts: [usize; 5],
    num_dabits: usize,
    num_rand_ohvs_u8_k4: usize,
    num_ring_edabits_u64: usize,
    num_ring_edabits_u128: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PreprocessingPool<F>>
where
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
{
    let mut pool = preprocess_pool_base(dir, counts, num_dabits, num_rand_ohvs_u8_k4, io)?;
    if num_ring_edabits_u64 > 0 {
        pool.set_ring_edabits_u64(super::edabits::random_edabits_ring_lazy::<u64, _>(
            num_ring_edabits_u64,
            io,
        )?);
    }
    if num_ring_edabits_u128 > 0 {
        pool.set_ring_edabits_u128(super::edabits::random_edabits_ring_lazy::<u128, _>(
            num_ring_edabits_u128,
            io,
        )?);
    }
    pool.save(dir)?;
    Ok(pool)
}

/// File-backed preprocessing: generate edaBits + daBits + wrap masks + ring edaBits into `dir`.
#[cfg(feature = "ring-msm")]
pub fn preprocess_pool<F, N>(
    dir: &Path,
    counts: [usize; 5],
    num_dabits: usize,
    num_rand_ohvs_u8_k4: usize,
    num_wrap_masks: usize,
    num_ring_edabits_u66: usize,
    num_ring_edabits_u64: usize,
    num_ring_edabits_u128: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PreprocessingPool<F>>
where
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
{
    let mut pool = preprocess_pool_base(dir, counts, num_dabits, num_rand_ohvs_u8_k4, io)?;
    if num_wrap_masks > 0 {
        pool.set_wrap_masks(super::wrap_mask::generate_wrap_masks_lazy(
            num_wrap_masks,
            io.main(),
        )?);
    }
    if num_ring_edabits_u66 > 0 {
        pool.set_ring_edabits_u66(super::edabits::random_edabits_ring_lazy::<U66, _>(
            num_ring_edabits_u66,
            io,
        )?);
    }
    if num_ring_edabits_u64 > 0 {
        pool.set_ring_edabits_u64(super::edabits::random_edabits_ring_lazy::<u64, _>(
            num_ring_edabits_u64,
            io,
        )?);
    }
    if num_ring_edabits_u128 > 0 {
        pool.set_ring_edabits_u128(super::edabits::random_edabits_ring_lazy::<u128, _>(
            num_ring_edabits_u128,
            io,
        )?);
    }
    pool.save(dir)?;
    Ok(pool)
}

/// Extend an existing pool with additional edaBits + daBits + ring edaBits.
#[cfg(not(feature = "ring-msm"))]
pub fn extend_pool_batched<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    pool: &mut PreprocessingPool<F>,
    deficit_counts: [usize; 5],
    deficit_dabits: usize,
    deficit_rand_ohvs_u8_k4: usize,
    deficit_ring_edabits_u64: usize,
    deficit_ring_edabits_u128: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<()> {
    extend_pool_batched_base(pool, deficit_counts, deficit_dabits, deficit_rand_ohvs_u8_k4, io)?;
    if deficit_ring_edabits_u64 > 0 {
        pool.set_ring_edabits_u64(super::edabits::random_edabits_ring_lazy::<u64, _>(
            deficit_ring_edabits_u64,
            io,
        )?);
    }
    if deficit_ring_edabits_u128 > 0 {
        pool.set_ring_edabits_u128(super::edabits::random_edabits_ring_lazy::<u128, _>(
            deficit_ring_edabits_u128,
            io,
        )?);
    }
    Ok(())
}

/// Extend an existing pool with additional edaBits + daBits + wrap masks + ring edaBits.
#[cfg(feature = "ring-msm")]
pub fn extend_pool_batched<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    pool: &mut PreprocessingPool<F>,
    deficit_counts: [usize; 5],
    deficit_dabits: usize,
    deficit_rand_ohvs_u8_k4: usize,
    deficit_wrap_masks: usize,
    deficit_ring_edabits_u66: usize,
    deficit_ring_edabits_u64: usize,
    deficit_ring_edabits_u128: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<()> {
    extend_pool_batched_base(
        pool,
        deficit_counts,
        deficit_dabits,
        deficit_rand_ohvs_u8_k4,
        io,
    )?;
    if deficit_wrap_masks > 0 {
        pool.set_wrap_masks(super::wrap_mask::generate_wrap_masks_lazy(
            deficit_wrap_masks,
            io.main(),
        )?);
    }
    if deficit_ring_edabits_u66 > 0 {
        pool.set_ring_edabits_u66(super::edabits::random_edabits_ring_lazy::<U66, _>(
            deficit_ring_edabits_u66,
            io,
        )?);
    }
    if deficit_ring_edabits_u64 > 0 {
        pool.set_ring_edabits_u64(super::edabits::random_edabits_ring_lazy::<u64, _>(
            deficit_ring_edabits_u64,
            io,
        )?);
    }
    if deficit_ring_edabits_u128 > 0 {
        pool.set_ring_edabits_u128(super::edabits::random_edabits_ring_lazy::<u128, _>(
            deficit_ring_edabits_u128,
            io,
        )?);
    }
    Ok(())
}
