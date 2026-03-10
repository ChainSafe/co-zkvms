use super::dabits::{DaBitBatch, LazyDaBits};
use super::edabits::{EdaBitsBatch, LazyEdaBits};
use crate::protocols::rep3::PartyID;
use mpc_types::field::PrimeField;
use mpc_types::protocols::rep3_ring::ring::int_ring::IntRing2k;
use rand::distributions::Standard;
use rand::prelude::Distribution;

#[cfg(feature = "ring-msm")]
use super::edabits::{EdaBitsRingBatch, LazyEdaBitsRing};
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
    #[cfg(feature = "ring-msm")]
    pub(crate) dapoints: super::daPoint::LazyDaPoints<C>,
    #[cfg(feature = "ring-msm")]
    pub(crate) wrap_masks: super::wrap_mask::LazyWrapMasks,
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
    ) -> Self {
        Self {
            edabits_u8,
            edabits_u16,
            edabits_u32,
            edabits_u64,
            edabits_u128,
            dabits,
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

    // --- always-available methods ---

    /// Drain `n` daBit tuples (Cheng23 Π₁) from the lazy source.
    #[tracing::instrument(skip(self))]
    pub fn take_dabits(&mut self, n: usize) -> eyre::Result<DaBitBatch<F>> {
        self.dabits.take_batch(n)
    }

    pub fn remaining_dabits(&self) -> usize {
        self.dabits.remaining()
    }

    pub fn is_empty(&self) -> bool {
        self.edabits_u8.remaining() == 0
            && self.edabits_u16.remaining() == 0
            && self.edabits_u32.remaining() == 0
            && self.edabits_u64.remaining() == 0
            && self.edabits_u128.remaining() == 0
            && self.dabits.remaining() == 0
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
    }

    /// Generic edaBits drain as flat batch, dispatched by `TypeId`.
    ///
    /// Returns an error if `T` is not one of u8, u16, u32, u64, u128, or if
    /// there are not enough edaBits remaining.
    #[tracing::instrument(skip(self))]
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
            #[cfg(feature = "ring-msm")]
            let h6 = s.spawn(|| self.wrap_masks.save(dir));
            #[cfg(feature = "ring-msm")]
            let h7 = s.spawn(|| self.ring_edabits_u66.save(dir));
            h0.join().unwrap()?;
            h1.join().unwrap()?;
            h2.join().unwrap()?;
            h3.join().unwrap()?;
            h4.join().unwrap()?;
            h5.join().unwrap()?;
            #[cfg(feature = "ring-msm")]
            h6.join().unwrap()?;
            #[cfg(feature = "ring-msm")]
            h7.join().unwrap()?;
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
