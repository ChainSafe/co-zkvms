//! Ring-MSM extensions for [`PreprocessingPool`].
//!
//! This module is only compiled when the `ring-msm` feature is enabled.
//! It adds daPoints, wrap masks, and Dory carry-ring edaBits support
//! to the base preprocessing pool.

use std::path::Path;

use super::backing_store;
use super::edabits::{EdaBitsRingBatch, LazyEdaBitsRing};
use super::pool::{
    DoryRingMsmInt, PreprocessingPool, configured_transport_lanes, extend_pool_batched_base, preproc_lanes,
    preproc_max_msg_mb, preproc_segment_mb, preprocess_pool_base,
};
use crate::field::PrimeField;
use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::network::Rep3RawFieldTransport;
use crate::protocols::rep3::network::{IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker};
use crate::protocols::rep3_ring::ring::int_ring::IntRing2k;
use crate::protocols::rep3_ring::ring::ring_impl::RingElement;
use crate::protocols::rep3_ring::ring::u34::U34;
use crate::protocols::rep3_ring::ring::u66::U66;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rayon::prelude::*;

// =============================================================================
// Ring-MSM methods on PreprocessingPool
// =============================================================================

impl<F: PrimeField, C: ark_ec::CurveGroup> PreprocessingPool<F, C> {
    /// Inject pre-generated daPoints into this pool.
    pub fn set_dapoints(&mut self, dp: super::dapoint::LazyDaPoints<C>) {
        self.dapoints = dp;
    }

    /// Drain `n` daPoint tuples from the lazy source.
    pub fn take_dapoints(&mut self, n: usize) -> eyre::Result<super::dapoint::DaPointsBatch<C>> {
        self.dapoints.take_batch(n)
    }

    pub fn remaining_dapoints(&self) -> usize {
        self.dapoints.remaining()
    }

    /// Discard cached daPoints if `num_columns` doesn't match.
    /// Returns `true` if daPoints were invalidated.
    pub fn validate_dapoints_num_columns(&mut self, num_columns: usize, party_id: PartyID) -> bool {
        let mut invalidated = false;
        if self.dapoints.num_columns() != Some(num_columns) && self.dapoints.remaining() > 0 {
            tracing::info!(
                stored = ?self.dapoints.num_columns(),
                current = num_columns,
                "daPoints num_columns mismatch; discarding cached daPoints"
            );
            self.dapoints = super::dapoint::LazyDaPoints::empty(party_id);
            invalidated = true;
        } else if self.dapoints.num_columns() != Some(num_columns) {
            self.dapoints = super::dapoint::LazyDaPoints::empty(party_id);
        }
        if self.dapoints_iring.num_columns() != Some(num_columns) && self.dapoints_iring.remaining() > 0 {
            tracing::info!(
                stored = ?self.dapoints_iring.num_columns(),
                current = num_columns,
                "daPoints_iring num_columns mismatch; discarding cached daPoints"
            );
            self.dapoints_iring = super::dapoint::LazyDaPoints::empty(party_id);
            invalidated = true;
        } else if self.dapoints_iring.num_columns() != Some(num_columns) {
            self.dapoints_iring = super::dapoint::LazyDaPoints::empty(party_id);
        }
        invalidated
    }

    /// Inject pre-generated lazy wrap masks into this pool.
    pub fn set_wrap_masks(&mut self, wm: super::wrap_mask::LazyWrapMasks<DoryRingMsmInt>) {
        self.wrap_masks = wm;
    }

    /// Drain `n` wrap masks from the lazy source.
    pub fn take_wrap_masks(&mut self, n: usize) -> eyre::Result<super::wrap_mask::WrapMaskBatch<DoryRingMsmInt>> {
        self.wrap_masks.take_batch(n)
    }

    pub fn remaining_wrap_masks(&self) -> usize {
        self.wrap_masks.remaining()
    }

    /// Inject pre-generated daPoints for IRingScalars into this pool.
    pub fn set_dapoints_iring(&mut self, dp: super::dapoint::LazyDaPoints<C>) {
        self.dapoints_iring = dp;
    }

    /// Drain `n` daPoint tuples for IRingScalars from the lazy source.
    pub fn take_dapoints_iring(&mut self, n: usize) -> eyre::Result<super::dapoint::DaPointsBatch<C>> {
        self.dapoints_iring.take_batch(n)
    }

    pub fn remaining_dapoints_iring(&self) -> usize {
        self.dapoints_iring.remaining()
    }

    /// Inject pre-generated lazy wrap masks for IRingScalars into this pool.
    pub fn set_wrap_masks_iring(&mut self, wm: super::wrap_mask::LazyWrapMasks<U66>) {
        self.wrap_masks_iring = wm;
    }

    /// Drain `n` wrap masks for IRingScalars from the lazy source.
    pub fn take_wrap_masks_iring(&mut self, n: usize) -> eyre::Result<super::wrap_mask::WrapMaskBatch<U66>> {
        self.wrap_masks_iring.take_batch(n)
    }

    pub fn remaining_wrap_masks_iring(&self) -> usize {
        self.wrap_masks_iring.remaining()
    }

    /// Inject pre-generated lazy Dory carry-ring edaBits into this pool.
    pub fn set_ring_edabits_dory(&mut self, eb: LazyEdaBitsRing<DoryRingMsmInt>) {
        #[cfg(feature = "rv64")]
        {
            self.ring_edabits_u66 = eb;
        }
        #[cfg(not(feature = "rv64"))]
        {
            self.ring_edabits_u34 = eb;
        }
    }

    /// Drain `n` Dory carry-ring edaBits from the lazy source.
    pub fn take_ring_edabits_dory(&mut self, n: usize) -> eyre::Result<EdaBitsRingBatch<DoryRingMsmInt>> {
        #[cfg(feature = "rv64")]
        {
            self.ring_edabits_u66.take_batch(n)
        }
        #[cfg(not(feature = "rv64"))]
        {
            self.ring_edabits_u34.take_batch(n)
        }
    }

    pub fn remaining_ring_edabits_dory(&self) -> usize {
        #[cfg(feature = "rv64")]
        {
            self.ring_edabits_u66.remaining()
        }
        #[cfg(not(feature = "rv64"))]
        {
            self.ring_edabits_u34.remaining()
        }
    }

    /// Remaining U66 ring edaBits (used for both rv64 Dory and IRingScalars).
    pub fn remaining_ring_edabits_u66(&self) -> usize {
        self.ring_edabits_u66.remaining()
    }

    /// Generate wrap masks and daPoints for ring-MSM preprocessing.
    ///
    /// Takes pre-computed Q column vectors (from Dory SRS) and budget counts.
    /// Generates wrap masks and daPoints in parallel when multiple tasks are active,
    /// falling back to sequential when only one task is needed.
    pub fn preprocess_wrap_masks_and_dapoints<N: Rep3Network + Rep3NetworkWorker>(
        &mut self,
        q0_xlen_cols: &[C],
        q1_xlen_cols: &[C],
        q0_64_cols: &[C],
        q1_64_cols: &[C],
        num_columns: usize,
        num_wrap_masks: usize,
        num_wrap_masks_iring: usize,
        num_dapoints: usize,
        num_dapoints_iring: usize,
        io: &mut IoContextPool<N>,
    ) -> eyre::Result<()>
    where
        Standard: Distribution<DoryRingMsmInt>,
        Standard: Distribution<U66>,
    {
        let need_wm = num_wrap_masks > 0;
        let need_wm_iring = num_wrap_masks_iring > 0;
        let need_dp = num_dapoints > 0;
        let need_dp_iring = num_dapoints_iring > 0;
        let active = [need_wm, need_wm_iring, need_dp, need_dp_iring].iter().filter(|&&b| b).count();

        if active == 0 {
            return Ok(());
        }

        if active >= 2 && io.max_forks() >= active {
            // Parallel: one task per fork.
            let mut tasks: Vec<(
                u8,
                Box<dyn FnOnce(&mut IoContext<N>) -> eyre::Result<Box<dyn std::any::Any + Send>> + Send>,
            )> = Vec::new();

            if need_wm {
                let n = num_wrap_masks;
                tasks.push((
                    0,
                    Box::new(move |ctx: &mut IoContext<N>| {
                        let wm = super::wrap_mask::generate_wrap_masks_lazy::<DoryRingMsmInt, _>(n, ctx)?;
                        Ok(Box::new(wm) as Box<dyn std::any::Any + Send>)
                    }),
                ));
            }
            if need_wm_iring {
                let n = num_wrap_masks_iring;
                tasks.push((
                    1,
                    Box::new(move |ctx: &mut IoContext<N>| {
                        let wm = super::wrap_mask::generate_wrap_masks_lazy::<U66, _>(n, ctx)?;
                        Ok(Box::new(wm) as Box<dyn std::any::Any + Send>)
                    }),
                ));
            }
            if need_dp {
                let num_coeffs = num_dapoints / 2;
                let nc = num_columns;
                let q0 = q0_xlen_cols.to_vec();
                let q1 = q1_xlen_cols.to_vec();
                tasks.push((
                    2,
                    Box::new(move |ctx: &mut IoContext<N>| {
                        let dp = super::dapoint::random_dapoints_from_columns(&q0, &q1, num_coeffs, nc, ctx)?;
                        Ok(Box::new(dp) as Box<dyn std::any::Any + Send>)
                    }),
                ));
            }
            if need_dp_iring {
                let num_coeffs = num_dapoints_iring / 2;
                let nc = num_columns;
                let q0 = q0_64_cols.to_vec();
                let q1 = q1_64_cols.to_vec();
                tasks.push((
                    3,
                    Box::new(move |ctx: &mut IoContext<N>| {
                        let dp = super::dapoint::random_dapoints_from_columns(&q0, &q1, num_coeffs, nc, ctx)?;
                        Ok(Box::new(dp) as Box<dyn std::any::Any + Send>)
                    }),
                ));
            }

            let forks = io.forks(tasks.len());
            let results: Vec<(u8, eyre::Result<Box<dyn std::any::Any + Send>>)> =
                tasks.into_par_iter().zip(forks.par_iter_mut()).map(|((tag, f), ctx)| (tag, f(ctx))).collect();

            for (tag, result) in results {
                let val: Box<dyn std::any::Any + Send> = result?;
                match tag {
                    0 => {
                        self.set_wrap_masks(*val.downcast::<super::wrap_mask::LazyWrapMasks<DoryRingMsmInt>>().unwrap())
                    }
                    1 => self.set_wrap_masks_iring(*val.downcast::<super::wrap_mask::LazyWrapMasks<U66>>().unwrap()),
                    2 => self.set_dapoints(*val.downcast::<super::dapoint::LazyDaPoints<C>>().unwrap()),
                    3 => self.set_dapoints_iring(*val.downcast::<super::dapoint::LazyDaPoints<C>>().unwrap()),
                    _ => unreachable!(),
                }
            }
        } else {
            // Sequential fallback.
            if need_wm {
                self.set_wrap_masks(super::wrap_mask::generate_wrap_masks_lazy::<DoryRingMsmInt, _>(
                    num_wrap_masks,
                    io.main(),
                )?);
            }
            if need_wm_iring {
                self.set_wrap_masks_iring(super::wrap_mask::generate_wrap_masks_lazy::<U66, _>(
                    num_wrap_masks_iring,
                    io.main(),
                )?);
            }
            if need_dp {
                let dp = super::dapoint::random_dapoints_from_columns(
                    q0_xlen_cols,
                    q1_xlen_cols,
                    num_dapoints / 2,
                    num_columns,
                    io.main(),
                )?;
                self.set_dapoints(dp);
            }
            if need_dp_iring {
                let dp = super::dapoint::random_dapoints_from_columns(
                    q0_64_cols,
                    q1_64_cols,
                    num_dapoints_iring / 2,
                    num_columns,
                    io.main(),
                )?;
                self.set_dapoints_iring(dp);
            }
        }
        Ok(())
    }
}

// =============================================================================
// Ring edaBit generation
// =============================================================================

/// Compute α₂ for a contiguous range of ring edaBits from seed snapshots.
///
/// Analogous to `edabit_alpha2_seed_chunk` but returns `Vec<RingElement<T>>` (ring domain).
/// rng1 stride = `(1+K)*t_bytes` (interleaved gamma + K alphas).
/// rng2 stride = `(1+K)*t_bytes` (same as rng1, both rngs advance equally).
fn ring_edabit_alpha2_seed_chunk<T: IntRing2k>(
    seed1: [u8; crate::SEED_SIZE],
    pos1: u128,
    seed2: [u8; crate::SEED_SIZE],
    pos2: u128,
    start_item: usize,
    num: usize,
    parallel: bool,
) -> Vec<RingElement<T>>
where
    Standard: Distribution<T>,
{
    use super::dabits;

    if num == 0 {
        return Vec::new();
    }
    let t_bytes = std::mem::size_of::<T>();
    let k = T::K;
    let stride = (1 + k) * t_bytes;
    let _span =
        tracing::trace_span!("ring_edabit_alpha2_p0_chunk", ring_bits = k, items = num, start_item, elems = num * k,)
            .entered();

    let all1 = dabits::seek_and_generate(seed1, pos1, start_item * stride, num * stride);
    let g2 = dabits::seek_and_generate(seed2, pos2, start_item * stride, num * stride);

    let mut out = vec![RingElement(T::zero()); num * k];
    let fill_chunk = |(i, chunk): (usize, &mut [RingElement<T>])| {
        let base = i * stride;
        let g1v = T::from_le_bytes(&all1[base..base + t_bytes]);
        let g2v = T::from_le_bytes(&g2[base..base + t_bytes]);
        let gamma = g1v ^ g2v;
        for j in 0..k {
            let a_start = base + t_bytes + j * t_bytes;
            let alpha_1 = RingElement(T::from_le_bytes(&all1[a_start..a_start + t_bytes]));
            let gamma_bit = ((gamma >> j) & T::one()) == T::one();
            chunk[j] = RingElement(T::from(gamma_bit)) - alpha_1;
        }
    };
    if parallel {
        out.par_chunks_mut(k).enumerate().with_min_len(256).for_each(fill_chunk);
    } else {
        out.chunks_mut(k).enumerate().for_each(fill_chunk);
    }
    out
}

/// Stream ring edaBits for one ring type: P0 computes α₂ in segments and sends to P2.
fn stream_ring_edabits<T: IntRing2k, N: Rep3Network>(
    seeds: ([u8; crate::SEED_SIZE], u128, [u8; crate::SEED_SIZE], u128),
    num: usize,
    store_path: &Path,
    party_id: PartyID,
    io: &mut IoContext<N>,
) -> eyre::Result<LazyEdaBitsRing<T>>
where
    Standard: Distribution<T>,
{
    let _span = tracing::info_span!("ring_edabits_stream", ring_bits = T::K, n = num).entered();
    let (seed1, pos1, seed2, pos2) = seeds;
    if num == 0 {
        return eyre::Result::Ok(LazyEdaBitsRing::empty(party_id));
    }
    let k = T::K;
    let elem_bytes = std::mem::size_of::<RingElement<T>>();
    // Items per message: target ~2MB of α₂ per message.
    let max_alpha2_per_msg = (preproc_max_msg_mb() * 1024 * 1024 / (k * elem_bytes).max(1)).max(1);
    // Items per segment for P0 computation: target ~64MB.
    let t_bytes = std::mem::size_of::<T>();
    let stride = (1 + k) * t_bytes;
    let segment_items = (preproc_segment_mb() * 1024 * 1024 / (stride).max(1)).max(max_alpha2_per_msg);

    match party_id {
        PartyID::ID0 => {
            let mut done = 0usize;
            while done < num {
                let seg_items = (num - done).min(segment_items);
                let alpha2 = ring_edabit_alpha2_seed_chunk::<T>(seed1, pos1, seed2, pos2, done, seg_items, true);
                // Send in bounded chunks.
                let mut offset = 0usize;
                let total_elems = alpha2.len();
                let max_elems = max_alpha2_per_msg * k;
                while offset < total_elems {
                    let chunk_len = (total_elems - offset).min(max_elems);
                    io.network.send_many(PartyID::ID2, &alpha2[offset..offset + chunk_len])?;
                    offset += chunk_len;
                }
                done += seg_items;
            }
        }
        PartyID::ID2 => {
            let total_elems = num * k;
            let mut store =
                backing_store::BackingStore::<RingElement<T>>::create_file_backed_sized(store_path, total_elems)?;
            let max_elems = max_alpha2_per_msg * k;
            let mut received = 0usize;
            while received < total_elems {
                let expect = (total_elems - received).min(max_elems);
                let chunk: Vec<RingElement<T>> = io.network.recv_many(PartyID::ID0)?;
                debug_assert_eq!(chunk.len(), expect);
                store.write_at(received, &chunk)?;
                received += chunk.len();
            }
            return eyre::Result::Ok(LazyEdaBitsRing::new_with_store(seed1, pos1, seed2, pos2, num, store, party_id));
        }
        PartyID::ID1 => {} // P1: seeds only, no IO.
    }
    eyre::Result::Ok(LazyEdaBitsRing::new(seed1, pos1, seed2, pos2, num, Vec::new(), party_id))
}

/// Orchestrate ring-edaBit generation: fork rngs upfront, dispatch in parallel when possible.
#[tracing::instrument(skip_all, name = "preprocess_ring_edabits")]
fn preprocess_ring_edabits<F, N>(
    pool: &mut PreprocessingPool<F>,
    dir: &Path,
    num_ring_edabits_u64: usize,
    num_ring_edabits_u128: usize,
    num_ring_edabits_dory: usize,
    num_ring_edabits_iring: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<()>
where
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
{
    let party_id = io.party_id();

    // Fork rngs from io.main() for each ring type (deterministic order across parties).
    // Must match the ordering in the sequential preprocess_pool so seeds are identical.
    let snap_u64 = io.main().rngs.rand.fork().snapshot();
    let snap_u128 = io.main().rngs.rand.fork().snapshot();
    #[cfg(feature = "rv64")]
    let combined_dory = num_ring_edabits_dory + num_ring_edabits_iring;
    #[cfg(feature = "rv64")]
    let snap_dory = io.main().rngs.rand.fork().snapshot();
    #[cfg(not(feature = "rv64"))]
    let snap_dory = io.main().rngs.rand.fork().snapshot();
    #[cfg(not(feature = "rv64"))]
    let snap_iring = io.main().rngs.rand.fork().snapshot();

    // Collect active ring-edabit types: (tag, count, seeds).
    type Seeds = ([u8; crate::SEED_SIZE], u128, [u8; crate::SEED_SIZE], u128);
    let mut tasks: Vec<(u8, usize, Seeds)> = Vec::new();
    if num_ring_edabits_u64 > 0 {
        tasks.push((0, num_ring_edabits_u64, snap_u64));
    }
    if num_ring_edabits_u128 > 0 {
        tasks.push((1, num_ring_edabits_u128, snap_u128));
    }
    {
        #[cfg(feature = "rv64")]
        if combined_dory > 0 {
            tasks.push((2, combined_dory, snap_dory));
        }
        #[cfg(not(feature = "rv64"))]
        {
            if num_ring_edabits_dory > 0 {
                tasks.push((2, num_ring_edabits_dory, snap_dory));
            }
            if num_ring_edabits_iring > 0 {
                tasks.push((3, num_ring_edabits_iring, snap_iring));
            }
        }
    }

    if tasks.is_empty() {
        return eyre::Result::Ok(());
    }

    let active_lanes = preproc_lanes().max(1).min(io.max_forks().max(1)).min(configured_transport_lanes().max(1));

    tracing::info!(
        tasks = tasks.len(),
        active_lanes,
        "ring edabits dispatch: {}",
        if tasks.len() > 1 && active_lanes >= tasks.len() { "PARALLEL" } else { "sequential" }
    );

    // Dispatch: one ring type per fork when enough lanes; sequential fallback otherwise.
    let mut dispatch = |tag: u8, count: usize, seeds: Seeds, ctx: &mut IoContext<N>| -> eyre::Result<()> {
        match tag {
            0 => {
                let path = dir.join(format!("ring_edabits_{}.alpha2", u64::K));
                pool.set_ring_edabits_u64(stream_ring_edabits::<u64, _>(seeds, count, &path, party_id, ctx)?);
            }
            1 => {
                let path = dir.join(format!("ring_edabits_{}.alpha2", u128::K));
                pool.set_ring_edabits_u128(stream_ring_edabits::<u128, _>(seeds, count, &path, party_id, ctx)?);
            }
            2 => {
                let path = dir.join(format!("ring_edabits_{}.alpha2", DoryRingMsmInt::K));
                pool.set_ring_edabits_dory(stream_ring_edabits::<DoryRingMsmInt, _>(
                    seeds, count, &path, party_id, ctx,
                )?);
            }
            #[cfg(not(feature = "rv64"))]
            3 => {
                let path = dir.join(format!("ring_edabits_{}.alpha2", U66::K));
                pool.ring_edabits_u66 = stream_ring_edabits::<U66, _>(seeds, count, &path, party_id, ctx)?;
            }
            _ => unreachable!(),
        }
        eyre::Result::Ok(())
    };

    if tasks.len() > 1 && active_lanes >= tasks.len() {
        // Parallel: one ring type per fork.
        let forks = io.forks(tasks.len());
        // Cannot use par_iter because dispatch borrows pool mutably.
        // Instead, collect results with Box<dyn Any> and apply sequentially.
        let results: Vec<(u8, eyre::Result<_>)> = tasks
            .into_par_iter()
            .zip(forks.par_iter_mut())
            .map(|((tag, count, seeds), ctx)| {
                let result = match tag {
                    0 => {
                        let path = dir.join(format!("ring_edabits_{}.alpha2", u64::K));
                        stream_ring_edabits::<u64, _>(seeds, count, &path, party_id, ctx)
                            .map(|r| Box::new(r) as Box<dyn std::any::Any + Send>)
                    }
                    1 => {
                        let path = dir.join(format!("ring_edabits_{}.alpha2", u128::K));
                        stream_ring_edabits::<u128, _>(seeds, count, &path, party_id, ctx)
                            .map(|r| Box::new(r) as Box<dyn std::any::Any + Send>)
                    }
                    2 => {
                        let path = dir.join(format!("ring_edabits_{}.alpha2", DoryRingMsmInt::K));
                        stream_ring_edabits::<DoryRingMsmInt, _>(seeds, count, &path, party_id, ctx)
                            .map(|r| Box::new(r) as Box<dyn std::any::Any + Send>)
                    }
                    #[cfg(not(feature = "rv64"))]
                    3 => {
                        let path = dir.join(format!("ring_edabits_{}.alpha2", U66::K));
                        stream_ring_edabits::<U66, _>(seeds, count, &path, party_id, ctx)
                            .map(|r| Box::new(r) as Box<dyn std::any::Any + Send>)
                    }
                    _ => unreachable!(),
                };
                (tag, result)
            })
            .collect::<Vec<_>>();

        for (tag, result) in results {
            let val = result?;
            match tag {
                0 => pool.set_ring_edabits_u64(*val.downcast::<LazyEdaBitsRing<u64>>().unwrap()),
                1 => pool.set_ring_edabits_u128(*val.downcast::<LazyEdaBitsRing<u128>>().unwrap()),
                2 => pool.set_ring_edabits_dory(*val.downcast::<LazyEdaBitsRing<DoryRingMsmInt>>().unwrap()),
                #[cfg(not(feature = "rv64"))]
                3 => pool.ring_edabits_u66 = *val.downcast::<LazyEdaBitsRing<U66>>().unwrap(),
                _ => unreachable!(),
            }
        }
    } else {
        // Sequential fallback.
        for (tag, count, seeds) in tasks {
            dispatch(tag, count, seeds, io.main())?;
        }
    }
    eyre::Result::Ok(())
}

// =============================================================================
// Public API functions
// =============================================================================

/// File-backed preprocessing: generate edaBits + daBits + ring edaBits into `dir`.
///
/// Wrap masks and daPoints are NOT generated here — the caller parallelises them
/// with daPoints on IoContextPool forks.
pub fn preprocess_pool<F, N>(
    dir: &Path,
    counts: [usize; 5],
    num_dabits: usize,
    num_ring_edabits_dory: usize,
    num_ring_edabits_u64: usize,
    num_ring_edabits_u128: usize,
    num_ring_edabits_iring: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<PreprocessingPool<F>>
where
    F: PrimeField + Copy,
    N: Rep3NetworkWorker + Rep3RawFieldTransport,
{
    let mut pool = preprocess_pool_base(dir, counts, num_dabits, io)?;

    // Ring edaBits: segmented + lane-parallel.
    preprocess_ring_edabits(
        &mut pool,
        dir,
        num_ring_edabits_u64,
        num_ring_edabits_u128,
        num_ring_edabits_dory,
        num_ring_edabits_iring,
        io,
    )?;

    // Wrap masks are NOT generated here — caller parallelises them with daPoints.
    // The caller must call pool.set_wrap_masks / set_wrap_masks_iring and pool.save.
    Ok(pool)
}

/// Extend an existing pool with additional edaBits + daBits + wrap masks + ring edaBits.
///
/// Regular edaBits + daBits are *appended* to existing data (cursor preserved).
/// Ring edaBits and wrap masks are *replaced* — the old source is discarded and a
/// fresh source of size `deficit + remaining` (= budget) is generated so the full
/// budget is available from cursor 0.
pub fn extend_pool_batched<F: PrimeField, N: Rep3NetworkWorker + Rep3RawFieldTransport>(
    pool: &mut PreprocessingPool<F>,
    deficit_counts: [usize; 5],
    deficit_dabits: usize,
    deficit_ring_edabits_dory: usize,
    deficit_ring_edabits_u64: usize,
    deficit_ring_edabits_u128: usize,
    deficit_ring_edabits_iring: usize,
    io: &mut IoContextPool<N>,
) -> eyre::Result<()> {
    extend_pool_batched_base(pool, deficit_counts, deficit_dabits, io)?;
    // Shared types first — must match non-ring-msm ordering (cross-feature cache compatibility).
    if deficit_ring_edabits_u64 > 0 {
        let total = deficit_ring_edabits_u64 + pool.remaining_ring_edabits_u64();
        pool.set_ring_edabits_u64(super::edabits::random_edabits_ring_lazy::<u64, _>(total, io)?);
    }
    if deficit_ring_edabits_u128 > 0 {
        let total = deficit_ring_edabits_u128 + pool.remaining_ring_edabits_u128();
        pool.set_ring_edabits_u128(super::edabits::random_edabits_ring_lazy::<u128, _>(total, io)?);
    }
    // ring-msm-only types after shared types.
    // Wrap masks are NOT generated here — caller parallelises them with daPoints.
    // For rv64: DoryRingMsmInt=U66, combine dory + iring U66 ring edaBits together.
    // For rv32: DoryRingMsmInt=U34, so ring_edabits_u66 is free for IRingScalars only.
    #[cfg(feature = "rv64")]
    {
        let combined_deficit = deficit_ring_edabits_dory + deficit_ring_edabits_iring;
        if combined_deficit > 0 {
            let total = combined_deficit + pool.remaining_ring_edabits_dory();
            pool.set_ring_edabits_dory(super::edabits::random_edabits_ring_lazy::<DoryRingMsmInt, _>(total, io)?);
        }
    }
    #[cfg(not(feature = "rv64"))]
    {
        if deficit_ring_edabits_dory > 0 {
            let total = deficit_ring_edabits_dory + pool.remaining_ring_edabits_dory();
            pool.set_ring_edabits_dory(super::edabits::random_edabits_ring_lazy::<DoryRingMsmInt, _>(total, io)?);
        }
        if deficit_ring_edabits_iring > 0 {
            // IRingScalars always uses U66 carry ring, even on rv32.
            let total = deficit_ring_edabits_iring + pool.remaining_ring_edabits_u66();
            pool.ring_edabits_u66 = super::edabits::random_edabits_ring_lazy::<U66, _>(total, io)?;
        }
    }
    // deficit_wrap_masks_iring also deferred to caller.
    Ok(())
}
