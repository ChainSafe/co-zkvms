//! `SuffixFutureBatch` — pre-bucketed suffix evaluation results.
//!
//! Values are pushed into typed buckets during suffix evaluation, then
//! fulfilled in a single batched pass per ring type.

use crate::utils::types::rep3_value::Rep3Value;
use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// B2ABucketExtend — compile-time dispatch for typed B2A bucket extension
// ---------------------------------------------------------------------------

/// Trait for compile-time dispatch of B2A bucket extension.
/// Each ring type maps to the correct typed bucket in `SuffixFutureBatch`.
pub trait B2ABucketExtend: IntRing2k {
    fn extend_bucket<F: JoltField>(
        batch: &mut SuffixFutureBatch<F>,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<Self>>,
    );
}

macro_rules! impl_b2a_bucket_extend {
    ($ring:ty, $idx_field:ident, $val_field:ident) => {
        impl B2ABucketExtend for $ring {
            fn extend_bucket<F: JoltField>(
                batch: &mut SuffixFutureBatch<F>,
                indices: impl IntoIterator<Item = usize>,
                vals: impl IntoIterator<Item = Rep3RingShare<Self>>,
            ) {
                batch.$idx_field.extend(indices);
                batch.$val_field.extend(vals);
            }
        }
    };
}

impl_b2a_bucket_extend!(u8, b2a_u8_idx, b2a_u8);
impl_b2a_bucket_extend!(u16, b2a_u16_idx, b2a_u16);
impl_b2a_bucket_extend!(u32, b2a_u32_idx, b2a_u32);
impl_b2a_bucket_extend!(u64, b2a_u64_idx, b2a_u64);
impl_b2a_bucket_extend!(u128, b2a_u128_idx, b2a_u128);

// ---------------------------------------------------------------------------
// SuffixFutureBatch
// ---------------------------------------------------------------------------

/// Pre-bucketed collection of suffix evaluation results, replacing the
/// `Vec<SuffixFuture>` + rayon fold/reduce classification scan.
///
/// Values are pushed into typed buckets during suffix evaluation, then
/// fulfilled in a single batched pass per ring type.
pub struct SuffixFutureBatch<F: JoltField> {
    pub(crate) len: usize,

    // Scatter indices (position in output vec)
    pub(crate) ready_idx: Vec<usize>,
    pub(crate) bitinject_idx: Vec<usize>,
    pub(crate) b2a_u8_idx: Vec<usize>,
    pub(crate) b2a_u16_idx: Vec<usize>,
    pub(crate) b2a_u32_idx: Vec<usize>,
    pub(crate) b2a_u64_idx: Vec<usize>,
    pub(crate) b2a_u128_idx: Vec<usize>,

    // Values
    pub(crate) ready: Vec<Rep3Value<F>>,
    pub(crate) bitinject: Vec<Rep3RingShare<Bit>>,
    /// Sparse map: bitinject position → post-injection scalar.
    /// Entries absent from this map get weight 1 (no scaling).
    pub(crate) bitinject_scalars: std::collections::BTreeMap<usize, F>,
    pub(crate) b2a_u8: Vec<Rep3RingShare<u8>>,
    pub(crate) b2a_u16: Vec<Rep3RingShare<u16>>,
    pub(crate) b2a_u32: Vec<Rep3RingShare<u32>>,
    pub(crate) b2a_u64: Vec<Rep3RingShare<u64>>,
    pub(crate) b2a_u128: Vec<Rep3RingShare<u128>>,
}

impl<F: JoltField> SuffixFutureBatch<F> {
    pub fn new() -> Self {
        Self {
            len: 0,
            ready_idx: Vec::new(),
            bitinject_idx: Vec::new(),
            b2a_u8_idx: Vec::new(),
            b2a_u16_idx: Vec::new(),
            b2a_u32_idx: Vec::new(),
            b2a_u64_idx: Vec::new(),
            b2a_u128_idx: Vec::new(),
            ready: Vec::new(),
            bitinject: Vec::new(),
            bitinject_scalars: std::collections::BTreeMap::new(),
            b2a_u8: Vec::new(),
            b2a_u16: Vec::new(),
            b2a_u32: Vec::new(),
            b2a_u64: Vec::new(),
            b2a_u128: Vec::new(),
        }
    }

    /// Reserve `n` output slots, returning the base index for this segment.
    pub fn reserve(&mut self, n: usize) -> usize {
        let base = self.len;
        self.len += n;
        base
    }

    /// Push Ready (field) values with their output indices.
    pub fn extend_ready(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3Value<F>>,
    ) {
        self.ready_idx.extend(indices);
        self.ready.extend(vals);
    }

    /// Push BitInject (single-bit) values with their output indices.
    pub fn extend_bitinject(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<Bit>>,
    ) {
        self.bitinject_idx.extend(indices);
        self.bitinject.extend(vals);
    }

    /// Push BitInject values that will be scaled by a public field element after injection.
    /// All bits in this call share the same scalar. Uses the same bitinject bucket
    /// with a sparse scalar map for the scaled entries.
    pub fn extend_bitinject_scaled(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<Bit>>,
        scalar: F,
    ) {
        let start = self.bitinject.len();
        self.bitinject_idx.extend(indices);
        self.bitinject.extend(vals);
        for pos in start..self.bitinject.len() {
            self.bitinject_scalars.insert(pos, scalar);
        }
    }

    /// Push B2A values into the correct typed bucket via compile-time dispatch.
    pub fn extend_b2a_ring<R: B2ABucketExtend>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        vals: impl IntoIterator<Item = Rep3RingShare<R>>,
    ) {
        R::extend_bucket(self, indices, vals);
    }

    /// Fulfill all pending conversions and scatter into output vec.
    #[tracing::instrument(skip_all, name = "Suffixes::fulfill")]
    pub fn fulfill_with_pool<N: Rep3NetworkWorker>(
        self,
        io_ctx: &mut IoContextPool<N>,
        pool: &mut PreprocessingPool<F>,
    ) -> eyre::Result<Vec<Rep3Value<F>>> {
        use mpc_core::protocols::rep3_ring::casts;
        use mpc_core::protocols::rep3_ring::conversion;

        let mut out = vec![Rep3Value::zero_share(); self.len];

        let b2a_outer_chunk: usize =
            std::env::var("SUFFIXES_B2A_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(8192).max(1);
        let b2a_min_inner_chunk: usize =
            std::env::var("SUFFIXES_B2A_MIN_INNER_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(2048).max(1);
        let b2a_max_forks_cap: Option<usize> =
            std::env::var("SUFFIXES_B2A_MAX_FORKS").ok().and_then(|s| s.parse().ok()).map(|v: usize| v.max(1));

        // Phase 1: Sequential conversions, collect all (idx, val) pairs.
        let mut scatter: Vec<(usize, Rep3Value<F>)> = Vec::with_capacity(self.len);

        // Ready — direct (already Rep3Value)
        scatter.extend(self.ready_idx.into_iter().zip(self.ready.into_iter()));

        // BitInject — single-bit → field via daBits
        if !self.bitinject.is_empty() {
            let dabits = pool.take_dabits(self.bitinject.len())?;
            let _span = tracing::info_span!("bit_inject_field_many", n = self.bitinject.len()).entered();
            let fields = conversion::bit_inject_field_preproc_many(&self.bitinject, &dabits, io_ctx.main())?;
            drop(_span);
            scatter.extend(self.bitinject_idx.into_iter().enumerate().zip(fields.into_iter()).map(
                |((pos, idx), f)| {
                    let val = match self.bitinject_scalars.get(&pos) {
                        Some(&w) => f * w,
                        None => f,
                    };
                    (idx, Rep3Value::Shared(val))
                },
            ));
        }

        // B2A per ring type
        macro_rules! fulfill_b2a {
            ($ring:ty, $idx:ident, $val:ident) => {
                if !self.$val.is_empty() {
                    debug_assert_eq!(self.$idx.len(), self.$val.len());
                    let total = self.$val.len();

                    for off in (0..total).step_by(b2a_outer_chunk) {
                        let end = (off + b2a_outer_chunk).min(total);
                        let chunk_len = end - off;

                        let batch = pool.take_edabits::<$ring>(chunk_len)?;
                        let _span = tracing::info_span!("ring_to_field_b2a_many", n = chunk_len).entered();

                        let chunk_vals: Vec<Rep3RingShare<$ring>> = self.$val[off..end].to_vec();
                        let max_forks_cap = b2a_max_forks_cap.unwrap_or(io_ctx.max_forks()).max(1);
                        let forks_by_size = chunk_len.div_ceil(b2a_min_inner_chunk);
                        let forks_effective = forks_by_size.clamp(1, max_forks_cap);

                        let fields = if io_ctx.max_forks() == 0 || forks_effective <= 1 {
                            casts::r2f_b2a_preproc_many::<$ring, F, _>(&chunk_vals, &batch, io_ctx.main())?
                        } else {
                            let inner_chunk_size = chunk_len.div_ceil(forks_effective);
                            io_ctx.par_chunks_preproc(chunk_vals, batch, Some(inner_chunk_size), |xs, b, ctx| {
                                casts::r2f_b2a_preproc_many::<$ring, F, _>(&xs, &b, ctx)
                            })?
                        };
                        drop(_span);

                        debug_assert_eq!(fields.len(), chunk_len);
                        scatter
                            .extend(self.$idx[off..end].iter().copied().zip(fields.into_iter().map(Rep3Value::Shared)));
                    }
                }
            };
        }
        // TODO: batch across rings?
        fulfill_b2a!(u8, b2a_u8_idx, b2a_u8);
        fulfill_b2a!(u16, b2a_u16_idx, b2a_u16);
        fulfill_b2a!(u32, b2a_u32_idx, b2a_u32);
        fulfill_b2a!(u64, b2a_u64_idx, b2a_u64);
        fulfill_b2a!(u128, b2a_u128_idx, b2a_u128);

        // Phase 2: Parallel scatter — all indices are disjoint by construction.
        use crate::utils::send_ptr::SendPtr;

        let ptr = SendPtr(out.as_mut_ptr());
        scatter.into_par_iter().for_each(|(idx, val)| {
            let p = ptr; // force capture of SendPtr wrapper, not the raw *mut field
                         // SAFETY: idx is unique across all entries; Rep3Value<F> is Copy.
            unsafe {
                p.0.add(idx).write(val);
            }
        });

        Ok(out)
    }
}
