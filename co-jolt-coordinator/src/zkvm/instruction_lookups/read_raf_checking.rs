use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::utils::fwht::{
    fwht_in_place, fwht_rep3_in_place, shift_eq_table_with_mask, unmask_histogram_public,
};
use crate::utils::types::{Either, Rep3Value};
use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use jolt2_common::constants::XLEN;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide};
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::expanding_table::ExpandingTable;
use jolt_core::utils::lookup_bits::LookupBits;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::instruction_lookups::{D, LOG_M};
use jolt_core::zkvm::lookup_table::prefixes::{PrefixCheckpoint, PrefixEval, Prefixes};
use jolt_core::zkvm::lookup_table::suffixes::Suffixes;
use jolt_core::zkvm::lookup_table::LookupTables;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::downcast;
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use num_traits::AsPrimitive;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rayon::prelude::*;
use std::sync::Arc;
use strum::{EnumCount, IntoEnumIterator};
use tracing::{info_span, trace_span};

use crate::poly::additive_dense_poly::AdditiveDensePoly;
use crate::utils::lagrange_interp_4;

const LOG_K: usize = XLEN * 2; // 128
const PHASES: usize = 8;
const M: usize = 1 << LOG_M; // 65536
const DEGREE: usize = 3;

fn fwht_unmask_rep3_to_additive<F: JoltField>(
    h: &mut [Rep3PrimeFieldShare<F>],
    ehat16: &[Rep3PrimeFieldShare<F>],
    inv_m: F,
) -> Vec<AdditiveShare<F>> {
    debug_assert_eq!(h.len(), M);
    debug_assert_eq!(ehat16.len(), M);

    fwht_rep3_in_place(h);
    let mut h_k: Vec<AdditiveShare<F>> = h
        .iter()
        .zip(ehat16.iter())
        .map(|(&a, &b)| (a * inv_m) * b)
        .collect();
    fwht_in_place(&mut h_k);
    h_k
}

fn reshare_and_unmask_additive_hists_chunked<F: JoltField, N: Rep3NetworkWorker>(
    hists: Vec<(usize, usize, Vec<AdditiveShare<F>>)>,
    ehat16: &[Rep3PrimeFieldShare<F>],
    inv_m: F,
    party_id: PartyID,
    io_ctx: &mut IoContextPool<N>,
    chunk_hists: usize,
) -> eyre::Result<Vec<(usize, usize, AdditiveDensePoly<F>)>> {
    if hists.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_hists = chunk_hists.max(1);
    let max_forks = io_ctx.max_forks();

    let _span = info_span!(
        "reshare_hists",
        n = hists.len(),
        chunk = chunk_hists,
        m = M,
        max_forks,
        party_id = ?party_id
    )
    .entered();

    let mut do_one_chunk = |chunk: Vec<(usize, usize, Vec<AdditiveShare<F>>)>,
                            ctx: &mut mpc_core::protocols::rep3::network::IoContext<N>|
     -> eyre::Result<Vec<(usize, usize, AdditiveDensePoly<F>)>> {
        let _chunk_span = trace_span!(
            "reshare_hists_chunk",
            n = chunk.len(),
            total_len = chunk.len() * M
        )
        .entered();

        let mut meta: Vec<(usize, usize)> = Vec::with_capacity(chunk.len());
        let mut flat: Vec<AdditiveShare<F>> = Vec::with_capacity(chunk.len() * M);
        for (ti, si, mut hist) in chunk {
            meta.push((ti, si));
            flat.append(&mut hist);
        }

        let mut flat_rep3 = rep3_arith::reshare_additive_many(&flat, ctx)?;
        drop(flat);

        let mut out: Vec<(usize, usize, AdditiveDensePoly<F>)> = Vec::with_capacity(meta.len());
        for (k, (ti, si)) in meta.into_iter().enumerate() {
            let seg = &mut flat_rep3[k * M..(k + 1) * M];
            let h_k = fwht_unmask_rep3_to_additive(seg, ehat16, inv_m);
            out.push((ti, si, AdditiveDensePoly::new(h_k)));
        }

        drop(_chunk_span);
        Ok(out)
    };

    if max_forks < 2 {
        let mut out = Vec::with_capacity(hists.len());
        let mut iter = hists.into_iter();
        loop {
            let mut chunk = Vec::with_capacity(chunk_hists);
            for _ in 0..chunk_hists {
                match iter.next() {
                    Some(v) => chunk.push(v),
                    None => break,
                }
            }
            if chunk.is_empty() {
                break;
            }
            out.extend(do_one_chunk(chunk, io_ctx.main())?);
        }
        return Ok(out);
    }

    let adjusted_chunk = chunk_hists.max(hists.len().div_ceil(max_forks)).max(1);
    debug_assert!(hists.len().div_ceil(adjusted_chunk) <= max_forks);

    io_ctx.par_chunks(
        hists.into_par_iter(),
        Some(adjusted_chunk),
        move |chunk, ctx| do_one_chunk(chunk, ctx),
    )
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute the public per-suffix weights for `LookupTables::combine(prefixes, suffixes)`.
///
/// `LookupTables::combine` is linear in `suffixes`, so:
///   combine(prefixes, suffixes) = Σ_i weights[i] * suffixes[i].
///
/// We obtain `weights[i] = combine(prefixes, e_i)` by probing with unit vectors.
#[inline]
fn combine_shared_weights<F: JoltField>(
    table: &LookupTables<XLEN>,
    prefixes: &[PrefixEval<F>],
    n: usize,
) -> [F; 8] {
    debug_assert!(n <= 8, "suffix count exceeds stack buffer size");

    let mut unit = [F::zero(); 8];
    let mut weights = [F::zero(); 8];
    for i in 0..n {
        unit[i] = F::one();
        weights[i] = table.combine(prefixes, &unit[..n]);
        unit[i] = F::zero();
    }
    weights
}

#[inline]
fn dot_weights_suffixes<F: JoltField>(
    weights: &[F; 8],
    suffixes: &[AdditiveShare<F>; 8],
    n: usize,
) -> AdditiveShare<F> {
    debug_assert!(n <= 8, "suffix count exceeds stack buffer size");
    let mut result = AdditiveShare::<F>::zero();
    for i in 0..n {
        result += suffixes[i] * weights[i];
    }
    result
}

/// MPC version of `PrefixSuffixDecomposition::sumcheck_evals`.
///
/// Given public P polynomial (from PrefixRegistry, ORDER=2: P[0] = Some(poly), P[1] = None)
/// and additive Q arrays, compute (eval_0, eval_2) at the given sumcheck index.
///
/// For P[i] = Some(prefix_poly): p_evals = (p[index], 2*p[index+len/2] - p[index])
/// For P[i] = None:              p_evals = (1, 1)
fn psd_sumcheck_evals_shared<F: JoltField>(
    p_poly: Option<
        &std::sync::Arc<std::sync::RwLock<jolt_core::poly::prefix_suffix::CachedPolynomial<F>>>,
    >,
    q: &[AdditiveDensePoly<F>; 2],
    index: usize,
    len: usize,
) -> (AdditiveShare<F>, AdditiveShare<F>) {
    let mut eval_0 = AdditiveShare::<F>::zero();
    let mut eval_2_left = AdditiveShare::<F>::zero();
    let mut eval_2_right = AdditiveShare::<F>::zero();

    // P[0] = p_poly (may be Some), P[1] = None (constant 1)
    let p_polys: [Option<
        &std::sync::Arc<std::sync::RwLock<jolt_core::poly::prefix_suffix::CachedPolynomial<F>>>,
    >; 2] = [p_poly, None];

    for (i, p) in p_polys.iter().enumerate() {
        let (p_0, p_2) = if let Some(p_arc) = p {
            let p_guard = p_arc.read().unwrap();
            let use_cache = std::sync::Arc::strong_count(p_arc) > 2;
            let evals = p_guard.cached_sumcheck_evals(index, 2, BindingOrder::HighToLow, use_cache);
            evals
        } else {
            (F::one(), F::one())
        };

        let q_left = q[i].get_coeff(index);
        let q_right = q[i].get_coeff(index + len / 2);

        eval_0 = eval_0 + q_left * p_0;
        eval_2_left = eval_2_left + q_left * p_2;
        eval_2_right = eval_2_right + q_right * p_2;
    }

    (eval_0, eval_2_right + eval_2_right - eval_2_left)
}

// ---------------------------------------------------------------------------
// Per-table suffix evaluation
// ---------------------------------------------------------------------------

/// Identifies a contiguous segment of suffix evaluation results in the flat output.
#[derive(Clone, Copy)]
struct EvalSegment {
    table_idx: usize,
    suffix_idx: usize,
    base: usize,
    n: usize,
}

/// Build `SuffixBitsBatch<T>` per table, evaluate all non-One suffixes per table,
/// and fulfill all B2A conversions in one batched pass.
///
/// Returns `(segments, all_field)` where each segment maps `(table_idx, suffix_idx)`
/// to a `[base..base+n)` slice of `all_field`.
fn table_suffixes_mle<T, F, N>(
    lookup_indices: &[Either<u128, Rep3RingShare<u128>>],
    lookup_indices_by_table: &[Vec<usize>],
    right_operand_public_mask: &[Option<u64>],
    suffix_len: usize,
    io_ctx: &mut IoContextPool<N>,
    party_id: PartyID,
    pool: &mut PreprocessingPool<F>,
) -> eyre::Result<(Vec<EvalSegment>, Vec<Rep3Value<F>>)>
where
    T: crate::zkvm::suffixes::Uninterleavable
        + AsPrimitive<mpc_core::protocols::rep3_ring::ring::bit::Bit>,
    Standard: Distribution<T> + Distribution<T::Half>,
    <T as crate::zkvm::suffixes::Uninterleavable>::Half:
        AsPrimitive<T> + AsPrimitive<mpc_core::protocols::rep3_ring::ring::bit::Bit>,
    F: JoltField,
    N: Rep3NetworkWorker,
{
    use crate::zkvm::suffixes::{
        evaluate_suffix_for_table, table_uses_interleaved_data, MixedBatch, SuffixBitsBatch,
        SuffixFutureBatch, Uninterleavable,
    };

    type H<T> = <T as Uninterleavable>::Half;

    let suffix_mask: u128 = if suffix_len >= 128 {
        u128::MAX
    } else {
        (1u128 << suffix_len) - 1
    };
    let half_bits = suffix_len / 2;

    let mut batch = SuffixFutureBatch::<F>::new();
    let mut segments = Vec::new();
    let _span = info_span!("suffixes_mle", n = lookup_indices.len()).entered();

    for (table_idx, table) in LookupTables::<XLEN>::iter().enumerate() {
        let table_cycles = &lookup_indices_by_table[table_idx];
        if table_cycles.is_empty() {
            continue;
        }

        let suffixes = table.suffixes();
        let uses_interleaved = table_uses_interleaved_data(&suffixes);

        // Build SuffixBitsBatch for this table
        let data: SuffixBitsBatch<T> = if uses_interleaved {
            let entries: Vec<Either<u128, Rep3RingShare<T>>> = table_cycles
                .iter()
                .map(|&j| match &lookup_indices[j] {
                    Either::Public(p) => Either::Public(*p & suffix_mask),
                    Either::Shared(s) => {
                        let masked = *s & RingElement(suffix_mask);
                        Either::Shared(Rep3RingShare {
                            a: RingElement(
                                T::try_from(masked.a.0).unwrap_or_else(|_| unreachable!()),
                            ),
                            b: RingElement(
                                T::try_from(masked.b.0).unwrap_or_else(|_| unreachable!()),
                            ),
                        })
                    }
                })
                .collect();
            SuffixBitsBatch::Interleaved(MixedBatch::classify(entries))
        } else {
            // Uninterleaved: split into left/right, check right_operand_public_mask
            let n = table_cycles.len();
            let mut left_entries: Vec<Either<u64, Rep3RingShare<H<T>>>> = Vec::with_capacity(n);
            let mut right_entries: Vec<Either<u64, Rep3RingShare<H<T>>>> = Vec::with_capacity(n);

            for &j in table_cycles {
                match &lookup_indices[j] {
                    Either::Public(p) => {
                        let masked = *p & suffix_mask;
                        let mut x = 0u64;
                        let mut y = 0u64;
                        for i in 0..half_bits {
                            x |= ((masked >> (2 * i + 1)) & 1) as u64 >> 0 << i;
                            y |= ((masked >> (2 * i)) & 1) as u64 >> 0 << i;
                        }
                        left_entries.push(Either::Public(x));
                        right_entries.push(Either::Public(y));
                    }
                    Either::Shared(s) => {
                        let masked = *s & RingElement(suffix_mask);
                        let masked_t = Rep3RingShare {
                            a: RingElement(
                                T::try_from(masked.a.0).unwrap_or_else(|_| unreachable!()),
                            ),
                            b: RingElement(
                                T::try_from(masked.b.0).unwrap_or_else(|_| unreachable!()),
                            ),
                        };
                        let (x_share, y_share) = T::uninterleave(masked_t);
                        left_entries.push(Either::Shared(x_share));

                        if let Some(mask_val) = right_operand_public_mask[j] {
                            let y_pub = if half_bits >= 64 {
                                mask_val
                            } else {
                                mask_val & ((1u64 << half_bits) - 1)
                            };
                            right_entries.push(Either::Public(y_pub));
                        } else {
                            right_entries.push(Either::Shared(y_share));
                        }
                    }
                }
            }

            SuffixBitsBatch::Uninterleaved(
                MixedBatch::classify(left_entries),
                MixedBatch::classify(right_entries),
            )
        };

        // Evaluate each non-One suffix for this table
        let n = table_cycles.len();
        for (suffix_idx, suffix) in suffixes.iter().enumerate() {
            let base = batch.reserve(n);
            segments.push(EvalSegment {
                table_idx,
                suffix_idx,
                base,
                n,
            });
            evaluate_suffix_for_table::<T, F, _>(
                suffix,
                &data,
                suffix_len,
                io_ctx.main(),
                party_id,
                base,
                &mut batch,
            )?;
        }
    }
    drop(_span);

    // Fulfill all pending B2A/BitInject conversions in one batch
    let all_field = batch.fulfill_with_pool(io_ctx, pool)?;
    Ok((segments, all_field))
}

/// Build weighted histograms for all (table, suffix) pairs.
///
/// For each pair, accumulates `u[j] * suffix_eval[j]` into a size-M histogram
/// indexed by public c16 values. Returns four groups:
/// - `pub_f`: histogram is fully public F (phase 0 with constant/One suffix)
/// - `rep3`: histogram is Rep3 (no reshare needed)
/// - `additive`: histogram is additive (needs reshare before FWHT)
/// - `zero`: table has no cycles or suffix is identically zero
fn build_suffix_polys_and_additive_hists<F: JoltField>(
    eval_segments: &[EvalSegment],
    all_field: &[Rep3Value<F>],
    u_evals: &Either<Vec<F>, Vec<Rep3PrimeFieldShare<F>>>,
    c16: &[Option<u16>],
    lookup_indices_by_table: &[Vec<usize>],
    suffix_len: usize,
    ehat16: &[Rep3PrimeFieldShare<F>],
    party_id: PartyID,
) -> (
    Vec<(usize, usize, AdditiveDensePoly<F>)>,
    Vec<(usize, usize, AdditiveDensePoly<F>)>,
    Vec<(usize, usize, Vec<AdditiveShare<F>>)>,
    Vec<(usize, usize)>,
) {
    let _span = info_span!("build_histograms", n = eval_segments.len()).entered();
    let inv_m = F::from(M as u64).inverse().expect("M invertible");

    // Build lookup: (table_idx, suffix_idx) → segment in all_field
    let segment_lookup: std::collections::HashMap<(usize, usize), (usize, usize)> = eval_segments
        .par_iter()
        .map(|seg| ((seg.table_idx, seg.suffix_idx), (seg.base, seg.n)))
        .collect();

    let work_items: Vec<(usize, usize, Suffixes)> = LookupTables::<XLEN>::iter()
        .enumerate()
        .flat_map(|(ti, table)| {
            table
                .suffixes()
                .into_iter()
                .enumerate()
                .map(move |(si, s)| (ti, si, s))
        })
        .collect();

    enum HistResult<F: JoltField> {
        PublicPoly(usize, usize, AdditiveDensePoly<F>),
        Rep3Poly(usize, usize, AdditiveDensePoly<F>),
        Additive(usize, usize, Vec<AdditiveShare<F>>),
        Zero(usize, usize),
    }

    let hist_results: Vec<HistResult<F>> = match u_evals {
        Either::Public(u_pub) => {
            // Phase 0: u is public
            work_items
                .par_iter()
                .map(|&(ti, si, ref suffix)| {
                    let table_cycles = &lookup_indices_by_table[ti];
                    if table_cycles.is_empty() {
                        return HistResult::Zero(ti, si);
                    }
                    if suffix_len == 0 {
                        let constant_u64 =
                            suffix.suffix_mle::<XLEN>(LookupBits::new(0u128, 0usize));
                        if constant_u64 == 0 {
                            return HistResult::Zero(ti, si);
                        }
                        let constant_f = F::from_u128(constant_u64 as u128);
                        let mut h = vec![F::zero(); M];
                        for &j in table_cycles {
                            if let Some(c) = c16[j] {
                                h[c as usize] += u_pub[j] * constant_f;
                            }
                        }
                        let unmasked = unmask_histogram_public(&mut h, ehat16, party_id);
                        HistResult::PublicPoly(ti, si, AdditiveDensePoly::new(unmasked))
                    } else if matches!(suffix, Suffixes::One) {
                        let mut h = vec![F::zero(); M];
                        for &j in table_cycles {
                            if let Some(c) = c16[j] {
                                h[c as usize] += u_pub[j];
                            }
                        }
                        let unmasked = unmask_histogram_public(&mut h, ehat16, party_id);
                        HistResult::PublicPoly(ti, si, AdditiveDensePoly::new(unmasked))
                    } else {
                        let &(seg_base, seg_n) =
                            segment_lookup.get(&(ti, si)).expect("missing eval segment");
                        let suffix_evals = &all_field[seg_base..seg_base + seg_n];

                        let mut h = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
                        for (local, &j) in table_cycles.iter().enumerate() {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                match suffix_evals[local] {
                                    Rep3Value::Public(f) => {
                                        h[ci] =
                                            rep3_arith::add_public(h[ci], u_pub[j] * f, party_id);
                                    }
                                    Rep3Value::Shared(s) => {
                                        h[ci] += rep3_arith::mul_public(s, u_pub[j]);
                                    }
                                    Rep3Value::Additive(_) => unreachable!(),
                                }
                            }
                        }
                        let h_k = fwht_unmask_rep3_to_additive(&mut h, ehat16, inv_m);
                        HistResult::Rep3Poly(ti, si, AdditiveDensePoly::new(h_k))
                    }
                })
                .collect()
        }
        Either::Shared(u_shared) => {
            // Phase 1+: u is shared
            work_items
                .par_iter()
                .map(|&(ti, si, ref suffix)| {
                    let table_cycles = &lookup_indices_by_table[ti];
                    if table_cycles.is_empty() {
                        return HistResult::Zero(ti, si);
                    }
                    if suffix_len == 0 {
                        let constant_u64 =
                            suffix.suffix_mle::<XLEN>(LookupBits::new(0u128, 0usize));
                        if constant_u64 == 0 {
                            return HistResult::Zero(ti, si);
                        }
                        let constant_f = F::from_u128(constant_u64 as u128);
                        let mut h = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
                        for &j in table_cycles {
                            if let Some(c) = c16[j] {
                                h[c as usize] += u_shared[j] * constant_f;
                            }
                        }
                        let h_k = fwht_unmask_rep3_to_additive(&mut h, ehat16, inv_m);
                        HistResult::Rep3Poly(ti, si, AdditiveDensePoly::new(h_k))
                    } else if matches!(suffix, Suffixes::One) {
                        let mut h = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
                        for &j in table_cycles {
                            if let Some(c) = c16[j] {
                                h[c as usize] += u_shared[j];
                            }
                        }
                        let h_k = fwht_unmask_rep3_to_additive(&mut h, ehat16, inv_m);
                        HistResult::Rep3Poly(ti, si, AdditiveDensePoly::new(h_k))
                    } else {
                        let &(seg_base, seg_n) =
                            segment_lookup.get(&(ti, si)).expect("missing eval segment");
                        let suffix_evals = &all_field[seg_base..seg_base + seg_n];

                        let all_suffix_public = suffix_evals
                            .iter()
                            .all(|v| matches!(v, Rep3Value::Public(_)));
                        if all_suffix_public {
                            let mut h = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
                            for (local, &j) in table_cycles.iter().enumerate() {
                                if let Some(c) = c16[j] {
                                    let f = suffix_evals[local].as_public();
                                    h[c as usize] += u_shared[j] * f;
                                }
                            }
                            let h_k = fwht_unmask_rep3_to_additive(&mut h, ehat16, inv_m);
                            HistResult::Rep3Poly(ti, si, AdditiveDensePoly::new(h_k))
                        } else {
                            let mut h = vec![AdditiveShare::<F>::zero(); M];
                            for (local, &j) in table_cycles.iter().enumerate() {
                                if let Some(c) = c16[j] {
                                    let w: AdditiveShare<F> = match suffix_evals[local] {
                                        Rep3Value::Public(f) => u_shared[j].into_additive() * f,
                                        Rep3Value::Shared(s) => u_shared[j] * s,
                                        Rep3Value::Additive(_) => unreachable!(),
                                    };
                                    h[c as usize] = h[c as usize] + w;
                                }
                            }
                            HistResult::Additive(ti, si, h)
                        }
                    }
                })
                .collect()
        }
    };

    let mut pub_polys = Vec::new();
    let mut rep3_polys = Vec::new();
    let mut additive = Vec::new();
    let mut zero = Vec::new();
    for result in hist_results {
        match result {
            HistResult::PublicPoly(ti, si, poly) => pub_polys.push((ti, si, poly)),
            HistResult::Rep3Poly(ti, si, poly) => rep3_polys.push((ti, si, poly)),
            HistResult::Additive(ti, si, h) => additive.push((ti, si, h)),
            HistResult::Zero(ti, si) => zero.push((ti, si)),
        }
    }
    (pub_polys, rep3_polys, additive, zero)
}

/// B2A conversion for operand Q: uninterleave interleaved shares into left/right halves,
/// skip B2A for public right operands, and convert identity shares at full ring width.
///
/// Returns `(s_left, s_right, s_identity)` where:
/// - `s_left`: field share for each interleaved cycle's left operand
/// - `s_right[i]`: `Some(share)` if right is shared, `None` if right is public (mixed)
/// - `s_identity`: field share for each identity cycle
fn q_polys_b2a<T, F, N>(
    interleaved_u128: Vec<Rep3RingShare<u128>>,
    shared_right_idx: &[usize],
    identity_u128: Vec<Rep3RingShare<u128>>,
    io_ctx: &mut IoContextPool<N>,
    pool: &mut PreprocessingPool<F>,
) -> eyre::Result<(
    Vec<Rep3PrimeFieldShare<F>>,
    Vec<Option<Rep3PrimeFieldShare<F>>>,
    Vec<Rep3PrimeFieldShare<F>>,
)>
where
    T: crate::zkvm::suffixes::Uninterleavable,
    T::Half: AsPrimitive<T>,
    u128: AsPrimitive<T>,
    Standard: Distribution<T> + Distribution<T::Half>,
    F: JoltField,
    N: Rep3NetworkWorker,
{
    use mpc_core::protocols::rep3_ring::edabits;

    let n_il = interleaved_u128.len();
    let n_id = identity_u128.len();
    let chunk_size = std::env::var("READRAF_Q_B2A_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8192)
        .max(1);

    let _span = trace_span!("q_polys_b2a", n_il, n_id, chunk = chunk_size, k = T::K).entered();

    let (xs, ys): (Vec<Rep3RingShare<T::Half>>, Vec<Rep3RingShare<T::Half>>) = interleaved_u128
        .par_iter()
        .map(|b| downcast::<u128, T>(*b))
        .map(|b| T::uninterleave(b))
        .unzip();

    let s_left;
    let mut s_right: Vec<Option<Rep3PrimeFieldShare<F>>> = vec![None; n_il];
    if n_il > 0 {
        let mut lr = Vec::with_capacity(n_il + shared_right_idx.len());
        lr.extend_from_slice(&xs);
        for &i in shared_right_idx {
            lr.push(ys[i]);
        }

        let _lr = trace_span!("q_polys_b2a_lr", n = lr.len()).entered();
        let mut lr_result: Vec<Rep3PrimeFieldShare<F>> = Vec::with_capacity(lr.len());
        for lr_chunk in lr.chunks(chunk_size) {
            let _c =
                trace_span!("q_polys_b2a_chunk", kind = "lr", chunk_len = lr_chunk.len()).entered();
            let lr_batch = pool.take_edabits::<T::Half>(lr_chunk.len());
            let out = edabits::ring_to_field_b2a_many::<T::Half, F, _>(
                lr_chunk,
                &lr_batch,
                io_ctx.main(),
            )?;
            lr_result.extend(out);
        }
        drop(_lr);

        let shared_rights = lr_result.split_off(n_il);
        s_left = lr_result;
        for (idx, &i) in shared_right_idx.iter().enumerate() {
            s_right[i] = Some(shared_rights[idx]);
        }
    } else {
        s_left = vec![];
    }

    let s_identity = if n_id > 0 {
        let _id = trace_span!("q_polys_b2a_id", n = n_id).entered();
        let mut out_all: Vec<Rep3PrimeFieldShare<F>> = Vec::with_capacity(n_id);
        for id_chunk in identity_u128.chunks(chunk_size) {
            let _c =
                trace_span!("q_polys_b2a_chunk", kind = "id", chunk_len = id_chunk.len()).entered();
            let id_shares: Vec<Rep3RingShare<T>> =
                id_chunk.iter().map(|b| downcast::<u128, T>(*b)).collect();
            let id_batch = pool.take_edabits::<T>(id_shares.len());
            let out =
                edabits::ring_to_field_b2a_many::<T, F, _>(&id_shares, &id_batch, io_ctx.main())?;
            out_all.extend(out);
        }
        drop(_id);
        out_all
    } else {
        vec![]
    };

    Ok((s_left, s_right, s_identity))
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3ReadRafSumcheck<F: JoltField> {
    gamma: F,
    gamma_squared: F,
    rv_claim: F,
    raf_claim: F,
    log_T: usize,
}

impl<F: JoltField> Rep3ReadRafSumcheck<F> {
    /// Construct the coordinator-side ReadRaf sumcheck instance.
    ///
    /// Draws gamma from transcript, then computes
    /// `raf_claim = left_operand_claim + gamma * right_operand_claim`.
    pub fn new<T: Transcript>(
        transcript: &mut T,
        rv_claim: F,
        left_operand_claim: F,
        right_operand_claim: F,
        log_T: usize,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let raf_claim = left_operand_claim + gamma * right_operand_claim;
        Self {
            gamma,
            gamma_squared: gamma.square(),
            rv_claim,
            raf_claim,
            log_T,
        }
    }

    pub fn gamma(&self) -> F {
        self.gamma
    }

    pub fn rv_claim(&self) -> F {
        self.rv_claim
    }

    pub fn raf_claim(&self) -> F {
        self.raf_claim
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim_public(&self) -> F {
        self.rv_claim + self.gamma * self.raf_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let (r_address_prime, r_cycle_prime) = r.split_at(LOG_K);

        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(r_address_prime);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(r_address_prime);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(r_address_prime);

        let val_evals: Vec<_> = LookupTables::<XLEN>::iter()
            .map(|table| table.evaluate_mle::<F, F::Challenge>(r_address_prime))
            .collect();

        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        let eq_eval_cycle = EqPolynomial::<F>::mle(&r_cycle, r_cycle_prime);

        let ra_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let table_flag_claims: Vec<F> = (0..LookupTables::<XLEN>::COUNT)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::LookupTableFlag(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .collect();

        let raf_flag_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRafFlag,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let rv_val_claim: F = val_evals
            .into_iter()
            .zip(table_flag_claims)
            .map(|(val, flag)| val * flag)
            .sum();

        let val_eval = rv_val_claim
            + (F::one() - raf_flag_claim)
                * (self.gamma * left_operand_eval + self.gamma_squared * right_operand_eval)
            + raf_flag_claim * self.gamma_squared * identity_poly_eval;

        eq_eval_cycle * ra_claim * val_eval
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        let (_r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);

        let num_tables = LookupTables::<XLEN>::COUNT;
        // Claims order: table_flags..., ra, raf_flag
        assert_eq!(claims.len(), num_tables + 2);

        for i in 0..num_tables {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
                claims[i],
            );
        }

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            r_sumcheck,
            claims[num_tables],
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle,
            claims[num_tables + 1],
        );
    }
}
