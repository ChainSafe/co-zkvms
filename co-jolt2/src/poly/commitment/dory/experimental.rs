//! Ring-MSM extensions for Dory commitment scheme.
//!
//! This module is only compiled when the `ring-msm` feature is enabled.
//! It contains the ring-MSM row commitment functions, preprocessing helpers,
//! the `Rep3CommitmentScheme` impl for ring-MSM, and daPoint Q-value
//! precomputation used by the Dory commitment scheme.

use crate::poly::compact_polynomial::Rep3CompactPolynomial;
use crate::poly::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use crate::utils::types::MaybeShared;
use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
use ark_ec::CurveGroup;
use ark_ff::{AdditiveGroup, PrimeField};
use ark_std::Zero;
use jolt_core::ark_bn254::{Fr, G1Projective};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use mpc_core::preprocessing::daPoint::DaPointsBatch;
use mpc_core::preprocessing::edabits::EdaBitsRingBatch;
use mpc_core::preprocessing::wrap_mask::WrapMaskBatch;
use mpc_core::protocols::rep3;
use mpc_core::protocols::rep3::network::{IoContext, IoContextPool, Rep3Network, Rep3NetworkWorker};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring;
use mpc_core::protocols::rep3_ring::conversion as ring_conv;
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::ring::u34::U34;
use mpc_core::protocols::rep3_ring::ring::u66::U66;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rayon::prelude::*;
use std::borrow::Borrow;

use super::commitment_scheme::{
    commit_local_rep3, combine_hints_rep3_impl, compute_nu, prove_rep3_impl, rows_to_commitment,
    setup_g1_projective, DoryCarryRing,
};
pub use jolt_core::poly::commitment::dory::*;
use super::commitment_scheme::rep3_local_coeffs_a;
use super::super::Rep3CommitmentScheme;

// =============================================================================
// Rep3CommitmentScheme impl for ring-MSM feature
// =============================================================================

impl<ProofTranscript: Transcript> Rep3CommitmentScheme<Fr, ProofTranscript> for DoryCommitmentScheme {
    fn commit_rep3<N: Rep3NetworkWorker>(
        poly: &Rep3MultilinearPolynomial<Fr>,
        setup: &Self::ProverSetup,
        commit_to_public: bool,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<Fr>,
    ) -> eyre::Result<(MaybeShared<Self::Commitment>, MaybeShared<Self::OpeningProofHint>)> {
        match poly {
            // Public polys and Dense/OneHot shared polys: no IO needed.
            Rep3MultilinearPolynomial::Public(_)
            | Rep3MultilinearPolynomial::Shared(
                Rep3SharedPoly::Dense(_) | Rep3SharedPoly::OneHot(_) | Rep3SharedPoly::RLC(_),
            ) => commit_local_rep3::<ProofTranscript>(poly, setup, commit_to_public),

            // Ring-scalar shared polys: require networked MPC commit.
            Rep3MultilinearPolynomial::Shared(shared_poly @ (Rep3SharedPoly::RingScalars(_) | Rep3SharedPoly::IRingScalars(_))) => {
                let sigma = DoryGlobals::get_num_columns().log_2();
                let (num_vars, rows) = match shared_poly {
                    Rep3SharedPoly::RingScalars(poly_ring) => {
                        let nu = compute_nu(poly_ring.get_num_vars(), sigma);
                        let rp = pretake_ring_preproc(poly_ring, shared_poly, preproc)?;
                        let rows = commit_ring_poly_inner(poly_ring, setup, nu, io_ctx.main(), rp)?;
                        (poly_ring.get_num_vars(), rows)
                    }
                    Rep3SharedPoly::IRingScalars(poly_inc) => {
                        let nu = compute_nu(poly_inc.get_num_vars(), sigma);
                        let rp = pretake_ring_preproc(poly_inc, shared_poly, preproc)?;
                        let rows = commit_ring_poly_inner(poly_inc, setup, nu, io_ctx.main(), rp)?;
                        (poly_inc.get_num_vars(), rows)
                    }
                    _ => unreachable!(),
                };
                rows_to_commitment(rows, num_vars, sigma, setup)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "Dory::batch_commit")]
    fn batch_commit_rep3<U, N>(
        polys: &[U],
        setup: &Self::ProverSetup,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<Fr>,
    ) -> eyre::Result<Vec<(MaybeShared<Self::Commitment>, MaybeShared<Self::OpeningProofHint>)>>
    where
        U: Borrow<Rep3MultilinearPolynomial<Fr>> + Sync,
        N: Rep3NetworkWorker,
    {
        let party_id = io_ctx.party_id();

        // Distribute public poly commits across workers: public poly i is
        // committed by worker (i % 3). This balances work evenly.
        let per_poly_commit_public: Vec<bool> = polys
            .par_iter()
            .enumerate()
            .map(|(i, p)| {
                if matches!(p.borrow(), Rep3MultilinearPolynomial::Public(_)) {
                    i % 3 == party_id as usize
                } else {
                    false // not applicable for shared polys
                }
            })
            .collect();

        // Partition into ring-MSM polys (need io_ctx/preproc) vs local-only.
        let mut ring_idxs = Vec::new();
        let mut local_idxs = Vec::new();
        for (i, p) in polys.iter().enumerate() {
            if matches!(
                p.borrow(),
                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(_) | Rep3SharedPoly::IRingScalars(_))
            ) {
                ring_idxs.push(i);
            } else {
                local_idxs.push(i);
            }
        }

        type CommitResult = eyre::Result<(MaybeShared<DoryCommitment>, MaybeShared<DoryOpeningProofHint>)>;

        // Pre-take preprocessing for each ring poly (sequential, no IO — fast).
        let sigma = DoryGlobals::get_num_columns().log_2();
        let ring_preprocs: Vec<(usize, RingCommitPreproc)> = ring_idxs
            .iter()
            .map(|&i| {
                let (poly_compact, shared_poly) = match polys[i].borrow() {
                    Rep3MultilinearPolynomial::Shared(sp @ Rep3SharedPoly::RingScalars(p)) => (p, sp),
                    Rep3MultilinearPolynomial::Shared(sp @ Rep3SharedPoly::IRingScalars(p)) => (p, sp),
                    _ => unreachable!(),
                };
                let rp = pretake_ring_preproc(poly_compact, shared_poly, preproc)?;
                Ok((i, rp))
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        let num_ring = ring_preprocs.len();
        let avail_forks = io_ctx.max_forks().min(num_ring);

        // Run local (CPU-only) and ring (forked IO) commits concurrently.
        // Sequential fallback when insufficient forks (can't share io_ctx.main() across join).
        let (local_results, ring_results): (Vec<CommitResult>, Vec<CommitResult>) =
            if num_ring > 0 && avail_forks < num_ring {
                // Insufficient forks — sequential (same perf as before).
                let local_results: Vec<CommitResult> = local_idxs
                    .par_iter()
                    .map(|&i| commit_local_rep3::<ProofTranscript>(polys[i].borrow(), setup, per_poly_commit_public[i]))
                    .collect();
                let ring_results: Vec<CommitResult> = ring_preprocs
                    .into_iter()
                    .map(|(idx, rp)| {
                        let (poly_compact, num_vars) = match polys[idx].borrow() {
                            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(p)) => (p, p.get_num_vars()),
                            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::IRingScalars(p)) => (p, p.get_num_vars()),
                            _ => unreachable!(),
                        };
                        let nu = compute_nu(num_vars, sigma);
                        let rows = commit_ring_poly_inner(poly_compact, setup, nu, io_ctx.main(), rp)?;
                        rows_to_commitment(rows, num_vars, sigma, setup)
                    })
                    .collect();
                (local_results, ring_results)
            } else {
                // Parallel: local commits on rayon, ring commits on forked IoContexts.
                let forks = io_ctx.forks(num_ring);
                rayon::join(
                    || {
                        local_idxs
                            .par_iter()
                            .map(|&i| {
                                commit_local_rep3::<ProofTranscript>(
                                    polys[i].borrow(),
                                    setup,
                                    per_poly_commit_public[i],
                                )
                            })
                            .collect()
                    },
                    || {
                        if num_ring == 0 {
                            return Vec::new();
                        }
                        ring_preprocs
                            .into_par_iter()
                            .zip(forks.par_iter_mut())
                            .map(|((idx, rp), io)| {
                                let (poly_compact, num_vars) = match polys[idx].borrow() {
                                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(p)) => {
                                        (p, p.get_num_vars())
                                    }
                                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::IRingScalars(p)) => {
                                        (p, p.get_num_vars())
                                    }
                                    _ => unreachable!(),
                                };
                                let nu = compute_nu(num_vars, sigma);
                                let rows = commit_ring_poly_inner(poly_compact, setup, nu, io, rp)?;
                                rows_to_commitment(rows, num_vars, sigma, setup)
                            })
                            .collect()
                    },
                )
            };

        // Merge results back into original order.
        let mut out: Vec<Option<(MaybeShared<DoryCommitment>, MaybeShared<DoryOpeningProofHint>)>> =
            (0..polys.len()).map(|_| None).collect();
        for (&i, r) in local_idxs.iter().zip(local_results) {
            out[i] = Some(r?);
        }
        for (&i, r) in ring_idxs.iter().zip(ring_results.into_iter()) {
            out[i] = Some(r?);
        }
        Ok(out.into_iter().map(|o| o.unwrap()).collect())
    }

    #[tracing::instrument(skip_all, name = "Dory::prove")]
    fn prove_rep3<Network>(
        poly: &Rep3MultilinearPolynomial<Fr>,
        setup: &Self::ProverSetup,
        opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        opening_hint: Option<Self::OpeningProofHint>,
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker,
    {
        prove_rep3_impl(poly, setup, opening_point, opening_hint, network)
    }

    fn combine_hints_rep3(
        hints: Vec<MaybeShared<Self::OpeningProofHint>>,
        coeffs: &[Fr],
        party_id: PartyID,
    ) -> Self::OpeningProofHint {
        combine_hints_rep3_impl(hints, coeffs, party_id)
    }
}

// =============================================================================
// Pre-taken preprocessing for parallel ring-MSM commits
// =============================================================================

/// Pre-extracted preprocessing batches for one ring-MSM poly commitment.
///
/// Created by sequential `take_*` calls on `PreprocessingPool` before
/// the parallel commit phase, so each thread owns its own data.
pub(super) enum RingCommitPreproc {
    Ring {
        ring_edabits: EdaBitsRingBatch<DoryCarryRing>,
        wrap_masks: WrapMaskBatch<DoryCarryRing>,
        dapoints: DaPointsBatch<G1Projective>,
    },
    IRing {
        ring_edabits: EdaBitsRingBatch<U66>,
        wrap_masks: WrapMaskBatch<U66>,
        dapoints: DaPointsBatch<G1Projective>,
    },
}

// =============================================================================
// Ring-MSM helper functions
// =============================================================================

/// Count how many shared coefficients a compact polynomial has.
fn count_shared_coeffs(poly: &Rep3CompactPolynomial) -> usize {
    use crate::zkvm::instruction::types::rep3_operand::Rep3Operand;
    poly.coeffs.iter().filter(|c| matches!(c, Rep3Operand::Shared { .. })).count()
}

/// Pre-take preprocessing batches from the pool for one ring poly.
pub(super) fn pretake_ring_preproc(
    poly: &Rep3CompactPolynomial,
    shared_poly: &Rep3SharedPoly<Fr>,
    preproc: &mut PreprocessingPool<Fr>,
) -> eyre::Result<RingCommitPreproc> {
    let num_shared = count_shared_coeffs(poly);
    let n = poly.coeffs.len();
    match shared_poly {
        Rep3SharedPoly::RingScalars(_) => Ok(RingCommitPreproc::Ring {
            ring_edabits: preproc.take_ring_edabits_dory(num_shared)?,
            wrap_masks: preproc.take_wrap_masks(num_shared)?,
            dapoints: preproc.take_dapoints(2 * n)?,
        }),
        Rep3SharedPoly::IRingScalars(_) => Ok(RingCommitPreproc::IRing {
            ring_edabits: preproc.take_ring_edabits::<U66>(n)?,
            wrap_masks: preproc.take_wrap_masks_iring(n)?,
            dapoints: preproc.take_dapoints_iring(2 * n)?,
        }),
        _ => unreachable!(),
    }
}

/// Commit a ring poly using pre-taken preprocessing and a single IoContext.
pub(super) fn commit_ring_poly_inner<N: Rep3Network>(
    poly: &Rep3CompactPolynomial,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    nu: usize,
    io: &mut IoContext<N>,
    preproc: RingCommitPreproc,
) -> eyre::Result<Vec<G1Projective>> {
    match preproc {
        RingCommitPreproc::Ring { ring_edabits, wrap_masks, dapoints } => {
            compute_row_commitment_shares_ring(poly, setup, nu, io, ring_edabits, wrap_masks, &dapoints)
        }
        RingCommitPreproc::IRing { ring_edabits, wrap_masks, dapoints } => {
            compute_row_commitment_shares_iring(poly, setup, nu, io, ring_edabits, wrap_masks, &dapoints)
        }
    }
}

// =============================================================================
// daPoint Q-value precomputation
// =============================================================================

/// Precompute the full Q-point array for daPoint preprocessing of U64Scalars wrap correction.
///
/// Returns `2 * num_coeffs` points ordered to match consumption in
/// `compute_row_commitment_shares_ring`: for each row, [q0_segment, q1_segment].
///
/// Q points: `q0[c] = 2^XLEN * g1_vec[c]`, `q1[c] = 2 * q0[c]` for c in 0..num_columns.
#[tracing::instrument(skip_all)]
pub fn precompute_dapoint_qs(
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    num_coeffs: usize,
    num_columns: usize,
) -> Vec<G1Projective> {
    let (q0, q1, _, _) = precompute_dapoint_q_columns(setup, num_columns);
    expand_q_columns(&q0, &q1, num_coeffs, num_columns)
}

/// Precompute the daPoint q-values for IRingScalars (biased inc) wrap correction.
///
/// IRingScalars uses u64 scalars regardless of XLEN, so the doublings are always
/// 64 (not XLEN). Q points: `q0[c] = 2^64 * g1_vec[c]`, `q1[c] = 2 * q0[c]`.
///
/// Returns `2 * num_coeffs` points ordered to match consumption in the ring-MSM
/// commit: for each row, [q0_segment, q1_segment].
#[tracing::instrument(skip_all)]
pub fn precompute_dapoint_qs_iring(
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    num_coeffs: usize,
    num_columns: usize,
) -> Vec<G1Projective> {
    let (_, _, q0, q1) = precompute_dapoint_q_columns(setup, num_columns);
    expand_q_columns(&q0, &q1, num_coeffs, num_columns)
}

/// Precompute column-template Q vectors for both RingScalars (XLEN doublings)
/// and IRingScalars (64 doublings).
///
/// Returns `(q0_xlen, q1_xlen, q0_64, q1_64)`, each of length `num_columns`.
/// Use with [`random_dapoints_from_columns`] to avoid materializing the full Q array.
#[tracing::instrument(skip_all)]
pub fn precompute_dapoint_q_columns(
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    num_columns: usize,
) -> (Vec<G1Projective>, Vec<G1Projective>, Vec<G1Projective>, Vec<G1Projective>) {
    use jolt_common::constants::XLEN;

    let g1_proj = &setup_g1_projective(setup)[..num_columns];

    let q0_xlen: Vec<G1Projective> = g1_proj
        .iter()
        .map(|b| {
            let mut p = *b;
            for _ in 0..XLEN {
                p.double_in_place();
            }
            p
        })
        .collect();
    let q1_xlen: Vec<G1Projective> = q0_xlen.iter().map(|p| *p + *p).collect();

    let q0_64: Vec<G1Projective> = g1_proj
        .iter()
        .map(|b| {
            let mut p = *b;
            for _ in 0..64 {
                p.double_in_place();
            }
            p
        })
        .collect();
    let q1_64: Vec<G1Projective> = q0_64.iter().map(|p| *p + *p).collect();

    (q0_xlen, q1_xlen, q0_64, q1_64)
}

fn expand_q_columns(
    q0: &[G1Projective],
    q1: &[G1Projective],
    num_coeffs: usize,
    num_columns: usize,
) -> Vec<G1Projective> {
    let mut all_q = Vec::with_capacity(2 * num_coeffs);
    let num_full_rows = num_coeffs / num_columns;
    let remainder = num_coeffs % num_columns;
    for _ in 0..num_full_rows {
        all_q.extend_from_slice(&q0[..num_columns]);
        all_q.extend_from_slice(&q1[..num_columns]);
    }
    if remainder > 0 {
        all_q.extend_from_slice(&q0[..remainder]);
        all_q.extend_from_slice(&q1[..remainder]);
    }
    all_q
}

// =============================================================================
// Core ring-MSM row commitment functions
// =============================================================================

/// Compute row commitment shares for a RingScalars polynomial.
///
/// Public coefficients (NoOp padding, immediates) skip ring B2A, wrap extraction,
/// and daPoint correction — only shared coefficients consume MPC preprocessing.
#[tracing::instrument(skip_all, name = "dense::commit_rows_ring", level = "trace")]
fn compute_row_commitment_shares_ring<N: Rep3Network>(
    poly: &Rep3CompactPolynomial,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    nu: usize,
    io: &mut mpc_core::protocols::rep3::network::IoContext<N>,
    ring_edabits: EdaBitsRingBatch<DoryCarryRing>,
    wrap_masks: WrapMaskBatch<DoryCarryRing>,
    dapoints: &DaPointsBatch<G1Projective>,
) -> eyre::Result<Vec<G1Projective>> {
    use crate::zkvm::instruction::types::rep3_operand::Rep3Operand;
    use jolt_common::constants::{XlenInt, XLEN};
    use mpc_core::protocols::rep3_ring::casts::downcast;

    let sigma = DoryGlobals::get_num_columns().log_2();
    let num_columns = 1usize << sigma;
    let num_rows_target = 1usize << nu;

    let g1_proj = &setup_g1_projective(setup)[..num_columns];
    let bases_aff = G1Projective::normalize_batch(g1_proj);

    let n = poly.coeffs.len();
    if n == 0 {
        return Ok(vec![G1Projective::zero(); num_rows_target]);
    }

    let party_id = io.id;

    // Partition coefficients: extract shared-only binary + arithmetic shares,
    // and build a position map from global index → shared index.
    let mut shared_pos_map: Vec<Option<usize>> = vec![None; n];
    let mut shared_bins: Vec<Rep3RingShare<XlenInt>> = Vec::new();
    let mut shared_ariths: Vec<Rep3RingShare<XlenInt>> = Vec::new();
    for (i, coeff) in poly.coeffs.iter().enumerate() {
        if let Rep3Operand::Shared { binary, arithmetic, .. } = coeff {
            shared_pos_map[i] = Some(shared_bins.len());
            shared_bins.push(*binary);
            shared_ariths.push(downcast(arithmetic.unwrap()));
        }
    }
    let num_shared = shared_bins.len();

    // Ring B2A + wrap extraction only for shared coefficients.
    let (m0_bin, m1_bin) = if num_shared > 0 {
        // Zero-extend shared xlen value → Dory carry ring.
        let arith_ext: Vec<Rep3RingShare<DoryCarryRing>> = shared_ariths
            .iter()
            .map(|s| Rep3RingShare {
                a: RingElement(DoryCarryRing::try_from(s.a.0 as u128).expect("xlen share fits in Dory carry ring")),
                b: RingElement(DoryCarryRing::try_from(s.b.0 as u128).expect("xlen share fits in Dory carry ring")),
            })
            .collect();
        let bin_ext: Vec<Rep3RingShare<DoryCarryRing>> = shared_bins
            .iter()
            .map(|s| Rep3RingShare {
                a: RingElement(DoryCarryRing::try_from(s.a.0 as u128).expect("xlen share fits in Dory carry ring")),
                b: RingElement(DoryCarryRing::try_from(s.b.0 as u128).expect("xlen share fits in Dory carry ring")),
            })
            .collect();

        // Ring B2A via edaBits Π₂ — 2 rounds
        let val_arith: Vec<Rep3RingShare<DoryCarryRing>> =
            rep3_ring::conversion::b2a_preproc_many(&bin_ext, &ring_edabits, io)?;
        let diff: Vec<Rep3RingShare<DoryCarryRing>> =
            arith_ext.iter().zip(val_arith.iter()).map(|(a, v)| *a - *v).collect();

        // Extract m bits via DaBit mask+open (1 round)
        let (m0, m1) = rep3_ring::wrap_mask::extract_wrap_m2_from_diff_many(&diff, &wrap_masks, io)?;
        (m0, m1)
    } else {
        (Vec::new(), Vec::new())
    };

    // Precompute q0/q1 for all columns: q0[c] = 2^XLEN*Γ1[c], q1[c] = 2*q0[c].
    let mut q0_cols: Vec<G1Projective> = Vec::with_capacity(num_columns);
    for b in g1_proj.iter() {
        let mut p = *b;
        for _ in 0..XLEN {
            p.double_in_place();
        }
        q0_cols.push(p);
    }
    let q1_cols: Vec<G1Projective> = q0_cols.iter().map(|p| *p + *p).collect();

    // MSM + correction, computed per row segment.
    let mut row_commitments = vec![G1Projective::zero(); num_rows_target];
    let last_row = (n - 1) / num_columns;
    let mut dp_offset = 0usize;

    for row in 0..=last_row {
        let row_start = row * num_columns;
        let seg_end = n.min(row_start + num_columns);
        let seg_len = seg_end - row_start;
        if seg_len == 0 {
            continue;
        }
        let local_start = row_start;

        // Build MSM scalars: shared → a-limb, public → trivial share for ID0.
        let scalars_u64: Vec<u64> = poly.coeffs[local_start..local_start + seg_len]
            .iter()
            .map(|op| match op {
                Rep3Operand::Shared { arithmetic, .. } => {
                    let arith_xlen: Rep3RingShare<XlenInt> = downcast(arithmetic.unwrap());
                    arith_xlen.a.0 as u64
                }
                Rep3Operand::Public(v) => {
                    if party_id == PartyID::ID0 {
                        (*v as XlenInt) as u64
                    } else {
                        0
                    }
                }
            })
            .collect();
        let msm: G1Projective = ArkVariableBaseMSM::msm_u64(&bases_aff[..seg_len], &scalars_u64, false);

        // daPoint correction only for shared coefficients in this segment.
        let batch = dapoints.slice(dp_offset, 2 * seg_len);
        dp_offset += 2 * seg_len;

        let mut bits_all: Vec<Rep3RingShare<Bit>> = Vec::new();
        let mut q_all: Vec<G1Projective> = Vec::new();
        let mut dp_selected: Vec<usize> = Vec::new();
        for seg_i in 0..seg_len {
            if let Some(shared_idx) = shared_pos_map[local_start + seg_i] {
                bits_all.push(m0_bin[shared_idx]);
                q_all.push(q0_cols[seg_i]);
                dp_selected.push(seg_i);
            }
        }
        for seg_i in 0..seg_len {
            if let Some(shared_idx) = shared_pos_map[local_start + seg_i] {
                bits_all.push(m1_bin[shared_idx]);
                q_all.push(q1_cols[seg_i]);
                dp_selected.push(seg_len + seg_i);
            }
        }

        if !bits_all.is_empty() {
            let filtered_batch = batch.select(&dp_selected);
            let corr_add = rep3::pointshare::dot_product_dapoints(&bits_all, &q_all, &filtered_batch, io)?;
            if row < row_commitments.len() {
                row_commitments[row] += msm - corr_add;
            }
        } else if row < row_commitments.len() {
            row_commitments[row] += msm;
        }
    }

    Ok(row_commitments)
}

/// Compute row commitment shares for an IRingScalars polynomial (biased inc, u64 scalars).
///
/// All coefficients are Shared (biased_inc = post - pre + 2^XLEN, always non-negative).
/// Uses U66 carry ring for wrap correction, 64-bit q doublings, and per-row bias correction
/// to account for the public 2^XLEN bias added to each scalar.
///
/// After MSM + wrap correction, each row subtracts: `2^XLEN * Σ bases[col_in_row]`.
#[tracing::instrument(skip_all, name = "dense::commit_rows_iring", level = "trace")]
fn compute_row_commitment_shares_iring<N: Rep3Network>(
    poly: &Rep3CompactPolynomial,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    nu: usize,
    io: &mut mpc_core::protocols::rep3::network::IoContext<N>,
    ring_edabits: EdaBitsRingBatch<U66>,
    wrap_masks: WrapMaskBatch<U66>,
    dapoints: &DaPointsBatch<G1Projective>,
) -> eyre::Result<Vec<G1Projective>> {
    use crate::zkvm::instruction::types::rep3_operand::Rep3Operand;
    use jolt_common::constants::XLEN;

    let sigma = DoryGlobals::get_num_columns().log_2();
    let num_columns = 1usize << sigma;
    let num_rows_target = 1usize << nu;

    let g1_proj = &setup_g1_projective(setup)[..num_columns];
    let bases_aff = G1Projective::normalize_batch(g1_proj);

    let n = poly.coeffs.len();
    if n == 0 {
        return Ok(vec![G1Projective::zero(); num_rows_target]);
    }

    let party_id = io.id;

    // Extract arithmetic u64 shares from all coefficients.
    let ariths_u64: Vec<Rep3RingShare<u64>> = poly
        .coeffs
        .iter()
        .map(|op| match op {
            Rep3Operand::Shared { arithmetic, .. } => {
                let wide = arithmetic.expect("IRingScalars: missing arithmetic share");
                Rep3RingShare { a: RingElement(wide.a.0 as u64), b: RingElement(wide.b.0 as u64) }
            }
            Rep3Operand::Public(_) => {
                unreachable!("IRingScalars should not contain Public operands")
            }
        })
        .collect();

    // A2B: arithmetic u64 → binary u64 (1 comm round)
    let bins_u64: Vec<Rep3RingShare<u64>> = ring_conv::a2b_many(&ariths_u64, io)?;

    // Zero-extend to U66 carry ring for B2A + wrap extraction.
    let arith_ext: Vec<Rep3RingShare<U66>> = ariths_u64
        .iter()
        .map(|s| Rep3RingShare {
            a: RingElement(U66::try_from(s.a.0 as u128).expect("u64 share fits in U66")),
            b: RingElement(U66::try_from(s.b.0 as u128).expect("u64 share fits in U66")),
        })
        .collect();
    let bin_ext: Vec<Rep3RingShare<U66>> = bins_u64
        .iter()
        .map(|s| Rep3RingShare {
            a: RingElement(U66::try_from(s.a.0 as u128).expect("u64 share fits in U66")),
            b: RingElement(U66::try_from(s.b.0 as u128).expect("u64 share fits in U66")),
        })
        .collect();

    // Ring B2A via edaBits Π₂ in U66 — 2 rounds
    let val_arith: Vec<Rep3RingShare<U66>> = rep3_ring::conversion::b2a_preproc_many(&bin_ext, &ring_edabits, io)?;
    let diff: Vec<Rep3RingShare<U66>> = arith_ext.iter().zip(val_arith.iter()).map(|(a, v)| *a - *v).collect();

    // Extract m bits via DaBit mask+open (1 round)
    let (m0_bin, m1_bin) = rep3_ring::wrap_mask::extract_wrap_m2_from_diff_many(&diff, &wrap_masks, io)?;

    // Precompute q0/q1 for all columns: q0[c] = 2^64 * Γ1[c], q1[c] = 2 * q0[c].
    let mut q0_cols: Vec<G1Projective> = Vec::with_capacity(num_columns);
    for b in g1_proj.iter() {
        let mut p = *b;
        for _ in 0..64 {
            p.double_in_place();
        }
        q0_cols.push(p);
    }
    let q1_cols: Vec<G1Projective> = q0_cols.iter().map(|p| *p + *p).collect();

    // Precompute bias correction base per column: bias_base[c] = 2^XLEN * Γ1[c].
    let bias_bases: Vec<G1Projective> = g1_proj
        .iter()
        .map(|b| {
            let mut p = *b;
            for _ in 0..XLEN {
                p.double_in_place();
            }
            p
        })
        .collect();

    // MSM + wrap correction + bias correction, per row segment.
    let mut row_commitments = vec![G1Projective::zero(); num_rows_target];
    let last_row = (n - 1) / num_columns;
    let mut dp_offset = 0usize;

    for row in 0..=last_row {
        let row_start = row * num_columns;
        let seg_end = n.min(row_start + num_columns);
        let seg_len = seg_end - row_start;
        if seg_len == 0 {
            continue;
        }

        let scalars_u64: Vec<u64> = ariths_u64[row_start..row_start + seg_len].iter().map(|s| s.a.0).collect();
        let msm: G1Projective = ArkVariableBaseMSM::msm_u64(&bases_aff[..seg_len], &scalars_u64, false);

        // daPoint wrap correction — all positions are shared (no filtering needed).
        let batch = dapoints.slice(dp_offset, 2 * seg_len);
        dp_offset += 2 * seg_len;

        let mut bits_all: Vec<Rep3RingShare<Bit>> = Vec::with_capacity(2 * seg_len);
        let mut q_all: Vec<G1Projective> = Vec::with_capacity(2 * seg_len);
        for seg_i in 0..seg_len {
            bits_all.push(m0_bin[row_start + seg_i]);
            q_all.push(q0_cols[seg_i]);
        }
        for seg_i in 0..seg_len {
            bits_all.push(m1_bin[row_start + seg_i]);
            q_all.push(q1_cols[seg_i]);
        }
        let corr_add = rep3::pointshare::dot_product_dapoints(&bits_all, &q_all, &batch, io)?;

        let bias_correction: G1Projective =
            if party_id == PartyID::ID0 { bias_bases[..seg_len].iter().copied().sum() } else { G1Projective::zero() };

        if row < row_commitments.len() {
            row_commitments[row] += msm - corr_add - bias_correction;
        }
    }

    Ok(row_commitments)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::compact_polynomial::Rep3CompactPolynomial;
    use crate::poly::{Rep3MultilinearPolynomial, Rep3SharedPoly};
    use crate::utils::types::MaybeShared;
    use jolt_core::ark_bn254::{Fr, G1Projective};
    use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
    use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
    use jolt_core::transcripts::Blake2bTranscript;
    use jolt_core::utils::math::Math;
    use mpc_core::protocols::rep3::network::IoContextPool;
    use mpc_core::protocols::rep3::test_utils::{run_rep3_local_test_with_coordinator, LocalRep3TestWorkerNet};
    use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
    use mpc_core::protocols::rep3_ring;
    use mpc_core::protocols::rep3_ring::conversion as ring_conv;
    use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
    use mpc_core::protocols::rep3_ring::ring::bit::Bit;
    use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
    use mpc_core::protocols::rep3_ring::Rep3RingShare;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;
    use super::super::super::Rep3CommitmentScheme;
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
    use ark_ec::CurveGroup;
    use ark_std::Zero;
    use mpc_core::protocols::rep3;
    use jolt_core::poly::commitment::dory::JoltGroupWrapper;

    #[test]
    fn dory_u64_scalars_commit_correct() {
        use jolt_common::constants::{ArithmeticWideInt, XlenInt};

        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let num_columns = DoryGlobals::get_num_columns();
        let sigma = num_columns.log_2();
        let num_vars = sigma;
        let num_rows = DoryGlobals::get_max_num_rows();

        let len = 1usize << num_vars;
        let values: Vec<u64> = (0..len).map(|_| rng.r#gen::<XlenInt>() as u64).collect();
        let coeffs_fr: Vec<Fr> = values.iter().copied().map(Fr::from).collect();

        let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        let public_poly = MultilinearPolynomial::from(coeffs_fr.clone());
        let (vanilla_commitment, mut vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);
        vanilla_hint.resize(num_rows, JoltGroupWrapper(G1Projective::zero()));

        // Share each value in both arithmetic (ArithmeticWideInt) and XOR (XlenInt) ring forms.
        let all_arith_shares: Vec<_> = values
            .iter()
            .map(|&v| rep3_ring::share_ring_element(RingElement(v as ArithmeticWideInt), &mut rng))
            .collect();
        let all_bin_shares: Vec<_> =
            values.iter().map(|&v| rep3_ring::share_ring_element_binary(RingElement(v as XlenInt), &mut rng)).collect();

        let polys_by_party: [Rep3MultilinearPolynomial<Fr>; 3] = std::array::from_fn(|pid| {
            let shares: Vec<Rep3RingShare<ArithmeticWideInt>> = all_arith_shares.iter().map(|s| s[pid]).collect();
            let shares_bin: Vec<Rep3RingShare<XlenInt>> = all_bin_shares.iter().map(|s| s[pid]).collect();
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(Rep3CompactPolynomial::from_shares(
                shares, shares_bin,
            )))
        });

        let (results, _) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| polys_by_party[party_idx].clone(),
            || (),
            move |poly, mut io_ctx| {
                use mpc_core::protocols::rep3_ring::edabits;

                let pool_dir = std::env::temp_dir().join(format!("co-jolt2-dory-test-{}", io_ctx.party_idx()));
                let mut preproc =
                    edabits::preprocess_pool::<Fr, _>(&pool_dir, [0, 0, 0, 0, 0], 0, len, len, 0, &mut io_ctx)?;

                // daPoints for Dory wrap correction (depend on SRS)
                let qs = precompute_dapoint_qs(&setup, len, num_columns);
                let lazy_dp = rep3_ring::daPoint::random_dapoints(&qs, &mut io_ctx)?;
                preproc.set_dapoints(lazy_dp);

                let polys = vec![&poly];
                let out = <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::batch_commit_rep3(
                    &polys,
                    &setup,
                    &mut io_ctx,
                    &mut preproc,
                )?;
                Ok(out[0].clone())
            },
            |(), _net| Ok(()),
        );

        let (c0, h0) = results[0].clone();
        let (c1, h1) = results[1].clone();
        let (c2, h2) = results[2].clone();

        let reconstructed_commitment =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_commitment_shares(&[&c0, &c1, &c2]);

        let reconstructed_hint =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_hint_shares(&[&h0, &h1, &h2]);

        assert_eq!(reconstructed_commitment, vanilla_commitment);
        assert_eq!(reconstructed_hint, vanilla_hint);
    }

    /// Same as `dory_u64_scalars_commit_correct` but with mixed Public + Shared operands.
    /// Exercises the daPoint Q-value alignment when public positions are skipped.
    #[test]
    fn dory_u64_scalars_mixed_public_shared_commit_correct() {
        use crate::zkvm::instruction::types::rep3_operand::Rep3Operand;
        use jolt_common::constants::{ArithmeticWideInt, XlenInt};
        use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;

        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let num_columns = DoryGlobals::get_num_columns();
        let sigma = num_columns.log_2();
        let num_vars = sigma;
        let num_rows = DoryGlobals::get_max_num_rows();

        let len = 1usize << num_vars;
        // Every other value is public (even indices), the rest are shared.
        let values: Vec<u64> = (0..len).map(|_| rng.r#gen::<XlenInt>() as u64).collect();
        let is_public: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
        let coeffs_fr: Vec<Fr> = values.iter().copied().map(Fr::from).collect();

        let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        let public_poly = MultilinearPolynomial::from(coeffs_fr.clone());
        let (vanilla_commitment, mut vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);
        vanilla_hint.resize(num_rows, JoltGroupWrapper(G1Projective::zero()));

        // Share shared values in both arithmetic and XOR ring forms.
        let all_arith_shares: Vec<Option<[Rep3RingShare<ArithmeticWideInt>; 3]>> = values
            .iter()
            .zip(is_public.iter())
            .map(|(&v, &pub_)| {
                if pub_ {
                    None
                } else {
                    Some(rep3_ring::share_ring_element(RingElement(v as ArithmeticWideInt), &mut rng))
                }
            })
            .collect();
        let all_bin_shares: Vec<Option<[Rep3RingShare<XlenInt>; 3]>> = values
            .iter()
            .zip(is_public.iter())
            .map(|(&v, &pub_)| {
                if pub_ {
                    None
                } else {
                    Some(rep3_ring::share_ring_element_binary(RingElement(v as XlenInt), &mut rng))
                }
            })
            .collect();

        let polys_by_party: [Rep3MultilinearPolynomial<Fr>; 3] = std::array::from_fn(|pid| {
            let operands: Vec<Rep3Operand> = (0..len)
                .map(|i| {
                    if is_public[i] {
                        Rep3Operand::Public(values[i] as i128)
                    } else {
                        let arith = all_arith_shares[i].as_ref().unwrap()[pid];
                        let bin = all_bin_shares[i].as_ref().unwrap()[pid];
                        Rep3Operand::Shared { binary: bin, arithmetic: Some(arith), public: None }
                    }
                })
                .collect();
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(Rep3CompactPolynomial::from_operands(
                operands,
            )))
        });

        let (results, _) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| polys_by_party[party_idx].clone(),
            || (),
            move |poly, mut io_ctx| {
                use mpc_core::protocols::rep3_ring::edabits;

                let pool_dir = std::env::temp_dir().join(format!("co-jolt2-dory-test2-{}", io_ctx.party_idx()));
                let mut preproc =
                    edabits::preprocess_pool::<Fr, _>(&pool_dir, [0, 0, 0, 0, 0], 0, len, len, 0, &mut io_ctx)?;

                // daPoints (depend on SRS)
                let qs = precompute_dapoint_qs(&setup, len, num_columns);
                let lazy_dp = rep3_ring::daPoint::random_dapoints(&qs, &mut io_ctx)?;
                preproc.set_dapoints(lazy_dp);

                let polys = vec![&poly];
                let out = <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::batch_commit_rep3(
                    &polys,
                    &setup,
                    &mut io_ctx,
                    &mut preproc,
                )?;
                Ok(out[0].clone())
            },
            |(), _net| Ok(()),
        );

        let (c0, h0) = results[0].clone();
        let (c1, h1) = results[1].clone();
        let (c2, h2) = results[2].clone();

        let reconstructed_commitment =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_commitment_shares(&[&c0, &c1, &c2]);

        let reconstructed_hint =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_hint_shares(&[&h0, &h1, &h2]);

        assert_eq!(reconstructed_commitment, vanilla_commitment);
        assert_eq!(reconstructed_hint, vanilla_hint);
    }

    /// Verify MPC wrap correction for arithmetic u32 shares embedded into Fr for MSM.
    /// Uses both arithmetic and binary u32 shares: B2A + subtract → open wrap count m → public correction.
    #[test]
    fn ring_shared_msm_correctness() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let n = 64;

        // Random G1 bases
        let bases_proj: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bases_aff = G1Projective::normalize_batch(&bases_proj);

        // Small random coefficients
        let values: Vec<u32> = (0..n).map(|_| rng.r#gen()).collect();

        // True MSM
        let scalars_fr: Vec<Fr> = values.iter().map(|&v| Fr::from(v as u32)).collect();
        let true_msm: G1Projective = ArkVariableBaseMSM::msm(&bases_aff, &scalars_fr).expect("true MSM should succeed");

        // --- MPC correction path ---
        let all_arith_shares: Vec<_> =
            values.iter().map(|&v| rep3_ring::share_ring_element(RingElement(v), &mut rng)).collect();
        let all_bin_shares: Vec<_> =
            values.iter().map(|&v| rep3_ring::share_ring_element_binary(RingElement(v), &mut rng)).collect();

        // Naive sum with arithmetic shares is also wrong
        let naive_arith_msms: [G1Projective; 3] = std::array::from_fn(|pid| {
            let scalars: Vec<Fr> = all_arith_shares.iter().map(|s| Fr::from(s[pid].a.0)).collect();
            ArkVariableBaseMSM::msm(&bases_aff, &scalars).unwrap()
        });
        let naive_arith_sum = naive_arith_msms[0] + naive_arith_msms[1] + naive_arith_msms[2];
        assert_ne!(naive_arith_sum, true_msm, "naive arithmetic sum should differ from true MSM");

        let (mpc_results, _) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| {
                let my_arith: Vec<Rep3RingShare<u32>> = all_arith_shares.iter().map(|s| s[party_idx]).collect();
                let my_bin: Vec<Rep3RingShare<u32>> = all_bin_shares.iter().map(|s| s[party_idx]).collect();
                (my_arith, my_bin, bases_aff.clone(), bases_proj.clone())
            },
            || (),
            |(arith_u32, bin_u32, bases_aff, bases_proj), mut io_ctx| {
                let arith_ext: Vec<Rep3RingShare<u64>> = arith_u32
                    .iter()
                    .map(|s| Rep3RingShare { a: RingElement(s.a.0 as u64), b: RingElement(s.b.0 as u64) })
                    .collect();

                let bin_ext: Vec<Rep3RingShare<u64>> = bin_u32
                    .iter()
                    .map(|s| Rep3RingShare { a: RingElement(s.a.0 as u64), b: RingElement(s.b.0 as u64) })
                    .collect();

                let val_arith: Vec<Rep3RingShare<u64>> = ring_conv::b2a_many(&bin_ext, io_ctx.main())?;

                let diff: Vec<Rep3RingShare<u64>> =
                    arith_ext.iter().zip(val_arith.iter()).map(|(a, v)| *a - *v).collect();

                let diff_bin: Vec<Rep3RingShare<u64>> = ring_conv::a2b_many(&diff, io_ctx.main())?;
                let m_bin_u64: Vec<Rep3RingShare<u64>> = diff_bin.iter().map(|d| d >> 32).collect();
                let m0_bin: Vec<Rep3RingShare<Bit>> = m_bin_u64.iter().map(|m| m.get_bit(0)).collect();
                let m1_bin: Vec<Rep3RingShare<Bit>> = m_bin_u64.iter().map(|m| m.get_bit(1)).collect();

                let scalars: Vec<Fr> = arith_u32.iter().map(|s| Fr::from(s.a.0)).collect();
                let party_msm: G1Projective = ArkVariableBaseMSM::msm(&bases_aff, &scalars).unwrap();

                let two_pow_32 = Fr::from(1u64 << 32);
                let q0: Vec<G1Projective> = bases_proj.iter().map(|b| *b * two_pow_32).collect();
                let q1: Vec<G1Projective> = q0.iter().map(|p| *p + *p).collect();
                let mut q_all: Vec<G1Projective> = Vec::with_capacity(2 * q0.len());
                q_all.extend(q0.iter().copied());
                q_all.extend(q1.iter().copied());

                let mut lazy_dapoints = rep3_ring::daPoint::random_dapoints(&q_all, &mut io_ctx)?;
                let batch = lazy_dapoints.take_batch(q_all.len())?;

                let mut bits_all: Vec<Rep3RingShare<Bit>> = Vec::with_capacity(2 * m0_bin.len());
                bits_all.extend(m0_bin.iter().copied());
                bits_all.extend(m1_bin.iter().copied());

                let total_corr_add = rep3::pointshare::dot_product_dapoints(&bits_all, &q_all, &batch, io_ctx.main())?;

                Ok(party_msm - total_corr_add)
            },
            |(), _net| Ok(()),
        );

        let mpc_sum = mpc_results[0] + mpc_results[1] + mpc_results[2];
        assert_eq!(mpc_sum, true_msm, "MPC-corrected MSM must equal true MSM");
    }
}
