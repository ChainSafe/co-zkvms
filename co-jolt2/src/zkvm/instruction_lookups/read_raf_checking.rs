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

    let _span = trace_span!(
        "reshare_hists_chunked",
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
// Worker
// ---------------------------------------------------------------------------

/// MPC prover state for the ReadRaf sumcheck.
///
/// Mirrors vanilla `ReadRafProverState` but replaces plaintext lookup indices
/// with masked-index + FWHT-based unmasking from the RandOHV witness.
struct ReadRafProverState<F: JoltField> {
    // -- Round tracking --
    r: Vec<F::Challenge>,

    // -- Per-cycle classification (all public) --
    /// Per-cycle lookup table variant (public; derived from opcode).
    lookup_tables: Vec<Option<LookupTables<XLEN>>>,
    /// Indices of cycles grouped by table type.
    lookup_indices_by_table: Vec<Vec<usize>>,
    /// Indices of cycles using interleaved operands (raf path).
    interleaved_cycles: Vec<usize>,
    /// Indices of cycles NOT using interleaved operands (identity path).
    identity_cycles: Vec<usize>,
    /// Per-cycle interleaved flag — used only in init_log_t_rounds for padding cycles
    /// that are outside interleaved_cycles/identity_cycles but still need combined_val_poly
    /// RAF contributions (because they may be paired with real cycles during sumcheck eval).
    is_interleaved_operands: Vec<bool>,

    // -- Shared per-cycle accumulators --
    /// u_evals[j] starts as public eq(r_cycle, j), becomes shared after first condensation.
    /// Either::Public = phase 0 (no mul_vec needed), Either::Shared = phase 1+.
    u_evals: Either<Vec<F>, Vec<Rep3PrimeFieldShare<F>>>,
    /// ra_acc[j] accumulates eq(k_j, r_address) across phases.
    /// None = phase 0 (all 1s for active, 0 for padding), Some = phase 1+.
    ra_acc: Option<Vec<Rep3PrimeFieldShare<F>>>,
    /// After LOG_K rounds, ra becomes a dense polynomial for the log_T rounds.
    ra: Option<Rep3DensePolynomial<F>>,

    // -- Phase polynomials (length M = 65536) --
    /// Current logical length of suffix polynomials (halves on each bind in address rounds).
    suffix_poly_len: usize,
    /// Per-table suffix polynomials stored as additive shares (no reshare needed).
    /// Built via local `Rep3 * Rep3 → Additive` in FWHT unmasking.
    suffix_polys: Vec<Vec<Option<AdditiveDensePoly<F>>>>,
    /// Additive Q arrays for operand prefix-suffix decompositions.
    left_operand_q: [AdditiveDensePoly<F>; 2],
    right_operand_q: [AdditiveDensePoly<F>; 2],
    identity_q: [AdditiveDensePoly<F>; 2],
    /// Public prefix-suffix decompositions — used only for P polynomial management.
    /// Their internal Q arrays are unused (we use our shared Q arrays above).
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    identity_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix checkpoints and registry (all public).
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    prefix_registry: PrefixRegistry<F>,
    /// Expanding EQ table for bound challenges (public).
    v: ExpandingTable<F>,

    // -- Lookup index data from witness --
    /// Full 128-bit lookup indices per cycle.
    /// `Either::Public` for control-only instructions, `Either::Shared` for secret operands.
    /// Used to extract suffix bits per phase.
    lookup_indices: Vec<Either<u128, Rep3RingShare<u128>>>,

    /// Per-cycle optional public right-operand bitmask (for SignExtension shortcut).
    /// `Some(mask)` for VirtualSRA/SRL/SRAI/SRLI cycles where the shift bitmask is public.
    right_operand_public_mask: Vec<Option<u64>>,

    // -- Mask data from witness --
    /// The D Rep3OneHotPolynomials (owned; reused across phases).
    one_hot_polys: Arc<[Rep3OneHotPolynomial<F>; D]>,
    /// Per-phase cached Ehat16 (FWHT of E16 tensor product). Length M.
    ehat16: Option<Vec<Rep3PrimeFieldShare<F>>>,
    /// Per-phase cached c16[j] (public masked 16-bit keys). None means inactive cycle.
    c16: Vec<Option<u16>>,
    // -- Cycle-round data (built after LOG_K rounds) --
    eq_r_cycle: MultilinearPolynomial<F>,
    combined_val_polynomial: Option<MultilinearPolynomial<F>>,

    party_id: PartyID,
}

pub struct Rep3ReadRafSumcheckWorker<F: JoltField, N: Rep3NetworkWorker> {
    gamma: F,
    gamma_squared: F,
    rv_claim: F,
    raf_claim: F,
    log_T: usize,
    state: ReadRafProverState<F>,
    _phantom: std::marker::PhantomData<N>,
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3ReadRafSumcheckWorker<F, N> {
    /// Construct the ReadRaf worker.
    ///
    /// # Arguments
    /// - `gamma`: the ReadRaf challenge (drawn by coordinator, broadcast to workers)
    /// - `rv_claim`, `raf_claim`: the virtual polynomial claims from Spartan outer sumcheck
    /// - `one_hot_polys`: the D Rep3OneHotPolynomials from witness generation
    /// - `eq_r_cycle_public`: eq(r_cycle, ·) evaluations (public, length T)
    /// - `lookup_tables`: per-cycle lookup table variant (from `cycle_witness`)
    /// - `is_interleaved_operands`: per-cycle interleaved flag (from `cycle_witness`)
    /// - `io_ctx`: IoContextPool for MPC operations during phase transitions
    /// - `party_id`: this party's ID
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "ReadRaf::new")]
    pub fn new(
        gamma: F,
        rv_claim: F,
        raf_claim: F,
        one_hot_polys: Arc<[Rep3OneHotPolynomial<F>; D]>,
        eq_r_cycle_public: &[F],
        lookup_tables: Vec<Option<LookupTables<XLEN>>>,
        is_interleaved_operands: Vec<bool>,
        lookup_indices: Vec<Either<u128, Rep3RingShare<u128>>>,
        right_operand_public_mask: Vec<Option<u64>>,
        io_ctx: &mut IoContextPool<N>,
        party_id: PartyID,
        preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<Self> {
        let num_cycles = lookup_tables.len();
        eyre::ensure!(
            num_cycles.is_power_of_two(),
            "ReadRaf requires power-of-two number of cycles, got {num_cycles}"
        );
        eyre::ensure!(
            eq_r_cycle_public.len() == num_cycles,
            "eq_r_cycle_public length mismatch: expected {num_cycles}, got {}",
            eq_r_cycle_public.len()
        );
        eyre::ensure!(PHASES * 2 == D, "expected PHASES*2 == D");
        eyre::ensure!(
            lookup_indices.len() == num_cycles,
            "lookup_indices length mismatch: expected {num_cycles}, got {}",
            lookup_indices.len()
        );
        for (i, ohv) in one_hot_polys.iter().enumerate() {
            eyre::ensure!(
                ohv.masked_indices_c.len() == num_cycles,
                "one_hot_polys[{i}].masked_indices_c length mismatch"
            );
            eyre::ensure!(
                ohv.rand_ohv_e_field.len() == 256,
                "one_hot_polys[{i}].rand_ohv_e_field must have length 256"
            );
        }
        let log_T = num_cycles.log_2();
        let num_tables = LookupTables::<XLEN>::COUNT;

        let mut lookup_indices_by_table: Vec<Vec<usize>> = vec![Vec::new(); num_tables];
        let mut interleaved_cycles = Vec::new();
        let mut identity_cycles = Vec::new();

        for (j, (table, &is_interleaved)) in lookup_tables
            .iter()
            .zip(is_interleaved_operands.iter())
            .enumerate()
        {
            if let Some(t) = table {
                let idx = LookupTables::<XLEN>::enum_index(t);
                lookup_indices_by_table[idx].push(j);
            }
            // Only classify cycles that have a valid one-hot entry (not padding NoOps).
            // Padding cycles have masked_indices_c[j] = None and c16[j] = None,
            // so they'd be skipped by the histogram scatter guard anyway, but§
            // filtering here avoids iterating over ~(padded_len - trace_len) entries.
            if one_hot_polys[0].masked_indices_c[j].is_some() {
                if is_interleaved {
                    interleaved_cycles.push(j);
                } else {
                    identity_cycles.push(j);
                }
            }
        }

        // -- Initialize u_evals and ra_acc --
        // NoOp padding cycles get zero so they contribute nothing to the sumcheck.
        // Phase 0: u_evals is public (eq(r_cycle, j)), avoiding trivial share promotion.
        let u_evals: Either<Vec<F>, Vec<Rep3PrimeFieldShare<F>>> = Either::Public(
            eq_r_cycle_public
                .iter()
                .zip(one_hot_polys[0].masked_indices_c.iter())
                .map(|(&v, opt)| if opt.is_some() { v } else { F::zero() })
                .collect(),
        );
        // Phase 0: ra_acc is None (implicit 1s for active cycles, 0 for padding)
        let ra_acc = None;

        // -- Initialize suffix polynomials (filled in init_phase) --
        //
        // Many tables/suffixes are unused for a given trace; keep them as `None` until a phase
        // materializes them. This avoids allocating O(#tables * #suffixes * M) upfront.
        let suffix_polys: Vec<Vec<Option<AdditiveDensePoly<F>>>> = LookupTables::<XLEN>::iter()
            .map(|table| vec![None; table.suffixes().len()])
            .collect();

        // -- Initialize prefix-suffix decompositions (public) --
        let right_operand_ps = PrefixSuffixDecomposition::new(
            Box::new(OperandPolynomial::new(LOG_K, OperandSide::Right)),
            LOG_M,
            LOG_K,
        );
        let left_operand_ps = PrefixSuffixDecomposition::new(
            Box::new(OperandPolynomial::new(LOG_K, OperandSide::Left)),
            LOG_M,
            LOG_K,
        );
        let identity_ps =
            PrefixSuffixDecomposition::new(Box::new(IdentityPolynomial::new(LOG_K)), LOG_M, LOG_K);

        let empty_q = || AdditiveDensePoly::empty();
        let mut state = ReadRafProverState {
            r: Vec::with_capacity(log_T + LOG_K),
            lookup_tables,
            lookup_indices_by_table,
            interleaved_cycles,
            identity_cycles,
            is_interleaved_operands,
            u_evals,
            ra_acc,
            ra: None,
            suffix_poly_len: M,
            suffix_polys,
            left_operand_q: [empty_q(), empty_q()],
            right_operand_q: [empty_q(), empty_q()],
            identity_q: [empty_q(), empty_q()],
            right_operand_ps,
            left_operand_ps,
            identity_ps,
            prefix_checkpoints: vec![None.into(); Prefixes::COUNT],
            prefix_registry: PrefixRegistry::new(),
            v: ExpandingTable::new(M),
            lookup_indices,
            right_operand_public_mask,
            one_hot_polys,
            ehat16: None,
            c16: vec![None; num_cycles],
            eq_r_cycle: MultilinearPolynomial::from(eq_r_cycle_public.to_vec()),
            combined_val_polynomial: None,
            party_id,
        };

        // Initialize phase 0
        state.init_phase(0, io_ctx, preproc)?;

        Ok(Self {
            gamma,
            gamma_squared: gamma.square(),
            rv_claim,
            raf_claim,
            log_T,
            state,
            _phantom: std::marker::PhantomData,
        })
    }
}

fn promote_to_trivial_share<F: JoltField>(val: F, party_id: PartyID) -> Rep3PrimeFieldShare<F> {
    Rep3PrimeFieldShare::promote_from_trivial(&val, party_id)
}

impl<F: JoltField> ReadRafProverState<F> {
    /// Initialize a new phase. Computes Ehat16, builds suffix polys and Q polys.
    ///
    /// For phase > 0, also does the condensation update on u_evals.
    #[tracing::instrument(
        skip(self, io_ctx, preproc),
        name = "ReadRaf::init_phase",
        fields(phase)
    )]
    fn init_phase<N: Rep3NetworkWorker>(
        &mut self,
        phase: usize,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<()> {
        eyre::ensure!(phase < PHASES, "phase out of range: {phase}");

        // Reset per-phase polynomial state (fresh unbound suffix domain of size M).
        self.suffix_poly_len = M;
        for table in self.suffix_polys.iter_mut() {
            for slot in table.iter_mut() {
                *slot = None;
            }
        }

        let hi = 2 * phase;
        let lo = 2 * phase + 1;
        eyre::ensure!(lo < D, "phase pair out of range: ({hi}, {lo})");
        eyre::ensure!(
            self.one_hot_polys[hi].masked_indices_c.len() == self.c16.len(),
            "one_hot_polys[{hi}].masked_indices_c length mismatch"
        );
        eyre::ensure!(
            self.one_hot_polys[lo].masked_indices_c.len() == self.c16.len(),
            "one_hot_polys[{lo}].masked_indices_c length mismatch"
        );
        eyre::ensure!(
            self.one_hot_polys[hi].rand_ohv_e_field.len() == 256,
            "one_hot_polys[{hi}].rand_ohv_e_field must have length 256"
        );
        eyre::ensure!(
            self.one_hot_polys[lo].rand_ohv_e_field.len() == 256,
            "one_hot_polys[{lo}].rand_ohv_e_field must have length 256"
        );
        let _ehat_span = tracing::info_span!("prefix_tensor_prod").entered();

        // derive c16
        for j in 0..self.c16.len() {
            let c_hi = self.one_hot_polys[hi].masked_indices_c[j];
            let c_lo = self.one_hot_polys[lo].masked_indices_c[j];
            self.c16[j] = match (c_hi, c_lo) {
                (Some(h), Some(l)) => Some(((h as u16) << 8) | (l as u16)),
                _ => None,
            };
        }

        // compute Ehat16 via tensor product
        let ehat16_prev = self.ehat16.take();

        let mut ehat8_hi: Vec<Rep3PrimeFieldShare<F>> =
            self.one_hot_polys[hi].rand_ohv_e_field.as_ref().clone();
        let mut ehat8_lo: Vec<Rep3PrimeFieldShare<F>> =
            self.one_hot_polys[lo].rand_ohv_e_field.as_ref().clone();
        fwht_rep3_in_place(&mut ehat8_hi);
        fwht_rep3_in_place(&mut ehat8_lo);

        // Tensor product: Ehat16[(a<<8)|b] = Ehat8_hi[a] * Ehat8_lo[b]
        let mut a_expanded = Vec::with_capacity(M);
        let mut b_expanded = Vec::with_capacity(M);
        for a_idx in 0..256 {
            for b_idx in 0..256 {
                a_expanded.push(ehat8_hi[a_idx]);
                b_expanded.push(ehat8_lo[b_idx]);
            }
        }

        let ehat16 = rep3_arith::mul_vec(&a_expanded, &b_expanded, io_ctx.main())?;
        self.ehat16 = Some(ehat16);
        drop(_ehat_span);

        // -- Condensation (phase > 0): u_evals *= eq_shifted[c16_prev] --
        let _cond_span = tracing::info_span!("condensation").entered();
        if phase > 0 {
            eyre::ensure!(
                self.r.len() >= phase * LOG_M,
                "init_phase({phase}) requires at least {} bound challenges, got {}",
                phase * LOG_M,
                self.r.len()
            );
            let prev_hi = 2 * (phase - 1);
            let prev_lo = 2 * (phase - 1) + 1;

            // Use cached ehat16 from the previous phase (avoids a 65536-element mul_vec)
            let ehat16_prev = ehat16_prev.expect("ehat16_prev must exist for phase > 0");

            // Build public EQ table from the 16 challenges of the previous phase
            let prev_start = (phase - 1) * LOG_M;
            let prev_challenges: Vec<F> = self.r[prev_start..prev_start + LOG_M]
                .iter()
                .map(|c| (*c).into())
                .collect();
            let eq16: Vec<F> = EqPolynomial::evals(&prev_challenges);

            // Shift into masked domain: eq_shifted[c] = eq16[c XOR r16_prev]
            let eq_shifted = shift_eq_table_with_mask(&eq16, &ehat16_prev);

            // c16_prev for each cycle
            let c16_prev: Vec<Option<u16>> = (0..self.c16.len())
                .map(|j| {
                    let c_hi = self.one_hot_polys[prev_hi].masked_indices_c[j];
                    let c_lo = self.one_hot_polys[prev_lo].masked_indices_c[j];
                    match (c_hi, c_lo) {
                        (Some(h), Some(l)) => Some(((h as u16) << 8) | (l as u16)),
                        _ => None,
                    }
                })
                .collect();

            // Only multiply active cycles (c16_prev != None) to reduce mul_vec size.
            let active_indices: Vec<usize> = c16_prev
                .iter()
                .enumerate()
                .filter_map(|(j, opt)| opt.map(|_| j))
                .collect();

            let num_cycles = self.c16.len();
            match &self.u_evals {
                Either::Public(u_pub) => {
                    // Phase 0→1: pub * shared → Shared (local, no mul_vec — saves 1 round)
                    let mut u_shared = vec![Rep3PrimeFieldShare::<F>::zero_share(); num_cycles];
                    for &j in &active_indices {
                        u_shared[j] = rep3_arith::mul_public(
                            eq_shifted[c16_prev[j].unwrap() as usize],
                            u_pub[j],
                        );
                    }
                    self.u_evals = Either::Shared(u_shared);
                }
                Either::Shared(u_shared) => {
                    // Phase 2+: existing mul_vec path
                    if !active_indices.is_empty() {
                        let u_active: Vec<Rep3PrimeFieldShare<F>> =
                            active_indices.iter().map(|&j| u_shared[j]).collect();
                        let eq_active: Vec<Rep3PrimeFieldShare<F>> = active_indices
                            .iter()
                            .map(|&j| eq_shifted[c16_prev[j].unwrap() as usize])
                            .collect();

                        let products = rep3_arith::mul_vec(&u_active, &eq_active, io_ctx.main())?;

                        let mut new_u = vec![Rep3PrimeFieldShare::<F>::zero_share(); num_cycles];
                        for (idx, &j) in active_indices.iter().enumerate() {
                            new_u[j] = products[idx];
                        }
                        self.u_evals = Either::Shared(new_u);
                    } else {
                        self.u_evals = Either::Shared(vec![
                            Rep3PrimeFieldShare::<F>::zero_share();
                            num_cycles
                        ]);
                    }
                }
            }
        }

        drop(_cond_span);

        // -- Build suffix polynomials --
        self.init_suffix_polys(phase, io_ctx, preproc)?;

        // -- Initialize prefix decompositions' Q polynomials --
        // We manage our own shared Q arrays ({left,right}_operand_q, identity_q) for
        // MPC computation. However, the PSDs' internal Q arrays are also bound every
        // round via PSD::bind(). Reset them to length-M zero polys so binding doesn't
        // underflow on subsequent phases. (Vanilla calls init_Q each phase, which
        // has the same effect.)
        self.identity_ps.init_Q(&[], &[]);
        self.right_operand_ps.init_Q(&[], &[]);
        self.left_operand_ps.init_Q(&[], &[]);

        self.init_operand_q_polys(phase, io_ctx, preproc)?;

        // Initialize P polynomials (public prefix polynomials)
        self.identity_ps.init_P(&mut self.prefix_registry);
        self.right_operand_ps.init_P(&mut self.prefix_registry);
        self.left_operand_ps.init_P(&mut self.prefix_registry);

        self.v.reset(F::one());

        Ok(())
    }

    /// Build suffix polynomials for the current phase using FWHT unmasking.
    ///
    /// For each table and each suffix, computes:
    ///   S[bucket] = Σ_{j ∈ table_cycles} u_evals[j] * suffix_mle(suffix_bits_j) * δ(prefix(k_j), bucket)
    ///
    /// Uses per-table suffix evaluation: each suffix is only evaluated for its
    /// table's cycles, not across all active cycles. Results are collected into a
    /// `SuffixFutureBatch` and fulfilled in one batched pass.
    ///
    /// Histogram building uses local `Rep3 × Rep3 → Additive` multiply, scatter
    /// into additive histogram, then `reshare_additive_many` (O(M) communication).
    #[tracing::instrument(skip_all, name = "ReadRaf::init_suffix_polys", fields(phase))]
    fn init_suffix_polys<N: Rep3NetworkWorker>(
        &mut self,
        phase: usize,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<()> {
        let ehat16 = self
            .ehat16
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ehat16 missing in init_suffix_polys"))?;
        let inv_m = F::from(M as u64).inverse().expect("M invertible");

        let suffix_len = (PHASES - 1 - phase) * LOG_M;

        // -- Step 1: Per-table suffix evaluation --
        // Dispatch to smallest ring type, build SuffixBitsBatch per table,
        // evaluate suffixes per table, fulfill all B2A conversions in one batch.
        let (eval_segments, all_field): (Vec<EvalSegment>, Vec<Rep3Value<F>>) = if suffix_len > 0 {
            match suffix_len {
                65..=128 => table_suffixes_mle::<u128, F, N>(
                    &self.lookup_indices,
                    &self.lookup_indices_by_table,
                    &self.right_operand_public_mask,
                    suffix_len,
                    io_ctx,
                    self.party_id,
                    preproc,
                )?,
                33..=64 => table_suffixes_mle::<u64, F, N>(
                    &self.lookup_indices,
                    &self.lookup_indices_by_table,
                    &self.right_operand_public_mask,
                    suffix_len,
                    io_ctx,
                    self.party_id,
                    preproc,
                )?,
                17..=32 => table_suffixes_mle::<u32, F, N>(
                    &self.lookup_indices,
                    &self.lookup_indices_by_table,
                    &self.right_operand_public_mask,
                    suffix_len,
                    io_ctx,
                    self.party_id,
                    preproc,
                )?,
                1..=16 => table_suffixes_mle::<u16, F, N>(
                    &self.lookup_indices,
                    &self.lookup_indices_by_table,
                    &self.right_operand_public_mask,
                    suffix_len,
                    io_ctx,
                    self.party_id,
                    preproc,
                )?,
                _ => unreachable!("suffix_len must be 1..=128"),
            }
        } else {
            (Vec::new(), Vec::new())
        };

        // -- Step 2: Build+unmask public/Rep3 histograms in parallel (no materialization) --
        // Returns:
        // - already-unmasked additive polys for public and rep3 histograms
        // - additive histograms that still require communication (reshare)
        let (pub_polys, rep3_polys, hist_entries_to_reshare, zero_polys) =
            build_suffix_polys_and_additive_hists(
                &eval_segments,
                &all_field,
                &self.u_evals,
                &self.c16,
                &self.lookup_indices_by_table,
                suffix_len,
                ehat16,
                self.party_id,
            );
        drop(zero_polys);

        for (table_idx, suffix_idx, poly) in pub_polys.into_iter().chain(rep3_polys) {
            self.suffix_polys[table_idx][suffix_idx] = Some(poly);
        }

        // -- Step 3: Chunked reshare+unmask additive histograms --
        let chunk_hists = std::env::var("RESHARE_HISTS_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8);

        let reshared_polys = reshare_and_unmask_additive_hists_chunked(
            hist_entries_to_reshare,
            ehat16,
            inv_m,
            self.party_id,
            io_ctx,
            chunk_hists,
        )?;
        for (table_idx, suffix_idx, poly) in reshared_polys {
            self.suffix_polys[table_idx][suffix_idx] = Some(poly);
        }

        Ok(())
    }

    /// Build the Q polynomials for operand prefix-suffix decompositions using FWHT.
    ///
    /// Mirrors vanilla `init_Q` / `init_Q_dual` but scatters into masked buckets
    /// (indexed by public c16) then unmaskes via FWHT convolution with Ehat16.
    ///
    /// Results are stored in `self.{left,right,identity}_operand_q` (our own shared
    /// Q arrays), NOT in the PrefixSuffixDecomposition's internal Q (which is public).
    ///
    /// The 6 Q polynomials (ORDER=2 for each of 3 decompositions) are:
    ///   left[0]  = ShiftHalfSuffix (constant: `1 << (suffix_len/2)`)
    ///   left[1]  = OperandPolynomial(Left) (secret: left operand value from suffix bits)
    ///   right[0] = ShiftHalfSuffix (constant)
    ///   right[1] = OperandPolynomial(Right) (secret: right operand value)
    ///   identity[0] = ShiftSuffix (constant: `1 << suffix_len`)
    ///   identity[1] = IdentityPolynomial (secret: full suffix value)
    ///
    /// For suffix_len == 0 (final phase), all are public constants.
    /// For suffix_len > 0, the secret operand/identity values are computed via
    /// FWHT-shifted table lookups on the future one-hot chunks.
    #[tracing::instrument(skip_all, name = "ReadRaf::init_operand_q_polys", fields(phase))]
    fn init_operand_q_polys<N: Rep3NetworkWorker>(
        &mut self,
        phase: usize,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<()> {
        use jolt_core::utils::uninterleave_bits;

        let ehat16 = self.ehat16.as_ref().unwrap();
        let suffix_len = (PHASES - 1 - phase) * LOG_M;

        // Helper: FWHT unmask a Rep3 histogram against Ehat16.
        let inv_m = F::from(M as u64).inverse().expect("M invertible");
        let fwht_unmask = |mut h: Vec<Rep3PrimeFieldShare<F>>| -> AdditiveDensePoly<F> {
            fwht_rep3_in_place(&mut h);
            let mut h_k: Vec<AdditiveShare<F>> = h
                .iter()
                .zip(ehat16.iter())
                .map(|(&a, &b)| (a * inv_m) * b)
                .collect();
            fwht_in_place(&mut h_k);
            AdditiveDensePoly::new(h_k)
        };

        if suffix_len == 0 {
            // All operand suffix values are zero. Only constant shift histograms matter.
            // histograms[0] == histograms[2], histograms[1,3,5] are zero.
            let _span = info_span!("build_histograms_final", phase).entered();
            let (q0, q4) = match &self.u_evals {
                Either::Public(u_pub) => {
                    // F histograms → unmask_histogram_public (FWHT on F, ~2x cheaper)
                    let mut h0_f = vec![F::zero(); M];
                    for &j in &self.interleaved_cycles {
                        if let Some(c) = self.c16[j] {
                            h0_f[c as usize] += u_pub[j];
                        }
                    }
                    let mut h4_f = vec![F::zero(); M];
                    for &j in &self.identity_cycles {
                        if let Some(c) = self.c16[j] {
                            h4_f[c as usize] += u_pub[j];
                        }
                    }
                    let party_id = self.party_id;
                    let q0 = AdditiveDensePoly::new(unmask_histogram_public(
                        &mut h0_f, ehat16, party_id,
                    ));
                    let q4 = AdditiveDensePoly::new(unmask_histogram_public(
                        &mut h4_f, ehat16, party_id,
                    ));
                    (q0, q4)
                }
                Either::Shared(u_shared) => {
                    let mut hist0 = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
                    for &j in &self.interleaved_cycles {
                        if let Some(c) = self.c16[j] {
                            hist0[c as usize] += u_shared[j];
                        }
                    }
                    let mut hist4 = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
                    for &j in &self.identity_cycles {
                        if let Some(c) = self.c16[j] {
                            hist4[c as usize] += u_shared[j];
                        }
                    }
                    let q0 = fwht_unmask(hist0);
                    let q4 = fwht_unmask(hist4);
                    (q0, q4)
                }
            };
            drop(_span);

            self.left_operand_q[0] = q0.clone();
            self.left_operand_q[1] = AdditiveDensePoly::zeros(M);
            self.right_operand_q[0] = q0;
            self.right_operand_q[1] = AdditiveDensePoly::zeros(M);
            self.identity_q[0] = q4;
            self.identity_q[1] = AdditiveDensePoly::zeros(M);
            return Ok(());
        }

        let _span = info_span!("build_histograms", phase).entered();

        // suffix_len > 0: classify cycles by group (interleaved vs identity)
        // and by secrecy (public vs shared).
        let mask_u128 = if suffix_len < 128 {
            (1u128 << suffix_len) - 1
        } else {
            u128::MAX
        };
        let mask = RingElement(mask_u128);

        // Interleaved group: pub (both operands public) vs shared (left shared).
        // For shared: right_operand_pub[i] = Some(F) if right is public (mixed), None if shared.
        let half_bits = suffix_len / 2;
        let mut pub_interleaved: Vec<(usize, F, F)> = Vec::new();
        let mut shared_interleaved_js: Vec<usize> = Vec::new();
        let mut shared_interleaved_masked: Vec<Rep3RingShare<u128>> = Vec::new();
        let mut right_operand_pub: Vec<Option<F>> = Vec::new();

        for &j in &self.interleaved_cycles {
            match &self.lookup_indices[j] {
                Either::Public(p) => {
                    let masked = *p & mask_u128;
                    let (x, y) = uninterleave_bits(masked);
                    pub_interleaved.push((j, F::from_u128(x as u128), F::from_u128(y as u128)));
                }
                Either::Shared(s) => {
                    shared_interleaved_js.push(j);
                    shared_interleaved_masked.push(*s & mask);
                    right_operand_pub.push(self.right_operand_public_mask[j].map(|right_val| {
                        let right_masked = if half_bits >= 64 {
                            right_val
                        } else {
                            right_val & ((1u64 << half_bits) - 1)
                        };
                        F::from_u64(right_masked)
                    }));
                }
            }
        }

        // Identity group: public → (j, id_val), shared → masked u128
        let (pub_identity, shared_identity_js, shared_identity_masked): (
            Vec<(usize, F)>,
            Vec<usize>,
            Vec<Rep3RingShare<u128>>,
        ) = self
            .identity_cycles
            .par_iter()
            .fold(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(mut pubs, mut js, mut masked), &j| {
                    match &self.lookup_indices[j] {
                        Either::Public(p) => {
                            pubs.push((j, F::from_u128(*p & mask_u128)));
                        }
                        Either::Shared(s) => {
                            js.push(j);
                            masked.push(*s & mask);
                        }
                    }
                    (pubs, js, masked)
                },
            )
            .reduce(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(mut a, mut b, mut c), (x, y, z)| {
                    a.extend(x);
                    b.extend(y);
                    c.extend(z);
                    (a, b, c)
                },
            );

        let shared_right_idx: Vec<usize> = right_operand_pub
            .par_iter()
            .enumerate()
            .filter_map(|(i, &r)| if !r.is_some() { Some(i) } else { None })
            .collect();
        tracing::info!(
            "identity (pub: {}, priv: {}); operands (pub: {}, mixed: {}, priv: {})",
            pub_identity.len(),
            shared_identity_js.len(),
            pub_interleaved.len(),
            shared_right_idx.len(),
            shared_interleaved_js.len() - shared_right_idx.len()
        );
        let (s_left, s_right, s_identity) = match suffix_len {
            65..=128 => q_polys_b2a::<u128, F, N>(
                shared_interleaved_masked,
                &shared_right_idx,
                shared_identity_masked,
                io_ctx,
                preproc,
            )?,
            33..=64 => q_polys_b2a::<u64, F, N>(
                shared_interleaved_masked,
                &shared_right_idx,
                shared_identity_masked,
                io_ctx,
                preproc,
            )?,
            17..=32 => q_polys_b2a::<u32, F, N>(
                shared_interleaved_masked,
                &shared_right_idx,
                shared_identity_masked,
                io_ctx,
                preproc,
            )?,
            1..=16 => q_polys_b2a::<u16, F, N>(
                shared_interleaved_masked,
                &shared_right_idx,
                shared_identity_masked,
                io_ctx,
                preproc,
            )?,
            _ => unreachable!("suffix_len must be 1..=128"),
        };

        // Constant suffix values
        let shift_half_val = F::from_u128(1u128 << (suffix_len / 2));
        let shift_val = F::from_u128(1u128 << suffix_len);
        let party_id = self.party_id;
        let c16 = &self.c16;

        // Build histograms. histograms[0] == histograms[2] always (both Σ u * shift_half_val
        // for interleaved cycles), so we compute it once and clone.
        // Phase 0: shift histograms stay as F (unmask_histogram_public is ~2x cheaper).
        let mut shift_half_f: Option<Vec<F>> = None;
        let mut shift_f: Option<Vec<F>> = None;
        let mut hist_shift_half = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
        let mut hist_shift = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
        let mut hist_left = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
        let mut hist_right = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
        let mut hist_identity = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];

        // Shift histogram accumulation is folded into the operand loops below:
        // pub_interleaved ∪ shared_interleaved == interleaved_cycles, and
        // pub_identity ∪ shared_identity == identity_cycles, so every active
        // cycle is visited exactly once.  The two groups (interleaved vs identity)
        // write to disjoint histograms and run in parallel via rayon::join.
        match &self.u_evals {
            Either::Public(u_pub) => {
                // Phase 0: u is public — shift histograms as plain F
                let mut hsh_f = vec![F::zero(); M];
                let mut hs_f = vec![F::zero(); M];
                rayon::join(
                    || {
                        // Interleaved: shift_half + left/right operands
                        for &(j, left_val, right_val) in &pub_interleaved {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                hsh_f[ci] += u_pub[j] * shift_half_val;
                                hist_left[ci] = rep3_arith::add_public(
                                    hist_left[ci],
                                    u_pub[j] * left_val,
                                    party_id,
                                );
                                hist_right[ci] = rep3_arith::add_public(
                                    hist_right[ci],
                                    u_pub[j] * right_val,
                                    party_id,
                                );
                            }
                        }
                        for (i, &j) in shared_interleaved_js.iter().enumerate() {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                hsh_f[ci] += u_pub[j] * shift_half_val;
                                hist_left[ci] += rep3_arith::mul_public(s_left[i], u_pub[j]);
                                match s_right[i] {
                                    Some(sr) => {
                                        hist_right[ci] += rep3_arith::mul_public(sr, u_pub[j]);
                                    }
                                    None => {
                                        hist_right[ci] = rep3_arith::add_public(
                                            hist_right[ci],
                                            u_pub[j] * right_operand_pub[i].unwrap(),
                                            party_id,
                                        );
                                    }
                                }
                            }
                        }
                    },
                    || {
                        // Identity: shift + identity operand
                        for &(j, id_val) in &pub_identity {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                hs_f[ci] += u_pub[j] * shift_val;
                                hist_identity[ci] = rep3_arith::add_public(
                                    hist_identity[ci],
                                    u_pub[j] * id_val,
                                    party_id,
                                );
                            }
                        }
                        for (i, &j) in shared_identity_js.iter().enumerate() {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                hs_f[ci] += u_pub[j] * shift_val;
                                hist_identity[ci] +=
                                    rep3_arith::mul_public(s_identity[i], u_pub[j]);
                            }
                        }
                    },
                );
                shift_half_f = Some(hsh_f);
                shift_f = Some(hs_f);
            }
            Either::Shared(u_shared) => {
                // Phase 1+: u is shared
                let has_shared_interleaved = !shared_interleaved_js.is_empty();
                let has_shared_identity = !shared_identity_js.is_empty();
                let has_fully_shared_right = right_operand_pub.iter().any(|r| r.is_none());

                // Additive accumulators allocated before parallel section.
                let (mut add_left, mut add_right, mut add_identity) =
                    if has_shared_interleaved || has_shared_identity {
                        (
                            vec![AdditiveShare::<F>::zero(); M],
                            if has_fully_shared_right {
                                vec![AdditiveShare::<F>::zero(); M]
                            } else {
                                vec![]
                            },
                            vec![AdditiveShare::<F>::zero(); M],
                        )
                    } else {
                        (vec![], vec![], vec![])
                    };

                rayon::join(
                    || {
                        // Interleaved: shift_half + pub operands + shared operands
                        for &(j, left_val, right_val) in &pub_interleaved {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                let u = u_shared[j];
                                hist_shift_half[ci] += u * shift_half_val;
                                hist_left[ci] += u * left_val;
                                hist_right[ci] += u * right_val;
                            }
                        }
                        for (i, &j) in shared_interleaved_js.iter().enumerate() {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                let u = u_shared[j];
                                hist_shift_half[ci] += u * shift_half_val;
                                add_left[ci] = add_left[ci] + (u * s_left[i]);
                                match s_right[i] {
                                    Some(sr) => {
                                        add_right[ci] = add_right[ci] + (u * sr);
                                    }
                                    None => {
                                        hist_right[ci] += u * right_operand_pub[i].unwrap();
                                    }
                                }
                            }
                        }
                    },
                    || {
                        // Identity: shift + pub operands + shared operands
                        for &(j, id_val) in &pub_identity {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                let u = u_shared[j];
                                hist_shift[ci] += u * shift_val;
                                hist_identity[ci] += u * id_val;
                            }
                        }
                        for (i, &j) in shared_identity_js.iter().enumerate() {
                            if let Some(c) = c16[j] {
                                let ci = c as usize;
                                let u = u_shared[j];
                                hist_shift[ci] += u * shift_val;
                                add_identity[ci] = add_identity[ci] + (u * s_identity[i]);
                            }
                        }
                    },
                );

                // Batch reshare additive → Rep3
                if has_shared_interleaved || has_shared_identity {
                    let _reshare = info_span!("q_reshare").entered();
                    let mut flat: Vec<AdditiveShare<F>> = Vec::with_capacity(3 * M);
                    flat.extend_from_slice(&add_left);
                    if has_fully_shared_right {
                        flat.extend_from_slice(&add_right);
                    }
                    flat.extend_from_slice(&add_identity);
                    let flat_rep3 = rep3_arith::reshare_additive_many(&flat, io_ctx.main())?;
                    drop(_reshare);

                    for i in 0..M {
                        hist_left[i] += flat_rep3[i];
                    }
                    if has_fully_shared_right {
                        for i in 0..M {
                            hist_right[i] += flat_rep3[M + i];
                        }
                        for i in 0..M {
                            hist_identity[i] += flat_rep3[2 * M + i];
                        }
                    } else {
                        for i in 0..M {
                            hist_identity[i] += flat_rep3[M + i];
                        }
                    }
                }
            }
        }
        drop(_span);

        // FWHT unmask. histograms[0]==[2] → 5 unique FWHTs (all independent).
        // Phase 0: shift histograms are F → use unmask_histogram_public (~2x cheaper).
        let _fwht_span = info_span!("fwht_unmask", phase).entered();

        let shift_in_rep3 = shift_half_f.is_none();
        let _seq = trace_span!(
            "init_operand_q_polys_unmask_seq",
            phase,
            shift_in_rep3
        )
        .entered();

        let (q02, q1, q3, q4, q5) = if shift_in_rep3 {
            // Phase 1+: all 5 from rep3; unmask sequentially and drop each histogram immediately.
            let q02 = fwht_unmask(hist_shift_half);
            let q1 = fwht_unmask(hist_left);
            let q3 = fwht_unmask(hist_right);
            let q4 = fwht_unmask(hist_shift);
            let q5 = fwht_unmask(hist_identity);
            (q02, q1, q3, q4, q5)
        } else {
            // Phase 0: shift histograms from F, operand histograms from Rep3.
            let party_id = self.party_id;
            let mut hsh = shift_half_f.unwrap();
            let mut hs = shift_f.unwrap();
            let q02 =
                AdditiveDensePoly::new(unmask_histogram_public(&mut hsh, ehat16, party_id));
            let q4 = AdditiveDensePoly::new(unmask_histogram_public(&mut hs, ehat16, party_id));

            let q1 = fwht_unmask(hist_left);
            let q3 = fwht_unmask(hist_right);
            let q5 = fwht_unmask(hist_identity);
            (q02, q1, q3, q4, q5)
        };
        drop(_seq);
        drop(_fwht_span);

        self.left_operand_q[0] = q02.clone();
        self.left_operand_q[1] = q1;
        self.right_operand_q[0] = q02;
        self.right_operand_q[1] = q3;
        self.identity_q[0] = q4;
        self.identity_q[1] = q5;

        Ok(())
    }

    /// Cache the phase after all LOG_M rounds are bound: update ra_acc.
    fn cache_phase<N: Rep3NetworkWorker>(
        &mut self,
        _phase: usize,
        io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<()> {
        // ra_acc[j] *= v[bucket] where bucket = prefix(k_j) % M for the current phase.
        // In masked domain: bucket = (c16[j] ^ r16) % M = c16[j] ^ r16 (since M = 2^16).
        // We use the shifted v table: v_shifted[c] = v[c ^ r16].
        let ehat16 = self.ehat16.as_ref().unwrap();
        let v_shifted = shift_eq_table_with_mask(self.v.values(), ehat16);

        // Only multiply active cycles (c16 != None) to reduce mul_vec size.
        // Inactive cycles have ra_acc = 0 already (multiplied by zero in condensation
        // or initialized to zero for padding), so skipping them is safe.
        let num_cycles = self.c16.len();
        match &self.ra_acc {
            None => {
                // Phase 0: ra_acc[j] = 1 for active, 0 for padding.
                // 1 * v_shifted[c16[j]] = v_shifted[c16[j]] — no mul_vec needed.
                let ra_shared: Vec<_> = (0..num_cycles)
                    .into_par_iter()
                    .map(|j| match self.c16[j] {
                        Some(c) => v_shifted[c as usize],
                        None => Rep3PrimeFieldShare::<F>::zero_share(),
                    })
                    .collect();
                self.ra_acc = Some(ra_shared);
            }
            Some(ra_shared) => {
                // Phase 1+: gather active (ra, v) pairs and indices in one parallel pass.
                let (active_indices, ra_active, v_active): (
                    Vec<usize>,
                    Vec<Rep3PrimeFieldShare<F>>,
                    Vec<Rep3PrimeFieldShare<F>>,
                ) = self
                    .c16
                    .par_iter()
                    .enumerate()
                    .filter_map(|(j, opt)| opt.map(|c| (j, ra_shared[j], v_shifted[c as usize])))
                    .fold(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |(mut idx, mut ra, mut v), (j, r, vs)| {
                            idx.push(j);
                            ra.push(r);
                            v.push(vs);
                            (idx, ra, v)
                        },
                    )
                    .reduce(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |(mut a, mut b, mut c), (x, y, z)| {
                            a.extend(x);
                            b.extend(y);
                            c.extend(z);
                            (a, b, c)
                        },
                    );

                if !active_indices.is_empty() {
                    let products = rep3_arith::mul_vec(&ra_active, &v_active, io_ctx.main())?;

                    // Parallel scatter — indices are disjoint by construction.
                    let mut new_ra = vec![Rep3PrimeFieldShare::<F>::zero_share(); num_cycles];
                    use crate::utils::send_ptr::SendPtr;
                    let ptr = SendPtr(new_ra.as_mut_ptr());
                    active_indices
                        .par_iter()
                        .zip(products.par_iter())
                        .for_each(|(&j, &val)| {
                            let p = ptr;
                            unsafe { p.0.add(j).write(val) };
                        });
                    self.ra_acc = Some(new_ra);
                }
            }
        }

        self.prefix_registry.update_checkpoints();
        Ok(())
    }

    /// Initialize the final log_T rounds.
    #[tracing::instrument(skip_all, name = "ReadRaf::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self, gamma: F, gamma_squared: F) {
        let prefixes: Vec<PrefixEval<F>> = std::mem::take(&mut self.prefix_checkpoints)
            .into_iter()
            .map(|checkpoint| checkpoint.unwrap())
            .collect();

        let t = self.c16.len();
        let mut combined_val_poly: Vec<F> = vec![F::zero(); t];

        // Precompute per-group RAF constants
        let left_right_contrib = gamma
            * self.prefix_registry.checkpoints[jolt_core::poly::prefix_suffix::Prefix::LeftOperand]
                .unwrap()
            + gamma_squared
                * self.prefix_registry.checkpoints
                    [jolt_core::poly::prefix_suffix::Prefix::RightOperand]
                    .unwrap();
        let identity_contrib = gamma_squared
            * self.prefix_registry.checkpoints[jolt_core::poly::prefix_suffix::Prefix::Identity]
                .unwrap();

        // Must iterate ALL cycles (including padding) because padding values
        // affect sumcheck evaluations at extrapolated points when paired with real cycles.
        for (j, table) in self.lookup_tables.iter().enumerate() {
            if let Some(table) = table {
                let suffixes: Vec<_> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| F::from_u64(suffix.suffix_mle::<XLEN>(LookupBits::new(0, 0))))
                    .collect();
                combined_val_poly[j] += table.combine(&prefixes, &suffixes);
            }
            if self.is_interleaved_operands[j] {
                combined_val_poly[j] += left_right_contrib;
            } else {
                combined_val_poly[j] += identity_contrib;
            }
        }

        self.combined_val_polynomial = Some(MultilinearPolynomial::from(combined_val_poly));

        // Build ra polynomial from ra_acc for the log_T rounds.
        // By this point, all 8 cache_phase calls have run, so ra_acc is Some(vec).
        let ra_vec = self
            .ra_acc
            .take()
            .expect("ra_acc must be Some after all cache_phase calls");
        self.ra = Some(Rep3DensePolynomial::new(ra_vec));

        // Address-round scratch is no longer needed for the cycle rounds.
        // Free it eagerly to reduce stage3 peak RSS.
        for table in self.suffix_polys.iter_mut() {
            for slot in table.iter_mut() {
                *slot = None;
            }
        }
        self.left_operand_q = [AdditiveDensePoly::empty(), AdditiveDensePoly::empty()];
        self.right_operand_q = [AdditiveDensePoly::empty(), AdditiveDensePoly::empty()];
        self.identity_q = [AdditiveDensePoly::empty(), AdditiveDensePoly::empty()];
        self.v = ExpandingTable::new(1);
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3ReadRafSumcheckWorker<F, N>
{
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self) -> Rep3Value<F> {
        Rep3Value::Public(self.rv_claim + self.gamma * self.raf_claim)
    }

    fn compute_prover_message_share(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        if round < LOG_K {
            // Address rounds: prefix-suffix phase.
            // Compute read-checking and RAF components.
            let [eval_0, eval_2] = self.compute_prefix_suffix_prover_message(round);
            let eval_1 = previous_claim - eval_0;

            // Degree 3 requires evals at {0, 2, 3}; we return {0, 2, 3, ..., max_degree}.
            let mut evals = vec![AdditiveShare::zero(); max_degree];
            evals[0] = eval_0;
            if max_degree >= 2 {
                evals[1] = eval_2;
            }
            // Degree-2 polynomial through (0, eval_0), (1, eval_1), (2, eval_2).
            // For x=3,...: use Lagrange interpolation.
            // p(x) = eval_0 * (x-1)(x-2)/2 - eval_1 * x(x-2) + eval_2 * x(x-1)/2
            for x in 3..=max_degree {
                let xf = F::from(x as u64);
                let xm1 = xf - F::one();
                let xm2 = xf - F::from(2u64);
                let inv2 = F::TWO_INV;
                evals[x - 1] =
                    eval_0 * (xm1 * xm2 * inv2) - eval_1 * (xf * xm2) + eval_2 * (xf * xm1 * inv2);
            }
            evals
        } else {
            // Cycle rounds: eq_r_cycle * ra * combined_val.
            let ps = &self.state;
            let ra = ps.ra.as_ref().unwrap();
            let combined_val = ps.combined_val_polynomial.as_ref().unwrap();
            let half = ps.eq_r_cycle.len() / 2;

            let evals = (0..half)
                .into_par_iter()
                .map(|i| {
                    let eq_vals = ps
                        .eq_r_cycle
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                    let val_vals =
                        combined_val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                    let ra_vals = ra.sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);

                    let mut local = [AdditiveShare::<F>::zero(); DEGREE];
                    for d in 0..DEGREE {
                        let eq_val: F = eq_vals[d] * val_vals[d];
                        local[d] = (ra_vals[d] * eq_val).into_additive();
                    }
                    local
                })
                .reduce(
                    || [AdditiveShare::<F>::zero(); DEGREE],
                    |a, b| std::array::from_fn(|d| a[d] + b[d]),
                );

            let mut result = vec![AdditiveShare::zero(); max_degree];
            for (i, &e) in evals.iter().enumerate().take(max_degree) {
                result[i] = e;
            }
            // Pad if max_degree > DEGREE
            if max_degree > DEGREE {
                // Degree-3 polynomial, extrapolate to higher points
                let eval_1 = previous_claim - result[0];
                for x in (DEGREE + 1)..=max_degree {
                    let xf = F::from(x as u64);
                    // Lagrange through (0, result[0]), (1, eval_1), (2, result[1]), (3, result[2])
                    result[x - 1] = lagrange_interp_4(result[0], eval_1, result[1], result[2], xf);
                }
            }
            result
        }
    }

    fn bind(
        &mut self,
        r_j: F::Challenge,
        round: usize,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
    ) {
        let ps = &mut self.state;
        ps.r.push(r_j);

        if round < LOG_K {
            let r: F = r_j.into();

            // Run suffix poly binding, Q poly binding, PSD binding, and v.update
            // concurrently — all are independent.
            rayon::scope(|s| {
                // Suffix polys: flatten all tables' suffix polys and bind in parallel
                s.spawn(|_| {
                    ps.suffix_polys.par_iter_mut().for_each(|polys| {
                        for poly in polys.iter_mut() {
                            if let Some(p) = poly.as_mut() {
                                p.bind(r);
                            }
                        }
                    });
                });

                // Q polys: bind all 6 Q arrays in parallel
                s.spawn(|_| {
                    let q_polys: Vec<&mut AdditiveDensePoly<F>> = ps
                        .left_operand_q
                        .iter_mut()
                        .chain(ps.right_operand_q.iter_mut())
                        .chain(ps.identity_q.iter_mut())
                        .collect();
                    q_polys.into_par_iter().for_each(|q| q.bind(r));
                });

                // PSD bindings (public, cheap — run together on one task)
                s.spawn(|_| {
                    ps.identity_ps.bind(r_j);
                    ps.right_operand_ps.bind(r_j);
                    ps.left_operand_ps.bind(r_j);
                });

                // Expanding table update
                s.spawn(|_| {
                    ps.v.update(r_j);
                });
            });

            // All suffix/Q polynomials have been bound once (halve their logical length).
            ps.suffix_poly_len /= 2;

            // Update prefix checkpoints every 2 rounds
            if ps.r.len().is_multiple_of(2) {
                Prefixes::update_checkpoints::<XLEN, F, F::Challenge>(
                    &mut ps.prefix_checkpoints,
                    ps.r[ps.r.len() - 2],
                    ps.r[ps.r.len() - 1],
                    round,
                );
            }

            // Phase transition at end of each LOG_M rounds
            if (round + 1).is_multiple_of(LOG_M) {
                let phase = round / LOG_M;
                ps.cache_phase(phase, io_ctx, preproc)
                    .expect("cache_phase failed");

                if phase != PHASES - 1 {
                    ps.init_phase(phase + 1, io_ctx, preproc)
                        .expect("init_phase failed");
                }
            }

            // Transition to log_T rounds
            if (round + 1) == LOG_K {
                ps.init_log_t_rounds(self.gamma, self.gamma_squared);
            }
        } else {
            // log_T rounds: bind ra, eq_r_cycle, combined_val
            ps.ra
                .as_mut()
                .unwrap()
                .bind(r_j.into(), BindingOrder::HighToLow);
            ps.eq_r_cycle.bind_parallel(r_j, BindingOrder::HighToLow);
            ps.combined_val_polynomial
                .as_mut()
                .unwrap()
                .bind_parallel(r_j, BindingOrder::HighToLow);
        }
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_worker(
        &mut self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let ps = &self.state;
        let (_r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);
        let eq_r_cycle_prime = EqPolynomial::<F>::evals(&r_cycle.r);

        // Table flag claims (public): Σ_j eq(r_cycle', j) for each table's cycles
        let flag_claims: Vec<F> = ps
            .lookup_indices_by_table
            .iter()
            .map(|table_cycles| table_cycles.iter().map(|&j| eq_r_cycle_prime[j]).sum::<F>())
            .collect();

        for (i, claim) in flag_claims.iter().enumerate() {
            accumulator.append_virtual_public(
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
                *claim,
                ps.party_id,
            );
        }

        // ra claim (shared): final value of ra polynomial after sumcheck
        let ra_claim = ps.ra.as_ref().unwrap().final_sumcheck_claim();
        accumulator.append_virtual(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            r_sumcheck,
            ra_claim,
        );

        // RAF flag claim (public): Σ_j eq(r_cycle', j) for identity cycles
        let raf_flag_claim: F = ps
            .identity_cycles
            .iter()
            .map(|&j| eq_r_cycle_prime[j])
            .sum();
        accumulator.append_virtual_public(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle,
            raf_flag_claim,
            ps.party_id,
        );

        // Return claims in deterministic order: table_flags..., ra, raf_flag
        let mut claims: Vec<Rep3PrimeFieldShare<F>> = flag_claims
            .into_iter()
            .map(|c| promote_to_trivial_share(c, ps.party_id))
            .collect();
        claims.push(ra_claim);
        claims.push(promote_to_trivial_share(raf_flag_claim, ps.party_id));
        claims
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3ReadRafSumcheckWorker<F, N> {
    /// Compute the prefix-suffix prover message for address rounds.
    fn compute_prefix_suffix_prover_message(&self, round: usize) -> [AdditiveShare<F>; 2] {
        // Capture fields directly to avoid Sync bound on PhantomData<N>.
        let ps = &self.state;
        let gamma = self.gamma;
        let gamma_squared = self.gamma_squared;
        let (read_checking, raf) = rayon::join(
            || Self::prover_msg_read_checking_inner(ps, round),
            || Self::prover_msg_raf_inner(ps, gamma, gamma_squared),
        );
        [read_checking[0] + raf[0], read_checking[1] + raf[1]]
    }

    /// Read-checking component of the prover message.
    fn prover_msg_read_checking_inner(
        ps: &ReadRafProverState<F>,
        j: usize,
    ) -> [AdditiveShare<F>; 2] {
        let len = ps.suffix_poly_len;
        debug_assert!(len > 0, "suffix_poly_len must be > 0 during address rounds");
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 {
            ps.r.last().copied()
        } else {
            None
        };

        let half = len / 2;

        // `Prefixes` is a small enum; materialize it once to avoid per-bucket iteration/allocation.
        // This is read-only and shared across Rayon threads.
        let prefix_kinds: Vec<Prefixes> = Prefixes::iter().collect();
        debug_assert_eq!(prefix_kinds.len(), Prefixes::COUNT);

        let (eval_0, eval_2_left, eval_2_right) = (0..half)
            .into_par_iter()
            .map(|b| {
                let b_bits = LookupBits::new(b as u128, log_len - 1);

                // Compute prefix evaluations into fixed-size stack arrays to avoid per-bucket Vecs.
                let prefixes_c0: [PrefixEval<F>; Prefixes::COUNT] = std::array::from_fn(|pi| {
                    let prefix = &prefix_kinds[pi];
                    prefix.prefix_mle::<XLEN, F, F::Challenge>(
                        &ps.prefix_checkpoints,
                        r_x,
                        0,
                        b_bits,
                        j,
                    )
                });
                let prefixes_c2: [PrefixEval<F>; Prefixes::COUNT] = std::array::from_fn(|pi| {
                    let prefix = &prefix_kinds[pi];
                    prefix.prefix_mle::<XLEN, F, F::Challenge>(
                        &ps.prefix_checkpoints,
                        r_x,
                        2,
                        b_bits,
                        j,
                    )
                });

                let mut e0 = AdditiveShare::<F>::zero();
                let mut e2l = AdditiveShare::<F>::zero();
                let mut e2r = AdditiveShare::<F>::zero();

                for (table, suffixes) in LookupTables::<XLEN>::iter().zip(ps.suffix_polys.iter()) {
                    let table = &table;
                    let n = suffixes.len();
                    debug_assert!(n <= 8, "suffix count exceeds stack buffer size");

                    // Gather suffix coefficients without allocating.
                    let mut suffixes_left: [AdditiveShare<F>; 8] = [AdditiveShare::<F>::zero(); 8];
                    let mut suffixes_right: [AdditiveShare<F>; 8] = [AdditiveShare::<F>::zero(); 8];
                    for si in 0..n {
                        // `get_coeff` is a by-value read of an AdditiveShare.
                        if let Some(poly) = suffixes[si].as_ref() {
                            suffixes_left[si] = poly.get_coeff(b);
                            suffixes_right[si] = poly.get_coeff(b + half);
                        }
                    }

                    // `LookupTables::combine(prefixes, suffixes)` is linear in `suffixes`.
                    // Compute per-suffix weights once and reuse across both left/right suffix
                    // coefficient vectors for the same (table, prefixes) pair.
                    let weights_c0 = combine_shared_weights::<F>(table, &prefixes_c0, n);
                    let weights_c2 = combine_shared_weights::<F>(table, &prefixes_c2, n);

                    e0 += dot_weights_suffixes::<F>(&weights_c0, &suffixes_left, n);
                    e2l += dot_weights_suffixes::<F>(&weights_c2, &suffixes_left, n);
                    e2r += dot_weights_suffixes::<F>(&weights_c2, &suffixes_right, n);
                }
                (e0, e2l, e2r)
            })
            .reduce(
                || {
                    (
                        AdditiveShare::<F>::zero(),
                        AdditiveShare::<F>::zero(),
                        AdditiveShare::<F>::zero(),
                    )
                },
                |(a0, a2l, a2r), (b0, b2l, b2r)| (a0 + b0, a2l + b2l, a2r + b2r),
            );

        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }

    /// RAF component of the prover message.
    ///
    /// Mirrors vanilla `prover_msg_raf` but uses our shared Q arrays
    /// (`{left,right,identity}_operand_q`) paired with public P polynomials
    /// from the `PrefixRegistry`.
    fn prover_msg_raf_inner(
        ps: &ReadRafProverState<F>,
        gamma: F,
        gamma_squared: F,
    ) -> [AdditiveShare<F>; 2] {
        let len = ps.identity_q[0].len();
        let half = len / 2;

        let left_p = ps.prefix_registry.polys[Prefix::LeftOperand as usize].as_ref();
        let right_p = ps.prefix_registry.polys[Prefix::RightOperand as usize].as_ref();
        let identity_p = ps.prefix_registry.polys[Prefix::Identity as usize].as_ref();

        let (left_0, left_2, right_0, right_2) = (0..half)
            .into_par_iter()
            .map(|b| {
                let (l0, l2) = psd_sumcheck_evals_shared(left_p, &ps.left_operand_q, b, len);
                let (r0, r2) = psd_sumcheck_evals_shared(right_p, &ps.right_operand_q, b, len);
                let (i0, i2) = psd_sumcheck_evals_shared(identity_p, &ps.identity_q, b, len);
                (l0, l2, i0 + r0, i2 + r2)
            })
            .reduce(
                || {
                    (
                        AdditiveShare::<F>::zero(),
                        AdditiveShare::<F>::zero(),
                        AdditiveShare::<F>::zero(),
                        AdditiveShare::<F>::zero(),
                    )
                },
                |(a0, a1, a2, a3), (b0, b1, b2, b3)| (a0 + b0, a1 + b1, a2 + b2, a3 + b3),
            );

        [
            left_0 * gamma + right_0 * gamma_squared,
            left_2 * gamma + right_2 * gamma_squared,
        ]
    }
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
    let _span =
        trace_span!("build_suffix_histograms_streaming", n = eval_segments.len()).entered();
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

    let _span = trace_span!(
        "q_polys_b2a",
        n_il,
        n_id,
        chunk = chunk_size,
        k = T::K
    )
    .entered();

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
            let _c = trace_span!("q_polys_b2a_chunk", kind = "lr", chunk_len = lr_chunk.len())
                .entered();
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
            let _c = trace_span!("q_polys_b2a_chunk", kind = "id", chunk_len = id_chunk.len())
                .entered();
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
