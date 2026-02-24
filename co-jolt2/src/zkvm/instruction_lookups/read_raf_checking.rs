use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::utils::fwht::{fwht_additive_in_place, fwht_rep3_in_place, shift_eq_table_with_mask};
use crate::utils::types::Rep3Value;
use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use crate::zkvm::instruction::Rep3Cycle;
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
use jolt_core::zkvm::instruction::{InstructionFlags, InstructionLookup, InterleavedBitsMarker};
use jolt_core::zkvm::instruction_lookups::{D, LOG_M};
use jolt_core::zkvm::lookup_table::prefixes::{PrefixCheckpoint, PrefixEval, Prefixes};
use jolt_core::zkvm::lookup_table::suffixes::{SuffixEval, Suffixes};
use jolt_core::zkvm::lookup_table::LookupTables;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use strum::{EnumCount, IntoEnumIterator};
use tracing::info_span;

// ---------------------------------------------------------------------------
// Additive dense polynomial — lightweight wrapper for suffix/Q polys
// ---------------------------------------------------------------------------

/// A dense multilinear polynomial stored as additive shares.
///
/// Used for suffix polys and operand Q polys where all downstream operations
/// are linear (bind with public challenge, read coefficients, scalar mult).
/// This avoids the `mul_vec` reshare cost — the FWHT pointwise product
/// `Rep3 * Rep3 → Additive` is computed locally with zero communication.
#[derive(Clone)]
struct AdditiveDensePoly<F: JoltField> {
    coeffs: Vec<AdditiveShare<F>>,
    /// After first bind, subsequent binds work on bound_coeffs.
    bound: Vec<AdditiveShare<F>>,
    /// Current logical length (halves on each bind).
    current_len: usize,
    is_bound: bool,
}

impl<F: JoltField> AdditiveDensePoly<F> {
    fn new(coeffs: Vec<AdditiveShare<F>>) -> Self {
        let len = coeffs.len();
        let n = len / 2;
        Self {
            coeffs,
            bound: vec![AdditiveShare::zero(); n],
            current_len: len,
            is_bound: false,
        }
    }

    fn zeros(len: usize) -> Self {
        Self::new(vec![AdditiveShare::zero(); len])
    }

    fn len(&self) -> usize {
        self.current_len
    }

    fn get_coeff(&self, index: usize) -> AdditiveShare<F> {
        if self.is_bound {
            self.bound[index]
        } else {
            self.coeffs[index]
        }
    }

    /// Bind the high variable with a public challenge (HighToLow order).
    /// Halves the polynomial: new[i] = left[i] + r * (right[i] - left[i]).
    fn bind(&mut self, r: F) {
        let n = self.current_len / 2;
        if self.is_bound {
            for i in 0..n {
                let left = self.bound[i];
                let right = self.bound[i + n];
                self.bound[i] = left + (right - left) * r;
            }
        } else {
            for i in 0..n {
                let left = self.coeffs[i];
                let right = self.coeffs[i + n];
                self.bound[i] = left + (right - left) * r;
            }
            self.is_bound = true;
        }
        self.current_len = n;
    }
}

const LOG_K: usize = XLEN * 2; // 128

/// Public per-cycle data extracted from the trace before it is cleared.
/// Stored in `Rep3LookupsDagWorker` and consumed when building the ReadRaf worker.
pub struct ReadRafCycleData {
    pub lookup_tables: Vec<Option<LookupTables<XLEN>>>,
    pub is_interleaved_operands: Vec<bool>,
}

impl ReadRafCycleData {
    /// Extract public per-cycle data from the MPC trace.
    pub fn from_rep3_trace(trace: &[Rep3Cycle]) -> Self {
        let (lookup_tables, is_interleaved_operands): (Vec<_>, Vec<_>) = trace
            .iter()
            .map(|cycle| {
                let table: Option<LookupTables<XLEN>> =
                    InstructionLookup::<XLEN>::lookup_table(cycle);
                let is_interleaved = cycle
                    .instruction()
                    .circuit_flags()
                    .is_interleaved_operands();
                (table, is_interleaved)
            })
            .unzip();
        Self {
            lookup_tables,
            is_interleaved_operands,
        }
    }
}

/// MPC-compatible version of `LookupTables::combine`.
///
/// The vanilla `combine(prefixes, suffixes) -> F` is linear in suffixes.
/// We extract per-suffix weights by probing with unit vectors, then apply
/// those weights (public F) to the additive suffix shares.
fn combine_shared<F: JoltField>(
    table: &LookupTables<XLEN>,
    prefixes: &[PrefixEval<F>],
    shared_suffixes: &[AdditiveShare<F>],
) -> AdditiveShare<F> {
    let n = shared_suffixes.len();
    // Extract weight for each suffix by evaluating combine with unit vector e_i
    let mut result = AdditiveShare::<F>::zero();
    for i in 0..n {
        let mut unit: Vec<SuffixEval<F>> = vec![F::zero(); n];
        unit[i] = F::one();
        let weight: F = table.combine(prefixes, &unit);
        result = result + shared_suffixes[i] * weight;
    }
    result
}
const PHASES: usize = 8;
const M: usize = 1 << LOG_M; // 65536
const DEGREE: usize = 3;

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
    /// Per-cycle: true if operands are interleaved.
    is_interleaved_operands: Vec<bool>,

    // -- Shared per-cycle accumulators --
    /// u_evals[j] starts as trivial share of eq(r_cycle, j), gets condensed each phase.
    u_evals: Vec<Rep3PrimeFieldShare<F>>,
    /// ra_acc[j] accumulates eq(k_j, r_address) across phases. Starts as trivial share of 1.
    ra_acc: Vec<Rep3PrimeFieldShare<F>>,
    /// After LOG_K rounds, ra becomes a dense polynomial for the log_T rounds.
    ra: Option<Rep3DensePolynomial<F>>,

    // -- Phase polynomials (length M = 65536) --
    /// Per-table suffix polynomials stored as additive shares (no reshare needed).
    /// Built via local `Rep3 * Rep3 → Additive` in FWHT unmasking.
    suffix_polys: Vec<Vec<AdditiveDensePoly<F>>>,
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
    /// Full 128-bit lookup indices per cycle (ring-shared, arithmetic domain).
    /// Used to extract suffix bits per phase.
    lookup_indices: Vec<Rep3RingShare<u128>>,

    // -- Mask data from witness --
    /// The D Rep3OneHotPolynomials (owned; reused across phases).
    one_hot_polys: [Rep3OneHotPolynomial<F>; D],
    /// Per-phase cached Ehat16 (FWHT of E16 tensor product). Length M.
    ehat16: Option<Vec<Rep3PrimeFieldShare<F>>>,
    /// Per-phase cached c16[j] (public masked 16-bit keys). None means inactive cycle.
    c16: Vec<Option<u16>>,
    /// Per-phase: which pair of one_hot_polys indices (hi, lo) form the 16-bit key.
    current_phase_pair: (usize, usize),

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
    io_ctx: IoContextPool<N>,
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3ReadRafSumcheckWorker<F, N> {
    /// Construct the ReadRaf worker.
    ///
    /// # Arguments
    /// - `gamma`: the ReadRaf challenge (drawn by coordinator, broadcast to workers)
    /// - `rv_claim`, `raf_claim`: the virtual polynomial claims from Spartan outer sumcheck
    /// - `one_hot_polys`: the D Rep3OneHotPolynomials from witness generation
    /// - `eq_r_cycle_public`: eq(r_cycle, ·) evaluations (public, length T)
    /// - `cycle_data`: pre-extracted public per-cycle data (lookup tables, interleaved flags)
    /// - `io_ctx`: a forked IoContext for MPC operations during phase transitions
    /// - `party_id`: this party's ID
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "ReadRafWorker::new")]
    pub fn new(
        gamma: F,
        rv_claim: F,
        raf_claim: F,
        one_hot_polys: [Rep3OneHotPolynomial<F>; D],
        eq_r_cycle_public: &[F],
        cycle_data: ReadRafCycleData,
        lookup_indices: Vec<Rep3RingShare<u128>>,
        mut io_ctx: IoContextPool<N>,
        party_id: PartyID,
    ) -> eyre::Result<Self> {
        let num_cycles = cycle_data.lookup_tables.len();
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

        let ReadRafCycleData {
            lookup_tables,
            is_interleaved_operands,
        } = cycle_data;

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
            if is_interleaved {
                interleaved_cycles.push(j);
            } else {
                identity_cycles.push(j);
            }
        }

        // -- Initialize u_evals and ra_acc --
        let u_evals: Vec<Rep3PrimeFieldShare<F>> = eq_r_cycle_public
            .iter()
            .map(|&v| promote_to_trivial_share(v, party_id))
            .collect();
        let ra_acc: Vec<Rep3PrimeFieldShare<F>> =
            vec![promote_to_trivial_share(F::one(), party_id); num_cycles];

        // -- Initialize suffix polynomials (empty, filled in init_phase) --
        let suffix_polys: Vec<Vec<AdditiveDensePoly<F>>> = LookupTables::<XLEN>::iter()
            .map(|table| {
                table
                    .suffixes()
                    .iter()
                    .map(|_| AdditiveDensePoly::zeros(M))
                    .collect()
            })
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

        let empty_q = || AdditiveDensePoly::zeros(M);
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
            one_hot_polys,
            ehat16: None,
            c16: vec![None; num_cycles],
            current_phase_pair: (0, 0),
            eq_r_cycle: MultilinearPolynomial::from(eq_r_cycle_public.to_vec()),
            combined_val_polynomial: None,
            party_id,
        };

        // Initialize phase 0
        state.init_phase(0, &mut io_ctx)?;

        Ok(Self {
            gamma,
            gamma_squared: gamma.square(),
            rv_claim,
            raf_claim,
            log_T,
            state,
            io_ctx,
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
    #[tracing::instrument(skip_all, name = "ReadRaf::init_phase", fields(phase))]
    fn init_phase<N: Rep3NetworkWorker>(
        &mut self,
        phase: usize,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<()> {
        eyre::ensure!(phase < PHASES, "phase out of range: {phase}");
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
        self.current_phase_pair = (hi, lo);

        // -- Step G1: derive c16 --
        for j in 0..self.c16.len() {
            let c_hi = self.one_hot_polys[hi].masked_indices_c[j];
            let c_lo = self.one_hot_polys[lo].masked_indices_c[j];
            self.c16[j] = match (c_hi, c_lo) {
                (Some(h), Some(l)) => Some(((h as u16) << 8) | (l as u16)),
                _ => None,
            };
        }

        // -- Step F: compute Ehat16 via tensor product --
        let _ehat_span = tracing::info_span!("ehat16_tensor_product").entered();
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
            for _b_idx in 0..256 {
                a_expanded.push(ehat8_hi[a_idx]);
            }
        }
        for _a_idx in 0..256 {
            for b_idx in 0..256 {
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

            // Build Ehat16_prev from previous phase's one-hot polys
            let mut ehat8_prev_hi: Vec<Rep3PrimeFieldShare<F>> = self.one_hot_polys[prev_hi]
                .rand_ohv_e_field
                .as_ref()
                .clone();
            let mut ehat8_prev_lo: Vec<Rep3PrimeFieldShare<F>> = self.one_hot_polys[prev_lo]
                .rand_ohv_e_field
                .as_ref()
                .clone();
            fwht_rep3_in_place(&mut ehat8_prev_hi);
            fwht_rep3_in_place(&mut ehat8_prev_lo);

            let mut a_exp = Vec::with_capacity(M);
            let mut b_exp = Vec::with_capacity(M);
            for a_idx in 0..256 {
                for _b_idx in 0..256 {
                    a_exp.push(ehat8_prev_hi[a_idx]);
                }
            }
            for _a_idx in 0..256 {
                for b_idx in 0..256 {
                    b_exp.push(ehat8_prev_lo[b_idx]);
                }
            }
            let ehat16_prev = rep3_arith::mul_vec(&a_exp, &b_exp, io_ctx.main())?;

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

            // Batch multiply: u_evals[j] *= eq_shifted[c16_prev[j]]
            // and ra_acc[j] *= eq_shifted[c16_prev[j]]
            let vals_to_mul: Vec<Rep3PrimeFieldShare<F>> = c16_prev
                .iter()
                .map(|opt| match opt {
                    Some(c) => eq_shifted[*c as usize],
                    None => Rep3PrimeFieldShare::zero_share(),
                })
                .collect();

            let u_products = rep3_arith::mul_vec(&self.u_evals, &vals_to_mul, io_ctx.main())?;

            // For inactive cycles (None), u_evals stays zero (mul by zero share).
            // For active cycles, update in place.
            // Note: only u_evals is updated in condensation (matching vanilla).
            // ra_acc is only updated in cache_phase.
            self.u_evals = u_products;
        }

        drop(_cond_span);

        // -- Build suffix polynomials --
        self.init_suffix_polys(phase, io_ctx)?;

        // -- Initialize prefix decompositions' Q polynomials --
        // We manage our own shared Q arrays ({left,right}_operand_q, identity_q) for
        // MPC computation. However, the PSDs' internal Q arrays are also bound every
        // round via PSD::bind(). Reset them to length-M zero polys so binding doesn't
        // underflow on subsequent phases. (Vanilla calls init_Q each phase, which
        // has the same effect.)
        self.identity_ps.init_Q(&[], &[]);
        self.right_operand_ps.init_Q(&[], &[]);
        self.left_operand_ps.init_Q(&[], &[]);

        self.init_operand_q_polys(phase, io_ctx)?;

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
    /// Optimizations vs naive per-(table,suffix) approach:
    /// 1. **Batch by suffix type**: evaluate each unique suffix MLE once over ALL cycles,
    ///    then each (table, suffix) pair picks its subset. Removes redundant MPC evaluations.
    /// 2. **Local multiply + histogram reshare**: instead of `mul_vec(u, t)` per table (O(N)
    ///    communication), do local `Rep3 × Rep3 → Additive` multiply, scatter into additive
    ///    histogram, then `reshare_additive_many` the histogram (O(M) communication).
    ///    Since M=65536 ≪ N (trace length), this is much cheaper.
    #[tracing::instrument(skip_all, name = "ReadRaf::init_suffix_polys", fields(phase))]
    fn init_suffix_polys<N: Rep3NetworkWorker>(
        &mut self,
        phase: usize,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<()> {
        use crate::utils::future_ring::Rep3RingFutureExt;
        use crate::zkvm::instruction::suffixes::{evaluate_suffix_mle_batched, SuffixFuture};
        use rayon::prelude::*;

        let ehat16 = self
            .ehat16
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ehat16 missing in init_suffix_polys"))?;
        let inv_m = F::from(M as u64).inverse().expect("M invertible");

        let suffix_len = (PHASES - 1 - phase) * LOG_M;
        let all_suffix_bits: Option<Vec<Rep3RingShare<u128>>> = if suffix_len > 0 {
            let mask = RingElement((1u128 << suffix_len) - 1);
            Some(
                self.lookup_indices
                    .par_iter()
                    .map(|idx| *idx & mask)
                    .collect(),
            )
        } else {
            None
        };

        // -- Step 1: Collect unique suffix types needed across all tables --
        let mut needed_suffixes: Vec<Suffixes> = Vec::new();
        for table in LookupTables::<XLEN>::iter() {
            for suffix in table.suffixes() {
                if !matches!(suffix, Suffixes::One)
                    && suffix_len > 0
                    && !needed_suffixes.contains(&suffix)
                {
                    needed_suffixes.push(suffix);
                }
            }
        }

        // -- Step 2: Batch-evaluate each unique suffix over ALL cycles --
        // Each suffix returns Vec<SuffixFuture<F>> with deferred ring→field conversion.
        // We flatten all futures and fulfill_batched once.
        let _span = info_span!("suffix_mle", n_suffixes = needed_suffixes.len()).entered();
        let mut suffix_futures_map: std::collections::HashMap<u8, Vec<SuffixFuture<F>>> =
            std::collections::HashMap::new();
        if let Some(ref all_bits) = all_suffix_bits {
            for &suffix in &needed_suffixes {
                // let _inner =
                //     info_span!("eval_suffix", suffix = ?suffix).entered();
                let futures = evaluate_suffix_mle_batched(
                    &suffix,
                    all_bits,
                    suffix_len,
                    io_ctx.main(),
                    self.party_id,
                )?;
                suffix_futures_map.insert(suffix as u8, futures);
            }
        }
        drop(_span);

        // Flatten all futures, fulfill_batched, then split back.
        let _span = info_span!("suffix_fulfill").entered();
        let num_cycles = self.lookup_indices.len();
        let mut suffix_keys: Vec<u8> = suffix_futures_map.keys().copied().collect();
        suffix_keys.sort_unstable(); // deterministic order across all parties
        let mut all_futures: Vec<SuffixFuture<F>> =
            Vec::with_capacity(suffix_keys.len() * num_cycles);
        for &key in &suffix_keys {
            all_futures.extend(suffix_futures_map.remove(&key).unwrap());
        }
        let all_field: Vec<Rep3PrimeFieldShare<F>> =
            all_futures.fulfill_batched(io_ctx, |share, ()| share)?;
        // Split back into per-suffix chunks (zero-copy via slicing)
        let suffix_eval_cache: std::collections::HashMap<u8, &[Rep3PrimeFieldShare<F>]> =
            suffix_keys
                .iter()
                .enumerate()
                .map(|(i, &key)| (key, &all_field[i * num_cycles..(i + 1) * num_cycles]))
                .collect();
        drop(_span);

        // Debug: verify suffix evals (can be enabled when needed)
        // #[cfg(debug_assertions)]
        // { ... }

        // -- Step 3: Build histograms in parallel --
        // Collect all (table_idx, suffix_idx, table, suffix) work items.
        let _span = info_span!("build_histograms").entered();

        // Enumerate all (table_idx, suffix_idx) pairs into a flat work list.
        struct WorkItem {
            table_idx: usize,
            suffix_idx: usize,
            suffix: Suffixes,
        }
        let mut work_items: Vec<WorkItem> = Vec::new();
        for (table_idx, table) in LookupTables::<XLEN>::iter().enumerate() {
            for (suffix_idx, suffix) in table.suffixes().into_iter().enumerate() {
                work_items.push(WorkItem {
                    table_idx,
                    suffix_idx,
                    suffix,
                });
            }
        }

        // Build histograms in parallel. Each work item produces either:
        //   - Rep3 histogram (constant suffix) → ready for FWHT
        //   - Additive histogram (secret suffix) → needs reshare
        enum HistResult<F: JoltField> {
            Rep3(usize, usize, Vec<Rep3PrimeFieldShare<F>>),
            Additive(usize, usize, Vec<AdditiveShare<F>>),
            ZeroPoly(usize, usize),
        }

        let u_evals = &self.u_evals;
        let c16 = &self.c16; // TODO: likely need to get c16 for current phase?
        let lookup_indices_by_table = &self.lookup_indices_by_table;

        let hist_results: Vec<HistResult<F>> = work_items
            .par_iter()
            .map(|item| {
                let table_cycles = &lookup_indices_by_table[item.table_idx];
                // Tables with no cycles: zero histogram, skip FWHT unmask.
                // Still emit a Rep3 entry so the poly gets reset to size M.
                if table_cycles.is_empty() {
                    return HistResult::ZeroPoly(item.table_idx, item.suffix_idx);
                }
                if suffix_len == 0 {
                    let mut h_c = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
                    let constant_u64 = item
                        .suffix
                        .suffix_mle::<XLEN>(LookupBits::new(0u128, 0usize));
                    if constant_u64 != 0 {
                        let constant_f = F::from_u128(constant_u64 as u128);
                        for &j in table_cycles {
                            if let Some(c) = c16[j] {
                                h_c[c as usize] += u_evals[j] * constant_f;
                            }
                        }
                    }
                    HistResult::Rep3(item.table_idx, item.suffix_idx, h_c)
                } else if matches!(item.suffix, Suffixes::One) {
                    let mut h_c = vec![Rep3PrimeFieldShare::<F>::zero_share(); M];
                    for &j in table_cycles {
                        if let Some(c) = c16[j] {
                            h_c[c as usize] += u_evals[j];
                        }
                    }
                    HistResult::Rep3(item.table_idx, item.suffix_idx, h_c)
                } else {
                    let suffix_evals = suffix_eval_cache.get(&(item.suffix as u8)).unwrap();
                    let mut h_add = vec![AdditiveShare::<F>::zero(); M];
                    for &j in table_cycles {
                        if let Some(c) = c16[j] {
                            let w: AdditiveShare<F> = u_evals[j] * suffix_evals[j];
                            h_add[c as usize] = h_add[c as usize] + w;
                        }
                    }
                    HistResult::Additive(item.table_idx, item.suffix_idx, h_add)
                }
            })
            .collect();

        // Partition into rep3 vs additive, and handle zero polys (no FWHT needed)
        let mut rep3_hist_entries: Vec<(usize, usize, Vec<Rep3PrimeFieldShare<F>>)> = Vec::new();
        let mut hist_entries_to_reshare: Vec<(usize, usize, Vec<AdditiveShare<F>>)> = Vec::new();
        for result in hist_results {
            match result {
                HistResult::Rep3(ti, si, h) => rep3_hist_entries.push((ti, si, h)),
                HistResult::Additive(ti, si, h) => hist_entries_to_reshare.push((ti, si, h)),
                HistResult::ZeroPoly(ti, si) => {
                    self.suffix_polys[ti][si] = AdditiveDensePoly::zeros(M);
                }
            }
        }
        drop(_span);

        // -- Step 4: Batch reshare all additive histograms to Rep3 in one round --
        let _span = info_span!("reshare_hists", n = hist_entries_to_reshare.len()).entered();
        let reshared_histograms = if !hist_entries_to_reshare.is_empty() {
            let total_len = hist_entries_to_reshare.len() * M;
            let mut flat_additive = Vec::with_capacity(total_len);
            for (_, _, ref hist) in &hist_entries_to_reshare {
                flat_additive.extend_from_slice(hist);
            }
            let flat_rep3 = rep3_arith::reshare_additive_many(&flat_additive, io_ctx.main())?;
            flat_rep3
                .chunks_exact(M)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        drop(_span);

        // -- Step 5: Merge reshared histograms back, then FWHT unmask all (parallel, local) --
        // Replace additive histograms with their reshared Rep3 versions, then
        // combine both lists into one unified FWHT pass.
        let mut all_hist_entries: Vec<(usize, usize, Vec<Rep3PrimeFieldShare<F>>)> =
            rep3_hist_entries;
        for ((ti, si, _additive), reshared) in hist_entries_to_reshare
            .into_iter()
            .zip(reshared_histograms.into_iter())
        {
            all_hist_entries.push((ti, si, reshared));
        }

        let _span = info_span!("fwht_unmask", n = all_hist_entries.len()).entered();

        let all_polys: Vec<(usize, usize, AdditiveDensePoly<F>)> = all_hist_entries
            .into_par_iter()
            .map(|(table_idx, suffix_idx, mut h_c)| {
                fwht_rep3_in_place(&mut h_c);
                let mut h: Vec<AdditiveShare<F>> = h_c
                    .iter()
                    .zip(ehat16.iter())
                    .map(|(&a, &b)| a * b)
                    .collect();
                fwht_additive_in_place(&mut h);
                for v in h.iter_mut() {
                    *v = *v * inv_m;
                }
                (table_idx, suffix_idx, AdditiveDensePoly::new(h))
            })
            .collect();

        for (table_idx, suffix_idx, poly) in all_polys {
            self.suffix_polys[table_idx][suffix_idx] = poly;
        }
        drop(_span);

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
    ) -> eyre::Result<()> {
        use crate::zkvm::instruction::suffixes::{
            compute_operand_q_suffix_evals, OperandQSuffixEvals,
        };

        let ehat16 = self.ehat16.as_ref().unwrap();
        let suffix_len = (PHASES - 1 - phase) * LOG_M;
        let num_cycles = self.c16.len();

        // Compute per-cycle suffix evaluations for the 3 secret suffix types.
        // The 2 constant types (ShiftHalf, Shift) are handled inline.
        let OperandQSuffixEvals {
            left_operand,
            right_operand,
            identity,
        } = compute_operand_q_suffix_evals(
            phase,
            &self.lookup_indices,
            num_cycles,
            io_ctx.main(),
            self.party_id,
        )?;

        // Constant suffix values
        let shift_half_val = if suffix_len > 0 {
            F::from_u128(1u128 << (suffix_len / 2))
        } else {
            F::one()
        };
        let shift_val = if suffix_len > 0 {
            F::from_u128(1u128 << suffix_len)
        } else {
            F::one()
        };

        // Build 6 histograms: left[0], left[1], right[0], right[1], identity[0], identity[1]
        let mut histograms: Vec<Vec<Rep3PrimeFieldShare<F>>> = (0..6)
            .map(|_| vec![Rep3PrimeFieldShare::<F>::zero_share(); M])
            .collect();

        if suffix_len == 0 {
            // All suffixes are public constants — no MPC multiplication needed.
            // ShiftHalf, OperandPoly(Left/Right), Shift, Identity all return constants
            // when suffix_len == 0 (the bitvector is empty).
            let left_op_val = F::zero(); // OperandPolynomial::suffix_mle(empty) = u128::from(empty) = 0
            let right_op_val = F::zero();
            let identity_val = F::zero(); // IdentityPolynomial::suffix_mle(empty) = 0

            for &j in &self.interleaved_cycles {
                if let Some(c) = self.c16[j] {
                    let u = self.u_evals[j];
                    histograms[0][c as usize] += u * shift_half_val;
                    if left_op_val != F::zero() {
                        histograms[1][c as usize] += u * left_op_val;
                    }
                    histograms[2][c as usize] += u * shift_half_val;
                    if right_op_val != F::zero() {
                        histograms[3][c as usize] += u * right_op_val;
                    }
                }
            }

            for &j in &self.identity_cycles {
                if let Some(c) = self.c16[j] {
                    let u = self.u_evals[j];
                    histograms[4][c as usize] += u * shift_val;
                    if identity_val != F::zero() {
                        histograms[5][c as usize] += u * identity_val;
                    }
                }
            }
        } else {
            // suffix_len > 0: secret suffix values for operand/identity suffixes.
            // left[0], right[0] are constant (ShiftHalf), identity[0] is constant (Shift).
            // left[1], right[1], identity[1] are secret (need Rep3 × Rep3 → Additive).
            //
            // Use local multiply into additive histograms + reshare (O(3M) communication)
            // instead of 3 × mul_vec (3 × O(N) communication).

            let mut additive_hists: Vec<Vec<AdditiveShare<F>>> = (0..3)
                .map(|_| vec![AdditiveShare::<F>::zero(); M])
                .collect();

            for &j in &self.interleaved_cycles {
                if let Some(c) = self.c16[j] {
                    let u = self.u_evals[j];
                    let ci = c as usize;
                    histograms[0][ci] += u * shift_half_val;
                    histograms[2][ci] += u * shift_half_val;
                    // Local Rep3 × Rep3 → Additive
                    additive_hists[0][ci] = additive_hists[0][ci] + (u * left_operand[j]);
                    additive_hists[1][ci] = additive_hists[1][ci] + (u * right_operand[j]);
                }
            }

            for &j in &self.identity_cycles {
                if let Some(c) = self.c16[j] {
                    let u = self.u_evals[j];
                    let ci = c as usize;
                    histograms[4][ci] += u * shift_val;
                    additive_hists[2][ci] = additive_hists[2][ci] + (u * identity[j]);
                }
            }

            // Batch reshare: flatten 3 additive histograms, reshare in one round
            let _reshare = info_span!("q_reshare").entered();
            let mut flat: Vec<AdditiveShare<F>> = Vec::with_capacity(3 * M);
            for h in &additive_hists {
                flat.extend_from_slice(h);
            }
            let flat_rep3 = rep3_arith::reshare_additive_many(&flat, io_ctx.main())?;
            drop(_reshare);

            // Scatter reshared histograms into the Rep3 histogram slots
            let left_hist = &flat_rep3[0..M];
            let right_hist = &flat_rep3[M..2 * M];
            let identity_hist = &flat_rep3[2 * M..3 * M];

            for i in 0..M {
                histograms[1][i] = left_hist[i];
                histograms[3][i] = right_hist[i];
                histograms[5][i] = identity_hist[i];
            }
        }

        // FWHT + local pointwise multiply with Ehat16 + inverse FWHT (parallel)
        use rayon::prelude::*;
        let inv_m = F::from(M as u64).inverse().expect("M invertible");
        let _fwht_span = info_span!("q_fwht_unmask").entered();
        let q_polys: Vec<AdditiveDensePoly<F>> = histograms
            .into_par_iter()
            .map(|mut h| {
                fwht_rep3_in_place(&mut h);
                let mut h_k: Vec<AdditiveShare<F>> =
                    h.iter().zip(ehat16.iter()).map(|(&a, &b)| a * b).collect();
                fwht_additive_in_place(&mut h_k);
                for v in h_k.iter_mut() {
                    *v = *v * inv_m;
                }
                AdditiveDensePoly::new(h_k)
            })
            .collect();
        drop(_fwht_span);

        let mut q_iter = q_polys.into_iter();
        self.left_operand_q[0] = q_iter.next().unwrap();
        self.left_operand_q[1] = q_iter.next().unwrap();
        self.right_operand_q[0] = q_iter.next().unwrap();
        self.right_operand_q[1] = q_iter.next().unwrap();
        self.identity_q[0] = q_iter.next().unwrap();
        self.identity_q[1] = q_iter.next().unwrap();

        Ok(())
    }

    /// Cache the phase after all LOG_M rounds are bound: update ra_acc.
    fn cache_phase<N: Rep3NetworkWorker>(
        &mut self,
        _phase: usize,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<()> {
        // ra_acc[j] *= v[bucket] where bucket = prefix(k_j) % M for the current phase.
        // In masked domain: bucket = (c16[j] ^ r16) % M = c16[j] ^ r16 (since M = 2^16).
        // We use the shifted v table: v_shifted[c] = v[c ^ r16].
        let ehat16 = self.ehat16.as_ref().unwrap();
        let v_values = self.v.clone_values();
        assert_eq!(v_values.len(), M);

        let v_shifted = shift_eq_table_with_mask(&v_values, ehat16);

        let vals_to_mul: Vec<Rep3PrimeFieldShare<F>> = self
            .c16
            .iter()
            .map(|opt| match opt {
                Some(c) => v_shifted[*c as usize],
                None => Rep3PrimeFieldShare::zero_share(),
            })
            .collect();

        let ra_products = rep3_arith::mul_vec(&self.ra_acc, &vals_to_mul, io_ctx.main())?;
        self.ra_acc = ra_products;

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

        for (j, (table, &is_interleaved)) in self
            .lookup_tables
            .iter()
            .zip(self.is_interleaved_operands.iter())
            .enumerate()
        {
            if let Some(table) = table {
                let suffixes: Vec<_> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| F::from_u64(suffix.suffix_mle::<XLEN>(LookupBits::new(0, 0))))
                    .collect();
                combined_val_poly[j] += table.combine(&prefixes, &suffixes);
            }

            if is_interleaved {
                combined_val_poly[j] += gamma
                    * self.prefix_registry.checkpoints
                        [jolt_core::poly::prefix_suffix::Prefix::LeftOperand]
                        .unwrap()
                    + gamma_squared
                        * self.prefix_registry.checkpoints
                            [jolt_core::poly::prefix_suffix::Prefix::RightOperand]
                            .unwrap();
            } else {
                combined_val_poly[j] += gamma_squared
                    * self.prefix_registry.checkpoints
                        [jolt_core::poly::prefix_suffix::Prefix::Identity]
                        .unwrap();
            }
        }

        self.combined_val_polynomial = Some(MultilinearPolynomial::from(combined_val_poly));

        // Build ra polynomial from ra_acc for the log_T rounds.
        self.ra = Some(Rep3DensePolynomial::new(std::mem::take(&mut self.ra_acc)));
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F>
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

            let mut evals = [AdditiveShare::<F>::zero(); DEGREE];
            for i in 0..half {
                let eq_vals = ps
                    .eq_r_cycle
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let val_vals =
                    combined_val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let ra_vals = ra.sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);

                for d in 0..DEGREE {
                    // eq * val is public×public = public
                    let eq_val: F = eq_vals[d] * val_vals[d];
                    // multiply by shared ra: public × shared → Rep3 share → additive
                    let contribution = (ra_vals[d] * eq_val).into_additive();
                    evals[d] += contribution;
                }
            }

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

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = &mut self.state;
        ps.r.push(r_j);

        if round < LOG_K {
            // Bind suffix polynomials (additive)
            let r: F = r_j.into();
            for polys in ps.suffix_polys.iter_mut() {
                for poly in polys.iter_mut() {
                    poly.bind(r);
                }
            }

            // Bind prefix-suffix decompositions (public P via PSD, additive Q arrays)
            ps.identity_ps.bind(r_j);
            ps.right_operand_ps.bind(r_j);
            ps.left_operand_ps.bind(r_j);
            for q in ps.left_operand_q.iter_mut() {
                q.bind(r);
            }
            for q in ps.right_operand_q.iter_mut() {
                q.bind(r);
            }
            for q in ps.identity_q.iter_mut() {
                q.bind(r);
            }
            ps.v.update(r_j);

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
                ps.cache_phase(phase, &mut self.io_ctx)
                    .expect("cache_phase failed");

                if phase != PHASES - 1 {
                    ps.init_phase(phase + 1, &mut self.io_ctx)
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
        let read_checking = self.prover_msg_read_checking(round);
        let raf = self.prover_msg_raf();
        [read_checking[0] + raf[0], read_checking[1] + raf[1]]
    }

    /// Read-checking component of the prover message.
    fn prover_msg_read_checking(&self, j: usize) -> [AdditiveShare<F>; 2] {
        let ps = &self.state;
        let lookup_tables: Vec<_> = LookupTables::<XLEN>::iter().collect();
        let len = ps.suffix_polys[0][0].len();
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 {
            ps.r.last().copied()
        } else {
            None
        };

        let mut eval_0 = AdditiveShare::<F>::zero();
        let mut eval_2_left = AdditiveShare::<F>::zero();
        let mut eval_2_right = AdditiveShare::<F>::zero();

        for b in 0..len / 2 {
            let b_bits = LookupBits::new(b as u128, log_len - 1);

            let prefixes_c0: Vec<_> = Prefixes::iter()
                .map(|prefix| {
                    prefix.prefix_mle::<XLEN, F, F::Challenge>(
                        &ps.prefix_checkpoints,
                        r_x,
                        0,
                        b_bits,
                        j,
                    )
                })
                .collect();
            let prefixes_c2: Vec<_> = Prefixes::iter()
                .map(|prefix| {
                    prefix.prefix_mle::<XLEN, F, F::Challenge>(
                        &ps.prefix_checkpoints,
                        r_x,
                        2,
                        b_bits,
                        j,
                    )
                })
                .collect();

            for (table, suffixes) in lookup_tables.iter().zip(ps.suffix_polys.iter()) {
                // suffix_left[s] = suffix_polys[table][s][b] (additive)
                // suffix_right[s] = suffix_polys[table][s][b + len/2] (additive)
                let suffixes_left: Vec<AdditiveShare<F>> =
                    suffixes.iter().map(|s| s.get_coeff(b)).collect();
                let suffixes_right: Vec<AdditiveShare<F>> =
                    suffixes.iter().map(|s| s.get_coeff(b + len / 2)).collect();

                // combine: public prefixes × additive suffixes → additive
                let combined_c0 = combine_shared(table, &prefixes_c0, &suffixes_left);
                let combined_c2_left = combine_shared(table, &prefixes_c2, &suffixes_left);
                let combined_c2_right = combine_shared(table, &prefixes_c2, &suffixes_right);

                eval_0 += combined_c0;
                eval_2_left += combined_c2_left;
                eval_2_right += combined_c2_right;
            }
        }

        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }

    /// RAF component of the prover message.
    ///
    /// Mirrors vanilla `prover_msg_raf` but uses our shared Q arrays
    /// (`{left,right,identity}_operand_q`) paired with public P polynomials
    /// from the `PrefixRegistry`.
    fn prover_msg_raf(&self) -> [AdditiveShare<F>; 2] {
        let ps = &self.state;
        let len = ps.identity_q[0].len();

        let mut left_0 = AdditiveShare::<F>::zero();
        let mut left_2 = AdditiveShare::<F>::zero();
        let mut right_0 = AdditiveShare::<F>::zero();
        let mut right_2 = AdditiveShare::<F>::zero();

        for b in 0..len / 2 {
            // For each decomposition:
            // P[0] = prefix polynomial (from registry), P[1] = None (constant 1)
            // Q[0], Q[1] are our shared polynomials
            //
            // sumcheck_evals(index) computes:
            //   For each (P_i, Q_i):
            //     p_evals = if P_i is Some: (p[index], 2*p[index+len/2] - p[index])
            //               else: (1, 1)
            //     q_left = Q_i[index], q_right = Q_i[index + len/2]
            //     accumulate (p_0 * q_left, p_2 * q_left, p_2 * q_right)
            //   Result: (Σ p_0*q_left, 2*(Σ p_2*q_right) - (Σ p_2*q_left))

            let (l0, l2) = psd_sumcheck_evals_shared(
                ps.prefix_registry.polys[Prefix::LeftOperand as usize].as_ref(),
                &ps.left_operand_q,
                b,
                len,
            );
            let (r0, r2) = psd_sumcheck_evals_shared(
                ps.prefix_registry.polys[Prefix::RightOperand as usize].as_ref(),
                &ps.right_operand_q,
                b,
                len,
            );
            let (i0, i2) = psd_sumcheck_evals_shared(
                ps.prefix_registry.polys[Prefix::Identity as usize].as_ref(),
                &ps.identity_q,
                b,
                len,
            );

            left_0 += l0;
            left_2 += l2;
            right_0 += i0 + r0;
            right_2 += i2 + r2;
        }

        [
            left_0 * self.gamma + right_0 * self.gamma_squared,
            left_2 * self.gamma + right_2 * self.gamma_squared,
        ]
    }

}

/// Lagrange interpolation through 4 points (0, y0), (1, y1), (2, y2), (3, y3) at x.
fn lagrange_interp_4<F: JoltField>(
    y0: AdditiveShare<F>,
    y1: AdditiveShare<F>,
    y2: AdditiveShare<F>,
    y3: AdditiveShare<F>,
    x: F,
) -> AdditiveShare<F> {
    let x0 = F::zero();
    let x1 = F::one();
    let x2 = F::from(2u64);
    let x3 = F::from(3u64);

    let l0 = (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3));
    let l1 = (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3));
    let l2 = (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3));
    let l3 = (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2));

    y0 * l0 + y1 * l1 + y2 * l2 + y3 * l3
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
