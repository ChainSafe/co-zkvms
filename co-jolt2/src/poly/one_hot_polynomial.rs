use std::sync::{Arc, RwLock};

use allocative::Allocative;
use ark_ec::AffineRepr;
use ark_ec::CurveGroup;
use jolt_core::msm::VariableBaseMSM;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use jolt_core::poly::one_hot_polynomial::EqAddressState;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::utils::math::Math;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::{binary, conversion, gadgets};
use mpc_core::protocols::{rep3::Rep3PrimeFieldShare, rep3_ring::Rep3RingShare};
use rayon::prelude::*;

use crate::poly::ra_poly::{shifted_table_from_rand_ohv, Rep3RaPolynomial};
use crate::utils::fwht::fwht_in_place;
use jolt_core::field::JoltField;

/// Represents a one-hot multilinear polynomial (ra/wa) used
/// in Twist/Shout. Perhaps somewhat unintuitively, the implementation
/// in this file is currently only used to compute the Dory
/// commitment and in the opening proof reduction sumcheck.
#[derive(Clone, Debug)] // Allocative
pub struct Rep3OneHotPolynomial<F: JoltField> {
    /// The size of the "address" space for this polynomial.
    pub K: usize,

    /// Public masked index per cycle: `Some(c)` where `c = open(k(j) XOR r)`;
    /// `None` means this cycle has no address (row is all-zero).
    pub masked_indices_c: Arc<Vec<Option<u8>>>,
    /// Field-injected RandOHV `E_field` (length K): secret-shared one-hot vector `e(r)`.
    /// Used to select `table[r XOR c]` from a public table via an inner product.
    pub rand_ohv_e_field: Arc<Vec<Rep3PrimeFieldShare<F>>>,
    /// The number of variables that have been bound over the
    /// course of sumcheck so far.
    num_variables_bound: usize,
    /// Helper array for address-variable rounds of opening reduction sumcheck.
    G: Vec<Rep3PrimeFieldShare<F>>,
    /// Helper polynomial for cycle-variable rounds: `H(j) = eq(k(j), u)` for the fully-bound address
    /// challenge vector `u`.
    H: Arc<RwLock<Rep3RaPolynomial<u8, F>>>,
    /// RandOHV mask `r` share — stored for reconstruction of plaintext indices.
    #[cfg(test)]
    r_share: Rep3RingShare<u8>,
}

impl<F: JoltField> Default for Rep3OneHotPolynomial<F> {
    fn default() -> Self {
        Self {
            K: 1,
            masked_indices_c: Arc::new(vec![]),
            rand_ohv_e_field: Arc::new(vec![]),
            num_variables_bound: 0,
            G: vec![],
            H: Arc::new(RwLock::new(Rep3RaPolynomial::None)),
            #[cfg(test)]
            r_share: Rep3RingShare::default(),
        }
    }
}

impl<F: JoltField> Rep3OneHotPolynomial<F> {
    /// Builds a RandOHV proving view for this polynomial:
    /// - samples a secret mask `r` and its one-hot vector `E = e(r)`,
    /// - opens `c[j] = open(k(j) XOR r)` once for all active cycles,
    /// - injects `E` into field shares once (length K).
    #[tracing::instrument(skip_all, name = "one_hot::from_indices", fields(K))]
    pub fn from_indices<N: Rep3Network>(
        nonzero_indices: Vec<Option<Rep3RingShare<u8>>>,
        K: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Self> {
        assert!(K.is_power_of_two(), "K must be a power of two");
        assert!(K <= 1 << 8, "K must be <= 256 for index to fit into u8");
        let log_k = K.log_2();

        // Sample one RandOHV for the mask `r` (binary-sharing domain).
        let (r_share, e_bits) = gadgets::ohv::rand_ohv::<u8, _>(log_k, io_ctx)?;

        // Open masked indices `c[j] = open(k(j) XOR r)` for active cycles only.
        //
        // The set of `None` positions is already known to all parties (it's in the
        // input `Vec<Option<...>>`), so we can save bandwidth by opening only the
        // active entries.
        let active_count = nonzero_indices.iter().filter(|opt| opt.is_some()).count();
        let mut masked_active: Vec<Rep3RingShare<u8>> = Vec::with_capacity(active_count);
        for opt in nonzero_indices.iter() {
            if let Some(kj) = opt {
                masked_active.push(*kj ^ r_share);
            }
        }
        let opened_active = binary::open_vec(&masked_active, io_ctx)?;
        debug_assert_eq!(opened_active.len(), active_count);

        let mut opened_iter = opened_active.into_iter();
        let masked_indices_c: Vec<Option<u8>> = nonzero_indices
            .iter()
            .map(|opt| opt.as_ref().map(|_| opened_iter.next().expect("active index open")))
            .collect();
        debug_assert!(opened_iter.next().is_none(), "opened_active length mismatch");

        // Inject the OHV bits into prime-field shares once.
        let rand_ohv_e_field: Vec<Rep3PrimeFieldShare<F>> =
            conversion::bit_inject_from_bits_to_field_many(&e_bits, io_ctx)?;

        Ok(Self {
            K,
            masked_indices_c: Arc::new(masked_indices_c),
            rand_ohv_e_field: Arc::new(rand_ohv_e_field),
            #[cfg(test)]
            r_share,
            ..Default::default()
        })
    }

    /// Build a `Rep3OneHotPolynomial` from pre-computed parts (useful for tests).
    #[cfg(feature = "test-utils")]
    pub fn from_parts(
        K: usize,
        masked_indices_c: Arc<Vec<Option<u8>>>,
        rand_ohv_e_field: Arc<Vec<Rep3PrimeFieldShare<F>>>,
    ) -> Self {
        Self { K, masked_indices_c, rand_ohv_e_field, ..Default::default() }
    }

    /// Reconstruct the plaintext `nonzero_indices` from 3 parties' Rep3OneHotPolynomial shares.
    ///
    /// Recovery: `k(j) = c[j] XOR r` where `r = s0.r_share.a XOR s1.r_share.a XOR s2.r_share.a`.
    /// All 3 parties share the same `masked_indices_c` (it's public), so we just need `r`.
    #[cfg(test)]
    pub fn reconstruct_indices(polys: [&Self; 3]) -> Vec<Option<u8>> {
        // Binary rep3 reconstruct: secret = a_0 XOR a_1 XOR a_2
        let r = (polys[0].r_share.a ^ polys[1].r_share.a ^ polys[2].r_share.a).0;
        polys[0].masked_indices_c.iter().map(|opt| opt.map(|c| c ^ r)).collect()
    }

    /// The number of rows in the coefficient matrix used to
    /// commit to this polynomial using Dory
    // pub fn num_rows(&self) -> usize {
    //     let T = self.nonzero_indices.len() as u128;
    //     let row_length = DoryGlobals::get_num_columns() as u128;
    //     (T * self.K as u128 / row_length) as usize
    // }

    /// Clone with a fresh (unshared) H. Use this when multiple prover openings
    /// need independent mutable H state — e.g. in the opening proof reduction
    /// where multiple entries for the same polynomial would otherwise share
    /// the same `Arc<RwLock<H>>`.
    pub fn clone_with_fresh_h(&self) -> Self {
        Self {
            K: self.K,
            masked_indices_c: self.masked_indices_c.clone(),
            rand_ohv_e_field: self.rand_ohv_e_field.clone(),
            num_variables_bound: self.num_variables_bound,
            G: self.G.clone(),
            H: Arc::new(RwLock::new(Rep3RaPolynomial::None)),
            #[cfg(test)]
            r_share: self.r_share,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.K.log_2() + self.masked_indices_c.len().log_2()
    }

    /// Computes additive group-element shares of the Dory row commitments for this one-hot
    /// polynomial, using the RandOHV representation.
    ///
    /// Return value is `Vec<G>` where each entry is *this party's additive share* of the
    /// corresponding row commitment. Reconstruction is `row = row0 + row1 + row2`.
    ///
    /// Preconditions:
    /// - `DoryGlobals` must be initialized and `T = DoryGlobals::get_T()` must match
    ///   `self.masked_indices_c.len()`.
    /// - `bases.len() == DoryGlobals::get_num_columns()`.
    /// - `self.rand_ohv_e_field.len() == self.K` (use `from_indices_randohv`).
    ///
    /// Perf note:
    /// Although `E` is *actually* a one-hot vector (so the plaintext operation is an index
    /// selection), in this arithmetic-sharing representation each party only sees dense-looking
    /// field elements and cannot exploit sparsity without switching to an oblivious-access
    /// technique (e.g. MPC lookup / switching network / DPF/FSS-style point function sharing).
    #[tracing::instrument(skip_all, name = "one_hot::commit_rows", level = "trace")]
    pub fn commit_rows<G>(&self, bases: &[G::Affine]) -> eyre::Result<Vec<G>>
    where
        G: CurveGroup<ScalarField = F> + VariableBaseMSM + Send + Sync,
    {
        let _guard = tracing::trace_span!(
            "commit_rows",
            K = %self.K,
            T = %self.masked_indices_c.len()
        )
        .entered();

        let t = DoryGlobals::get_T();
        let row_len = DoryGlobals::get_num_columns();

        // eyre::ensure!(
        //     self.masked_indices_c.len() == t,
        //     "masked_indices_c length mismatch: got {}, want {t}",
        //     self.masked_indices_c.len()
        // );
        // eyre::ensure!(
        //     bases.len() == row_len,
        //     "bases length mismatch: got {}, want {row_len}",
        //     bases.len()
        // );
        eyre::ensure!(
            self.rand_ohv_e_field.len() == self.K,
            "RandOHV E_field must be initialized (use from_indices_randohv); got {}, want {}",
            self.rand_ohv_e_field.len(),
            self.K
        );

        // eyre::ensure!(
        //     (t as u128 * self.K as u128) % (row_len as u128) == 0,
        //     "invalid Dory sizing: T*K must be divisible by num_columns"
        // );
        let num_rows = (t * self.K) / row_len;
        let mut out = vec![G::zero(); num_rows];

        // Fast path: row alignment per k when `T % row_len == 0` (matches vanilla commit_rows).
        if t % row_len == 0 {
            let bases_group: Vec<G> = bases.iter().map(|b| b.into_group()).collect();

            debug_assert!(self.K.is_power_of_two(), "K must be power-of-two for FWHT");
            let g: Vec<F> = self.rand_ohv_e_field.iter().map(|s| s.a).collect();
            let mut g_hat = g;
            fwht_in_place(&mut g_hat);
            let inv_k = F::from(self.K as u64).inverse().expect("K invertible in field");

            let rows_per_k = t / row_len;
            let chunk_commitments: Vec<Vec<G>> = {
                let _guard = tracing::trace_span!("aligned_par").entered();

                (0..rows_per_k)
                    .into_par_iter()
                    .map(|chunk_index| -> eyre::Result<Vec<G>> {
                        let _guard = tracing::trace_span!(
                            "aligned_chunk",
                            chunk_index,
                            chunk_start = (chunk_index * row_len),
                            rows_per_k
                        )
                        .entered();

                        // Public aggregation of bases by masked index `c` within this chunk.
                        let mut s: Vec<G> = vec![G::zero(); self.K];
                        {
                            let _guard = tracing::trace_span!("aggregate_bases").entered();
                            let chunk_start = chunk_index * row_len;
                            for col in 0..row_len {
                                let idx_t = chunk_start + col;
                                if let Some(c) = self.masked_indices_c[idx_t] {
                                    s[c as usize] += bases_group[col];
                                }
                            }
                        }

                        // Compute all rows for this chunk via FWHT XOR-convolution:
                        // out[k] = Σ_c s[c] * g_hat_fwht[c XOR k].
                        {
                            let _guard = tracing::trace_span!("fwht").entered();
                            fwht_in_place(&mut s);
                            for (si, &gi) in s.iter_mut().zip(g_hat.iter()) {
                                *si = *si * gi;
                            }
                            fwht_in_place(&mut s);
                            for si in s.iter_mut() {
                                *si = *si * inv_k;
                            }
                        }

                        Ok(s)
                    })
                    .collect::<eyre::Result<Vec<_>>>()?
            };

            for (chunk_index, rows_for_chunk) in chunk_commitments.into_iter().enumerate() {
                for (k, share) in rows_for_chunk.into_iter().enumerate() {
                    out[k * rows_per_k + chunk_index] = share;
                }
            }
            return Ok(out);
        }

        // Fallback: general Dory layout when `T % row_len != 0`.
        //
        // This directly accumulates (row, col) contributions for each candidate `k` using the
        // shared bit `E_field[c XOR k]` and the public base at `col`.
        let _guard = tracing::trace_span!("fallback").entered();
        let active_t = self.masked_indices_c.iter().filter(|opt| opt.is_some()).count();

        // Note: parallelizing `msm_field_elements` here via rayon can overflow the default
        // rayon worker thread stack on some platforms/configs. Keep this sequential for now;
        // if we want parallelism, use a dedicated rayon pool with a larger stack size.
        {
            let _guard = tracing::trace_span!("row_msm").entered();
            for (row, out_row) in out.iter_mut().enumerate() {
                let _guard = tracing::trace_span!("row", row).entered();

                // Fill the per-row scalar vector for this row's MSM.
                let mut scalars = vec![F::zero(); row_len];
                for col in 0..row_len {
                    let global_index = row * row_len + col;
                    let k = global_index / t;
                    let t_idx = global_index % t;

                    let Some(c) = self.masked_indices_c[t_idx] else {
                        continue;
                    };
                    scalars[col] = self.rand_ohv_e_field[(c ^ (k as u8)) as usize].a;
                }

                *out_row = {
                    let _guard = tracing::info_span!("msm").entered();
                    G::msm_field_elements(bases, &scalars)?
                };
            }
        }

        tracing::info!(active_t, num_scalar_muls = (active_t * self.K));

        Ok(out)
    }

    /// Computes this party's additive share of the one-hot polynomial's contribution to the
    /// Dory v_vec (vector-matrix product), scaled by `coeff`.
    ///
    /// For each cycle `t` with `masked_indices_c[t] = Some(c)`, and for each `k` in `0..K`:
    ///   global_index = k * T + t
    ///   row = global_index / ncols
    ///   col = global_index % ncols
    ///   v_vec[col] += coeff * E_field[k XOR c].a * l_vec[row]
    ///
    /// Mirrors vanilla `OneHotPolynomial::vector_matrix_product` but operates on `.a` shares.
    #[tracing::instrument(skip_all, name = "one_hot::compute_v_vec_share", level = "trace")]
    pub fn compute_v_vec_share(&self, coeff: F, l_vec: &[F], v_vec: &mut [F]) {
        let t = self.masked_indices_c.len();
        let num_columns = DoryGlobals::get_num_columns();
        let row_len = num_columns;

        if t >= row_len {
            // Typical case: T >= row_len.
            //
            // The naive inner loop computes, for each active (col_index, row_offset, c):
            //   Σ_k E_field[k XOR c].a * l_vec[k * rows_per_k + row_offset]
            //
            // This is an XOR-convolution evaluated at index c. We precompute the
            // full convolution per row_offset via FWHT, turning O(T·K) into
            // O(rows_per_k · K log K + T).
            let rows_per_k = t / row_len;

            debug_assert!(self.K.is_power_of_two(), "K must be power-of-two for FWHT");

            // FWHT of the E_field .a shares (computed once, reused for all row_offsets).
            let g: Vec<F> = self.rand_ohv_e_field.iter().map(|s| s.a).collect();
            let mut g_hat = g;
            fwht_in_place(&mut g_hat);
            let inv_k = F::from(self.K as u64).inverse().expect("K invertible in field");

            // For each row_offset, compute conv[c] = Σ_k g[k XOR c] * l_vec[k * rows_per_k + row_offset]
            // via FWHT XOR-convolution: conv = IFWHT(FWHT(g) · FWHT(h)) / K
            let convolutions: Vec<Vec<F>> = (0..rows_per_k)
                .into_par_iter()
                .map(|row_offset| {
                    let mut h: Vec<F> = (0..self.K)
                        .map(|k| {
                            let row_index = k * rows_per_k + row_offset;
                            if row_index < l_vec.len() {
                                l_vec[row_index]
                            } else {
                                F::zero()
                            }
                        })
                        .collect();
                    fwht_in_place(&mut h);
                    for (hi, &gi) in h.iter_mut().zip(g_hat.iter()) {
                        *hi *= gi;
                    }
                    fwht_in_place(&mut h);
                    for hi in h.iter_mut() {
                        *hi *= inv_k;
                    }
                    h
                })
                .collect();

            // Accumulate into v_vec using precomputed convolution lookups.
            v_vec.par_iter_mut().enumerate().for_each(|(col_index, dest)| {
                let mut col_dot_product = F::zero();
                for (row_offset, t_idx) in (col_index..t).step_by(row_len).enumerate() {
                    if let Some(c) = self.masked_indices_c[t_idx] {
                        col_dot_product += convolutions[row_offset][c as usize];
                    }
                }
                *dest += coeff * col_dot_product;
            });
        } else {
            // T < row_len case
            let num_chunks = rayon::current_num_threads().next_power_of_two();
            let chunk_size = std::cmp::max(1, num_columns / num_chunks);

            v_vec.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_index, chunk)| {
                let min_col_index = chunk_index * chunk_size;
                let max_col_index = min_col_index + chunk_size;
                for (t_idx, opt_c) in self.masked_indices_c.iter().enumerate() {
                    if let Some(c) = opt_c {
                        for k in 0..self.K {
                            let global_index = k as u128 * t as u128 + t_idx as u128;
                            let col_index = (global_index % row_len as u128) as usize;
                            if col_index >= min_col_index && col_index < max_col_index {
                                let row_index = (global_index / row_len as u128) as usize;
                                let e_idx = k ^ (*c as usize);
                                if row_index < l_vec.len() {
                                    chunk[col_index % chunk_size] +=
                                        coeff * self.rand_ohv_e_field[e_idx].a * l_vec[row_index];
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    /// Evaluates the true MLE value:
    /// `Σ_j eq(j, r_cycle) * eq(k(j), r_address)`.
    #[tracing::instrument(skip_all, name = "one_hot::evaluate", level = "trace")]
    pub fn evaluate<C>(&self, r_address: &[C], r_cycle: &[C]) -> Rep3PrimeFieldShare<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        assert_eq!(r_address.len(), self.K.log_2());
        // assert_eq!(r_cycle.len(), self.masked_indices_c.len().log_2());

        let eq_addr: Vec<F> = EqPolynomial::<F>::evals(r_address);
        let eq_cycle: Vec<F> = EqPolynomial::<F>::evals(r_cycle);

        // Precompute shifted EQ table once:
        //   shifted[c] = eq_addr[r XOR c]  (in shares), where r is the RandOHV mask.
        // This turns per-cycle masked selection from O(K) to O(1).
        let shifted = shifted_table_from_rand_ohv(&eq_addr, &self.rand_ohv_e_field);

        self.masked_indices_c
            .par_iter()
            .enumerate()
            .fold(Rep3PrimeFieldShare::zero_share, |mut acc, (j, opt_c)| {
                let Some(c) = opt_c else { return acc };
                acc += shifted[*c as usize] * eq_cycle[j];
                acc
            })
            .reduce(Rep3PrimeFieldShare::zero_share, |mut a, b| {
                a += b;
                a
            })
    }
}

/// State related to the cycle variable terms in the opening reduction sumcheck.
#[derive(Clone, Debug, Allocative)]
pub struct EqCycleState<F: JoltField> {
    pub D: GruenSplitEqPolynomial<F>,
    pub w: Vec<F::Challenge>,
    pub num_variables_bound: usize,
}

impl<F: JoltField> EqCycleState<F> {
    pub fn new(r_cycle: &[F::Challenge]) -> Self {
        Self {
            D: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::HighToLow),
            w: r_cycle.to_vec(),
            num_variables_bound: 0,
        }
    }
}

/// Worker-side opening reduction prover for `Rep3OneHotPolynomial`, using the RandOHV representation.
#[derive(Clone, Debug)]
pub struct Rep3OneHotPolynomialProverOpening<F: JoltField> {
    pub polynomial: Rep3OneHotPolynomial<F>,
    pub eq_address_state: EqAddressState<F>,
    pub eq_cycle_state: EqCycleState<F>,
    pub log_T: usize,
    pub party_id: PartyID,
}

impl<F: JoltField> Rep3OneHotPolynomialProverOpening<F> {
    pub fn new(
        mut polynomial: Rep3OneHotPolynomial<F>,
        r_address: &[F::Challenge],
        r_cycle: &[F::Challenge],
        party_id: PartyID,
    ) -> Self {
        assert_eq!(polynomial.K, 1 << r_address.len());
        assert_eq!(polynomial.masked_indices_c.len(), 1 << r_cycle.len());

        let eq_cycle: Vec<F> = EqPolynomial::<F>::evals(r_cycle);
        polynomial.G = compute_g_from_masked_indices(&polynomial, &eq_cycle);

        Self {
            log_T: r_cycle.len(),
            polynomial,
            eq_address_state: EqAddressState::new(r_address),
            eq_cycle_state: EqCycleState::new(r_cycle),
            party_id,
        }
    }

    /// Returns the sumcheck univariate evaluations at {0, 2} for the current round.
    pub fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> [Rep3PrimeFieldShare<F>; 2] {
        let log_k = self.polynomial.K.log_2();

        if round < log_k {
            // Address-variable rounds.
            let num_unbound_address_variables = log_k - round;
            let B = &self.eq_address_state.B;
            let F_table = &self.eq_address_state.F;
            let G = &self.polynomial.G;

            let half = B.len() / 2;
            let (eval0, eval2) = (0..half)
                .into_par_iter()
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals_array::<2>(k_prime, BindingOrder::HighToLow);

                    let mut inner0 = Rep3PrimeFieldShare::zero_share();
                    let mut inner2 = Rep3PrimeFieldShare::zero_share();

                    for (k, &G_k) in G.iter().enumerate().skip(k_prime).step_by(half) {
                        let k_m = (k >> (num_unbound_address_variables - 1)) & 1;
                        let F_k = F_table[k >> num_unbound_address_variables];
                        let g_times_f = G_k * F_k;

                        if k_m == 0 {
                            inner0 += g_times_f;
                            inner2 -= g_times_f;
                        } else {
                            inner2 += g_times_f + g_times_f;
                        }
                    }

                    (inner0 * B_evals[0], inner2 * B_evals[1])
                })
                .reduce(
                    || (Rep3PrimeFieldShare::zero_share(), Rep3PrimeFieldShare::zero_share()),
                    |(mut a0, mut a2), (b0, b2)| {
                        a0 += b0;
                        a2 += b2;
                        (a0, a2)
                    },
                );

            [eval0, eval2]
        } else {
            // Cycle-variable rounds.
            let B = &self.eq_address_state.B;
            let eq_r_address_claim = B.final_sumcheck_claim();

            // Compute q(0) for Gruen/Dao-Thaler:
            // this is the inner sum term *without* the current linear eq factor.
            // Mirrors vanilla `OneHotPolynomialProverOpening` (but with secret-shared H).
            let d_gruen = &self.eq_cycle_state.D;
            let h_guard = self.polynomial.H.read().unwrap();
            let H = &*h_guard;

            let q0 = if d_gruen.E_in_current_len() == 1 {
                let e_out = d_gruen.E_out_current();
                (0..(d_gruen.len() / 2))
                    .into_par_iter()
                    .fold(Rep3PrimeFieldShare::zero_share, |mut acc, j| {
                        acc += H.get_bound_coeff(j) * e_out[j];
                        acc
                    })
                    .reduce(Rep3PrimeFieldShare::zero_share, |mut a, b| {
                        a += b;
                        a
                    })
            } else {
                let d_e_in = d_gruen.E_in_current();
                let d_e_out = d_gruen.E_out_current();
                let num_x_in = d_gruen.E_in_current_len();
                let num_x_out = d_gruen.E_out_current_len();
                let num_x_out_bits = num_x_out.log_2();

                (0..num_x_in)
                    .into_par_iter()
                    .fold(Rep3PrimeFieldShare::zero_share, |mut acc, x_in| {
                        let mut inner = Rep3PrimeFieldShare::zero_share();
                        for x_out in 0..num_x_out {
                            let j = (x_in << num_x_out_bits) | x_out;
                            inner += H.get_bound_coeff(j) * d_e_out[x_out];
                        }
                        acc += inner * d_e_in[x_in];
                        acc
                    })
                    .reduce(Rep3PrimeFieldShare::zero_share, |mut a, b| {
                        a += b;
                        a
                    })
            };

            // Normalize by eq_r_address_claim (public), matching vanilla.
            let previous_norm = previous_claim / eq_r_address_claim;
            let previous_norm_share = Rep3PrimeFieldShare::promote_from_trivial(&previous_norm, self.party_id);

            // Compute eq evals for the current round from the cycle-state scalar and the next w entry.
            let current_scalar = self.eq_cycle_state.D.get_current_scalar();
            let wi: F = self.eq_cycle_state.w[self.eq_cycle_state.num_variables_bound].into();
            let eq_eval_1 = current_scalar * wi;
            let eq_eval_0 = current_scalar - eq_eval_1;
            let eq_m = eq_eval_1 - eq_eval_0;
            let eq_eval_2 = eq_eval_1 + eq_m;

            let eval0 = q0 * eq_eval_0;
            let eval1 = previous_norm_share - eval0;
            let inv_eq1 = eq_eval_1.inverse().unwrap();
            let q1 = eval1 * inv_eq1;
            let q2 = (q1 + q1) - q0;
            let eval2 = q2 * eq_eval_2;

            [eval0 * eq_r_address_claim, eval2 * eq_r_address_claim]
        }
    }

    /// Like `compute_prover_message`, but with an `AdditiveShare<F>` previous_claim.
    /// Used in the batched opening proof reduction where per-instance claims are additive shares.
    pub fn compute_prover_message_shared(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
    ) -> [AdditiveShare<F>; 2] {
        let log_k = self.polynomial.K.log_2();

        if round < log_k {
            // Address-variable rounds: previous_claim is not used.
            // Compute the same as compute_prover_message but convert to additive.
            let [eval0, eval2] = self.compute_prover_message(round, F::zero());
            [eval0.into_additive(), eval2.into_additive()]
        } else {
            // Cycle-variable rounds: adapt Gruen computation for AdditiveShare previous_claim.
            let B = &self.eq_address_state.B;
            let eq_r_address_claim = B.final_sumcheck_claim();

            let d_gruen = &self.eq_cycle_state.D;
            let h_guard = self.polynomial.H.read().unwrap();
            let H = &*h_guard;

            let q0 = if d_gruen.E_in_current_len() == 1 {
                let e_out = d_gruen.E_out_current();
                let loop_bound = d_gruen.len() / 2;
                debug_assert!(loop_bound <= H.len());
                (0..loop_bound)
                    .into_par_iter()
                    .fold(Rep3PrimeFieldShare::zero_share, |mut acc, j| {
                        acc += H.get_bound_coeff(j) * e_out[j];
                        acc
                    })
                    .reduce(Rep3PrimeFieldShare::zero_share, |mut a, b| {
                        a += b;
                        a
                    })
            } else {
                let d_e_in = d_gruen.E_in_current();
                let d_e_out = d_gruen.E_out_current();
                let num_x_in = d_gruen.E_in_current_len();
                let num_x_out = d_gruen.E_out_current_len();
                let num_x_out_bits = num_x_out.log_2();
                debug_assert!(num_x_in == 0 || ((num_x_in - 1) << num_x_out_bits | (num_x_out - 1)) < H.len());

                (0..num_x_in)
                    .into_par_iter()
                    .fold(Rep3PrimeFieldShare::zero_share, |mut acc, x_in| {
                        let mut inner = Rep3PrimeFieldShare::zero_share();
                        for x_out in 0..num_x_out {
                            let j = (x_in << num_x_out_bits) | x_out;
                            inner += H.get_bound_coeff(j) * d_e_out[x_out];
                        }
                        acc += inner * d_e_in[x_in];
                        acc
                    })
                    .reduce(Rep3PrimeFieldShare::zero_share, |mut a, b| {
                        a += b;
                        a
                    })
            };

            // previous_claim is AdditiveShare<F>; eq_r_address_claim is public F.
            let inv_eq_r = eq_r_address_claim.inverse().unwrap();
            let previous_norm = previous_claim * inv_eq_r;

            let current_scalar = self.eq_cycle_state.D.get_current_scalar();
            let wi: F = self.eq_cycle_state.w[self.eq_cycle_state.num_variables_bound].into();
            let eq_eval_1 = current_scalar * wi;
            let eq_eval_0 = current_scalar - eq_eval_1;
            let eq_m = eq_eval_1 - eq_eval_0;
            let eq_eval_2 = eq_eval_1 + eq_m;

            let eval0_rep3 = q0 * eq_eval_0;
            let eval0 = eval0_rep3.into_additive();
            let eval1 = previous_norm - eval0;
            let inv_eq1 = eq_eval_1.inverse().unwrap();
            let q1 = eval1 * inv_eq1;
            let q2 = q1 + q1 - q0.into_additive();
            let eval2 = q2 * eq_eval_2;

            [eval0 * eq_r_address_claim, eval2 * eq_r_address_claim]
        }
    }

    pub fn bind(&mut self, r: F::Challenge, round: usize) {
        let log_k = self.polynomial.K.log_2();

        if round < log_k {
            self.eq_address_state.B.bind_parallel(r, BindingOrder::HighToLow);
            self.eq_address_state.F.update(r);
            self.eq_address_state.num_variables_bound += 1;

            // Transition after final address bit: build H(j) = eq(k(j), u).
            if round == log_k - 1 {
                let eq_u = self.eq_address_state.F.clone_values();
                assert_eq!(eq_u.len(), self.polynomial.K);

                let table_shifted = shifted_table_from_rand_ohv(&eq_u, &self.polynomial.rand_ohv_e_field);
                *self.polynomial.H.write().unwrap() =
                    Rep3RaPolynomial::new(self.polynomial.masked_indices_c.clone(), table_shifted);
                self.polynomial.G.clear();
            }
        } else {
            self.eq_cycle_state.D.bind(r);
            self.eq_cycle_state.num_variables_bound += 1;
            self.polynomial.H.write().unwrap().bind_parallel(r, BindingOrder::HighToLow);
        }
    }

    pub fn final_sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        self.polynomial.H.read().unwrap().final_sumcheck_claim()
    }
}

pub(crate) fn compute_g_from_masked_indices<F: JoltField>(
    polynomial: &Rep3OneHotPolynomial<F>,
    eq_cycle: &[F],
) -> Vec<Rep3PrimeFieldShare<F>> {
    assert_eq!(eq_cycle.len(), polynomial.masked_indices_c.len());
    assert_eq!(polynomial.rand_ohv_e_field.len(), polynomial.K);

    // Histogram in masked index space.
    let k_len = polynomial.K;
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(polynomial.masked_indices_c.len()).max(1);
    let chunk_size = (polynomial.masked_indices_c.len() / num_chunks).max(1);

    let g_c = polynomial
        .masked_indices_c
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut local = vec![F::zero(); k_len];
            let chunk_start = chunk_index * chunk_size;
            for (offset, opt_c) in chunk.iter().enumerate() {
                let j = chunk_start + offset;
                if let Some(c) = opt_c {
                    local[*c as usize] += eq_cycle[j];
                }
            }
            local
        })
        .reduce(
            || vec![F::zero(); k_len],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += bi;
                }
                a
            },
        );

    // Convert to k-space: G[k] = Σ_c G_c[c] * E_field[c XOR k].
    (0..polynomial.K)
        .into_par_iter()
        .map(|k| {
            let mut acc = Rep3PrimeFieldShare::zero_share();
            for c in 0..polynomial.K {
                let idx = (c as u8) ^ (k as u8);
                acc += polynomial.rand_ohv_e_field[idx as usize] * g_c[c];
            }
            acc
        })
        .collect()
}

pub fn compute_g_from_masked_indices_many<F: JoltField, const D: usize>(
    polynomials: &[Rep3OneHotPolynomial<F>; D],
    eq_cycle: &[F],
) -> [Arc<Vec<Rep3PrimeFieldShare<F>>>; D] {
    debug_assert_eq!(eq_cycle.len(), polynomials[0].masked_indices_c.len());
    debug_assert_eq!(polynomials[0].rand_ohv_e_field.len(), polynomials[0].K);

    let t = eq_cycle.len();
    let k_len = polynomials[0].K;

    for i in 1..D {
        debug_assert_eq!(polynomials[i].K, k_len, "K mismatch across chunks");
        debug_assert_eq!(polynomials[i].masked_indices_c.len(), t, "masked indices length mismatch");
        debug_assert_eq!(polynomials[i].rand_ohv_e_field.len(), k_len, "E_field length mismatch");
    }

    // Histogram in masked index space for all D chunks in one trace pass.
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(t).max(1);
    let chunk_size = (t / num_chunks).max(1);

    let g_c: [Vec<F>; D] = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_index| {
            let start = chunk_index * chunk_size;
            let end = ((chunk_index + 1) * chunk_size).min(t);

            let mut local: [Vec<F>; D] = std::array::from_fn(|_| vec![F::zero(); k_len]);
            for j in start..end {
                let eq = eq_cycle[j];
                for i in 0..D {
                    if let Some(c) = polynomials[i].masked_indices_c[j] {
                        local[i][c as usize] += eq;
                    }
                }
            }
            local
        })
        .reduce(
            || std::array::from_fn(|_| vec![F::zero(); k_len]),
            |mut a, b| {
                for i in 0..D {
                    for (ai, bi) in a[i].iter_mut().zip(b[i].iter()) {
                        *ai += *bi;
                    }
                }
                a
            },
        );

    // Convert to k-space: G[k] = Σ_c G_c[c] * E_field[c XOR k].
    std::array::from_fn(|i| {
        let g_c_i = &g_c[i];
        let e_field = &polynomials[i].rand_ohv_e_field;
        let g_i: Vec<Rep3PrimeFieldShare<F>> = (0..k_len)
            .into_par_iter()
            .map(|k| {
                let mut acc = Rep3PrimeFieldShare::zero_share();
                for c in 0..k_len {
                    let idx = (c as u8) ^ (k as u8);
                    acc += e_field[idx as usize] * g_c_i[c];
                }
                acc
            })
            .collect();
        Arc::new(g_i)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;
    use ark_std::UniformRand;
    use jolt_core::ark_bn254::Fr;
    use jolt_core::ark_bn254::{G1Affine, G1Projective};
    use jolt_core::poly::commitment::dory::DoryGlobals;
    use jolt_core::poly::dense_mlpoly::DensePolynomial;
    use jolt_core::poly::one_hot_polynomial as vanilla;
    use jolt_core::poly::unipoly::UniPoly;
    use mpc_core::protocols::rep3::combine_field_element;
    use num_traits::{One, Zero};
    use rand::RngCore;
    use std::path::Path;
    use std::sync::RwLock;

    fn share_field_element_rep3<F: JoltField, R: rand::Rng + rand::CryptoRng>(val: F, rng: &mut R) -> [Rep3PrimeFieldShare<F>; 3] {
        mpc_core::protocols::rep3::share_field_element(val, rng)
    }

    fn build_matching_polys<F: JoltField, R: rand::RngCore + rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        k: usize,
        t: usize,
    ) -> (Vec<Option<u8>>, vanilla::OneHotPolynomial<F>, [Rep3OneHotPolynomial<F>; 3]) {
        // Plaintext nonzero indices (allow some None entries).
        let mut nonzero_indices_plain: Vec<Option<u8>> = (0..t)
            .map(|_| if (rng.next_u32() & 3) == 0 { None } else { Some((rng.next_u32() as u8) & 0xff) })
            .collect();
        if nonzero_indices_plain.iter().all(|x| x.is_none()) {
            nonzero_indices_plain[0] = Some(7);
        }

        let vanilla_poly = vanilla::OneHotPolynomial::<F>::from_indices(nonzero_indices_plain.clone(), k);

        // Choose a fixed RandOHV mask index r (plaintext) and build the public masked indices
        // c[j] = k(j) XOR r.
        let r_mask: u8 = (rng.next_u32() as u8) & 0xff;
        let masked_indices_c: Arc<Vec<Option<u8>>> =
            Arc::new(nonzero_indices_plain.iter().map(|opt| opt.map(|kj| kj ^ r_mask)).collect());

        // Replicated arithmetic shares of E_field = e(r_mask).
        let mut e_field_party: [Vec<Rep3PrimeFieldShare<F>>; 3] = std::array::from_fn(|_| Vec::with_capacity(k));
        for i in 0..k {
            let bit = if i as u8 == r_mask { F::one() } else { F::zero() };
            let shares = share_field_element_rep3(bit, rng);
            for pid in 0..3 {
                e_field_party[pid].push(shares[pid]);
            }
        }
        let e_field_party: [Arc<Vec<Rep3PrimeFieldShare<F>>>; 3] =
            std::array::from_fn(|pid| Arc::new(e_field_party[pid].clone()));

        // Secret shares of the indices.
        //
        // NOTE: The RandOHV proving path in these tests uses `masked_indices_c` + `E_field` and
        // does not read `nonzero_indices`. For real usage via `from_indices_randohv`, the indices
        // must be provided as binary/XOR `Rep3RingShare<u8>`.
        let mut nonzero_indices_shares: [Vec<Option<Rep3RingShare<u8>>>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(t));
        for opt in nonzero_indices_plain.iter().copied() {
            match opt {
                None => {
                    for pid in 0..3 {
                        nonzero_indices_shares[pid].push(None);
                    }
                }
                Some(kj) => {
                    // Indices are binary/XOR shared for the RandOHV construction.
                    let shares = mpc_core::protocols::rep3_ring::share_ring_element_binary(mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement(kj), rng);
                    for pid in 0..3 {
                        nonzero_indices_shares[pid].push(Some(shares[pid]));
                    }
                }
            }
        }

        let rep3_polys: [Rep3OneHotPolynomial<F>; 3] = std::array::from_fn(|pid| Rep3OneHotPolynomial {
            K: k,
            masked_indices_c: masked_indices_c.clone(),
            rand_ohv_e_field: e_field_party[pid].clone(),
            num_variables_bound: 0,
            G: vec![],
            H: Arc::new(RwLock::new(Rep3RaPolynomial::None)),
            r_share: Rep3RingShare::default(),
        });

        (nonzero_indices_plain, vanilla_poly, rep3_polys)
    }

    #[test]
    fn one_hot_masked_select() {
        type F = Fr;
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        let K = 256usize;
        let r: u8 = (rng.next_u32() as u8) & 0xff;
        let c: u8 = (rng.next_u32() as u8) & 0xff;

        let table: Vec<F> = (0..K).map(|_| F::random(&mut rng)).collect();

        let mut e_field_party: [Vec<Rep3PrimeFieldShare<F>>; 3] = std::array::from_fn(|_| Vec::with_capacity(K));
        for i in 0..K {
            let bit = if i as u8 == r { F::one() } else { F::zero() };
            let shares = share_field_element_rep3(bit, &mut rng);
            for pid in 0..3 {
                e_field_party[pid].push(shares[pid]);
            }
        }

        let masked_indices_c = Arc::new(vec![Some(c)]);

        let polys: [Rep3OneHotPolynomial<F>; 3] = std::array::from_fn(|pid| Rep3OneHotPolynomial {
            K,
            masked_indices_c: masked_indices_c.clone(),
            rand_ohv_e_field: Arc::new(e_field_party[pid].clone()),
            num_variables_bound: 0,
            G: vec![],
            H: Arc::new(RwLock::new(Rep3RaPolynomial::None)),
            r_share: Rep3RingShare::default(),
        });

        let shares: [Rep3PrimeFieldShare<F>; 3] = std::array::from_fn(|pid| {
            let shifted = shifted_table_from_rand_ohv(&table, &polys[pid].rand_ohv_e_field);
            shifted[c as usize]
        });

        let got = combine_field_element(shares[0], shares[1], shares[2]);
        let want = table[(r ^ c) as usize];
        assert_eq!(got, want);
    }

    #[test]
    fn one_hot_eval_correct() {
        type F = Fr;
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        let K = 256usize;
        let log_k = 8usize;
        let log_t = 5usize;
        let T = 1usize << log_t;

        let r_mask: u8 = (rng.next_u32() as u8) & 0xff;

        // Random indices with some None entries.
        let mut k_plain: Vec<Option<u8>> = (0..T)
            .map(|_| if (rng.next_u32() & 3) == 0 { None } else { Some((rng.next_u32() as u8) & 0xff) })
            .collect();
        // Ensure at least one active entry.
        if k_plain.iter().all(|x| x.is_none()) {
            k_plain[0] = Some(7);
        }

        let masked_indices_c: Vec<Option<u8>> = k_plain.iter().map(|opt| opt.map(|k| k ^ r_mask)).collect();

        let mut e_field_party: [Vec<Rep3PrimeFieldShare<F>>; 3] = std::array::from_fn(|_| Vec::with_capacity(K));
        for i in 0..K {
            let bit = if i as u8 == r_mask { F::one() } else { F::zero() };
            let shares = share_field_element_rep3(bit, &mut rng);
            for pid in 0..3 {
                e_field_party[pid].push(shares[pid]);
            }
        }

        let polys: [Rep3OneHotPolynomial<F>; 3] = std::array::from_fn(|pid| Rep3OneHotPolynomial {
            K,
            masked_indices_c: Arc::new(masked_indices_c.clone()),
            rand_ohv_e_field: Arc::new(e_field_party[pid].clone()),
            num_variables_bound: 0,
            G: vec![],
            H: Arc::new(RwLock::new(Rep3RaPolynomial::None)),
            r_share: Rep3RingShare::default(),
        });

        // Random challenge points.
        let r_address: Vec<F> = (0..log_k).map(|_| F::random(&mut rng)).collect();
        let r_cycle: Vec<F> = (0..log_t).map(|_| F::random(&mut rng)).collect();

        let shares: [Rep3PrimeFieldShare<F>; 3] = std::array::from_fn(|pid| polys[pid].evaluate(&r_address, &r_cycle));
        let got = combine_field_element(shares[0], shares[1], shares[2]);

        // Plaintext reference.
        let eq_addr = EqPolynomial::<F>::evals(&r_address);
        let eq_cycle = EqPolynomial::<F>::evals(&r_cycle);
        let mut want = F::zero();
        for (j, opt_k) in k_plain.iter().enumerate() {
            if let Some(k) = opt_k {
                want += eq_cycle[j] * eq_addr[*k as usize];
            }
        }

        assert_eq!(got, want);
    }

    #[test]
    fn one_hot_eval_open_sumcheck() {
        type F = Fr;
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        let log_k = 8usize;
        let log_t = 9usize;
        let k = 1usize << log_k;
        let t = 1usize << log_t;

        // Vanilla OneHotPolynomial depends on DoryGlobals for sizing assertions and its Dory-backed evaluate().
        crate::poly::commitment::dory::test_support::init_dory_globals(k, t);
        let (nonzero_indices_plain, vanilla_poly, polys) = build_matching_polys::<F, _>(&mut rng, k, t);

        // Random opening points.
        let r_address: Vec<<F as jolt_core::field::JoltField>::Challenge> =
            std::iter::repeat_with(|| <F as jolt_core::field::JoltField>::Challenge::random(&mut rng))
                .take(log_k)
                .collect();
        let r_cycle: Vec<<F as jolt_core::field::JoltField>::Challenge> =
            std::iter::repeat_with(|| <F as jolt_core::field::JoltField>::Challenge::random(&mut rng))
                .take(log_t)
                .collect();
        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();

        // Evaluate: Rep3 (reconstructed) matches vanilla OneHotPolynomial::evaluate (Dory-backed).
        let rep3_eval_shares: [Rep3PrimeFieldShare<F>; 3] =
            std::array::from_fn(|pid| polys[pid].evaluate(&r_address, &r_cycle));
        let rep3_eval = combine_field_element(rep3_eval_shares[0], rep3_eval_shares[1], rep3_eval_shares[2]);
        let vanilla_eval = vanilla_poly.evaluate(&r_concat);
        assert_eq!(rep3_eval, vanilla_eval, "evaluate mismatch");

        // Also sanity-check: vanilla evaluate matches dense-dot-eq (the "true MLE" evaluation).
        let eq_all = EqPolynomial::<F>::evals(&r_concat);
        let dense_eval: F = nonzero_indices_plain
            .iter()
            .enumerate()
            .filter_map(|(j, opt_k)| opt_k.map(|kk| (j, kk)))
            .map(|(j, kk)| eq_all[(kk as usize) * t + j])
            .sum();
        assert_eq!(vanilla_eval, dense_eval, "vanilla Dory evaluate != true MLE");

        // Sumcheck message equivalence (compute_prover_message + bind), round-by-round.
        let eq_address_state = vanilla::EqAddressState::<F>::new(&r_address);
        let mut eq_cycle_state = vanilla::EqCycleState::<F>::new(&r_cycle);
        eq_cycle_state.merge_D();
        let mut vanilla_opening = vanilla::OneHotPolynomialProverOpening::<F>::new(
            Arc::new(RwLock::new(eq_address_state)),
            Arc::new(RwLock::new(eq_cycle_state)),
        );
        vanilla_opening.initialize(vanilla_poly.clone());

        let mut rep3_openings: [Rep3OneHotPolynomialProverOpening<F>; 3] = std::array::from_fn(|pid| {
            Rep3OneHotPolynomialProverOpening::new(
                polys[pid].clone(),
                &r_address,
                &r_cycle,
                match pid {
                    0 => PartyID::ID0,
                    1 => PartyID::ID1,
                    _ => PartyID::ID2,
                },
            )
        });

        // Dense reference for the product polynomial `one_hot * eq(r_concat, ·)`.
        let mut dense_coeffs = vec![F::zero(); k * t];
        for (j, opt_k) in nonzero_indices_plain.iter().enumerate() {
            if let Some(kk) = opt_k {
                dense_coeffs[(*kk as usize) * t + j] = F::one();
            }
        }
        let mut dense_poly = DensePolynomial::<F>::new(dense_coeffs);
        let mut eq_poly = DensePolynomial::<F>::new(EqPolynomial::<F>::evals(&r_concat));

        let input_claim: F =
            dense_poly.Z.iter().zip(eq_poly.Z.iter()).take(dense_poly.len()).map(|(a, b)| *a * *b).sum();
        let mut previous_claim = input_claim;

        for round in 0..(log_k + log_t) {
            let vanilla_msg = vanilla_opening.compute_prover_message(round, previous_claim);
            assert_eq!(vanilla_msg.len(), 2);

            let rep3_msgs: [[Rep3PrimeFieldShare<F>; 2]; 3] =
                std::array::from_fn(|pid| rep3_openings[pid].compute_prover_message(round, previous_claim));
            let rep3_msg0 = combine_field_element(rep3_msgs[0][0], rep3_msgs[1][0], rep3_msgs[2][0]);
            let rep3_msg2 = combine_field_element(rep3_msgs[0][1], rep3_msgs[1][1], rep3_msgs[2][1]);

            // Dense reference sumcheck message at {0,2}.
            let mut expected0 = F::zero();
            let mut expected2 = F::zero();
            let half = dense_poly.len() / 2;
            for i in 0..half {
                expected0 += dense_poly.Z[i] * eq_poly.Z[i];

                let poly_bound_point = dense_poly.Z[i + half] + dense_poly.Z[i + half] - dense_poly.Z[i];
                let eq_bound_point = eq_poly.Z[i + half] + eq_poly.Z[i + half] - eq_poly.Z[i];
                expected2 += poly_bound_point * eq_bound_point;
            }

            assert_eq!(vanilla_msg[0], expected0, "round {round} vanilla eval(0) mismatch");
            assert_eq!(vanilla_msg[1], expected2, "round {round} vanilla eval(2) mismatch");
            assert_eq!(rep3_msg0, expected0, "round {round} rep3 eval(0) mismatch");
            assert_eq!(rep3_msg2, expected2, "round {round} rep3 eval(2) mismatch");

            // Bind next challenge and update the running claim.
            let r = <F as jolt_core::field::JoltField>::Challenge::random(&mut rng);
            let eval_at_1 = previous_claim - expected0;
            let univariate_poly = UniPoly::from_evals(&[expected0, eval_at_1, expected2]);
            previous_claim = univariate_poly.evaluate(&r);

            vanilla_opening.bind(r, round);
            for pid in 0..3 {
                rep3_openings[pid].bind(r, round);
            }
            dense_poly.bind_parallel(r, BindingOrder::HighToLow);
            eq_poly.bind_parallel(r, BindingOrder::HighToLow);
        }

        assert_eq!(vanilla_opening.final_sumcheck_claim(), dense_poly.Z[0], "vanilla final sumcheck claim");
        let rep3_final = combine_field_element(
            rep3_openings[0].final_sumcheck_claim(),
            rep3_openings[1].final_sumcheck_claim(),
            rep3_openings[2].final_sumcheck_claim(),
        );
        assert_eq!(rep3_final, dense_poly.Z[0], "rep3 final sumcheck claim");
    }

    #[test]
    fn one_hot_commit_rows_correct() {
        let _tracing_guard = crate::utils::tracing::init_tracing(
            "rep3_commit_rows_reconstructs_to_vanilla.json",
            Path::new("/tmp/co-jolt2-traces"),
        );

        type F = Fr;
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        let log_k = 8usize;
        let log_t = 9usize;
        let k = 1usize << log_k;
        let t = 1usize << log_t;

        crate::poly::commitment::dory::test_support::init_dory_globals(k, t);
        let row_len = DoryGlobals::get_num_columns();

        let (_nonzero_indices_plain, vanilla_poly, polys) = build_matching_polys::<F, _>(&mut rng, k, t);

        let bases: Vec<G1Affine> = (0..row_len).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();

        let vanilla_rows = vanilla_poly.commit_rows::<G1Projective>(&bases);
        let rep3_rows: [Vec<G1Projective>; 3] =
            std::array::from_fn(|pid| polys[pid].commit_rows::<G1Projective>(&bases).expect("rep3 commit_rows"));

        assert_eq!(rep3_rows[0].len(), vanilla_rows.len());
        assert_eq!(rep3_rows[1].len(), vanilla_rows.len());
        assert_eq!(rep3_rows[2].len(), vanilla_rows.len());

        for i in 0..vanilla_rows.len() {
            let reconstructed = rep3_rows[0][i] + rep3_rows[1][i] + rep3_rows[2][i];
            assert_eq!(reconstructed, vanilla_rows[i].0, "row {i} mismatch");
        }
    }
}
