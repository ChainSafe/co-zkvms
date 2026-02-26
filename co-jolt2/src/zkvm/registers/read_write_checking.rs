use jolt2_common::constants::REGISTER_COUNT;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding,
};
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN};
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::utils::types::Rep3Value;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};
use crate::zkvm::instruction_lookups::booleanity::{extend_degree_3_evals, gruen_evals_deg_3};

const K: usize = REGISTER_COUNT as usize;
const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Per-chunk data buffers for phase1 of register ReadWriteChecking.
/// Mirrors vanilla `DataBuffers` but with `Rep3PrimeFieldShare<F>` for val entries
/// (val tracks accumulated increments which are SHARED).
struct DataBuffers<F: JoltField> {
    /// Val(k, j', 0, ..., 0) — accumulated checkpoint + increments (SHARED)
    val_j_0: [Rep3PrimeFieldShare<F>; K],
    /// val_j_r[b][k] — partially-bound Val values (SHARED)
    val_j_r: [[Rep3PrimeFieldShare<F>; K]; 2],
    /// rs1_ra[b][k] — read-address histogram for rs1 (PUBLIC)
    rs1_ra: [[F; K]; 2],
    /// rs2_ra[b][k] — read-address histogram for rs2 (PUBLIC)
    rs2_ra: [[F; K]; 2],
    /// rd_wa[b][k] — write-address histogram for rd (PUBLIC)
    rd_wa: [[F; K]; 2],
    dirty_indices: Vec<bool>,
}

struct ReadWriteCheckingProverState<F: JoltField> {
    addresses: Vec<(u8, u8, u8)>,
    chunk_size: usize,
    /// Checkpoints: Val(k) at chunk boundaries (SHARED)
    val_checkpoints: Vec<Rep3PrimeFieldShare<F>>,
    data_buffers: Vec<DataBuffers<F>>,
    /// I[chunk][row_in_chunk] = (j_prime, k, inc_lt, inc)
    /// inc_lt and inc are SHARED
    I: Vec<Vec<(usize, u8, Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>)>>,
    /// EQ table (PUBLIC)
    A: Vec<F>,
    gruens_eq_r_prime: GruenSplitEqPolynomial<F>,
    /// Committed RdInc polynomial (SHARED)
    inc_cycle: Rep3DensePolynomial<F>,
    // Materialized after phase1
    eq_r_prime: Option<MultilinearPolynomial<F>>,
    rs1_ra: Option<MultilinearPolynomial<F>>,
    rs2_ra: Option<MultilinearPolynomial<F>>,
    rd_wa: Option<MultilinearPolynomial<F>>,
    val: Option<Rep3DensePolynomial<F>>,
}

impl<F: JoltField> ReadWriteCheckingProverState<F> {
    fn initialize<PCS: CommitmentScheme<Field = F>>(sm: &mut StateManagerWorker<'_, F, PCS>,
        r_prime: &[F::Challenge],
    ) -> Self {
        let cycle_witness = &sm.prover_state.cycle_witness;
        let T = cycle_witness.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        // Compute deltas per chunk from the inc_cycle polynomial (SHARED).
        // inc_cycle[j] = rd_write_post[j] - rd_write_pre[j] in field (SHARED).
        let inc_cycle = sm
            .prover_state
            .cycle_witness
            .rd_inc
            .clone()
            .expect("rd_inc not populated on cycle_witness");

        let rd_addr = &cycle_witness.rd_addr;

        let deltas: Vec<[Rep3PrimeFieldShare<F>; K]> = rd_addr[..T - chunk_size]
            .par_chunks_exact(chunk_size)
            .enumerate()
            .map(|(chunk_index, addr_chunk)| {
                let mut delta = [Rep3PrimeFieldShare::<F>::zero_share(); K];
                let base = chunk_index * chunk_size;
                for (i, &k) in addr_chunk.iter().enumerate() {
                    delta[k as usize] += inc_cycle.get_bound_coeff(base + i);
                }
                delta
            })
            .collect();

        // Compute checkpoints: Val(k) at chunk boundaries (SHARED)
        let mut checkpoints: Vec<[Rep3PrimeFieldShare<F>; K]> = Vec::with_capacity(num_chunks);
        checkpoints.push([Rep3PrimeFieldShare::<F>::zero_share(); K]);

        for (chunk_index, delta) in deltas.into_iter().enumerate() {
            let next: [Rep3PrimeFieldShare<F>; K] =
                std::array::from_fn(|k| checkpoints[chunk_index][k] + delta[k]);
            checkpoints.push(next);
        }

        let mut val_checkpoints: Vec<Rep3PrimeFieldShare<F>> =
            vec![Rep3PrimeFieldShare::zero_share(); K * num_chunks];
        val_checkpoints
            .par_chunks_mut(K)
            .zip(checkpoints.into_par_iter())
            .for_each(|(dest, src)| dest.copy_from_slice(&src));

        // EQ table (PUBLIC)
        let mut A: Vec<F> = unsafe_allocate_zero_vec(chunk_size);
        A[0] = F::one();

        // Build I data structure (inc values are SHARED)
        let rs1_addr = &cycle_witness.rs1_addr;
        let rs2_addr = &cycle_witness.rs2_addr;
        let I: Vec<Vec<(usize, u8, Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>)>> = rd_addr
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, addr_chunk)| {
                let mut j = chunk_index * chunk_size;
                addr_chunk
                    .iter()
                    .map(|&k| {
                        let inc_val = inc_cycle.get_bound_coeff(j);
                        let entry = (j, k, Rep3PrimeFieldShare::zero_share(), inc_val);
                        j += 1;
                        entry
                    })
                    .collect()
            })
            .collect();

        let gruens_eq_r_prime = GruenSplitEqPolynomial::<F>::new(r_prime, BindingOrder::LowToHigh);

        let addresses: Vec<(u8, u8, u8)> = (0..T)
            .into_par_iter()
            .map(|j| (rs1_addr[j], rs2_addr[j], rd_addr[j]))
            .collect();

        let data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: [Rep3PrimeFieldShare::zero_share(); K],
                val_j_r: [
                    [Rep3PrimeFieldShare::zero_share(); K],
                    [Rep3PrimeFieldShare::zero_share(); K],
                ],
                rs1_ra: [[F::zero(); K]; 2],
                rs2_ra: [[F::zero(); K]; 2],
                rd_wa: [[F::zero(); K]; 2],
                dirty_indices: vec![false; K],
            })
            .collect();

        ReadWriteCheckingProverState {
            addresses,
            chunk_size,
            val_checkpoints,
            data_buffers,
            I,
            A,
            gruens_eq_r_prime,
            inc_cycle,
            eq_r_prime: None,
            rs1_ra: None,
            rs2_ra: None,
            rd_wa: None,
            val: None,
        }
    }
}

pub struct Rep3RegistersReadWriteCheckingWorker<F: JoltField> {
    party_id: PartyID,
    T: usize,
    gamma: F,
    gamma_sqr: F,
    sumcheck_switch_index: usize,
    prover_state: ReadWriteCheckingProverState<F>,
    input_claim: F,
}

impl<F: JoltField> Rep3RegistersReadWriteCheckingWorker<F> {
    pub fn new<PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerWorker<'_, F, PCS>,
        gamma: F,
        input_claim: F,
    ) -> Self {
        let party_id = sm.party_id;
        let T = sm.prover_state.cycle_witness.len();

        let (r_cycle, _) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);

        let gamma_sqr = gamma.square();

        let prover_state = ReadWriteCheckingProverState::initialize(sm, &r_cycle.r);

        Self {
            party_id,
            T,
            gamma,
            gamma_sqr,
            sumcheck_switch_index: sm.twist_sumcheck_switch_index,
            prover_state,
            input_claim,
        }
    }

    /// Phase1: Gruen-optimized sumcheck over cycle variables (low-to-high binding).
    ///
    /// Inner sum formula for registers:
    ///   rd_wa(PUB) * (inc_cycle(SHARED) + val(SHARED))
    ///   + gamma * rs1_ra(PUB) * val(SHARED)
    ///   + gamma^2 * rs2_ra(PUB) * val(SHARED)
    /// All products are PUBLIC * SHARED = Rep3PrimeFieldShare.
    fn phase1_compute_prover_message(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
    ) -> Vec<AdditiveShare<F>> {
        let party_id = self.party_id;
        let gamma = self.gamma;
        let gamma_sqr = self.gamma_sqr;
        let ReadWriteCheckingProverState {
            addresses,
            I,
            data_buffers,
            A,
            val_checkpoints,
            inc_cycle,
            gruens_eq_r_prime,
            ..
        } = &mut self.prover_state;

        // Compute quadratic coefficients: [q0, q_inf] as Rep3Value
        let quadratic_coeffs: [Rep3Value<F>; 2] = if gruens_eq_r_prime.E_in_current_len() == 1 {
            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [Rep3Value::zero_share(); 2];

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        rs1_ra,
                        rs2_ra,
                        rd_wa,
                        dirty_indices,
                    } = buffers;

                    val_j_0.copy_from_slice(checkpoint);

                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0;

                            // Build public ra/wa histograms for the pair of rows
                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                let (k_rs1, k_rs2, k_rd) = addresses[j];
                                dirty_indices[k_rs1 as usize] = true;
                                rs1_ra[0][k_rs1 as usize] += A[j_bound];
                                dirty_indices[k_rs2 as usize] = true;
                                rs2_ra[0][k_rs2 as usize] += A[j_bound];
                                dirty_indices[k_rd as usize] = true;
                                rd_wa[0][k_rd as usize] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                let (k_rs1, k_rs2, k_rd) = addresses[j];
                                dirty_indices[k_rs1 as usize] = true;
                                rs1_ra[1][k_rs1 as usize] += A[j_bound];
                                dirty_indices[k_rs2 as usize] = true;
                                rs2_ra[1][k_rs2 as usize] += A[j_bound];
                                dirty_indices[k_rd as usize] = true;
                                rd_wa[1][k_rd as usize] += A[j_bound];
                            }

                            for k in (0..K).filter(|&k| dirty_indices[k]) {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First row
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col as usize] += *inc_lt;
                                val_j_0[*col as usize] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for k in (0..K).filter(|&k| dirty_indices[k]) {
                                val_j_r[1][k] = val_j_0[k];
                            }

                            // Second row
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col as usize] += inc_lt;
                                val_j_0[col as usize] += inc;
                            }

                            let eq_r_prime_eval = gruens_eq_r_prime.E_out_current()[j_prime / 2];
                            let inc_cycle_evals = {
                                let inc_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                [inc_0, inc_1 - inc_0] // [eval_at_0, slope]
                            };

                            // Compute inner sum over dirty indices
                            let mut rd_inner = [Rep3PrimeFieldShare::<F>::zero_share(); 2];
                            let mut rs1_inner = [Rep3PrimeFieldShare::<F>::zero_share(); 2];
                            let mut rs2_inner = [Rep3PrimeFieldShare::<F>::zero_share(); 2];

                            for k in (0..K).filter(|&k| dirty_indices[k]) {
                                let val_0 = val_j_r[0][k]; // SHARED
                                let val_slope = val_j_r[1][k] - val_0; // SHARED

                                if !rd_wa[0][k].is_zero() || !rd_wa[1][k].is_zero() {
                                    let wa_0 = rd_wa[0][k]; // PUBLIC
                                    let wa_slope = rd_wa[1][k] - wa_0;
                                    // wa(PUB) * (inc(SHARED) + val(SHARED))
                                    rd_inner[0] +=
                                        rep3_arith::mul_public(inc_cycle_evals[0] + val_0, wa_0);
                                    rd_inner[1] += rep3_arith::mul_public(
                                        inc_cycle_evals[1] + val_slope,
                                        wa_slope,
                                    );
                                    rd_wa[0][k] = F::zero();
                                    rd_wa[1][k] = F::zero();
                                }

                                if !rs1_ra[0][k].is_zero() || !rs1_ra[1][k].is_zero() {
                                    let ra_0 = rs1_ra[0][k];
                                    let ra_slope = rs1_ra[1][k] - ra_0;
                                    rs1_inner[0] += rep3_arith::mul_public(val_0, ra_0);
                                    rs1_inner[1] += rep3_arith::mul_public(val_slope, ra_slope);
                                    rs1_ra[0][k] = F::zero();
                                    rs1_ra[1][k] = F::zero();
                                }

                                if !rs2_ra[0][k].is_zero() || !rs2_ra[1][k].is_zero() {
                                    let ra_0 = rs2_ra[0][k];
                                    let ra_slope = rs2_ra[1][k] - ra_0;
                                    rs2_inner[0] += rep3_arith::mul_public(val_0, ra_0);
                                    rs2_inner[1] += rep3_arith::mul_public(val_slope, ra_slope);
                                    rs2_ra[0][k] = F::zero();
                                    rs2_ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = Rep3PrimeFieldShare::zero_share();
                                val_j_r[1][k] = Rep3PrimeFieldShare::zero_share();
                            }
                            dirty_indices.fill(false);

                            // Combine with gamma (PUBLIC)
                            let sum_0 = rd_inner[0]
                                + rep3_arith::mul_public(rs1_inner[0], gamma)
                                + rep3_arith::mul_public(rs2_inner[0], gamma_sqr);
                            let sum_1 = rd_inner[1]
                                + rep3_arith::mul_public(rs1_inner[1], gamma)
                                + rep3_arith::mul_public(rs2_inner[1], gamma_sqr);

                            // eq_r_prime(PUB) * sum(SHARED)
                            evals[0] = evals[0].add(
                                &Rep3Value::Shared(rep3_arith::mul_public(sum_0, eq_r_prime_eval)),
                                party_id,
                            );
                            evals[1] = evals[1].add(
                                &Rep3Value::Shared(rep3_arith::mul_public(sum_1, eq_r_prime_eval)),
                                party_id,
                            );
                        });
                    evals
                })
                .reduce(
                    || [Rep3Value::zero_share(); 2],
                    |running, new| {
                        [
                            running[0].add(&new[0], party_id),
                            running[1].add(&new[1], party_id),
                        ]
                    },
                )
        } else {
            // E_in not fully bound — handle E_in and E_out
            let num_x_in_bits = gruens_eq_r_prime.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [Rep3Value::zero_share(); 2];
                    let mut evals_for_current_E_out = [Rep3Value::zero_share(); 2];
                    let mut x_out_prev: Option<usize> = None;

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        rs1_ra,
                        rs2_ra,
                        rd_wa,
                        dirty_indices,
                    } = buffers;
                    val_j_0.copy_from_slice(checkpoint);

                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0;

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                let (k_rs1, k_rs2, k_rd) = addresses[j];
                                dirty_indices[k_rs1 as usize] = true;
                                rs1_ra[0][k_rs1 as usize] += A[j_bound];
                                dirty_indices[k_rs2 as usize] = true;
                                rs2_ra[0][k_rs2 as usize] += A[j_bound];
                                dirty_indices[k_rd as usize] = true;
                                rd_wa[0][k_rd as usize] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                let (k_rs1, k_rs2, k_rd) = addresses[j];
                                dirty_indices[k_rs1 as usize] = true;
                                rs1_ra[1][k_rs1 as usize] += A[j_bound];
                                dirty_indices[k_rs2 as usize] = true;
                                rs2_ra[1][k_rs2 as usize] += A[j_bound];
                                dirty_indices[k_rd as usize] = true;
                                rd_wa[1][k_rd as usize] += A[j_bound];
                            }

                            for k in (0..K).filter(|&k| dirty_indices[k]) {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col as usize] += *inc_lt;
                                val_j_0[*col as usize] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for k in (0..K).filter(|&k| dirty_indices[k]) {
                                val_j_r[1][k] = val_j_0[k];
                            }
                            for entry in inc_iter {
                                let (row, col, inc_lt, inc) = *entry;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col as usize] += inc_lt;
                                val_j_0[col as usize] += inc;
                            }

                            let x_in = (j_prime / 2) & x_bitmask;
                            let x_out = (j_prime / 2) >> num_x_in_bits;
                            let E_in_eval = gruens_eq_r_prime.E_in_current()[x_in];

                            let inc_cycle_evals = {
                                let inc_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                [inc_0, inc_1 - inc_0]
                            };

                            match x_out_prev {
                                None => {
                                    x_out_prev = Some(x_out);
                                }
                                Some(x) if x_out != x => {
                                    x_out_prev = Some(x_out);
                                    let E_out_eval = gruens_eq_r_prime.E_out_current()[x];
                                    evals[0] = evals[0].add(
                                        &evals_for_current_E_out[0].mul_public(E_out_eval),
                                        party_id,
                                    );
                                    evals[1] = evals[1].add(
                                        &evals_for_current_E_out[1].mul_public(E_out_eval),
                                        party_id,
                                    );
                                    evals_for_current_E_out = [Rep3Value::zero_share(); 2];
                                }
                                _ => (),
                            }

                            let mut rd_inner = [Rep3PrimeFieldShare::<F>::zero_share(); 2];
                            let mut rs1_inner = [Rep3PrimeFieldShare::<F>::zero_share(); 2];
                            let mut rs2_inner = [Rep3PrimeFieldShare::<F>::zero_share(); 2];

                            for k in (0..K).filter(|&k| dirty_indices[k]) {
                                let val_0 = val_j_r[0][k];
                                let val_slope = val_j_r[1][k] - val_0;

                                if !rd_wa[0][k].is_zero() || !rd_wa[1][k].is_zero() {
                                    let wa_0 = rd_wa[0][k];
                                    let wa_slope = rd_wa[1][k] - wa_0;
                                    rd_inner[0] +=
                                        rep3_arith::mul_public(inc_cycle_evals[0] + val_0, wa_0);
                                    rd_inner[1] += rep3_arith::mul_public(
                                        inc_cycle_evals[1] + val_slope,
                                        wa_slope,
                                    );
                                    rd_wa[0][k] = F::zero();
                                    rd_wa[1][k] = F::zero();
                                }

                                if !rs1_ra[0][k].is_zero() || !rs1_ra[1][k].is_zero() {
                                    let ra_0 = rs1_ra[0][k];
                                    let ra_slope = rs1_ra[1][k] - ra_0;
                                    rs1_inner[0] += rep3_arith::mul_public(val_0, ra_0);
                                    rs1_inner[1] += rep3_arith::mul_public(val_slope, ra_slope);
                                    rs1_ra[0][k] = F::zero();
                                    rs1_ra[1][k] = F::zero();
                                }

                                if !rs2_ra[0][k].is_zero() || !rs2_ra[1][k].is_zero() {
                                    let ra_0 = rs2_ra[0][k];
                                    let ra_slope = rs2_ra[1][k] - ra_0;
                                    rs2_inner[0] += rep3_arith::mul_public(val_0, ra_0);
                                    rs2_inner[1] += rep3_arith::mul_public(val_slope, ra_slope);
                                    rs2_ra[0][k] = F::zero();
                                    rs2_ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = Rep3PrimeFieldShare::zero_share();
                                val_j_r[1][k] = Rep3PrimeFieldShare::zero_share();
                            }
                            dirty_indices.fill(false);

                            let sum_0 = rd_inner[0]
                                + rep3_arith::mul_public(rs1_inner[0], gamma)
                                + rep3_arith::mul_public(rs2_inner[0], gamma_sqr);
                            let sum_1 = rd_inner[1]
                                + rep3_arith::mul_public(rs1_inner[1], gamma)
                                + rep3_arith::mul_public(rs2_inner[1], gamma_sqr);

                            // E_in(PUB) * sum(SHARED)
                            evals_for_current_E_out[0] = evals_for_current_E_out[0].add(
                                &Rep3Value::Shared(rep3_arith::mul_public(sum_0, E_in_eval)),
                                party_id,
                            );
                            evals_for_current_E_out[1] = evals_for_current_E_out[1].add(
                                &Rep3Value::Shared(rep3_arith::mul_public(sum_1, E_in_eval)),
                                party_id,
                            );
                        });

                    if let Some(x) = x_out_prev {
                        let E_out_eval = gruens_eq_r_prime.E_out_current()[x];
                        evals[0] = evals[0].add(
                            &evals_for_current_E_out[0].mul_public(E_out_eval),
                            party_id,
                        );
                        evals[1] = evals[1].add(
                            &evals_for_current_E_out[1].mul_public(E_out_eval),
                            party_id,
                        );
                    }
                    evals
                })
                .reduce(
                    || [Rep3Value::zero_share(); 2],
                    |running, new| {
                        [
                            running[0].add(&new[0], party_id),
                            running[1].add(&new[1], party_id),
                        ]
                    },
                )
        };

        // Use gruen_evals_deg_3 to convert quadratic coefficients to degree-3 evaluations
        gruen_evals_deg_3(
            &self.prover_state.gruens_eq_r_prime,
            quadratic_coeffs[0],
            quadratic_coeffs[1],
            previous_claim,
            party_id,
        )
    }

    fn phase2_compute_prover_message(&self) -> Vec<AdditiveShare<F>> {
        let ReadWriteCheckingProverState {
            inc_cycle,
            eq_r_prime,
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            ..
        } = &self.prover_state;
        let rs1_ra = rs1_ra.as_ref().unwrap();
        let rs2_ra = rs2_ra.as_ref().unwrap();
        let rd_wa = rd_wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();
        let eq_r_prime = eq_r_prime.as_ref().unwrap();

        let gamma = self.gamma;
        let gamma_sqr = self.gamma_sqr;

        // eq_r_prime(PUB), rs1_ra/rs2_ra/rd_wa(PUB), inc_cycle(SHARED), val(SHARED)
        //
        // sumcheck_evals with DEGREE=3 returns evaluations at {0, 2, 3} (skipping 1).
        // The manual interpolation for public polys must match these evaluation points.
        // For a linear poly with values (v0, v1) at (0, 1), the evals at {0, 2, 3} are:
        //   f(0) = v0,  f(2) = 2*v1 - v0,  f(3) = 3*v1 - 2*v0
        const EVAL_POINTS: [u64; DEGREE] = [0, 2, 3];
        let evals: Vec<AdditiveShare<F>> = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|j| {
                let eq_0 = eq_r_prime.get_bound_coeff(j);
                let eq_1 = eq_r_prime.get_bound_coeff(j + eq_r_prime.len() / 2);
                let eq_m = eq_1 - eq_0;

                let inc_evals = inc_cycle.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
                // Inner sum over K registers
                let mut inner = [Rep3PrimeFieldShare::<F>::zero_share(); DEGREE];
                for k in 0..K {
                    let index = j * K + k;
                    let rs1_0 = rs1_ra.get_bound_coeff(index);
                    let rs1_m = rs1_ra.get_bound_coeff(index + rs1_ra.len() / 2) - rs1_0;
                    let rs2_0 = rs2_ra.get_bound_coeff(index);
                    let rs2_m = rs2_ra.get_bound_coeff(index + rs2_ra.len() / 2) - rs2_0;
                    let wa_0 = rd_wa.get_bound_coeff(index);
                    let wa_m = rd_wa.get_bound_coeff(index + rd_wa.len() / 2) - wa_0;
                    let val_evals = val.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);

                    for d in 0..DEGREE {
                        let x = F::from_u64(EVAL_POINTS[d]);
                        let rs1_e = rs1_0 + rs1_m * x;
                        let rs2_e = rs2_0 + rs2_m * x;
                        let wa_e = wa_0 + wa_m * x;
                        // wa(PUB) * (inc(SHARED) + val(SHARED)) + gamma * rs1(PUB) * val(SHARED) + gamma^2 * rs2(PUB) * val(SHARED)
                        inner[d] += rep3_arith::mul_public(inc_evals[d] + val_evals[d], wa_e)
                            + rep3_arith::mul_public(val_evals[d], gamma * rs1_e)
                            + rep3_arith::mul_public(val_evals[d], gamma_sqr * rs2_e);
                    }
                }

                // eq_r_prime(PUB) * inner(SHARED) → AdditiveShare
                let mut result = [AdditiveShare::<F>::zero(); DEGREE];
                for d in 0..DEGREE {
                    let x = F::from_u64(EVAL_POINTS[d]);
                    let eq_e = eq_0 + eq_m * x;
                    // PUB * SHARED → SHARED → AdditiveShare
                    let prod = rep3_arith::mul_public(inner[d], eq_e);
                    result[d] = prod.into_additive();
                }
                result
            })
            .reduce(
                || [AdditiveShare::<F>::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            )
            .to_vec();

        evals
    }

    fn phase3_compute_prover_message(&self) -> Vec<AdditiveShare<F>> {
        let ReadWriteCheckingProverState {
            inc_cycle,
            eq_r_prime,
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            ..
        } = &self.prover_state;
        let rs1_ra = rs1_ra.as_ref().unwrap();
        let rs2_ra = rs2_ra.as_ref().unwrap();
        let rd_wa = rd_wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        let eq_r_prime_eval = eq_r_prime.as_ref().unwrap().final_sumcheck_claim();
        let inc_eval = inc_cycle.final_sumcheck_claim();

        let gamma = self.gamma;
        let gamma_sqr = self.gamma_sqr;

        // sumcheck_evals returns evaluations at {0, 2, 3}; match these for public polys.
        const EVAL_POINTS: [u64; DEGREE] = [0, 2, 3];

        let evals: [AdditiveShare<F>; DEGREE] = (0..rs1_ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let rs1_0 = rs1_ra.get_bound_coeff(k);
                let rs1_m = rs1_ra.get_bound_coeff(k + rs1_ra.len() / 2) - rs1_0;
                let rs2_0 = rs2_ra.get_bound_coeff(k);
                let rs2_m = rs2_ra.get_bound_coeff(k + rs2_ra.len() / 2) - rs2_0;
                let wa_0 = rd_wa.get_bound_coeff(k);
                let wa_m = rd_wa.get_bound_coeff(k + rd_wa.len() / 2) - wa_0;
                let val_evals = val.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);

                let mut result = [AdditiveShare::<F>::zero(); DEGREE];
                for d in 0..DEGREE {
                    let x = F::from_u64(EVAL_POINTS[d]);
                    let rs1_e = rs1_0 + rs1_m * x;
                    let rs2_e = rs2_0 + rs2_m * x;
                    let wa_e = wa_0 + wa_m * x;
                    // wa(PUB) * (inc(SHARED) + val(SHARED))
                    let term = rep3_arith::mul_public(inc_eval + val_evals[d], wa_e)
                        + rep3_arith::mul_public(val_evals[d], gamma * rs1_e)
                        + rep3_arith::mul_public(val_evals[d], gamma_sqr * rs2_e);
                    result[d] = term.into_additive();
                }
                result
            })
            .reduce(
                || [AdditiveShare::<F>::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        // Multiply by public eq_r_prime_eval
        evals.iter().map(|e| *e * eq_r_prime_eval).collect()
    }

    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let ReadWriteCheckingProverState {
            addresses,
            I,
            A,
            inc_cycle,
            gruens_eq_r_prime,
            eq_r_prime,
            chunk_size,
            val_checkpoints,
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            ..
        } = &mut self.prover_state;

        // Bind I
        I.par_iter_mut().for_each(|I_chunk| {
            let mut next_bound_index = 0;
            let mut bound_indices: Vec<Option<usize>> = vec![None; K];

            for i in 0..I_chunk.len() {
                let (j_prime, k, inc_lt, inc) = I_chunk[i];
                if let Some(bound_index) = bound_indices[k as usize] {
                    if I_chunk[bound_index].0 == j_prime / 2 {
                        debug_assert!(j_prime % 2 == 1);
                        I_chunk[bound_index].2 += rep3_arith::mul_public(inc_lt, r_j.into());
                        I_chunk[bound_index].3 += inc;
                        continue;
                    }
                }
                let bound_value = if j_prime % 2 == 0 {
                    inc_lt + rep3_arith::mul_public(inc - inc_lt, r_j.into())
                } else {
                    rep3_arith::mul_public(inc_lt, r_j.into())
                };
                I_chunk[next_bound_index] = (j_prime / 2, k, bound_value, inc);
                bound_indices[k as usize] = Some(next_bound_index);
                next_bound_index += 1;
            }
            I_chunk.truncate(next_bound_index);
        });

        gruens_eq_r_prime.bind(r_j);
        inc_cycle.bind(r_j.into(), BindingOrder::LowToHigh);

        // Update A
        let (A_left, A_right) = A.split_at_mut(1 << round);
        A_left
            .par_iter_mut()
            .zip(A_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });

        if round == chunk_size.log_2() - 1 {
            // Materialize full polynomials
            let num_chunks = addresses.len() / *chunk_size;

            // rs1_ra (PUBLIC)
            let mut rs1_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            rs1_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    for (jb, (k, _, _)) in addresses[ci * *chunk_size..(ci + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        chunk[*k as usize] += A[jb];
                    }
                });
            *rs1_ra = Some(MultilinearPolynomial::from(rs1_evals));

            // rs2_ra (PUBLIC)
            let mut rs2_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            rs2_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    for (jb, (_, k, _)) in addresses[ci * *chunk_size..(ci + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        chunk[*k as usize] += A[jb];
                    }
                });
            *rs2_ra = Some(MultilinearPolynomial::from(rs2_evals));

            // rd_wa (PUBLIC)
            let mut wa_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            wa_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    for (jb, (_, _, k)) in addresses[ci * *chunk_size..(ci + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        chunk[*k as usize] += A[jb];
                    }
                });
            *rd_wa = Some(MultilinearPolynomial::from(wa_evals));

            // Val (SHARED) — from checkpoints + I increments
            let mut val_evals: Vec<Rep3PrimeFieldShare<F>> = std::mem::take(val_checkpoints);
            val_evals
                .par_chunks_mut(K)
                .zip(I.into_par_iter())
                .enumerate()
                .for_each(|(ci, (val_chunk, I_chunk))| {
                    for (j, k, inc_lt, _inc) in I_chunk.iter_mut() {
                        debug_assert_eq!(*j, ci);
                        val_chunk[*k as usize] += *inc_lt;
                    }
                });
            *val = Some(Rep3DensePolynomial::new(val_evals));

            // eq_r_prime (PUBLIC)
            let eq_evals: Vec<F> =
                EqPolynomial::<F>::evals(&gruens_eq_r_prime.w[..gruens_eq_r_prime.current_index])
                    .par_iter()
                    .map(|x| *x * gruens_eq_r_prime.current_scalar)
                    .collect();
            *eq_r_prime = Some(MultilinearPolynomial::from(eq_evals));
        }
    }

    fn phase2_bind(&mut self, r_j: F::Challenge) {
        let ps = &mut self.prover_state;
        let rs1 = ps.rs1_ra.as_mut().unwrap();
        let rs2 = ps.rs2_ra.as_mut().unwrap();
        let wa = ps.rd_wa.as_mut().unwrap();
        let eq = ps.eq_r_prime.as_mut().unwrap();

        // PUBLIC polys
        [rs1, rs2, wa, eq]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        // SHARED polys
        rayon::join(
            || {
                ps.val
                    .as_mut()
                    .unwrap()
                    .bind(r_j.into(), BindingOrder::HighToLow)
            },
            || ps.inc_cycle.bind(r_j.into(), BindingOrder::HighToLow),
        );
    }

    fn phase3_bind(&mut self, r_j: F::Challenge) {
        let ps = &mut self.prover_state;
        let rs1 = ps.rs1_ra.as_mut().unwrap();
        let rs2 = ps.rs2_ra.as_mut().unwrap();
        let wa = ps.rd_wa.as_mut().unwrap();

        [rs1, rs2, wa]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        ps.val
            .as_mut()
            .unwrap()
            .bind(r_j.into(), BindingOrder::HighToLow);
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N> for Rep3RegistersReadWriteCheckingWorker<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> Rep3Value<F> {
        Rep3Value::Public(self.input_claim)
    }

    fn compute_prover_message_share(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        let chunk_size_log = self.prover_state.chunk_size.log_2();
        let log_T = self.T.log_2();

        let evals = if round < chunk_size_log {
            self.phase1_compute_prover_message(round, previous_claim)
        } else if round < log_T {
            self.phase2_compute_prover_message()
        } else {
            self.phase3_compute_prover_message()
        };

        // Pad to max_degree if needed
        if evals.len() < max_degree {
            extend_degree_3_evals(previous_claim, &evals, max_degree)
        } else {
            evals
        }
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize, _io_ctx: &mut IoContextPool<N>) {
        let chunk_size_log = self.prover_state.chunk_size.log_2();
        let log_T = self.T.log_2();

        if round < chunk_size_log {
            self.phase1_bind(r_j, round);
        } else if round < log_T {
            self.phase2_bind(r_j);
        } else {
            self.phase3_bind(r_j);
        }
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let log_T = self.T.log_2();
        let mut r_cycle = opening_point[self.sumcheck_switch_index..log_T].to_vec();
        r_cycle.extend(opening_point[..self.sumcheck_switch_index].iter().rev());
        let r_address = opening_point[log_T..].to_vec();
        [r_address, r_cycle].concat().into()
    }

    fn cache_openings_worker(
        &mut self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let ps = &self.prover_state;

        let val_claim = ps.val.as_ref().unwrap().final_sumcheck_claim();
        let rs1_ra_claim = ps.rs1_ra.as_ref().unwrap().final_sumcheck_claim();
        let rs2_ra_claim = ps.rs2_ra.as_ref().unwrap().final_sumcheck_claim();
        let rd_wa_claim = ps.rd_wa.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = ps.inc_cycle.final_sumcheck_claim();

        // val is SHARED
        accumulator.append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            val_claim,
        );

        // rs1_ra, rs2_ra, rd_wa are PUBLIC
        accumulator.append_virtual_public(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rs1_ra_claim,
            self.party_id,
        );
        accumulator.append_virtual_public(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rs2_ra_claim,
            self.party_id,
        );
        accumulator.append_virtual_public(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rd_wa_claim,
            self.party_id,
        );

        // inc is SHARED, committed
        let (_, r_cycle) = opening_point.split_at(K.log_2());
        accumulator.append_dense(
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
            &[inc_claim],
        );

        vec![
            val_claim,
            rep3_arith::promote_to_trivial_share(self.party_id, rs1_ra_claim),
            rep3_arith::promote_to_trivial_share(self.party_id, rs2_ra_claim),
            rep3_arith::promote_to_trivial_share(self.party_id, rd_wa_claim),
            inc_claim,
        ]
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3RegistersReadWriteChecking<F: JoltField> {
    T: usize,
    gamma: F,
    gamma_sqr: F,
    sumcheck_switch_index: usize,
    input_claim: F,
}

impl<F: JoltField> Rep3RegistersReadWriteChecking<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (r_point, rs1_rv_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rs2_rv_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        let (_, rd_wv_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );

        let gamma: F = sm.transcript.challenge_scalar();
        let input_claim = rd_wv_claim + gamma * rs1_rv_claim + gamma.square() * rs2_rv_claim;

        let T = 1 << r_point.r.len();

        Self {
            T,
            gamma,
            gamma_sqr: gamma.square(),
            sumcheck_switch_index: sm.twist_sumcheck_switch_index,
            input_claim,
        }
    }

    /// Gamma challenge for broadcasting to workers.
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// Input claim for broadcasting to workers.
    pub fn input_claim(&self) -> F {
        self.input_claim
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3RegistersReadWriteChecking<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        K.log_2() + self.T.log_2()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let (r_prime, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);

        let mut r_cycle = r[..self.sumcheck_switch_index].to_vec();
        r_cycle.extend(r[self.sumcheck_switch_index..self.T.log_2()].iter().rev());
        let r_cycle = OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle);

        let eq_eval_cycle = EqPolynomial::mle_endian(&r_prime, &r_cycle);

        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rd_wa_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );

        eq_eval_cycle
            * (rd_wa_claim * (inc_claim + val_claim)
                + self.gamma * rs1_ra_claim * val_claim
                + self.gamma_sqr * rs2_ra_claim * val_claim)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let log_T = self.T.log_2();
        let mut r_cycle = opening_point[self.sumcheck_switch_index..log_T].to_vec();
        r_cycle.extend(opening_point[..self.sumcheck_switch_index].iter().rev());
        let r_address = opening_point[log_T..].to_vec();
        [r_address, r_cycle].concat().into()
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        // claims: [val, rs1_ra, rs2_ra, rd_wa, inc]
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            claims[0],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            claims[1],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            claims[2],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            claims[3],
        );

        let (_, r_cycle) = opening_point.split_at(K.log_2());
        accumulator.append_dense(
            transcript,
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
            vec![claims[4]],
        );
    }
}
