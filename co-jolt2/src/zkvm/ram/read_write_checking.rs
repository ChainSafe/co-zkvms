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
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::utils::types::Rep3Value;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::Rep3SumcheckInstanceWorker;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::instruction_lookups::booleanity::{extend_degree_3_evals, gruen_evals_deg_3};

const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Per-chunk data buffers for phase1 of RAM ReadWriteChecking.
struct DataBuffers<F: JoltField> {
    val_j_0: Vec<Rep3PrimeFieldShare<F>>,
    val_j_r: [Vec<Rep3PrimeFieldShare<F>>; 2],
    ra: [Vec<F>; 2],
    dirty_indices: Vec<usize>,
}

struct ReadWriteCheckingProverState<F: JoltField> {
    ram_addresses: Vec<Option<u64>>,
    K: usize,
    chunk_size: usize,
    val_checkpoints: Vec<Rep3PrimeFieldShare<F>>,
    data_buffers: Vec<DataBuffers<F>>,
    I: Vec<Vec<(usize, usize, Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>)>>,
    A: Vec<F>,
    gruens_eq_r_prime: GruenSplitEqPolynomial<F>,
    inc_cycle: Rep3DensePolynomial<F>,
    // Materialized after phase1
    eq_r_prime: Option<MultilinearPolynomial<F>>,
    ra: Option<MultilinearPolynomial<F>>,
    val: Option<Rep3DensePolynomial<F>>,
}

impl<F: JoltField> ReadWriteCheckingProverState<F> {
    fn initialize<PCS: CommitmentScheme<Field = F>>(
        initial_memory_state: &[Rep3PrimeFieldShare<F>],
        K: usize,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Self {
        let cycle_witness = &sm.prover_state.cycle_witness;
        let memory_layout = &sm.program_io.memory_layout;
        let T = cycle_witness.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        // Clone is cheap (Rep3DensePolynomial is Arc-backed); we avoid lifetime plumbing here.
        let inc_cycle = sm.prover_state.cycle_witness.ram_inc_ref().clone();

        let r_prime = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamReadValue,
                SumcheckId::SpartanOuter,
            )
            .0;

        // Compute ram_addresses (PUBLIC)
        let ram_addresses: Vec<Option<u64>> = cycle_witness
            .meta()
            .par_iter()
            .map(|m| remap_address(m.ram_addr, memory_layout))
            .collect();

        // Compute checkpoints (val at each chunk start) without materializing per-chunk deltas.
        //
        // val_checkpoints[chunk][k] = initial_memory_state[k] + Σ_{j in cycles < chunk_start} inc(j) * [addr(j)==k]
        let mut val_checkpoints: Vec<Rep3PrimeFieldShare<F>> =
            vec![Rep3PrimeFieldShare::zero_share(); K * num_chunks];
        val_checkpoints[..K].copy_from_slice(initial_memory_state);

        let mut running: Vec<Rep3PrimeFieldShare<F>> = initial_memory_state.to_vec();
        for chunk_index in 0..(num_chunks.saturating_sub(1)) {
            let base = chunk_index * chunk_size;
            for offset in 0..chunk_size {
                let j = base + offset;
                let k = ram_addresses[j].unwrap_or(0) as usize;
                running[k] += inc_cycle.get_bound_coeff(j);
            }
            let start = (chunk_index + 1) * K;
            val_checkpoints[start..start + K].copy_from_slice(&running);
        }

        // EQ table (PUBLIC)
        let mut A: Vec<F> = unsafe_allocate_zero_vec(chunk_size);
        A[0] = F::one();

        // Build I data structure
        let I: Vec<Vec<(usize, usize, Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>)>> =
            ram_addresses
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_index, addr_chunk)| {
                    let mut j = chunk_index * chunk_size;
                    addr_chunk
                        .iter()
                        .map(|addr| {
                            let k = addr.unwrap_or(0) as usize;
                            let inc_val = inc_cycle.get_bound_coeff(j);
                            let entry = (j, k, Rep3PrimeFieldShare::zero_share(), inc_val);
                            j += 1;
                            entry
                        })
                        .collect()
                })
                .collect();

        let gruens_eq_r_prime = GruenSplitEqPolynomial::new(&r_prime.r, BindingOrder::LowToHigh);

        let data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: vec![Rep3PrimeFieldShare::zero_share(); K],
                val_j_r: [
                    vec![Rep3PrimeFieldShare::zero_share(); K],
                    vec![Rep3PrimeFieldShare::zero_share(); K],
                ],
                ra: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                dirty_indices: Vec::with_capacity(K),
            })
            .collect();

        ReadWriteCheckingProverState {
            ram_addresses,
            K,
            chunk_size,
            val_checkpoints,
            data_buffers,
            I,
            A,
            gruens_eq_r_prime,
            inc_cycle,
            eq_r_prime: None,
            ra: None,
            val: None,
        }
    }
}

pub struct Rep3RamReadWriteCheckingWorker<F: JoltField> {
    party_id: PartyID,
    K: usize,
    T: usize,
    gamma: F,
    sumcheck_switch_index: usize,
    prover_state: ReadWriteCheckingProverState<F>,
    input_claim: F,
}

impl<F: JoltField> Rep3RamReadWriteCheckingWorker<F> {
    pub fn new<PCS: CommitmentScheme<Field = F>>(
        initial_memory_state: &[Rep3PrimeFieldShare<F>],
        sm: &mut StateManagerWorker<'_, F, PCS>,
        gamma: F,
        input_claim: F,
    ) -> Self {
        let party_id = sm.party_id;
        let K = sm.ram_K;
        let T = sm.prover_state.cycle_witness.len();

        let prover_state = ReadWriteCheckingProverState::initialize(initial_memory_state, K, sm);

        Self {
            party_id,
            K,
            T,
            gamma,
            sumcheck_switch_index: sm.twist_sumcheck_switch_index,
            prover_state,
            input_claim,
        }
    }

    fn phase1_compute_prover_message(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
    ) -> Vec<AdditiveShare<F>> {
        let K = self.K;
        let ReadWriteCheckingProverState {
            ram_addresses,
            I,
            data_buffers,
            A,
            val_checkpoints,
            inc_cycle,
            gruens_eq_r_prime,
            ..
        } = &mut self.prover_state;

        let gamma = self.gamma;
        let party_id = self.party_id;

        let quadratic_coeffs: [Rep3Value<F>; 2] = if gruens_eq_r_prime.E_in_current_len() == 1 {
            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [Rep3Value::zero_share(); 2];

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        ra,
                        dirty_indices,
                    } = buffers;
                    val_j_0.copy_from_slice(checkpoint);

                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0;

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                if let Some(k) = ram_addresses[j] {
                                    let k = k as usize;
                                    if ra[0][k].is_zero() {
                                        dirty_indices.push(k);
                                    }
                                    ra[0][k] += A[j_bound];
                                }
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                if let Some(k) = ram_addresses[j] {
                                    let k = k as usize;
                                    if ra[0][k].is_zero() && ra[1][k].is_zero() {
                                        dirty_indices.push(k);
                                    }
                                    ra[1][k] += A[j_bound];
                                }
                            }

                            for &k in dirty_indices.iter() {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col] += *inc_lt;
                                val_j_0[*col] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for &k in dirty_indices.iter() {
                                val_j_r[1][k] = val_j_0[k];
                            }
                            for entry in inc_iter {
                                let (row, col, inc_lt, inc) = *entry;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] += inc;
                            }

                            let eq_r_prime_eval = gruens_eq_r_prime.E_out_current()[j_prime / 2];
                            let inc_cycle_evals = {
                                let inc_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                [inc_0, inc_1 - inc_0]
                            };

                            let mut inner = [Rep3PrimeFieldShare::<F>::zero_share(); 2];
                            for k in dirty_indices.drain(..) {
                                if !ra[0][k].is_zero() || !ra[1][k].is_zero() {
                                    let ra_0 = ra[0][k]; // PUBLIC
                                    let ra_slope = ra[1][k] - ra_0;
                                    let val_0 = val_j_r[0][k]; // SHARED
                                    let val_slope = val_j_r[1][k] - val_0;

                                    // ra(PUB) * (val(SHARED) + gamma * (inc(SHARED) + val(SHARED)))
                                    inner[0] += rep3_arith::mul_public(
                                        val_0
                                            + rep3_arith::mul_public(
                                                inc_cycle_evals[0] + val_0,
                                                gamma,
                                            ),
                                        ra_0,
                                    );
                                    inner[1] += rep3_arith::mul_public(
                                        val_slope
                                            + rep3_arith::mul_public(
                                                inc_cycle_evals[1] + val_slope,
                                                gamma,
                                            ),
                                        ra_slope,
                                    );

                                    ra[0][k] = F::zero();
                                    ra[1][k] = F::zero();
                                }
                                val_j_r[0][k] = Rep3PrimeFieldShare::zero_share();
                                val_j_r[1][k] = Rep3PrimeFieldShare::zero_share();
                            }

                            evals[0] = evals[0].add(
                                &Rep3Value::Shared(rep3_arith::mul_public(
                                    inner[0],
                                    eq_r_prime_eval,
                                )),
                                party_id,
                            );
                            evals[1] = evals[1].add(
                                &Rep3Value::Shared(rep3_arith::mul_public(
                                    inner[1],
                                    eq_r_prime_eval,
                                )),
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
            // E_in not fully bound
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
                        ra,
                        dirty_indices,
                    } = buffers;
                    val_j_0.copy_from_slice(checkpoint);

                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0;

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                if let Some(k) = ram_addresses[j] {
                                    let k = k as usize;
                                    if ra[0][k].is_zero() {
                                        dirty_indices.push(k);
                                    }
                                    ra[0][k] += A[j_bound];
                                }
                            }
                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                if let Some(k) = ram_addresses[j] {
                                    let k = k as usize;
                                    if ra[0][k].is_zero() && ra[1][k].is_zero() {
                                        dirty_indices.push(k);
                                    }
                                    ra[1][k] += A[j_bound];
                                }
                            }

                            for &k in dirty_indices.iter() {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col] += *inc_lt;
                                val_j_0[*col] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for &k in dirty_indices.iter() {
                                val_j_r[1][k] = val_j_0[k];
                            }
                            for entry in inc_iter {
                                let (row, col, inc_lt, inc) = *entry;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] += inc;
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

                            let mut inner = [Rep3PrimeFieldShare::<F>::zero_share(); 2];
                            for k in dirty_indices.drain(..) {
                                if !ra[0][k].is_zero() || !ra[1][k].is_zero() {
                                    let ra_0 = ra[0][k];
                                    let ra_slope = ra[1][k] - ra_0;
                                    let val_0 = val_j_r[0][k];
                                    let val_slope = val_j_r[1][k] - val_0;

                                    inner[0] += rep3_arith::mul_public(
                                        val_0
                                            + rep3_arith::mul_public(
                                                inc_cycle_evals[0] + val_0,
                                                gamma,
                                            ),
                                        ra_0,
                                    );
                                    inner[1] += rep3_arith::mul_public(
                                        val_slope
                                            + rep3_arith::mul_public(
                                                inc_cycle_evals[1] + val_slope,
                                                gamma,
                                            ),
                                        ra_slope,
                                    );

                                    ra[0][k] = F::zero();
                                    ra[1][k] = F::zero();
                                }
                                val_j_r[0][k] = Rep3PrimeFieldShare::zero_share();
                                val_j_r[1][k] = Rep3PrimeFieldShare::zero_share();
                            }

                            evals_for_current_E_out[0] = evals_for_current_E_out[0].add(
                                &Rep3Value::Shared(rep3_arith::mul_public(inner[0], E_in_eval)),
                                party_id,
                            );
                            evals_for_current_E_out[1] = evals_for_current_E_out[1].add(
                                &Rep3Value::Shared(rep3_arith::mul_public(inner[1], E_in_eval)),
                                party_id,
                            );
                        });

                    if let Some(x) = x_out_prev {
                        let E_out_eval = gruens_eq_r_prime.E_out_current()[x];
                        evals[0] = evals[0]
                            .add(&evals_for_current_E_out[0].mul_public(E_out_eval), party_id);
                        evals[1] = evals[1]
                            .add(&evals_for_current_E_out[1].mul_public(E_out_eval), party_id);
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

        gruen_evals_deg_3(
            &self.prover_state.gruens_eq_r_prime,
            quadratic_coeffs[0],
            quadratic_coeffs[1],
            previous_claim,
            self.party_id,
        )
    }

    fn phase2_compute_prover_message(&self) -> Vec<AdditiveShare<F>> {
        let ps = &self.prover_state;
        let ra = ps.ra.as_ref().unwrap();
        let val = ps.val.as_ref().unwrap();
        let eq_r_prime = ps.eq_r_prime.as_ref().unwrap();
        let inc_cycle = &ps.inc_cycle;
        let K = self.K;
        let gamma = self.gamma;

        let evals: [AdditiveShare<F>; DEGREE] = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|j| {
                let eq_evals: [F; DEGREE] =
                    eq_r_prime.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);
                let inc_evals = inc_cycle.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);

                let mut inner = [Rep3PrimeFieldShare::<F>::zero_share(); DEGREE];
                for k in 0..K {
                    let index = j * K + k;
                    let ra_evals: [F; DEGREE] =
                        ra.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                    let val_evals = val.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);

                    for d in 0..DEGREE {
                        // ra(PUB) * (val(SHARED) + gamma * (inc(SHARED) + val(SHARED)))
                        inner[d] += rep3_arith::mul_public(
                            val_evals[d]
                                + rep3_arith::mul_public(inc_evals[d] + val_evals[d], gamma),
                            ra_evals[d],
                        );
                    }
                }

                let mut result = [AdditiveShare::<F>::zero(); DEGREE];
                for d in 0..DEGREE {
                    result[d] = rep3_arith::mul_public(inner[d], eq_evals[d]).into_additive();
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

        evals.to_vec()
    }

    fn phase3_compute_prover_message(&self) -> Vec<AdditiveShare<F>> {
        let ps = &self.prover_state;
        let ra = ps.ra.as_ref().unwrap();
        let val = ps.val.as_ref().unwrap();
        let eq_r_prime_eval = ps.eq_r_prime.as_ref().unwrap().final_sumcheck_claim();
        let inc_eval = ps.inc_cycle.final_sumcheck_claim();
        let gamma = self.gamma;

        let evals: [AdditiveShare<F>; DEGREE] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let ra_evals: [F; DEGREE] =
                    ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let val_evals = val.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);

                let mut result = [AdditiveShare::<F>::zero(); DEGREE];
                for d in 0..DEGREE {
                    let term = rep3_arith::mul_public(
                        val_evals[d] + rep3_arith::mul_public(val_evals[d] + inc_eval, gamma),
                        ra_evals[d],
                    );
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

        evals.iter().map(|e| *e * eq_r_prime_eval).collect()
    }

    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = &mut self.prover_state;
        let K = ps.K;

        ps.I.par_iter_mut().for_each(|I_chunk| {
            let mut next_bound_index = 0;
            let mut bound_indices: Vec<Option<usize>> = vec![None; K];

            for i in 0..I_chunk.len() {
                let (j_prime, k, inc_lt, inc) = I_chunk[i];
                if let Some(bound_index) = bound_indices[k] {
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
                bound_indices[k] = Some(next_bound_index);
                next_bound_index += 1;
            }
            I_chunk.truncate(next_bound_index);
        });

        ps.gruens_eq_r_prime.bind(r_j);
        ps.inc_cycle.bind(r_j.into(), BindingOrder::LowToHigh);

        let (A_left, A_right) = ps.A.split_at_mut(1 << round);
        A_left
            .par_iter_mut()
            .zip(A_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });

        if round == ps.chunk_size.log_2() - 1 {
            let num_chunks = ps.ram_addresses.len() / ps.chunk_size;

            // Materialize ra (PUBLIC)
            let mut ra_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            ra_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    for (jb, addr) in ps.ram_addresses[ci * ps.chunk_size..(ci + 1) * ps.chunk_size]
                        .iter()
                        .enumerate()
                    {
                        if let Some(k) = addr {
                            chunk[*k as usize] += ps.A[jb];
                        }
                    }
                });
            ps.ra = Some(MultilinearPolynomial::from(ra_evals));

            // Materialize val (SHARED)
            let mut val_evals = std::mem::take(&mut ps.val_checkpoints);
            val_evals
                .par_chunks_mut(K)
                .zip(ps.I.par_iter_mut())
                .enumerate()
                .for_each(|(ci, (val_chunk, I_chunk))| {
                    for (j, k, inc_lt, _inc) in I_chunk.iter_mut() {
                        debug_assert_eq!(*j, ci);
                        val_chunk[*k] += *inc_lt;
                    }
                });
            ps.val = Some(Rep3DensePolynomial::new(val_evals));

            // Materialize eq_r_prime (PUBLIC)
            let eq_evals: Vec<F> = EqPolynomial::<F>::evals(
                &ps.gruens_eq_r_prime.w[..ps.gruens_eq_r_prime.current_index],
            )
            .par_iter()
            .map(|x| *x * ps.gruens_eq_r_prime.current_scalar)
            .collect();
            ps.eq_r_prime = Some(MultilinearPolynomial::from(eq_evals));
        }
    }

    fn phase2_bind(&mut self, r_j: F::Challenge) {
        let ps = &mut self.prover_state;
        let ra = ps.ra.as_mut().unwrap();
        let eq = ps.eq_r_prime.as_mut().unwrap();
        [ra, eq]
            .into_par_iter()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::HighToLow));
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
        ps.ra
            .as_mut()
            .unwrap()
            .bind_parallel(r_j, BindingOrder::HighToLow);
        ps.val
            .as_mut()
            .unwrap()
            .bind(r_j.into(), BindingOrder::HighToLow);
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3RamReadWriteCheckingWorker<F>
{
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
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
        let chunk_log = self.prover_state.chunk_size.log_2();
        let log_T = self.T.log_2();

        let evals = if round < chunk_log {
            self.phase1_compute_prover_message(round, previous_claim)
        } else if round < log_T {
            self.phase2_compute_prover_message()
        } else {
            self.phase3_compute_prover_message()
        };

        if evals.len() < max_degree {
            extend_degree_3_evals(previous_claim, &evals, max_degree)
        } else {
            evals
        }
    }

    fn bind(
        &mut self,
        r_j: F::Challenge,
        round: usize,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) {
        let chunk_log = self.prover_state.chunk_size.log_2();
        let log_T = self.T.log_2();

        if round < chunk_log {
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
        let ra_claim = ps.ra.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = ps.inc_cycle.final_sumcheck_claim();

        accumulator.append_virtual(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            val_claim,
        );
        accumulator.append_virtual_public(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            ra_claim,
            self.party_id,
        );

        let (_, r_cycle) = opening_point.split_at(self.K.log_2());
        accumulator.append_dense(
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
            &[inc_claim],
        );

        vec![
            val_claim,
            rep3_arith::promote_to_trivial_share(self.party_id, ra_claim),
            inc_claim,
        ]
    }
}

// ---------------------------------------------------------------------------
