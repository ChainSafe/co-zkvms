#![allow(clippy::too_many_arguments)]

use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::r1cs::constraints::LC;
use jolt_core::zkvm::r1cs::constraints::UNIFORM_R1CS;
use jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs;
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;

use crate::utils::types::Rep3Value;
use crate::zkvm::r1cs::inputs::Rep3R1CSCycleInputs;
use jolt_core::field::JoltField;

/// Sparse interleaved representation of the Stage 1 Spartan outer sumcheck polynomials.
///
/// We represent the three multilinear polynomials `Az`, `Bz`, `Cz` over the row index `x`
/// interleaved in a single sparse vector:
/// - row `k` has coefficients at indices `3k+0`, `3k+1`, `3k+2` (Az/Bz/Cz).
///
/// When binding variables in **LowToHigh** order, sumcheck rounds pair adjacent rows in `k`,
/// so a block of 6 entries (two rows × 3 polys) shares the same `index / 6`.
#[derive(Clone, Debug)]
pub struct Rep3SpartanInterleavedPolynomial<F: JoltField> {
    cycle_inputs: Option<Vec<Rep3R1CSCycleInputs<F>>>,
    pub(crate) bound_coeffs: BoundCoeffs<F>,
    dense_len: usize,
    padded_num_constraints: usize,
}

#[derive(Clone, Debug)]
pub enum BoundCoeffs<F: JoltField> {
    None,
    Sharded(Vec<Vec<SparseCoefficient<Rep3Value<F>>>>),
    Flat(Vec<SparseCoefficient<Rep3Value<F>>>),
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseCoefficient<T> {
    pub index: usize,
    pub value: T,
}

impl<T> From<(usize, T)> for SparseCoefficient<T> {
    fn from(x: (usize, T)) -> Self {
        Self { index: x.0, value: x.1 }
    }
}

impl<F: JoltField> Rep3SpartanInterleavedPolynomial<F> {
    #[tracing::instrument(
        skip_all,
        name = "spartan_stage1_interleaved_new",
        level = "trace",
        fields(
            num_steps = key.num_steps,
            num_constraints = UNIFORM_R1CS.len(),
            padded_num_constraints = key.padded_row_constraint_per_step()
        )
    )]
    pub fn new(key: &UniformSpartanKey<F>, cycle_inputs: Vec<Rep3R1CSCycleInputs<F>>) -> eyre::Result<Self> {
        eyre::ensure!(
            cycle_inputs.len() == key.num_steps,
            "cycle_inputs length mismatch: got {}, expected {}",
            cycle_inputs.len(),
            key.num_steps
        );
        let padded_num_constraints = key.padded_row_constraint_per_step();
        eyre::ensure!(padded_num_constraints >= UNIFORM_R1CS.len(), "padded constraints too small");

        let num_steps = key.num_steps;
        let dense_len = num_steps * padded_num_constraints;
        eyre::ensure!(dense_len.is_power_of_two(), "dense_len must be pow2");

        Ok(Self {
            cycle_inputs: Some(cycle_inputs),
            bound_coeffs: BoundCoeffs::None,
            dense_len,
            padded_num_constraints,
        })
    }

    pub fn is_bound(&self) -> bool {
        !matches!(self.bound_coeffs, BoundCoeffs::None)
    }

    #[tracing::instrument(skip_all, name = "SpartanInterleavedPoly::streaming_sumcheck_round", level = "trace")]
    pub fn streaming_sumcheck_round<Network: Rep3NetworkWorker>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        r: &mut Vec<F::Challenge>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.party_id();
        eyre::ensure!(!self.is_bound(), "expected unbound coefficients");

        let cycle_inputs = self.cycle_inputs.as_ref().ok_or_else(|| eyre::eyre!("missing cycle_inputs for round0"))?;

        let (t0, t_inf) =
            round0_quadratic_evals_from_cycle_inputs(cycle_inputs, eq_poly, party_id, self.padded_num_constraints);
        io_ctx.network().send_response((t0, t_inf))?;

        let r_i: F::Challenge = io_ctx.network().receive_request()?;
        r.push(r_i);
        eq_poly.bind(r_i);

        let bound_shards = round0_build_bound_shards_from_cycle_inputs(
            cycle_inputs,
            party_id,
            r_i.into(),
            self.padded_num_constraints,
        );
        self.cycle_inputs = None;
        self.bound_coeffs = BoundCoeffs::Sharded(bound_shards);
        self.maybe_flatten_bound_shards(eq_poly, party_id);
        self.dense_len /= 2;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "SpartanInterleavedPoly::remaining_sumcheck_round", level = "trace")]
    pub fn remaining_sumcheck_round<Network: Rep3NetworkWorker>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        r: &mut Vec<F::Challenge>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.party_id();
        eyre::ensure!(self.is_bound(), "expected bound coefficients");

        let (t0, t_inf) = match &self.bound_coeffs {
            BoundCoeffs::None => return Err(eyre::eyre!("expected bound coefficients")),
            BoundCoeffs::Sharded(shards) => {
                debug_assert!(eq_poly.E_in_current_len() > 1, "sharded bound coeffs require E_in not fully bound");
                quadratic_evals_from_bound_sharded(shards, eq_poly, party_id)
            }
            BoundCoeffs::Flat(coeffs) => quadratic_evals_from_bound(coeffs, eq_poly, party_id),
        };
        io_ctx.network().send_response((t0, t_inf))?;

        let r_i: F::Challenge = io_ctx.network().receive_request()?;
        r.push(r_i);
        eq_poly.bind(r_i);

        match &mut self.bound_coeffs {
            BoundCoeffs::None => return Err(eyre::eyre!("expected bound coefficients")),
            BoundCoeffs::Sharded(shards) => {
                bind_shards_low_to_high_in_place(shards, party_id, r_i.into());
                self.maybe_flatten_bound_shards(eq_poly, party_id);
            }
            BoundCoeffs::Flat(coeffs) => {
                bind_sparse_coeffs_low_to_high_in_place(coeffs, party_id, r_i.into());
            }
        }
        self.dense_len /= 2;

        Ok(())
    }

    pub fn final_evals_additive(&self, party_id: PartyID) -> [AdditiveShare<F>; 3] {
        debug_assert_eq!(self.dense_len, 1);
        let mut out = [AdditiveShare::<F>::zero(), AdditiveShare::<F>::zero(), AdditiveShare::<F>::zero()];
        match &self.bound_coeffs {
            BoundCoeffs::None => {}
            BoundCoeffs::Sharded(shards) => {
                for shard in shards {
                    for coeff in shard.iter() {
                        let which = coeff.index % 3;
                        out[which] += coeff.value.into_additive(party_id);
                    }
                }
            }
            BoundCoeffs::Flat(coeffs) => {
                for coeff in coeffs.iter() {
                    let which = coeff.index % 3;
                    out[which] += coeff.value.into_additive(party_id);
                }
            }
        }
        out
    }

    fn maybe_flatten_bound_shards(&mut self, eq_poly: &GruenSplitEqPolynomial<F>, party_id: PartyID) {
        if eq_poly.E_in_current_len() > 1 {
            return;
        }

        self.maybe_flatten_bound_shards_force(party_id);
    }

    fn maybe_flatten_bound_shards_force(&mut self, party_id: PartyID) {
        let BoundCoeffs::Sharded(shards) = &mut self.bound_coeffs else {
            return;
        };
        let mut shards = core::mem::take(shards);

        let total_len: usize = shards.iter().map(|v| v.len()).sum();
        let _span =
            tracing::trace_span!("SpartanInterleavedPoly::flatten_bound_shards", shards = shards.len(), total_len)
                .entered();

        let mut flat: Vec<SparseCoefficient<Rep3Value<F>>> = Vec::with_capacity(total_len);
        for shard in shards.iter_mut() {
            flat.append(shard);
        }
        flat.par_sort_unstable_by_key(|coeff| coeff.index);

        let mut write = 0usize;
        for i in 0..flat.len() {
            if write > 0 && flat[write - 1].index == flat[i].index {
                let summed = flat[write - 1].value.add(&flat[i].value, party_id);
                flat[write - 1].value = summed;
            } else {
                if write != i {
                    flat[write] = flat[i];
                }
                write += 1;
            }
        }
        flat.truncate(write);
        self.bound_coeffs = BoundCoeffs::Flat(flat);
        drop(_span);
    }
}

fn round0_quadratic_evals_from_cycle_inputs<F: JoltField>(
    cycle_inputs: &[Rep3R1CSCycleInputs<F>],
    eq_poly: &GruenSplitEqPolynomial<F>,
    party_id: PartyID,
    padded_num_constraints: usize,
) -> (AdditiveShare<F>, AdditiveShare<F>) {
    let e_in_len = eq_poly.E_in_current_len();
    let num_x_in_bits = if e_in_len > 0 { e_in_len.log_2() } else { 0 };
    let x_in_mask = if num_x_in_bits > 0 { (1usize << num_x_in_bits) - 1 } else { 0 };

    let num_steps = cycle_inputs.len();
    let num_constraints = UNIFORM_R1CS.len();
    let num_pairs = padded_num_constraints / 2;

    let num_chunks =
        core::cmp::min(rayon::current_num_threads().next_power_of_two() * 16, core::cmp::max(1, num_steps / 2));
    let chunk_size = num_steps.div_ceil(num_chunks);

    let _span = tracing::trace_span!(
        "SpartanInterleavedPoly::round0_quadratic_evals_from_trace",
        num_steps,
        num_constraints,
        padded_num_constraints,
        num_chunks,
        chunk_size
    )
    .entered();

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let step_start = chunk_idx * chunk_size;
            let step_end = core::cmp::min((chunk_idx + 1) * chunk_size, num_steps);

            let mut t0 = AdditiveShare::<F>::zero();
            let mut t_inf = AdditiveShare::<F>::zero();

            for step_idx in step_start..step_end {
                let inputs = &cycle_inputs[step_idx];
                for pair in 0..num_pairs {
                    let c0_idx = pair * 2;
                    let c1_idx = c0_idx + 1;

                    let (az0, bz0, cz0) = if c0_idx < num_constraints {
                        let named = &UNIFORM_R1CS[c0_idx];
                        (
                            eval_lc_rep3(named.cons.a, inputs, party_id),
                            eval_lc_rep3(named.cons.b, inputs, party_id),
                            eval_lc_rep3(named.cons.c, inputs, party_id),
                        )
                    } else {
                        (Rep3Value::<F>::zero_public(), Rep3Value::<F>::zero_public(), Rep3Value::<F>::zero_public())
                    };

                    let (az1, bz1, _cz1) = if c1_idx < num_constraints {
                        let named = &UNIFORM_R1CS[c1_idx];
                        (
                            eval_lc_rep3(named.cons.a, inputs, party_id),
                            eval_lc_rep3(named.cons.b, inputs, party_id),
                            eval_lc_rep3(named.cons.c, inputs, party_id),
                        )
                    } else {
                        (Rep3Value::<F>::zero_public(), Rep3Value::<F>::zero_public(), Rep3Value::<F>::zero_public())
                    };

                    let block = step_idx * num_pairs + pair;
                    let x_in = block & x_in_mask;
                    let x_out = block >> num_x_in_bits;
                    let weight = if eq_poly.E_in_current_len() == 0 {
                        eq_poly.E_out_current()[block]
                    } else {
                        eq_poly.E_in_current()[x_in] * eq_poly.E_out_current()[x_out]
                    };

                    let az_inf = az1.sub(&az0, party_id);
                    let bz_inf = bz1.sub(&bz0, party_id);

                    let abc0 = az0.mul(&bz0).into_additive(party_id) - cz0.into_additive(party_id);
                    let ab_inf = az_inf.mul(&bz_inf).into_additive(party_id);

                    t0 += abc0 * weight;
                    t_inf += ab_inf * weight;
                }
            }

            (t0, t_inf)
        })
        .reduce(|| (AdditiveShare::<F>::zero(), AdditiveShare::<F>::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
}

fn round0_build_bound_shards_from_cycle_inputs<F: JoltField>(
    cycle_inputs: &[Rep3R1CSCycleInputs<F>],
    party_id: PartyID,
    r: F,
    padded_num_constraints: usize,
) -> Vec<Vec<SparseCoefficient<Rep3Value<F>>>> {
    let num_steps = cycle_inputs.len();
    let num_constraints = UNIFORM_R1CS.len();
    let num_pairs = padded_num_constraints / 2;

    let num_chunks =
        core::cmp::min(rayon::current_num_threads().next_power_of_two() * 16, core::cmp::max(1, num_steps / 2));
    let chunk_size = num_steps.div_ceil(num_chunks);

    let _span = tracing::trace_span!(
        "SpartanInterleavedPoly::round0_build_bound_shards",
        num_steps,
        num_constraints,
        padded_num_constraints,
        num_chunks,
        chunk_size,
        r_is_zero = (r == F::zero())
    )
    .entered();

    let shards: Vec<Vec<SparseCoefficient<Rep3Value<F>>>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let step_start = chunk_idx * chunk_size;
            let step_end = core::cmp::min((chunk_idx + 1) * chunk_size, num_steps);

            let mut out: Vec<SparseCoefficient<Rep3Value<F>>> =
                Vec::with_capacity((step_end - step_start) * num_constraints.saturating_div(2) * 3);

            for step_idx in step_start..step_end {
                let inputs = &cycle_inputs[step_idx];
                for pair in 0..num_pairs {
                    let c0_idx = pair * 2;
                    let c1_idx = c0_idx + 1;

                    let (_az0, _bz0, _cz0, a0_present, b0_present, c0_present) = if c0_idx < num_constraints {
                        let named = &UNIFORM_R1CS[c0_idx];
                        let az0 = eval_lc_rep3(named.cons.a, inputs, party_id);
                        let bz0 = eval_lc_rep3(named.cons.b, inputs, party_id);
                        let cz0 = eval_lc_rep3(named.cons.c, inputs, party_id);
                        (az0, bz0, cz0, az0.shared_or_not_zero(), bz0.shared_or_not_zero(), cz0.shared_or_not_zero())
                    } else {
                        (
                            Rep3Value::<F>::zero_public(),
                            Rep3Value::<F>::zero_public(),
                            Rep3Value::<F>::zero_public(),
                            false,
                            false,
                            false,
                        )
                    };

                    let (_az1, _bz1, _cz1, a1_present, b1_present, c1_present) = if c1_idx < num_constraints {
                        let named = &UNIFORM_R1CS[c1_idx];
                        let az1 = eval_lc_rep3(named.cons.a, inputs, party_id);
                        let bz1 = eval_lc_rep3(named.cons.b, inputs, party_id);
                        let cz1 = eval_lc_rep3(named.cons.c, inputs, party_id);
                        (az1, bz1, cz1, az1.shared_or_not_zero(), bz1.shared_or_not_zero(), cz1.shared_or_not_zero())
                    } else {
                        (
                            Rep3Value::<F>::zero_public(),
                            Rep3Value::<F>::zero_public(),
                            Rep3Value::<F>::zero_public(),
                            false,
                            false,
                            false,
                        )
                    };

                    let block = step_idx * num_pairs + pair;
                    let base = 3 * block;

                    if a0_present || a1_present {
                        let low = eval_lc_rep3_shared(UNIFORM_R1CS[c0_idx].cons.a, inputs, party_id);
                        let high = if c1_idx < num_constraints {
                            eval_lc_rep3_shared(UNIFORM_R1CS[c1_idx].cons.a, inputs, party_id)
                        } else {
                            mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share()
                        };
                        let v = low + mpc_core::protocols::rep3::arithmetic::mul_public(high - low, r);
                        out.push((base, Rep3Value::Shared(v)).into());
                    }
                    if b0_present || b1_present {
                        let low = eval_lc_rep3_shared(UNIFORM_R1CS[c0_idx].cons.b, inputs, party_id);
                        let high = if c1_idx < num_constraints {
                            eval_lc_rep3_shared(UNIFORM_R1CS[c1_idx].cons.b, inputs, party_id)
                        } else {
                            mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share()
                        };
                        let v = low + mpc_core::protocols::rep3::arithmetic::mul_public(high - low, r);
                        out.push((base + 1, Rep3Value::Shared(v)).into());
                    }
                    if c0_present || c1_present {
                        let low = eval_lc_rep3_shared(UNIFORM_R1CS[c0_idx].cons.c, inputs, party_id);
                        let high = if c1_idx < num_constraints {
                            eval_lc_rep3_shared(UNIFORM_R1CS[c1_idx].cons.c, inputs, party_id)
                        } else {
                            mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share()
                        };
                        let v = low + mpc_core::protocols::rep3::arithmetic::mul_public(high - low, r);
                        out.push((base + 2, Rep3Value::Shared(v)).into());
                    }
                }
            }

            out
        })
        .collect();

    drop(_span);
    shards
}

fn quadratic_evals_from_bound<F: JoltField>(
    coeffs: &[SparseCoefficient<Rep3Value<F>>],
    eq_poly: &GruenSplitEqPolynomial<F>,
    party_id: PartyID,
) -> (AdditiveShare<F>, AdditiveShare<F>) {
    let mut t0 = AdditiveShare::<F>::zero();
    let mut t_inf = AdditiveShare::<F>::zero();
    let e_in_len = eq_poly.E_in_current_len();
    let num_x_in_bits = if e_in_len > 1 { e_in_len.log_2() } else { 0 };
    let x_in_mask = if num_x_in_bits > 0 { (1usize << num_x_in_bits) - 1 } else { 0 };

    let mut i = 0;
    while i < coeffs.len() {
        let block = coeffs[i].index / 6;

        let mut az0 = Rep3Value::<F>::zero_public();
        let mut bz0 = Rep3Value::<F>::zero_public();
        let mut cz0 = Rep3Value::<F>::zero_public();
        let mut az1 = Rep3Value::<F>::zero_public();
        let mut bz1 = Rep3Value::<F>::zero_public();
        let mut _cz1 = Rep3Value::<F>::zero_public();

        while i < coeffs.len() && coeffs[i].index / 6 == block {
            match coeffs[i].index % 6 {
                0 => az0 = coeffs[i].value,
                1 => bz0 = coeffs[i].value,
                2 => cz0 = coeffs[i].value,
                3 => az1 = coeffs[i].value,
                4 => bz1 = coeffs[i].value,
                5 => _cz1 = coeffs[i].value,
                _ => unreachable!(),
            }
            i += 1;
        }

        let weight = if e_in_len <= 1 {
            eq_poly.E_out_current()[block]
        } else {
            let x_in = block & x_in_mask;
            let x_out = block >> num_x_in_bits;
            eq_poly.E_in_current()[x_in] * eq_poly.E_out_current()[x_out]
        };

        let az_inf = az1.sub(&az0, party_id);
        let bz_inf = bz1.sub(&bz0, party_id);

        let abc0 = az0.mul(&bz0).into_additive(party_id) - cz0.into_additive(party_id);
        let ab_inf = az_inf.mul(&bz_inf).into_additive(party_id);

        t0 += abc0 * weight;
        t_inf += ab_inf * weight;
    }

    (t0, t_inf)
}

fn quadratic_evals_from_bound_sharded<F: JoltField>(
    shards: &[Vec<SparseCoefficient<Rep3Value<F>>>],
    eq_poly: &GruenSplitEqPolynomial<F>,
    party_id: PartyID,
) -> (AdditiveShare<F>, AdditiveShare<F>) {
    shards
        .par_iter()
        .map(|coeffs| quadratic_evals_from_bound(coeffs, eq_poly, party_id))
        .reduce(|| (AdditiveShare::<F>::zero(), AdditiveShare::<F>::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
}

fn bind_shards_low_to_high_in_place<F: JoltField>(
    shards: &mut [Vec<SparseCoefficient<Rep3Value<F>>>],
    party_id: PartyID,
    r: F,
) {
    let _span = tracing::trace_span!(
        "SpartanInterleavedPoly::bind_shards_in_place",
        shards = shards.len(),
        r_is_zero = (r == F::zero())
    )
    .entered();

    shards.par_iter_mut().for_each(|coeffs| bind_sparse_coeffs_low_to_high_in_place(coeffs, party_id, r));
    drop(_span);
}

fn bind_sparse_coeffs_low_to_high_in_place<F: JoltField>(
    coeffs: &mut Vec<SparseCoefficient<Rep3Value<F>>>,
    party_id: PartyID,
    r: F,
) {
    let _span = tracing::trace_span!(
        "bind_sparse_coeffs_low_to_high_in_place",
        in_len = coeffs.len(),
        r_is_zero = (r == F::zero())
    )
    .entered();

    // We want in-place binding without allocating a new Vec. This requires
    // simultaneously reading (immutable) and writing (mutable) within `coeffs`.
    // Rust borrowing rules don't allow that directly, but it's safe here because:
    // - we only write to indices `< write`, and
    // - we only read from indices `>= i` as `i` monotonically increases,
    // so we never overwrite an element that we will read later.
    let input_ptr = coeffs.as_ptr();
    let input_len = coeffs.len();

    let mut write = 0usize;
    let mut i = 0usize;
    while i < input_len {
        let block = unsafe { (*input_ptr.add(i)).index / 6 };

        let mut a0: Option<Rep3Value<F>> = None;
        let mut b0: Option<Rep3Value<F>> = None;
        let mut c0: Option<Rep3Value<F>> = None;
        let mut a1: Option<Rep3Value<F>> = None;
        let mut b1: Option<Rep3Value<F>> = None;
        let mut c1: Option<Rep3Value<F>> = None;

        while i < input_len {
            let coeff = unsafe { *input_ptr.add(i) };
            if coeff.index / 6 != block {
                break;
            }
            match coeff.index % 6 {
                0 => a0 = Some(coeff.value),
                1 => b0 = Some(coeff.value),
                2 => c0 = Some(coeff.value),
                3 => a1 = Some(coeff.value),
                4 => b1 = Some(coeff.value),
                5 => c1 = Some(coeff.value),
                _ => unreachable!(),
            }
            i += 1;
        }

        let base = 3 * block;

        if a0.is_some() || a1.is_some() {
            let low = a0.unwrap_or_else(Rep3Value::zero_public);
            let high = a1.unwrap_or_else(Rep3Value::zero_public);
            let v = low.add(&high.sub(&low, party_id).mul_public(r), party_id);
            coeffs[write] = (base, v).into();
            write += 1;
        }

        if b0.is_some() || b1.is_some() {
            let low = b0.unwrap_or_else(Rep3Value::zero_public);
            let high = b1.unwrap_or_else(Rep3Value::zero_public);
            let v = low.add(&high.sub(&low, party_id).mul_public(r), party_id);
            coeffs[write] = (base + 1, v).into();
            write += 1;
        }

        if c0.is_some() || c1.is_some() {
            let low = c0.unwrap_or_else(Rep3Value::zero_public);
            let high = c1.unwrap_or_else(Rep3Value::zero_public);
            let v = low.add(&high.sub(&low, party_id).mul_public(r), party_id);
            coeffs[write] = (base + 2, v).into();
            write += 1;
        }
    }

    coeffs.truncate(write);
    drop(_span);
}

fn eval_lc_rep3<F: JoltField>(lc: LC, inputs: &Rep3R1CSCycleInputs<F>, party_id: PartyID) -> Rep3Value<F> {
    let mut acc = Rep3Value::<F>::zero_public();
    lc.for_each_term(|input_index, coeff| {
        let scalar = F::from_i128(coeff.to_i128());
        let val = input_as_value(inputs, JoltR1CSInputs::from_index(input_index), party_id);
        acc.add_assign(&val.mul_public(scalar), party_id);
    });
    if let Some(c) = lc.const_term() {
        let scalar = F::from_i128(c.to_i128());
        acc.add_public_assign(scalar, party_id);
    }
    acc
}

fn eval_lc_rep3_shared<F: JoltField>(
    lc: LC,
    inputs: &Rep3R1CSCycleInputs<F>,
    party_id: PartyID,
) -> mpc_core::protocols::rep3::Rep3PrimeFieldShare<F> {
    let mut acc = mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share();
    lc.for_each_term(|input_index, coeff| {
        let scalar = F::from_i128(coeff.to_i128());
        let val = input_as_value(inputs, JoltR1CSInputs::from_index(input_index), party_id).into_shared_rep3(party_id);
        acc += mpc_core::protocols::rep3::arithmetic::mul_public(val, scalar);
    });
    if let Some(c) = lc.const_term() {
        acc = mpc_core::protocols::rep3::arithmetic::add_public(acc, F::from_i128(c.to_i128()), party_id);
    }
    acc
}

fn input_as_value<F: JoltField>(
    inputs: &Rep3R1CSCycleInputs<F>,
    input: JoltR1CSInputs,
    party_id: PartyID,
) -> Rep3Value<F> {
    match input {
        JoltR1CSInputs::LeftInstructionInput => Rep3Value::Shared(inputs.left_input),
        JoltR1CSInputs::RightInstructionInput => Rep3Value::Shared(inputs.right_input),
        JoltR1CSInputs::Product => Rep3Value::Shared(inputs.product),
        JoltR1CSInputs::LeftLookupOperand => Rep3Value::Shared(inputs.left_lookup),
        JoltR1CSInputs::RightLookupOperand => Rep3Value::Shared(inputs.right_lookup),
        JoltR1CSInputs::LookupOutput => Rep3Value::Shared(inputs.lookup_output),
        JoltR1CSInputs::Rs1Value => Rep3Value::Shared(inputs.rs1_read_value),
        JoltR1CSInputs::Rs2Value => Rep3Value::Shared(inputs.rs2_read_value),
        JoltR1CSInputs::RdWriteValue => Rep3Value::Shared(inputs.rd_write_value),
        JoltR1CSInputs::RamReadValue => Rep3Value::Shared(inputs.ram_read_value),
        JoltR1CSInputs::RamWriteValue => Rep3Value::Shared(inputs.ram_write_value),
        JoltR1CSInputs::ShouldBranch => Rep3Value::Shared(inputs.should_branch),

        JoltR1CSInputs::WriteLookupOutputToRD => {
            let _ = party_id;
            Rep3Value::Public(F::from_u64(inputs.write_lookup_output_to_rd_addr as u64))
        }
        JoltR1CSInputs::WritePCtoRD => {
            let _ = party_id;
            Rep3Value::Public(F::from_u64(inputs.write_pc_to_rd_addr as u64))
        }

        JoltR1CSInputs::PC => Rep3Value::Public(F::from_u64(inputs.pc)),
        JoltR1CSInputs::NextPC => Rep3Value::Public(F::from_u64(inputs.next_pc)),
        JoltR1CSInputs::UnexpandedPC => Rep3Value::Public(F::from_u64(inputs.unexpanded_pc)),
        JoltR1CSInputs::NextUnexpandedPC => Rep3Value::Public(F::from_u64(inputs.next_unexpanded_pc)),
        JoltR1CSInputs::Imm => Rep3Value::Public(F::from_i128(inputs.imm)),
        JoltR1CSInputs::Rd => Rep3Value::Public(F::from_u64(inputs.rd_addr as u64)),
        JoltR1CSInputs::RamAddress => Rep3Value::Public(F::from_u64(inputs.ram_addr)),
        JoltR1CSInputs::NextIsNoop => Rep3Value::Public(F::from_u64(inputs.next_is_noop as u64)),
        JoltR1CSInputs::ShouldJump => Rep3Value::Public(F::from_u64(inputs.should_jump as u64)),
        JoltR1CSInputs::OpFlags(flag) => Rep3Value::Public(F::from_u64(inputs.flags[flag as usize] as u64)),
    }
}
