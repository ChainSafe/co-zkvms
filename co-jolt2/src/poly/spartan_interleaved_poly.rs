#![allow(clippy::too_many_arguments)]

use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::r1cs::constraints::LC;
use jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs;
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::utils::types::Rep3Value;
use crate::zkvm::r1cs::inputs::Rep3R1CSCycleInputs;

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
    pub(crate) unbound_coeffs_shards: Vec<Vec<SparseCoefficient<Rep3Value<F>>>>,
    pub(crate) bound_coeffs: Vec<SparseCoefficient<Rep3Value<F>>>,
    dense_len: usize,
    padded_num_constraints: usize,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseCoefficient<T> {
    pub index: usize,
    pub value: T,
}

impl<T> From<(usize, T)> for SparseCoefficient<T> {
    fn from(x: (usize, T)) -> Self {
        Self {
            index: x.0,
            value: x.1,
        }
    }
}

impl<F: JoltField> Rep3SpartanInterleavedPolynomial<F> {
    #[tracing::instrument(
        skip_all,
        name = "spartan_stage1_interleaved_new",
        level = "trace",
        fields(
            num_steps = key.num_steps,
            num_constraints = uniform_constraints.len(),
            padded_num_constraints = key.padded_row_constraint_per_step()
        )
    )]
    pub fn new(
        key: &UniformSpartanKey<F>,
        cycle_inputs: &[Rep3R1CSCycleInputs<F>],
        uniform_constraints: &[jolt_core::zkvm::r1cs::constraints::NamedConstraint],
        party_id: PartyID,
    ) -> eyre::Result<Self> {
        eyre::ensure!(
            cycle_inputs.len() == key.num_steps,
            "cycle_inputs length mismatch: got {}, expected {}",
            cycle_inputs.len(),
            key.num_steps
        );
        let padded_num_constraints = key.padded_row_constraint_per_step();
        eyre::ensure!(
            padded_num_constraints >= uniform_constraints.len(),
            "padded constraints too small"
        );

        let num_steps = key.num_steps;
        let dense_len = num_steps * padded_num_constraints;
        eyre::ensure!(dense_len.is_power_of_two(), "dense_len must be pow2");

        // Chunk by steps; each shard produces sparse (Az,Bz,Cz) coefficients.
        let num_chunks = core::cmp::min(
            rayon::current_num_threads().next_power_of_two() * 16,
            core::cmp::max(1, num_steps / 2),
        );
        let chunk_size = num_steps.div_ceil(num_chunks);

        let _build_span =
            tracing::trace_span!("SpartanInterleavedPoly::build_shards", num_chunks, chunk_size)
                .entered();
        let shards: Vec<Vec<SparseCoefficient<Rep3Value<F>>>> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = core::cmp::min((chunk_idx + 1) * chunk_size, num_steps);
                let mut coeffs: Vec<SparseCoefficient<Rep3Value<F>>> =
                    Vec::with_capacity((end - start) * uniform_constraints.len() * 3);

                for step_idx in start..end {
                    let inputs = &cycle_inputs[step_idx];
                    for (constraint_idx, named) in uniform_constraints.iter().enumerate() {
                        let row_index = step_idx * padded_num_constraints + constraint_idx;
                        let base = 3 * row_index;

                        let az = eval_lc_rep3(named.cons.a, inputs, party_id);
                        if az.shared_or_not_zero() {
                            coeffs.push((base, az).into());
                        }

                        let bz = eval_lc_rep3(named.cons.b, inputs, party_id);
                        if bz.shared_or_not_zero() {
                            coeffs.push((base + 1, bz).into());
                        }

                        let cz = eval_lc_rep3(named.cons.c, inputs, party_id);
                        if cz.shared_or_not_zero() {
                            coeffs.push((base + 2, cz).into());
                        }
                    }
                }

                coeffs
            })
            .collect();
        drop(_build_span);

        Ok(Self {
            unbound_coeffs_shards: shards,
            bound_coeffs: vec![],
            dense_len,
            padded_num_constraints,
        })
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPoly::streaming_sumcheck_round",
        level = "trace"
    )]
    pub fn streaming_sumcheck_round<Network: Rep3NetworkWorker>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        r: &mut Vec<F::Challenge>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.party_id();
        eyre::ensure!(!self.is_bound(), "expected unbound coefficients");

        let (t0, t_inf) = quadratic_evals_from_unbound(
            &self.unbound_coeffs_shards,
            eq_poly,
            party_id,
            self.padded_num_constraints,
        );
        io_ctx.network().send_response((t0, t_inf))?;

        let r_i: F::Challenge = io_ctx.network().receive_request()?;
        r.push(r_i);
        eq_poly.bind(r_i);

        self.bound_coeffs =
            bind_sparse_shards_into_bound(&self.unbound_coeffs_shards, party_id, r_i.into());
        self.unbound_coeffs_shards.clear();
        self.unbound_coeffs_shards.shrink_to_fit();
        self.dense_len /= 2;

        Ok(())
    }

    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPoly::remaining_sumcheck_round",
        level = "trace"
    )]
    pub fn remaining_sumcheck_round<Network: Rep3NetworkWorker>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        r: &mut Vec<F::Challenge>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.party_id();
        eyre::ensure!(self.is_bound(), "expected bound coefficients");

        let (t0, t_inf) = quadratic_evals_from_bound(&self.bound_coeffs, eq_poly, party_id);
        io_ctx.network().send_response((t0, t_inf))?;

        let r_i: F::Challenge = io_ctx.network().receive_request()?;
        r.push(r_i);
        eq_poly.bind(r_i);

        bind_sparse_coeffs_low_to_high_in_place(&mut self.bound_coeffs, party_id, r_i.into());
        self.dense_len /= 2;

        Ok(())
    }

    pub fn final_evals_additive(&self, party_id: PartyID) -> [AdditiveShare<F>; 3] {
        debug_assert_eq!(self.dense_len, 1);
        let mut out = [
            AdditiveShare::<F>::zero(),
            AdditiveShare::<F>::zero(),
            AdditiveShare::<F>::zero(),
        ];
        for coeff in self.bound_coeffs.iter() {
            let which = coeff.index % 3;
            out[which] = coeff.value.into_additive(party_id);
        }
        out
    }
}

fn quadratic_evals_from_unbound<F: JoltField>(
    shards: &[Vec<SparseCoefficient<Rep3Value<F>>>],
    eq_poly: &GruenSplitEqPolynomial<F>,
    party_id: PartyID,
    _padded_num_constraints: usize,
) -> (AdditiveShare<F>, AdditiveShare<F>) {
    let e_in_len = eq_poly.E_in_current_len();
    let num_x_in_bits = if e_in_len > 0 { e_in_len.log_2() } else { 0 };
    let x_in_mask = if num_x_in_bits > 0 {
        (1usize << num_x_in_bits) - 1
    } else {
        0
    };

    shards
        .par_iter()
        .map(|coeffs| {
            let mut t0 = AdditiveShare::<F>::zero();
            let mut t_inf = AdditiveShare::<F>::zero();

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

            (t0, t_inf)
        })
        .reduce(
            || (AdditiveShare::<F>::zero(), AdditiveShare::<F>::zero()),
            |a, b| (a.0 + b.0, a.1 + b.1),
        )
}

fn quadratic_evals_from_bound<F: JoltField>(
    coeffs: &[SparseCoefficient<Rep3Value<F>>],
    eq_poly: &GruenSplitEqPolynomial<F>,
    party_id: PartyID,
) -> (AdditiveShare<F>, AdditiveShare<F>) {
    let e_in_len = eq_poly.E_in_current_len();
    let num_x_in_bits = if e_in_len > 0 { e_in_len.log_2() } else { 0 };
    let x_in_mask = if num_x_in_bits > 0 {
        (1usize << num_x_in_bits) - 1
    } else {
        0
    };

    let mut t0 = AdditiveShare::<F>::zero();
    let mut t_inf = AdditiveShare::<F>::zero();

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

    (t0, t_inf)
}

fn bind_sparse_shards_into_bound<F: JoltField>(
    shards: &[Vec<SparseCoefficient<Rep3Value<F>>>],
    party_id: PartyID,
    r: F,
) -> Vec<SparseCoefficient<Rep3Value<F>>> {
    let _span = tracing::trace_span!(
        "bind_sparse_shards_into_bound",
        shards = shards.len(),
        r_is_zero = (r == F::zero())
    )
    .entered();

    // Compute total output length so we can preallocate.
    let output_lens: Vec<usize> = shards
        .par_iter()
        .map(|coeffs| binding_output_length(coeffs))
        .collect();
    let total_len: usize = output_lens.iter().sum();
    let mut out: Vec<SparseCoefficient<Rep3Value<F>>> = Vec::with_capacity(total_len);

    for (coeffs, expected_len) in shards.iter().zip(output_lens.into_iter()) {
        let before = out.len();
        bind_sparse_coeffs_low_to_high_into(coeffs, party_id, r, &mut out);
        debug_assert_eq!(out.len() - before, expected_len);
    }

    out
}

fn binding_output_length<F: JoltField>(coeffs: &[SparseCoefficient<Rep3Value<F>>]) -> usize {
    let mut out = 0usize;
    let mut i = 0;
    while i < coeffs.len() {
        let block = coeffs[i].index / 6;
        let mut has_a = false;
        let mut has_b = false;
        let mut has_c = false;
        while i < coeffs.len() && coeffs[i].index / 6 == block {
            match coeffs[i].index % 6 {
                0 | 3 => has_a = true,
                1 | 4 => has_b = true,
                2 | 5 => has_c = true,
                _ => unreachable!(),
            }
            i += 1;
        }
        out += has_a as usize + has_b as usize + has_c as usize;
    }
    out
}

fn bind_sparse_coeffs_low_to_high_into<F: JoltField>(
    coeffs: &[SparseCoefficient<Rep3Value<F>>],
    party_id: PartyID,
    r: F,
    out: &mut Vec<SparseCoefficient<Rep3Value<F>>>,
) {
    let _span = tracing::trace_span!(
        "bind_sparse_coeffs_low_to_high_into",
        in_len = coeffs.len(),
        r_is_zero = (r == F::zero())
    )
    .entered();

    out.reserve(binding_output_length(coeffs));

    bind_sparse_coeffs_low_to_high_visit(coeffs, party_id, r, |index, value| {
        out.push((index, value).into());
    });
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

fn bind_sparse_coeffs_low_to_high_visit<F: JoltField>(
    coeffs: &[SparseCoefficient<Rep3Value<F>>],
    party_id: PartyID,
    r: F,
    mut emit: impl FnMut(usize, Rep3Value<F>),
) {
    let mut i = 0usize;
    while i < coeffs.len() {
        let block = coeffs[i].index / 6;

        let mut a0: Option<Rep3Value<F>> = None;
        let mut b0: Option<Rep3Value<F>> = None;
        let mut c0: Option<Rep3Value<F>> = None;
        let mut a1: Option<Rep3Value<F>> = None;
        let mut b1: Option<Rep3Value<F>> = None;
        let mut c1: Option<Rep3Value<F>> = None;

        while i < coeffs.len() && coeffs[i].index / 6 == block {
            match coeffs[i].index % 6 {
                0 => a0 = Some(coeffs[i].value),
                1 => b0 = Some(coeffs[i].value),
                2 => c0 = Some(coeffs[i].value),
                3 => a1 = Some(coeffs[i].value),
                4 => b1 = Some(coeffs[i].value),
                5 => c1 = Some(coeffs[i].value),
                _ => unreachable!(),
            }
            i += 1;
        }

        let base = 3 * block;

        if a0.is_some() || a1.is_some() {
            let low = a0.unwrap_or_else(Rep3Value::zero_public);
            let high = a1.unwrap_or_else(Rep3Value::zero_public);
            let v = low.add(&high.sub(&low, party_id).mul_public(r), party_id);
            emit(base, v);
        }

        if b0.is_some() || b1.is_some() {
            let low = b0.unwrap_or_else(Rep3Value::zero_public);
            let high = b1.unwrap_or_else(Rep3Value::zero_public);
            let v = low.add(&high.sub(&low, party_id).mul_public(r), party_id);
            emit(base + 1, v);
        }

        if c0.is_some() || c1.is_some() {
            let low = c0.unwrap_or_else(Rep3Value::zero_public);
            let high = c1.unwrap_or_else(Rep3Value::zero_public);
            let v = low.add(&high.sub(&low, party_id).mul_public(r), party_id);
            emit(base + 2, v);
        }
    }
}

fn eval_lc_rep3<F: JoltField>(
    lc: LC,
    inputs: &Rep3R1CSCycleInputs<F>,
    party_id: PartyID,
) -> Rep3Value<F> {
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
        JoltR1CSInputs::NextUnexpandedPC => {
            Rep3Value::Public(F::from_u64(inputs.next_unexpanded_pc))
        }
        JoltR1CSInputs::Imm => Rep3Value::Public(F::from_i128(inputs.imm)),
        JoltR1CSInputs::Rd => Rep3Value::Public(F::from_u64(inputs.rd_addr as u64)),
        JoltR1CSInputs::RamAddress => Rep3Value::Public(F::from_u64(inputs.ram_addr)),
        JoltR1CSInputs::NextIsNoop => Rep3Value::Public(F::from_u64(inputs.next_is_noop as u64)),
        JoltR1CSInputs::ShouldJump => Rep3Value::Public(F::from_u64(inputs.should_jump as u64)),
        JoltR1CSInputs::OpFlags(flag) => {
            Rep3Value::Public(F::from_u64(inputs.flags[flag as usize] as u64))
        }
    }
}
