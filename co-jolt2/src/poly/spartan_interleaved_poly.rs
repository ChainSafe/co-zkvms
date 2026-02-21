#![allow(clippy::too_many_arguments)]

use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::zkvm::r1cs::constraints::{LC, UNIFORM_R1CS};
use jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs;
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
use mpc_core::protocols::rep3::network::{IoContext, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use rayon::prelude::*;
use snarks_core::math::Math;

use crate::field::JoltField;
use crate::zkvm::r1cs::inputs::Rep3R1CSCycleInputs;

/// Dense representation of the Stage 1 Spartan outer sumcheck polynomials.
///
/// Holds evaluations (over `{0,1}^n`) of the multilinear polynomials:
/// - `Az(x)`, `Bz(x)`, `Cz(x)` for the R1CS row index `x`
/// - `eq(tau, x)` weights (public)
///
/// The evaluation ordering is row-major: `row = step_idx * rows_per_step_padded + constraint_idx`.
/// This matches a variable ordering where step variables are the high bits and constraint variables
/// are the low bits, so binding low-to-high corresponds to folding adjacent pairs.
pub struct Rep3SpartanInterleavedPolynomial<F: JoltField> {
    pub a: Vec<Rep3PrimeFieldShare<F>>,
    pub b: Vec<Rep3PrimeFieldShare<F>>,
    pub c: Vec<Rep3PrimeFieldShare<F>>,
    pub eq: Vec<F>,
    pub rows_per_step_padded: usize,
}

impl<F: JoltField> Rep3SpartanInterleavedPolynomial<F> {
    pub fn new(
        party_id: PartyID,
        key: &UniformSpartanKey<F>,
        tau: &[F::Challenge],
        cycle_inputs: &[Rep3R1CSCycleInputs<F>],
    ) -> eyre::Result<Self> {
        let rows_per_step_padded = key.padded_row_constraint_per_step();
        eyre::ensure!(
            cycle_inputs.len() == key.num_steps,
            "cycle_inputs length mismatch: got {}, expected {}",
            cycle_inputs.len(),
            key.num_steps
        );

        let total_rows = key.num_steps * rows_per_step_padded;
        eyre::ensure!(
            total_rows.is_power_of_two(),
            "total rows must be power-of-two"
        );

        let mut a = vec![Rep3PrimeFieldShare::zero_share(); total_rows];
        let mut b = vec![Rep3PrimeFieldShare::zero_share(); total_rows];
        let mut c = vec![Rep3PrimeFieldShare::zero_share(); total_rows];

        // Evaluate constraints per step for the unpadded subset, leave padding rows as zero.
        let num_constraints = UNIFORM_R1CS.len();
        eyre::ensure!(
            num_constraints <= rows_per_step_padded,
            "rows_per_step_padded too small"
        );

        a.par_chunks_mut(rows_per_step_padded)
            .zip(b.par_chunks_mut(rows_per_step_padded))
            .zip(c.par_chunks_mut(rows_per_step_padded))
            .zip(cycle_inputs.par_iter())
            .for_each(|(((a_chunk, b_chunk), c_chunk), inputs)| {
                for (i, named) in UNIFORM_R1CS.iter().enumerate() {
                    let row = &named.cons;
                    a_chunk[i] = eval_lc_rep3(row.a, inputs, party_id);
                    b_chunk[i] = eval_lc_rep3(row.b, inputs, party_id);
                    c_chunk[i] = eval_lc_rep3(row.c, inputs, party_id);
                }
            });

        let eq = EqPolynomial::<F>::evals(tau);
        eyre::ensure!(
            eq.len() == total_rows,
            "eq length mismatch: got {}, expected {}",
            eq.len(),
            total_rows
        );

        Ok(Self {
            a,
            b,
            c,
            eq,
            rows_per_step_padded,
        })
    }

    /// Compute `(t(0), t(∞))` as additive shares for the current sumcheck round, where:
    /// - `t(0) = Σ_rest eq_rest * (A(0)*B(0) - C(0))`
    /// - `t(∞) = Σ_rest eq_rest * (ΔA * ΔB)` (quadratic coefficient)
    pub fn quadratic_evals<N: Rep3NetworkWorker>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<(AdditiveShare<F>, AdditiveShare<F>)> {
        debug_assert_eq!(self.a.len(), self.b.len());
        debug_assert_eq!(self.a.len(), self.c.len());
        debug_assert_eq!(self.a.len(), self.eq.len());
        debug_assert!(self.a.len().is_power_of_two());

        let half = self.a.len() / 2;
        let mut a0_vec = Vec::with_capacity(half);
        let mut b0_vec = Vec::with_capacity(half);
        let mut da_vec = Vec::with_capacity(half);
        let mut db_vec = Vec::with_capacity(half);

        for j in 0..half {
            let idx0 = 2 * j;
            let idx1 = idx0 + 1;

            let a0 = self.a[idx0];
            let b0 = self.b[idx0];
            a0_vec.push(a0);
            b0_vec.push(b0);

            da_vec.push(self.a[idx1] - a0);
            db_vec.push(self.b[idx1] - b0);
        }

        // Multiplications are secret-shared: use MPC (batched) to obtain replicated shares.
        let a0b0 = rep3_arithmetic::mul_vec_par(&a0_vec, &b0_vec, io_ctx)?;
        let d_ad_b = rep3_arithmetic::mul_vec_par(&da_vec, &db_vec, io_ctx)?;

        let mut t0 = AdditiveShare::<F>::zero();
        let mut t_inf = AdditiveShare::<F>::zero();
        for j in 0..half {
            let idx0 = 2 * j;
            let idx1 = idx0 + 1;
            let eq_rest = self.eq[idx0] + self.eq[idx1];

            let p0 = a0b0[j].into_additive() - self.c[idx0].into_additive();
            t0 += p0 * eq_rest;

            t_inf += d_ad_b[j].into_additive() * eq_rest;
        }

        Ok((t0, t_inf))
    }

    /// Bind the current sumcheck variable to `r`, folding `(a,b,c,eq)` in place.
    pub fn bind(&mut self, party_id: PartyID, r: F::Challenge) {
        let r_f: F = r.into();
        let half = self.a.len() / 2;
        let mut next_a = Vec::with_capacity(half);
        let mut next_b = Vec::with_capacity(half);
        let mut next_c = Vec::with_capacity(half);
        let mut next_eq = Vec::with_capacity(half);

        for j in 0..half {
            let idx0 = 2 * j;
            let idx1 = idx0 + 1;

            next_a.push(bind_pair_rep3(
                self.a[idx0],
                self.a[idx1],
                party_id,
                r_f,
            ));
            next_b.push(bind_pair_rep3(
                self.b[idx0],
                self.b[idx1],
                party_id,
                r_f,
            ));
            next_c.push(bind_pair_rep3(
                self.c[idx0],
                self.c[idx1],
                party_id,
                r_f,
            ));
            next_eq.push(bind_pair_public(self.eq[idx0], self.eq[idx1], r_f));
        }

        self.a = next_a;
        self.b = next_b;
        self.c = next_c;
        self.eq = next_eq;
    }

    pub fn final_evals_additive(&self) -> [AdditiveShare<F>; 3] {
        debug_assert_eq!(self.a.len(), 1);
        [
            self.a[0].into_additive(),
            self.b[0].into_additive(),
            self.c[0].into_additive(),
        ]
    }
}

fn bind_pair_public<F: JoltField>(low: F, high: F, r: F) -> F {
    low + (high - low) * r
}

fn bind_pair_rep3<F: JoltField>(
    low: Rep3PrimeFieldShare<F>,
    high: Rep3PrimeFieldShare<F>,
    party_id: PartyID,
    r: F,
) -> Rep3PrimeFieldShare<F> {
    low + rep3_arithmetic::mul_public(high - low, r)
}

fn eval_lc_rep3<F: JoltField>(
    lc: LC,
    inputs: &Rep3R1CSCycleInputs<F>,
    party_id: PartyID,
) -> Rep3PrimeFieldShare<F> {
    let mut acc = Rep3PrimeFieldShare::<F>::zero_share();
    lc.for_each_term(|input_index, coeff| {
        let scalar = F::from_i128(coeff.to_i128());
        let val = input_as_share::<F>(inputs, JoltR1CSInputs::from_index(input_index), party_id);
        acc += rep3_arithmetic::mul_public(val, scalar);
    });
    if let Some(c) = lc.const_term() {
        let scalar = F::from_i128(c.to_i128());
        acc = rep3_arithmetic::add_public(acc, scalar, party_id);
    }
    acc
}

fn input_as_share<F: JoltField>(
    inputs: &Rep3R1CSCycleInputs<F>,
    input: JoltR1CSInputs,
    party_id: PartyID,
) -> Rep3PrimeFieldShare<F> {
    use mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share;

    match input {
        JoltR1CSInputs::LeftInstructionInput => inputs.left_input,
        JoltR1CSInputs::RightInstructionInput => inputs.right_input,
        JoltR1CSInputs::Product => inputs.product,
        JoltR1CSInputs::LeftLookupOperand => inputs.left_lookup,
        JoltR1CSInputs::RightLookupOperand => inputs.right_lookup,
        JoltR1CSInputs::LookupOutput => inputs.lookup_output,
        JoltR1CSInputs::Rs1Value => inputs.rs1_read_value,
        JoltR1CSInputs::Rs2Value => inputs.rs2_read_value,
        JoltR1CSInputs::RdWriteValue => inputs.rd_write_value,
        JoltR1CSInputs::RamReadValue => inputs.ram_read_value,
        JoltR1CSInputs::RamWriteValue => inputs.ram_write_value,
        JoltR1CSInputs::ShouldBranch => inputs.should_branch,

        JoltR1CSInputs::WriteLookupOutputToRD => promote_to_trivial_share(
            party_id,
            F::from_u64(inputs.write_lookup_output_to_rd_addr as u64),
        ),
        JoltR1CSInputs::WritePCtoRD => {
            promote_to_trivial_share(party_id, F::from_u64(inputs.write_pc_to_rd_addr as u64))
        }
        JoltR1CSInputs::PC => promote_to_trivial_share(party_id, F::from_u64(inputs.pc)),
        JoltR1CSInputs::UnexpandedPC => {
            promote_to_trivial_share(party_id, F::from_u64(inputs.unexpanded_pc))
        }
        JoltR1CSInputs::Rd => promote_to_trivial_share(party_id, F::from_u64(inputs.rd_addr as u64)),
        JoltR1CSInputs::Imm => promote_to_trivial_share(party_id, F::from_i128(inputs.imm)),
        JoltR1CSInputs::RamAddress => {
            promote_to_trivial_share(party_id, F::from_u64(inputs.ram_addr))
        }
        JoltR1CSInputs::NextUnexpandedPC => {
            promote_to_trivial_share(party_id, F::from_u64(inputs.next_unexpanded_pc))
        }
        JoltR1CSInputs::NextPC => promote_to_trivial_share(party_id, F::from_u64(inputs.next_pc)),
        JoltR1CSInputs::NextIsNoop => {
            promote_to_trivial_share(party_id, F::from_bool(inputs.next_is_noop))
        }
        JoltR1CSInputs::ShouldJump => {
            promote_to_trivial_share(party_id, F::from_bool(inputs.should_jump))
        }
        JoltR1CSInputs::OpFlags(flag) => {
            promote_to_trivial_share(party_id, F::from_bool(inputs.flags[flag as usize]))
        }
    }
}
