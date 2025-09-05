use std::{
    collections::BTreeMap,
    marker::PhantomData,
    sync::atomic::{AtomicBool, Ordering},
};

use jolt_core::r1cs::{
    builder::{Constraint, OffsetEqConstraint},
    inputs::ConstraintInput,
    key::{
        CrossStepR1CS, CrossStepR1CSConstraint, SparseConstraints, SparseEqualityItem, UniformR1CS,
        UniformSpartanKey,
    },
    ops::{Term, Variable, LC},
};
use mpc_core::protocols::rep3::{
    network::{IoContextPool, Rep3NetworkWorker},
    PartyID, Rep3PrimeFieldShare,
};

use crate::{
    field::JoltField,
    jolt::vm::witness::Rep3JoltPolynomials,
    poly::{spartan_interleaved_poly::Rep3SpartanInterleavedPolynomial, Rep3MultilinearPolynomial},
    utils::{
        future::{FutureExt, FutureRep3},
        types::Rep3Value,
    },
};

use rayon::prelude::*;

type AuxComputationFunction<F: JoltField> =
    dyn Fn(&[Rep3Value<F>], PartyID) -> FutureRep3<F, Rep3PrimeFieldShare<F>> + Send + Sync;

pub struct AuxComputation<F: JoltField> {
    pub symbolic_inputs: Vec<LC>,
    pub compute: Box<AuxComputationFunction<F>>,
    _field: PhantomData<F>,
}

impl<F: JoltField> AuxComputation<F> {
    fn new(symbolic_inputs: Vec<LC>, compute: Box<AuxComputationFunction<F>>) -> Self {
        Self {
            symbolic_inputs,
            compute,
            _field: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, level = "trace")]
    fn compute_aux_poly_fut<const C: usize, I: ConstraintInput>(
        &self,
        jolt_polynomials: &Rep3JoltPolynomials<F>,
        poly_len: usize,
        party_id: PartyID,
    ) -> Vec<FutureRep3<F, Rep3PrimeFieldShare<F>>> {
        let flattened_polys: Vec<&Rep3MultilinearPolynomial<F>> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(jolt_polynomials))
            .collect();

        let mut aux_poly_fut = vec![FutureRep3::Ready(Rep3PrimeFieldShare::zero_share()); poly_len];
        let num_threads = rayon::current_num_threads();
        let chunk_size = poly_len.div_ceil(num_threads);

        aux_poly_fut
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                chunk.iter_mut().enumerate().for_each(|(offset, aux_val)| {
                    let global_index = chunk_index * chunk_size + offset;
                    let compute_inputs: Vec<_> = self
                        .symbolic_inputs
                        .iter()
                        .map(|lc| {
                            let mut input = Rep3Value::zero_public();
                            for term in lc.terms().iter() {
                                match term.0 {
                                    Variable::Input(index) | Variable::Auxiliary(index) => {
                                        input.add_assign(
                                            &flattened_polys[index]
                                                .get_coeff(global_index)
                                                .mul_public(F::from(term.1)),
                                            party_id,
                                        );
                                    }
                                    Variable::Constant => {
                                        input.add_public_assign(F::from(term.1), party_id)
                                    }
                                }
                            }
                            input
                        })
                        .collect();
                    *aux_val = (self.compute)(&compute_inputs, party_id);
                });
            });

        aux_poly_fut
    }
}

pub struct R1CSBuilder<const C: usize, F: JoltField, I: ConstraintInput> {
    _inputs: PhantomData<I>,
    pub constraints: Vec<Constraint>,
    pub aux_computations: BTreeMap<usize, AuxComputation<F>>,
}

impl<const C: usize, F: JoltField, I: ConstraintInput> Default for R1CSBuilder<C, F, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize, F: JoltField, I: ConstraintInput> R1CSBuilder<C, F, I> {
    pub fn new() -> Self {
        Self {
            _inputs: PhantomData,
            constraints: vec![],
            aux_computations: BTreeMap::new(),
        }
    }

    fn allocate_aux(
        &mut self,
        aux_symbol: I,
        symbolic_inputs: Vec<LC>,
        compute: Box<AuxComputationFunction<F>>,
    ) -> Variable {
        let aux_index = aux_symbol.to_index::<C>();
        let new_aux = Variable::Auxiliary(aux_index);
        let computation = AuxComputation::new(symbolic_inputs, compute);
        self.aux_computations.insert(aux_index, computation);

        new_aux
    }

    pub fn constrain_eq(&mut self, left: impl Into<LC>, right: impl Into<LC>) {
        // left - right == 0
        let left: LC = left.into();
        let right: LC = right.into();

        let a = left - right.clone();
        let b = Variable::Constant.into();
        let constraint = Constraint {
            a,
            b,
            c: LC::zero(),
        };
        self.constraints.push(constraint);
    }

    pub fn constrain_eq_conditional(
        &mut self,
        condition: impl Into<LC>,
        left: impl Into<LC>,
        right: impl Into<LC>,
    ) {
        // condition  * (left - right) == 0
        let condition: LC = condition.into();
        let left: LC = left.into();
        let right: LC = right.into();

        let a = condition;
        let b = left - right;
        let c = LC::zero();
        let constraint = Constraint { a, b, c }; // TODO(sragss): Can do better on middle term.
        self.constraints.push(constraint);
    }

    pub fn constrain_binary(&mut self, value: impl Into<LC>) {
        let one: LC = Variable::Constant.into();
        let a: LC = value.into();
        let b = one - a.clone();
        // value * (1 - value) == 0
        let constraint = Constraint {
            a,
            b,
            c: LC::zero(),
        };
        self.constraints.push(constraint);
    }

    pub fn constrain_if_else(
        &mut self,
        condition: impl Into<LC>,
        result_true: impl Into<LC>,
        result_false: impl Into<LC>,
        alleged_result: impl Into<LC>,
    ) {
        let condition: LC = condition.into();
        let result_true: LC = result_true.into();
        let result_false: LC = result_false.into();
        let alleged_result: LC = alleged_result.into();

        // result == condition * true_coutcome + (1 - condition) * false_outcome
        // simplify to single mul, single constraint => condition * (true_outcome - false_outcome) == (result - false_outcome)

        let constraint = Constraint {
            a: condition.clone(),
            b: (result_true - result_false.clone()),
            c: (alleged_result - result_false),
        };
        self.constraints.push(constraint);
    }

    #[must_use]
    pub fn allocate_if_else(
        &mut self,
        aux_symbol: I,
        condition: impl Into<LC>,
        result_true: impl Into<LC>,
        result_false: impl Into<LC>,
    ) -> Variable {
        let (condition, result_true, result_false) =
            (condition.into(), result_true.into(), result_false.into());

        let aux_var = self.aux_if_else(aux_symbol, &condition, &result_true, &result_false);

        self.constrain_if_else(condition, result_true, result_false, aux_var);
        aux_var
    }

    fn aux_if_else(
        &mut self,
        aux_symbol: I,
        condition: &LC,
        result_true: &LC,
        result_false: &LC,
    ) -> Variable {
        // aux = (condition == 1) ? result_true : result_false;
        let if_else =
            |values: &[Rep3Value<F>], party_id: PartyID| -> FutureRep3<F, Rep3PrimeFieldShare<F>> {
                assert_eq!(values.len(), 3);
                let condition = values[0];
                let result_true = values[1].into_rep3_local(party_id);
                let result_false = values[2].into_rep3_local(party_id);

                match condition {
                    Rep3Value::Public(cond) => {
                        if cond.is_one() {
                            FutureRep3::Ready(result_true)
                        } else {
                            FutureRep3::Ready(result_false)
                        }
                    }
                    Rep3Value::Shared(cond) => FutureRep3::cmux(cond, result_true, result_false),
                    _ => unreachable!(),
                }
            };

        let symbolic_inputs = vec![condition.clone(), result_true.clone(), result_false.clone()];
        let compute = Box::new(if_else);
        self.allocate_aux(aux_symbol, symbolic_inputs, compute)
    }

    pub fn constrain_pack_le(
        &mut self,
        unpacked: Vec<Variable>,
        result: impl Into<LC>,
        operand_bits: usize,
    ) {
        // Pack unpacked via a simple weighted linear combination
        // A + 2 * B + 4 * C + 8 * D, ...
        let packed: Vec<Term> = unpacked
            .into_iter()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        self.constrain_eq(packed, result);
    }

    pub fn constrain_pack_be(
        &mut self,
        unpacked: Vec<Variable>,
        result: impl Into<LC>,
        operand_bits: usize,
    ) {
        // Pack unpacked via a simple weighted linear combination
        // A + 2 * B + 4 * C + 8 * D, ...
        // Note: Packing order is reversed from constrain_pack_le
        let packed: Vec<Term> = unpacked
            .into_iter()
            .rev()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        self.constrain_eq(packed, result);
    }

    /// Constrain x * y == z
    pub fn constrain_prod(&mut self, x: impl Into<LC>, y: impl Into<LC>, z: impl Into<LC>) {
        let constraint = Constraint {
            a: x.into(),
            b: y.into(),
            c: z.into(),
        };
        self.constraints.push(constraint);
    }

    #[must_use]
    pub fn allocate_prod(&mut self, aux_symbol: I, x: impl Into<LC>, y: impl Into<LC>) -> Variable {
        let (x, y) = (x.into(), y.into());
        let z = self.aux_prod(aux_symbol, &x, &y);

        self.constrain_prod(x, y, z);
        z
    }

    fn aux_prod(&mut self, aux_symbol: I, x: &LC, y: &LC) -> Variable {
        let prod = |values: &[Rep3Value<F>], party_id| -> FutureRep3<F, Rep3PrimeFieldShare<F>> {
            assert_eq!(values.len(), 2);
            match (values[0], values[1]) {
                (Rep3Value::Shared(x), Rep3Value::Shared(y)) => FutureRep3::mul(x, y),
                _ => FutureRep3::Ready(values[0].mul(&values[1]).into_rep3_local(party_id)),
            }
        };

        let symbolic_inputs = vec![x.clone(), y.clone()];
        let compute = Box::new(prod);
        self.allocate_aux(aux_symbol, symbolic_inputs, compute)
    }

    fn materialize(&self) -> UniformR1CS<F> {
        let a_len: usize = self.constraints.iter().map(|c| c.a.num_vars()).sum();
        let b_len: usize = self.constraints.iter().map(|c| c.b.num_vars()).sum();
        let c_len: usize = self.constraints.iter().map(|c| c.c.num_vars()).sum();
        let mut a_sparse = SparseConstraints::empty_with_capacity(a_len, self.constraints.len());
        let mut b_sparse = SparseConstraints::empty_with_capacity(b_len, self.constraints.len());
        let mut c_sparse = SparseConstraints::empty_with_capacity(c_len, self.constraints.len());

        let update_sparse = |row_index: usize, lc: &LC, sparse: &mut SparseConstraints<F>| {
            lc.terms().iter().for_each(|term| {
                match term.0 {
                    Variable::Input(inner) | Variable::Auxiliary(inner) => {
                        sparse.vars.push((row_index, inner, F::from_i64(term.1)))
                    }
                    Variable::Constant => {}
                };
            });
            if let Some(term) = lc.constant_term() {
                sparse.consts.push((row_index, F::from_i64(term.1)));
            }
        };

        for (row_index, constraint) in self.constraints.iter().enumerate() {
            update_sparse(row_index, &constraint.a, &mut a_sparse);
            update_sparse(row_index, &constraint.b, &mut b_sparse);
            update_sparse(row_index, &constraint.c, &mut c_sparse);
        }

        assert_eq!(a_sparse.vars.len(), a_len);
        assert_eq!(b_sparse.vars.len(), b_len);
        assert_eq!(c_sparse.vars.len(), c_len);

        UniformR1CS::<F> {
            a: a_sparse,
            b: b_sparse,
            c: c_sparse,
            num_vars: I::num_inputs::<C>(),
            num_rows: self.constraints.len(),
        }
    }

    pub fn get_constraints(&self) -> Vec<Constraint> {
        self.constraints.clone()
    }

    pub fn pack_be(unpacked: Vec<Variable>, operand_bits: usize) -> LC {
        let packed: Vec<Term> = unpacked
            .into_iter()
            .rev()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        packed.into()
    }
}

pub struct CombinedUniformBuilder<const C: usize, F: JoltField, I: ConstraintInput> {
    pub uniform_builder: R1CSBuilder<C, F, I>,

    /// Padded to the nearest power of 2
    pub uniform_repeat: usize, // TODO: Remove padding of steps

    pub offset_equality_constraints: Vec<OffsetEqConstraint>,
}

impl<const C: usize, F: JoltField, I: ConstraintInput> CombinedUniformBuilder<C, F, I> {
    pub fn construct(
        uniform_builder: R1CSBuilder<C, F, I>,
        uniform_repeat: usize,
        offset_equality_constraints: Vec<OffsetEqConstraint>,
    ) -> Self {
        assert!(uniform_repeat.is_power_of_two());
        Self {
            uniform_builder,
            uniform_repeat,
            offset_equality_constraints,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn compute_aux<N: Rep3NetworkWorker>(
        &self,
        jolt_polynomials: &mut Rep3JoltPolynomials<F>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<()> {
        let flattened_vars = I::flatten::<C>();
        for (aux_index, aux_compute) in self.uniform_builder.aux_computations.iter() {
            let coeffs: Vec<Rep3PrimeFieldShare<F>> = aux_compute
                .compute_aux_poly_fut::<C, I>(
                    jolt_polynomials,
                    self.uniform_repeat,
                    io_ctx.party_id(),
                )
                .fulfill_batched(io_ctx.main(), |r, _| r)?;
            *flattened_vars[*aux_index].get_ref_mut(jolt_polynomials) =
                Rep3MultilinearPolynomial::from(coeffs);
        }

        Ok(())
    }

    /// Number of constraint rows per step, padded to the next power of two.
    pub fn padded_rows_per_step(&self) -> usize {
        let num_constraints =
            self.uniform_builder.constraints.len() + self.offset_equality_constraints.len();
        num_constraints.next_power_of_two()
    }

    /// Total number of rows used across all repeated constraints. Not padded to nearest power of two.
    pub fn constraint_rows(&self) -> usize {
        self.uniform_repeat * self.padded_rows_per_step()
    }

    pub fn uniform_repeat(&self) -> usize {
        self.uniform_repeat
    }

    /// Materializes the uniform constraints into sparse (value != 0) A, B, C matrices represented in (row, col, value) format.
    pub fn materialize_uniform(&self) -> UniformR1CS<F> {
        self.uniform_builder.materialize()
    }

    /// Converts builder::OffsetEqConstraints into key::CrossStepR1CSConstraint
    pub fn materialize_offset_eq(&self) -> CrossStepR1CS<F> {
        // (a - b) * condition == 0
        // A: a - b
        // B: condition
        // C: 0

        let mut constraints = Vec::with_capacity(self.offset_equality_constraints.len());
        for constraint in &self.offset_equality_constraints {
            let mut eq = SparseEqualityItem::<F>::empty();
            let mut condition = SparseEqualityItem::<F>::empty();

            constraint
                .cond
                .1
                .terms()
                .iter()
                .for_each(|term| match term.0 {
                    Variable::Input(inner) | Variable::Auxiliary(inner) => condition
                        .offset_vars
                        .push((inner, constraint.cond.0, F::from_i64(term.1))),
                    Variable::Constant => {}
                });
            if let Some(term) = constraint.cond.1.constant_term() {
                condition.constant = F::from_i64(term.1);
            }

            // Can't simply combine like terms because of the offset
            let lhs = constraint.a.1.clone();
            let rhs = -constraint.b.1.clone();

            lhs.terms().iter().for_each(|term| match term.0 {
                Variable::Input(inner) | Variable::Auxiliary(inner) => {
                    eq.offset_vars
                        .push((inner, constraint.a.0, F::from_i64(term.1)))
                }
                Variable::Constant => {}
            });
            rhs.terms().iter().for_each(|term| match term.0 {
                Variable::Input(inner) | Variable::Auxiliary(inner) => {
                    eq.offset_vars
                        .push((inner, constraint.b.0, F::from_i64(term.1)))
                }
                Variable::Constant => {}
            });

            // Handle constants
            lhs.terms().iter().for_each(|term| {
                assert!(
                    !matches!(term.0, Variable::Constant),
                    "Constants only supported in RHS"
                )
            });
            if let Some(term) = rhs.constant_term() {
                eq.constant = F::from_i64(term.1);
            }

            constraints.push(CrossStepR1CSConstraint::new(eq, condition));
        }

        CrossStepR1CS { constraints }
    }

    #[tracing::instrument(skip_all)]
    pub fn compute_spartan_Az_Bz_Cz(
        &self,
        flattened_polynomials: &[&Rep3MultilinearPolynomial<F>], // N variables of (S steps)
        party_id: PartyID,
    ) -> Rep3SpartanInterleavedPolynomial<F> {
        Rep3SpartanInterleavedPolynomial::new(
            &self.uniform_builder.constraints,
            &self.offset_equality_constraints,
            flattened_polynomials,
            self.padded_rows_per_step(),
            party_id,
        )
    }
}

impl<const C: usize, I: ConstraintInput, F: JoltField> From<&CombinedUniformBuilder<C, F, I>>
    for UniformSpartanKey<C, I, F>
{
    fn from(constraint_builder: &CombinedUniformBuilder<C, F, I>) -> Self {
        let uniform_r1cs = constraint_builder.materialize_uniform();
        let offset_eq_r1cs = constraint_builder.materialize_offset_eq();

        let total_rows = constraint_builder.constraint_rows().next_power_of_two();
        let num_steps = constraint_builder.uniform_repeat().next_power_of_two(); // TODO(JP): Number of steps no longer need to be padded.

        let vk_digest = Self::digest(&uniform_r1cs, &offset_eq_r1cs, num_steps);

        Self {
            _inputs: PhantomData,
            uniform_r1cs,
            offset_eq_r1cs,
            num_cons_total: total_rows,
            num_steps,
            vk_digest,
        }
    }
}
