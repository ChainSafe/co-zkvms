use std::{cell::RefCell, iter::once, rc::Rc, sync::Arc};

use num_traits::Zero;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable, math::Math, small_scalar::SmallScalar,
        thread::unsafe_allocate_zero_vec,
    },
    zkvm::{
        instruction::{
            CircuitFlags, InstructionFlags, InstructionLookup, InterleavedBitsMarker,
            NUM_CIRCUIT_FLAGS,
        },
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
use common::constants::{REGISTER_COUNT, XLEN};
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::NormalizedInstruction;

/// Number of batched read-checking sumchecks bespokely
const STAGES: usize = 3;

#[derive(Allocative)]
struct ReadCheckingProverState<F: JoltField> {
    F: [MultilinearPolynomial<F>; STAGES],
    ra: Vec<RaPolynomial<u8, F>>,
    v: Vec<ExpandingTable<F>>,
    eq_polys: [MultilinearPolynomial<F>; STAGES],
    val_gamma: Option<[F; STAGES]>,
    pc: Vec<usize>,
}

#[derive(Allocative)]
pub struct ReadRafSumcheck<F: JoltField> {
    gamma: [F; STAGES],
    gamma_cub: F,
    gamma_sqr: F,
    rv_claim: F,
    log_K_chunk: usize,
    K_chunk: usize,
    log_K: usize,
    log_T: usize,
    d: usize,
    prover_state: Option<ReadCheckingProverState<F>>,
    val_polys: [MultilinearPolynomial<F>; STAGES],
    int_poly: IdentityPolynomial<F>,
}

#[derive(Debug, Clone, Copy)]
pub enum ReadCheckingValType {
    /// Spartan outer sumcheck
    Stage1,
    /// Registers read-write sumcheck
    Stage2,
    /// Registers val sumcheck wa, PCSumcheck, Instruction Lookups
    Stage3,
}

impl<F: JoltField> ReadRafSumcheck<F> {
    /// Construct a prover instance from pre-extracted parts.
    ///
    /// * `gamma` — the scalar challenge (raw, not powered)
    /// * `rv_claim` — the combined input claim
    /// * `log_K`, `log_T`, `d` — bytecode parameters
    /// * `val_polys` — 3 precomputed val polynomials
    /// * `r_cycles` — 3 cycle challenge vectors (from accumulator)
    /// * `trace`, `preprocessing` — for computing F arrays and pc indices
    #[allow(clippy::too_many_arguments)]
    pub fn new_prover_from_parts(
        gamma: F,
        rv_claim: F,
        log_K: usize,
        log_T: usize,
        d: usize,
        val_polys: [Vec<F>; STAGES],
        r_cycles: [Vec<F::Challenge>; STAGES],
        pc: Vec<usize>,
        F_polys: [Vec<F>; STAGES],
    ) -> Self {
        let log_K_chunk = log_K.div_ceil(d);
        let K_chunk = 1 << log_K_chunk;
        let K = 1 << log_K;
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;

        let eq_evals = [
            EqPolynomial::evals(&r_cycles[0]),
            EqPolynomial::evals(&r_cycles[1]),
            EqPolynomial::evals(&r_cycles[2]),
        ];

        let eq_polys = eq_evals
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let F_mlps = [
            MultilinearPolynomial::from(F_polys[0].clone()),
            MultilinearPolynomial::from(F_polys[1].clone()),
            MultilinearPolynomial::from(F_polys[2].clone()),
        ];

        let mut v = (0..d)
            .map(|_| ExpandingTable::new(K_chunk))
            .collect::<Vec<_>>();
        v.par_iter_mut().for_each(|v| v.reset(F::one()));

        let int_poly = IdentityPolynomial::<F>::new(log_K);

        Self {
            rv_claim,
            log_K,
            log_K_chunk,
            K_chunk,
            d,
            log_T,
            prover_state: Some(ReadCheckingProverState {
                F: F_mlps,
                ra: Vec::with_capacity(d),
                v,
                eq_polys,
                val_gamma: None,
                pc,
            }),
            val_polys: [
                MultilinearPolynomial::from(val_polys[0].clone()),
                MultilinearPolynomial::from(val_polys[1].clone()),
                MultilinearPolynomial::from(val_polys[2].clone()),
            ],
            int_poly,
            gamma: [F::one(), gamma, gamma_sqr],
            gamma_sqr,
            gamma_cub,
        }
    }

    /// Construct a verifier-like instance from pre-extracted parts.
    #[allow(clippy::too_many_arguments)]
    pub fn new_verifier_from_parts(
        gamma: F,
        rv_claim: F,
        log_K: usize,
        log_T: usize,
        d: usize,
        val_polys: [Vec<F>; STAGES],
    ) -> Self {
        let log_K_chunk = log_K.div_ceil(d);
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;
        let int_poly = IdentityPolynomial::new(log_K);

        Self {
            gamma: [F::one(), gamma, gamma_sqr],
            gamma_sqr,
            gamma_cub,
            rv_claim,
            log_K,
            log_K_chunk,
            K_chunk: 1 << log_K_chunk,
            d,
            log_T,
            prover_state: None,
            val_polys: [
                MultilinearPolynomial::from(val_polys[0].clone()),
                MultilinearPolynomial::from(val_polys[1].clone()),
                MultilinearPolynomial::from(val_polys[2].clone()),
            ],
            int_poly,
        }
    }

    pub fn get_r_cycle(acc: &Rc<RefCell<ProverOpeningAccumulator<F>>>) -> [Vec<F::Challenge>; STAGES] {
        let (r_cycle_1, _) = acc
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_2) = r.split_at((REGISTER_COUNT as usize).log_2());
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_3) = r.split_at((REGISTER_COUNT as usize).log_2());
        let result = [r_cycle_1.r, r_cycle_2.r, r_cycle_3.r];
        result
    }

    pub fn get_r_cycle_verif(
        acc: &Rc<RefCell<VerifierOpeningAccumulator<F>>>,
    ) -> [Vec<F::Challenge>; STAGES] {
        let (r_cycle_1, _) = acc
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_2) = r.split_at((REGISTER_COUNT as usize).log_2());
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_3) = r.split_at((REGISTER_COUNT as usize).log_2());
        let result = [r_cycle_1.r, r_cycle_2.r, r_cycle_3.r];
        result
    }

    /// Computes `Val_1` from `&[Instruction]` directly.
    pub fn compute_val_1_from_bytecode(
        bytecode: &[tracer::instruction::Instruction],
        gamma_powers: &[F],
    ) -> Vec<F> {
        bytecode
            .par_iter()
            .map(|instruction| {
                let NormalizedInstruction {
                    address: unexpanded_pc,
                    operands,
                    ..
                } = instruction.normalize();
                let flags = instruction.circuit_flags();

                #[cfg(not(feature = "rv64"))]
                let imm_i128 = if flags[CircuitFlags::Branch] {
                    operands.imm as i32 as i128
                } else {
                    operands.imm as common::constants::XlenInt as i128
                };
                #[cfg(feature = "rv64")]
                let imm_i128 = operands.imm;

                let mut linear_combination = F::zero();
                linear_combination += F::from_u64(unexpanded_pc as u64);
                linear_combination += imm_i128.field_mul(gamma_powers[1]);
                linear_combination += (operands.rd as u64).field_mul(gamma_powers[2]);
                assert!(
                    !flags[CircuitFlags::IsCompressed]
                        || !flags[CircuitFlags::DoNotUpdateUnexpandedPC]
                );
                for (flag, gamma_power) in flags.iter().zip(gamma_powers[3..].iter()) {
                    if *flag {
                        linear_combination += *gamma_power;
                    }
                }

                linear_combination
            })
            .collect()
    }

    /// Like `compute_val_2` but takes `&[Instruction]` and `eq_r_register` directly.
    pub fn compute_val_2_from_bytecode(
        bytecode: &[tracer::instruction::Instruction],
        gamma_powers: &[F],
        eq_r_register: &[F],
    ) -> Vec<F> {
        bytecode
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();

                std::iter::empty()
                    .chain(once(instr.operands.rd))
                    .chain(once(instr.operands.rs1))
                    .chain(once(instr.operands.rs2))
                    .map(|r| eq_r_register[r as usize])
                    .zip(gamma_powers)
                    .map(|(claim, gamma)| claim * gamma)
                    .sum::<F>()
            })
            .collect()
    }

    /// Like `compute_val_3` but takes `&[Instruction]` and `eq_r_register` directly.
    pub fn compute_val_3_from_bytecode(
        bytecode: &[tracer::instruction::Instruction],
        gamma_powers: &[F],
        eq_r_register: &[F],
    ) -> Vec<F> {
        bytecode
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();
                let flags = instruction.circuit_flags();
                let unexpanded_pc = instr.address;

                let mut linear_combination: F = F::zero();

                linear_combination += eq_r_register[instr.operands.rd as usize];
                linear_combination += gamma_powers[1].mul_u64(unexpanded_pc as u64);
                if flags[CircuitFlags::IsNoop] {
                    linear_combination += gamma_powers[2];
                }
                if !flags.is_interleaved_operands() {
                    linear_combination += gamma_powers[3];
                }

                if let Some(table) = instruction.lookup_table() {
                    let table_index = LookupTables::enum_index(&table);
                    linear_combination += gamma_powers[4 + table_index];
                }

                linear_combination
            })
            .collect()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        self.degree()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self) -> F {
        self.input_claim()
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (r_address_prime, r_cycle_prime) = r.split_at(self.log_K);
        // r_cycle is bound LowToHigh, so reverse
        let r_cycle_prime = r_cycle_prime
            .iter()
            .rev()
            .copied()
            .collect::<Vec<F::Challenge>>();

        let int_poly = self.int_poly.evaluate(r_address_prime);

        let ra_claims = (0..self.d).map(|i| {
            accumulator
                .borrow()
                .get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                )
                .1
        });
        let r_cycles = Self::get_r_cycle_verif(accumulator);

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^3 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^2 * Int)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * raf_1 + gamma^4 * raf_3
        let val = self
            .val_polys
            .iter()
            .zip(r_cycles.iter())
            .zip(self.gamma.iter())
            .zip([
                int_poly * self.gamma_cub, // RAF for Stage1
                F::zero(),                 // There's no raf for Stage2
                int_poly * self.gamma_sqr, // RAF for Stage3
            ])
            .map(|(((val, r_cycle), gamma), int_poly)| {
                (val.evaluate(r_address_prime) + int_poly)
                    * EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime)
                    * gamma
            })
            .sum::<F>();

        ra_claims.fold(val, |running, ra_claim| running * ra_claim)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        self.normalize_opening_point(opening_point)
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (r_address, r_cycle) = opening_point.split_at(self.log_K);
        (0..self.d).for_each(|i| {
            let r_address = &r_address.r[self.log_K_chunk * i..self.log_K_chunk * (i + 1)];
            accumulator.borrow_mut().append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                [r_address, &r_cycle.r].concat(),
            );
        });
    }
}

impl<F: JoltField> ReadRafSumcheck<F> {
    pub fn d(&self) -> usize {
        self.d
    }

    pub fn log_K(&self) -> usize {
        self.log_K
    }

    pub fn log_K_chunk(&self) -> usize {
        self.log_K_chunk
    }

    /// Returns the final sumcheck claim for `ra[i]` (prover only).
    pub fn ra_final_claim(&self, i: usize) -> F {
        self.prover_state
            .as_ref()
            .expect("ra_final_claim called on verifier instance")
            .ra[i]
            .final_sumcheck_claim()
    }

    pub fn gamma_sqr(&self) -> F {
        self.gamma_sqr
    }

    pub fn gamma_cub(&self) -> F {
        self.gamma_cub
    }

    /// Returns `[gamma^0, gamma^1, gamma^2]` (the per-stage gamma weights).
    pub fn gamma_stages(&self) -> [F; 3] {
        self.gamma
    }

    /// Evaluates `int_poly` (the IdentityPolynomial) at the given point.
    pub fn int_poly_evaluate(&self, r: &[F::Challenge]) -> F {
        use crate::poly::multilinear_polynomial::PolynomialEvaluation;
        self.int_poly.evaluate(r)
    }

    /// Evaluates each `val_poly` at `r_address` and returns the three evaluations.
    pub fn val_polys_evaluate(&self, r_address: &[F::Challenge]) -> [F; 3] {
        use crate::poly::multilinear_polynomial::PolynomialEvaluation;
        std::array::from_fn(|i| self.val_polys[i].evaluate(r_address))
    }

    pub fn degree(&self) -> usize {
        self.d + 1
    }

    pub fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    pub fn input_claim(&self) -> F {
        self.rv_claim
    }

    pub fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = opening_point.to_vec();
        r[self.log_K..].reverse();
        OpeningPoint::new(r)
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::compute_prover_message")]
    pub fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        if round < self.log_K {
            const DEGREE: usize = 2;

            let result: Vec<F> = (0..self.val_polys[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let ra_evals = ps.F.iter().map(|poly| {
                        poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow)
                    });
                    let int_evals =
                        self.int_poly
                            .sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);
                    let val_evals = self
                        .val_polys
                        .iter()
                        .map(|val| val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow))
                        .zip([Some(&int_evals), None, Some(&int_evals)])
                        .zip([Some(self.gamma_cub), None, Some(self.gamma_sqr)])
                        .map(|((val_evals, int_evals), gamma)| {
                            std::array::from_fn::<F, DEGREE, _>(|j| {
                                val_evals[j]
                                    + int_evals.map_or(F::zero(), |int_evals| {
                                        int_evals[j] * gamma.unwrap()
                                    })
                            })
                        });

                    ra_evals
                        .zip(val_evals)
                        .zip(self.gamma.iter())
                        .map(|((ra_evals, val_evals), gamma)| {
                            std::array::from_fn::<F::Unreduced<9>, DEGREE, _>(|j| {
                                let val_gamma = val_evals[j] * gamma;
                                ra_evals[j].mul_unreduced::<9>(val_gamma)
                            })
                        })
                        .fold([F::Unreduced::zero(); DEGREE], |mut running, new| {
                            for i in 0..DEGREE {
                                running[i] += new[i];
                            }
                            running
                        })
                })
                .reduce(
                    || [F::Unreduced::zero(); DEGREE],
                    |mut running, new| {
                        for i in 0..DEGREE {
                            running[i] += new[i];
                        }
                        running
                    },
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect();
            result
        } else {
            let degree = self.degree();
            (0..ps.ra[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let eq_evals = ps
                        .eq_polys
                        .iter()
                        .map(|eq| eq.sumcheck_evals(i, degree, BindingOrder::LowToHigh));
                    let ra_evals = ps
                        .ra
                        .iter()
                        .map(|ra| ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh));
                    let eq_times_val = eq_evals
                        .zip(ps.val_gamma.as_ref().unwrap().iter())
                        .map(|(eq_evals, val_evals)| {
                            eq_evals
                                .into_iter()
                                .map(|eq_eval| eq_eval * val_evals)
                                .collect()
                        })
                        .fold(
                            vec![F::zero(); degree],
                            |mut running: Vec<F>, new: Vec<F>| {
                                for i in 0..degree {
                                    running[i] += new[i];
                                }
                                running
                            },
                        );

                    let ra_evals = ra_evals.fold(vec![F::one(); degree], |mut running, new| {
                        for i in 0..degree {
                            running[i] *= new[i];
                        }
                        running
                    });

                    ra_evals
                        .into_iter()
                        .zip(eq_times_val)
                        .map(|(ra, eq)| ra.mul_unreduced::<9>(eq))
                        .collect::<Vec<_>>()
                })
                .reduce(
                    || vec![F::Unreduced::zero(); degree],
                    |mut running, new| {
                        for i in 0..degree {
                            running[i] += new[i];
                        }
                        running
                    },
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect()
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < self.log_K {
            rayon::scope(|s| {
                s.spawn(|_| {
                    self.val_polys
                        .par_iter_mut()
                        .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                });
                s.spawn(|_| {
                    self.int_poly.bind_parallel(r_j, BindingOrder::HighToLow);
                });
                s.spawn(|_| {
                    ps.F.par_iter_mut()
                        .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
                });
                s.spawn(|_| {
                    ps.v[round / self.log_K_chunk].update(r_j);
                });
            });
            if round == self.log_K - 1 {
                self.init_log_t_rounds();
            }
        } else {
            ps.ra
                .par_iter_mut()
                .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
            ps.eq_polys
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn init_log_t_rounds(&mut self) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        let int_poly = self.int_poly.final_sumcheck_claim();

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^3 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^2 * Int)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * raf_1 + gamma^4 * raf_3
        ps.val_gamma = Some(
            self.val_polys
                .iter()
                .zip(self.gamma.iter())
                .zip([
                    int_poly * self.gamma_cub,
                    F::zero(),
                    int_poly * self.gamma_sqr,
                ])
                .map(|((poly, gamma), int_poly)| (poly.final_sumcheck_claim() + int_poly) * gamma)
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        );

        ps.v.par_iter()
            .enumerate()
            .map(|(i, v)| {
                let ra_i: Vec<Option<u8>> = ps
                    .pc
                    .par_iter()
                    .map(|k| {
                        let k = (k >> (self.log_K_chunk * (self.d - i - 1))) % self.K_chunk;
                        Some(k as u8)
                    })
                    .collect();
                RaPolynomial::new(Arc::new(ra_i), v.clone_values())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|ra| {
                ps.ra.push(ra);
            });
    }
}

pub fn get_gamma_powers<F: JoltField>(transcript: &mut impl Transcript, amount: usize) -> Vec<F> {
    let mut gamma_powers = vec![F::one()];
    let gamma: F = transcript.challenge_scalar();
    for _ in 1..amount {
        gamma_powers.push(gamma * gamma_powers.last().unwrap());
    }
    gamma_powers
}
