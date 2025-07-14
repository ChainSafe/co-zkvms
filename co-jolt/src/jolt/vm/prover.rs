use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly_commit::multilinear_pc::data_structures::Proof;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Context;
use itertools::{chain, interleave};
use jolt_core::lasso::memory_checking::{
    MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, NoPreprocessing,
};
use jolt_core::poly::commitment::commitment_scheme::CommitShape;
use jolt_core::poly::structured_poly::StructuredOpeningProof;
use jolt_core::{
    jolt::vm::instruction_lookups::{InstructionCommitment, PrimarySumcheckOpenings},
    lasso::memory_checking::MultisetHashes,
    poly::{
        commitment::commitment_scheme::{BatchType, CommitmentScheme},
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        field::JoltField,
        identity_poly::IdentityPolynomial,
        structured_poly::StructuredCommitment,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::{
        grand_product::{
            BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
        },
        sumcheck::SumcheckInstanceProof,
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        mul_0_1_optimized, split_poly_flagged,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};
use tracing::trace_span;

use super::{LassoPolynomials, InstructionLookupsPreprocessing};
use crate::{instructions::LookupSet, subtables::SubtableSet};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LassoProof<const C: usize, const M: usize, F, CS, Lookups, Subtables>
where
    F: JoltField,
    CS: CommitmentScheme<Field = F>,
    Lookups: LookupSet<F>,
    Subtables: SubtableSet<F>,
{
    pub(crate) _marker: PhantomData<Lookups>,
    pub primary_sumcheck: PrimarySumcheck<F, CS>,
    pub(crate) memory_checking: MemoryCheckingProof<
        F,
        CS,
        Polynomials<F, CS>,
        InstructionReadWriteOpenings<F>,
        InstructionFinalOpenings<F, Subtables>,
    >,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct PrimarySumcheck<F: JoltField, CS: CommitmentScheme<Field = F>> {
    pub(crate) sumcheck_proof: SumcheckInstanceProof<F>,
    pub(crate) num_rounds: usize,
    pub(crate) openings: PrimarySumcheckOpenings<F>,
    pub opening_proof: CS::BatchedProof,
}

type Preprocessing<F> = InstructionLookupsPreprocessing<F>;
type Polynomials<F, C> = LassoPolynomials<F, C>;

impl<const C: usize, const M: usize, F: JoltField, CS, Lookups, Subtables>
    LassoProof<C, M, F, CS, Lookups, Subtables>
where
    CS: CommitmentScheme<Field = F>,
    Lookups: LookupSet<F>,
    Subtables: SubtableSet<F>,
{
    #[tracing::instrument(skip_all, name = "LassoProver::prove")]
    pub fn prove(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F, CS>,
        setup: &CS::Setup,
        transcript: &mut ProofTranscript,
    ) -> LassoProof<C, M, F, CS, Lookups, Subtables> {
        let trace_length = polynomials.dim[0].len();
        let r_eq = transcript.challenge_vector(b"Jolt instruction lookups", trace_length.log_2());

        let eq_evals: Vec<F> = EqPolynomial::new(r_eq.to_vec()).evals();
        let mut eq_poly = DensePolynomial::new(eq_evals);
        let num_rounds = trace_length.log_2();

        // TODO: compartmentalize all primary sumcheck logic
        let (primary_sumcheck_proof, r_primary_sumcheck, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck(
                preprocessing,
                num_rounds,
                &mut eq_poly,
                &polynomials.E_polys,
                &polynomials.instruction_flag_polys,
                &mut polynomials.lookup_outputs.clone(),
                Self::sumcheck_poly_degree(),
                transcript,
            );

        let sumcheck_openings = PrimarySumcheckOpenings {
            E_poly_openings: E_evals,
            flag_openings: flag_evals,
            lookup_outputs_opening: outputs_eval,
        };

        let opening_proof = Self::prove_openings(
            polynomials,
            &r_primary_sumcheck,
            &sumcheck_openings,
            setup,
            transcript,
        );

        let primary_sumcheck = PrimarySumcheck {
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            openings: sumcheck_openings,
            opening_proof,
        };

        let memory_checking =
            Self::prove_memory_checking(preprocessing, polynomials, setup, transcript);

        LassoProof {
            _marker: PhantomData,
            primary_sumcheck,
            memory_checking,
        }
    }

    /// Converts instruction flag polynomials into memory flag polynomials. A memory flag polynomial
    /// can be computed by summing over the instructions that use that memory: if a given execution step
    /// accesses the memory, it must be executing exactly one of those instructions.
    pub fn memory_flag_polys(
        preprocessing: &Preprocessing<F>,
        instruction_flag_bitvectors: &[Vec<u64>],
    ) -> Vec<DensePolynomial<F>> {
        let m = instruction_flag_bitvectors[0].len();

        (0..preprocessing.num_memories)
            .into_par_iter()
            .map(|memory_index| {
                let mut memory_flag_bitvector = vec![0u64; m];
                for instruction_index in 0..Lookups::COUNT {
                    if preprocessing.instruction_to_memory_indices[instruction_index]
                        .contains(&memory_index)
                    {
                        memory_flag_bitvector
                            .iter_mut()
                            .zip(&instruction_flag_bitvectors[instruction_index])
                            .for_each(|(memory_flag, instruction_flag)| {
                                *memory_flag += instruction_flag
                            });
                    }
                }
                DensePolynomial::from_u64(&memory_flag_bitvector)
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "LassoProver::prove_primary_sumcheck")]
    fn prove_primary_sumcheck(
        preprocessing: &Preprocessing<F>,
        num_rounds: usize,
        eq_poly: &mut DensePolynomial<F>,
        memory_polys: &Vec<DensePolynomial<F>>,
        flag_polys: &Vec<DensePolynomial<F>>,
        lookup_outputs_poly: &mut DensePolynomial<F>,
        degree: usize,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, Vec<F>, Vec<F>, F) {
        // Check all polys are the same size
        let poly_len = eq_poly.len();
        memory_polys
            .iter()
            .for_each(|E_poly| debug_assert_eq!(E_poly.len(), poly_len));
        flag_polys
            .iter()
            .for_each(|flag_poly| debug_assert_eq!(flag_poly.len(), poly_len));
        debug_assert_eq!(lookup_outputs_poly.len(), poly_len);

        let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        let num_eval_points = degree + 1;

        let round_uni_poly = Self::primary_sumcheck_inner_loop(
            preprocessing,
            eq_poly,
            flag_polys,
            memory_polys,
            lookup_outputs_poly,
            num_eval_points,
        );
        compressed_polys.push(round_uni_poly.compress());
        let r_j = Self::update_primary_sumcheck_transcript(round_uni_poly, transcript);
        random_vars.push(r_j);

        let _bind_span = trace_span!("BindPolys");
        let _bind_enter = _bind_span.enter();
        rayon::join(
            || eq_poly.bound_poly_var_top(&r_j),
            || lookup_outputs_poly.bound_poly_var_top_many_ones(&r_j),
        );
        let mut flag_polys_updated: Vec<DensePolynomial<F>> = flag_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_bound_poly_var_top_flags(&r_j))
            .collect();
        let mut memory_polys_updated: Vec<DensePolynomial<F>> = memory_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_bound_poly_var_top(&r_j))
            .collect();
        drop(_bind_enter);
        drop(_bind_span);

        for _round in 1..num_rounds {
            let round_uni_poly = Self::primary_sumcheck_inner_loop(
                preprocessing,
                eq_poly,
                &flag_polys_updated,
                &memory_polys_updated,
                lookup_outputs_poly,
                num_eval_points,
            );
            compressed_polys.push(round_uni_poly.compress());
            let r_j = Self::update_primary_sumcheck_transcript(round_uni_poly, transcript);
            random_vars.push(r_j);

            // Bind all polys
            let _bind_span = trace_span!("BindPolys");
            let _bind_enter = _bind_span.enter();
            rayon::join(
                || eq_poly.bound_poly_var_top(&r_j),
                || lookup_outputs_poly.bound_poly_var_top_many_ones(&r_j),
            );
            flag_polys_updated
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j));
            memory_polys_updated
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j));

            drop(_bind_enter);
            drop(_bind_span);
        } // End rounds

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();
        let flag_evals = flag_polys_updated.iter().map(|poly| poly[0]).collect();
        let memory_evals = memory_polys_updated.iter().map(|poly| poly[0]).collect();
        let outputs_eval = lookup_outputs_poly[0];

        (
            SumcheckInstanceProof::new(compressed_polys),
            random_vars,
            flag_evals,
            memory_evals,
            outputs_eval,
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "LassoProver::primary_sumcheck_inner_loop",
        level = "trace"
    )]
    fn primary_sumcheck_inner_loop(
        preprocessing: &Preprocessing<F>,
        eq_poly: &DensePolynomial<F>,
        flag_polys: &[DensePolynomial<F>],
        memory_polys: &[DensePolynomial<F>],
        lookup_outputs_poly: &DensePolynomial<F>,
        num_eval_points: usize,
    ) -> UniPoly<F> {
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        // Loop over half MLE size (size of MLE next round)
        //   - Compute evaluations of eq, flags, E, at p {0, 1, ..., degree}:
        //       eq(p, _boolean_hypercube_), flags(p, _boolean_hypercube_), E(p, _boolean_hypercube_)
        // After: Sum over MLE elements (with combine)
        let evaluations: Vec<F> = (0..mle_half)
            .into_par_iter()
            .map(|low_index| {
                let high_index = mle_half + low_index;

                let mut eq_evals: Vec<F> = vec![F::zero(); num_eval_points];
                let mut outputs_evals: Vec<F> = vec![F::zero(); num_eval_points];
                let mut multi_flag_evals: Vec<Vec<F>> =
                    vec![vec![F::zero(); Lookups::COUNT]; num_eval_points];
                let mut multi_memory_evals: Vec<Vec<F>> =
                    vec![vec![F::zero(); preprocessing.num_memories]; num_eval_points];

                eq_evals[0] = eq_poly[low_index];
                eq_evals[1] = eq_poly[high_index];
                let eq_m = eq_poly[high_index] - eq_poly[low_index];
                for eval_index in 2..num_eval_points {
                    eq_evals[eval_index] = eq_evals[eval_index - 1] + eq_m;
                }

                outputs_evals[0] = lookup_outputs_poly[low_index];
                outputs_evals[1] = lookup_outputs_poly[high_index];
                let outputs_m = lookup_outputs_poly[high_index] - lookup_outputs_poly[low_index];
                for eval_index in 2..num_eval_points {
                    outputs_evals[eval_index] = outputs_evals[eval_index - 1] + outputs_m;
                }

                // TODO: Exactly one flag across NUM_INSTRUCTIONS is non-zero
                for flag_instruction_index in 0..Lookups::COUNT {
                    multi_flag_evals[0][flag_instruction_index] =
                        flag_polys[flag_instruction_index][low_index];
                    multi_flag_evals[1][flag_instruction_index] =
                        flag_polys[flag_instruction_index][high_index];
                    let flag_m = flag_polys[flag_instruction_index][high_index]
                        - flag_polys[flag_instruction_index][low_index];
                    for eval_index in 2..num_eval_points {
                        let flag_eval =
                            multi_flag_evals[eval_index - 1][flag_instruction_index] + flag_m;
                        multi_flag_evals[eval_index][flag_instruction_index] = flag_eval;
                    }
                }

                // TODO: Some of these intermediates need not be computed if flags is computed
                for memory_index in 0..preprocessing.num_memories {
                    multi_memory_evals[0][memory_index] = memory_polys[memory_index][low_index];

                    multi_memory_evals[1][memory_index] = memory_polys[memory_index][high_index];
                    let memory_m = memory_polys[memory_index][high_index]
                        - memory_polys[memory_index][low_index];
                    for eval_index in 2..num_eval_points {
                        multi_memory_evals[eval_index][memory_index] =
                            multi_memory_evals[eval_index - 1][memory_index] + memory_m;
                    }
                }

                // Accumulate inner terms.
                // S({0,1,... num_eval_points}) = eq * [ INNER TERMS ]
                //            = eq[000] * [ flags_0[000] * g_0(E_0)[000] + flags_1[000] * g_1(E_1)[000]]
                //            + eq[001] * [ flags_0[001] * g_0(E_0)[001] + flags_1[001] * g_1(E_1)[001]]
                //            + ...
                //            + eq[111] * [ flags_0[111] * g_0(E_0)[111] + flags_1[111] * g_1(E_1)[111]]
                let mut inner_sum = vec![F::zero(); num_eval_points];
                for instruction in Lookups::iter() {
                    let instruction_index = Lookups::enum_index(&instruction);
                    let memory_indices = &preprocessing.instruction_to_memory_indices[instruction_index];

                    for eval_index in 0..num_eval_points {
                        let flag_eval = multi_flag_evals[eval_index][instruction_index];
                        if flag_eval.is_zero() {
                            continue;
                        }; // Early exit if no contribution.

                        let terms: Vec<F> = memory_indices
                            .iter()
                            .map(|memory_index| multi_memory_evals[eval_index][*memory_index])
                            .collect();
                        let instruction_collation_eval = instruction.combine_lookups(&terms, C, M);

                        // TODO(sragss): Could sum all shared inner terms before multiplying by the flag eval
                        inner_sum[eval_index] += flag_eval * instruction_collation_eval;
                    }
                }
                let evaluations: Vec<F> = (0..num_eval_points)
                    .map(|eval_index| {
                        eq_evals[eval_index] * (inner_sum[eval_index] - outputs_evals[eval_index])
                    })
                    .collect();
                evaluations
            })
            .reduce(
                || vec![F::zero(); num_eval_points],
                |running, new| {
                    debug_assert_eq!(running.len(), new.len());
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + n)
                        .collect()
                },
            );

        UniPoly::from_evals(&evaluations)
    }

    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &Polynomials<F, CS>,
        opening_point: &[F],
        openings: &PrimarySumcheckOpenings<F>,
        setup: &CS::Setup,
        transcript: &mut ProofTranscript,
    ) -> CS::BatchedProof {
        let primary_sumcheck_polys = chain![
            polynomials.E_polys.iter(),
            polynomials.instruction_flag_polys.iter(),
            [&polynomials.lookup_outputs].into_iter()
        ]
        .collect::<Vec<_>>();
        let mut primary_sumcheck_openings: Vec<F> = [
            openings.E_poly_openings.as_slice(),
            openings.flag_openings.as_slice(),
        ]
        .concat();
        primary_sumcheck_openings.push(openings.lookup_outputs_opening);

        CS::batch_prove(
            &primary_sumcheck_polys,
            opening_point,
            &primary_sumcheck_openings,
            setup,
            BatchType::Big,
            transcript,
        )
    }

    fn update_primary_sumcheck_transcript(
        round_uni_poly: UniPoly<F>,
        transcript: &mut ProofTranscript,
    ) -> F {
        round_uni_poly.append_to_transcript(b"poly", transcript);

        transcript.challenge_scalar::<F>(b"challenge_nextround")
    }

    fn combine_lookups(preprocessing: &Preprocessing<F>, vals: &[F], flags: &[F]) -> F {
        assert_eq!(vals.len(), preprocessing.num_memories);
        assert_eq!(flags.len(), Lookups::COUNT);

        let mut sum = F::zero();
        for instruction in Lookups::iter() {
            let instruction_index = Lookups::enum_index(&instruction);
            let memory_indices = &preprocessing.instruction_to_memory_indices[instruction_index];
            let filtered_operands: Vec<F> = memory_indices.iter().map(|i| vals[*i]).collect();
            sum += flags[instruction_index] * instruction.combine_lookups(&filtered_operands, C, M);
        }

        sum
    }

    // Converts instruction flag values into memory flag values. A memory flag value
    /// can be computed by summing over the instructions that use that memory: if a given execution step
    /// accesses the memory, it must be executing exactly one of those instructions.
    fn memory_flags(
        preprocessing: &InstructionLookupsPreprocessing<F>,
        instruction_flags: &[F],
    ) -> Vec<F> {
        let mut memory_flags = vec![F::zero(); preprocessing.num_memories];
        for instruction_index in 0..Lookups::COUNT {
            for memory_index in &preprocessing.instruction_to_memory_indices[instruction_index] {
                memory_flags[*memory_index] += instruction_flags[instruction_index];
            }
        }
        memory_flags
    }

    /// Returns the sumcheck polynomial degree for the "primary" sumcheck. Since the primary sumcheck expression
    /// is \sum_x \tilde{eq}(r, x) * \sum_i flag_i(x) * g_i(E_1(x), ..., E_\alpha(x)), the degree is
    /// the max over all the instructions' `g_i` polynomial degrees, plus two (one for \tilde{eq}, one for flag_i)
    fn sumcheck_poly_degree() -> usize {
        Lookups::iter()
            .map(|instruction| instruction.g_poly_degree(C))
            .max()
            .unwrap()
            + 2 // eq and flag
    }

    pub fn commitment_shapes(
        preprocessing: &Preprocessing<F>,
        max_trace_length: usize,
    ) -> Vec<CommitShape> {
        let max_trace_length = max_trace_length.next_power_of_two();
        // { dim, read_cts, E_polys, instruction_flag_polys, lookup_outputs }
        let read_write_generator_shape = CommitShape::new(max_trace_length, BatchType::Big);
        let init_final_generator_shape = CommitShape::new(
            M * preprocessing.num_memories.next_power_of_two(),
            BatchType::Small,
        );

        vec![read_write_generator_shape, init_final_generator_shape]
    }

    pub fn verify(
        setup: &CS::Setup,
        preprocessing: &Preprocessing<F>,
        proof: Self,
        commitment: &InstructionCommitment<CS>,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<()> {
        let r_eq = transcript.challenge_vector::<F>(
            b"Jolt instruction lookups",
            proof.primary_sumcheck.num_rounds,
        );

        // TODO: compartmentalize all primary sumcheck logic
        let (claim_last, r_primary_sumcheck) = proof.primary_sumcheck.sumcheck_proof.verify(
            F::zero(),
            proof.primary_sumcheck.num_rounds,
            Self::sumcheck_poly_degree(),
            transcript,
        )?;

        // Verify that eq(r, r_z) * [f_1(r_z) * g(E_1(r_z)) + ... + f_F(r_z) * E_F(r_z))] = claim_last
        let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_primary_sumcheck);
        assert_eq!(
            eq_eval
                * (Self::combine_lookups(
                    preprocessing,
                    &proof.primary_sumcheck.openings.E_poly_openings,
                    &proof.primary_sumcheck.openings.flag_openings,
                ) - proof.primary_sumcheck.openings.lookup_outputs_opening),
            claim_last,
            "Primary sumcheck check failed."
        );

        Self::verify_openings(
            &proof.primary_sumcheck.openings,
            setup,
            &proof.primary_sumcheck.opening_proof,
            commitment,
            &r_primary_sumcheck,
            transcript,
        )?;

        Self::verify_memory_checking(preprocessing, &setup, proof.memory_checking, commitment, transcript)?;

        Ok(())
    }

    fn verify_openings(
        openings: &PrimarySumcheckOpenings<F>,
        setup: &CS::Setup,
        opening_proof: &CS::BatchedProof,
        commitment: &InstructionCommitment<CS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<()> {
        let mut primary_sumcheck_openings: Vec<F> = [
            openings.E_poly_openings.as_slice(),
            openings.flag_openings.as_slice(),
        ]
        .concat();
        primary_sumcheck_openings.push(openings.lookup_outputs_opening);
        let primary_sumcheck_commitments = commitment.trace_commitment
            [commitment.trace_commitment.len() - primary_sumcheck_openings.len()..]
            .iter()
            .collect::<Vec<_>>();

        CS::batch_verify(
            opening_proof,
            setup,
            opening_point,
            &primary_sumcheck_openings,
            &primary_sumcheck_commitments,
            transcript,
        )
        .context("while verifying primary sumcheck openings")?;

        Ok(())
    }
}

impl<const C: usize, const M: usize, F, CS, Lookups, Subtables>
    MemoryCheckingProver<F, CS, LassoPolynomials<F, CS>>
    for LassoProof<C, M, F, CS, Lookups, Subtables>
where
    F: JoltField,
    CS: CommitmentScheme<Field = F>,
    Lookups: LookupSet<F>,
    Subtables: SubtableSet<F>,
{
    type Preprocessing = InstructionLookupsPreprocessing<F>;
    type ReadWriteOpenings = InstructionReadWriteOpenings<F>;
    type InitFinalOpenings = InstructionFinalOpenings<F, Subtables>;

    type MemoryTuple = (F, F, F, Option<F>); // (a, v, t, flag)

    fn fingerprint(inputs: &(F, F, F, Option<F>), gamma: &F, tau: &F) -> F {
        let (a, v, t, flag) = *inputs;
        match flag {
            Some(val) => val * (t * gamma.square() + v * *gamma + a - tau) + F::one() - val,
            None => t * gamma.square() + v * *gamma + a - tau,
        }
    }

    fn compute_leaves(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F, CS>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dim[0].len();

        let read_write_leaves = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];

                let read_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| {
                        let a = &polynomials.dim[dim_index][i];
                        let v = &polynomials.E_polys[memory_index][i];
                        let t = &polynomials.read_cts[memory_index][i];
                        mul_0_1_optimized(t, &gamma_squared) + mul_0_1_optimized(v, gamma) + a - tau
                    })
                    .collect();
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();
                [
                    DensePolynomial::new(read_fingerprints),
                    DensePolynomial::new(write_fingerprints),
                ]
            })
            .collect();

        let init_final_leaves: Vec<DensePolynomial<F>> = preprocessing
            .materialized_subtables
            .par_iter()
            .enumerate()
            .flat_map_iter(|(subtable_index, subtable)| {
                let init_fingerprints: Vec<F> = (0..M)
                    .map(|i| {
                        let a = &F::from_u64(i as u64).unwrap();
                        let v = &subtable[i];
                        // let t = F::zero();
                        // Compute h(a,v,t) where t == 0
                        mul_0_1_optimized(v, gamma) + a - tau
                    })
                    .collect();

                let final_leaves: Vec<DensePolynomial<F>> = preprocessing
                    .subtable_to_memory_indices[subtable_index]
                    .iter()
                    .map(|memory_index| {
                        let final_cts = &polynomials.final_cts[*memory_index];
                        let final_fingerprints = (0..M)
                            .map(|i| {
                                init_fingerprints[i]
                                    + mul_0_1_optimized(&final_cts[i], &gamma_squared)
                            })
                            .collect();
                        DensePolynomial::new(final_fingerprints)
                    })
                    .collect();

                let mut polys = Vec::with_capacity(C + 1);
                polys.push(DensePolynomial::new(init_fingerprints));
                polys.extend(final_leaves);
                polys
            })
            .collect();

        (read_write_leaves, init_final_leaves)
    }

    fn interleave_hashes(
        preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) -> (Vec<F>, Vec<F>) {
        // R W R W R W ...
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();

        // I F F F F I F F F F ...
        let mut init_final_hashes = Vec::with_capacity(
            multiset_hashes.init_hashes.len() + multiset_hashes.final_hashes.len(),
        );
        for subtable_index in 0..Subtables::COUNT {
            init_final_hashes.push(multiset_hashes.init_hashes[subtable_index]);
            let memory_indices = &preprocessing.subtable_to_memory_indices[subtable_index];
            memory_indices
                .iter()
                .for_each(|i| init_final_hashes.push(multiset_hashes.final_hashes[*i]));
        }

        (read_write_hashes, init_final_hashes)
    }


    fn uninterleave_hashes(
        preprocessing: &Self::Preprocessing,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len(), 2 * preprocessing.num_memories);
        assert_eq!(
            init_final_hashes.len(),
            Subtables::COUNT + preprocessing.num_memories
        );

        let mut read_hashes = Vec::with_capacity(preprocessing.num_memories);
        let mut write_hashes = Vec::with_capacity(preprocessing.num_memories);
        for i in 0..preprocessing.num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(Subtables::COUNT);
        let mut final_hashes = Vec::with_capacity(preprocessing.num_memories);
        let mut init_final_hashes = init_final_hashes.iter();
        for subtable_index in 0..Subtables::COUNT {
            // I
            init_hashes.push(*init_final_hashes.next().unwrap());
            // F F F F
            let memory_indices = &preprocessing.subtable_to_memory_indices[subtable_index];
            for _ in memory_indices {
                final_hashes.push(*init_final_hashes.next().unwrap());
            }
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        assert_eq!(multiset_hashes.init_hashes.len(), Subtables::COUNT);
        assert_eq!(
            multiset_hashes.read_hashes.len(),
            preprocessing.num_memories
        );
        assert_eq!(
            multiset_hashes.write_hashes.len(),
            preprocessing.num_memories
        );
        assert_eq!(
            multiset_hashes.final_hashes.len(),
            preprocessing.num_memories
        );

        (0..preprocessing.num_memories)
            .into_par_iter()
            .for_each(|i| {
                let read_hash = multiset_hashes.read_hashes[i];
                let write_hash = multiset_hashes.write_hashes[i];
                let init_hash =
                    multiset_hashes.init_hashes[preprocessing.memory_to_subtable_index[i]];
                let final_hash = multiset_hashes.final_hashes[i];
                assert_eq!(
                    init_hash * write_hash,
                    final_hash * read_hash,
                    "Multiset hashes don't match"
                );
            });
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        preprocessing: &Self::Preprocessing,
        polynomials: &Polynomials<F, CS>,
        read_write_leaves: Vec<DensePolynomial<F>>,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>) {
        assert_eq!(read_write_leaves.len(), 2 * preprocessing.num_memories);

        let _span = trace_span!("LassoLookups: construct circuits");
        let _enter = _span.enter();

        let memory_flag_polys =
            Self::memory_flag_polys(preprocessing, &polynomials.instruction_flag_bitvectors);

        let read_write_circuits = read_write_leaves
            .par_iter()
            .enumerate()
            .map(|(i, leaves_poly)| {
                // Split while cloning to save on future cloning in GrandProductCircuit
                let memory_index = i / 2;
                let flag: &DensePolynomial<F> = &memory_flag_polys[memory_index];
                let (toggled_leaves_l, toggled_leaves_r) = split_poly_flagged(leaves_poly, flag);
                GrandProductCircuit::new_split(
                    DensePolynomial::new(toggled_leaves_l),
                    DensePolynomial::new(toggled_leaves_r),
                )
            })
            .collect::<Vec<GrandProductCircuit<F>>>();

        drop(_enter);
        drop(_span);

        let _span = trace_span!("InstructionLookups: compute hashes");
        let _enter = _span.enter();

        let read_write_hashes: Vec<F> = read_write_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        drop(_enter);
        drop(_span);

        let _span = trace_span!("InstructionLookups: the rest");
        let _enter = _span.enter();

        // Prover has access to memory_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
        let batched_circuits = BatchedGrandProductCircuit::new_batch_flags(
            read_write_circuits,
            memory_flag_polys,
            read_write_leaves,
        );

        drop(_enter);
        drop(_span);

        (batched_circuits, read_write_hashes)
    }

    fn protocol_name() -> &'static [u8] {
        b"Instruction lookups memory checking"
    }
}

impl<F, CS, InstructionSet, Subtables, const C: usize, const M: usize>
    MemoryCheckingVerifier<F, CS, LassoPolynomials<F, CS>>
    for LassoProof<C, M, F, CS, InstructionSet, Subtables>
where
    F: JoltField,
    CS: CommitmentScheme<Field = F>,
    InstructionSet: LookupSet<F>,
    Subtables: SubtableSet<F>,
{
    fn read_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let memory_flags = Self::memory_flags(preprocessing, &openings.flag_openings);
        (0..preprocessing.num_memories)
            .map(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                (
                    openings.dim_openings[dim_index],
                    openings.E_poly_openings[memory_index],
                    openings.read_openings[memory_index],
                    Some(memory_flags[memory_index]),
                )
            })
            .collect()
    }
    fn write_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        Self::read_tuples(preprocessing, openings)
            .iter()
            .map(|(a, v, t, flag)| (*a, *v, *t + F::one(), *flag))
            .collect()
    }
    fn init_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Subtables::COUNT)
            .map(|subtable_index| (a_init, v_init[subtable_index], F::zero(), None))
            .collect()
    }
    fn final_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..preprocessing.num_memories)
            .map(|memory_index| {
                (
                    a_init,
                    v_init[preprocessing.memory_to_subtable_index[memory_index]],
                    openings.final_openings[memory_index],
                    None,
                )
            })
            .collect()
    }
}


#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct InstructionReadWriteOpenings<F>
where
    F: JoltField,
{
    /// Evaluations of the dim_i polynomials at the opening point. Vector is of length C.
    pub(crate) dim_openings: Vec<F>,
    /// Evaluations of the read_cts_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    pub(crate) read_openings: Vec<F>,
    /// Evaluations of the E_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    pub(crate) E_poly_openings: Vec<F>,
    /// Evaluations of the flag polynomials at the opening point. Vector is of length NUM_INSTRUCTIONS.
    pub(crate) flag_openings: Vec<F>,
}

impl<F, C> StructuredOpeningProof<F, C, LassoPolynomials<F, C>> for InstructionReadWriteOpenings<F>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Proof = C::BatchedProof;
    type Preprocessing = NoPreprocessing;

    #[tracing::instrument(skip_all, name = "InstructionReadWriteOpenings::open")]
    fn open(polynomials: &LassoPolynomials<F, C>, opening_point: &[F]) -> Self {
        // All of these evaluations share the lagrange basis polynomials.
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();

        let dim_openings = polynomials
            .dim
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chis))
            .collect();
        let read_openings = polynomials
            .read_cts
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chis))
            .collect();
        let E_poly_openings = polynomials
            .E_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chis))
            .collect();
        let flag_openings = polynomials
            .instruction_flag_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chis))
            .collect();

        Self {
            dim_openings,
            read_openings,
            E_poly_openings,
            flag_openings,
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &LassoPolynomials<F, C>,
        opening_point: &[F],
        openings: &Self,
        setup: &C::Setup,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let read_write_polys = chain![
            polynomials.dim.iter(),
            polynomials.read_cts.iter(),
            polynomials.E_polys.iter(),
            polynomials.instruction_flag_polys.iter()
        ]
        .collect::<Vec<_>>();

        let read_write_openings: Vec<F> = [
            openings.dim_openings.as_slice(),
            openings.read_openings.as_slice(),
            openings.E_poly_openings.as_slice(),
            openings.flag_openings.as_slice(),
        ]
        .concat();

        C::batch_prove(
            &read_write_polys,
            opening_point,
            &read_write_openings,
            setup,
            BatchType::Big,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        generators: &C::Setup,
        opening_proof: &Self::Proof,
        commitment: &InstructionCommitment<C>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let read_write_openings: Vec<F> = [
            self.dim_openings.as_slice(),
            self.read_openings.as_slice(),
            self.E_poly_openings.as_slice(),
            self.flag_openings.as_slice(),
        ]
        .concat();
        C::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &read_write_openings,
            &commitment.trace_commitment[..read_write_openings.len()]
                .iter()
                .collect::<Vec<_>>(),
            transcript,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct InstructionFinalOpenings<F, Subtables>
where
    F: JoltField,
    Subtables: SubtableSet<F>,
{
    pub(crate) _subtables: PhantomData<Subtables>,
    /// Evaluations of the final_cts_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    pub(crate) final_openings: Vec<F>,
    /// Evaluation of the a_init/final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    pub(crate) a_init_final: Option<F>,
    /// Evaluation of the v_init/final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    pub(crate) v_init_final: Option<Vec<F>>,
}

impl<F, C, Subtables> StructuredOpeningProof<F, C, LassoPolynomials<F, C>>
    for InstructionFinalOpenings<F, Subtables>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
    Subtables: SubtableSet<F>,
{
    type Preprocessing = InstructionLookupsPreprocessing<F>;
    type Proof = C::BatchedProof;

    #[tracing::instrument(skip_all, name = "InstructionFinalOpenings::open")]
    fn open(polynomials: &LassoPolynomials<F, C>, opening_point: &[F]) -> Self {
        // All of these evaluations share the lagrange basis polynomials.
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let final_openings = polynomials
            .final_cts
            .par_iter()
            .map(|final_cts_i| final_cts_i.evaluate_at_chi_low_optimized(&chis))
            .collect();
        Self {
            _subtables: PhantomData,
            final_openings,
            a_init_final: None,
            v_init_final: None,
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &LassoPolynomials<F, C>,
        opening_point: &[F],
        openings: &Self,
        setup: &C::Setup,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        C::batch_prove(
            &polynomials.final_cts.iter().collect::<Vec<_>>(),
            opening_point,
            &openings.final_openings,
            setup,
            BatchType::Big,
            transcript,
        )
    }

    fn compute_verifier_openings(
        &mut self,
        _preprocessing: &Self::Preprocessing,
        opening_point: &[F],
    ) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
        self.v_init_final = Some(
            Subtables::iter()
                .map(|subtable| subtable.evaluate_mle(&opening_point))
                .collect(),
        );
    }

    fn verify_openings(
        &self,
        generators: &C::Setup,
        opening_proof: &Self::Proof,
        commitment: &InstructionCommitment<C>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        C::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &self.final_openings,
            &commitment.final_commitment.iter().collect::<Vec<_>>(),
            transcript,
        )
    }
}
