use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Context;
use itertools::interleave;
use jolt_core::{
    lasso::memory_checking::MultisetHashes,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        field::JoltField,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::{
        grand_product::{
            BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
        },
        sumcheck::SumcheckInstanceProof,
    },
    utils::{
        math::Math,
        mul_0_1_optimized, split_poly_flagged,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};
use tracing::trace_span;

use super::{LassoPolynomials, LassoPreprocessing};
use crate::{instructions::LookupSet, subtables::SubtableSet};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct LassoProver<const C: usize, const M: usize, F: JoltField, Lookups, Subtables> {
    pub _marker: PhantomData<(F, Lookups, Subtables)>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LassoProof<const C: usize, const M: usize, F>
where
    F: JoltField,
{
    pub(crate) primary_sumcheck: PrimarySumcheck<F>,
    pub(crate) memory_checking: MemoryCheckingProof<
        F,
        // CS,
        // InstructionPolynomials<F, CS>,
        // InstructionReadWriteOpenings<F>,
        // InstructionFinalOpenings<F, Subtables>,
    >,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct PrimarySumcheck<F: JoltField> {
    pub(crate) sumcheck_proof: SumcheckInstanceProof<F>,
    pub(crate) num_rounds: usize,
    // openings: PrimarySumcheckOpenings<F>,
    // opening_proof: CS::BatchedProof,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCheckingProof<F>
where
    F: JoltField,
    // C: CommitmentScheme<Field = F>,
    // Polynomials: StructuredCommitment<C>,
    // ReadWriteOpenings: StructuredOpeningProof<F, C, Polynomials>,
    // InitFinalOpenings: StructuredOpeningProof<F, C, Polynomials>,
{
    /// Read/write/init/final multiset hashes for each memory
    pub multiset_hashes: MultisetHashes<F>,
    /// The read and write grand products for every memory has the same size,
    /// so they can be batched.
    pub read_write_grand_product: BatchedGrandProductArgument<F>,
    /// The init and final grand products for every memory has the same size,
    /// so they can be batched.
    pub init_final_grand_product: BatchedGrandProductArgument<F>,
    // /// The opening proofs associated with the read/write grand product.
    // pub read_write_openings: ReadWriteOpenings,
    // pub read_write_opening_proof: ReadWriteOpenings::Proof,
    // /// The opening proofs associated with the init/final grand product.
    // pub init_final_openings: InitFinalOpenings,
    // pub init_final_opening_proof: InitFinalOpenings::Proof,
}

type Preprocessing<F> = LassoPreprocessing<F>;
type Polynomials<F> = LassoPolynomials<F>;

impl<const C: usize, const M: usize, F: JoltField, Lookups, Subtables>
    LassoProver<C, M, F, Lookups, Subtables>
where
    Lookups: LookupSet<F>,
    Subtables: SubtableSet<F>,
{
    #[tracing::instrument(skip_all, name = "LassoProver::prove")]
    pub fn prove(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        transcript: &mut ProofTranscript,
    ) -> LassoProof<C, M, F> {
        let trace_length = polynomials.dims[0].len();
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
                &polynomials.e_polys,
                &polynomials.lookup_flag_polys,
                &mut polynomials.lookup_outputs.clone(),
                Self::sumcheck_poly_degree(),
                transcript,
            );

        let primary_sumcheck = PrimarySumcheck {
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            // openings: sumcheck_openings,
            // opening_proof: sumcheck_opening_proof,
        };

        let memory_checking = Self::prove_memory_checking(preprocessing, polynomials, transcript);

        LassoProof {
            primary_sumcheck,
            memory_checking,
        }
    }
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_memory_checking")]
    pub fn prove_memory_checking(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        // network: &mut N,
        transcript: &mut ProofTranscript,
    ) -> MemoryCheckingProof<F> {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products(preprocessing, polynomials, transcript);

        MemoryCheckingProof {
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
        }
    }

    fn prove_grand_products(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        // network: &mut N,
        transcript: &mut ProofTranscript,
    ) -> (
        BatchedGrandProductArgument<F>,
        BatchedGrandProductArgument<F>,
        MultisetHashes<F>,
        Vec<F>,
        Vec<F>,
    ) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar::<F>(b"Memory checking gamma");
        let tau: F = transcript.challenge_scalar::<F>(b"Memory checking tau");

        let (read_write_leaves, init_final_leaves) =
            Self::compute_leaves(preprocessing, polynomials, &gamma, &tau);

        let (read_write_circuit, read_write_hashes) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves);
        let (init_final_circuit, init_final_hashes) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves);

        let multiset_hashes =
            Self::uninterleave_hashes(preprocessing, read_write_hashes, init_final_hashes);
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let (read_write_grand_product, r_read_write) =
            BatchedGrandProductArgument::prove(read_write_circuit, transcript);
        let (init_final_grand_product, r_init_final) =
            BatchedGrandProductArgument::prove(init_final_circuit, transcript);

        (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        )
    }

    fn compute_leaves(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dims[0].len();

        let read_write_leaves = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];

                let read_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| {
                        let a = &polynomials.dims[dim_index][i];
                        let v = &polynomials.e_polys[memory_index][i];
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

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        read_write_leaves: Vec<DensePolynomial<F>>,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>) {
        assert_eq!(read_write_leaves.len(), 2 * preprocessing.num_memories);

        let _span = trace_span!("LassoLookups: construct circuits");
        let _enter = _span.enter();

        let memory_flag_polys =
            Self::memory_flag_polys(preprocessing, &polynomials.lookup_flag_bitvectors);

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
                    if preprocessing.lookup_to_memory_indices[instruction_index]
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

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::init_final_grand_product")]
    fn init_final_grand_product(
        _preprocessing: &Preprocessing<F>,
        _polynomials: &Polynomials<F>,
        init_final_leaves: Vec<DensePolynomial<F>>,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>) {
        let init_final_circuits: Vec<GrandProductCircuit<F>> = init_final_leaves
            .par_iter()
            .map(|leaves| GrandProductCircuit::new(leaves))
            .collect();
        let init_final_hashes: Vec<F> = init_final_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        (
            BatchedGrandProductCircuit::new_batch(init_final_circuits),
            init_final_hashes,
        )
    }

    fn uninterleave_hashes(
        _preprocessing: &Preprocessing<F>,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len() % 2, 0);
        let num_memories = read_write_hashes.len() / 2;

        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(num_memories);
        let mut final_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            init_hashes.push(init_final_hashes[2 * i]);
            final_hashes.push(init_final_hashes[2 * i + 1]);
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        _preprocessing: &Preprocessing<F>,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        let num_memories = multiset_hashes.read_hashes.len();
        assert_eq!(multiset_hashes.final_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.write_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.init_hashes.len(), Subtables::COUNT);
    }

    fn interleave_hashes(
        _preprocessing: &Preprocessing<F>,
        multiset_hashes: &MultisetHashes<F>,
    ) -> (Vec<F>, Vec<F>) {
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();
        let init_final_hashes = interleave(
            multiset_hashes.init_hashes.clone(),
            multiset_hashes.final_hashes.clone(),
        )
        .collect();

        (read_write_hashes, init_final_hashes)
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

    #[tracing::instrument(skip_all, name = "LassoProver::primary_sumcheck_inner_loop", level = "trace")]
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
                    let memory_indices = &preprocessing.lookup_to_memory_indices[instruction_index];

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
            let memory_indices = &preprocessing.lookup_to_memory_indices[instruction_index];
            let filtered_operands: Vec<F> = memory_indices.iter().map(|i| vals[*i]).collect();
            sum += flags[instruction_index] * instruction.combine_lookups(&filtered_operands, C, M);
        }

        sum
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

    pub fn verify(
        preprocessing: &Preprocessing<F>,
        proof: LassoProof<C, M, F>,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<()> {
        let _r_eq = transcript.challenge_vector::<F>(
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
        // let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_primary_sumcheck);
        // assert_eq!(
        //     eq_eval
        //         * (Self::combine_lookups(
        //             preprocessing,
        //             &proof.primary_sumcheck.openings.E_poly_openings,
        //             &proof.primary_sumcheck.openings.flag_openings,
        //         ) - proof.primary_sumcheck.openings.lookup_outputs_opening),
        //     claim_last,
        //     "Primary sumcheck check failed."
        // );

        // proof.primary_sumcheck.openings.verify_openings(
        //     generators,
        //     &proof.primary_sumcheck.opening_proof,
        //     commitment,
        //     &r_primary_sumcheck,
        //     transcript,
        // )?;

        Self::verify_memory_checking(preprocessing, proof.memory_checking, transcript)?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::verify")]
    pub fn verify_memory_checking(
        preprocessing: &Preprocessing<F>,
        mut proof: MemoryCheckingProof<F>,
        // commitments: &Polynomials::Commitment,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<()> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar(b"Memory checking gamma");
        let tau: F = transcript.challenge_scalar(b"Memory checking tau");

        // transcript.append_protocol_name(Self::protocol_name());

        Self::check_multiset_equality(preprocessing, &proof.multiset_hashes);
        proof.multiset_hashes.append_to_transcript(transcript);

        let (read_write_hashes, init_final_hashes) =
            Self::interleave_hashes(preprocessing, &proof.multiset_hashes);

        let (claims_read_write, r_read_write) = proof
            .read_write_grand_product
            .verify(&read_write_hashes, transcript)
            .context("while verifying read_write_grand_product")?;
        let (claims_init_final, r_init_final) = proof
            .init_final_grand_product
            .verify(&init_final_hashes, transcript)
            .context("while verifying init_final_grand_product")?;

        // proof.read_write_openings.verify_openings(
        //     generators,
        //     &proof.read_write_opening_proof,
        //     commitments,
        //     &r_read_write,
        //     transcript,
        // )?;
        // proof.init_final_openings.verify_openings(
        //     generators,
        //     &proof.init_final_opening_proof,
        //     commitments,
        //     &r_init_final,
        //     transcript,
        // )?;

        // proof
        //     .read_write_openings
        //     .compute_verifier_openings(&NoPreprocessing, &r_read_write);
        // proof
        //     .init_final_openings
        //     .compute_verifier_openings(preprocessing, &r_init_final);

        // Self::check_fingerprints(
        //     preprocessing,
        //     claims_read_write,
        //     claims_init_final,
        //     &proof.read_write_openings,
        //     &proof.init_final_openings,
        //     &gamma,
        //     &tau,
        // );

        Ok(())
    }
}
