use crate::{
    lasso::memory_checking::worker::MemoryCheckingProverRep3Worker,
    poly::{
        commitment::Rep3CommitmentScheme, opening_proof::Rep3ProverOpeningAccumulator,
        unipoly::CompressedUniPoly, Rep3DensePolynomial, Rep3MultilinearPolynomial,
        Rep3PolysConversion,
    },
    subprotocols::{
        grand_product::{Rep3BatchedDenseGrandProduct, Rep3BatchedGrandProductWorker},
        sparse_grand_product::Rep3ToggledBatchedGrandProduct,
    },
    utils::{
        split_rep3_poly_flagged,
        transcript::{KeccakTranscript, Transcript},
    },
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::{chain, Itertools};
use jolt_core::{
    field::JoltField,
    jolt::subtable::JoltSubtableSet,
    jolt::vm::instruction_lookups::InstructionLookupStuff,
    lasso::memory_checking::NoExogenousOpenings,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::{CompactPolynomial, SmallScalar},
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::UniPoly,
    },
    utils::{math::Math, mul_0_1_optimized},
};
use mpc_core::protocols::{additive, rep3::Rep3PrimeFieldShare};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        self,
        network::{IoContext, Rep3Network, Rep3NetworkWorker},
        PartyID,
    },
};
use mpc_net::mpc_star::MpcStarNetWorker;
use std::{iter, marker::PhantomData};
use tracing::trace_span;

use super::{witness::Rep3InstructionLookupPolynomials, InstructionLookupsPreprocessing};
use crate::jolt::{
    instruction::{JoltInstructionSet, Rep3JoltInstructionSet},
    vm::{
        instruction_lookups::InstructionLookupsProof, witness::Rep3JoltPolynomials, JoltPolynomials,
    },
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type Rep3InstructionLookupOpenings<F> = InstructionLookupStuff<Rep3PrimeFieldShare<F>>;

pub struct Rep3InstructionLookupsProver<
    const C: usize,
    const M: usize,
    F,
    Instructions,
    Subtables,
    Network,
> where
    F: JoltField,
    Network: Rep3Network,
{
    pub _marker: PhantomData<(F, Instructions, Subtables, Network)>,
}

impl<const C: usize, const M: usize, F, InstructionSet, Subtables, Network>
    Rep3InstructionLookupsProver<C, M, F, InstructionSet, Subtables, Network>
where
    F: JoltField,
    InstructionSet: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    Network: Rep3NetworkWorker,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "Rep3InstructionLookups::prove")]
    pub fn prove<PCS, ProofTranscript>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        polynomials: &mut Rep3JoltPolynomials<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        pcs_setup: &PCS::Setup,
        io_ctx: &mut IoContext<Network>,
    ) -> Result<()>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
    {
        let trace_length = polynomials.instruction_lookups.dim[0].len();
        let r_eq = io_ctx.network.receive_request::<Vec<F>>()?;

        let eq_evals: Vec<F> = EqPolynomial::evals(&r_eq);
        let eq_poly = MultilinearPolynomial::from(eq_evals);
        let num_rounds = trace_length.log_2();

        let (mut r_primary_sumcheck, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck(
                preprocessing,
                num_rounds,
                eq_poly,
                &mut polynomials.instruction_lookups.E_polys.clone(),
                &mut polynomials.instruction_lookups.instruction_flags.clone(),
                &mut polynomials.instruction_lookups.lookup_outputs.clone(),
                io_ctx,
            )?;
        r_primary_sumcheck = r_primary_sumcheck.into_iter().rev().collect();

        let primary_sumcheck_polys = polynomials
            .instruction_lookups
            .E_polys
            .iter()
            .chain(polynomials.instruction_lookups.instruction_flags.iter())
            .chain([&polynomials.instruction_lookups.lookup_outputs].into_iter())
            .collect::<Vec<_>>();

        let mut primary_sumcheck_openings: Vec<F> = [E_evals, flag_evals].concat();
        primary_sumcheck_openings.push(outputs_eval);

        let eq_primary_sumcheck = DensePolynomial::new(EqPolynomial::evals(&r_primary_sumcheck));
        opening_accumulator.append(
            &primary_sumcheck_polys,
            eq_primary_sumcheck,
            r_primary_sumcheck,
            &primary_sumcheck_openings,
            io_ctx,
        )?;

        <Self as MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>>::prove_memory_checking(
            pcs_setup,
            preprocessing,
            &polynomials.instruction_lookups,
            &polynomials,
            opening_accumulator,
            io_ctx,
        )?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "InstructionLookups::prove_primary_sumcheck")]
    fn prove_primary_sumcheck(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        num_rounds: usize,
        mut eq_poly: MultilinearPolynomial<F>,
        memory_polys: &mut [Rep3MultilinearPolynomial<F>],
        flag_polys: &mut [Rep3MultilinearPolynomial<F>],
        lookup_outputs_poly: &mut Rep3MultilinearPolynomial<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<(
        Vec<AdditiveShare<F>>,
        Vec<AdditiveShare<F>>,
        Vec<AdditiveShare<F>>,
        AdditiveShare<F>,
    )> {
        // Check all polys are the same size
        let poly_len = eq_poly.len();
        memory_polys
            .iter()
            .for_each(|E_poly| debug_assert_eq!(E_poly.len(), poly_len));
        flag_polys
            .iter()
            .for_each(|flag_poly| debug_assert_eq!(flag_poly.len(), poly_len));
        debug_assert_eq!(lookup_outputs_poly.len(), poly_len);

        let mut previous_claim = F::zero();
        let mut r: Vec<F> = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            let univariate_poly = Self::primary_sumcheck_prover_message(
                preprocessing,
                &eq_poly,
                &flag_polys,
                memory_polys,
                lookup_outputs_poly,
                previous_claim,
                io_ctx,
                _round,
            )?;

            io_ctx.network.send_response(univariate_poly.as_vec())?;

            let (r_j, new_claim) = io_ctx.network.receive_request::<(F, F)>()?;
            r.push(r_j);

            previous_claim = additive::promote_to_trivial_share(new_claim, io_ctx.id);

            // Bind all polys
            let _bind_span = trace_span!("bind");
            let _bind_enter = _bind_span.enter();
            flag_polys
                .par_iter_mut()
                .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
            eq_poly.bind(r_j, BindingOrder::LowToHigh);
            memory_polys
                .par_iter_mut()
                .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
            lookup_outputs_poly.bind(r_j, BindingOrder::LowToHigh);

            drop(_bind_enter);
        } // End rounds

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();
        let flag_evals = flag_polys
            .iter()
            .map(|poly| {
                additive::promote_to_trivial_share(
                    poly.as_public().final_sumcheck_claim(),
                    io_ctx.id,
                )
            })
            .collect();
        let memory_evals = memory_polys
            .iter()
            .map(|poly| poly.as_shared()[0].into_additive())
            .collect();
        let outputs_eval = lookup_outputs_poly.as_shared()[0].into_additive();

        Ok((r, flag_evals, memory_evals, outputs_eval))
    }

    #[tracing::instrument(skip_all, level = "trace")]
    fn primary_sumcheck_prover_message(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        eq_poly: &MultilinearPolynomial<F>,
        flag_polys: &[Rep3MultilinearPolynomial<F>],
        subtable_polys: &[Rep3MultilinearPolynomial<F>],
        lookup_outputs_poly: &mut Rep3MultilinearPolynomial<F>,
        previous_claim: F,
        io_ctx: &mut IoContext<Network>,
        round: usize,
    ) -> eyre::Result<UniPoly<F>> {
        let degree = Self::sumcheck_poly_degree();
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        let mut evaluations: Vec<_> = crate::utils::try_fork_chunks(
            0..mle_half,
            io_ctx,
            8, // TODO: make configurable
            |i, io_ctx| {
                let eq_evals = eq_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let output_evals = lookup_outputs_poly.as_shared().sumcheck_evals(
                    i,
                    degree,
                    BindingOrder::LowToHigh,
                );
                let flag_evals: Vec<Vec<F>> = flag_polys
                    .iter()
                    .map(|poly| {
                        poly.as_public()
                            .sumcheck_evals(i, degree, BindingOrder::LowToHigh)
                    })
                    .collect();
                // Subtable evals are lazily computed in the for-loop below
                let mut subtable_evals: Vec<Vec<_>> = vec![vec![]; subtable_polys.len()];

                let mut inner_sum = vec![Rep3PrimeFieldShare::zero_share(); degree];
                for instruction in InstructionSet::iter() {
                    let instruction_index =
                        <InstructionSet as Rep3JoltInstructionSet<F>>::enum_index(&instruction);
                    let memory_indices =
                        &preprocessing.instruction_to_memory_indices[instruction_index];

                    for j in 0..degree {
                        let flag_eval = flag_evals[instruction_index][j];
                        if flag_eval.is_zero() {
                            continue;
                        }; // Early exit if no contribution.

                        let subtable_terms: Vec<_> = memory_indices
                            .iter()
                            .map(|memory_index| {
                                if subtable_evals[*memory_index].is_empty() {
                                    subtable_evals[*memory_index] = subtable_polys[*memory_index]
                                        .as_shared()
                                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                                }
                                subtable_evals[*memory_index][j]
                            })
                            .collect();

                        let instruction_collation_eval =
                            instruction.combine_lookups_rep3(&subtable_terms, C, M, io_ctx)?;
                        #[cfg(feature = "debug")]
                        {
                            let instruction_collation_eval_open =
                                rep3::arithmetic::open_vec::<F, _>(
                                    &[instruction_collation_eval],
                                    io_ctx,
                                )
                                .unwrap()[0];
                            let subtable_terms_open =
                                rep3::arithmetic::open_vec::<F, _>(&subtable_terms, io_ctx)
                                    .unwrap();
                            let instruction_collation_eval_check =
                                instruction.combine_lookups(&subtable_terms_open, C, M);
                            assert_eq!(
                                instruction_collation_eval_check,
                                instruction_collation_eval_open,
                                "instruction {:?} combine_lookups_rep3 != combine_lookups",
                                Rep3JoltInstructionSet::<F>::name(&instruction).to_string()
                            );
                        }

                        inner_sum[j] +=
                            rep3::arithmetic::mul_public(instruction_collation_eval, flag_eval);
                    }
                }

                let evaluations: Vec<_> = (0..degree)
                    .map(|eval_index| {
                        rep3::arithmetic::mul_public(
                            inner_sum[eval_index] - output_evals[eval_index],
                            eq_evals[eval_index],
                        )
                        .into_additive()
                    })
                    .collect();
                Ok(evaluations)
            },
        )?
        .into_par_iter()
        .reduce(
            || vec![F::zero(); degree],
            |running, new| {
                debug_assert_eq!(running.len(), new.len());
                running
                    .iter()
                    .zip(new.iter())
                    .map(|(r, n)| *r + *n)
                    .collect()
            },
        );

        evaluations.insert(1, previous_claim - evaluations[0]);
        Ok(UniPoly::from_evals(&evaluations))
    }

    /// Returns the sumcheck polynomial degree for the "primary" sumcheck. Since the primary sumcheck expression
    /// is \sum_x \tilde{eq}(r, x) * \sum_i flag_i(x) * g_i(E_1(x), ..., E_\alpha(x)), the degree is
    /// the max over all the instructions' `g_i` polynomial degrees, plus two (one for \tilde{eq}, one for flag_i)
    fn sumcheck_poly_degree() -> usize {
        InstructionSet::iter()
            .map(|lookup| lookup.g_poly_degree(C))
            .max()
            .unwrap()
            + 2 // eq and flag
    }
}

impl<
        F,
        const C: usize,
        const M: usize,
        PCS,
        ProofTranscript,
        InstructionSet,
        Subtables,
        Network,
    > MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>
    for Rep3InstructionLookupsProver<C, M, F, InstructionSet, Subtables, Network>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    InstructionSet: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    Network: Rep3NetworkWorker,
{
    type Rep3Polynomials = Rep3InstructionLookupPolynomials<F>;
    type Preprocessing = InstructionLookupsPreprocessing<C, F>;

    type ReadWriteGrandProduct = Rep3ToggledBatchedGrandProduct<F>;
    type InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    type Openings = InstructionLookupStuff<F>;

    // type Commitments;

    type ExogenousOpenings = NoExogenousOpenings;

    #[tracing::instrument(skip_all, name = "Rep3InstructionLookupsProver::compute_leaves")]
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        _jolt_polynomials: &Rep3JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
        io_ctx: &mut IoContext<Network>,
    ) -> Result<(
        <Self::ReadWriteGrandProduct as Rep3BatchedGrandProductWorker<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::Leaves,
        <Self::InitFinalGrandProduct as Rep3BatchedGrandProductWorker<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::Leaves,
    )> {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dim[0].len();
        let party_id = io_ctx.network.party_id();

        let read_write_leaves: Vec<_> = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let dim = polynomials.dim[dim_index].as_shared();
                let e_polys = polynomials.E_polys[memory_index].as_shared();
                let read_cts = polynomials.read_cts[memory_index].as_shared();

                let read_fingerprints: Vec<_> = (0..num_lookups)
                    .map(|i| {
                        let a = &dim[i];
                        let v = &e_polys[i];
                        let t = &read_cts[i];
                        rep3::arithmetic::sub_shared_by_public(
                            (t * gamma_squared) + (v * *gamma) + *a,
                            *tau,
                            party_id,
                        )
                    })
                    .collect();
                let write_fingerprints: Vec<Rep3PrimeFieldShare<F>> = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| {
                        rep3::arithmetic::add_public(*read_fingerprint, gamma_squared, party_id)
                    })
                    .collect();
                [read_fingerprints, write_fingerprints]
            })
            .collect();

        let init_final_leaves: Vec<_> = preprocessing
            .materialized_subtables
            .par_iter()
            .enumerate()
            .flat_map_iter(|(subtable_index, subtable)| {
                let mut leaves =
                    vec![
                        Rep3PrimeFieldShare::zero_share();
                        M * (preprocessing.subtable_to_memory_indices[subtable_index].len() + 1)
                    ];
                // Init leaves
                (0..M).for_each(|i| {
                    let a = &F::from_u16(i as u16);
                    let v: u32 = subtable[i];
                    // let t = F::zero();
                    // Compute h(a,v,t) where t == 0
                    leaves[i] = rep3::arithmetic::promote_to_trivial_share(
                        party_id,
                        v.field_mul(*gamma) + *a - *tau,
                    );
                });

                // Final leaves
                let mut leaf_index = M;
                for memory_index in &preprocessing.subtable_to_memory_indices[subtable_index] {
                    let final_cts = &polynomials.final_cts[*memory_index].as_shared();
                    (0..M).for_each(|i| {
                        leaves[leaf_index] =
                            leaves[i] + rep3::arithmetic::mul_public(final_cts[i], gamma_squared);
                        leaf_index += 1;
                    });
                }

                leaves
            })
            .collect();

        let memory_flags = InstructionLookupsProof::<
            C,
            M,
            F,
            PCS,
            InstructionSet,
            Subtables,
            ProofTranscript,
        >::memory_flag_indices(
            preprocessing,
            polynomials
                .instruction_flags
                .try_into_public()
                .into_iter()
                .map(|p| p.try_into().unwrap())
                .collect(),
        );

        Ok((
            (memory_flags, read_write_leaves),
            (
                init_final_leaves,
                // # init = # subtables; # final = # memories
                Subtables::COUNT + preprocessing.num_memories,
            ),
        ))
    }
}
