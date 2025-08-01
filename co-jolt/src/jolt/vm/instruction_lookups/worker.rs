use crate::{
    lasso::memory_checking::worker::MemoryCheckingProverRep3Worker,
    poly::{
        commitment::Rep3CommitmentScheme, opening_proof::Rep3ProverOpeningAccumulator,
        split_public_poly, Rep3MultilinearPolynomial, Rep3PolysConversion,
    },
    subprotocols::{
        grand_product::{Rep3BatchedDenseGrandProduct, Rep3BatchedGrandProductWorker},
        sparse_grand_product::Rep3ToggledBatchedGrandProduct,
    },
    utils::transcript::Transcript,
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    jolt::subtable::JoltSubtableSet,
    jolt::vm::instruction_lookups::InstructionLookupStuff,
    lasso::memory_checking::NoExogenousOpenings,
    poly::{
        compact_polynomial::SmallScalar,
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

        let log_num_workers = io_ctx.network.log_num_workers_per_party();

        let eq_poly_chunks = split_public_poly(&eq_poly, log_num_workers);
        let flag_poly_chunks = Rep3MultilinearPolynomial::split_poly_vec(
            &polynomials.instruction_lookups.instruction_flags,
            log_num_workers,
        );
        let memory_poly_chunks = Rep3MultilinearPolynomial::split_poly_vec(
            &polynomials.instruction_lookups.E_polys,
            log_num_workers,
        );
        let lookup_outputs_poly_chunks = Rep3MultilinearPolynomial::split_poly(
            &polynomials.instruction_lookups.lookup_outputs,
            log_num_workers,
        );
        let num_flag_polys = polynomials.instruction_lookups.instruction_flags.len();
        let num_memory_polys = polynomials.instruction_lookups.E_polys.len();

        let worker_polys = itertools::multizip((
            eq_poly_chunks,
            flag_poly_chunks,
            memory_poly_chunks,
            lookup_outputs_poly_chunks,
        ))
        .collect::<Vec<_>>();

        let (mut r_primary_sumchecks, eq_poly, flag_polys, E_polys, outputs_poly): (
            Vec<_>,
            MultilinearPolynomial<F>,
            Vec<_>,
            Vec<_>,
            Rep3MultilinearPolynomial<F>,
        ) =
            crate::utils::try_map_chunks_with_worker_subnets(
                worker_polys,
                io_ctx,
                1 << log_num_workers,
                |(eq_poly_chunk, flag_poly_chunk, memory_poly_chunk, lookup_outputs_poly_chunk),
                 io_ctx| {
                    Self::prove_primary_sumcheck(
                        preprocessing,
                        num_rounds - log_num_workers,
                        eq_poly_chunk,
                        memory_poly_chunk,
                        flag_poly_chunk,
                        lookup_outputs_poly_chunk,
                        io_ctx,
                    )
                },
            )
            .context("while proving primary sumcheck rounds")?
            .into_iter()
            .enumerate()
            .fold(
                (
                    vec![],
                    MultilinearPolynomial::from(vec![F::zero(); 1 << log_num_workers]),
                    vec![
                        Rep3MultilinearPolynomial::public_zero(1 << log_num_workers);
                        num_flag_polys
                    ],
                    vec![Rep3MultilinearPolynomial::shared(Default::default()); num_memory_polys],
                    Rep3MultilinearPolynomial::shared(Default::default()),
                ),
                |(_, mut eq_poly, mut flag_polys, mut E_polys, mut outputs_poly),
                 (
                    i,
                    (r_primary_sumcheck, eq_eval, flag_evals_chunk, E_evals_chunk, outputs_eval),
                )| {
                    eq_poly.as_dense_poly_mut().Z[i] = eq_eval;
                    flag_polys
                        .par_iter_mut()
                        .zip(flag_evals_chunk.into_par_iter())
                        .for_each(|(flag_poly, flag_eval)| {
                            flag_poly.as_public_mut().as_dense_poly_mut().Z[i] = flag_eval;
                        });
                    E_polys
                        .par_iter_mut()
                        .zip(E_evals_chunk.into_par_iter())
                        .for_each(|(E_poly, E_eval)| {
                            E_poly.as_shared_mut().evals.push(E_eval);
                        });
                    outputs_poly.as_shared_mut().evals.push(outputs_eval);
                    (
                        r_primary_sumcheck, // same for each worker
                        eq_poly,
                        flag_polys,
                        E_polys,
                        outputs_poly,
                    )
                },
            );

        let (flag_evals, E_evals, outputs_eval) = if log_num_workers > 0 {
            // Remaining sumcheck rounds
            let (r_primary_sumchecks_final, _, flag_evals, E_evals, outputs_eval) =
                Self::prove_primary_sumcheck(
                    preprocessing,
                    log_num_workers,
                    eq_poly,
                    E_polys,
                    flag_polys,
                    outputs_poly,
                    io_ctx,
                )
                .context("while proving remaining primary sumcheck rounds")?;
            r_primary_sumchecks.extend(r_primary_sumchecks_final);
            (flag_evals, E_evals, outputs_eval)
        } else {
            let flag_evals = flag_polys
                .iter()
                .map(|poly| poly.as_public().final_sumcheck_claim())
                .collect();
            let memory_evals = E_polys.iter().map(|poly| poly.as_shared()[0]).collect();
            (flag_evals, memory_evals, outputs_poly.as_shared()[0])
        };

        let r_primary_sumcheck = r_primary_sumchecks.into_iter().rev().collect::<Vec<_>>();

        let primary_sumcheck_polys = polynomials
            .instruction_lookups
            .E_polys
            .iter()
            .chain(polynomials.instruction_lookups.instruction_flags.iter())
            .chain([&polynomials.instruction_lookups.lookup_outputs].into_iter())
            .collect::<Vec<_>>();

        let primary_sumcheck_openings: Vec<F> = E_evals
            .into_iter()
            .map(|e| e.into_additive())
            .chain(
                flag_evals
                    .into_iter()
                    .map(|e| additive::promote_to_trivial_share(e, io_ctx.id)),
            )
            .chain(iter::once(outputs_eval.into_additive()))
            .collect();

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
    #[tracing::instrument(skip_all, name = "InstructionLookups::prove_primary_sumcheck", fields(worker_id = io_ctx.network.worker_idx()))]
    fn prove_primary_sumcheck(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        num_rounds: usize,
        mut eq_poly: MultilinearPolynomial<F>,
        mut memory_polys: Vec<Rep3MultilinearPolynomial<F>>,
        mut flag_polys: Vec<Rep3MultilinearPolynomial<F>>,
        mut lookup_outputs_poly: Rep3MultilinearPolynomial<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<(
        Vec<F>,
        F,
        Vec<F>,
        Vec<Rep3PrimeFieldShare<F>>,
        Rep3PrimeFieldShare<F>,
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

        let mut r: Vec<F> = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            let round_evaluations = Self::primary_sumcheck_prover_message(
                preprocessing,
                &eq_poly,
                &flag_polys,
                &mut memory_polys,
                &mut lookup_outputs_poly,
                io_ctx,
            )?;

            io_ctx.network.send_response(round_evaluations)?;

            let r_j = io_ctx
                .network
                .receive_request::<F>()
                .context("while receiving new claim")?;
            r.push(r_j);

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
            .map(|poly| poly.as_public().final_sumcheck_claim())
            .collect();
        let memory_evals = memory_polys
            .iter()
            .map(|poly| poly.as_shared()[0])
            .collect();
        let outputs_eval = lookup_outputs_poly.as_shared()[0];
        let eq_eval = eq_poly.final_sumcheck_claim();

        Ok((r, eq_eval, flag_evals, memory_evals, outputs_eval))
    }

    #[tracing::instrument(skip_all)]
    fn primary_sumcheck_prover_message(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        eq_poly: &MultilinearPolynomial<F>,
        flag_polys: &[Rep3MultilinearPolynomial<F>],
        subtable_polys: &[Rep3MultilinearPolynomial<F>],
        lookup_outputs_poly: &mut Rep3MultilinearPolynomial<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<Vec<F>> {
        let degree = Self::sumcheck_poly_degree();
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        let max_threads_per_worker = rayon::current_num_threads() / ((1 << io_ctx.network.log_num_workers_per_party()) * 3);
        let max_forks = std::cmp::min(max_threads_per_worker, 16);

        let evaluations: Vec<_> =
            crate::utils::try_fork_chunks(0..mle_half, io_ctx, max_forks, |i, io_ctx| {
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

                // let span = tracing::info_span!("instructions");
                // let _span_enter = span.enter();
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
                        // #[cfg(test)]
                        // {
                        //     let instruction_collation_eval_open =
                        //         rep3::arithmetic::open_vec::<F, _>(
                        //             &[instruction_collation_eval],
                        //             io_ctx,
                        //         )
                        //         .unwrap()[0];
                        //     let subtable_terms_open =
                        //         rep3::arithmetic::open_vec::<F, _>(&subtable_terms, io_ctx)
                        //             .unwrap();
                        //     let instruction_collation_eval_check =
                        //         instruction.combine_lookups(&subtable_terms_open, C, M);
                        //     assert_eq!(
                        //         instruction_collation_eval_check,
                        //         instruction_collation_eval_open,
                        //         "instruction {:?} combine_lookups_rep3 != combine_lookups",
                        //         Rep3JoltInstructionSet::<F>::name(&instruction).to_string()
                        //     );
                        // }

                        inner_sum[j] +=
                            rep3::arithmetic::mul_public(instruction_collation_eval, flag_eval);
                    }
                }
                // drop(_span_enter);
                // drop(span);

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
            })?
            .into_par_iter()
            .reduce_with(|a, b| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| *x + *y)
                    .collect::<Vec<F>>()
            })
            .unwrap_or(vec![F::zero(); degree]);

        // subtracing privious claim for each party/worker will break reconstraction of round poly,
        // so we let coordinator do it instead of workers
        // evaluations.insert(1, previous_claim - evaluations[0]);

        Ok(evaluations)
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
