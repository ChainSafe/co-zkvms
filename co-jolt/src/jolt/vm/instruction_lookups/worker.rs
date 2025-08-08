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
    utils::{
        element::SharedOrPublic,
        instruction_utils::{transpose_flatten, transpose_hashmap},
        transcript::Transcript,
    },
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::{chain, Itertools};
use jolt_core::{
    field::JoltField,
    jolt::{subtable::JoltSubtableSet, vm::instruction_lookups::InstructionLookupStuff},
    lasso::memory_checking::NoExogenousOpenings,
    poly::{
        compact_polynomial::SmallScalar,
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
    },
    utils::{math::Math, thread::drop_in_background_thread},
};
use mpc_core::protocols::rep3::{
    network::{IoContextPool, WorkerIoContext},
    Rep3PrimeFieldShare,
};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        self,
        network::{Rep3Network, Rep3NetworkWorker},
        PartyID,
    },
};
use mpc_net::mpc_star::MpcStarNetWorker;
use std::{collections::HashMap, iter::once, marker::PhantomData, sync::Arc};
use tracing::trace_span;

use super::{witness::Rep3InstructionLookupPolynomials, InstructionLookupsPreprocessing};
use crate::jolt::{
    instruction::{JoltInstructionSet, Rep3JoltInstructionSet},
    vm::{instruction_lookups::InstructionLookupsProof, witness::Rep3JoltPolynomials},
};

use rayon::{prelude::*, ThreadPoolBuilder};

type Rep3InstructionLookupOpenings<F> = InstructionLookupStuff<Rep3PrimeFieldShare<F>>;

use once_cell::sync::Lazy;
use rayon::ThreadPool;
pub static CPU_ONLY_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        // .num_threads(16) // tune
        .thread_name(|i| format!("cpu-only-{}", i))
        .build()
        .unwrap()
});

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
        preprocessing: &Arc<InstructionLookupsPreprocessing<C, F>>,
        polynomials: &mut Rep3JoltPolynomials<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        pcs_setup: &PCS::Setup,
        io_ctx: &mut IoContextPool<Network>,
    ) -> Result<()>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
    {
        let trace_length = polynomials.instruction_lookups.dim[0].len();
        let r_eq = io_ctx.network().receive_request::<Vec<F>>()?;

        let eq_evals: Vec<F> = EqPolynomial::evals(&r_eq);
        let eq_poly = MultilinearPolynomial::from(eq_evals);
        let num_rounds = trace_length.log_2();

        let (r_primary_sumchecks, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck(
                preprocessing,
                num_rounds,
                eq_poly,
                polynomials.instruction_lookups.instruction_flags.clone(),
                polynomials.instruction_lookups.E_polys.clone(),
                polynomials.instruction_lookups.lookup_outputs.clone(),
                io_ctx,
            )?;

        let r_primary_sumcheck = r_primary_sumchecks.into_iter().rev().collect::<Vec<_>>();

        let primary_sumcheck_polys = polynomials
            .instruction_lookups
            .E_polys
            .iter()
            .chain(polynomials.instruction_lookups.instruction_flags.iter())
            .chain([&polynomials.instruction_lookups.lookup_outputs].into_iter())
            .collect::<Vec<_>>();

        let primary_sumcheck_openings: Vec<F> =
            chain![E_evals, flag_evals, once(outputs_eval)].collect();

        let eq_primary_sumcheck = DensePolynomial::new(EqPolynomial::evals(&r_primary_sumcheck));
        opening_accumulator.append(
            &primary_sumcheck_polys,
            eq_primary_sumcheck,
            r_primary_sumcheck,
            &primary_sumcheck_openings,
            io_ctx.main(),
        )?;

        <Self as MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>>::prove_memory_checking(
            pcs_setup,
            preprocessing,
            &polynomials.instruction_lookups,
            &polynomials,
            opening_accumulator,
            io_ctx,
        )?;

        // drop polynomials that won't be used anymore
        drop_in_background_thread(std::mem::take(&mut polynomials.instruction_lookups.E_polys));
        drop_in_background_thread(std::mem::take(
            &mut polynomials.instruction_lookups.read_cts,
        ));
        drop_in_background_thread(std::mem::take(
            &mut polynomials.instruction_lookups.final_cts,
        ));
        polynomials
            .instruction_lookups
            .instruction_flags
            .par_iter_mut()
            .for_each(|poly| match poly {
                Rep3MultilinearPolynomial::Public { trivial_share, .. } => {
                    drop_in_background_thread(trivial_share.take());
                }
                _ => unreachable!(),
            });

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "InstructionLookups::prove_primary_sumcheck")]
    fn prove_primary_sumcheck(
        preprocessing: &Arc<InstructionLookupsPreprocessing<C, F>>,
        num_rounds: usize,
        eq_poly: MultilinearPolynomial<F>,
        instruction_flags: Vec<Rep3MultilinearPolynomial<F>>,
        E_polys: Vec<Rep3MultilinearPolynomial<F>>,
        lookup_outputs_poly: Rep3MultilinearPolynomial<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<(
        Vec<AdditiveShare<F>>,
        Vec<AdditiveShare<F>>,
        Vec<AdditiveShare<F>>,
        AdditiveShare<F>,
    )> {
        let log_num_workers = io_ctx.log_num_workers_per_party();

        let num_flag_polys = instruction_flags.len();

        let E_polys_zero = E_polys
            .iter()
            .map(|poly| match poly {
                Rep3MultilinearPolynomial::Public { .. } => {
                    Rep3MultilinearPolynomial::public_zero(1 << log_num_workers)
                }
                Rep3MultilinearPolynomial::Shared(_) => {
                    Rep3MultilinearPolynomial::from_shared_bound_coeffs(
                        vec![Rep3PrimeFieldShare::zero_share(); 1 << log_num_workers],
                    )
                }
            })
            .collect();

        let (eq_poly_chunks, flag_poly_chunks, E_poly_chunks, lookup_outputs_poly_chunks) =
            tracing::info_span!("split_polys").in_scope(|| {
                let eq_poly_chunks = split_public_poly(eq_poly, log_num_workers);
                let flag_poly_chunks =
                    Rep3MultilinearPolynomial::split_poly_vec(instruction_flags, log_num_workers);
                let E_poly_chunks =
                    Rep3MultilinearPolynomial::split_poly_vec(E_polys, log_num_workers);
                let lookup_outputs_poly_chunks =
                    Rep3MultilinearPolynomial::split_poly(lookup_outputs_poly, log_num_workers);
                (
                    eq_poly_chunks,
                    flag_poly_chunks,
                    E_poly_chunks,
                    lookup_outputs_poly_chunks,
                )
            });

        let worker_polys = itertools::multizip((
            eq_poly_chunks,
            flag_poly_chunks,
            E_poly_chunks,
            lookup_outputs_poly_chunks,
        ))
        .collect::<Vec<_>>();

        let span = tracing::info_span!("prove_primary_sumcheck_distributed");
        let _span_enter = span.enter();
        let (mut r_primary_sumchecks, eq_poly, flag_polys, E_polys, outputs_poly): (
            Vec<_>,
            MultilinearPolynomial<F>,
            Vec<_>,
            Vec<_>,
            Rep3MultilinearPolynomial<F>,
        ) = worker_polys
            .into_par_iter()
            .zip(io_ctx.workers.par_iter_mut())
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(
                |(
                    (eq_poly_chunk, flag_poly_chunk, E_poly_chunk, lookup_outputs_poly_chunk),
                    io_ctx,
                )| {
                    Self::prove_primary_sumcheck_inner(
                        preprocessing,
                        num_rounds - log_num_workers,
                        eq_poly_chunk,
                        E_poly_chunk,
                        flag_poly_chunk,
                        lookup_outputs_poly_chunk,
                        io_ctx,
                    )
                },
            )
            .collect::<eyre::Result<Vec<_>>>()?
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
                    E_polys_zero,
                    Rep3MultilinearPolynomial::from_shared_bound_coeffs(vec![
                        Rep3PrimeFieldShare::zero_share();
                        1 << log_num_workers
                    ]),
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
                            flag_poly.as_public_mut().as_dense_poly_mut().Z[i] =
                                flag_eval.as_public();
                        });
                    E_polys
                        .par_iter_mut()
                        .zip(E_evals_chunk.into_par_iter())
                        .for_each(|(E_poly, E_eval)| {
                            E_poly.set_bound_coeff(i, E_eval);
                        });
                    outputs_poly.set_bound_coeff(i, outputs_eval.into());
                    (
                        r_primary_sumcheck, // same for each worker
                        eq_poly,
                        flag_polys,
                        E_polys,
                        outputs_poly,
                    )
                },
            );
        drop(_span_enter);
        drop(span);

        let span = tracing::info_span!("prove_primary_sumcheck_remaining");
        let _span_enter = span.enter();
        let (flag_evals, E_evals, outputs_eval) = if log_num_workers > 0 {
            // Remaining sumcheck rounds
            let (r_primary_sumchecks_final, _, flag_evals, E_evals, outputs_eval) =
                Self::prove_primary_sumcheck_inner(
                    preprocessing,
                    log_num_workers,
                    eq_poly,
                    E_polys,
                    flag_polys,
                    outputs_poly,
                    io_ctx.worker(0),
                )
                .context("while proving remaining primary sumcheck rounds")?;
            r_primary_sumchecks.extend(r_primary_sumchecks_final);
            let flag_evals = flag_evals
                .par_iter()
                .map(|eval| eval.into_additive(io_ctx.id))
                .collect();

            let E_evals = E_evals
                .par_iter()
                .map(|e| e.into_additive(io_ctx.id))
                .collect();
            (flag_evals, E_evals, outputs_eval.into_additive())
        } else {
            let flag_evals = flag_polys
                .par_iter()
                .map(|poly| poly.final_sumcheck_claim().into_additive(io_ctx.id))
                .collect();
            let E_evals = E_polys
                .par_iter()
                .map(|poly| poly.final_sumcheck_claim().into_additive(io_ctx.id))
                .collect();
            (
                flag_evals,
                E_evals,
                outputs_poly.final_sumcheck_claim().into_additive(io_ctx.id),
            )
        };
        drop(_span_enter);
        drop(span);

        Ok((r_primary_sumchecks, flag_evals, E_evals, outputs_eval))
    }

    #[tracing::instrument(skip_all, name = "InstructionLookups::prove_primary_sumcheck_inner", fields(worker_id = io_ctx.network().worker_idx()))]
    fn prove_primary_sumcheck_inner(
        preprocessing: &Arc<InstructionLookupsPreprocessing<C, F>>,
        num_rounds: usize,
        mut eq_poly: MultilinearPolynomial<F>,
        mut E_polys: Vec<Rep3MultilinearPolynomial<F>>,
        mut flag_polys: Vec<Rep3MultilinearPolynomial<F>>,
        mut lookup_outputs_poly: Rep3MultilinearPolynomial<F>,
        io_ctx: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<(
        Vec<F>,
        F,
        Vec<SharedOrPublic<F>>,
        Vec<SharedOrPublic<F>>,
        Rep3PrimeFieldShare<F>,
    )> {
        // Check all polys are the same size
        let poly_len = eq_poly.len();
        E_polys
            .iter()
            .for_each(|E_poly| debug_assert_eq!(E_poly.len(), poly_len));
        flag_polys
            .iter()
            .for_each(|flag_poly| debug_assert_eq!(flag_poly.len(), poly_len));
        debug_assert_eq!(lookup_outputs_poly.len(), poly_len);

        let mut r: Vec<F> = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            let r_j = rayon::scope(|_| {
                let round_evaluations = Self::primary_sumcheck_prover_message(
                    preprocessing,
                    &eq_poly,
                    &flag_polys,
                    &mut E_polys,
                    &lookup_outputs_poly,
                    io_ctx,
                )?;

                let span = tracing::trace_span!("coordinator_io");
                let _span_enter = span.enter();
                io_ctx.network().send_response(round_evaluations)?;

                io_ctx
                    .network()
                    .receive_request::<F>()
                    .context("while receiving new claim")
            })?;

            r.push(r_j);

            // Bind all polys
            let _bind_span = trace_span!("bind polys");
            let _bind_enter = _bind_span.enter();
            let (tx, rx) = tokio::sync::oneshot::channel();
            CPU_ONLY_POOL.spawn(move || {
                flag_polys
                    .par_iter_mut()
                    .chain(E_polys.par_iter_mut())
                    .chain(rayon::iter::once(&mut lookup_outputs_poly))
                    .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
                eq_poly.bind(r_j, BindingOrder::LowToHigh);
                tx.send((flag_polys, E_polys, lookup_outputs_poly, eq_poly))
                    .unwrap();
            });
            (flag_polys, E_polys, lookup_outputs_poly, eq_poly) = rx.blocking_recv().unwrap();

            drop(_bind_enter);
        }

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();

        let flag_evals = flag_polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();
        let E_evals = E_polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();
        let outputs_eval = lookup_outputs_poly.final_sumcheck_claim().as_shared();
        let eq_eval = eq_poly.final_sumcheck_claim();

        Ok((r, eq_eval, flag_evals, E_evals, outputs_eval))
    }

    #[tracing::instrument(skip_all, fields(party_id = io_ctx.party_idx()))]
    fn primary_sumcheck_prover_message(
        preprocessing: &Arc<InstructionLookupsPreprocessing<C, F>>,
        eq_poly: &MultilinearPolynomial<F>,
        flag_polys: &[Rep3MultilinearPolynomial<F>],
        subtable_polys: &[Rep3MultilinearPolynomial<F>],
        lookup_outputs_poly: &Rep3MultilinearPolynomial<F>,
        io_ctx: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<Vec<F>> {
        let party_id = io_ctx.party_id();
        let degree = Self::sumcheck_poly_degree();
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        let precomputed_evals = if io_ctx.total_num_workers() > 1 {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let preprocessing = preprocessing.clone();
            let eq_poly = eq_poly.clone();
            let flag_polys = Rep3MultilinearPolynomial::clone_with_bound_coeffs_vec(flag_polys);
            let subtable_polys = subtable_polys.to_vec();
            let lookup_outputs_poly = lookup_outputs_poly.clone();

            CPU_ONLY_POOL.spawn(move || {
                let evals = Self::precompute_evals(
                    mle_half,
                    &preprocessing,
                    &eq_poly,
                    &flag_polys,
                    &subtable_polys,
                    &lookup_outputs_poly,
                    party_id,
                );
                tx.send(evals).unwrap();
            });
            rx.blocking_recv()
                .context("while receiving precomputed evals")?
        } else {
            Self::precompute_evals(
                mle_half,
                &preprocessing,
                &eq_poly,
                &flag_polys,
                &subtable_polys,
                &lookup_outputs_poly,
                party_id,
            )
        };

        let evaluations: Vec<_> = io_ctx
            .par_chunks(precomputed_evals, None, |chunk, io_ctx| {
                let chunk_size = chunk.len();
                let (
                    mle_indices,                            // [i: mle_index]
                    eq_evals,                               // [i: [degree: eq_eval]]
                    output_evals,                           // [i: [degree: output_eval]]
                    flag_evals,        // [i: [instruction_index: [degree: flag_eval]]]
                    used_flag_indices, // [i: [instruction_index: [idx for flag_evals[i][instruction_index][idx] != 0]]]
                    subtable_eval_batches_per_mem_by_instr, // [i: instruction_index -> [memory_index: [subtable_eval for used_flag_indices[i][instruction_index]]]]
                ): (
                    Vec<usize>,
                    Vec<Vec<F>>,
                    Vec<Vec<Rep3PrimeFieldShare<F>>>,
                    Vec<Vec<Vec<F>>>,
                    Vec<Vec<Vec<usize>>>,
                    Vec<HashMap<usize, Vec<Vec<Rep3PrimeFieldShare<F>>>>>,
                ) = chunk.into_iter().multiunzip();

                // `subtable_eval_batches_per_mem_batches_by_instr`: instruction_index -> [i: [memory_index: [idx: subtable_eval]]]
                // subtable_eval_batches_per_mem_batches_by_instr` doesn't align by mle_index i.e. an instruction may be inactive for some mle_index i.e. mle_index in tuple != position in vector
                let mut subtable_eval_batches_per_mem_batches_by_instr =
                    transpose_hashmap(subtable_eval_batches_per_mem_by_instr);

                let mut inner_sums =
                    vec![vec![Rep3PrimeFieldShare::zero_share(); degree]; chunk_size];

                for (instruction_index, instruction) in InstructionSet::iter().enumerate() {
                    if let Some(subtable_eval_batches_per_mem_batches) =
                        subtable_eval_batches_per_mem_batches_by_instr.remove(&instruction_index)
                    {
                        let subtable_eval_greater_batches_per_mem =
                            transpose_flatten(subtable_eval_batches_per_mem_batches);

                        // instruction_collation_evals: [i * degree: instruction_collation_eval]
                        let instruction_collation_evals = instruction
                            .combine_lookups_rep3_batched(
                                subtable_eval_greater_batches_per_mem,
                                C,
                                M,
                                io_ctx,
                            )?;

                        let mut offset = 0;

                        mle_indices.iter().enumerate().for_each(|(i, _)| {
                            used_flag_indices[i][instruction_index]
                                .iter()
                                .enumerate()
                                .for_each(|(index_in_terms_batch, &degree_index)| {
                                    let degrees_used =
                                        used_flag_indices[i][instruction_index].len();
                                    if degrees_used > 0 {
                                        inner_sums[i][degree_index] += rep3::arithmetic::mul_public(
                                            instruction_collation_evals
                                                [offset + index_in_terms_batch],
                                            flag_evals[i][instruction_index][degree_index],
                                        );
                                    }
                                });
                            offset += used_flag_indices[i][instruction_index].len();
                        });
                    }
                }

                let evaluations_in_chunk: Vec<Vec<F>> = (0..chunk_size)
                    .map(|i| {
                        (0..degree)
                            .map(|eval_index| {
                                rep3::arithmetic::mul_public(
                                    inner_sums[i][eval_index] - output_evals[i][eval_index],
                                    eq_evals[i][eval_index],
                                )
                                .into_additive()
                            })
                            .collect()
                    })
                    .collect();

                Ok(evaluations_in_chunk)
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

    fn precompute_evals(
        mle_half: usize,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        eq_poly: &MultilinearPolynomial<F>,
        flag_polys: &[Rep3MultilinearPolynomial<F>],
        subtable_polys: &[Rep3MultilinearPolynomial<F>],
        lookup_outputs_poly: &Rep3MultilinearPolynomial<F>,
        party_id: PartyID,
    ) -> Vec<(
        usize,
        Vec<F>,
        Vec<Rep3PrimeFieldShare<F>>,
        Vec<Vec<F>>,
        Vec<Vec<usize>>,
        HashMap<usize, Vec<Vec<Rep3PrimeFieldShare<F>>>>,
    )> {
        let degree = Self::sumcheck_poly_degree();

        (0..mle_half)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let output_evals = lookup_outputs_poly.as_shared().sumcheck_evals(
                    i,
                    degree,
                    BindingOrder::LowToHigh,
                );
                // flag_evals: [[flag_eval; degree]; flag_poly_index]
                let flag_evals: Vec<Vec<F>> = flag_polys
                    .iter()
                    .map(|poly| {
                        poly.as_public()
                            .sumcheck_evals(i, degree, BindingOrder::LowToHigh)
                    })
                    .collect();
                // Subtable evals are lazily computed in the for-loop below
                let mut subtable_evals: Vec<Vec<_>> = vec![vec![]; subtable_polys.len()];

                // used_flag_indices: [[degree index where instruction is used]; flag_poly_index]
                let used_flag_indices: Vec<Vec<usize>> = flag_evals
                    .iter()
                    .map(|evals| evals.iter().positions(|eval| !eval.is_zero()).collect())
                    .collect::<Vec<_>>();

                // instruction_index -> [[subtable_eval; memory_index]; degree]
                let used_subtable_terms_batches_per_instruction: HashMap<usize, Vec<_>> =
                    InstructionSet::iter()
                        .filter_map(|instruction| {
                            let instruction_index =
                                <InstructionSet as Rep3JoltInstructionSet<F>>::enum_index(
                                    &instruction,
                                );
                            let memory_indices =
                                &preprocessing.instruction_to_memory_indices[instruction_index];

                            if used_flag_indices[instruction_index].is_empty() {
                                return None;
                            }

                            let toggled_subtable_terms_batches: Vec<Vec<Rep3PrimeFieldShare<F>>> =
                                memory_indices
                                    .iter()
                                    .map(|memory_index| {
                                        if !used_flag_indices[instruction_index].is_empty()
                                            && subtable_evals[*memory_index].is_empty()
                                        {
                                            subtable_evals[*memory_index] = subtable_polys
                                                [*memory_index]
                                                .sumcheck_evals_into_share(
                                                    i,
                                                    degree,
                                                    BindingOrder::LowToHigh,
                                                    party_id,
                                                );
                                        }
                                        used_flag_indices[instruction_index]
                                            .iter()
                                            .map(|&j| subtable_evals[*memory_index][j])
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>();
                            Some((instruction_index, toggled_subtable_terms_batches))
                        })
                        .collect();

                (
                    i,
                    eq_evals,
                    output_evals,
                    flag_evals,
                    used_flag_indices,
                    used_subtable_terms_batches_per_instruction,
                )
            })
            .collect()
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
        io_ctx: &mut IoContextPool<Network>,
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
        let party_id = io_ctx.party_id();

        let read_write_leaves: Vec<_> = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let dim = polynomials.dim[dim_index].as_shared();
                let e_polys = &polynomials.E_polys[memory_index];
                let read_cts = &polynomials.read_cts[memory_index];

                let read_fingerprints: Vec<_> = (0..num_lookups)
                    .map(|i| {
                        let a = dim[i];
                        let v = e_polys.get_coeff(i);
                        let t = read_cts.get_coeff(i);
                        t.mul_public(gamma_squared)
                            .add(&v.mul_public(*gamma), party_id)
                            .add_shared(a, party_id)
                            .sub_public(&*tau, party_id)
                            .as_shared()
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
