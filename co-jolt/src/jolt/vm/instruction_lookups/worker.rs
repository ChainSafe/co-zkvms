use crate::{
    jolt::vm::instruction_lookups::witness,
    lasso::memory_checking::worker::MemoryCheckingProverRep3Worker,
    poly::{
        commitment::Rep3CommitmentScheme,
        opening_proof::{Rep3OpeningAccumulatorWorker, Rep3ProverOpening},
        Rep3MultilinearPolynomial, Rep3PolysConversion,
    },
    subprotocols::{
        grand_product::{Rep3BatchedDenseGrandProduct, Rep3BatchedGrandProductWorker},
        sparse_grand_product::Rep3ToggledBatchedGrandProduct,
    },
    utils::{transcript::Transcript, transpose_flatten, transpose_hashmap, types::Rep3Value},
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::{chain, Itertools};
use jolt_core::{
    jolt::{
        subtable::JoltSubtableSet,
        vm::{instruction_lookups::InstructionLookupStuff, JoltStuff},
    },
    lasso::memory_checking::{NoExogenousOpenings, StructuredPolynomialData},
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
use mpc_core::protocols::{
    additive,
    rep3::{network::IoContextPool, Rep3PrimeFieldShare},
};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        self,
        network::{Rep3Network, Rep3NetworkWorker},
        PartyID,
    },
};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
use tracing::trace_span;

use super::{witness::Rep3InstructionLookupPolynomials, InstructionLookupsPreprocessing};
use crate::field::JoltField;
use crate::jolt::{
    instruction::Rep3JoltInstructionSet,
    vm::{instruction_lookups::InstructionLookupsProof, witness::Rep3JoltPolynomials},
};

use rayon::{iter::once, prelude::*, ThreadPoolBuilder};

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
    InstructionSet: Rep3JoltInstructionSet,
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
        opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        pcs_setup: &PCS::Setup,
        io_ctx: &mut IoContextPool<Network>,
    ) -> Result<()>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
    {
        let r_eq = io_ctx.network().receive_request::<Vec<F>>()?;
        let num_rounds = r_eq.len() - io_ctx.log_num_workers();

        let worker_idx = io_ctx.worker_idx();
        let eq_chunk_size = 1usize << num_rounds;
        let eq_evals: Vec<F> = EqPolynomial::evals(&r_eq)
            .drain(worker_idx * eq_chunk_size..(worker_idx + 1) * eq_chunk_size)
            .collect();
        let eq_poly = MultilinearPolynomial::from(eq_evals);

        // let mut instruction_flags = polynomials
        //     .instruction_lookups
        //     .instruction_flags
        //     .iter()
        //     .map(|p| Rep3MultilinearPolynomial::poly_shard_for_worker(&p, num_rounds, worker_idx))
        //     .collect::<Vec<_>>();

        let r_primary_sumchecks = Self::prove_primary_sumcheck(
            preprocessing,
            num_rounds,
            eq_poly,
            &mut polynomials.instruction_lookups.instruction_flags,
            &mut polynomials.instruction_lookups.E_polys,
            &mut polynomials.instruction_lookups.lookup_outputs,
            io_ctx,
        )?;

        let r_primary_sumcheck = r_primary_sumchecks.into_iter().rev().collect::<Vec<_>>();

        let primary_sumcheck_polys = chain![
            &polynomials.instruction_lookups.E_polys,
            &polynomials.instruction_lookups.instruction_flags,
            [&polynomials.instruction_lookups.lookup_outputs]
        ]
        .collect::<Vec<_>>();

        let eq_primary_sumcheck = DensePolynomial::new(EqPolynomial::evals(&r_primary_sumcheck));
        opening_accumulator.append_sharded(
            &primary_sumcheck_polys,
            eq_primary_sumcheck,
            r_primary_sumcheck,
            io_ctx.main(),
        )?;

        let read_memories = witness::read_write_memories_for_worker(
            preprocessing.num_memories,
            io_ctx.num_workers(),
            io_ctx.worker_idx(),
        );
        polynomials.instruction_lookups.E_polys =
            std::mem::take(&mut polynomials.instruction_lookups.E_polys)
                .into_iter()
                .enumerate()
                .filter_map(|(i, p)| read_memories.contains(&i).then_some(p.as_full_poly()))
                .collect();

        // println!(
        //     "instruction_flags {}",
        //     polynomials.instruction_lookups.instruction_flags.len()
        // );

        // println!(
        //     "instruction_to_memory_indices {:?}",
        //     preprocessing.instruction_to_memory_indices
        // );

        // polynomials.instruction_lookups.instruction_flags =
        //     std::mem::take(&mut polynomials.instruction_lookups.instruction_flags)
        //         .into_iter()
        //         .enumerate()
        //         .filter_map(|(i, p)| {
        //             // preprocessing.instruction_to_memory_indices[i]
        //             read_memories
        //                 .as_ref()
        //                 .is_some_and(|m| m.contains(&i))
        //                 .then_some(p)
        //         })
        //         .collect();

        // println!(
        //     "instruction_flags worker {}: {}",
        //     io_ctx.worker_idx(),
        //     polynomials.instruction_lookups.instruction_flags.len()
        // );

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
        instruction_flags: &mut [Rep3MultilinearPolynomial<F>],
        E_polys: &mut [Rep3MultilinearPolynomial<F>],
        lookup_outputs_poly: &mut Rep3MultilinearPolynomial<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Vec<F>> {
        let (mut r_primary_sumchecks, eq_evals, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck_inner(
                preprocessing,
                num_rounds,
                eq_poly,
                E_polys,
                instruction_flags,
                lookup_outputs_poly,
                io_ctx,
            )?;

        let E_evals: Vec<_> = E_evals
            .into_par_iter()
            .map(|eval| eval.into_additive())
            .collect();

        if io_ctx.network().is_distributed() {
            // Coordinator runs remaining sumcheck rounds
            io_ctx.network().send_response((
                eq_evals,
                flag_evals,
                E_evals,
                outputs_eval.into_additive(),
            ))?;

            let r_final: Vec<F> = io_ctx.network().receive_request()?;

            r_primary_sumchecks.extend(r_final);
            Ok(r_primary_sumchecks)
        } else {
            io_ctx
                .network()
                .send_response((flag_evals, E_evals, outputs_eval.into_additive()))?;

            Ok(r_primary_sumchecks)
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionLookups::prove_primary_sumcheck_inner")]
    fn prove_primary_sumcheck_inner(
        preprocessing: &Arc<InstructionLookupsPreprocessing<C, F>>,
        num_rounds: usize,
        mut eq_poly: MultilinearPolynomial<F>,
        E_polys: &mut [Rep3MultilinearPolynomial<F>],
        flag_polys: &mut [Rep3MultilinearPolynomial<F>],
        lookup_outputs_poly: &mut Rep3MultilinearPolynomial<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<(
        Vec<F>,
        F,
        Vec<F>,
        Vec<Rep3PrimeFieldShare<F>>,
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
                    E_polys,
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
            // let (tx, rx) = tokio::sync::oneshot::channel();
            // CPU_ONLY_POOL.spawn(move || {
            //     flag_polys
            //         .par_iter_mut()
            //         .chain(E_polys.par_iter_mut())
            //         .chain(rayon::iter::once(&mut lookup_outputs_poly))
            //         .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
            //     eq_poly.bind(r_j, BindingOrder::LowToHigh);
            //     tx.send((flag_polys, E_polys, lookup_outputs_poly, eq_poly))
            //         .unwrap();
            // });
            // (flag_polys, E_polys, lookup_outputs_poly, eq_poly) = rx.blocking_recv().unwrap();
            flag_polys
                .par_iter_mut()
                .chain(E_polys.par_iter_mut())
                .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));

            rayon::join(
                || lookup_outputs_poly.bind(r_j, BindingOrder::LowToHigh),
                || eq_poly.bind(r_j, BindingOrder::LowToHigh),
            );

            drop(_bind_enter);
        }

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();

        // println!("flag_evals {}", flag_polys[0].len());
        // println!("E_evals {}", E_polys[0].len());
        // println!("lookup_outputs_eval {}", lookup_outputs_poly.len());
        // println!("eq_eval {}", eq_poly.len());
        let flag_evals = flag_polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim().as_public())
            .collect();
        let E_evals = E_polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim().as_shared())
            .collect();
        let outputs_eval = lookup_outputs_poly.final_sumcheck_claim().as_shared();
        let eq_eval = eq_poly.final_sumcheck_claim();

        Ok((r, eq_eval, flag_evals, E_evals, outputs_eval))
    }

    #[tracing::instrument(skip_all, level = "trace")]
    fn primary_sumcheck_prover_message(
        preprocessing: &Arc<InstructionLookupsPreprocessing<C, F>>,
        eq_poly: &MultilinearPolynomial<F>,
        flag_polys: &[Rep3MultilinearPolynomial<F>],
        subtable_polys: &[Rep3MultilinearPolynomial<F>],
        lookup_outputs_poly: &Rep3MultilinearPolynomial<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Vec<AdditiveShare<F>>> {
        let party_id = io_ctx.party_id();
        let degree = Self::sumcheck_poly_degree();
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        let precomputed_evals = Self::precompute_evals(
            mle_half,
            &preprocessing,
            &eq_poly,
            &flag_polys,
            &subtable_polys,
            &lookup_outputs_poly,
            party_id,
        );

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

                let mut inner_sums = vec![vec![AdditiveShare::<F>::zero(); degree]; chunk_size];

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

                        let span = tracing::trace_span!("inner_sums");
                        let _span_enter = span.enter();
                        mle_indices.iter().enumerate().for_each(|(i, _)| {
                            used_flag_indices[i][instruction_index]
                                .iter()
                                .enumerate()
                                .for_each(|(index_in_terms_batch, &degree_index)| {
                                    let degrees_used =
                                        used_flag_indices[i][instruction_index].len();
                                    if degrees_used > 0 {
                                        inner_sums[i][degree_index] += instruction_collation_evals
                                            [offset + index_in_terms_batch]
                                            .into_additive()
                                            * flag_evals[i][instruction_index][degree_index];
                                    }
                                });
                            offset += used_flag_indices[i][instruction_index].len();
                        });
                        drop(_span_enter);
                    }
                }

                let span = tracing::trace_span!("evaluations_in_chunk");
                let _span_enter = span.enter();
                let evaluations_in_chunk: Vec<Vec<_>> = (0..chunk_size)
                    .map(|i| {
                        (0..degree)
                            .map(|eval_index| {
                                (inner_sums[i][eval_index]
                                    - output_evals[i][eval_index].into_additive())
                                    * eq_evals[i][eval_index]
                            })
                            .collect()
                    })
                    .collect();
                drop(_span_enter);

                eyre::Ok(evaluations_in_chunk)
            })?
            .into_par_iter()
            .reduce_with(|a, b| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| *x + *y)
                    .collect::<Vec<_>>()
            })
            .unwrap_or(vec![AdditiveShare::<F>::zero(); degree]);

        // subtracing privious claim for each party/worker will break reconstraction of round poly,
        // so we let coordinator do it instead of workers
        // evaluations.insert(1, previous_claim - evaluations[0]);

        Ok(evaluations)
    }

    #[tracing::instrument(skip_all, name = "precompute_evals", level = "trace")]
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
                                <InstructionSet as Rep3JoltInstructionSet>::enum_index(
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
    InstructionSet: Rep3JoltInstructionSet,
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
        (Vec<Vec<usize>>, Vec<Vec<Rep3PrimeFieldShare<F>>>, usize),
        (Vec<Rep3PrimeFieldShare<F>>, usize, usize),
    )> {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.read_cts[0].len();
        let party_id = io_ctx.party_id();

        let num_workers = io_ctx.num_workers();
        let worker_idx = io_ctx.worker_idx();

        let read_memories = witness::read_write_memories_for_worker(
            preprocessing.num_memories,
            num_workers,
            worker_idx,
        );

        let final_subtables = witness::init_final_subtables_for_worker(
            &preprocessing.subtable_to_memory_indices,
            num_workers,
            worker_idx,
        );

        if io_ctx.party_idx() == 0 {
            tracing::info!(
                "Worker {} final_subtables: {:?}",
                worker_idx,
                final_subtables
            );
        }

        let offset = read_memories[0];
        let read_write_leaves = read_memories
            .par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[*memory_index];
                let dim = polynomials.dim[dim_index].as_shared();
                let e_polys = &polynomials.E_polys[*memory_index - offset].as_shared();
                let read_cts = &polynomials.read_cts[*memory_index - offset];

                let read_fingerprints: Vec<_> = (0..num_lookups)
                    .map(|i| {
                        let a = dim[i];
                        let v: Rep3Value<F> = e_polys[i].into();
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
            .collect::<Vec<_>>();

        let offset = final_subtables[0].2[0];
        let init_final_leaves = final_subtables
            .par_iter()
            .flat_map_iter(|(subtable_index, has_init, memories)| {
                let subtable = &preprocessing.materialized_subtables[*subtable_index];
                let mut leaves = vec![Rep3PrimeFieldShare::zero_share(); M * (memories.len() + 1)];

                // Init leaves
                (0..M).for_each(|i| {
                    let a = &F::from_u16(i as u16);
                    let v: u32 = subtable[i];
                    // Compute h(a,v,t) where t == 0
                    leaves[i] = rep3::arithmetic::promote_to_trivial_share(
                        party_id,
                        v.field_mul(*gamma) + *a - *tau,
                    );
                });
                let mut leaf_index = M;

                // Final leaves
                for memory_index in memories {
                    let final_cts = &polynomials.final_cts[memory_index - offset].as_shared();

                    (0..M).for_each(|i| {
                        leaves[leaf_index] =
                            leaves[i] + rep3::arithmetic::mul_public(final_cts[i], gamma_squared);
                        leaf_index += 1;
                    });
                }

                if !has_init {
                    leaves = leaves.split_off(M);
                }

                leaves
            })
            .collect::<Vec<_>>();

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
            Some(read_memories),
        );

        // # init = # subtables; # final = # memories
        let init_final_batch_size = if io_ctx.log_num_workers() != 0 {
            final_subtables
                .iter()
                .map(|(_, init, m)| *init as usize + m.len())
                .sum()
        } else {
            Subtables::COUNT + preprocessing.num_memories
        };

        tracing::info!(
            "worker {} init_final_batch_size: {}",
            io_ctx.worker_idx(),
            init_final_batch_size
        );

        Ok((
            (
                memory_flags,
                read_write_leaves,
                preprocessing.num_memories * 2,
            ),
            (
                init_final_leaves,
                init_final_batch_size,
                Subtables::COUNT + preprocessing.num_memories,
            ),
        ))
    }

    fn compute_openings(
        preprocessing: &Self::Preprocessing,
        opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        polynomials: &Self::Rep3Polynomials,
        _: &JoltStuff<Rep3MultilinearPolynomial<F>>,
        r_read_write: &[F],
        r_init_final: &[F],
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.party_id();

        let read_write_polys = polynomials.read_write_values_grand_product();

        let (read_write_evals, eq_read_write) =
            Rep3MultilinearPolynomial::batch_evaluate_full(&read_write_polys, &r_read_write);

        {
            io_ctx.network().send_response(
                read_write_evals
                    .iter()
                    .map(|x| x.into_additive(party_id))
                    .collect::<Vec<_>>(),
            )?;

            let (rho, rho_offsets, batched_claim): (F, Vec<usize>, F) =
                io_ctx.network().receive_request()?;

            let num_memories_worker = polynomials.read_cts.len();
            let total_num_polys = 4 + preprocessing.num_memories * 2 + InstructionSet::COUNT;
            let mut rho_powers = vec![F::one()];
            for i in 1..total_num_polys {
                rho_powers.push(rho_powers[i - 1] * rho);
            }

            if io_ctx.party_idx() == 0 {
                // tracing::info!(
                //     "rho_powers: {:?}\n{:?}\n{:?}\n{:?}",
                //     &rho_powers[..4],
                //     &rho_powers[rho_offsets[0]..rho_offsets[0] + num_memories_worker],
                //     &rho_powers[rho_offsets[1]..rho_offsets[1] + num_memories_worker],
                //     &rho_powers[4 + preprocessing.num_memories * 2..],
                // );
            }

            rho_powers = chain![
                &rho_powers[..4],
                &rho_powers[rho_offsets[0]..rho_offsets[0] + num_memories_worker],
                &rho_powers[rho_offsets[1]..rho_offsets[1] + num_memories_worker],
                &rho_powers[4 + preprocessing.num_memories * 2..],
            ]
            .copied()
            .collect();

            // let read_write_polys = polynomials
            //     .read_cts
            //     .iter()
            //     // .chain(polynomials.read_cts.iter())
            //     .chain(polynomials.E_polys.iter())
            //     .collect::<Vec<_>>();

            let masked_polys = read_write_polys
                .iter()
                .map(|p| p.into_masked_shard_mle()) // breaks with E_polys
                .collect::<Vec<_>>();

            tracing::info!(
                "E_polys len: {:?} num_memories_worker {:?}",
                polynomials.E_polys.len(),
                num_memories_worker
            );

            let batched_poly = Rep3MultilinearPolynomial::linear_combination(
                // &read_write_polys,
                &masked_polys
                    .iter()
                    .chain(masked_polys.iter())
                    .collect::<Vec<_>>(),
                &rho_powers,
                io_ctx.party_id(),
            );

            let batched_poly_open = match &batched_poly {
                Rep3MultilinearPolynomial::Public {
                    poly,
                    trivial_share,
                } => poly.clone(),
                Rep3MultilinearPolynomial::Shared(rep3_dense_polynomial) => {
                    MultilinearPolynomial::from(
                        rep3::arithmetic::open_vec(
                            rep3_dense_polynomial.coeffs_ref(),
                            io_ctx.main(),
                        )
                        .unwrap(),
                    )
                }
            };
            if io_ctx.party_idx() == 0 {
                tracing::info!(
                    "batched_poly_open len: {:?} nv {}",
                    batched_poly_open.len(),
                    batched_poly_open.get_num_vars()
                );

                io_ctx
                    .network()
                    .send_response(batched_poly_open.evaluate(&r_read_write))
                    .unwrap();
            }
            tracing::info!("batched_poly: {:?}", batched_poly.len());

            opening_accumulator.append_opening(
                batched_poly,
                DensePolynomial::new(eq_read_write),
                r_read_write.to_vec(),
                additive::promote_to_trivial_share(batched_claim, io_ctx.party_id()),
            );
        }

        let init_final_polys = polynomials.init_final_values();
        let (init_final_evals, eq_init_final) =
            Rep3MultilinearPolynomial::batch_evaluate_full(&init_final_polys, &r_init_final);

        opening_accumulator.append_batched_(
            &polynomials.init_final_values(),
            DensePolynomial::new(eq_init_final),
            r_init_final.to_vec(),
            &init_final_evals
                .iter()
                .map(|x| x.into_additive(party_id))
                .collect::<Vec<_>>(),
            io_ctx.main(),
        )?;

        Ok(())
    }
}
