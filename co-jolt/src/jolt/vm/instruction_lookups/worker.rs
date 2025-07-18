use co_lasso::{
    memory_checking::worker::MemoryCheckingProverRep3Worker,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::Rep3ProverOpeningAccumulator, Rep3DensePolynomial,
    },
    subprotocols::{
        commitment::DistributedCommitmentScheme,
        grand_product::{Rep3BatchedDenseGrandProduct, Rep3BatchedGrandProductWorker}, sparse_grand_product::Rep3ToggledBatchedGrandProduct,
    },
    utils::{split_rep3_poly_flagged, transcript::KeccakTranscript},
    Rep3Polynomials,
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::chain;
use jolt_core::{
    field::JoltField,
    jolt::vm::instruction_lookups::InstructionLookupStuff,
    lasso::memory_checking::NoExogenousOpenings,
    poly::{
        compact_polynomial::{CompactPolynomial, SmallScalar},
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        unipoly::UniPoly,
    },
    utils::{math::Math, mul_0_1_optimized},
};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network, Rep3NetworkWorker},
    PartyID,
};
use mpc_net::mpc_star::MpcStarNetWorker;
use std::{iter, marker::PhantomData};
use tracing::trace_span;

use super::{witness::Rep3InstructionLookupPolynomials, InstructionLookupsPreprocessing};
use crate::jolt::{
    instruction::{JoltInstructionSet, Rep3JoltInstructionSet},
    subtable::JoltSubtableSet,
    vm::{instruction_lookups::InstructionLookupsProof, JoltPolynomials},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type ProofTranscript = KeccakTranscript;
type Rep3InstructionLookupOpenings<F> = InstructionLookupStuff<Rep3PrimeFieldShare<F>>;

pub struct Rep3InstructionLookupsProver<
    const C: usize,
    const M: usize,
    F,
    PCS,
    Lookups,
    Subtables,
    Network,
> where
    F: JoltField,
    Network: Rep3Network,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
{
    pub io_ctx: IoContext<Network>,
    pub _marker: PhantomData<(F, Lookups, Subtables, PCS)>,
}

impl<const C: usize, const M: usize, F, PCS, InstructionSet, Subtables, Network>
    Rep3InstructionLookupsProver<C, M, F, PCS, InstructionSet, Subtables, Network>
where
    F: JoltField,
    InstructionSet: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    Network: Rep3NetworkWorker,
{
    pub fn new(net: Network) -> color_eyre::Result<Self> {
        let io_ctx = IoContext::init(net).context("failed to initialize io context")?;
        Ok(Self {
            io_ctx,
            _marker: PhantomData,
        })
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove")]
    pub fn prove(
        &mut self,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        polynomials: &Rep3InstructionLookupPolynomials<F>,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        pcs_setup: &PCS::Setup,
    ) -> Result<()> {
        let trace_length = polynomials.dim[0].len();
        let r_eq = self.io_ctx.network.receive_request::<Vec<F>>()?;

        let eq_evals: Vec<F> = EqPolynomial::evals(&r_eq);
        let mut eq_poly = DensePolynomial::new(eq_evals);
        let num_rounds = trace_length.log_2();

        let (r_primary_sumcheck, flag_evals, memory_evals, outputs_eval) = self
            .prove_primary_sumcheck(
                preprocessing,
                num_rounds,
                &mut eq_poly,
                &polynomials.E_polys,
                &polynomials.instruction_flags,
                &mut polynomials.lookup_outputs.clone(),
                Self::sumcheck_poly_degree(),
            )?;

        if self.io_ctx.network.party_id() == PartyID::ID0 {
            self.io_ctx.network.send_response(flag_evals)?;
        }
        self.io_ctx
            .network
            .send_response((memory_evals, outputs_eval))?;

        // self.prove_openings(&polynomials, &r_primary_sumcheck)?;

        Self::prove_memory_checking(
            pcs_setup,
            preprocessing,
            polynomials,
            &jolt_polynomials,
            opening_accumulator,
            &mut self.io_ctx,
        )?;

        Ok(())
    }

    /// Prove Jolt primary sumcheck including instruction collation.
    ///
    /// Computes \sum{ eq(r,x) * [ flags_0(x) * g_0(E(x)) + flags_1(x) * g_1(E(x)) + ... + flags_{NUM_INSTRUCTIONS}(E(x)) * g_{NUM_INSTRUCTIONS}(E(x)) ]}
    /// via the sumcheck protocol.
    /// Note: These E(x) terms differ from term to term depending on the memories used in the instruction.
    ///
    /// Returns: (SumcheckProof, Random evaluation point, claimed evaluations of polynomials)
    ///
    /// Params:
    /// - `claim`: Claimed sumcheck evaluation.
    /// - `num_rounds`: Number of rounds to run sumcheck. Corresponds to the number of free bits or free variables in the polynomials.
    /// - `memory_polys`: Each of the `E` polynomials or "dereferenced memory" polynomials.
    /// - `flag_polys`: Each of the flag selector polynomials describing which instruction is used at a given step of the CPU.
    /// - `degree`: Degree of the inner sumcheck polynomial. Corresponds to number of evaluation points per round.
    /// - `transcript`: Fiat-shamir transcript.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_primary_sumcheck")]
    fn prove_primary_sumcheck(
        &mut self,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        num_rounds: usize,
        eq_poly: &mut DensePolynomial<F>,
        memory_polys: &Vec<Rep3DensePolynomial<F>>,
        flag_polys: &Vec<MultilinearPolynomial<F>>,
        lookup_outputs_poly: &mut Rep3DensePolynomial<F>,
        degree: usize,
    ) -> eyre::Result<(
        Vec<F>,
        Vec<F>,
        Vec<Rep3PrimeFieldShare<F>>,
        Rep3PrimeFieldShare<F>,
    )> {
        let mut flag_polys_updated: Vec<MultilinearPolynomial<F>> = flag_polys.clone();
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
        let num_eval_points = degree + 1;

        let round_uni_poly = self.primary_sumcheck_inner_loop(
            preprocessing,
            eq_poly,
            flag_polys,
            memory_polys,
            lookup_outputs_poly,
            num_eval_points,
        )?;
        self.io_ctx
            .network
            .send_response(round_uni_poly.compress().coeffs_except_linear_term)?;
        let r_j = self.io_ctx.network.receive_request::<F>()?;
        random_vars.push(r_j);

        let _bind_span = trace_span!("BindPolys");
        let _bind_enter = _bind_span.enter();
        rayon::join(
            || eq_poly.bound_poly_var_top(&r_j),
            || lookup_outputs_poly.fix_var_top_many_ones(&r_j),
        );
        flag_polys_updated
            .par_iter_mut()
            .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
        let mut memory_polys_updated: Vec<_> = memory_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_fix_var_top(&r_j))
            .collect();
        drop(_bind_enter);
        drop(_bind_span);

        for _round in 1..num_rounds {
            let round_uni_poly = self.primary_sumcheck_inner_loop(
                preprocessing,
                eq_poly,
                &flag_polys_updated,
                &memory_polys_updated,
                lookup_outputs_poly,
                num_eval_points,
            )?;
            // compressed_polys.push(round_uni_poly.compress());
            self.io_ctx
                .network
                .send_response(round_uni_poly.compress().coeffs_except_linear_term)?;
            let r_j = self.io_ctx.network.receive_request::<F>()?;
            random_vars.push(r_j);

            // Bind all polys
            let _bind_span = trace_span!("BindPolys");
            let _bind_enter = _bind_span.enter();
            rayon::join(
                || eq_poly.bound_poly_var_top(&r_j),
                || lookup_outputs_poly.fix_var_top_many_ones(&r_j),
            );
            flag_polys_updated
                .par_iter_mut()
                .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
            memory_polys_updated
                .par_iter_mut()
                .for_each(|poly| poly.fix_var_top_many_ones(&r_j));

            drop(_bind_enter);
            drop(_bind_span);
        } // End rounds

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        let flag_evals: Vec<_> = flag_polys_updated
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();
        let memory_evals: Vec<_> = memory_polys_updated.iter().map(|poly| poly[0]).collect();
        let outputs_eval = lookup_outputs_poly[0];

        Ok((random_vars, flag_evals, memory_evals, outputs_eval))
    }

    fn primary_sumcheck_inner_loop(
        &mut self,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        eq_poly: &DensePolynomial<F>,
        flag_polys: &[MultilinearPolynomial<F>],
        memory_polys: &[Rep3DensePolynomial<F>],
        lookup_outputs_poly: &Rep3DensePolynomial<F>,
        num_eval_points: usize,
    ) -> eyre::Result<UniPoly<F>> {
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        let flag_polys: Vec<&CompactPolynomial<u32, F>> = flag_polys
            .iter()
            .map(|poly| poly.try_into().unwrap())
            .collect();

        // Loop over half MLE size (size of MLE next round)
        //   - Compute evaluations of eq, flags, E, at p {0, 1, ..., degree}:
        //       eq(p, _boolean_hypercube_), flags(p, _boolean_hypercube_), E(p, _boolean_hypercube_)
        // After: Sum over MLE elements (with combine)
        let evaluations: Vec<_> = co_lasso::utils::try_fork_chunks(
            0..mle_half,
            &mut self.io_ctx,
            8, // TODO: make configurable
            |low_index, io_ctx| {
                let high_index = mle_half + low_index;

                let mut eq_evals: Vec<F> = vec![F::zero(); num_eval_points];
                let mut outputs_evals = vec![Rep3PrimeFieldShare::zero_share(); num_eval_points];
                let mut multi_flag_evals: Vec<Vec<F>> =
                    vec![vec![F::zero(); InstructionSet::COUNT]; num_eval_points];
                let mut multi_memory_evals =
                    vec![
                        vec![Rep3PrimeFieldShare::zero_share(); preprocessing.num_memories];
                        num_eval_points
                    ];

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
                for flag_instruction_index in 0..InstructionSet::COUNT {
                    multi_flag_evals[0][flag_instruction_index] =
                        flag_polys[flag_instruction_index][low_index].into();
                    multi_flag_evals[1][flag_instruction_index] =
                        flag_polys[flag_instruction_index][high_index].into();
                    let flag_m: F = (flag_polys[flag_instruction_index][high_index]
                        - flag_polys[flag_instruction_index][low_index])
                        .into();
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
                // TODO: convert to additive
                let mut inner_sum = vec![Rep3PrimeFieldShare::zero_share(); num_eval_points];
                for instruction in InstructionSet::iter() {
                    let instruction_index =
                        <InstructionSet as Rep3JoltInstructionSet<F>>::enum_index(&instruction);
                    let memory_indices =
                        &preprocessing.instruction_to_memory_indices[instruction_index];

                    for eval_index in 0..num_eval_points {
                        let flag_eval = multi_flag_evals[eval_index][instruction_index];
                        if flag_eval.is_zero() {
                            continue;
                        }; // Early exit if no contribution.

                        let terms: Vec<_> = memory_indices
                            .iter()
                            .map(|memory_index| multi_memory_evals[eval_index][*memory_index])
                            .collect();
                        let instruction_collation_eval =
                            instruction.combine_lookups_rep3(&terms, C, M, io_ctx)?;

                        // TODO(sragss): Could sum all shared inner terms before multiplying by the flag eval
                        inner_sum[eval_index] +=
                            rep3::arithmetic::mul_public(instruction_collation_eval, flag_eval);
                    }
                }
                let evaluations: Vec<_> = (0..num_eval_points)
                    .map(|eval_index| {
                        rep3::arithmetic::mul_public(
                            inner_sum[eval_index] - outputs_evals[eval_index],
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
            || vec![F::zero(); num_eval_points],
            |running, new| {
                debug_assert_eq!(running.len(), new.len());
                running
                    .iter()
                    .zip(new.iter())
                    .map(|(r, n)| *r + *n)
                    .collect()
            },
        );

        Ok(UniPoly::from_evals(&evaluations))
    }

    // #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::prove_openings")]
    // fn prove_openings(
    //     &mut self,
    //     polynomials: &Rep3InstructionLookupPolynomials<F>,
    //     opening_point: &[F],
    // ) -> eyre::Result<()> {
    //     // let e_polys_a = polynomials
    //     //     .e_polys
    //     //     .iter()
    //     //     .map(|poly| poly.copy_share_a())
    //     //     .collect::<Vec<_>>();
    //     // let lookup_output_a = polynomials.lookup_outputs.copy_share_a();
    //     let lookup_flag_polys = polynomials
    //         .instruction_flag_polys
    //         .iter()
    //         .map(|p| {
    //             Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
    //                 p.evals(),
    //                 self.io_ctx.network.party_id(),
    //             ))
    //         })
    //         .collect::<Vec<_>>();
    //     let primary_sumcheck_polys = chain![
    //         polynomials.E_polys.iter(),
    //         lookup_flag_polys.iter(),
    //         iter::once(&polynomials.lookup_outputs),
    //     ]
    //     .collect::<Vec<_>>();

    //     PCS::distributed_batch_open_worker(
    //         &primary_sumcheck_polys,
    //         &self.ck,
    //         opening_point,
    //         &mut self.io_ctx.network,
    //     )
    // }

    /// Converts instruction flag polynomials into memory flag polynomials. A memory flag polynomial
    /// can be computed by summing over the instructions that use that memory: if a given execution step
    /// accesses the memory, it must be executing exactly one of those instructions.
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::memory_flag_polys")]
    fn memory_flag_polys(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        flag_bitvectors: &[Vec<u64>],
    ) -> Vec<DensePolynomial<F>> {
        let m = flag_bitvectors[0].len();

        (0..preprocessing.num_memories)
            .into_par_iter()
            .map(|memory_index| {
                let mut memory_flag_bitvector = vec![0u64; m];
                for instruction_index in 0..InstructionSet::COUNT {
                    if preprocessing.instruction_to_memory_indices[instruction_index]
                        .contains(&memory_index)
                    {
                        memory_flag_bitvector
                            .iter_mut()
                            .zip(&flag_bitvectors[instruction_index])
                            .for_each(|(memory_flag, instruction_flag)| {
                                *memory_flag += instruction_flag
                            });
                    }
                }
                DensePolynomial::from_u64(&memory_flag_bitvector)
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

impl<F: JoltField, const C: usize, const M: usize, PCS, InstructionSet, Subtables, Network>
    MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>
    for Rep3InstructionLookupsProver<C, M, F, PCS, InstructionSet, Subtables, Network>
where
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
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
        _jolt_polynomials: &JoltPolynomials<F>,
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
        let num_lookups = polynomials.num_lookups();
        let party_id = io_ctx.network.party_id();

        let read_write_leaves: Vec<_> = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];

                let read_fingerprints: Vec<_> = (0..num_lookups)
                    .map(|i| {
                        let a = &polynomials.dim[dim_index][i];
                        let v = &polynomials.E_polys[memory_index][i];
                        let t = &polynomials.read_cts[memory_index][i];
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
                    let final_cts = &polynomials.final_cts[*memory_index];
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
                .iter()
                .map(|poly| poly.try_into().unwrap())
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
