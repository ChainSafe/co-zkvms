use co_spartan::mpc::rep3::Rep3PrimeFieldShare;
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::Itertools;
use jolt_core::{
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        field::JoltField,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{math::Math, mul_0_1_optimized},
};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    PartyID,
};
use mpc_net::mpc_star::MpcStarNetWorker;
use std::marker::PhantomData;
use tracing::trace_span;

use crate::{
    grand_product::{
        BatchedGrandProductProver, BatchedRep3GrandProductCircuit, Rep3GrandProductCircuit,
    },
    instructions::Rep3LookupSet,
    lasso::LassoPreprocessing,
    poly::Rep3DensePolynomial,
    subtables::SubtableSet,
    utils::{self, split_rep3_poly_flagged},
    witness_solver::Rep3LassoPolynomials,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Rep3LassoProver<const C: usize, const M: usize, F, Lookups, Subtables, Network>
where
    Network: Rep3Network,
{
    pub _marker: PhantomData<(F, Lookups, Subtables)>,
    pub io_ctx: IoContext<Network>,
}

type Preprocessing<F> = LassoPreprocessing<F>;
type Polynomials<F> = Rep3LassoPolynomials<F>;

impl<const C: usize, const M: usize, F: JoltField, Lookups, Subtables, Network>
    Rep3LassoProver<C, M, F, Lookups, Subtables, Network>
where
    Lookups: Rep3LookupSet<F>,
    Subtables: SubtableSet<F>,
    Network: Rep3Network + MpcStarNetWorker,
{
    pub fn new(net: Network) -> color_eyre::Result<Self> {
        let io_ctx = IoContext::init(net).context("failed to initialize io context")?;
        Ok(Self {
            _marker: PhantomData,
            io_ctx,
        })
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove")]
    pub fn prove(
        &mut self,
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
    ) -> Result<()> {
        let trace_length = polynomials.dims[0].len();
        let r_eq = self.io_ctx.network.receive_request::<Vec<F>>()?;

        let eq_evals: Vec<F> = EqPolynomial::new(r_eq.to_vec()).evals();
        let mut eq_poly = DensePolynomial::new(eq_evals);
        let num_rounds = trace_length.log_2();

        self.prove_primary_sumcheck(
            preprocessing,
            num_rounds,
            &mut eq_poly,
            &polynomials.e_polys,
            &polynomials.lookup_flag_polys,
            &mut polynomials.lookup_outputs.clone(),
            Self::sumcheck_poly_degree(),
        )?;

        let _ = self.prove_memory_checking(preprocessing, polynomials)?;

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
        preprocessing: &Preprocessing<F>,
        num_rounds: usize,
        eq_poly: &mut DensePolynomial<F>,
        memory_polys: &Vec<Rep3DensePolynomial<F>>,
        flag_polys: &Vec<DensePolynomial<F>>,
        lookup_outputs_poly: &mut Rep3DensePolynomial<F>,
        degree: usize,
    ) -> eyre::Result<()> {
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
        );
        // compressed_polys.push(round_uni_poly.compress());
        self.io_ctx
            .network
            .send_response(round_uni_poly.compress())?;
        let r_j = self.io_ctx.network.receive_request::<F>()?;
        random_vars.push(r_j);

        let _bind_span = trace_span!("BindPolys");
        let _bind_enter = _bind_span.enter();
        rayon::join(
            || eq_poly.bound_poly_var_top(&r_j),
            || lookup_outputs_poly.fix_var_top_many_ones(&r_j),
        );
        let mut flag_polys_updated: Vec<DensePolynomial<F>> = flag_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_bound_poly_var_top_flags(&r_j))
            .collect();
        let mut memory_polys_updated: Vec<_> = memory_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_fix_var_top_flags(&r_j))
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
            );
            // compressed_polys.push(round_uni_poly.compress());
            self.io_ctx.network.send_response(round_uni_poly)?;
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
                .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j));
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
        let flag_evals: Vec<_> = flag_polys_updated.iter().map(|poly| poly[0]).collect();
        let memory_evals: Vec<_> = memory_polys_updated.iter().map(|poly| poly[0]).collect();
        let outputs_eval = lookup_outputs_poly[0];
        if self.io_ctx.network.party_id() == PartyID::ID0 {
            self.io_ctx.network.send_response(flag_evals)?;
        }
        self.io_ctx
            .network
            .send_response((memory_evals, outputs_eval))?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::primary_sumcheck_inner_loop")]
    fn primary_sumcheck_inner_loop(
        &mut self,
        preprocessing: &Preprocessing<F>,
        eq_poly: &DensePolynomial<F>,
        flag_polys: &[DensePolynomial<F>],
        memory_polys: &[Rep3DensePolynomial<F>],
        lookup_outputs_poly: &Rep3DensePolynomial<F>,
        num_eval_points: usize,
    ) -> UniPoly<F> {
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        // Loop over half MLE size (size of MLE next round)
        //   - Compute evaluations of eq, flags, E, at p {0, 1, ..., degree}:
        //       eq(p, _boolean_hypercube_), flags(p, _boolean_hypercube_), E(p, _boolean_hypercube_)
        // After: Sum over MLE elements (with combine)
        let evaluations: Vec<_> = (0..mle_half)
            .into_par_iter()
            .map(|low_index| {
                let high_index = mle_half + low_index;

                let mut eq_evals: Vec<F> = vec![F::zero(); num_eval_points];
                let mut outputs_evals = vec![Rep3PrimeFieldShare::zero_share(); num_eval_points];
                let mut multi_flag_evals: Vec<Vec<F>> =
                    vec![vec![F::zero(); Lookups::COUNT]; num_eval_points];
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
                // TODO: convert to additive
                let mut inner_sum = vec![Rep3PrimeFieldShare::zero_share(); num_eval_points];
                for instruction in Lookups::iter() {
                    let instruction_index = Lookups::enum_index(&instruction);
                    let memory_indices = &preprocessing.lookup_to_memory_indices[instruction_index];

                    for eval_index in 0..num_eval_points {
                        let flag_eval = multi_flag_evals[eval_index][instruction_index];
                        if flag_eval.is_zero() {
                            continue;
                        }; // Early exit if no contribution.

                        let terms: Vec<_> = memory_indices
                            .iter()
                            .map(|memory_index| multi_memory_evals[eval_index][*memory_index])
                            .collect();
                        let instruction_collation_eval = instruction.combine_lookups(&terms, C, M);

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
                evaluations
            })
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

        UniPoly::from_evals(&evaluations)
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_memory_checking")]
    pub fn prove_memory_checking(
        &mut self,
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
    ) -> Result<()> {
        let _ = self.prove_grand_products(preprocessing, polynomials)?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_grand_products")]
    fn prove_grand_products(
        &mut self,
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
    ) -> Result<(Vec<F>, Vec<F>)> {
        let (gamma, tau) = self.io_ctx.network.receive_request()?;
        self.io_ctx
            .network
            .send_response(polynomials.dims[0].len())?;

        let (read_write_leaves, init_final_leaves) = Self::compute_leaves(
            preprocessing,
            polynomials,
            &gamma,
            &tau,
            &mut self.io_ctx.network,
        );

        let (read_write_circuit, read_write_hashes) =
            self.read_write_grand_product(preprocessing, polynomials, read_write_leaves)?;
        let (init_final_circuit, init_final_hashes) =
            self.init_final_grand_product(preprocessing, polynomials, init_final_leaves)?;

        self.io_ctx
            .network
            .send_response((read_write_hashes.clone(), init_final_hashes.clone()))?;

        let r_read_write =
            BatchedGrandProductProver::prove_worker(read_write_circuit, &mut self.io_ctx.network)?;
        let r_init_final =
            BatchedGrandProductProver::prove_worker(init_final_circuit, &mut self.io_ctx.network)?;

        Ok((r_read_write, r_init_final))
    }

    fn compute_leaves(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        gamma: &F,
        tau: &F,
        network: &mut Network,
    ) -> (Vec<Rep3DensePolynomial<F>>, Vec<Rep3DensePolynomial<F>>) {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dims[0].len();
        let party_id = network.party_id();

        let read_write_leaves = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];

                let read_fingerprints: Vec<_> = (0..num_lookups)
                    .map(|i| {
                        let a = &polynomials.dims[dim_index][i];
                        let v = &polynomials.e_polys[memory_index][i];
                        let t = &polynomials.read_cts[memory_index][i];
                        rep3::arithmetic::sub_shared_by_public(
                            (t * gamma_squared) + (v * *gamma) + *a,
                            *tau,
                            party_id,
                        )
                    })
                    .collect();
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| {
                        rep3::arithmetic::add_public(*read_fingerprint, gamma_squared, party_id)
                    })
                    .collect();
                [
                    Rep3DensePolynomial::new(read_fingerprints),
                    Rep3DensePolynomial::new(write_fingerprints),
                ]
            })
            .collect();

        let init_final_leaves = preprocessing
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
                        mul_0_1_optimized(gamma, v) + a - tau
                    })
                    .collect();
                let init_fingerprints =
                    rep3::arithmetic::promote_to_trivial_shares(init_fingerprints, party_id);

                let final_leaves: Vec<_> = preprocessing.subtable_to_memory_indices[subtable_index]
                    .iter()
                    .map(|memory_index| {
                        let final_cts = &polynomials.final_cts[*memory_index];
                        let final_fingerprints = (0..M)
                            .map(|i| init_fingerprints[i] + (final_cts[i] * gamma_squared))
                            .collect();
                        Rep3DensePolynomial::new(final_fingerprints)
                    })
                    .collect();

                let mut polys = Vec::with_capacity(C + 1);
                polys.push(Rep3DensePolynomial::new(init_fingerprints));
                polys.extend(final_leaves);
                polys
            })
            .collect();

        (read_write_leaves, init_final_leaves)
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::read_write_grand_product")]
    fn read_write_grand_product(
        &mut self,
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        read_write_leaves: Vec<Rep3DensePolynomial<F>>,
    ) -> Result<(BatchedRep3GrandProductCircuit<F>, Vec<F>)> {
        assert_eq!(read_write_leaves.len(), 2 * preprocessing.num_memories);

        let _span = trace_span!("construct_circuits");
        _span.enter();

        let memory_flag_polys =
            Self::memory_flag_polys(preprocessing, &polynomials.lookup_flag_bitvectors);

        let read_write_circuits = read_write_leaves
            // .par_iter()
            .iter()
            .enumerate()
            .map(|(i, leaves_poly)| {
                // Split while cloning to save on future cloning in GrandProductCircuit
                let memory_index = i / 2;
                let flag: &DensePolynomial<F> = &memory_flag_polys[memory_index];
                let (toggled_leaves_l, toggled_leaves_r) =
                    split_rep3_poly_flagged(leaves_poly, flag, self.io_ctx.network.party_id());
                Rep3GrandProductCircuit::new_split(
                    toggled_leaves_l,
                    toggled_leaves_r,
                    &mut self.io_ctx,
                )
            })
            .collect::<Result<Vec<Rep3GrandProductCircuit<F>>>>()?;

        drop(_span);

        let read_write_hashes: Vec<F> = trace_span!("compute_hashes").in_scope(|| {
            read_write_circuits
                .par_iter()
                .map(|circuit| circuit.evaluate())
                .collect()
        });

        // Prover has access to memory_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
        let batched_circuits = BatchedRep3GrandProductCircuit::new_batch_flags(
            read_write_circuits,
            memory_flag_polys,
            read_write_leaves,
        );

        Ok((batched_circuits, read_write_hashes))
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::init_final_grand_product")]
    fn init_final_grand_product(
        &mut self,
        _preprocessing: &Preprocessing<F>,
        _polynomials: &Polynomials<F>,
        init_final_leaves: Vec<Rep3DensePolynomial<F>>,
    ) -> Result<(BatchedRep3GrandProductCircuit<F>, Vec<F>)> {
        let init_final_circuits: Vec<Rep3GrandProductCircuit<F>> =
            utils::fork_map(init_final_leaves, &mut self.io_ctx, |leaves, io_ctx| {
                Rep3GrandProductCircuit::new(&leaves, io_ctx).unwrap()
            })?;

        let init_final_hashes: Vec<F> = init_final_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        Ok((
            BatchedRep3GrandProductCircuit::new_batch(init_final_circuits),
            init_final_hashes,
        ))
    }

    /// Converts instruction flag polynomials into memory flag polynomials. A memory flag polynomial
    /// can be computed by summing over the instructions that use that memory: if a given execution step
    /// accesses the memory, it must be executing exactly one of those instructions.
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::memory_flag_polys")]
    fn memory_flag_polys(
        preprocessing: &Preprocessing<F>,
        flag_bitvectors: &[Vec<u64>],
    ) -> Vec<DensePolynomial<F>> {
        let m = flag_bitvectors[0].len();

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
        Lookups::iter()
            .map(|lookup| lookup.g_poly_degree(C))
            .max()
            .unwrap()
            + 2 // eq and flag
    }
}
