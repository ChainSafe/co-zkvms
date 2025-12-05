use crate::field::JoltField;
use crate::jolt::vm::instruction_lookups::witness;
use crate::poly::opening_proof::{Rep3CoordinatorOpening, Rep3OpeningAccumulatorCoordinator};
use crate::{
    lasso::memory_checking::Rep3MemoryCheckingProver,
    poly::{commitment::Rep3CommitmentScheme, opening_proof::Rep3OpeningAccumulatorWorker},
    subprotocols::{
        grand_product::Rep3BatchedDenseGrandProduct,
        sparse_grand_product::Rep3ToggledBatchedGrandProduct,
    },
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::{chain, izip, Itertools};
use jolt_core::lasso::memory_checking::{
    ExogenousOpenings, Initializable, StructuredPolynomialData,
};
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding,
};
use jolt_core::utils::transcript::{AppendToTranscript, Transcript};
use jolt_core::{
    jolt::subtable::JoltSubtableSet,
    poly::unipoly::{CompressedUniPoly, UniPoly},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::math::Math,
};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;
use std::iter::once;
use std::marker::PhantomData;

use super::{
    InstructionLookupsPreprocessing, InstructionLookupsProof, PrimarySumcheck,
    PrimarySumcheckOpenings,
};
use crate::jolt::instruction::JoltInstructionSet;

impl<F, const C: usize, const M: usize, PCS, ProofTranscript, Instructions, Subtables>
    InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    Instructions: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Rep3InstructionLookups::prove")]
    pub fn prove_rep3<Network: Rep3NetworkCoordinator>(
        num_ops: usize,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        network: &mut Network,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>>
    {
        transcript.append_message(Self::protocol_name());

        let num_rounds = num_ops.log_2();
        let r_eq = transcript.challenge_vector::<F>(num_rounds);
        tracing::info!("commitment check/r_eq: {:?}", r_eq[0]);
        // tracing::info!("r_eq: {:?}", r_eq);
        network.broadcast_request(r_eq)?;

        let (primary_sumcheck_proof, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck_rep3(
                num_rounds,
                preprocessing,
                F::zero(),
                transcript,
                network,
            )
            .context("while proving primary sumcheck")?;

        let primary_sumcheck_claims: Vec<_> =
            chain![E_evals, flag_evals, once(outputs_eval)].collect();

        let mut flag_evals = vec![F::zero(); Self::NUM_INSTRUCTIONS];
        let mut E_evals = vec![F::zero(); preprocessing.num_memories];
        let mut outputs_eval = F::zero();

        opening_accumulator.append_with_claims(
            num_rounds,
            &primary_sumcheck_claims,
            transcript,
            network,
        )?;

        E_evals
            .iter_mut()
            .chain(flag_evals.iter_mut())
            .chain([&mut outputs_eval])
            .zip(primary_sumcheck_claims.into_iter())
            .for_each(|(eval, claim)| {
                *eval = claim;
            });

        // Create a single opening proof for the flag_evals and memory_evals
        let sumcheck_openings = PrimarySumcheckOpenings {
            E_poly_openings: E_evals,
            flag_openings: flag_evals,
            lookup_outputs_opening: outputs_eval,
        };

        let primary_sumcheck = PrimarySumcheck::<F, ProofTranscript> {
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            openings: sumcheck_openings,
            _marker: PhantomData,
        };

        let memory_checking = Self::coordinate_memory_checking(
            preprocessing,
            num_ops,
            M,
            opening_accumulator,
            transcript,
            network,
        )
        .context("while proving memory checking")?;

        Ok(InstructionLookupsProof {
            primary_sumcheck,
            memory_checking,
            _instructions: PhantomData,
            _subtables: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_primary_sumcheck")]
    fn prove_primary_sumcheck_rep3<Network: Rep3NetworkCoordinator>(
        num_rounds: usize,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        mut previous_claim: F,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>, F)> {
        let log_num_workers = network.log_num_workers();

        let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        for _ in 0..num_rounds - log_num_workers {
            let mut round_evals = if network.is_distributed() {
                let subnet_responces =
                    network.receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?;
                let degree = subnet_responces[0][0].len();
                subnet_responces
                    .into_iter()
                    .map(|shares| additive::combine_additive_vec(shares))
                    .fold(vec![F::zero(); degree], |mut acc, coeff| {
                        acc.iter_mut().zip(coeff.iter()).for_each(|(acc, coeff)| {
                            *acc += coeff;
                        });
                        acc
                    })
            } else {
                additive::combine_additive_vec(network.receive_responses()?)
            };

            // tracing::info!("Round evaluations: {:?}", round_evals);

            round_evals.insert(1, previous_claim - round_evals[0]);
            let round_poly = UniPoly::from_evals(&round_evals);

            let compressed_round_poly = round_poly.compress();
            compressed_round_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_round_poly);
            let r_j = transcript.challenge_scalar::<F>();
            network.broadcast_request(r_j)?;
            random_vars.push(r_j);

            let new_claim = round_poly.evaluate(&r_j);
            previous_claim = new_claim;
        }

        // Remaining rounds
        if network.is_distributed() {
            let (remaining_polys, flag_evals, E_evals, outputs_eval) =
                Self::prove_remaining_primary_sumcheck(
                    preprocessing,
                    previous_claim,
                    transcript,
                    network,
                )?;

            compressed_polys.extend(remaining_polys);
            Ok((
                SumcheckInstanceProof::new(compressed_polys),
                flag_evals,
                E_evals,
                outputs_eval,
            ))
        } else {
            let (mut flag_evals, E_evals_shares, output_eval_shares): (Vec<_>, Vec<_>, Vec<_>) =
                network
                    .receive_responses::<(Vec<F>, Vec<AdditiveShare<F>>, AdditiveShare<F>)>()?
                    .into_iter()
                    .multiunzip();
            let flag_evals = flag_evals.pop().unwrap();
            let E_evals = additive::combine_additive_vec(E_evals_shares);
            let output_eval = additive::combine_additive_share(output_eval_shares);

            Ok((
                SumcheckInstanceProof::new(compressed_polys),
                flag_evals,
                E_evals,
                output_eval,
            ))
        }
    }

    fn prove_remaining_primary_sumcheck<Network: Rep3NetworkCoordinator>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        mut previous_claim: F,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(Vec<CompressedUniPoly<F>>, Vec<F>, Vec<F>, F)> {
        let log_num_workers = network.log_num_workers();

        let mut r: Vec<F> = Vec::with_capacity(log_num_workers);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(log_num_workers);

        let (eq_evals, flag_evals, E_evals, output_evals) = network
            .receive_responses_from_subnets::<(F, Vec<F>, Vec<AdditiveShare<F>>, AdditiveShare<F>)>(
            )?
            .into_iter()
            .map(|shares| {
                let (eq_evals, mut flag_evals, E_evals, output_evals): (
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                ) = shares.into_iter().multiunzip();

                (
                    eq_evals[0],
                    flag_evals.pop().unwrap(),
                    additive::combine_additive_vec(E_evals),
                    additive::combine_additive_share(output_evals),
                )
            })
            .fold(
                (
                    vec![],
                    vec![vec![]; Instructions::COUNT],
                    vec![vec![]; preprocessing.num_memories],
                    vec![],
                ),
                |(mut eq_evals, mut flag_evals, mut E_evals, mut output_evals),
                 (eq_eval, flag_eval, E_eval, output_eval)| {
                    eq_evals.push(eq_eval);
                    izip!(&mut flag_evals, flag_eval).for_each(|(es, e)| es.push(e));
                    izip!(&mut E_evals, E_eval).for_each(|(es, e)| es.push(e));
                    output_evals.push(output_eval);
                    (eq_evals, flag_evals, E_evals, output_evals)
                },
            );

        let mut E_polys = E_evals
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect_vec();

        let mut flag_polys = flag_evals
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect_vec();

        let mut eq_poly: MultilinearPolynomial<F> = eq_evals.into();
        let mut outputs_poly: MultilinearPolynomial<F> = output_evals.into();

        for _round in 0..log_num_workers {
            let univariate_poly = Self::primary_sumcheck_prover_message(
                preprocessing,
                &eq_poly,
                &flag_polys,
                &E_polys,
                &outputs_poly,
                previous_claim,
            );

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r.push(r_j);

            previous_claim = univariate_poly.evaluate(&r_j);

            // Bind all polys
            flag_polys
                .par_iter_mut()
                .chain(E_polys.par_iter_mut())
                .chain([&mut eq_poly, &mut outputs_poly].into_par_iter())
                .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
        } // End rounds

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();
        let flag_evals = flag_polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();
        let memory_evals = E_polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();
        let outputs_eval = outputs_poly.final_sumcheck_claim();

        network.broadcast_request(r)?;

        Ok((compressed_polys, flag_evals, memory_evals, outputs_eval))
    }
}

use crate::subprotocols::grand_product::Rep3BatchedGrandProduct;

impl<F, const C: usize, const M: usize, PCS, ProofTranscript, Instructions, Subtables, Network>
    Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
    for InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Instructions: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    Network: Rep3NetworkCoordinator,
{
    type Rep3ReadWriteGrandProduct = Rep3ToggledBatchedGrandProduct<F>;
    type Rep3InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    fn receive_read_write_openings(
        num_ops: usize,
        preprocessing: &Self::Preprocessing,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(Self::Openings, Self::ExogenousOpenings)> {
        let exogenous_openings = Self::ExogenousOpenings::default();
        let mut openings = Self::Openings::initialize(preprocessing);

        if !network.is_distributed() {
            let read_write_evals: Vec<F> =
                opening_accumulator.append(num_ops.log_2(), transcript, network)?;

            openings
                .read_write_values_grand_product_mut()
                .into_par_iter()
                .zip(read_write_evals.par_iter())
                .for_each(|(opening, eval)| {
                    *opening = *eval;
                });

            // tracing::info!("instruction_flags claims: {:?}", openings.instruction_flags);

            return Ok((openings, exogenous_openings));
        }

        let num_workers = 1 << network.log_num_workers();
        let (batch_lens, openings) = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
            .into_iter()
            .enumerate()
            .map(|(worker_idx, shares)| {
                let evals = additive::combine_additive_vec(shares);
                let mut openings = Self::Openings::initialize(preprocessing);

                let read_memories = witness::read_write_memories_for_worker(
                    preprocessing.num_memories,
                    num_workers,
                    worker_idx,
                );

                openings.E_polys.truncate(read_memories.len());
                openings.read_cts.truncate(read_memories.len());

                openings
                    .read_write_values_grand_product_mut()
                    .into_par_iter()
                    .zip(evals.par_iter())
                    .for_each(|(opening, eval)| {
                        *opening = *eval;
                    });

                (vec![read_memories.len()], openings)
            })
            .reduce(|(mut lens_acc, mut openings_acc), (len, opening)| {
                openings_acc.E_polys.extend(opening.E_polys);
                openings_acc.read_cts.extend(opening.read_cts);
                lens_acc.push(len[0]);
                (lens_acc, openings_acc)
            })
            .unwrap();

        let claims = openings
            .read_write_values_grand_product()
            .into_iter()
            .copied()
            .collect::<Vec<_>>();

        // tracing::info!("instruction_flags claims: {:?}", openings.instruction_flags);

        let rho: F = transcript.challenge_scalar();
        let _span = tracing::trace_span!("rho_powers").entered();
        let mut rho_powers = vec![F::one()];
        for i in 1..claims.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }
        drop(_span);

        // Compute the random linear combination of the claims
        // let claim: F = rho_powers
        //     .iter()
        //     .zip(claims.iter())
        //     .map(|(scalar, eval)| *scalar * *eval)
        //     .sum();
        let claim: F = rho_powers[..4 /*claims.len() - Instructions::COUNT*/]
            .iter()
            .zip(&claims[..4 /*claims.len() - Instructions::COUNT*/])
            // .zip(openings.read_cts.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();

        // tracing::info!("rho_powers: {:?}", rho_powers);

        let mut offset = batch_lens[0];
        let mut rho_offsets = vec![vec![
            4,                              // [dim]..
            4 + preprocessing.num_memories, // [dim][read_cts]..
        ]];
        for len in &batch_lens[1..] {
            let prev = rho_offsets.last().unwrap();
            rho_offsets.push(vec![prev[0] + offset, prev[1] + offset]);
            offset += len;
        }
        network.send_requests_to_workers(
            rho_offsets
                .into_iter()
                .map(|offsets| (rho, offsets, claim))
                .collect(),
        )?;

        let claim_check: F = network
            .receive_response_from_workers::<F>(PartyID::ID0)
            .unwrap()
            .into_iter()
            .sum();
        assert_eq!(claim_check, claim);

        opening_accumulator.append_opening(Rep3CoordinatorOpening {
            poly_num_vars: num_ops.log_2(),
            claim,
        });

        Ok((openings, exogenous_openings))
    }

    fn read_write_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
        num_lookups: usize,
    ) -> Rep3ToggledBatchedGrandProduct<F> {
        <Rep3ToggledBatchedGrandProduct<F> as Rep3BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::construct(num_lookups.log_2() + 1)
    }
}
