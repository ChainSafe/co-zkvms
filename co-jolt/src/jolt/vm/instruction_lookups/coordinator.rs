use crate::field::JoltField;
use crate::{
    lasso::memory_checking::Rep3MemoryCheckingProver,
    poly::{commitment::Rep3CommitmentScheme, opening_proof::Rep3ProverOpeningAccumulator},
    subprotocols::{
        grand_product::Rep3BatchedDenseGrandProduct,
        sparse_grand_product::Rep3ToggledBatchedGrandProduct,
    },
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::{chain, izip, Itertools};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::transcript::{AppendToTranscript, Transcript};
use jolt_core::{
    jolt::subtable::JoltSubtableSet,
    poly::unipoly::{CompressedUniPoly, UniPoly},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::math::Math,
};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use std::iter::once;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

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
        transcript: &mut ProofTranscript,
    ) -> Result<InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>>
    {
        transcript.append_message(Self::protocol_name());

        let r_eq = transcript.challenge_vector::<F>(num_ops.log_2());
        network.broadcast_request(r_eq)?;

        let num_rounds = num_ops.log_2();

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

        Rep3ProverOpeningAccumulator::<F>::coordinate_with_known_claims(
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

        let memory_checking =
            Self::coordinate_memory_checking(preprocessing, num_ops, M, transcript, network)
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
        let log_num_workers = network.log_num_workers_per_party();
        if log_num_workers > 0 {
            network.extend_with_worker_subnets(1 << log_num_workers)?;
        }

        let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        for _ in 0..num_rounds - log_num_workers {
            let mut round_evals = if log_num_workers == 0 {
                additive::combine_additive_vec(network.receive_responses()?)
            } else {
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
            };
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
        if log_num_workers > 0 {
            let (eq_evals, flag_evals, E_evals, output_evals) = network
                .receive_responses_from_subnets::<(
                    F,
                    Vec<F>,
                    Vec<AdditiveShare<F>>,
                    AdditiveShare<F>,
                )>()?
                .into_iter()
                .map(|shares| {
                    let (eq_evals, mut flag_evals, E_evals, output_evals):  (Vec<_>, Vec<_>, Vec<_>, Vec<_>)= shares.into_iter().multiunzip();

                    (eq_evals[0], flag_evals.pop().unwrap(), additive::combine_additive_vec(E_evals), additive::combine_additive_share(output_evals))
                })
                .fold((vec![], vec![vec![]; Instructions::COUNT], vec![vec![]; preprocessing.num_memories], vec![]), |(mut eq_evals, mut flag_evals, mut E_evals, mut output_evals), (eq_eval, flag_eval, E_eval, output_eval)| {
                    eq_evals.push(eq_eval);
                    izip!(&mut flag_evals, flag_eval).for_each(|(es, e)| es.push(e));
                    izip!(&mut E_evals, E_eval).for_each(|(es, e)| es.push(e));
                    output_evals.push(output_eval);
                    (eq_evals, flag_evals, E_evals, output_evals)
                });

            let mut E_polys = E_evals
                .into_iter()
                .map(MultilinearPolynomial::from)
                .collect_vec();

            let mut flag_polys = flag_evals
                .into_iter()
                .map(MultilinearPolynomial::from)
                .collect_vec();

            let eq_poly: MultilinearPolynomial<F> = eq_evals.into();

            let (proof, r_final, flag_evals, E_evals, outputs_eval) = InstructionLookupsProof::<
                C,
                M,
                F,
                PCS,
                Instructions,
                Subtables,
                ProofTranscript,
            >::prove_primary_sumcheck(
                preprocessing,
                log_num_workers,
                eq_poly,
                &mut E_polys,
                &mut flag_polys,
                &mut output_evals.into(),
                transcript,
            );

            network.broadcast_request(())?;

            compressed_polys.extend(proof.compressed_polys);
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
