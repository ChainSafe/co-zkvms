#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::split_eq_poly::DistributedSplitEqPolynomial;
use crate::poly::unipoly::unipoly_from_additive_evals;
use crate::utils::types::Rep3Value;
use jolt_core::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::BatchedCubicSumcheck;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::{additive, rep3::Rep3PrimeFieldShare};
use rayon::prelude::*;

use crate::poly::PolyDegree;
use jolt_core::poly::split_eq_poly::SplitEqPolynomial;
use jolt_core::utils::transcript::{AppendToTranscript, Transcript};

pub use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;

pub trait Rep3Bindable<F: JoltField>: Sync {
    fn bind(&mut self, r: F, party_id: PartyID);
}

pub trait Rep3BatchedCubicSumcheck<F, ProofTranscript, Network>: Rep3Bindable<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedCubicSumcheck::prove_sumcheck",
        level = "trace"
    )]
    fn coordinate_prove_sumcheck(
        &self,
        claim: &F,
        num_rounds: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, (F, F))> {
        let log_num_workers = network.log_num_workers();
        let mut previous_claim = *claim;

        let worker_num_rounds = if log_num_workers > 0 {
            num_rounds - log_num_workers - 2
        } else {
            num_rounds
        };

        let (mut sumcheck_proof, mut r) = coordinate_prove_arbitrary_distributed(
            &mut previous_claim,
            worker_num_rounds,
            transcript,
            network,
        )?;

        let final_claims = if network.log_num_workers() > 0 {
            self.prove_remaining_rounds(
                &mut r,
                previous_claim,
                &mut sumcheck_proof,
                transcript,
                network,
            )?
        } else {
            self.receive_final_claims(network)?
        };

        Ok((sumcheck_proof, r, final_claims))
    }

    fn receive_final_claims(&self, network: &mut Network) -> eyre::Result<(F, F)> {
        let final_claims: Vec<_> = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
            .into_iter()
            .flat_map(additive::combine_additive_vec)
            .collect();

        Ok((final_claims[0], final_claims[1]))
    }

    fn prove_remaining_rounds(
        &self,
        r: &mut Vec<F>,
        previous_claim: F,
        proof: &mut SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(F, F)> {
        let evals = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
            .into_iter()
            .flat_map(additive::combine_additive_vec)
            .collect();

        // Assumption: At round N-log_num_workers E_1 is completely bound,
        // meaning we switched over to the linear-time sumcheck prover, using E_2 := E_1 * E_2
        let E2 = network
            .receive_response_from_workers::<Vec<F>>(PartyID::ID0)?
            .into_iter()
            .flatten()
            .collect();
        let E1 = vec![F::zero()];
        let mut eq_poly = SplitEqPolynomial::new_bound(E1, E2);

        let mut layer = DenseInterleavedPolynomial::new(evals);

        let (proof_, r_, final_claims) =
            layer.prove_sumcheck(&previous_claim, &mut eq_poly, transcript);

        network.broadcast_request(r_.clone())?;
        proof.compressed_polys.extend(proof_.compressed_polys);
        r.extend(r_);

        Ok(final_claims)
    }
}

pub trait Rep3BatchedCubicSumcheckWorker<F: JoltField, Network: Rep3NetworkWorker>:
    Rep3Bindable<F>
{
    fn compute_cubic(
        &self,
        eq_poly: &DistributedSplitEqPolynomial<F>,
        party_id: PartyID,
    ) -> [AdditiveShare<F>; 3];

    fn final_evals(&self, party_id: PartyID) -> Vec<AdditiveShare<F>>;

    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedCubicSumcheck::prove_sumcheck_worker",
        level = "trace"
    )]
    fn prove_sumcheck(
        &mut self,
        eq_poly: &mut DistributedSplitEqPolynomial<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Vec<F>> {
        let mut num_rounds = eq_poly.get_num_vars();

        if io_ctx.log_num_workers() > 0 {
            num_rounds -= 2;
        };

        // let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let party_id = io_ctx.party_id();
        for _round in 0..num_rounds {
            let cubic_poly = self.compute_cubic(eq_poly, party_id);
            tracing::info!("compute_cubic round {}", _round);
            // append the prover's message to the transcript
            io_ctx.network().send_response(cubic_poly)?;
            let r_j = io_ctx.network().receive_request()?;

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(r_j, party_id);
            eq_poly.bind(r_j);

            tracing::info!("bind round {}", _round);
            // poly coeffs are additive shares but evaluation requires multiplication
            // e = poly.evaluate(&r_j);
            // since we sent coeffs shares earlier, we can just receive the evaluation from coordinator
            // previous_claim = additive::promote_to_trivial_share(next_claim, party_id);
        }

        debug_assert_eq!(eq_poly.len(), 1);

        let final_evals = self.final_evals(party_id);
        io_ctx.network().send_response(final_evals)?;

        if io_ctx.log_num_workers() > 0 {
            tracing::trace!("send remaining");
            if io_ctx.party_id() == PartyID::ID0 {
                io_ctx.network().send_response(eq_poly.E2.clone())?;
            }

            // Coordinator runs remaining sumcheck rounds
            r.extend(io_ctx.network().receive_request::<Vec<F>>()?);
        }

        Ok(r)
    }
}

#[tracing::instrument(
    skip_all,
    name = "coordinate_prove_arbitrary_distributed",
    level = "trace"
)]
pub fn coordinate_prove_arbitrary_distributed<F: JoltField, ProofTranscript, Network>(
    claim: &mut F,
    num_rounds: usize,
    transcript: &mut ProofTranscript,
    network: &mut Network,
) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>)>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    let mut r: Vec<F> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

    let mut tmp_e = vec![];
    for _round in 0..num_rounds {
        let mut round_evals = if network.log_num_workers() == 0 {
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
        round_evals.insert(1, *claim - round_evals[0]);

        let round_poly = UniPoly::<F>::from_evals(&round_evals);
        let compressed_poly = round_poly.compress();

        // append the prover's message to the transcript
        compressed_poly.append_to_transcript(transcript);
        // derive the verifier's challenge for the next round
        let r_j = transcript.challenge_scalar();
        r.push(r_j);

        *claim = round_poly.evaluate(&r_j);
        tmp_e.push(*claim);

        network.broadcast_request((r_j, *claim))?;

        cubic_polys.push(compressed_poly);
    }

    Ok((SumcheckInstanceProof::new(cubic_polys), r))
}

#[tracing::instrument(skip_all, name = "coordinate_prove_arbitrary", level = "trace")]
pub fn coordinate_prove_arbitrary<F: JoltField, ProofTranscript, Network>(
    num_rounds: usize,
    transcript: &mut ProofTranscript,
    network: &mut Network,
) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>)>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    let mut r: Vec<F> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

    for _round in 0..num_rounds {
        let round_poly =
            UniPoly::<F>::from_coeff(additive::combine_additive_vec(network.receive_responses()?));
        let compressed_poly = round_poly.compress();

        // append the prover's message to the transcript
        compressed_poly.append_to_transcript(transcript);
        // derive the verifier's challenge for the next round
        let r_j = transcript.challenge_scalar();
        r.push(r_j);

        let claim = round_poly.evaluate(&r_j);

        network.broadcast_request((r_j, claim))?;

        cubic_polys.push(compressed_poly);
    }

    Ok((SumcheckInstanceProof::new(cubic_polys), r))
}

#[tracing::instrument(skip_all, name = "sumcheck::prove_arbitrary_worker")]
pub fn prove_arbitrary_worker<F, Poly, Func, Network>(
    claim: &AdditiveShare<F>,
    num_rounds: usize,
    polys: &mut Vec<Poly>,
    comb_func: Func,
    combined_degree: usize,
    io_ctx: &mut IoContextPool<Network>,
) -> eyre::Result<(Vec<F>, Vec<AdditiveShare<F>>)>
where
    F: JoltField,
    Poly: PolynomialBinding<F, Rep3Value<F>>
        + PolynomialEvaluation<F, Rep3Value<F>>
        + PolyDegree
        + Send
        + Sync,
    Func: Fn(&[Rep3Value<F>]) -> AdditiveShare<F> + std::marker::Sync,
    Network: Rep3NetworkWorker,
{
    let mut previous_claim = *claim;
    let mut r: Vec<F> = Vec::new();

    for _round in 0..num_rounds {
        // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
        // for points {0, ..., |g(x)|}
        let mut eval_points = vec![AdditiveShare::<F>::zero(); combined_degree];

        let mle_half = polys[0].len() / 2;

        let accum: Vec<Vec<AdditiveShare<F>>> = (0..mle_half)
            .into_par_iter()
            .map(|poly_term_i| {
                let mut accum = vec![AdditiveShare::<F>::zero(); combined_degree];
                // TODO Optimize
                let evals: Vec<_> = polys
                    .iter()
                    .map(|poly| {
                        poly.sumcheck_evals(poly_term_i, combined_degree, BindingOrder::HighToLow)
                    })
                    .collect();
                for j in 0..combined_degree {
                    let evals_j: Vec<_> = evals.iter().map(|x| x[j]).collect();
                    accum[j] += comb_func(&evals_j);
                }

                accum
            })
            .collect();

        eval_points
            .par_iter_mut()
            .enumerate()
            .for_each(|(poly_i, eval_point)| {
                *eval_point = accum.par_iter().take(mle_half).map(|mle| mle[poly_i]).sum();
            });

        eval_points.insert(1, previous_claim - eval_points[0]);
        let univariate_poly = unipoly_from_additive_evals(&eval_points);
        io_ctx.network().send_response(univariate_poly.coeffs)?;

        // append the prover's message to the transcript
        // compressed_poly.append_to_transcript(transcript);
        // let r_j = transcript.challenge_scalar();
        let (r_j, next_claim) = io_ctx.network().receive_request()?;
        r.push(r_j);

        // bound all tables to the verifier's challenge
        polys
            .par_iter_mut()
            .for_each(|poly| poly.bind(r_j, BindingOrder::HighToLow));
        previous_claim = additive::promote_to_trivial_share(next_claim, io_ctx.party_id());
    }

    let final_evals = polys
        .iter()
        .map(|poly| poly.final_sumcheck_claim().into_additive(io_ctx.party_id()))
        .collect();

    Ok((r, final_evals))
}
