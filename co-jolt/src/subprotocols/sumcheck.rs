#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::utils::element::SharedOrPublic;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::poly::{
    dense_mlpoly::DensePolynomial,
    unipoly::{CompressedUniPoly, UniPoly},
};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContext, IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, PartyID};
use mpc_core::protocols::{additive, rep3::Rep3PrimeFieldShare};
use rayon::prelude::*;

use crate::poly::{PolyDegree, Rep3DensePolynomial, Rep3MultilinearPolynomial};
use jolt_core::poly::split_eq_poly::{GruenSplitEqPolynomial, SplitEqPolynomial};
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
        num_rounds: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, (F, F))> {
        let (sumcheck_proof, r) = coordinate_prove_arbitrary(num_rounds, transcript, network)?;

        let final_claims = self.receive_final_claims(network)?;

        Ok((sumcheck_proof, r, final_claims))
    }

    fn receive_final_claims(&self, network: &mut Network) -> eyre::Result<(F, F)> {
        let (final_claims_shares_l, final_claims_shares_r): (
            Vec<Rep3PrimeFieldShare<F>>,
            Vec<Rep3PrimeFieldShare<F>>,
        ) = network
            .receive_responses::<(Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>)>()?
            .into_iter()
            .unzip();

        let final_claims = (
            rep3::combine_field_element(
                final_claims_shares_l[0],
                final_claims_shares_l[1],
                final_claims_shares_l[2],
            ),
            rep3::combine_field_element(
                final_claims_shares_r[0],
                final_claims_shares_r[1],
                final_claims_shares_r[2],
            ),
        );

        Ok(final_claims)
    }
}

pub trait Rep3BatchedCubicSumcheckWorker<F: JoltField, Network: Rep3NetworkWorker>:
    Rep3Bindable<F>
{
    fn compute_cubic(
        &self,
        eq_poly: &SplitEqPolynomial<F>,
        previous_round_claim: F,
        party_id: PartyID,
    ) -> UniPoly<F>;
    
    fn final_claims(&self, party_id: PartyID) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>);

    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedCubicSumcheck::prove_sumcheck_worker",
        level = "trace"
    )]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        eq_poly: &mut SplitEqPolynomial<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<(Vec<F>, (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>))> {
        let num_rounds = eq_poly.get_num_vars();

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let party_id = io_ctx.party_id();
        for _round in 0..num_rounds {
            let cubic_poly = self.compute_cubic(eq_poly, previous_claim, party_id);
            // append the prover's message to the transcript
            io_ctx.network().send_response(cubic_poly.as_vec())?;
            let (r_j, next_claim) = io_ctx.network().receive_request()?;

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(r_j, party_id);
            eq_poly.bind(r_j);

            // poly coeffs are additive shares but evaluation requires multiplication
            // e = poly.evaluate(&r_j);
            // since we sent coeffs shares earlier, we can just receive the evaluation from coordinator
            previous_claim = additive::promote_to_trivial_share(next_claim, party_id);
        }

        debug_assert_eq!(eq_poly.len(), 1);

        let final_claims = self.final_claims(party_id);
        io_ctx.network().send_response(final_claims)?;

        Ok((r, final_claims))
    }
}

#[tracing::instrument(skip_all, name = "Sumcheck.prove", level = "trace")]
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
        let round_poly = UniPoly::<F>::from_coeff(additive::combine_field_element_vec(
            network.receive_responses()?,
        ));
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
) -> eyre::Result<(Vec<F>, Vec<F>)>
where
    F: JoltField,
    Poly: PolynomialBinding<F>
        + PolynomialEvaluation<F, SharedOrPublic<F>>
        + PolyDegree
        + Send
        + Sync,
    Func: Fn(&[SharedOrPublic<F>]) -> AdditiveShare<F> + std::marker::Sync,
    Network: Rep3NetworkWorker,
{
    let mut previous_claim = *claim;
    let mut r: Vec<F> = Vec::new();

    for _round in 0..num_rounds {
        // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
        // for points {0, ..., |g(x)|}
        let mut eval_points = vec![F::zero(); combined_degree];

        let mle_half = polys[0].len() / 2;

        let accum: Vec<Vec<F>> = (0..mle_half)
            .into_par_iter()
            .map(|poly_term_i| {
                let mut accum = vec![F::zero(); combined_degree];
                // TODO(moodlezoup): Optimize
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
                *eval_point = accum
                    .par_iter()
                    .take(mle_half)
                    .map(|mle| mle[poly_i])
                    .sum::<F>();
            });

        eval_points.insert(1, previous_claim - eval_points[0]);
        let univariate_poly = UniPoly::from_evals(&eval_points);
        io_ctx.network().send_response(univariate_poly.as_vec())?;

        // append the prover's message to the transcript
        // compressed_poly.append_to_transcript(transcript);
        // let r_j = transcript.challenge_scalar();
        let (r_j, next_claim) = io_ctx.network().receive_request()?;
        r.push(r_j);

        // bound all tables to the verifier's challenge
        polys
            .par_iter_mut()
            .for_each(|poly| poly.bind(r_j, BindingOrder::HighToLow));
        previous_claim = additive::promote_to_trivial_share(next_claim, io_ctx.id);
    }

    let final_evals = polys
        .iter()
        .map(|poly| poly.final_sumcheck_claim())
        .collect();

    Ok((r, final_evals))
}
