#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use itertools::multizip;
use jolt_core::subprotocols::sumcheck::Bindable;
use jolt_core::utils::thread::drop_in_background_thread;
use mpc_core::protocols::rep3::network::{IoContext, Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, PartyID};
use mpc_core::protocols::{additive, rep3::Rep3PrimeFieldShare};
use mpc_net::mpc_star::MpcStarNetWorker;
use rayon::prelude::*;
use tracing::trace_span;

use crate::poly::Rep3DensePolynomial;
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
    #[tracing::instrument(skip_all, name = "Rep3BatchedCubicSumcheck::prove_sumcheck")]
    fn coordinate_prove_sumcheck(
        &self,
        num_rounds: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, (F, F))> {
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for round in 0..num_rounds {
            let round_poly = UniPoly::<F>::from_coeff(additive::combine_field_element_vec(
                network.receive_responses(Vec::new())?,
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

        let final_claims = self.receive_final_claims(network)?;

        Ok((SumcheckInstanceProof::new(cubic_polys), r, final_claims))
    }

    fn receive_final_claims(&self, network: &mut Network) -> eyre::Result<(F, F)> {
        let (final_claims_shares_l, final_claims_shares_r): (
            Vec<Rep3PrimeFieldShare<F>>,
            Vec<Rep3PrimeFieldShare<F>>,
        ) = network
            .receive_responses::<(Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>)>((
                Default::default(),
                Default::default(),
            ))?
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

    #[tracing::instrument(skip_all, name = "Rep3BatchedCubicSumcheck::prove_sumcheck_worker")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        eq_poly: &mut SplitEqPolynomial<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<(Vec<F>, (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>))> {
        let num_rounds = eq_poly.get_num_vars();

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let party_id = io_ctx.network.get_id();
        for _round in 0..num_rounds {
            let cubic_poly = self.compute_cubic(eq_poly, previous_claim, party_id);
            // append the prover's message to the transcript
            io_ctx.network.send_response(cubic_poly.as_vec())?;
            let (r_j, next_claim) = io_ctx.network.receive_request()?;

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
        io_ctx.network.send_response(final_claims)?;

        Ok((r, final_claims))
    }
}
