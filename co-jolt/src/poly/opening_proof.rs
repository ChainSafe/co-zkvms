use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding,
};
pub use jolt_core::poly::opening_proof::*;
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::utils::transcript::AppendToTranscript;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::{
    additive,
    rep3::{
        self,
        network::{IoContext, Rep3NetworkCoordinator, Rep3NetworkWorker},
    },
};

use crate::{
    field::JoltField,
    poly::{commitment::Rep3CommitmentScheme, Rep3MultilinearPolynomial},
    utils::transcript::Transcript,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// An opening computed by the prover.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3ProverOpening<F: JoltField> {
    /// The polynomial being opened. May be a random linear combination
    /// of multiple polynomials all being opened at the same point.
    pub polynomial: Rep3MultilinearPolynomial<F>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: MultilinearPolynomial<F>,
    /// The point at which the `polynomial` is being evaluated.
    pub opening_point: Vec<F>,
    /// The claimed opening.
    pub claim: Rep3PrimeFieldShare<F>,
}

impl<F: JoltField> Rep3ProverOpening<F> {
    fn new(
        polynomial: Rep3MultilinearPolynomial<F>,
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claim: Rep3PrimeFieldShare<F>,
    ) -> Self {
        Rep3ProverOpening {
            polynomial,
            eq_poly: MultilinearPolynomial::LargeScalars(eq_poly),
            opening_point,
            claim,
        }
    }
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
pub struct Rep3ProverOpeningAccumulator<F: JoltField> {
    openings: Vec<Rep3ProverOpening<F>>,
}

impl<F: JoltField> Rep3ProverOpeningAccumulator<F> {
    pub fn new() -> Self {
        Self { openings: vec![] }
    }

    pub fn len(&self) -> usize {
        self.openings.len()
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append")]
    pub fn append<Network: Rep3NetworkWorker>(
        &mut self,
        polynomials: &[&Rep3MultilinearPolynomial<F>],
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claims: &[AdditiveShare<F>],
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        assert_eq!(polynomials.len(), claims.len());

        io_ctx.network.send_response(claims.to_vec())?;
        let (rho, batched_claim): (F, F) = io_ctx.network.receive_request()?;

        // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
        let mut rho_powers = vec![F::one()];
        for i in 1..polynomials.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let batched_poly =
            Rep3MultilinearPolynomial::linear_combination(polynomials, &rho_powers, io_ctx.id);

        let batched_claim_rep3 =
            rep3::arithmetic::promote_to_trivial_share(io_ctx.id, batched_claim);
        let opening =
            Rep3ProverOpening::new(batched_poly, eq_poly, opening_point, batched_claim_rep3);
        self.openings.push(opening);

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "Rep3ProverOpeningAccumulator::receive_claims")]
    pub fn receive_claims<ProofTranscript: Transcript, Network: Rep3NetworkCoordinator>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Vec<F>> {
        let claims = additive::combine_field_element_vec(network.receive_responses()?);
        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..claims.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }
        // Compute the random linear combination of the claims
        let batched_claim: F = rho_powers
            .iter()
            .zip(claims.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();
        network.broadcast_request((rho, batched_claim))?;
        Ok(claims)
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_public")]
    pub fn append_public<Network: Rep3NetworkWorker>(
        &mut self,
        opening: &ProverOpening<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let ProverOpening {
            polynomial,
            eq_poly,
            opening_point,
            claim,
            ..
        } = opening;

        let eq_poly: DensePolynomial<F> = eq_poly.clone().try_into().unwrap();

        let opening_for = |party_id: PartyID| -> Rep3ProverOpening<F> {
            // TODO: should we promote to shared? would reduce communication between parties but computation overhead during reduce_and_prove?
            let polynomial = Rep3MultilinearPolynomial::public(polynomial.clone());
            let claim = rep3::arithmetic::promote_to_trivial_share(party_id, *claim);

            Rep3ProverOpening::new(
                polynomial.clone(),
                eq_poly.clone(),
                opening_point.clone(),
                claim,
            )
        };

        self.openings.push(opening_for(io_ctx.id));

        io_ctx
            .network
            .send(io_ctx.id.next_id(), opening_for(io_ctx.id.next_id()))?;
        io_ctx
            .network
            .send(io_ctx.id.prev_id(), opening_for(io_ctx.id.prev_id()))?;

        Ok(())
    }

    pub fn receive_public_opening<Network: Rep3NetworkWorker>(
        &mut self,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let opening = io_ctx.network.recv(PartyID::ID0)?;
        self.openings.push(opening);
        Ok(())
    }

    /// Reduces the multiple openings accumulated into a single opening proof,
    /// using a single sumcheck.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove<PCS, ProofTranscript, Network>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<ReducedOpeningProof<F, PCS, ProofTranscript>>
    where
        Network: Rep3NetworkCoordinator,
        ProofTranscript: Transcript,
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    {
        // Generate coefficients for random linear combination
        let rho: F = transcript.challenge_scalar();
        network.broadcast_request(rho)?;

        let max_num_vars = network.receive_response(PartyID::ID0, 0)?;

        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..max_num_vars {
            let uni_poly = UniPoly::from_coeff(additive::combine_field_element_vec(
                network.receive_responses()?,
            ));
            let compressed_poly = uni_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);
            let new_claim = uni_poly.evaluate(&r_j);

            network.broadcast_request((r_j, new_claim))?;

            compressed_polys.push(compressed_poly);
        }

        let sumcheck_proof = SumcheckInstanceProof::new(compressed_polys);

        let sumcheck_claims = additive::combine_field_element_vec(network.receive_responses()?);

        transcript.append_scalars(&sumcheck_claims);

        let gamma: F = transcript.challenge_scalar();
        network.broadcast_request(gamma)?;

        // Reduced opening proof
        let joint_opening_proof = PCS::coordinate_prove(network)?;

        Ok(ReducedOpeningProof {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        })
    }

    /// Reduces the multiple openings accumulated into a single opening proof,
    /// using a single sumcheck.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove_worker<PCS, ProofTranscript, Network>(
        &mut self,
        pcs_setup: &PCS::Setup,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker,
        ProofTranscript: Transcript,
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    {
        // Generate coefficients for random linear combination
        let rho: F = io_ctx.network.receive_request()?;
        let mut rho_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        // TODO(moodlezoup): surely there's a better way to do this
        let unbound_polys = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.clone())
            .collect::<Vec<_>>();

        // Use sumcheck reduce many openings to one
        let (r_sumcheck, sumcheck_claims) =
            self.prove_batch_opening_reduction(&rho_powers, io_ctx)?;

        io_ctx.network.send_response(sumcheck_claims)?;

        let gamma: F = io_ctx.network.receive_request()?;
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let joint_poly = Rep3MultilinearPolynomial::linear_combination(
            &unbound_polys.iter().collect::<Vec<_>>(),
            &gamma_powers,
            io_ctx.id,
        );

        let joint_poly = match joint_poly {
            Rep3MultilinearPolynomial::Shared(poly) => poly,
            Rep3MultilinearPolynomial::Public { .. } => {
                panic!("Joint polynomial is expected to be shared")
            }
        };

        // Reduced opening proof
        PCS::prove_rep3(&joint_poly, pcs_setup, &r_sumcheck, &mut io_ctx.network)?;

        Ok(())
    }

    /// Proves the sumcheck used to prove the reduction of many openings into one.
    #[tracing::instrument(skip_all, name = "prove_batch_opening_reduction")]
    pub fn prove_batch_opening_reduction<Network: Rep3NetworkWorker>(
        &mut self,
        coeffs: &[F],
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<(Vec<F>, Vec<F>)> {
        let max_num_vars = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.get_num_vars())
            .max()
            .unwrap();

        if io_ctx.id == PartyID::ID0 {
            io_ctx.network.send_response(max_num_vars)?;
        }

        // Compute random linear combination of the claims, accounting for the fact that the
        // polynomials may be of different sizes
        let mut e: F = coeffs
            .par_iter()
            .zip(self.openings.par_iter_mut())
            .map(|(coeff, opening)| {
                let scaled_claim = if opening.polynomial.get_num_vars() != max_num_vars {
                    rep3::arithmetic::mul_public(
                        opening.claim,
                        F::from_u64_unchecked(
                            1 << (max_num_vars - opening.polynomial.get_num_vars()),
                        ),
                    )
                } else {
                    opening.claim
                };
                rep3::arithmetic::mul_public(scaled_claim, *coeff).into_additive()
            })
            .sum();

        let mut r: Vec<F> = Vec::new();

        for round in 0..max_num_vars {
            let remaining_rounds = max_num_vars - round;
            let uni_poly = self.compute_quadratic(coeffs, remaining_rounds, e, io_ctx.id);
            io_ctx.network.send_response(uni_poly.as_vec())?;

            // append the prover's message to the transcript
            let (r_j, new_claim) = io_ctx.network.receive_request()?;
            r.push(r_j);
            e = additive::promote_to_trivial_share(new_claim, io_ctx.id);

            self.openings.par_iter_mut().for_each(|opening| {
                if remaining_rounds <= opening.opening_point.len() {
                    rayon::join(
                        || opening.eq_poly.bind(r_j, BindingOrder::HighToLow),
                        || opening.polynomial.bind(r_j, BindingOrder::HighToLow),
                    );
                }
            });
        }

        let claims: Vec<_> = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.get_bound_coeff(0).into_additive(io_ctx.id))
            .collect();

        Ok((r, claims))
    }

    /// Computes the univariate (quadratic) polynomial that serves as the
    /// prover's message in each round of the sumcheck in `prove_batch_opening_reduction`.
    #[tracing::instrument(
        skip_all,
        name = "Rep3ProverOpeningAccumulator::compute_quadratic",
        level = "trace"
    )]
    fn compute_quadratic(
        &self,
        coeffs: &[F],
        remaining_sumcheck_rounds: usize,
        previous_round_claim: F,
        party_id: PartyID,
    ) -> UniPoly<F> {
        let evals: Vec<(F, F)> = self
            .openings
            .par_iter()
            .zip(coeffs.par_iter())
            .map(|(opening, coeff)| {
                if remaining_sumcheck_rounds <= opening.opening_point.len() {
                    let mle_half = opening.polynomial.len() / 2;
                    let eval_0 = (0..mle_half)
                        .map(|i| {
                            opening
                                .polynomial
                                .get_bound_coeff(i)
                                .mul_public(opening.eq_poly.get_bound_coeff(i) * coeff)
                                .into_additive(party_id)
                        })
                        .sum();
                    let eval_2 = (0..mle_half)
                        .map(|i| {
                            let poly_bound_point = opening
                                .polynomial
                                .get_bound_coeff(i + mle_half)
                                .add(&opening.polynomial.get_bound_coeff(i + mle_half), party_id)
                                .sub(&opening.polynomial.get_bound_coeff(i), party_id);
                            let eq_bound_point = opening.eq_poly.get_bound_coeff(i + mle_half)
                                + opening.eq_poly.get_bound_coeff(i + mle_half)
                                - opening.eq_poly.get_bound_coeff(i);
                            poly_bound_point
                                .mul_public(eq_bound_point * coeff)
                                .into_additive(party_id)
                        })
                        .sum();
                    (eval_0, eval_2)
                } else {
                    // debug_assert!(!opening.polynomial.is_bound());
                    let remaining_variables =
                        remaining_sumcheck_rounds - opening.opening_point.len() - 1;
                    let scaled_claim = rep3::arithmetic::mul_public(
                        opening.claim,
                        F::from_u64_unchecked(1 << remaining_variables) * coeff,
                    )
                    .into_additive();
                    (scaled_claim, scaled_claim)
                }
            })
            .collect();

        let evals_combined_0: F = (0..evals.len()).map(|i| evals[i].0).sum();
        let evals_combined_2: F = (0..evals.len()).map(|i| evals[i].1).sum();
        let evals = vec![
            evals_combined_0,
            previous_round_claim - evals_combined_0,
            evals_combined_2,
        ];

        UniPoly::from_evals(&evals)
    }
}
