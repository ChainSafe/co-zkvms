use std::os::unix::net;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::izip;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding,
};
pub use jolt_core::poly::opening_proof::*;
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::utils::transcript::AppendToTranscript;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::IoContextPool;
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
        Ok(())
        // self.append_with_known_claim(polynomials, eq_poly, opening_point, io_ctx)
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append")]
    pub fn append_with_known_claim<Network: Rep3NetworkWorker>(
        &mut self,
        polynomials: &[&Rep3MultilinearPolynomial<F>],
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
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
        let claims = if network.is_distributed() {
            network
                .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
                .into_iter()
                .flat_map(additive::combine_additive_vec)
                .collect()
        } else {
            additive::combine_additive_vec(network.receive_responses()?)
        };
        Self::coordinate_with_known_claims(&claims, transcript, network)?;
        Ok(claims)
    }

    #[tracing::instrument(skip_all, name = "Rep3ProverOpeningAccumulator::receive_claims")]
    pub fn coordinate_with_known_claims<
        ProofTranscript: Transcript,
        Network: Rep3NetworkCoordinator,
    >(
        claims: &[F],
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<()> {
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
        Ok(())
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
        pcs_setup: &PCS::Setup,
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
        tracing::info!("rho = {}", rho);
        network.broadcast_request(rho)?;

        let max_num_vars: usize = network
            .receive_response_from_workers::<usize>(PartyID::ID0)?
            .into_iter()
            .max()
            .unwrap();

        tracing::info!("max_num_vars: {}", max_num_vars);

        let mut combined_claim = network
            .receive_responses_from_subnets::<AdditiveShare<F>>()?
            .into_iter()
            .map(additive::combine_additive_share)
            .sum::<F>();

        tracing::info!("combined_claim: {}", combined_claim);

        // network.broadcast_request(e)?;

        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let log_num_workers = network.log_num_workers();

        for _round in 0..max_num_vars {
            let mut round_evals = if network.is_distributed() {
                network
                    .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
                    .into_iter()
                    .map(|shares| additive::combine_additive_vec(shares))
                    .fold(vec![F::zero(); 2], |mut acc, coeff| {
                        acc.iter_mut().zip(coeff.iter()).for_each(|(acc, coeff)| {
                            *acc += coeff;
                        });
                        acc
                    })
            } else {
                additive::combine_additive_vec(network.receive_responses()?)
            };
            tracing::info!("round evals: {:?}", round_evals);
            round_evals.insert(1, combined_claim - round_evals[0]);
            let uni_poly = UniPoly::from_evals(&round_evals);
            let compressed_poly = uni_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);
            combined_claim = uni_poly.evaluate(&r_j);
            // tracing::info!("next_claim: {:?} r_j: {:?}", combined_claim, r_j);

            network.broadcast_request(r_j)?;

            compressed_polys.push(compressed_poly);
        }

        if network.is_distributed() {
            for _round in 0..log_num_workers {}
        }

        let sumcheck_proof = SumcheckInstanceProof::new(compressed_polys);

        let sumcheck_claims = if network.is_distributed() {
            network
                .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
                .into_iter()
                .map(|shares| additive::combine_additive_vec(shares))
                .reduce(|acc, coeff| izip!(acc, coeff).map(|(a, b)| a + b).collect())
                .unwrap()
        } else {
            additive::combine_additive_vec(network.receive_responses()?)
        };
        transcript.append_scalars(&sumcheck_claims);

        let gamma: F = transcript.challenge_scalar();
        network.broadcast_request(gamma)?;

        // Reduced opening proof
        let joint_opening_proof = if network.is_distributed() {
            PCS::merge_proofs_rep3(pcs_setup, &r, network)?
        } else {
            PCS::coordinate_prove(network)?
        };

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
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker,
        ProofTranscript: Transcript,
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    {
        // Generate coefficients for random linear combination
        let rho: F = io_ctx.network().receive_request()?;
        let mut rho_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        // TODO: surely there's a better way to do this
        let unbound_polys = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.clone())
            .collect::<Vec<_>>();

        // Use sumcheck reduce many openings to one
        let (r_sumcheck, sumcheck_claims) =
            self.prove_batch_opening_reduction(&rho_powers, io_ctx)?;

        io_ctx.network().send_response(sumcheck_claims)?;

        let gamma: F = io_ctx.network().receive_request()?;
        tracing::info!("gamma: {:?}", gamma);
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        tracing::info!(
            "unbound_polys: {:?}",
            unbound_polys
                .iter()
                .map(|poly| poly.len())
                .collect::<Vec<_>>()
        );

        let joint_poly = Rep3MultilinearPolynomial::linear_combination(
            &unbound_polys.iter().collect::<Vec<_>>(),
            &gamma_powers,
            io_ctx.party_id(),
        );

        tracing::info!("joint polynomial: {:?}", joint_poly.len());

        let joint_poly = match joint_poly {
            Rep3MultilinearPolynomial::Shared(poly) => poly,
            Rep3MultilinearPolynomial::Public { .. } => {
                panic!("Joint polynomial is expected to be shared")
            }
        };

        // Reduced opening proof
        if io_ctx.network().is_distributed() {
            PCS::distributed_prove_rep3(&joint_poly, pcs_setup, &r_sumcheck, io_ctx.network())?;
        } else {
            PCS::prove_rep3(&joint_poly, pcs_setup, &r_sumcheck, io_ctx.network())?;
        }

        Ok(())
    }

    /// Proves the sumcheck used to prove the reduction of many openings into one.
    #[tracing::instrument(skip_all, name = "prove_batch_opening_reduction")]
    pub fn prove_batch_opening_reduction<Network: Rep3NetworkWorker>(
        &mut self,
        coeffs: &[F],
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<(Vec<F>, Vec<AdditiveShare<F>>)> {
        tracing::info!(
            "openings: {:?}",
            self.openings
                .iter()
                .map(|p| p.polynomial.get_num_vars())
                .collect::<Vec<_>>()
        );

        let max_num_vars = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.get_num_vars())
            .max()
            .unwrap();

        if io_ctx.party_idx() == 0 {
            tracing::info!("max_num_vars: {}", max_num_vars);

            io_ctx.network().send_response(max_num_vars)?;
        }

        // Compute random linear combination of the claims, accounting for the fact that the
        // polynomials may be of different sizes
        let e: AdditiveShare<F> = coeffs
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
                scaled_claim.into_additive() * *coeff
            })
            .sum();

        io_ctx.network().send_response(e)?;
        // e = io_ctx.network.receive_request()?;

        let mut r: Vec<F> = Vec::new();

        for round in 0..max_num_vars {
            let remaining_rounds = max_num_vars - round;
            let evals = self.compute_quadratic(coeffs, remaining_rounds, io_ctx.party_id());
            io_ctx.network().send_response(evals.to_vec())?;

            // append the prover's message to the transcript
            let r_j = io_ctx.network().receive_request()?;
            r.push(r_j);
            // e = additive::promote_to_trivial_share(new_claim, io_ctx.id);

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
            .map(|opening| {
                opening
                    .polynomial
                    .get_bound_coeff(0)
                    .into_additive(io_ctx.party_id())
            })
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
        party_id: PartyID,
    ) -> [AdditiveShare<F>; 2] {
        let evals: Vec<(AdditiveShare<F>, AdditiveShare<F>)> = self
            .openings
            .par_iter()
            .map(|opening| {
                if remaining_sumcheck_rounds <= opening.opening_point.len() {
                    let mle_half = opening.polynomial.len() / 2;
                    let eval_0 = (0..mle_half)
                        .map(|i| {
                            opening
                                .polynomial
                                .get_bound_coeff(i)
                                .mul_public(opening.eq_poly.get_bound_coeff(i))
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
                                .mul_public(eq_bound_point)
                                .into_additive(party_id)
                        })
                        .sum();
                    (eval_0, eval_2)
                } else {
                    // debug_assert!(!opening.polynomial.is_bound());
                    let remaining_variables =
                        remaining_sumcheck_rounds - opening.opening_point.len() - 1;
                    let scaled_claim = opening.claim.into_additive()
                        * F::from_u64_unchecked(1 << remaining_variables);
                    (scaled_claim, scaled_claim)
                }
            })
            .collect();

        let evals_combined_0: AdditiveShare<F> =
            (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
        let evals_combined_2: AdditiveShare<F> =
            (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();

        [evals_combined_0, evals_combined_2]
    }
}
