use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::{izip, Itertools};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
pub use jolt_core::poly::opening_proof::*;
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::utils::transcript::AppendToTranscript;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::IoContextPool;
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::{
    additive,
    rep3::network::{IoContext, Rep3NetworkCoordinator, Rep3NetworkWorker},
};

use crate::utils::types::Rep3Value;
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
    pub claim: AdditiveShare<F>,
}

impl<F: JoltField> Rep3ProverOpening<F> {
    fn new(
        polynomial: Rep3MultilinearPolynomial<F>,
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claim: AdditiveShare<F>,
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
pub struct Rep3OpeningAccumulatorWorker<F: JoltField> {
    openings: Vec<Rep3ProverOpening<F>>,
}

impl<F: JoltField> Rep3OpeningAccumulatorWorker<F> {
    pub fn new() -> Self {
        Self { openings: vec![] }
    }

    pub fn len(&self) -> usize {
        self.openings.len()
    }

    pub fn append_opening(
        &mut self,
        polynomial: Rep3MultilinearPolynomial<F>,
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claim: AdditiveShare<F>,
    ) {
        self.openings.push(Rep3ProverOpening::new(
            polynomial,
            eq_poly,
            opening_point,
            claim,
        ));
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append")]
    pub fn append<Network: Rep3NetworkWorker>(
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
            Rep3MultilinearPolynomial::linear_combination(&polynomials, &rho_powers, io_ctx.id);
        tracing::info!("batched_poly: {:?}", batched_poly.len());

        self.openings.push(Rep3ProverOpening::new(
            batched_poly,
            eq_poly,
            opening_point,
            additive::promote_to_trivial_share(batched_claim, io_ctx.id),
        ));

        Ok(())
    }

    pub fn append_send_claims<Network: Rep3NetworkWorker>(
        &mut self,
        polynomials: &[&Rep3MultilinearPolynomial<F>],
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claims: &[Rep3Value<F>],
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.id;
        io_ctx.network.send_response(
            claims
                .par_iter()
                .map(|x| x.into_additive(party_id))
                .collect::<Vec<_>>(),
        )?;
        self.append(polynomials, eq_poly, opening_point, io_ctx)
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_batched")]
    pub fn append_batched<Network: Rep3NetworkWorker>(
        &mut self,
        polynomials: &[&Rep3MultilinearPolynomial<F>],
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claims: &[AdditiveShare<F>],
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        assert!(io_ctx.network.is_distributed());
        io_ctx.network.send_response(claims.to_vec())?;
        let (rho, first_rho_power, batched_claim): (F, F, F) = io_ctx.network.receive_request()?;

        let mut rho_powers = vec![first_rho_power];
        for i in 1..polynomials.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let batched_poly =
            Rep3MultilinearPolynomial::linear_combination(&polynomials, &rho_powers, io_ctx.id);

        self.openings.push(Rep3ProverOpening::new(
            batched_poly,
            eq_poly,
            opening_point,
            additive::promote_to_trivial_share(batched_claim, io_ctx.id),
        ));

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
            let claim = additive::promote_to_trivial_share(*claim, party_id);

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
}

/// An opening computed by the prover.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3CoordinatorOpening<F: JoltField> {
    /// Number of variables in the polynomial being opened.
    pub poly_num_vars: usize,
    /// The claimed opening.
    pub claim: F,
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
pub struct Rep3OpeningAccumulatorCoordinator<F: JoltField> {
    openings: Vec<Rep3CoordinatorOpening<F>>,
}

impl<F: JoltField> Rep3OpeningAccumulatorCoordinator<F> {
    pub fn new() -> Self {
        Self { openings: vec![] }
    }

    pub fn len(&self) -> usize {
        self.openings.len()
    }

    pub fn append_opening(&mut self, opening: Rep3CoordinatorOpening<F>) {
        self.openings.push(opening);
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append")]
    pub fn append<ProofTranscript: Transcript, Network: Rep3NetworkCoordinator>(
        &mut self,
        poly_num_vars: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Vec<F>> {
        let claims = if network.is_distributed() {
            network
                .receive_responses_from_subnets()?
                .into_iter()
                .map(additive::combine_additive_vec)
                .reduce(|prev, next| izip!(prev, next).map(|(a, b)| a + b).collect())
                .unwrap()
        } else {
            additive::combine_additive_vec(network.receive_responses()?)
        };
        self.append_with_claims(poly_num_vars, &claims, transcript, network)?;
        Ok(claims)
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_partial")]
    pub fn append_partial<ProofTranscript: Transcript, Network: Rep3NetworkCoordinator>(
        &mut self,
        poly_num_vars: usize,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Vec<F>> {
        let claims = if network.is_distributed() {
            let polys = network
                .receive_responses_from_subnets()?
                .into_iter()
                .map(|shares| {
                    additive::combine_additive_vec::<F>(shares)
                        .into_iter()
                        .map(|e| vec![e])
                        .collect_vec()
                })
                .reduce(|mut acc, next| {
                    izip!(&mut acc, next).for_each(|(coeffs, c)| coeffs.push(c[0]));
                    acc
                })
                .unwrap()
                .into_iter()
                .map(MultilinearPolynomial::from)
                .collect_vec();
            let (_, r_merge) =
                opening_point.split_at(opening_point.len() - network.log_num_workers());
            let (claims, _) =
                MultilinearPolynomial::batch_evaluate(&polys.iter().collect_vec(), r_merge);
            claims
        } else {
            additive::combine_additive_vec(network.receive_responses()?)
        };
        self.append_with_claims(poly_num_vars, &claims, transcript, network)?;
        Ok(claims)
    }

    pub fn append_with_claims<ProofTranscript: Transcript, Network: Rep3NetworkCoordinator>(
        &mut self,
        poly_num_vars: usize,
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
        let claim: F = rho_powers
            .iter()
            .zip(claims.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();
        tracing::info!("append claim: {:?}", claim);

        network.broadcast_request((rho, claim))?;

        self.openings.push(Rep3CoordinatorOpening {
            poly_num_vars,
            claim,
        });
        Ok(())
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_batched")]
    pub fn append_batched<ProofTranscript: Transcript, Network: Rep3NetworkCoordinator>(
        &mut self,
        poly_num_vars: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Vec<F>> {
        assert!(network.is_distributed());

        let (batch_lens, claims): (Vec<_>, Vec<_>) = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
            .into_iter()
            .map(|shares| (shares[0].len(), additive::combine_additive_vec(shares)))
            .unzip();
        let claims = claims.into_iter().flatten().collect_vec();

        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..claims.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        // Compute the random linear combination of the claims
        let claim: F = rho_powers
            .iter()
            .zip(claims.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();

        // tracing::info!("IF combined_claim: {:?}", claim);

        let mut worker_offset_rho_power = vec![F::one()];
        let mut offset = batch_lens[0];
        for len in &batch_lens[1..] {
            worker_offset_rho_power.push(rho_powers[offset]);
            offset += len;
        }
        network.send_requests_to_workers(
            worker_offset_rho_power
                .into_iter()
                .map(|offset_rho_pow| (rho, offset_rho_pow, claim))
                .collect(),
        )?;

        self.openings.push(Rep3CoordinatorOpening {
            poly_num_vars,
            claim,
        });
        Ok(claims)
    }

    /// Reduces the multiple openings accumulated into a single opening proof,
    /// using a single sumcheck.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove<PCS, ProofTranscript, Network>(
        &self,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<ReducedOpeningProof<F, PCS, ProofTranscript>>
    where
        Network: Rep3NetworkCoordinator,
        ProofTranscript: Transcript,
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    {
        let rho: F = transcript.challenge_scalar();
        tracing::info!("rho: {:?}", rho);
        network.broadcast_request(rho)?;
        let _span = tracing::trace_span!("rho_powers").entered();
        let mut rho_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }
        drop(_span);

        let max_num_vars = self
            .openings
            .iter()
            .map(|opening| opening.poly_num_vars)
            .max()
            .unwrap();

        tracing::info!("max_num_vars: {:?}", max_num_vars);

        let mut combined_claim: F = rho_powers
            .par_iter()
            .zip(self.openings.par_iter())
            .map(|(coeff, opening)| {
                let scaled_claim = if opening.poly_num_vars != max_num_vars {
                    F::from_u64_unchecked(1 << (max_num_vars - opening.poly_num_vars))
                        * opening.claim
                } else {
                    opening.claim
                };

                scaled_claim * coeff
            })
            .sum();
        tracing::info!("combined_claim: {:?}", combined_claim);
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

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
            tracing::info!("round_evals: {:?}", round_evals);

            round_evals.insert(1, combined_claim - round_evals[0]);
            let uni_poly = UniPoly::from_evals(&round_evals);
            let compressed_poly = uni_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            tracing::info!("opening r_j: {:?}", r_j);
            r.push(r_j);
            combined_claim = uni_poly.evaluate(&r_j);

            network.broadcast_request(r_j)?;

            compressed_polys.push(compressed_poly);
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
        let joint_opening_proof = PCS::coordinate_prove(network)?;

        Ok(ReducedOpeningProof {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        })
    }
}

impl<F: JoltField> Rep3OpeningAccumulatorWorker<F> {
    /// Reduces the multiple openings accumulated into a single opening proof,
    /// using a single sumcheck.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove<PCS, ProofTranscript, Network>(
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
        tracing::info!("rho: {:?}", rho);
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
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let joint_poly = Rep3MultilinearPolynomial::linear_combination(
            &unbound_polys.iter().collect::<Vec<_>>(),
            &gamma_powers,
            io_ctx.party_id(),
        );

        // Reduced opening proof
        if let Rep3MultilinearPolynomial::Shared(joint_poly) = joint_poly {
            PCS::prove_rep3(&joint_poly, pcs_setup, &r_sumcheck, io_ctx.network())?;
        } else {
            panic!("Joint polynomial is expected to be shared")
        }

        Ok(())
    }

    /// Proves the sumcheck used to prove the reduction of many openings into one.
    #[tracing::instrument(skip_all, name = "prove_batch_opening_reduction")]
    fn prove_batch_opening_reduction<Network: Rep3NetworkWorker>(
        &mut self,
        coeffs: &[F],
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<(Vec<F>, Vec<AdditiveShare<F>>)> {
        let max_num_vars = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.get_num_vars())
            .max()
            .unwrap();

        tracing::info!("max_num_vars: {:?}", max_num_vars);

        let mut r: Vec<F> = Vec::new();

        for round in 0..max_num_vars {
            let remaining_rounds = max_num_vars - round;
            let evals = self.compute_quadratic(
                coeffs,
                remaining_rounds,
                io_ctx.party_id(),
                io_ctx.worker_idx(),
            );
            io_ctx.network().send_response(evals.to_vec())?;

            let r_j = io_ctx.network().receive_request()?;
            r.push(r_j);

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
        worker_idx: usize,
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
                    if worker_idx != 0 {
                        // must preserve linearity in coordinator partial summation, so only one worker contributes scaled claims
                        return (AdditiveShare::zero(), AdditiveShare::zero());
                    }
                    let remaining_variables =
                        remaining_sumcheck_rounds - opening.opening_point.len() - 1;
                    let scaled_claim =
                        opening.claim * F::from_u64_unchecked(1 << remaining_variables);
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
