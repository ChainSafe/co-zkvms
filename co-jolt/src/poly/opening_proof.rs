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
            // .sum::<F>();
            .collect::<Vec<F>>()[0]; // TODO: move openings batched_claims into coordinator flow

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

        tracing::info!("rho_powers: {:?}", rho_powers);

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
            // .par_iter()
            .iter()
            // .zip(self.openings.par_iter_mut())
            .zip(self.openings.iter_mut())
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
                // let claim_open = rep3::arithmetic::open(scaled_claim, io_ctx.main()).unwrap();
                // tracing::info!(
                //     "worker {} opening claim {}",
                //     io_ctx.worker_idx(),
                //     claim_open
                // );
                scaled_claim.into_additive() * *coeff
            })
            .sum();

        io_ctx.network().send_response(e)?;
        // e = io_ctx.network.receive_request()?;

        let mut r: Vec<F> = Vec::new();

        for round in 0..max_num_vars {
            let remaining_rounds = max_num_vars - round;
            let evals =
                self.compute_quadratic(coeffs, remaining_rounds, io_ctx.party_id(), io_ctx.main());
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
    fn compute_quadratic<Network: Rep3NetworkWorker>(
        &self,
        coeffs: &[F],
        remaining_sumcheck_rounds: usize,
        party_id: PartyID,
        io_ctx: &mut IoContext<Network>,
    ) -> [AdditiveShare<F>; 2] {
        let evals: Vec<(AdditiveShare<F>, AdditiveShare<F>)> = self
            .openings
            // .par_iter()
            .iter()
            .map(|opening| {
                if remaining_sumcheck_rounds <= opening.opening_point.len() {
                    let mle_half = opening.polynomial.len() / 2;
                    let eval_0 = (0..mle_half)
                        .map(|i| {
                            let e = opening
                                .polynomial
                                .get_bound_coeff(i)
                                .mul_public(opening.eq_poly.get_bound_coeff(i))
                                .into_additive(party_id);
                            let e_open = additive::open(e.into_fe(), io_ctx).unwrap();
                            tracing::info!("mle_half_i {} eval_0: {}", i, e_open);
                            e
                        })
                        .sum();
                    let eval_2 = (0..mle_half)
                        .map(|i| {
                            let poly_bound_point = opening
                                .polynomial
                                .get_bound_coeff(i + mle_half)
                                .add(&opening.polynomial.get_bound_coeff(i + mle_half), party_id)
                                .sub(&opening.polynomial.get_bound_coeff(i), party_id);
                            let open = additive::open(
                                poly_bound_point.into_additive(party_id).into_fe(),
                                io_ctx,
                            )
                            .unwrap();
                            let eq_bound_point = opening.eq_poly.get_bound_coeff(i + mle_half)
                                + opening.eq_poly.get_bound_coeff(i + mle_half)
                                - opening.eq_poly.get_bound_coeff(i);
                            tracing::info!(
                                "mle_half_i {} poly_bound_point: {} eq_bound_point: {}",
                                i,
                                open,
                                eq_bound_point
                            );

                            poly_bound_point
                                .mul_public(eq_bound_point)
                                .into_additive(party_id)
                        })
                        .sum();
                    (eval_0, eval_2)
                } else {
                    tracing::warn!("Scaling claim!");
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

#[cfg(test)]
mod simulation {
    use std::env;

    use ark_ec::pairing::Pairing;
    use ark_ec::CurveGroup;
    use ark_ff::AdditiveGroup;
    use ark_poly_commit::multilinear_pc::data_structures::Proof;
    use ark_std::test_rng;
    use ark_std::One;
    use itertools::Itertools;
    use jolt_core::{
        msm::Icicle,
        poly::{
            commitment::commitment_scheme::CommitmentScheme, eq_poly::EqPolynomial,
            multilinear_polynomial::PolynomialEvaluation,
        },
        utils::transcript::KeccakTranscript,
    };
    use snarks_core::math::Math;

    use crate::poly::commitment::{
        mock::MockCommitScheme,
        pst13::{PST13Setup, PST13},
    };

    use super::*;

    #[test]
    fn test_simulate_local_open_reduce() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();

        type F = ark_bn254::Fr;
        let NUM_POLYS = 1usize;
        let P: usize = env::var("P")
            .unwrap_or_else(|_| "8".to_string())
            .parse()
            .unwrap();
        // let W: usize = env::var("NUM_WORKERS")
        //     .unwrap_or_else(|_| "2".to_string())
        //     .parse()
        //     .unwrap();

        let polys = (0..NUM_POLYS)
            .map(|_| MultilinearPolynomial::from((0..P).map(|e| F::from(e as u64)).collect_vec()))
            .collect_vec();
        println!(
            "polys: {:?}",
            polys
                .iter()
                .map(|p| p.coeffs_as_field_elements())
                .collect_vec()
        );
        let polys_ref = polys.iter().collect_vec();

        println!("\n/------------ Prover ------------/");

        let comms = MockCommitScheme::<F, KeccakTranscript>::batch_commit(&polys_ref, &());

        let mut transcript = KeccakTranscript::new(&[]);
        let mut opening_accum = ProverOpeningAccumulator::<F, KeccakTranscript>::new();

        let r_point = transcript.challenge_vector(P.log_2());
        println!("r_point: {:?}", r_point);

        // {
        //     let r_point = r_point.iter().rev().copied().collect_vec();
        //     let mut p = DensePolynomial::new(polys[0].clone().coeffs_as_field_elements());
        //     r_point[..r_point.len() - 1]
        //         .iter()
        //         .for_each(|r| p.bind(*r, BindingOrder::LowToHigh));
        //     println!("rem_evals: {:?}", &p.Z[..p.len()]);
        //     println!(
        //         "eval full: {:?}",
        //         &DensePolynomial::new(p.Z[..p.len()].to_vec())
        //             .evaluate(&r_point[r_point.len() - 1..])
        //     );
        // }
        let (claims, eq_r) = MultilinearPolynomial::batch_evaluate(&polys_ref, &r_point);
        println!("claims: {:?}", claims);
        opening_accum.append(
            &polys_ref,
            DensePolynomial::new(eq_r),
            r_point.clone(),
            &claims,
            &mut transcript,
        );

        let proof = opening_accum
            .reduce_and_prove::<MockCommitScheme<F, KeccakTranscript>>(&(), &mut transcript);

        println!("\n/------------ Verifier ------------/");

        let mut transcript = KeccakTranscript::new(&[]);
        let mut opening_accum = VerifierOpeningAccumulator::<
            F,
            MockCommitScheme<F, KeccakTranscript>,
            KeccakTranscript,
        >::new();

        let r_point = transcript.challenge_vector(P.log_2());

        opening_accum.append(
            &comms.iter().collect_vec(),
            r_point,
            &claims.iter().collect_vec(),
            &mut transcript,
        );

        opening_accum
            .reduce_and_verify(&(), &proof, &mut transcript)
            .unwrap();
    }

    #[test]
    fn test_simulate_distributed_open_reduce() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();

        type E = ark_bn254::Bn254;
        type F = ark_bn254::Fr;
        let NUM_POLYS = 1usize;
        let P: usize = env::var("P")
            .unwrap_or_else(|_| "8".to_string())
            .parse()
            .unwrap();
        let P_nv = P.log_2();
        let W: usize = env::var("NUM_WORKERS")
            .unwrap_or_else(|_| "2".to_string())
            .parse()
            .unwrap();
        let W_log2 = W.log_2();
        let P_worker = P / W;

        let worker_polys = (0..W)
            .map(|w| {
                (0..NUM_POLYS)
                    .map(|_| {
                        MultilinearPolynomial::from(
                            (w * P_worker..P_worker * (w + 1))
                                .map(|e| F::from(e as u64))
                                .collect_vec(),
                        )
                    })
                    .collect_vec()
            })
            .collect_vec();

        println!(
            "worker_polys: {:?}",
            worker_polys
                .iter()
                .map(|wp| wp
                    .iter()
                    .map(|p| p.coeffs_as_field_elements())
                    .collect_vec())
                .collect_vec()
        );

        let worker_polys_ref = worker_polys
            .iter()
            .map(|w| w.iter().collect_vec())
            .collect_vec();

        println!("\n/------------ Prover ------------/");

        let full_polys = (0..NUM_POLYS)
            .map(|i| {
                MultilinearPolynomial::from(
                    (0..W)
                        .flat_map(|w| worker_polys_ref[w][i].coeffs_as_field_elements())
                        .collect_vec(),
                )
            })
            .collect_vec();

        let mut rng = test_rng();
        let setup = PST13::setup(P, &mut rng);
        let comms =
            <PST13<E> as CommitmentScheme<KeccakTranscript>>::batch_commit(&full_polys, &setup);

        let mut transcript = KeccakTranscript::new(&[]);
        let mut accums = (0..W)
            .map(|_| ProverOpeningAccumulator::<F, KeccakTranscript>::new())
            .collect_vec();

        let r_point = transcript.challenge_vector(P_nv);

        let rem_evals = (0..W)
            .map(|w| {
                let (claims, _) =
                    MultilinearPolynomial::batch_evaluate(&worker_polys_ref[w], &r_point[W_log2..]);

                claims
            })
            .fold(vec![vec![]; NUM_POLYS], |mut acc, next| {
                izip!(acc.iter_mut(), next).for_each(|(a, b)| {
                    a.push(b);
                });
                acc
            });

        println!("rem_evals: {:?}", rem_evals);

        let claims = rem_evals
            .iter()
            .cloned()
            .map(|evals| MultilinearPolynomial::from(evals).evaluate(&r_point[..W_log2]))
            .collect_vec();

        println!("claims: {:?}", claims);

        for w in 0..W {
            let mut tr_clone = transcript.clone();
            let transcript = if w != 0 {
                &mut tr_clone
            } else {
                &mut transcript
            };
            let chunk = 1usize << (P_nv - W_log2);
            let worker_polys_padded = worker_polys[w]
                .iter()
                .map(|p| {
                    let mut padded_evals = vec![F::ZERO; P];
                    padded_evals[w * chunk..(w + 1) * chunk]
                        .copy_from_slice(&p.coeffs_as_field_elements());
                    MultilinearPolynomial::from(padded_evals)
                })
                .collect_vec();
            accums[w].append(
                &worker_polys_padded.iter().collect_vec(),
                DensePolynomial::new(EqPolynomial::evals(&r_point)),
                r_point.clone(),
                &claims,
                transcript,
            );
        }

        let proof = simulate_distributed_reduce_and_prove(&mut accums, &setup, &mut transcript);

        println!("\n/------------ Verifier ------------/");

        let mut transcript = KeccakTranscript::new(&[]);
        let mut opening_accum = VerifierOpeningAccumulator::<F, PST13<E>, KeccakTranscript>::new();

        let r_point = transcript.challenge_vector(P.log_2());

        opening_accum.append(
            &comms.iter().collect_vec(),
            r_point,
            &claims.iter().collect_vec(),
            &mut transcript,
        );

        opening_accum
            .reduce_and_verify(&setup, &proof, &mut transcript)
            .unwrap();
    }

    pub fn simulate_distributed_reduce_and_prove<
        // F: JoltField,
        E: Pairing<ScalarField: JoltField, G1: Icicle>,
        // PCS: CommitmentScheme<KeccakTranscript, Field = F>,
    >(
        accums: &mut [ProverOpeningAccumulator<E::ScalarField, KeccakTranscript>],
        pcs_setup: &PST13Setup<E>,
        transcript: &mut KeccakTranscript,
    ) -> ReducedOpeningProof<E::ScalarField, PST13<E>, KeccakTranscript> {
        // Generate coefficients for random linear combination
        let rho: E::ScalarField = transcript.challenge_scalar();
        let mut rho_powers = vec![E::ScalarField::one()];
        for i in 1..accums[0].openings.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let coeffs = rho_powers;

        let worker_unbound_polys = accums
            .iter()
            .map(|a| {
                a.openings
                    .iter()
                    .map(|opening| opening.polynomial.clone())
                    .collect::<Vec<_>>()
            })
            .collect_vec();

        let max_num_vars = accums[0]
            .openings
            .iter()
            .map(|opening| opening.polynomial.get_num_vars())
            .max()
            .unwrap();

        // Compute random linear combination of the claims, accounting for the fact that the
        // polynomials may be of different sizes
        let mut e: E::ScalarField = coeffs
            .par_iter()
            .zip(accums[0].openings.par_iter())
            .map(|(coeff, opening)| {
                let scaled_claim = if opening.polynomial.get_num_vars() != max_num_vars {
                    E::ScalarField::from_u64_unchecked(
                        1 << (max_num_vars - opening.polynomial.get_num_vars()),
                    ) * opening.claim
                } else {
                    opening.claim
                };
                scaled_claim * coeff
            })
            .sum();

        println!("combined_claim: {e}");

        let mut r_sumcheck: Vec<_> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<_>> = Vec::new();

        // let log_num_workers = accums.len().log_2();
        let worker_rounds = max_num_vars; // - log_num_workers;

        for round in 0..worker_rounds {
            let remaining_rounds = max_num_vars - round;
            let mut evals = accums
                .iter()
                .enumerate()
                .map(|(w, w_acc)| {
                    println!("worker {w}:");
                    let evals = compute_quadratic(w_acc, &coeffs, remaining_rounds);
                    evals
                })
                .reduce(|p, n| izip!(p, n).map(|(a, b)| a + b).collect_vec())
                .unwrap();
            println!("round evals: {:?}", evals);
            println!("----------------------");
            evals.insert(1, e - evals[0]);
            let uni_poly = UniPoly::from_evals(&evals);
            let compressed_poly = uni_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            println!("r_j: {:?}", r_j);
            r_sumcheck.push(r_j);

            for acc in accums.iter_mut() {
                acc.openings.par_iter_mut().for_each(|opening| {
                    if remaining_rounds <= opening.opening_point.len() {
                        rayon::join(
                            || opening.eq_poly.bind(r_j, BindingOrder::HighToLow),
                            || opening.polynomial.bind(r_j, BindingOrder::HighToLow),
                        );
                    }
                });
            }

            e = uni_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let sumcheck_claims: Vec<_> = accums
            .iter()
            .map(|acc| {
                acc.openings
                    .iter()
                    .map(|opening| opening.polynomial.final_sumcheck_claim())
                    .collect_vec()
            })
            .reduce(|acc, item| izip!(acc, item).map(|(a, b)| a + b).collect_vec())
            .unwrap();

        transcript.append_scalars(&sumcheck_claims);

        let sumcheck_proof = SumcheckInstanceProof::new(compressed_polys);

        let gamma: E::ScalarField = transcript.challenge_scalar();
        let mut gamma_powers = vec![E::ScalarField::one()];
        for i in 1..accums[0].openings.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let joint_opening_proof = (0..accums.len())
            .map(|w| {
                let joint_poly = MultilinearPolynomial::linear_combination(
                    &worker_unbound_polys[w].iter().collect::<Vec<_>>(),
                    &gamma_powers,
                );

                // Reduced opening proof
                PST13::prove(pcs_setup, &joint_poly, &r_sumcheck, transcript)
            })
            .reduce(|acc, item| Proof {
                proofs: izip!(acc.proofs, item.proofs)
                    .map(|(p, n)| (p + n).into_affine())
                    .collect(),
            })
            .unwrap();

        // #[cfg(test)]
        // self.openings
        //     .iter_mut()
        //     .zip(unbound_polys.into_iter())
        //     .for_each(|(opening, poly)| opening.polynomial = poly);

        ReducedOpeningProof {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        }
    }

    fn compute_quadratic<F: JoltField>(
        acc: &ProverOpeningAccumulator<F, KeccakTranscript>,
        coeffs: &[F],
        remaining_rounds: usize,
    ) -> Vec<F> {
        let evals: Vec<(F, F)> = acc
            .openings
            .par_iter()
            .map(|opening| {
                println!("--------------");
                if remaining_rounds <= opening.opening_point.len() {
                    let mle_half = opening.polynomial.len() / 2;
                    let eval_0: F = (0..mle_half)
                        .map(|i| {
                            println!(
                                "eval_0: {:?}",
                                [
                                    opening.eq_poly.get_bound_coeff(i),
                                    opening.polynomial.get_bound_coeff(i),
                                ]
                            );
                            println!("-----");
                            opening.polynomial.get_bound_coeff(i)
                                * opening.eq_poly.get_bound_coeff(i)
                        })
                        .sum();
                    let eval_2: F = (0..mle_half)
                        .map(|i| {
                            let poly_bound_point = opening.polynomial.get_bound_coeff(i + mle_half)
                                + opening.polynomial.get_bound_coeff(i + mle_half)
                                - opening.polynomial.get_bound_coeff(i);
                            let eq_bound_point = opening.eq_poly.get_bound_coeff(i + mle_half)
                                + opening.eq_poly.get_bound_coeff(i + mle_half)
                                - opening.eq_poly.get_bound_coeff(i);

                            println!("eval_2: {:?}", [eq_bound_point, poly_bound_point]);
                            println!("-----");
                            poly_bound_point * eq_bound_point
                        })
                        .sum();
                    println!("----------");
                    // println!("summed eval_0 {} eval_2 {}", eval_0, eval_2);
                    (eval_0, eval_2)
                } else {
                    debug_assert!(!opening.polynomial.is_bound());
                    let remaining_variables = remaining_rounds - opening.opening_point.len() - 1;
                    let scaled_claim =
                        F::from_u64_unchecked(1 << remaining_variables) * opening.claim;
                    (scaled_claim, scaled_claim)
                }
            })
            .collect();

        let evals_combined_0: F = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
        let evals_combined_2: F = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
        vec![evals_combined_0, evals_combined_2]
    }
}
