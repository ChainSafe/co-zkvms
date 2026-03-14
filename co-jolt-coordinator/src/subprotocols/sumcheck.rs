#![allow(clippy::type_complexity)]

use eyre::Context;
use jolt_core::curve::Bn254Curve;
#[cfg(feature = "zk")]
use jolt_core::curve::Bn254G1;
#[cfg(feature = "zk")]
use jolt_core::poly::commitment::pedersen::PedersenGenerators;
#[cfg(feature = "zk")]
use jolt_core::poly::opening_proof::OpeningId;
use jolt_core::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
#[cfg(feature = "zk")]
use jolt_core::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::{AppendToTranscript, Transcript};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
#[cfg(feature = "zk")]
use rand::{CryptoRng, RngCore};

use jolt_core::field::JoltField;

use crate::poly::opening_proof::Rep3OpeningAccumulator;

pub trait Rep3SumcheckInstance<F: JoltField, T: Transcript> {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;
    fn input_claim_public(&self) -> F;

    fn expected_output_claim(&self, accumulator: &Rep3OpeningAccumulator<F>, r: &[F::Challenge]) -> F;

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F>;

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    );

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _accumulator: &Rep3OpeningAccumulator<F>) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        None
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        Vec::new()
    }
}

pub trait PublicSumcheckInstance<F: JoltField, T: Transcript> {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;
    fn input_claim_public(&self) -> F;

    fn expected_output_claim(&self, accumulator: &Rep3OpeningAccumulator<F>, r: &[F::Challenge]) -> F;

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F>;

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    );

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _accumulator: &Rep3OpeningAccumulator<F>) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        None
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        Vec::new()
    }
}

pub enum BatchedSumcheckInstance<F: JoltField, T: Transcript> {
    Secret(Box<dyn Rep3SumcheckInstance<F, T>>),
    Public(Box<dyn PublicSumcheckInstance<F, T>>),
}

impl<F: JoltField, T: Transcript> BatchedSumcheckInstance<F, T> {
    pub fn degree(&self) -> usize {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.degree(),
            BatchedSumcheckInstance::Public(s) => s.degree(),
        }
    }

    pub fn num_rounds(&self) -> usize {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.num_rounds(),
            BatchedSumcheckInstance::Public(s) => s.num_rounds(),
        }
    }

    pub fn input_claim_public(&self) -> F {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.input_claim_public(),
            BatchedSumcheckInstance::Public(s) => s.input_claim_public(),
        }
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.normalize_opening_point(opening_point),
            BatchedSumcheckInstance::Public(s) => s.normalize_opening_point(opening_point),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.cache_openings(accumulator, transcript, opening_point, claims),
            BatchedSumcheckInstance::Public(s) => s.cache_openings(accumulator, transcript, opening_point, claims),
        }
    }

    #[cfg(feature = "zk")]
    pub fn input_claim_constraint(&self) -> InputClaimConstraint {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.input_claim_constraint(),
            BatchedSumcheckInstance::Public(s) => s.input_claim_constraint(),
        }
    }

    #[cfg(feature = "zk")]
    pub fn input_constraint_challenge_values(&self, accumulator: &Rep3OpeningAccumulator<F>) -> Vec<F> {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.input_constraint_challenge_values(accumulator),
            BatchedSumcheckInstance::Public(s) => s.input_constraint_challenge_values(accumulator),
        }
    }

    #[cfg(feature = "zk")]
    pub fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.output_claim_constraint(),
            BatchedSumcheckInstance::Public(s) => s.output_claim_constraint(),
        }
    }

    #[cfg(feature = "zk")]
    pub fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        match self {
            BatchedSumcheckInstance::Secret(s) => s.output_constraint_challenge_values(sumcheck_challenges),
            BatchedSumcheckInstance::Public(s) => s.output_constraint_challenge_values(sumcheck_challenges),
        }
    }
}

pub struct Rep3BatchedSumcheck;

impl Rep3BatchedSumcheck {
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::prove")]
    pub fn prove<F, ProofTranscript, N>(
        instances: &[Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>],
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        network: &mut N,
    ) -> eyre::Result<(SumcheckInstanceProof<F, Bn254Curve, ProofTranscript>, Vec<F::Challenge>)>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        N: Rep3NetworkCoordinator,
    {
        eyre::ensure!(!instances.is_empty(), "Batched sumcheck requires >= 1 instance");

        let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
        let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(instances.len());
        network.broadcast_request(batching_coeffs.clone())?;

        let individual_claims: Vec<F> = instances
            .iter()
            .map(|instance| {
                let input_claim = instance.input_claim_public();
                transcript.append_scalar(&input_claim);
                input_claim.mul_pow_2(max_num_rounds - instance.num_rounds())
            })
            .collect();

        let mut batched_claim: F =
            individual_claims.iter().zip(batching_coeffs.iter()).map(|(claim, coeff)| *claim * coeff).sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for _round in 0..max_num_rounds {
            let round_evals = receive_batched_round_evals::<F, N>(network)?;
            eyre::ensure!(
                round_evals.len() == max_degree,
                "round evals len mismatch: expected {max_degree}, got {}",
                round_evals.len()
            );

            let mut full_evals = Vec::with_capacity(max_degree + 1);
            full_evals.push(round_evals[0]);
            full_evals.push(batched_claim - round_evals[0]);
            full_evals.extend(round_evals.into_iter().skip(1));

            let mut round_poly = UniPoly::<F>::from_evals(&full_evals);
            while round_poly.coeffs.len() > 1 && round_poly.coeffs.last() == Some(&F::zero()) {
                round_poly.coeffs.pop();
            }
            let compressed_poly = round_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);
            batched_claim = round_poly.evaluate(&r_j);
            network.broadcast_request(r_j)?;
        }

        let opening_claims = receive_opening_claims::<F, N>(network)?;
        eyre::ensure!(
            opening_claims.len() == instances.len(),
            "opening claims instance count mismatch: expected {}, got {}",
            instances.len(),
            opening_claims.len()
        );

        for (i, instance) in instances.iter().enumerate() {
            let num_rounds = instance.num_rounds();
            let r_slice = &r_sumcheck[max_num_rounds - num_rounds..];
            let opening_point = instance.normalize_opening_point(r_slice);
            instance.cache_openings(accumulator, transcript, opening_point, opening_claims[i].clone());
        }

        Ok((SumcheckInstanceProof::<F, Bn254Curve, ProofTranscript>::new(compressed_polys), r_sumcheck))
    }
}

pub struct HybridBatchedSumcheck;

type HybridRoundMsg<F> = (Vec<AdditiveShare<F>>, Option<Vec<F>>);
type HybridOpeningsMsg<F> = Vec<(Vec<AdditiveShare<F>>, Option<Vec<F>>)>;

#[cfg(feature = "zk")]
pub struct HybridZkProofMaterial<F: JoltField> {
    pub initial_claim: F,
    pub batching_coefficients: Vec<F>,
    pub challenges: Vec<F::Challenge>,
    pub round_commitments: Vec<Bn254G1>,
    pub poly_coeffs: Vec<Vec<F>>,
    pub blinding_factors: Vec<F>,
    pub output_claims: Vec<(OpeningId, F)>,
    pub output_claims_blindings: Vec<F>,
    pub output_claims_commitments: Vec<Bn254G1>,
}

impl HybridBatchedSumcheck {
    #[tracing::instrument(skip_all, name = "HybridSumcheck::prove")]
    pub fn prove<F, ProofTranscript, N>(
        instances: &[BatchedSumcheckInstance<F, ProofTranscript>],
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        network: &mut N,
    ) -> eyre::Result<(SumcheckInstanceProof<F, Bn254Curve, ProofTranscript>, Vec<F::Challenge>)>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        N: Rep3NetworkCoordinator,
    {
        eyre::ensure!(!instances.is_empty(), "Batched sumcheck requires >= 1 instance");

        let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
        let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(instances.len());
        network.broadcast_request(batching_coeffs.clone())?;

        let individual_claims: Vec<F> = instances
            .iter()
            .map(|instance| {
                let input_claim = instance.input_claim_public();
                transcript.append_scalar(&input_claim);
                input_claim.mul_pow_2(max_num_rounds - instance.num_rounds())
            })
            .collect();

        let mut batched_claim: F =
            individual_claims.iter().zip(batching_coeffs.iter()).map(|(claim, coeff)| *claim * coeff).sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for _round in 0..max_num_rounds {
            let round_evals = receive_hybrid_round_evals::<F, N>(network)?;
            eyre::ensure!(
                round_evals.len() == max_degree,
                "round evals len mismatch: expected {max_degree}, got {}",
                round_evals.len()
            );

            let mut full_evals = Vec::with_capacity(max_degree + 1);
            full_evals.push(round_evals[0]);
            full_evals.push(batched_claim - round_evals[0]);
            full_evals.extend(round_evals.into_iter().skip(1));

            let mut round_poly = UniPoly::<F>::from_evals(&full_evals);
            while round_poly.coeffs.len() > 1 && round_poly.coeffs.last() == Some(&F::zero()) {
                round_poly.coeffs.pop();
            }
            let compressed_poly = round_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);
            batched_claim = round_poly.evaluate(&r_j);
            network.broadcast_request(r_j)?;
        }

        let opening_claims = receive_hybrid_opening_claims::<F, N>(network)?;
        eyre::ensure!(
            opening_claims.len() == instances.len(),
            "opening claims instance count mismatch: expected {}, got {}",
            instances.len(),
            opening_claims.len()
        );

        for (i, instance) in instances.iter().enumerate() {
            let num_rounds = instance.num_rounds();
            let r_slice = &r_sumcheck[max_num_rounds - num_rounds..];
            let opening_point = instance.normalize_opening_point(r_slice);
            instance.cache_openings(accumulator, transcript, opening_point, opening_claims[i].clone());
        }

        Ok((SumcheckInstanceProof::<F, Bn254Curve, ProofTranscript>::new(compressed_polys), r_sumcheck))
    }

    #[cfg(feature = "zk")]
    #[tracing::instrument(skip_all, name = "HybridSumcheck::prove_zk")]
    pub fn prove_zk<F, ProofTranscript, N, R>(
        instances: &[BatchedSumcheckInstance<F, ProofTranscript>],
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        network: &mut N,
        pedersen_gens: &PedersenGenerators<Bn254Curve>,
        rng: &mut R,
    ) -> eyre::Result<(
        SumcheckInstanceProof<F, Bn254Curve, ProofTranscript>,
        Vec<F::Challenge>,
        HybridZkProofMaterial<F>,
    )>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        N: Rep3NetworkCoordinator,
        R: CryptoRng + RngCore,
    {
        eyre::ensure!(!instances.is_empty(), "Batched sumcheck requires >= 1 instance");

        let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
        let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(instances.len());
        network.broadcast_request(batching_coeffs.clone())?;

        let individual_claims: Vec<F> = instances
            .iter()
            .map(|instance| instance.input_claim_public().mul_pow_2(max_num_rounds - instance.num_rounds()))
            .collect();
        let initial_batched_claim: F =
            individual_claims.iter().zip(batching_coeffs.iter()).map(|(claim, coeff)| *claim * coeff).sum();

        let mut batched_claim = initial_batched_claim;
        let mut r_sumcheck = Vec::with_capacity(max_num_rounds);
        let mut round_commitments = Vec::with_capacity(max_num_rounds);
        let mut poly_coeffs = Vec::with_capacity(max_num_rounds);
        let mut blinding_factors = Vec::with_capacity(max_num_rounds);
        let mut poly_degrees = Vec::with_capacity(max_num_rounds);

        for _round in 0..max_num_rounds {
            let round_evals = receive_hybrid_round_evals::<F, N>(network)?;
            eyre::ensure!(
                round_evals.len() == max_degree,
                "round evals len mismatch: expected {max_degree}, got {}",
                round_evals.len()
            );

            let mut full_evals = Vec::with_capacity(max_degree + 1);
            full_evals.push(round_evals[0]);
            full_evals.push(batched_claim - round_evals[0]);
            full_evals.extend(round_evals.into_iter().skip(1));

            let mut round_poly = UniPoly::<F>::from_evals(&full_evals);
            while round_poly.coeffs.len() > 1 && round_poly.coeffs.last() == Some(&F::zero()) {
                round_poly.coeffs.pop();
            }

            let blinding = F::random(rng);
            let commitment = pedersen_gens.commit(&round_poly.coeffs, &blinding);
            transcript.append_message(b"sumcheck_commitment");
            transcript.append_serializable(&commitment);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);
            batched_claim = round_poly.evaluate(&r_j);
            network.broadcast_request(r_j)?;

            poly_degrees.push(round_poly.coeffs.len() - 1);
            round_commitments.push(commitment);
            poly_coeffs.push(round_poly.coeffs);
            blinding_factors.push(blinding);
        }

        let opening_claims = receive_hybrid_opening_claims::<F, N>(network)?;
        eyre::ensure!(
            opening_claims.len() == instances.len(),
            "opening claims instance count mismatch: expected {}, got {}",
            instances.len(),
            opening_claims.len()
        );

        accumulator.set_zk_mode(true);
        for (i, instance) in instances.iter().enumerate() {
            let num_rounds = instance.num_rounds();
            let r_slice = &r_sumcheck[max_num_rounds - num_rounds..];
            let opening_point = instance.normalize_opening_point(r_slice);
            instance.cache_openings(accumulator, transcript, opening_point, opening_claims[i].clone());
        }
        let output_claim_values = accumulator.take_pending_claims();
        let output_claim_ids = accumulator.take_pending_claim_ids();
        accumulator.set_zk_mode(false);

        let output_claims: Vec<(OpeningId, F)> = output_claim_ids.into_iter().zip(output_claim_values).collect();
        let committed_output_claims =
            pedersen_gens.commit_chunked(&output_claims.iter().map(|(_, value)| *value).collect::<Vec<_>>(), rng);
        let (output_claims_commitments, output_claims_blindings): (Vec<_>, Vec<_>) =
            committed_output_claims.into_iter().unzip();
        transcript.append_message(b"output_claims_coms");
        output_claims_commitments.iter().for_each(|commitment| transcript.append_serializable(commitment));

        Ok((
            SumcheckInstanceProof::<F, Bn254Curve, ProofTranscript>::new_zk(
                round_commitments.clone(),
                poly_degrees,
                output_claims_commitments.clone(),
            ),
            r_sumcheck.clone(),
            HybridZkProofMaterial {
                initial_claim: initial_batched_claim,
                batching_coefficients: batching_coeffs,
                challenges: r_sumcheck,
                round_commitments,
                poly_coeffs,
                blinding_factors,
                output_claims,
                output_claims_blindings,
                output_claims_commitments,
            },
        ))
    }
}

fn receive_hybrid_round_evals<F, N>(network: &mut N) -> eyre::Result<Vec<F>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    if network.is_distributed() {
        let subnet_responses = network
            .receive_responses_from_subnets::<HybridRoundMsg<F>>()
            .context("receive round evals from subnets")?;

        let max_degree = subnet_responses.first().and_then(|s| s.first()).map(|msg| msg.0.len()).unwrap_or(0);

        let mut total = vec![F::zero(); max_degree];
        for shares_by_party in subnet_responses {
            eyre::ensure!(shares_by_party.len() == 3, "expected 3 parties per subnet");
            let secret_parts: Vec<Vec<AdditiveShare<F>>> = shares_by_party.iter().map(|m| m.0.clone()).collect();
            let secret_open = additive::combine_additive_vec(secret_parts);
            total.iter_mut().zip(secret_open.into_iter()).for_each(|(dst, src)| *dst += src);

            if let Some(ref public) = shares_by_party[0].1 {
                eyre::ensure!(public.len() == max_degree, "public evals len mismatch");
                total.iter_mut().zip(public.iter()).for_each(|(dst, src)| *dst += *src);
            }
        }
        Ok(total)
    } else {
        let responses = network.receive_responses::<HybridRoundMsg<F>>().context("receive round evals")?;
        eyre::ensure!(responses.len() == 3, "expected 3 parties");

        let secret_parts: Vec<Vec<AdditiveShare<F>>> = responses.iter().map(|m| m.0.clone()).collect();
        let mut total = additive::combine_additive_vec(secret_parts);

        if let Some(ref public) = responses[0].1 {
            eyre::ensure!(public.len() == total.len(), "public evals len mismatch");
            total.iter_mut().zip(public.iter()).for_each(|(dst, src)| *dst += *src);
        }
        Ok(total)
    }
}

fn receive_hybrid_opening_claims<F, N>(network: &mut N) -> eyre::Result<Vec<Vec<F>>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    if network.is_distributed() {
        let subnet_responses = network
            .receive_responses_from_subnets::<HybridOpeningsMsg<F>>()
            .context("receive opening claims from subnets")?;

        let mut acc: Option<Vec<Vec<F>>> = None;
        for shares_by_party in subnet_responses {
            eyre::ensure!(shares_by_party.len() == 3, "expected 3 parties per subnet");
            let m = shares_by_party[0].len();
            let mut opened: Vec<Vec<F>> = Vec::with_capacity(m);
            for i in 0..m {
                let secret_lists: Vec<Vec<AdditiveShare<F>>> =
                    shares_by_party.iter().map(|party| party[i].0.clone()).collect();
                let mut claims = if secret_lists.iter().all(|v| v.is_empty()) {
                    vec![]
                } else {
                    additive::combine_additive_vec(secret_lists)
                };
                if let Some(ref public_claims) = shares_by_party[0][i].1 {
                    if claims.is_empty() {
                        claims = public_claims.clone();
                    } else {
                        eyre::ensure!(claims.len() == public_claims.len(), "opening claims len mismatch");
                        for (c, p) in claims.iter_mut().zip(public_claims.iter()) {
                            *c += *p;
                        }
                    }
                }
                opened.push(claims);
            }

            match &mut acc {
                None => acc = Some(opened),
                Some(total) => {
                    eyre::ensure!(total.len() == opened.len(), "opening claims instance count mismatch across subnets");
                    for (t, o) in total.iter_mut().zip(opened.into_iter()) {
                        if t.is_empty() {
                            *t = o;
                            continue;
                        }
                        eyre::ensure!(t.len() == o.len(), "opening claim len mismatch across subnets");
                        for (tv, ov) in t.iter_mut().zip(o.into_iter()) {
                            *tv += ov;
                        }
                    }
                }
            }
        }
        Ok(acc.unwrap_or_default())
    } else {
        let responses = network.receive_responses::<HybridOpeningsMsg<F>>().context("receive opening claims")?;
        eyre::ensure!(responses.len() == 3, "expected 3 parties");
        let m = responses[0].len();
        let mut opened: Vec<Vec<F>> = Vec::with_capacity(m);
        for i in 0..m {
            let secret_lists: Vec<Vec<AdditiveShare<F>>> = responses.iter().map(|party| party[i].0.clone()).collect();
            let mut claims = if secret_lists.iter().all(|v| v.is_empty()) {
                vec![]
            } else {
                additive::combine_additive_vec(secret_lists)
            };
            if let Some(ref public_claims) = responses[0][i].1 {
                if claims.is_empty() {
                    claims = public_claims.clone();
                } else {
                    eyre::ensure!(claims.len() == public_claims.len(), "opening claim len mismatch");
                    for (c, p) in claims.iter_mut().zip(public_claims.iter()) {
                        *c += *p;
                    }
                }
            }
            opened.push(claims);
        }
        Ok(opened)
    }
}

fn receive_batched_round_evals<F, N>(network: &mut N) -> eyre::Result<Vec<F>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    if network.is_distributed() {
        let subnet_responses = network
            .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()
            .context("receive round eval shares from subnets")?;
        let degree = subnet_responses.first().and_then(|s| s.first()).map(|v| v.len()).unwrap_or(0);

        Ok(subnet_responses.into_iter().map(additive::combine_additive_vec).fold(
            vec![F::zero(); degree],
            |mut acc, evals| {
                acc.iter_mut().zip(evals.into_iter()).for_each(|(dst, src)| *dst += src);
                acc
            },
        ))
    } else {
        let shares = network.receive_responses::<Vec<AdditiveShare<F>>>().context("receive round eval shares")?;
        Ok(additive::combine_additive_vec(shares))
    }
}

fn receive_opening_claims<F, N>(network: &mut N) -> eyre::Result<Vec<Vec<F>>>
where
    F: JoltField,
    N: Rep3NetworkCoordinator,
{
    if network.is_distributed() {
        let subnet_responses = network
            .receive_responses_from_subnets::<Vec<Vec<AdditiveShare<F>>>>()
            .context("receive opening claim shares from subnets")?;

        let mut acc: Option<Vec<Vec<F>>> = None;
        for shares_by_party in subnet_responses {
            let m = shares_by_party.first().map(|v| v.len()).unwrap_or_default();
            let opened: Vec<Vec<F>> = (0..m)
                .map(|i| {
                    let shares_for_instance: Vec<Vec<AdditiveShare<F>>> =
                        shares_by_party.iter().map(|party| party[i].clone()).collect();
                    additive::combine_additive_vec(shares_for_instance)
                })
                .collect();

            match &mut acc {
                None => acc = Some(opened),
                Some(total) => {
                    eyre::ensure!(total.len() == opened.len(), "opening claims instance count mismatch across subnets");
                    for (t, o) in total.iter_mut().zip(opened.into_iter()) {
                        eyre::ensure!(t.len() == o.len(), "opening claim len mismatch across subnets");
                        for (tv, ov) in t.iter_mut().zip(o.into_iter()) {
                            *tv += ov;
                        }
                    }
                }
            }
        }
        Ok(acc.unwrap_or_default())
    } else {
        let responses =
            network.receive_responses::<Vec<Vec<AdditiveShare<F>>>>().context("receive opening claim shares")?;
        let m = responses.first().map(|v| v.len()).unwrap_or_default();
        Ok((0..m)
            .map(|i| {
                let shares_for_instance: Vec<Vec<AdditiveShare<F>>> =
                    responses.iter().map(|party| party[i].clone()).collect();
                additive::combine_additive_vec(shares_for_instance)
            })
            .collect())
    }
}
