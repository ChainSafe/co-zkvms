use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use rayon::prelude::*;

use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::{AppendToTranscript, Transcript};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::multilinear_polynomial::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomialProverOpening;
use crate::utils::types::MaybeShared;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Worker-side opening accumulator. Stores polynomial opening claims as
/// secret-shared field elements (`Rep3PrimeFieldShare`).
///
/// Mirrors vanilla `ProverOpeningAccumulator` but without transcript interaction
/// (worker has no transcript) and without `reduce_and_prove` (deferred).
pub struct Rep3OpeningAccumulatorWorker<F: JoltField> {
    pub openings: BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, F>, Rep3PrimeFieldShare<F>)>,
    pub sumchecks: Vec<Rep3OpeningProofReductionSumcheck<F>>,
    party_id: PartyID,
}

impl<F: JoltField> Rep3OpeningAccumulatorWorker<F> {
    pub fn new(party_id: PartyID) -> Self {
        Self {
            openings: BTreeMap::new(),
            sumchecks: Vec::new(),
            party_id,
        }
    }

    /// Append sparse polynomial openings (one-hot style) at `[r_address, r_cycle]`.
    /// Each polynomial gets its own sumcheck entry.
    pub fn append_sparse(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: &[F::Challenge],
        r_cycle: &[F::Challenge],
        claims: Vec<Rep3PrimeFieldShare<F>>,
    ) {
        assert_eq!(polynomials.len(), claims.len());
        let r_concat: Vec<F::Challenge> = r_address
            .iter()
            .copied()
            .chain(r_cycle.iter().copied())
            .collect();

        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let point = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings.insert(key, (point, *claim));

            self.sumchecks.push(
                Rep3OpeningProofReductionSumcheck::new_prover_instance_one_hot(
                    *label,
                    sumcheck,
                    r_address,
                    r_cycle,
                    *claim,
                    self.party_id,
                ),
            );
        }
    }

    /// Append a virtual polynomial opening (not committed, just stores the claim).
    pub fn append_virtual(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: Rep3PrimeFieldShare<F>,
    ) {
        self.openings.insert(
            OpeningId::Virtual(polynomial, sumcheck),
            (opening_point, claim),
        );
    }

    /// Append a virtual polynomial opening where the claim is PUBLIC.
    /// Promotes the public claim to a trivial rep3 share internally.
    pub fn append_virtual_public(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
        party_id: PartyID,
    ) {
        self.append_virtual(
            polynomial,
            sumcheck,
            opening_point,
            rep3_arith::promote_to_trivial_share(party_id, claim),
        );
    }

    /// Append dense (committed) polynomial openings at a shared opening point.
    pub fn append_dense(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claims: &[Rep3PrimeFieldShare<F>],
    ) {
        assert_eq!(polynomials.len(), claims.len());

        self.sumchecks.push(
            Rep3OpeningProofReductionSumcheck::new_prover_instance_dense(
                polynomials.clone(),
                sumcheck,
                opening_point.clone(),
                claims.to_vec(),
                self.party_id,
            ),
        );

        for (label, claim) in polynomials.into_iter().zip(claims.iter()) {
            let point = OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone());
            let key = OpeningId::Committed(label, sumcheck);
            self.openings.insert(key, (point, *claim));
        }
    }

    pub fn get_opening(&self, key: OpeningId) -> Rep3PrimeFieldShare<F> {
        self.openings.get(&key).unwrap().1
    }

    pub fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, Rep3PrimeFieldShare<F>) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Virtual(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        (point.clone(), *claim)
    }

    pub fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, Rep3PrimeFieldShare<F>) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Committed(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        (point.clone(), *claim)
    }

    /// Reduce all accumulated openings into a single PCS opening proof.
    ///
    /// Protocol:
    /// 1. Receive gammas from coordinator, prepare sumcheck instances
    /// 2. Receive rho powers, run batched opening reduction sumcheck
    /// 3. Send final claims, receive gamma for joint polynomial RLC
    /// 4. Build joint polynomial and combined hint, call PCS::prove_rep3
    #[tracing::instrument(skip_all, name = "Rep3OpeningAccumulatorWorker::reduce_and_prove")]
    pub fn reduce_and_prove<PCS, ProofTranscript, N>(
        &mut self,
        polynomials: &HashMap<CommittedPolynomial, Arc<Rep3MultilinearPolynomial<F>>>,
        mut opening_hints: HashMap<CommittedPolynomial, MaybeShared<PCS::OpeningProofHint>>,
        pcs_setup: &PCS::ProverSetup,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        PCS: crate::poly::commitment::Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        N: mpc_core::protocols::rep3::network::Rep3NetworkWorker,
    {
        // a. Receive gammas from coordinator
        let all_gammas: Vec<F> = network.receive_request()?;

        // b. Prepare sumchecks
        let mut gamma_offsets = vec![0usize];
        for sumcheck in self.sumchecks.iter() {
            let num_gammas = if sumcheck.polynomials.len() > 1 {
                sumcheck.polynomials.len()
            } else {
                1
            };
            gamma_offsets.push(gamma_offsets.last().unwrap() + num_gammas);
        }

        for (idx, sumcheck) in self.sumchecks.iter_mut().enumerate() {
            let offset = gamma_offsets[idx];
            let num_gammas = gamma_offsets[idx + 1] - offset;
            sumcheck.prepare_sumcheck(polynomials, &all_gammas[offset..offset + num_gammas]);
        }

        // c. Receive rho powers from coordinator
        let coeffs: Vec<F> = network.receive_request()?;

        // d. Run batched opening reduction sumcheck
        let (r_sumcheck, _claims) = self.prove_batch_opening_reduction(&coeffs, network)?;

        // e. Send sumcheck claims to coordinator
        let claim_shares: Vec<AdditiveShare<F>> = self
            .sumchecks
            .iter()
            .map(|s| s.sumcheck_claim().into_additive())
            .collect();
        network.send_response(claim_shares)?;

        // f. Receive gamma for joint poly RLC from coordinator
        let gamma: F = network.receive_request()?;
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.sumchecks.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        // g. Compute per-polynomial RLC coefficients
        let mut rlc_map: BTreeMap<CommittedPolynomial, F> = BTreeMap::new();
        for (gamma_power, sumcheck) in gamma_powers.iter().zip(self.sumchecks.iter()) {
            for (coeff, polynomial) in sumcheck.rlc_coeffs.iter().zip(sumcheck.polynomials.iter()) {
                *rlc_map.entry(*polynomial).or_insert(F::zero()) += *coeff * gamma_power;
            }
        }

        // h. Combine hints using PCS::combine_hints_rep3
        let (coeffs_for_hints, hints): (Vec<F>, Vec<MaybeShared<PCS::OpeningProofHint>>) = rlc_map
            .iter()
            .map(|(k, v)| (*v, opening_hints.remove(k).unwrap()))
            .unzip();
        let combined_hint = PCS::combine_hints_rep3(hints, &coeffs_for_hints, self.party_id);

        // i. Build Rep3RLCPolynomial
        let (rlc_coeffs, rlc_polys): (Vec<F>, Vec<Arc<Rep3MultilinearPolynomial<F>>>) = rlc_map
            .into_iter()
            .map(|(k, v)| (v, polynomials.get(&k).unwrap().clone()))
            .unzip();
        let rlc = crate::poly::rlc_polynomial::Rep3RLCPolynomial::linear_combination(
            rlc_polys,
            &rlc_coeffs,
            self.party_id,
        );
        let joint_poly = Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RLC(rlc));

        // j. Call PCS::prove_rep3 with joint polynomial and pre-combined hint
        PCS::prove_rep3(
            &joint_poly,
            pcs_setup,
            &r_sumcheck,
            Some(combined_hint),
            network,
        )?;

        Ok(())
    }

    /// Run the batched sumcheck that reduces many openings to one.
    /// Returns the sumcheck point and per-instance final claims.
    #[tracing::instrument(
        skip_all,
        name = "Rep3OpeningAccumulatorWorker::prove_batch_opening_reduction"
    )]
    fn prove_batch_opening_reduction<N>(
        &mut self,
        coeffs: &[F],
        network: &mut N,
    ) -> eyre::Result<(Vec<F::Challenge>, Vec<Rep3PrimeFieldShare<F>>)>
    where
        N: mpc_core::protocols::rep3::network::Rep3NetworkWorker,
    {
        let max_num_rounds = self
            .sumchecks
            .iter()
            .map(|s| s.num_rounds())
            .max()
            .unwrap_or(0);

        // Per-instance claims as additive shares, scaled by 2^(max - num_rounds)
        // to match vanilla's BatchedSumcheck::prove initialization.
        let mut individual_claims: Vec<AdditiveShare<F>> = self
            .sumchecks
            .iter()
            .map(|s| {
                let scale = max_num_rounds - s.num_rounds();
                let claim = s.input_claim();
                // mul_pow_2 on Rep3PrimeFieldShare: scale both components
                let scaled =
                    Rep3PrimeFieldShare::new(claim.a.mul_pow_2(scale), claim.b.mul_pow_2(scale));
                scaled.into_additive()
            })
            .collect();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let two_inv = F::TWO_INV;

        for round in 0..max_num_rounds {
            let remaining_rounds = max_num_rounds - round;

            let mut batched_e0 = AdditiveShare::<F>::zero();
            let mut batched_e2 = AdditiveShare::<F>::zero();
            let mut per_instance_e0: Vec<AdditiveShare<F>> =
                Vec::with_capacity(self.sumchecks.len());
            let mut per_instance_e2: Vec<AdditiveShare<F>> =
                Vec::with_capacity(self.sumchecks.len());

            for (i, sumcheck) in self.sumchecks.iter_mut().enumerate() {
                let num_rounds = sumcheck.num_rounds();
                if remaining_rounds > num_rounds {
                    // Inactive: constant polynomial = individual_claims[i] / 2
                    let e = individual_claims[i] * two_inv;
                    per_instance_e0.push(e);
                    per_instance_e2.push(e);
                    batched_e0 += e * coeffs[i];
                    batched_e2 += e * coeffs[i];
                } else {
                    // Active: compute from polynomial
                    let inst_round = round - (max_num_rounds - num_rounds);
                    let [e0, e2] =
                        sumcheck.compute_prover_message(inst_round, individual_claims[i]);
                    per_instance_e0.push(e0);
                    per_instance_e2.push(e2);
                    batched_e0 += e0 * coeffs[i];
                    batched_e2 += e2 * coeffs[i];
                }
            }

            // Send batched eval shares to coordinator
            network.send_response((batched_e0, batched_e2))?;

            // Receive challenge from coordinator
            let r_j: F::Challenge = network.receive_request()?;
            r_sumcheck.push(r_j);
            let r: F = r_j.into();

            // Update per-instance claims and bind active instances
            for (i, sumcheck) in self.sumchecks.iter_mut().enumerate() {
                let num_rounds = sumcheck.num_rounds();
                if remaining_rounds > num_rounds {
                    // Inactive: claim = constant (just halves)
                    individual_claims[i] = per_instance_e0[i]; // constant value
                } else {
                    let inst_round = round - (max_num_rounds - num_rounds);

                    // Interpolate degree-2 univariate at r_j to get new claim
                    let e0 = per_instance_e0[i];
                    let e2 = per_instance_e2[i];
                    let e1 = individual_claims[i] - e0;
                    // Lagrange basis at {0, 1, 2}
                    let l0 = (r - F::one()) * (r - F::from_u64(2)) * two_inv;
                    let l1 = r * (F::from_u64(2) - r);
                    let l2 = r * (r - F::one()) * two_inv;
                    individual_claims[i] = e0 * l0 + e1 * l1 + e2 * l2;

                    sumcheck.bind(r_j, inst_round);
                }
            }
        }

        // Collect final claims from each instance's bound polynomial
        for sumcheck in self.sumchecks.iter_mut() {
            sumcheck.cache_sumcheck_claim();
        }
        let claims: Vec<Rep3PrimeFieldShare<F>> =
            self.sumchecks.iter().map(|s| s.sumcheck_claim()).collect();

        Ok((r_sumcheck, claims))
    }
}

impl<F: JoltField> Default for Rep3OpeningAccumulatorWorker<F> {
    fn default() -> Self {
        Self::new(PartyID::ID0)
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

/// Coordinator-side opening accumulator. Stores polynomial opening claims as
/// public field elements and interacts with the Fiat-Shamir transcript.
///
/// Mirrors vanilla `ProverOpeningAccumulator` from the coordinator's perspective.
pub struct Rep3OpeningAccumulator<F: JoltField> {
    pub openings: BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, F>, F)>,
    pub sumchecks: Vec<Rep3CoordinatorReductionSumcheck<F>>,
}

impl<F: JoltField> Rep3OpeningAccumulator<F> {
    pub fn new() -> Self {
        Self {
            openings: BTreeMap::new(),
            sumchecks: Vec::new(),
        }
    }

    /// Append sparse polynomial openings with transcript interaction.
    /// Each polynomial gets its own sumcheck entry.
    pub fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: &[F::Challenge],
        r_cycle: &[F::Challenge],
        claims: Vec<F>,
    ) {
        assert_eq!(polynomials.len(), claims.len());
        claims
            .iter()
            .for_each(|claim| transcript.append_scalar(claim));

        let r_concat: Vec<F::Challenge> = r_address
            .iter()
            .copied()
            .chain(r_cycle.iter().copied())
            .collect();

        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let point = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings.insert(key, (point, *claim));

            self.sumchecks
                .push(Rep3CoordinatorReductionSumcheck::new_one_hot(
                    *label, sumcheck, r_address, r_cycle, *claim,
                ));
        }
    }

    /// Append dense (committed) polynomial openings with transcript interaction.
    pub fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claims: Vec<F>,
    ) {
        assert_eq!(polynomials.len(), claims.len());
        transcript.append_scalars(&claims);

        self.sumchecks
            .push(Rep3CoordinatorReductionSumcheck::new_dense(
                polynomials.clone(),
                sumcheck,
                opening_point.clone(),
                claims.clone(),
            ));

        for (label, claim) in polynomials.into_iter().zip(claims.into_iter()) {
            let point = OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone());
            let key = OpeningId::Committed(label, sumcheck);
            self.openings.insert(key, (point, claim));
        }
    }

    /// Append a virtual polynomial opening with transcript interaction.
    pub fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);
        self.openings.insert(
            OpeningId::Virtual(polynomial, sumcheck),
            (opening_point, claim),
        );
    }

    pub fn get_opening(&self, key: OpeningId) -> F {
        self.openings.get(&key).unwrap().1
    }

    pub fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Virtual(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        (point.clone(), *claim)
    }

    pub fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Committed(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        (point.clone(), *claim)
    }

    /// Reduce all accumulated openings into a single PCS opening proof.
    ///
    /// Coordinator drives the Fiat-Shamir transcript and reconstructs
    /// additive shares from workers to build round polynomials.
    #[tracing::instrument(skip_all, name = "Rep3OpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove<PCS, ProofTranscript, N>(
        &mut self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
        network: &mut N,
    ) -> eyre::Result<ReducedOpeningProof<F, PCS, ProofTranscript>>
    where
        PCS: crate::poly::commitment::Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        N: mpc_core::protocols::rep3::network::Rep3NetworkCoordinator,
    {
        // a. Count gammas needed and sample from transcript
        let total_gammas: usize = self
            .sumchecks
            .iter()
            .map(|s| {
                if s.polynomials.len() > 1 {
                    s.polynomials.len()
                } else {
                    1
                }
            })
            .sum();
        let all_gammas: Vec<F> = transcript.challenge_vector(total_gammas);
        network.broadcast_request(all_gammas.clone())?;

        // b. Prepare coordinator sumcheck entries
        let mut gamma_offsets = vec![0usize];
        for sumcheck in self.sumchecks.iter() {
            let num_gammas = if sumcheck.polynomials.len() > 1 {
                sumcheck.polynomials.len()
            } else {
                1
            };
            gamma_offsets.push(gamma_offsets.last().unwrap() + num_gammas);
        }
        for (idx, sumcheck) in self.sumchecks.iter_mut().enumerate() {
            let offset = gamma_offsets[idx];
            let num_gammas = gamma_offsets[idx + 1] - offset;
            sumcheck.prepare_sumcheck(&all_gammas[offset..offset + num_gammas]);
        }

        // c. Sample batching coefficients (independent, matching vanilla's challenge_vector)
        let coeffs: Vec<F> = transcript.challenge_vector(self.sumchecks.len());
        network.broadcast_request(coeffs.clone())?;

        // d. Compute combined_claim
        let max_num_rounds = self
            .sumchecks
            .iter()
            .map(|s| s.num_rounds())
            .max()
            .unwrap_or(0);

        // Append input claims to transcript and scale by 2^(max - num_rounds)
        // (matches vanilla's BatchedSumcheck::prove transcript sequence)
        let mut combined_claim: F = F::zero();
        for (i, sumcheck) in self.sumchecks.iter().enumerate() {
            let input_claim = sumcheck.input_claim();
            transcript.append_scalar(&input_claim);
            let scaled = input_claim.mul_pow_2(max_num_rounds - sumcheck.num_rounds());
            combined_claim += coeffs[i] * scaled;
        }

        // e. Round-by-round sumcheck (forward iteration, matching vanilla)
        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut polys: Vec<jolt_core::poly::unipoly::CompressedUniPoly<F>> =
            Vec::with_capacity(max_num_rounds);

        for _round in 0..max_num_rounds {
            // Receive batched eval shares from workers
            let round_shares: Vec<(AdditiveShare<F>, AdditiveShare<F>)> =
                network.receive_responses()?;
            eyre::ensure!(round_shares.len() == 3, "expected 3 parties");

            let eval_0 = mpc_core::protocols::additive::combine_additive_share(
                round_shares.iter().map(|x| x.0).collect(),
            );
            let eval_2 = mpc_core::protocols::additive::combine_additive_share(
                round_shares.into_iter().map(|x| x.1).collect(),
            );

            // eval_1 = combined_claim - eval_0
            let eval_1 = combined_claim - eval_0;

            // Build UniPoly from evals, compress, append to transcript
            let uni_poly = jolt_core::poly::unipoly::UniPoly::from_evals(&[eval_0, eval_1, eval_2]);
            let compressed = uni_poly.compress();
            compressed.append_to_transcript(transcript);
            polys.push(compressed);

            // Challenge for this round
            let r_j: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            network.broadcast_request(r_j)?;
            r_sumcheck.push(r_j);

            // Update combined claim
            combined_claim = uni_poly.evaluate(&r_j);
        }

        let sumcheck_proof = jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::new(polys);

        // f. Receive final claims from workers, reconstruct
        let claim_shares: Vec<Vec<AdditiveShare<F>>> = network.receive_responses()?;
        let sumcheck_claims: Vec<F> =
            mpc_core::protocols::additive::combine_additive_vec(claim_shares);
        transcript.append_scalars(&sumcheck_claims);

        // g. Sample gamma for joint polynomial RLC, broadcast
        let gamma: F = transcript.challenge_scalar();
        network.broadcast_request(gamma)?;

        let mut gamma_powers = vec![F::one()];
        for i in 1..self.sumchecks.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        // h. Compute joint_commitment = RLC of commitments
        let mut rlc_map: BTreeMap<CommittedPolynomial, F> = BTreeMap::new();
        for (gamma_power, sumcheck) in gamma_powers.iter().zip(self.sumchecks.iter()) {
            for (coeff, polynomial) in sumcheck.rlc_coeffs.iter().zip(sumcheck.polynomials.iter()) {
                *rlc_map.entry(*polynomial).or_insert(F::zero()) += *coeff * gamma_power;
            }
        }

        let (rlc_coeffs_vec, commitments_vec): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
            .iter()
            .map(|(poly_label, rlc_coeff)| {
                let commitment = commitment_map
                    .get(poly_label)
                    .unwrap_or_else(|| panic!("Missing commitment for {poly_label:?}"))
                    .clone();
                (*rlc_coeff, commitment)
            })
            .unzip();
        let joint_commitment = PCS::combine_commitments(&commitments_vec, &rlc_coeffs_vec);

        // i. Compute joint_claim = sum gamma^i * sumcheck_claims[i]
        let joint_claim: F = gamma_powers
            .iter()
            .zip(sumcheck_claims.iter())
            .map(|(g, c)| *g * *c)
            .sum();

        // j. Call PCS::coordinate_prove
        let joint_opening_proof = PCS::coordinate_prove(
            pcs_setup,
            transcript,
            network,
            &r_sumcheck,
            &joint_claim,
            &joint_commitment,
        )?;

        Ok(ReducedOpeningProof {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        })
    }
}

impl<F: JoltField> Default for Rep3OpeningAccumulator<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ReducedOpeningProof
// ---------------------------------------------------------------------------

/// The result of the batched opening proof reduction.
pub struct ReducedOpeningProof<
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub sumcheck_claims: Vec<F>,
    pub joint_opening_proof: PCS::Proof,
}

// ---------------------------------------------------------------------------
// Opening proof reduction types (worker-side)
// ---------------------------------------------------------------------------

/// Worker-side dense polynomial prover opening.
/// Mirrors vanilla `DensePolynomialProverOpening`, adapted for MPC.
pub struct Rep3DensePolynomialProverOpening<F: JoltField> {
    /// The secret-shared polynomial being opened. `None` until `prepare_sumcheck`.
    pub polynomial: Option<Rep3DensePolynomial<F>>,
    /// Public eq polynomial: EQ(opening_point, ·). Coefficients stored in `Z`.
    pub eq_poly: DensePolynomial<F>,
}

impl<F: JoltField> Rep3DensePolynomialProverOpening<F> {
    /// Compute additive shares of eval_0 and eval_2 for the current sumcheck round.
    ///
    /// eval_0 = Σ_{j=0}^{half-1} poly[j] * eq[j]
    /// eval_2 = Σ_{j=0}^{half-1} (2*poly[j+half] - poly[j]) * (2*eq[j+half] - eq[j])
    pub fn compute_prover_message(
        &mut self,
        _round: usize,
        _previous_claim: AdditiveShare<F>,
    ) -> [AdditiveShare<F>; 2] {
        let polynomial = self.polynomial.as_ref().unwrap();
        let mle_half = polynomial.len() / 2;

        let eval_0: AdditiveShare<F> = (0..mle_half)
            .into_par_iter()
            .map(|j| (polynomial.get_bound_coeff(j) * self.eq_poly.Z[j]).into_additive())
            .sum();

        let eval_2: AdditiveShare<F> = (0..mle_half)
            .into_par_iter()
            .map(|j| {
                let poly_j = polynomial.get_bound_coeff(j);
                let poly_j_half = polynomial.get_bound_coeff(j + mle_half);
                let poly_2 = poly_j_half + poly_j_half - poly_j;

                let eq_j = self.eq_poly.Z[j];
                let eq_j_half = self.eq_poly.Z[j + mle_half];
                let eq_2 = eq_j_half + eq_j_half - eq_j;

                (poly_2 * eq_2).into_additive()
            })
            .sum();

        [eval_0, eval_2]
    }

    pub fn bind(&mut self, r_j: F::Challenge) {
        self.eq_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        self.polynomial
            .as_mut()
            .unwrap()
            .bind(r_j.into(), BindingOrder::HighToLow);
    }

    pub fn final_sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        self.polynomial.as_ref().unwrap().final_sumcheck_claim()
    }
}

/// Worker-side prover opening enum. Dispatches to dense or one-hot variant.
pub enum Rep3ProverOpening<F: JoltField> {
    Dense(Rep3DensePolynomialProverOpening<F>),
    OneHot(Rep3OneHotPolynomialProverOpening<F>),
}

impl<F: JoltField> Rep3ProverOpening<F> {
    pub fn compute_prover_message(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
    ) -> [AdditiveShare<F>; 2] {
        match self {
            Rep3ProverOpening::Dense(d) => d.compute_prover_message(round, previous_claim),
            Rep3ProverOpening::OneHot(oh) => {
                oh.compute_prover_message_shared(round, previous_claim)
            }
        }
    }

    pub fn bind(&mut self, r_j: F::Challenge, round: usize) {
        match self {
            Rep3ProverOpening::Dense(d) => d.bind(r_j),
            Rep3ProverOpening::OneHot(oh) => oh.bind(r_j, round),
        }
    }

    pub fn final_sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        match self {
            Rep3ProverOpening::Dense(d) => d.final_sumcheck_claim(),
            Rep3ProverOpening::OneHot(oh) => oh.final_sumcheck_claim(),
        }
    }
}

/// Tracks whether a reduction sumcheck entry is for dense or one-hot polynomials.
enum OpeningKind {
    Dense,
    OneHot { address_len: usize },
}

/// Worker-side opening proof reduction sumcheck entry.
/// Mirrors vanilla `OpeningProofReductionSumcheck`, adapted for MPC.
pub struct Rep3OpeningProofReductionSumcheck<F: JoltField> {
    prover_state: Option<Rep3ProverOpening<F>>,
    opening_kind: OpeningKind,
    pub polynomials: Vec<CommittedPolynomial>,
    pub sumcheck_id: SumcheckId,
    pub rlc_coeffs: Vec<F>,
    input_claims: Vec<Rep3PrimeFieldShare<F>>,
    pub opening_point: Vec<F::Challenge>,
    party_id: PartyID,
    sumcheck_claim: Option<Rep3PrimeFieldShare<F>>,
}

impl<F: JoltField> Rep3OpeningProofReductionSumcheck<F> {
    pub fn new_prover_instance_dense(
        polynomials: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claims: Vec<Rep3PrimeFieldShare<F>>,
        party_id: PartyID,
    ) -> Self {
        Self {
            prover_state: None,
            opening_kind: OpeningKind::Dense,
            polynomials,
            sumcheck_id,
            rlc_coeffs: vec![],
            input_claims: claims,
            opening_point,
            party_id,
            sumcheck_claim: None,
        }
    }

    pub fn new_prover_instance_one_hot(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        r_address: &[F::Challenge],
        r_cycle: &[F::Challenge],
        claim: Rep3PrimeFieldShare<F>,
        party_id: PartyID,
    ) -> Self {
        let opening_point: Vec<F::Challenge> =
            r_address.iter().chain(r_cycle.iter()).copied().collect();
        Self {
            prover_state: None,
            opening_kind: OpeningKind::OneHot {
                address_len: r_address.len(),
            },
            polynomials: vec![polynomial],
            sumcheck_id,
            rlc_coeffs: vec![F::one()],
            input_claims: vec![claim],
            opening_point,
            party_id,
            sumcheck_claim: None,
        }
    }

    /// Initialize the prover state from the polynomial map and RLC gammas.
    /// Must be called before the sumcheck rounds begin.
    pub fn prepare_sumcheck(
        &mut self,
        polynomials_map: &HashMap<CommittedPolynomial, Arc<Rep3MultilinearPolynomial<F>>>,
        gammas: &[F],
    ) {
        match &self.opening_kind {
            OpeningKind::Dense => {
                // Set RLC coefficients
                if self.polynomials.len() > 1 {
                    assert_eq!(gammas.len(), self.polynomials.len());
                    self.rlc_coeffs = gammas.to_vec();
                } else {
                    assert_eq!(gammas.len(), 1);
                    self.rlc_coeffs = vec![F::one()];
                }

                // Create eq polynomial from the public opening point
                let eq_evals = EqPolynomial::<F>::evals(&self.opening_point);
                let eq_poly = DensePolynomial::new(eq_evals);

                if self.polynomials.len() > 1 {
                    // Reduce claims
                    let reduced_claim: Rep3PrimeFieldShare<F> = self
                        .rlc_coeffs
                        .iter()
                        .zip(self.input_claims.iter())
                        .map(|(gamma, claim)| *claim * *gamma)
                        .sum();
                    self.input_claims = vec![reduced_claim];

                    // Create RLC dense polynomial
                    let dense_polys: Vec<&Rep3DensePolynomial<F>> = self
                        .polynomials
                        .iter()
                        .map(|label| {
                            let poly = polynomials_map.get(label).unwrap();
                            match poly.as_ref() {
                                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(d)) => d,
                                _ => panic!("Expected shared dense polynomial for {label:?}"),
                            }
                        })
                        .collect();

                    let num_coeffs = dense_polys[0].len();
                    let coeffs = &self.rlc_coeffs;
                    let combined: Vec<Rep3PrimeFieldShare<F>> = (0..num_coeffs)
                        .into_par_iter()
                        .map(|i| {
                            coeffs
                                .iter()
                                .zip(dense_polys.iter())
                                .map(|(gamma, poly)| poly.get_bound_coeff(i) * *gamma)
                                .sum()
                        })
                        .collect();

                    self.prover_state =
                        Some(Rep3ProverOpening::Dense(Rep3DensePolynomialProverOpening {
                            polynomial: Some(Rep3DensePolynomial::new(combined)),
                            eq_poly,
                        }));
                } else {
                    // Single polynomial
                    let poly = polynomials_map.get(&self.polynomials[0]).unwrap();
                    let dense = match poly.as_ref() {
                        Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(d)) => d.clone(),
                        _ => panic!("Expected shared dense polynomial"),
                    };
                    self.prover_state =
                        Some(Rep3ProverOpening::Dense(Rep3DensePolynomialProverOpening {
                            polynomial: Some(dense),
                            eq_poly,
                        }));
                }
            }
            OpeningKind::OneHot { address_len } => {
                assert_eq!(gammas.len(), 1);
                assert_eq!(self.polynomials.len(), 1);

                let poly = polynomials_map.get(&self.polynomials[0]).unwrap();
                let one_hot = match poly.as_ref() {
                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(oh)) => oh.clone(),
                    _ => panic!("Expected shared one-hot polynomial"),
                };

                let (r_address, r_cycle) = self.opening_point.split_at(*address_len);
                let oh_opening = Rep3OneHotPolynomialProverOpening::new(
                    one_hot,
                    r_address,
                    r_cycle,
                    self.party_id,
                );
                self.prover_state = Some(Rep3ProverOpening::OneHot(oh_opening));
            }
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    pub fn input_claim(&self) -> Rep3PrimeFieldShare<F> {
        assert_eq!(
            self.input_claims.len(),
            1,
            "Input claims should have been reduced"
        );
        self.input_claims[0]
    }

    pub fn compute_prover_message(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
    ) -> [AdditiveShare<F>; 2] {
        self.prover_state
            .as_mut()
            .unwrap()
            .compute_prover_message(round, previous_claim)
    }

    pub fn bind(&mut self, r_j: F::Challenge, round: usize) {
        self.prover_state.as_mut().unwrap().bind(r_j, round);
    }

    pub fn cache_sumcheck_claim(&mut self) {
        debug_assert!(self.sumcheck_claim.is_none());
        self.sumcheck_claim = Some(self.prover_state.as_ref().unwrap().final_sumcheck_claim());
    }

    pub fn sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        self.sumcheck_claim.unwrap()
    }
}

// ---------------------------------------------------------------------------
// Opening proof reduction types (coordinator-side)
// ---------------------------------------------------------------------------

/// Coordinator-side reduction sumcheck entry. Tracks polynomials, claims,
/// and RLC coefficients without holding secret-shared state.
pub struct Rep3CoordinatorReductionSumcheck<F: JoltField> {
    pub polynomials: Vec<CommittedPolynomial>,
    pub sumcheck_id: SumcheckId,
    pub opening_point: Vec<F::Challenge>,
    pub claims: Vec<F>,
    pub rlc_coeffs: Vec<F>,
}

impl<F: JoltField> Rep3CoordinatorReductionSumcheck<F> {
    pub fn new_dense(
        polynomials: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claims: Vec<F>,
    ) -> Self {
        Self {
            polynomials,
            sumcheck_id,
            opening_point,
            claims,
            rlc_coeffs: vec![],
        }
    }

    pub fn new_one_hot(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        r_address: &[F::Challenge],
        r_cycle: &[F::Challenge],
        claim: F,
    ) -> Self {
        let opening_point: Vec<F::Challenge> =
            r_address.iter().chain(r_cycle.iter()).copied().collect();
        Self {
            polynomials: vec![polynomial],
            sumcheck_id,
            opening_point,
            claims: vec![claim],
            rlc_coeffs: vec![F::one()],
        }
    }

    /// Set RLC coefficients and reduce multi-polynomial claims.
    pub fn prepare_sumcheck(&mut self, gammas: &[F]) {
        if self.polynomials.len() > 1 {
            assert_eq!(gammas.len(), self.polynomials.len());
            self.rlc_coeffs = gammas.to_vec();

            let reduced: F = self
                .rlc_coeffs
                .iter()
                .zip(self.claims.iter())
                .map(|(gamma, claim)| *gamma * *claim)
                .sum();
            self.claims = vec![reduced];
        } else {
            assert_eq!(gammas.len(), 1);
            self.rlc_coeffs = vec![F::one()];
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    pub fn input_claim(&self) -> F {
        assert_eq!(self.claims.len(), 1, "Claims should have been reduced");
        self.claims[0]
    }
}
