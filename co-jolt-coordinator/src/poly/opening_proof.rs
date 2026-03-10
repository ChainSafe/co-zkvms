use std::collections::{BTreeMap, HashMap};

use crate::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::{OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use crate::poly::commitment::Rep3CommitmentScheme;
use crate::subprotocols::sumcheck::{Rep3BatchedSumcheck, Rep3SumcheckInstance};

pub struct Rep3OpeningAccumulator<F: JoltField> {
    pub openings: BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, F>, F)>,
    pub sumchecks: Vec<Rep3CoordinatorReductionSumcheck<F>>,
    pub opening_proof_reduction_claims: Vec<F>,
}

impl<F: JoltField> Rep3OpeningAccumulator<F> {
    pub fn new() -> Self {
        Self {
            openings: BTreeMap::new(),
            sumchecks: Vec::new(),
            opening_proof_reduction_claims: Vec::new(),
        }
    }

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
            self.sumchecks.push(Rep3CoordinatorReductionSumcheck::new_one_hot(
                *label, sumcheck, r_address, r_cycle, *claim,
            ));
        }
    }

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

        self.sumchecks.push(Rep3CoordinatorReductionSumcheck::new_dense(
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

    pub fn get_opening(&self, key: OpeningId) -> F {
        self.openings.get(&key).unwrap().1
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

    #[tracing::instrument(skip_all, name = "OpeningAcc::reduce_and_prove")]
    pub fn reduce_and_prove<PCS, ProofTranscript, N>(
        &mut self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
        network: &mut N,
    ) -> eyre::Result<ReducedOpeningProof<F, PCS, ProofTranscript>>
    where
        PCS: CommitmentScheme<Field = F>
            + Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        N: mpc_core::protocols::rep3::network::Rep3NetworkCoordinator,
    {
        let total_gammas: usize = self
            .sumchecks
            .iter()
            .map(|s| if s.polynomials.len() > 1 { s.polynomials.len() } else { 1 })
            .sum();
        let all_gammas: Vec<F> = transcript.challenge_vector(total_gammas);
        network.broadcast_request(all_gammas.clone())?;

        let mut gamma_offsets = vec![0usize];
        for sumcheck in &self.sumchecks {
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

        let saved_meta: Vec<(Vec<F>, Vec<CommittedPolynomial>)> = self
            .sumchecks
            .iter()
            .map(|s| (s.rlc_coeffs.clone(), s.polynomials.clone()))
            .collect();
        let num_sumchecks = self.sumchecks.len();

        let instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> = self
            .sumchecks
            .drain(..)
            .map(|s| Box::new(s) as Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>)
            .collect();
        let (sumcheck_proof, r_sumcheck) =
            Rep3BatchedSumcheck::prove(&instances, self, transcript, network)?;

        let sumcheck_claims = std::mem::take(&mut self.opening_proof_reduction_claims);
        assert_eq!(sumcheck_claims.len(), num_sumchecks);

        transcript.append_scalars(&sumcheck_claims);
        let gamma: F = transcript.challenge_scalar();
        network.broadcast_request(gamma)?;

        let mut gamma_powers = vec![F::one()];
        for i in 1..num_sumchecks {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let mut rlc_map: BTreeMap<CommittedPolynomial, F> = BTreeMap::new();
        for (gamma_power, (rlc_coeffs, poly_labels)) in gamma_powers.iter().zip(saved_meta.iter()) {
            for (coeff, polynomial) in rlc_coeffs.iter().zip(poly_labels.iter()) {
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

        let joint_claim: F = gamma_powers
            .iter()
            .zip(sumcheck_claims.iter())
            .map(|(g, c)| *g * *c)
            .sum();

        let joint_opening_proof =
            <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::coordinate_prove(
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

pub struct ReducedOpeningProof<
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub sumcheck_claims: Vec<F>,
    pub joint_opening_proof: PCS::Proof,
}

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

    fn input_claim(&self) -> F {
        assert_eq!(self.claims.len(), 1, "Claims should have been reduced");
        self.claims[0]
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T>
    for Rep3CoordinatorReductionSumcheck<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim()
    }

    fn expected_output_claim(
        &self,
        _accumulator: &Rep3OpeningAccumulator<F>,
        _r: &[F::Challenge],
    ) -> F {
        unimplemented!("not used in prove path")
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        assert_eq!(claims.len(), 1);
        accumulator.opening_proof_reduction_claims.push(claims[0]);
    }
}
