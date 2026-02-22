use std::collections::BTreeMap;

use jolt_core::poly::opening_proof::{OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};

use crate::field::JoltField;

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
}

impl<F: JoltField> Rep3OpeningAccumulatorWorker<F> {
    pub fn new() -> Self {
        Self {
            openings: BTreeMap::new(),
        }
    }

    /// Append sparse polynomial openings (one-hot style) at `[r_address, r_cycle]`.
    pub fn append_sparse(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: &[F::Challenge],
        r_cycle: &[F::Challenge],
        claims: Vec<Rep3PrimeFieldShare<F>>,
    ) {
        assert_eq!(polynomials.len(), claims.len());
        let r_concat: Vec<F::Challenge> =
            r_address.iter().copied().chain(r_cycle.iter().copied()).collect();

        for (label, claim) in polynomials.into_iter().zip(claims.into_iter()) {
            let point = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(label, sumcheck);
            self.openings.insert(key, (point, claim));
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
}

impl<F: JoltField> Default for Rep3OpeningAccumulatorWorker<F> {
    fn default() -> Self {
        Self::new()
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
}

impl<F: JoltField> Rep3OpeningAccumulator<F> {
    pub fn new() -> Self {
        Self {
            openings: BTreeMap::new(),
        }
    }

    /// Append sparse polynomial openings with transcript interaction.
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
        claims.iter().for_each(|claim| transcript.append_scalar(claim));

        let r_concat: Vec<F::Challenge> =
            r_address.iter().copied().chain(r_cycle.iter().copied()).collect();

        for (label, claim) in polynomials.into_iter().zip(claims.into_iter()) {
            let point = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(label, sumcheck);
            self.openings.insert(key, (point, claim));
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
}

impl<F: JoltField> Default for Rep3OpeningAccumulator<F> {
    fn default() -> Self {
        Self::new()
    }
}
