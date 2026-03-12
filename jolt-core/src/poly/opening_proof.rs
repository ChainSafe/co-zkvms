//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use allocative::Allocative;
use num_derive::FromPrimitive;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::{
    commitment::commitment_scheme::CommitmentScheme, eq_poly::EqPolynomial,
    multilinear_polynomial::MultilinearPolynomial,
};
use crate::{
    curve::JoltCurve,
    field::JoltField,
    subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance, SumcheckInstanceProof},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

#[derive(Clone, Debug, PartialEq, Default, Allocative)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F::Challenge>,
}

impl<const E: Endianness, F: JoltField> std::ops::Index<usize> for OpeningPoint<E, F> {
    type Output = F::Challenge;

    fn index(&self, index: usize) -> &Self::Output {
        &self.r[index]
    }
}

impl<const E: Endianness, F: JoltField> std::ops::Index<std::ops::RangeFull> for OpeningPoint<E, F> {
    type Output = [F::Challenge];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.r[..]
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn split_at_r(&self, mid: usize) -> (&[F::Challenge], &[F::Challenge]) {
        self.r.split_at(mid)
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.r.split_at(mid);
        (Self::new(left.to_vec()), Self::new(right.to_vec()))
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn new(r: Vec<F::Challenge>) -> Self {
        Self { r }
    }

    pub fn endianness(&self) -> &'static str {
        if E == BIG_ENDIAN {
            "big"
        } else {
            "little"
        }
    }

    pub fn match_endianness<const SWAPPED_E: Endianness>(&self) -> OpeningPoint<SWAPPED_E, F>
    where
        F: Clone,
    {
        let mut reversed = self.r.clone();
        if E != SWAPPED_E {
            reversed.reverse();
        }
        OpeningPoint::<SWAPPED_E, F>::new(reversed)
    }
}

impl<F: JoltField> From<Vec<F::Challenge>> for OpeningPoint<LITTLE_ENDIAN, F> {
    fn from(r: Vec<F::Challenge>) -> Self {
        Self::new(r)
    }
}

impl<F: JoltField> From<Vec<F::Challenge>> for OpeningPoint<BIG_ENDIAN, F> {
    fn from(r: Vec<F::Challenge>) -> Self {
        Self::new(r)
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F::Challenge>> for OpeningPoint<E, F> {
    fn into(self) -> Vec<F::Challenge> {
        self.r
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F::Challenge>> for &OpeningPoint<E, F>
where
    F: Clone,
{
    fn into(self) -> Vec<F::Challenge> {
        self.r.clone()
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, FromPrimitive, Allocative)]
#[repr(u8)]
pub enum SumcheckId {
    SpartanOuter,
    SpartanInner,
    SpartanShift,
    ProductVirtualization,
    InstructionBooleanity,
    InstructionHammingWeight,
    InstructionReadRaf,
    InstructionRaVirtualization,
    RamReadWriteChecking,
    RamRafEvaluation,
    RamHammingWeight,
    RamHammingBooleanity,
    RamBooleanity,
    RamRaVirtualization,
    RamOutputCheck,
    RamValEvaluation,
    RamValFinalEvaluation,
    RegistersReadWriteChecking,
    RegistersValEvaluation,
    BytecodeReadRaf,
    BytecodeBooleanity,
    BytecodeHammingWeight,
    OpeningReduction,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum OpeningId {
    Committed(CommittedPolynomial, SumcheckId),
    Virtual(VirtualPolynomial, SumcheckId),
    ReducedOpeningClaim(u32),
    UntrustedAdvice,
    TrustedAdvice,
}

pub type Openings<F> = BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, F>, F)>;

#[derive(Clone, Debug)]
pub struct BlindfoldOpeningData<F: JoltField> {
    pub opening_ids: Vec<OpeningId>,
    pub constraint_coeffs: Vec<F>,
    pub joint_claim: F,
}

pub trait OpeningAccumulator<F: JoltField> {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);
}

#[derive(Clone, Allocative)]
pub struct OpeningProofReductionSumcheck<F>
where
    F: JoltField,
{
    pub polynomials: Vec<CommittedPolynomial>,
    sumcheck_id: SumcheckId,
    rlc_coeffs: Vec<F>,
    input_claims: Vec<F>,
    opening_point: Vec<F::Challenge>,
    sumcheck_claim: Option<F>,
}

impl<F> OpeningProofReductionSumcheck<F>
where
    F: JoltField,
{
    fn new_verifier_instance(
        polynomials: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claims: Vec<F>,
    ) -> Self {
        let rlc_coeffs = if polynomials.len() == 1 {
            vec![F::one()]
        } else {
            vec![] // Will be populated later
        };
        Self { polynomials, sumcheck_id, input_claims: claims, rlc_coeffs, opening_point, sumcheck_claim: None }
    }

    fn prepare_sumcheck(&mut self, gammas: &[F]) {
        if self.polynomials.len() > 1 {
            assert_eq!(
                gammas.len(),
                self.polynomials.len(),
                "Expected {} gammas but got {}",
                self.polynomials.len(),
                gammas.len()
            );
            self.rlc_coeffs = gammas.to_vec();
        } else {
            assert_eq!(gammas.len(), 1, "Expected 1 gamma but got {}", gammas.len());
            self.rlc_coeffs = vec![F::one()];
        }

        if self.polynomials.len() > 1 {
            let reduced_claim =
                self.rlc_coeffs.par_iter().zip(self.input_claims.par_iter()).map(|(gamma, claim)| *gamma * claim).sum();
            self.input_claims = vec![reduced_claim];
        }
    }
}

impl<F, T: Transcript> SumcheckInstance<F, T> for OpeningProofReductionSumcheck<F>
where
    F: JoltField,
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    fn input_claim(&self) -> F {
        assert_eq!(self.input_claims.len(), 1, "Input claims should have been reduced by now");
        self.input_claims[0]
    }

    fn expected_output_claim(
        &self,
        _: Option<std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let eq_eval = EqPolynomial::<F>::mle(&self.opening_point, r);
        eq_eval * self.sumcheck_claim.unwrap()
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_verifier(
        &self,
        _accumulator: std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        unimplemented!("Unused")
    }
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
#[derive(Clone, Allocative)]
pub struct ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub openings: Openings<F>,
    pending_claims: Vec<F>,
    pending_claim_ids: Vec<OpeningId>,
    zk_mode: bool,
    #[cfg(test)]
    pub appended_virtual_openings: std::rc::Rc<std::cell::RefCell<Vec<OpeningId>>>,
}

impl<F> Default for ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new() -> Self {
        Self {
            openings: BTreeMap::new(),
            pending_claims: vec![],
            pending_claim_ids: vec![],
            zk_mode: false,
            #[cfg(test)]
            appended_virtual_openings: std::rc::Rc::new(std::cell::RefCell::new(vec![])),
        }
    }

    pub fn new_zk() -> Self {
        Self {
            openings: BTreeMap::new(),
            pending_claims: vec![],
            pending_claim_ids: vec![],
            zk_mode: true,
            #[cfg(test)]
            appended_virtual_openings: std::rc::Rc::new(std::cell::RefCell::new(vec![])),
        }
    }

    pub fn evaluation_openings(&self) -> &Openings<F> {
        &self.openings
    }

    pub fn evaluation_openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.openings
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
        #[cfg(test)]
        {
            let mut virtual_openings = self.appended_virtual_openings.borrow_mut();
            if let Some(index) = virtual_openings.iter().position(|id| id == &OpeningId::Virtual(polynomial, sumcheck))
            {
                virtual_openings.remove(index);
            }
        }
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

    pub fn get_untrusted_advice_opening(&self) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::UntrustedAdvice)?;
        Some((point.clone(), *claim))
    }

    pub fn get_trusted_advice_opening(&self) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::TrustedAdvice)?;
        Some((point.clone(), *claim))
    }

    pub fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claims: &[F],
    ) {
        assert_eq!(polynomials.len(), claims.len());
        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings.insert(key, (opening_point_struct, *claim));
            if self.zk_mode {
                self.pending_claims.push(*claim);
                self.pending_claim_ids.push(key);
            }
        }
        if !self.zk_mode {
            transcript.append_scalars(claims);
        }
    }

    pub fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: Vec<F::Challenge>,
        r_cycle: Vec<F::Challenge>,
        claims: Vec<F>,
    ) {
        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();
        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings.insert(key, (opening_point_struct, *claim));
            if self.zk_mode {
                self.pending_claims.push(*claim);
                self.pending_claim_ids.push(key);
            }
        }
        if !self.zk_mode {
            claims.iter().for_each(|claim| {
                transcript.append_scalar(claim);
            });
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
        let key = OpeningId::Virtual(polynomial, sumcheck);
        self.openings.insert(key, (opening_point, claim));
        if self.zk_mode {
            self.pending_claims.push(claim);
            self.pending_claim_ids.push(key);
        } else {
            transcript.append_scalar(&claim);
        }
        #[cfg(test)]
        self.appended_virtual_openings.borrow_mut().push(key);
    }

    pub fn append_untrusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        self.openings.insert(OpeningId::UntrustedAdvice, (opening_point, claim));
        if self.zk_mode {
            self.pending_claims.push(claim);
            self.pending_claim_ids.push(OpeningId::UntrustedAdvice);
        } else {
            transcript.append_scalar(&claim);
        }
    }

    pub fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        self.openings.insert(OpeningId::TrustedAdvice, (opening_point, claim));
        if self.zk_mode {
            self.pending_claims.push(claim);
            self.pending_claim_ids.push(OpeningId::TrustedAdvice);
        } else {
            transcript.append_scalar(&claim);
        }
    }

    pub fn flush_to_transcript<T: Transcript>(&mut self, transcript: &mut T) {
        for claim in self.pending_claims.drain(..) {
            transcript.append_scalar(&claim);
        }
        self.pending_claim_ids.clear();
    }

    pub fn take_pending_claims(&mut self) -> Vec<F> {
        std::mem::take(&mut self.pending_claims)
    }

    pub fn take_pending_claim_ids(&mut self) -> Vec<OpeningId> {
        std::mem::take(&mut self.pending_claim_ids)
    }
}

impl<F: JoltField> OpeningAccumulator<F> for ProverOpeningAccumulator<F> {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        ProverOpeningAccumulator::get_virtual_polynomial_opening(self, polynomial, sumcheck)
    }

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        ProverOpeningAccumulator::get_committed_polynomial_opening(self, polynomial, sumcheck)
    }
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    sumchecks: Vec<OpeningProofReductionSumcheck<F>>,
    pub openings: Openings<F>,
    zk_mode: bool,
    pending_claims: Vec<F>,
    pending_claim_ids: Vec<OpeningId>,
    blindfold_opening_data: Option<BlindfoldOpeningData<F>>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct ReducedOpeningProof<
    F: JoltField,
    C: JoltCurve,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub sumcheck_proof: SumcheckInstanceProof<F, C, ProofTranscript>,
    pub sumcheck_claims: Vec<F>,
    pub joint_opening_proof: PCS::Proof,
    #[cfg(test)]
    joint_poly: MultilinearPolynomial<F>,
    #[cfg(test)]
    joint_commitment: PCS::Commitment,
}

impl<F> Default for VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new() -> Self {
        Self {
            sumchecks: vec![],
            openings: BTreeMap::new(),
            zk_mode: false,
            pending_claims: vec![],
            pending_claim_ids: vec![],
            blindfold_opening_data: None,
        }
    }

    pub fn new_zk() -> Self {
        Self {
            sumchecks: vec![],
            openings: BTreeMap::new(),
            zk_mode: true,
            pending_claims: vec![],
            pending_claim_ids: vec![],
            blindfold_opening_data: None,
        }
    }

    pub fn set_zk_mode(&mut self, zk_mode: bool) {
        self.zk_mode = zk_mode;
    }

    pub fn len(&self) -> usize {
        self.sumchecks.len()
    }

    pub fn openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.openings
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
            .unwrap_or_else(|| panic!("No opening found for {sumcheck:?} {polynomial:?}"));
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
            .unwrap_or_else(|| panic!("No opening found for {sumcheck:?} {polynomial:?}"));
        (point.clone(), *claim)
    }

    pub fn get_untrusted_advice_opening(&self) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::UntrustedAdvice)?;
        Some((point.clone(), *claim))
    }

    pub fn get_trusted_advice_opening(&self) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::TrustedAdvice)?;
        Some((point.clone(), *claim))
    }

    pub fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
    ) {
        let claims: Vec<F> = polynomials
            .iter()
            .map(|poly| {
                let key = OpeningId::Committed(*poly, sumcheck);
                let claim = self.openings.get(&key).map(|opening| opening.1).unwrap_or(F::zero());
                self.openings.insert(key, (OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()), claim));
                if self.zk_mode {
                    self.pending_claim_ids.push(key);
                    self.pending_claims.push(claim);
                }
                claim
            })
            .collect();
        if !self.zk_mode {
            transcript.append_scalars(&claims);
        }

        self.sumchecks.push(OpeningProofReductionSumcheck::new_verifier_instance(
            polynomials,
            sumcheck,
            opening_point,
            claims,
        ));
    }

    pub fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
    ) {
        for label in polynomials.into_iter() {
            let key = OpeningId::Committed(label, sumcheck);
            let claim = self.openings.get(&key).map(|opening| opening.1).unwrap_or(F::zero());
            self.openings.insert(key, (OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()), claim));
            if self.zk_mode {
                self.pending_claim_ids.push(key);
                self.pending_claims.push(claim);
            }
            if !self.zk_mode {
                transcript.append_scalar(&claim);
            }

            self.sumchecks.push(OpeningProofReductionSumcheck::new_verifier_instance(
                vec![label],
                sumcheck,
                opening_point.clone(),
                vec![claim],
            ));
        }
    }

    pub fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::Virtual(polynomial, sumcheck);
        if let Some((_, claim)) = self.openings.get(&key) {
            let claim = *claim;
            self.openings.insert(key, (opening_point.clone(), claim));
            if self.zk_mode {
                self.pending_claims.push(claim);
                self.pending_claim_ids.push(key);
            } else {
                transcript.append_scalar(&claim);
            }
        } else {
            self.openings.insert(key, (opening_point.clone(), F::zero()));
            self.pending_claims.push(F::zero());
            self.pending_claim_ids.push(key);
        }
    }

    pub fn append_untrusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        if let Some((_, claim)) = self.openings.get(&OpeningId::UntrustedAdvice) {
            let claim = *claim;
            self.openings.insert(OpeningId::UntrustedAdvice, (opening_point.clone(), claim));
            if self.zk_mode {
                self.pending_claims.push(claim);
                self.pending_claim_ids.push(OpeningId::UntrustedAdvice);
            } else {
                transcript.append_scalar(&claim);
            }
        } else {
            self.openings.insert(OpeningId::UntrustedAdvice, (opening_point.clone(), F::zero()));
            self.pending_claims.push(F::zero());
            self.pending_claim_ids.push(OpeningId::UntrustedAdvice);
        }
    }

    pub fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        if let Some((_, claim)) = self.openings.get(&OpeningId::TrustedAdvice) {
            let claim = *claim;
            self.openings.insert(OpeningId::TrustedAdvice, (opening_point.clone(), claim));
            if self.zk_mode {
                self.pending_claims.push(claim);
                self.pending_claim_ids.push(OpeningId::TrustedAdvice);
            } else {
                transcript.append_scalar(&claim);
            }
        } else {
            self.openings.insert(OpeningId::TrustedAdvice, (opening_point.clone(), F::zero()));
            self.pending_claims.push(F::zero());
            self.pending_claim_ids.push(OpeningId::TrustedAdvice);
        }
    }

    pub fn flush_to_transcript<T: Transcript>(&mut self, transcript: &mut T) {
        for claim in self.pending_claims.drain(..) {
            transcript.append_scalar(&claim);
        }
        self.pending_claim_ids.clear();
    }

    pub fn take_pending_claims(&mut self) -> Vec<F> {
        std::mem::take(&mut self.pending_claims)
    }

    pub fn take_pending_claim_ids(&mut self) -> Vec<OpeningId> {
        std::mem::take(&mut self.pending_claim_ids)
    }

    pub fn take_blindfold_opening_data(&mut self) -> Option<BlindfoldOpeningData<F>> {
        self.blindfold_opening_data.take()
    }

    /// Verifies that the given `reduced_opening_proof` (consisting of a sumcheck proof
    /// and a single opening proof) indeed proves the openings accumulated.
    pub fn reduce_and_verify<C: JoltCurve, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        &mut self,
        pcs_setup: &PCS::VerifierSetup,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        reduced_opening_proof: &ReducedOpeningProof<F, C, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let total_challenges_needed: usize = self
            .sumchecks
            .iter()
            .map(|sumcheck| if sumcheck.polynomials.len() > 1 { sumcheck.polynomials.len() } else { 1 })
            .sum();

        let all_gammas: Vec<F> = transcript.challenge_vector(total_challenges_needed);

        let mut gamma_offsets = vec![0];
        for sumcheck in self.sumchecks.iter() {
            let num_gammas = if sumcheck.polynomials.len() > 1 { sumcheck.polynomials.len() } else { 1 };
            gamma_offsets.push(gamma_offsets.last().unwrap() + num_gammas);
        }

        self.sumchecks.par_iter_mut().zip(gamma_offsets.par_iter()).for_each(|(sumcheck, &offset)| {
            let num_gammas = if sumcheck.polynomials.len() > 1 { sumcheck.polynomials.len() } else { 1 };
            let gammas_slice = &all_gammas[offset..offset + num_gammas];
            sumcheck.prepare_sumcheck(gammas_slice);
        });

        let num_sumcheck_rounds = self.sumchecks.iter().map(|opening| opening.opening_point.len()).max().unwrap();

        self.sumchecks
            .iter_mut()
            .zip(reduced_opening_proof.sumcheck_claims.iter())
            .for_each(|(opening, claim)| opening.sumcheck_claim = Some(*claim));

        // Verify the sumcheck
        let r_sumcheck = self.verify_batch_opening_reduction(&reduced_opening_proof.sumcheck_proof, transcript)?;

        transcript.append_scalars(&reduced_opening_proof.sumcheck_claims);

        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.sumchecks.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let opening_ids: Vec<OpeningId> =
            (0..self.sumchecks.len()).map(|idx| OpeningId::ReducedOpeningClaim(idx as u32)).collect();
        for (opening_id, claim) in opening_ids.iter().zip(reduced_opening_proof.sumcheck_claims.iter()) {
            self.openings.insert(*opening_id, (OpeningPoint::new(r_sumcheck.clone()), *claim));
        }
        // Compute the commitment for the reduced opening proof by homomorphically combining
        // the commitments of the individual polynomials.
        let joint_commitment = {
            let mut rlc_map = HashMap::new();
            for (gamma, sumcheck) in gamma_powers.iter().zip(self.sumchecks.iter()) {
                for (coeff, polynomial) in sumcheck.rlc_coeffs.iter().zip(sumcheck.polynomials.iter()) {
                    if let Some(value) = rlc_map.get_mut(&polynomial) {
                        *value += *coeff * gamma;
                    } else {
                        rlc_map.insert(polynomial, *coeff * gamma);
                    }
                }
            }

            let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) =
                rlc_map.into_iter().map(|(k, v)| (v, commitment_map.remove(k).unwrap())).unzip();
            debug_assert!(commitment_map.is_empty(), "Every commitment should be used");

            PCS::combine_commitments(&commitments, &coeffs)
        };

        #[cfg(test)]
        assert_eq!(joint_commitment, reduced_opening_proof.joint_commitment, "joint commitment mismatch");

        // Compute joint claim = ∑ᵢ γⁱ⋅ claimᵢ
        let joint_claim: F = gamma_powers
            .iter()
            .zip(reduced_opening_proof.sumcheck_claims.iter())
            .zip(self.sumchecks.iter())
            .map(|((coeff, claim), opening)| {
                let r_slice = &r_sumcheck[..num_sumcheck_rounds - opening.opening_point.len()];
                let lagrange_eval: F = r_slice.iter().map(|r| F::one() - r).product();
                *coeff * claim * lagrange_eval
            })
            .sum();
        let constraint_coeffs: Vec<F> = gamma_powers
            .iter()
            .zip(self.sumchecks.iter())
            .map(|(coeff, opening)| {
                let r_slice = &r_sumcheck[..num_sumcheck_rounds - opening.opening_point.len()];
                let lagrange_eval: F = r_slice.iter().map(|r| F::one() - r).product();
                *coeff * lagrange_eval
            })
            .collect();

        self.blindfold_opening_data = Some(BlindfoldOpeningData {
            opening_ids,
            constraint_coeffs,
            joint_claim,
        });

        // Verify the reduced opening proof
        PCS::verify(
            &reduced_opening_proof.joint_opening_proof,
            pcs_setup,
            transcript,
            &r_sumcheck,
            &joint_claim,
            &joint_commitment,
        )
    }

    /// Verifies the sumcheck proven in batch opening reduction.
    fn verify_batch_opening_reduction<C: JoltCurve, ProofTranscript: Transcript>(
        &self,
        sumcheck_proof: &SumcheckInstanceProof<F, C, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let instances: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = self
            .sumchecks
            .iter()
            .map(|opening| {
                let instance: &dyn SumcheckInstance<F, ProofTranscript> = opening;
                instance
            })
            .collect();
        BatchedSumcheck::verify(sumcheck_proof, instances, None, transcript).map(|(_, r, _)| r)
    }
}

impl<F: JoltField> OpeningAccumulator<F> for VerifierOpeningAccumulator<F> {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        VerifierOpeningAccumulator::get_virtual_polynomial_opening(self, polynomial, sumcheck)
    }

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        VerifierOpeningAccumulator::get_committed_polynomial_opening(self, polynomial, sumcheck)
    }
}
