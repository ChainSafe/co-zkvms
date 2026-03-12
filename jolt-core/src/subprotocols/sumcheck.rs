#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::opening_proof::{
    OpeningId, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;

use ark_serialize::*;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

/// Verifier-side trait for a sumcheck instance that can be batched with other instances.
pub trait SumcheckInstance<F: JoltField, T: Transcript> {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;
    fn input_claim(&self) -> F;

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F;

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F>;

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    );

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
    ) -> Vec<F> {
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

    // Prover-side methods with default panic implementations.
    // These exist to allow files that implement both prover and verifier
    // logic to compile. Only the verifier methods above are required.
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        unimplemented!("prover not available")
    }
    fn bind(&mut self, _r_j: F::Challenge, _round: usize) {
        unimplemented!("prover not available")
    }
    fn cache_openings_prover(
        &self,
        _accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        unimplemented!("prover not available")
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, _flamegraph: &mut allocative::FlameGraphBuilder) {}
}

pub enum SingleSumcheck {}
impl SingleSumcheck {
    /// Verifies a single sumcheck instance.
    pub fn verify<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>(
        sumcheck_instance: &dyn SumcheckInstance<F, ProofTranscript>,
        proof: &SumcheckInstanceProof<F, C, ProofTranscript>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let input_claim = sumcheck_instance.input_claim();
        transcript.append_scalar(&input_claim); // Append input claim

        let (output_claim, r) = proof.verify(
            input_claim,
            sumcheck_instance.num_rounds(),
            sumcheck_instance.degree(),
            transcript,
        )?;

        if !proof.is_zk()
            && output_claim
                != sumcheck_instance.expected_output_claim(opening_accumulator.clone(), &r)
        {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        sumcheck_instance.cache_openings_verifier(
            opening_accumulator.unwrap(),
            transcript,
            sumcheck_instance.normalize_opening_point(&r),
        );

        Ok(r)
    }
}

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    pub fn verify<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, C, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstance<F, ProofTranscript>>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, Vec<F::Challenge>, Vec<OpeningId>), ProofVerifyError> {
        let max_degree = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.degree())
            .max()
            .unwrap();
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let is_zk = proof.is_zk();
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        let claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim();
                if !is_zk {
                    transcript.append_scalar(&input_claim);
                }
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        if is_zk {
            if let Some(opening_accumulator) = &opening_accumulator {
                opening_accumulator.borrow_mut().set_zk_mode(true);
            }
        }

        let expected_output_claim = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

                if let Some(opening_accumulator) = &opening_accumulator {
                    sumcheck.cache_openings_verifier(
                        opening_accumulator.clone(),
                        transcript,
                        sumcheck.normalize_opening_point(r_slice),
                    );
                }
                let claim = sumcheck.expected_output_claim(opening_accumulator.clone(), r_slice);

                claim * coeff
            })
            .sum();

        if !is_zk && output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        if let Some(opening_accumulator) = &opening_accumulator {
            if is_zk {
                if let SumcheckInstanceProof::Zk(zk_proof) = proof {
                    let mut accumulator = opening_accumulator.borrow_mut();
                    transcript.append_message(b"output_claims_coms");
                    for commitment in &zk_proof.output_claims_commitments {
                        transcript.append_serializable(commitment);
                    }
                    let _ = accumulator.take_pending_claims();
                    accumulator.set_zk_mode(false);
                }
            } else {
                opening_accumulator.borrow_mut().flush_to_transcript(transcript);
            }
        }

        let output_claim_ids = if let Some(opening_accumulator) = &opening_accumulator {
            opening_accumulator.borrow_mut().take_pending_claim_ids()
        } else {
            Vec::new()
        };

        Ok((batching_coeffs, r_sumcheck, output_claim_ids))
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ClearSumcheckProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> ClearSumcheckProof<F, ProofTranscript> {
    pub fn new(compressed_polys: Vec<CompressedUniPoly<F>>) -> ClearSumcheckProof<F, ProofTranscript> {
        ClearSumcheckProof {
            compressed_polys,
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F::Challenge> = Vec::new();

        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            self.compressed_polys[i].append_to_transcript(transcript);

            let r_i = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);

            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

#[derive(Debug, Clone)]
pub struct ZkSumcheckProof<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> {
    pub round_commitments: Vec<C::G1>,
    pub poly_degrees: Vec<usize>,
    pub output_claims_commitments: Vec<C::G1>,
    _marker: PhantomData<(F, ProofTranscript)>,
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> CanonicalSerialize
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.round_commitments
            .serialize_with_mode(&mut writer, compress)?;
        self.poly_degrees
            .serialize_with_mode(&mut writer, compress)?;
        self.output_claims_commitments
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.round_commitments.serialized_size(compress)
            + self.poly_degrees.serialized_size(compress)
            + self.output_claims_commitments.serialized_size(compress)
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> Valid
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.round_commitments.check()?;
        self.poly_degrees.check()?;
        self.output_claims_commitments.check()
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> CanonicalDeserialize
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let round_commitments =
            Vec::<C::G1>::deserialize_with_mode(&mut reader, compress, validate)?;
        let poly_degrees = Vec::<usize>::deserialize_with_mode(&mut reader, compress, validate)?;
        let output_claims_commitments =
            Vec::<C::G1>::deserialize_with_mode(reader, compress, validate)?;
        Ok(Self {
            round_commitments,
            poly_degrees,
            output_claims_commitments,
            _marker: PhantomData,
        })
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>
    ZkSumcheckProof<F, C, ProofTranscript>
{
    pub fn new(
        round_commitments: Vec<C::G1>,
        poly_degrees: Vec<usize>,
        output_claims_commitments: Vec<C::G1>,
    ) -> Self {
        Self {
            round_commitments,
            poly_degrees,
            output_claims_commitments,
            _marker: PhantomData,
        }
    }

    pub fn verify_transcript_only(
        &self,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        if self.round_commitments.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.round_commitments.len(),
            ));
        }
        if self.poly_degrees.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.poly_degrees.len(),
            ));
        }
        for &degree in &self.poly_degrees {
            if degree > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(degree_bound, degree));
            }
        }

        let mut r = Vec::with_capacity(num_rounds);
        for commitment in &self.round_commitments {
            transcript.append_message(b"sumcheck_commitment");
            transcript.append_serializable(commitment);
            let r_i = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);
        }
        Ok(r)
    }
}

#[derive(Debug, Clone)]
pub enum SumcheckInstanceProof<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> {
    Clear(ClearSumcheckProof<F, ProofTranscript>),
    Zk(ZkSumcheckProof<F, C, ProofTranscript>),
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> CanonicalSerialize
    for SumcheckInstanceProof<F, C, ProofTranscript>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Clear(proof) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
            Self::Zk(proof) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::Clear(proof) => proof.serialized_size(compress),
            Self::Zk(proof) => proof.serialized_size(compress),
        }
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> Valid
    for SumcheckInstanceProof<F, C, ProofTranscript>
{
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Clear(proof) => proof.check(),
            Self::Zk(proof) => proof.check(),
        }
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> CanonicalDeserialize
    for SumcheckInstanceProof<F, C, ProofTranscript>
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => Ok(Self::Clear(ClearSumcheckProof::deserialize_with_mode(
                reader, compress, validate,
            )?)),
            1 => Ok(Self::Zk(ZkSumcheckProof::deserialize_with_mode(
                reader, compress, validate,
            )?)),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>
    SumcheckInstanceProof<F, C, ProofTranscript>
{
    pub fn new(compressed_polys: Vec<CompressedUniPoly<F>>) -> Self {
        Self::Clear(ClearSumcheckProof::new(compressed_polys))
    }

    pub fn new_standard(compressed_polys: Vec<CompressedUniPoly<F>>) -> Self {
        Self::new(compressed_polys)
    }

    pub fn new_zk(
        round_commitments: Vec<C::G1>,
        poly_degrees: Vec<usize>,
        output_claims_commitments: Vec<C::G1>,
    ) -> Self {
        Self::Zk(ZkSumcheckProof::new(
            round_commitments,
            poly_degrees,
            output_claims_commitments,
        ))
    }

    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        match self {
            Self::Clear(proof) => proof.verify(claim, num_rounds, degree_bound, transcript),
            Self::Zk(proof) => Ok((F::zero(), proof.verify_transcript_only(num_rounds, degree_bound, transcript)?)),
        }
    }

    pub fn is_zk(&self) -> bool {
        matches!(self, Self::Zk(_))
    }

    pub fn num_rounds(&self) -> usize {
        match self {
            Self::Clear(proof) => proof.compressed_polys.len(),
            Self::Zk(proof) => proof.round_commitments.len(),
        }
    }
}

/// Helper function to encapsulate the common subroutine for sumcheck with eq poly factor.
/// Returns the derived challenge.
#[inline]
pub fn process_eq_sumcheck_round<F: JoltField, ProofTranscript: Transcript>(
    quadratic_evals: (F, F), // (t_i(0), t_i(infty))
    eq_poly: &mut GruenSplitEqPolynomial<F>,
    polys: &mut Vec<CompressedUniPoly<F>>,
    r: &mut Vec<F::Challenge>,
    claim: &mut F,
    transcript: &mut ProofTranscript,
) -> F::Challenge {
    let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];

    let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
        [
            eq_poly.current_scalar - scalar_times_w_i,
            scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
        ],
        quadratic_evals.0,
        quadratic_evals.1,
        *claim,
    );

    let compressed_poly = cubic_poly.compress();
    compressed_poly.append_to_transcript(transcript);

    let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
    r.push(r_i);
    polys.push(compressed_poly);

    *claim = cubic_poly.evaluate(&r_i);

    eq_poly.bind(r_i);

    r_i
}

#[cfg(test)]
mod tests {
    use super::{BatchedSumcheck, SumcheckInstance, SumcheckInstanceProof, ZkSumcheckProof};
    use crate::curve::Bn254Curve;
    use crate::field::JoltField;
    use crate::poly::commitment::pedersen::PedersenGenerators;
    use crate::poly::opening_proof::{
        OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
    };
    use crate::transcripts::{KeccakTranscript, Transcript};
    use crate::zkvm::witness::VirtualPolynomial;
    use ark_bn254::Fr;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct MockZkInstance {
        input_claim: Fr,
        output_claim: Fr,
        sumcheck_id: SumcheckId,
        poly: VirtualPolynomial,
    }

    impl SumcheckInstance<Fr, KeccakTranscript> for MockZkInstance {
        fn degree(&self) -> usize {
            1
        }

        fn num_rounds(&self) -> usize {
            1
        }

        fn input_claim(&self) -> Fr {
            self.input_claim
        }

        fn expected_output_claim(
            &self,
            opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<Fr>>>>,
            _r: &[<Fr as JoltField>::Challenge],
        ) -> Fr {
            let accumulator = opening_accumulator.unwrap();
            let claim = accumulator
                .borrow()
                .get_virtual_polynomial_opening(self.poly, self.sumcheck_id)
                .1;
            claim
        }

        fn normalize_opening_point(
            &self,
            opening_point: &[<Fr as JoltField>::Challenge],
        ) -> OpeningPoint<BIG_ENDIAN, Fr> {
            OpeningPoint::new(opening_point.to_vec())
        }

        fn cache_openings_verifier(
            &self,
            accumulator: Rc<RefCell<VerifierOpeningAccumulator<Fr>>>,
            transcript: &mut KeccakTranscript,
            opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
        ) {
            let mut accumulator = accumulator.borrow_mut();
            accumulator.openings.insert(
                crate::poly::opening_proof::OpeningId::Virtual(self.poly, self.sumcheck_id),
                (opening_point.clone(), self.output_claim),
            );
            accumulator.append_virtual(transcript, self.poly, self.sumcheck_id, opening_point);
        }
    }

    #[test]
    fn zk_sumcheck_proof_replays_round_commitments() {
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(4);
        let commitment_0 = gens.commit(&[Fr::from(3_u64), Fr::from(5_u64)], &Fr::from(7_u64));
        let commitment_1 = gens.commit(&[Fr::from(11_u64)], &Fr::from(13_u64));

        let proof =
            ZkSumcheckProof::<Fr, Bn254Curve, KeccakTranscript>::new(
                vec![commitment_0, commitment_1],
                vec![1, 0],
                vec![],
            );

        let mut expected = KeccakTranscript::new(b"zk_sumcheck_rounds");
        expected.append_message(b"sumcheck_commitment");
        expected.append_serializable(&commitment_0);
        let _: <Fr as JoltField>::Challenge = expected.challenge_scalar_optimized::<Fr>();
        expected.append_message(b"sumcheck_commitment");
        expected.append_serializable(&commitment_1);
        let _: <Fr as JoltField>::Challenge = expected.challenge_scalar_optimized::<Fr>();

        let mut verifier = KeccakTranscript::new(b"zk_sumcheck_rounds");
        verifier.compare_to(expected);
        let challenges = proof
            .verify_transcript_only(2, 1, &mut verifier)
            .expect("transcript replay should succeed");
        assert_eq!(challenges.len(), 2);
    }

    #[test]
    fn zk_batched_sumcheck_uses_output_claim_commitments() {
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(4);
        let round_commitment = gens.commit(&[Fr::from(17_u64), Fr::from(19_u64)], &Fr::from(23_u64));
        let output_commitment = gens.commit(&[Fr::from(29_u64)], &Fr::from(31_u64));
        let proof = SumcheckInstanceProof::<Fr, Bn254Curve, KeccakTranscript>::new_zk(
            vec![round_commitment],
            vec![1],
            vec![output_commitment],
        );

        let instance = MockZkInstance {
            input_claim: Fr::from(37_u64),
            output_claim: Fr::from(41_u64),
            sumcheck_id: SumcheckId::ProductVirtualization,
            poly: VirtualPolynomial::Product,
        };

        let mut expected = KeccakTranscript::new(b"zk_output_claim_commitments");
        let _: Vec<Fr> = expected.challenge_vector(1);
        expected.append_message(b"sumcheck_commitment");
        expected.append_serializable(&round_commitment);
        let _: <Fr as JoltField>::Challenge = expected.challenge_scalar_optimized::<Fr>();
        expected.append_message(b"output_claims_coms");
        expected.append_serializable(&output_commitment);

        let mut verifier = KeccakTranscript::new(b"zk_output_claim_commitments");
        verifier.compare_to(expected);

        let accumulator = Rc::new(RefCell::new(VerifierOpeningAccumulator::<Fr>::new_zk()));
        let (batching_coeffs, r_sumcheck, output_claim_ids) = BatchedSumcheck::verify(
            &proof,
            vec![&instance as &dyn SumcheckInstance<Fr, KeccakTranscript>],
            Some(accumulator.clone()),
            &mut verifier,
        )
        .expect("zk verification should replay transcript");

        assert_eq!(batching_coeffs.len(), 1);
        assert_eq!(r_sumcheck.len(), 1);
        assert_eq!(output_claim_ids.len(), 1);
        assert!(accumulator.borrow_mut().take_pending_claims().is_empty());
        assert!(accumulator.borrow_mut().take_pending_claim_ids().is_empty());
    }
}
