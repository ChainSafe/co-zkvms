use co_jolt2::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use co_jolt2::subprotocols::sumcheck::{Rep3BatchedSumcheck, Rep3BatchedSumcheckWorker};
use co_jolt2::utils::test_utils::run_rep3_local_test_with_coordinator;
use co_jolt2::utils::types::Rep3Value;
use co_jolt2::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};

use ark_bn254::Fr;
use ark_ff::{One, Zero};
use jolt_core::field::JoltField as _;
use jolt_core::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::IoContextPool;
use mpc_core::protocols::rep3::PartyID;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

// ---------------------------------------------------------------------------
// Toy linear multilinear sumcheck (degree 1)
// ---------------------------------------------------------------------------

struct ToyLinearWorker {
    party_id: PartyID,
    coeffs: Vec<Fr>, // evaluations on {0,1}^n in low-to-high variable order
}

impl ToyLinearWorker {
    fn new(party_id: PartyID, coeffs: Vec<Fr>) -> Self {
        Self { party_id, coeffs }
    }

    fn num_rounds_from_len(len: usize) -> usize {
        len.trailing_zeros() as usize
    }

    fn promote(&self, x: Fr) -> AdditiveShare<Fr> {
        additive::promote_to_trivial_share(x, self.party_id)
    }
}

impl Rep3SumcheckInstanceWorker<Fr> for ToyLinearWorker {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        Self::num_rounds_from_len(self.coeffs.len())
    }

    fn input_claim(&self) -> Rep3Value<Fr> {
        Rep3Value::Public(self.coeffs.iter().copied().sum())
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        _previous_claim: AdditiveShare<Fr>,
        max_degree: usize,
    ) -> Vec<AdditiveShare<Fr>> {
        let mut eval0 = Fr::zero();
        let mut eval1 = Fr::zero();
        for pair in self.coeffs.chunks_exact(2) {
            eval0 += pair[0];
            eval1 += pair[1];
        }

        let slope = eval1 - eval0;
        let mut evals = vec![AdditiveShare::zero(); max_degree];
        evals[0] = self.promote(eval0);
        for x in 2..=max_degree {
            let fx = eval0 + slope * Fr::from(x as u64);
            evals[x - 1] = self.promote(fx);
        }
        evals
    }

    fn bind(&mut self, r_j: <Fr as jolt_core::field::JoltField>::Challenge, _round: usize) {
        let r: Fr = r_j.into();
        let one_minus_r = Fr::one() - r;

        let mut next = Vec::with_capacity(self.coeffs.len() / 2);
        for pair in self.coeffs.chunks_exact(2) {
            next.push(pair[0] * one_minus_r + pair[1] * r);
        }
        self.coeffs = next;
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fr> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_worker(
        &mut self,
        _accumulator: &mut Rep3OpeningAccumulatorWorker<Fr>,
        _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
    ) -> Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<Fr>> {
        vec![]
    }
}

struct ToyLinearCoordinator {
    coeffs: Vec<Fr>,
}

impl ToyLinearCoordinator {
    fn new(coeffs: Vec<Fr>) -> Self {
        Self { coeffs }
    }

    fn eval_mle_at_r(&self, r: &[<Fr as jolt_core::field::JoltField>::Challenge]) -> Fr {
        let mut v = self.coeffs.clone();
        for &r_i in r {
            let r_i: Fr = r_i.into();
            let one_minus_r = Fr::one() - r_i;
            let mut next = Vec::with_capacity(v.len() / 2);
            for pair in v.chunks_exact(2) {
                next.push(pair[0] * one_minus_r + pair[1] * r_i);
            }
            v = next;
        }
        v[0]
    }
}

impl Rep3SumcheckInstance<Fr, KeccakTranscript> for ToyLinearCoordinator {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        ToyLinearWorker::num_rounds_from_len(self.coeffs.len())
    }

    fn input_claim_public(&self) -> Fr {
        self.coeffs.iter().copied().sum()
    }

    fn expected_output_claim(
        &self,
        _accumulator: &Rep3OpeningAccumulator<Fr>,
        r: &[<Fr as jolt_core::field::JoltField>::Challenge],
    ) -> Fr {
        self.eval_mle_at_r(r)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fr> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings(
        &self,
        _accumulator: &mut Rep3OpeningAccumulator<Fr>,
        _transcript: &mut KeccakTranscript,
        _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
        claims: Vec<Fr>,
    ) {
        assert!(claims.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Toy cubic sumcheck: product of three multilinear polynomials (degree 3)
// ---------------------------------------------------------------------------

struct ToyCubicProductWorker {
    party_id: PartyID,
    a: Vec<Fr>,
    b: Vec<Fr>,
    c: Vec<Fr>,
}

impl ToyCubicProductWorker {
    fn new(party_id: PartyID, a: Vec<Fr>, b: Vec<Fr>, c: Vec<Fr>) -> Self {
        assert_eq!(a.len(), b.len());
        assert_eq!(b.len(), c.len());
        Self { party_id, a, b, c }
    }

    fn promote(&self, x: Fr) -> AdditiveShare<Fr> {
        additive::promote_to_trivial_share(x, self.party_id)
    }

    fn num_rounds(&self) -> usize {
        self.a.len().trailing_zeros() as usize
    }

    fn sum_product_at_x(&self, x: Fr) -> Fr {
        let mut acc = Fr::zero();
        for ((ap, bp), cp) in self
            .a
            .chunks_exact(2)
            .zip(self.b.chunks_exact(2))
            .zip(self.c.chunks_exact(2))
        {
            let a0 = ap[0];
            let a1 = ap[1];
            let b0 = bp[0];
            let b1 = bp[1];
            let c0 = cp[0];
            let c1 = cp[1];
            let ax = a0 + (a1 - a0) * x;
            let bx = b0 + (b1 - b0) * x;
            let cx = c0 + (c1 - c0) * x;
            acc += ax * bx * cx;
        }
        acc
    }

    fn bind_vec(v: &mut Vec<Fr>, r: Fr) {
        let one_minus_r = Fr::one() - r;
        let mut next = Vec::with_capacity(v.len() / 2);
        for pair in v.chunks_exact(2) {
            next.push(pair[0] * one_minus_r + pair[1] * r);
        }
        *v = next;
    }
}

impl Rep3SumcheckInstanceWorker<Fr> for ToyCubicProductWorker {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self) -> Rep3Value<Fr> {
        Rep3Value::Public(self.a
            .iter()
            .zip(self.b.iter())
            .zip(self.c.iter())
            .map(|((&a, &b), &c)| a * b * c)
            .sum())
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        _previous_claim: AdditiveShare<Fr>,
        max_degree: usize,
    ) -> Vec<AdditiveShare<Fr>> {
        let mut evals = vec![AdditiveShare::zero(); max_degree];
        evals[0] = self.promote(self.sum_product_at_x(Fr::zero()));
        for x in 2..=max_degree {
            evals[x - 1] = self.promote(self.sum_product_at_x(Fr::from(x as u64)));
        }
        evals
    }

    fn bind(&mut self, r_j: <Fr as jolt_core::field::JoltField>::Challenge, _round: usize) {
        let r: Fr = r_j.into();
        Self::bind_vec(&mut self.a, r);
        Self::bind_vec(&mut self.b, r);
        Self::bind_vec(&mut self.c, r);
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fr> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_worker(
        &mut self,
        _accumulator: &mut Rep3OpeningAccumulatorWorker<Fr>,
        _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
    ) -> Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<Fr>> {
        vec![]
    }
}

struct ToyCubicProductCoordinator {
    a: Vec<Fr>,
    b: Vec<Fr>,
    c: Vec<Fr>,
}

impl ToyCubicProductCoordinator {
    fn new(a: Vec<Fr>, b: Vec<Fr>, c: Vec<Fr>) -> Self {
        Self { a, b, c }
    }

    fn eval_mle(mut v: Vec<Fr>, r: &[<Fr as jolt_core::field::JoltField>::Challenge]) -> Fr {
        for &r_i in r {
            let r_i: Fr = r_i.into();
            let one_minus_r = Fr::one() - r_i;
            let mut next = Vec::with_capacity(v.len() / 2);
            for pair in v.chunks_exact(2) {
                next.push(pair[0] * one_minus_r + pair[1] * r_i);
            }
            v = next;
        }
        v[0]
    }
}

impl Rep3SumcheckInstance<Fr, KeccakTranscript> for ToyCubicProductCoordinator {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.a.len().trailing_zeros() as usize
    }

    fn input_claim_public(&self) -> Fr {
        self.a
            .iter()
            .zip(self.b.iter())
            .zip(self.c.iter())
            .map(|((&a, &b), &c)| a * b * c)
            .sum()
    }

    fn expected_output_claim(
        &self,
        _accumulator: &Rep3OpeningAccumulator<Fr>,
        r: &[<Fr as jolt_core::field::JoltField>::Challenge],
    ) -> Fr {
        let ar = Self::eval_mle(self.a.clone(), r);
        let br = Self::eval_mle(self.b.clone(), r);
        let cr = Self::eval_mle(self.c.clone(), r);
        ar * br * cr
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fr> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings(
        &self,
        _accumulator: &mut Rep3OpeningAccumulator<Fr>,
        _transcript: &mut KeccakTranscript,
        _opening_point: OpeningPoint<BIG_ENDIAN, Fr>,
        claims: Vec<Fr>,
    ) {
        assert!(claims.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn rep3_batched_sumcheck_mixed_degree_front_loaded_correct() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE);

    let lin_len = 4; // 2 rounds
    let cub_len = 16; // 4 rounds

    let lin_coeffs: Vec<Fr> = (0..lin_len).map(|_| rng.gen()).collect();
    let a: Vec<Fr> = (0..cub_len).map(|_| rng.gen()).collect();
    let b: Vec<Fr> = (0..cub_len).map(|_| rng.gen()).collect();
    let c: Vec<Fr> = (0..cub_len).map(|_| rng.gen()).collect();

    let worker_input = (lin_coeffs.clone(), a.clone(), b.clone(), c.clone());
    let coordinator_input = (lin_coeffs, a, b, c);

    let (_worker_results, _coordinator_result) = run_rep3_local_test_with_coordinator(
        0,
        |_party| worker_input.clone(),
        || coordinator_input,
        |(lin_coeffs, a, b, c), mut io_ctx: IoContextPool<_>| {
            let party_id = io_ctx.party_id();

            let mut instances: Vec<Box<dyn Rep3SumcheckInstanceWorker<Fr>>> = vec![
                Box::new(ToyLinearWorker::new(party_id, lin_coeffs)),
                Box::new(ToyCubicProductWorker::new(party_id, a, b, c)),
            ];

            let mut accumulator = Rep3OpeningAccumulatorWorker::<Fr>::new(party_id);
            let _r = Rep3BatchedSumcheckWorker::prove(&mut instances, &mut accumulator, &mut io_ctx)?;
            Ok(())
        },
        |(lin_coeffs, a, b, c), network| {
            let instances: Vec<Box<dyn Rep3SumcheckInstance<Fr, KeccakTranscript>>> = vec![
                Box::new(ToyLinearCoordinator::new(lin_coeffs)),
                Box::new(ToyCubicProductCoordinator::new(a, b, c)),
            ];

            let mut transcript_prove = KeccakTranscript::new(b"rep3_batched_sumcheck");
            let mut transcript_verify = transcript_prove.clone();
            let mut accumulator = Rep3OpeningAccumulator::<Fr>::new();

            let (proof, r_prove) = Rep3BatchedSumcheck::prove(
                &instances,
                &mut accumulator,
                &mut transcript_prove,
                network,
            )?;

            let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
            let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();
            let batching_coeffs: Vec<Fr> = transcript_verify.challenge_vector(instances.len());

            let mut batched_input_claim = Fr::zero();
            for (instance, coeff) in instances.iter().zip(batching_coeffs.iter()) {
                let input = instance.input_claim_public();
                transcript_verify.append_scalar(&input);
                let scaled = input.mul_pow_2(max_num_rounds - instance.num_rounds());
                batched_input_claim += scaled * coeff;
            }

            let (output_claim, r_verify) = proof
                .verify(
                    batched_input_claim,
                    max_num_rounds,
                    max_degree,
                    &mut transcript_verify,
                )
                .unwrap();

            assert_eq!(r_prove, r_verify);

            let expected_output: Fr = instances
                .iter()
                .zip(batching_coeffs.iter())
                .map(|(instance, coeff)| {
                    let slice = &r_verify[max_num_rounds - instance.num_rounds()..];
                    instance.expected_output_claim(&accumulator, slice) * coeff
                })
                .sum();

            assert_eq!(output_claim, expected_output);
            Ok(())
        },
    );
}

#[test]
fn rep3_batched_sumcheck_degree3_smoke_correct() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xBADA55);
    let len = 8; // 3 rounds

    let a: Vec<Fr> = (0..len).map(|_| rng.gen()).collect();
    let b: Vec<Fr> = (0..len).map(|_| rng.gen()).collect();
    let c: Vec<Fr> = (0..len).map(|_| rng.gen()).collect();

    let worker_input = (a.clone(), b.clone(), c.clone());
    let coordinator_input = (a, b, c);

    let (_worker_results, _coordinator_result) = run_rep3_local_test_with_coordinator(
        0,
        |_party| worker_input.clone(),
        || coordinator_input,
        |(a, b, c), mut io_ctx: IoContextPool<_>| {
            let party_id = io_ctx.party_id();
            let mut instances: Vec<Box<dyn Rep3SumcheckInstanceWorker<Fr>>> =
                vec![Box::new(ToyCubicProductWorker::new(party_id, a, b, c))];

            let mut accumulator = Rep3OpeningAccumulatorWorker::<Fr>::new(party_id);
            let _r = Rep3BatchedSumcheckWorker::prove(&mut instances, &mut accumulator, &mut io_ctx)?;
            Ok(())
        },
        |(a, b, c), network| {
            let instances: Vec<Box<dyn Rep3SumcheckInstance<Fr, KeccakTranscript>>> =
                vec![Box::new(ToyCubicProductCoordinator::new(a, b, c))];

            let mut transcript_prove = KeccakTranscript::new(b"rep3_batched_sumcheck");
            let mut transcript_verify = transcript_prove.clone();
            let mut accumulator = Rep3OpeningAccumulator::<Fr>::new();

            let (proof, r_prove) = Rep3BatchedSumcheck::prove(
                &instances,
                &mut accumulator,
                &mut transcript_prove,
                network,
            )?;

            let max_num_rounds = instances[0].num_rounds();
            let max_degree = instances[0].degree();
            let batching_coeffs: Vec<Fr> = transcript_verify.challenge_vector(instances.len());
            assert_eq!(batching_coeffs.len(), 1);
            let coeff = batching_coeffs[0];

            let input = instances[0].input_claim_public();
            transcript_verify.append_scalar(&input);
            let batched_input_claim = input * coeff;

            let (output_claim, r_verify) = proof
                .verify(
                    batched_input_claim,
                    max_num_rounds,
                    max_degree,
                    &mut transcript_verify,
                )
                .unwrap();
            assert_eq!(r_prove, r_verify);

            let expected = instances[0].expected_output_claim(&accumulator, &r_verify) * coeff;
            assert_eq!(output_claim, expected);
            Ok(())
        },
    );
}
