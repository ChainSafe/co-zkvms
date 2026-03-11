use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::AdditiveShare;

use crate::poly::opening_proof::Rep3OpeningAccumulator;
use jolt_core::field::JoltField;

use crate::zkvm::dag::stage::Rep3SumcheckInstance;

const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3BooleanitySumcheck<F: JoltField> {
    gamma: [F; D],
    r_address: Vec<F::Challenge>,
    log_T: usize,
}

impl<F: JoltField> Rep3BooleanitySumcheck<F> {
    pub fn new<T: Transcript>(transcript: &mut T, log_T: usize) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let r_address: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(LOG_K_CHUNK);

        Self {
            gamma: gamma_powers,
            r_address,
            log_T,
        }
    }

    /// Return gamma powers so the worker can use them.
    pub fn gamma(&self) -> [F; D] {
        self.gamma
    }

    /// Return r_address so the worker can use them.
    pub fn r_address(&self) -> &[F::Challenge] {
        &self.r_address
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK + self.log_T
    }

    fn input_claim_public(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r_prime: &[F::Challenge],
    ) -> F {
        let ra_claims = (0..D).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionBooleanity,
                )
                .1
        });
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();

        EqPolynomial::<F>::mle(
            r_prime,
            &self
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(r_cycle.iter().cloned().rev())
                .collect::<Vec<F::Challenge>>(),
        ) * self
            .gamma
            .iter()
            .zip(ra_claims)
            .fold(F::zero(), |acc, (gamma, ra)| {
                (ra.square() - ra) * gamma + acc
            })
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address, r_cycle) = opening_point.split_at(LOG_K_CHUNK);
        let mut r_big_endian: Vec<F::Challenge> = r_address.iter().rev().copied().collect();
        r_big_endian.extend(r_cycle.iter().copied().rev());
        OpeningPoint::new(r_big_endian)
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        accumulator.append_sparse(
            transcript,
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionBooleanity,
            &r_sumcheck.r[..LOG_K_CHUNK],
            &r_sumcheck.r[LOG_K_CHUNK..],
            claims,
        );
    }
}

pub(crate) fn extend_degree_3_evals<F: JoltField>(
    previous_claim: AdditiveShare<F>,
    base: &[AdditiveShare<F>],
    max_degree: usize,
) -> Vec<AdditiveShare<F>> {
    debug_assert_eq!(base.len(), DEGREE);
    debug_assert!(max_degree >= DEGREE);

    if max_degree == DEGREE {
        return base.to_vec();
    }

    // Nodes for degree-3 polynomial at x=0..3.
    let y0 = base[0];
    let y1 = previous_claim - y0;
    let y2 = base[1]; // eval at 2
    let y3 = base[2]; // eval at 3

    let mut evals = vec![AdditiveShare::<F>::zero(); max_degree];
    evals[0] = y0;
    evals[1] = y2;
    evals[2] = y3;

    // Evaluate at x = 4..=max_degree via Lagrange on nodes 0..3.
    for x in 4..=max_degree {
        let xf = F::from(x as u64);
        let coeffs = lagrange_coeffs_consecutive_3::<F>(xf);
        evals[x - 1] = y0 * coeffs[0] + y1 * coeffs[1] + y2 * coeffs[2] + y3 * coeffs[3];
    }

    evals
}

fn lagrange_coeffs_consecutive_3<F: JoltField>(x: F) -> [F; 4] {
    // degree=3 nodes {0,1,2,3}. Precompute denominators and compute numerators on the fly.
    // denom(k) = Π_{m!=k} (k - m).
    let den0 = (F::from(0u64) - F::from(1u64))
        * (F::from(0u64) - F::from(2u64))
        * (F::from(0u64) - F::from(3u64));
    let den1 = (F::from(1u64) - F::from(0u64))
        * (F::from(1u64) - F::from(2u64))
        * (F::from(1u64) - F::from(3u64));
    let den2 = (F::from(2u64) - F::from(0u64))
        * (F::from(2u64) - F::from(1u64))
        * (F::from(2u64) - F::from(3u64));
    let den3 = (F::from(3u64) - F::from(0u64))
        * (F::from(3u64) - F::from(1u64))
        * (F::from(3u64) - F::from(2u64));

    let num0 = (x - F::from(1u64)) * (x - F::from(2u64)) * (x - F::from(3u64));
    let num1 = (x - F::from(0u64)) * (x - F::from(2u64)) * (x - F::from(3u64));
    let num2 = (x - F::from(0u64)) * (x - F::from(1u64)) * (x - F::from(3u64));
    let num3 = (x - F::from(0u64)) * (x - F::from(1u64)) * (x - F::from(2u64));

    [
        num0 * den0.inverse().unwrap(),
        num1 * den1.inverse().unwrap(),
        num2 * den2.inverse().unwrap(),
        num3 * den3.inverse().unwrap(),
    ]
}
