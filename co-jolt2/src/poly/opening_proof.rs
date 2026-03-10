use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use rayon::prelude::*;

use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use jolt_core::poly::opening_proof::{OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;

use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::multilinear_polynomial::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomialProverOpening;
use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::subprotocols::sumcheck::Rep3SumcheckInstanceWorker;
use crate::utils::types::{MaybeShared, Rep3Value};

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

    /// Append sparse polynomial openings for **public** one-hot polynomials.
    /// Accepts plain `F` claims (no promote). Skips `self.openings` storage
    /// because worker-side callers never read committed polynomial claims for
    /// public one-hots (all `get_committed_polynomial_opening` calls for
    /// BytecodeRa/RamRa are coordinator-side only).
    pub fn append_sparse_public(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: &[F::Challenge],
        r_cycle: &[F::Challenge],
        claims: Vec<F>,
    ) {
        assert_eq!(polynomials.len(), claims.len());
        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            self.sumchecks.push(
                Rep3OpeningProofReductionSumcheck::new_prover_instance_one_hot_public(
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
    /// 2. Delegate batched sumcheck to `Rep3BatchedSumcheckWorker::prove`
    /// 3. Receive gamma for joint polynomial RLC
    /// 4. Build joint polynomial and combined hint, call PCS::prove_rep3
    #[tracing::instrument(skip_all, name = "OpeningAcc::reduce_and_prove")]
    pub fn reduce_and_prove<PCS, ProofTranscript, N>(
        &mut self,
        polynomials: &HashMap<CommittedPolynomial, Arc<Rep3MultilinearPolynomial<F>>>,
        mut opening_hints: HashMap<CommittedPolynomial, MaybeShared<PCS::OpeningProofHint>>,
        pcs_setup: &PCS::ProverSetup,
        io_ctx: &mut mpc_core::protocols::rep3::network::IoContextPool<N>,
    ) -> eyre::Result<()>
    where
        PCS: crate::poly::commitment::Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        N: mpc_core::protocols::rep3::network::Rep3NetworkWorker,
    {
        // a. Receive gammas from coordinator
        let all_gammas: Vec<F> = io_ctx.network().receive_request()?;

        let _span = tracing::info_span!("prepare_sumchecks").entered();
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

        self.sumchecks
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, sumcheck)| {
                let offset = gamma_offsets[idx];
                let num_gammas = gamma_offsets[idx + 1] - offset;
                sumcheck.prepare_sumcheck(polynomials, &all_gammas[offset..offset + num_gammas]);
            });

        // c. Save rlc_coeffs + polynomials before draining into boxed instances
        let saved_meta: Vec<(Vec<F>, Vec<CommittedPolynomial>)> = self
            .sumchecks
            .iter()
            .map(|s| (s.rlc_coeffs.clone(), s.polynomials.clone()))
            .collect();
        let num_sumchecks = self.sumchecks.len();

        // d. Drain sumchecks into boxed trait objects
        let mut instances: Vec<Box<dyn Rep3SumcheckInstanceWorker<F, N>>> = self
            .sumchecks
            .drain(..)
            .map(|s| Box::new(s) as Box<dyn Rep3SumcheckInstanceWorker<F, N>>)
            .collect();
        drop(_span);

        // e. Run batched sumcheck
        let mut preproc = PreprocessingPool::empty(self.party_id);
        let r_sumcheck = crate::subprotocols::sumcheck::Rep3BatchedSumcheckWorker::prove(
            &mut instances,
            self,
            io_ctx,
            &mut preproc,
        )?;

        let _span = tracing::info_span!("rlc_and_hints").entered();

        // f. Receive gamma for joint poly RLC from coordinator
        let gamma: F = io_ctx.network().receive_request()?;
        let mut gamma_powers = vec![F::one()];
        for i in 1..num_sumchecks {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        // g. Compute per-polynomial RLC coefficients (using saved metadata)
        let mut rlc_map: BTreeMap<CommittedPolynomial, F> = BTreeMap::new();
        for (gamma_power, (rlc_coeffs, poly_labels)) in gamma_powers.iter().zip(saved_meta.iter()) {
            for (coeff, polynomial) in rlc_coeffs.iter().zip(poly_labels.iter()) {
                *rlc_map.entry(*polynomial).or_insert(F::zero()) += *coeff * gamma_power;
            }
        }

        // h. Combine hints using PCS::combine_hints_rep3
        // Use Public(None) as default for polynomials without hints (e.g. public
        // polynomials on non-ID0 workers).
        let (coeffs_for_hints, hints): (Vec<F>, Vec<MaybeShared<PCS::OpeningProofHint>>) = rlc_map
            .iter()
            .map(|(k, v)| {
                (
                    *v,
                    opening_hints.remove(k).unwrap_or(MaybeShared::Public(None)),
                )
            })
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
        drop(_span);

        // j. Call PCS::prove_rep3 with joint polynomial and pre-combined hint
        PCS::prove_rep3(
            &joint_poly,
            pcs_setup,
            &r_sumcheck,
            Some(combined_hint),
            io_ctx.network(),
        )?;

        Ok(())
    }
}

impl<F: JoltField> Default for Rep3OpeningAccumulatorWorker<F> {
    fn default() -> Self {
        Self::new(PartyID::ID0)
    }
}

// ---------------------------------------------------------------------------
// Opening proof reduction types (worker-side)
// ---------------------------------------------------------------------------

/// Worker-side dense polynomial prover opening.
/// Mirrors vanilla `DensePolynomialProverOpening`, adapted for MPC.
pub struct Rep3DensePolynomialProverOpening<F: JoltField> {
    /// The secret-shared (dense) polynomial being opened, possibly an RLC of multiple shared polys.
    /// `None` until `prepare_sumcheck`.
    pub shared_polynomial: Option<Rep3DensePolynomial<F>>,
    /// Public dense polynomials participating in this opening, stored in the vanilla representation
    /// to avoid per-coefficient `promote_to_trivial_share` overhead. Each entry is (gamma, poly).
    pub public_polynomials: Vec<(F, MultilinearPolynomial<F>)>,
    /// Public eq polynomial: EQ(opening_point, ·). Coefficients stored in `Z`.
    pub eq_poly: DensePolynomial<F>,
    pub party_id: PartyID,
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
        let mle_half = if let Some(poly) = self.shared_polynomial.as_ref() {
            poly.len() / 2
        } else if let Some((_, poly)) = self.public_polynomials.first() {
            poly.len() / 2
        } else {
            unreachable!("dense prover opening has no polynomials");
        };

        let shared = self.shared_polynomial.as_ref();
        let public_polys = &self.public_polynomials;

        let (eval_0_shared, eval_2_shared, eval_0_public, eval_2_public) = (0..mle_half)
            .into_par_iter()
            .map(|j| {
                let eq_j = self.eq_poly.Z[j];
                let eq_j_half = self.eq_poly.Z[j + mle_half];
                let eq_2 = (eq_j_half + eq_j_half) - eq_j;

                let (mut e0_shared, mut e2_shared) = (AdditiveShare::zero(), AdditiveShare::zero());
                if let Some(shared) = shared {
                    let poly_j = shared.get_bound_coeff(j);
                    e0_shared = (poly_j * eq_j).into_additive();

                    let poly_j_half = shared.get_bound_coeff(j + mle_half);
                    let poly_2 = (poly_j_half + poly_j_half) - poly_j;
                    e2_shared = (poly_2 * eq_2).into_additive();
                }

                let mut e0_public = F::zero();
                let mut e2_public = F::zero();
                for (gamma, poly) in public_polys.iter() {
                    let poly_j = poly.get_bound_coeff(j);
                    e0_public += *gamma * poly_j * eq_j;

                    let poly_j_half = poly.get_bound_coeff(j + mle_half);
                    let poly_2 = (poly_j_half + poly_j_half) - poly_j;
                    e2_public += *gamma * poly_2 * eq_2;
                }

                (e0_shared, e2_shared, e0_public, e2_public)
            })
            .reduce(
                || {
                    (
                        AdditiveShare::zero(),
                        AdditiveShare::zero(),
                        F::zero(),
                        F::zero(),
                    )
                },
                |(a0, a2, ap0, ap2), (b0, b2, bp0, bp2)| (a0 + b0, a2 + b2, ap0 + bp0, ap2 + bp2),
            );

        let eval_0 = eval_0_shared
            + mpc_core::protocols::additive::promote_to_trivial_share(eval_0_public, self.party_id);
        let eval_2 = eval_2_shared
            + mpc_core::protocols::additive::promote_to_trivial_share(eval_2_public, self.party_id);

        [eval_0, eval_2]
    }

    pub fn bind(&mut self, r_j: F::Challenge) {
        self.eq_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        if let Some(shared) = self.shared_polynomial.as_mut() {
            shared.bind(r_j.into(), BindingOrder::HighToLow);
        }
        self.public_polynomials
            .iter_mut()
            .for_each(|(_gamma, poly)| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    pub fn final_sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        let mut acc = Rep3PrimeFieldShare::zero_share();
        if let Some(shared) = self.shared_polynomial.as_ref() {
            acc += shared.final_sumcheck_claim();
        }
        if !self.public_polynomials.is_empty() {
            let public_claim: F = self
                .public_polynomials
                .iter()
                .map(|(gamma, poly)| *gamma * poly.final_sumcheck_claim())
                .sum();
            acc += rep3_arith::promote_to_trivial_share(self.party_id, public_claim);
        }
        acc
    }
}

/// Worker-side prover opening enum. Dispatches to dense, shared one-hot,
/// or public one-hot variant.
pub enum Rep3ProverOpening<F: JoltField> {
    Dense(Rep3DensePolynomialProverOpening<F>),
    OneHot(Rep3OneHotPolynomialProverOpening<F>),
    /// Public one-hot polynomial — wraps vanilla `OneHotPolynomialProverOpening`
    /// to avoid O(K×T) dense expansion. Uses the efficient O(K+T) algorithm,
    /// promoting only the 2 per-round scalars to trivial additive shares.
    PublicOneHot(
        jolt_core::poly::one_hot_polynomial::OneHotPolynomialProverOpening<F>,
        PartyID,
    ),
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
            Rep3ProverOpening::PublicOneHot(vanilla, party_id) => {
                // Reconstruct the previous_claim as a plain F for vanilla.
                // In the trivial-share convention, only ID0 holds the value.
                let prev_f = previous_claim.into_fe();
                let vanilla_evals = vanilla.compute_prover_message(round, prev_f);
                // Promote the 2 result scalars to trivial additive shares.
                [
                    mpc_core::protocols::additive::promote_to_trivial_share(
                        vanilla_evals[0],
                        *party_id,
                    ),
                    mpc_core::protocols::additive::promote_to_trivial_share(
                        vanilla_evals[1],
                        *party_id,
                    ),
                ]
            }
        }
    }

    pub fn bind(&mut self, r_j: F::Challenge, round: usize) {
        match self {
            Rep3ProverOpening::Dense(d) => d.bind(r_j),
            Rep3ProverOpening::OneHot(oh) => oh.bind(r_j, round),
            Rep3ProverOpening::PublicOneHot(vanilla, _) => vanilla.bind(r_j, round),
        }
    }

    pub fn final_sumcheck_claim(&self) -> Rep3PrimeFieldShare<F> {
        match self {
            Rep3ProverOpening::Dense(d) => d.final_sumcheck_claim(),
            Rep3ProverOpening::OneHot(oh) => oh.final_sumcheck_claim(),
            Rep3ProverOpening::PublicOneHot(vanilla, party_id) => {
                rep3_arith::promote_to_trivial_share(*party_id, vanilla.final_sumcheck_claim())
            }
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
    input_claims: Vec<Rep3Value<F>>,
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
            input_claims: claims.into_iter().map(Rep3Value::Shared).collect(),
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
            input_claims: vec![Rep3Value::Shared(claim)],
            opening_point,
            party_id,
            sumcheck_claim: None,
        }
    }

    pub fn new_prover_instance_one_hot_public(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        r_address: &[F::Challenge],
        r_cycle: &[F::Challenge],
        claim: F,
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
            input_claims: vec![Rep3Value::Public(claim)],
            opening_point,
            party_id,
            sumcheck_claim: None,
        }
    }

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
                    // Reduce claims (Dense claims are always Shared)
                    let reduced_claim: Rep3PrimeFieldShare<F> = self
                        .rlc_coeffs
                        .iter()
                        .zip(self.input_claims.iter())
                        .map(|(gamma, claim)| match claim {
                            Rep3Value::Shared(s) => *s * *gamma,
                            _ => unreachable!("Dense claims are always Shared"),
                        })
                        .sum();
                    self.input_claims = vec![Rep3Value::Shared(reduced_claim)];
                }

                let mut shared_terms: Vec<(F, Rep3DensePolynomial<F>)> = Vec::new();
                let mut public_terms: Vec<(F, MultilinearPolynomial<F>)> = Vec::new();
                for (gamma, label) in self.rlc_coeffs.iter().zip(self.polynomials.iter()) {
                    let poly = polynomials_map.get(label).unwrap();
                    match poly.as_ref() {
                        Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(d)) => {
                            shared_terms.push((*gamma, d.clone()))
                        }
                        Rep3MultilinearPolynomial::Public(mlp) => {
                            assert!(
                                !matches!(mlp, MultilinearPolynomial::OneHot(_)),
                                "dense opening reduction received public one-hot {label:?}"
                            );
                            public_terms.push((*gamma, mlp.clone()));
                        }
                        _ => panic!(
                            "dense opening reduction received non-dense poly {label:?}: {:?}",
                            std::mem::discriminant(poly.as_ref())
                        ),
                    }
                }

                let shared_polynomial: Option<Rep3DensePolynomial<F>> = if shared_terms.is_empty() {
                    None
                } else if shared_terms.len() == 1 && shared_terms[0].0 == F::one() {
                    Some(shared_terms.pop().unwrap().1)
                } else {
                    let num_coeffs = shared_terms[0].1.len();
                    let combined: Vec<Rep3PrimeFieldShare<F>> = (0..num_coeffs)
                        .into_par_iter()
                        .map(|i| {
                            shared_terms
                                .iter()
                                .map(|(gamma, poly)| poly.get_bound_coeff(i) * *gamma)
                                .sum()
                        })
                        .collect();
                    Some(Rep3DensePolynomial::new(combined))
                };

                // Combine multiple public polynomials into a single public dense polynomial.
                // This mirrors vanilla behavior and avoids retaining many large public polynomials
                // in memory across all sumcheck rounds.
                let public_polynomials: Vec<(F, MultilinearPolynomial<F>)> =
                    if public_terms.len() <= 1 {
                        public_terms
                    } else {
                        let (coeffs, polys): (Vec<F>, Vec<MultilinearPolynomial<F>>) =
                            public_terms.into_iter().unzip();
                        let refs: Vec<&MultilinearPolynomial<F>> = polys.iter().collect();
                        let combined = DensePolynomial::linear_combination(&refs, &coeffs);
                        vec![(F::one(), MultilinearPolynomial::LargeScalars(combined))]
                    };

                self.prover_state =
                    Some(Rep3ProverOpening::Dense(Rep3DensePolynomialProverOpening {
                        shared_polynomial,
                        public_polynomials,
                        eq_poly,
                        party_id: self.party_id,
                    }));
            }
            OpeningKind::OneHot { address_len } => {
                assert_eq!(gammas.len(), 1);
                assert_eq!(self.polynomials.len(), 1);

                let poly = polynomials_map.get(&self.polynomials[0]).unwrap();
                match poly.as_ref() {
                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(oh)) => {
                        let (r_address, r_cycle) = self.opening_point.split_at(*address_len);
                        // Use clone_with_fresh_h so this prover opening gets its own
                        // independent H. Multiple reduction entries for the same
                        // CommittedPolynomial would otherwise share the same
                        // Arc<RwLock<H>>, causing each entry's bind() to mutate the
                        // shared H and advance it too many rounds.
                        let oh_opening = Rep3OneHotPolynomialProverOpening::new(
                            oh.clone_with_fresh_h(),
                            r_address,
                            r_cycle,
                            self.party_id,
                        );
                        self.prover_state = Some(Rep3ProverOpening::OneHot(oh_opening));
                    }
                    Rep3MultilinearPolynomial::Public(
                        jolt_core::poly::multilinear_polynomial::MultilinearPolynomial::OneHot(oh),
                    ) => {
                        // Public one-hot polynomial (e.g., BytecodeRa, RamRa):
                        // Use vanilla OneHotPolynomialProverOpening for O(K+T)
                        // per round instead of dense-expanding to O(K×T).
                        use jolt_core::poly::one_hot_polynomial::{
                            EqAddressState, EqCycleState, OneHotPolynomialProverOpening,
                        };
                        use std::sync::{Arc, RwLock};

                        let (r_address, r_cycle) = self.opening_point.split_at(*address_len);
                        let eq_address = Arc::new(RwLock::new(EqAddressState::new(r_address)));
                        let eq_cycle = Arc::new(RwLock::new(EqCycleState::new(r_cycle)));
                        eq_cycle.write().unwrap().merge_D();

                        let mut vanilla_opening =
                            OneHotPolynomialProverOpening::new(eq_address, eq_cycle);
                        vanilla_opening.initialize(oh.clone());
                        self.prover_state = Some(Rep3ProverOpening::PublicOneHot(
                            vanilla_opening,
                            self.party_id,
                        ));
                    }
                    _ => panic!(
                        "Expected shared or public one-hot polynomial for {:?}",
                        self.polynomials[0]
                    ),
                }
            }
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    pub fn input_claim_value(&self) -> Rep3Value<F> {
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

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3OpeningProofReductionSumcheck<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    fn input_claim(&self) -> Rep3Value<F> {
        self.input_claim_value()
    }

    fn compute_prover_message_share(
        &mut self,
        round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        let [e0, e2] = self.compute_prover_message(round, previous_claim);
        let mut result = vec![AdditiveShare::zero(); max_degree];
        result[0] = e0;
        if max_degree > 1 {
            result[1] = e2;
        }
        result
    }

    fn bind(
        &mut self,
        r_j: F::Challenge,
        round: usize,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) {
        self.prover_state.as_mut().unwrap().bind(r_j, round);
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_worker(
        &mut self,
        _accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        self.cache_sumcheck_claim();
        vec![self.sumcheck_claim()]
    }
}
