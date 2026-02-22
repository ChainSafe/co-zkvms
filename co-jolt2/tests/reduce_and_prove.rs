use std::collections::HashMap;
use std::sync::Arc;

use ark_bn254::Fr;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use co_jolt2::poly::multilinear_polynomial::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use co_jolt2::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use co_jolt2::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use co_jolt2::utils::test_utils::run_rep3_local_test_with_coordinator;
use co_jolt2::utils::types::MaybeShared;

use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::poly::opening_proof::{
    OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
    BIG_ENDIAN,
};
use jolt_core::transcripts::{Blake2bTranscript, Transcript};
use jolt_core::zkvm::witness::CommittedPolynomial;

use mpc_core::protocols::rep3::network::IoContextPool;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;
type Challenge = <F as JoltField>::Challenge;

/// Worker input: everything a worker needs for reduce_and_prove.
#[derive(Clone)]
struct WorkerInput {
    polys: HashMap<CommittedPolynomial, Arc<Rep3MultilinearPolynomial<F>>>,
    hints: HashMap<CommittedPolynomial, MaybeShared<<PCS as CommitmentScheme>::OpeningProofHint>>,
    claims: Vec<Rep3PrimeFieldShare<F>>,
    opening_point: Vec<Challenge>,
    setup: <PCS as CommitmentScheme>::ProverSetup,
    labels: Vec<CommittedPolynomial>,
    sumcheck_id: SumcheckId,
}

/// Helper: share a field element into 3 Rep3PrimeFieldShares.
fn share_field<R: Rng>(val: F, rng: &mut R) -> [Rep3PrimeFieldShare<F>; 3] {
    mpc_core::protocols::rep3::arithmetic::generate_shares_rep3(val, rng)
        .try_into()
        .unwrap()
}

/// Run both dense-only and mixed (dense+one-hot) reduce_and_prove tests
/// under a single DoryGlobals initialization (since DoryGlobals is process-wide).
#[test]
fn reduce_and_prove_correct() {
    // DoryGlobals: K=2, T=16 → 8×8 matrix (sigma=3).
    // Dense test: 16 coefficients → 2 rows in 8 columns, fits.
    // Mixed test: one-hot K=2, T=16 = 32 entries → 4 rows in 8 columns, fits.
    // setup_prover(5) → n=8 bases matches 8 columns.
    let _dory_guard = DoryGlobals::initialize(2, 16);

    eprintln!(
        "DoryGlobals: T={}, num_columns={}, max_num_rows={}",
        DoryGlobals::get_T(),
        DoryGlobals::get_num_columns(),
        DoryGlobals::get_max_num_rows(),
    );
    eprintln!("=== Starting dense-only test ===");
    test_dense_only();
    eprintln!("=== Dense-only test PASSED ===");
    eprintln!("=== Starting mixed test ===");
    test_mixed_dense_one_hot();
    eprintln!("=== Mixed test PASSED ===");
}

fn test_dense_only() {
    let num_vars = 4;
    let num_coeffs = 1usize << num_vars;
    let mut rng = ChaCha20Rng::seed_from_u64(0xDEADBEEF);

    // Use max_num_vars=5 (shared with mixed test for the one-hot polynomial).
    let prover_setup = PCS::setup_prover(5);
    let verifier_setup = PCS::setup_verifier(&prover_setup);

    // --- Create two random polynomials ---
    let poly_a_evals: Vec<F> = (0..num_coeffs).map(|_| rng.gen()).collect();
    let poly_b_evals: Vec<F> = (0..num_coeffs).map(|_| rng.gen()).collect();

    let vanilla_poly_a = MultilinearPolynomial::from(poly_a_evals.clone());
    let vanilla_poly_b = MultilinearPolynomial::from(poly_b_evals.clone());

    let label_a = CommittedPolynomial::LeftInstructionInput;
    let label_b = CommittedPolynomial::RightInstructionInput;
    let labels = vec![label_a, label_b];
    let sumcheck_id = SumcheckId::SpartanOuter;

    // Commit vanilla-side
    let (commitment_a, hint_a) = PCS::commit(&vanilla_poly_a, &prover_setup);
    let (commitment_b, hint_b) = PCS::commit(&vanilla_poly_b, &prover_setup);

    // Generate opening point from a test transcript (to get proper Challenge type)
    let opening_point: Vec<Challenge> = {
        let mut t = FS::new(b"test_opening_point");
        t.challenge_vector_optimized::<F>(num_vars)
    };
    let claim_a: F = vanilla_poly_a.evaluate(&opening_point);
    let claim_b: F = vanilla_poly_b.evaluate(&opening_point);

    // === Part 1: Vanilla prover + verifier (sanity check) ===
    let vanilla_sumcheck_claims = {
        let mut prover_acc = ProverOpeningAccumulator::<F>::new();
        let mut prover_transcript = FS::new(b"reduce_and_prove_test");
        prover_acc.append_dense(
            &mut prover_transcript,
            labels.clone(),
            sumcheck_id,
            opening_point.clone(),
            &[claim_a, claim_b],
        );

        let mut poly_map = HashMap::new();
        poly_map.insert(label_a, vanilla_poly_a.clone());
        poly_map.insert(label_b, vanilla_poly_b.clone());

        let mut hint_map = HashMap::new();
        hint_map.insert(label_a, hint_a.clone());
        hint_map.insert(label_b, hint_b.clone());

        let proof = prover_acc.reduce_and_prove::<FS, PCS>(
            poly_map,
            hint_map,
            &prover_setup,
            &mut prover_transcript,
        );

        // Verify vanilla proof
        let mut verifier_acc = VerifierOpeningAccumulator::<F>::new();
        verifier_acc.openings.insert(
            OpeningId::Committed(label_a, sumcheck_id),
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim_a,
            ),
        );
        verifier_acc.openings.insert(
            OpeningId::Committed(label_b, sumcheck_id),
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim_b,
            ),
        );
        let mut verifier_transcript = FS::new(b"reduce_and_prove_test");
        verifier_acc.append_dense(
            &mut verifier_transcript,
            labels.clone(),
            sumcheck_id,
            opening_point.clone(),
        );

        let mut commitment_map = HashMap::new();
        commitment_map.insert(label_a, commitment_a.clone());
        commitment_map.insert(label_b, commitment_b.clone());

        verifier_acc
            .reduce_and_verify::<FS, PCS>(
                &verifier_setup,
                &mut commitment_map,
                &proof,
                &mut verifier_transcript,
            )
            .expect("Vanilla proof should verify (sanity check)");

        proof.sumcheck_claims.clone()
    };

    // === Part 2: MPC reduce_and_prove ===

    // Create Rep3 shares of polynomials
    let [shares_a_0, shares_a_1, shares_a_2] =
        Rep3MultilinearPolynomial::generate_shares_from_coeffs(&poly_a_evals, &mut rng);
    let [shares_b_0, shares_b_1, shares_b_2] =
        Rep3MultilinearPolynomial::generate_shares_from_coeffs(&poly_b_evals, &mut rng);

    // Commit MPC-side
    let party_shares_a = [shares_a_0, shares_a_1, shares_a_2];
    let party_shares_b = [shares_b_0, shares_b_1, shares_b_2];

    let mut party_hints_a = Vec::new();
    let mut party_hints_b = Vec::new();

    for i in 0..3 {
        let (_, ha) = <PCS as co_jolt2::poly::commitment::Rep3CommitmentScheme<F, FS>>::commit_rep3(
            &party_shares_a[i],
            &prover_setup,
            true,
        );
        let (_, hb) = <PCS as co_jolt2::poly::commitment::Rep3CommitmentScheme<F, FS>>::commit_rep3(
            &party_shares_b[i],
            &prover_setup,
            true,
        );
        party_hints_a.push(ha);
        party_hints_b.push(hb);
    }

    // Share the claims
    let claim_a_shares =
        mpc_core::protocols::rep3::arithmetic::generate_shares_rep3(claim_a, &mut rng);
    let claim_b_shares =
        mpc_core::protocols::rep3::arithmetic::generate_shares_rep3(claim_b, &mut rng);

    // Build per-party worker inputs
    let worker_inputs: Vec<WorkerInput> = (0..3)
        .map(|i| {
            let mut polys = HashMap::new();
            polys.insert(label_a, Arc::new(party_shares_a[i].clone()));
            polys.insert(label_b, Arc::new(party_shares_b[i].clone()));

            let mut hints = HashMap::new();
            hints.insert(label_a, party_hints_a[i].clone());
            hints.insert(label_b, party_hints_b[i].clone());

            WorkerInput {
                polys,
                hints,
                claims: vec![claim_a_shares[i], claim_b_shares[i]],
                opening_point: opening_point.clone(),
                setup: prover_setup.clone(),
                labels: labels.clone(),
                sumcheck_id,
            }
        })
        .collect();

    #[derive(Clone)]
    struct CoordinatorInput {
        claims: Vec<F>,
        opening_point: Vec<Challenge>,
        setup: <PCS as CommitmentScheme>::ProverSetup,
        commitments: HashMap<CommittedPolynomial, <PCS as CommitmentScheme>::Commitment>,
        labels: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
    }

    let coordinator_input = CoordinatorInput {
        claims: vec![claim_a, claim_b],
        opening_point: opening_point.clone(),
        setup: prover_setup.clone(),
        commitments: {
            let mut m = HashMap::new();
            m.insert(label_a, commitment_a.clone());
            m.insert(label_b, commitment_b.clone());
            m
        },
        labels: labels.clone(),
        sumcheck_id,
    };

    let (_worker_results, mpc_sumcheck_claims) = run_rep3_local_test_with_coordinator(
        0,
        move |party_idx| worker_inputs[party_idx].clone(),
        move || coordinator_input,
        // Worker
        |input: WorkerInput, mut io_ctx: IoContextPool<_>| {
            let party_id = io_ctx.party_id();
            let mut accumulator = Rep3OpeningAccumulatorWorker::<F>::new(party_id);

            accumulator.append_dense(
                input.labels,
                input.sumcheck_id,
                input.opening_point,
                &input.claims,
            );

            accumulator.reduce_and_prove::<PCS, FS, _>(
                &input.polys,
                input.hints,
                &input.setup,
                io_ctx.network(),
            )?;

            Ok(())
        },
        // Coordinator
        |input: CoordinatorInput, network| {
            let mut transcript = FS::new(b"reduce_and_prove_test");
            let mut accumulator = Rep3OpeningAccumulator::<F>::new();

            accumulator.append_dense(
                &mut transcript,
                input.labels,
                input.sumcheck_id,
                input.opening_point,
                input.claims,
            );

            let mut commitment_map = input.commitments;

            let proof = accumulator.reduce_and_prove::<PCS, FS, _>(
                &mut commitment_map,
                &input.setup,
                &mut transcript,
                network,
            )?;

            Ok(proof.sumcheck_claims)
        },
    );

    // === Part 3: Compare sumcheck claims ===
    assert_eq!(
        vanilla_sumcheck_claims.len(),
        mpc_sumcheck_claims.len(),
        "Sumcheck claims count mismatch"
    );
    for (i, (vanilla, mpc)) in vanilla_sumcheck_claims
        .iter()
        .zip(mpc_sumcheck_claims.iter())
        .enumerate()
    {
        assert_eq!(
            vanilla, mpc,
            "Sumcheck claim {i} mismatch: vanilla={vanilla:?}, mpc={mpc:?}"
        );
    }
}

#[cfg(test)]
fn test_mixed_dense_one_hot() {
    let dense_num_vars = 3;
    let dense_num_coeffs = 1usize << dense_num_vars;
    let log_k = 1usize;
    let log_t = 4usize; // ≥3 so vanilla RaPolynomial reaches RoundN
    let K = 1usize << log_k; // 2
    let T = 1usize << log_t; // 16 (matches DoryGlobals::get_T())
    let oh_num_vars = log_k + log_t; // 5
    let max_num_vars = oh_num_vars.max(dense_num_vars);

    let mut rng = ChaCha20Rng::seed_from_u64(0xCAFEBABE);

    // Dory globals already initialized by the caller.
    let prover_setup = PCS::setup_prover(max_num_vars);
    let verifier_setup = PCS::setup_verifier(&prover_setup);

    // --- Labels ---
    let dense_label = CommittedPolynomial::LeftInstructionInput;
    let oh_label = CommittedPolynomial::RamRa(0);
    let dense_sumcheck_id = SumcheckId::SpartanOuter;
    let oh_sumcheck_id = SumcheckId::RamHammingWeight;

    // --- Dense polynomial ---
    let dense_evals: Vec<F> = (0..dense_num_coeffs).map(|_| rng.gen()).collect();
    let vanilla_dense = MultilinearPolynomial::from(dense_evals.clone());
    let (dense_commitment, dense_hint) = PCS::commit(&vanilla_dense, &prover_setup);

    // --- One-hot polynomial ---
    // Plaintext indices (some None = inactive cycles)
    let nonzero_indices: Vec<Option<u8>> = (0..T)
        .map(|i| {
            if i == 0 {
                None // first cycle inactive
            } else {
                Some((rng.next_u32() as u8) % (K as u8))
            }
        })
        .collect();

    let vanilla_oh = OneHotPolynomial::<F>::from_indices(nonzero_indices.clone(), K);
    let vanilla_oh_ml = MultilinearPolynomial::OneHot(vanilla_oh.clone());
    let (oh_commitment, oh_hint) = PCS::commit(&vanilla_oh_ml, &prover_setup);

    // Build Rep3 one-hot polynomials: replicated shares of the RandOHV E_field vector
    let r_mask: u8 = (rng.next_u32() as u8) % (K as u8);
    let masked_indices_c: Arc<Vec<Option<u8>>> = Arc::new(
        nonzero_indices
            .iter()
            .map(|opt| opt.map(|kj| kj ^ r_mask))
            .collect(),
    );

    // E_field[i] = share(if i == r_mask { 1 } else { 0 })
    let mut e_field_party: [Vec<Rep3PrimeFieldShare<F>>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(K));
    for i in 0..K {
        let bit: F = if i as u8 == r_mask {
            F::from_u64(1)
        } else {
            F::from_u64(0)
        };
        let shares = share_field(bit, &mut rng);
        for pid in 0..3 {
            e_field_party[pid].push(shares[pid]);
        }
    }
    let e_field_arcs: [Arc<Vec<Rep3PrimeFieldShare<F>>>; 3] =
        std::array::from_fn(|pid| Arc::new(e_field_party[pid].clone()));

    let rep3_oh_polys: [Rep3OneHotPolynomial<F>; 3] = std::array::from_fn(|pid| {
        Rep3OneHotPolynomial::from_parts(K, masked_indices_c.clone(), e_field_arcs[pid].clone())
    });

    // --- Opening points ---
    let dense_opening_point: Vec<Challenge> = {
        let mut t = FS::new(b"test_dense_point");
        t.challenge_vector_optimized::<F>(dense_num_vars)
    };
    let oh_r_address: Vec<Challenge> = {
        let mut t = FS::new(b"test_oh_address");
        t.challenge_vector_optimized::<F>(log_k)
    };
    let oh_r_cycle: Vec<Challenge> = {
        let mut t = FS::new(b"test_oh_cycle");
        t.challenge_vector_optimized::<F>(log_t)
    };
    let oh_opening_point: Vec<Challenge> = oh_r_address
        .iter()
        .chain(oh_r_cycle.iter())
        .copied()
        .collect();

    // --- Evaluation claims ---
    let dense_claim: F = vanilla_dense.evaluate(&dense_opening_point);
    let oh_claim: F = vanilla_oh.evaluate(&oh_opening_point);

    // === Part 1: Vanilla prover + verifier (sanity check) ===
    let vanilla_sumcheck_claims = {
        let mut prover_acc = ProverOpeningAccumulator::<F>::new();
        let mut prover_transcript = FS::new(b"reduce_and_prove_mixed_test");

        // Dense opening
        prover_acc.append_dense(
            &mut prover_transcript,
            vec![dense_label],
            dense_sumcheck_id,
            dense_opening_point.clone(),
            &[dense_claim],
        );
        // One-hot opening
        prover_acc.append_sparse(
            &mut prover_transcript,
            vec![oh_label],
            oh_sumcheck_id,
            oh_r_address.clone(),
            oh_r_cycle.clone(),
            vec![oh_claim],
        );

        let mut poly_map = HashMap::new();
        poly_map.insert(dense_label, vanilla_dense.clone());
        poly_map.insert(oh_label, vanilla_oh_ml.clone());

        let mut hint_map = HashMap::new();
        hint_map.insert(dense_label, dense_hint.clone());
        hint_map.insert(oh_label, oh_hint.clone());

        let proof = prover_acc.reduce_and_prove::<FS, PCS>(
            poly_map,
            hint_map,
            &prover_setup,
            &mut prover_transcript,
        );

        // Verify vanilla proof
        let mut verifier_acc = VerifierOpeningAccumulator::<F>::new();
        verifier_acc.openings.insert(
            OpeningId::Committed(dense_label, dense_sumcheck_id),
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(dense_opening_point.clone()),
                dense_claim,
            ),
        );
        verifier_acc.openings.insert(
            OpeningId::Committed(oh_label, oh_sumcheck_id),
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(oh_opening_point.clone()),
                oh_claim,
            ),
        );

        let mut verifier_transcript = FS::new(b"reduce_and_prove_mixed_test");
        verifier_acc.append_dense(
            &mut verifier_transcript,
            vec![dense_label],
            dense_sumcheck_id,
            dense_opening_point.clone(),
        );
        verifier_acc.append_sparse(
            &mut verifier_transcript,
            vec![oh_label],
            oh_sumcheck_id,
            oh_opening_point.clone(),
        );

        let mut commitment_map = HashMap::new();
        commitment_map.insert(dense_label, dense_commitment.clone());
        commitment_map.insert(oh_label, oh_commitment.clone());

        verifier_acc
            .reduce_and_verify::<FS, PCS>(
                &verifier_setup,
                &mut commitment_map,
                &proof,
                &mut verifier_transcript,
            )
            .expect("Vanilla mixed proof should verify");

        proof.sumcheck_claims.clone()
    };

    // === Part 2: MPC reduce_and_prove ===

    // Create Rep3 shares of dense polynomial
    let [dense_s0, dense_s1, dense_s2] =
        Rep3MultilinearPolynomial::generate_shares_from_coeffs(&dense_evals, &mut rng);
    let dense_party_shares = [dense_s0, dense_s1, dense_s2];

    // Wrap one-hot polynomials as Rep3MultilinearPolynomial
    let oh_party_mlps: [Rep3MultilinearPolynomial<F>; 3] = std::array::from_fn(|pid| {
        Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(rep3_oh_polys[pid].clone()))
    });

    // Commit MPC-side (dense)
    let mut dense_party_hints = Vec::new();
    for i in 0..3 {
        let (_, h) = <PCS as co_jolt2::poly::commitment::Rep3CommitmentScheme<F, FS>>::commit_rep3(
            &dense_party_shares[i],
            &prover_setup,
            true,
        );
        dense_party_hints.push(h);
    }

    // Commit MPC-side (one-hot)
    let mut oh_party_hints = Vec::new();
    for i in 0..3 {
        let (_, h) = <PCS as co_jolt2::poly::commitment::Rep3CommitmentScheme<F, FS>>::commit_rep3(
            &oh_party_mlps[i],
            &prover_setup,
            true,
        );
        oh_party_hints.push(h);
    }

    // Share evaluation claims
    let dense_claim_shares = share_field(dense_claim, &mut rng);
    let oh_claim_shares = share_field(oh_claim, &mut rng);

    // Build per-party worker inputs
    #[derive(Clone)]
    struct MixedWorkerInput {
        polys: HashMap<CommittedPolynomial, Arc<Rep3MultilinearPolynomial<F>>>,
        hints:
            HashMap<CommittedPolynomial, MaybeShared<<PCS as CommitmentScheme>::OpeningProofHint>>,
        dense_claim: Rep3PrimeFieldShare<F>,
        oh_claim: Rep3PrimeFieldShare<F>,
        dense_opening_point: Vec<Challenge>,
        oh_r_address: Vec<Challenge>,
        oh_r_cycle: Vec<Challenge>,
        setup: <PCS as CommitmentScheme>::ProverSetup,
        dense_label: CommittedPolynomial,
        dense_sumcheck_id: SumcheckId,
        oh_label: CommittedPolynomial,
        oh_sumcheck_id: SumcheckId,
    }

    let worker_inputs: Vec<MixedWorkerInput> = (0..3)
        .map(|i| {
            let mut polys = HashMap::new();
            polys.insert(dense_label, Arc::new(dense_party_shares[i].clone()));
            polys.insert(oh_label, Arc::new(oh_party_mlps[i].clone()));

            let mut hints = HashMap::new();
            hints.insert(dense_label, dense_party_hints[i].clone());
            hints.insert(oh_label, oh_party_hints[i].clone());

            MixedWorkerInput {
                polys,
                hints,
                dense_claim: dense_claim_shares[i],
                oh_claim: oh_claim_shares[i],
                dense_opening_point: dense_opening_point.clone(),
                oh_r_address: oh_r_address.clone(),
                oh_r_cycle: oh_r_cycle.clone(),
                setup: prover_setup.clone(),
                dense_label,
                dense_sumcheck_id,
                oh_label,
                oh_sumcheck_id,
            }
        })
        .collect();

    #[derive(Clone)]
    struct MixedCoordinatorInput {
        dense_claim: F,
        oh_claim: F,
        dense_opening_point: Vec<Challenge>,
        oh_r_address: Vec<Challenge>,
        oh_r_cycle: Vec<Challenge>,
        setup: <PCS as CommitmentScheme>::ProverSetup,
        commitments: HashMap<CommittedPolynomial, <PCS as CommitmentScheme>::Commitment>,
        dense_label: CommittedPolynomial,
        dense_sumcheck_id: SumcheckId,
        oh_label: CommittedPolynomial,
        oh_sumcheck_id: SumcheckId,
    }

    let coordinator_input = MixedCoordinatorInput {
        dense_claim,
        oh_claim,
        dense_opening_point: dense_opening_point.clone(),
        oh_r_address: oh_r_address.clone(),
        oh_r_cycle: oh_r_cycle.clone(),
        setup: prover_setup.clone(),
        commitments: {
            let mut m = HashMap::new();
            m.insert(dense_label, dense_commitment.clone());
            m.insert(oh_label, oh_commitment.clone());
            m
        },
        dense_label,
        dense_sumcheck_id,
        oh_label,
        oh_sumcheck_id,
    };

    let (_worker_results, mpc_sumcheck_claims) = run_rep3_local_test_with_coordinator(
        0,
        move |party_idx| worker_inputs[party_idx].clone(),
        move || coordinator_input,
        // Worker
        |input: MixedWorkerInput, mut io_ctx: IoContextPool<_>| {
            let party_id = io_ctx.party_id();
            let mut accumulator = Rep3OpeningAccumulatorWorker::<F>::new(party_id);

            // Dense opening
            accumulator.append_dense(
                vec![input.dense_label],
                input.dense_sumcheck_id,
                input.dense_opening_point,
                &[input.dense_claim],
            );
            // One-hot opening
            accumulator.append_sparse(
                vec![input.oh_label],
                input.oh_sumcheck_id,
                &input.oh_r_address,
                &input.oh_r_cycle,
                vec![input.oh_claim],
            );

            accumulator.reduce_and_prove::<PCS, FS, _>(
                &input.polys,
                input.hints,
                &input.setup,
                io_ctx.network(),
            )?;

            Ok(())
        },
        // Coordinator
        |input: MixedCoordinatorInput, network| {
            let mut transcript = FS::new(b"reduce_and_prove_mixed_test");
            let mut accumulator = Rep3OpeningAccumulator::<F>::new();

            // Dense opening
            accumulator.append_dense(
                &mut transcript,
                vec![input.dense_label],
                input.dense_sumcheck_id,
                input.dense_opening_point,
                vec![input.dense_claim],
            );
            // One-hot opening
            accumulator.append_sparse(
                &mut transcript,
                vec![input.oh_label],
                input.oh_sumcheck_id,
                &input.oh_r_address,
                &input.oh_r_cycle,
                vec![input.oh_claim],
            );

            let mut commitment_map = input.commitments;

            let proof = accumulator.reduce_and_prove::<PCS, FS, _>(
                &mut commitment_map,
                &input.setup,
                &mut transcript,
                network,
            )?;

            Ok(proof.sumcheck_claims)
        },
    );

    // === Part 3: Compare sumcheck claims ===
    assert_eq!(
        vanilla_sumcheck_claims.len(),
        mpc_sumcheck_claims.len(),
        "Sumcheck claims count mismatch"
    );
    for (i, (vanilla, mpc)) in vanilla_sumcheck_claims
        .iter()
        .zip(mpc_sumcheck_claims.iter())
        .enumerate()
    {
        assert_eq!(
            vanilla, mpc,
            "Sumcheck claim {i} mismatch: vanilla={vanilla:?}, mpc={mpc:?}"
        );
    }
}
