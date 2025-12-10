use crate::field::JoltField;
use crate::poly::opening_proof::*;
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ff::AdditiveGroup;
use ark_poly_commit::multilinear_pc::data_structures::Proof;
use ark_std::test_rng;
use ark_std::One;
use itertools::chain;
use itertools::izip;
use itertools::Itertools;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::multilinear_polynomial::PolynomialBinding;
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::utils::transcript::AppendToTranscript;
use jolt_core::utils::transcript::Transcript;
use jolt_core::{
    msm::Icicle,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, eq_poly::EqPolynomial,
        multilinear_polynomial::PolynomialEvaluation,
    },
    utils::transcript::KeccakTranscript,
};
use rayon::prelude::*;
use snarks_core::math::Math;
use std::env;

use crate::poly::commitment::{
    mock::MockCommitScheme,
    pst13::{PST13Setup, PST13},
};

#[test]
fn test_simulate_local_open_reduce() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    type F = ark_bn254::Fr;
    let NUM_CHUNKED_POLYS = env::var("NUM_CHUNKED_POLYS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();
    let W: usize = env::var("NUM_WORKERS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();
    let NUM_FULL_POLYS = env::var("NUM_FULL_POLYS")
        .unwrap_or_else(|_| "2".to_string())
        .parse::<usize>()
        .unwrap()
        * W;
    let P: usize = env::var("P")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let P2: usize = P * 2;
    // let W: usize = env::var("NUM_WORKERS")
    //     .unwrap_or_else(|_| "2".to_string())
    //     .parse()
    //     .unwrap();

    let polys1 = (0..NUM_CHUNKED_POLYS)
        .map(|i| MultilinearPolynomial::from((0..P).map(|e| F::from((e + i) as u64)).collect_vec()))
        .collect_vec();

    let polys2 = (0..NUM_FULL_POLYS)
        .map(|i| {
            MultilinearPolynomial::from(
                (0..P2)
                    .map(|e| F::from((e + i + NUM_CHUNKED_POLYS) as u64))
                    .collect_vec(),
            )
        })
        .collect_vec();

    let polys1_ref = polys1.iter().collect_vec();
    let polys2_ref = polys2.iter().collect_vec();

    let comms = MockCommitScheme::<F, KeccakTranscript>::batch_commit(&polys1_ref, &());
    let comms2 = MockCommitScheme::<F, KeccakTranscript>::batch_commit(&polys2_ref, &());

    println!("\n/------------ Append ------------/");

    println!(
        "polys1: {:?}",
        polys1
            .iter()
            .map(|p| p.coeffs_as_field_elements())
            .collect_vec()
    );

    let mut transcript = KeccakTranscript::new(&[]);
    let mut opening_accum = ProverOpeningAccumulator::<F, KeccakTranscript>::new();

    let r_point = transcript.challenge_vector(P.log_2());

    // {
    //     let r_point = r_point.iter().rev().copied().collect_vec();
    //     let mut p = DensePolynomial::new(polys[0].clone().coeffs_as_field_elements());
    //     r_point[..r_point.len() - 1]
    //         .iter()
    //         .for_each(|r| p.bind(*r, BindingOrder::LowToHigh));
    //     println!("rem_evals: {:?}", &p.Z[..p.len()]);
    //     println!(
    //         "eval full: {:?}",
    //         &DensePolynomial::new(p.Z[..p.len()].to_vec())
    //             .evaluate(&r_point[r_point.len() - 1..])
    //     );
    // }
    let (claims1, eq_r1) = MultilinearPolynomial::batch_evaluate(&polys1_ref, &r_point);
    println!("claims1: {:?}", claims1);
    opening_accum.append(
        &polys1_ref,
        DensePolynomial::new(eq_r1),
        r_point.clone(),
        &claims1,
        &mut transcript,
    );

    println!("\n/------------ Append 2 ------------/");
    println!(
        "polys2: {:?}",
        polys2
            .iter()
            .map(|p| p.coeffs_as_field_elements())
            .collect_vec()
    );
    let r2_point = transcript.challenge_vector(P2.log_2());
    let (claims2, eq_r2) = MultilinearPolynomial::batch_evaluate(&polys2_ref, &r2_point);
    println!("claims2: {:?}", claims2);

    opening_accum.append(
        &polys2_ref,
        DensePolynomial::new(eq_r2),
        r2_point.clone(),
        &claims2,
        &mut transcript,
    );

    println!("\n/------------ Prover ------------/");
    let proof = opening_accum
        .reduce_and_prove::<MockCommitScheme<F, KeccakTranscript>>(&(), &mut transcript);

    println!("\n/------------ Verifier ------------/");

    let mut transcript = KeccakTranscript::new(&[]);
    let mut opening_accum = VerifierOpeningAccumulator::<
        F,
        MockCommitScheme<F, KeccakTranscript>,
        KeccakTranscript,
    >::new();

    let r_point = transcript.challenge_vector(P.log_2());

    opening_accum.append(
        &comms.iter().collect_vec(),
        r_point,
        &claims1.iter().collect_vec(),
        &mut transcript,
    );

    let r2_point = transcript.challenge_vector(P2.log_2());

    opening_accum.append(
        &comms2.iter().collect_vec(),
        r2_point,
        &claims2.iter().collect_vec(),
        &mut transcript,
    );

    opening_accum
        .reduce_and_verify(&(), &proof, &mut transcript)
        .unwrap();
}

#[test]
fn test_simulate_distributed_open_reduce() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    type E = ark_bn254::Bn254;
    type F = ark_bn254::Fr;
    let NUM_CHUNKED_POLYS = env::var("NUM_CHUNKED_POLYS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();
    let NUM_FULL_POLYS = env::var("NUM_FULL_POLYS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();
    let P: usize = env::var("P")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let P2: usize = P * 2;
    let P_nv = P.log_2();
    let W: usize = env::var("NUM_WORKERS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();
    let W_log2 = W.log_2();
    let P_worker = P / W;

    println!(
        "W={W}, P={P}, P2={P2}, NUM_CHUNKED_POLYS={NUM_CHUNKED_POLYS}, NUM_FULL_POLYS={NUM_FULL_POLYS}"
    );

    let worker_polys1 = (0..W)
        .map(|w| {
            (0..NUM_CHUNKED_POLYS)
                .map(|i| {
                    MultilinearPolynomial::from(
                        (w * P_worker..P_worker * (w + 1))
                            .map(|e| F::from((e + i) as u64))
                            .collect_vec(),
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let worker_polys2 = (0..W)
        .map(|w| {
            (0..NUM_FULL_POLYS)
                .map(|i| {
                    MultilinearPolynomial::from(
                        (0..P2)
                            .map(|e| F::from((NUM_CHUNKED_POLYS + i + e + w) as u64))
                            .collect_vec(),
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let full_polys = chain![
        (0..NUM_CHUNKED_POLYS)
            .map(|i| {
                MultilinearPolynomial::from(
                    (0..W)
                        .flat_map(|w| worker_polys1[w][i].coeffs_as_field_elements())
                        .collect_vec(),
                )
            })
            .collect_vec(),
        (0..W).flat_map(|w| (0..NUM_FULL_POLYS)
            .map(|i| worker_polys2[w][i].clone())
            .collect_vec())
    ]
    .collect_vec();

    let mut rng = test_rng();
    let setup = PST13::setup(P2, &mut rng);
    let comms_p = <PST13<E> as CommitmentScheme<KeccakTranscript>>::batch_commit(
        &full_polys[..NUM_CHUNKED_POLYS],
        &setup,
    );
    let comms_p2 = <PST13<E> as CommitmentScheme<KeccakTranscript>>::batch_commit(
        &full_polys[NUM_CHUNKED_POLYS..],
        &setup,
    );

    let mut transcript = KeccakTranscript::new(&[]);
    let mut accums = (0..W)
        .map(|_| ProverOpeningAccumulator::<F, KeccakTranscript>::new())
        .collect_vec();

    println!("\n/------------ Append 1 ------------/");

    println!(
        "worker_polys1: {:?}",
        worker_polys1
            .iter()
            .map(|wp| wp
                .iter()
                .map(|p| p.coeffs_as_field_elements())
                .collect_vec())
            .collect_vec()
    );

    let r_point = transcript.challenge_vector(P_nv);

    let rem_evals = (0..W)
        .map(|w| {
            let (claims, _) = MultilinearPolynomial::batch_evaluate(
                &worker_polys1[w].iter().collect_vec(),
                &r_point[W_log2..],
            );

            claims
        })
        .fold(vec![vec![]; NUM_CHUNKED_POLYS], |mut acc, next| {
            izip!(acc.iter_mut(), next).for_each(|(a, b)| {
                a.push(b);
            });
            acc
        });

    println!("rem_evals: {:?}", rem_evals);

    let claims1 = rem_evals
        .iter()
        .cloned()
        .map(|evals| MultilinearPolynomial::from(evals).evaluate(&r_point[..W_log2]))
        .collect_vec();

    println!("claims: {:?}", claims1);

    let tr_clone = transcript.clone();
    for w in 0..W {
        let mut tr_tmp = tr_clone.clone();
        let transcript = if w != 0 { &mut tr_tmp } else { &mut transcript };
        let chunk = 1usize << (P_nv - W_log2);
        let worker_polys_padded = worker_polys1[w]
            .iter()
            .map(|p| {
                let mut padded_evals = vec![F::ZERO; P];
                padded_evals[w * chunk..(w + 1) * chunk]
                    .copy_from_slice(&p.coeffs_as_field_elements());
                MultilinearPolynomial::from(padded_evals)
            })
            .collect_vec();
        accums[w].append(
            &worker_polys_padded.iter().collect_vec(),
            DensePolynomial::new(EqPolynomial::evals(&r_point)),
            r_point.clone(),
            &claims1,
            transcript,
        );
    }

    println!("\n/------------ Append 2 ------------/");

    println!(
        "worker_polys2: {:?}",
        worker_polys2
            .iter()
            .map(|wp| wp
                .iter()
                .map(|p| p.coeffs_as_field_elements())
                .collect_vec())
            .collect_vec()
    );

    let r2_point = transcript.challenge_vector(P2.log_2());

    let claims2 = (0..W)
        .flat_map(|w| {
            (0..NUM_FULL_POLYS)
                .map(|i| worker_polys2[w][i].evaluate(&r2_point))
                .collect_vec()
        })
        .collect_vec();

    println!("claims2: {:?}", claims2);

    {
        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..NUM_FULL_POLYS * W {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let batched_claim: F = rho_powers
            .iter()
            .zip(claims2.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();

        for w in 0..W {
            let batched_poly = MultilinearPolynomial::linear_combination(
                &worker_polys2[w].iter().collect_vec(),
                &rho_powers[NUM_FULL_POLYS * w..NUM_FULL_POLYS * (w + 1)],
            );
            let opening = ProverOpening::new(
                batched_poly,
                DensePolynomial::new(EqPolynomial::evals(&r2_point)),
                r2_point.clone(),
                batched_claim,
            );
            accums[w].openings.push(opening);
        }
    }

    println!("\n/------------ Prover ------------/");

    let proof = simulate_distributed_reduce_and_prove(&mut accums, &setup, &mut transcript);

    println!("\n/------------ Verifier ------------/");

    let mut transcript = KeccakTranscript::new(&[]);
    let mut opening_accum = VerifierOpeningAccumulator::<F, PST13<E>, KeccakTranscript>::new();

    let r_point = transcript.challenge_vector(P.log_2());

    opening_accum.append(
        &comms_p.iter().collect_vec(),
        r_point,
        &claims1.iter().collect_vec(),
        &mut transcript,
    );

    let r_point2 = transcript.challenge_vector(P2.log_2());

    opening_accum.append(
        &comms_p2.iter().collect_vec(),
        r_point2,
        &claims2.iter().collect_vec(),
        &mut transcript,
    );

    opening_accum
        .reduce_and_verify(&setup, &proof, &mut transcript)
        .unwrap();
}

pub fn simulate_distributed_reduce_and_prove<
    // F: JoltField,
    E: Pairing<ScalarField: JoltField, G1: Icicle>,
    // PCS: CommitmentScheme<KeccakTranscript, Field = F>,
>(
    accums: &mut [ProverOpeningAccumulator<E::ScalarField, KeccakTranscript>],
    pcs_setup: &PST13Setup<E>,
    transcript: &mut KeccakTranscript,
) -> ReducedOpeningProof<E::ScalarField, PST13<E>, KeccakTranscript> {
    // Generate coefficients for random linear combination
    let rho: E::ScalarField = transcript.challenge_scalar();
    let mut rho_powers = vec![E::ScalarField::one()];
    for i in 1..accums[0].openings.len() {
        rho_powers.push(rho_powers[i - 1] * rho);
    }

    let coeffs = rho_powers;

    let worker_unbound_polys = accums
        .iter()
        .map(|a| {
            a.openings
                .iter()
                .map(|opening| opening.polynomial.clone())
                .collect::<Vec<_>>()
        })
        .collect_vec();

    let max_num_vars = accums[0]
        .openings
        .iter()
        .map(|opening| opening.polynomial.get_num_vars())
        .max()
        .unwrap();

    // Compute random linear combination of the claims, accounting for the fact that the
    // polynomials may be of different sizes
    let mut e: E::ScalarField = coeffs
        .par_iter()
        .zip(accums[0].openings.par_iter())
        .map(|(coeff, opening)| {
            let scaled_claim = if opening.polynomial.get_num_vars() != max_num_vars {
                E::ScalarField::from_u64_unchecked(
                    1 << (max_num_vars - opening.polynomial.get_num_vars()),
                ) * opening.claim
            } else {
                opening.claim
            };
            scaled_claim * coeff
        })
        .sum();

    println!("combined_claim: {e}");

    let mut r_sumcheck: Vec<_> = Vec::new();
    let mut compressed_polys = Vec::new();

    // let log_num_workers = accums.len().log_2();
    let worker_rounds = max_num_vars; // - log_num_workers;

    for round in 0..worker_rounds {
        let remaining_rounds = max_num_vars - round;
        let mut evals = accums
            .iter()
            .enumerate()
            .map(|(w, w_acc)| {
                println!("worker {w}:");
                let evals = compute_quadratic(w_acc, &coeffs, remaining_rounds, w);
                evals
            })
            .reduce(|p, n| izip!(p, n).map(|(a, b)| a + b).collect_vec())
            .unwrap();
        println!("round evals: {:?}", evals);
        println!("----------------------");
        evals.insert(1, e - evals[0]);
        let uni_poly = UniPoly::from_evals(&evals);
        let compressed_poly = uni_poly.compress();

        // append the prover's message to the transcript
        compressed_poly.append_to_transcript(transcript);
        let r_j = transcript.challenge_scalar();
        println!("r_j: {:?}", r_j);
        r_sumcheck.push(r_j);

        for acc in accums.iter_mut() {
            acc.openings.par_iter_mut().for_each(|opening| {
                if remaining_rounds <= opening.opening_point.len() {
                    rayon::join(
                        || opening.eq_poly.bind(r_j, BindingOrder::HighToLow),
                        || opening.polynomial.bind(r_j, BindingOrder::HighToLow),
                    );
                }
            });
        }

        e = uni_poly.evaluate(&r_j);
        compressed_polys.push(compressed_poly);
    }

    let sumcheck_claims: Vec<_> = accums
        .iter()
        .map(|acc| {
            acc.openings
                .iter()
                .map(|opening| opening.polynomial.final_sumcheck_claim())
                .collect_vec()
        })
        .reduce(|acc, item| izip!(acc, item).map(|(a, b)| a + b).collect_vec())
        .unwrap();

    transcript.append_scalars(&sumcheck_claims);

    let sumcheck_proof = SumcheckInstanceProof::new(compressed_polys);

    let gamma: E::ScalarField = transcript.challenge_scalar();
    let mut gamma_powers = vec![E::ScalarField::one()];
    for i in 1..accums[0].openings.len() {
        gamma_powers.push(gamma_powers[i - 1] * gamma);
    }

    let joint_opening_proof = (0..accums.len())
        .map(|w| {
            let joint_poly = MultilinearPolynomial::linear_combination(
                &worker_unbound_polys[w].iter().collect::<Vec<_>>(),
                &gamma_powers,
            );

            // Reduced opening proof
            PST13::prove(pcs_setup, &joint_poly, &r_sumcheck, transcript)
        })
        .reduce(|acc, item| Proof {
            proofs: izip!(acc.proofs, item.proofs)
                .map(|(p, n)| (p + n).into_affine())
                .collect(),
        })
        .unwrap();

    ReducedOpeningProof {
        sumcheck_proof,
        sumcheck_claims,
        joint_opening_proof,
    }
}

fn compute_quadratic<F: JoltField>(
    acc: &ProverOpeningAccumulator<F, KeccakTranscript>,
    coeffs: &[F],
    remaining_rounds: usize,
    w: usize,
) -> Vec<F> {
    let evals: Vec<(F, F)> = acc
        .openings
        .par_iter()
        .map(|opening| {
            println!("--------------");
            println!("poly: {:?}", opening.polynomial.coeffs_as_field_elements());
            if remaining_rounds <= opening.opening_point.len() {
                let mle_half = opening.polynomial.len() / 2;
                let eval_0: F = (0..mle_half)
                    .map(|i| {
                        println!(
                            "eval_0: {:?}",
                            [
                                opening.eq_poly.get_bound_coeff(i),
                                opening.polynomial.get_bound_coeff(i),
                            ]
                        );
                        println!("-----");
                        opening.polynomial.get_bound_coeff(i) * opening.eq_poly.get_bound_coeff(i)
                    })
                    .sum();
                let eval_2: F = (0..mle_half)
                    .map(|i| {
                        let poly_bound_point = opening.polynomial.get_bound_coeff(i + mle_half)
                            + opening.polynomial.get_bound_coeff(i + mle_half)
                            - opening.polynomial.get_bound_coeff(i);
                        let eq_bound_point = opening.eq_poly.get_bound_coeff(i + mle_half)
                            + opening.eq_poly.get_bound_coeff(i + mle_half)
                            - opening.eq_poly.get_bound_coeff(i);

                        println!("eval_2: {:?}", [eq_bound_point, poly_bound_point]);
                        println!("-----");
                        poly_bound_point * eq_bound_point
                    })
                    .sum();
                println!("----------");
                // println!("summed eval_0 {} eval_2 {}", eval_0, eval_2);
                (eval_0, eval_2)
            } else {
                debug_assert!(!opening.polynomial.is_bound());
                if w != 0 {
                    return (F::zero(), F::zero());
                }
                let remaining_variables = remaining_rounds - opening.opening_point.len() - 1;
                let scaled_claim = F::from_u64_unchecked(1 << remaining_variables) * opening.claim;
                println!(
                    "remaining_variables {} claim {} scaled_claim: {:?}",
                    remaining_variables, opening.claim, scaled_claim
                );
                println!("-----");
                (scaled_claim, scaled_claim)
            }
        })
        .collect();

    let evals_combined_0: F = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
    let evals_combined_2: F = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
    vec![evals_combined_0, evals_combined_2]
}
