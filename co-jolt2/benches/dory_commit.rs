use ark_bn254::Fr;
use co_jolt2::poly::commitment::dory::{
    commit_local_rep3, precompute_dapoint_qs, test_support::init_dory_globals, DoryCommitmentScheme, DoryGlobals,
};
use co_jolt2::poly::commitment::Rep3CommitmentScheme;
use co_jolt2::poly::{Rep3CompactPolynomial, Rep3MultilinearPolynomial, Rep3SharedPoly};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::utils::math::Math;
use mpc_core::protocols::rep3::network::IoContextPool;
use mpc_core::protocols::rep3::test_utils::{run_rep3_local_test_with_coordinator, LocalRep3TestWorkerNet};
use mpc_core::protocols::rep3_ring;
use mpc_core::protocols::rep3_ring::edabits;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::ring::u66::U66;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::time::{Duration, Instant};

const N: usize = 1 << 18;
const ITERS: usize = 10;
const WARMUP: usize = 3;

#[cfg(feature = "ring-msm")]
fn main() {
    let mut rng = ChaCha12Rng::seed_from_u64(0);

    // Sample u64 coefficients
    let values: Vec<u64> = (0..N).map(|_| rng.gen()).collect();
    let coeffs_fr: Vec<Fr> = values.iter().copied().map(Fr::from).collect();

    // Dory SRS setup
    init_dory_globals(256, 512);
    let num_columns = DoryGlobals::get_num_columns();
    let sigma = num_columns.log_2();
    let num_vars = N.log_2();
    let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

    // Generate field-share polynomials (one per party)
    let polys_f = Rep3MultilinearPolynomial::generate_shares_from_coeffs(&coeffs_fr, &mut rng);

    // Generate ring-share polynomials (arith + bin, one per party)
    use jolt_common::constants::{ArithmeticWideInt, XlenInt};
    let all_arith: Vec<_> =
        values.iter().map(|&v| rep3_ring::share_ring_element(RingElement(v as ArithmeticWideInt), &mut rng)).collect();
    let all_bin: Vec<_> =
        values.iter().map(|&v| rep3_ring::share_ring_element_binary(RingElement(v as XlenInt), &mut rng)).collect();
    let polys_u64: [Rep3MultilinearPolynomial<Fr>; 3] = std::array::from_fn(|pid| {
        let shares: Vec<Rep3RingShare<ArithmeticWideInt>> = all_arith.iter().map(|s| s[pid]).collect();
        let shares_bin: Vec<Rep3RingShare<XlenInt>> = all_bin.iter().map(|s| s[pid]).collect();
        Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(Rep3CompactPolynomial::from_shares(
            shares, shares_bin,
        )))
    });

    // =========================================================================
    // Bench 1: commit_rep3 (field shares, local, no network)
    // =========================================================================
    let setup1 = setup.clone();
    let (durations_f, _) = run_rep3_local_test_with_coordinator(
        0,
        |pid| polys_f[pid].clone(),
        || (),
        move |poly, _io_ctx: IoContextPool<LocalRep3TestWorkerNet>| {
            // Warmup
            for _ in 0..WARMUP {
                let _ = commit_local_rep3::<Blake2bTranscript>(&poly, &setup1, false);
            }
            let start = Instant::now();
            for _ in 0..ITERS {
                let _ = commit_local_rep3::<Blake2bTranscript>(&poly, &setup1, false);
            }
            Ok(start.elapsed())
        },
        |(), _net| Ok(()),
    );
    let per_iter_f = durations_f[0] / ITERS as u32;
    println!("commit_rep3 (field shares): {:?} / iter  ({ITERS} iters, total {:?})", per_iter_f, durations_f[0]);

    // =========================================================================
    // Bench 2: batch_commit_rep3 (ring shares, MPC)
    // =========================================================================
    let setup2 = setup.clone();
    let (durations_u64, _) = run_rep3_local_test_with_coordinator(
        0,
        |pid| polys_u64[pid].clone(),
        || (),
        move |poly, mut io_ctx: IoContextPool<LocalRep3TestWorkerNet>| {
            let total_coeffs = (WARMUP + ITERS) * N;

            // Measure preprocessing time
            let preproc_start = Instant::now();
            let pool_dir = std::env::temp_dir().join(format!("co-jolt2-bench-{}", io_ctx.party_idx()));
            let mut preproc = edabits::preprocess_pool::<Fr, _>(
                &pool_dir,
                [0, 0, 0, 0, 0],
                0,
                total_coeffs,
                total_coeffs,
                0,
                0,
                &mut io_ctx,
            )?;

            let qs = precompute_dapoint_qs(&setup2, total_coeffs, num_columns);
            let lazy_dp = rep3_ring::preprocessing::daPoint::random_dapoints(&qs, &mut io_ctx)?;
            preproc.set_dapoints(lazy_dp);
            let preproc_elapsed = preproc_start.elapsed();

            // Warmup
            for _ in 0..WARMUP {
                let polys = vec![&poly];
                let _ = <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::batch_commit_rep3(
                    &polys,
                    &setup2,
                    &mut io_ctx,
                    &mut preproc,
                )?;
            }

            let start = Instant::now();
            for _ in 0..ITERS {
                let polys = vec![&poly];
                let _ = <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::batch_commit_rep3(
                    &polys,
                    &setup2,
                    &mut io_ctx,
                    &mut preproc,
                )?;
            }
            Ok((start.elapsed(), preproc_elapsed))
        },
        |(), _net| Ok(()),
    );
    let (online_u64, preproc_u64) = durations_u64[0];
    let per_iter_u64 = online_u64 / ITERS as u32;
    let preproc_per_commit = preproc_u64 / (WARMUP + ITERS) as u32;
    println!("batch_commit_rep3 (ring shares): {:?} / iter  ({ITERS} iters, total {:?})", per_iter_u64, online_u64);
    println!(
        "  preprocessing: {:?} total for {} commits ({:?} / commit)",
        preproc_u64,
        WARMUP + ITERS,
        preproc_per_commit
    );

    // Summary
    println!("\n--- Summary (N={N}, {ITERS} iters) ---");
    println!("  Field shares (commit_rep3):               {:?} / iter", per_iter_f);
    println!("  Ring shares  (batch_commit_rep3):  {:?} / iter (online only)", per_iter_u64);
    println!("  Ring shares  preprocessing:               {:?} / commit", preproc_per_commit);

    let speedup_online = per_iter_f.as_secs_f64() / per_iter_u64.as_secs_f64();
    println!("  Speedup (online only): {speedup_online:.2}x");
}
