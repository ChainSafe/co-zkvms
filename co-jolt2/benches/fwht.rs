use ark_bn254::Fr;
use co_jolt2::utils::fwht::{fwht_in_place, fwht_rep3_in_place};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use std::time::Instant;

const M: usize = 65536;
const ITERS: usize = 100;

fn make_field_vec() -> Vec<Fr> {
    (0..M).map(|i| Fr::from(i as u64 + 1)).collect()
}

fn make_rep3_vec() -> Vec<Rep3PrimeFieldShare<Fr>> {
    (0..M)
        .map(|i| Rep3PrimeFieldShare::new(Fr::from(i as u64 + 1), Fr::from(i as u64 + 2)))
        .collect()
}

fn bench(name: &str, f: impl Fn()) {
    // Warmup
    for _ in 0..5 {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / ITERS as u32;
    println!("{name}: {per_iter:?} / iter  ({ITERS} iters, total {elapsed:?})");
}

fn main() {
    bench("fwht_in_place<Fr>", || {
        let mut v = make_field_vec();
        fwht_in_place(&mut v);
    });

    bench("fwht_rep3_in_place<Fr>", || {
        let mut v = make_rep3_vec();
        fwht_rep3_in_place(&mut v);
    });
}
