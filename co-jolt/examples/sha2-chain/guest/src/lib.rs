#![cfg_attr(feature = "guest", no_std)]

use jolt::UntrustedAdvice;
use sha2::{Digest, Sha256};

#[jolt::provable]
fn sha2_chain(num_iters: u32, input: UntrustedAdvice<[u8; 32]>) -> [u8; 32] {
    let mut hash = input.value;
    for _ in 0..num_iters {
        let mut hasher = Sha256::new();
        hasher.update(hash);
        let res = &hasher.finalize();
        hash = Into::<[u8; 32]>::into(*res);
    }

    hash
}
