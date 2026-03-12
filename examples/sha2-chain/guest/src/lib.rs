#![cfg_attr(feature = "guest", no_std)]

use sha2::{Digest, Sha256};

#[jolt::provable(stack_size = 65536, memory_size = 10240)]
fn sha2_chain(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    let mut hash = input;
    for _ in 0..num_iters {
        let mut hasher = Sha256::new();
        hasher.update(hash);
        let res = &hasher.finalize();
        hash = Into::<[u8; 32]>::into(*res);
    }
    hash
}
