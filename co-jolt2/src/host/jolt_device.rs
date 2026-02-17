use jolt2_common::jolt_device::MemoryLayout;
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use serde::{Deserialize, Serialize};
use tracer::JoltDevice;

use crate::utils::transpose;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rep3ProgramIOInput {
    pub trusted_advice: Vec<Rep3RingShare<u8>>,
    pub untrusted_advice: Vec<Rep3RingShare<u8>>,
    pub inputs: Vec<u8>,
    pub outputs: Vec<Rep3RingShare<u8>>,
    pub panic: Rep3RingShare<Bit>,
    pub memory_layout: MemoryLayout,
}

impl Rep3ProgramIOInput {
    pub fn generate_secret_shares<R: rand::Rng>(program_io: JoltDevice, rng: &mut R) -> Vec<Self> {
        let JoltDevice {
            inputs,
            trusted_advice,
            untrusted_advice,
            outputs,
            panic,
            memory_layout,
        } = program_io;

        let trusted_advice_shares = if trusted_advice.is_empty() {
            vec![vec![]; 3]
        } else {
            transpose(
                trusted_advice
                    .into_iter()
                    .map(|byte| rep3_ring::binary::generate_shares_rep3(byte, rng))
                    .collect::<Vec<_>>(),
            )
        };

        let untrusted_advice_shares = if untrusted_advice.is_empty() {
            vec![vec![]; 3]
        } else {
            transpose(
                untrusted_advice
                    .into_iter()
                    .map(|byte| rep3_ring::binary::generate_shares_rep3(byte, rng))
                    .collect::<Vec<_>>(),
            )
        };

        let output_shares = if outputs.is_empty() {
            vec![vec![]; 3]
        } else {
            transpose(
                outputs
                    .into_iter()
                    .map(|byte| rep3_ring::binary::generate_shares_rep3(byte, rng))
                    .collect::<Vec<_>>(),
            )
        };

        let panic_shares: Vec<Rep3RingShare<Bit>> =
            rep3_ring::binary::generate_shares_rep3(Bit::from(panic), rng);

        itertools::izip!(
            trusted_advice_shares,
            untrusted_advice_shares,
            output_shares,
            panic_shares
        )
        .map(|(trusted_advice, untrusted_advice, outputs, panic)| Self {
            trusted_advice,
            untrusted_advice,
            inputs: inputs.clone(),
            outputs,
            panic,
            memory_layout: memory_layout.clone(),
        })
        .collect()
    }
}
