use jolt_common::jolt_device::MemoryLayout;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use serde::{Deserialize, Serialize};
use tracer::JoltDevice;

use crate::utils::transpose;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rep3ProgramIOInput {
    pub trusted_advice: Vec<Rep3RingShare<u8>>,
    pub untrusted_advice: Vec<Rep3RingShare<u8>>,
    pub inputs: Vec<u8>,
    pub outputs: Vec<u8>,
    pub panic: bool,
    pub memory_layout: MemoryLayout,
}

impl Rep3ProgramIOInput {
    pub fn generate_secret_shares<R: rand::Rng>(program_io: JoltDevice, rng: &mut R) -> Vec<Self> {
        let JoltDevice { inputs, trusted_advice, untrusted_advice, outputs, panic, memory_layout } = program_io;

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

        itertools::izip!(trusted_advice_shares, untrusted_advice_shares)
            .map(|(trusted_advice, untrusted_advice)| Self {
                trusted_advice,
                untrusted_advice,
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                panic,
                memory_layout: memory_layout.clone(),
            })
            .collect()
    }

    pub fn pack_advice_words(advice: &[Rep3RingShare<u8>]) -> Vec<Rep3RingShare<u64>> {
        advice.chunks(8).map(Rep3RingShare::<u64>::from_le_bytes).collect()
    }
}

#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use mpc_core::protocols::rep3_ring::combine_ring_element_binary;
    use tracer::JoltDevice;

    use super::Rep3ProgramIOInput;

    #[test]
    fn generate_secret_shares_preserves_public_metadata_and_packs_advice_words() {
        let program_io = JoltDevice {
            inputs: vec![1, 2, 3],
            trusted_advice: vec![0x11, 0x22, 0x33, 0x44, 0x55],
            untrusted_advice: vec![0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x12, 0x34, 0x56],
            outputs: vec![9, 8],
            panic: true,
            memory_layout: Default::default(),
        };

        let mut rng = test_rng();
        let shares = Rep3ProgramIOInput::generate_secret_shares(program_io.clone(), &mut rng);
        let [share0, share1, share2]: [Rep3ProgramIOInput; 3] = shares.try_into().expect("expected 3 shares");

        for share in [&share0, &share1, &share2] {
            assert_eq!(share.inputs, program_io.inputs);
            assert_eq!(share.outputs, program_io.outputs);
            assert_eq!(share.panic, program_io.panic);
            assert_eq!(share.memory_layout, program_io.memory_layout);
        }

        let trusted_words0 = Rep3ProgramIOInput::pack_advice_words(&share0.trusted_advice);
        let trusted_words1 = Rep3ProgramIOInput::pack_advice_words(&share1.trusted_advice);
        let trusted_words2 = Rep3ProgramIOInput::pack_advice_words(&share2.trusted_advice);
        let trusted_reconstructed: Vec<u64> = trusted_words0
            .into_iter()
            .zip(trusted_words1)
            .zip(trusted_words2)
            .map(|((w0, w1), w2)| combine_ring_element_binary(w0, w1, w2).0)
            .collect();
        assert_eq!(trusted_reconstructed, vec![0x0000_0055_4433_2211]);

        let untrusted_words0 = Rep3ProgramIOInput::pack_advice_words(&share0.untrusted_advice);
        let untrusted_words1 = Rep3ProgramIOInput::pack_advice_words(&share1.untrusted_advice);
        let untrusted_words2 = Rep3ProgramIOInput::pack_advice_words(&share2.untrusted_advice);
        let untrusted_reconstructed: Vec<u64> = untrusted_words0
            .into_iter()
            .zip(untrusted_words1)
            .zip(untrusted_words2)
            .map(|((w0, w1), w2)| combine_ring_element_binary(w0, w1, w2).0)
            .collect();
        assert_eq!(untrusted_reconstructed, vec![0x3412_ffee_ddcc_bbaa, 0x0000_0000_0000_0056]);
    }
}
