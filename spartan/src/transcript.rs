//! Transcript utilities, mostly taken from
use std::vec::Vec;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge, FieldElementSize};
use ark_ff::{BigInteger, Field, PrimeField};
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_serialize::CanonicalSerialize;
use rand::RngCore;

#[derive(Clone)]
pub struct TranscriptMerlin(merlin::Transcript);

impl TranscriptMerlin {
    pub fn new(label: &'static [u8]) -> Self {
        Self(merlin::Transcript::new(label))
    }
}

/// A Transcript with some shorthands for feeding scalars, group elements, and obtaining challenges as field elements.
pub trait Transcript: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error> {
    fn append_scalar<F: Field>(&mut self, label: &'static [u8], msg: &F);

    fn append_serializable<S: CanonicalSerialize>(&mut self, label: &'static [u8], msg: &S);

    /// Compute a `label`ed challenge scalar from the given commitments and the choice bit.
    fn challenge_scalar<F: Field>(&mut self, label: &'static [u8]) -> F;

    fn challenge_vector<F: Field>(&mut self, label: &'static [u8], size: usize) -> Vec<F>;

    fn fork(&self) -> Self;
}

impl RngCore for TranscriptMerlin {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0; 4];
        self.0.challenge_bytes(b"", &mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0; 8];
        self.0.challenge_bytes(b"", &mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.challenge_bytes(b"", dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.0.challenge_bytes(b"", dest);
        Ok(())
    }
}

impl FeedableRNG for TranscriptMerlin {
    type Error = ark_linear_sumcheck::Error;

    fn feed<M: CanonicalSerialize>(&mut self, msg: &M) -> Result<(), Self::Error> {
        self.append_serializable(b"", msg);
        Ok(())
    }

    fn setup() -> Self {
        Self(merlin::Transcript::new(b"dfs"))
    }
}

impl Transcript for TranscriptMerlin {
    fn append_scalar<F: Field>(&mut self, label: &'static [u8], msg: &F) {
        self.append_serializable(label, msg);
    }

    fn append_serializable<S: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        serializable: &S,
    ) {
        let mut message = Vec::new();
        serializable.serialize_uncompressed(&mut message).unwrap();
        self.0.append_message(label, &message)
    }

    fn challenge_scalar<F: Field>(&mut self, label: &'static [u8]) -> F {
        loop {
            let mut bytes = [0; 64];
            self.0.challenge_bytes(label, &mut bytes);
            if let Some(e) = F::from_random_bytes(&bytes) {
                return e;
            }
        }
    }

    fn challenge_vector<F: Field>(&mut self, label: &'static [u8], size: usize) -> Vec<F> {
        (0..size).map(|_| self.challenge_scalar(label)).collect()
    }

    fn fork(&self) -> Self {
        Self(self.0.clone())
    }
}

impl CryptographicSponge for TranscriptMerlin {
    type Config = ();

    fn new(_params: &Self::Config) -> Self {
        unimplemented!()
    }

    fn absorb(&mut self, input: &impl Absorb) {
        self.0.append_message(b"", &input.to_sponge_bytes_as_vec());
    }

    fn squeeze_bytes(&mut self, num_bytes: usize) -> Vec<u8> {
        let mut bytes = vec![0; num_bytes];
        self.0.challenge_bytes(b"", &mut bytes);
        bytes
    }

    fn squeeze_field_elements_with_sizes<F: PrimeField>(
        &mut self,
        sizes: &[FieldElementSize],
    ) -> Vec<F> {
        if sizes.len() == 0 {
            return Vec::new();
        }

        let field_elements = self.challenge_vector::<F>(b"", sizes.len());

        let mut output = Vec::with_capacity(sizes.len());
        for (elem, size) in field_elements.into_iter().zip(sizes.iter()) {
            if let FieldElementSize::Truncated(num_bits) = *size {
                let mut bits: Vec<bool> = elem.into_bigint().to_bits_le();
                bits.truncate(num_bits);
                let emulated_bytes = bits
                    .chunks(8)
                    .map(|bits| {
                        let mut byte = 0u8;
                        for (i, &bit) in bits.into_iter().enumerate() {
                            if bit {
                                byte += 1 << i;
                            }
                        }
                        byte
                    })
                    .collect::<Vec<_>>();

                output.push(F::from_le_bytes_mod_order(emulated_bytes.as_slice()));
            } else {
                output.push(elem);
            }
        }

        output
    }

    fn squeeze_bits(&mut self, _num_bits: usize) -> Vec<bool> {
        unimplemented!()
    }
}

pub fn get_scalar_challenge<
    R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
    F: Field,
>(
    rng: &mut R,
) -> F {
    loop {
        let mut bytes = [0; 64];
        rng.fill_bytes(&mut bytes);
        if let Some(e) = F::from_random_bytes(&bytes) {
            return e;
        }
    }
}

pub fn get_vector_challenge<
    R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
    F: Field,
>(
    rng: &mut R,
    size: usize,
) -> Vec<F> {
    (0..size).map(|_| get_scalar_challenge(rng)).collect()
}
