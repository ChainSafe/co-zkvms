//! Transcript utilities, mostly taken from
use std::vec::Vec;

use ark_ff::Field;
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_serialize::CanonicalSerialize;
use rand::RngCore;

pub struct TranscriptMerlin(merlin::Transcript);

impl TranscriptMerlin {
    pub fn new(label: &'static [u8]) -> Self {
        Self(merlin::Transcript::new(label))
    }
}

/// A Transcript with some shorthands for feeding scalars, group elements, and obtaining challenges as field elements.
pub trait Transcript: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error> {
    fn append_serializable<S: CanonicalSerialize>(&mut self, label: &'static [u8], msg: &S);

    /// Compute a `label`ed challenge scalar from the given commitments and the choice bit.
    fn get_scalar_challenge<F: Field>(&mut self, label: &'static [u8]) -> F;

    fn get_vector_challenge<F: Field>(&mut self, label: &'static [u8], size: usize) -> Vec<F>;
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
        // self.append_serializable(b"", msg);
        Ok(())
    }

    fn setup() -> Self {
        Self(merlin::Transcript::new(b"dfs"))
    }
}

impl Transcript for TranscriptMerlin {
    fn append_serializable<S: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        serializable: &S,
    ) {
        let mut message = Vec::new();
        serializable.serialize_uncompressed(&mut message).unwrap();
        // self.0.append_message(label, &message)
    }

    fn get_scalar_challenge<F: Field>(&mut self, label: &'static [u8]) -> F {
        loop {
            let mut bytes = [0; 64];
            self.0.challenge_bytes(label, &mut bytes);
            if let Some(e) = F::from_random_bytes(&bytes) {
                return e;
            }
        }
    }

    fn get_vector_challenge<F: Field>(&mut self, label: &'static [u8], size: usize) -> Vec<F> {
        (0..size)
            .map(|_| self.get_scalar_challenge(label))
            .collect()
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
