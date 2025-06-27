//! Transcript utilities, mostly taken from
use ark_ff::Field;
use ark_linear_sumcheck::rng::FeedableRNG;
use ark_serialize::CanonicalSerialize;
use merlin::Transcript;
use rand::RngCore;
use std::vec::Vec;

/// A Transcript with some shorthands for feeding scalars, group elements, and obtaining challenges as field elements.
pub trait LookupTranscript {
    fn append_serializable<S: CanonicalSerialize>(&mut self, label: &'static [u8], msg: &S);

    /// Compute a `label`ed challenge scalar from the given commitments and the choice bit.
    fn get_scalar_challenge<F: Field>(&mut self, label: &'static [u8]) -> F;

    fn get_vector_challenge<F: Field>(&mut self, label: &'static [u8], size: usize) -> Vec<F>;
}

impl LookupTranscript for Transcript {
    fn append_serializable<S: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        serializable: &S,
    ) {
        let mut message = Vec::new();
        serializable.serialize_uncompressed(&mut message).unwrap();
        self.append_message(label, &message)
    }

    fn get_scalar_challenge<F: Field>(&mut self, label: &'static [u8]) -> F {
        loop {
            let mut bytes = [0; 64];
            self.challenge_bytes(label, &mut bytes);
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
