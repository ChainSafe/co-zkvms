use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
pub use jolt_core::utils::transcript::*;

pub trait TranscriptExt: Transcript {
    type State: Clone + CanonicalSerialize + CanonicalDeserialize;
    fn state(&self) -> Self::State;

    fn from_state(state: Self::State) -> Self;
}

impl TranscriptExt for KeccakTranscript {
    type State = [u8; 32];

    fn from_state(state: Self::State) -> Self {
        let mut transcript = KeccakTranscript::new(b"");
        transcript.state = state;
        transcript
    }

    fn state(&self) -> Self::State {
        self.state
    }
}
