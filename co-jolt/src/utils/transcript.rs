use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
pub use jolt_core::utils::transcript::*;

pub trait TranscriptExt: Transcript {
    type State: Clone + CanonicalSerialize + CanonicalDeserialize;
    fn state(&self) -> Self::State;

    fn from_state(state: Self::State) -> Self;

    fn update_state(&mut self, state: Self::State);
}

impl TranscriptExt for KeccakTranscript {
    type State = ([u8; 32], u32);

    fn from_state(state: Self::State) -> Self {
        let (state, n_rounds) = state;
        let mut transcript = KeccakTranscript::new(b"");
        transcript.state = state;
        transcript.n_rounds = n_rounds;
        transcript
    }

    fn state(&self) -> Self::State {
        (self.state, self.n_rounds)
    }

    fn update_state(&mut self, state: Self::State) {
        let (state, n_rounds) = state;
        self.state = state;
        self.n_rounds = n_rounds;
    }
}
