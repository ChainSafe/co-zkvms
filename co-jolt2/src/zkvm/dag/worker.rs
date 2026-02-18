use std::collections::HashMap;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::witness::generate_witness_batch_rep3;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::witness::{
    compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
};
use mpc_core::protocols::rep3::network::{Rep3Network, Rep3NetworkWorker};
use mpc_core::protocols::rep3::PartyID;
use rand::distributions::{Distribution, Standard};

/// Worker side of the MPC DAG prover.
///
/// Generates shared witness polynomials, commits shares, and participates
/// in distributed sumcheck rounds driven by the coordinator.
pub struct Rep3JoltDAGWorker;

impl Rep3JoltDAGWorker {
    /// Generate witness polynomials, commit, send commitment shares to coordinator,
    /// and open hint shares across parties.
    ///
    /// Returns the opened hint map (keyed by `CommittedPolynomial`).
    #[tracing::instrument(skip_all, name = "Rep3JoltDAGWorker::prove")]
    pub fn prove<F, PCS, ProofTranscript, N>(
        mut state: StateManagerWorker<'_, F, PCS, N>,
    ) -> eyre::Result<HashMap<CommittedPolynomial, PCS::OpeningProofHint>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        PCS::OpeningProofHint: CanonicalSerialize + CanonicalDeserialize,
        N: Rep3NetworkWorker,
        Standard: Distribution<u64> + Distribution<u8> + Distribution<u128>,
    {
        let trace_length = state.prover_state.trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        let party_id = state.io_ctx.party_id();

        // --- Send trace_length to coordinator (first message) ---
        state.io_ctx.network().send_response(trace_length)?;

        // --- Initialize DoryGlobals and AllCommittedPolynomials ---
        let ram_K = state.ram_K;
        let bytecode_d = state.prover_state.preprocessing.shared.bytecode.d;
        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length);
        let _poly_guard =
            AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d);

        let hint_map = Self::generate_and_commit_polynomials::<F, PCS, ProofTranscript, N>(
            party_id, &mut state,
        )?;

        // Future stages (sumcheck, opening proof) will go here...
        Ok(hint_map)
    }

    /// Generate witness polynomials, commit shares, send commitments to
    /// coordinator, and open hint shares across parties.
    fn generate_and_commit_polynomials<F, PCS, ProofTranscript, N>(
        party_id: PartyID,
        state: &mut StateManagerWorker<'_, F, PCS, N>,
    ) -> eyre::Result<HashMap<CommittedPolynomial, PCS::OpeningProofHint>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        PCS::OpeningProofHint: CanonicalSerialize + CanonicalDeserialize,
        N: Rep3NetworkWorker,
        Standard: Distribution<u64> + Distribution<u8> + Distribution<u128>,
    {
        // Skip InstructionRa polys (one-hot commitment deferred)
        let polys_to_generate: Vec<CommittedPolynomial> = AllCommittedPolynomials::iter()
            .copied()
            .filter(|p| !matches!(p, CommittedPolynomial::InstructionRa(_)))
            .collect();

        // Access fields directly to avoid overlapping borrows on `state`
        let witness_polys = generate_witness_batch_rep3(
            &polys_to_generate,
            state.prover_state.preprocessing,
            &state.prover_state.trace,
            &mut state.io_ctx,
        )?;

        // Iterate in AllCommittedPolynomials order for alignment with coordinator.
        // Missing polys (InstructionRa) get (Public(None), Public(None)) placeholder.
        let commit_to_public = party_id == PartyID::ID0;
        let generators = &state.prover_state.preprocessing.generators;

        let poly_keys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();

        let commit_results: Vec<(
            MaybeShared<PCS::Commitment>,
            MaybeShared<PCS::OpeningProofHint>,
        )> = poly_keys
            .iter()
            .map(|poly_key| match witness_polys.get(poly_key) {
                Some(poly) => <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::commit_rep3(
                    poly,
                    generators,
                    commit_to_public,
                ),
                None => (MaybeShared::Public(None), MaybeShared::Public(None)),
            })
            .collect();

        let (commitment_shares, hint_shares): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();

        // Send commitment shares to coordinator
        state.io_ctx.network().send_response(commitment_shares)?;

        // Open hints across parties (without coordinator)
        Self::open_hints::<F, PCS, ProofTranscript, N>(&poly_keys, hint_shares, state)
    }

    /// Open hint shares across all 3 parties using two rounds of `reshare_many`.
    ///
    /// After two reshares each party holds all 3 additive shares and can
    /// reconstruct the full hint via `combine_hint_shares`.
    fn open_hints<F, PCS, ProofTranscript, N>(
        poly_keys: &[CommittedPolynomial],
        hint_shares: Vec<MaybeShared<PCS::OpeningProofHint>>,
        state: &mut StateManagerWorker<'_, F, PCS, N>,
    ) -> eyre::Result<HashMap<CommittedPolynomial, PCS::OpeningProofHint>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        PCS::OpeningProofHint: CanonicalSerialize + CanonicalDeserialize,
        N: Rep3NetworkWorker,
    {
        // Collect shared hint shares for resharing
        let mut shared_indices: Vec<usize> = Vec::new();
        let mut own_shared: Vec<PCS::OpeningProofHint> = Vec::new();

        for (i, hint) in hint_shares.iter().enumerate() {
            if let MaybeShared::Shared(h) = hint {
                shared_indices.push(i);
                own_shared.push(h.clone());
            }
        }

        if own_shared.is_empty() {
            // No shared hints — all public, return directly
            let mut hint_map = HashMap::with_capacity(poly_keys.len());
            for (key, hint) in poly_keys.iter().zip(hint_shares) {
                if let MaybeShared::Public(Some(h)) = hint {
                    hint_map.insert(*key, h);
                }
            }
            return Ok(hint_map);
        }

        // Round 1: send own shares to next party, receive prev party's shares
        let prev_shared: Vec<PCS::OpeningProofHint> =
            state.io_ctx.main().network.reshare_many(&own_shared)?;

        // Round 2: forward prev's shares, receive the third party's shares
        let prev_prev_shared: Vec<PCS::OpeningProofHint> =
            state.io_ctx.main().network.reshare_many(&prev_shared)?;

        // Combine all 3 shares per polynomial
        let mut hint_map = HashMap::with_capacity(poly_keys.len());
        let mut shared_idx = 0;

        for (i, key) in poly_keys.iter().enumerate() {
            match &hint_shares[i] {
                MaybeShared::Shared(_) => {
                    let own = MaybeShared::Shared(own_shared[shared_idx].clone());
                    let prev = MaybeShared::Shared(prev_shared[shared_idx].clone());
                    let prev_prev = MaybeShared::Shared(prev_prev_shared[shared_idx].clone());
                    let combined =
                        <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::combine_hint_shares(&[
                            &own, &prev, &prev_prev,
                        ]);
                    hint_map.insert(*key, combined);
                    shared_idx += 1;
                }
                MaybeShared::Public(Some(h)) => {
                    hint_map.insert(*key, h.clone());
                }
                MaybeShared::Public(None) => {
                    // Skipped polynomial (e.g. InstructionRa) — no hint
                }
            }
        }

        Ok(hint_map)
    }
}
