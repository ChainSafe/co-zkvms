use std::collections::HashMap;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::Rep3MultilinearPolynomial;
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::witness::generate_witness_batch_rep3;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
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

        // --- Send trace_length to coordinator ---
        state.io_ctx.network().send_response(trace_length)?;

        let ram_K = state.ram_K;
        let bytecode_d = state.prover_state.preprocessing.shared.bytecode.d;

        // --- Commit untrusted advice (uses separate DoryGlobals, must come first) ---
        Self::commit_untrusted_advice::<F, PCS, N>(&mut state)?;

        // --- Initialize main DoryGlobals and AllCommittedPolynomials ---
        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length);
        let _poly_guard =
            AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d);

        let hint_map = Self::generate_and_commit_polynomials::<F, PCS, ProofTranscript, N>(
            party_id, &mut state,
        )?;

        // --- Compute trusted advice polynomial (after witness commit, matching vanilla) ---
        Self::compute_trusted_advice_poly::<F, PCS, N>(&mut state);

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
        let poly_keys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();

        // Access fields directly to avoid overlapping borrows on `state`
        let witness_polys = generate_witness_batch_rep3(
            &poly_keys,
            state.prover_state.preprocessing,
            &state.prover_state.trace,
            &mut state.io_ctx,
        )?;

        // Collect polys in AllCommittedPolynomials order for alignment with coordinator.
        let commit_to_public = party_id == PartyID::ID0;
        let generators = &state.prover_state.preprocessing.generators;

        let ordered_polys: Vec<Rep3MultilinearPolynomial<F>> = poly_keys
            .iter()
            .map(|key| witness_polys.get(key).cloned().unwrap_or_default())
            .collect();

        let commit_results = <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::batch_commit_rep3(
            &ordered_polys,
            generators,
            commit_to_public,
        );

        let (commitment_shares, hint_shares): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();

        // Send commitment shares to coordinator
        state.io_ctx.network().send_response(commitment_shares)?;

        // Send untrusted advice commitment to coordinator (all workers computed
        // the same public commitment; coordinator verifies consistency).
        state
            .io_ctx
            .network()
            .send_response(state.untrusted_advice_commitment.clone())?;

        // Open hints across parties (without coordinator)
        Self::open_hints::<F, PCS, ProofTranscript, N>(&poly_keys, hint_shares, state)
    }

    /// Commit the untrusted advice polynomial (if non-empty).
    ///
    /// Mirrors vanilla `JoltDAG::commit_untrusted_advice`: packs advice bytes
    /// into u64 words, builds a `MultilinearPolynomial`, commits with standard
    /// `PCS::commit` (advice is public data, not secret-shared), and stores the
    /// commitment + polynomial on state.
    ///
    /// Uses a separate `DoryGlobals(K=1, T=max_advice_size/8)` that is dropped
    /// before the main witness DoryGlobals is initialized.
    fn commit_untrusted_advice<F, PCS, N>(
        state: &mut StateManagerWorker<'_, F, PCS, N>,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        N: Rep3NetworkWorker,
    {
        if state.program_io.untrusted_advice.is_empty() {
            return Ok(());
        }

        let max_size = state.program_io.memory_layout.max_untrusted_advice_size as usize / 8;
        let _advice_dory_guard = DoryGlobals::initialize(1, max_size);

        let mut initial_memory_state = vec![0u64; max_size];
        let mut index = 1;
        for chunk in state.program_io.untrusted_advice.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            initial_memory_state[index] = u64::from_le_bytes(word);
            index += 1;
        }

        let poly = MultilinearPolynomial::from(initial_memory_state);
        let (commitment, _hint) = PCS::commit(&poly, &state.prover_state.preprocessing.generators);

        state.untrusted_advice_commitment = Some(commitment);
        state.prover_state.untrusted_advice_polynomial =
            Some(Rep3MultilinearPolynomial::Public(poly));

        Ok(())
    }

    /// Compute the trusted advice polynomial (if non-empty).
    ///
    /// Mirrors vanilla `JoltDAG::compute_trusted_advice_poly`: packs advice
    /// bytes into u64 words, builds a `MultilinearPolynomial`, and stores it
    /// on prover state. No commitment is computed here — the trusted advice
    /// commitment comes from an external source.
    fn compute_trusted_advice_poly<F, PCS, N>(state: &mut StateManagerWorker<'_, F, PCS, N>)
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        N: Rep3NetworkWorker,
    {
        if state.program_io.trusted_advice.is_empty() {
            return;
        }

        let max_size = state.program_io.memory_layout.max_trusted_advice_size as usize / 8;

        let mut initial_memory_state = vec![0u64; max_size];
        let mut index = 1;
        for chunk in state.program_io.trusted_advice.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            initial_memory_state[index] = u64::from_le_bytes(word);
            index += 1;
        }

        let poly = MultilinearPolynomial::from(initial_memory_state);
        state.prover_state.trusted_advice_polynomial =
            Some(Rep3MultilinearPolynomial::Public(poly));
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
