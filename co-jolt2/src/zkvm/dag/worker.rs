use std::collections::HashMap;
use std::sync::Arc;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tracing::info_span;

use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::multilinear_polynomial::Rep3SharedPoly;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::Rep3MultilinearPolynomial;
use crate::subprotocols::sumcheck::HybridBatchedSumcheckWorker;
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::stage::{Rep3JoltDagStagesWorker, SumcheckStagesWorker};
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::spartan::Rep3SpartanDagWorker;
use crate::zkvm::witness::{generate_witness_batch_rep3, populate_cycle_witness_rep3};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::D;
use jolt_core::zkvm::witness::{
    compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
};
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
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
        mut state: StateManagerWorker<'_, F, PCS>,
        mut io_ctx: &mut IoContextPool<N>,
        edabits_pool: mpc_core::protocols::rep3_ring::edabits::PreprocessingPool<F>,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        PCS::OpeningProofHint: CanonicalSerialize + CanonicalDeserialize,
        N: Rep3NetworkWorker,
        Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
    {
        let trace_length = state.trace_len();
        let padded_trace_length = trace_length.next_power_of_two();
        let party_id = io_ctx.party_id();

        // --- Send trace_length to coordinator ---
        io_ctx.network().send_response(trace_length)?;

        let ram_K = state.ram_K;
        let bytecode_d = state.prover_state.preprocessing.shared.bytecode.d;

        let _guard = (
            DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
            AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d),
        );

        // --- Commit untrusted advice (must use the same DoryGlobals T) ---
        Self::commit_untrusted_advice::<F, PCS>(&mut state, padded_trace_length)?;

        let (opening_hints, polynomials_map, instruction_one_hot_polys) =
            Self::generate_and_commit_polynomials::<F, PCS, ProofTranscript, N>(
                party_id,
                &mut state,
                &mut io_ctx,
            )?;

        // --- Compute trusted advice polynomial (after witness commit, matching vanilla) ---
        Self::compute_trusted_advice_poly::<F, PCS>(&mut state);

        // Stage 1 (Spartan outer sumcheck)
        let (outer_sumcheck_r, claimed_witness_evals) =
            Rep3SpartanDagWorker::stage1_prove::<F, PCS, N>(&mut state, &mut io_ctx)?;

        // Stage 1 witness vectors are no longer needed after Spartan stage 1:
        // later stages use (meta, inc polynomials, read_raf witness, product inputs).
        state.prover_state.cycle_witness.drop_stage1();

        let mut stages = Rep3JoltDagStagesWorker::new(
            outer_sumcheck_r,
            claimed_witness_evals,
            padded_trace_length,
            instruction_one_hot_polys,
            edabits_pool,
        );

        let mut stage2_instances = stages.stage2_instances(&mut state, &mut io_ctx)?;
        let _stage2 = info_span!("stage2_prove").entered();
        HybridBatchedSumcheckWorker::prove(
            &mut stage2_instances,
            &mut state.accumulator,
            &mut io_ctx,
        )?;
        drop(_stage2);

        // // -------------------------------------------------------------------
        // // Stage 2: batched sumcheck (secret instances only)
        // // -------------------------------------------------------------------

        // let mut registers = Rep3RegistersDagWorker::<F>::new();
        // let mut ram = Rep3RamDagWorker::<F>::new(&mut state, &mut io_ctx)?;
        // let mut lookups = Rep3LookupsDagWorker::<F>::new(instruction_one_hot_polys);

        // let (registers_gamma, registers_input_claim): (F, F) =
        //     io_ctx.network().receive_request()?;
        // registers.set_stage2_init(registers_gamma, registers_input_claim);

        // let (ram_gamma, ram_input_claim, ram_r_address): (F, F, Vec<F::Challenge>) =
        //     io_ctx.network().receive_request()?;
        // ram.set_stage2_init(ram_gamma, ram_input_claim, ram_r_address);

        // let (lookup_gamma, lookup_r_address): ([F; D], Vec<F::Challenge>) =
        //     io_ctx.network().receive_request()?;
        // lookups.set_stage2_init(lookup_gamma, lookup_r_address);

        // let mut stage2_instances: Vec<BatchedSumcheckWorkerInstance<F>> = vec![];
        // stage2_instances.extend(registers.stage2_instances(&mut state));
        // stage2_instances.extend(ram.stage2_instances(&mut state));
        // stage2_instances.extend(lookups.stage2_instances(&mut state));

        // HybridBatchedSumcheckWorker::prove(
        //     &mut stage2_instances,
        //     &mut state.accumulator,
        //     &mut io_ctx,
        // )?;

        // -------------------------------------------------------------------
        // Stage 3: batched sumcheck (secret + public instances)
        // -------------------------------------------------------------------

        let mut stage3_instances = stages.stage3_instances(&mut state, &mut io_ctx)?;
        let _stage3 = info_span!("stage3_prove").entered();
        HybridBatchedSumcheckWorker::prove(
            &mut stage3_instances,
            &mut state.accumulator,
            &mut io_ctx,
        )?;
        drop(_stage3);

        // -------------------------------------------------------------------
        // Stage 4: batched sumcheck (RAM + Bytecode public, Lookups RA secret)
        // -------------------------------------------------------------------
        let mut stage4_instances = stages.stage4_instances(&mut state, &mut io_ctx)?;
        if !stage4_instances.is_empty() {
            let _stage4 = info_span!("stage4_prove").entered();
            HybridBatchedSumcheckWorker::prove(
                &mut stage4_instances,
                &mut state.accumulator,
                &mut io_ctx,
            )?;
        }
        // -------------------------------------------------------------------
        // Stage 5: opening proof reduction
        // -------------------------------------------------------------------
        let _stage5 = info_span!("stage5_reduce_and_prove").entered();
        state
            .accumulator
            .reduce_and_prove::<PCS, ProofTranscript, N>(
                &polynomials_map,
                opening_hints,
                &state.prover_state.preprocessing.generators,
                &mut io_ctx,
            )?;
        drop(_stage5);

        Ok(())
    }

    /// Generate witness polynomials, commit shares, send commitments to
    /// coordinator, and open hint shares across parties.
    fn generate_and_commit_polynomials<F, PCS, ProofTranscript, N>(
        party_id: PartyID,
        state: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<(
        HashMap<CommittedPolynomial, MaybeShared<PCS::OpeningProofHint>>,
        HashMap<CommittedPolynomial, Arc<Rep3MultilinearPolynomial<F>>>,
        [Rep3OneHotPolynomial<F>; D],
    )>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        PCS::OpeningProofHint: CanonicalSerialize + CanonicalDeserialize,
        N: Rep3NetworkWorker,
        Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
    {
        let poly_keys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();

        // Populate the field-domain per-cycle witness cache (used for Spartan Stage1 and later).
        populate_cycle_witness_rep3(state, io_ctx)?;

        let witness_polys = generate_witness_batch_rep3(&poly_keys, state, io_ctx)?;

        let instruction_one_hot_polys: [Rep3OneHotPolynomial<F>; D] = std::array::from_fn(|i| {
            let key = CommittedPolynomial::InstructionRa(i);
            let poly = witness_polys
                .get(&key)
                .unwrap_or_else(|| panic!("missing witness poly for {key:?}"));
            match poly {
                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(one_hot)) => {
                    one_hot.clone()
                }
                _ => panic!("witness poly for {key:?} is not a shared OneHot polynomial"),
            }
        });

        // Collect polys in AllCommittedPolynomials order for alignment with coordinator.
        let commit_to_public = party_id == PartyID::ID0;
        let generators = &state.prover_state.preprocessing.generators;

        // Avoid cloning large polynomials just to commit: commit borrows them.
        let default_poly = Rep3MultilinearPolynomial::<F>::default();
        let ordered_polys: Vec<&Rep3MultilinearPolynomial<F>> = poly_keys
            .iter()
            .map(|key| witness_polys.get(key).unwrap_or(&default_poly))
            .collect();

        let commit_results = <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::batch_commit_rep3(
            &ordered_polys,
            generators,
            commit_to_public,
        );

        let (commitment_shares, hint_shares): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();

        // Send commitment shares to coordinator
        io_ctx.network().send_response(commitment_shares)?;

        // Send untrusted advice commitment to coordinator (all workers computed
        // the same public commitment; coordinator verifies consistency).
        io_ctx
            .network()
            .send_response(state.untrusted_advice_commitment.clone())?;

        // Build hint map from raw MaybeShared hint shares (used by reduce_and_prove).
        let hint_map: HashMap<CommittedPolynomial, MaybeShared<PCS::OpeningProofHint>> = poly_keys
            .iter()
            .zip(hint_shares)
            .filter_map(|(key, hint)| match &hint {
                MaybeShared::Public(None) => None,
                _ => Some((*key, hint)),
            })
            .collect();

        // Build Arc-wrapped polynomial map for reduce_and_prove.
        let polynomials_map: HashMap<CommittedPolynomial, Arc<Rep3MultilinearPolynomial<F>>> =
            witness_polys
                .into_iter()
                .map(|(k, v)| (k, Arc::new(v)))
                .collect();

        // Ring-shared trace is no longer needed after witness generation; drop it to free memory.
        state.prover_state.trace = None;

        Ok((hint_map, polynomials_map, instruction_one_hot_polys))
    }

    /// Commit the untrusted advice polynomial (if non-empty).
    ///
    /// Mirrors vanilla `JoltDAG::commit_untrusted_advice`: packs advice bytes
    /// into u64 words, builds a `MultilinearPolynomial`, commits with standard
    /// `PCS::commit` (advice is public data, not secret-shared), and stores the
    /// commitment + polynomial on state.
    fn commit_untrusted_advice<F, PCS>(
        state: &mut StateManagerWorker<'_, F, PCS>,
        padded_trace_length: usize,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        if state.program_io.untrusted_advice.is_empty() {
            return Ok(());
        }

        let max_size = state.program_io.memory_layout.max_untrusted_advice_size as usize / 8;
        eyre::ensure!(
            max_size <= padded_trace_length,
            "max_untrusted_advice_size/8 ({max_size}) exceeds padded_trace_length ({padded_trace_length}); \
             current PCS generators/DoryGlobals are built for padded_trace_length"
        );

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
    fn compute_trusted_advice_poly<F, PCS>(state: &mut StateManagerWorker<'_, F, PCS>)
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
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
}
