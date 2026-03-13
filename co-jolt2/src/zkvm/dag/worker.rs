use std::collections::HashMap;
use std::sync::Arc;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use tracing::info_span;

use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::multilinear_polynomial::Rep3SharedPoly;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::Rep3MultilinearPolynomial;
use crate::subprotocols::sumcheck::HybridBatchedSumcheckWorker;
use crate::utils::memory::maybe_purge_jemalloc;
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::stage::{Rep3JoltDagStagesWorker, SumcheckStagesWorker};
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::spartan::Rep3SpartanDagWorker;
use crate::zkvm::witness::{generate_witness_batch_rep3, populate_cycle_witness_rep3};
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryContext, DoryGlobals};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::D;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K};
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::r2f_b2a_many;
use rand::distributions::{Distribution, Standard};

/// Worker side of the MPC DAG prover.
///
/// Generates shared witness polynomials, commits shares, and participates
/// in distributed sumcheck rounds driven by the coordinator.
pub struct Rep3JoltDagWorker;

impl Rep3JoltDagWorker {
    /// Generate witness polynomials, commit, send commitment shares to coordinator,
    /// and open hint shares across parties.
    ///
    /// Returns the opened hint map (keyed by `CommittedPolynomial`).
    #[tracing::instrument(skip_all, name = "JoltDag::prove")]
    pub fn prove<F, PCS, ProofTranscript, N>(
        mut state: StateManagerWorker<'_, F, PCS>,
        mut io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
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

        // --- Commit untrusted advice under its own DoryGlobals (K=1, T=advice_size) ---
        // Must happen before the Main DoryGlobals init because the advice poly
        // is smaller and needs its own sigma/nu dimensions.
        Self::commit_untrusted_advice::<F, PCS, ProofTranscript, N>(
            &mut state,
            padded_trace_length,
            &mut io_ctx,
            preproc,
        )?;

        // In-process tests share DoryGlobals across all 3 worker threads.
        // Barrier ensures all workers finish the advice commit (which uses
        // advice-sized DoryGlobals) before any worker initializes Main globals.
        #[cfg(feature = "test-utils")]
        io_ctx.sync_with_parties()?;

        let _guard = (
            DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
            AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d),
        );

        let (opening_hints, polynomials_map, instruction_one_hot_polys) =
            Self::generate_and_commit_polynomials::<F, PCS, ProofTranscript, N>(
                party_id,
                &mut state,
                &mut io_ctx,
                preproc,
            )?;

        // --- Compute trusted advice polynomial (after witness commit, matching vanilla) ---
        Self::compute_trusted_advice_poly::<F, PCS, N>(&mut state, &mut io_ctx)?;

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
        );

        let stage2_instances = stages.stage2_instances(&mut state, &mut io_ctx)?;
        let _stage2 = info_span!("stage2_prove").entered();
        HybridBatchedSumcheckWorker::prove(stage2_instances, &mut state.accumulator, &mut io_ctx, preproc)?;
        drop(_stage2);
        maybe_purge_jemalloc();

        // -------------------------------------------------------------------
        // Stage 3: batched sumcheck (secret + public instances)
        // -------------------------------------------------------------------

        let stage3_instances = stages.stage3_instances(&mut state, &mut io_ctx, preproc)?;
        let _stage3 = info_span!("stage3_prove").entered();
        HybridBatchedSumcheckWorker::prove(stage3_instances, &mut state.accumulator, &mut io_ctx, preproc)?;
        drop(_stage3);
        maybe_purge_jemalloc();

        // -------------------------------------------------------------------
        // Stage 4: batched sumcheck (RAM + Bytecode public, Lookups RA secret)
        // -------------------------------------------------------------------
        let stage4_instances = stages.stage4_instances(&mut state, &mut io_ctx)?;
        if !stage4_instances.is_empty() {
            let _stage4 = info_span!("stage4_prove").entered();
            HybridBatchedSumcheckWorker::prove(stage4_instances, &mut state.accumulator, &mut io_ctx, preproc)?;
            drop(_stage4);
        }

        // Stage 2-4 DAG state can be dropped before the opening reduction.
        drop(stages);
        maybe_purge_jemalloc();

        // -------------------------------------------------------------------
        // Untrusted advice opening proof (if advice is non-empty)
        // -------------------------------------------------------------------
        if state.prover_state.untrusted_advice_polynomial.is_some() {
            Self::prove_untrusted_advice_opening::<F, PCS, ProofTranscript, N>(&mut state, &mut io_ctx)?;
        }

        // In-process tests share DoryGlobals across all 3 worker threads.
        // Barrier ensures all workers finish the advice opening proof (which uses
        // UntrustedAdvice DoryContext) before any worker enters stage5 (Main context).
        #[cfg(feature = "test-utils")]
        io_ctx.sync_with_parties()?;

        // -------------------------------------------------------------------
        // Stage 5: opening proof reduction
        // -------------------------------------------------------------------
        let _stage5 = info_span!("stage5_reduce_and_prove").entered();
        state.accumulator.reduce_and_prove::<PCS, ProofTranscript, N>(
            &polynomials_map,
            opening_hints,
            &state.prover_state.preprocessing.generators,
            &mut io_ctx,
        )?;
        drop(_stage5);
        maybe_purge_jemalloc();

        Ok(())
    }

    /// Generate witness polynomials, commit shares, send commitments to
    /// coordinator, and open hint shares across parties.
    fn generate_and_commit_polynomials<F, PCS, ProofTranscript, N>(
        party_id: PartyID,
        state: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
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
        let poly_keys: Vec<CommittedPolynomial> = AllCommittedPolynomials::iter().copied().collect();

        // Populate the field-domain per-cycle witness cache (used for Spartan Stage1 and later).
        populate_cycle_witness_rep3(state, io_ctx, preproc)?;

        let mut witness_polys = generate_witness_batch_rep3(&poly_keys, state, io_ctx, preproc)?;

        let instruction_one_hot_polys: [Rep3OneHotPolynomial<F>; D] = std::array::from_fn(|i| {
            let key = CommittedPolynomial::InstructionRa(i);
            let poly = witness_polys.get(&key).unwrap_or_else(|| panic!("missing witness poly for {key:?}"));
            match poly {
                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(one_hot)) => one_hot.clone(),
                _ => panic!("witness poly for {key:?} is not a shared OneHot polynomial"),
            }
        });

        // Collect polys in AllCommittedPolynomials order for alignment with coordinator.
        let generators = &state.prover_state.preprocessing.generators;

        // Avoid cloning large polynomials just to commit: commit borrows them.
        let default_poly = Rep3MultilinearPolynomial::<F>::default();
        let ordered_polys: Vec<&Rep3MultilinearPolynomial<F>> =
            poly_keys.iter().map(|key| witness_polys.get(key).unwrap_or(&default_poly)).collect();

        let commit_results = <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::batch_commit_rep3(
            &ordered_polys,
            generators,
            io_ctx,
            preproc,
        )?;

        let (commitment_shares, hint_shares): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();

        // Send commitment shares to coordinator
        io_ctx.network().send_response(commitment_shares)?;

        // Send untrusted advice commitment share to coordinator.
        io_ctx.network().send_response(state.untrusted_advice_commitment.clone())?;

        // Build hint map from raw MaybeShared hint shares (used by reduce_and_prove).
        let hint_map: HashMap<CommittedPolynomial, MaybeShared<PCS::OpeningProofHint>> = poly_keys
            .iter()
            .zip(hint_shares)
            .filter_map(|(key, hint)| match &hint {
                MaybeShared::Public(None) => None,
                _ => Some((*key, hint)),
            })
            .collect();

        // Replace U64Scalars (used for ring MSM commit) with Dense field-share polys
        // for opening proof evaluation. The U64Scalars variant cannot evaluate in the field.
        #[cfg(feature = "ring-msm")]
        {
            use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
            let n = state.prover_state.cycle_witness.len();
            for key in [CommittedPolynomial::LeftInstructionInput, CommittedPolynomial::RightInstructionInput] {
                if let Some(poly) = witness_polys.get(&key) {
                    if matches!(poly, Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::CompactRing(_))) {
                        let mut field_shares: Vec<Rep3PrimeFieldShare<F>> = Vec::with_capacity(n);
                        for t in 0..n {
                            let (l, r) = state.prover_state.cycle_witness.row_stage1(t).to_instruction_inputs(party_id);
                            field_shares.push(if key == CommittedPolynomial::LeftInstructionInput { l } else { r });
                        }
                        witness_polys.insert(key, Rep3MultilinearPolynomial::from(field_shares));
                    }
                }
            }
        }

        // Build Arc-wrapped polynomial map for reduce_and_prove.
        let polynomials_map: HashMap<CommittedPolynomial, Arc<Rep3MultilinearPolynomial<F>>> =
            witness_polys.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();

        // Ring-shared trace is no longer needed after witness generation; drop it to free memory.
        state.prover_state.trace = None;

        Ok((hint_map, polynomials_map, instruction_one_hot_polys))
    }

    fn shared_advice_polynomial<F, PCS, N>(
        advice: &[mpc_core::protocols::rep3_ring::Rep3RingShare<u8>],
        max_size: usize,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<Rep3MultilinearPolynomial<F>>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        N: Rep3NetworkWorker,
    {
        let mut coeffs = vec![Rep3PrimeFieldShare::zero_share(); max_size];
        let words = crate::host::jolt_device::Rep3ProgramIOInput::pack_advice_words(advice);
        let field_words = r2f_b2a_many(&words, io_ctx.main())?;
        for (i, share) in field_words.into_iter().enumerate() {
            coeffs[i + 1] = share;
        }
        Ok(Rep3MultilinearPolynomial::from_shared_coeffs(coeffs))
    }

    /// Commit the untrusted advice polynomial (if non-empty) using Rep3 shares.
    ///
    /// Initializes `DoryContext::UntrustedAdvice` with advice dimensions (K=1, T=max_size).
    /// The hint is kept for the separate advice opening proof produced after stage4.
    fn commit_untrusted_advice<F, PCS, ProofTranscript, N>(
        state: &mut StateManagerWorker<'_, F, PCS>,
        padded_trace_length: usize,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        N: Rep3NetworkWorker,
    {
        if state.program_io.untrusted_advice.is_empty() {
            return Ok(());
        }

        let ws = jolt_common::constants::RAM_WORD_SIZE as usize;
        let max_size = state.program_io.memory_layout.max_untrusted_advice_size as usize / ws;
        eyre::ensure!(
            max_size <= padded_trace_length,
            "max_untrusted_advice_size/{ws} ({max_size}) exceeds padded_trace_length ({padded_trace_length}); \
             current PCS generators/DoryGlobals are built for padded_trace_length"
        );

        // Initialize UntrustedAdvice DoryContext (persists alongside Main context).
        // Use set_context (not with_context) — guard-based restore races with
        // other worker threads sharing the same process-global CURRENT_CONTEXT.
        DoryGlobals::initialize_context(1, max_size, DoryContext::UntrustedAdvice, None);
        DoryGlobals::set_context(DoryContext::UntrustedAdvice);

        let poly = Self::shared_advice_polynomial::<F, PCS, N>(&state.program_io.untrusted_advice, max_size, io_ctx)?;
        let (commitment, hint) = <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::commit_rep3(
            &poly,
            &state.prover_state.preprocessing.generators,
            false,
            io_ctx,
            preproc,
        )?;

        state.untrusted_advice_commitment = Some(commitment);
        state.prover_state.untrusted_advice_polynomial = Some(poly);
        state.prover_state.untrusted_advice_hint = Some(hint);

        DoryGlobals::set_context(DoryContext::Main);
        Ok(())
    }

    /// Produce the untrusted advice opening proof.
    ///
    /// The coordinator sends the opening point (derived from the stage2 accumulator).
    /// Workers evaluate the advice polynomial at that point, send an additive share
    /// of the evaluation, then participate in a coordinated Dory prove under
    /// `DoryContext::UntrustedAdvice`.
    fn prove_untrusted_advice_opening<F, PCS, ProofTranscript, N>(
        state: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        N: Rep3NetworkWorker,
    {
        // Receive the advice opening point from the coordinator.
        let opening_point: Vec<F::Challenge> = io_ctx.network().receive_request()?;

        // Evaluate the shared advice polynomial at the opening point.
        let poly = state.prover_state.untrusted_advice_polynomial.as_ref().unwrap();
        let point_f: Vec<F> = opening_point.iter().map(|c| (*c).into()).collect();
        let eval = poly.evaluate(&point_f);
        let eval_f: F = match eval {
            crate::utils::types::Rep3Value::Additive(additive) => additive.into_fe(),
            crate::utils::types::Rep3Value::Public(v) => v,
            crate::utils::types::Rep3Value::Shared(_) => {
                unreachable!("advice polynomial should produce Additive eval")
            }
        };
        io_ctx.network().send_response(eval_f)?;

        // Switch to UntrustedAdvice DoryContext for the prove.
        // NOTE: We use set_context instead of with_context because the guard
        // pattern is not safe when multiple threads share the same global
        // CURRENT_CONTEXT (the last guard to drop restores the wrong context).
        DoryGlobals::set_context(DoryContext::UntrustedAdvice);
        let hint = state.prover_state.untrusted_advice_hint.take().map(|h| match h {
            MaybeShared::Shared(v) => v,
            MaybeShared::Public(Some(v)) => v,
            MaybeShared::Public(None) => unreachable!("advice hint should not be None"),
        });
        let result = <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::prove_rep3(
            poly,
            &state.prover_state.preprocessing.generators,
            &opening_point,
            hint,
            io_ctx.network(),
        );
        DoryGlobals::set_context(DoryContext::Main);
        result?;

        Ok(())
    }

    /// Compute the trusted advice polynomial (if non-empty) from Rep3 shares.
    fn compute_trusted_advice_poly<F, PCS, N>(
        state: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        N: Rep3NetworkWorker,
    {
        if state.program_io.trusted_advice.is_empty() {
            return Ok(());
        }

        let max_size = state.program_io.memory_layout.max_trusted_advice_size as usize / 8;
        let poly = Self::shared_advice_polynomial::<F, PCS, N>(&state.program_io.trusted_advice, max_size, io_ctx)?;
        state.prover_state.trusted_advice_polynomial = Some(poly);
        Ok(())
    }
}
