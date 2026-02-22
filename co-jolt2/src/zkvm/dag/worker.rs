use std::collections::HashMap;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::multilinear_polynomial::Rep3SharedPoly;
use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::Rep3MultilinearPolynomial;
use crate::subprotocols::sumcheck::{
    BatchedSumcheckInstance, BatchedSumcheckWorkerInstance, HybridBatchedSumcheckWorker,
};
use crate::subprotocols::sumcheck::{Rep3BatchedSumcheckWorker, Rep3SumcheckInstanceWorker};
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::stage::SumcheckStagesWorker;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::instruction_lookups::Rep3LookupsDagWorker;
use crate::zkvm::ram::Rep3RamDagWorker;
use crate::zkvm::registers::Rep3RegistersDagWorker;
use crate::zkvm::spartan::product::Rep3ProductVirtualizationSumcheckWorker;
use crate::zkvm::spartan::Rep3InnerSumcheckWorker;
use crate::zkvm::witness::{generate_witness_batch_rep3, populate_cycle_witness_rep3};
use crate::zkvm::{dag::Rep3DagStop, spartan::Rep3SpartanDagWorker};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::eq_poly::EqPlusOnePolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction::CircuitFlags;
use jolt_core::zkvm::instruction_lookups::D;
use jolt_core::zkvm::spartan::pc::PCSumcheck;
use jolt_core::zkvm::witness::VirtualPolynomial;
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
        mut io_ctx: IoContextPool<N>,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        PCS::OpeningProofHint: CanonicalSerialize + CanonicalDeserialize,
        N: Rep3NetworkWorker,
        Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
    {
        Self::prove_with_stop::<F, PCS, ProofTranscript, N>(state, io_ctx, Rep3DagStop::AfterStage3)
    }

    #[tracing::instrument(skip_all, name = "Rep3JoltDAGWorker::prove_with_stop")]
    pub fn prove_with_stop<F, PCS, ProofTranscript, N>(
        mut state: StateManagerWorker<'_, F, PCS>,
        mut io_ctx: IoContextPool<N>,
        stop: Rep3DagStop,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        PCS::OpeningProofHint: CanonicalSerialize + CanonicalDeserialize,
        N: Rep3NetworkWorker,
        Standard: Distribution<u32> + Distribution<u64> + Distribution<u8> + Distribution<u128>,
    {
        let trace_length = state.prover_state.trace.len();
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

        let (_hint_map, instruction_one_hot_polys) = Self::generate_and_commit_polynomials::<
            F,
            PCS,
            ProofTranscript,
            N,
        >(party_id, &mut state, &mut io_ctx)?;

        // --- Compute trusted advice polynomial (after witness commit, matching vanilla) ---
        Self::compute_trusted_advice_poly::<F, PCS>(&mut state);

        if stop == Rep3DagStop::AfterCommitments {
            return Ok(());
        }

        // Stage 1 (Spartan outer sumcheck)
        let (outer_sumcheck_r, claimed_witness_evals) =
            Rep3SpartanDagWorker::stage1_prove::<F, PCS, N>(&mut state, &mut io_ctx)?;

        // // Cache outer sumcheck openings in the worker accumulator so downstream
        // // subsystems (registers, RAM, lookups) can read r_cycle and claims.
        // {
        //     use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId};
        //     use jolt_core::zkvm::r1cs::inputs::ALL_R1CS_INPUTS;
        //     use jolt_core::zkvm::witness::VirtualPolynomial;

        //     let num_steps_bits = padded_trace_length.ilog2() as usize;
        //     let r_cycle = &outer_sumcheck_r[..num_steps_bits];
        //     let r_cycle_point = OpeningPoint::new(r_cycle.to_vec());

        //     for (input, eval) in ALL_R1CS_INPUTS.iter().zip(claimed_witness_evals.iter()) {
        //         if let Ok(poly) = VirtualPolynomial::try_from(input) {
        //             state.accumulator.append_virtual(
        //                 poly,
        //                 SumcheckId::SpartanOuter,
        //                 r_cycle_point.clone(),
        //                 *eval,
        //             );
        //         }
        //     }
        // }

        // Future stages (opening proof, etc.) will go here...
        if stop == Rep3DagStop::AfterStage1 {
            return Ok(());
        }

        // --- Prepare RAM worker (ring→field conversion requires MPC communication) ---
        let mut ram_dag = Rep3RamDagWorker::new(&mut state, &mut io_ctx)?;

        // Stage 2: collect instances from all subsystems in vanilla ordering.

        // 1) Spartan inner sumcheck — receive (gamma, input_claim) from coordinator
        let (spartan_gamma, spartan_input_claim): (F, F) = io_ctx.network().receive_request()?;
        let inner = Rep3InnerSumcheckWorker::new(
            spartan_gamma,
            spartan_input_claim,
            &outer_sumcheck_r,
            claimed_witness_evals,
            padded_trace_length,
            party_id,
        );

        // 2) Registers read-write checking — receive (gamma, input_claim)
        let (reg_gamma, reg_input_claim): (F, F) = io_ctx.network().receive_request()?;
        let mut registers_dag = Rep3RegistersDagWorker::new();
        registers_dag.set_stage2_init(reg_gamma, reg_input_claim);
        let registers_instances = registers_dag.stage2_instances(&mut state);

        // 3) RAM — receive (gamma, input_claim, r_address)
        let (ram_gamma, ram_input_claim, ram_r_address): (F, F, Vec<F::Challenge>) =
            io_ctx.network().receive_request()?;
        ram_dag.set_stage2_init(ram_gamma, ram_input_claim, ram_r_address);
        let ram_instances = ram_dag.stage2_instances(&mut state);

        // 4) Lookups booleanity — receive (gamma_powers, r_address)
        let (lookups_gamma, lookups_r_address): ([F; D], Vec<F::Challenge>) =
            io_ctx.network().receive_request()?;
        let mut lookups_dag = Rep3LookupsDagWorker::new(instruction_one_hot_polys);
        lookups_dag.set_stage2_init(lookups_gamma, lookups_r_address);
        let lookups_instances = lookups_dag.stage2_instances(&mut state);

        // Collect all instances in vanilla order
        let mut instances: Vec<BatchedSumcheckWorkerInstance<F>> = std::iter::empty()
            .chain(std::iter::once(BatchedSumcheckWorkerInstance::Secret(
                Box::new(inner),
            )))
            .chain(registers_instances)
            .chain(ram_instances)
            .chain(lookups_instances)
            .collect();

        HybridBatchedSumcheckWorker::prove(&mut instances, &mut state.accumulator, &mut io_ctx)?;

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

        // // -------------------------------------------------------------------
        // // Stage 3: batched sumcheck (secret + public instances)
        // // -------------------------------------------------------------------

        // let registers_val_claim: F = io_ctx.network().receive_request()?;
        // registers.set_stage3_init(registers_val_claim);

        // let ram_stage3_input_claim: F = io_ctx.network().receive_request()?;
        // ram.set_stage3_init(ram_stage3_input_claim);

        // let lookup_stage3_gamma: [F; D] = io_ctx.network().receive_request()?;
        // lookups.set_stage3_init(lookup_stage3_gamma);

        // let (gamma_pc, input_claim_pc, input_claim_product): (F, F, F) =
        //     io_ctx.network().receive_request()?;

        // let party_id = io_ctx.party_id();
        // let log_T = state
        //     .accumulator
        //     .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter)
        //     .0
        //     .r
        //     .len();

        // let pc_sumcheck = if party_id == PartyID::ID0 {
        //     let cycle_witness = &state.prover_state.cycle_witness;
        //     let unexpanded_pc_poly: MultilinearPolynomial<F> =
        //         cycle_witness.unexpanded_pc.clone().into();
        //     let pc_poly: MultilinearPolynomial<F> = cycle_witness.pc.clone().into();

        //     let mask = 1u32 << (CircuitFlags::IsNoop as usize);
        //     let is_noop: Vec<u8> = cycle_witness
        //         .flags_bits
        //         .iter()
        //         .map(|bits| ((bits & mask) != 0) as u8)
        //         .collect();
        //     let is_noop_poly: MultilinearPolynomial<F> = is_noop.into();

        //     let r_cycle = state
        //         .accumulator
        //         .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter)
        //         .0
        //         .r;
        //     let (_, eq_plus_one_evals) = EqPlusOnePolynomial::<F>::evals(&r_cycle, None);
        //     let eq_plus_one_poly = MultilinearPolynomial::from(eq_plus_one_evals);

        //     PCSumcheck::<F>::new_prover_from_polys(
        //         input_claim_pc,
        //         gamma_pc,
        //         log_T,
        //         unexpanded_pc_poly,
        //         pc_poly,
        //         is_noop_poly,
        //         eq_plus_one_poly,
        //     )
        // } else {
        //     PCSumcheck::<F>::new_verifier_from_openings(input_claim_pc, gamma_pc, log_T)
        // };

        // let product_sumcheck =
        //     Rep3ProductVirtualizationSumcheckWorker::<F>::new(&mut state, input_claim_product);

        // let mut stage3_instances: Vec<BatchedSumcheckWorkerInstance<F>> = vec![];
        // stage3_instances.extend(registers.stage3_instances(&mut state));
        // stage3_instances.extend(ram.stage3_instances(&mut state));
        // stage3_instances.extend(lookups.stage3_instances(&mut state));
        // stage3_instances.push(BatchedSumcheckWorkerInstance::Public(Box::new(pc_sumcheck)));
        // stage3_instances.push(BatchedSumcheckWorkerInstance::Secret(Box::new(
        //     product_sumcheck,
        // )));

        // HybridBatchedSumcheckWorker::prove(
        //     &mut stage3_instances,
        //     &mut state.accumulator,
        //     &mut io_ctx,
        // )?;

        // if stop == Rep3DagStop::AfterStage3 {
        //     return Ok(());
        // }

        // Future stages (opening proof) will go here...
        Ok(())
    }

    /// Generate witness polynomials, commit shares, send commitments to
    /// coordinator, and open hint shares across parties.
    fn generate_and_commit_polynomials<F, PCS, ProofTranscript, N>(
        party_id: PartyID,
        state: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<(
        HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
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

        // Drop the committed witness polynomials after committing for now.
        // (Future stages that need PCS openings will store them on state.)

        // Send commitment shares to coordinator
        io_ctx.network().send_response(commitment_shares)?;

        // Send untrusted advice commitment to coordinator (all workers computed
        // the same public commitment; coordinator verifies consistency).
        io_ctx
            .network()
            .send_response(state.untrusted_advice_commitment.clone())?;

        // Open hints across parties (without coordinator)
        let hint_map =
            Self::open_hints::<F, PCS, ProofTranscript, N>(&poly_keys, hint_shares, state, io_ctx)?;

        // Ring-shared trace is no longer needed after witness generation; drop it to free memory.
        state.prover_state.trace.clear();
        state.prover_state.trace.shrink_to_fit();

        Ok((hint_map, instruction_one_hot_polys))
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

    /// Open hint shares across all 3 parties using two rounds of `reshare_many`.
    ///
    /// After two reshares each party holds all 3 additive shares and can
    /// reconstruct the full hint via `combine_hint_shares`.
    fn open_hints<F, PCS, ProofTranscript, N>(
        poly_keys: &[CommittedPolynomial],
        hint_shares: Vec<MaybeShared<PCS::OpeningProofHint>>,
        state: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
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
            io_ctx.main().network.reshare_many(&own_shared)?;

        // Round 2: forward prev's shares, receive the third party's shares
        let prev_prev_shared: Vec<PCS::OpeningProofHint> =
            io_ctx.main().network.reshare_many(&prev_shared)?;

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
