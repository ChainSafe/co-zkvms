use std::collections::HashMap;

use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::subprotocols::sumcheck::{BatchedSumcheckInstance, HybridBatchedSumcheck};
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::stage::Rep3SumcheckInstance;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManagerCoordinator};
use crate::zkvm::instruction_lookups::booleanity::Rep3BooleanitySumcheck;
use crate::zkvm::instruction_lookups::hamming_weight::Rep3HammingWeightSumcheck;
use crate::zkvm::instruction_lookups::read_raf_checking::Rep3ReadRafSumcheck;
use crate::zkvm::ram::output_check::{Rep3OutputSumcheck, Rep3ValFinalSumcheck};
use crate::zkvm::ram::raf_evaluation::Rep3RafEvaluation;
use crate::zkvm::ram::read_write_checking::Rep3RamReadWriteChecking;
use crate::zkvm::registers::read_write_checking::Rep3RegistersReadWriteChecking;
use crate::zkvm::registers::val_evaluation::Rep3ValEvaluation;
use crate::zkvm::spartan::product::Rep3ProductVirtualizationSumcheck;
use crate::zkvm::spartan::Rep3SpartanDag;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::dag::proof_serialization::{Claims, JoltProof};
use jolt_core::zkvm::spartan::pc::PCSumcheck;
use jolt_core::zkvm::witness::VirtualPolynomial;
use jolt_core::zkvm::witness::{
    compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

/// Coordinator side of the MPC DAG prover.
///
/// Owns the Fiat-Shamir transcript, drives sumcheck rounds by broadcasting
/// challenges, receives evaluation shares from workers, and assembles the
/// final proof.
pub struct Rep3JoltDAGCoordinator;

impl Rep3JoltDAGCoordinator {
    #[tracing::instrument(skip_all, name = "Rep3JoltDAGCoordinator::prove")]
    pub fn prove<'a, F, ProofTranscript, PCS, N>(
        mut state: StateManagerCoordinator<'a, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<JoltProof<F, PCS, ProofTranscript>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        // --- Receive trace_length from workers ---
        let trace_lengths: Vec<usize> = network.receive_responses()?;
        let trace_length = trace_lengths[0];
        eyre::ensure!(
            trace_lengths.iter().all(|&t| t == trace_length),
            "trace_length mismatch across parties"
        );
        state.trace_length = trace_length;
        let padded_trace_length = trace_length.next_power_of_two();

        // --- Fiat-Shamir preamble ---
        state.fiat_shamir_preamble(trace_length);

        // --- Initialize DoryGlobals and AllCommittedPolynomials ---
        let ram_K = state.ram_K;
        let bytecode_d = state.preprocessing.shared.bytecode.d;
        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length);
        let _poly_guard =
            AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d);

        // --- Receive, combine, and store commitments ---
        Self::receive_commitments::<F, PCS, ProofTranscript, N>(&mut state, network)?;

        // --- Receive untrusted advice commitment from workers ---
        Self::receive_untrusted_advice_commitment::<F, PCS, ProofTranscript, N>(
            &mut state, network,
        )?;

        // --- Append advice commitments to transcript (matching vanilla ordering) ---
        if let Some(ref untrusted_advice_commitment) = state.untrusted_advice_commitment {
            state
                .transcript
                .append_serializable(untrusted_advice_commitment);
        }

        if let Some(ref trusted_advice_commitment) = state.trusted_advice_commitment {
            state
                .transcript
                .append_serializable(trusted_advice_commitment);
        }

        Rep3SpartanDag::stage1_prove(&mut state, network)?;

        // -------------------------------------------------------------------
        // Stage 2: batched sumcheck
        // -------------------------------------------------------------------

        let stage2_instances = Self::stage2_collect_instances(&mut state, network)?;

        let stage2_hybrid: Vec<BatchedSumcheckInstance<F, ProofTranscript>> = stage2_instances
            .into_iter()
            .map(BatchedSumcheckInstance::Secret)
            .collect();

        let (proof, _r_stage2) = HybridBatchedSumcheck::prove(
            &stage2_hybrid,
            &mut state.accumulator,
            &mut state.transcript,
            network,
        )?;
        state
            .proofs
            .insert(ProofKeys::Stage2Sumcheck, ProofData::SumcheckProof(proof));

        // -------------------------------------------------------------------
        // Stage 3: batched sumcheck (secret + public instances)
        // -------------------------------------------------------------------

        let stage3_instances = Self::stage3_collect_instances(&mut state, network)?;

        let (stage3_proof, _r_stage3) = HybridBatchedSumcheck::prove(
            &stage3_instances,
            &mut state.accumulator,
            &mut state.transcript,
            network,
        )?;
        state.proofs.insert(
            ProofKeys::Stage3Sumcheck,
            ProofData::SumcheckProof(stage3_proof),
        );

        // -------------------------------------------------------------------
        // Stage 4: batched sumcheck (RAM + Bytecode public, Lookups RA secret)
        // -------------------------------------------------------------------

        let stage4_instances = Self::stage4_collect_instances(&mut state, network)?;

        if !stage4_instances.is_empty() {
            let (stage4_proof, _r_stage4) = HybridBatchedSumcheck::prove(
                &stage4_instances,
                &mut state.accumulator,
                &mut state.transcript,
                network,
            )?;
            state.proofs.insert(
                ProofKeys::Stage4Sumcheck,
                ProofData::SumcheckProof(stage4_proof),
            );
        }

        // --- Construct stub JoltProof with real commitments, deferred stages ---
        // -------------------------------------------------------------------
        // Stage 5: opening proof reduction
        // -------------------------------------------------------------------

        let poly_keys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();
        let mut commitment_map: HashMap<CommittedPolynomial, PCS::Commitment> = poly_keys
            .into_iter()
            .zip(state.commitments.iter().cloned())
            .collect();

        let pcs_setup = state
            .pcs_setup
            .expect("StateManagerCoordinator::pcs_setup must be set for reduce_and_prove (stage5)");
        let reduced = state
            .accumulator
            .reduce_and_prove::<PCS, ProofTranscript, N>(
                &mut commitment_map,
                pcs_setup,
                &mut state.transcript,
                network,
            )?;
        state.proofs.insert(
            ProofKeys::ReducedOpeningProof,
            ProofData::ReducedOpeningProof(
                jolt_core::poly::opening_proof::ReducedOpeningProof {
                    sumcheck_proof: reduced.sumcheck_proof,
                    sumcheck_claims: reduced.sumcheck_claims,
                    joint_opening_proof: reduced.joint_opening_proof,
                },
            ),
        );

        // --- Construct JoltProof ---
        let proof = JoltProof {
            opening_claims: Claims(std::mem::take(&mut state.accumulator.openings)),
            commitments: std::mem::take(&mut state.commitments),
            proofs: std::mem::take(&mut state.proofs),
            untrusted_advice_commitment: state.untrusted_advice_commitment.take(),
            trace_length,
            ram_K: state.ram_K,
            bytecode_d: state.preprocessing.shared.bytecode.d,
            twist_sumcheck_switch_index: state.twist_sumcheck_switch_index,
        };
        Ok(proof)
    }

    /// Collect all stage2 sumcheck instances from every subsystem.
    ///
    /// Creates coordinator-side instances (which derive transcript challenges in
    /// the correct order), broadcasts init data to workers, and returns the
    /// instances in vanilla ordering.
    fn stage2_collect_instances<F, ProofTranscript, PCS, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        use crate::zkvm::instruction_lookups::booleanity::Rep3BooleanitySumcheck;
        use crate::zkvm::ram::output_check::Rep3OutputSumcheck as Rep3OutputSumcheckCoord;
        use crate::zkvm::ram::raf_evaluation::Rep3RafEvaluation;
        use crate::zkvm::ram::read_write_checking::Rep3RamReadWriteChecking;
        use crate::zkvm::registers::read_write_checking::Rep3RegistersReadWriteChecking;
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::zkvm::instruction_lookups::D;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        // 1) Spartan inner sumcheck — derives gamma, input_claim from transcript;
        //    broadcasts (gamma, input_claim) to workers internally.
        let spartan_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            Rep3SpartanDag::stage2_instances(state, network)?;

        // 2) Registers read-write checking — derives gamma from transcript.
        //    Create concrete type, extract init data, broadcast, then box.
        let reg_rwc = Rep3RegistersReadWriteChecking::<F>::new::<ProofTranscript, PCS>(state);
        let reg_gamma = reg_rwc.gamma();
        let reg_input_claim = reg_rwc.input_claim();
        network.broadcast_request((reg_gamma, reg_input_claim))?;
        let registers_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            vec![Box::new(reg_rwc)];

        // 3) RAM (raf, read-write, output) — create each concrete type,
        //    extract init data from RamRWC (gamma) and OutputSumcheck (r_address),
        //    broadcast combined init bundle, then box all.
        let raf = Rep3RafEvaluation::<F>::new::<ProofTranscript, PCS>(state);
        let ram_rwc = Rep3RamReadWriteChecking::<F>::new::<ProofTranscript, PCS>(state);
        let output = Rep3OutputSumcheckCoord::<F>::new::<ProofTranscript, PCS>(state);
        let ram_gamma = ram_rwc.gamma();
        let ram_input_claim = ram_rwc.input_claim();
        let ram_r_address = output.r_address().to_vec();
        network.broadcast_request((ram_gamma, ram_input_claim, ram_r_address))?;
        let ram_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            vec![Box::new(raf), Box::new(ram_rwc), Box::new(output)];

        // 4) Lookups booleanity — derives gamma + r_address from transcript.
        let log_T = state
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .len();
        let booleanity = Rep3BooleanitySumcheck::<F>::new(&mut state.transcript, log_T);
        let lookups_gamma = booleanity.gamma();
        let lookups_r_address = booleanity.r_address().to_vec();
        network.broadcast_request((lookups_gamma, lookups_r_address))?;
        let lookups_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            vec![Box::new(booleanity)];

        // Collect all instances in vanilla order
        let stage2_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            std::iter::empty()
                .chain(spartan_instances)
                .chain(registers_instances)
                .chain(ram_instances)
                .chain(lookups_instances)
                .collect();

        Ok(stage2_instances)
    }

    /// Collect all stage3 sumcheck instances from every subsystem.
    ///
    /// Creates coordinator-side instances in vanilla ordering (spartan → registers
    /// → lookups → ram), derives transcript challenges, broadcasts init data to
    /// workers, and returns the instances for batched proving.
    fn stage3_collect_instances<F, ProofTranscript, PCS, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        use jolt_core::zkvm::instruction_lookups::D;

        // Pre-read accumulator values needed for both broadcasting and instance creation.
        let registers_val_claim = state
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RegistersVal,
                SumcheckId::RegistersReadWriteChecking,
            )
            .1;
        let product_input_claim = state
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter)
            .1;
        let val_init_eval = state
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValInit,
                SumcheckId::RamOutputCheck,
            )
            .1;
        let val_final_claim_eval = state
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;
        let ram_val_final_input_claim = val_final_claim_eval - val_init_eval;

        // === 1) Spartan: PCSumcheck (public) + ProductVirtualization (secret) ===
        // PCSumcheck draws gamma from transcript.
        let gamma_pc: F = state.transcript.challenge_scalar();
        let (r_cycle_point, next_pc_eval) = state
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, next_unexpanded_pc_eval) = state.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_noop_eval) = state.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanOuter,
        );
        let input_claim_pc = next_unexpanded_pc_eval
            + gamma_pc * next_pc_eval
            + gamma_pc.square() * next_is_noop_eval;
        let spartan_pc = PCSumcheck::<F>::new_verifier_from_openings(
            input_claim_pc,
            gamma_pc,
            r_cycle_point.r.len(),
        );
        let spartan_product = Rep3ProductVirtualizationSumcheck::<F>::new(state);

        // === 2) Registers: ValEvaluation (secret) — no transcript draw ===
        let registers_val = Rep3ValEvaluation::<F>::new(state);

        // === 3) Lookups: ReadRaf (secret) + HammingWeight (secret) ===
        // ReadRaf draws gamma from transcript. Must be created before HammingWeight.
        let (_, rv_claim) = state.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_operand_claim) = state.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let (_, right_operand_claim) = state.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let log_T = state
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .len();
        let lookup_read_raf = Rep3ReadRafSumcheck::<F>::new(
            &mut state.transcript,
            rv_claim,
            left_operand_claim,
            right_operand_claim,
            log_T,
        );
        let read_raf_gamma = lookup_read_raf.gamma();
        let read_raf_rv_claim = lookup_read_raf.rv_claim();
        let read_raf_raf_claim = lookup_read_raf.raf_claim();
        // HammingWeight draws its own gamma from transcript.
        let lookup_hamming = Rep3HammingWeightSumcheck::<F>::new(&mut state.transcript);
        let lookups_gamma: [F; D] = lookup_hamming.gamma();

        // === 4) RAM: ValEvaluation (secret) + ValFinal (secret) + HammingBooleanity (public) ===
        // ValEvaluation needs init_eval computed from public initial_ram_state.
        // No transcript draw for any of these.
        let (ram_val_eval, ram_val_eval_input_claim) = {
            use crate::zkvm::ram::{build_initial_memory_state, val_evaluation::Rep3RamValEvaluation};
            use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};

            let initial_ram_state = build_initial_memory_state(
                &state.preprocessing.shared.ram,
                &state.program_io,
                state.ram_K,
            );
            let (r_val_point, _) = state.accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            );
            let (r_address, _) = r_val_point.split_at(state.ram_K.log_2());
            let val_init_poly: MultilinearPolynomial<F> =
                MultilinearPolynomial::from(initial_ram_state);
            let ram_init_eval = val_init_poly.evaluate(&r_address.r);

            let val_eval = Rep3RamValEvaluation::<F>::new::<ProofTranscript, PCS>(state, ram_init_eval);
            let claim = val_eval.input_claim();
            (val_eval, claim)
        };
        let ram_val_final = Rep3ValFinalSumcheck::<F>::new(state);
        let log_T = state.trace_length.log_2();
        let ram_hamming_bool = jolt_core::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheck::<
            F,
        >::new_verifier_from_parts(log_T);

        // Broadcast stage3 init data to workers in four messages
        // (ark-serialize tuple impls only go up to small arities).
        network.broadcast_request((gamma_pc, input_claim_pc, product_input_claim))?;
        let lookups_gamma_vec: Vec<F> = lookups_gamma.to_vec();
        network.broadcast_request((
            registers_val_claim,
            lookups_gamma_vec,
            ram_val_final_input_claim,
            ram_val_eval_input_claim,
        ))?;
        // ReadRaf init data: gamma, rv_claim, raf_claim
        network.broadcast_request((read_raf_gamma, read_raf_rv_claim, read_raf_raf_claim))?;

        // Collect all instances in vanilla ordering:
        // spartan(PC, Product) → registers(Val) → lookups(ReadRaf, HammingWeight)
        // → ram(ValEvaluation, ValFinal, HammingBooleanity)
        let stage3_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> = vec![
            BatchedSumcheckInstance::Public(Box::new(spartan_pc)),
            BatchedSumcheckInstance::Secret(Box::new(spartan_product)),
            BatchedSumcheckInstance::Secret(Box::new(registers_val)),
            BatchedSumcheckInstance::Secret(Box::new(lookup_read_raf)),
            BatchedSumcheckInstance::Secret(Box::new(lookup_hamming)),
            BatchedSumcheckInstance::Secret(Box::new(ram_val_eval)),
            BatchedSumcheckInstance::Secret(Box::new(ram_val_final)),
            BatchedSumcheckInstance::Public(Box::new(ram_hamming_bool)),
        ];

        Ok(stage3_instances)
    }

    /// Collect all stage4 sumcheck instances from RAM, Bytecode, and Lookups.
    ///
    /// Creates coordinator-side instances (advancing the transcript in vanilla order),
    /// broadcasts init data to workers, and returns all instances for batched proving.
    ///
    /// Vanilla ordering: RAM(HammingWeight, Booleanity, Ra) → Bytecode(ReadRaf, Booleanity,
    /// HammingWeight) → Lookups(InstructionRa).
    fn stage4_collect_instances<F, ProofTranscript, PCS, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        use crate::zkvm::bytecode::Rep3BytecodeDag;
        use crate::zkvm::instruction_lookups::ra_virtual::Rep3InstructionRaSumcheck;
        use crate::zkvm::ram::Rep3RamDag;
        use jolt_core::poly::opening_proof::OpeningId;
        use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};

        // === 1) RAM: HammingWeight, Booleanity, Ra (all public) ===
        let (ram_instances, ram_init) =
            Rep3RamDag::stage4_instances_with_init::<F, ProofTranscript, PCS>(state);

        // === 2) Bytecode: ReadRaf, Booleanity, HammingWeight (all public) ===
        let (bytecode_instances, bytecode_init) =
            Rep3BytecodeDag::stage4_instances_with_init::<F, ProofTranscript, PCS>(state);

        // === 3) Lookups: InstructionRa (secret, conditional) ===
        let ra_key = OpeningId::Virtual(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        let has_ra_opening = state.accumulator.openings.contains_key(&ra_key);

        let lookups_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> = if has_ra_opening
        {
            let (ra_point, ra_claim) = state.accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa,
                SumcheckId::InstructionReadRaf,
            );
            let (r_address, r_cycle) = ra_point.r.split_at(D * LOG_K_CHUNK);
            let r_address_chunks: Vec<Vec<F::Challenge>> =
                r_address.chunks(LOG_K_CHUNK).map(|c| c.to_vec()).collect();

            let ra_coord =
                Rep3InstructionRaSumcheck::new(ra_claim, r_cycle.to_vec(), r_address_chunks);

            vec![BatchedSumcheckInstance::Secret(Box::new(ra_coord))]
        } else {
            vec![]
        };

        // Broadcast init data to workers.
        // Message 1: has_ra_opening flag
        network.broadcast_request(has_ra_opening)?;
        // Message 2: RAM init (split into two messages due to tuple size limits)
        network.broadcast_request((
            ram_init.hamming_gamma_powers,
            ram_init.hamming_input_claim,
            ram_init.bool_r_cycle,
            ram_init.bool_r_address,
            ram_init.bool_gamma_powers,
        ))?;
        network.broadcast_request((
            ram_init.ra_gamma,
            ram_init.ra_claim,
            ram_init.ra_r_cycle,
            ram_init.ra_r_address_chunks,
        ))?;
        // Message 3: Bytecode init (split into two messages)
        network.broadcast_request((
            bytecode_init.read_raf_gamma,
            bytecode_init.rv_claim,
            bytecode_init.val_polys,
            bytecode_init.r_cycles,
        ))?;
        network.broadcast_request((
            bytecode_init.bool_gamma_powers,
            bytecode_init.bool_r_address,
            bytecode_init.hw_gamma_powers,
        ))?;
        // Message 4: Lookups RA init (only if active)
        if has_ra_opening {
            let (ra_point, ra_claim) = state.accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa,
                SumcheckId::InstructionReadRaf,
            );
            let (r_address, r_cycle) = ra_point.r.split_at(D * LOG_K_CHUNK);
            network.broadcast_request((ra_claim, r_address.to_vec(), r_cycle.to_vec()))?;
        }

        let mut all_instances = Vec::new();
        all_instances.extend(ram_instances);
        all_instances.extend(bytecode_instances);
        all_instances.extend(lookups_instances);

        Ok(all_instances)
    }

    fn receive_commitments<F, PCS, ProofTranscript, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        // Receive commitment shares from all 3 parties
        // Each party sends Vec<MaybeShared<Commitment>> aligned with AllCommittedPolynomials
        let all_commitment_shares: Vec<Vec<MaybeShared<PCS::Commitment>>> =
            network.receive_responses()?;

        eyre::ensure!(
            all_commitment_shares.len() == 3,
            "expected commitment shares from 3 parties, got {}",
            all_commitment_shares.len()
        );
        let num_polys = all_commitment_shares[0].len();

        // Combine commitment shares
        let combined_commitments: Vec<PCS::Commitment> = (0..num_polys)
            .map(|i| {
                let shares: Vec<&MaybeShared<PCS::Commitment>> = all_commitment_shares
                    .iter()
                    .map(|party_shares| &party_shares[i])
                    .collect();
                <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::combine_commitment_shares(
                    &shares,
                )
            })
            .collect();

        // Store commitments and append to transcript
        state.commitments = combined_commitments;
        for commitment in &state.commitments {
            state.transcript.append_serializable(commitment);
        }

        Ok(())
    }

    /// Receive the untrusted advice commitment from all 3 workers.
    ///
    /// All workers compute the same public commitment, so we verify consistency
    /// and store the first non-None value.
    fn receive_untrusted_advice_commitment<F, PCS, ProofTranscript, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        let commitments: Vec<Option<PCS::Commitment>> = network.receive_responses()?;
        eyre::ensure!(
            commitments.len() == 3,
            "expected untrusted advice commitment from 3 parties, got {}",
            commitments.len()
        );

        // All workers should produce identical results (public computation)
        eyre::ensure!(
            commitments[0] == commitments[1] && commitments[1] == commitments[2],
            "untrusted advice commitment mismatch across parties"
        );

        state.untrusted_advice_commitment = commitments.into_iter().next().unwrap();
        Ok(())
    }
}
