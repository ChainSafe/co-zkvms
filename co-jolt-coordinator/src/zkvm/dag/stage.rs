use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

use jolt_core::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
pub use crate::subprotocols::sumcheck::{
    BatchedSumcheckInstance, PublicSumcheckInstance, Rep3SumcheckInstance,
};
use crate::zkvm::dag::state_manager::StateManager;

// ---------------------------------------------------------------------------
// Staged sumcheck pipeline trait (coordinator side)
// ---------------------------------------------------------------------------

/// Coordinator side of the staged sumcheck pipeline.
///
/// Each subsystem DAG node (e.g. `Rep3LookupsDag`)
/// implements this trait to drive sumcheck rounds via the Fiat-Shamir transcript.
pub trait SumcheckStagesCoordinator<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkCoordinator,
>
{
    fn stage1_prove(
        &mut self,
        _sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), eyre::Report> {
        Ok(())
    }

    fn stage2_instances(
        &mut self,
        _sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        Ok(vec![])
    }

    fn stage3_instances(
        &mut self,
        _sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        Ok(vec![])
    }

    fn stage4_instances(
        &mut self,
        _sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        Ok(vec![])
    }
}

// ---------------------------------------------------------------------------
// Top-level DAG stage instance wiring (Jolt DAG) - Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3JoltDagStages;

impl<F, ProofTranscript, PCS, N> SumcheckStagesCoordinator<F, ProofTranscript, PCS, N>
    for Rep3JoltDagStages
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
    N: Rep3NetworkCoordinator,
{
    #[tracing::instrument(skip_all)]
    fn stage2_instances(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        use crate::zkvm::instruction_lookups::booleanity::Rep3BooleanitySumcheck;
        use crate::zkvm::ram::output_check::Rep3OutputSumcheck;
        use crate::zkvm::ram::raf_evaluation::Rep3RafEvaluation;
        use crate::zkvm::ram::read_write_checking::Rep3RamReadWriteChecking;
        use crate::zkvm::registers::read_write_checking::Rep3RegistersReadWriteChecking;
        use crate::zkvm::spartan::Rep3SpartanDag;

        // 1) Spartan inner sumcheck (secret)
        let spartan_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            Rep3SpartanDag::stage2_instances(sm, network)?;
        let spartan_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> = spartan_instances
            .into_iter()
            .map(BatchedSumcheckInstance::Secret)
            .collect();

        // 2) Registers read-write checking
        let reg_rwc = Rep3RegistersReadWriteChecking::<F>::new::<ProofTranscript, PCS>(sm);
        network.broadcast_request((reg_rwc.gamma(), reg_rwc.input_claim()))?;
        let registers_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> =
            vec![BatchedSumcheckInstance::Secret(Box::new(reg_rwc))];

        // 3) RAM (raf, read-write, output)
        let raf = Rep3RafEvaluation::<F>::new::<ProofTranscript, PCS>(sm);
        let ram_rwc = Rep3RamReadWriteChecking::<F>::new::<ProofTranscript, PCS>(sm);
        let output = Rep3OutputSumcheck::<F>::new::<ProofTranscript, PCS>(sm);
        network.broadcast_request((
            ram_rwc.gamma(),
            ram_rwc.input_claim(),
            output.r_address().to_vec(),
        ))?;
        let ram_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> = vec![
            BatchedSumcheckInstance::Secret(Box::new(raf)),
            BatchedSumcheckInstance::Secret(Box::new(ram_rwc)),
            BatchedSumcheckInstance::Secret(Box::new(output)),
        ];

        // 4) Lookups booleanity
        let log_T = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .len();
        let booleanity = Rep3BooleanitySumcheck::<F>::new(&mut sm.transcript, log_T);
        network.broadcast_request((booleanity.gamma(), booleanity.r_address().to_vec()))?;
        let lookups_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> =
            vec![BatchedSumcheckInstance::Secret(Box::new(booleanity))];

        // Vanilla ordering: spartan → registers → ram → lookups
        let mut stage2_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> =
            Vec::with_capacity(
                spartan_instances.len()
                    + registers_instances.len()
                    + ram_instances.len()
                    + lookups_instances.len(),
            );
        stage2_instances.extend(spartan_instances);
        stage2_instances.extend(registers_instances);
        stage2_instances.extend(ram_instances);
        stage2_instances.extend(lookups_instances);

        Ok(stage2_instances)
    }

    #[tracing::instrument(skip_all)]
    fn stage3_instances(
        &mut self,
        state: &mut StateManager<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::utils::math::Math;
        use jolt_core::zkvm::instruction_lookups::D;
        use jolt_core::zkvm::spartan::pc::PCSumcheck;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        use crate::zkvm::instruction_lookups::hamming_weight::Rep3HammingWeightSumcheck;
        use crate::zkvm::instruction_lookups::read_raf_checking::Rep3ReadRafSumcheck;
        use crate::zkvm::ram::output_check::Rep3ValFinalSumcheck;
        use crate::zkvm::registers::val_evaluation::Rep3ValEvaluation;
        use crate::zkvm::spartan::product::Rep3ProductVirtualizationSumcheck;

        // Pre-read accumulator values
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

        // === 2) Registers: ValEvaluation (secret) ===
        let registers_val = Rep3ValEvaluation::<F>::new(state);

        // === 3) Lookups: ReadRaf (secret) + HammingWeight (secret) ===
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
        let lookup_hamming = Rep3HammingWeightSumcheck::<F>::new(&mut state.transcript);
        let lookups_gamma: [F; D] = lookup_hamming.gamma();

        // === 4) RAM: ValEvaluation (secret) + ValFinal (secret) + HammingBooleanity (public) ===
        let (ram_val_eval, ram_val_eval_input_claim) = {
            use jolt_core::poly::multilinear_polynomial::{
                MultilinearPolynomial, PolynomialEvaluation,
            };

            use crate::zkvm::ram::{build_initial_memory_state, val_evaluation::Rep3RamValEvaluation};

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

            let val_eval =
                Rep3RamValEvaluation::<F>::new::<ProofTranscript, PCS>(state, ram_init_eval);
            let claim = val_eval.input_claim();
            (val_eval, claim)
        };
        let ram_val_final = Rep3ValFinalSumcheck::<F>::new(state);
        let log_T = state.trace_length.log_2();
        let ram_hamming_bool = jolt_core::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheck::<
            F,
        >::new_verifier_from_parts(log_T);

        // Broadcast stage3 init data to workers
        network.broadcast_request((gamma_pc, input_claim_pc, product_input_claim))?;
        network.broadcast_request((
            registers_val_claim,
            lookups_gamma.to_vec(),
            ram_val_final_input_claim,
            ram_val_eval_input_claim,
        ))?;
        network.broadcast_request((read_raf_gamma, read_raf_rv_claim, read_raf_raf_claim))?;

        Ok(vec![
            BatchedSumcheckInstance::Public(Box::new(spartan_pc)),
            BatchedSumcheckInstance::Secret(Box::new(spartan_product)),
            BatchedSumcheckInstance::Secret(Box::new(registers_val)),
            BatchedSumcheckInstance::Secret(Box::new(lookup_read_raf)),
            BatchedSumcheckInstance::Secret(Box::new(lookup_hamming)),
            BatchedSumcheckInstance::Secret(Box::new(ram_val_eval)),
            BatchedSumcheckInstance::Secret(Box::new(ram_val_final)),
            BatchedSumcheckInstance::Public(Box::new(ram_hamming_bool)),
        ])
    }

    #[tracing::instrument(skip_all)]
    fn stage4_instances(
        &mut self,
        state: &mut StateManager<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
        use jolt_core::zkvm::witness::VirtualPolynomial;

        use crate::zkvm::bytecode::Rep3BytecodeDag;
        use crate::zkvm::instruction_lookups::ra_virtual::Rep3InstructionRaSumcheck;
        use crate::zkvm::ram::Rep3RamDag;

        // === 1) RAM: HammingWeight, Booleanity, Ra (all public) ===
        let (ram_instances, ram_init) =
            Rep3RamDag::stage4_instances_with_init::<F, ProofTranscript, PCS>(state);

        // === 2) Bytecode: ReadRaf, Booleanity, HammingWeight (all public) ===
        let (bytecode_instances, bytecode_init) =
            Rep3BytecodeDag::stage4_instances_with_init::<F, ProofTranscript, PCS>(state);

        // === 3) Lookups: InstructionRa (secret) ===
        eyre::ensure!(
            state.accumulator.openings.contains_key(
                &jolt_core::poly::opening_proof::OpeningId::Virtual(
                    VirtualPolynomial::InstructionRa,
                    SumcheckId::InstructionReadRaf
                )
            ),
            "missing InstructionRa opening (expected from stage3 ReadRaf)"
        );
        let (ra_point, ra_claim) = state.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        let (r_address, r_cycle) = ra_point.r.split_at(D * LOG_K_CHUNK);
        let r_address_chunks: Vec<Vec<F::Challenge>> =
            r_address.chunks(LOG_K_CHUNK).map(|c| c.to_vec()).collect();
        let ra_coord = Rep3InstructionRaSumcheck::new(ra_claim, r_cycle.to_vec(), r_address_chunks);
        let lookups_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> =
            vec![BatchedSumcheckInstance::Secret(Box::new(ra_coord))];

        // Broadcast init data to workers
        network.broadcast_request((
            ram_init,
            bytecode_init,
            ra_claim,
            r_address.to_vec(),
            r_cycle.to_vec(),
        ))?;

        let mut all_instances = Vec::new();
        all_instances.extend(ram_instances);
        all_instances.extend(bytecode_instances);
        all_instances.extend(lookups_instances);
        Ok(all_instances)
    }
}
