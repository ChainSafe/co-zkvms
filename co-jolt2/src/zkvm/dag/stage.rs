use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};

use crate::field::JoltField;
pub use crate::subprotocols::sumcheck::{
    BatchedSumcheckInstance, BatchedSumcheckWorkerInstance, PublicSumcheckInstance,
    PublicSumcheckInstanceWorker, Rep3SumcheckInstance, Rep3SumcheckInstanceWorker,
};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

// ---------------------------------------------------------------------------
// Staged sumcheck pipeline traits (per-subsystem interface)
// ---------------------------------------------------------------------------

/// Worker side of the staged sumcheck pipeline.
///
/// Each subsystem DAG node (e.g. `Rep3LookupsDagWorker`)
/// implements this trait to contribute sumcheck instances from shared polynomials.
pub trait SumcheckStagesWorker<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
{
    fn stage1_prove(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Result<(), eyre::Report> {
        Ok(())
    }

    fn stage2_instances(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        Ok(vec![])
    }

    fn stage3_instances(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        Ok(vec![])
    }

    fn stage4_instances(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        Ok(vec![])
    }
}

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
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), eyre::Report> {
        Ok(())
    }

    fn stage2_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        Ok(vec![])
    }

    fn stage3_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        Ok(vec![])
    }

    fn stage4_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        Ok(vec![])
    }
}

// ---------------------------------------------------------------------------
// Top-level DAG stage instance wiring (Jolt DAG)
// ---------------------------------------------------------------------------

pub struct Rep3JoltDagStagesWorker<F: JoltField> {
    // Spartan stage1 outputs, needed for stage2 inner sumcheck.
    outer_sumcheck_r: Vec<F::Challenge>,
    claimed_witness_evals: Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>,
    padded_trace_length: usize,

    // Per-subsystem DAG workers kept across stages.
    registers_dag: crate::zkvm::registers::Rep3RegistersDagWorker<F>,
    ram_dag: Option<crate::zkvm::ram::Rep3RamDagWorker<F>>,
    lookups_dag: Option<crate::zkvm::instruction_lookups::Rep3LookupsDagWorker<F>>,

    // Stage3 needs edabits ownership to construct ReadRaf worker.
    edabits_pool: Option<mpc_core::protocols::rep3_ring::edabits::EdaBitsPool<F>>,

    // Witness-time lookup polynomials (consumed when we create lookups_dag).
    instruction_one_hot_polys: Option<
        [crate::poly::one_hot_polynomial::Rep3OneHotPolynomial<F>;
            jolt_core::zkvm::instruction_lookups::D],
    >,
}

impl<F: JoltField> Rep3JoltDagStagesWorker<F> {
    pub fn new(
        outer_sumcheck_r: Vec<F::Challenge>,
        claimed_witness_evals: Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>,
        padded_trace_length: usize,
        instruction_one_hot_polys: [crate::poly::one_hot_polynomial::Rep3OneHotPolynomial<F>;
            jolt_core::zkvm::instruction_lookups::D],
        edabits_pool: mpc_core::protocols::rep3_ring::edabits::EdaBitsPool<F>,
    ) -> Self {
        Self {
            outer_sumcheck_r,
            claimed_witness_evals,
            padded_trace_length,
            registers_dag: crate::zkvm::registers::Rep3RegistersDagWorker::new(),
            ram_dag: None,
            lookups_dag: None,
            edabits_pool: Some(edabits_pool),
            instruction_one_hot_polys: Some(instruction_one_hot_polys),
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
    SumcheckStagesWorker<F, PCS, N> for Rep3JoltDagStagesWorker<F>
where
    rand::distributions::Standard: rand::distributions::Distribution<u32>
        + rand::distributions::Distribution<u64>
        + rand::distributions::Distribution<u8>
        + rand::distributions::Distribution<u128>,
{
    #[tracing::instrument(skip_all)]
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        use crate::subprotocols::sumcheck::BatchedSumcheckWorkerInstance;
        use crate::zkvm::spartan::Rep3InnerSumcheckWorker;

        if self.ram_dag.is_none() {
            self.ram_dag = Some(crate::zkvm::ram::Rep3RamDagWorker::new(sm, io_ctx)?);
        }
        if self.lookups_dag.is_none() {
            let polys = self
                .instruction_one_hot_polys
                .take()
                .expect("instruction_one_hot_polys already consumed");
            self.lookups_dag = Some(crate::zkvm::instruction_lookups::Rep3LookupsDagWorker::new(
                polys,
            ));
        }

        let party_id = io_ctx.party_id();
        let ram_dag = self.ram_dag.as_mut().expect("ram_dag missing");
        let lookups_dag = self.lookups_dag.as_mut().expect("lookups_dag missing");

        // 1) Spartan inner sumcheck — receive (gamma, input_claim) from coordinator
        let (spartan_gamma, spartan_input_claim): (F, F) = io_ctx.network().receive_request()?;
        let inner = Rep3InnerSumcheckWorker::new(
            spartan_gamma,
            spartan_input_claim,
            &self.outer_sumcheck_r,
            std::mem::take(&mut self.claimed_witness_evals),
            self.padded_trace_length,
            party_id,
        );

        // 2) Registers read-write checking — receive (gamma, input_claim)
        let (reg_gamma, reg_input_claim): (F, F) = io_ctx.network().receive_request()?;
        self.registers_dag
            .set_stage2_init(reg_gamma, reg_input_claim);
        let registers_instances = self.registers_dag.stage2_instances(sm, io_ctx)?;

        // 3) RAM — receive (gamma, input_claim, r_address)
        let (ram_gamma, ram_input_claim, ram_r_address): (F, F, Vec<F::Challenge>) =
            io_ctx.network().receive_request()?;
        ram_dag.set_stage2_init(ram_gamma, ram_input_claim, ram_r_address);
        let ram_instances = ram_dag.stage2_instances(sm, io_ctx)?;

        // 4) Lookups booleanity — receive (gamma_powers, r_address)
        let (lookups_gamma, lookups_r_address): (
            [F; jolt_core::zkvm::instruction_lookups::D],
            Vec<F::Challenge>,
        ) = io_ctx.network().receive_request()?;
        lookups_dag.set_stage2_init(lookups_gamma, lookups_r_address);
        let lookups_instances = lookups_dag.stage2_instances(sm, io_ctx)?;

        // Collect all instances in vanilla order
        let mut instances: Vec<BatchedSumcheckWorkerInstance<F, N>> = Vec::with_capacity(
            1 + registers_instances.len() + ram_instances.len() + lookups_instances.len(),
        );
        instances.push(BatchedSumcheckWorkerInstance::Secret(Box::new(inner)));
        instances.extend(registers_instances);
        instances.extend(ram_instances);
        instances.extend(lookups_instances);
        Ok(instances)
    }

    #[tracing::instrument(skip_all)]
    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        use crate::subprotocols::sumcheck::BatchedSumcheckWorkerInstance;
        use crate::zkvm::spartan::product::Rep3ProductVirtualizationSumcheckWorker;
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::zkvm::instruction::CircuitFlags;
        use jolt_core::zkvm::spartan::pc::PCSumcheck;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        let party_id = io_ctx.party_id();
        let ram_dag = self.ram_dag.as_mut().expect("ram_dag missing");
        let lookups_dag = self.lookups_dag.as_mut().expect("lookups_dag missing");

        // Receive stage3 init data from coordinator (three messages).
        let (gamma_pc, input_claim_pc, input_claim_product): (F, F, F) =
            io_ctx.network().receive_request()?;
        let (
            registers_val_claim,
            lookups_gamma_vec,
            ram_val_final_input_claim,
            ram_val_eval_input_claim,
        ): (F, Vec<F>, F, F) = io_ctx.network().receive_request()?;
        let lookups_gamma: [F; jolt_core::zkvm::instruction_lookups::D] = lookups_gamma_vec
            .try_into()
            .map_err(|_| eyre::eyre!("lookups gamma vec has wrong length"))?;
        let (read_raf_gamma, read_raf_rv_claim, read_raf_raf_claim): (F, F, F) =
            io_ctx.network().receive_request()?;

        // 1) Spartan: PCSumcheck (public) + ProductVirtualization (secret)
        let log_T = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter)
            .0
            .r
            .len();

        let pc_sumcheck = if party_id == mpc_core::protocols::rep3::PartyID::ID0 {
            let cycle_witness = &sm.prover_state.cycle_witness;
            let unexpanded_pc_poly: jolt_core::poly::multilinear_polynomial::MultilinearPolynomial<
                F,
            > = cycle_witness.pc_sumcheck_unexpanded_pc().to_vec().into();
            let pc_indices: Vec<u64> = cycle_witness.meta().iter().map(|m| m.pc_index).collect();
            let pc_poly: jolt_core::poly::multilinear_polynomial::MultilinearPolynomial<F> =
                pc_indices.into();

            let mask = 1u32 << (CircuitFlags::IsNoop as usize);
            let is_noop: Vec<u8> = cycle_witness
                .pc_sumcheck_flags_bits()
                .iter()
                .map(|bits| ((bits & mask) != 0) as u8)
                .collect();
            let is_noop_poly: jolt_core::poly::multilinear_polynomial::MultilinearPolynomial<F> =
                is_noop.into();

            let r_cycle = sm
                .accumulator
                .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter)
                .0
                .r;
            let (_, eq_plus_one_evals) =
                jolt_core::poly::eq_poly::EqPlusOnePolynomial::<F>::evals(&r_cycle, None);
            let eq_plus_one_poly =
                jolt_core::poly::multilinear_polynomial::MultilinearPolynomial::from(
                    eq_plus_one_evals,
                );

            PCSumcheck::<F>::new_prover_from_polys(
                input_claim_pc,
                gamma_pc,
                log_T,
                unexpanded_pc_poly,
                pc_poly,
                is_noop_poly,
                eq_plus_one_poly,
            )
        } else {
            PCSumcheck::<F>::new_verifier_from_openings(input_claim_pc, gamma_pc, log_T)
        };

        // PCSumcheck inputs are fully materialized into owned polynomials above.
        sm.prover_state.cycle_witness.drop_pc_sumcheck_inputs();

        let product_sumcheck =
            Rep3ProductVirtualizationSumcheckWorker::<F>::new(sm, input_claim_product);

        // 2) Registers: ValEvaluation (secret)
        self.registers_dag.set_stage3_init(registers_val_claim);
        let registers_stage3 = self.registers_dag.stage3_instances(sm, io_ctx)?;

        // 3) Lookups: ReadRaf (secret) + HammingWeight (secret)
        lookups_dag.set_stage3_init(
            lookups_gamma,
            read_raf_gamma,
            read_raf_rv_claim,
            read_raf_raf_claim,
        );
        let edabits_pool = self
            .edabits_pool
            .take()
            .expect("edabits_pool already taken");
        let lookups_stage3 = lookups_dag.stage3_instances(sm, io_ctx, edabits_pool);

        // 4) RAM: ValEvaluation (secret) + ValFinal (secret) + HammingBooleanity (public)
        ram_dag.set_stage3_init(ram_val_final_input_claim, ram_val_eval_input_claim);
        let ram_stage3 = ram_dag.stage3_instances(sm, io_ctx)?;

        // Collect all instances in vanilla ordering:
        // spartan(PC, Product) → registers(Val) → lookups(ReadRaf, HammingWeight)
        // → ram(ValEvaluation, ValFinal, HammingBooleanity)
        let mut stage3_instances: Vec<BatchedSumcheckWorkerInstance<F, N>> = Vec::with_capacity(
            2 + registers_stage3.len() + lookups_stage3.len() + ram_stage3.len(),
        );
        stage3_instances.push(BatchedSumcheckWorkerInstance::Public(Box::new(pc_sumcheck)));
        stage3_instances.push(BatchedSumcheckWorkerInstance::Secret(Box::new(
            product_sumcheck,
        )));
        stage3_instances.extend(registers_stage3);
        stage3_instances.extend(lookups_stage3);
        stage3_instances.extend(ram_stage3);

        Ok(stage3_instances)
    }

    #[tracing::instrument(skip_all)]
    fn stage4_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        use crate::subprotocols::sumcheck::BatchedSumcheckWorkerInstance;
        use crate::zkvm::instruction_lookups::ra_virtual::Rep3InstructionRaSumcheckWorker;

        let ram_dag = self.ram_dag.as_mut().expect("ram_dag missing");
        let lookups_dag = self.lookups_dag.as_mut().expect("lookups_dag missing");

        // Single init message (bundle): RAM init + Bytecode init + Lookups RA init.
        let (ram_init, bytecode_init, ra_input_claim, ra_r_address, ra_r_cycle): (
            crate::zkvm::ram::RamStage4Init<F>,
            crate::zkvm::bytecode::BytecodeStage4Init<F>,
            F,
            Vec<F::Challenge>,
            Vec<F::Challenge>,
        ) = io_ctx.network().receive_request()?;

        ram_dag.set_stage4_init(ram_init);
        let ram_stage4 = ram_dag.stage4_instances(sm, io_ctx)?;

        let mut bytecode_dag = crate::zkvm::bytecode::Rep3BytecodeDagWorker::<F>::new();
        bytecode_dag.set_stage4_init(bytecode_init);
        let bytecode_stage4 = bytecode_dag.stage4_instances(sm, io_ctx)?;

        // Lookups RA init (always active; mirrors vanilla stage4).
        let ra_worker = Rep3InstructionRaSumcheckWorker::new(
            &lookups_dag.one_hot_polys,
            &ra_r_address,
            ra_r_cycle,
            ra_input_claim,
        );
        let lookups_stage4: Vec<BatchedSumcheckWorkerInstance<F, N>> =
            vec![BatchedSumcheckWorkerInstance::Secret(Box::new(ra_worker))];

        let mut stage4_instances: Vec<BatchedSumcheckWorkerInstance<F, N>> =
            Vec::with_capacity(ram_stage4.len() + bytecode_stage4.len() + lookups_stage4.len());
        stage4_instances.extend(ram_stage4);
        stage4_instances.extend(bytecode_stage4);
        stage4_instances.extend(lookups_stage4);
        Ok(stage4_instances)
    }
}

pub struct Rep3JoltDagStagesCoordinator;

impl<F, ProofTranscript, PCS, N> SumcheckStagesCoordinator<F, ProofTranscript, PCS, N>
    for Rep3JoltDagStagesCoordinator
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>
        + crate::poly::commitment::Rep3CommitmentScheme<F, ProofTranscript>,
    N: Rep3NetworkCoordinator,
{
    #[tracing::instrument(skip_all)]
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        use crate::zkvm::instruction_lookups::booleanity::Rep3BooleanitySumcheck;
        use crate::zkvm::ram::output_check::Rep3OutputSumcheck as Rep3OutputSumcheckCoord;
        use crate::zkvm::ram::raf_evaluation::Rep3RafEvaluation;
        use crate::zkvm::ram::read_write_checking::Rep3RamReadWriteChecking;
        use crate::zkvm::registers::read_write_checking::Rep3RegistersReadWriteChecking;
        use crate::zkvm::spartan::Rep3SpartanDag;
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        // 1) Spartan inner sumcheck (secret) — derives gamma,input_claim and broadcasts internally.
        let spartan_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            Rep3SpartanDag::stage2_instances(sm, network)?;
        let spartan_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> = spartan_instances
            .into_iter()
            .map(BatchedSumcheckInstance::Secret)
            .collect();

        // 2) Registers read-write checking — derive gamma from transcript, broadcast init.
        let reg_rwc = Rep3RegistersReadWriteChecking::<F>::new::<ProofTranscript, PCS>(sm);
        network.broadcast_request((reg_rwc.gamma(), reg_rwc.input_claim()))?;
        let registers_instances: Vec<BatchedSumcheckInstance<F, ProofTranscript>> =
            vec![BatchedSumcheckInstance::Secret(Box::new(reg_rwc))];

        // 3) RAM (raf, read-write, output) — broadcast combined init bundle (gamma + r_address).
        let raf = Rep3RafEvaluation::<F>::new::<ProofTranscript, PCS>(sm);
        let ram_rwc = Rep3RamReadWriteChecking::<F>::new::<ProofTranscript, PCS>(sm);
        let output = Rep3OutputSumcheckCoord::<F>::new::<ProofTranscript, PCS>(sm);
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

        // 4) Lookups booleanity — derives gamma + r_address from transcript, broadcast init.
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
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        use crate::zkvm::instruction_lookups::hamming_weight::Rep3HammingWeightSumcheck;
        use crate::zkvm::instruction_lookups::read_raf_checking::Rep3ReadRafSumcheck;
        use crate::zkvm::ram::output_check::Rep3ValFinalSumcheck;
        use crate::zkvm::registers::val_evaluation::Rep3ValEvaluation;
        use crate::zkvm::spartan::product::Rep3ProductVirtualizationSumcheck;
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::utils::math::Math;
        use jolt_core::zkvm::instruction_lookups::D;
        use jolt_core::zkvm::spartan::pc::PCSumcheck;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        // Pre-read accumulator values needed for broadcasting and instance creation.
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
            use crate::zkvm::ram::{
                build_initial_memory_state, val_evaluation::Rep3RamValEvaluation,
            };
            use jolt_core::poly::multilinear_polynomial::{
                MultilinearPolynomial, PolynomialEvaluation,
            };

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

        // Broadcast stage3 init data to workers (split due to tuple size limits).
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
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        use crate::zkvm::bytecode::Rep3BytecodeDag;
        use crate::zkvm::instruction_lookups::ra_virtual::Rep3InstructionRaSumcheck;
        use crate::zkvm::ram::Rep3RamDag;
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
        use jolt_core::zkvm::witness::VirtualPolynomial;

        // === 1) RAM: HammingWeight, Booleanity, Ra (all public) ===
        let (ram_instances, ram_init) =
            Rep3RamDag::stage4_instances_with_init::<F, ProofTranscript, PCS>(state);

        // === 2) Bytecode: ReadRaf, Booleanity, HammingWeight (all public) ===
        let (bytecode_instances, bytecode_init) =
            Rep3BytecodeDag::stage4_instances_with_init::<F, ProofTranscript, PCS>(state);

        // === 3) Lookups: InstructionRa (secret, always present in vanilla stage4) ===
        eyre::ensure!(
            state
                .accumulator
                .openings
                .contains_key(&jolt_core::poly::opening_proof::OpeningId::Virtual(
                    VirtualPolynomial::InstructionRa,
                    SumcheckId::InstructionReadRaf
                )),
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

        // Broadcast init data to workers (single bundled message).
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
