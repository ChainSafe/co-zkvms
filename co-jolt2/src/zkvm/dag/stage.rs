use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;

use crate::field::JoltField;
pub use crate::subprotocols::sumcheck::{
    BatchedSumcheckWorkerInstance, PublicSumcheckInstanceWorker, Rep3SumcheckInstanceWorker,
};
use crate::zkvm::dag::state_manager::StateManagerWorker;

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
        _preproc: &mut PreprocessingPool<F>,
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
    ) -> Self {
        Self {
            outer_sumcheck_r,
            claimed_witness_evals,
            padded_trace_length,
            registers_dag: crate::zkvm::registers::Rep3RegistersDagWorker::new(),
            ram_dag: None,
            lookups_dag: None,
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

        if let Some(limit) = std::env::var("CO_JOLT2_STAGE2_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            instances.truncate(limit.min(instances.len()));
        }

        Ok(instances)
    }

    #[tracing::instrument(skip_all)]
    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
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
        let registers_stage3 = self.registers_dag.stage3_instances(sm, io_ctx, preproc)?;

        // 3) Lookups: ReadRaf (secret) + HammingWeight (secret)
        lookups_dag.set_stage3_init(
            lookups_gamma,
            read_raf_gamma,
            read_raf_rv_claim,
            read_raf_raf_claim,
        );
        let lookups_stage3 = lookups_dag.stage3_instances(sm, io_ctx, preproc);

        // 4) RAM: ValEvaluation (secret) + ValFinal (secret) + HammingBooleanity (public)
        ram_dag.set_stage3_init(ram_val_final_input_claim, ram_val_eval_input_claim);
        let ram_stage3 = ram_dag.stage3_instances(sm, io_ctx, preproc)?;

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

        if let Some(limit) = std::env::var("CO_JOLT2_STAGE3_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            stage3_instances.truncate(limit.min(stage3_instances.len()));
        }

        Ok(stage3_instances)
    }

    #[tracing::instrument(skip_all, name = "stage4_instances")]
    fn stage4_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        use crate::subprotocols::sumcheck::BatchedSumcheckWorkerInstance;
        use crate::zkvm::instruction_lookups::ra_virtual::Rep3InstructionRaSumcheckWorker;

        let ram_dag = self.ram_dag.as_mut().expect("ram_dag missing");
        let lookups_dag = self.lookups_dag.take().expect("lookups_dag missing");

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
        let one_hot_polys = lookups_dag.one_hot_polys.clone();
        drop(lookups_dag);
        let ra_worker = Rep3InstructionRaSumcheckWorker::new(
            one_hot_polys,
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
