use crate::{
    jolt::vm::{
        bytecode::worker::Rep3BytecodeProver,
        jolt::witness::JoltWitnessMeta,
        read_write_memory::{
            witness::{Rep3ProgramIO, Rep3ProgramIOInput},
            worker::Rep3ReadWriteMemoryProver,
        },
    },
    lasso::memory_checking::worker::MemoryCheckingProverRep3Worker,
    poly::{commitment::Rep3CommitmentScheme, opening_proof::Rep3OpeningAccumulatorWorker},
    r1cs::{
        builder::CombinedUniformBuilder, constraints::R1CSConstraints,
        spartan::worker::Rep3UniformSpartanProver,
    },
    utils::transcript::{Transcript, TranscriptExt},
};
use mpc_core::protocols::rep3::{
    network::{IoContextPool, Rep3NetworkWorker},
    PartyID,
};
use mpc_net::topology::MpcRingNetWorkerExt;
use snarks_core::math::Math;

use crate::field::JoltField;
use crate::jolt::{
    instruction::Rep3JoltInstructionSet,
    vm::{
        instruction_lookups::worker::Rep3InstructionLookupsProver,
        witness::{Rep3JoltPolynomials, Rep3JoltPolynomialsExt, Rep3Polynomials},
        JoltTraceStep,
    },
};
use jolt_core::{
    jolt::subtable::JoltSubtableSet, jolt::vm::JoltProverPreprocessing,
    r1cs::key::UniformSpartanKey,
};

pub struct JoltRep3Prover<
    F,
    const C: usize,
    const M: usize,
    Instructions,
    Subtables,
    Constraints,
    PCS,
    ProofTranscript,
    Network,
> where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    Constraints: R1CSConstraints<C, F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    pub io_ctx: IoContextPool<Network>,
    pub preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
    pub polynomials: Rep3JoltPolynomials<F>,
    pub r1cs_builder: CombinedUniformBuilder<C, F, Constraints::Inputs>,
    pub spartan_key: UniformSpartanKey<C, <Constraints as R1CSConstraints<C, F>>::Inputs, F>,
    pub program_io: Rep3ProgramIO<F>,
    pub padded_trace_length: usize,
    _instruction_lookups: Rep3InstructionLookupsProver<C, M, F, Instructions, Subtables, Network>,
    _spartan_prover:
        Rep3UniformSpartanProver<F, PCS, ProofTranscript, Constraints::Inputs, Network>,
}

impl<
        F,
        const C: usize,
        const M: usize,
        Instructions,
        Subtables,
        Constraints,
        PCS,
        ProofTranscript,
        Network,
    > JoltRep3Prover<F, C, M, Instructions, Subtables, Constraints, PCS, ProofTranscript, Network>
where
    F: JoltField,
    Instructions: Rep3JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    Constraints: R1CSConstraints<C, F>,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker + MpcRingNetWorkerExt,
{
    pub fn init(
        mut trace: Vec<JoltTraceStep<Instructions>>,
        program_io: Rep3ProgramIOInput,
        preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
        network: Network,
    ) -> eyre::Result<Self>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
    {
        let mut io_ctx = IoContextPool::init(network, rayon::current_num_threads() as u32)?;

        let _guard = tracing::info_span!(
            "JoltRep3Prover::init",
            worker = io_ctx.worker_idx(),
            party = io_ctx.party_idx()
        )
        .entered();

        JoltTraceStep::pad(&mut trace);
        let padded_trace_length = trace.len() / io_ctx.num_workers();

        let memory_layout = program_io.memory_layout;

        let program_io =
            Rep3ProgramIO::<F>::generate_witness_rep3(program_io, &trace, &mut io_ctx)?;

        let mut polynomials = Rep3JoltPolynomials::generate_witness_rep3(
            &preprocessing.shared,
            &mut trace,
            &program_io,
            M,
            &mut io_ctx,
        )?;

        let r1cs_builder = Constraints::construct_constraints(
            padded_trace_length,
            program_io.memory_layout.input_start,
        );
        let spartan_key = UniformSpartanKey::from(&r1cs_builder);

        r1cs_builder.compute_aux(&mut polynomials, &mut io_ctx)?;

        assert_eq!(
            polynomials.instruction_lookups.dim[0].len(),
            padded_trace_length
        );
        // assert_eq!(
        //     polynomials.read_write_memory.a_ram.len(),
        //     padded_trace_length
        // );
        // assert_eq!(polynomials.bytecode.a_read_write.len(), padded_trace_length);

        if io_ctx.party_id() == PartyID::ID0 {
            let meta = JoltWitnessMeta {
                padded_trace_length,
                read_write_memory_size: polynomials.read_write_memory.v_final.len(),
                memory_layout,
            };

            io_ctx.network().send_response(meta)?;
        }

        Ok(Self {
            io_ctx,
            polynomials,
            program_io,
            preprocessing,
            padded_trace_length,
            r1cs_builder,
            spartan_key,
            _instruction_lookups: Rep3InstructionLookupsProver::new(),
            _spartan_prover: Rep3UniformSpartanProver::new(),
        })
    }

    #[tracing::instrument(skip_all, name = "JoltRep3Prover::prove", fields(worker = self.io_ctx.worker_idx(), party = self.io_ctx.party_idx()))]
    pub fn prove(&mut self) -> eyre::Result<()>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: TranscriptExt,
    {
        self.io_ctx.sync_with_coordinator()?;
        let preprocessing = &mut self.preprocessing;
        let polynomials = &mut self.polynomials;

        let srs_size = PCS::srs_size(&preprocessing.shared.generators);

        if self.padded_trace_length > srs_size {
            return Err(eyre::eyre!(
                "Padded trace length {} (2^{}) exceeds SRS size {srs_size} (2^{}). Consider increasing the max_trace_length.",
                self.padded_trace_length, self.padded_trace_length.log_2(), srs_size.log_2()
            ));
        }

        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        polynomials
            .commit::<C, PCS, ProofTranscript, _>(&preprocessing.shared, &mut self.io_ctx)?;

        self.io_ctx.sync_with_coordinator()?;

        let mut opening_accumulator = Rep3OpeningAccumulatorWorker::<F>::new();

        // let span = tracing::span!(tracing::Level::INFO, "Rep3BytecodeProver::prove");
        // let _guard = span.enter();
        // Rep3BytecodeProver::<F, PCS, ProofTranscript, Network>::prove_memory_checking(
        //     &preprocessing.shared.generators,
        //     &preprocessing.shared.bytecode,
        //     &polynomials.bytecode,
        //     &polynomials,
        //     &mut opening_accumulator,
        //     &mut self.io_ctx,
        // )?;
        // drop(_guard);
        // drop(span);

        // self.io_ctx.sync_with_parties()?;

        Rep3InstructionLookupsProver::<C, M, F, Instructions, Subtables, Network>::prove::<
            PCS,
            ProofTranscript,
        >(
            &preprocessing.shared.instruction_lookups,
            polynomials,
            &mut opening_accumulator,
            &preprocessing.shared.generators,
            &mut self.io_ctx,
        )?;

        // self.io_ctx.sync_with_parties()?;

        // Rep3ReadWriteMemoryProver::<F, PCS, ProofTranscript, Network>::prove(
        //     &preprocessing.shared.generators,
        //     &preprocessing.shared.read_write_memory,
        //     polynomials,
        //     &self.program_io,
        //     &mut opening_accumulator,
        //     &mut self.io_ctx,
        // )?;

        // self.io_ctx.sync_with_parties()?;

        // Rep3UniformSpartanProver::<F, PCS, ProofTranscript, Constraints::Inputs, Network>::prove(
        //     &self.r1cs_builder,
        //     &self.spartan_key,
        //     polynomials,
        //     &mut opening_accumulator,
        //     &mut self.io_ctx,
        // )?;

        // self.io_ctx.sync_with_parties()?;

        // Batch-prove all openings
        opening_accumulator.reduce_and_prove::<PCS, ProofTranscript, _>(
            &preprocessing.shared.generators,
            &mut self.io_ctx,
        )?;

        Ok(())
    }

    // pub fn switch_network(&mut self, network: Network) -> eyre::Result<()> {
    //     let io_ctx = IoContext::init(network).context("failed to initialize io context")?;
    //     self.io_ctx = io_ctx;
    //     Ok(())
    // }
}
