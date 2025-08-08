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
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, Rep3CommitmentScheme},
        opening_proof::Rep3ProverOpeningAccumulator,
    },
    r1cs::spartan::worker::Rep3UniformSpartanProver,
    utils::transcript::{Transcript, TranscriptExt},
};
use eyre::Context;
use mpc_core::protocols::rep3::{
    network::{IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker},
    PartyID,
};
use snarks_core::math::Math;

use crate::jolt::{
    instruction::{JoltInstructionSet, Rep3JoltInstructionSet},
    vm::{
        instruction_lookups::worker::Rep3InstructionLookupsProver,
        witness::{Rep3JoltPolynomials, Rep3JoltPolynomialsExt, Rep3Polynomials},
        Jolt, JoltTraceStep,
    },
};
use jolt_core::r1cs::constraints::R1CSConstraints;
use jolt_core::{
    field::JoltField,
    jolt::subtable::JoltSubtableSet,
    jolt::vm::JoltProverPreprocessing,
    r1cs::{builder::CombinedUniformBuilder, key::UniformSpartanKey},
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
    pub trace_length: usize,
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
    Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    Constraints: R1CSConstraints<C, F>,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    #[tracing::instrument(skip_all, name = "JoltRep3Prover::init")]
    pub fn init(
        witness: Option<(Vec<JoltTraceStep<F, Instructions>>, Rep3ProgramIOInput<F>)>,
        preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
        network: Network,
    ) -> eyre::Result<Self>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
    {
        let num_workers = 1 << network.log_num_workers_per_party();
        let mut io_ctx = IoContextPool::init(network, rayon::current_num_threads() / num_workers)?;

        let generate_witness = witness.is_some();

        let (mut polynomials, program_io, trace_length) = match witness {
            Some((mut trace, program_io)) => {
                JoltTraceStep::pad(&mut trace);
                let memory_layout = program_io.memory_layout;

                let polynomials = Rep3JoltPolynomials::generate_witness_rep3(
                    &preprocessing.shared,
                    &mut trace,
                    M,
                    io_ctx.fork().context("failed to fork io context")?,
                )?;

                let program_io = Rep3ProgramIO::<F>::generate_witness_rep3(
                    &preprocessing.shared.read_write_memory,
                    program_io,
                    io_ctx.main(),
                )?;

                let trace_length = trace.len();
                let padded_trace_length = trace_length.next_power_of_two();
                assert_eq!(
                    polynomials.instruction_lookups.dim[0].len(),
                    padded_trace_length
                );
                assert_eq!(
                    polynomials.read_write_memory.a_ram.len(),
                    padded_trace_length
                );
                assert_eq!(polynomials.bytecode.a_read_write.len(), padded_trace_length);

                if io_ctx.id == PartyID::ID0 {
                    let meta = JoltWitnessMeta {
                        padded_trace_length,
                        read_write_memory_size: polynomials.read_write_memory.v_final.len(),
                        memory_layout,
                    };

                    io_ctx.network().send_response(meta)?;
                }

                (polynomials, program_io, trace_length)
            }
            None => tracing::trace_span!("recieve_witness_polys")
                .in_scope(|| io_ctx.network().receive_request())?,
        };
        let r1cs_builder = Constraints::construct_constraints(
            trace_length.next_power_of_two(),
            program_io.memory_layout.input_start,
        );
        let spartan_key = UniformSpartanKey::from_builder(&r1cs_builder);

        if generate_witness {
            polynomials.compute_aux::<C, Constraints::Inputs>(&r1cs_builder);
        }

        Ok(Self {
            io_ctx,
            polynomials,
            program_io,
            preprocessing,
            trace_length,
            r1cs_builder,
            spartan_key,
            _instruction_lookups: Rep3InstructionLookupsProver::new(),
            _spartan_prover: Rep3UniformSpartanProver::new(),
        })
    }

    #[tracing::instrument(skip_all, name = "JoltRep3Prover::prove")]
    pub fn prove(&mut self) -> eyre::Result<()>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: TranscriptExt,
    {
        self.io_ctx.sync_with_coordinator()?;
        let preprocessing = &self.preprocessing;
        let polynomials = &mut self.polynomials;

        let trace_length = self.trace_length;
        let padded_trace_length = trace_length.next_power_of_two();
        let srs_size = PCS::srs_size(&preprocessing.shared.generators);
        let padded_log2 = padded_trace_length.log_2();
        let srs_log2 = srs_size.log_2();

        // println!(
        //     "Trace length: {trace_length} (2^{})",
        //     trace_length.next_power_of_two().log_2()
        // );

        if padded_trace_length > srs_size {
            panic!(
                "Padded trace length {padded_trace_length} (2^{padded_log2}) exceeds SRS size {srs_size} (2^{srs_log2}). Consider increasing the max_trace_length."
            );
        }

        // F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        polynomials
            .commit::<C, PCS, ProofTranscript, _>(&preprocessing.shared, &mut self.io_ctx)?;

        self.io_ctx.sync_with_coordinator()?;

        let mut opening_accumulator = Rep3ProverOpeningAccumulator::<F>::new();

        let span = tracing::span!(tracing::Level::INFO, "Rep3BytecodeProver::prove");
        let _guard = span.enter();
        Rep3BytecodeProver::<F, PCS, ProofTranscript, Network>::prove_memory_checking(
            &preprocessing.shared.generators,
            &preprocessing.shared.bytecode,
            &polynomials.bytecode,
            &polynomials,
            &mut opening_accumulator,
            &mut self.io_ctx,
        )?;
        drop(_guard);
        drop(span);

        self.io_ctx.sync_with_parties()?;

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

        self.io_ctx.sync_with_parties()?;

        Rep3ReadWriteMemoryProver::<F, PCS, ProofTranscript, Network>::prove(
            &preprocessing.shared.generators,
            &preprocessing.shared.read_write_memory,
            polynomials,
            &self.program_io,
            &mut opening_accumulator,
            &mut self.io_ctx,
        )?;

        self.io_ctx.sync_with_parties()?;

        Rep3UniformSpartanProver::<F, PCS, ProofTranscript, Constraints::Inputs, Network>::prove(
            &self.r1cs_builder,
            &self.spartan_key,
            polynomials,
            &mut opening_accumulator,
            &mut self.io_ctx,
        )?;

        self.io_ctx.sync_with_parties()?;

        // Batch-prove all openings
        opening_accumulator.reduce_and_prove_worker::<PCS, ProofTranscript, _>(
            &preprocessing.shared.generators,
            self.io_ctx.main(),
        )?;

        Ok(())
    }

    // pub fn switch_network(&mut self, network: Network) -> eyre::Result<()> {
    //     let io_ctx = IoContext::init(network).context("failed to initialize io context")?;
    //     self.io_ctx = io_ctx;
    //     Ok(())
    // }
}
