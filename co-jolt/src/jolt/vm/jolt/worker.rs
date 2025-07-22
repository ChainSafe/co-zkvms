use crate::{
    jolt::vm::jolt::witness::JoltWitnessMeta,
    lasso::memory_checking::StructuredPolynomialData,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{
            ProverOpeningAccumulator, ReducedOpeningProof, Rep3ProverOpeningAccumulator,
        },
    },
    subprotocols::commitment::DistributedCommitmentScheme,
    utils::{thread::drop_in_background_thread, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Context;
use jolt_tracer::JoltDevice;
use mpc_core::protocols::rep3::{
    network::{IoContext, Rep3NetworkCoordinator, Rep3NetworkWorker},
    PartyID,
};
use snarks_core::math::Math;
use strum::EnumCount;

use crate::jolt::{
    instruction::{JoltInstructionSet, Rep3JoltInstructionSet},
    subtable::JoltSubtableSet,
    vm::{
        instruction_lookups::{worker::Rep3InstructionLookupsProver, InstructionLookupsProof},
        rv32i_vm::{RV32IJoltVM, RV32I},
        witness::{Rep3JoltPolynomials, Rep3JoltPolynomialsExt, Rep3Polynomials},
        Jolt, JoltCommitments, JoltPolynomials, JoltProof, JoltTraceStep,
    },
};
use jolt_core::utils::transcript::AppendToTranscript;
use jolt_core::{
    field::JoltField,
    jolt::vm::{JoltProverPreprocessing, JoltStuff, ProverDebugInfo},
    poly::multilinear_polynomial::MultilinearPolynomial,
    r1cs::inputs::ConstraintInput,
};

pub struct JoltRep3Prover<
    F,
    const C: usize,
    const M: usize,
    Instructions,
    Subtables,
    PCS,
    ProofTranscript,
    Network,
> where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    pub io_ctx: IoContext<Network>,
    pub preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
    pub polynomials: Rep3JoltPolynomials<F>,
    pub trace_length: usize,
    instruction_lookups: Rep3InstructionLookupsProver<C, M, F, Instructions, Subtables, Network>,
}

impl<F, const C: usize, const M: usize, Instructions, Subtables, PCS, ProofTranscript, Network>
    JoltRep3Prover<F, C, M, Instructions, Subtables, PCS, ProofTranscript, Network>
where
    F: JoltField,
    Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    pub fn init(
        trace: Option<Vec<JoltTraceStep<F, Instructions>>>,
        preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
        network: Network,
    ) -> eyre::Result<Self>
    where
        PCS: DistributedCommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
    {
        let mut io_ctx = IoContext::init(network).context("failed to initialize io context")?;

        let (trace_length, polynomials) = match trace {
            Some(mut trace) => {
                JoltTraceStep::pad(&mut trace);

                let polynomials = Rep3JoltPolynomials::generate_witness_rep3(
                    &preprocessing.shared,
                    &mut trace,
                    M,
                    io_ctx.fork().context("failed to fork io context")?,
                )?;

                let trace_length = trace.len();

                if io_ctx.id == PartyID::ID0 {
                    let meta = JoltWitnessMeta {
                        trace_length,
                        read_write_memory_size: polynomials.read_write_memory.v_final.len(),
                    };

                    io_ctx.network.send_response(meta)?;
                }

                (trace_length, polynomials)
            }
            None => {
                let polynomials = io_ctx.network.receive_request()?;
                let trace_length = io_ctx.network.receive_request()?;
                (trace_length, polynomials)
            }
        };

        Ok(Self {
            io_ctx,
            polynomials,
            preprocessing,
            trace_length,
            instruction_lookups: Rep3InstructionLookupsProver::new(),
        })
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    pub fn prove(&mut self) -> eyre::Result<()>
    where
        PCS: DistributedCommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
    {
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

        let mut opening_accumulator = Rep3ProverOpeningAccumulator::<F>::new();

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

        // Batch-prove all openings
        opening_accumulator.reduce_and_prove_worker::<PCS, ProofTranscript, _>(
            &preprocessing.shared.generators,
            &mut self.io_ctx,
        )?;

        Ok(())
    }

    pub fn switch_network(&mut self, network: Network) -> eyre::Result<()> {
        let io_ctx = IoContext::init(network).context("failed to initialize io context")?;
        self.io_ctx = io_ctx;
        Ok(())
    }
}
