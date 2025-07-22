use crate::lasso::memory_checking::StructuredPolynomialData;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::{Rep3MultilinearPolynomial, Rep3PolysConversion};
use crate::subprotocols::commitment::DistributedCommitmentScheme;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::{multizip, Itertools};
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::instruction_lookups::{
    InstructionLookupCommitments, InstructionLookupStuff,
};
use jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryStuff;
use jolt_core::jolt::vm::timestamp_range_check::{
    TimestampRangeCheckPolynomials, TimestampRangeCheckStuff,
};
use jolt_core::jolt::vm::{JoltCommitments, JoltPolynomials, JoltStuff, JoltVerifierPreprocessing};
use jolt_core::lasso::memory_checking::Initializable;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use mpc_core::protocols::rep3::network::{
    IoContext, Rep3Network, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::PartyID;
use rand::Rng;

use crate::jolt::instruction::{JoltInstructionSet, Rep3JoltInstructionSet};
use crate::jolt::vm::instruction_lookups::witness::Rep3InstructionLookupPolynomials;
use crate::jolt::vm::JoltTraceStep;

#[derive(Debug, Clone, Copy, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltWitnessMeta {
    pub trace_length: usize,
    pub read_write_memory_size: usize,
}

pub type Rep3JoltPolynomials<F: JoltField> = JoltStuff<Rep3MultilinearPolynomial<F>>;

pub trait Rep3Polynomials<F: JoltField, Preprocessing>: Sized {
    type PublicPolynomials;

    fn combine_polynomials(
        preprocessing: &Preprocessing,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials>;

    fn generate_secret_shares<R: Rng>(
        preprocessing: &Preprocessing,
        polynomials: &Self::PublicPolynomials,
        rng: &mut R,
    ) -> Vec<Self>;

    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &Preprocessing,
        ops: &mut [JoltTraceStep<F, Instructions>],
        M: usize,
        network: IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
        Network: Rep3Network;
}

impl<F: JoltField, const C: usize, PCS, ProofTranscript>
    Rep3Polynomials<F, JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>>
    for Rep3JoltPolynomials<F>
where
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
{
    type PublicPolynomials = JoltPolynomials<F>;

    fn combine_polynomials(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials> {
        let instructions_shares: Vec<_> = polynomials_shares
            .into_iter()
            .map(|p| {
                let Rep3JoltPolynomials {
                    instruction_lookups,
                    ..
                } = p;
                instruction_lookups
            })
            .collect();

        let instruction_lookups = Rep3InstructionLookupPolynomials::combine_polynomials(
            &preprocessing.instruction_lookups,
            instructions_shares,
        )?;

        Ok(JoltPolynomials {
            instruction_lookups,
            ..Default::default()
        })
    }

    fn generate_secret_shares<R: Rng>(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        polynomials: &Self::PublicPolynomials,
        rng: &mut R,
    ) -> Vec<Self> {
        let instruction_lookups = Rep3InstructionLookupPolynomials::generate_secret_shares(
            &preprocessing.instruction_lookups,
            &polynomials.instruction_lookups,
            rng,
        );

        let jolt_polys_shares = instruction_lookups
            .into_iter()
            .map(|instruction_lookups| Self {
                instruction_lookups,
                ..Default::default()
            })
            .collect_vec();

        jolt_polys_shares
    }

    #[tracing::instrument(skip_all, name = "Rep3JoltPolynomials::generate_witness_rep3")]
    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        ops: &mut [JoltTraceStep<F, Instructions>],
        M: usize,
        io_ctx: IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
        Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
        Network: Rep3Network,
    {
        let instruction_lookups = Rep3InstructionLookupPolynomials::generate_witness_rep3(
            &preprocessing.instruction_lookups,
            ops,
            M,
            io_ctx,
        )?;

        Ok(Self {
            instruction_lookups,
            ..Default::default()
        })
    }
}

// pub struct Rep3JoltTraceStep<F: JoltField, InstructionSet: Rep3JoltInstructionSet<F>> {
//     pub instruction_lookup: Option<InstructionSet<F>>,
//     // pub bytecode_row: BytecodeRow,
//     // pub memory_ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION],
//     // pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
//     // _field: PhantomData<F>,
// }

pub trait Rep3JoltPolynomialsExt<F: JoltField> {
    fn commit<const C: usize, PCS, ProofTranscript, Network>(
        &self,
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()>
    where
        PCS: DistributedCommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkWorker;

    fn receive_commitments<const C: usize, PCS, ProofTranscript, Network>(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        network: &mut Network,
    ) -> eyre::Result<JoltCommitments<PCS, ProofTranscript>>
    where
        PCS: DistributedCommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkCoordinator,
    {
        let mut commitments = JoltCommitments::<PCS, ProofTranscript>::initialize(preprocessing);

        let instruction_commitments = &mut commitments.instruction_lookups;

        let [share1, share2, share3] = network
            .receive_responses(InstructionLookupCommitments::<PCS, ProofTranscript>::default())?
            .try_into()
            .map_err(|_| eyre::eyre!("failed to receive commitments"))?;

        // lookup flag polys commitment are not secret shared
        let lookup_flag_polys_commitments: Vec<PCS::Commitment> =
            network.receive_response(PartyID::ID0, 0, Default::default())?;

        let trace_commitments = multizip((
            read_write_values_except_flags(&share1),
            read_write_values_except_flags(&share2),
            read_write_values_except_flags(&share3),
        ))
        .map(|(trace1, trace2, trace3)| PCS::combine_commitment_shares(&[trace1, trace2, trace3]))
        .collect_vec();

        instruction_commitments
            .dim
            .iter_mut()
            .chain(instruction_commitments.read_cts.iter_mut())
            .chain(instruction_commitments.E_polys.iter_mut())
            .chain([&mut instruction_commitments.lookup_outputs])
            .chain(instruction_commitments.instruction_flags.iter_mut())
            .zip(
                trace_commitments
                    .into_iter()
                    .chain(lookup_flag_polys_commitments),
            )
            .for_each(|(dest, src)| *dest = src);

        instruction_commitments.final_cts =
            multizip((share1.final_cts, share2.final_cts, share3.final_cts))
                .map(|(final1, final2, final3)| {
                    PCS::combine_commitment_shares(&[&final1, &final2, &final3])
                })
                .collect_vec();

        Ok(commitments)
    }

    fn get_timestamp_range_check_polynomials(&mut self) -> TimestampRangeCheckPolynomials<F>;

    fn get_exogenous_polynomials_for_timestamp_range_check(&mut self) -> JoltPolynomials<F>;
}

impl<F: JoltField> Rep3JoltPolynomialsExt<F> for Rep3JoltPolynomials<F> {
    fn commit<const C: usize, PCS, ProofTranscript, Network>(
        &self,
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()>
    where
        PCS: DistributedCommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkWorker,
    {
        let mut commitments = JoltCommitments::<PCS, ProofTranscript>::initialize(preprocessing);

        let trace_polys = read_write_values_except_flags(&self.instruction_lookups)
            .map(|poly| MultilinearPolynomial::LargeScalars(poly.as_shared().copy_share_a()))
            .collect_vec();

        let trace_polys_ref = trace_polys.iter().collect::<Vec<_>>();

        let trace_commitments = PCS::batch_commit(&trace_polys_ref, &preprocessing.generators);

        read_write_values_except_flags_mut(&mut commitments.instruction_lookups)
            .zip(trace_commitments.into_iter())
            .for_each(|(dest, src)| *dest = src);

        commitments.instruction_lookups.final_cts = PCS::batch_commit(
            &self
                .instruction_lookups
                .final_cts
                .iter()
                .map(|poly| MultilinearPolynomial::LargeScalars(poly.as_shared().copy_share_a()))
                .collect_vec(),
            &preprocessing.generators,
        );

        io_ctx
            .network
            .send_response(commitments.instruction_lookups)?;

        if io_ctx.id == PartyID::ID0 {
            let lookup_flag_polys_commitment = PCS::batch_commit(
                &self.instruction_lookups.instruction_flags.try_into_public(),
                &preprocessing.generators,
            );
            io_ctx.network.send_response(lookup_flag_polys_commitment)?;
        }

        Ok(())
    }

    fn get_timestamp_range_check_polynomials(&mut self) -> TimestampRangeCheckPolynomials<F> {
        let TimestampRangeCheckStuff {
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
            identity,
        } = std::mem::take(&mut self.timestamp_range_check);

        let read_cts_read_timestamp = read_cts_read_timestamp.map(|poly| poly.try_into().unwrap());
        let read_cts_global_minus_read =
            read_cts_global_minus_read.map(|poly| poly.try_into().unwrap());
        let final_cts_read_timestamp =
            final_cts_read_timestamp.map(|poly| poly.try_into().unwrap());
        let final_cts_global_minus_read =
            final_cts_global_minus_read.map(|poly| poly.try_into().unwrap());

        let identity = identity.map(|poly| poly.try_into().unwrap());
        TimestampRangeCheckPolynomials {
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
            identity,
        }
    }

    fn get_exogenous_polynomials_for_timestamp_range_check(&mut self) -> JoltPolynomials<F> {
        let t_read_rd = std::mem::take(&mut self.read_write_memory.t_read_rd)
            .try_into()
            .unwrap();
        let t_read_rs1 = std::mem::take(&mut self.read_write_memory.t_read_rs1)
            .try_into()
            .unwrap();
        let t_read_rs2 = std::mem::take(&mut self.read_write_memory.t_read_rs2)
            .try_into()
            .unwrap();
        let t_final = std::mem::take(&mut self.read_write_memory.t_read_ram)
            .try_into()
            .unwrap();

        JoltPolynomials {
            read_write_memory: ReadWriteMemoryStuff {
                t_read_rd,
                t_read_rs1,
                t_read_rs2,
                t_final,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

fn read_write_values_except_flags<T>(stuff: &InstructionLookupStuff<T>) -> impl Iterator<Item = &T>
where
    T: CanonicalSerialize + CanonicalDeserialize,
{
    stuff
        .dim
        .iter()
        .chain(stuff.read_cts.iter())
        .chain(stuff.E_polys.iter())
        .chain([&stuff.lookup_outputs])
}

fn read_write_values_except_flags_mut<T>(
    stuff: &mut InstructionLookupStuff<T>,
) -> impl Iterator<Item = &mut T>
where
    T: CanonicalSerialize + CanonicalDeserialize,
{
    stuff
        .dim
        .iter_mut()
        .chain(stuff.read_cts.iter_mut())
        .chain(stuff.E_polys.iter_mut())
        .chain([&mut stuff.lookup_outputs])
}
