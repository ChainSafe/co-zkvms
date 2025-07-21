use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use co_lasso::memory_checking::StructuredPolynomialData;
use co_lasso::poly::commitment::commitment_scheme::CommitmentScheme;
use co_lasso::subprotocols::commitment::DistributedCommitmentScheme;
use co_lasso::utils::transcript::{KeccakTranscript, Transcript};
use itertools::{multizip, Itertools};
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::instruction_lookups::{
    InstructionLookupCommitments, InstructionLookupStuff,
};
use jolt_core::jolt::vm::{JoltCommitments, JoltPolynomials, JoltStuff, JoltVerifierPreprocessing};
use jolt_core::lasso::memory_checking::Initializable;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use mpc_core::protocols::rep3::network::{
    IoContext, Rep3Network, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use rand::Rng;

use crate::jolt::instruction::{JoltInstructionSet, Rep3JoltInstructionSet};
use crate::jolt::vm::instruction_lookups::witness::Rep3InstructionLookupPolynomials;
use crate::jolt::vm::JoltTraceStep;

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3JoltPolynomials<F: JoltField> {
    pub instruction_lookups: Rep3InstructionLookupPolynomials<F>,
}

pub trait Rep3Polynomials<F: JoltField, Preprocessing>: Sized {
    type PublicPolynomials;

    fn combine_polynomials(
        preprocessing: &Preprocessing,
        polynomials_shares: Vec<Self>,
    ) -> Self::PublicPolynomials;

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
    ) -> Self::PublicPolynomials {
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
        );

        JoltPolynomials {
            instruction_lookups,
            ..Default::default()
        }
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

impl<F: JoltField> Rep3JoltPolynomials<F> {
    pub fn commit<const C: usize, PCS, ProofTranscript, Network>(
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

        let trace_polys = Self::read_write_values_except_flags(&self.instruction_lookups)
            .map(|poly| MultilinearPolynomial::LargeScalars(poly.copy_share_a()))
            .collect_vec();

        let trace_polys_ref = trace_polys.iter().collect::<Vec<_>>();

        let trace_commitments = PCS::batch_commit(&trace_polys_ref, &preprocessing.generators);

        Self::read_write_values_except_flags_mut(&mut commitments.instruction_lookups)
            .zip(trace_commitments.into_iter())
            .for_each(|(dest, src)| *dest = src);

        commitments.instruction_lookups.final_cts = PCS::batch_commit(
            &self
                .instruction_lookups
                .final_cts
                .iter()
                .map(|poly| MultilinearPolynomial::LargeScalars(poly.copy_share_a()))
                .collect_vec(),
            &preprocessing.generators,
        );

        io_ctx
            .network
            .send_response(commitments.instruction_lookups)?;

        if io_ctx.id == PartyID::ID0 {
            let lookup_flag_polys_commitment = PCS::batch_commit(
                &self
                    .instruction_lookups
                    .aux_stuff
                    .instruction_flags
                    .iter()
                    .collect::<Vec<_>>(),
                &preprocessing.generators,
            );
            io_ctx.network.send_response(lookup_flag_polys_commitment)?;
        }

        Ok(())
    }

    pub fn receive_commitments<const C: usize, PCS, ProofTranscript, Network>(
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
            Self::read_write_values_except_flags(&share1),
            Self::read_write_values_except_flags(&share2),
            Self::read_write_values_except_flags(&share3),
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

    fn read_write_values_except_flags<T, U>(
        stuff: &InstructionLookupStuff<T, U>,
    ) -> impl Iterator<Item = &T>
    where
        T: CanonicalSerialize + CanonicalDeserialize,
        U: CanonicalSerialize + CanonicalDeserialize + Default,
    {
        stuff
            .dim
            .iter()
            .chain(stuff.read_cts.iter())
            .chain(stuff.E_polys.iter())
            .chain([&stuff.lookup_outputs])
    }

    fn read_write_values_except_flags_mut<T, U>(
        stuff: &mut InstructionLookupStuff<T, U>,
    ) -> impl Iterator<Item = &mut T>
    where
        T: CanonicalSerialize + CanonicalDeserialize,
        U: CanonicalSerialize + CanonicalDeserialize + Default,
    {
        stuff
            .dim
            .iter_mut()
            .chain(stuff.read_cts.iter_mut())
            .chain(stuff.E_polys.iter_mut())
            .chain([&mut stuff.lookup_outputs])
    }
}
