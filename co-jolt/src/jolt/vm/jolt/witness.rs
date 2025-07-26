use std::marker::PhantomData;

use crate::jolt::vm::bytecode::witness::Rep3BytecodePolynomials;
use crate::jolt::vm::read_write_memory::witness::Rep3ReadWriteMemoryPolynomials;
use crate::jolt::vm::timestamp_range_check::Rep3TimestampRangeCheckPolynomials;
use crate::lasso::memory_checking::StructuredPolynomialData;
use crate::poly::commitment::{commitment_scheme::CommitmentScheme, Rep3CommitmentScheme};
use crate::poly::{Rep3MultilinearPolynomial, Rep3PolysConversion};
use crate::r1cs::inputs::Rep3R1CSPolynomials;
use crate::utils::element::MaybeShared;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::{multizip, Itertools};
use jolt_common::rv_trace::MemoryLayout;
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::instruction_lookups::{
    InstructionLookupCommitments, InstructionLookupStuff, InstructionLookupsPreprocessing,
};
use jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryStuff;
use jolt_core::jolt::vm::timestamp_range_check::{
    TimestampRangeCheckPolynomials, TimestampRangeCheckStuff,
};
use jolt_core::jolt::vm::{JoltCommitments, JoltPolynomials, JoltStuff, JoltVerifierPreprocessing};
use jolt_core::lasso::memory_checking::{Initializable, NoPreprocessing};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::r1cs::builder::CombinedUniformBuilder;
use jolt_core::r1cs::inputs::ConstraintInput;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use jolt_tracer::JoltDevice;
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
    pub memory_layout: MemoryLayout,
}

pub type Rep3JoltPolynomials<F: JoltField> = JoltStuff<Rep3MultilinearPolynomial<F>>;

pub trait Rep3Polynomials<F: JoltField, Preprocessing>: Sized {
    type PublicPolynomials;

    fn generate_secret_shares<R: Rng>(
        _preprocessing: &Preprocessing,
        polynomials: Self::PublicPolynomials,
        rng: &mut R,
    ) -> Vec<Self>;

    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &Preprocessing,
        trace: &mut [JoltTraceStep<F, Instructions>],
        M: usize,
        network: IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
        Network: Rep3Network;

    fn combine_polynomials(
        preprocessing: &Preprocessing,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials>;
}

impl<F: JoltField, const C: usize, PCS, ProofTranscript>
    Rep3Polynomials<F, JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>>
    for Rep3JoltPolynomials<F>
where
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
{
    type PublicPolynomials = JoltPolynomials<F>;

    #[tracing::instrument(skip_all, name = "Rep3JoltPolynomials::generate_secret_shares")]
    fn generate_secret_shares<R: Rng>(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        polynomials: Self::PublicPolynomials,
        rng: &mut R,
    ) -> Vec<Self> {
        let JoltPolynomials {
            instruction_lookups,
            read_write_memory,
            timestamp_range_check,
            r1cs,
            bytecode,
        } = polynomials;

        let instruction_lookups = Rep3InstructionLookupPolynomials::generate_secret_shares(
            &preprocessing.instruction_lookups,
            instruction_lookups,
            rng,
        );

        let read_write_memory = Rep3ReadWriteMemoryPolynomials::generate_secret_shares(
            &preprocessing.read_write_memory,
            read_write_memory,
            rng,
        );

        let timestamp_range_check = Rep3TimestampRangeCheckPolynomials::generate_secret_shares(
            &NoPreprocessing,
            timestamp_range_check,
            rng,
        );

        let bytecode =
            Rep3BytecodePolynomials::generate_secret_shares(&preprocessing.bytecode, bytecode, rng);

        let r1cs = Rep3R1CSPolynomials::generate_secret_shares(&NoPreprocessing, r1cs, rng);

        let jolt_polys_shares = itertools::multizip((
            instruction_lookups,
            read_write_memory,
            timestamp_range_check,
            bytecode,
            r1cs,
        ))
        .into_iter()
        .map(
            |(instruction_lookups, read_write_memory, timestamp_range_check, bytecode, r1cs)| {
                Self {
                    instruction_lookups,
                    read_write_memory,
                    timestamp_range_check,
                    bytecode,
                    r1cs,
                }
            },
        )
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
}
pub trait Rep3JoltPolynomialsExt<F: JoltField> {
    fn commit<const C: usize, PCS, ProofTranscript, Network>(
        &self,
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkWorker;

    #[tracing::instrument(skip_all, name = "Rep3JoltPolynomials::receive_commitments")]
    fn receive_commitments<const C: usize, PCS, ProofTranscript, Network>(
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        network: &mut Network,
    ) -> eyre::Result<JoltCommitments<PCS, ProofTranscript>>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkCoordinator,
    {
        let mut commitments = JoltCommitments::<PCS, ProofTranscript>::initialize(preprocessing);

        let mut commitments_shares: Vec<JoltMaybeSharedCommitments<PCS, ProofTranscript>> = network
            .receive_responses()?
            .try_into()
            .map_err(|_| eyre::eyre!("failed to receive commitments"))?;

        multizip((
            commitments_shares[0].read_write_values(),
            commitments_shares[1].read_write_values(),
            commitments_shares[2].read_write_values(),
        ))
        .map(|(c0, c1, c2)| PCS::combine_commitment_shares(&[c0, c1, c2]))
        .zip(commitments.read_write_values_mut())
        .for_each(|(commitment, dest)| *dest = commitment);

        commitments.instruction_lookups.final_cts = multizip((
            &commitments_shares[0].instruction_lookups.final_cts,
            &commitments_shares[1].instruction_lookups.final_cts,
            &commitments_shares[2].instruction_lookups.final_cts,
        ))
        .map(|(c0, c1, c2)| PCS::combine_commitment_shares(&[c0, c1, c2]))
        .collect_vec();

        commitments.bytecode.t_final = std::mem::take(
            commitments_shares[0]
                .bytecode
                .t_final
                .try_into_public_mut()
                .expect("party 0 must compute commitment to public t_final"),
        );

        commitments.read_write_memory.v_final = PCS::combine_commitment_shares(&[
            &commitments_shares[0].read_write_memory.v_final,
            &commitments_shares[1].read_write_memory.v_final,
            &commitments_shares[2].read_write_memory.v_final,
        ]);

        commitments.read_write_memory.t_final = std::mem::take(
            commitments_shares[0]
                .read_write_memory
                .t_final
                .try_into_public_mut()
                .expect("party 0 must compute commitment to public t_final"),
        );

        Ok(commitments)
    }

    fn get_timestamp_range_check_polynomials(&mut self) -> TimestampRangeCheckPolynomials<F>;

    fn get_exogenous_polynomials_for_timestamp_range_check(&mut self) -> JoltPolynomials<F>;

    fn compute_aux<const C: usize, I: ConstraintInput>(
        &mut self,
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
    );
}

type JoltMaybeSharedCommitments<
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
> = JoltStuff<MaybeShared<PCS::Commitment>>;

impl<F: JoltField> Rep3JoltPolynomialsExt<F> for Rep3JoltPolynomials<F> {
    #[tracing::instrument(skip_all, name = "Rep3JoltPolynomials::commit")]
    fn commit<const C: usize, PCS, ProofTranscript, Network>(
        &self,
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkWorker,
    {
        let mut commitments =
            JoltMaybeSharedCommitments::<PCS, ProofTranscript>::initialize(preprocessing);

        let trace_polys = self.read_write_values();

        let trace_commitments = PCS::batch_commit_rep3(
            &trace_polys,
            &preprocessing.generators,
            io_ctx.id == PartyID::ID0,
        );

        commitments
            .read_write_values_mut()
            .into_iter()
            .zip(trace_commitments.into_iter())
            .for_each(|(dest, src)| *dest = src);

        commitments.instruction_lookups.final_cts = PCS::batch_commit_rep3(
            &self.instruction_lookups.final_cts,
            &preprocessing.generators,
            false, // no public polys in final_cts
        );

        io_ctx.network.send_response(commitments)
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

    fn compute_aux<const C: usize, I: ConstraintInput>(
        &mut self,
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
    ) {
        // use crate::r1cs::spartan::worker::compute_aux_poly;
        // let flattened_vars = I::flatten::<C>();
        // for (aux_index, aux_compute) in constraint_builder.uniform_builder.aux_computations.iter() {
        //     *flattened_vars[*aux_index].get_ref_mut(self) =
        //         aux_compute.compute_aux_poly::<C, I>(self, constraint_builder.uniform_repeat);
        // }
        todo!()
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
