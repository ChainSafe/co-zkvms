use std::cmp::min;

use crate::field::JoltField;
use crate::jolt::vm::bytecode::witness::Rep3BytecodePolynomials;
use crate::jolt::vm::read_write_memory::witness::{Rep3ProgramIO, Rep3ReadWriteMemoryPolynomials};
use crate::jolt::vm::timestamp_range_check::{self};
use crate::lasso::memory_checking::StructuredPolynomialData;
use crate::poly::commitment::{commitment_scheme::CommitmentScheme, Rep3CommitmentScheme};
use crate::poly::Rep3MultilinearPolynomial;
use crate::r1cs::inputs::{R1CSPreprocessing, Rep3R1CSPolynomials};
use crate::utils::types::MaybeShared;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::{izip, multizip, Itertools};
use jolt_common::rv_trace::MemoryLayout;
use jolt_core::jolt::vm::bytecode::BytecodeStuff;
use jolt_core::jolt::vm::instruction_lookups::InstructionLookupStuff;
use jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryStuff;
use jolt_core::jolt::vm::timestamp_range_check::TimestampRangeCheckStuff;
use jolt_core::jolt::vm::{JoltCommitments, JoltPolynomials, JoltStuff, JoltVerifierPreprocessing};
use jolt_core::lasso::memory_checking::{Initializable, NoPreprocessing};
use jolt_core::r1cs::inputs::R1CSStuff;
use jolt_core::utils::transcript::Transcript;
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::PartyID;
use mpc_net::topology::MpcRingNetWorkerExt;
use snarks_core::field::FieldExt;

use crate::jolt::instruction::Rep3JoltInstructionSet;
use crate::jolt::vm::instruction_lookups::witness::Rep3InstructionLookupPolynomials;
use crate::jolt::vm::{JoltTraceStep, JoltWorkerPreprocessing};

#[derive(Debug, Clone, Copy, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltWitnessMeta {
    pub padded_trace_length: usize,
    pub read_write_memory_size: usize,
    pub memory_layout: MemoryLayout,
}

pub type Rep3JoltPolynomials<F> = JoltStuff<Rep3MultilinearPolynomial<F>>;

pub trait Rep3Polynomials<F: JoltField, Preprocessing>: Sized {
    #[cfg(feature = "debug")]
    type PublicPolynomials;

    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &Preprocessing,
        trace: &mut [JoltTraceStep<Instructions>],
        program_io: &Rep3ProgramIO<F>,
        network: &mut IoContextPool<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker + MpcRingNetWorkerExt;

    #[cfg(feature = "debug")]
    fn combine_polynomials(
        preprocessing: &Preprocessing,
        polynomials_shares: Vec<Self>,
    ) -> Self::PublicPolynomials;
}

impl<F: JoltField, const C: usize, PCS, ProofTranscript>
    Rep3Polynomials<F, JoltWorkerPreprocessing<C, F, PCS, ProofTranscript>>
    for Rep3JoltPolynomials<F>
where
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
{
    #[cfg(feature = "debug")]
    type PublicPolynomials = JoltPolynomials<F>;

    #[tracing::instrument(skip_all, name = "Rep3JoltPolynomials::generate_witness_rep3")]
    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &JoltWorkerPreprocessing<C, F, PCS, ProofTranscript>,
        ops: &mut [JoltTraceStep<Instructions>],
        program_io: &Rep3ProgramIO<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Self>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
        Instructions: Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker + MpcRingNetWorkerExt,
    {
        let instruction_lookups = Rep3InstructionLookupPolynomials::generate_witness_rep3(
            &preprocessing.instruction_lookups,
            ops,
            program_io,
            io_ctx,
        )?;

        // let worker_idx = io_ctx.worker_idx();
        // let m_worker = ops.len().next_power_of_two() / io_ctx.num_workers();
        // let trace_worker_range =
        //     (worker_idx * m_worker)..min((worker_idx + 1) * m_worker, ops.len());

        let r1cs = Rep3R1CSPolynomials::generate_witness_rep3(
            &preprocessing.r1cs,
            ops,
            program_io,
            io_ctx,
        )?;

        let mut read_write_memory = Rep3ReadWriteMemoryPolynomials::generate_witness_rep3(
            &preprocessing.read_write_memory,
            ops,
            program_io,
            io_ctx,
        )?;

        let bytecode = Rep3BytecodePolynomials::generate_witness_rep3(
            &preprocessing.bytecode,
            ops,
            program_io,
            io_ctx,
        )?;

        let timestamp_range_check =
            timestamp_range_check::get_timestamp_range_check_polynomials_rep3::<
                F,
                PCS,
                ProofTranscript,
            >(&mut read_write_memory);

        Ok(Self {
            instruction_lookups,
            r1cs,
            read_write_memory,
            bytecode,
            timestamp_range_check,
        })
    }

    #[cfg(feature = "debug")]
    fn combine_polynomials(
        preprocessing: &JoltWorkerPreprocessing<C, F, PCS, ProofTranscript>,
        polynomials_shares: Vec<Self>,
    ) -> Self::PublicPolynomials {
        let (instructions_shares, r1cs, read_write_memory, bytecode): (
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = polynomials_shares
            .into_iter()
            .map(|p| {
                let Rep3JoltPolynomials {
                    instruction_lookups,
                    bytecode,
                    read_write_memory,
                    r1cs,
                    ..
                } = p;
                (instruction_lookups, r1cs, read_write_memory, bytecode)
            })
            .multiunzip();

        let instruction_lookups = Rep3InstructionLookupPolynomials::combine_polynomials(
            &preprocessing.instruction_lookups,
            instructions_shares,
        );

        let r1cs = Rep3R1CSPolynomials::combine_polynomials(&preprocessing.r1cs, r1cs);

        let read_write_memory = Rep3ReadWriteMemoryPolynomials::combine_polynomials(
            &preprocessing.read_write_memory,
            read_write_memory,
        );

        let bytecode =
            Rep3BytecodePolynomials::combine_polynomials(&preprocessing.bytecode, bytecode);

        JoltPolynomials {
            instruction_lookups,
            r1cs,
            read_write_memory,
            bytecode,
            ..Default::default()
        }
    }
}
pub trait Rep3JoltPolynomialsExt<F: JoltField> {
    fn commit<const C: usize, PCS, ProofTranscript, Network>(
        &self,
        preprocessing: &JoltWorkerPreprocessing<C, F, PCS, ProofTranscript>,
        io_ctx: &mut IoContextPool<Network>,
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
        // PCS::Commitment: AddAssign<PCS::Commitment>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkCoordinator,
    {
        // let mut commitments = JoltCommitments::<PCS, ProofTranscript>::initialize(preprocessing);

        let worker_commitments_shares: Vec<Vec<JoltMaybeSharedCommitments<PCS, ProofTranscript>>> =
            network
                .receive_responses_from_subnets()?
                .try_into()
                .map_err(|_| eyre::eyre!("failed to receive commitments"))?;

        let commitments = worker_commitments_shares
            .into_iter()
            .map(|mut commitments_shares| {
                // can use `initialize_for_worker` but we can't truncate `read_cts` and `final_cts` based on shares form instead
                let mut commitments =
                    JoltCommitments::<PCS, ProofTranscript>::initialize(preprocessing);

                commitments
                    .instruction_lookups
                    .read_cts
                    .truncate(commitments_shares[0].instruction_lookups.read_cts.len());
                commitments
                    .instruction_lookups
                    .final_cts
                    .truncate(commitments_shares[0].instruction_lookups.final_cts.len());

                let span = tracing::span!(tracing::Level::INFO, "combine_read_write_values");
                let _guard = span.enter();

                multizip((
                    commitments_shares[0].read_write_values(),
                    commitments_shares[1].read_write_values(),
                    commitments_shares[2].read_write_values(),
                ))
                .map(|(c0, c1, c2)| PCS::combine_commitment_shares(&[c0, c1, c2]))
                .zip(commitments.read_write_values_mut())
                .for_each(|(commitment, dest)| *dest = commitment);
                drop(_guard);

                let span = tracing::span!(tracing::Level::INFO, "combine_final_cts");
                let _guard = span.enter();
                commitments.instruction_lookups.final_cts = multizip((
                    &commitments_shares[0].instruction_lookups.final_cts,
                    &commitments_shares[1].instruction_lookups.final_cts,
                    &commitments_shares[2].instruction_lookups.final_cts,
                ))
                .map(|(c0, c1, c2)| PCS::combine_commitment_shares(&[c0, c1, c2]))
                .collect_vec();
                drop(_guard);

                let span = tracing::span!(tracing::Level::INFO, "combine_t_final");
                let _guard = span.enter();
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
                commitments
            })
            .reduce(|mut acc, next| {
                let read_cts_start = 19 + C;
                let acc_read_cts_len = acc.instruction_lookups.read_cts.len();
                let mut next_rw_comms = next.read_write_values();
                let mut acc_rw_comms = acc.read_write_values_mut();

                let _: Vec<_> = next_rw_comms
                    .drain(read_cts_start..read_cts_start + next.instruction_lookups.read_cts.len())
                    .collect();
                let _: Vec<_> = acc_rw_comms
                    .drain(read_cts_start..read_cts_start + acc_read_cts_len)
                    .collect();

                assert_eq!(acc_rw_comms.len(), next_rw_comms.len());

                izip!(acc_rw_comms, next_rw_comms)
                    .for_each(|(acc, comm)| *acc = PCS::concat_commitments(acc, comm));
                acc.instruction_lookups
                    .read_cts
                    .extend(next.instruction_lookups.read_cts);
                acc.instruction_lookups
                    .final_cts
                    .extend(next.instruction_lookups.final_cts);
                izip!(
                    acc.read_write_memory.init_final_values_mut(),
                    next.read_write_memory.init_final_values()
                )
                .for_each(|(acc, comm)| *acc = PCS::concat_commitments(acc, comm));

                acc.bytecode.t_final =
                    PCS::concat_commitments(&acc.bytecode.t_final, &next.bytecode.t_final);
                acc
            })
            .unwrap();

        tracing::info!(
            "instructions final_cts comms: {:?}",
            commitments.instruction_lookups.final_cts.len()
        );

        Ok(commitments)
    }

    fn take_exogenous_polynomials_for_timestamp_range_check(&mut self) -> JoltPolynomials<F>;
}

type JoltMaybeSharedCommitments<
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
> = JoltStuff<MaybeShared<PCS::Commitment>>;

impl<F: JoltField> Rep3JoltPolynomialsExt<F> for Rep3JoltPolynomials<F> {
    #[tracing::instrument(skip_all, name = "Rep3JoltPolynomials::commit")]
    fn commit<const C: usize, PCS, ProofTranscript, Network>(
        &self,
        preprocessing: &JoltWorkerPreprocessing<C, F, PCS, ProofTranscript>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkWorker,
    {
        let id = io_ctx.party_id();
        let mut commitments =
            JoltMaybeSharedCommitments::<PCS, ProofTranscript>::worker_initialize(preprocessing);

        if io_ctx.network().is_distributed() {
            // `worker_initialize` assumes that worker holds all `preprocessing.instruction_lookups.read_memories_worker.len()` E_polys as for Memory checking
            // But during commit and instructions primary sumcheck, workers hold all E_polys *shards*, so we must resize to original `num_memories`
            commitments
                .instruction_lookups
                .E_polys
                .resize_with(preprocessing.instruction_lookups.num_memories, || {
                    Default::default()
                });
        }

        let span = tracing::span!(tracing::Level::INFO, "commit::trace_polys");
        let _guard = span.enter();
        let trace_polys = self.read_write_values();

        let trace_commitments =
            PCS::batch_commit_rep3(&trace_polys, &preprocessing.generators, id == PartyID::ID0);

        commitments
            .read_write_values_mut()
            .into_iter()
            .zip(trace_commitments.into_iter())
            .for_each(|(dest, src)| *dest = src);
        drop(_guard);
        drop(span);

        let id = io_ctx.party_id();
        let span = tracing::span!(tracing::Level::INFO, "commit::t_final");
        let _guard = span.enter();
        commitments.bytecode.t_final = PCS::commit_rep3(
            &self.bytecode.t_final,
            &preprocessing.generators,
            id == PartyID::ID0,
        );
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "commit::read_write_memory");
        let _guard = span.enter();
        (
            commitments.read_write_memory.v_final,
            commitments.read_write_memory.t_final,
        ) = rayon::join(
            || {
                PCS::commit_rep3(
                    &self.read_write_memory.v_final,
                    &preprocessing.generators,
                    id == PartyID::ID0,
                )
            },
            || {
                PCS::commit_rep3(
                    &self.read_write_memory.t_final,
                    &preprocessing.generators,
                    id == PartyID::ID0,
                )
            },
        );
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "commit::instruction_final_cts");
        let _guard = span.enter();
        commitments.instruction_lookups.final_cts = PCS::batch_commit_rep3(
            &self.instruction_lookups.final_cts,
            &preprocessing.generators,
            false, // no public polys in final_cts
        );
        drop(_guard);
        drop(span);

        io_ctx.sync_with_parties()?;

        io_ctx.network().send_response(commitments)
    }

    fn take_exogenous_polynomials_for_timestamp_range_check(&mut self) -> JoltPolynomials<F> {
        let t_read_rd = std::mem::take(&mut self.read_write_memory.t_read_rd)
            .try_into()
            .unwrap();
        let t_read_rs1 = std::mem::take(&mut self.read_write_memory.t_read_rs1)
            .try_into()
            .unwrap();
        let t_read_rs2 = std::mem::take(&mut self.read_write_memory.t_read_rs2)
            .try_into()
            .unwrap();
        let t_read_ram = std::mem::take(&mut self.read_write_memory.t_read_ram)
            .try_into()
            .unwrap();

        JoltPolynomials {
            read_write_memory: ReadWriteMemoryStuff {
                t_read_rd,
                t_read_rs1,
                t_read_rs2,
                t_read_ram,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

pub trait WorkerInitializable<T, Preprocessing>: StructuredPolynomialData<T> + Default {
    type VerifierPreprocessing;

    /// This function is used in lieu of `Default::default()` to initialize a
    /// `StructuredPolynomialData` struct, which may contain `Vec` fields
    /// whose lengths depend on some preprocessing.
    ///
    /// Note that the default implementation of initialize, however, does
    /// just return `Default::default()`.
    fn worker_initialize(_preprocessing: &Preprocessing) -> Self {
        Default::default()
    }

    fn initialize_for_worker(
        preprocessing: &Self::VerifierPreprocessing,
        _num_workers: usize,
        _worker_idx: usize,
    ) -> Self
    where
        Self: Initializable<T, Self::VerifierPreprocessing>,
    {
        Self::initialize(preprocessing)
    }
}

impl<
        const C: usize,
        T: CanonicalSerialize + CanonicalDeserialize + Default + Sync,
        PCS: CommitmentScheme<ProofTranscript>,
        ProofTranscript: Transcript,
    > WorkerInitializable<T, JoltWorkerPreprocessing<C, PCS::Field, PCS, ProofTranscript>>
    for JoltStuff<T>
where
    PCS::Field: FieldExt,
{
    type VerifierPreprocessing = JoltVerifierPreprocessing<C, PCS::Field, PCS, ProofTranscript>;

    fn worker_initialize(
        preprocessing: &JoltWorkerPreprocessing<C, PCS::Field, PCS, ProofTranscript>,
    ) -> Self {
        Self {
            bytecode: BytecodeStuff::worker_initialize(&preprocessing.bytecode),
            read_write_memory: ReadWriteMemoryStuff::worker_initialize(
                &preprocessing.read_write_memory,
            ),
            instruction_lookups: InstructionLookupStuff::worker_initialize(
                &preprocessing.instruction_lookups,
            ),
            timestamp_range_check: <TimestampRangeCheckStuff<T> as Initializable<
                T,
                NoPreprocessing,
            >>::initialize(&NoPreprocessing),
            r1cs: <R1CSStuff<T> as Initializable<T, usize>>::initialize(&C),
        }
    }

    fn initialize_for_worker(
        preprocessing: &Self::VerifierPreprocessing,
        num_workers: usize,
        worker_id: usize,
    ) -> Self
    where
        Self: Initializable<T, Self::VerifierPreprocessing>,
    {
        Self {
            bytecode: BytecodeStuff::initialize_for_worker(
                &preprocessing.bytecode,
                num_workers,
                worker_id,
            ),
            read_write_memory: ReadWriteMemoryStuff::initialize_for_worker(
                &preprocessing.read_write_memory,
                num_workers,
                worker_id,
            ),
            instruction_lookups: InstructionLookupStuff::initialize_for_worker(
                &preprocessing.instruction_lookups,
                num_workers,
                worker_id,
            ),
            timestamp_range_check: <TimestampRangeCheckStuff<T> as Initializable<
                T,
                NoPreprocessing,
            >>::initialize(&NoPreprocessing),
            r1cs: <R1CSStuff<T> as Initializable<T, usize>>::initialize(&C),
        }
    }
}
