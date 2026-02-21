use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::subprotocols::sumcheck::Rep3BatchedSumcheck;
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::stage::Rep3SumcheckInstance;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManagerCoordinator};
use crate::zkvm::dag::Rep3DagStop;
use crate::zkvm::spartan::Rep3SpartanDag;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::dag::proof_serialization::{Claims, JoltProof};
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

/// Coordinator side of the MPC DAG prover.
///
/// Owns the Fiat-Shamir transcript, drives sumcheck rounds by broadcasting
/// challenges, receives evaluation shares from workers, and assembles the
/// final proof.
pub struct Rep3JoltDAGCoordinator;

impl Rep3JoltDAGCoordinator {
    #[tracing::instrument(skip_all, name = "Rep3JoltDAGCoordinator::prove")]
    pub fn prove<'a, F, ProofTranscript, PCS, N>(
        mut state: StateManagerCoordinator<'a, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<JoltProof<F, PCS, ProofTranscript>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        Self::prove_with_stop(state, network, Rep3DagStop::AfterStage1)
    }

    #[tracing::instrument(skip_all, name = "Rep3JoltDAGCoordinator::prove_with_stop")]
    pub fn prove_with_stop<'a, F, ProofTranscript, PCS, N>(
        mut state: StateManagerCoordinator<'a, F, ProofTranscript, PCS>,
        network: &mut N,
        stop: Rep3DagStop,
    ) -> eyre::Result<JoltProof<F, PCS, ProofTranscript>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        // --- Receive trace_length from workers ---
        let trace_lengths: Vec<usize> = network.receive_responses()?;
        let trace_length = trace_lengths[0];
        eyre::ensure!(
            trace_lengths.iter().all(|&t| t == trace_length),
            "trace_length mismatch across parties"
        );
        state.trace_length = trace_length;
        let padded_trace_length = trace_length.next_power_of_two();

        // --- Fiat-Shamir preamble ---
        state.fiat_shamir_preamble(trace_length);

        // --- Initialize DoryGlobals and AllCommittedPolynomials ---
        let ram_K = state.ram_K;
        let bytecode_d = state.preprocessing.shared.bytecode.d;
        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length);
        let _poly_guard =
            AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d);

        // --- Receive, combine, and store commitments ---
        Self::receive_commitments::<F, PCS, ProofTranscript, N>(&mut state, network)?;

        // --- Receive untrusted advice commitment from workers ---
        Self::receive_untrusted_advice_commitment::<F, PCS, ProofTranscript, N>(
            &mut state, network,
        )?;

        // --- Append advice commitments to transcript (matching vanilla ordering) ---
        if let Some(ref untrusted_advice_commitment) = state.untrusted_advice_commitment {
            state
                .transcript
                .append_serializable(untrusted_advice_commitment);
        }

        if let Some(ref trusted_advice_commitment) = state.trusted_advice_commitment {
            state
                .transcript
                .append_serializable(trusted_advice_commitment);
        }

        if stop != Rep3DagStop::AfterCommitments {
            Rep3SpartanDag::stage1_prove(&mut state, network)?;
        }

        if stop != Rep3DagStop::AfterCommitments && stop != Rep3DagStop::AfterStage1 {
            // Stage 2: collect instances from all subsystems (matching vanilla ordering).
            // Each subsystem derives its init data from the transcript/accumulator,
            // then we broadcast the init bundles to workers.
            let stage2_instances =
                Self::stage2_collect_instances(&mut state, network)?;

            let (proof, _r_stage2) = Rep3BatchedSumcheck::prove(
                &stage2_instances,
                &mut state.accumulator,
                &mut state.transcript,
                network,
            )?;
            state.proofs.insert(
                ProofKeys::Stage2Sumcheck,
                ProofData::SumcheckProof(proof),
            );
        }

        // --- Construct stub JoltProof with real commitments, deferred stages ---
        let proof = JoltProof {
            opening_claims: Claims(std::mem::take(&mut state.accumulator.openings)),
            commitments: std::mem::take(&mut state.commitments),
            proofs: std::mem::take(&mut state.proofs),
            untrusted_advice_commitment: state.untrusted_advice_commitment.take(),
            trace_length,
            ram_K: state.ram_K,
            bytecode_d: state.preprocessing.shared.bytecode.d,
            twist_sumcheck_switch_index: state.twist_sumcheck_switch_index,
        };
        Ok(proof)
    }

    /// Collect all stage2 sumcheck instances from every subsystem.
    ///
    /// Creates coordinator-side instances (which derive transcript challenges in
    /// the correct order), broadcasts init data to workers, and returns the
    /// instances in vanilla ordering.
    fn stage2_collect_instances<F, ProofTranscript, PCS, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        use crate::zkvm::instruction_lookups::booleanity::Rep3BooleanitySumcheck;
        use crate::zkvm::ram::output_check::Rep3OutputSumcheck as Rep3OutputSumcheckCoord;
        use crate::zkvm::ram::raf_evaluation::Rep3RafEvaluation;
        use crate::zkvm::ram::read_write_checking::Rep3RamReadWriteChecking;
        use crate::zkvm::registers::read_write_checking::Rep3RegistersReadWriteChecking;
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::zkvm::instruction_lookups::D;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        // 1) Spartan inner sumcheck — derives gamma, input_claim from transcript;
        //    broadcasts (gamma, input_claim) to workers internally.
        let spartan_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            Rep3SpartanDag::stage2_instances(state, network)?;

        // 2) Registers read-write checking — derives gamma from transcript.
        //    Create concrete type, extract init data, broadcast, then box.
        let reg_rwc = Rep3RegistersReadWriteChecking::<F>::new::<ProofTranscript, PCS>(state);
        let reg_gamma = reg_rwc.gamma();
        let reg_input_claim = reg_rwc.input_claim();
        network.broadcast_request((reg_gamma, reg_input_claim))?;
        let registers_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            vec![Box::new(reg_rwc)];

        // 3) RAM (raf, read-write, output) — create each concrete type,
        //    extract init data from RamRWC (gamma) and OutputSumcheck (r_address),
        //    broadcast combined init bundle, then box all.
        let raf = Rep3RafEvaluation::<F>::new::<ProofTranscript, PCS>(state);
        let ram_rwc = Rep3RamReadWriteChecking::<F>::new::<ProofTranscript, PCS>(state);
        let output = Rep3OutputSumcheckCoord::<F>::new::<ProofTranscript, PCS>(state);
        let ram_gamma = ram_rwc.gamma();
        let ram_input_claim = ram_rwc.input_claim();
        let ram_r_address = output.r_address().to_vec();
        network.broadcast_request((ram_gamma, ram_input_claim, ram_r_address))?;
        let ram_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            vec![Box::new(raf), Box::new(ram_rwc), Box::new(output)];

        // // 4) Lookups booleanity — derives gamma + r_address from transcript.
        // let log_T = state
        //     .accumulator
        //     .get_virtual_polynomial_opening(
        //         VirtualPolynomial::LookupOutput,
        //         SumcheckId::SpartanOuter,
        //     )
        //     .0
        //     .r
        //     .len();
        // let booleanity = Rep3BooleanitySumcheck::<F>::new(&mut state.transcript, log_T);
        // let lookups_gamma = booleanity.gamma();
        // let lookups_r_address = booleanity.r_address().to_vec();
        // network.broadcast_request((lookups_gamma, lookups_r_address))?;
        // let lookups_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
        //     vec![Box::new(booleanity)];

        // Collect all instances in vanilla order
        let stage2_instances: Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> =
            std::iter::empty()
                .chain(spartan_instances)
                .chain(registers_instances)
                .chain(ram_instances)
                // .chain(lookups_instances)
                .collect();

        Ok(stage2_instances)
    }

    fn receive_commitments<F, PCS, ProofTranscript, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        // Receive commitment shares from all 3 parties
        // Each party sends Vec<MaybeShared<Commitment>> aligned with AllCommittedPolynomials
        let all_commitment_shares: Vec<Vec<MaybeShared<PCS::Commitment>>> =
            network.receive_responses()?;

        eyre::ensure!(
            all_commitment_shares.len() == 3,
            "expected commitment shares from 3 parties, got {}",
            all_commitment_shares.len()
        );
        let num_polys = all_commitment_shares[0].len();

        // Combine commitment shares
        let combined_commitments: Vec<PCS::Commitment> = (0..num_polys)
            .map(|i| {
                let shares: Vec<&MaybeShared<PCS::Commitment>> = all_commitment_shares
                    .iter()
                    .map(|party_shares| &party_shares[i])
                    .collect();
                <PCS as Rep3CommitmentScheme<F, ProofTranscript>>::combine_commitment_shares(
                    &shares,
                )
            })
            .collect();

        // Store commitments and append to transcript
        state.commitments = combined_commitments;
        for commitment in &state.commitments {
            state.transcript.append_serializable(commitment);
        }

        Ok(())
    }

    /// Receive the untrusted advice commitment from all 3 workers.
    ///
    /// All workers compute the same public commitment, so we verify consistency
    /// and store the first non-None value.
    fn receive_untrusted_advice_commitment<F, PCS, ProofTranscript, N>(
        state: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<()>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F> + Rep3CommitmentScheme<F, ProofTranscript>,
        N: Rep3NetworkCoordinator,
    {
        let commitments: Vec<Option<PCS::Commitment>> = network.receive_responses()?;
        eyre::ensure!(
            commitments.len() == 3,
            "expected untrusted advice commitment from 3 parties, got {}",
            commitments.len()
        );

        // All workers should produce identical results (public computation)
        eyre::ensure!(
            commitments[0] == commitments[1] && commitments[1] == commitments[2],
            "untrusted advice commitment mismatch across parties"
        );

        state.untrusted_advice_commitment = commitments.into_iter().next().unwrap();
        Ok(())
    }
}
