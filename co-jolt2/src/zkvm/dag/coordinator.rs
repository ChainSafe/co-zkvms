use std::collections::BTreeMap;

use crate::field::JoltField;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::utils::types::MaybeShared;
use crate::zkvm::dag::state_manager::StateManagerCoordinator;
use crate::subprotocols::sumcheck::Rep3BatchedSumcheck;
use crate::zkvm::instruction_lookups::booleanity::Rep3BooleanitySumcheck;
use crate::zkvm::instruction_lookups::hamming_weight::Rep3HammingWeightSumcheck;
use crate::zkvm::ram::output_check::{Rep3OutputSumcheck, Rep3ValFinalSumcheck};
use crate::zkvm::ram::raf_evaluation::Rep3RafEvaluation;
use crate::zkvm::ram::read_write_checking::Rep3RamReadWriteChecking;
use crate::zkvm::registers::read_write_checking::Rep3RegistersReadWriteChecking;
use crate::zkvm::registers::val_evaluation::Rep3ValEvaluation;
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

        // --- Stage 2 ---
        // NOTE: Assumes Stage 1 (SpartanOuter) has already populated `state.accumulator`
        // with the necessary virtual polynomial openings.
        let log_T = state
            .accumulator
            .get_virtual_polynomial_opening(
                jolt_core::zkvm::witness::VirtualPolynomial::LookupOutput,
                jolt_core::poly::opening_proof::SumcheckId::SpartanOuter,
            )
            .0
            .r
            .len();

        let registers_rwc = Rep3RegistersReadWriteChecking::new(&mut state);
        let ram_raf = Rep3RafEvaluation::new(&mut state);
        let ram_rwc = Rep3RamReadWriteChecking::new(&mut state);
        let ram_output = Rep3OutputSumcheck::new(&mut state);
        let lookups_booleanity = Rep3BooleanitySumcheck::new(&mut state.transcript, log_T);

        let stage2_init = (
            (
                lookups_booleanity.gamma(),
                lookups_booleanity.r_address().to_vec(),
                registers_rwc.gamma(),
                registers_rwc.input_claim(),
                ram_rwc.gamma(),
            ),
            (ram_rwc.input_claim(), ram_output.r_address().to_vec()),
        );
        network.broadcast_request(stage2_init)?;

        let mut stage2_instances: Vec<Box<dyn crate::subprotocols::sumcheck::Rep3SumcheckInstance<F, ProofTranscript>>> =
            vec![
                Box::new(registers_rwc),
                Box::new(ram_raf),
                Box::new(ram_rwc),
                Box::new(ram_output),
                Box::new(lookups_booleanity),
            ];

        let (stage2_proof, _r_stage2) = Rep3BatchedSumcheck::prove(
            &stage2_instances,
            &mut state.accumulator,
            &mut state.transcript,
            network,
        )?;
        state.proofs.insert(
            jolt_core::zkvm::dag::state_manager::ProofKeys::Stage2Sumcheck,
            jolt_core::zkvm::dag::state_manager::ProofData::SumcheckProof(stage2_proof),
        );

        // --- Stage 3 ---
        let registers_val = Rep3ValEvaluation::new(&mut state);
        let lookups_hamming = Rep3HammingWeightSumcheck::new(&mut state.transcript);
        let ram_val_final = Rep3ValFinalSumcheck::new(&mut state);

        let stage3_init = (
            registers_val.val_claim(),
            lookups_hamming.gamma(),
            ram_val_final.input_claim(),
        );
        network.broadcast_request(stage3_init)?;

        let mut stage3_instances: Vec<Box<dyn crate::subprotocols::sumcheck::Rep3SumcheckInstance<F, ProofTranscript>>> =
            vec![
                Box::new(registers_val),
                Box::new(lookups_hamming),
                Box::new(ram_val_final),
            ];

        let (stage3_proof, _r_stage3) = Rep3BatchedSumcheck::prove(
            &stage3_instances,
            &mut state.accumulator,
            &mut state.transcript,
            network,
        )?;
        state.proofs.insert(
            jolt_core::zkvm::dag::state_manager::ProofKeys::Stage3Sumcheck,
            jolt_core::zkvm::dag::state_manager::ProofData::SumcheckProof(stage3_proof),
        );

        // --- Construct stub JoltProof with real commitments, deferred stages ---
        let proof = JoltProof {
            opening_claims: Claims(BTreeMap::new()),
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
