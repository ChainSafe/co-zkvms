use std::marker::PhantomData;

use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

use crate::zkvm::dag::stage::{BatchedSumcheckInstance, SumcheckStagesCoordinator};
use crate::zkvm::dag::state_manager::StateManager;
use jolt_core::field::JoltField;

use self::booleanity::Rep3BooleanitySumcheck;
use self::hamming_weight::Rep3HammingWeightSumcheck;
use self::ra_virtual::Rep3InstructionRaSumcheck;
use self::read_raf_checking::Rep3ReadRafSumcheck;

pub mod booleanity;
pub mod hamming_weight;
pub mod ra_virtual;
pub mod read_raf_checking;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3LookupsDag<F: JoltField> {
    _phantom: PhantomData<F>,
}

impl<F: JoltField> Rep3LookupsDag<F> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<F: JoltField> Default for Rep3LookupsDag<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: JoltField> Rep3LookupsDag<F> {
    /// Run the coordinator-side RA virtualization sumcheck (stage 4).
    ///
    /// Reads the InstructionRa opening from the accumulator, broadcasts init
    /// data to workers, and runs the dedicated proving loop.
    pub fn stage4_prove_coordinator<ProofTranscript, PCS, N>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<Option<(SumcheckInstanceProof<F, Bn254Curve, ProofTranscript>, Vec<F::Challenge>)>>
    where
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
        N: Rep3NetworkCoordinator,
    {
        use jolt_core::poly::opening_proof::OpeningId;

        let ra_key = OpeningId::Virtual(VirtualPolynomial::InstructionRa, SumcheckId::InstructionReadRaf);
        let has_ra_opening = sm.accumulator.openings.contains_key(&ra_key);

        // Tell workers whether stage 4 is active.
        network.broadcast_request(has_ra_opening)?;

        if !has_ra_opening {
            return Ok(None);
        }

        let (ra_point, ra_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::InstructionRa, SumcheckId::InstructionReadRaf);

        let (r_address, r_cycle) = ra_point.r.split_at(D * LOG_K_CHUNK);
        let r_address_chunks: Vec<Vec<F::Challenge>> = r_address.chunks(LOG_K_CHUNK).map(|c| c.to_vec()).collect();

        network.broadcast_request((ra_claim, r_address.to_vec(), r_cycle.to_vec()))?;

        let ra_coord = Rep3InstructionRaSumcheck::new(ra_claim, r_cycle.to_vec(), r_address_chunks);
        let result = ra_virtual::prove_coordinator(&ra_coord, &mut sm.accumulator, &mut sm.transcript, network)?;

        Ok(Some(result))
    }
}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>, N>
    SumcheckStagesCoordinator<F, ProofTranscript, PCS, N> for Rep3LookupsDag<F>
where
    N: Rep3NetworkCoordinator,
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        let log_T = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter)
            .0
            .r
            .len();

        let booleanity = Rep3BooleanitySumcheck::new(&mut sm.transcript, log_T);

        Ok(vec![BatchedSumcheckInstance::Secret(Box::new(booleanity))])
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        // ReadRaf (created before HammingWeight, matching vanilla ordering).
        // Draws gamma from transcript internally.
        let (_, rv_claim) =
            sm.accumulator.get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter);
        let (_, left_operand_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::LeftLookupOperand, SumcheckId::SpartanOuter);
        let (_, right_operand_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RightLookupOperand, SumcheckId::SpartanOuter);
        let log_T = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter)
            .0
            .r
            .len();

        let read_raf =
            Rep3ReadRafSumcheck::new(&mut sm.transcript, rv_claim, left_operand_claim, right_operand_claim, log_T);

        let hamming_weight = Rep3HammingWeightSumcheck::new(&mut sm.transcript);

        Ok(vec![
            BatchedSumcheckInstance::Secret(Box::new(read_raf)),
            BatchedSumcheckInstance::Secret(Box::new(hamming_weight)),
        ])
    }
}
