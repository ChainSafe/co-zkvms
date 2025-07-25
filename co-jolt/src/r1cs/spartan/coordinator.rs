use jolt_core::r1cs::spartan::UniformSpartanProof;
use mpc_core::protocols::additive;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use std::marker::PhantomData;

use jolt_core::field::JoltField;
use jolt_core::r1cs::key::UniformSpartanKey;
use jolt_core::utils::math::Math;
use jolt_core::utils::transcript::Transcript;

use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;

use crate::poly::opening_proof::Rep3ProverOpeningAccumulator;
use crate::subprotocols::commitment::DistributedCommitmentScheme;
use crate::subprotocols::sumcheck;
use crate::subprotocols::sumcheck_spartan::coordinate_eq_sumcheck_round;
use jolt_core::r1cs::inputs::ConstraintInput;

pub trait Rep3UniformSpartanCoordinator<const C: usize, I, F, ProofTranscript, Network>
where
    F: JoltField,
    ProofTranscript: Transcript,
    I: ConstraintInput,
    Network: Rep3NetworkCoordinator,
{
    #[tracing::instrument(skip_all, name = "Rep3UniformSpartan::prove")]
    fn prove_rep3<PCS>(
        key: &UniformSpartanKey<C, I, F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<UniformSpartanProof<C, I, F, ProofTranscript>>
    where
        PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    {
        let num_rounds_x = key.num_rows_bits();

        /* Sumcheck 1: Outer sumcheck */
        let span = tracing::info_span!("outer_sumcheck");
        let _guard = span.enter();

        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();
        network.broadcast_request(tau)?;

        let mut outer_sumcheck_r = Vec::new();
        let mut claim = F::zero();
        let mut polys = Vec::new();

        for _round in 0..num_rounds_x {
            coordinate_eq_sumcheck_round(
                &mut polys,
                &mut outer_sumcheck_r,
                &mut claim,
                transcript,
                network,
            )?
        }

        let outer_sumcheck_proof = SumcheckInstanceProof::new(polys);
        let outer_sumcheck_claims =
            additive::combine_field_element_vec(network.receive_responses()?);

        transcript.append_scalars(&outer_sumcheck_claims);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );
        drop(_guard);
        drop(span);

        /* Sumcheck 2: Inner sumcheck
            RLC of claims Az, Bz, Cz
            where claim_Az = \sum_{y_var} A(rx, y_var || rx_step) * z(y_var || rx_step)
                                + A_shift(..) * z_shift(..)
            and shift denotes the values at the next time step "rx_step+1" for cross-step constraints
            - A_shift(rx, y_var || rx_step) = \sum_t A(rx, y_var || t) * eq_plus_one(rx_step, t)
            - z_shift(y_var || rx_step) = \sum z(y_var || rx_step) * eq_plus_one(rx_step, t)
        */

        let span = tracing::info_span!("inner_sumcheck");
        let _guard = span.enter();

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = claim_Az
            + inner_sumcheck_RLC * claim_Bz
            + inner_sumcheck_RLC * inner_sumcheck_RLC * claim_Cz;

        network.broadcast_request((inner_sumcheck_RLC, claim_inner_joint))?;

        let num_rounds_inner_sumcheck = (key.uniform_r1cs.num_vars.next_power_of_two() * 4).log_2();

        let (inner_sumcheck_proof, _) =
            sumcheck::coordinate_prove_arbitrary(num_rounds_inner_sumcheck, transcript, network)?;

        drop(_guard);
        drop(span);

        /*  Sumcheck 3: Shift sumcheck
            sumcheck claim is = z_shift(ry_var || rx_step) = \sum_t z(ry_var || t) * eq_plus_one(rx_step, t)
        */
        let span = tracing::info_span!("shift_sumcheck");
        let _guard = span.enter();
        let num_rounds_shift_sumcheck = key.num_steps.log_2();

        let shift_sumcheck_claim = network.receive_responses::<F>()?.into_iter().sum();

        let (shift_sumcheck_proof, _) =
            sumcheck::coordinate_prove_arbitrary(num_rounds_shift_sumcheck, transcript, network)?;
        drop(_guard);
        drop(span);

        let claimed_witness_evals =
            Rep3ProverOpeningAccumulator::receive_claims(transcript, network)?;

        let shift_sumcheck_witness_evals =
            Rep3ProverOpeningAccumulator::receive_claims(transcript, network)?;

        let outer_sumcheck_claims = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        Ok(UniformSpartanProof {
            _inputs: PhantomData,
            outer_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_proof,
            shift_sumcheck_proof,
            shift_sumcheck_claim,
            claimed_witness_evals,
            shift_sumcheck_witness_evals,
            _marker: PhantomData,
        })
    }
}

impl<const C: usize, I, F, ProofTranscript, Network>
    Rep3UniformSpartanCoordinator<C, I, F, ProofTranscript, Network>
    for UniformSpartanProof<C, I, F, ProofTranscript>
where
    I: ConstraintInput,
    F: JoltField,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
}
