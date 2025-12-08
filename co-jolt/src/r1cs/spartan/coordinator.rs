use jolt_core::poly::sparse_interleaved_poly::SparseCoefficient;
use jolt_core::poly::spartan_interleaved_poly::SpartanInterleavedPolynomial;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::r1cs::spartan::UniformSpartanProof;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use std::marker::PhantomData;

use crate::field::JoltField;
use jolt_core::r1cs::key::UniformSpartanKey;
use jolt_core::utils::math::Math;
use jolt_core::utils::transcript::Transcript;

use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;

use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::opening_proof::Rep3OpeningAccumulatorCoordinator;
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
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<UniformSpartanProof<C, I, F, ProofTranscript>>
    where
        PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    {
        let log_num_workers = network.log_num_workers();
        let num_rounds_x = key.num_rows_bits();

        /* Sumcheck 1: Outer sumcheck */
        let span = tracing::info_span!("outer_sumcheck");
        let _guard = span.enter();

        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>(); // TODO transcript.challenge_scalars
        let mut eq_poly = GruenSplitEqPolynomial::new(&tau);

        network.broadcast_request(tau)?;

        let mut outer_sumcheck_r = Vec::new();
        let mut claim = F::zero();
        let mut polys = Vec::new();

        for _round in 0..num_rounds_x - log_num_workers {
            coordinate_eq_sumcheck_round(
                &mut polys,
                &mut outer_sumcheck_r,
                &mut claim,
                &mut eq_poly, // TODO: avoid instantiating full eq_poly
                transcript,
                network,
            )?
        }

        let outer_sumcheck_claims = if network.is_distributed() {
            let bound_coeffs = network
                .receive_responses_from_subnets::<Vec<AdditiveShare<F>>>()?
                .into_iter()
                .flat_map(additive::combine_additive_vec)
                .enumerate()
                .map(SparseCoefficient::from)
                .collect();
            let mut az_bz_cz_poly = SpartanInterleavedPolynomial::new_bound(bound_coeffs);

            for _round in 0..log_num_workers {
                az_bz_cz_poly.subsequent_sumcheck_round(
                    &mut eq_poly,
                    transcript,
                    &mut outer_sumcheck_r,
                    &mut polys,
                    &mut claim,
                );
            }

            network.broadcast_request(
                outer_sumcheck_r[outer_sumcheck_r.len() - log_num_workers..].to_vec(),
            )?;
            az_bz_cz_poly.final_sumcheck_evals()
        } else {
            additive::combine_additive_vec(network.receive_responses()?)
                .try_into()
                .unwrap()
        };

        tracing::info!("outer_sumcheck_claims: {:?}", outer_sumcheck_claims);
        let outer_sumcheck_proof = SumcheckInstanceProof::new(polys);
        transcript.append_scalars(&outer_sumcheck_claims);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let [claim_Az, claim_Bz, claim_Cz] = outer_sumcheck_claims;

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
        tracing::info!("inner_sumcheck_RLC: {}", inner_sumcheck_RLC);

        let mut claim_inner_joint = claim_Az
            + inner_sumcheck_RLC * claim_Bz
            + inner_sumcheck_RLC * inner_sumcheck_RLC * claim_Cz;

        tracing::info!("claim_inner_joint: {}", claim_inner_joint);

        network.broadcast_request(inner_sumcheck_RLC)?;

        let num_rounds_inner_sumcheck = (key.uniform_r1cs.num_vars.next_power_of_two() * 4).log_2();

        let comb_func = |poly_evals: &[F]| -> F {
            assert_eq!(poly_evals.len(), 2);
            poly_evals[0] * poly_evals[1]
        };

        let (inner_sumcheck_proof, _inner_sumcheck_r) = sumcheck::coordinate_prove_arbitrary(
            &mut claim_inner_joint,
            num_rounds_inner_sumcheck,
            transcript,
            network,
        )?;

        drop(_guard);
        drop(span);

        /*  Sumcheck 3: Shift sumcheck
            sumcheck claim is = z_shift(ry_var || rx_step) = \sum_t z(ry_var || t) * eq_plus_one(rx_step, t)
        */
        let span = tracing::info_span!("shift_sumcheck");
        let _guard = span.enter();
        let num_rounds_shift_sumcheck = key.num_steps.log_2();

        let mut shift_sumcheck_claim = if network.is_distributed() {
            network
                .receive_responses_from_subnets::<AdditiveShare<F>>()?
                .into_iter()
                .map(additive::combine_additive_share)
                .sum::<F>()
        } else {
            additive::combine_additive_share(network.receive_responses()?)
        };

        tracing::info!("shift_sumcheck_claim: {}", shift_sumcheck_claim);

        unsafe {
            std::env::set_var("SUMCHECK_LOG", "true");
        }
        let (shift_sumcheck_proof, _, _) = sumcheck::coordinate_distributed_prove_arbitrary(
            &mut shift_sumcheck_claim,
            num_rounds_shift_sumcheck,
            2,
            2,
            comb_func,
            transcript,
            network,
        )?;
        drop(_guard);
        drop(span);

        unsafe {
            std::env::remove_var("SUMCHECK_LOG");
        }
        // let num_steps = key.num_steps;
        // let num_steps_bits = num_steps.ilog2() as usize;
        // let (rx_step, rx_constr) = outer_sumcheck_r.split_at(num_steps_bits);

        let claimed_witness_evals =
            opening_accumulator.append(num_rounds_x, transcript, network)?;

        let shift_sumcheck_witness_evals =
            opening_accumulator.append(num_rounds_shift_sumcheck, transcript, network)?;

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
