use ark_ff::BigInteger;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use mpc_core::protocols::additive;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::IoContextPool;
use mpc_core::protocols::rep3::network::Rep3NetworkWorker;
use mpc_core::protocols::rep3::PartyID;
use std::marker::PhantomData;
use tracing::{span, Level};

use jolt_core::field::JoltField;
use jolt_core::poly::multilinear_polynomial::PolynomialEvaluation;
use jolt_core::r1cs::key::UniformSpartanKey;
use jolt_core::utils::math::Math;
use jolt_core::utils::thread::drop_in_background_thread;

use jolt_core::utils::transcript::Transcript;

use jolt_core::poly::{
    dense_mlpoly::DensePolynomial,
    eq_poly::{EqPlusOnePolynomial, EqPolynomial},
};

use crate::jolt::vm::witness::Rep3JoltPolynomials;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::mixed_polynomial::MixedPolynomial;
use crate::poly::opening_proof::Rep3ProverOpeningAccumulator;
use crate::poly::spartan_interleaved_poly::Rep3SpartanInterleavedPolynomial;
use crate::poly::PolyDegree;
use crate::poly::Rep3MultilinearPolynomial;
use crate::subprotocols::sumcheck;
use crate::utils::element::SharedOrPublic;
use crate::utils::element::SharedOrPublicIter;
use jolt_core::r1cs::builder::CombinedUniformBuilder;
use jolt_core::r1cs::inputs::ConstraintInput;

use rayon::prelude::*;

#[derive(Debug, Default)]
pub struct Rep3UniformSpartanProver<F, PCS, ProofTranscript, I, Network> {
    _marker: PhantomData<(F, PCS, ProofTranscript, I, Network)>,
}

impl<F, PCS, ProofTranscript, I, Network>
    Rep3UniformSpartanProver<F, PCS, ProofTranscript, I, Network>
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F, PCS, ProofTranscript, I, Network>
    Rep3UniformSpartanProver<F, PCS, ProofTranscript, I, Network>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    I: ConstraintInput,
    Network: Rep3NetworkWorker,
{
    #[tracing::instrument(skip_all, name = "Rep3UniformSpartan::prove")]
    pub fn prove<const C: usize>(
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
        key: &UniformSpartanKey<C, I, F>,
        polynomials: &Rep3JoltPolynomials<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.id;
        let flattened_polys: Vec<&Rep3MultilinearPolynomial<F>> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(polynomials))
            .collect();

        let num_rounds_x = key.num_rows_bits();

        /* Sumcheck 1: Outer sumcheck */
        let span = span!(Level::INFO, "outer_sumcheck");
        let _guard = span.enter();
        let tau = io_ctx.network().receive_request::<Vec<F>>()?;
        let mut eq_tau = GruenSplitEqPolynomial::new(&tau);

        let mut az_bz_cz_poly =
            compute_spartan_Az_Bz_Cz(constraint_builder, &flattened_polys, party_id);

        let (outer_sumcheck_r, _outer_sumcheck_claims) =
            prove_spartan_cubic_sumcheck(num_rounds_x, &mut eq_tau, &mut az_bz_cz_poly, io_ctx)?;
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        drop_in_background_thread((az_bz_cz_poly, eq_tau));
        drop(_guard);
        drop(span);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness

        /* Sumcheck 2: Inner sumcheck
            RLC of claims Az, Bz, Cz
            where claim_Az = \sum_{y_var} A(rx, y_var || rx_step) * z(y_var || rx_step)
                                + A_shift(..) * z_shift(..)
            and shift denotes the values at the next time step "rx_step+1" for cross-step constraints
            - A_shift(rx, y_var || rx_step) = \sum_t A(rx, y_var || t) * eq_plus_one(rx_step, t)
            - z_shift(y_var || rx_step) = \sum z(y_var || rx_step) * eq_plus_one(rx_step, t)
        */

        let span = span!(Level::INFO, "inner_sumcheck");
        let _guard = span.enter();
        let num_steps = key.num_steps;
        let num_steps_bits = num_steps.ilog2() as usize;
        let num_vars_uniform = key.num_vars_uniform_padded().next_power_of_two();

        let (inner_sumcheck_RLC, claim_inner_joint) =
            io_ctx.network().receive_request::<(F, F)>()?;

        let (rx_step, rx_constr) = outer_sumcheck_r.split_at(num_steps_bits);

        let (eq_rx_step, eq_plus_one_rx_step) = EqPlusOnePolynomial::evals(rx_step, None);

        /* Compute the two polynomials provided as input to the second sumcheck:
           - poly_ABC: A(r_x, y_var || rx_step), A_shift(..) at all variables y_var
           - poly_z: z(y_var || rx_step), z_shift(..)
        */

        let poly_ABC = MixedPolynomial::from_public_evals(
            key.evaluate_matrix_mle_partial(rx_constr, rx_step, inner_sumcheck_RLC),
            party_id,
        );

        // Binding z and z_shift polynomials at point rx_step
        let binding_span = span!(Level::INFO, "binding_z_and_shift_z");
        let binding_guard = binding_span.enter();

        let mut bind_z = vec![SharedOrPublic::zero_public(); num_vars_uniform * 2];
        let mut bind_shift_z = vec![SharedOrPublic::zero_public(); num_vars_uniform * 2];

        flattened_polys
            .par_iter()
            .zip(bind_z.par_iter_mut().zip(bind_shift_z.par_iter_mut()))
            .for_each(|(poly, (eval, eval_shifted))| {
                *eval = poly.dot_product_with_public(&eq_rx_step);
                *eval_shifted = poly.dot_product_with_public(&eq_plus_one_rx_step);
            });

        bind_z[num_vars_uniform] = F::one().into();

        drop(binding_guard);
        drop(binding_span);

        let poly_z = MixedPolynomial::new(
            bind_z.into_iter().chain(bind_shift_z.into_iter()).collect(),
            party_id,
        );

        assert_eq!(poly_z.len(), poly_ABC.len());

        let num_rounds_inner_sumcheck = poly_ABC.len().log_2();

        let mut polys = vec![poly_ABC, poly_z];

        let comb_func = |poly_evals: &[SharedOrPublic<F>]| -> AdditiveShare<F> {
            assert_eq!(poly_evals.len(), 2);
            poly_evals[0].mul(&poly_evals[1]).into_additive(party_id)
        };

        let (inner_sumcheck_r, _claims_inner) = sumcheck::prove_arbitrary_worker(
            &additive::promote_to_trivial_share(claim_inner_joint, party_id),
            num_rounds_inner_sumcheck,
            &mut polys,
            comb_func,
            2,
            io_ctx,
        )?;
        drop(_guard);
        drop(span);
        drop_in_background_thread(polys);

        /*  Sumcheck 3: Shift sumcheck
            sumcheck claim is = z_shift(ry_var || rx_step) = \sum_t z(ry_var || t) * eq_plus_one(rx_step, t)
        */

        let span = span!(Level::INFO, "shift_sumcheck");
        let _guard = span.enter();
        let ry_var = inner_sumcheck_r[1..].to_vec();
        let eq_ry_var = EqPolynomial::evals(&ry_var);
        let eq_ry_var_r2 = EqPolynomial::evals(&ry_var);

        let mut bind_z_ry_var: Vec<SharedOrPublic<F>> = Vec::with_capacity(num_steps);

        let bind_span = span!(Level::INFO, "bind_z_ry_var");
        let bind_guard = bind_span.enter();
        let num_steps_unpadded = constraint_builder.uniform_repeat();
        (0..num_steps_unpadded) // unpadded number of steps is sufficient
            .into_par_iter()
            .map(|t| {
                flattened_polys
                    .iter()
                    .enumerate()
                    .map(|(i, poly)| poly.scale_coeff(t, eq_ry_var[i], eq_ry_var_r2[i]))
                    .sum_for(party_id)
            })
            .collect_into_vec(&mut bind_z_ry_var);
        drop(bind_guard);
        drop(bind_span);

        let num_rounds_shift_sumcheck = num_steps_bits;
        assert_eq!(bind_z_ry_var.len(), eq_plus_one_rx_step.len());

        let mut shift_sumcheck_polys = vec![
            MixedPolynomial::new(bind_z_ry_var, party_id),
            MixedPolynomial::from_public_evals(eq_plus_one_rx_step, party_id),
        ];

        let shift_sumcheck_claim = tracing::trace_span!("shift_sumcheck_claim").in_scope(|| {
            (0..1 << num_rounds_shift_sumcheck)
                .into_par_iter()
                .map(|i| {
                    let params: Vec<_> = shift_sumcheck_polys.iter().map(|poly| poly[i]).collect();
                    comb_func(&params)
                })
                .reduce(|| F::zero(), |acc, x| acc + x)
        });

        io_ctx.network().send_response(shift_sumcheck_claim)?;

        let (shift_sumcheck_r, _shift_sumcheck_claims) = sumcheck::prove_arbitrary_worker(
            &shift_sumcheck_claim,
            num_rounds_shift_sumcheck,
            &mut shift_sumcheck_polys,
            comb_func,
            2,
            io_ctx,
        )?;

        drop(_guard);
        drop(span);

        drop_in_background_thread(shift_sumcheck_polys);

        // Inner sumcheck evaluations: evaluate z on rx_step
        let (claimed_witness_evals, chis) =
            Rep3MultilinearPolynomial::batch_evaluate(&flattened_polys, rx_step);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis),
            rx_step.to_vec(),
            &claimed_witness_evals
                .iter()
                .map(|x| x.into_additive(party_id))
                .collect::<Vec<_>>(),
            io_ctx.main(),
        )?;

        // Shift sumcheck evaluations: evaluate z on ry_var
        let (shift_sumcheck_witness_evals, chis2) =
            Rep3MultilinearPolynomial::batch_evaluate(&flattened_polys, &shift_sumcheck_r);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis2),
            shift_sumcheck_r.to_vec(),
            &shift_sumcheck_witness_evals
                .iter()
                .map(|x| x.into_additive(party_id))
                .collect::<Vec<_>>(),
            io_ctx.main(),
        )?;

        Ok(())
    }
}

#[tracing::instrument(skip_all, name = "Spartan::sumcheck::prove_spartan_cubic")]
fn prove_spartan_cubic_sumcheck<F: JoltField, Network: Rep3NetworkWorker>(
    num_rounds: usize,
    eq_poly: &mut GruenSplitEqPolynomial<F>,
    az_bz_cz_poly: &mut Rep3SpartanInterleavedPolynomial<F>,
    io_ctx: &mut IoContextPool<Network>,
) -> eyre::Result<(Vec<F>, [AdditiveShare<F>; 3])> {
    let mut r: Vec<F> = Vec::new();
    let mut claim = F::zero();

    // TODO: parallelize into subnets
    for round in 0..num_rounds {
        if round == 0 {
            az_bz_cz_poly.first_sumcheck_round(eq_poly, &mut r, &mut claim, io_ctx.main())?;
        } else {
            az_bz_cz_poly.subsequent_sumcheck_round(eq_poly, &mut r, &mut claim, io_ctx.main())?;
        }
    }

    let final_evals = az_bz_cz_poly.final_sumcheck_evals(io_ctx.id);

    io_ctx.network().send_response(final_evals.to_vec())?;

    Ok((r, final_evals))
}

#[tracing::instrument(skip_all)]
pub fn compute_spartan_Az_Bz_Cz<const C: usize, F: JoltField, I: ConstraintInput>(
    constraint_builder: &CombinedUniformBuilder<C, F, I>,
    flattened_polynomials: &[&Rep3MultilinearPolynomial<F>], // N variables of (S steps)
    party_id: PartyID,
) -> Rep3SpartanInterleavedPolynomial<F> {
    Rep3SpartanInterleavedPolynomial::new(
        &constraint_builder.uniform_builder.constraints,
        &constraint_builder.offset_equality_constraints,
        flattened_polynomials,
        constraint_builder.padded_rows_per_step(),
        party_id,
    )
}

// pub fn compute_aux_poly<const C: usize, I: ConstraintInput, F: JoltField>(
//     aux_compute: &AuxComputation<F>,
//     jolt_polynomials: &Rep3JoltPolynomials<F>,
//     poly_len: usize,
//     party_id: PartyID,
// ) -> MultilinearPolynomial<F> {
//     let flattened_polys: Vec<&Rep3MultilinearPolynomial<F>> = I::flatten::<C>()
//         .iter()
//         .map(|var| var.get_ref(jolt_polynomials))
//         .collect();

//     let mut aux_poly: Vec<i64> = vec![0; poly_len];
//     let num_threads = rayon::current_num_threads();
//     let chunk_size = poly_len.div_ceil(num_threads);
//     let contains_negative_values = AtomicBool::new(false);

//     aux_poly
//         .par_chunks_mut(chunk_size)
//         .enumerate()
//         .for_each(|(chunk_index, chunk)| {
//             chunk.iter_mut().enumerate().for_each(|(offset, result)| {
//                 let global_index = chunk_index * chunk_size + offset;
//                 let compute_inputs: Vec<_> = aux_compute
//                     .symbolic_inputs
//                     .iter()
//                     .map(|lc| {
//                         let mut input = SharedOrPublic::<F>::Public(F::zero());
//                         for term in lc.terms().iter() {
//                             match term.0 {
//                                 Variable::Input(index) | Variable::Auxiliary(index) => {
//                                     input.add_assign(flattened_polys[index]
//                                         .get_coeff(global_index)
//                                         .mul_public(F::from_i64(term.1)), party_id)
//                                 }
//                                 Variable::Constant => input.add_assign(F::from_i64(term.1), party_id),
//                             }
//                         }
//                         input
//                     })
//                     .collect();
//                 let aux_value = (self.compute)(&compute_inputs);
//                 if aux_value.is_negative() {
//                     contains_negative_values.store(true, Ordering::Relaxed);
//                 }
//                 *result = aux_value as i64;
//             });
//         });

//     if contains_negative_values.into_inner() {
//         MultilinearPolynomial::from(aux_poly)
//     } else {
//         let aux_poly: Vec<_> = aux_poly.into_iter().map(|x| x as u64).collect();
//         MultilinearPolynomial::from(aux_poly)
//     }
// }
