use std::marker::PhantomData;

use ark_serialize::*;
use co_spartan::mpc::{additive, rep3::Rep3PrimeFieldShare};
use eyre::Context;
use jolt_core::{
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, field::JoltField, unipoly::UniPoly,
    },
    subprotocols::{
        grand_product::{
            BatchedGrandProductArgument, BatchedGrandProductCircuit, LayerProofBatched,
        },
        sumcheck::{CubicSumcheckType, SumcheckInstanceProof},
    },
    utils::{
        math::Math,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    rngs::Rep3CorrelatedRng,
    PartyID,
};
use mpc_net::mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker};

use crate::poly::{combine_poly_shares, Rep3DensePolynomial};
use crate::sumcheck::{self, rep3_prove_cubic_batched, Rep3CubicSumcheckParams};

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductProver<F: JoltField> {
    _marker: PhantomData<F>,
}

impl<F: JoltField> BatchedGrandProductProver<F> {
    #[tracing::instrument(skip_all, name = "BatchedGrandProductArgument.prove")]
    pub fn prove<N: MpcStarNetCoordinator>(
        claims_to_verify: Vec<F>,
        num_layers: usize,
        network: &mut N,
        transcript: &mut ProofTranscript,
    ) -> eyre::Result<(BatchedGrandProductArgument<F>, Vec<F>)> {
        let mut proof_layers: Vec<LayerProofBatched<F>> = Vec::new();
        let num_circuits = claims_to_verify.len();
        let mut coeff_vec: Vec<F> =
            transcript.challenge_vector::<F>(b"rand_coeffs_next_layer", num_circuits);
        let first_claim: F = (0..claims_to_verify.len())
            .map(|i| (claims_to_verify[i] * coeff_vec[i]))
            .sum();
        tracing::info!("claims_to_verify: {:?}", claims_to_verify);
        tracing::info!("first_claim: {:?}", first_claim);
        network.broadcast_request((first_claim, coeff_vec.clone()))?;

        let mut rand = Vec::new();
        for layer_id in (0..num_layers).rev() {
            tracing::info!("layer_id: {:?}", layer_id);
            let span = tracing::span!(tracing::Level::TRACE, "grand_product_layer", layer_id);
            let _enter = span.enter();

            // // approach 2
            // let claim: F = (0..claims_to_verify.len())
            //     .map(|i| claims_to_verify[i] * coeff_vec[i])
            //     .sum(); // recv: num_circuits * 3

            let (num_vars, sumcheck_type, eq) = network
                .receive_responses((0usize, 0u8, DensePolynomial::new(vec![F::zero()])))
                .context("while receiving sumcheck params")?[0].clone();
            if num_vars < 3 {
                tracing::info!("layer_id: {:?} eq len: {:?} eq: {:?}", layer_id, eq.len(), eq);
            }
            let sumcheck_type = CubicSumcheckType::from(sumcheck_type);

            let mut cubic_polys = Vec::new();
            let mut rand_prod = Vec::new();
            for _round in 0..num_vars {
                tracing::info!("round: {:?} sumcheck_type: {:?}", _round, sumcheck_type);
                // if sumcheck_type == CubicSumcheckType::ProdOnes {
                //     let evals = network
                //         .receive_responses::<Vec<(F, F, F)>>(Vec::new())
                //         .context("while receiving evals")?;
                //     let [evals_0, evals_1, evals_2] = evals.try_into().unwrap();
                //     let evals: Vec<(F, F, F)> = evals_0
                //         .into_iter()
                //         .zip(evals_1.into_iter())
                //         .zip(evals_2.into_iter())
                //         .enumerate()
                //         .map(|(i, ((a0, a1), a2))| {
                //             (
                //                 additive::combine_field_element(&a0.0, &a1.0, &a2.0),
                //                 additive::combine_field_element(&a0.1, &a1.1, &a2.1),
                //                 additive::combine_field_element(&a0.2, &a1.2, &a2.2),
                //             )
                //         })
                //         .collect();

                //     tracing::info!("evals: {:?}", evals);
                // }

                let poly_shares = network
                    .receive_responses(Vec::new())
                    .context("while receiving poly shares")?;
                assert_eq!(poly_shares.len(), 3);
                let coeffs: Vec<F> = additive::combine_field_elements(
                    &poly_shares[0],
                    &poly_shares[1],
                    &poly_shares[2],
                );
                // let poly = UniPoly::from_coeff(coeffs);
                let poly = UniPoly::from_coeff(coeffs);
                // tracing::info!("poly coeffs: {:?}", poly.as_vec());
                poly.append_to_transcript(b"poly", transcript);
                cubic_polys.push(poly.compress());
                let r_j = transcript.challenge_scalar(b"challenge_nextround");
                network.broadcast_request(r_j)?;
                rand_prod.push(r_j);

                // tracing::info!("e: {:?}", poly.evaluate(&r_j));
                network.broadcast_request(poly.evaluate(&r_j))?;
                tracing::info!("---------------");
            }
            if cubic_polys.len() > 0 {
                tracing::info!(
                    "{sumcheck_type:?} proof poly[0]: {:?}",
                    cubic_polys.last().unwrap().coeffs_except_linear_term
                );
            }
            let claims_prod: (
                Vec<Vec<Rep3PrimeFieldShare<F>>>,
                Vec<Vec<Rep3PrimeFieldShare<F>>>,
            ) = network
                .receive_responses((Vec::new(), Vec::new()))
                .context("while receiving claims prod")?
                .into_iter()
                .unzip();
            let (claims_poly_A, claims_poly_B) = {
                let (claims_poly_a_shares, claims_poly_b_shares) = claims_prod;
                assert_eq!(claims_poly_a_shares.len(), 3);
                assert_eq!(claims_poly_b_shares.len(), 3);
                let claims_poly_a = rep3::combine_field_elements(
                    claims_poly_a_shares[0].as_slice(),
                    claims_poly_a_shares[1].as_slice(),
                    claims_poly_a_shares[2].as_slice(),
                );
                let claims_poly_b = rep3::combine_field_elements(
                    claims_poly_b_shares[0].as_slice(),
                    claims_poly_b_shares[1].as_slice(),
                    claims_poly_b_shares[2].as_slice(),
                );
                (claims_poly_a, claims_poly_b)
            };

            let proof = SumcheckInstanceProof::new(cubic_polys);

            for i in 0..num_circuits {
                transcript.append_scalar(b"claim_prod_left", &claims_poly_A[i]);

                transcript.append_scalar(b"claim_prod_right", &claims_poly_B[i]);
            }

            tracing::info!("claims_poly_A: {:?}", claims_poly_A);
            tracing::info!("claims_poly_B: {:?}", claims_poly_B);

            if sumcheck_type == CubicSumcheckType::Prod
                || sumcheck_type == CubicSumcheckType::ProdOnes
            {
                // Prod layers must generate an additional random coefficient. The sumcheck randomness indexes into the current layer,
                // but the resulting randomness and claims are about the next layer. The next layer is indexed by an additional variable
                // in the MSB. We use the evaluations V_i(r,0), V_i(r,1) to compute V_i(r, r').

                // produce a random challenge to condense two claims into a single claim

                // approach 1
                let r_layer = transcript.challenge_scalar(b"challenge_r_layer");
                // tracing::info!("r_layer: {:?}", r_layer);

                network.broadcast_request(r_layer)?; // recv: 3 * num_layers send: 1 * num_layers

                // // approach 2
                // let claims_to_verify = (0..num_circuits)
                //     .map(|i| claims_poly_A[i] + r_layer * (claims_poly_B[i] - claims_poly_A[i]))
                //     .collect::<Vec<F>>();
                // network.broadcast_request(claims_to_verify)?; // recv: num_circuits * 3 send: num_circuits * num_layers

                let mut ext = vec![r_layer];
                ext.extend(rand_prod);
                rand = ext;

                proof_layers.push(LayerProofBatched {
                    proof,
                    claims_poly_A,
                    claims_poly_B,
                    combine_prod: true,
                });
            } else {
                // CubicSumcheckType::Flags
                // Flag layers do not need the additional bit as the randomness from the previous layers have already fully determined
                assert_eq!(layer_id, 0);
                rand = rand_prod;

                proof_layers.push(LayerProofBatched {
                    proof,
                    claims_poly_A,
                    claims_poly_B,
                    combine_prod: false,
                });
            }

            tracing::info!("--------------------------------");

            if layer_id != 0 {
                coeff_vec =
                    transcript.challenge_vector::<F>(b"rand_coeffs_next_layer", num_circuits);
                tracing::info!("coeff_vec: {:?}", coeff_vec);
                network.broadcast_request(coeff_vec.clone())?;

                // approach 1
                // let claim_shares = network
                //     .receive_responses(F::zero())
                //     .context("while receiving claim shares")?;
                // assert_eq!(claim_shares.len(), 3);
                // let claim = additive::combine_field_element(
                //     &claim_shares[0],
                //     &claim_shares[1],
                //     &claim_shares[2],
                // ); // recv: 3 * num_layers
                // network.broadcast_request(claim)?;
                // tracing::info!("claim: {:?}", claim);
            }
            drop(_enter);
        }

        Ok((
            BatchedGrandProductArgument {
                proof: proof_layers,
            },
            rand,
        ))
    }

    pub fn prove_worker<N: MpcStarNetWorker>(
        mut batch: BatchedRep3GrandProductCircuit<F>,
        network: &mut N,
    ) -> eyre::Result<Vec<F>> {
        // In first layer coordinator can compute claim since we sent circuits hashes (claims_to_verify) earlier
        let (mut claim, mut coeff_vec) = network.receive_request::<(F, Vec<F>)>()?;
        let mut claims_to_verify = Vec::new();

        tracing::info!("num_layers: {:?}", batch.num_layers());
        tracing::info!("num_circuits: {:?}", batch.circuits.len());

        let mut rand = Vec::new();
        for layer in (0..batch.num_layers()).rev() {
            let span = tracing::trace_span!("grand_product_layer", layer);
            let _enter = span.enter();

            let eq = DensePolynomial::new(EqPolynomial::<F>::new(rand.clone()).evals());

            let params = batch.sumcheck_layer_params(layer, eq.clone());
            network
                .send_response::<(usize, u8, DensePolynomial<F>)>((params.num_rounds, params.sumcheck_type.into(), eq))?;

            let sumcheck_type = params.sumcheck_type;
            let (rand_prod, claims_prod) =
                sumcheck::rep3_prove_cubic_batched(&claim, params, &coeff_vec, network)?;

            let (claims_poly_a, claims_poly_b, _) = claims_prod;

            network.send_response((claims_poly_a.clone(), claims_poly_b.clone()))?;

            if sumcheck_type == CubicSumcheckType::Prod
                || sumcheck_type == CubicSumcheckType::ProdOnes
            {
                // Prod layers must generate an additional random coefficient. The sumcheck randomness indexes into the current layer,
                // but the resulting randomness and claims are about the next layer. The next layer is indexed by an additional variable
                // in the MSB. We use the evaluations V_i(r,0), V_i(r,1) to compute V_i(r, r').

                // produce a random challenge to condense two claims into a single claim
                // approach 1
                let r_layer = network.receive_request()?;
                claims_to_verify = (0..batch.circuits.len())
                    .map(|i| claims_poly_a[i] + (claims_poly_b[i] - claims_poly_a[i]) * r_layer)
                    .collect::<Vec<_>>();

                // // approach 2
                // claims_to_verify = network.receive_request()?; // 1 + num_circuits * num_layers

                let mut ext = vec![r_layer];
                ext.extend(rand_prod);
                rand = ext;
            } else {
                // CubicSumcheckType::Flags
                // Flag layers do not need the additional bit as the randomness from the previous layers have already fully determined
                assert_eq!(layer, 0);
                rand = rand_prod;
            }

            // In all other layers we need to receive coeffs and coordinator help to open claim
            if layer != 0 {
                // produce a fresh set of coeffs and a joint claim
                coeff_vec = network.receive_request()?;
                claim = (0..claims_to_verify.len())
                    .map(|i| (claims_to_verify[i] * coeff_vec[i]).into_additive())
                    .sum(); // okay to remain additive since in the next round we just use it for substraction `e - evals_combined_0`
                            // network.send_response(claim_share)?;
                            // claim = network.receive_request()?;
            }
            drop(_enter);
        }

        Ok(rand)
    }
}

#[derive(Debug, Clone)]
pub struct Rep3GrandProductCircuit<F: JoltField> {
    pub left_vec: Vec<Rep3DensePolynomial<F>>,
    pub right_vec: Vec<Rep3DensePolynomial<F>>,
}

impl<F: JoltField> Rep3GrandProductCircuit<F> {
    #[tracing::instrument(skip_all, name = "GrandProductCircuit::new", level = "trace")]
    pub fn new<N: Rep3Network>(
        leaves: &Rep3DensePolynomial<F>,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Self> {
        let mut left_vec = Vec::new();
        let mut right_vec = Vec::new();

        let num_layers = leaves.len().log_2();
        let (outp_left, outp_right) = leaves.split(leaves.len() / 2);

        left_vec.push(outp_left);
        right_vec.push(outp_right);

        for i in 0..num_layers - 1 {
            let (outp_left, outp_right) = Self::compute_layer(&left_vec[i], &right_vec[i], io_ctx)?;
            left_vec.push(outp_left);
            right_vec.push(outp_right);
        }

        Ok(Self {
            left_vec,
            right_vec,
        })
    }

    #[tracing::instrument(skip_all, name = "GrandProductCircuit::new_split", level = "trace")]
    pub fn new_split<N: Rep3Network>(
        left_leaves: Rep3DensePolynomial<F>,
        right_leaves: Rep3DensePolynomial<F>,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Self> {
        let num_layers = left_leaves.len().log_2() + 1;
        let mut left_vec = Vec::with_capacity(num_layers);
        let mut right_vec = Vec::with_capacity(num_layers);

        left_vec.push(left_leaves);
        right_vec.push(right_leaves);

        for i in 0..num_layers - 1 {
            let (outp_left, outp_right) = Self::compute_layer(&left_vec[i], &right_vec[i], io_ctx)?;
            left_vec.push(outp_left);
            right_vec.push(outp_right);
        }

        Ok(Self {
            left_vec,
            right_vec,
        })
    }

    pub fn evaluate(&self) -> F {
        let len = self.left_vec.len();
        assert_eq!(self.left_vec[len - 1].num_vars(), 0);
        assert_eq!(self.right_vec[len - 1].num_vars(), 0);
        self.left_vec[len - 1][0] * self.right_vec[len - 1][0]
    }

    pub fn take_layer(
        &mut self,
        layer_id: usize,
    ) -> (Rep3DensePolynomial<F>, Rep3DensePolynomial<F>) {
        let left = std::mem::replace(
            &mut self.left_vec[layer_id],
            Rep3DensePolynomial::new(vec![Rep3PrimeFieldShare::zero_share()]),
        );
        let right = std::mem::replace(
            &mut self.right_vec[layer_id],
            Rep3DensePolynomial::new(vec![Rep3PrimeFieldShare::zero_share()]),
        );
        (left, right)
    }

    fn compute_layer<N: Rep3Network>(
        inp_left: &Rep3DensePolynomial<F>,
        inp_right: &Rep3DensePolynomial<F>,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<(Rep3DensePolynomial<F>, Rep3DensePolynomial<F>)> {
        let len = inp_left.len() + inp_right.len();

        let outp_left = rep3::arithmetic::mul_vec(
            &inp_left.evals_ref()[0..len / 4],
            &inp_right.evals_ref()[0..len / 4],
            io_ctx,
        )
        .context("while multiplying left")?;
        let outp_right = rep3::arithmetic::mul_vec(
            &inp_left.evals_ref()[len / 4..len / 2],
            &inp_right.evals_ref()[len / 4..len / 2],
            io_ctx,
        )
        .context("while multiplying right")?;

        Ok((
            Rep3DensePolynomial::new(outp_left),
            Rep3DensePolynomial::new(outp_right),
        ))
    }
}

pub struct BatchedRep3GrandProductCircuit<F: JoltField> {
    pub circuits: Vec<Rep3GrandProductCircuit<F>>,

    pub flags_present: bool,
    pub flags: Option<Vec<DensePolynomial<F>>>,
    pub fingerprint_polys: Option<Vec<Rep3DensePolynomial<F>>>,
}

impl<F: JoltField> BatchedRep3GrandProductCircuit<F> {
    pub fn new_batch(circuits: Vec<Rep3GrandProductCircuit<F>>) -> Self {
        Self {
            circuits,
            flags_present: false,
            flags: None,
            fingerprint_polys: None,
        }
    }

    pub fn new_batch_flags(
        circuits: Vec<Rep3GrandProductCircuit<F>>,
        flags: Vec<DensePolynomial<F>>,
        fingerprint_polys: Vec<Rep3DensePolynomial<F>>,
    ) -> Self {
        assert_eq!(circuits.len(), fingerprint_polys.len());

        Self {
            circuits,
            flags_present: true,
            flags: Some(flags),
            fingerprint_polys: Some(fingerprint_polys),
        }
    }

    pub fn num_layers(&self) -> usize {
        let prod_layers = self.circuits[0].left_vec.len();

        if self.flags.is_some() {
            prod_layers + 1
        } else {
            prod_layers
        }
    }

    pub fn sumcheck_layer_params(
        &mut self,
        layer_id: usize,
        eq: DensePolynomial<F>,
    ) -> Rep3CubicSumcheckParams<F> {
        if self.flags_present && layer_id == 0 {
            let flags = self.flags.as_ref().unwrap();
            debug_assert_eq!(flags[0].len(), eq.len());

            let num_rounds = eq.get_num_vars();

            // Each of these is needed exactly once, transfer ownership rather than clone.
            let fingerprint_polys = self.fingerprint_polys.take().unwrap();
            let flags = self.flags.take().unwrap();
            Rep3CubicSumcheckParams::new_flags(fingerprint_polys, flags, eq, num_rounds)
        } else {
            // If flags is present layer_id 1 corresponds to circuits.left_vec/right_vec[0]
            let layer_id = if self.flags_present {
                layer_id - 1
            } else {
                layer_id
            };

            let num_rounds = self.circuits[0].left_vec[layer_id].num_vars();

            let (lefts, rights): (Vec<Rep3DensePolynomial<F>>, Vec<Rep3DensePolynomial<F>>) = self
                .circuits
                .iter_mut()
                .map(|circuit| circuit.take_layer(layer_id))
                .unzip();
            if self.flags_present {
                Rep3CubicSumcheckParams::new_prod_ones(lefts, rights, eq, num_rounds)
            } else {
                Rep3CubicSumcheckParams::new_prod(lefts, rights, eq, num_rounds)
            }
        }
    }
}
