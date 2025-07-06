use ark_serialize::*;
use jolt_core::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, field::JoltField},
    subprotocols::{
        grand_product::{BatchedGrandProductCircuit, LayerProofBatched},
        sumcheck::{CubicSumcheckParams, CubicSumcheckType, SumcheckInstanceProof},
    },
    utils::transcript::ProofTranscript,
};
use spartan::transcript::Transcript;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductArgument<F: JoltField> {
    proof: Vec<LayerProofBatched<F>>,
}

impl<F: JoltField> BatchedGrandProductArgument<F> {
    #[tracing::instrument(skip_all, name = "BatchedGrandProductArgument.prove")]
    pub fn prove(
        mut batch: BatchedGrandProductCircuit<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>) {
        let mut proof_layers: Vec<LayerProofBatched<F>> = Vec::new();
        let mut claims_to_verify = (0..batch.circuits.len())
            .map(|i| batch.circuits[i].evaluate())
            .collect::<Vec<F>>();

        let mut rand = Vec::new();
        for layer_id in (0..batch.num_layers()).rev() {
            let span = tracing::span!(tracing::Level::TRACE, "grand_product_layer", layer_id);
            let _enter = span.enter();

            // produce a fresh set of coeffs and a joint claim
            let coeff_vec: Vec<F> =
                transcript.challenge_vector::<F>(b"rand_coeffs_next_layer", claims_to_verify.len());
            let claim = (0..claims_to_verify.len())
                .map(|i| claims_to_verify[i] * coeff_vec[i])
                .sum();

            let eq = DensePolynomial::new(EqPolynomial::<F>::new(rand.clone()).evals());
            let params = batch.sumcheck_layer_params(layer_id, eq);
            let sumcheck_type = params.sumcheck_type.clone();
            let (proof, rand_prod, claims_prod) =
                SumcheckInstanceProof::prove_cubic_batched(&claim, params, &coeff_vec, transcript);

            let (claims_poly_A, claims_poly_B, _claim_eq) = claims_prod;
            for i in 0..batch.circuits.len() {
                transcript.append_scalar(b"claim_prod_left", &claims_poly_A[i]);

                transcript.append_scalar(b"claim_prod_right", &claims_poly_B[i]);
            }

            if sumcheck_type == CubicSumcheckType::Prod
                || sumcheck_type == CubicSumcheckType::ProdOnes
            {
                // Prod layers must generate an additional random coefficient. The sumcheck randomness indexes into the current layer,
                // but the resulting randomness and claims are about the next layer. The next layer is indexed by an additional variable
                // in the MSB. We use the evaluations V_i(r,0), V_i(r,1) to compute V_i(r, r').

                // produce a random challenge to condense two claims into a single claim
                let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

                claims_to_verify = (0..batch.circuits.len())
                    .map(|i| claims_poly_A[i] + r_layer * (claims_poly_B[i] - claims_poly_A[i]))
                    .collect::<Vec<F>>();

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
            drop(_enter);
        }

        (
            BatchedGrandProductArgument {
                proof: proof_layers,
            },
            rand,
        )
    }

    pub fn verify(
        &self,
        claims_prod_vec: &Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (Vec<F>, Vec<F>) {
        let mut rand: Vec<F> = Vec::new();
        let num_layers = self.proof.len();

        let mut claims_to_verify = claims_prod_vec.to_owned();
        for (num_rounds, i) in (0..num_layers).enumerate() {
            // produce random coefficients, one for each instance
            let coeff_vec =
                transcript.challenge_vector::<F>(b"rand_coeffs_next_layer", claims_to_verify.len());

            // produce a joint claim
            let claim = (0..claims_to_verify.len())
                .map(|i| claims_to_verify[i] * coeff_vec[i])
                .sum();

            let (claim_last, rand_prod) = self.proof[i].verify(claim, num_rounds, 3, transcript);

            let claims_prod_left = &self.proof[i].claims_poly_A;
            let claims_prod_right = &self.proof[i].claims_poly_B;
            assert_eq!(claims_prod_left.len(), claims_prod_vec.len());
            assert_eq!(claims_prod_right.len(), claims_prod_vec.len());

            for i in 0..claims_prod_vec.len() {
                transcript.append_scalar(b"claim_prod_left", &claims_prod_left[i]);
                transcript.append_scalar(b"claim_prod_right", &claims_prod_right[i]);
            }

            assert_eq!(rand.len(), rand_prod.len());
            let eq: F = (0..rand.len())
                .map(|i| rand[i] * rand_prod[i] + (F::one() - rand[i]) * (F::one() - rand_prod[i]))
                .product();

            // Compute the claim_expected which is a random linear combination of the batched evaluations.
            // The evaluation is the combination of eq / A / B depending on the cubic layer type (flags / prod).
            // We also compute claims_to_verify which computes sumcheck_cubic_poly(r, r') from
            // sumcheck_cubic_poly(r, 0), sumcheck_subic_poly(r, 1)
            let claim_expected = if self.proof[i].combine_prod {
                let claim_expected: F = (0..claims_prod_vec.len())
                    .map(|i| {
                        coeff_vec[i]
                            * CubicSumcheckParams::combine_prod(
                                &claims_prod_left[i],
                                &claims_prod_right[i],
                                &eq,
                            )
                    })
                    .sum();

                // produce a random challenge
                let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

                claims_to_verify = (0..claims_prod_left.len())
                    .map(|i| {
                        claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i])
                    })
                    .collect::<Vec<F>>();

                let mut ext = vec![r_layer];
                ext.extend(rand_prod);
                rand = ext;

                claim_expected
            } else {
                let claim_expected: F = (0..claims_prod_vec.len())
                    .map(|i| {
                        coeff_vec[i]
                            * CubicSumcheckParams::combine_flags(
                                &claims_prod_left[i],
                                &claims_prod_right[i],
                                &eq,
                            )
                    })
                    .sum();

                rand = rand_prod;

                claims_to_verify = (0..claims_prod_left.len())
                    .map(|i| {
                        claims_prod_left[i] * claims_prod_right[i]
                            + (F::one() - claims_prod_right[i])
                    })
                    .collect::<Vec<F>>();

                claim_expected
            };

            assert_eq!(claim_expected, claim_last);
        }
        (claims_to_verify, rand)
    }
}
