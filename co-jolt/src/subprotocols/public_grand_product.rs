use std::iter;

use crate::{
    field::JoltField,
    poly::split_eq_poly::DistributedSplitEqPolynomial,
    subprotocols::{
        grand_product::{
            Rep3BatchedGrandProduct, Rep3BatchedGrandProductLayer,
            Rep3BatchedGrandProductLayerWorker, Rep3BatchedGrandProductWorker,
        },
        sumcheck::{Rep3BatchedCubicSumcheck, Rep3BatchedCubicSumcheckWorker, Rep3Bindable},
    },
};
use eyre::Context;
use itertools::Itertools;
use jolt_core::{
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        dense_interleaved_poly::DenseInterleavedPolynomial,
        dense_mlpoly::DensePolynomial,
        split_eq_poly::SplitEqPolynomial,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::{
        grand_product::{
            BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductLayerProof,
            BatchedGrandProductProof,
        },
        sumcheck::{Bindable, SumcheckInstanceProof},
    },
    utils::transcript::{AppendToTranscript, Transcript},
};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        network::{IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker},
        PartyID,
    },
};

use rayon::prelude::*;
use snarks_core::math::Math;

impl<F: JoltField, PCS, ProofTranscript, Network>
    Rep3BatchedGrandProductWorker<F, PCS, ProofTranscript, Network> for BatchedDenseGrandProduct<F>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    type Leaves = (Vec<F>, usize, usize);

    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedDenseGrandProduct::construct",
        level = "trace"
    )]
    fn construct(
        leaves: Self::Leaves,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<(Self, usize)> {
        let (leaves, batch_size, full_batch_size) = leaves;

        if io_ctx.party_idx() != 0 {
            return Ok((
                Self {
                    batch_size: 0,
                    layers: Vec::new(),
                },
                full_batch_size,
            ));
        }
        Ok((
            <Self as BatchedGrandProduct<F, PCS, ProofTranscript>>::construct((leaves, batch_size)),
            full_batch_size,
        ))
    }

    fn batch_size_minus_delta(&self) -> usize {
        self.batch_size
    }

    fn is_worker_symmetric(&self) -> bool {
        true
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedDenseGrandProduct::claimed_outputs",
        level = "trace"
    )]
    fn claimed_outputs(&self) -> Option<Vec<AdditiveShare<F>>> {
        if self.layers.is_empty() {
            return None;
        }
        let last_layer = &self.layers[self.layers.len() - 1];
        Some(
            last_layer
                .par_chunks(2)
                .map(|chunk| AdditiveShare::from_fe(chunk[0] * chunk[1]))
                .collect(),
        )
    }

    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn Rep3BatchedGrandProductLayerWorker<F, Network>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn Rep3BatchedGrandProductLayerWorker<F, Network>)
            .rev()
    }

    fn prove_grand_product_worker(
        &mut self,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Vec<F>> {
        if io_ctx.party_idx() != 0 {
            return io_ctx
                .network()
                .recv(PartyID::ID0)
                .context("while receiving r");
        }
        let mut r = io_ctx.network().receive_request()?;
        let mut eq_chunk_size = self.batch_size;

        for layer in self.layers.iter_mut().rev() {
            layer.prove_layer(&mut r, eq_chunk_size, true, io_ctx)?;
            eq_chunk_size *= 2;
        }

        io_ctx.network().send(PartyID::ID0.next_id(), r.clone())?;
        io_ctx.network().send(PartyID::ID0.prev_id(), r.clone())?;

        Ok(r)
    }
}

impl<F: JoltField, PCS, ProofTranscript, Network>
    Rep3BatchedGrandProduct<F, PCS, ProofTranscript, Network> for BatchedDenseGrandProduct<F>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn construct(num_layers: usize, batch_size: usize) -> Self {
        Self {
            layers: vec![DenseInterleavedPolynomial::default(); num_layers],
            batch_size,
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn is_worker_symmetric(&self) -> bool {
        true
    }

    fn layers(
        &'_ self,
    ) -> impl Iterator<Item = &'_ dyn Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>>
    {
        iter::empty()
    }

    fn receive_hashes(network: &mut Network) -> eyre::Result<Vec<F>> {
        Ok(network
            .receive_response_from_workers::<Vec<F>>(PartyID::ID0)
            .context("while receiving hashes")?
            .concat())
    }

    fn cooridinate_prove_grand_product(
        &self,
        claimed_outputs: Vec<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(BatchedGrandProductProof<PCS, ProofTranscript>, Vec<F>)> {
        let mut proof_layers = Vec::with_capacity(self.layers.len());

        // Evaluate the MLE of the output layer at a random point to reduce the outputs to
        // a single claim.
        transcript.append_scalars(&claimed_outputs);
        let output_mle = DensePolynomial::new_padded(claimed_outputs);
        let mut r_grand_product: Vec<F> = transcript.challenge_vector(output_mle.get_num_vars());
        let mut claim = output_mle.evaluate(&r_grand_product);
        network.send_request_to_workers(PartyID::ID0, r_grand_product.clone())?;

        for layer in self.layers.iter().rev() {
            proof_layers.push(layer.coordinate_prove_layer(
                &mut claim,
                &mut r_grand_product,
                true,
                transcript,
                network,
            )?);
        }

        Ok((
            BatchedGrandProductProof {
                gkr_layers: proof_layers,
                quark_proof: None,
            },
            r_grand_product,
        ))
    }
}

impl<F: JoltField, ProofTranscript, Network> Rep3BatchedCubicSumcheck<F, ProofTranscript, Network>
    for DenseInterleavedPolynomial<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    #[tracing::instrument(
        skip_all,
        name = "BatchedCubicSumcheck::prove_sumcheck",
        level = "trace"
    )]
    fn coordinate_prove_sumcheck(
        &self,
        claim: &F,
        r_grand_product: &[F],
        num_rounds: usize,
        _: bool,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, (F, F))> {
        let mut previous_claim = *claim;
        let log_num_workers = network.active_num_workers().log_2();

        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..num_rounds - log_num_workers {
            let mut round_evals = if network.is_distributed() {
                let subnet_responces =
                    network.receive_response_from_workers::<Vec<F>>(PartyID::ID0)?;
                let degree = subnet_responces[0].len();

                subnet_responces
                    .into_iter()
                    .fold(vec![F::zero(); degree], |mut acc, coeff| {
                        acc.iter_mut().zip(coeff.iter()).for_each(|(acc, coeff)| {
                            *acc += coeff;
                        });
                        acc
                    })
            } else {
                network.receive_response(PartyID::ID0, 0)?
            };

            round_evals.insert(1, previous_claim - round_evals[0]);

            let round_poly = UniPoly::<F>::from_evals(&round_evals);
            let compressed_poly = round_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            // derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();

            r.push(r_j);

            previous_claim = round_poly.evaluate(&r_j);

            network.send_request_to_workers(PartyID::ID0, r_j)?;
            cubic_polys.push(compressed_poly);
        }

        let mut sumcheck_proof = SumcheckInstanceProof::new(cubic_polys);

        let final_claims = if network.is_distributed() {
            self.prove_remaining_rounds(
                r_grand_product,
                &mut r,
                previous_claim,
                &mut sumcheck_proof,
                transcript,
                network,
            )?
        } else {
            let final_claims = network.receive_response::<Vec<F>>(PartyID::ID0, 0)?;

            (final_claims[0], final_claims[1])
        };

        Ok((sumcheck_proof, r, final_claims))
    }

    fn receive_final_claims(&self, _: &mut Network) -> eyre::Result<(F, F)> {
        unimplemented!()
    }

    fn prove_remaining_rounds(
        &self,
        r_grand_product: &[F],
        r: &mut Vec<F>,
        previous_claim: F,
        proof: &mut SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(F, F)> {
        let evals = network
            .receive_response_from_workers::<Vec<F>>(PartyID::ID0)?
            .into_iter()
            .flatten()
            .collect_vec();

        let mut eq_poly = SplitEqPolynomial::new_bind(r_grand_product, r);

        let mut layer = DenseInterleavedPolynomial::new(evals);

        let (proof_, r_, final_claims) = <DenseInterleavedPolynomial<F> as jolt_core::subprotocols::sumcheck::BatchedCubicSumcheck<
            F,
            ProofTranscript,
        >>::prove_sumcheck(
            &mut layer, &previous_claim, &mut eq_poly, transcript
        );

        network.send_request_to_workers(PartyID::ID0, r_.clone())?;
        proof.compressed_polys.extend(proof_.compressed_polys);
        r.extend(r_);

        Ok(final_claims)
    }
}

impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedGrandProductLayerWorker<F, Network>
    for DenseInterleavedPolynomial<F>
{
}

impl<F: JoltField, ProofTranscript, Network>
    Rep3BatchedGrandProductLayer<F, ProofTranscript, Network> for DenseInterleavedPolynomial<F>
where
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    /// Proves a single layer of a batched grand product circuit
    fn coordinate_prove_layer(
        &self,
        claim: &mut F,
        r_grand_product: &mut Vec<F>,
        worker_symmetric: bool,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<BatchedGrandProductLayerProof<F, ProofTranscript>> {
        let num_rounds = r_grand_product.len();

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) = self.coordinate_prove_sumcheck(
            claim,
            r_grand_product,
            num_rounds,
            worker_symmetric,
            transcript,
            network,
        )?;

        let (left_claim, right_claim) = sumcheck_claims;
        transcript.append_scalar(&left_claim);
        transcript.append_scalar(&right_claim);

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        // produce a random challenge to condense two claims into a single claim
        let r_layer: F = transcript.challenge_scalar();

        *claim = left_claim + r_layer * (right_claim - left_claim);

        network.send_request_to_workers(PartyID::ID0, r_layer)?;
        r_grand_product.push(r_layer);

        Ok(BatchedGrandProductLayerProof {
            proof: sumcheck_proof,
            left_claim,
            right_claim,
        })
    }
}

impl<F: JoltField> Rep3Bindable<F> for DenseInterleavedPolynomial<F> {
    #[tracing::instrument(skip_all, name = "DenseInterleavedPolynomial::bind", level = "trace")]
    fn bind(&mut self, r: F, _: PartyID) {
        <Self as Bindable<F>>::bind(self, r);
    }
}

impl<F: JoltField, Network: Rep3NetworkWorker> Rep3BatchedCubicSumcheckWorker<F, Network>
    for DenseInterleavedPolynomial<F>
{
    #[tracing::instrument(
        skip_all,
        name = "Rep3DenseInterleavedPolynomial::compute_cubic",
        level = "trace"
    )]
    fn compute_cubic(
        &self,
        eq_poly: &DistributedSplitEqPolynomial<F>,
        _: PartyID,
    ) -> [AdditiveShare<F>; 3] {
        let cubic_evals = if eq_poly.E1_len == 1 {
            self.coeffs[..self.len()]
                .par_chunks(4)
                .zip(eq_poly.E2.par_chunks(2))
                .map(|(layer_chunk, eq_chunk)| {
                    let eq_evals = {
                        let eval_point_0 = eq_chunk[0];
                        let m_eq = eq_chunk[1] - eq_chunk[0];
                        let eval_point_2 = eq_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };
                    let left = (
                        *layer_chunk.first().unwrap_or(&F::zero()),
                        *layer_chunk.get(2).unwrap_or(&F::zero()),
                    );
                    let right = (
                        *layer_chunk.get(1).unwrap_or(&F::zero()),
                        *layer_chunk.get(3).unwrap_or(&F::zero()),
                    );

                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let left_eval_2 = left.1 + m_left;
                    let left_eval_3 = left_eval_2 + m_left;

                    let right_eval_2 = right.1 + m_right;
                    let right_eval_3 = right_eval_2 + m_right;

                    (
                        eq_evals.0 * left.0 * right.0,
                        eq_evals.1 * left_eval_2 * right_eval_2,
                        eq_evals.2 * left_eval_3 * right_eval_3,
                    )
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
        } else {
            let E1_evals: Vec<_> = eq_poly.E1[..eq_poly.E1_len]
                .par_chunks(2)
                .map(|E1_chunk| {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            let chunk_size = (self.len().next_power_of_two() / eq_poly.E2_len).max(1);
            eq_poly.E2[..eq_poly.E2_len]
                .par_iter()
                .zip(self.par_chunks(chunk_size))
                .map(|(E2_eval, P_x2)| {
                    let mut inner_sum = (F::zero(), F::zero(), F::zero());
                    for (E1_evals, P_chunk) in E1_evals.iter().zip(P_x2.chunks(4)) {
                        let left = (
                            *P_chunk.first().unwrap_or(&F::zero()),
                            *P_chunk.get(2).unwrap_or(&F::zero()),
                        );
                        let right = (
                            *P_chunk.get(1).unwrap_or(&F::zero()),
                            *P_chunk.get(3).unwrap_or(&F::zero()),
                        );
                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        inner_sum.0 += E1_evals.0 * left.0 * right.0;
                        inner_sum.1 += E1_evals.1 * left_eval_2 * right_eval_2;
                        inner_sum.2 += E1_evals.2 * left_eval_3 * right_eval_3;
                    }

                    (
                        *E2_eval * inner_sum.0,
                        *E2_eval * inner_sum.1,
                        *E2_eval * inner_sum.2,
                    )
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
        };

        AdditiveShare::from_fe_vec(vec![cubic_evals.0, cubic_evals.1, cubic_evals.2])
            .try_into()
            .unwrap()
    }

    fn final_evals(&self, _: usize, _: PartyID) -> Vec<AdditiveShare<F>> {
        self.coeffs[..self.len()]
            .par_iter()
            .map(|c| AdditiveShare::from_fe(*c))
            .collect()
    }
}
