use jolt_core::{
    poly::{commitment::commitment_scheme::CommitmentScheme, split_eq_poly::SplitEqPolynomial},
    subprotocols::grand_product::{
        BatchedGrandProductLayer, BatchedGrandProductLayerProof, BatchedGrandProductProof,
    },
    utils::thread::drop_in_background_thread,
};
use jolt_core::{
    poly::{dense_interleaved_poly::DenseInterleavedPolynomial, dense_mlpoly::DensePolynomial},
    utils::{math::Math, transcript::Transcript},
};
use mpc_core::protocols::{
    additive,
    rep3::{network::IoContextPool, Rep3PrimeFieldShare},
};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::{
        self,
        network::{Rep3NetworkCoordinator, Rep3NetworkWorker},
    },
};

use rayon::prelude::*;

use crate::field::JoltField;
use crate::{
    poly::{
        dense_interleaved_poly::Rep3DenseInterleavedPolynomial,
        opening_proof::Rep3ProverOpeningAccumulator,
    },
    subprotocols::sumcheck::{Rep3BatchedCubicSumcheck, Rep3BatchedCubicSumcheckWorker},
};

pub trait Rep3BatchedGrandProduct<F, PCS, ProofTranscript, Network: Rep3NetworkCoordinator>:
    Sized
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// Constructs the grand product circuit(s) from `leaves` with the default configuration
    fn construct(num_layers: usize) -> Self;

    /// The number of layers in the grand product.
    fn num_layers(&self) -> usize;

    /// Returns an iterator over the layers of this batched grand product circuit.
    /// Each layer is mutable so that its polynomials can be bound over the course
    /// of proving.
    fn layers(
        &'_ self,
    ) -> impl Iterator<Item = &'_ dyn Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>>;

    /// Computes a batched grand product proof, layer by layer.
    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::prove_grand_product")]
    fn cooridinate_prove_grand_product(
        &self,
        claimed_outputs: Vec<F>,
        remaining_layers: Option<Vec<impl BatchedGrandProductLayer<F, ProofTranscript>>>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(BatchedGrandProductProof<PCS, ProofTranscript>, Vec<F>)> {
        let mut proof_layers = Vec::with_capacity(
            self.num_layers() + remaining_layers.as_ref().map_or(0, |v| v.len()),
        );

        // Evaluate the MLE of the output layer at a random point to reduce the outputs to
        // a single claim.
        transcript.append_scalars(&claimed_outputs);
        let output_mle = DensePolynomial::new_padded(claimed_outputs);
        let mut r: Vec<F> = transcript.challenge_vector(output_mle.get_num_vars());
        let mut claim = output_mle.evaluate(&r);
        println!("coordinator | initial claim: {} r: {:?}", claim, r);

        if let Some(remaining_layers) = remaining_layers {
            for mut layer in remaining_layers {
                proof_layers.push(layer.prove_layer(&mut claim, &mut r, transcript));
            }
        }

        println!(
            "coordinator | remaining layers r: [{}..{}] claim: {}",
            r[0],
            r.last().unwrap(),
            claim
        );

        println!("--------------------");

        network.broadcast_request((r.clone(), claim))?;

        for layer in self.layers() {
            proof_layers
                .push(layer.coordinate_prove_layer(&mut claim, &mut r, transcript, network)?);
        }

        Ok((
            BatchedGrandProductProof {
                gkr_layers: proof_layers,
                quark_proof: None,
            },
            r,
        ))
    }
}

pub trait Rep3BatchedGrandProductWorker<F: JoltField, PCS, ProofTranscript, Network>:
    Sized
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    type Leaves;

    /// Constructs the grand product circuit(s) from `leaves` with the default configuration
    fn construct(leaves: Self::Leaves, io_ctx: &mut IoContextPool<Network>) -> eyre::Result<Self>;

    /// The number of layers in the grand product.
    fn num_layers(&self) -> usize;

    /// The claimed outputs of the grand products.
    fn claimed_outputs(&self) -> Vec<AdditiveShare<F>>;

    /// Returns an iterator over the layers of this batched grand product circuit.
    /// Each layer is mutable so that its polynomials can be bound over the course
    /// of proving.
    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn Rep3BatchedGrandProductLayerWorker<F, Network>>;

    /// Computes a batched grand product proof, layer by layer.
    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::prove_grand_product")]
    fn prove_grand_product_worker(
        &mut self,
        _opening_accumulator: Option<&mut Rep3ProverOpeningAccumulator<F>>,
        _setup: Option<&PCS::Setup>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Vec<F>> {
        let (mut r, claim): (Vec<F>, F) = io_ctx.network().receive_request()?;
        let mut claim = additive::promote_to_trivial_share(claim, io_ctx.party_id());
        for layer in self.layers().into_iter() {
            layer.prove_layer(&mut claim, &mut r, io_ctx)?;
        }

        Ok(r)
    }
}

pub trait Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>:
    Rep3BatchedCubicSumcheck<F, ProofTranscript, Network> + std::fmt::Debug
where
    F: JoltField,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    /// Proves a single layer of a batched grand product circuit
    fn coordinate_prove_layer(
        &self,
        claim: &mut F,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<BatchedGrandProductLayerProof<F, ProofTranscript>> {
        // let num_rounds = network.receive_response::<usize>(rep3::PartyID::ID0, 0)?;
        let num_rounds = r_grand_product.len();

        println!("coordinator | previous_claim: {:?}", claim);

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.coordinate_prove_sumcheck(claim, num_rounds, transcript, network)?;

        println!("coordinator | r_sumcheck: {:?}", r_sumcheck);

        // println!("r_grand_product: {:?}", r_grand_product);
        // println!("r_sumcheck: {:?}", r_sumcheck);

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
        println!(
            "r_layer: [{}..{}] claim {}",
            r_grand_product[0], r_layer, claim
        );
        network.broadcast_request(r_layer)?;
        r_grand_product.push(r_layer);

        Ok(BatchedGrandProductLayerProof {
            proof: sumcheck_proof,
            left_claim,
            right_claim,
        })
    }
}

pub trait Rep3BatchedGrandProductLayerWorker<F: JoltField, Network: Rep3NetworkWorker>:
    Rep3BatchedCubicSumcheckWorker<F, Network> + std::fmt::Debug
{
    /// Proves a single layer of a batched grand product circuit
    #[tracing::instrument(
        skip_all,
        name = "BatchedGrandProductLayer::prove_layer",
        level = "trace"
    )]
    fn prove_layer(
        &mut self,
        claim: &mut AdditiveShare<F>,
        r_grand_product: &mut Vec<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let mut eq_poly = SplitEqPolynomial::new_chunk(
            r_grand_product,
            io_ctx.log_num_workers_per_party(),
            io_ctx.worker_idx(),
        );
        // if io_ctx.worker_idx() == 0 {
        //     let eq_evals = SplitEqPolynomial::new(r_grand_product).merge().Z;
        //     let eq_evals_1 = SplitEqPolynomial::new_chunk(
        //         r_grand_product,
        //         io_ctx.log_num_workers_per_party(),
        //         0,
        //     )
        //     .merge()
        //     .Z;
        //     let eq_evals_2 = SplitEqPolynomial::new_chunk(
        //         r_grand_product,
        //         io_ctx.log_num_workers_per_party(),
        //         1,
        //     )
        //     .merge()
        //     .Z;
        //     assert_eq!(eq_evals_1.len() + eq_evals_2.len(), eq_evals.len());
        //     assert_eq!([eq_evals_1, eq_evals_2].concat(), eq_evals);
        //     println!("eq_poly split correct");
        // }
        // let mut eq_poly = SplitEqPolynomial::new(r_grand_product);

        let (r_sumcheck, sumcheck_claims) = self.prove_sumcheck(claim, &mut eq_poly, io_ctx)?;

        drop_in_background_thread(eq_poly);

        let (left_claim, right_claim) = sumcheck_claims;

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = io_ctx.network().receive_request()?;
        *claim = rep3::arithmetic::add_mul_public(left_claim, right_claim - left_claim, r_layer)
            .into_additive();

        r_grand_product.push(r_layer);

        Ok(())
    }
}

pub struct Rep3BatchedDenseGrandProduct<F: JoltField> {
    layers: Vec<Rep3DenseInterleavedPolynomial<F>>,
}

impl<F: JoltField, PCS, ProofTranscript, Network>
    Rep3BatchedGrandProductWorker<F, PCS, ProofTranscript, Network>
    for Rep3BatchedDenseGrandProduct<F>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    type Leaves = (Vec<Rep3PrimeFieldShare<F>>, usize);

    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedDenseGrandProduct::construct",
        level = "trace"
    )]
    fn construct(leaves: Self::Leaves, io_ctx: &mut IoContextPool<Network>) -> eyre::Result<Self> {
        let (leaves, batch_size) = leaves;
        assert!(leaves.len() % batch_size == 0);
        assert!((leaves.len() / batch_size).is_power_of_two());

        let num_layers = (leaves.len() / batch_size).log_2();
        let mut layers: Vec<Rep3DenseInterleavedPolynomial<F>> = Vec::with_capacity(num_layers);
        layers.push(Rep3DenseInterleavedPolynomial::new(leaves));

        for i in 0..num_layers - 1 {
            let previous_layer = &layers[i];
            let new_layer = previous_layer.layer_output(io_ctx)?;
            layers.push(new_layer);
        }

        Ok(Self { layers })
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedDenseGrandProduct::claimed_outputs",
        level = "trace"
    )]
    fn claimed_outputs(&self) -> Vec<AdditiveShare<F>> {
        let last_layer = &self.layers[self.layers.len() - 1];
        last_layer
            .par_chunks(2)
            .map(|chunk| chunk[0] * chunk[1])
            .collect()
    }

    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn Rep3BatchedGrandProductLayerWorker<F, Network>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn Rep3BatchedGrandProductLayerWorker<F, Network>)
            .rev()
    }
}

impl<F: JoltField, PCS, ProofTranscript, Network>
    Rep3BatchedGrandProduct<F, PCS, ProofTranscript, Network> for Rep3BatchedDenseGrandProduct<F>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn construct(num_layers: usize) -> Self {
        Self {
            layers: vec![Rep3DenseInterleavedPolynomial::default(); num_layers],
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn layers(
        &'_ self,
    ) -> impl Iterator<Item = &'_ dyn Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>>
    {
        self.layers
            .iter()
            .map(|layer| layer as &'_ dyn Rep3BatchedGrandProductLayer<F, ProofTranscript, Network>)
    }
}
