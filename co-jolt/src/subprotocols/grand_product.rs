use jolt_core::{
    poly::dense_mlpoly::DensePolynomial,
    utils::{math::Math, transcript::Transcript},
};
use jolt_core::{
    poly::{commitment::commitment_scheme::CommitmentScheme, split_eq_poly::SplitEqPolynomial},
    subprotocols::grand_product::{
        BatchedGrandProductLayer, BatchedGrandProductLayerProof, BatchedGrandProductProof,
    },
    utils::thread::drop_in_background_thread,
};
use mpc_core::protocols::rep3::{network::IoContextPool, Rep3PrimeFieldShare};
use mpc_core::protocols::{
    additive::AdditiveShare,
    rep3::network::{Rep3NetworkCoordinator, Rep3NetworkWorker},
};

use rayon::prelude::*;

use crate::{field::JoltField, poly::split_eq_poly::DistributedSplitEqPolynomial};
use crate::{
    poly::{
        dense_interleaved_poly::Rep3DenseInterleavedPolynomial,
        opening_proof::Rep3OpeningAccumulatorWorker,
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
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(BatchedGrandProductProof<PCS, ProofTranscript>, Vec<F>)> {
        let mut proof_layers = Vec::with_capacity(self.num_layers());

        // Evaluate the MLE of the output layer at a random point to reduce the outputs to
        // a single claim.
        transcript.append_scalars(&claimed_outputs);
        let output_mle = DensePolynomial::new_padded(claimed_outputs);
        let mut r_grand_product: Vec<F> = transcript.challenge_vector(output_mle.get_num_vars());
        let mut claim = output_mle.evaluate(&r_grand_product);
        network.broadcast_request(r_grand_product.clone())?;

        tracing::info!("num layers {}", self.num_layers());
        for layer in self.layers() {
            proof_layers.push(layer.coordinate_prove_layer(
                &mut claim,
                &mut r_grand_product,
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

    fn batch_size_minus_delta(&self) -> usize;

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
        _opening_accumulator: Option<&mut Rep3OpeningAccumulatorWorker<F>>,
        _setup: Option<&PCS::Setup>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Vec<F>> {
        let mut r = io_ctx.network().receive_request()?;
        let mut eq_chunk_size = self.batch_size_minus_delta();
        for (i, layer) in self.layers().into_iter().enumerate() {
            layer.prove_layer(&mut r, eq_chunk_size, io_ctx)?;
            eq_chunk_size *= 2;
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
        let num_rounds = r_grand_product.len();

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) = self.coordinate_prove_sumcheck(
            claim,
            r_grand_product,
            num_rounds,
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
        r_grand_product: &mut Vec<F>,
        eq_chunk_size: usize,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let mut eq_poly = DistributedSplitEqPolynomial::new(
            &r_grand_product,
            io_ctx.log_num_workers(),
            io_ctx.worker_idx(),
            eq_chunk_size,
        );

        let r_sumcheck = self.prove_sumcheck(&mut eq_poly, io_ctx)?;

        drop_in_background_thread(eq_poly);

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = io_ctx.network().receive_request()?;
        r_grand_product.push(r_layer);

        Ok(())
    }
}

pub struct Rep3BatchedDenseGrandProduct<F: JoltField> {
    layers: Vec<Rep3DenseInterleavedPolynomial<F>>,
    batch_size_minus_delta: usize,
}

impl<F: JoltField, PCS, ProofTranscript, Network>
    Rep3BatchedGrandProductWorker<F, PCS, ProofTranscript, Network>
    for Rep3BatchedDenseGrandProduct<F>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    type Leaves = (Vec<Rep3PrimeFieldShare<F>>, usize, usize);

    #[tracing::instrument(
        skip_all,
        name = "Rep3BatchedDenseGrandProduct::construct",
        level = "trace"
    )]
    fn construct(leaves: Self::Leaves, io_ctx: &mut IoContextPool<Network>) -> eyre::Result<Self> {
        let (leaves, batch_size, batch_size_full) = leaves;
        assert!(leaves.len() % batch_size == 0);
        assert!((leaves.len() / batch_size).is_power_of_two());

        // Number of chunks allocated to each worker except the last one, need to calculate equal poly chunk offset
        let batch_size_minus_delta =
            if io_ctx.log_num_workers() > 0 && io_ctx.worker_idx() == io_ctx.num_workers() - 1 {
                (batch_size_full - batch_size) / (io_ctx.num_workers() - 1)
            } else {
                batch_size
            };

        let num_layers = (leaves.len() / batch_size).log_2();
        let mut layers: Vec<Rep3DenseInterleavedPolynomial<F>> = Vec::with_capacity(num_layers);
        layers.push(Rep3DenseInterleavedPolynomial::new(leaves));

        for i in 0..num_layers - 1 {
            let previous_layer = &layers[i];
            let new_layer = previous_layer.layer_output(io_ctx)?;
            layers.push(new_layer);
        }

        Ok(Self {
            layers,
            batch_size_minus_delta,
        })
    }

    fn batch_size_minus_delta(&self) -> usize {
        self.batch_size_minus_delta
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
            batch_size_minus_delta: 0,
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
