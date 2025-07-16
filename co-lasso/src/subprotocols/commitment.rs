use ark_ec::ScalarMul;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{One, PrimeField, Zero};
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, CommitterKey, Proof, UniversalParams, VerifierKey},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter_mut, test_rng};
use jolt_core::{
    poly::{
        commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
        dense_mlpoly::DensePolynomial,
        field::JoltField,
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_net::mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker};
use rand::RngCore;
use snarks_core::poly::commitment::{aggregate_comm, aggregate_eval};
use std::marker::PhantomData;
use std::ops::Mul;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::poly::Rep3DensePolynomial;

pub trait DistributedCommitmentScheme<F: JoltField>: CommitmentScheme<Field = F> {
    fn distributed_batch_open<Network: MpcStarNetCoordinator>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Self::BatchedProof>
    where
        Network: MpcStarNetCoordinator;

    fn distributed_batch_open_worker<Network: MpcStarNetWorker>(
        polys: &[&Rep3DensePolynomial<F>],
        ck: &Self::Setup,
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: MpcStarNetWorker;

    fn combine_commitments(commitments: &[Self::Commitment]) -> Self::Commitment;
}

#[derive(Clone)]
pub struct PST13<E: Pairing> {
    _marker: PhantomData<E>,
}

impl<E: Pairing> PST13<E>
where
    E::ScalarField: JoltField,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    pub fn setup<R: RngCore>(commitment_shapes: &[CommitShape], rng: &mut R) -> PST13Setup<E> {
        let mut max_len: usize = 0;
        for shape in commitment_shapes {
            max_len = max_len.max(shape.input_length);
        }
        let num_vars = max_len.log_2();
        let uni_params = MultilinearPC::setup(num_vars, rng);
        PST13Setup { uni_params }
    }
}

impl<E: Pairing> DistributedCommitmentScheme<E::ScalarField> for PST13<E>
where
    E::ScalarField: JoltField,
{
    fn distributed_batch_open<Network: MpcStarNetCoordinator>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Proof<E>>
    where
        Network: MpcStarNetCoordinator,
    {
        let eta: E::ScalarField = transcript.challenge_scalar(b"eta");
        network.broadcast_request(eta)?;

        let [pf0, pf1, pf2]: [Vec<E::G1Affine>; 3] =
            network.receive_responses(Vec::new())?.try_into().unwrap();

        let proofs = itertools::multizip((pf0, pf1, pf2))
            .map(|(a, b, c)| (a + b + c).into_affine())
            .collect::<Vec<_>>();

        Ok(Proof { proofs })
    }

    fn distributed_batch_open_worker<Network: MpcStarNetWorker>(
        polys: &[&Rep3DensePolynomial<E::ScalarField>],
        setup: &Self::Setup,
        opening_point: &[E::ScalarField],
        network: &mut Network,
    ) -> eyre::Result<()> {
        let polys_a = polys.iter().map(|p| p.copy_share_a()).collect::<Vec<_>>();
        let eta: E::ScalarField = network.receive_request()?;

        let agg_poly = aggregate_poly(eta, &polys_a.iter().collect::<Vec<_>>());

        let opening_point_rev = opening_point.iter().copied().rev().collect::<Vec<_>>();
        let (pf, _) = open(
            &setup.ck(agg_poly.get_num_vars()),
            &agg_poly,
            &opening_point_rev,
        );

        // let mut evals = Vec::new();
        // for p in polys.iter() {
        //     evals.push(p.evaluate(&point[0..num_var - log_num_workers].to_vec()));
        // }
        // let response = PartialProof {
        //     proofs: pf,
        //     val: r,
        //     evals: openings.to_vec(),
        // };

        network.send_response(pf.proofs)
    }

    fn combine_commitments(commitments: &[PST13Commitment<E>]) -> PST13Commitment<E> {
        let mut g_product = E::G1::zero();
        for commitment in commitments {
            g_product += commitment.g_product;
        }
        PST13Commitment {
            nv: commitments[0].nv,
            g_product: g_product.into_affine(),
        }
    }
}

#[derive(Clone)]
pub struct PST13Setup<E: Pairing> {
    pub uni_params: UniversalParams<E>,
}

impl<E: Pairing> PST13Setup<E> {
    pub fn ck(&self, num_vars: usize) -> CommitterKey<E> {
        MultilinearPC::trim(&self.uni_params, num_vars).0
    }

    pub fn vk(&self, num_vars: usize) -> VerifierKey<E> {
        MultilinearPC::trim(&self.uni_params, num_vars).1
    }
}

impl<E: Pairing> CommitmentScheme for PST13<E>
where
    E::ScalarField: JoltField,
{
    type Setup = PST13Setup<E>;
    type Field = E::ScalarField;
    type Proof = Proof<E>;
    type BatchedProof = Proof<E>;
    type Commitment = PST13Commitment<E>;

    fn setup(
        shapes: &[jolt_core::poly::commitment::commitment_scheme::CommitShape],
    ) -> Self::Setup {
        let mut rng = test_rng();
        PST13::setup(shapes, &mut rng)
    }

    fn commit(poly: &DensePolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let nv = poly.get_num_vars();
        let scalars: Vec<_> = poly.evals_ref().iter().map(|x| x.into_bigint()).collect();
        let g_product = <E::G1 as VariableBaseMSM>::msm_bigint(
            &setup.ck(nv).powers_of_g[0],
            scalars.as_slice(),
        )
        .into_affine();
        PST13Commitment { nv, g_product }
    }

    fn batch_commit(
        evals: &[&[Self::Field]],
        setup: &Self::Setup,
        _batch_type: jolt_core::poly::commitment::commitment_scheme::BatchType,
    ) -> Vec<Self::Commitment> {
        let mut commitments = Vec::new();
        for evals in evals {
            let commitment = Self::commit(&DensePolynomial::new(evals.to_vec()), setup);
            commitments.push(commitment);
        }
        commitments
    }

    fn commit_slice(evals: &[Self::Field], setup: &Self::Setup) -> Self::Commitment {
        todo!()
    }

    fn prove(
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        setup: &Self::Setup,
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        assert_eq!(poly.get_num_vars(), opening_point.len());
        // reverse becasue evaluations via `DensePolynomial::bound_poly_var_top`
        let opening_point_rev = opening_point.iter().copied().rev().collect::<Vec<_>>();
        open(&setup.ck(opening_point.len()), poly, &opening_point_rev).0
    }

    fn batch_prove(
        polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        _openings: &[Self::Field],
        setup: &Self::Setup,
        _batch_type: jolt_core::poly::commitment::commitment_scheme::BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        let eta: Self::Field = transcript.challenge_scalar(b"eta");
        let batch_poly = aggregate_poly(eta, &polynomials);
        Self::prove(&batch_poly, opening_point, setup, transcript)
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        _transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        // reverse becasue evaluations via `DensePolynomial::bound_poly_var_top`

        let opening_point_rev = opening_point.iter().copied().rev().collect::<Vec<_>>();
        MultilinearPC::check(
            &setup.vk(opening_point.len()),
            &commitment.into(),
            &opening_point_rev,
            *opening,
            &proof,
        )
        .ok_or(ProofVerifyError::InternalError)
    }

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        setup: &Self::Setup,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let eta = transcript.challenge_scalar(b"eta");

        let batch_comm = aggregate_comm(
            eta,
            &commitments.iter().map(|&c| c.into()).collect::<Vec<_>>(),
        );

        let batch_eval = aggregate_eval(eta, openings);

        let opening_point_rev = opening_point.iter().copied().rev().collect::<Vec<_>>();

        MultilinearPC::check(
            &setup.vk(opening_point.len()),
            &batch_comm,
            &opening_point_rev,
            batch_eval,
            &batch_proof,
        )
        .ok_or(ProofVerifyError::InternalError)
    }

    fn protocol_name() -> &'static [u8] {
        todo!()
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize, Eq, PartialEq)]
pub struct PST13Commitment<E: Pairing> {
    pub(crate) nv: usize,
    pub(crate) g_product: E::G1Affine,
}

impl<E: Pairing> AppendToTranscript for PST13Commitment<E> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_point(label, &self.g_product.into_group());
    }
}

impl<E: Pairing> Into<Commitment<E>> for &PST13Commitment<E> {
    fn into(self) -> Commitment<E> {
        Commitment {
            nv: self.nv,
            g_product: self.g_product,
        }
    }
}

pub fn aggregate_poly<F: JoltField>(eta: F, polys: &[&DensePolynomial<F>]) -> DensePolynomial<F> {
    let mut vars = 0;
    for p in polys {
        let num_vars = p.get_num_vars();
        if num_vars > vars {
            vars = num_vars;
        }
    }
    let mut evals = vec![F::zero(); 1 << vars];
    let mut x = F::one();
    for p in polys {
        cfg_iter_mut!(evals)
            .zip(p.evals_ref())
            .for_each(|(a, b)| *a += x * b);
        x *= eta
    }
    assert_eq!(evals.len(), 1 << vars);
    DensePolynomial::new(evals)
}

fn open<E: Pairing>(
    ck: &CommitterKey<E>,
    polynomial: &DensePolynomial<E::ScalarField>,
    point: &[E::ScalarField],
) -> (Proof<E>, E::ScalarField)
where
    E::ScalarField: JoltField,
{
    let nv = polynomial.get_num_vars();
    assert_eq!(nv, ck.nv, "Invalid size of polynomial");
    let mut r: Vec<Vec<E::ScalarField>> = (0..nv + 1).map(|_| Vec::new()).collect();
    let mut q: Vec<Vec<E::ScalarField>> = (0..nv + 1).map(|_| Vec::new()).collect();

    r[nv] = polynomial.evals();

    let mut proofs = Vec::new();
    for i in 0..nv {
        let k = nv - i;
        let point_at_k = point[i];
        q[k] = (0..(1 << (k - 1)))
            .map(|_| E::ScalarField::zero())
            .collect();
        r[k - 1] = (0..(1 << (k - 1)))
            .map(|_| E::ScalarField::zero())
            .collect();
        for b in 0..(1 << (k - 1)) {
            q[k][b] = r[k][(b << 1) + 1] - &r[k][b << 1];
            r[k - 1][b] = r[k][b << 1] * &(E::ScalarField::one() - &point_at_k)
                + &(r[k][(b << 1) + 1] * &point_at_k);
        }
        let scalars: Vec<_> = (0..(1 << k)).map(|x| q[k][x >> 1].into_bigint()).collect();

        let pi_g =
            <E::G1 as VariableBaseMSM>::msm_bigint(&ck.powers_of_g[i], &scalars).into_affine(); // no need to move outside and partition
        proofs.push(pi_g);
    }

    (Proof { proofs }, r[0][0])
}

#[cfg(test)]
mod tests {
    use std::iter;

    use ark_ff::UniformRand;
    use ark_poly::DenseMultilinearExtension;
    use ark_poly::MultilinearExtension;
    use ark_poly::Polynomial;
    use jolt_core::utils::math::Math;
    use mpc_core::protocols::rep3;

    use crate::poly::Rep3DensePolynomial;

    use super::*;

    type E = ark_bn254::Bn254;
    type F = ark_bn254::Fr;

    #[test]
    fn test_open_plain() {
        const NUM_INPUTS: usize = 8;
        let mut rng = test_rng();
        let commitment_shapes = vec![CommitShape::new(NUM_INPUTS, BatchType::Big)];
        let ck = PST13::<E>::setup(&commitment_shapes, &mut rng);
        let poly = DensePolynomial::new(
            iter::repeat_with(|| F::rand(&mut rng))
                .take(NUM_INPUTS)
                .collect(),
        );
        let commitment = PST13::commit(&poly, &ck);
        let point = iter::repeat_with(|| F::rand(&mut rng))
            .take(NUM_INPUTS.log_2())
            .collect::<Vec<_>>();
        let proof = PST13::prove(&poly, &point, &ck, &mut ProofTranscript::new(b"test"));
        let opening = poly.evaluate(&point);
        let mut transcript = ProofTranscript::new(b"test");
        PST13::<E>::verify(&proof, &ck, &mut transcript, &point, &opening, &commitment).unwrap();
    }

    #[test]
    fn test_open_rep3() {
        const NUM_INPUTS: usize = 8;
        let mut rng = test_rng();
        let commitment_shapes = vec![CommitShape::new(NUM_INPUTS, BatchType::Big)];
        let setup = PST13::<E>::setup(&commitment_shapes, &mut rng);
        let evals = iter::repeat_with(|| F::rand(&mut rng))
            .take(NUM_INPUTS)
            .collect::<Vec<_>>();
        let poly = DensePolynomial::new(evals.clone());
        let poly_share0 = Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
            evals.clone(),
            rep3::PartyID::ID0,
        ));

        let poly_share1 = Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
            evals.clone(),
            rep3::PartyID::ID1,
        ));

        let poly_share2 = Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
            evals,
            rep3::PartyID::ID2,
        ));

        let commitment = PST13::commit(&poly, &setup);
        let point = iter::repeat_with(|| F::rand(&mut rng))
            .take(NUM_INPUTS.log_2())
            .collect::<Vec<_>>();
        let mut transcript = ProofTranscript::new(b"test");
        let eta = transcript.challenge_scalar(b"eta");
        let batch_poly = aggregate_poly(
            eta,
            &[
                &poly_share0.copy_share_a(),
                &poly_share1.copy_share_a(),
                &poly_share2.copy_share_a(),
            ],
        );
        let ck = setup.ck(NUM_INPUTS.log_2());
        let [pf0, pf1, pf2] = {
            let opening_point_rev = point.iter().copied().rev().collect::<Vec<_>>();
            let (pf0, _) = open(&ck, &poly_share0.copy_share_a(), &opening_point_rev);
            let (pf1, _) = open(&ck, &poly_share1.copy_share_a(), &opening_point_rev);
            let (pf2, _) = open(&ck, &poly_share2.copy_share_a(), &opening_point_rev);
            [pf0.proofs, pf1.proofs, pf2.proofs]
        };
        let proofs = itertools::multizip((pf0, pf1, pf2))
            .map(|(a, b, c)| (a + b + c).into_affine())
            .collect::<Vec<_>>();
        let proof_combined = Proof { proofs };
        let opening = poly.evaluate(&point);
        let mut transcript = ProofTranscript::new(b"test");
        PST13::<E>::verify(
            &proof_combined,
            &setup,
            &mut transcript,
            &point,
            &opening,
            &commitment,
        )
        .unwrap();
    }
}
