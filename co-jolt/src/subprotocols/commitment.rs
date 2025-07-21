use crate::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, dense_mlpoly::DensePolynomial},
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{One, PrimeField, Zero};
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, CommitterKey, Proof, UniversalParams, VerifierKey},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter_mut, test_rng};
use jolt_core::{
    poly::multilinear_polynomial::MultilinearPolynomial, utils::transcript::KeccakTranscript,
};
use mpc_core::protocols::rep3::network::{Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_net::mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker};
use rand::RngCore;
use snarks_core::poly::commitment::{aggregate_comm, aggregate_comm_with_powers, aggregate_eval};
use std::{borrow::Borrow, marker::PhantomData};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::poly::Rep3DensePolynomial;

pub trait DistributedCommitmentScheme<F: JoltField, ProofTranscript: Transcript = KeccakTranscript>:
    CommitmentScheme<ProofTranscript, Field = F>
{
    fn distributed_batch_open<Network>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Self::Proof>
    where
        Network: MpcStarNetCoordinator;

    fn distributed_batch_open_worker<Network>(
        polys: &[&Rep3DensePolynomial<F>],
        ck: &Self::Setup,
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: MpcStarNetWorker;

    fn recieve_prove<Network>(network: &mut Network) -> eyre::Result<Self::Proof>
    where
        Network: Rep3NetworkCoordinator;

    fn prove_rep3<Network>(
        poly: &Rep3DensePolynomial<F>,
        ck: &Self::Setup,
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker;

    fn combine_commitment_shares(commitments: &[&Self::Commitment]) -> Self::Commitment;
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

    pub fn setup<R: RngCore>(max_len: usize, rng: &mut R) -> PST13Setup<E> {
        let num_vars = max_len.log_2();
        let uni_params = MultilinearPC::setup(num_vars, rng);
        PST13Setup { uni_params }
    }

    fn batch_prove<U, ProofTranscript: Transcript>(
        setup: &PST13Setup<E>,
        polynomials: &[U],
        opening_point: &[E::ScalarField],
        _openings: &[E::ScalarField],
        transcript: &mut ProofTranscript,
    ) -> Proof<E>
    where
        U: Borrow<DensePolynomial<E::ScalarField>> + Sync,
    {
        let eta: E::ScalarField = transcript.challenge_scalar();
        let batch_poly = aggregate_poly(eta, polynomials);
        Self::prove(
            setup,
            &MultilinearPolynomial::LargeScalars(batch_poly),
            opening_point,
            transcript,
        )
    }

    fn batch_verify<ProofTranscript: Transcript>(
        batch_proof: &Proof<E>,
        setup: &PST13Setup<E>,
        opening_point: &[E::ScalarField],
        openings: &[E::ScalarField],
        commitments: &[&PST13Commitment<E>],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let eta = transcript.challenge_scalar();

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
}

impl<E: Pairing, ProofTranscript: Transcript>
    DistributedCommitmentScheme<E::ScalarField, ProofTranscript> for PST13<E>
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
        let eta: E::ScalarField = transcript.challenge_scalar();
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

        network.send_response(pf.proofs)
    }

    fn combine_commitment_shares(commitments: &[&PST13Commitment<E>]) -> PST13Commitment<E> {
        let mut g_product = E::G1::zero();
        for commitment in commitments {
            g_product += commitment.g_product;
        }
        PST13Commitment {
            nv: commitments[0].nv,
            g_product: g_product.into_affine(),
        }
    }

    fn recieve_prove<Network>(network: &mut Network) -> eyre::Result<Proof<E>>
    where
        Network: Rep3NetworkCoordinator,
    {
        let [pf0, pf1, pf2]: [Vec<E::G1Affine>; 3] =
            network.receive_responses(Vec::new())?.try_into().unwrap();

        let proofs = itertools::multizip((pf0, pf1, pf2))
            .map(|(a, b, c)| (a + b + c).into_affine())
            .collect::<Vec<_>>();

        Ok(Proof { proofs })
    }

    fn prove_rep3<Network>(
        poly: &Rep3DensePolynomial<E::ScalarField>,
        setup: &Self::Setup,
        opening_point: &[E::ScalarField],
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker,
    {
        let opening_point_rev = opening_point.iter().copied().rev().collect::<Vec<_>>();
        let (pf, _) = open(
            &setup.ck(poly.get_num_vars()),
            &poly.copy_share_a(),
            &opening_point_rev,
        );
        network.send_response(pf.proofs)
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PST13Setup<E: Pairing> {
    pub uni_params: UniversalParams<E>,
}

impl<E: Pairing> Default for PST13Setup<E> {
    fn default() -> Self {
        Self {
            uni_params: UniversalParams {
                num_vars: 0,
                powers_of_g: vec![],
                powers_of_h: vec![],
                g: E::G1Affine::zero(),
                h: E::G2Affine::zero(),
                h_mask: vec![],
            },
        }
    }
}

impl<E: Pairing> PST13Setup<E> {
    pub fn ck(&self, num_vars: usize) -> CommitterKey<E> {
        MultilinearPC::trim(&self.uni_params, num_vars).0
    }

    pub fn vk(&self, num_vars: usize) -> VerifierKey<E> {
        MultilinearPC::trim(&self.uni_params, num_vars).1
    }
}

impl<E: Pairing, ProofTranscript: Transcript> CommitmentScheme<ProofTranscript> for PST13<E>
where
    E::ScalarField: JoltField,
{
    type Setup = PST13Setup<E>;
    type Field = E::ScalarField;
    type Proof = Proof<E>;
    type BatchedProof = Proof<E>;
    type Commitment = PST13Commitment<E>;

    fn setup(max_len: usize) -> Self::Setup {
        let mut rng = test_rng();
        PST13::setup(max_len, &mut rng)
    }

    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let nv = poly.get_num_vars();
        let poly = DensePolynomial::new(poly.coeffs_as_field_elements());
        let scalars: Vec<_> = poly.evals_ref().iter().map(|x| x.into_bigint()).collect();
        let g_product = <E::G1 as VariableBaseMSM>::msm_bigint(
            &setup.ck(nv).powers_of_g[0],
            scalars.as_slice(),
        )
        .into_affine();
        PST13Commitment { nv, g_product }
    }

    fn batch_commit<U>(evals: &[U], setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        let mut commitments = Vec::new();
        for evals in evals {
            let commitment =
                <Self as CommitmentScheme<ProofTranscript>>::commit(evals.borrow(), setup);
            commitments.push(commitment);
        }
        commitments
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let comm = aggregate_comm_with_powers::<E>(
            coeffs,
            &commitments.iter().map(|&c| c.into()).collect::<Vec<_>>(),
        );
        PST13Commitment {
            nv: commitments[0].nv,
            g_product: comm.g_product,
        }
    }

    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        assert_eq!(poly.get_num_vars(), opening_point.len());
        // reverse becasue evaluations via `DensePolynomial::bound_poly_var_top`
        let opening_point_rev = opening_point.iter().copied().rev().collect::<Vec<_>>();
        match poly {
            MultilinearPolynomial::LargeScalars(poly) => {
                open(&setup.ck(opening_point.len()), poly, &opening_point_rev).0
            }
            _ => todo!(),
        }
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

    fn protocol_name() -> &'static [u8] {
        b"PST13"
    }

    fn srs_size(setup: &Self::Setup) -> usize {
        1 << setup.uni_params.num_vars
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize, Eq, PartialEq)]
pub struct PST13Commitment<E: Pairing> {
    pub(crate) nv: usize,
    pub(crate) g_product: E::G1Affine,
}

impl<E: Pairing> Default for PST13Commitment<E> {
    fn default() -> Self {
        Self {
            nv: 0,
            g_product: E::G1Affine::zero(),
        }
    }
}

impl<E: Pairing> AppendToTranscript for PST13Commitment<E> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"g_product");
        transcript.append_point(&self.g_product.into_group());
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

pub fn aggregate_poly<F: JoltField, U>(eta: F, polys: &[U]) -> DensePolynomial<F>
where
    U: Borrow<DensePolynomial<F>> + Sync,
{
    let mut vars = 0;
    for p in polys {
        let num_vars = p.borrow().get_num_vars();
        if num_vars > vars {
            vars = num_vars;
        }
    }
    let mut evals = vec![F::zero(); 1 << vars];
    let mut x = F::one();
    for p in polys {
        cfg_iter_mut!(evals)
            .zip(p.borrow().evals_ref())
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
    use itertools::Itertools;
    use jolt_core::poly::multilinear_polynomial::PolynomialEvaluation;
    use jolt_core::utils::math::Math;
    use mpc_core::protocols::rep3;

    use crate::poly::Rep3DensePolynomial;

    use super::*;

    type E = ark_bn254::Bn254;
    type F = ark_bn254::Fr;
    type ProofTranscript = KeccakTranscript;

    #[test]
    fn test_combine_commitments() {
        let mut rng = test_rng();
        let setup = PST13::<E>::setup(1 << 3, &mut rng);

        let rho = F::rand(&mut rng);
        let mut rho_powers = vec![F::one()];
        for i in 1..3 {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let polys = iter::repeat_with(|| {
            MultilinearPolynomial::<F>::from(
                iter::repeat_with(|| F::rand(&mut rng))
                    .take(1 << 3)
                    .collect::<Vec<_>>(),
            )
        })
        .take(3)
        .collect::<Vec<_>>();

        let commitments = polys
            .iter()
            .map(|p| <PST13<E> as CommitmentScheme<ProofTranscript>>::commit(p, &setup))
            .collect::<Vec<_>>();

        let combined = <PST13<E> as CommitmentScheme<ProofTranscript>>::combine_commitments(
            &commitments.iter().collect::<Vec<_>>(),
            &rho_powers,
        );

        let agg_poly = MultilinearPolynomial::linear_combination(
            &polys.iter().collect::<Vec<_>>(),
            &rho_powers,
        );
        let agg_commitment =
            <PST13<E> as CommitmentScheme<ProofTranscript>>::commit(&agg_poly, &setup);

        assert_eq!(combined, agg_commitment);

        let r = iter::repeat_with(|| F::rand(&mut rng)).take(3).collect_vec();

        let pf = PST13::prove(&setup, &agg_poly, &r, &mut ProofTranscript::new(b"test"));
        let opening = agg_poly.evaluate(&r);
        let mut transcript = ProofTranscript::new(b"test");
        PST13::<E>::verify(&pf, &setup, &mut transcript, &r, &opening, &agg_commitment).unwrap();
    }


    // #[test]
    // fn test_open_plain() {
    //     const NUM_INPUTS: usize = 8;
    //     let mut rng = test_rng();
    //     let commitment_shapes = vec![CommitShape::new(NUM_INPUTS, BatchType::Big)];
    //     let ck = PST13::<E>::setup(&commitment_shapes, &mut rng);
    //     let poly = DensePolynomial::new(
    //         iter::repeat_with(|| F::rand(&mut rng))
    //             .take(NUM_INPUTS)
    //             .collect(),
    //     );
    //     let commitment = PST13::commit(&poly, &ck);
    //     let point = iter::repeat_with(|| F::rand(&mut rng))
    //         .take(NUM_INPUTS.log_2())
    //         .collect::<Vec<_>>();
    //     let proof = PST13::prove(&poly, &point, &ck, &mut ProofTranscript::new(b"test"));
    //     let opening = poly.evaluate(&point);
    //     let mut transcript = ProofTranscript::new(b"test");
    //     PST13::<E>::verify(&proof, &ck, &mut transcript, &point, &opening, &commitment).unwrap();
    // }

    // #[test]
    // fn test_open_rep3() {
    //     const NUM_INPUTS: usize = 8;
    //     let mut rng = test_rng();
    //     let commitment_shapes = vec![CommitShape::new(NUM_INPUTS, BatchType::Big)];
    //     let setup = PST13::<E>::setup(&commitment_shapes, &mut rng);
    //     let evals = iter::repeat_with(|| F::rand(&mut rng))
    //         .take(NUM_INPUTS)
    //         .collect::<Vec<_>>();
    //     let poly = DensePolynomial::new(evals.clone());
    //     let poly_share0 = Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
    //         evals.clone(),
    //         rep3::PartyID::ID0,
    //     ));

    //     let poly_share1 = Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
    //         evals.clone(),
    //         rep3::PartyID::ID1,
    //     ));

    //     let poly_share2 = Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
    //         evals,
    //         rep3::PartyID::ID2,
    //     ));

    //     let commitment = PST13::commit(&poly, &setup);
    //     let point = iter::repeat_with(|| F::rand(&mut rng))
    //         .take(NUM_INPUTS.log_2())
    //         .collect::<Vec<_>>();
    //     let mut transcript = ProofTranscript::new(b"test");
    //     let eta = transcript.challenge_scalar(b"eta");
    //     let batch_poly = aggregate_poly(
    //         eta,
    //         &[
    //             &poly_share0.copy_share_a(),
    //             &poly_share1.copy_share_a(),
    //             &poly_share2.copy_share_a(),
    //         ],
    //     );
    //     let ck = setup.ck(NUM_INPUTS.log_2());
    //     let [pf0, pf1, pf2] = {
    //         let opening_point_rev = point.iter().copied().rev().collect::<Vec<_>>();
    //         let (pf0, _) = open(&ck, &poly_share0.copy_share_a(), &opening_point_rev);
    //         let (pf1, _) = open(&ck, &poly_share1.copy_share_a(), &opening_point_rev);
    //         let (pf2, _) = open(&ck, &poly_share2.copy_share_a(), &opening_point_rev);
    //         [pf0.proofs, pf1.proofs, pf2.proofs]
    //     };
    //     let proofs = itertools::multizip((pf0, pf1, pf2))
    //         .map(|(a, b, c)| (a + b + c).into_affine())
    //         .collect::<Vec<_>>();
    //     let proof_combined = Proof { proofs };
    //     let opening = poly.evaluate(&point);
    //     let mut transcript = ProofTranscript::new(b"test");
    //     PST13::<E>::verify(
    //         &proof_combined,
    //         &setup,
    //         &mut transcript,
    //         &point,
    //         &opening,
    //         &commitment,
    //     )
    //     .unwrap();
    // }
}
